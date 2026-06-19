use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use clap::{Parser, ValueEnum};
use image::{DynamicImage, Rgb, RgbImage};
use serde::Serialize;
use yolo_seg::{
    BoundingBox, Detection, DevicePreference, PrecisionPreference, StageTiming, Yolo26Config,
    Yolo26Segmenter,
};

#[derive(Debug, Clone, Copy, ValueEnum)]
enum DeviceArg {
    Auto,
    Cpu,
    Mps,
}

impl From<DeviceArg> for DevicePreference {
    fn from(value: DeviceArg) -> Self {
        match value {
            DeviceArg::Auto => Self::Auto,
            DeviceArg::Cpu => Self::Cpu,
            DeviceArg::Mps => Self::Mps,
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum PrecisionArg {
    Auto,
    F32,
    F16,
}

impl From<PrecisionArg> for PrecisionPreference {
    fn from(value: PrecisionArg) -> Self {
        match value {
            PrecisionArg::Auto => Self::Auto,
            PrecisionArg::F32 => Self::F32,
            PrecisionArg::F16 => Self::F16,
        }
    }
}

/// YOLO26 instance segmentation with Rust and LibTorch.
#[derive(Debug, Parser)]
#[command(name = "yolo26-seg", version)]
struct Args {
    /// Exported TorchScript model (not the original Ultralytics checkpoint).
    #[arg(long)]
    model: PathBuf,

    #[arg(long)]
    input: PathBuf,

    #[arg(long)]
    output: PathBuf,

    #[arg(long)]
    json: PathBuf,

    #[arg(long)]
    classes: Option<PathBuf>,

    #[arg(long, default_value_t = 0.25)]
    conf: f32,

    #[arg(long = "mask-threshold", default_value_t = 0.5)]
    mask_threshold: f32,

    #[arg(long, value_enum, default_value_t = DeviceArg::Auto)]
    device: DeviceArg,

    #[arg(long, value_enum, default_value_t = PrecisionArg::Auto)]
    precision: PrecisionArg,

    #[arg(long, default_value_t = 1)]
    warmup: u32,

    #[arg(long, default_value_t = 1)]
    iterations: u32,

    /// LibTorch CPU worker threads. Zero keeps the LibTorch default.
    #[arg(long, default_value_t = 0)]
    threads: i32,
}

#[derive(Serialize)]
struct JsonDetection<'a> {
    class_id: i64,
    class_name: &'a str,
    confidence: f32,
    #[serde(rename = "box")]
    box_: BoundingBox,
    mask_pixels: usize,
}

#[derive(Serialize)]
struct ImageSize {
    width: u32,
    height: u32,
}

#[derive(Serialize)]
struct JsonOutput<'a> {
    schema_version: u32,
    implementation: &'static str,
    backend: String,
    model: String,
    input: String,
    image_size: ImageSize,
    detections: Vec<JsonDetection<'a>>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    validate_args(&args)?;
    if args.threads > 0 {
        tch::set_num_threads(args.threads);
        tch::set_num_interop_threads(1);
    }

    let classes = load_classes(args.classes.as_deref())?;
    let image = image::open(&args.input)
        .with_context(|| format!("cannot decode input image: {}", args.input.display()))?;
    let config = Yolo26Config {
        confidence_threshold: args.conf,
        mask_threshold: args.mask_threshold,
        device: args.device.into(),
        precision: args.precision.into(),
        ..Yolo26Config::default()
    };
    let mut segmenter = Yolo26Segmenter::new(&args.model, config)
        .with_context(|| format!("cannot load model: {}", args.model.display()))?;
    println!("Selected backend: {}", segmenter.backend());

    for _ in 0..args.warmup {
        segmenter.predict(&image)?;
    }

    let mut prediction = segmenter.predict(&image)?;
    let mut sum = prediction.timing;
    for _ in 1..args.iterations {
        prediction = segmenter.predict(&image)?;
        add_timing(&mut sum, prediction.timing);
    }
    let average = divide_timing(sum, f64::from(args.iterations));

    ensure_parent(&args.output)?;
    ensure_parent(&args.json)?;
    render(&image, &prediction.detections).save(&args.output)?;
    write_json(
        &args,
        &image,
        &prediction.detections,
        &classes,
        &prediction.backend.to_string(),
    )?;

    println!("Backend: {}", prediction.backend);
    println!("Detections: {}", prediction.detections.len());
    println!(
        "Average over {} iteration(s): preprocess {:.2} ms, inference {:.2} ms, postprocess {:.2} ms, total {:.2} ms",
        args.iterations,
        average.preprocess_ms,
        average.inference_ms,
        average.postprocess_ms,
        average.total_ms
    );
    println!("Image: {}", args.output.display());
    println!("JSON: {}", args.json.display());
    Ok(())
}

fn validate_args(args: &Args) -> Result<()> {
    if !args.model.is_file() {
        bail!("model does not exist: {}", args.model.display());
    }
    if !args.input.is_file() {
        bail!("input image does not exist: {}", args.input.display());
    }
    if args.iterations == 0 {
        bail!("--iterations must be at least 1");
    }
    if args.threads < 0 {
        bail!("--threads cannot be negative");
    }
    Ok(())
}

fn load_classes(path: Option<&Path>) -> Result<Vec<String>> {
    let Some(path) = path else {
        return Ok(Vec::new());
    };
    let text = fs::read_to_string(path)
        .with_context(|| format!("cannot read class list: {}", path.display()))?;
    Ok(text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(str::to_owned)
        .collect())
}

fn class_name(class_id: i64, classes: &[String]) -> &str {
    usize::try_from(class_id)
        .ok()
        .and_then(|index| classes.get(index))
        .map(String::as_str)
        .unwrap_or("")
}

fn ensure_parent(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }
    Ok(())
}

fn write_json(
    args: &Args,
    image: &DynamicImage,
    detections: &[Detection],
    classes: &[String],
    backend: &str,
) -> Result<()> {
    let detections = detections
        .iter()
        .map(|detection| JsonDetection {
            class_id: detection.class_id,
            class_name: class_name(detection.class_id, classes),
            confidence: detection.confidence,
            box_: detection.bbox,
            mask_pixels: detection.mask_pixels(),
        })
        .collect();
    let output = JsonOutput {
        schema_version: 1,
        implementation: "yolo26-rust-libtorch",
        backend: backend.to_owned(),
        model: file_name(&args.model),
        input: file_name(&args.input),
        image_size: ImageSize {
            width: image.width(),
            height: image.height(),
        },
        detections,
    };
    fs::write(&args.json, serde_json::to_string_pretty(&output)? + "\n")?;
    Ok(())
}

fn file_name(path: &Path) -> String {
    path.file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default()
}

fn render(source: &DynamicImage, detections: &[Detection]) -> RgbImage {
    let mut image = source.to_rgb8();
    for detection in detections {
        let color = color_for_class(detection.class_id);
        for local_y in 0..detection.bbox.height {
            for local_x in 0..detection.bbox.width {
                let mask_index = (local_y * detection.bbox.width + local_x) as usize;
                if detection.mask.get(mask_index).copied().unwrap_or(0) != 0 {
                    let pixel =
                        image.get_pixel_mut(detection.bbox.x + local_x, detection.bbox.y + local_y);
                    pixel.0 = [
                        ((u16::from(pixel[0]) + u16::from(color[0])) / 2) as u8,
                        ((u16::from(pixel[1]) + u16::from(color[1])) / 2) as u8,
                        ((u16::from(pixel[2]) + u16::from(color[2])) / 2) as u8,
                    ];
                }
            }
        }
        draw_box(&mut image, detection.bbox, color, 2);
    }
    image
}

fn color_for_class(class_id: i64) -> Rgb<u8> {
    let channel = |multiplier: i64, offset: i64| {
        (multiplier.saturating_mul(class_id).saturating_add(offset)).rem_euclid(256) as u8
    };
    Rgb([channel(29, 97), channel(17, 149), channel(37, 53)])
}

fn draw_box(image: &mut RgbImage, bbox: BoundingBox, color: Rgb<u8>, thickness: u32) {
    let right = bbox.x + bbox.width - 1;
    let bottom = bbox.y + bbox.height - 1;
    for offset in 0..thickness {
        let x0 = (bbox.x + offset).min(right);
        let y0 = (bbox.y + offset).min(bottom);
        let x1 = right.saturating_sub(offset).max(x0);
        let y1 = bottom.saturating_sub(offset).max(y0);
        for x in x0..=x1 {
            image.put_pixel(x, y0, color);
            image.put_pixel(x, y1, color);
        }
        for y in y0..=y1 {
            image.put_pixel(x0, y, color);
            image.put_pixel(x1, y, color);
        }
    }
}

fn add_timing(total: &mut StageTiming, value: StageTiming) {
    total.preprocess_ms += value.preprocess_ms;
    total.inference_ms += value.inference_ms;
    total.postprocess_ms += value.postprocess_ms;
    total.total_ms += value.total_ms;
}

fn divide_timing(value: StageTiming, denominator: f64) -> StageTiming {
    StageTiming {
        preprocess_ms: value.preprocess_ms / denominator,
        inference_ms: value.inference_ms / denominator,
        postprocess_ms: value.postprocess_ms / denominator,
        total_ms: value.total_ms / denominator,
    }
}
