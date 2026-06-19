use std::path::{Path, PathBuf};
use std::time::Instant;

use image::{DynamicImage, RgbImage};
use serde::Serialize;
use tch::{CModule, Device, IValue, IndexOp, Kind, Tensor};

use crate::{Error, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DevicePreference {
    Auto,
    Cpu,
    Mps,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrecisionPreference {
    Auto,
    F32,
    F16,
}

#[derive(Debug, Clone, Copy)]
pub struct Yolo26Config {
    pub input_width: u32,
    pub input_height: u32,
    pub confidence_threshold: f32,
    pub mask_threshold: f32,
    pub device: DevicePreference,
    pub precision: PrecisionPreference,
}

impl Default for Yolo26Config {
    fn default() -> Self {
        Self {
            input_width: 640,
            input_height: 640,
            confidence_threshold: 0.25,
            mask_threshold: 0.5,
            device: DevicePreference::Auto,
            precision: PrecisionPreference::Auto,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct BoundingBox {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

#[derive(Debug)]
pub struct Detection {
    pub class_id: i64,
    pub confidence: f32,
    pub bbox: BoundingBox,
    /// Row-major binary mask local to `bbox`; values are 0 or 1.
    pub mask: Vec<u8>,
}

impl Detection {
    pub fn mask_pixels(&self) -> usize {
        self.mask.iter().map(|&value| usize::from(value)).sum()
    }
}

#[derive(Debug, Default, Clone, Copy, Serialize)]
pub struct StageTiming {
    pub preprocess_ms: f64,
    pub inference_ms: f64,
    pub postprocess_ms: f64,
    pub total_ms: f64,
}

#[derive(Debug)]
pub struct Prediction {
    pub detections: Vec<Detection>,
    pub timing: StageTiming,
    pub backend: Backend,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Backend {
    CpuF32,
    MpsF32,
    MpsF16,
}

impl std::fmt::Display for Backend {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CpuF32 => formatter.write_str("cpu-f32"),
            Self::MpsF32 => formatter.write_str("mps-f32"),
            Self::MpsF16 => formatter.write_str("mps-f16"),
        }
    }
}

impl Backend {
    fn device(self) -> Device {
        match self {
            Self::CpuF32 => Device::Cpu,
            Self::MpsF32 | Self::MpsF16 => Device::Mps,
        }
    }

    fn kind(self) -> Kind {
        match self {
            Self::MpsF16 => Kind::Half,
            Self::CpuF32 | Self::MpsF32 => Kind::Float,
        }
    }
}

#[derive(Debug)]
struct Letterbox {
    scale: f64,
    left: u32,
    top: u32,
    right: u32,
    bottom: u32,
    resized_width: u32,
    resized_height: u32,
}

pub struct Yolo26Segmenter {
    config: Yolo26Config,
    model_path: PathBuf,
    module: CModule,
    backend: Backend,
    automatic_fallback: bool,
}

impl Yolo26Segmenter {
    pub fn new(model_path: impl AsRef<Path>, config: Yolo26Config) -> Result<Self> {
        validate_config(&config)?;
        let model_path = model_path.as_ref().to_path_buf();
        let automatic_fallback = config.device == DevicePreference::Auto;
        let mut backend = choose_backend(config.device, config.precision)?;
        let module = match load_module(&model_path, backend) {
            Ok(module) => module,
            Err(error) if automatic_fallback && backend.device() == Device::Mps => {
                eprintln!("warning: model cannot be loaded on MPS ({error}); falling back to CPU");
                backend = Backend::CpuF32;
                load_module(&model_path, backend)?
            }
            Err(error) => return Err(error),
        };

        Ok(Self {
            config,
            model_path,
            module,
            backend,
            automatic_fallback,
        })
    }

    pub fn backend(&self) -> Backend {
        self.backend
    }

    pub fn predict(&mut self, source: &DynamicImage) -> Result<Prediction> {
        match self.predict_once(source) {
            Ok(prediction) => Ok(prediction),
            Err(first_error) if self.automatic_fallback && self.backend == Backend::MpsF16 => {
                eprintln!("warning: MPS FP16 failed ({first_error}); retrying with MPS FP32");
                self.switch_backend(Backend::MpsF32)?;
                match self.predict_once(source) {
                    Ok(prediction) => Ok(prediction),
                    Err(second_error) => {
                        eprintln!("warning: MPS FP32 failed ({second_error}); falling back to CPU");
                        self.switch_backend(Backend::CpuF32)?;
                        self.predict_once(source)
                    }
                }
            }
            Err(error) if self.automatic_fallback && self.backend == Backend::MpsF32 => {
                eprintln!("warning: MPS failed ({error}); falling back to CPU");
                self.switch_backend(Backend::CpuF32)?;
                self.predict_once(source)
            }
            Err(error) => Err(error),
        }
    }

    fn switch_backend(&mut self, backend: Backend) -> Result<()> {
        self.module = load_module(&self.model_path, backend)?;
        self.backend = backend;
        Ok(())
    }

    fn predict_once(&self, source: &DynamicImage) -> Result<Prediction> {
        let total_start = Instant::now();
        let preprocess_start = Instant::now();
        let owned_source;
        let source = if let Some(rgb) = source.as_rgb8() {
            rgb
        } else {
            owned_source = source.to_rgb8();
            &owned_source
        };
        let transform = letterbox(source, self.config.input_width, self.config.input_height);
        let input = image_tensor(source, &transform, self.backend)?;
        let _ = f32::try_from(input.i((0, 0, 0, 0)))?;
        let preprocess_ms = elapsed_ms(preprocess_start);

        let inference_start = Instant::now();
        let output = tch::no_grad(|| {
            self.module
                .forward_is(&[IValue::Tensor(input)])
                .map_err(Error::from)
        })?;
        let (detection_tensor, prototypes) = extract_outputs(&output)?;
        // MPS dispatch is asynchronous. Reading one scalar makes the inference
        // timing honest without copying the large prototype tensor to the CPU.
        let _ = f32::try_from(detection_tensor.i((0, 0, 0)))?;
        let inference_ms = elapsed_ms(inference_start);

        let postprocess_start = Instant::now();
        let detections = decode(
            &detection_tensor,
            &prototypes,
            &transform,
            source.width(),
            source.height(),
            &self.config,
        )?;
        let postprocess_ms = elapsed_ms(postprocess_start);

        Ok(Prediction {
            detections,
            timing: StageTiming {
                preprocess_ms,
                inference_ms,
                postprocess_ms,
                total_ms: elapsed_ms(total_start),
            },
            backend: self.backend,
        })
    }
}

fn validate_config(config: &Yolo26Config) -> Result<()> {
    if config.input_width == 0 || config.input_height == 0 {
        return Err(Error::Config("input dimensions must be positive".into()));
    }
    if !(0.0..=1.0).contains(&config.confidence_threshold) {
        return Err(Error::Config(
            "confidence threshold must be between 0 and 1".into(),
        ));
    }
    if !(0.0..=1.0).contains(&config.mask_threshold) {
        return Err(Error::Config(
            "mask threshold must be between 0 and 1".into(),
        ));
    }
    if config.device == DevicePreference::Cpu && config.precision == PrecisionPreference::F16 {
        return Err(Error::Config(
            "FP16 is only supported by the MPS backend in this application".into(),
        ));
    }
    Ok(())
}

fn choose_backend(device: DevicePreference, precision: PrecisionPreference) -> Result<Backend> {
    let mps_available = tch::utils::has_mps();
    match device {
        DevicePreference::Cpu => Ok(Backend::CpuF32),
        DevicePreference::Mps => {
            if !mps_available {
                return Err(Error::Config(
                    "MPS was requested but this LibTorch build has no MPS backend".into(),
                ));
            }
            match precision {
                PrecisionPreference::F32 => Ok(Backend::MpsF32),
                PrecisionPreference::Auto | PrecisionPreference::F16 => Ok(Backend::MpsF16),
            }
        }
        DevicePreference::Auto => {
            if !mps_available {
                return Ok(Backend::CpuF32);
            }
            match precision {
                PrecisionPreference::F32 => Ok(Backend::MpsF32),
                PrecisionPreference::Auto | PrecisionPreference::F16 => Ok(Backend::MpsF16),
            }
        }
    }
}

fn load_module(path: &Path, backend: Backend) -> Result<CModule> {
    // Loading the whole archive on the target device also maps TorchScript
    // graph constants (YOLO anchors/strides), not only parameters and buffers.
    let mut module = CModule::load_on_device(path, backend.device())?;
    module.f_set_eval()?;
    if backend == Backend::MpsF16 {
        module.to(Device::Mps, Kind::Half, false);
    }
    Ok(module)
}

fn elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1_000.0
}

fn letterbox(source: &RgbImage, target_width: u32, target_height: u32) -> Letterbox {
    let scale = f64::min(
        f64::from(target_width) / f64::from(source.width()),
        f64::from(target_height) / f64::from(source.height()),
    );
    let resized_width = (f64::from(source.width()) * scale).round() as u32;
    let resized_height = (f64::from(source.height()) * scale).round() as u32;
    let left = ((f64::from(target_width - resized_width) / 2.0) - 0.1).round() as u32;
    let top = ((f64::from(target_height - resized_height) / 2.0) - 0.1).round() as u32;
    let right = ((f64::from(target_width - resized_width) / 2.0) + 0.1).round() as u32;
    let bottom = ((f64::from(target_height - resized_height) / 2.0) + 0.1).round() as u32;
    Letterbox {
        scale,
        left,
        top,
        right,
        bottom,
        resized_width,
        resized_height,
    }
}

fn image_tensor(image: &RgbImage, transform: &Letterbox, backend: Backend) -> Result<Tensor> {
    let tensor = Tensor::f_from_slice(image.as_raw())?
        .f_reshape([i64::from(image.height()), i64::from(image.width()), 3])?
        .f_permute([2, 0, 1])?
        .f_unsqueeze(0)?
        .f_to_device(backend.device())?
        .f_to_kind(backend.kind())?
        .f_div_scalar(255.0)?
        .f_upsample_bilinear2d(
            [
                i64::from(transform.resized_height),
                i64::from(transform.resized_width),
            ],
            false,
            None,
            None,
        )?
        .f_pad(
            [
                i64::from(transform.left),
                i64::from(transform.right),
                i64::from(transform.top),
                i64::from(transform.bottom),
            ],
            "constant",
            Some(114.0 / 255.0),
        )?;
    Ok(tensor)
}

fn collect_tensors(value: &IValue, tensors: &mut Vec<Tensor>) {
    match value {
        IValue::Tensor(tensor) => tensors.push(tensor.shallow_clone()),
        IValue::TensorList(values) => {
            tensors.extend(values.iter().map(Tensor::shallow_clone));
        }
        IValue::Tuple(values) | IValue::GenericList(values) => {
            for value in values {
                collect_tensors(value, tensors);
            }
        }
        IValue::GenericDict(values) => {
            for (key, value) in values {
                collect_tensors(key, tensors);
                collect_tensors(value, tensors);
            }
        }
        _ => {}
    }
}

fn extract_outputs(output: &IValue) -> Result<(Tensor, Tensor)> {
    let mut tensors = Vec::new();
    collect_tensors(output, &mut tensors);
    let prototypes = tensors
        .iter()
        .find(|tensor| {
            let shape = tensor.size();
            shape.len() == 4 && shape[0] == 1 && shape[1] > 0
        })
        .ok_or_else(|| {
            Error::ModelOutput(format!(
                "prototype tensor [1,C,H,W] not found; tensors: {}",
                shapes(&tensors)
            ))
        })?
        .shallow_clone();
    let channels = prototypes.size()[1];
    let mut detections = tensors
        .iter()
        .find(|tensor| {
            let shape = tensor.size();
            shape.len() == 3
                && shape[0] == 1
                && (shape[2] == channels + 6 || shape[1] == channels + 6)
        })
        .ok_or_else(|| {
            Error::ModelOutput(format!(
                "detection tensor [1,N,{}] not found; tensors: {}. Export with scripts/export_torchscript.py",
                channels + 6,
                shapes(&tensors)
            ))
        })?
        .shallow_clone();
    if detections.size()[2] != channels + 6 {
        detections = detections.f_transpose(1, 2)?;
    }
    Ok((detections, prototypes))
}

fn shapes(tensors: &[Tensor]) -> String {
    tensors
        .iter()
        .map(|tensor| format!("{:?}", tensor.size()))
        .collect::<Vec<_>>()
        .join(", ")
}

fn decode(
    detection_tensor: &Tensor,
    prototypes: &Tensor,
    transform: &Letterbox,
    source_width: u32,
    source_height: u32,
    config: &Yolo26Config,
) -> Result<Vec<Detection>> {
    let rows_tensor = detection_tensor
        .f_squeeze_dim(0)?
        .f_to_device(Device::Cpu)?
        .f_to_kind(Kind::Float)?;
    let rows = Vec::<Vec<f32>>::try_from(&rows_tensor)?;
    let channels = prototypes.size()[1] as usize;
    let mut kept_rows = Vec::new();
    let mut boxes = Vec::new();

    for row in rows {
        if row.len() != channels + 6 || !row[4].is_finite() || row[4] < config.confidence_threshold
        {
            continue;
        }
        if let Some(bbox) = map_box(&row, transform, source_width, source_height) {
            boxes.push((bbox, row[5].round() as i64, row[4]));
            kept_rows.extend_from_slice(&row[6..]);
        }
    }

    if boxes.is_empty() {
        return Ok(Vec::new());
    }

    // Mask reconstruction runs on the CPU. The per-detection bilinear upsample
    // below uses a different output size for every box; on MPS those dynamic
    // shapes trigger repeated Metal kernel recompilation that dominates video
    // latency. Inference (fixed 640x640) stays on the GPU where it is fast.
    let coefficients =
        Tensor::f_from_slice(&kept_rows)?.f_reshape([boxes.len() as i64, channels as i64])?;
    let proto_shape = prototypes.size();
    let proto_height = proto_shape[2];
    let proto_width = proto_shape[3];
    let proto_matrix = prototypes
        .f_squeeze_dim(0)?
        .f_flatten(1, 2)?
        .f_to_device(Device::Cpu)?
        .f_to_kind(Kind::Float)?;
    let mask_grids = coefficients
        .f_matmul(&proto_matrix)?
        .f_sigmoid()?
        .f_reshape([boxes.len() as i64, proto_height, proto_width])?;

    let mut detections = Vec::with_capacity(boxes.len());
    for (index, (bbox, class_id, confidence)) in boxes.into_iter().enumerate() {
        let fx = proto_width as f64 / f64::from(config.input_width);
        let fy = proto_height as f64 / f64::from(config.input_height);
        let to_proto_x = |x: u32| (f64::from(transform.left) + f64::from(x) * transform.scale) * fx;
        let to_proto_y = |y: u32| (f64::from(transform.top) + f64::from(y) * transform.scale) * fy;
        let x0 = (to_proto_x(bbox.x).floor() as i64).clamp(0, proto_width - 1);
        let y0 = (to_proto_y(bbox.y).floor() as i64).clamp(0, proto_height - 1);
        let x1 = (to_proto_x(bbox.x + bbox.width).ceil() as i64).clamp(x0 + 1, proto_width);
        let y1 = (to_proto_y(bbox.y + bbox.height).ceil() as i64).clamp(y0 + 1, proto_height);
        let crop = mask_grids
            .i((index as i64, y0..y1, x0..x1))
            .f_unsqueeze(0)?
            .f_unsqueeze(0)?;
        let mask = crop
            .f_upsample_bilinear2d(
                [i64::from(bbox.height), i64::from(bbox.width)],
                false,
                None,
                None,
            )?
            .f_gt(config.mask_threshold as f64)?
            .f_to_kind(Kind::Uint8)?
            .f_reshape([-1])?;
        detections.push(Detection {
            class_id,
            confidence,
            bbox,
            mask: Vec::<u8>::try_from(&mask)?,
        });
    }
    Ok(detections)
}

fn map_box(
    row: &[f32],
    transform: &Letterbox,
    source_width: u32,
    source_height: u32,
) -> Option<BoundingBox> {
    let x1 = (f64::from(row[0]) - f64::from(transform.left)) / transform.scale;
    let y1 = (f64::from(row[1]) - f64::from(transform.top)) / transform.scale;
    let x2 = (f64::from(row[2]) - f64::from(transform.left)) / transform.scale;
    let y2 = (f64::from(row[3]) - f64::from(transform.top)) / transform.scale;
    let left = (x1.floor() as i64).clamp(0, i64::from(source_width)) as u32;
    let top = (y1.floor() as i64).clamp(0, i64::from(source_height)) as u32;
    let right = (x2.ceil() as i64).clamp(0, i64::from(source_width)) as u32;
    let bottom = (y2.ceil() as i64).clamp(0, i64::from(source_height)) as u32;
    (right > left && bottom > top).then_some(BoundingBox {
        x: left,
        y: top,
        width: right - left,
        height: bottom - top,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn letterbox_and_inverse_box_match_portrait_source() {
        let image = RgbImage::new(810, 1080);
        let transform = letterbox(&image, 640, 640);
        assert_eq!(
            (
                transform.resized_width + transform.left + transform.right,
                transform.resized_height + transform.top + transform.bottom,
            ),
            (640, 640)
        );
        assert_eq!((transform.left, transform.top), (80, 0));
        let bbox = map_box(
            &[80.0, 0.0, 560.0, 640.0],
            &transform,
            image.width(),
            image.height(),
        )
        .unwrap();
        assert_eq!((bbox.x, bbox.y, bbox.width, bbox.height), (0, 0, 810, 1080));
    }
}
