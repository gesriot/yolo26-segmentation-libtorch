//! Rust/LibTorch YOLO26 instance-segmentation inference.

mod error;
mod segmenter;

pub use error::Error;
pub use segmenter::{
    Backend, BoundingBox, Detection, DevicePreference, PrecisionPreference, Prediction,
    StageTiming, Yolo26Config, Yolo26Segmenter,
};

pub type Result<T> = std::result::Result<T, Error>;
