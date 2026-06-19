use thiserror::Error;

#[derive(Debug, Error)]
#[non_exhaustive]
pub enum Error {
    #[error("LibTorch error: {0}")]
    Torch(#[from] tch::TchError),

    #[error("cannot decode image: {0}")]
    Image(#[from] image::ImageError),

    #[error("invalid configuration: {0}")]
    Config(String),

    #[error("unexpected model output: {0}")]
    ModelOutput(String),
}
