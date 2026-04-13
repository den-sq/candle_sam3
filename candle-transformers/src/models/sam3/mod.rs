#![allow(dead_code)]

//! SAM 3 scaffold.
//!
//! This module defines the intended Candle layout for a SAM 3 port:
//! a ViTDet-style visual trunk, a lightweight FPN neck, a CLIP-like
//! text encoder, a prompt/geometry encoder, a fusion encoder, a
//! DETR-style decoder with presence-token scoring, and a MaskFormer-like
//! segmentation head.
//!
//! The current implementation is deliberately a scaffold. The public
//! API and file layout are in place so the actual port can land
//! incrementally without reshaping the module tree later.

mod checkpoint;
pub mod config;
mod debug;
mod decoder;
mod encoder;
mod geometry;
mod image;
mod neck;
mod segmentation;
mod text;
mod video;
mod vitdet;

pub use checkpoint::{
    load_upstream_detector_var_builder, map_image_tensor_to_upstream_checkpoint_name,
    Sam3CheckpointSource, UPSTREAM_SAM3_DETECTOR_PREFIX, UPSTREAM_SAM3_STATE_KEY,
};
pub use config::{
    Config, DecoderConfig, EncoderConfig, GeometryConfig, ImageConfig, NeckConfig,
    SegmentationConfig, TextConfig, VisionConfig,
};
pub use debug::{capture_tensor, finish, set_exporter, DebugExporter};
pub use decoder::{DecoderOutput, Sam3TransformerDecoder};
pub use encoder::{FusionEncoderOutput, Sam3FusionEncoder};
pub use geometry::{EncodedPrompt, GeometryPrompt, SequenceGeometryEncoder};
pub use image::{GroundingOutput, ImageSize, Sam3ImageModel, Sam3ImageState, Sam3PromptState};
pub use neck::VisualBackboneOutput;
pub use segmentation::{SegmentationOutput, UniversalSegmentationHead};
pub use text::{Sam3TextEncoder, TextEncoding};
pub use video::{
    Sam3VideoPredictor, Sam3VideoSession, SessionPrompt, TrackedObject, VideoConfig,
    VideoOutput, PropagationDirection,
};
pub use vitdet::{Sam3ViTDetTrunk, ViTDetTrunkOutput};
