//! Feature-gated re-exports for the external SAM3 parity repo.
//!
//! This module is intentionally small in the first migration step. It gives the
//! parity repo a stable namespace to target while parity-specific test helpers
//! continue to be lifted out of private in-tree test modules.

pub use super::config::{
    Config, DecoderConfig, EncoderConfig, GeometryConfig, ImageConfig, NeckConfig,
    SegmentationConfig, TextConfig, VisionConfig,
};
pub use super::geometry::{EncodedPrompt, GeometryPrompt, SequenceGeometryEncoder};
pub use super::image::{GroundingOutput, ImageSize, Sam3ImageModel, Sam3ImageState};
pub use super::tracker::{Sam3TrackerConfig, Sam3TrackerModel, TrackerFrameState, TrackerStepOutput};
pub use super::video::{
    FrameSource, ObjectFrameOutput, PropagationDirection, PropagationOptions, Sam3VideoPredictor,
    Sam3VideoSession, SessionCacheStats, SessionPrompt, TrackedObject, VideoConfig,
    VideoDebugConfig, VideoFrameOutput, VideoOutput, VideoSessionOptions, VideoSource,
};
