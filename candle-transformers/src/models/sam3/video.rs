// Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::path::Path;
#[cfg(test)]
use std::process::Command;
#[cfg(test)]
use std::time::{SystemTime, UNIX_EPOCH};
#[cfg(test)]
use std::{fs, path::PathBuf};

use candle::{DType, Device, IndexOp, Result, Tensor};
use tokenizers::{PaddingDirection, PaddingParams, Tokenizer, TruncationParams};

use super::{
    geometry::{EncodedPrompt, GeometryPrompt},
    image::{GroundingOutput, ImageSize},
    neck::VisualBackboneOutput,
    text::TextEncoding,
    torch_ops::{
        interpolate::resize_bilinear2d_antialias,
        masks::{binary_planes_iou, mask_to_bool_plane},
    },
    Sam3ImageModel, Sam3TrackerModel, TrackerFrameState,
};
#[cfg(test)]
#[allow(unused_imports)]
use super::{Config, Sam3TrackerConfig};

const CLIP_EOT_TOKEN: &str = "<|endoftext|>";

mod config;
pub use config::*;
mod sources;
#[cfg(test)]
#[allow(unused_imports)]
pub(crate) use sources::{frame_blob_from_rgb_image_with_filter, FrameBlob};
pub use sources::{FrameSource, VideoSource};
mod propagation;
use propagation::*;
pub use propagation::{
    ObjectFrameOutput, Sam3VideoPredictor, Sam3VideoTrackerCore, SessionPrompt, VideoFrameOutput,
    VideoOutput,
};
mod session;
pub use session::{Sam3VideoSession, SessionCacheStats, TrackedObject};
mod temporal_disambiguation;
use temporal_disambiguation::{TemporalDisambiguationFrameMetadata, TemporalDisambiguationState};
mod postprocess;
use postprocess::*;
mod prompting;
use prompting::*;
mod debug;
#[cfg(test)]
#[allow(unused_imports)]
use debug::{count_foreground_pixels, tensor_to_mask_probs_2d};
use debug::{debug_prompt_metadata, VideoDebugRecorder};

const VIDEO_DEBUG_MANIFEST_FILE: &str = "debug_manifest.json";
const VIDEO_DEBUG_MASK_THRESHOLD: f32 = 0.5;
const VIDEO_PROPAGATION_FILL_HOLE_AREA: usize = 0;
const VIDEO_PROPAGATION_HOLE_FILL_LOGIT: f32 = 0.1;
const VIDEO_PROPAGATION_SPRINKLE_REMOVE_LOGIT: f32 = -0.1;

