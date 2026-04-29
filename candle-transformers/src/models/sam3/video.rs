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

#[cfg(feature = "sam3-parity-support")]
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ParityTemporalDisambiguationFrameMetadata {
    pub removed_obj_ids: BTreeSet<u32>,
    pub suppressed_obj_ids: BTreeSet<u32>,
    pub unconfirmed_obj_ids: BTreeSet<u32>,
    pub matched_obj_ids: BTreeSet<u32>,
    pub unmatched_obj_ids: BTreeSet<u32>,
}

#[cfg(feature = "sam3-parity-support")]
pub trait Sam3VideoPredictorParityExt {
    fn parity_video_config(&self) -> &VideoConfig;
    fn parity_video_config_mut(&mut self) -> &mut VideoConfig;
    fn parity_session(&self, session_id: &str) -> Option<&Sam3VideoSession>;
    fn parity_session_mut(&mut self, session_id: &str) -> Option<&mut Sam3VideoSession>;
}

#[cfg(feature = "sam3-parity-support")]
pub trait Sam3VideoSessionParityExt {
    fn parity_tracked_objects(&self) -> &BTreeMap<u32, TrackedObject>;
    fn parity_tracked_objects_mut(&mut self) -> &mut BTreeMap<u32, TrackedObject>;
    fn parity_frame_outputs(&self) -> &BTreeMap<usize, BTreeMap<u32, ObjectFrameOutput>>;
    fn parity_frame_outputs_mut(
        &mut self,
    ) -> &mut BTreeMap<usize, BTreeMap<u32, ObjectFrameOutput>>;
    fn parity_temporal_disambiguation_metadata(
        &self,
    ) -> BTreeMap<usize, ParityTemporalDisambiguationFrameMetadata>;
}

#[cfg(feature = "sam3-parity-support")]
pub trait Sam3VideoTrackerCoreParityExt {
    fn parity_process_frame(
        &self,
        model: &Sam3ImageModel,
        compute_device: &Device,
        config: &VideoConfig,
        session: &mut Sam3VideoSession,
        frame_idx: usize,
        direction: PropagationDirection,
        output_threshold: f32,
    ) -> Result<VideoFrameOutput>;
}

#[cfg(feature = "sam3-parity-support")]
pub trait ObjectFrameOutputParityExt {
    fn parity_score_value(&self) -> Result<f32>;
}

#[cfg(feature = "sam3-parity-support")]
pub fn parity_replay_temporal_disambiguation_for_outputs(
    session: &mut Sam3VideoSession,
    outputs: &[VideoFrameOutput],
    matched_obj_ids_by_frame: &BTreeMap<usize, BTreeSet<u32>>,
    direction: PropagationDirection,
    video_config: &VideoConfig,
    confirmation_threshold: usize,
) -> Result<BTreeMap<usize, ParityTemporalDisambiguationFrameMetadata>> {
    let mut temporal_disambiguation_state = TemporalDisambiguationState::default();
    session.clear_temporal_disambiguation_metadata();
    if let Some(first_output) = outputs.first() {
        temporal_disambiguation_state.seed_prompt_frame_confirmation(
            session,
            first_output.frame_idx,
            direction,
            confirmation_threshold,
        );
    }
    for output in outputs {
        let frame_idx = output.frame_idx;
        let matched_obj_ids = matched_obj_ids_by_frame
            .get(&frame_idx)
            .cloned()
            .unwrap_or_default();
        temporal_disambiguation_state.record_confirmation_status(
            session,
            frame_idx,
            direction,
            &matched_obj_ids,
            confirmation_threshold,
        );
        let unmatched_obj_ids = output
            .objects
            .iter()
            .map(|object| object.obj_id)
            .filter(|obj_id| !matched_obj_ids.contains(obj_id))
            .collect::<BTreeSet<_>>();
        let empty_mask_obj_ids = output
            .objects
            .iter()
            .filter_map(|object| match mask_has_foreground(&object.mask_logits, 0.0) {
                Ok(true) => None,
                Ok(false) => Some(Ok(object.obj_id)),
                Err(err) => Some(Err(err)),
            })
            .collect::<Result<BTreeSet<_>>>()?;
        let previous_removed_obj_ids = temporal_disambiguation_state.hidden_obj_ids().clone();
        let unmatched_suppressed_obj_ids = if video_config.hotstart_delay > 0 {
            temporal_disambiguation_state.record_unmatched_outputs(
                session,
                frame_idx,
                &matched_obj_ids,
                &unmatched_obj_ids,
                &empty_mask_obj_ids,
                direction,
                video_config.hotstart_delay,
                video_config.hotstart_unmatch_thresh,
                video_config.suppress_unmatched_only_within_hotstart,
                video_config.max_trk_keep_alive,
                video_config.min_trk_keep_alive,
                video_config.decrease_trk_keep_alive_for_empty_masklets,
            )
        } else {
            BTreeSet::new()
        };
        let current_removed_obj_ids = temporal_disambiguation_state.hidden_obj_ids().clone();
        let newly_removed_obj_ids = current_removed_obj_ids
            .difference(&previous_removed_obj_ids)
            .copied()
            .collect::<BTreeSet<_>>();
        let current_suppressed_obj_ids = temporal_disambiguation_state
            .record_recent_occlusion_suppression(
                output,
                frame_idx,
                direction,
                video_config.suppress_overlapping_based_on_recent_occlusion_threshold,
                &newly_removed_obj_ids,
            )?;
        let effective_suppressed_obj_ids = current_suppressed_obj_ids
            .union(&unmatched_suppressed_obj_ids)
            .copied()
            .collect::<BTreeSet<_>>();
        let mut current_unconfirmed_obj_ids = temporal_disambiguation_state
            .unconfirmed_obj_ids_per_frame
            .get(&frame_idx)
            .cloned()
            .unwrap_or_default();
        current_unconfirmed_obj_ids.retain(|obj_id| !current_removed_obj_ids.contains(obj_id));
        session.temporal_disambiguation_metadata.insert(
            frame_idx,
            TemporalDisambiguationFrameMetadata {
                removed_obj_ids: current_removed_obj_ids,
                suppressed_obj_ids: effective_suppressed_obj_ids,
                unconfirmed_obj_ids: current_unconfirmed_obj_ids,
                matched_obj_ids: matched_obj_ids.clone(),
                unmatched_obj_ids: unmatched_obj_ids.clone(),
            },
        );
    }
    Ok(session.parity_temporal_disambiguation_metadata())
}
