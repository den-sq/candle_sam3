use std::{
    fmt::Debug,
    path::{Path, PathBuf},
    sync::Arc,
};

use candle::Result;

use super::super::Sam3TrackerConfig;

#[derive(Debug, Clone)]
pub struct VideoConfig {
    pub score_threshold: f32,
    pub score_threshold_detection: f32,
    /// Number of outputs retained before streaming begins. Values above zero
    /// delay prompt and non-prompt callbacks through the same bounded queue;
    /// the final partial queue is emitted as an ordered callback burst.
    pub hotstart_delay: usize,
    pub hotstart_unmatch_thresh: usize,
    pub suppress_unmatched_only_within_hotstart: bool,
    pub init_trk_keep_alive: isize,
    pub max_trk_keep_alive: isize,
    pub min_trk_keep_alive: isize,
    pub decrease_trk_keep_alive_for_empty_masklets: bool,
    pub suppress_overlapping_based_on_recent_occlusion_threshold: f32,
    pub max_objects: usize,
    pub memory_frame_count: usize,
    pub max_memory_boxes: usize,
    pub derive_mask_centroid_points: bool,
    pub fill_hole_area: usize,
    pub max_point_num_in_prompt_enc: usize,
    pub non_overlap_masks_for_output: bool,
}

impl Default for VideoConfig {
    fn default() -> Self {
        Self {
            score_threshold: 0.5,
            score_threshold_detection: 0.5,
            hotstart_delay: 0,
            hotstart_unmatch_thresh: 0,
            suppress_unmatched_only_within_hotstart: true,
            init_trk_keep_alive: 0,
            max_trk_keep_alive: 8,
            min_trk_keep_alive: -4,
            decrease_trk_keep_alive_for_empty_masklets: false,
            suppress_overlapping_based_on_recent_occlusion_threshold: 0.7,
            max_objects: usize::MAX,
            memory_frame_count: 6,
            max_memory_boxes: 2,
            derive_mask_centroid_points: true,
            fill_hole_area: 16,
            max_point_num_in_prompt_enc: 16,
            non_overlap_masks_for_output: false,
        }
    }
}

impl VideoConfig {
    pub(crate) fn from_tracker_config(config: &Sam3TrackerConfig) -> Self {
        Self {
            score_threshold: 0.5,
            score_threshold_detection: 0.5,
            hotstart_delay: config.predictor.hotstart_delay,
            hotstart_unmatch_thresh: config.predictor.hotstart_unmatch_thresh,
            suppress_unmatched_only_within_hotstart: true,
            init_trk_keep_alive: 0,
            max_trk_keep_alive: 8,
            min_trk_keep_alive: -4,
            decrease_trk_keep_alive_for_empty_masklets: false,
            suppress_overlapping_based_on_recent_occlusion_threshold: config
                .predictor
                .suppress_overlapping_based_on_recent_occlusion_threshold,
            max_objects: usize::MAX,
            memory_frame_count: config.num_maskmem.saturating_sub(1),
            max_memory_boxes: 2,
            derive_mask_centroid_points: true,
            fill_hole_area: config.predictor.fill_hole_area,
            max_point_num_in_prompt_enc: config.predictor.max_point_num_in_prompt_enc,
            non_overlap_masks_for_output: config.predictor.non_overlap_masks_for_output,
        }
    }
}

#[derive(Debug, Clone)]
pub enum VideoMemoryProfile {
    Balanced,
    LowMemory,
}

#[derive(Debug, Clone)]
pub struct VideoSessionOptions {
    pub tokenizer_path: Option<PathBuf>,
    pub memory_profile: VideoMemoryProfile,
    pub offload_frames_to_cpu: bool,
    pub offload_state_to_cpu: bool,
    pub prefetch_ahead: usize,
    pub prefetch_behind: usize,
    pub max_feature_cache_entries: usize,
    /// Maximum retained non-conditioning tracker states per object.
    ///
    /// `None` preserves the unbounded compatibility behavior. When set, the
    /// predictor rejects values smaller than the tracker memory/refinement
    /// windows and evicts older non-conditioning states after callbacks.
    pub max_non_cond_tracker_states: Option<usize>,
}

impl Default for VideoSessionOptions {
    fn default() -> Self {
        Self {
            tokenizer_path: None,
            memory_profile: VideoMemoryProfile::Balanced,
            offload_frames_to_cpu: false,
            offload_state_to_cpu: false,
            prefetch_ahead: 2,
            prefetch_behind: 1,
            max_feature_cache_entries: 2,
            max_non_cond_tracker_states: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct VideoDebugConfig {
    pub enabled: bool,
    pub capture_obj_ids: Vec<u32>,
    pub capture_frame_indices: Vec<usize>,
    pub capture_first_propagated_only: bool,
    pub output_root: Option<PathBuf>,
    /// Caller-owned encoder for binary mask artifacts.
    ///
    /// The sink receives a path relative to `output_root`, row-major binary
    /// pixels, and their dimensions. Core SAM3 code never depends on an image
    /// codec implementation.
    pub artifact_sink: Option<Arc<dyn VideoDebugArtifactSink>>,
}

pub trait VideoDebugArtifactSink: Debug + Send + Sync {
    fn write_binary_mask(
        &self,
        relative_path: &Path,
        width: usize,
        height: usize,
        pixels: &[u8],
    ) -> Result<()>;
}

impl Default for VideoDebugConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            capture_obj_ids: Vec::new(),
            capture_frame_indices: Vec::new(),
            capture_first_propagated_only: true,
            output_root: None,
            artifact_sink: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PropagationOptions {
    pub direction: PropagationDirection,
    pub start_frame_idx: Option<usize>,
    pub max_frame_num_to_track: Option<usize>,
    pub output_prob_threshold: Option<f32>,
}

impl Default for PropagationOptions {
    fn default() -> Self {
        Self {
            direction: PropagationDirection::Forward,
            start_frame_idx: None,
            max_frame_num_to_track: None,
            output_prob_threshold: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PropagationDirection {
    Forward,
    Backward,
    Both,
}
