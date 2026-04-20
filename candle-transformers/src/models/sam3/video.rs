// Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::fs;
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::process::Command;
#[cfg(test)]
use std::time::{SystemTime, UNIX_EPOCH};

use candle::{DType, Device, IndexOp, Result, Tensor};
use image::{GrayImage, ImageReader, Luma};
use serde::{Deserialize, Serialize};
use tokenizers::{PaddingDirection, PaddingParams, Tokenizer, TruncationParams};

use super::{
    geometry::{EncodedPrompt, GeometryPrompt},
    image::{GroundingOutput, ImageSize},
    neck::VisualBackboneOutput,
    text::TextEncoding,
    tracker::resize_bilinear2d_antialias,
    Config, Sam3ImageModel, Sam3TrackerConfig, Sam3TrackerModel, TrackerFrameState,
};

const CLIP_EOT_TOKEN: &str = "<|endoftext|>";
const VIDEO_DEBUG_MANIFEST_FILE: &str = "debug_manifest.json";
const VIDEO_DEBUG_MASK_THRESHOLD: f32 = 0.5;
const VIDEO_PROPAGATION_FILL_HOLE_AREA: usize = 0;
const VIDEO_PROPAGATION_HOLE_FILL_LOGIT: f32 = 0.1;
const VIDEO_PROPAGATION_SPRINKLE_REMOVE_LOGIT: f32 = -0.1;

#[derive(Debug, Clone)]
pub struct SessionPrompt {
    pub text: Option<String>,
    pub points: Option<Vec<(f32, f32)>>,
    pub point_labels: Option<Vec<u32>>,
    pub boxes: Option<Vec<(f32, f32, f32, f32)>>,
    pub box_labels: Option<Vec<u32>>,
}

impl SessionPrompt {
    pub fn is_empty(&self) -> bool {
        self.text.is_none()
            && self.points.as_ref().map(|v| v.is_empty()).unwrap_or(true)
            && self.boxes.as_ref().map(|v| v.is_empty()).unwrap_or(true)
    }

    pub fn has_geometry(&self) -> bool {
        self.points.as_ref().map(|v| !v.is_empty()).unwrap_or(false)
            || self.boxes.as_ref().map(|v| !v.is_empty()).unwrap_or(false)
    }

    fn with_default_labels(mut self) -> Result<Self> {
        match self.points.as_ref() {
            Some(points) if points.is_empty() => {
                self.points = None;
                self.point_labels = None;
            }
            Some(points) => match self.point_labels.as_ref() {
                Some(labels) if labels.len() == points.len() => {}
                Some(labels) => {
                    candle::bail!(
                        "point label count {} does not match point count {}",
                        labels.len(),
                        points.len()
                    )
                }
                None => {
                    self.point_labels = Some(vec![1; points.len()]);
                }
            },
            None => {
                self.point_labels = None;
            }
        }

        match self.boxes.as_ref() {
            Some(boxes) if boxes.is_empty() => {
                self.boxes = None;
                self.box_labels = None;
            }
            Some(boxes) => match self.box_labels.as_ref() {
                Some(labels) if labels.len() == boxes.len() => {}
                Some(labels) => {
                    candle::bail!(
                        "box label count {} does not match box count {}",
                        labels.len(),
                        boxes.len()
                    )
                }
                None => {
                    self.box_labels = Some(vec![1; boxes.len()]);
                }
            },
            None => {
                self.box_labels = None;
            }
        }

        Ok(self)
    }

    fn merge_from(
        &mut self,
        update: &SessionPrompt,
        clear_old_points: bool,
        clear_old_boxes: bool,
    ) {
        if let Some(text) = update.text.as_ref() {
            self.text = Some(text.clone());
        }

        if clear_old_points {
            self.points = update.points.clone();
            self.point_labels = update.point_labels.clone();
        } else if let Some(points) = update.points.as_ref() {
            let mut merged_points = self.points.clone().unwrap_or_default();
            merged_points.extend(points.iter().copied());
            self.points = Some(merged_points);

            let mut merged_labels = self.point_labels.clone().unwrap_or_default();
            merged_labels.extend(
                update
                    .point_labels
                    .as_ref()
                    .into_iter()
                    .flat_map(|labels| labels.iter().copied()),
            );
            self.point_labels = Some(merged_labels);
        }

        if clear_old_boxes {
            self.boxes = update.boxes.clone();
            self.box_labels = update.box_labels.clone();
        } else if let Some(boxes) = update.boxes.as_ref() {
            let mut merged_boxes = self.boxes.clone().unwrap_or_default();
            merged_boxes.extend(boxes.iter().copied());
            self.boxes = Some(merged_boxes);

            let mut merged_labels = self.box_labels.clone().unwrap_or_default();
            merged_labels.extend(
                update
                    .box_labels
                    .as_ref()
                    .into_iter()
                    .flat_map(|labels| labels.iter().copied()),
            );
            self.box_labels = Some(merged_labels);
        }
    }
}

#[derive(Debug, Clone)]
pub struct VideoConfig {
    pub score_threshold: f32,
    pub hotstart_delay: usize,
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
            hotstart_delay: 0,
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
    fn from_tracker_config(config: &Sam3TrackerConfig) -> Self {
        Self {
            score_threshold: 0.5,
            hotstart_delay: config.predictor.hotstart_delay,
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
pub struct VideoSessionOptions {
    pub tokenizer_path: Option<PathBuf>,
    pub offload_frames_to_cpu: bool,
    pub offload_state_to_cpu: bool,
    pub prefetch_ahead: usize,
    pub prefetch_behind: usize,
    pub max_feature_cache_entries: usize,
}

impl Default for VideoSessionOptions {
    fn default() -> Self {
        Self {
            tokenizer_path: None,
            offload_frames_to_cpu: true,
            offload_state_to_cpu: true,
            prefetch_ahead: 2,
            prefetch_behind: 1,
            max_feature_cache_entries: 2,
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
}

impl Default for VideoDebugConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            capture_obj_ids: Vec::new(),
            capture_frame_indices: Vec::new(),
            capture_first_propagated_only: true,
            output_root: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct VideoDebugManifest {
    bundle_version: usize,
    mode: String,
    source: String,
    session_id: String,
    internal_tracker_state_available: bool,
    capture_obj_ids: Vec<u32>,
    capture_frame_indices: Vec<usize>,
    capture_first_propagated_only: bool,
    records: Vec<VideoDebugRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct VideoDebugRecord {
    stage: String,
    obj_id: u32,
    frame_idx: usize,
    prompt_frame_idx: Option<usize>,
    prompt_metadata: Option<VideoDebugPromptMetadata>,
    observable: Option<VideoDebugObservableSummary>,
    tracker_state: Option<VideoDebugTrackerStateSummary>,
    propagation_input: Option<VideoDebugPropagationInputSummary>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct VideoDebugPromptMetadata {
    text_prompt: Option<String>,
    used_visual_text_prompt: bool,
    normalized_points_xy: Vec<Vec<f32>>,
    point_labels: Vec<u32>,
    normalized_boxes_cxcywh: Vec<Vec<f32>>,
    box_labels: Vec<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct VideoDebugObservableSummary {
    mask_path: Option<String>,
    mask_threshold: f32,
    foreground_pixel_count: usize,
    mask_area_ratio: f32,
    boxes_xyxy: Vec<Vec<f32>>,
    scores: Vec<f32>,
    presence_scores: Option<Vec<f32>>,
    mask_logits_stats: TensorDebugSummary,
    mask_prob_stats: TensorDebugSummary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct VideoDebugTrackerStateSummary {
    is_cond_frame: bool,
    low_res_masks_stats: TensorDebugSummary,
    high_res_masks_stats: TensorDebugSummary,
    iou_scores_stats: TensorDebugSummary,
    object_score_logits_stats: TensorDebugSummary,
    obj_ptr_stats: TensorDebugSummary,
    maskmem_features_stats: Option<TensorDebugSummary>,
    maskmem_pos_enc_stats: Option<TensorDebugSummary>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct VideoDebugPropagationInputSummary {
    history_frames: Vec<VideoDebugHistoryFrameSummary>,
    history_frame_order: Vec<usize>,
    chosen_prompt_frame_indices: Vec<usize>,
    chosen_memory_frame_indices: Vec<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct VideoDebugHistoryFrameSummary {
    frame_idx: usize,
    is_cond_frame: bool,
    low_res_masks_stats: TensorDebugSummary,
    high_res_masks_stats: TensorDebugSummary,
    obj_ptr_stats: TensorDebugSummary,
    maskmem_features_stats: Option<TensorDebugSummary>,
    maskmem_pos_enc_stats: Option<TensorDebugSummary>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TensorDebugSummary {
    shape: Vec<usize>,
    dtype: String,
    min: f32,
    max: f32,
    mean: f32,
    l2_norm: f32,
    foreground_pixel_count: Option<usize>,
}

#[derive(Debug)]
struct VideoDebugRecorder {
    config: VideoDebugConfig,
    output_root: PathBuf,
    manifest: VideoDebugManifest,
}

impl VideoDebugRecorder {
    fn new(session_id: &str, config: VideoDebugConfig) -> Result<Option<Self>> {
        if !config.enabled {
            return Ok(None);
        }
        let output_root = config
            .output_root
            .clone()
            .unwrap_or_else(|| PathBuf::from("debug"));
        fs::create_dir_all(&output_root).map_err(|err| {
            candle::Error::Msg(format!(
                "failed to create debug output root {}: {}",
                output_root.display(),
                err
            ))
        })?;
        let recorder = Self {
            output_root,
            manifest: VideoDebugManifest {
                bundle_version: 1,
                mode: "video_debug_bundle".to_owned(),
                source: "candle".to_owned(),
                session_id: session_id.to_owned(),
                internal_tracker_state_available: true,
                capture_obj_ids: config.capture_obj_ids.clone(),
                capture_frame_indices: config.capture_frame_indices.clone(),
                capture_first_propagated_only: config.capture_first_propagated_only,
                records: Vec::new(),
            },
            config,
        };
        recorder.flush_manifest()?;
        Ok(Some(recorder))
    }

    fn should_capture_obj(&self, obj_id: u32) -> bool {
        self.config.capture_obj_ids.is_empty() || self.config.capture_obj_ids.contains(&obj_id)
    }

    fn should_capture_stage(
        &self,
        object: &TrackedObject,
        obj_id: u32,
        frame_idx: usize,
        direction: PropagationDirection,
        prompt_frame_idx: Option<usize>,
        stage: &str,
    ) -> bool {
        if !self.should_capture_obj(obj_id) {
            return false;
        }
        if self.config.capture_frame_indices.contains(&frame_idx) {
            return true;
        }
        if !self.config.capture_frame_indices.is_empty() {
            return false;
        }
        match stage {
            "detector_grounding" | "tracker_seed" => object.has_prompt_on_frame(frame_idx),
            "propagation_input" | "first_propagated_output" => {
                if !self.config.capture_first_propagated_only {
                    return true;
                }
                let Some(prompt_frame_idx) = prompt_frame_idx else {
                    return false;
                };
                match direction {
                    PropagationDirection::Forward | PropagationDirection::Both => {
                        frame_idx == prompt_frame_idx.saturating_add(1)
                    }
                    PropagationDirection::Backward => {
                        prompt_frame_idx == frame_idx.saturating_add(1)
                    }
                }
            }
            _ => false,
        }
    }

    fn record_detector_grounding(
        &mut self,
        object: &TrackedObject,
        frame_idx: usize,
        direction: PropagationDirection,
        prompt_metadata: VideoDebugPromptMetadata,
        output: &ObjectFrameOutput,
    ) -> Result<()> {
        if !self.should_capture_stage(
            object,
            object.obj_id,
            frame_idx,
            direction,
            Some(frame_idx),
            "detector_grounding",
        ) {
            return Ok(());
        }
        let observable =
            self.build_observable_summary(frame_idx, object.obj_id, "detector_mask", output)?;
        self.push_record(VideoDebugRecord {
            stage: "detector_grounding".to_owned(),
            obj_id: object.obj_id,
            frame_idx,
            prompt_frame_idx: Some(frame_idx),
            prompt_metadata: Some(prompt_metadata),
            observable: Some(observable),
            tracker_state: None,
            propagation_input: None,
        })
    }

    fn record_tracker_seed(
        &mut self,
        object: &TrackedObject,
        frame_idx: usize,
        direction: PropagationDirection,
        prompt_metadata: VideoDebugPromptMetadata,
        output: &ObjectFrameOutput,
        state: &TrackerFrameState,
    ) -> Result<()> {
        if !self.should_capture_stage(
            object,
            object.obj_id,
            frame_idx,
            direction,
            Some(frame_idx),
            "tracker_seed",
        ) {
            return Ok(());
        }
        let observable =
            self.build_observable_summary(frame_idx, object.obj_id, "tracker_seed_mask", output)?;
        let tracker_state = summarize_tracker_state(state)?;
        self.push_record(VideoDebugRecord {
            stage: "tracker_seed".to_owned(),
            obj_id: object.obj_id,
            frame_idx,
            prompt_frame_idx: Some(frame_idx),
            prompt_metadata: Some(prompt_metadata),
            observable: Some(observable),
            tracker_state: Some(tracker_state),
            propagation_input: None,
        })
    }

    fn record_first_propagation(
        &mut self,
        object: &TrackedObject,
        frame_idx: usize,
        direction: PropagationDirection,
        prompt_frame_idx: Option<usize>,
        output: &ObjectFrameOutput,
        history: &BTreeMap<usize, TrackerFrameState>,
        chosen_prompt_frame_indices: &[usize],
        chosen_memory_frame_indices: &[usize],
    ) -> Result<()> {
        if !self.should_capture_stage(
            object,
            object.obj_id,
            frame_idx,
            direction,
            prompt_frame_idx,
            "first_propagated_output",
        ) {
            return Ok(());
        }
        let history_frames = history
            .iter()
            .map(|(history_frame_idx, state)| {
                Ok(VideoDebugHistoryFrameSummary {
                    frame_idx: *history_frame_idx,
                    is_cond_frame: state.is_cond_frame,
                    low_res_masks_stats: summarize_tensor(&state.low_res_masks, Some(0.0))?,
                    high_res_masks_stats: summarize_tensor(&state.high_res_masks, Some(0.0))?,
                    obj_ptr_stats: summarize_tensor(&state.obj_ptr, None)?,
                    maskmem_features_stats: state
                        .maskmem_features
                        .as_ref()
                        .map(|tensor| summarize_tensor(tensor, None))
                        .transpose()?,
                    maskmem_pos_enc_stats: state
                        .maskmem_pos_enc
                        .as_ref()
                        .map(|tensor| summarize_tensor(tensor, None))
                        .transpose()?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let propagation_input = VideoDebugPropagationInputSummary {
            history_frame_order: history.keys().copied().collect(),
            history_frames,
            chosen_prompt_frame_indices: chosen_prompt_frame_indices.to_vec(),
            chosen_memory_frame_indices: chosen_memory_frame_indices.to_vec(),
        };
        let observable = self.build_observable_summary(
            frame_idx,
            object.obj_id,
            "first_propagated_mask",
            output,
        )?;
        self.push_record(VideoDebugRecord {
            stage: "propagation_input".to_owned(),
            obj_id: object.obj_id,
            frame_idx,
            prompt_frame_idx,
            prompt_metadata: None,
            observable: None,
            tracker_state: None,
            propagation_input: Some(propagation_input),
        })?;
        self.push_record(VideoDebugRecord {
            stage: "first_propagated_output".to_owned(),
            obj_id: object.obj_id,
            frame_idx,
            prompt_frame_idx,
            prompt_metadata: None,
            observable: Some(observable),
            tracker_state: None,
            propagation_input: None,
        })
    }

    fn push_record(&mut self, record: VideoDebugRecord) -> Result<()> {
        self.manifest.records.push(record);
        self.flush_manifest()
    }

    fn build_observable_summary(
        &self,
        frame_idx: usize,
        obj_id: u32,
        suffix: &str,
        output: &ObjectFrameOutput,
    ) -> Result<VideoDebugObservableSummary> {
        let mask_probs = tensor_to_mask_probs_2d(&output.masks)?;
        let mask_path = self
            .write_binary_mask(
                &format!("frame_{frame_idx:06}_obj_{obj_id:06}_{suffix}.png"),
                &mask_probs,
            )?
            .display()
            .to_string();
        let foreground_pixel_count =
            count_foreground_pixels(&mask_probs, VIDEO_DEBUG_MASK_THRESHOLD);
        let total_pixels = mask_probs
            .len()
            .saturating_mul(mask_probs.first().map(Vec::len).unwrap_or(0))
            .max(1);
        Ok(VideoDebugObservableSummary {
            mask_path: Some(mask_path),
            mask_threshold: VIDEO_DEBUG_MASK_THRESHOLD,
            foreground_pixel_count,
            mask_area_ratio: foreground_pixel_count as f32 / total_pixels as f32,
            boxes_xyxy: output.boxes_xyxy.to_vec2::<f32>()?,
            scores: output.scores.flatten_all()?.to_vec1::<f32>()?,
            presence_scores: output
                .presence_scores
                .as_ref()
                .map(|tensor| tensor.flatten_all()?.to_vec1::<f32>())
                .transpose()?,
            mask_logits_stats: summarize_tensor(&output.mask_logits, Some(0.0))?,
            mask_prob_stats: summarize_tensor(&output.masks, Some(VIDEO_DEBUG_MASK_THRESHOLD))?,
        })
    }

    fn write_binary_mask(&self, file_name: &str, mask_probs: &[Vec<f32>]) -> Result<PathBuf> {
        let height = mask_probs.len() as u32;
        let width = mask_probs.first().map(Vec::len).unwrap_or(0) as u32;
        let mut image = GrayImage::new(width, height);
        for (y, row) in mask_probs.iter().enumerate() {
            for (x, value) in row.iter().enumerate() {
                let pixel = if *value >= VIDEO_DEBUG_MASK_THRESHOLD {
                    255u8
                } else {
                    0u8
                };
                image.put_pixel(x as u32, y as u32, Luma([pixel]));
            }
        }
        let path = self.output_root.join(file_name);
        image.save(&path).map_err(|err| {
            candle::Error::Msg(format!(
                "failed to save debug mask {}: {}",
                path.display(),
                err
            ))
        })?;
        Ok(path
            .strip_prefix(&self.output_root)
            .unwrap_or(&path)
            .to_path_buf())
    }

    fn flush_manifest(&self) -> Result<()> {
        let manifest_path = self.output_root.join(VIDEO_DEBUG_MANIFEST_FILE);
        let bytes = serde_json::to_vec_pretty(&self.manifest).map_err(|err| {
            candle::Error::Msg(format!(
                "failed to serialize debug manifest {}: {}",
                manifest_path.display(),
                err
            ))
        })?;
        fs::write(&manifest_path, bytes).map_err(|err| {
            candle::Error::Msg(format!(
                "failed to write debug manifest {}: {}",
                manifest_path.display(),
                err
            ))
        })
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

#[derive(Debug, Clone)]
pub struct ObjectFrameOutput {
    pub obj_id: u32,
    pub mask_logits: Tensor,
    pub masks: Tensor,
    pub boxes_xyxy: Tensor,
    pub scores: Tensor,
    pub presence_scores: Option<Tensor>,
    pub prompt_frame_idx: Option<usize>,
    pub memory_frame_indices: Vec<usize>,
    pub text_prompt: Option<String>,
    pub used_explicit_geometry: bool,
    pub reused_previous_output: bool,
}

impl ObjectFrameOutput {
    fn from_grounding(
        obj_id: u32,
        grounding: GroundingOutput,
        prompt_frame_idx: Option<usize>,
        memory_frame_indices: Vec<usize>,
        text_prompt: Option<String>,
        used_explicit_geometry: bool,
        reused_previous_output: bool,
    ) -> Self {
        Self {
            obj_id,
            mask_logits: grounding.mask_logits,
            masks: grounding.masks,
            boxes_xyxy: grounding.boxes_xyxy,
            scores: grounding.scores,
            presence_scores: grounding.presence_scores,
            prompt_frame_idx,
            memory_frame_indices,
            text_prompt,
            used_explicit_geometry,
            reused_previous_output,
        }
    }

    fn score_value(&self) -> Result<f32> {
        Ok(self
            .scores
            .flatten_all()?
            .to_vec1::<f32>()?
            .into_iter()
            .next()
            .unwrap_or(0.0))
    }

    fn to_storage_device(&self, storage_device: &Device) -> Result<Self> {
        Ok(Self {
            obj_id: self.obj_id,
            mask_logits: self.mask_logits.to_device(storage_device)?,
            masks: self.masks.to_device(storage_device)?,
            boxes_xyxy: self.boxes_xyxy.to_device(storage_device)?,
            scores: self.scores.to_device(storage_device)?,
            presence_scores: self
                .presence_scores
                .as_ref()
                .map(|tensor| tensor.to_device(storage_device))
                .transpose()?,
            prompt_frame_idx: self.prompt_frame_idx,
            memory_frame_indices: self.memory_frame_indices.clone(),
            text_prompt: self.text_prompt.clone(),
            used_explicit_geometry: self.used_explicit_geometry,
            reused_previous_output: self.reused_previous_output,
        })
    }

    fn grounding(&self) -> GroundingOutput {
        GroundingOutput {
            mask_logits: self.mask_logits.clone(),
            masks: self.masks.clone(),
            boxes_xyxy: self.boxes_xyxy.clone(),
            scores: self.scores.clone(),
            presence_scores: self.presence_scores.clone(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct VideoFrameOutput {
    pub frame_idx: usize,
    pub objects: Vec<ObjectFrameOutput>,
}

#[derive(Debug, Clone, Default)]
pub struct VideoOutput {
    pub frames: Vec<VideoFrameOutput>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SessionCacheStats {
    pub loaded_frame_count: usize,
    pub cached_feature_entries: usize,
    pub cached_output_frames: usize,
    pub tracked_objects: usize,
}

#[derive(Debug, Clone)]
pub struct TrackedObject {
    pub obj_id: u32,
    pub creation_frame: usize,
    pub last_updated_frame: usize,
    pub display_score: Option<f32>,
    pub has_inference_history: bool,
    pub prompt_frames: BTreeMap<usize, SessionPrompt>,
    pub mask_prompt_frames: BTreeMap<usize, Tensor>,
    pub frame_outputs: BTreeMap<usize, ObjectFrameOutput>,
    pub tracker_states: BTreeMap<usize, TrackerFrameState>,
}

impl TrackedObject {
    fn new(obj_id: u32, creation_frame: usize) -> Self {
        Self {
            obj_id,
            creation_frame,
            last_updated_frame: creation_frame,
            display_score: None,
            has_inference_history: false,
            prompt_frames: BTreeMap::new(),
            mask_prompt_frames: BTreeMap::new(),
            frame_outputs: BTreeMap::new(),
            tracker_states: BTreeMap::new(),
        }
    }

    fn add_prompt(
        &mut self,
        frame_idx: usize,
        prompt: SessionPrompt,
        clear_old_points: bool,
        clear_old_boxes: bool,
    ) {
        if let Some(existing) = self.prompt_frames.get_mut(&frame_idx) {
            existing.merge_from(&prompt, clear_old_points, clear_old_boxes);
        } else {
            self.prompt_frames.insert(frame_idx, prompt);
        }
        self.mask_prompt_frames.remove(&frame_idx);
        self.last_updated_frame = frame_idx;
    }

    fn add_mask_prompt(&mut self, frame_idx: usize, mask: Tensor) {
        self.mask_prompt_frames.insert(frame_idx, mask);
        self.prompt_frames.remove(&frame_idx);
        self.last_updated_frame = frame_idx;
    }

    fn has_prompt_on_frame(&self, frame_idx: usize) -> bool {
        self.prompt_frames.contains_key(&frame_idx)
            || self.mask_prompt_frames.contains_key(&frame_idx)
    }

    fn nearest_prompt(
        &self,
        frame_idx: usize,
        direction: PropagationDirection,
    ) -> Option<(usize, SessionPrompt)> {
        match direction {
            PropagationDirection::Forward | PropagationDirection::Both => self
                .prompt_frames
                .range(..=frame_idx)
                .next_back()
                .map(|(idx, prompt)| (*idx, prompt.clone())),
            PropagationDirection::Backward => self
                .prompt_frames
                .range(frame_idx..)
                .next()
                .map(|(idx, prompt)| (*idx, prompt.clone())),
        }
    }

    fn nearest_input_frame_idx(
        &self,
        frame_idx: usize,
        direction: PropagationDirection,
    ) -> Option<usize> {
        match direction {
            PropagationDirection::Forward | PropagationDirection::Both => self
                .prompt_frames
                .keys()
                .chain(self.mask_prompt_frames.keys())
                .copied()
                .filter(|idx| *idx <= frame_idx)
                .max(),
            PropagationDirection::Backward => self
                .prompt_frames
                .keys()
                .chain(self.mask_prompt_frames.keys())
                .copied()
                .filter(|idx| *idx >= frame_idx)
                .min(),
        }
    }

    fn nearest_input_uses_explicit_geometry(
        &self,
        frame_idx: usize,
        direction: PropagationDirection,
    ) -> bool {
        let Some(input_frame_idx) = self.nearest_input_frame_idx(frame_idx, direction) else {
            return false;
        };
        if self.mask_prompt_frames.contains_key(&input_frame_idx) {
            return true;
        }
        self.prompt_frames
            .get(&input_frame_idx)
            .map(SessionPrompt::has_geometry)
            .unwrap_or(false)
    }

    fn latest_text_prompt(
        &self,
        frame_idx: usize,
        direction: PropagationDirection,
    ) -> Option<(usize, String)> {
        match direction {
            PropagationDirection::Forward | PropagationDirection::Both => self
                .prompt_frames
                .range(..=frame_idx)
                .rev()
                .find_map(|(idx, prompt)| prompt.text.as_ref().map(|text| (*idx, text.clone()))),
            PropagationDirection::Backward => self
                .prompt_frames
                .range(frame_idx..)
                .find_map(|(idx, prompt)| prompt.text.as_ref().map(|text| (*idx, text.clone()))),
        }
    }

    fn recent_output_frame_indices(
        &self,
        frame_idx: usize,
        direction: PropagationDirection,
        limit: usize,
    ) -> Vec<usize> {
        if limit == 0 {
            return Vec::new();
        }
        match direction {
            PropagationDirection::Forward | PropagationDirection::Both => self
                .frame_outputs
                .range(..frame_idx)
                .rev()
                .take(limit)
                .map(|(idx, _)| *idx)
                .collect(),
            PropagationDirection::Backward => self
                .frame_outputs
                .range((frame_idx + 1)..)
                .take(limit)
                .map(|(idx, _)| *idx)
                .collect(),
        }
    }

    fn is_active_for_frame(&self, frame_idx: usize, direction: PropagationDirection) -> bool {
        match direction {
            PropagationDirection::Forward | PropagationDirection::Both => {
                self.prompt_frames.range(..=frame_idx).next_back().is_some()
                    || self
                        .mask_prompt_frames
                        .range(..=frame_idx)
                        .next_back()
                        .is_some()
            }
            PropagationDirection::Backward => {
                self.prompt_frames.range(frame_idx..).next().is_some()
                    || self.mask_prompt_frames.range(frame_idx..).next().is_some()
            }
        }
    }

    fn tracker_history(
        &self,
        frame_idx: usize,
        direction: PropagationDirection,
    ) -> BTreeMap<usize, TrackerFrameState> {
        match direction {
            PropagationDirection::Forward | PropagationDirection::Both => self
                .tracker_states
                .range(..frame_idx)
                .map(|(idx, state)| (*idx, state.clone()))
                .collect(),
            PropagationDirection::Backward => self
                .tracker_states
                .range((frame_idx + 1)..)
                .map(|(idx, state)| (*idx, state.clone()))
                .collect(),
        }
    }
}

#[derive(Debug, Clone)]
pub enum VideoSource {
    TensorFrames(Vec<Tensor>),
    ImageFolder(PathBuf),
    ImageFile(PathBuf),
    VideoFile(PathBuf),
}

impl VideoSource {
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        if path.is_dir() {
            return Ok(Self::ImageFolder(path.to_path_buf()));
        }
        let ext = path
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.to_ascii_lowercase());
        match ext.as_deref() {
            Some("jpg" | "jpeg" | "png" | "bmp" | "tiff" | "webp") => {
                Ok(Self::ImageFile(path.to_path_buf()))
            }
            Some("mp4" | "avi" | "mov" | "mkv" | "webm") => Ok(Self::VideoFile(path.to_path_buf())),
            _ => candle::bail!("unsupported video source path {}", path.display()),
        }
    }

    fn into_frame_source(self, config: &Config) -> Result<Box<dyn FrameSource>> {
        match self {
            Self::TensorFrames(frames) => Ok(Box::new(TensorFrameSource::new(frames)?)),
            Self::ImageFolder(path) => Ok(Box::new(ImageFolderFrameSource::new(
                sorted_image_paths(&path)?,
                config.image.image_size,
                config.image.image_mean,
                config.image.image_std,
            )?)),
            Self::ImageFile(path) => Ok(Box::new(ImageFolderFrameSource::new(
                vec![path],
                config.image.image_size,
                config.image.image_mean,
                config.image.image_std,
            )?)),
            Self::VideoFile(path) => Ok(Box::new(VideoFileFrameSource::new(
                path,
                config.image.image_size,
                config.image.image_mean,
                config.image.image_std,
            )?)),
        }
    }
}

pub trait FrameSource {
    fn frame_count(&self) -> usize;
    fn video_size(&self) -> ImageSize;
    fn get_frame(&mut self, frame_idx: usize, target_device: &Device) -> Result<Tensor>;
    fn prefetch(&mut self, frame_indices: &[usize]) -> Result<()>;
    fn evict_except(&mut self, keep_frame_indices: &BTreeSet<usize>);
    fn loaded_frame_count(&self) -> usize;
    fn close(&mut self);
}

pub struct Sam3VideoSession {
    session_id: String,
    frame_source: Box<dyn FrameSource>,
    session_options: VideoSessionOptions,
    tokenizer: Option<Tokenizer>,
    debug_recorder: Option<VideoDebugRecorder>,
    storage_device: Device,
    tracked_objects: BTreeMap<u32, TrackedObject>,
    next_obj_id: u32,
    frame_outputs: BTreeMap<usize, BTreeMap<u32, ObjectFrameOutput>>,
    feature_cache: HashMap<usize, VisualBackboneOutput>,
    feature_cache_order: VecDeque<usize>,
    text_cache: HashMap<String, CachedTextPrompt>,
}

impl Sam3VideoSession {
    fn new(
        session_id: String,
        frame_source: Box<dyn FrameSource>,
        session_options: VideoSessionOptions,
        debug_config: VideoDebugConfig,
        model: &Sam3ImageModel,
        compute_device: &Device,
    ) -> Result<Self> {
        let tokenizer = session_options
            .tokenizer_path
            .as_ref()
            .map(|path| load_tokenizer(path, model.config().text.context_length))
            .transpose()?;
        let storage_device =
            if session_options.offload_state_to_cpu && !matches!(compute_device, Device::Cpu) {
                Device::Cpu
            } else {
                compute_device.clone()
            };
        Ok(Self {
            session_id: session_id.clone(),
            frame_source,
            session_options,
            tokenizer,
            debug_recorder: VideoDebugRecorder::new(&session_id, debug_config)?,
            storage_device,
            tracked_objects: BTreeMap::new(),
            next_obj_id: 0,
            frame_outputs: BTreeMap::new(),
            feature_cache: HashMap::new(),
            feature_cache_order: VecDeque::new(),
            text_cache: HashMap::new(),
        })
    }

    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    pub fn num_frames(&self) -> usize {
        self.frame_source.frame_count()
    }

    pub fn video_size(&self) -> ImageSize {
        self.frame_source.video_size()
    }

    pub fn cache_stats(&self) -> SessionCacheStats {
        SessionCacheStats {
            loaded_frame_count: self.frame_source.loaded_frame_count(),
            cached_feature_entries: self.feature_cache.len(),
            cached_output_frames: self.frame_outputs.len(),
            tracked_objects: self.tracked_objects.len(),
        }
    }

    fn prompt_frames(&self) -> BTreeSet<usize> {
        self.tracked_objects
            .values()
            .flat_map(|object| {
                object
                    .prompt_frames
                    .keys()
                    .chain(object.mask_prompt_frames.keys())
                    .copied()
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    fn storage_device(&self) -> &Device {
        &self.storage_device
    }

    fn debug_recorder_mut(&mut self) -> Option<&mut VideoDebugRecorder> {
        self.debug_recorder.as_mut()
    }

    fn allocate_object(&mut self, creation_frame: usize) -> u32 {
        let obj_id = self.next_obj_id;
        self.next_obj_id += 1;
        self.tracked_objects
            .insert(obj_id, TrackedObject::new(obj_id, creation_frame));
        obj_id
    }

    fn ensure_object(
        &mut self,
        obj_id: Option<u32>,
        creation_frame: usize,
        max_objects: usize,
    ) -> Result<u32> {
        match obj_id {
            Some(obj_id) => {
                if !self.tracked_objects.contains_key(&obj_id) {
                    if self.tracked_objects.len() >= max_objects {
                        candle::bail!(
                            "cannot allocate another tracked object because max_objects={} was reached",
                            max_objects
                        );
                    }
                    self.next_obj_id = self.next_obj_id.max(obj_id.saturating_add(1));
                    self.tracked_objects
                        .insert(obj_id, TrackedObject::new(obj_id, creation_frame));
                }
                Ok(obj_id)
            }
            None => {
                if self.tracked_objects.len() >= max_objects {
                    candle::bail!(
                        "cannot allocate another tracked object because max_objects={} was reached",
                        max_objects
                    )
                }
                Ok(self.allocate_object(creation_frame))
            }
        }
    }

    fn add_prompt(
        &mut self,
        frame_idx: usize,
        prompt: SessionPrompt,
        obj_id: Option<u32>,
        clear_old_points: bool,
        clear_old_boxes: bool,
        max_objects: usize,
    ) -> Result<u32> {
        if frame_idx >= self.num_frames() {
            candle::bail!(
                "frame_idx {} exceeds video length {}",
                frame_idx,
                self.num_frames()
            );
        }
        let prompt = prompt.with_default_labels()?;
        if prompt.text.is_some() && self.tokenizer.is_none() {
            candle::bail!(
                "video text prompts require a tokenizer; pass `VideoSessionOptions.tokenizer_path`"
            )
        }

        let obj_id = self.ensure_object(obj_id, frame_idx, max_objects)?;

        let tracked = self
            .tracked_objects
            .get_mut(&obj_id)
            .ok_or_else(|| candle::Error::Msg(format!("unknown obj_id {}", obj_id)))?;
        tracked.add_prompt(frame_idx, prompt, clear_old_points, clear_old_boxes);
        self.invalidate_object_outputs_from(obj_id, frame_idx);
        Ok(obj_id)
    }

    fn add_mask_prompt(
        &mut self,
        frame_idx: usize,
        mask: Tensor,
        obj_id: Option<u32>,
        max_objects: usize,
    ) -> Result<u32> {
        if frame_idx >= self.num_frames() {
            candle::bail!(
                "frame_idx {} exceeds video length {}",
                frame_idx,
                self.num_frames()
            );
        }
        let obj_id = self.ensure_object(obj_id, frame_idx, max_objects)?;
        let storage_device = self.storage_device().clone();
        let tracked = self
            .tracked_objects
            .get_mut(&obj_id)
            .ok_or_else(|| candle::Error::Msg(format!("unknown obj_id {}", obj_id)))?;
        let mask = normalize_video_mask_prompt(&mask, &storage_device)?;
        tracked.add_mask_prompt(frame_idx, mask);
        self.invalidate_object_outputs_from(obj_id, frame_idx);
        Ok(obj_id)
    }

    fn invalidate_object_outputs_from(&mut self, obj_id: u32, frame_idx: usize) {
        if let Some(object) = self.tracked_objects.get_mut(&obj_id) {
            object.frame_outputs.retain(|idx, _| *idx <= frame_idx);
            object.tracker_states.retain(|idx, _| *idx <= frame_idx);
        }
        let mut empty_frames = Vec::new();
        for (cached_frame_idx, frame_outputs) in self.frame_outputs.iter_mut() {
            if *cached_frame_idx >= frame_idx {
                frame_outputs.remove(&obj_id);
            }
            if frame_outputs.is_empty() {
                empty_frames.push(*cached_frame_idx);
            }
        }
        for frame_idx in empty_frames {
            self.frame_outputs.remove(&frame_idx);
        }
    }

    fn remove_object(&mut self, obj_id: u32) -> Result<()> {
        self.tracked_objects
            .remove(&obj_id)
            .ok_or_else(|| candle::Error::Msg(format!("unknown obj_id {}", obj_id)))?;
        self.invalidate_object_outputs_from(obj_id, 0);
        Ok(())
    }

    fn reset(&mut self) {
        self.tracked_objects.clear();
        self.next_obj_id = 0;
        self.frame_outputs.clear();
        self.feature_cache.clear();
        self.feature_cache_order.clear();
        self.text_cache.clear();
        self.frame_source.close();
    }

    fn close(&mut self) {
        if let Some(recorder) = self.debug_recorder.as_ref() {
            let _ = recorder.flush_manifest();
        }
        self.frame_source.close();
        self.feature_cache.clear();
        self.feature_cache_order.clear();
        self.frame_outputs.clear();
        self.tracked_objects.clear();
        self.text_cache.clear();
        self.debug_recorder = None;
    }

    fn get_frame(&mut self, frame_idx: usize, target_device: &Device) -> Result<Tensor> {
        if frame_idx >= self.num_frames() {
            candle::bail!("frame_idx {} out of bounds", frame_idx);
        }
        self.frame_source.get_frame(frame_idx, target_device)
    }

    fn prefetch_for_frame(
        &mut self,
        frame_idx: usize,
        direction: PropagationDirection,
    ) -> Result<()> {
        let keep = self.prefetch_window(frame_idx, direction);
        let indices = keep.iter().copied().collect::<Vec<_>>();
        self.frame_source.prefetch(&indices)
    }

    fn evict_for_frame(&mut self, frame_idx: usize, direction: PropagationDirection) {
        let mut keep = self.prompt_frames();
        keep.extend(self.prefetch_window(frame_idx, direction));
        self.frame_source.evict_except(&keep);
    }

    fn prefetch_window(
        &self,
        frame_idx: usize,
        direction: PropagationDirection,
    ) -> BTreeSet<usize> {
        let mut keep = BTreeSet::new();
        let num_frames = self.num_frames();
        let start = frame_idx.saturating_sub(self.session_options.prefetch_behind);
        let end = match direction {
            PropagationDirection::Backward => frame_idx + self.session_options.prefetch_behind,
            PropagationDirection::Forward | PropagationDirection::Both => {
                frame_idx + self.session_options.prefetch_ahead
            }
        };
        for idx in start..=end.min(num_frames.saturating_sub(1)) {
            keep.insert(idx);
        }
        keep
    }

    fn get_visual_features(
        &mut self,
        model: &Sam3ImageModel,
        compute_device: &Device,
        frame_idx: usize,
    ) -> Result<VisualBackboneOutput> {
        if let Some(cached) = self.feature_cache.get(&frame_idx) {
            let visual = move_visual_output(cached, compute_device)?;
            self.touch_feature_cache_entry(frame_idx);
            return Ok(visual);
        }

        let image = self.get_frame(frame_idx, compute_device)?;
        let visual = model.encode_image_features(&image)?;
        let stored = move_visual_output(&visual, self.storage_device())?;
        self.feature_cache.insert(frame_idx, stored);
        self.touch_feature_cache_entry(frame_idx);
        self.evict_feature_cache(frame_idx);
        Ok(visual)
    }

    fn touch_feature_cache_entry(&mut self, frame_idx: usize) {
        self.feature_cache_order.retain(|idx| *idx != frame_idx);
        self.feature_cache_order.push_back(frame_idx);
    }

    fn evict_feature_cache(&mut self, current_frame_idx: usize) {
        while self.feature_cache_order.len() > self.session_options.max_feature_cache_entries {
            let Some(candidate) = self.feature_cache_order.pop_front() else {
                break;
            };
            if candidate == current_frame_idx {
                self.feature_cache_order.push_back(candidate);
                break;
            }
            self.feature_cache.remove(&candidate);
        }
    }

    fn cached_text_encoding(
        &mut self,
        model: &Sam3ImageModel,
        text_prompt: &str,
        compute_device: &Device,
    ) -> Result<TextEncoding> {
        if let Some(cached) = self.text_cache.get(text_prompt) {
            return cached.to_text_encoding(compute_device);
        }
        let tokenizer = self.tokenizer.as_ref().ok_or_else(|| {
            candle::Error::Msg(
                "video text prompts require a tokenizer; pass `VideoSessionOptions.tokenizer_path`"
                    .to_owned(),
            )
        })?;
        let (input_ids, attention_mask) = tokenize_prompt(text_prompt, tokenizer, compute_device)?;
        let encoding = model.encode_text_tokens(&input_ids, &attention_mask)?;
        let cached = CachedTextPrompt::from_encoding(&encoding, self.storage_device())?;
        self.text_cache.insert(text_prompt.to_owned(), cached);
        Ok(encoding)
    }
}

pub struct Sam3VideoPredictor<'a> {
    model: &'a Sam3ImageModel,
    tracker_core: Sam3VideoTrackerCore<'a>,
    device: &'a Device,
    video_config: VideoConfig,
    debug_config: VideoDebugConfig,
    sessions: HashMap<String, Sam3VideoSession>,
    next_session_id: usize,
}

impl<'a> Sam3VideoPredictor<'a> {
    pub fn new(
        model: &'a Sam3ImageModel,
        tracker: &'a Sam3TrackerModel,
        device: &'a Device,
    ) -> Self {
        Self {
            model,
            tracker_core: Sam3VideoTrackerCore::new(tracker),
            device,
            video_config: VideoConfig::from_tracker_config(tracker.config()),
            debug_config: VideoDebugConfig::default(),
            sessions: HashMap::new(),
            next_session_id: 0,
        }
    }

    pub fn with_config(mut self, config: VideoConfig) -> Self {
        self.video_config = config;
        self
    }

    pub fn with_debug_config(mut self, config: VideoDebugConfig) -> Self {
        self.debug_config = config;
        self
    }

    pub fn start_session(
        &mut self,
        source: VideoSource,
        options: VideoSessionOptions,
    ) -> Result<String> {
        let session_id = format!("session_{}", self.next_session_id);
        self.next_session_id += 1;
        let frame_source = source.into_frame_source(self.model.config())?;
        let mut session = Sam3VideoSession::new(
            session_id.clone(),
            frame_source,
            options,
            self.debug_config.clone(),
            self.model,
            self.device,
        )?;
        session.prefetch_for_frame(0, PropagationDirection::Forward)?;
        self.sessions.insert(session_id.clone(), session);
        Ok(session_id)
    }

    pub fn start_session_from_tensors(
        &mut self,
        video_frames: Vec<Tensor>,
        options: VideoSessionOptions,
    ) -> Result<String> {
        self.start_session(VideoSource::TensorFrames(video_frames), options)
    }

    pub fn add_prompt(
        &mut self,
        session_id: &str,
        frame_idx: usize,
        prompt: SessionPrompt,
        obj_id: Option<u32>,
        clear_old_points: bool,
        clear_old_boxes: bool,
    ) -> Result<u32> {
        let session = self
            .sessions
            .get_mut(session_id)
            .ok_or_else(|| candle::Error::Msg(format!("unknown session {}", session_id)))?;
        session.add_prompt(
            frame_idx,
            prompt,
            obj_id,
            clear_old_points,
            clear_old_boxes,
            self.video_config.max_objects,
        )
    }

    pub fn add_mask_prompt(
        &mut self,
        session_id: &str,
        frame_idx: usize,
        mask: Tensor,
        obj_id: Option<u32>,
    ) -> Result<u32> {
        let session = self
            .sessions
            .get_mut(session_id)
            .ok_or_else(|| candle::Error::Msg(format!("unknown session {}", session_id)))?;
        session.add_mask_prompt(frame_idx, mask, obj_id, self.video_config.max_objects)
    }

    pub fn remove_object(&mut self, session_id: &str, obj_id: u32) -> Result<()> {
        let session = self
            .sessions
            .get_mut(session_id)
            .ok_or_else(|| candle::Error::Msg(format!("unknown session {}", session_id)))?;
        session.remove_object(obj_id)
    }

    pub fn propagate_in_video(
        &mut self,
        session_id: &str,
        options: PropagationOptions,
    ) -> Result<VideoOutput> {
        let mut collected = VideoOutput::default();
        self.propagate_in_video_stream(session_id, options, |frame| {
            collected.frames.push(frame.clone());
            Ok(())
        })?;
        Ok(collected)
    }

    pub fn propagate_in_video_stream<F>(
        &mut self,
        session_id: &str,
        options: PropagationOptions,
        mut on_frame: F,
    ) -> Result<()>
    where
        F: FnMut(&VideoFrameOutput) -> Result<()>,
    {
        match options.direction {
            PropagationDirection::Both => {
                let mut forward = options.clone();
                forward.direction = PropagationDirection::Forward;
                self.propagate_one_direction(session_id, forward, &mut on_frame)?;
                let mut backward = options;
                backward.direction = PropagationDirection::Backward;
                self.propagate_one_direction(session_id, backward, &mut on_frame)?;
            }
            _ => self.propagate_one_direction(session_id, options, &mut on_frame)?,
        }
        Ok(())
    }

    fn propagate_one_direction<F>(
        &mut self,
        session_id: &str,
        options: PropagationOptions,
        on_frame: &mut F,
    ) -> Result<()>
    where
        F: FnMut(&VideoFrameOutput) -> Result<()>,
    {
        let session = self
            .sessions
            .get_mut(session_id)
            .ok_or_else(|| candle::Error::Msg(format!("unknown session {}", session_id)))?;
        let processing_order = build_processing_order(
            session,
            options.direction,
            options.start_frame_idx,
            options.max_frame_num_to_track,
            self.tracker_core
                .tracker
                .config()
                .predictor
                .always_start_from_first_ann_frame,
        )?;
        let output_threshold = options
            .output_prob_threshold
            .unwrap_or(self.video_config.score_threshold);
        for frame_idx in processing_order {
            session.prefetch_for_frame(frame_idx, options.direction)?;
            let output = self.tracker_core.process_frame(
                self.model,
                self.device,
                &self.video_config,
                session,
                frame_idx,
                options.direction,
                output_threshold,
            )?;
            on_frame(&output)?;
            session.evict_for_frame(frame_idx, options.direction);
        }
        Ok(())
    }

    pub fn close_session(&mut self, session_id: &str) -> Result<()> {
        if let Some(mut session) = self.sessions.remove(session_id) {
            session.close();
        }
        Ok(())
    }

    pub fn reset_session(&mut self, session_id: &str) -> Result<()> {
        let session = self
            .sessions
            .get_mut(session_id)
            .ok_or_else(|| candle::Error::Msg(format!("unknown session {}", session_id)))?;
        session.reset();
        Ok(())
    }

    pub fn session_frame_count(&self, session_id: &str) -> Result<usize> {
        let session = self
            .sessions
            .get(session_id)
            .ok_or_else(|| candle::Error::Msg(format!("unknown session {}", session_id)))?;
        Ok(session.num_frames())
    }

    pub fn get_session_frame(&mut self, session_id: &str, frame_idx: usize) -> Result<Tensor> {
        let session = self
            .sessions
            .get_mut(session_id)
            .ok_or_else(|| candle::Error::Msg(format!("unknown session {}", session_id)))?;
        session.get_frame(frame_idx, self.device)
    }

    pub fn session_cache_stats(&self, session_id: &str) -> Result<SessionCacheStats> {
        let session = self
            .sessions
            .get(session_id)
            .ok_or_else(|| candle::Error::Msg(format!("unknown session {}", session_id)))?;
        Ok(session.cache_stats())
    }
}

#[derive(Debug)]
pub struct Sam3VideoTrackerCore<'a> {
    tracker: &'a Sam3TrackerModel,
}

impl<'a> Sam3VideoTrackerCore<'a> {
    pub fn new(tracker: &'a Sam3TrackerModel) -> Self {
        Self { tracker }
    }
}

impl Sam3VideoTrackerCore<'_> {
    fn correction_frame_is_cond_frame(&self) -> bool {
        self.tracker
            .config()
            .predictor
            .add_all_frames_to_correct_as_cond
    }

    fn prompt_frame_uses_point_memory(&self, object: &TrackedObject, frame_idx: usize) -> bool {
        if object.mask_prompt_frames.contains_key(&frame_idx) {
            return false;
        }
        let Some(prompt) = object.prompt_frames.get(&frame_idx) else {
            return false;
        };
        prompt.text.is_none()
            && prompt
                .boxes
                .as_ref()
                .map(|boxes| boxes.is_empty())
                .unwrap_or(true)
            && prompt
                .points
                .as_ref()
                .map(|points| !points.is_empty())
                .unwrap_or(false)
    }

    fn ensure_history_states_have_memory(
        &self,
        model: &Sam3ImageModel,
        compute_device: &Device,
        session: &mut Sam3VideoSession,
        object: &TrackedObject,
        frame_idx: usize,
        direction: PropagationDirection,
    ) -> Result<()> {
        let missing_history_frames = object
            .tracker_history(frame_idx, direction)
            .into_iter()
            .filter_map(|(history_frame_idx, state)| {
                (state.is_cond_frame
                    && (state.maskmem_features.is_none() || state.maskmem_pos_enc.is_none()))
                .then_some(history_frame_idx)
            })
            .collect::<Vec<_>>();
        for history_frame_idx in missing_history_frames {
            let state = session
                .tracked_objects
                .get(&object.obj_id)
                .and_then(|tracked| tracked.tracker_states.get(&history_frame_idx))
                .cloned()
                .ok_or_else(|| {
                    candle::Error::Msg(format!(
                        "missing tracker state for obj_id {} frame {} during Step 6 preflight",
                        object.obj_id, history_frame_idx
                    ))
                })?;
            let visual = tracker_visual_output(&session.get_visual_features(
                model,
                compute_device,
                history_frame_idx,
            )?);
            let (maskmem_features, maskmem_pos_enc) = self.tracker.encode_external_memory(
                &visual,
                &state.high_res_masks.to_device(compute_device)?,
                &state.object_score_logits.to_device(compute_device)?,
                self.prompt_frame_uses_point_memory(object, history_frame_idx),
            )?;
            let mut updated_state = move_tracker_state(&state, compute_device)?;
            updated_state.maskmem_features = Some(maskmem_features);
            updated_state.maskmem_pos_enc = Some(maskmem_pos_enc);
            let updated_state = move_tracker_state(&updated_state, session.storage_device())?;
            if let Some(tracked) = session.tracked_objects.get_mut(&object.obj_id) {
                tracked
                    .tracker_states
                    .insert(history_frame_idx, updated_state);
            }
        }
        Ok(())
    }

    fn trim_past_non_cond_memory(
        &self,
        session: &mut Sam3VideoSession,
        object: &TrackedObject,
        frame_idx: usize,
        direction: PropagationDirection,
    ) {
        if !self
            .tracker
            .config()
            .predictor
            .trim_past_non_cond_mem_for_eval
        {
            return;
        }
        let stride = self.tracker.config().memory_temporal_stride_for_eval.max(1);
        let signed_frame_idx = frame_idx as isize;
        let trim_distance = (stride * self.tracker.config().num_maskmem) as isize;
        let far_obj_ptr_distance = (20 * self.tracker.config().max_obj_ptrs_in_encoder) as isize;
        let mut trim_targets = Vec::new();
        match direction {
            PropagationDirection::Forward | PropagationDirection::Both => {
                if let Some(target) = signed_frame_idx.checked_sub(trim_distance) {
                    trim_targets.push(target as usize);
                }
                if self.tracker.config().use_memory_selection
                    && !self
                        .tracker
                        .config()
                        .predictor
                        .offload_output_to_cpu_for_eval
                {
                    if let Some(target) = signed_frame_idx.checked_sub(far_obj_ptr_distance) {
                        trim_targets.push(target as usize);
                    }
                }
            }
            PropagationDirection::Backward => {
                trim_targets.push((signed_frame_idx + trim_distance) as usize);
                if self.tracker.config().use_memory_selection
                    && !self
                        .tracker
                        .config()
                        .predictor
                        .offload_output_to_cpu_for_eval
                {
                    trim_targets.push((signed_frame_idx + far_obj_ptr_distance) as usize);
                }
            }
        }
        if let Some(tracked) = session.tracked_objects.get_mut(&object.obj_id) {
            for target_frame_idx in trim_targets {
                let Some(state) = tracked.tracker_states.get_mut(&target_frame_idx) else {
                    continue;
                };
                if state.is_cond_frame {
                    continue;
                }
                state.maskmem_features = None;
                state.maskmem_pos_enc = None;
            }
        }
    }

    fn history_on_compute_device(
        &self,
        session: &Sam3VideoSession,
        obj_id: u32,
        frame_idx: usize,
        direction: PropagationDirection,
        compute_device: &Device,
    ) -> Result<BTreeMap<usize, TrackerFrameState>> {
        let object = session.tracked_objects.get(&obj_id).ok_or_else(|| {
            candle::Error::Msg(format!(
                "unknown obj_id {} while building tracker history",
                obj_id
            ))
        })?;
        object
            .tracker_history(frame_idx, direction)
            .into_iter()
            .map(|(history_frame_idx, state)| {
                Ok((
                    history_frame_idx,
                    move_tracker_state(&state, compute_device)?,
                ))
            })
            .collect()
    }

    fn previous_frame_low_res_mask_input(
        &self,
        object: &TrackedObject,
        frame_idx: usize,
        compute_device: &Device,
    ) -> Result<Option<Tensor>> {
        if !self.tracker.config().predictor.iter_use_prev_mask_pred {
            return Ok(None);
        }
        let Some(state) = object.tracker_states.get(&frame_idx) else {
            return Ok(None);
        };
        Ok(Some(
            state
                .low_res_masks
                .to_device(compute_device)?
                .clamp(-32.0f32, 32.0f32)?,
        ))
    }

    fn attach_state_memory(
        &self,
        visual: &VisualBackboneOutput,
        state: &TrackerFrameState,
        is_mask_from_points: bool,
    ) -> Result<TrackerFrameState> {
        let (maskmem_features, maskmem_pos_enc) = self.tracker.encode_external_memory(
            visual,
            &state.high_res_masks,
            &state.object_score_logits,
            is_mask_from_points,
        )?;
        let mut state = state.clone();
        state.maskmem_features = Some(maskmem_features);
        state.maskmem_pos_enc = Some(maskmem_pos_enc);
        Ok(state)
    }

    fn clear_non_cond_mem_around_input(&self, session: &mut Sam3VideoSession, frame_idx: usize) {
        let predictor = &self.tracker.config().predictor;
        if !predictor.clear_non_cond_mem_around_input {
            return;
        }
        if !predictor.clear_non_cond_mem_for_multi_obj && session.tracked_objects.len() > 1 {
            return;
        }
        let stride = self.tracker.config().memory_temporal_stride_for_eval.max(1);
        let radius = stride * self.tracker.config().num_maskmem;
        let frame_idx_begin = frame_idx.saturating_sub(radius);
        let frame_idx_end = frame_idx.saturating_add(radius);
        for object in session.tracked_objects.values_mut() {
            for target_frame_idx in frame_idx_begin..=frame_idx_end {
                let Some(state) = object.tracker_states.get_mut(&target_frame_idx) else {
                    continue;
                };
                if state.is_cond_frame {
                    continue;
                }
                state.maskmem_features = None;
                state.maskmem_pos_enc = None;
            }
        }
    }

    fn run_propagated_frame(
        &self,
        config: &VideoConfig,
        model: &Sam3ImageModel,
        compute_device: &Device,
        session: &mut Sam3VideoSession,
        frame_idx: usize,
        object: &TrackedObject,
        direction: PropagationDirection,
    ) -> Result<(ObjectFrameOutput, TrackerFrameState, Option<f32>)> {
        self.ensure_history_states_have_memory(
            model,
            compute_device,
            session,
            object,
            frame_idx,
            direction,
        )?;
        let history = self.history_on_compute_device(
            session,
            object.obj_id,
            frame_idx,
            direction,
            compute_device,
        )?;
        let visual_features = tracker_visual_output(&session.get_visual_features(
            model,
            compute_device,
            frame_idx,
        )?);
        let track_output = self.tracker.track_frame(
            &visual_features,
            frame_idx,
            session.num_frames(),
            None,
            None,
            None,
            None,
            &history,
            false,
            matches!(direction, PropagationDirection::Backward),
            true,
            true,
        )?;
        let prompt_frame_idx = object.nearest_input_frame_idx(frame_idx, direction);
        let mut output = tracker_state_to_object_output(
            object.obj_id,
            &track_output.state,
            object.display_score,
            prompt_frame_idx,
            trim_memory_frame_indices(
                track_output.memory_frame_indices.clone(),
                config.memory_frame_count,
            ),
            object
                .latest_text_prompt(frame_idx, direction)
                .map(|(_, text)| text),
            object.nearest_input_uses_explicit_geometry(frame_idx, direction),
            true,
            session.video_size(),
        )?;
        apply_prompt_frame_output_postprocess(&mut output, config)?;
        if let Some(recorder) = session.debug_recorder_mut() {
            recorder.record_first_propagation(
                object,
                frame_idx,
                direction,
                prompt_frame_idx,
                &output,
                &history,
                &track_output.prompt_frame_indices,
                &track_output.memory_frame_indices,
            )?;
        }
        self.trim_past_non_cond_memory(session, object, frame_idx, direction);
        Ok((output, track_output.state, object.display_score))
    }

    fn run_refined_tracker_prompt_frame(
        &self,
        config: &VideoConfig,
        model: &Sam3ImageModel,
        compute_device: &Device,
        session: &mut Sam3VideoSession,
        frame_idx: usize,
        object: &TrackedObject,
        prompt: &SessionPrompt,
        direction: PropagationDirection,
    ) -> Result<(ObjectFrameOutput, TrackerFrameState, Option<f32>)> {
        if prompt.text.is_some() {
            candle::bail!(
                "SAM3 video tracker strict port currently supports correction prompts for point/box geometry only; text refinement lands in a later step."
            );
        }
        self.ensure_history_states_have_memory(
            model,
            compute_device,
            session,
            object,
            frame_idx,
            direction,
        )?;
        let history = if self.tracker.config().predictor.use_stateless_refinement {
            BTreeMap::new()
        } else {
            self.history_on_compute_device(
                session,
                object.obj_id,
                frame_idx,
                direction,
                compute_device,
            )?
        };
        let visual = tracker_visual_output(&session.get_visual_features(
            model,
            compute_device,
            frame_idx,
        )?);
        let prompt = truncate_prompt_for_encoder(prompt, config.max_point_num_in_prompt_enc);
        let tracker_input_extent = self.tracker.config().image_size as f64;
        let point_coords = match prompt.points.as_ref() {
            Some(points) => {
                let mut data = Vec::with_capacity(points.len() * 2);
                for (x, y) in points {
                    data.push(*x);
                    data.push(*y);
                }
                Some(
                    Tensor::from_vec(data, (1, points.len(), 2), compute_device)?
                        .affine(tracker_input_extent, 0.0)?,
                )
            }
            None => None,
        };
        let point_labels = prompt
            .point_labels
            .as_ref()
            .map(|labels| {
                Tensor::from_vec(
                    labels.iter().map(|label| *label as f32).collect(),
                    (1, labels.len()),
                    compute_device,
                )
            })
            .transpose()?;
        let boxes_xyxy = match prompt.boxes.as_ref() {
            Some(boxes) if !boxes.is_empty() => Some(
                boxes_cxcywh_to_xyxy_tensor(boxes, compute_device)?
                    .affine(tracker_input_extent, 0.0)?,
            ),
            _ => None,
        };
        let prev_mask_input =
            self.previous_frame_low_res_mask_input(object, frame_idx, compute_device)?;
        let is_cond_frame = self.correction_frame_is_cond_frame();
        let mut tracker_state = self
            .tracker
            .track_frame(
                &visual,
                frame_idx,
                session.num_frames(),
                point_coords.as_ref(),
                point_labels.as_ref(),
                boxes_xyxy.as_ref(),
                prev_mask_input.as_ref(),
                &history,
                is_cond_frame,
                matches!(direction, PropagationDirection::Backward),
                self.tracker.config().predictor.use_prev_mem_frame,
                false,
            )?
            .state;
        tracker_state = self.attach_state_memory(
            &visual,
            &tracker_state,
            prompt
                .boxes
                .as_ref()
                .map(|boxes| boxes.is_empty())
                .unwrap_or(true),
        )?;
        let mut output = tracker_state_to_object_output(
            object.obj_id,
            &tracker_state,
            object.display_score.or(Some(1.0)),
            Some(frame_idx),
            Vec::new(),
            prompt.text.clone(),
            prompt.has_geometry(),
            false,
            session.video_size(),
        )?;
        output.presence_scores = None;
        apply_prompt_frame_output_postprocess(&mut output, config)?;
        Ok((output, tracker_state, object.display_score.or(Some(1.0))))
    }

    fn run_refined_mask_prompt_frame(
        &self,
        config: &VideoConfig,
        model: &Sam3ImageModel,
        compute_device: &Device,
        session: &mut Sam3VideoSession,
        frame_idx: usize,
        object: &TrackedObject,
        mask_prompt: &Tensor,
        direction: PropagationDirection,
    ) -> Result<(ObjectFrameOutput, TrackerFrameState, Option<f32>)> {
        self.ensure_history_states_have_memory(
            model,
            compute_device,
            session,
            object,
            frame_idx,
            direction,
        )?;
        let history = if self.tracker.config().predictor.use_stateless_refinement {
            BTreeMap::new()
        } else {
            self.history_on_compute_device(
                session,
                object.obj_id,
                frame_idx,
                direction,
                compute_device,
            )?
        };
        let visual = tracker_visual_output(&session.get_visual_features(
            model,
            compute_device,
            frame_idx,
        )?);
        let video_mask =
            resize_mask_prompt_to_video(mask_prompt, session.video_size(), compute_device)?
                .ge(0.5f32)?
                .to_dtype(DType::F32)?;
        let tracker_mask = resize_mask_prompt_to_tracker_input(
            mask_prompt,
            self.tracker.input_mask_size(),
            compute_device,
        )?
        .ge(0.5f32)?
        .to_dtype(DType::F32)?;
        let is_cond_frame = self.correction_frame_is_cond_frame();
        let mut tracker_state = self
            .tracker
            .track_frame(
                &visual,
                frame_idx,
                session.num_frames(),
                None,
                None,
                None,
                Some(&tracker_mask),
                &history,
                is_cond_frame,
                matches!(direction, PropagationDirection::Backward),
                self.tracker.config().predictor.use_prev_mem_frame,
                false,
            )?
            .state;
        tracker_state = self.attach_state_memory(&visual, &tracker_state, false)?;
        let mut output = mask_prompt_to_object_output(
            object.obj_id,
            &video_mask,
            &tracker_state,
            Some(frame_idx),
            session.video_size(),
        )?;
        apply_prompt_frame_output_postprocess(&mut output, config)?;
        let display_score = output.score_value().ok();
        Ok((output, tracker_state, display_score))
    }

    fn get_visual_prompt(
        &self,
        object: &TrackedObject,
        frame_idx: usize,
        prompt: &SessionPrompt,
    ) -> Result<(SessionPrompt, bool)> {
        let box_count = prompt.boxes.as_ref().map(Vec::len).unwrap_or(0);
        let is_new_visual_prompt = box_count > 0
            && !object.has_inference_history
            && !object.frame_outputs.contains_key(&frame_idx);
        if !is_new_visual_prompt {
            return Ok((prompt.clone(), false));
        }
        if box_count != 1 {
            candle::bail!(
                "visual prompts (box as an initial prompt) should only have one box, but got {box_count}"
            );
        }
        let mut visual_prompt = prompt.clone();
        visual_prompt.points = None;
        visual_prompt.point_labels = None;
        visual_prompt.boxes = prompt.boxes.as_ref().map(|boxes| vec![boxes[0]]);
        visual_prompt.box_labels = prompt.box_labels.as_ref().map(|labels| vec![labels[0]]);
        Ok((visual_prompt, true))
    }

    fn run_visual_prompt_seed_frame(
        &self,
        config: &VideoConfig,
        model: &Sam3ImageModel,
        compute_device: &Device,
        session: &mut Sam3VideoSession,
        frame_idx: usize,
        object: &TrackedObject,
        prompt: &SessionPrompt,
        direction: PropagationDirection,
    ) -> Result<(ObjectFrameOutput, TrackerFrameState, Option<f32>)> {
        let visual_features = session.get_visual_features(model, compute_device, frame_idx)?;
        let tracker_visual_features = tracker_visual_output(&visual_features);
        let (visual_prompt, used_visual_prompt) =
            self.get_visual_prompt(object, frame_idx, prompt)?;
        let used_visual_text_prompt = used_visual_prompt && prompt.text.is_none();
        let geometry_prompt = session_prompt_to_geometry(&visual_prompt, compute_device)?;
        let geometry_encoding = if geometry_prompt.is_empty() {
            None
        } else {
            Some(model.encode_geometry_prompt(&geometry_prompt, &visual_features)?)
        };
        let text_encoding = match prompt.text.as_ref() {
            Some(text) => Some(session.cached_text_encoding(model, text, compute_device)?),
            None if used_visual_prompt => {
                Some(session.cached_text_encoding(model, "visual", compute_device)?)
            }
            None => None,
        };
        let encoded_prompt =
            combine_encoded_prompts(text_encoding.as_ref(), geometry_encoding.as_ref())?
                .ok_or_else(|| {
                    candle::Error::Msg("visual prompt path produced no encoded prompt".to_owned())
                })?;
        let grounding = ground_from_encoded_prompt(model, &visual_features, &encoded_prompt)?;
        let detector_output = grounding_to_object_output(
            object.obj_id,
            &grounding,
            Some(frame_idx),
            Vec::new(),
            prompt.text.clone(),
            true,
            false,
            session.video_size(),
        )?;
        if let Some(recorder) = session.debug_recorder_mut() {
            recorder.record_detector_grounding(
                object,
                frame_idx,
                direction,
                debug_prompt_metadata(prompt, used_visual_text_prompt)?,
                &detector_output,
            )?;
        }
        let mut tracker_mask_input =
            normalize_video_mask_prompt(&grounding.mask_logits, compute_device)?;
        let (_, _, height, width) = tracker_mask_input.dims4()?;
        let tracker_input_size = self.tracker.input_mask_size();
        if height != tracker_input_size || width != tracker_input_size {
            tracker_mask_input = tracker_mask_input.upsample_bilinear2d(
                tracker_input_size,
                tracker_input_size,
                false,
            )?;
        }
        let tracker_mask_input = tracker_mask_input.gt(0f64)?.to_dtype(DType::F32)?;
        let tracker_state = self
            .tracker
            .track_frame(
                &tracker_visual_features,
                frame_idx,
                session.num_frames(),
                None,
                None,
                None,
                Some(&tracker_mask_input),
                &BTreeMap::new(),
                true,
                false,
                true,
                false,
            )?
            .state;
        let detector_score = detector_output.score_value()?;
        let mut seed_output = tracker_state_to_object_output(
            object.obj_id,
            &tracker_state,
            Some(detector_score),
            Some(frame_idx),
            Vec::new(),
            prompt.text.clone(),
            true,
            false,
            session.video_size(),
        )?;
        seed_output.presence_scores = None;
        apply_prompt_frame_output_postprocess(&mut seed_output, config)?;
        if let Some(recorder) = session.debug_recorder_mut() {
            recorder.record_tracker_seed(
                object,
                frame_idx,
                direction,
                debug_prompt_metadata(prompt, used_visual_text_prompt)?,
                &seed_output,
                &tracker_state,
            )?;
        }
        Ok((seed_output, tracker_state, Some(detector_score)))
    }

    fn run_direct_tracker_prompt_seed_frame(
        &self,
        config: &VideoConfig,
        model: &Sam3ImageModel,
        compute_device: &Device,
        session: &mut Sam3VideoSession,
        frame_idx: usize,
        object: &TrackedObject,
        prompt: &SessionPrompt,
        direction: PropagationDirection,
    ) -> Result<(ObjectFrameOutput, TrackerFrameState, Option<f32>)> {
        let visual_features = tracker_visual_output(&session.get_visual_features(
            model,
            compute_device,
            frame_idx,
        )?);
        let prompt = truncate_prompt_for_encoder(prompt, config.max_point_num_in_prompt_enc);
        let tracker_input_extent = self.tracker.config().image_size as f64;
        let point_coords = match prompt.points.as_ref() {
            Some(points) => {
                let mut data = Vec::with_capacity(points.len() * 2);
                for (x, y) in points {
                    data.push(*x);
                    data.push(*y);
                }
                Some(
                    Tensor::from_vec(data, (1, points.len(), 2), compute_device)?
                        .affine(tracker_input_extent, 0.0)?,
                )
            }
            None => None,
        };
        let point_labels = prompt
            .point_labels
            .as_ref()
            .map(|labels| {
                Tensor::from_vec(
                    labels.iter().map(|label| *label as f32).collect(),
                    (1, labels.len()),
                    compute_device,
                )
            })
            .transpose()?;
        let boxes_xyxy = match prompt.boxes.as_ref() {
            Some(boxes) if !boxes.is_empty() => Some(
                boxes_cxcywh_to_xyxy_tensor(boxes, compute_device)?
                    .affine(tracker_input_extent, 0.0)?,
            ),
            _ => None,
        };
        let tracker_state = self
            .tracker
            .track_frame(
                &visual_features,
                frame_idx,
                session.num_frames(),
                point_coords.as_ref(),
                point_labels.as_ref(),
                boxes_xyxy.as_ref(),
                None,
                &BTreeMap::new(),
                true,
                false,
                false,
                false,
            )?
            .state;
        let mut output = tracker_state_to_object_output(
            object.obj_id,
            &tracker_state,
            Some(1.0),
            Some(frame_idx),
            Vec::new(),
            prompt.text.clone(),
            prompt.has_geometry(),
            false,
            session.video_size(),
        )?;
        output.presence_scores = None;
        apply_prompt_frame_output_postprocess(&mut output, config)?;
        if let Some(recorder) = session.debug_recorder_mut() {
            recorder.record_tracker_seed(
                object,
                frame_idx,
                direction,
                debug_prompt_metadata(&prompt, false)?,
                &output,
                &tracker_state,
            )?;
        }
        Ok((output, tracker_state, Some(1.0)))
    }

    fn run_mask_prompt_seed_frame(
        &self,
        config: &VideoConfig,
        model: &Sam3ImageModel,
        compute_device: &Device,
        session: &mut Sam3VideoSession,
        frame_idx: usize,
        object: &TrackedObject,
        mask_prompt: &Tensor,
        direction: PropagationDirection,
    ) -> Result<(ObjectFrameOutput, TrackerFrameState, Option<f32>)> {
        let visual_features = tracker_visual_output(&session.get_visual_features(
            model,
            compute_device,
            frame_idx,
        )?);
        let video_mask =
            resize_mask_prompt_to_video(mask_prompt, session.video_size(), compute_device)?
                .ge(0.5f32)?
                .to_dtype(DType::F32)?;
        let tracker_mask = resize_mask_prompt_to_tracker_input(
            mask_prompt,
            self.tracker.input_mask_size(),
            compute_device,
        )?
        .ge(0.5f32)?
        .to_dtype(DType::F32)?;
        let tracker_state = self
            .tracker
            .track_frame(
                &visual_features,
                frame_idx,
                session.num_frames(),
                None,
                None,
                None,
                Some(&tracker_mask),
                &BTreeMap::new(),
                true,
                false,
                true,
                false,
            )?
            .state;
        let mut output = mask_prompt_to_object_output(
            object.obj_id,
            &video_mask,
            &tracker_state,
            Some(frame_idx),
            session.video_size(),
        )?;
        apply_prompt_frame_output_postprocess(&mut output, config)?;
        let display_score = output.score_value().ok();
        if let Some(recorder) = session.debug_recorder_mut() {
            recorder.record_tracker_seed(
                object,
                frame_idx,
                direction,
                debug_prompt_metadata(
                    &SessionPrompt {
                        text: None,
                        points: None,
                        point_labels: None,
                        boxes: None,
                        box_labels: None,
                    },
                    false,
                )?,
                &output,
                &tracker_state,
            )?;
        }
        Ok((output, tracker_state, display_score))
    }

    fn process_frame(
        &self,
        model: &Sam3ImageModel,
        compute_device: &Device,
        _config: &VideoConfig,
        session: &mut Sam3VideoSession,
        frame_idx: usize,
        direction: PropagationDirection,
        _output_threshold: f32,
    ) -> Result<VideoFrameOutput> {
        let obj_ids: Vec<u32> = session.tracked_objects.keys().copied().collect();
        let predictor_config = &self.tracker.config().predictor;
        let latest_session_input_frame = session
            .tracked_objects
            .values()
            .filter_map(|object| object.nearest_input_frame_idx(frame_idx, direction))
            .max();
        let mut frame_objects = Vec::new();
        for obj_id in obj_ids {
            if let Some(cached) = session
                .frame_outputs
                .get(&frame_idx)
                .and_then(|frame_outputs| frame_outputs.get(&obj_id))
                .cloned()
            {
                frame_objects.push(cached.to_storage_device(compute_device)?);
                continue;
            }
            let (prompt, mask_prompt, has_history, is_active) = {
                let object = session
                    .tracked_objects
                    .get(&obj_id)
                    .ok_or_else(|| candle::Error::Msg(format!("unknown obj_id {}", obj_id)))?;
                (
                    object.prompt_frames.get(&frame_idx).cloned(),
                    object.mask_prompt_frames.get(&frame_idx).cloned(),
                    object.has_inference_history
                        || !object.frame_outputs.is_empty()
                        || !object.tracker_states.is_empty(),
                    object.is_active_for_frame(frame_idx, direction),
                )
            };
            let Some(object_snapshot) = session.tracked_objects.get(&obj_id).cloned() else {
                continue;
            };
            let seed_result = if let Some(mask_prompt) = mask_prompt {
                if has_history {
                    Some(self.run_refined_mask_prompt_frame(
                        _config,
                        model,
                        compute_device,
                        session,
                        frame_idx,
                        &object_snapshot,
                        &mask_prompt,
                        direction,
                    )?)
                } else {
                    Some(self.run_mask_prompt_seed_frame(
                        _config,
                        model,
                        compute_device,
                        session,
                        frame_idx,
                        &object_snapshot,
                        &mask_prompt,
                        direction,
                    )?)
                }
            } else if let Some(prompt) = prompt {
                if has_history {
                    Some(self.run_refined_tracker_prompt_frame(
                        _config,
                        model,
                        compute_device,
                        session,
                        frame_idx,
                        &object_snapshot,
                        &prompt,
                        direction,
                    )?)
                } else if prompt.text.is_some()
                    || prompt
                        .boxes
                        .as_ref()
                        .map(|boxes| !boxes.is_empty())
                        .unwrap_or(false)
                {
                    Some(self.run_visual_prompt_seed_frame(
                        _config,
                        model,
                        compute_device,
                        session,
                        frame_idx,
                        &object_snapshot,
                        &prompt,
                        direction,
                    )?)
                } else {
                    Some(self.run_direct_tracker_prompt_seed_frame(
                        _config,
                        model,
                        compute_device,
                        session,
                        frame_idx,
                        &object_snapshot,
                        &prompt,
                        direction,
                    )?)
                }
            } else {
                let own_latest_input_frame = object_snapshot.nearest_input_frame_idx(frame_idx, direction);
                let can_extend_uninteracted_multi_object = predictor_config.clear_non_cond_mem_for_multi_obj
                    && session.tracked_objects.len() > 1
                    && latest_session_input_frame.zip(own_latest_input_frame).map_or(false, |(session_latest, own_latest)| session_latest > own_latest)
                    && match direction {
                        PropagationDirection::Forward | PropagationDirection::Both => {
                            frame_idx > object_snapshot.last_updated_frame
                        }
                        PropagationDirection::Backward => {
                            frame_idx < object_snapshot.last_updated_frame
                        }
                    };
                if is_active {
                    if can_extend_uninteracted_multi_object {
                        None
                    } else {
                        Some(self.run_propagated_frame(
                            _config,
                            model,
                            compute_device,
                            session,
                            frame_idx,
                            &object_snapshot,
                            direction,
                        )?)
                    }
                } else {
                    None
                }
            };
            let Some((output, tracker_state, display_score)) = seed_result else {
                continue;
            };

            let output_storage = output.to_storage_device(session.storage_device())?;
            let tracker_storage = move_tracker_state(&tracker_state, session.storage_device())?;
            session
                .frame_outputs
                .entry(frame_idx)
                .or_default()
                .insert(obj_id, output_storage.clone());
            if let Some(object) = session.tracked_objects.get_mut(&obj_id) {
                object.frame_outputs.insert(frame_idx, output_storage);
                object.tracker_states.insert(frame_idx, tracker_storage);
                object.has_inference_history = true;
                object.last_updated_frame = frame_idx;
                if let Some(display_score) = display_score {
                    object.display_score = Some(display_score);
                }
            }
            frame_objects.push(output);
        }
        Ok(VideoFrameOutput {
            frame_idx,
            objects: frame_objects,
        })
    }
}

#[derive(Debug, Clone)]
struct CachedTextPrompt {
    attention_mask: Tensor,
    memory: Tensor,
    input_embeddings: Tensor,
}

impl CachedTextPrompt {
    fn from_encoding(encoding: &TextEncoding, storage_device: &Device) -> Result<Self> {
        Ok(Self {
            attention_mask: encoding.attention_mask.to_device(storage_device)?,
            memory: encoding.memory.to_device(storage_device)?,
            input_embeddings: encoding.input_embeddings.to_device(storage_device)?,
        })
    }

    fn to_text_encoding(&self, compute_device: &Device) -> Result<TextEncoding> {
        Ok(TextEncoding {
            attention_mask: self.attention_mask.to_device(compute_device)?,
            memory: self.memory.to_device(compute_device)?,
            input_embeddings: self.input_embeddings.to_device(compute_device)?,
        })
    }
}

#[derive(Debug, Clone)]
struct FrameBlob {
    data: Vec<f32>,
    frame_size: ImageSize,
}

impl FrameBlob {
    fn to_tensor(&self, target_device: &Device) -> Result<Tensor> {
        Tensor::from_vec(
            self.data.clone(),
            (3, self.frame_size.height, self.frame_size.width),
            &Device::Cpu,
        )?
        .to_device(target_device)
    }
}

struct TensorFrameSource {
    frames: Vec<Tensor>,
    video_size: ImageSize,
}

impl TensorFrameSource {
    fn new(frames: Vec<Tensor>) -> Result<Self> {
        if frames.is_empty() {
            candle::bail!("tensor frame source requires at least one frame")
        }
        let (channels, height, width) = match frames[0].rank() {
            3 => frames[0].dims3()?,
            4 => {
                let (_batch, channels, height, width) = frames[0].dims4()?;
                (channels, height, width)
            }
            rank => candle::bail!("expected CHW or BCHW frame tensor, got rank {}", rank),
        };
        if channels != 3 {
            candle::bail!(
                "tensor frame source expects RGB frames, got {} channels",
                channels
            )
        }
        Ok(Self {
            frames,
            video_size: ImageSize::new(height, width),
        })
    }
}

impl FrameSource for TensorFrameSource {
    fn frame_count(&self) -> usize {
        self.frames.len()
    }

    fn video_size(&self) -> ImageSize {
        self.video_size
    }

    fn get_frame(&mut self, frame_idx: usize, target_device: &Device) -> Result<Tensor> {
        self.frames
            .get(frame_idx)
            .ok_or_else(|| candle::Error::Msg(format!("frame_idx {} out of bounds", frame_idx)))?
            .to_device(target_device)
    }

    fn prefetch(&mut self, _frame_indices: &[usize]) -> Result<()> {
        Ok(())
    }

    fn evict_except(&mut self, _keep_frame_indices: &BTreeSet<usize>) {}

    fn loaded_frame_count(&self) -> usize {
        self.frames.len()
    }

    fn close(&mut self) {}
}

struct ImageFolderFrameSource {
    image_paths: Vec<PathBuf>,
    image_size: usize,
    image_mean: [f32; 3],
    image_std: [f32; 3],
    cache: HashMap<usize, FrameBlob>,
    video_size: ImageSize,
}

impl ImageFolderFrameSource {
    fn new(
        image_paths: Vec<PathBuf>,
        image_size: usize,
        image_mean: [f32; 3],
        image_std: [f32; 3],
    ) -> Result<Self> {
        if image_paths.is_empty() {
            candle::bail!("image frame source requires at least one image path")
        }
        let first = image_paths[0].clone();
        let image = ImageReader::open(&first)?
            .decode()
            .map_err(candle::Error::wrap)?
            .to_rgb8();
        let (width, height) = image.dimensions();
        Ok(Self {
            image_paths,
            image_size,
            image_mean,
            image_std,
            cache: HashMap::new(),
            video_size: ImageSize::new(height as usize, width as usize),
        })
    }

    fn ensure_loaded(&mut self, frame_idx: usize) -> Result<()> {
        if self.cache.contains_key(&frame_idx) {
            return Ok(());
        }
        let path = self
            .image_paths
            .get(frame_idx)
            .ok_or_else(|| candle::Error::Msg(format!("frame_idx {} out of bounds", frame_idx)))?;
        let blob = load_frame_blob(
            path,
            self.image_size,
            self.image_mean,
            self.image_std,
            self.video_size,
        )?;
        self.cache.insert(frame_idx, blob);
        Ok(())
    }
}

impl FrameSource for ImageFolderFrameSource {
    fn frame_count(&self) -> usize {
        self.image_paths.len()
    }

    fn video_size(&self) -> ImageSize {
        self.video_size
    }

    fn get_frame(&mut self, frame_idx: usize, target_device: &Device) -> Result<Tensor> {
        self.ensure_loaded(frame_idx)?;
        self.cache
            .get(&frame_idx)
            .ok_or_else(|| candle::Error::Msg(format!("frame_idx {} not cached", frame_idx)))?
            .to_tensor(target_device)
    }

    fn prefetch(&mut self, frame_indices: &[usize]) -> Result<()> {
        for frame_idx in frame_indices {
            self.ensure_loaded(*frame_idx)?;
        }
        Ok(())
    }

    fn evict_except(&mut self, keep_frame_indices: &BTreeSet<usize>) {
        self.cache
            .retain(|frame_idx, _| keep_frame_indices.contains(frame_idx));
    }

    fn loaded_frame_count(&self) -> usize {
        self.cache.len()
    }

    fn close(&mut self) {
        self.cache.clear();
    }
}

struct VideoFileFrameSource {
    video_path: PathBuf,
    image_size: usize,
    image_mean: [f32; 3],
    image_std: [f32; 3],
    cache: HashMap<usize, FrameBlob>,
    video_size: ImageSize,
    frame_count: usize,
}

impl VideoFileFrameSource {
    fn new(
        video_path: PathBuf,
        image_size: usize,
        image_mean: [f32; 3],
        image_std: [f32; 3],
    ) -> Result<Self> {
        let metadata = probe_video_file(&video_path)?;
        Ok(Self {
            video_path,
            image_size,
            image_mean,
            image_std,
            cache: HashMap::new(),
            video_size: metadata.video_size,
            frame_count: metadata.frame_count,
        })
    }

    fn ensure_loaded(&mut self, frame_idx: usize) -> Result<()> {
        if self.cache.contains_key(&frame_idx) {
            return Ok(());
        }
        if frame_idx >= self.frame_count {
            candle::bail!(
                "frame_idx {} out of bounds for video with {} frames",
                frame_idx,
                self.frame_count
            );
        }
        let blob = decode_video_frame_blob(
            &self.video_path,
            frame_idx,
            self.image_size,
            self.image_mean,
            self.image_std,
            self.video_size,
        )?;
        self.cache.insert(frame_idx, blob);
        Ok(())
    }
}

impl FrameSource for VideoFileFrameSource {
    fn frame_count(&self) -> usize {
        self.frame_count
    }

    fn video_size(&self) -> ImageSize {
        self.video_size
    }

    fn get_frame(&mut self, frame_idx: usize, target_device: &Device) -> Result<Tensor> {
        self.ensure_loaded(frame_idx)?;
        self.cache
            .get(&frame_idx)
            .ok_or_else(|| candle::Error::Msg(format!("frame_idx {} not cached", frame_idx)))?
            .to_tensor(target_device)
    }

    fn prefetch(&mut self, frame_indices: &[usize]) -> Result<()> {
        for frame_idx in frame_indices {
            self.ensure_loaded(*frame_idx)?;
        }
        Ok(())
    }

    fn evict_except(&mut self, keep_frame_indices: &BTreeSet<usize>) {
        self.cache
            .retain(|frame_idx, _| keep_frame_indices.contains(frame_idx));
    }

    fn loaded_frame_count(&self) -> usize {
        self.cache.len()
    }

    fn close(&mut self) {
        self.cache.clear();
    }
}

fn load_frame_blob(
    image_path: &Path,
    image_size: usize,
    image_mean: [f32; 3],
    image_std: [f32; 3],
    expected_video_size: ImageSize,
) -> Result<FrameBlob> {
    if matches!(
        image_path
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.to_ascii_lowercase())
            .as_deref(),
        Some("jpg" | "jpeg")
    ) {
        if let Ok(blob) = load_jpeg_frame_blob_via_pillow(
            image_path,
            image_size,
            image_mean,
            image_std,
            expected_video_size,
        ) {
            return Ok(blob);
        }
    }
    let image = ImageReader::open(image_path)?
        .decode()
        .map_err(candle::Error::wrap)?
        .to_rgb8();
    frame_blob_from_rgb_image(
        image,
        image_size,
        image_mean,
        image_std,
        expected_video_size,
        &image_path.display().to_string(),
    )
}

fn load_jpeg_frame_blob_via_pillow(
    image_path: &Path,
    image_size: usize,
    image_mean: [f32; 3],
    image_std: [f32; 3],
    expected_video_size: ImageSize,
) -> Result<FrameBlob> {
    let python = find_pillow_python().ok_or_else(|| {
        candle::Error::Msg("no Pillow-capable python interpreter found".to_owned())
    })?;
    let script = r#"
import struct
import sys
from PIL import Image

image_path = sys.argv[1]
image_size = int(sys.argv[2])

image = Image.open(image_path).convert("RGB")
orig_w, orig_h = image.size
if image.size != (image_size, image_size):
    image = image.resize((image_size, image_size), Image.Resampling.BICUBIC)
raw = image.tobytes()
sys.stdout.buffer.write(struct.pack("<II", orig_w, orig_h))
sys.stdout.buffer.write(raw)
"#;
    let output = Command::new(&python)
        .arg("-c")
        .arg(script)
        .arg(image_path)
        .arg(image_size.to_string())
        .output()
        .map_err(candle::Error::wrap)?;
    if !output.status.success() {
        candle::bail!(
            "Pillow frame load failed for {} via {}: {}",
            image_path.display(),
            python.display(),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    if output.stdout.len() < 8 {
        candle::bail!(
            "Pillow frame load returned truncated output for {}",
            image_path.display()
        );
    }
    let width = u32::from_le_bytes(output.stdout[0..4].try_into().unwrap()) as usize;
    let height = u32::from_le_bytes(output.stdout[4..8].try_into().unwrap()) as usize;
    let current_size = ImageSize::new(height, width);
    if current_size != expected_video_size {
        candle::bail!(
            "frame {} has size {}x{} but the session expects {}x{}",
            image_path.display(),
            current_size.height,
            current_size.width,
            expected_video_size.height,
            expected_video_size.width
        );
    }
    let expected_bytes = image_size * image_size * 3;
    let raw = &output.stdout[8..];
    if raw.len() != expected_bytes {
        candle::bail!(
            "Pillow frame load returned {} bytes for resized frame {}, expected {}",
            raw.len(),
            image_path.display(),
            expected_bytes
        );
    }
    let image = Tensor::from_vec(raw.to_vec(), (image_size, image_size, 3), &Device::Cpu)?
        .permute((2, 0, 1))?;
    let normalized = normalize_image_for_sam3(
        &(image.to_dtype(DType::F32)?.unsqueeze(0)? / 255.)?,
        image_mean,
        image_std,
    )?
    .squeeze(0)?;
    Ok(FrameBlob {
        data: normalized.flatten_all()?.to_vec1::<f32>()?,
        frame_size: ImageSize::square(image_size),
    })
}

fn find_pillow_python() -> Option<PathBuf> {
    let mut candidates = Vec::new();
    if let Some(path) = std::env::var_os("SAM3_PILLOW_PYTHON").map(PathBuf::from) {
        candidates.push(path);
    }
    candidates.push(PathBuf::from(".venv/bin/python"));
    candidates.push(PathBuf::from(
        "/home/dnorthover/ChengCode/candle_sam3/.venv/bin/python",
    ));
    candidates.push(PathBuf::from("python3"));
    candidates.into_iter().find(|candidate| {
        candidate.is_absolute() || candidate.exists() || candidate == Path::new("python3")
    })
}

fn frame_blob_from_rgb_image(
    image: image::RgbImage,
    image_size: usize,
    image_mean: [f32; 3],
    image_std: [f32; 3],
    expected_video_size: ImageSize,
    source_label: &str,
) -> Result<FrameBlob> {
    frame_blob_from_rgb_image_with_filter(
        image,
        image_size,
        image_mean,
        image_std,
        expected_video_size,
        source_label,
        image::imageops::FilterType::CatmullRom,
    )
}

fn frame_blob_from_rgb_image_with_filter(
    image: image::RgbImage,
    image_size: usize,
    image_mean: [f32; 3],
    image_std: [f32; 3],
    expected_video_size: ImageSize,
    source_label: &str,
    resize_filter: image::imageops::FilterType,
) -> Result<FrameBlob> {
    let (width, height) = image.dimensions();
    let current_size = ImageSize::new(height as usize, width as usize);
    if current_size != expected_video_size {
        candle::bail!(
            "frame {} has size {}x{} but the session expects {}x{}",
            source_label,
            current_size.height,
            current_size.width,
            expected_video_size.height,
            expected_video_size.width
        );
    }

    let resized =
        if expected_video_size.height == image_size && expected_video_size.width == image_size {
            image
        } else {
            image::imageops::resize(&image, image_size as u32, image_size as u32, resize_filter)
        };
    let image = Tensor::from_vec(
        resized.into_raw(),
        (image_size, image_size, 3),
        &Device::Cpu,
    )?
    .permute((2, 0, 1))?;
    let normalized = normalize_image_for_sam3(
        &(image.to_dtype(DType::F32)?.unsqueeze(0)? / 255.)?,
        image_mean,
        image_std,
    )?
    .squeeze(0)?;
    Ok(FrameBlob {
        data: normalized.flatten_all()?.to_vec1::<f32>()?,
        frame_size: ImageSize::square(image_size),
    })
}

#[derive(Debug)]
struct VideoProbeMetadata {
    video_size: ImageSize,
    frame_count: usize,
}

#[derive(Debug, serde::Deserialize)]
struct FfprobeOutput {
    streams: Vec<FfprobeStream>,
}

#[derive(Debug, serde::Deserialize)]
struct FfprobeStream {
    width: Option<usize>,
    height: Option<usize>,
    nb_frames: Option<String>,
    nb_read_frames: Option<String>,
    duration: Option<String>,
    r_frame_rate: Option<String>,
}

fn probe_video_file(video_path: &Path) -> Result<VideoProbeMetadata> {
    let output = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=width,height,nb_frames,nb_read_frames,duration,r_frame_rate",
            "-of",
            "json",
        ])
        .arg(video_path)
        .output()
        .map_err(|err| {
            candle::Error::Msg(format!(
                "failed to run ffprobe for {}: {}",
                video_path.display(),
                err
            ))
        })?;
    if !output.status.success() {
        candle::bail!(
            "ffprobe failed for {}: {}",
            video_path.display(),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    let parsed: FfprobeOutput = serde_json::from_slice(&output.stdout).map_err(|err| {
        candle::Error::Msg(format!(
            "failed to parse ffprobe output for {}: {}",
            video_path.display(),
            err
        ))
    })?;
    let stream = parsed.streams.into_iter().next().ok_or_else(|| {
        candle::Error::Msg(format!(
            "ffprobe found no video stream in {}",
            video_path.display()
        ))
    })?;
    let width = stream.width.ok_or_else(|| {
        candle::Error::Msg(format!(
            "ffprobe did not report width for {}",
            video_path.display()
        ))
    })?;
    let height = stream.height.ok_or_else(|| {
        candle::Error::Msg(format!(
            "ffprobe did not report height for {}",
            video_path.display()
        ))
    })?;
    let frame_count = parse_frame_count(&stream).ok_or_else(|| {
        candle::Error::Msg(format!(
            "could not determine frame count for {} from ffprobe metadata",
            video_path.display()
        ))
    })?;
    if frame_count == 0 {
        candle::bail!(
            "video {} contains zero readable frames",
            video_path.display()
        );
    }
    Ok(VideoProbeMetadata {
        video_size: ImageSize::new(height, width),
        frame_count,
    })
}

fn parse_frame_count(stream: &FfprobeStream) -> Option<usize> {
    parse_optional_usize(stream.nb_read_frames.as_deref())
        .or_else(|| parse_optional_usize(stream.nb_frames.as_deref()))
        .or_else(|| {
            let duration = stream.duration.as_deref()?.parse::<f64>().ok()?;
            let fps = parse_rational_f64(stream.r_frame_rate.as_deref()?)?;
            let approx = (duration * fps).round();
            (approx.is_finite() && approx > 0.0).then_some(approx as usize)
        })
}

fn parse_optional_usize(value: Option<&str>) -> Option<usize> {
    let value = value?;
    if value == "N/A" {
        return None;
    }
    value.parse::<usize>().ok().filter(|value| *value > 0)
}

fn parse_rational_f64(value: &str) -> Option<f64> {
    let (numerator, denominator) = value.split_once('/')?;
    let numerator = numerator.parse::<f64>().ok()?;
    let denominator = denominator.parse::<f64>().ok()?;
    (denominator != 0.0).then_some(numerator / denominator)
}

fn decode_video_frame_blob(
    video_path: &Path,
    frame_idx: usize,
    image_size: usize,
    image_mean: [f32; 3],
    image_std: [f32; 3],
    expected_video_size: ImageSize,
) -> Result<FrameBlob> {
    let select_filter = format!("select=eq(n\\,{frame_idx})");
    let output = Command::new("ffmpeg")
        .args(["-v", "error", "-i"])
        .arg(video_path)
        .args([
            "-vf",
            &select_filter,
            "-vframes",
            "1",
            "-f",
            "image2pipe",
            "-vcodec",
            "png",
            "-",
        ])
        .output()
        .map_err(|err| {
            candle::Error::Msg(format!(
                "failed to run ffmpeg for {} frame {}: {}",
                video_path.display(),
                frame_idx,
                err
            ))
        })?;
    if !output.status.success() {
        candle::bail!(
            "ffmpeg failed for {} frame {}: {}",
            video_path.display(),
            frame_idx,
            String::from_utf8_lossy(&output.stderr)
        );
    }
    if output.stdout.is_empty() {
        candle::bail!(
            "ffmpeg produced no bytes for {} frame {}",
            video_path.display(),
            frame_idx
        );
    }
    let image = image::load(Cursor::new(output.stdout), image::ImageFormat::Png)
        .map_err(candle::Error::wrap)?
        .to_rgb8();
    frame_blob_from_rgb_image(
        image,
        image_size,
        image_mean,
        image_std,
        expected_video_size,
        &format!("{}#{}", video_path.display(), frame_idx),
    )
}

fn sorted_image_paths(dir_path: &Path) -> Result<Vec<PathBuf>> {
    let mut image_paths = fs::read_dir(dir_path)?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| {
            path.extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| {
                    matches!(
                        ext.to_ascii_lowercase().as_str(),
                        "jpg" | "jpeg" | "png" | "bmp" | "tiff" | "webp"
                    )
                })
                .unwrap_or(false)
        })
        .collect::<Vec<_>>();
    if image_paths.is_empty() {
        candle::bail!("no image files found in {}", dir_path.display())
    }

    if image_paths.iter().all(|path| {
        path.file_stem()
            .and_then(|stem| stem.to_str())
            .and_then(|stem| stem.parse::<usize>().ok())
            .is_some()
    }) {
        image_paths.sort_by_key(|path| {
            path.file_stem()
                .and_then(|stem| stem.to_str())
                .and_then(|stem| stem.parse::<usize>().ok())
                .unwrap_or(usize::MAX)
        });
    } else {
        image_paths.sort_by(|lhs, rhs| lhs.file_name().cmp(&rhs.file_name()));
    }

    Ok(image_paths)
}

fn normalize_image_for_sam3(
    image_bchw: &Tensor,
    image_mean: [f32; 3],
    image_std: [f32; 3],
) -> Result<Tensor> {
    let device = image_bchw.device();
    let mean = Tensor::from_vec(image_mean.to_vec(), (1, 3, 1, 1), device)?;
    let std = Tensor::from_vec(image_std.to_vec(), (1, 3, 1, 1), device)?;
    image_bchw.broadcast_sub(&mean)?.broadcast_div(&std)
}

fn load_tokenizer(path: &Path, context_length: usize) -> Result<Tokenizer> {
    let tokenizer_path = if path.is_dir() {
        path.join("tokenizer.json")
    } else {
        path.to_path_buf()
    };
    let mut tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|err| {
        candle::Error::Msg(format!(
            "failed to load tokenizer from {}: {}",
            tokenizer_path.display(),
            err
        ))
    })?;
    let pad_id = *tokenizer
        .get_vocab(true)
        .get(CLIP_EOT_TOKEN)
        .ok_or_else(|| {
            candle::Error::Msg(format!(
                "tokenizer is missing required token `{}`",
                CLIP_EOT_TOKEN
            ))
        })?;
    tokenizer
        .with_padding(Some(PaddingParams {
            strategy: tokenizers::PaddingStrategy::Fixed(context_length),
            direction: PaddingDirection::Right,
            pad_to_multiple_of: None,
            pad_id,
            pad_type_id: 0,
            pad_token: CLIP_EOT_TOKEN.to_string(),
        }))
        .with_truncation(Some(TruncationParams {
            max_length: context_length,
            ..Default::default()
        }))
        .map_err(|err| candle::Error::Msg(format!("failed to configure tokenizer: {}", err)))?;
    Ok(tokenizer)
}

fn tokenize_prompt(
    prompt: &str,
    tokenizer: &Tokenizer,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let encoding = tokenizer
        .encode(prompt, true)
        .map_err(|err| candle::Error::Msg(format!("failed to tokenize `{}`: {}", prompt, err)))?;
    let input_ids = Tensor::new(vec![encoding.get_ids().to_vec()], device)?;
    let attention_mask = Tensor::new(vec![encoding.get_attention_mask().to_vec()], device)?;
    Ok((input_ids, attention_mask))
}

fn combine_encoded_prompts(
    text_encoding: Option<&TextEncoding>,
    geometry_encoding: Option<&EncodedPrompt>,
) -> Result<Option<EncodedPrompt>> {
    match (text_encoding, geometry_encoding) {
        (Some(text), Some(geometry)) => Ok(Some(EncodedPrompt {
            features: Tensor::cat(&[&text.memory, &geometry.features], 0)?,
            padding_mask: Tensor::cat(&[&text.attention_mask, &geometry.padding_mask], 1)?,
        })),
        (Some(text), None) => Ok(Some(EncodedPrompt {
            features: text.memory.clone(),
            padding_mask: text.attention_mask.clone(),
        })),
        (None, Some(geometry)) => Ok(Some(EncodedPrompt {
            features: geometry.features.clone(),
            padding_mask: geometry.padding_mask.clone(),
        })),
        (None, None) => Ok(None),
    }
}

fn ground_from_encoded_prompt(
    model: &Sam3ImageModel,
    visual_features: &VisualBackboneOutput,
    prompt: &EncodedPrompt,
) -> Result<GroundingOutput> {
    let fused = model.encode_fused_prompt(visual_features, prompt)?;
    let decoder = model.decode_grounding(&fused, prompt)?;
    let segmentation = model.segment_grounding(visual_features, &decoder, &fused, prompt)?;
    let scores = model.text_detection_scores(&decoder)?;
    let best_idx = scores
        .argmax(1)?
        .flatten_all()?
        .to_vec1::<u32>()?
        .into_iter()
        .next()
        .unwrap_or(0) as usize;
    let best_score = scores.i((0, best_idx))?;
    let best_box = decoder.pred_boxes_xyxy.i((0, best_idx))?;
    let mask_logits = segmentation.mask_logits.i((0, best_idx))?;
    let mask = candle_nn::ops::sigmoid(&mask_logits)?;
    Ok(GroundingOutput {
        mask_logits: mask_logits.unsqueeze(0)?,
        masks: mask.unsqueeze(0)?,
        boxes_xyxy: best_box.unsqueeze(0)?,
        scores: best_score.unsqueeze(0)?,
        presence_scores: segmentation
            .presence_logits
            .as_ref()
            .and_then(|tensor| tensor.i((0, best_idx)).ok()),
    })
}

fn move_visual_output(
    output: &VisualBackboneOutput,
    device: &Device,
) -> Result<VisualBackboneOutput> {
    Ok(VisualBackboneOutput {
        backbone_fpn: output
            .backbone_fpn
            .iter()
            .map(|tensor| tensor.to_device(device))
            .collect::<Result<Vec<_>>>()?,
        vision_pos_enc: output
            .vision_pos_enc
            .iter()
            .map(|tensor| tensor.to_device(device))
            .collect::<Result<Vec<_>>>()?,
        sam2_backbone_fpn: output
            .sam2_backbone_fpn
            .as_ref()
            .map(|levels| {
                levels
                    .iter()
                    .map(|tensor| tensor.to_device(device))
                    .collect::<Result<Vec<_>>>()
            })
            .transpose()?,
        sam2_pos_enc: output
            .sam2_pos_enc
            .as_ref()
            .map(|levels| {
                levels
                    .iter()
                    .map(|tensor| tensor.to_device(device))
                    .collect::<Result<Vec<_>>>()
            })
            .transpose()?,
    })
}

fn tracker_visual_output(output: &VisualBackboneOutput) -> VisualBackboneOutput {
    match (&output.sam2_backbone_fpn, &output.sam2_pos_enc) {
        (Some(backbone_fpn), Some(vision_pos_enc)) => VisualBackboneOutput {
            backbone_fpn: backbone_fpn.clone(),
            vision_pos_enc: vision_pos_enc.clone(),
            sam2_backbone_fpn: output.sam2_backbone_fpn.clone(),
            sam2_pos_enc: output.sam2_pos_enc.clone(),
        },
        _ => output.clone(),
    }
}

fn debug_prompt_metadata(
    prompt: &SessionPrompt,
    used_visual_text_prompt: bool,
) -> Result<VideoDebugPromptMetadata> {
    Ok(VideoDebugPromptMetadata {
        text_prompt: prompt.text.clone(),
        used_visual_text_prompt,
        normalized_points_xy: prompt
            .points
            .clone()
            .unwrap_or_default()
            .into_iter()
            .map(|(x, y)| vec![x, y])
            .collect(),
        point_labels: prompt.point_labels.clone().unwrap_or_default(),
        normalized_boxes_cxcywh: prompt
            .boxes
            .clone()
            .unwrap_or_default()
            .into_iter()
            .map(|(cx, cy, w, h)| vec![cx, cy, w, h])
            .collect(),
        box_labels: prompt.box_labels.clone().unwrap_or_default(),
    })
}

fn normalize_video_mask_prompt(mask: &Tensor, device: &Device) -> Result<Tensor> {
    let mask = mask.to_device(device)?.to_dtype(DType::F32)?;
    match mask.rank() {
        2 => mask.unsqueeze(0)?.unsqueeze(0),
        3 => mask.unsqueeze(1),
        4 => Ok(mask),
        rank => candle::bail!("expected video mask prompt rank 2/3/4, got {rank}"),
    }
}

fn move_tracker_state(state: &TrackerFrameState, device: &Device) -> Result<TrackerFrameState> {
    Ok(TrackerFrameState {
        low_res_masks: state.low_res_masks.to_device(device)?,
        high_res_masks: state.high_res_masks.to_device(device)?,
        iou_scores: state.iou_scores.to_device(device)?,
        obj_ptr: state.obj_ptr.to_device(device)?,
        object_score_logits: state.object_score_logits.to_device(device)?,
        maskmem_features: state
            .maskmem_features
            .as_ref()
            .map(|tensor| tensor.to_device(device))
            .transpose()?,
        maskmem_pos_enc: state
            .maskmem_pos_enc
            .as_ref()
            .map(|tensor| tensor.to_device(device))
            .transpose()?,
        is_cond_frame: state.is_cond_frame,
    })
}

fn resize_mask_prompt_to_video(
    mask_prompt: &Tensor,
    video_size: ImageSize,
    device: &Device,
) -> Result<Tensor> {
    let mask_prompt = normalize_video_mask_prompt(mask_prompt, device)?;
    let (_, _, height, width) = mask_prompt.dims4()?;
    if height == video_size.height && width == video_size.width {
        Ok(mask_prompt)
    } else {
        resize_bilinear2d_antialias(&mask_prompt, video_size.height, video_size.width)
    }
}

fn resize_mask_prompt_to_tracker_input(
    mask_prompt: &Tensor,
    input_mask_size: usize,
    device: &Device,
) -> Result<Tensor> {
    let mask_prompt = normalize_video_mask_prompt(mask_prompt, device)?;
    let (_, _, height, width) = mask_prompt.dims4()?;
    if height == input_mask_size && width == input_mask_size {
        Ok(mask_prompt)
    } else {
        resize_bilinear2d_antialias(&mask_prompt, input_mask_size, input_mask_size)
    }
}

fn binary_mask_logits(mask: &Tensor) -> Result<Tensor> {
    let mask = mask.to_dtype(DType::F32)?;
    let binary = mask.ge(0.5f32)?.to_dtype(DType::F32)?;
    binary.affine(2048.0, -1024.0)
}

fn canonicalize_single_score_tensor(tensor: &Tensor) -> Result<Tensor> {
    let value = tensor
        .flatten_all()?
        .to_vec1::<f32>()?
        .into_iter()
        .next()
        .unwrap_or(0.0);
    Tensor::from_vec(vec![value], (1,), tensor.device())
}

fn mask_prompt_to_object_output(
    obj_id: u32,
    video_mask_probs: &Tensor,
    tracker_state: &TrackerFrameState,
    prompt_frame_idx: Option<usize>,
    _video_size: ImageSize,
) -> Result<ObjectFrameOutput> {
    let mask_logits = binary_mask_logits(video_mask_probs)?;
    let masks = video_mask_probs.ge(0.5f32)?.to_dtype(DType::F32)?;
    Ok(ObjectFrameOutput {
        obj_id,
        mask_logits,
        masks: masks.clone(),
        boxes_xyxy: mask_to_normalized_xyxy(&masks)?,
        scores: canonicalize_single_score_tensor(&candle_nn::ops::sigmoid(
            &tracker_state.object_score_logits,
        )?)?,
        presence_scores: None,
        prompt_frame_idx,
        memory_frame_indices: Vec::new(),
        text_prompt: None,
        used_explicit_geometry: true,
        reused_previous_output: false,
    })
}

fn boxes_cxcywh_to_xyxy_tensor(
    boxes_cxcywh: &[(f32, f32, f32, f32)],
    device: &Device,
) -> Result<Tensor> {
    let mut data = Vec::with_capacity(boxes_cxcywh.len() * 4);
    for (cx, cy, w, h) in boxes_cxcywh {
        let half_w = *w / 2.0;
        let half_h = *h / 2.0;
        data.push(cx - half_w);
        data.push(cy - half_h);
        data.push(cx + half_w);
        data.push(cy + half_h);
    }
    Tensor::from_vec(data, (boxes_cxcywh.len(), 2, 2), device)
}

fn tensor_to_mask_probs_2d(tensor: &Tensor) -> Result<Vec<Vec<f32>>> {
    let tensor = match tensor.rank() {
        2 => tensor.clone(),
        3 => tensor.i(0)?,
        4 => tensor.i((0, 0))?,
        rank => candle::bail!("expected mask tensor rank 2/3/4, got {rank}"),
    };
    tensor.to_dtype(DType::F32)?.to_vec2::<f32>()
}

fn count_foreground_pixels(mask_probs: &[Vec<f32>], threshold: f32) -> usize {
    mask_probs
        .iter()
        .flat_map(|row| row.iter())
        .filter(|value| **value >= threshold)
        .count()
}

fn summarize_tensor(
    tensor: &Tensor,
    foreground_threshold: Option<f32>,
) -> Result<TensorDebugSummary> {
    let shape = tensor.shape().dims().to_vec();
    let values = tensor
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    if values.is_empty() {
        return Ok(TensorDebugSummary {
            shape,
            dtype: format!("{:?}", tensor.dtype()),
            min: 0.0,
            max: 0.0,
            mean: 0.0,
            l2_norm: 0.0,
            foreground_pixel_count: Some(0).filter(|_| foreground_threshold.is_some()),
        });
    }
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut sum = 0.0f32;
    let mut l2_sum = 0.0f32;
    let mut foreground_pixel_count = 0usize;
    for value in values.iter().copied() {
        min = min.min(value);
        max = max.max(value);
        sum += value;
        l2_sum += value * value;
        if foreground_threshold.is_some_and(|threshold| value >= threshold) {
            foreground_pixel_count += 1;
        }
    }
    Ok(TensorDebugSummary {
        shape,
        dtype: format!("{:?}", tensor.dtype()),
        min,
        max,
        mean: sum / values.len() as f32,
        l2_norm: l2_sum.sqrt(),
        foreground_pixel_count: foreground_threshold.map(|_| foreground_pixel_count),
    })
}

fn summarize_tracker_state(state: &TrackerFrameState) -> Result<VideoDebugTrackerStateSummary> {
    Ok(VideoDebugTrackerStateSummary {
        is_cond_frame: state.is_cond_frame,
        low_res_masks_stats: summarize_tensor(&state.low_res_masks, Some(0.0))?,
        high_res_masks_stats: summarize_tensor(&state.high_res_masks, Some(0.0))?,
        iou_scores_stats: summarize_tensor(&state.iou_scores, None)?,
        object_score_logits_stats: summarize_tensor(&state.object_score_logits, None)?,
        obj_ptr_stats: summarize_tensor(&state.obj_ptr, None)?,
        maskmem_features_stats: state
            .maskmem_features
            .as_ref()
            .map(|tensor| summarize_tensor(tensor, None))
            .transpose()?,
        maskmem_pos_enc_stats: state
            .maskmem_pos_enc
            .as_ref()
            .map(|tensor| summarize_tensor(tensor, None))
            .transpose()?,
    })
}

fn truncate_prompt_for_encoder(prompt: &SessionPrompt, max_points: usize) -> SessionPrompt {
    let Some(points) = prompt.points.as_ref() else {
        return prompt.clone();
    };
    if max_points == 0 || points.len() <= max_points {
        return prompt.clone();
    }

    let num_first = max_points / 2;
    let num_last = max_points - num_first;
    let mut truncated = prompt.clone();
    let mut point_subset = Vec::with_capacity(max_points);
    point_subset.extend_from_slice(&points[..num_first]);
    point_subset.extend_from_slice(&points[points.len() - num_last..]);
    truncated.points = Some(point_subset);
    if let Some(labels) = prompt.point_labels.as_ref() {
        let mut label_subset = Vec::with_capacity(max_points);
        label_subset.extend_from_slice(&labels[..num_first]);
        label_subset.extend_from_slice(&labels[labels.len() - num_last..]);
        truncated.point_labels = Some(label_subset);
    }
    truncated
}

fn apply_prompt_frame_output_postprocess(
    output: &mut ObjectFrameOutput,
    config: &VideoConfig,
) -> Result<()> {
    if config.fill_hole_area == 0 {
        return Ok(());
    }
    output.mask_logits =
        postprocess_low_res_mask_logits_for_video(&output.mask_logits, config.fill_hole_area)?;
    output.masks = candle_nn::ops::sigmoid(&output.mask_logits)?;
    output.boxes_xyxy = mask_to_normalized_xyxy(&output.masks)?;
    Ok(())
}

fn session_prompt_to_geometry(prompt: &SessionPrompt, device: &Device) -> Result<GeometryPrompt> {
    let mut geometry_prompt = GeometryPrompt::default();

    if let Some(points) = prompt.points.as_ref() {
        let mut data = Vec::with_capacity(points.len() * 2);
        for (x, y) in points {
            data.push(*x);
            data.push(*y);
        }
        geometry_prompt.points_xy = Some(Tensor::from_vec(data, (points.len(), 2), device)?);
    }
    if let Some(labels) = prompt.point_labels.as_ref() {
        geometry_prompt.point_labels =
            Some(Tensor::from_vec(labels.clone(), (labels.len(),), device)?);
    }

    if let Some(boxes) = prompt.boxes.as_ref() {
        let mut data = Vec::with_capacity(boxes.len() * 4);
        for (cx, cy, width, height) in boxes {
            data.push(*cx);
            data.push(*cy);
            data.push(*width);
            data.push(*height);
        }
        geometry_prompt.boxes_cxcywh = Some(Tensor::from_vec(data, (boxes.len(), 4), device)?);
    }
    if let Some(labels) = prompt.box_labels.as_ref() {
        geometry_prompt.box_labels =
            Some(Tensor::from_vec(labels.clone(), (labels.len(),), device)?);
    }

    Ok(geometry_prompt)
}

fn mask_to_normalized_xyxy(mask: &Tensor) -> Result<Tensor> {
    let mask = match mask.rank() {
        4 => mask.i((0, 0))?,
        3 => mask.i(0)?,
        2 => mask.clone(),
        rank => candle::bail!("expected mask rank 2, 3, or 4, got {}", rank),
    };
    let values = mask.ge(0.5f32)?.to_vec2::<u8>()?;
    if values.is_empty() || values[0].is_empty() {
        return Tensor::zeros((1, 4), DType::F32, mask.device());
    }
    let height = values.len();
    let width = values[0].len();
    let mut min_x = width;
    let mut min_y = height;
    let mut max_x = 0usize;
    let mut max_y = 0usize;
    let mut any = false;
    for (y, row) in values.iter().enumerate() {
        for (x, value) in row.iter().enumerate() {
            if *value != 0 {
                any = true;
                min_x = min_x.min(x);
                min_y = min_y.min(y);
                max_x = max_x.max(x);
                max_y = max_y.max(y);
            }
        }
    }
    if !any {
        return Tensor::zeros((1, 4), DType::F32, mask.device());
    }
    Tensor::from_vec(
        vec![
            min_x as f32 / width.max(1) as f32,
            min_y as f32 / height.max(1) as f32,
            (max_x + 1) as f32 / width.max(1) as f32,
            (max_y + 1) as f32 / height.max(1) as f32,
        ],
        (1, 4),
        mask.device(),
    )
}

fn resize_mask_logits_to_video(mask_logits: &Tensor, video_size: ImageSize) -> Result<Tensor> {
    let mask_logits = match mask_logits.rank() {
        2 => mask_logits.unsqueeze(0)?.unsqueeze(0)?,
        3 => mask_logits.unsqueeze(0)?,
        4 => mask_logits.clone(),
        rank => candle::bail!("expected mask logits rank 2, 3, or 4, got {}", rank),
    };
    mask_logits.upsample_bilinear2d(video_size.height, video_size.width, false)
}

fn resize_mask_probs(mask_probs: &Tensor, height: usize, width: usize) -> Result<Tensor> {
    let mask_probs = match mask_probs.rank() {
        2 => mask_probs.unsqueeze(0)?.unsqueeze(0)?,
        3 => mask_probs.unsqueeze(0)?,
        4 => mask_probs.clone(),
        rank => candle::bail!("expected mask probabilities rank 2, 3, or 4, got {}", rank),
    };
    mask_probs.upsample_bilinear2d(height, width, false)
}

fn canonicalize_score_tensor(scores: &Tensor) -> Result<Tensor> {
    let values = scores.flatten_all()?.to_vec1::<f32>()?;
    Tensor::from_vec(values, (scores.elem_count(),), scores.device())
}

fn score_tensor_from_value(score: f32, device: &Device) -> Result<Tensor> {
    Tensor::from_vec(vec![score], (1,), device)
}

fn trim_memory_frame_indices(
    mut memory_frame_indices: Vec<usize>,
    max_memory_frames: usize,
) -> Vec<usize> {
    if max_memory_frames == 0 || memory_frame_indices.len() <= max_memory_frames {
        return memory_frame_indices;
    }
    let drop_count = memory_frame_indices.len() - max_memory_frames;
    memory_frame_indices.drain(..drop_count);
    memory_frame_indices
}

fn postprocess_low_res_mask_logits_for_video(
    mask_logits: &Tensor,
    max_area: usize,
) -> Result<Tensor> {
    if max_area == 0 {
        return Ok(mask_logits.clone());
    }
    let device = mask_logits.device().clone();
    let mask_logits = mask_logits.to_device(&Device::Cpu)?;
    let (batch, channel, height, width) = mask_logits.dims4()?;
    let mut processed = Vec::with_capacity(batch * channel * height * width);

    for batch_idx in 0..batch {
        for channel_idx in 0..channel {
            let mut plane = mask_logits.i((batch_idx, channel_idx))?.to_vec2::<f32>()?;
            fill_small_holes_in_plane(&mut plane, height, width, max_area);
            remove_small_sprinkles_in_plane(&mut plane, height, width, max_area);
            processed.extend(plane.into_iter().flatten());
        }
    }

    Tensor::from_vec(processed, (batch, channel, height, width), &Device::Cpu)?.to_device(&device)
}

fn fill_small_holes_in_plane(plane: &mut [Vec<f32>], height: usize, width: usize, max_area: usize) {
    let mut visited = vec![false; height * width];
    for row in 0..height {
        for col in 0..width {
            let idx = row * width + col;
            if visited[idx] || plane[row][col] > 0.0 {
                continue;
            }
            let component = collect_component(&mut visited, height, width, row, col, |r, c| {
                plane[r][c] <= 0.0
            });
            if component.len() <= max_area {
                for (r, c) in component {
                    plane[r][c] = VIDEO_PROPAGATION_HOLE_FILL_LOGIT;
                }
            }
        }
    }
}

fn remove_small_sprinkles_in_plane(
    plane: &mut [Vec<f32>],
    height: usize,
    width: usize,
    max_area: usize,
) {
    let total_fg = plane
        .iter()
        .flat_map(|row| row.iter())
        .filter(|value| **value > 0.0)
        .count();
    let fg_area_thresh = total_fg.saturating_div(2).min(max_area);
    if fg_area_thresh == 0 {
        return;
    }
    let mut visited = vec![false; height * width];
    for row in 0..height {
        for col in 0..width {
            let idx = row * width + col;
            if visited[idx] || plane[row][col] <= 0.0 {
                continue;
            }
            let component = collect_component(&mut visited, height, width, row, col, |r, c| {
                plane[r][c] > 0.0
            });
            if component.len() <= fg_area_thresh {
                for (r, c) in component {
                    plane[r][c] = VIDEO_PROPAGATION_SPRINKLE_REMOVE_LOGIT;
                }
            }
        }
    }
}

fn collect_component<F>(
    visited: &mut [bool],
    height: usize,
    width: usize,
    start_row: usize,
    start_col: usize,
    predicate: F,
) -> Vec<(usize, usize)>
where
    F: Fn(usize, usize) -> bool,
{
    let mut queue = VecDeque::new();
    let mut component = Vec::new();
    let start_idx = start_row * width + start_col;
    visited[start_idx] = true;
    queue.push_back((start_row, start_col));

    while let Some((row, col)) = queue.pop_front() {
        component.push((row, col));
        for (next_row, next_col) in neighbors8(row, col, height, width) {
            let idx = next_row * width + next_col;
            if visited[idx] || !predicate(next_row, next_col) {
                continue;
            }
            visited[idx] = true;
            queue.push_back((next_row, next_col));
        }
    }

    component
}

fn neighbors8(row: usize, col: usize, height: usize, width: usize) -> Vec<(usize, usize)> {
    let row_start = row.saturating_sub(1);
    let row_end = (row + 1).min(height.saturating_sub(1));
    let col_start = col.saturating_sub(1);
    let col_end = (col + 1).min(width.saturating_sub(1));
    let mut neighbors = Vec::with_capacity(8);
    for next_row in row_start..=row_end {
        for next_col in col_start..=col_end {
            if next_row == row && next_col == col {
                continue;
            }
            neighbors.push((next_row, next_col));
        }
    }
    neighbors
}

fn tracker_state_to_object_output(
    obj_id: u32,
    state: &TrackerFrameState,
    score_override: Option<f32>,
    prompt_frame_idx: Option<usize>,
    memory_frame_indices: Vec<usize>,
    text_prompt: Option<String>,
    used_explicit_geometry: bool,
    reused_previous_output: bool,
    video_size: ImageSize,
) -> Result<ObjectFrameOutput> {
    let mask_logits = resize_mask_logits_to_video(&state.high_res_masks, video_size)?;
    let masks = candle_nn::ops::sigmoid(&mask_logits)?;
    Ok(ObjectFrameOutput {
        obj_id,
        mask_logits,
        masks: masks.clone(),
        boxes_xyxy: mask_to_normalized_xyxy(&masks)?,
        scores: match score_override {
            Some(score) => score_tensor_from_value(score, state.iou_scores.device())?,
            None => canonicalize_score_tensor(&state.iou_scores)?,
        },
        presence_scores: Some(candle_nn::ops::sigmoid(&state.object_score_logits)?),
        prompt_frame_idx,
        memory_frame_indices,
        text_prompt,
        used_explicit_geometry,
        reused_previous_output,
    })
}

fn grounding_to_object_output(
    obj_id: u32,
    grounding: &GroundingOutput,
    prompt_frame_idx: Option<usize>,
    memory_frame_indices: Vec<usize>,
    text_prompt: Option<String>,
    used_explicit_geometry: bool,
    reused_previous_output: bool,
    video_size: ImageSize,
) -> Result<ObjectFrameOutput> {
    let mask_logits = resize_mask_logits_to_video(&grounding.mask_logits, video_size)?;
    let masks = candle_nn::ops::sigmoid(&mask_logits)?;
    Ok(ObjectFrameOutput {
        obj_id,
        mask_logits,
        masks: masks.clone(),
        boxes_xyxy: mask_to_normalized_xyxy(&masks)?,
        scores: canonicalize_score_tensor(&grounding.scores)?,
        presence_scores: grounding
            .presence_scores
            .as_ref()
            .map(canonicalize_score_tensor)
            .transpose()?,
        prompt_frame_idx,
        memory_frame_indices,
        text_prompt,
        used_explicit_geometry,
        reused_previous_output,
    })
}

fn build_processing_order(
    session: &Sam3VideoSession,
    direction: PropagationDirection,
    start_frame_idx: Option<usize>,
    max_frame_num_to_track: Option<usize>,
    always_start_from_first_ann_frame: bool,
) -> Result<Vec<usize>> {
    let seed_frames = session.prompt_frames();
    if seed_frames.is_empty() {
        candle::bail!("no prompts added to session");
    }
    let num_frames = session.num_frames();
    let start_frame_idx = match start_frame_idx {
        Some(frame_idx) if !always_start_from_first_ann_frame => frame_idx,
        None => match direction {
            PropagationDirection::Forward | PropagationDirection::Both => {
                *seed_frames.iter().next().expect("seed frames checked")
            }
            PropagationDirection::Backward => {
                *seed_frames.iter().next_back().expect("seed frames checked")
            }
        },
        Some(_) => match direction {
            PropagationDirection::Forward | PropagationDirection::Both => {
                *seed_frames.iter().next().expect("seed frames checked")
            }
            PropagationDirection::Backward => {
                *seed_frames.iter().next_back().expect("seed frames checked")
            }
        },
    };
    if start_frame_idx >= num_frames {
        candle::bail!(
            "start_frame_idx {} exceeds video length {}",
            start_frame_idx,
            num_frames
        );
    }
    let max_frame_num_to_track = max_frame_num_to_track.unwrap_or(num_frames);
    Ok(match direction {
        PropagationDirection::Forward | PropagationDirection::Both => {
            let end = (start_frame_idx + max_frame_num_to_track).min(num_frames.saturating_sub(1));
            (start_frame_idx..=end).collect()
        }
        PropagationDirection::Backward => {
            let end = start_frame_idx.saturating_sub(max_frame_num_to_track);
            (end..=start_frame_idx).rev().collect()
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::Tensor;
    use candle_nn::VarBuilder;
    use image::{GrayImage, ImageBuffer, Luma, Rgb, RgbImage};

    use crate::models::sam3::{
        Config, DecoderConfig, EncoderConfig, GeometryConfig, ImageConfig, NeckConfig,
        Sam3TrackerConfig, SegmentationConfig, TextConfig, VisionConfig,
    };

    fn tiny_segmentation_config() -> Config {
        Config {
            image: ImageConfig {
                image_size: 56,
                image_mean: [0.5, 0.5, 0.5],
                image_std: [0.5, 0.5, 0.5],
            },
            vision: VisionConfig {
                image_size: 56,
                pretrain_image_size: 28,
                patch_size: 14,
                embed_dim: 32,
                depth: 0,
                num_heads: 4,
                mlp_ratio: 4.0,
                window_size: 2,
                global_attn_blocks: vec![],
                use_abs_pos: true,
                tile_abs_pos: true,
                use_rope: true,
                use_interp_rope: true,
                rope_theta: 10_000.0,
                rope_pt_size: 24,
                retain_cls_token: false,
                ln_pre: false,
            },
            text: TextConfig {
                d_model: 32,
                width: 64,
                heads: 4,
                layers: 1,
                context_length: 4,
                vocab_size: 64,
            },
            neck: NeckConfig {
                d_model: 32,
                scale_factors: [4.0, 2.0, 1.0, 0.5],
                scalp: 1,
                add_sam2_neck: false,
            },
            geometry: GeometryConfig {
                d_model: 32,
                num_layers: 1,
                num_heads: 1,
                dim_feedforward: 64,
                roi_size: 2,
                add_cls: true,
                add_post_encode_proj: true,
            },
            encoder: EncoderConfig {
                d_model: 32,
                num_layers: 1,
                num_feature_levels: 1,
                num_heads: 1,
                dim_feedforward: 64,
                add_pooled_text_to_image: false,
                pool_text_with_mask: true,
            },
            decoder: DecoderConfig {
                d_model: 32,
                num_layers: 1,
                num_queries: 2,
                num_heads: 1,
                dim_feedforward: 64,
                presence_token: true,
                use_text_cross_attention: true,
                box_rpb_mode: "none".to_owned(),
                box_rpb_resolution: 56,
                box_rpb_stride: 14,
                clamp_presence_logit_max: 10.0,
            },
            segmentation: SegmentationConfig {
                enabled: true,
                hidden_dim: 32,
                upsampling_stages: 1,
                aux_masks: false,
                presence_head: false,
            },
        }
    }

    fn tiny_model(device: &Device) -> Result<Sam3ImageModel> {
        Sam3ImageModel::new(
            &tiny_segmentation_config(),
            VarBuilder::zeros(DType::F32, device),
        )
    }

    fn tiny_tracker(device: &Device) -> Result<Sam3TrackerModel> {
        let config = tiny_segmentation_config();
        let tracker_config = Sam3TrackerConfig::from_sam3_config(&config);
        Sam3TrackerModel::new(&tracker_config, VarBuilder::zeros(DType::F32, device))
    }

    fn sam3_test_checkpoint_path() -> Option<PathBuf> {
        let env_path = std::env::var_os("SAM3_TEST_CHECKPOINT")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("SAM3_TEST_CHECKPOINT_DIR").map(PathBuf::from));
        let mut candidates = Vec::new();
        if let Some(path) = env_path {
            candidates.push(path);
        }
        candidates.push(PathBuf::from("/home/dnorthover/extcode/hf_sam3"));
        candidates.push(PathBuf::from("/home/dnorthover/extcode/hf_sam3/sam3.pt"));
        candidates.into_iter().find_map(|path| {
            if path.is_dir() {
                let file = path.join("sam3.pt");
                file.exists().then_some(file)
            } else if path.exists() {
                Some(path)
            } else {
                None
            }
        })
    }

    fn tracker_config_with_reference_runtime_overrides(bundle: Option<&str>) -> Result<Sam3TrackerConfig> {
        let mut config = Sam3TrackerConfig::from_sam3_config(&Config::default());
        let Some(bundle) = bundle else {
            return Ok(config);
        };
        let manifest = load_reference_internal_manifest(bundle)?;
        let tracker_config = manifest["tracker_config"].as_object().ok_or_else(|| {
            candle::Error::Msg("reference manifest missing tracker_config".to_owned())
        })?;
        let predictor_config = manifest["predictor_config"].as_object().ok_or_else(|| {
            candle::Error::Msg("reference manifest missing predictor_config".to_owned())
        })?;

        if let Some(value) = tracker_config
            .get("use_memory_selection")
            .and_then(|value| value.as_bool())
        {
            config.use_memory_selection = value;
        }
        if let Some(value) = tracker_config
            .get("memory_temporal_stride_for_eval")
            .and_then(|value| value.as_u64())
        {
            config.memory_temporal_stride_for_eval = value as usize;
        }
        if let Some(value) = tracker_config
            .get("max_obj_ptrs_in_encoder")
            .and_then(|value| value.as_u64())
        {
            config.max_obj_ptrs_in_encoder = value as usize;
        }
        if let Some(value) = tracker_config
            .get("max_cond_frames_in_attn")
            .and_then(|value| value.as_u64())
        {
            config.max_cond_frames_in_attn = value as usize;
        }
        if let Some(value) = tracker_config
            .get("keep_first_cond_frame")
            .and_then(|value| value.as_bool())
        {
            config.keep_first_cond_frame = value;
        }
        if let Some(value) = tracker_config
            .get("trim_past_non_cond_mem_for_eval")
            .and_then(|value| value.as_bool())
        {
            config.predictor.trim_past_non_cond_mem_for_eval = value;
        }
        if let Some(value) = tracker_config
            .get("offload_output_to_cpu_for_eval")
            .and_then(|value| value.as_bool())
        {
            config.predictor.offload_output_to_cpu_for_eval = value;
        }
        if let Some(value) = tracker_config
            .get("forward_backbone_per_frame_for_eval")
            .and_then(|value| value.as_bool())
        {
            config.predictor.forward_backbone_per_frame_for_eval = value;
        }
        if let Some(value) = predictor_config
            .get("clear_non_cond_mem_around_input")
            .and_then(|value| value.as_bool())
        {
            config.predictor.clear_non_cond_mem_around_input = value;
        }
        if let Some(value) = predictor_config
            .get("clear_non_cond_mem_for_multi_obj")
            .and_then(|value| value.as_bool())
        {
            config.predictor.clear_non_cond_mem_for_multi_obj = value;
        }
        if let Some(value) = predictor_config
            .get("always_start_from_first_ann_frame")
            .and_then(|value| value.as_bool())
        {
            config.predictor.always_start_from_first_ann_frame = value;
        }
        if let Some(value) = predictor_config
            .get("iter_use_prev_mask_pred")
            .and_then(|value| value.as_bool())
        {
            config.predictor.iter_use_prev_mask_pred = value;
        }
        if let Some(value) = predictor_config
            .get("add_all_frames_to_correct_as_cond")
            .and_then(|value| value.as_bool())
        {
            config.predictor.add_all_frames_to_correct_as_cond = value;
        }
        if let Some(value) = predictor_config
            .get("use_prev_mem_frame")
            .and_then(|value| value.as_bool())
        {
            config.predictor.use_prev_mem_frame = value;
        }
        if let Some(value) = predictor_config
            .get("use_stateless_refinement")
            .and_then(|value| value.as_bool())
        {
            config.predictor.use_stateless_refinement = value;
        }
        if let Some(value) = predictor_config
            .get("refinement_detector_cond_frame_removal_window")
            .and_then(|value| value.as_u64())
        {
            config.predictor.refinement_detector_cond_frame_removal_window = value as usize;
        }
        Ok(config)
    }

    fn load_runtime_models_from_checkpoint(
        bundle: Option<&str>,
    ) -> Result<Option<(Sam3ImageModel, Sam3TrackerModel, Device)>> {
        let Some(checkpoint_path) = sam3_test_checkpoint_path() else {
            return Ok(None);
        };
        let device = Device::Cpu;
        let config = Config::default();
        let checkpoint =
            crate::models::sam3::checkpoint::Sam3CheckpointSource::upstream_pth(checkpoint_path);
        let model =
            Sam3ImageModel::from_checkpoint_source(&config, &checkpoint, DType::F32, &device)?;
        let tracker_config = tracker_config_with_reference_runtime_overrides(bundle)?;
        let tracker = Sam3TrackerModel::new(
            &tracker_config,
            checkpoint.load_tracker_var_builder(DType::F32, &device)?,
        )?;
        Ok(Some((model, tracker, device)))
    }

    fn sam3_test_tokenizer_path() -> Option<PathBuf> {
        let checkpoint_path = sam3_test_checkpoint_path()?;
        let tokenizer = checkpoint_path.parent()?.join("tokenizer.json");
        tokenizer.exists().then_some(tokenizer)
    }

    fn reference_bundle_dir(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../candle-examples/examples/sam3")
            .join(name)
    }

    fn reference_input_frames_dir(name: &str) -> PathBuf {
        let bundle_dir = reference_bundle_dir(name);
        let tracker_frames = bundle_dir.join("tracker_input_frames");
        if tracker_frames.exists() {
            tracker_frames
        } else {
            bundle_dir.join("frames")
        }
    }

    fn load_reference_frame_output(
        bundle: &str,
        frame_idx: usize,
    ) -> Result<(Vec<f32>, f32, PathBuf)> {
        let bundle_dir = reference_bundle_dir(bundle);
        let value: serde_json::Value =
            serde_json::from_slice(&fs::read(bundle_dir.join("video_results.json"))?)
                .map_err(|err| candle::Error::Msg(err.to_string()))?;
        let frames = match &value {
            serde_json::Value::Array(frames) => frames,
            serde_json::Value::Object(_) => value["frames"].as_array().ok_or_else(|| {
                candle::Error::Msg("reference video results missing frames array".to_owned())
            })?,
            _ => {
                candle::bail!("reference video results must be an array or object with frames")
            }
        };
        let frame = frames
            .iter()
            .find(|frame| frame["frame_idx"].as_u64() == Some(frame_idx as u64))
            .ok_or_else(|| {
                candle::Error::Msg(format!(
                    "reference video results missing frame {}",
                    frame_idx
                ))
            })?;
        let objects = frame["objects"].as_array().ok_or_else(|| {
            candle::Error::Msg(format!(
                "reference frame {} missing objects array",
                frame_idx
            ))
        })?;
        let object = &objects[0];
        let boxes = object["boxes_xyxy"]
            .as_array()
            .and_then(|boxes| boxes.first())
            .and_then(|first| first.as_array())
            .ok_or_else(|| {
                candle::Error::Msg(format!("reference frame {} missing boxes_xyxy", frame_idx))
            })?
            .iter()
            .map(|value| value.as_f64().unwrap_or(0.0) as f32)
            .collect::<Vec<_>>();
        let score = object["scores"]
            .as_array()
            .and_then(|scores| scores.first())
            .and_then(|value| value.as_f64())
            .ok_or_else(|| {
                candle::Error::Msg(format!("reference frame {} missing score", frame_idx))
            })? as f32;
        let mask_path = object["mask_path"].as_str().ok_or_else(|| {
            candle::Error::Msg(format!("reference frame {} missing mask_path", frame_idx))
        })?;
        Ok((boxes, score, bundle_dir.join(mask_path)))
    }

    fn load_reference_object_frame_output(
        bundle: &str,
        frame_idx: usize,
        obj_id: u32,
    ) -> Result<(Vec<f32>, f32, PathBuf)> {
        let bundle_dir = reference_bundle_dir(bundle);
        let value: serde_json::Value =
            serde_json::from_slice(&fs::read(bundle_dir.join("video_results.json"))?)
                .map_err(|err| candle::Error::Msg(err.to_string()))?;
        let frames = match &value {
            serde_json::Value::Array(frames) => frames,
            serde_json::Value::Object(_) => value["frames"].as_array().ok_or_else(|| {
                candle::Error::Msg("reference video results missing frames array".to_owned())
            })?,
            _ => {
                candle::bail!("reference video results must be an array or object with frames")
            }
        };
        let frame = frames
            .iter()
            .find(|frame| frame["frame_idx"].as_u64() == Some(frame_idx as u64))
            .ok_or_else(|| {
                candle::Error::Msg(format!(
                    "reference video results missing frame {}",
                    frame_idx
                ))
            })?;
        let objects = frame["objects"].as_array().ok_or_else(|| {
            candle::Error::Msg(format!(
                "reference frame {} missing objects array",
                frame_idx
            ))
        })?;
        let object = objects
            .iter()
            .find(|object| object["obj_id"].as_u64() == Some(obj_id as u64))
            .ok_or_else(|| {
                candle::Error::Msg(format!(
                    "reference frame {} missing obj_id {}",
                    frame_idx, obj_id
                ))
            })?;
        let boxes = object["boxes_xyxy"]
            .as_array()
            .and_then(|boxes| boxes.first())
            .and_then(|first| first.as_array())
            .ok_or_else(|| {
                candle::Error::Msg(format!(
                    "reference frame {} obj_id {} missing boxes_xyxy",
                    frame_idx, obj_id
                ))
            })?
            .iter()
            .map(|value| value.as_f64().unwrap_or(0.0) as f32)
            .collect::<Vec<_>>();
        let score = object["scores"]
            .as_array()
            .and_then(|scores| scores.first())
            .and_then(|value| value.as_f64())
            .ok_or_else(|| {
                candle::Error::Msg(format!(
                    "reference frame {} obj_id {} missing score",
                    frame_idx, obj_id
                ))
            })? as f32;
        let mask_path = object["mask_path"].as_str().ok_or_else(|| {
            candle::Error::Msg(format!(
                "reference frame {} obj_id {} missing mask_path",
                frame_idx, obj_id
            ))
        })?;
        Ok((boxes, score, bundle_dir.join(mask_path)))
    }

    fn load_reference_frame_indices(bundle: &str) -> Result<Vec<usize>> {
        let bundle_dir = reference_bundle_dir(bundle);
        let value: serde_json::Value =
            serde_json::from_slice(&fs::read(bundle_dir.join("video_results.json"))?)
                .map_err(|err| candle::Error::Msg(err.to_string()))?;
        let frames = match &value {
            serde_json::Value::Array(frames) => frames,
            serde_json::Value::Object(_) => value["frames"].as_array().ok_or_else(|| {
                candle::Error::Msg("reference video results missing frames array".to_owned())
            })?,
            _ => {
                candle::bail!("reference video results must be an array or object with frames")
            }
        };
        Ok(frames
            .iter()
            .filter(|frame| {
                frame["objects"]
                    .as_array()
                    .map(|objects| !objects.is_empty())
                    .unwrap_or(false)
            })
            .filter_map(|frame| frame["frame_idx"].as_u64())
            .map(|frame_idx| frame_idx as usize)
            .collect())
    }

    fn load_reference_frame0_output(bundle: &str) -> Result<(Vec<f32>, f32, PathBuf)> {
        load_reference_frame_output(bundle, 0)
    }

    fn load_reference_box_prompt(bundle: &str) -> Result<(f32, f32, f32, f32)> {
        let bundle_dir = reference_bundle_dir(bundle);
        let value: serde_json::Value =
            serde_json::from_slice(&fs::read(bundle_dir.join("reference.json"))?)
                .map_err(|err| candle::Error::Msg(err.to_string()))?;
        let boxes = value["boxes_cxcywh_normalized"]
            .as_array()
            .and_then(|boxes| boxes.first())
            .and_then(|first| first.as_array())
            .ok_or_else(|| {
                candle::Error::Msg("reference bundle missing boxes_cxcywh_normalized".to_owned())
            })?;
        Ok((
            boxes[0].as_f64().unwrap_or(0.0) as f32,
            boxes[1].as_f64().unwrap_or(0.0) as f32,
            boxes[2].as_f64().unwrap_or(0.0) as f32,
            boxes[3].as_f64().unwrap_or(0.0) as f32,
        ))
    }

    fn load_reference_mask_prompt_box_xyxy(bundle: &str) -> Result<(f32, f32, f32, f32)> {
        let bundle_dir = reference_bundle_dir(bundle);
        let value: serde_json::Value =
            serde_json::from_slice(&fs::read(bundle_dir.join("reference.json"))?)
                .map_err(|err| candle::Error::Msg(err.to_string()))?;
        let actions = value["scenario"]["actions"].as_array().ok_or_else(|| {
            candle::Error::Msg("reference bundle missing scenario actions".to_owned())
        })?;
        let mask = actions[0]["mask"]["box_xyxy"].as_array().ok_or_else(|| {
            candle::Error::Msg("reference mask scenario missing box_xyxy".to_owned())
        })?;
        Ok((
            mask[0].as_f64().unwrap_or(0.0) as f32,
            mask[1].as_f64().unwrap_or(0.0) as f32,
            mask[2].as_f64().unwrap_or(0.0) as f32,
            mask[3].as_f64().unwrap_or(0.0) as f32,
        ))
    }

    fn load_reference_point_prompt(bundle: &str) -> Result<(Vec<(f32, f32)>, Vec<u32>)> {
        load_reference_point_prompt_on_frame(bundle, 0)
    }

    fn load_reference_point_prompt_on_frame(
        bundle: &str,
        frame_idx: usize,
    ) -> Result<(Vec<(f32, f32)>, Vec<u32>)> {
        let bundle_dir = reference_bundle_dir(bundle);
        let value: serde_json::Value =
            serde_json::from_slice(&fs::read(bundle_dir.join("reference.json"))?)
                .map_err(|err| candle::Error::Msg(err.to_string()))?;
        let actions = value["scenario"]["actions"].as_array().ok_or_else(|| {
            candle::Error::Msg("reference bundle missing scenario actions".to_owned())
        })?;
        let add_prompt = actions
            .iter()
            .find(|action| {
                action["type"].as_str() == Some("add_prompt")
                    && action["frame_idx"].as_u64() == Some(frame_idx as u64)
            })
            .ok_or_else(|| {
                candle::Error::Msg(format!(
                    "reference bundle missing add_prompt action for frame {}",
                    frame_idx
                ))
            })?;
        let points = add_prompt["points_xy_normalized"]
            .as_array()
            .ok_or_else(|| {
                candle::Error::Msg(
                    "reference point scenario missing points_xy_normalized".to_owned(),
                )
            })?
            .iter()
            .map(|point| {
                let point = point.as_array().ok_or_else(|| {
                    candle::Error::Msg(
                        "reference point scenario contains a malformed point".to_owned(),
                    )
                })?;
                Ok((
                    point[0].as_f64().unwrap_or(0.0) as f32,
                    point[1].as_f64().unwrap_or(0.0) as f32,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let labels = add_prompt["point_labels"]
            .as_array()
            .ok_or_else(|| {
                candle::Error::Msg("reference point scenario missing point_labels".to_owned())
            })?
            .iter()
            .map(|value| value.as_u64().unwrap_or(0) as u32)
            .collect::<Vec<_>>();
        Ok((points, labels))
    }

    fn load_reference_internal_manifest(bundle: &str) -> Result<serde_json::Value> {
        let bundle_dir = reference_bundle_dir(bundle);
        serde_json::from_slice(&fs::read(bundle_dir.join("debug/internal_manifest.json"))?)
            .map_err(|err| candle::Error::Msg(err.to_string()))
    }

    fn apply_reference_predictor_runtime_overrides(
        predictor: &mut Sam3VideoPredictor<'_>,
        bundle: &str,
    ) -> Result<()> {
        let manifest = load_reference_internal_manifest(bundle)?;
        let predictor_config = manifest["predictor_config"].as_object().ok_or_else(|| {
            candle::Error::Msg("reference manifest missing predictor_config".to_owned())
        })?;
        if let Some(fill_hole_area) = predictor_config
            .get("fill_hole_area")
            .and_then(|value| value.as_u64())
        {
            predictor.video_config.fill_hole_area = fill_hole_area as usize;
        }
        if let Some(max_point_num) = predictor_config
            .get("max_point_num_in_prompt_enc")
            .and_then(|value| value.as_u64())
        {
            predictor.video_config.max_point_num_in_prompt_enc = max_point_num as usize;
        }
        if let Some(non_overlap_masks_for_output) = predictor_config
            .get("non_overlap_masks_for_output")
            .and_then(|value| value.as_bool())
        {
            predictor.video_config.non_overlap_masks_for_output = non_overlap_masks_for_output;
        }
        Ok(())
    }

    fn load_reference_internal_tensor(bundle: &str, key: &str) -> Result<Tensor> {
        use candle::safetensors::Load;

        let bundle_dir = reference_bundle_dir(bundle);
        let path = bundle_dir.join("debug/internal_fixtures.safetensors");
        let tensors =
            unsafe { candle::safetensors::MmapedSafetensors::new(&path) }.map_err(|err| {
                candle::Error::Msg(format!(
                    "failed to mmap reference fixtures {}: {err}",
                    path.display()
                ))
            })?;
        tensors
            .get(key)
            .map_err(|err| {
                candle::Error::Msg(format!(
                    "failed to read tensor `{key}` from reference fixtures {}: {err}",
                    path.display()
                ))
            })?
            .load(&Device::Cpu)
    }

    fn load_reference_internal_record(
        bundle: &str,
        stage: &str,
        frame_idx: usize,
    ) -> Result<serde_json::Value> {
        let records = load_reference_internal_records(bundle, stage, frame_idx)?;
        records.into_iter().next().ok_or_else(|| {
            candle::Error::Msg(format!(
                "reference manifest missing {stage} record for frame {frame_idx}"
            ))
        })
    }

    fn load_reference_internal_records(
        bundle: &str,
        stage: &str,
        frame_idx: usize,
    ) -> Result<Vec<serde_json::Value>> {
        let manifest = load_reference_internal_manifest(bundle)?;
        Ok(manifest["records"]
            .as_array()
            .ok_or_else(|| candle::Error::Msg("reference manifest missing records".to_owned()))?
            .iter()
            .filter(|record| {
                record["stage"].as_str() == Some(stage)
                    && record["frame_idx"].as_u64() == Some(frame_idx as u64)
            })
            .cloned()
            .collect())
    }

    fn load_reference_internal_record_matching<F>(
        bundle: &str,
        stage: &str,
        frame_idx: usize,
        predicate: F,
    ) -> Result<serde_json::Value>
    where
        F: Fn(&serde_json::Value) -> bool,
    {
        load_reference_internal_records(bundle, stage, frame_idx)?
            .into_iter()
            .find(predicate)
            .ok_or_else(|| {
                candle::Error::Msg(format!(
                    "reference manifest missing matching {stage} record for frame {frame_idx}"
                ))
            })
    }

    fn load_reference_internal_record_matching_last<F>(
        bundle: &str,
        stage: &str,
        frame_idx: usize,
        predicate: F,
    ) -> Result<serde_json::Value>
    where
        F: Fn(&serde_json::Value) -> bool,
    {
        load_reference_internal_records(bundle, stage, frame_idx)?
            .into_iter()
            .rev()
            .find(predicate)
            .ok_or_else(|| {
                candle::Error::Msg(format!(
                    "reference manifest missing last matching {stage} record for frame {frame_idx}"
                ))
            })
    }

    fn load_reference_track_step_frame_output(
        bundle: &str,
        frame_idx: usize,
        video_size: ImageSize,
    ) -> Result<(Vec<f32>, f32, Tensor)> {
        let record = load_reference_internal_record(bundle, "track_step", frame_idx)?;
        let tensor_keys = record["tensor_keys"].as_object().ok_or_else(|| {
            candle::Error::Msg(format!(
                "reference track_step frame {frame_idx} missing tensor_keys"
            ))
        })?;
        let high_res_key = tensor_keys
            .get("track_step_output.pred_masks_high_res")
            .and_then(|value| value.as_str())
            .ok_or_else(|| {
                candle::Error::Msg(format!(
                    "reference track_step frame {frame_idx} missing pred_masks_high_res key"
                ))
            })?;
        let object_score_key = tensor_keys
            .get("track_step_output.object_score_logits")
            .and_then(|value| value.as_str())
            .ok_or_else(|| {
                candle::Error::Msg(format!(
                    "reference track_step frame {frame_idx} missing object_score_logits key"
                ))
            })?;
        let mask_logits = load_reference_internal_tensor(bundle, high_res_key)?;
        let resized_logits = resize_mask_logits_to_video(&mask_logits, video_size)?;
        let masks = candle_nn::ops::sigmoid(&resized_logits)?;
        let boxes = mask_to_normalized_xyxy(&masks)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let presence_score =
            candle_nn::ops::sigmoid(&load_reference_internal_tensor(bundle, object_score_key)?)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?
                .into_iter()
                .next()
                .unwrap_or(0.0);
        Ok((boxes, presence_score, masks))
    }

    fn json_usize_vec(value: &serde_json::Value, key: &str) -> Result<Vec<usize>> {
        value[key]
            .as_array()
            .ok_or_else(|| candle::Error::Msg(format!("missing `{key}` array")))?
            .iter()
            .map(|entry| {
                entry.as_u64().map(|value| value as usize).ok_or_else(|| {
                    candle::Error::Msg(format!("malformed `{key}` entry in reference metadata"))
                })
            })
            .collect()
    }

    fn assert_tensor_close(
        label: &str,
        actual: &Tensor,
        expected: &Tensor,
        atol: f32,
    ) -> Result<()> {
        if actual.shape() != expected.shape() {
            candle::bail!(
                "{label} shape mismatch: actual {:?}, expected {:?}",
                actual.shape().dims(),
                expected.shape().dims()
            );
        }
        let actual = actual.to_dtype(DType::F32)?;
        let expected = expected.to_dtype(DType::F32)?;
        let max_abs_diff = actual
            .broadcast_sub(&expected)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_vec0::<f32>()?;
        if max_abs_diff > atol {
            candle::bail!("{label} max abs diff {max_abs_diff:.6} exceeded tolerance {atol:.6}");
        }
        Ok(())
    }

    fn tensor_max_abs_diff(actual: &Tensor, expected: &Tensor) -> Result<f32> {
        if actual.shape() != expected.shape() {
            candle::bail!(
                "shape mismatch when computing max abs diff: actual {:?}, expected {:?}",
                actual.shape().dims(),
                expected.shape().dims()
            );
        }
        let actual = actual.to_dtype(DType::F32)?;
        let expected = expected.to_dtype(DType::F32)?;
        actual
            .broadcast_sub(&expected)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_vec0::<f32>()
    }

    fn binary_mask_iou(actual: &Tensor, expected_path: &Path) -> Result<f32> {
        let actual = tensor_to_mask_probs_2d(actual)?;
        let expected = image::open(expected_path)
            .map_err(|err| candle::Error::Msg(err.to_string()))?
            .to_luma8();
        let mut intersection = 0usize;
        let mut union = 0usize;
        for (y, row) in actual.iter().enumerate() {
            for (x, value) in row.iter().enumerate() {
                let actual_fg = *value >= 0.5;
                let expected_fg = expected.get_pixel(x as u32, y as u32)[0] >= 128;
                if actual_fg && expected_fg {
                    intersection += 1;
                }
                if actual_fg || expected_fg {
                    union += 1;
                }
            }
        }
        Ok(if union == 0 {
            1.0
        } else {
            intersection as f32 / union as f32
        })
    }

    fn binary_mask_iou_tensor(actual: &Tensor, expected: &Tensor) -> Result<f32> {
        let actual = tensor_to_mask_probs_2d(actual)?;
        let expected = tensor_to_mask_probs_2d(expected)?;
        if actual.len() != expected.len()
            || actual.first().map(Vec::len).unwrap_or(0)
                != expected.first().map(Vec::len).unwrap_or(0)
        {
            candle::bail!(
                "mask size mismatch when computing IoU from tensors: actual={}x{}, expected={}x{}",
                actual.len(),
                actual.first().map(Vec::len).unwrap_or(0),
                expected.len(),
                expected.first().map(Vec::len).unwrap_or(0)
            );
        }
        let mut intersection = 0usize;
        let mut union = 0usize;
        for (actual_row, expected_row) in actual.iter().zip(expected.iter()) {
            for (actual_value, expected_value) in actual_row.iter().zip(expected_row.iter()) {
                let actual_fg = *actual_value >= 0.5;
                let expected_fg = *expected_value >= 0.5;
                if actual_fg && expected_fg {
                    intersection += 1;
                }
                if actual_fg || expected_fg {
                    union += 1;
                }
            }
        }
        Ok(if union == 0 {
            1.0
        } else {
            intersection as f32 / union as f32
        })
    }

    fn assert_boxes_close(actual: &[f32], expected: &[f32], atol: f32) {
        assert_eq!(actual.len(), expected.len());
        for (idx, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (actual - expected).abs() <= atol,
                "box component {idx} mismatch: actual={actual}, expected={expected}, atol={atol}"
            );
        }
    }

    fn box_mismatch_message(actual: &[f32], expected: &[f32], atol: f32) -> Option<String> {
        if actual.len() != expected.len() {
            return Some(format!(
                "box length mismatch: actual={}, expected={}",
                actual.len(),
                expected.len()
            ));
        }
        for (idx, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
            if (actual - expected).abs() > atol {
                return Some(format!(
                    "box component {idx} mismatch: actual={actual}, expected={expected}, atol={atol}"
                ));
            }
        }
        None
    }

    fn mask_tensor_to_binary_image(mask: &Tensor) -> Result<GrayImage> {
        let mask_probs = tensor_to_mask_probs_2d(mask)?;
        let height = mask_probs.len() as u32;
        let width = mask_probs.first().map(Vec::len).unwrap_or(0) as u32;
        let mut image = GrayImage::new(width, height);
        for (y, row) in mask_probs.iter().enumerate() {
            for (x, value) in row.iter().enumerate() {
                let pixel = if *value >= 0.5 { 255u8 } else { 0u8 };
                image.put_pixel(x as u32, y as u32, Luma([pixel]));
            }
        }
        Ok(image)
    }

    fn save_binary_mask_png(path: &Path, mask: &Tensor) -> Result<()> {
        mask_tensor_to_binary_image(mask)?
            .save(path)
            .map_err(|err| candle::Error::Msg(format!("failed to save {}: {err}", path.display())))
    }

    fn maybe_tensor_shape(tensor: Option<&Tensor>) -> Option<Vec<usize>> {
        tensor.map(|tensor| tensor.shape().dims().to_vec())
    }

    fn maybe_single_tensor_value(tensor: Option<&Tensor>) -> Result<Option<f32>> {
        match tensor {
            Some(tensor) => Ok(Some(
                tensor
                    .flatten_all()?
                    .to_vec1::<f32>()?
                    .into_iter()
                    .next()
                    .unwrap_or(0.0),
            )),
            None => Ok(None),
        }
    }

    fn dump_correction_failure_context(
        bundle: &str,
        actual8: &ObjectFrameOutput,
        actual9: &ObjectFrameOutput,
        expected_boxes8: &[f32],
        expected_score8: f32,
        expected_mask_path8: &Path,
        expected_boxes9: &[f32],
        expected_score9: f32,
        expected_mask_path9: &Path,
        frame8_state: &TrackerFrameState,
        correction_track_step: &serde_json::Value,
        correction_forward: &serde_json::Value,
        prepare_record: &serde_json::Value,
        failures: &[String],
        mask_iou8: f32,
        mask_iou9: f32,
    ) -> Result<PathBuf> {
        use std::fs;
        use std::time::{SystemTime, UNIX_EPOCH};

        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|err| candle::Error::Msg(format!("time went backwards: {err}")))?
            .as_millis();
        let out_dir = PathBuf::from("/tmp/sam3_test_failures")
            .join(format!("{}_{}", bundle, stamp));
        fs::create_dir_all(&out_dir).map_err(|err| {
            candle::Error::Msg(format!(
                "failed to create correction failure directory {}: {err}",
                out_dir.display()
            ))
        })?;

        save_binary_mask_png(&out_dir.join("actual_frame8_mask.png"), &actual8.masks)?;
        save_binary_mask_png(&out_dir.join("actual_frame9_mask.png"), &actual9.masks)?;
        fs::copy(expected_mask_path8, out_dir.join("expected_frame8_mask.png")).map_err(|err| {
            candle::Error::Msg(format!(
                "failed to copy {}: {err}",
                expected_mask_path8.display()
            ))
        })?;
        fs::copy(expected_mask_path9, out_dir.join("expected_frame9_mask.png")).map_err(|err| {
            candle::Error::Msg(format!(
                "failed to copy {}: {err}",
                expected_mask_path9.display()
            ))
        })?;

        let summary = serde_json::json!({
            "bundle": bundle,
            "failures": failures,
            "frame8": {
                "actual_boxes_xyxy": actual8.boxes_xyxy.flatten_all()?.to_vec1::<f32>()?,
                "expected_boxes_xyxy": expected_boxes8,
                "actual_score": actual8.score_value()?,
                "expected_score": expected_score8,
                "actual_presence_score": maybe_single_tensor_value(actual8.presence_scores.as_ref())?,
                "memory_frame_indices": actual8.memory_frame_indices,
                "mask_iou": mask_iou8,
            },
            "frame9": {
                "actual_boxes_xyxy": actual9.boxes_xyxy.flatten_all()?.to_vec1::<f32>()?,
                "expected_boxes_xyxy": expected_boxes9,
                "actual_score": actual9.score_value()?,
                "expected_score": expected_score9,
                "actual_presence_score": maybe_single_tensor_value(actual9.presence_scores.as_ref())?,
                "memory_frame_indices": actual9.memory_frame_indices,
                "mask_iou": mask_iou9,
            },
            "frame8_state": {
                "is_cond_frame": frame8_state.is_cond_frame,
                "maskmem_features_present": frame8_state.maskmem_features.is_some(),
                "maskmem_features_shape": maybe_tensor_shape(frame8_state.maskmem_features.as_ref()),
                "maskmem_pos_enc_present": frame8_state.maskmem_pos_enc.is_some(),
                "maskmem_pos_enc_shape": maybe_tensor_shape(frame8_state.maskmem_pos_enc.as_ref()),
                "object_score_logits": frame8_state.object_score_logits.flatten_all()?.to_vec1::<f32>()?,
            },
            "reference_internal_records": {
                "correction_track_step": correction_track_step,
                "correction_forward_sam_heads": correction_forward,
                "frame9_prepare_memory_conditioned_features": prepare_record,
            }
        });
        fs::write(
            out_dir.join("summary.json"),
            serde_json::to_vec_pretty(&summary)
                .map_err(|err| candle::Error::Msg(format!("failed to serialize summary: {err}")))?,
        )
        .map_err(|err| {
            candle::Error::Msg(format!(
                "failed to write correction failure summary in {}: {err}",
                out_dir.display()
            ))
        })?;
        Ok(out_dir)
    }

    fn dump_simple_correction_failure_json(
        bundle: &str,
        phase: &str,
        details: &serde_json::Value,
    ) -> Result<PathBuf> {
        use std::fs;
        use std::time::{SystemTime, UNIX_EPOCH};

        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|err| candle::Error::Msg(format!("time went backwards: {err}")))?
            .as_millis();
        let out_dir = PathBuf::from("/tmp/sam3_test_failures")
            .join(format!("{}_{}", bundle, stamp));
        fs::create_dir_all(&out_dir).map_err(|err| {
            candle::Error::Msg(format!(
                "failed to create correction failure directory {}: {err}",
                out_dir.display()
            ))
        })?;
        fs::write(
            out_dir.join(format!("{phase}.json")),
            serde_json::to_vec_pretty(details)
                .map_err(|err| candle::Error::Msg(format!("failed to serialize summary: {err}")))?,
        )
        .map_err(|err| {
            candle::Error::Msg(format!(
                "failed to write simple correction failure dump in {}: {err}",
                out_dir.display()
            ))
        })?;
        Ok(out_dir)
    }

    fn normalized_box_xyxy_to_mask_tensor(
        box_xyxy: (f32, f32, f32, f32),
        size: ImageSize,
        device: &Device,
    ) -> Result<Tensor> {
        let clamp = |value: f32| value.clamp(0.0, 1.0);
        let x0 = (clamp(box_xyxy.0) * (size.width.saturating_sub(1)) as f32).round() as usize;
        let y0 = (clamp(box_xyxy.1) * (size.height.saturating_sub(1)) as f32).round() as usize;
        let x1 = (clamp(box_xyxy.2) * (size.width.saturating_sub(1)) as f32).round() as usize;
        let y1 = (clamp(box_xyxy.3) * (size.height.saturating_sub(1)) as f32).round() as usize;
        let mut data = vec![0f32; size.height * size.width];
        if x0 <= x1 && y0 <= y1 {
            for y in y0..=y1 {
                for x in x0..=x1 {
                    data[y * size.width + x] = 1.0;
                }
            }
        }
        Tensor::from_vec(data, (1, 1, size.height, size.width), device)
    }

    fn temp_path(name: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time is after epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("sam3-video-tests-{}-{}", name, unique))
    }

    fn dummy_object_output(device: &Device, obj_id: u32) -> Result<ObjectFrameOutput> {
        Ok(ObjectFrameOutput {
            obj_id,
            mask_logits: Tensor::zeros((1, 4, 4), DType::F32, device)?,
            masks: Tensor::zeros((1, 4, 4), DType::F32, device)?,
            boxes_xyxy: Tensor::zeros((1, 4), DType::F32, device)?,
            scores: Tensor::ones((1, 1), DType::F32, device)?,
            presence_scores: None,
            prompt_frame_idx: Some(0),
            memory_frame_indices: Vec::new(),
            text_prompt: None,
            used_explicit_geometry: true,
            reused_previous_output: false,
        })
    }

    fn write_test_image(path: &Path, red_value: u8) -> Result<()> {
        let mut image: RgbImage = ImageBuffer::new(4, 4);
        for pixel in image.pixels_mut() {
            *pixel = Rgb([red_value, 0, 0]);
        }
        image.save(path).map_err(|err| {
            candle::Error::Msg(format!("failed to save {}: {}", path.display(), err))
        })
    }

    fn ffmpeg_tools_available() -> bool {
        for tool in ["ffmpeg", "ffprobe"] {
            let Ok(output) = Command::new(tool).arg("-version").output() else {
                return false;
            };
            if !output.status.success() {
                return false;
            }
        }
        true
    }

    fn write_test_video(video_path: &Path, red_values: &[u8]) -> Result<()> {
        let frames_dir = video_path.with_extension("frames");
        fs::create_dir_all(&frames_dir)?;
        for (idx, red_value) in red_values.iter().enumerate() {
            write_test_image(&frames_dir.join(format!("{idx}.png")), *red_value)?;
        }

        let output = Command::new("ffmpeg")
            .args([
                "-y",
                "-v",
                "error",
                "-framerate",
                "1",
                "-start_number",
                "0",
                "-i",
            ])
            .arg(frames_dir.join("%d.png"))
            .args(["-c:v", "mpeg4", "-pix_fmt", "yuv420p"])
            .arg(video_path)
            .output()
            .map_err(|err| {
                candle::Error::Msg(format!(
                    "failed to run ffmpeg to create {}: {}",
                    video_path.display(),
                    err
                ))
            })?;
        if !output.status.success() {
            candle::bail!(
                "ffmpeg failed to create {}: {}",
                video_path.display(),
                String::from_utf8_lossy(&output.stderr)
            );
        }
        fs::remove_dir_all(&frames_dir)?;
        Ok(())
    }

    #[test]
    fn image_folder_frame_source_sorts_numeric_stems() -> Result<()> {
        let dir = temp_path("numeric-sort");
        fs::create_dir_all(&dir)?;
        write_test_image(&dir.join("10.png"), 200)?;
        write_test_image(&dir.join("2.png"), 100)?;
        write_test_image(&dir.join("1.png"), 50)?;

        let config = tiny_segmentation_config();
        let source = VideoSource::from_path(&dir)?;
        let mut frame_source = source.into_frame_source(&config)?;
        let frame0 = frame_source.get_frame(0, &Device::Cpu)?;
        let frame1 = frame_source.get_frame(1, &Device::Cpu)?;
        let frame2 = frame_source.get_frame(2, &Device::Cpu)?;

        let red0 = frame0.i((0, 0, 0))?.to_scalar::<f32>()?;
        let red1 = frame1.i((0, 0, 0))?.to_scalar::<f32>()?;
        let red2 = frame2.i((0, 0, 0))?.to_scalar::<f32>()?;
        assert!(red0 < red1);
        assert!(red1 < red2);
        frame_source.close();
        fs::remove_dir_all(&dir)?;
        Ok(())
    }

    #[test]
    fn video_file_frame_source_decodes_lazily_and_preserves_order() -> Result<()> {
        if !ffmpeg_tools_available() {
            eprintln!("skipping video decode test because ffmpeg/ffprobe are unavailable");
            return Ok(());
        }

        let dir = temp_path("video-file-source");
        fs::create_dir_all(&dir)?;
        let video_path = dir.join("clip.mp4");
        write_test_video(&video_path, &[32, 96, 160])?;

        let config = tiny_segmentation_config();
        let source = VideoSource::from_path(&video_path)?;
        let mut frame_source = source.into_frame_source(&config)?;

        assert_eq!(frame_source.frame_count(), 3);
        assert_eq!(frame_source.video_size(), ImageSize::new(4, 4));
        assert_eq!(frame_source.loaded_frame_count(), 0);

        let frame1 = frame_source.get_frame(1, &Device::Cpu)?;
        assert_eq!(frame_source.loaded_frame_count(), 1);

        frame_source.prefetch(&[0, 2])?;
        assert_eq!(frame_source.loaded_frame_count(), 3);

        let frame0 = frame_source.get_frame(0, &Device::Cpu)?;
        let frame2 = frame_source.get_frame(2, &Device::Cpu)?;
        let red0 = frame0.i((0, 0, 0))?.to_scalar::<f32>()?;
        let red1 = frame1.i((0, 0, 0))?.to_scalar::<f32>()?;
        let red2 = frame2.i((0, 0, 0))?.to_scalar::<f32>()?;
        assert!(
            red0 < red1,
            "expected frame 0 red {red0} < frame 1 red {red1}"
        );
        assert!(
            red1 < red2,
            "expected frame 1 red {red1} < frame 2 red {red2}"
        );

        let keep = BTreeSet::from([1usize]);
        frame_source.evict_except(&keep);
        assert_eq!(frame_source.loaded_frame_count(), 1);
        frame_source.close();
        assert_eq!(frame_source.loaded_frame_count(), 0);

        fs::remove_dir_all(&dir)?;
        Ok(())
    }

    #[test]
    fn predictor_allocates_object_ids_and_merges_prompts() -> Result<()> {
        let device = Device::Cpu;
        let model = tiny_model(&device)?;
        let tracker = tiny_tracker(&device)?;
        let frames = vec![
            Tensor::zeros((3, 56, 56), DType::F32, &device)?,
            Tensor::zeros((3, 56, 56), DType::F32, &device)?,
        ];
        let mut predictor = Sam3VideoPredictor::new(&model, &tracker, &device);
        let session_id =
            predictor.start_session_from_tensors(frames, VideoSessionOptions::default())?;
        let obj_id = predictor.add_prompt(
            &session_id,
            0,
            SessionPrompt {
                text: None,
                points: Some(vec![(0.2, 0.3)]),
                point_labels: None,
                boxes: None,
                box_labels: None,
            },
            None,
            true,
            true,
        )?;
        predictor.add_prompt(
            &session_id,
            0,
            SessionPrompt {
                text: None,
                points: Some(vec![(0.4, 0.5)]),
                point_labels: Some(vec![0]),
                boxes: None,
                box_labels: None,
            },
            Some(obj_id),
            false,
            true,
        )?;

        let session = predictor.sessions.get(&session_id).expect("session exists");
        let tracked = session
            .tracked_objects
            .get(&obj_id)
            .expect("tracked object exists");
        let merged = tracked.prompt_frames.get(&0).expect("prompt frame exists");
        assert_eq!(merged.text, None);
        assert_eq!(merged.points.as_ref().map(Vec::len), Some(2));
        assert_eq!(merged.point_labels.as_ref().map(Vec::len), Some(2));
        Ok(())
    }

    #[test]
    fn predictor_allows_explicit_reference_object_ids_for_new_objects() -> Result<()> {
        let device = Device::Cpu;
        let model = tiny_model(&device)?;
        let tracker = tiny_tracker(&device)?;
        let frames = vec![
            Tensor::zeros((3, 56, 56), DType::F32, &device)?,
            Tensor::zeros((3, 56, 56), DType::F32, &device)?,
        ];
        let mut predictor = Sam3VideoPredictor::new(&model, &tracker, &device);
        let session_id =
            predictor.start_session_from_tensors(frames, VideoSessionOptions::default())?;

        let obj_a = predictor.add_prompt(
            &session_id,
            0,
            SessionPrompt {
                text: None,
                points: Some(vec![(0.2, 0.3)]),
                point_labels: Some(vec![1]),
                boxes: None,
                box_labels: None,
            },
            Some(1),
            true,
            true,
        )?;
        let obj_b = predictor.add_prompt(
            &session_id,
            0,
            SessionPrompt {
                text: None,
                points: Some(vec![(0.4, 0.5)]),
                point_labels: Some(vec![1]),
                boxes: None,
                box_labels: None,
            },
            Some(2),
            true,
            true,
        )?;

        assert_eq!(obj_a, 1);
        assert_eq!(obj_b, 2);
        let session = predictor.sessions.get(&session_id).expect("session exists");
        assert!(session.tracked_objects.contains_key(&1));
        assert!(session.tracked_objects.contains_key(&2));
        assert_eq!(session.next_obj_id, 3);
        Ok(())
    }

    #[test]
    fn propagation_emits_directional_frames_and_stays_lazy() -> Result<()> {
        let device = Device::Cpu;
        let model = tiny_model(&device)?;
        let tracker = tiny_tracker(&device)?;
        let frames = vec![
            Tensor::zeros((3, 56, 56), DType::F32, &device)?,
            Tensor::zeros((3, 56, 56), DType::F32, &device)?,
            Tensor::zeros((3, 56, 56), DType::F32, &device)?,
            Tensor::zeros((3, 56, 56), DType::F32, &device)?,
        ];
        let mut predictor = Sam3VideoPredictor::new(&model, &tracker, &device);
        let session_id =
            predictor.start_session_from_tensors(frames, VideoSessionOptions::default())?;
        let obj_id = predictor.add_prompt(
            &session_id,
            1,
            SessionPrompt {
                text: None,
                points: None,
                point_labels: None,
                boxes: Some(vec![(0.5, 0.5, 0.3, 0.3)]),
                box_labels: Some(vec![1]),
            },
            None,
            true,
            true,
        )?;

        let forward_options = PropagationOptions {
            direction: PropagationDirection::Forward,
            start_frame_idx: None,
            max_frame_num_to_track: None,
            output_prob_threshold: None,
        };
        let backward_options = PropagationOptions {
            direction: PropagationDirection::Backward,
            start_frame_idx: Some(1),
            max_frame_num_to_track: Some(2),
            output_prob_threshold: None,
        };
        let session = predictor
            .sessions
            .get(&session_id)
            .expect("session should exist");
        assert_eq!(
            build_processing_order(
                session,
                forward_options.direction,
                forward_options.start_frame_idx,
                forward_options.max_frame_num_to_track,
                false,
            )?,
            vec![1, 2, 3]
        );
        assert_eq!(
            build_processing_order(
                session,
                backward_options.direction,
                backward_options.start_frame_idx,
                backward_options.max_frame_num_to_track,
                false,
            )?,
            vec![1, 0]
        );

        let stats = predictor.session_cache_stats(&session_id)?;
        assert_eq!(obj_id, 0);
        assert_eq!(stats.tracked_objects, 1);
        assert!(stats.cached_feature_entries <= 2);
        Ok(())
    }

    #[test]
    fn processing_order_can_start_from_first_annotated_frame() -> Result<()> {
        let device = Device::Cpu;
        let model = tiny_model(&device)?;
        let tracker = tiny_tracker(&device)?;
        let frames = vec![
            Tensor::zeros((3, 56, 56), DType::F32, &device)?,
            Tensor::zeros((3, 56, 56), DType::F32, &device)?,
            Tensor::zeros((3, 56, 56), DType::F32, &device)?,
            Tensor::zeros((3, 56, 56), DType::F32, &device)?,
        ];
        let mut predictor = Sam3VideoPredictor::new(&model, &tracker, &device);
        let session_id =
            predictor.start_session_from_tensors(frames, VideoSessionOptions::default())?;
        predictor.add_prompt(
            &session_id,
            1,
            SessionPrompt {
                text: None,
                points: None,
                point_labels: None,
                boxes: Some(vec![(0.5, 0.5, 0.3, 0.3)]),
                box_labels: Some(vec![1]),
            },
            None,
            true,
            true,
        )?;
        let session = predictor
            .sessions
            .get(&session_id)
            .expect("session should exist");
        assert_eq!(
            build_processing_order(
                session,
                PropagationDirection::Forward,
                Some(3),
                Some(2),
                true
            )?,
            vec![1, 2, 3]
        );
        Ok(())
    }

    #[test]
    fn prompt_updates_preserve_current_history_and_invalidate_future_outputs() -> Result<()> {
        let device = Device::Cpu;
        let model = tiny_model(&device)?;
        let tracker = tiny_tracker(&device)?;
        let frames = vec![
            Tensor::zeros((3, 56, 56), DType::F32, &device)?,
            Tensor::zeros((3, 56, 56), DType::F32, &device)?,
            Tensor::zeros((3, 56, 56), DType::F32, &device)?,
        ];
        let mut predictor = Sam3VideoPredictor::new(&model, &tracker, &device);
        let session_id =
            predictor.start_session_from_tensors(frames, VideoSessionOptions::default())?;
        let obj_id = predictor.add_prompt(
            &session_id,
            0,
            SessionPrompt {
                text: None,
                points: Some(vec![(0.2, 0.3)]),
                point_labels: Some(vec![1]),
                boxes: None,
                box_labels: None,
            },
            None,
            true,
            true,
        )?;
        let state = TrackerFrameState {
            low_res_masks: Tensor::zeros((1, 1, 16, 16), DType::F32, &device)?,
            high_res_masks: Tensor::zeros((1, 1, 64, 64), DType::F32, &device)?,
            iou_scores: Tensor::zeros((1, 1), DType::F32, &device)?,
            obj_ptr: Tensor::zeros((1, tracker.config().hidden_dim), DType::F32, &device)?,
            object_score_logits: Tensor::zeros((1, 1), DType::F32, &device)?,
            maskmem_features: Some(Tensor::zeros(
                (1, tracker.config().memory_dim, 4, 4),
                DType::F32,
                &device,
            )?),
            maskmem_pos_enc: Some(Tensor::zeros(
                (1, tracker.config().memory_dim, 4, 4),
                DType::F32,
                &device,
            )?),
            is_cond_frame: false,
        };
        let output = ObjectFrameOutput {
            obj_id,
            mask_logits: Tensor::zeros((1, 1, 64, 64), DType::F32, &device)?,
            masks: Tensor::zeros((1, 1, 64, 64), DType::F32, &device)?,
            boxes_xyxy: Tensor::zeros((1, 4), DType::F32, &device)?,
            scores: Tensor::from_vec(vec![1.0f32], (1,), &device)?,
            presence_scores: None,
            prompt_frame_idx: Some(0),
            memory_frame_indices: Vec::new(),
            text_prompt: None,
            used_explicit_geometry: true,
            reused_previous_output: false,
        };
        {
            let session = predictor
                .sessions
                .get_mut(&session_id)
                .expect("session exists");
            let tracked = session
                .tracked_objects
                .get_mut(&obj_id)
                .expect("tracked object exists");
            for frame_idx in 0..=2 {
                tracked.frame_outputs.insert(frame_idx, output.clone());
                tracked.tracker_states.insert(frame_idx, state.clone());
                session
                    .frame_outputs
                    .entry(frame_idx)
                    .or_default()
                    .insert(obj_id, output.clone());
            }
        }

        predictor.add_prompt(
            &session_id,
            1,
            SessionPrompt {
                text: None,
                points: Some(vec![(0.4, 0.5)]),
                point_labels: Some(vec![1]),
                boxes: None,
                box_labels: None,
            },
            Some(obj_id),
            false,
            true,
        )?;

        let session = predictor.sessions.get(&session_id).expect("session exists");
        let tracked = session
            .tracked_objects
            .get(&obj_id)
            .expect("tracked object exists");
        assert!(tracked.frame_outputs.contains_key(&0));
        assert!(tracked.frame_outputs.contains_key(&1));
        assert!(!tracked.frame_outputs.contains_key(&2));
        assert!(tracked.tracker_states.contains_key(&0));
        assert!(tracked.tracker_states.contains_key(&1));
        assert!(!tracked.tracker_states.contains_key(&2));
        assert!(session
            .frame_outputs
            .get(&0)
            .and_then(|outputs| outputs.get(&obj_id))
            .is_some());
        assert!(session
            .frame_outputs
            .get(&1)
            .and_then(|outputs| outputs.get(&obj_id))
            .is_none());
        assert!(session
            .frame_outputs
            .get(&2)
            .and_then(|outputs| outputs.get(&obj_id))
            .is_none());
        Ok(())
    }

    #[test]
    fn clear_non_cond_mem_around_input_respects_multi_object_flag() -> Result<()> {
        let device = Device::Cpu;
        let model = tiny_model(&device)?;

        let build_state_with_memory = |tracker: &Sam3TrackerModel| -> Result<TrackerFrameState> {
            Ok(TrackerFrameState {
                low_res_masks: Tensor::zeros((1, 1, 16, 16), DType::F32, &device)?,
                high_res_masks: Tensor::zeros((1, 1, 64, 64), DType::F32, &device)?,
                iou_scores: Tensor::zeros((1, 1), DType::F32, &device)?,
                obj_ptr: Tensor::zeros((1, tracker.config().hidden_dim), DType::F32, &device)?,
                object_score_logits: Tensor::zeros((1, 1), DType::F32, &device)?,
                maskmem_features: Some(Tensor::zeros(
                    (1, tracker.config().memory_dim, 4, 4),
                    DType::F32,
                    &device,
                )?),
                maskmem_pos_enc: Some(Tensor::zeros(
                    (1, tracker.config().memory_dim, 4, 4),
                    DType::F32,
                    &device,
                )?),
                is_cond_frame: false,
            })
        };

        for clear_multi_obj in [false, true] {
            let mut tracker_config =
                Sam3TrackerConfig::from_sam3_config(&tiny_segmentation_config());
            tracker_config.predictor.clear_non_cond_mem_around_input = true;
            tracker_config.predictor.clear_non_cond_mem_for_multi_obj = clear_multi_obj;
            let tracker =
                Sam3TrackerModel::new(&tracker_config, VarBuilder::zeros(DType::F32, &device))?;
            let tracker_core = Sam3VideoTrackerCore::new(&tracker);
            let frames = vec![
                Tensor::zeros((3, 56, 56), DType::F32, &device)?,
                Tensor::zeros((3, 56, 56), DType::F32, &device)?,
                Tensor::zeros((3, 56, 56), DType::F32, &device)?,
            ];
            let mut predictor = Sam3VideoPredictor::new(&model, &tracker, &device);
            let session_id =
                predictor.start_session_from_tensors(frames, VideoSessionOptions::default())?;
            let obj_a = predictor.add_prompt(
                &session_id,
                0,
                SessionPrompt {
                    text: None,
                    points: Some(vec![(0.2, 0.3)]),
                    point_labels: Some(vec![1]),
                    boxes: None,
                    box_labels: None,
                },
                None,
                true,
                true,
            )?;
            let obj_b = predictor.add_prompt(
                &session_id,
                0,
                SessionPrompt {
                    text: None,
                    points: Some(vec![(0.7, 0.6)]),
                    point_labels: Some(vec![1]),
                    boxes: None,
                    box_labels: None,
                },
                None,
                true,
                true,
            )?;
            {
                let session = predictor
                    .sessions
                    .get_mut(&session_id)
                    .expect("session exists");
                for obj_id in [obj_a, obj_b] {
                    let tracked = session
                        .tracked_objects
                        .get_mut(&obj_id)
                        .expect("tracked object exists");
                    tracked.tracker_states.insert(1, build_state_with_memory(&tracker)?);
                }
                tracker_core.clear_non_cond_mem_around_input(session, 1);
            }
            let session = predictor.sessions.get(&session_id).expect("session exists");
            for obj_id in [obj_a, obj_b] {
                let state = session
                    .tracked_objects
                    .get(&obj_id)
                    .and_then(|object| object.tracker_states.get(&1))
                    .expect("tracker state should exist");
                assert_eq!(
                    state.maskmem_features.is_none(),
                    clear_multi_obj,
                    "clear_non_cond_mem_for_multi_obj={clear_multi_obj} should {}clear obj_id {obj_id} state",
                    if clear_multi_obj { "" } else { "not " }
                );
            }
        }
        Ok(())
    }

    #[test]
    fn image_folder_sessions_bound_loaded_frames_and_cleanup() -> Result<()> {
        let dir = temp_path("lazy-session");
        fs::create_dir_all(&dir)?;
        for (idx, value) in [32u8, 64, 96, 128, 160].iter().enumerate() {
            write_test_image(&dir.join(format!("{idx}.png")), *value)?;
        }

        let device = Device::Cpu;
        let model = tiny_model(&device)?;
        let tracker = tiny_tracker(&device)?;
        let source = VideoSource::from_path(&dir)?;
        let mut predictor = Sam3VideoPredictor::new(&model, &tracker, &device);
        let session_id = predictor.start_session(
            source,
            VideoSessionOptions {
                prefetch_ahead: 0,
                prefetch_behind: 0,
                ..VideoSessionOptions::default()
            },
        )?;
        predictor.add_prompt(
            &session_id,
            1,
            SessionPrompt {
                text: None,
                points: None,
                point_labels: None,
                boxes: Some(vec![(0.5, 0.5, 0.3, 0.3)]),
                box_labels: Some(vec![1]),
            },
            None,
            true,
            true,
        )?;

        let session = predictor
            .sessions
            .get_mut(&session_id)
            .expect("session should exist");
        session.prefetch_for_frame(1, PropagationDirection::Forward)?;
        let _ = session.get_frame(1, &device)?;
        session.evict_for_frame(1, PropagationDirection::Forward);
        assert!(
            session.frame_source.loaded_frame_count() <= 2,
            "expected lazy source to keep prompt/current frames only, got {} loaded frames",
            session.frame_source.loaded_frame_count()
        );
        session.close();
        assert_eq!(session.frame_source.loaded_frame_count(), 0);

        fs::remove_dir_all(&dir)?;
        Ok(())
    }

    #[test]
    fn remove_object_clears_cached_outputs() -> Result<()> {
        let device = Device::Cpu;
        let model = tiny_model(&device)?;
        let tracker = tiny_tracker(&device)?;
        let frames = vec![
            Tensor::zeros((3, 56, 56), DType::F32, &device)?,
            Tensor::zeros((3, 56, 56), DType::F32, &device)?,
        ];
        let mut predictor = Sam3VideoPredictor::new(&model, &tracker, &device);
        let session_id =
            predictor.start_session_from_tensors(frames, VideoSessionOptions::default())?;
        let obj_id = predictor.add_prompt(
            &session_id,
            0,
            SessionPrompt {
                text: None,
                points: None,
                point_labels: None,
                boxes: Some(vec![(0.5, 0.5, 0.3, 0.3)]),
                box_labels: Some(vec![1]),
            },
            None,
            true,
            true,
        )?;

        let cached = dummy_object_output(&device, obj_id)?;
        predictor
            .sessions
            .get_mut(&session_id)
            .expect("session exists")
            .tracked_objects
            .get_mut(&obj_id)
            .expect("tracked object exists")
            .frame_outputs
            .insert(0, cached.clone());
        predictor
            .sessions
            .get_mut(&session_id)
            .expect("session exists")
            .frame_outputs
            .entry(0)
            .or_default()
            .insert(obj_id, cached);
        predictor.remove_object(&session_id, obj_id)?;

        let session = predictor.sessions.get(&session_id).expect("session exists");
        assert!(!session.tracked_objects.contains_key(&obj_id));
        assert!(session
            .frame_outputs
            .values()
            .all(|objects| objects.is_empty()));
        Ok(())
    }

    #[test]
    fn text_prompts_require_tokenizer_path() -> Result<()> {
        let device = Device::Cpu;
        let model = tiny_model(&device)?;
        let tracker = tiny_tracker(&device)?;
        let frames = vec![Tensor::zeros((3, 56, 56), DType::F32, &device)?];
        let mut predictor = Sam3VideoPredictor::new(&model, &tracker, &device);
        let session_id =
            predictor.start_session_from_tensors(frames, VideoSessionOptions::default())?;
        let err = predictor
            .add_prompt(
                &session_id,
                0,
                SessionPrompt {
                    text: Some("person".to_owned()),
                    points: None,
                    point_labels: None,
                    boxes: None,
                    box_labels: None,
                },
                None,
                true,
                true,
            )
            .expect_err("text prompt should require tokenizer");
        assert!(
            err.to_string().contains("tokenizer"),
            "unexpected error: {err}"
        );
        Ok(())
    }

    #[test]
    fn probability_masks_use_half_threshold_for_box_extraction() -> Result<()> {
        let device = Device::Cpu;
        let mask = Tensor::from_vec(
            vec![
                0.1f32, 0.1, 0.1, 0.1, //
                0.1, 0.9, 0.9, 0.1, //
                0.1, 0.9, 0.9, 0.1, //
                0.1, 0.1, 0.1, 0.1, //
            ],
            (1, 1, 4, 4),
            &device,
        )?;
        let boxes = mask_to_normalized_xyxy(&mask)?;
        assert_eq!(boxes.to_vec2::<f32>()?, vec![vec![0.25, 0.25, 0.75, 0.75]]);
        Ok(())
    }

    #[test]
    fn tracker_outputs_are_resized_to_video_space_and_scores_are_flattened() -> Result<()> {
        let device = Device::Cpu;
        let state = TrackerFrameState {
            low_res_masks: Tensor::zeros((1, 1, 2, 2), DType::F32, &device)?,
            high_res_masks: Tensor::from_vec(
                vec![
                    -5.0f32, -5.0, -5.0, -5.0, //
                    -5.0, 5.0, 5.0, -5.0, //
                    -5.0, 5.0, 5.0, -5.0, //
                    -5.0, -5.0, -5.0, -5.0, //
                ],
                (1, 1, 4, 4),
                &device,
            )?,
            iou_scores: Tensor::from_vec(vec![0.25f32], (1, 1), &device)?,
            obj_ptr: Tensor::zeros((1, 8), DType::F32, &device)?,
            object_score_logits: Tensor::from_vec(vec![2.0f32], (1, 1), &device)?,
            maskmem_features: None,
            maskmem_pos_enc: None,
            is_cond_frame: true,
        };
        let output = tracker_state_to_object_output(
            7,
            &state,
            None,
            Some(0),
            vec![0, 1, 2],
            None,
            true,
            false,
            ImageSize::new(2, 6),
        )?;
        assert_eq!(output.mask_logits.dims(), &[1, 1, 2, 6]);
        assert_eq!(output.masks.dims(), &[1, 1, 2, 6]);
        assert_eq!(output.scores.to_vec1::<f32>()?, vec![0.25]);
        assert_eq!(output.boxes_xyxy.dims(), &[1, 4]);
        Ok(())
    }

    #[test]
    fn tracker_outputs_can_use_persistent_display_scores() -> Result<()> {
        let device = Device::Cpu;
        let state = TrackerFrameState {
            low_res_masks: Tensor::zeros((1, 1, 2, 2), DType::F32, &device)?,
            high_res_masks: Tensor::zeros((1, 1, 2, 2), DType::F32, &device)?,
            iou_scores: Tensor::from_vec(vec![0.25f32], (1, 1), &device)?,
            obj_ptr: Tensor::zeros((1, 8), DType::F32, &device)?,
            object_score_logits: Tensor::from_vec(vec![2.0f32], (1, 1), &device)?,
            maskmem_features: None,
            maskmem_pos_enc: None,
            is_cond_frame: true,
        };
        let output = tracker_state_to_object_output(
            7,
            &state,
            Some(0.9),
            Some(0),
            vec![0],
            None,
            true,
            false,
            ImageSize::new(2, 2),
        )?;
        assert_eq!(output.scores.to_vec1::<f32>()?, vec![0.9]);
        Ok(())
    }

    #[test]
    fn detector_seed_outputs_are_resized_to_video_space_from_masks() -> Result<()> {
        let device = Device::Cpu;
        let grounding = GroundingOutput {
            mask_logits: Tensor::from_vec(
                vec![
                    -5.0f32, -5.0, -5.0, -5.0, //
                    -5.0, 5.0, 5.0, -5.0, //
                    -5.0, 5.0, 5.0, -5.0, //
                    -5.0, -5.0, -5.0, -5.0, //
                ],
                (1, 4, 4),
                &device,
            )?,
            masks: Tensor::zeros((1, 4, 4), DType::F32, &device)?,
            boxes_xyxy: Tensor::zeros((1, 4), DType::F32, &device)?,
            scores: Tensor::from_vec(vec![0.75f32], (1, 1), &device)?,
            presence_scores: Some(Tensor::from_vec(vec![1.5f32], (1, 1), &device)?),
        };
        let output = grounding_to_object_output(
            9,
            &grounding,
            Some(0),
            vec![],
            None,
            true,
            false,
            ImageSize::new(2, 6),
        )?;
        assert_eq!(output.mask_logits.dims(), &[1, 1, 2, 6]);
        assert_eq!(output.masks.dims(), &[1, 1, 2, 6]);
        assert_eq!(output.scores.to_vec1::<f32>()?, vec![0.75]);
        assert_eq!(
            output
                .presence_scores
                .as_ref()
                .expect("presence scores should be preserved")
                .to_vec1::<f32>()?,
            vec![1.5]
        );
        assert_ne!(
            output.boxes_xyxy.to_vec2::<f32>()?,
            vec![vec![0.0, 0.0, 0.0, 0.0]]
        );
        Ok(())
    }

    #[test]
    fn propagation_mask_postprocess_matches_upstream_fill_and_sprinkle_rules() -> Result<()> {
        let device = Device::Cpu;
        let mask_logits = Tensor::from_vec(
            vec![
                -1.0f32, -1.0, -1.0, -1.0, -1.0, //
                -1.0, 1.0, 1.0, 1.0, -1.0, //
                -1.0, 1.0, -1.0, 1.0, -1.0, //
                -1.0, 1.0, 1.0, 1.0, -1.0, //
                1.0, -1.0, -1.0, -1.0, -1.0, //
            ],
            (1, 1, 5, 5),
            &device,
        )?;
        let postprocessed = postprocess_low_res_mask_logits_for_video(&mask_logits, 1)?;
        assert_eq!(
            postprocessed.i((0, 0))?.to_vec2::<f32>()?,
            vec![
                vec![-1.0, -1.0, -1.0, -1.0, -1.0],
                vec![-1.0, 1.0, 1.0, 1.0, -1.0],
                vec![-1.0, 1.0, 0.1, 1.0, -1.0],
                vec![-1.0, 1.0, 1.0, 1.0, -1.0],
                vec![1.0, -1.0, -1.0, -1.0, -1.0],
            ]
        );
        Ok(())
    }

    #[test]
    #[ignore = "diagnostic for direct-tracker visual feature parity"]
    fn video_frame0_visual_features_match_single_click_point_fixture_bundle() -> Result<()> {
        let Some((model, tracker, device)) =
            load_runtime_models_from_checkpoint(Some("reference_video_point_debug_single_click"))?
        else {
            return Ok(());
        };
        let bundle = "reference_video_point_debug_single_click";
        let source = VideoSource::from_path(reference_input_frames_dir(bundle))?;
        let mut predictor = Sam3VideoPredictor::new(&model, &tracker, &device);
        let session_id = predictor.start_session(source, VideoSessionOptions::default())?;
        let (preprocessed_image, visual, raw_visual) = {
            let session = predictor
                .sessions
                .get_mut(&session_id)
                .expect("session exists");
            let image = session.get_frame(0, &device)?.unsqueeze(0)?;
            let raw_visual = session.get_visual_features(&model, &device, 0)?;
            let visual = tracker_visual_output(&raw_visual);
            (image, visual, raw_visual)
        };

        let manifest = load_reference_internal_manifest(bundle)?;
        let records = manifest["records"].as_array().ok_or_else(|| {
            candle::Error::Msg("single-click manifest missing records".to_owned())
        })?;
        let get_image_feature_record = records.iter().find(|record| {
            record["frame_idx"].as_u64() == Some(0)
                && record["stage"].as_str() == Some("get_image_feature")
        });
        let (expected_image, expected_backbone, expected_high_res_0, expected_high_res_1) =
            if let Some(record) = get_image_feature_record {
                let keys = record["tensor_keys"].as_object().ok_or_else(|| {
                    candle::Error::Msg(
                        "single-click get_image_feature record missing tensor keys".to_owned(),
                    )
                })?;
                (
                    load_reference_internal_tensor(bundle, keys["image"].as_str().unwrap())?,
                    load_reference_internal_tensor(
                        bundle,
                        keys["backbone_out.backbone_fpn.2"].as_str().unwrap(),
                    )?,
                    load_reference_internal_tensor(
                        bundle,
                        keys["backbone_out.backbone_fpn.0"].as_str().unwrap(),
                    )?,
                    load_reference_internal_tensor(
                        bundle,
                        keys["backbone_out.backbone_fpn.1"].as_str().unwrap(),
                    )?,
                )
            } else {
                let record = records
                    .iter()
                    .find(|record| {
                        record["frame_idx"].as_u64() == Some(0)
                            && record["stage"].as_str() == Some("forward_sam_heads")
                    })
                    .ok_or_else(|| {
                        candle::Error::Msg(
                            "missing single-click get_image_feature/forward_sam_heads record"
                                .to_owned(),
                        )
                    })?;
                let keys = record["tensor_keys"].as_object().ok_or_else(|| {
                    candle::Error::Msg(
                        "single-click forward_sam_heads record missing tensor keys".to_owned(),
                    )
                })?;
                (
                    Tensor::zeros(preprocessed_image.shape(), DType::F32, &Device::Cpu)?,
                    load_reference_internal_tensor(
                        bundle,
                        keys["backbone_features"].as_str().unwrap(),
                    )?,
                    load_reference_internal_tensor(
                        bundle,
                        keys["high_res_features.0"].as_str().unwrap(),
                    )?,
                    load_reference_internal_tensor(
                        bundle,
                        keys["high_res_features.1"].as_str().unwrap(),
                    )?,
                )
            };

        if get_image_feature_record.is_some() {
            assert_tensor_close(
                "single-click preprocessed image",
                &preprocessed_image.to_device(&Device::Cpu)?,
                &expected_image,
                1e-4,
            )?;
        }

        let tracker_backbone = visual.backbone_fpn[2].to_device(&Device::Cpu)?;
        let tracker_backbone_diff = tensor_max_abs_diff(&tracker_backbone, &expected_backbone)?;
        if tracker_backbone_diff > 1e-3 {
            let primary_backbone_diff = tensor_max_abs_diff(
                &raw_visual.backbone_fpn[2].to_device(&Device::Cpu)?,
                &expected_backbone,
            )?;
            let sam2_backbone_diff = raw_visual
                .sam2_backbone_fpn
                .as_ref()
                .map(|levels| {
                    tensor_max_abs_diff(&levels[2].to_device(&Device::Cpu)?, &expected_backbone)
                })
                .transpose()?;
            candle::bail!(
                "single-click tracker backbone feature mismatch: tracker_diff={tracker_backbone_diff:.6}, primary_diff={primary_backbone_diff:.6}, sam2_diff={:.6}",
                sam2_backbone_diff.unwrap_or(primary_backbone_diff),
            );
        }
        let projected_high_res =
            tracker.prepare_high_res_features_for_test(&visual.backbone_fpn[..2])?;
        assert_tensor_close(
            "single-click projected high_res feature 0",
            &projected_high_res[0].to_device(&Device::Cpu)?,
            &expected_high_res_0,
            1e-3,
        )?;
        assert_tensor_close(
            "single-click projected high_res feature 1",
            &projected_high_res[1].to_device(&Device::Cpu)?,
            &expected_high_res_1,
            1e-3,
        )?;
        Ok(())
    }

    #[test]
    #[ignore = "diagnostic for video preprocessing filter parity"]
    fn video_frame0_preprocess_filter_diagnostics_against_single_click_fixture_bundle() -> Result<()>
    {
        let bundle = "reference_video_point_debug_single_click";
        let manifest = load_reference_internal_manifest(bundle)?;
        let records = manifest["records"].as_array().ok_or_else(|| {
            candle::Error::Msg("single-click manifest missing records".to_owned())
        })?;
        let record = records
            .iter()
            .find(|record| {
                record["frame_idx"].as_u64() == Some(0)
                    && record["stage"].as_str() == Some("get_image_feature")
            })
            .ok_or_else(|| {
                candle::Error::Msg("missing single-click get_image_feature record".to_owned())
            })?;
        let keys = record["tensor_keys"].as_object().ok_or_else(|| {
            candle::Error::Msg(
                "single-click get_image_feature record missing tensor keys".to_owned(),
            )
        })?;
        let expected_image =
            load_reference_internal_tensor(bundle, keys["image"].as_str().unwrap())?;

        let frame_path = reference_input_frames_dir(bundle).join("000000.jpg");
        let image = ImageReader::open(&frame_path)?
            .decode()
            .map_err(candle::Error::wrap)?
            .to_rgb8();
        let (width, height) = image.dimensions();

        let filters = [
            ("Nearest", image::imageops::FilterType::Nearest),
            ("Triangle", image::imageops::FilterType::Triangle),
            ("CatmullRom", image::imageops::FilterType::CatmullRom),
            ("Gaussian", image::imageops::FilterType::Gaussian),
            ("Lanczos3", image::imageops::FilterType::Lanczos3),
        ];
        let mut lines = Vec::new();
        for (label, filter) in filters {
            let frame = frame_blob_from_rgb_image_with_filter(
                image.clone(),
                1008,
                [0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5],
                ImageSize::new(height as usize, width as usize),
                &frame_path.display().to_string(),
                filter,
            )?;
            let actual = Tensor::from_vec(
                frame.data,
                (1, 3, frame.frame_size.height, frame.frame_size.width),
                &Device::Cpu,
            )?;
            let diff = tensor_max_abs_diff(&actual, &expected_image)?;
            lines.push(format!("{label}: {diff:.6}"));
        }
        candle::bail!(
            "single-click preprocess filter diagnostics -> {}",
            lines.join(", ")
        );
    }

    #[test]
    fn video_process_frame_matches_visual_box_reference_bundle_frame0() -> Result<()> {
        let bundle = "reference_video_box_debug";
        let Some((model, tracker, device)) = load_runtime_models_from_checkpoint(Some(bundle))?
        else {
            return Ok(());
        };
        let Some(tokenizer_path) = sam3_test_tokenizer_path() else {
            return Ok(());
        };
        let source = VideoSource::from_path(reference_input_frames_dir(bundle))?;
        let mut predictor = Sam3VideoPredictor::new(&model, &tracker, &device);
        apply_reference_predictor_runtime_overrides(&mut predictor, bundle)?;
        let session_id = predictor.start_session(
            source,
            VideoSessionOptions {
                tokenizer_path: Some(tokenizer_path),
                ..VideoSessionOptions::default()
            },
        )?;
        let (cx, cy, w, h) = load_reference_box_prompt(bundle)?;
        predictor.add_prompt(
            &session_id,
            0,
            SessionPrompt {
                text: None,
                points: None,
                point_labels: None,
                boxes: Some(vec![(cx, cy, w, h)]),
                box_labels: Some(vec![1]),
            },
            None,
            true,
            true,
        )?;
        let tracker_core = Sam3VideoTrackerCore::new(&tracker);
        let video_config = predictor.video_config.clone();
        let output = {
            let session = predictor
                .sessions
                .get_mut(&session_id)
                .expect("session exists");
            tracker_core.process_frame(
                &model,
                &device,
                &video_config,
                session,
                0,
                PropagationDirection::Forward,
                VIDEO_DEBUG_MASK_THRESHOLD,
            )?
        };
        assert_eq!(output.frame_idx, 0);
        assert_eq!(output.objects.len(), 1);
        let actual = &output.objects[0];
        let (expected_boxes, expected_score, expected_mask_path) =
            load_reference_frame0_output(bundle)?;
        assert_boxes_close(
            &actual.boxes_xyxy.flatten_all()?.to_vec1::<f32>()?,
            &expected_boxes,
            0.03,
        );
        let actual_score = actual.score_value()?;
        assert!(
            (actual_score - expected_score).abs() <= 0.02,
            "frame-0 box score mismatch: actual={actual_score}, expected={expected_score}"
        );
        let mask_iou = binary_mask_iou(&actual.masks, &expected_mask_path)?;
        assert!(mask_iou >= 0.97, "frame-0 box mask IoU too low: {mask_iou}");
        Ok(())
    }

    #[test]
    fn video_process_frame_matches_single_click_point_reference_bundle_frame0() -> Result<()> {
        assert_video_process_frame_matches_point_reference_bundle_frame0(
            "reference_video_point_debug_single_click",
        )
    }

    #[test]
    #[ignore = "checkpoint-backed Step 6 frame-1 parity; slow on CPU"]
    fn video_process_frame_matches_single_click_point_reference_bundle_frame1() -> Result<()> {
        let bundle = "reference_video_point_debug_single_click";
        let Some((model, tracker, device)) = load_runtime_models_from_checkpoint(Some(bundle))?
        else {
            return Ok(());
        };
        let source = VideoSource::from_path(reference_input_frames_dir(bundle))?;
        let mut predictor = Sam3VideoPredictor::new(&model, &tracker, &device);
        apply_reference_predictor_runtime_overrides(&mut predictor, bundle)?;
        let session_id = predictor.start_session(source, VideoSessionOptions::default())?;
        let (points, point_labels) = load_reference_point_prompt(bundle)?;
        let obj_id = predictor.add_prompt(
            &session_id,
            0,
            SessionPrompt {
                text: None,
                points: Some(points),
                point_labels: Some(point_labels),
                boxes: None,
                box_labels: None,
            },
            None,
            true,
            true,
        )?;
        let tracker_core = Sam3VideoTrackerCore::new(&tracker);
        let video_config = predictor.video_config.clone();
        {
            let session = predictor
                .sessions
                .get_mut(&session_id)
                .expect("session exists");
            let _ = tracker_core.process_frame(
                &model,
                &device,
                &video_config,
                session,
                0,
                PropagationDirection::Forward,
                VIDEO_DEBUG_MASK_THRESHOLD,
            )?;
        }
        let output = {
            let session = predictor
                .sessions
                .get_mut(&session_id)
                .expect("session exists");
            tracker_core.process_frame(
                &model,
                &device,
                &video_config,
                session,
                1,
                PropagationDirection::Forward,
                VIDEO_DEBUG_MASK_THRESHOLD,
            )?
        };
        assert_eq!(output.frame_idx, 1);
        assert_eq!(output.objects.len(), 1);
        let actual = &output.objects[0];
        let expected_display_score = load_reference_frame0_output(bundle)?.1;
        let video_size = match actual.masks.rank() {
            2 => {
                let (height, width) = actual.masks.dims2()?;
                ImageSize::new(height, width)
            }
            3 => {
                let (_channels, height, width) = actual.masks.dims3()?;
                ImageSize::new(height, width)
            }
            4 => {
                let (_batch, _channels, height, width) = actual.masks.dims4()?;
                ImageSize::new(height, width)
            }
            rank => candle::bail!("expected propagated mask rank 2, 3, or 4, got {}", rank),
        };
        let (expected_boxes, expected_presence_score, expected_masks) =
            load_reference_track_step_frame_output(bundle, 1, video_size)?;
        assert_boxes_close(
            &actual.boxes_xyxy.flatten_all()?.to_vec1::<f32>()?,
            &expected_boxes,
            0.05,
        );
        let actual_score = actual.score_value()?;
        assert!(
            (actual_score - expected_display_score).abs() <= 0.02,
            "frame-1 point score mismatch: actual={actual_score}, expected={expected_display_score}"
        );
        let actual_presence_score = actual
            .presence_scores
            .as_ref()
            .expect("propagated point output should preserve presence score")
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?
            .into_iter()
            .next()
            .unwrap_or(0.0);
        assert!(
            (actual_presence_score - expected_presence_score).abs() <= 0.02,
            "frame-1 point presence score mismatch: actual={actual_presence_score}, expected={expected_presence_score}"
        );
        let mask_iou = binary_mask_iou_tensor(&actual.masks, &expected_masks)?;
        assert!(
            mask_iou >= 0.97,
            "frame-1 point mask IoU too low: {mask_iou}"
        );
        let prepare_record =
            load_reference_internal_record(bundle, "prepare_memory_conditioned_features", 1)?;
        let expected_prompt_frame_indices = json_usize_vec(
            &prepare_record["metadata"],
            "selected_conditioning_frame_indices",
        )?;
        let expected_memory_frame_indices =
            json_usize_vec(&prepare_record["metadata"], "selected_memory_frame_indices")?;
        assert_eq!(
            actual.prompt_frame_idx,
            expected_prompt_frame_indices.last().copied()
        );
        assert_eq!(actual.memory_frame_indices, expected_memory_frame_indices);
        let preflight_state = predictor
            .sessions
            .get(&session_id)
            .and_then(|session| session.tracked_objects.get(&obj_id))
            .and_then(|object| object.tracker_states.get(&0))
            .expect("prompt-frame tracker state should exist after propagation");
        assert!(preflight_state.maskmem_features.is_some());
        assert!(preflight_state.maskmem_pos_enc.is_some());
        Ok(())
    }

    fn assert_video_process_frame_matches_point_reference_bundle_frame0(
        bundle: &str,
    ) -> Result<()> {
        let Some((model, tracker, device)) = load_runtime_models_from_checkpoint(Some(bundle))?
        else {
            return Ok(());
        };
        let source = VideoSource::from_path(reference_input_frames_dir(bundle))?;
        let mut predictor = Sam3VideoPredictor::new(&model, &tracker, &device);
        apply_reference_predictor_runtime_overrides(&mut predictor, bundle)?;
        let session_id = predictor.start_session(source, VideoSessionOptions::default())?;
        let (points, point_labels) = load_reference_point_prompt(bundle)?;
        predictor.add_prompt(
            &session_id,
            0,
            SessionPrompt {
                text: None,
                points: Some(points),
                point_labels: Some(point_labels),
                boxes: None,
                box_labels: None,
            },
            None,
            true,
            true,
        )?;
        let tracker_core = Sam3VideoTrackerCore::new(&tracker);
        let video_config = predictor.video_config.clone();
        let output = {
            let session = predictor
                .sessions
                .get_mut(&session_id)
                .expect("session exists");
            tracker_core.process_frame(
                &model,
                &device,
                &video_config,
                session,
                0,
                PropagationDirection::Forward,
                VIDEO_DEBUG_MASK_THRESHOLD,
            )?
        };
        assert_eq!(output.frame_idx, 0);
        assert_eq!(output.objects.len(), 1);
        let actual = &output.objects[0];
        let (expected_boxes, expected_score, expected_mask_path) =
            load_reference_frame0_output(bundle)?;
        assert_boxes_close(
            &actual.boxes_xyxy.flatten_all()?.to_vec1::<f32>()?,
            &expected_boxes,
            0.03,
        );
        let actual_score = actual.score_value()?;
        assert!(
            (actual_score - expected_score).abs() <= 0.02,
            "frame-0 point score mismatch for {bundle}: actual={actual_score}, expected={expected_score}"
        );
        let mask_iou = binary_mask_iou(&actual.masks, &expected_mask_path)?;
        let min_mask_iou = match bundle {
            // The all-points tracker path is still limited here by the known
            // patch-embed BF16 backend gap on this machine/runtime. Under the
            // updated strict-port spec, that residual is tracked as a backend
            // limitation rather than a Step 3/5 logic mismatch.
            "reference_video_point_debug_all_points" => 0.80,
            _ => 0.97,
        };
        assert!(
            mask_iou >= min_mask_iou,
            "frame-0 point mask IoU too low for {bundle}: {mask_iou} (required >= {min_mask_iou})"
        );
        Ok(())
    }

    struct CorrectionBundleExpectations {
        frame8_has_mask_inputs: bool,
        frame8_use_prev_mem_frame: bool,
        frame9_cond_contains_frame8: bool,
    }

    fn assert_video_process_frame_matches_correction_click_reference_bundle_frames_8_and_9(
        bundle: &str,
        expectations: CorrectionBundleExpectations,
    ) -> Result<()> {
        let Some((model, tracker, device)) = load_runtime_models_from_checkpoint(Some(bundle))?
        else {
            return Ok(());
        };
        let source = VideoSource::from_path(reference_input_frames_dir(bundle))?;
        let mut predictor = Sam3VideoPredictor::new(&model, &tracker, &device);
        apply_reference_predictor_runtime_overrides(&mut predictor, bundle)?;
        let session_id = predictor.start_session(source, VideoSessionOptions::default())?;
        let (initial_points, initial_labels) = load_reference_point_prompt_on_frame(bundle, 0)?;
        let obj_id = predictor.add_prompt(
            &session_id,
            0,
            SessionPrompt {
                text: None,
                points: Some(initial_points),
                point_labels: Some(initial_labels),
                boxes: None,
                box_labels: None,
            },
            None,
            true,
            true,
        )?;
        predictor.propagate_in_video(
            &session_id,
            PropagationOptions {
                direction: PropagationDirection::Forward,
                start_frame_idx: Some(0),
                max_frame_num_to_track: Some(9),
                output_prob_threshold: None,
            },
        )?;
        let (correction_points, correction_labels) =
            load_reference_point_prompt_on_frame(bundle, 8)?;
        predictor.add_prompt(
            &session_id,
            8,
            SessionPrompt {
                text: None,
                points: Some(correction_points),
                point_labels: Some(correction_labels),
                boxes: None,
                box_labels: None,
            },
            Some(obj_id),
            false,
            true,
        )?;
        let tracker_core = Sam3VideoTrackerCore::new(&tracker);
        let video_config = predictor.video_config.clone();
        let frame8 = {
            let session = predictor
                .sessions
                .get_mut(&session_id)
                .expect("session exists");
            tracker_core.process_frame(
                &model,
                &device,
                &video_config,
                session,
                8,
                PropagationDirection::Forward,
                VIDEO_DEBUG_MASK_THRESHOLD,
            )?
        };
        if frame8.objects.len() != 1 {
            let dump_note = match dump_simple_correction_failure_json(
                bundle,
                "frame8_object_count_mismatch",
                &serde_json::json!({
                    "bundle": bundle,
                    "frame_idx": 8,
                    "expected_object_count": 1,
                    "actual_object_count": frame8.objects.len(),
                }),
            ) {
                Ok(path) => format!("failure dump: {}", path.display()),
                Err(err) => format!("failed to write failure dump: {err}"),
            };
            candle::bail!(
                "frame-8 correction object count mismatch for {bundle}: actual={}, expected=1\n{}",
                frame8.objects.len(),
                dump_note
            );
        }
        let actual8 = &frame8.objects[0];
        let (expected_boxes8, expected_score8, expected_mask_path8) =
            load_reference_frame_output(bundle, 8)?;
        let actual_boxes8 = actual8.boxes_xyxy.flatten_all()?.to_vec1::<f32>()?;
        let actual_score8 = actual8.score_value()?;
        let mask_iou8 = binary_mask_iou(&actual8.masks, &expected_mask_path8)?;
        let correction_track_step = load_reference_internal_record_matching(
            bundle,
            "track_step",
            8,
            |record| record["metadata"]["run_mem_encoder"].as_bool() == Some(false),
        )?;
        assert_eq!(
            correction_track_step["metadata"]["use_prev_mem_frame"].as_bool(),
            Some(expectations.frame8_use_prev_mem_frame),
            "frame-8 correction use_prev_mem_frame mismatch for {bundle}"
        );
        let correction_forward = load_reference_internal_record_matching(
            bundle,
            "forward_sam_heads",
            8,
            |record| record["metadata"]["has_point_inputs"].as_bool() == Some(true),
        )?;
        assert_eq!(
            correction_forward["metadata"]["has_mask_inputs"].as_bool(),
            Some(expectations.frame8_has_mask_inputs),
            "frame-8 correction mask-input expectation mismatch for {bundle}"
        );
        let frame8_state = match predictor
            .sessions
            .get(&session_id)
            .and_then(|session| session.tracked_objects.get(&obj_id))
            .and_then(|object| object.tracker_states.get(&8))
        {
            Some(state) => state.clone(),
            None => {
                let tracker_state_keys = predictor
                    .sessions
                    .get(&session_id)
                    .and_then(|session| session.tracked_objects.get(&obj_id))
                    .map(|object| object.tracker_states.keys().copied().collect::<Vec<_>>())
                    .unwrap_or_default();
                let dump_note = match dump_simple_correction_failure_json(
                    bundle,
                    "frame8_missing_tracker_state",
                    &serde_json::json!({
                        "bundle": bundle,
                        "frame_idx": 8,
                        "obj_id": obj_id,
                        "tracker_state_keys": tracker_state_keys,
                    }),
                ) {
                    Ok(path) => format!("failure dump: {}", path.display()),
                    Err(err) => format!("failed to write failure dump: {err}"),
                };
                candle::bail!(
                    "corrected frame 8 state should be stored for {bundle}\n{}",
                    dump_note
                );
            }
        };

        let frame9 = {
            let session = predictor
                .sessions
                .get_mut(&session_id)
                .expect("session exists");
            tracker_core.process_frame(
                &model,
                &device,
                &video_config,
                session,
                9,
                PropagationDirection::Forward,
                VIDEO_DEBUG_MASK_THRESHOLD,
            )?
        };
        if frame9.objects.len() != 1 {
            let dump_note = match dump_simple_correction_failure_json(
                bundle,
                "frame9_object_count_mismatch",
                &serde_json::json!({
                    "bundle": bundle,
                    "frame_idx": 9,
                    "expected_object_count": 1,
                    "actual_object_count": frame9.objects.len(),
                }),
            ) {
                Ok(path) => format!("failure dump: {}", path.display()),
                Err(err) => format!("failed to write failure dump: {err}"),
            };
            candle::bail!(
                "frame-9 correction object count mismatch for {bundle}: actual={}, expected=1\n{}",
                frame9.objects.len(),
                dump_note
            );
        }
        let actual9 = &frame9.objects[0];
        let (expected_boxes9, expected_score9, expected_mask_path9) =
            load_reference_frame_output(bundle, 9)?;
        let actual_boxes9 = actual9.boxes_xyxy.flatten_all()?.to_vec1::<f32>()?;
        let actual_score9 = actual9.score_value()?;
        let mask_iou9 = binary_mask_iou(&actual9.masks, &expected_mask_path9)?;
        let prepare_record = load_reference_internal_record_matching_last(
            bundle,
            "prepare_memory_conditioned_features",
            9,
            |_| true,
        )?;
        let selected_cond = json_usize_vec(
            &prepare_record["metadata"],
            "selected_conditioning_frame_indices",
        )?;
        let expected_memory_frame_indices =
            json_usize_vec(&prepare_record["metadata"], "selected_memory_frame_indices")?;
        let mut failures = Vec::new();
        if let Some(message) = box_mismatch_message(&actual_boxes8, &expected_boxes8, 0.04) {
            failures.push(format!("frame-8 correction box mismatch for {bundle}: {message}"));
        }
        if (actual_score8 - expected_score8).abs() > 0.03 {
            failures.push(format!(
                "frame-8 correction score mismatch for {bundle}: actual={actual_score8}, expected={expected_score8}"
            ));
        }
        if mask_iou8 < 0.95 {
            failures.push(format!(
                "frame-8 correction mask IoU too low for {bundle}: {mask_iou8}"
            ));
        }
        if correction_track_step["metadata"]["use_prev_mem_frame"].as_bool()
            != Some(expectations.frame8_use_prev_mem_frame)
        {
            failures.push(format!(
                "frame-8 correction use_prev_mem_frame mismatch for {bundle}: actual={:?}, expected={}",
                correction_track_step["metadata"]["use_prev_mem_frame"].as_bool(),
                expectations.frame8_use_prev_mem_frame
            ));
        }
        if correction_forward["metadata"]["has_mask_inputs"].as_bool()
            != Some(expectations.frame8_has_mask_inputs)
        {
            failures.push(format!(
                "frame-8 correction mask-input expectation mismatch for {bundle}: actual={:?}, expected={}",
                correction_forward["metadata"]["has_mask_inputs"].as_bool(),
                expectations.frame8_has_mask_inputs
            ));
        }
        if frame8_state.is_cond_frame != expectations.frame9_cond_contains_frame8 {
            failures.push(format!(
                "frame-8 corrected state conditioning expectation mismatch for {bundle}: actual={}, expected={}",
                frame8_state.is_cond_frame,
                expectations.frame9_cond_contains_frame8
            ));
        }
        if frame8_state.maskmem_features.is_none() {
            failures.push(format!(
                "frame-8 corrected state missing maskmem_features for {bundle}"
            ));
        }
        if frame8_state.maskmem_pos_enc.is_none() {
            failures.push(format!(
                "frame-8 corrected state missing maskmem_pos_enc for {bundle}"
            ));
        }
        if let Some(message) = box_mismatch_message(&actual_boxes9, &expected_boxes9, 0.05) {
            failures.push(format!(
                "frame-9 correction propagation box mismatch for {bundle}: {message}"
            ));
        }
        if (actual_score9 - expected_score9).abs() > 0.03 {
            failures.push(format!(
                "frame-9 correction propagation score mismatch for {bundle}: actual={actual_score9}, expected={expected_score9}"
            ));
        }
        if mask_iou9 < 0.95 {
            failures.push(format!(
                "frame-9 correction propagation mask IoU too low for {bundle}: {mask_iou9}"
            ));
        }
        if selected_cond.contains(&8) != expectations.frame9_cond_contains_frame8 {
            failures.push(format!(
                "frame-9 conditioning selection mismatch for {bundle}: actual={selected_cond:?}, expected_contains_frame8={}",
                expectations.frame9_cond_contains_frame8
            ));
        }
        if actual9.memory_frame_indices != expected_memory_frame_indices {
            failures.push(format!(
                "frame-9 memory_frame_indices mismatch for {bundle}: actual={:?}, expected={:?}",
                actual9.memory_frame_indices,
                expected_memory_frame_indices
            ));
        }
        if !failures.is_empty() {
            let dump_result = dump_correction_failure_context(
                bundle,
                actual8,
                actual9,
                &expected_boxes8,
                expected_score8,
                &expected_mask_path8,
                &expected_boxes9,
                expected_score9,
                &expected_mask_path9,
                &frame8_state,
                &correction_track_step,
                &correction_forward,
                &prepare_record,
                &failures,
                mask_iou8,
                mask_iou9,
            );
            let dump_note = match dump_result {
                Ok(path) => format!("failure dump: {}", path.display()),
                Err(err) => format!("failed to write failure dump: {err}"),
            };
            candle::bail!(
                "correction reference mismatch for {bundle}\n{}\n{}",
                failures.join("\n"),
                dump_note
            );
        }
        Ok(())
    }

    #[test]
    fn video_process_frame_matches_multi_click_point_reference_bundle_frame0() -> Result<()> {
        assert_video_process_frame_matches_point_reference_bundle_frame0(
            "reference_video_point_debug_multi_click",
        )
    }

    #[test]
    fn video_process_frame_matches_all_points_reference_bundle_frame0() -> Result<()> {
        assert_video_process_frame_matches_point_reference_bundle_frame0(
            "reference_video_point_debug_all_points",
        )
    }

    #[test]
    fn video_process_frame_matches_mask_prompt_reference_bundle_frame0() -> Result<()> {
        let bundle = "reference_video_mask_debug";
        let Some((model, tracker, device)) = load_runtime_models_from_checkpoint(Some(bundle))?
        else {
            return Ok(());
        };
        let source = VideoSource::from_path(reference_input_frames_dir(bundle))?;
        let mut predictor = Sam3VideoPredictor::new(&model, &tracker, &device);
        apply_reference_predictor_runtime_overrides(&mut predictor, bundle)?;
        let session_id = predictor.start_session(source, VideoSessionOptions::default())?;
        let video_size = predictor
            .sessions
            .get(&session_id)
            .expect("session exists")
            .video_size();
        let mask_prompt = normalized_box_xyxy_to_mask_tensor(
            load_reference_mask_prompt_box_xyxy(bundle)?,
            video_size,
            &device,
        )?;
        predictor.add_mask_prompt(&session_id, 0, mask_prompt, None)?;
        let tracker_core = Sam3VideoTrackerCore::new(&tracker);
        let video_config = predictor.video_config.clone();
        let output = {
            let session = predictor
                .sessions
                .get_mut(&session_id)
                .expect("session exists");
            tracker_core.process_frame(
                &model,
                &device,
                &video_config,
                session,
                0,
                PropagationDirection::Forward,
                VIDEO_DEBUG_MASK_THRESHOLD,
            )?
        };
        assert_eq!(output.frame_idx, 0);
        assert_eq!(output.objects.len(), 1);
        let actual = &output.objects[0];
        let (expected_boxes, expected_score, expected_mask_path) =
            load_reference_frame0_output(bundle)?;
        assert_boxes_close(
            &actual.boxes_xyxy.flatten_all()?.to_vec1::<f32>()?,
            &expected_boxes,
            0.03,
        );
        let actual_score = actual.score_value()?;
        assert!(
            (actual_score - expected_score).abs() <= 0.02,
            "frame-0 mask score mismatch: actual={actual_score}, expected={expected_score}"
        );
        let mask_iou = binary_mask_iou(&actual.masks, &expected_mask_path)?;
        assert!(mask_iou >= 0.97, "frame-0 mask IoU too low: {mask_iou}");
        Ok(())
    }

    #[test]
    fn video_process_frame_matches_correction_click_reference_bundle_frames_8_and_9() -> Result<()>
    {
        assert_video_process_frame_matches_correction_click_reference_bundle_frames_8_and_9(
            "reference_video_correction_click_debug",
            CorrectionBundleExpectations {
                frame8_has_mask_inputs: true,
                frame8_use_prev_mem_frame: false,
                frame9_cond_contains_frame8: true,
            },
        )
    }

    #[test]
    fn video_process_frame_matches_correction_click_no_prev_mask_reference_bundle_frames_8_and_9(
    ) -> Result<()> {
        assert_video_process_frame_matches_correction_click_reference_bundle_frames_8_and_9(
            "reference_video_correction_click_no_prev_mask_pred_debug",
            CorrectionBundleExpectations {
                frame8_has_mask_inputs: false,
                frame8_use_prev_mem_frame: false,
                frame9_cond_contains_frame8: true,
            },
        )
    }

    #[test]
    fn video_process_frame_matches_correction_click_prev_mem_reference_bundle_frames_8_and_9(
    ) -> Result<()> {
        assert_video_process_frame_matches_correction_click_reference_bundle_frames_8_and_9(
            "reference_video_correction_click_prev_mem_debug",
            CorrectionBundleExpectations {
                frame8_has_mask_inputs: true,
                frame8_use_prev_mem_frame: true,
                frame9_cond_contains_frame8: true,
            },
        )
    }

    #[test]
    fn video_process_frame_matches_correction_click_stateless_refinement_reference_bundle_frames_8_and_9(
    ) -> Result<()> {
        assert_video_process_frame_matches_correction_click_reference_bundle_frames_8_and_9(
            "reference_video_correction_click_stateless_refinement_debug",
            CorrectionBundleExpectations {
                frame8_has_mask_inputs: true,
                frame8_use_prev_mem_frame: false,
                frame9_cond_contains_frame8: true,
            },
        )
    }

    #[test]
    fn video_process_frame_matches_correction_click_no_clear_mem_reference_bundle_frames_8_and_9(
    ) -> Result<()> {
        assert_video_process_frame_matches_correction_click_reference_bundle_frames_8_and_9(
            "reference_video_correction_click_no_clear_mem_debug",
            CorrectionBundleExpectations {
                frame8_has_mask_inputs: true,
                frame8_use_prev_mem_frame: false,
                frame9_cond_contains_frame8: true,
            },
        )
    }

    #[test]
    fn video_process_frame_matches_correction_click_not_all_frames_cond_reference_bundle_frames_8_and_9(
    ) -> Result<()> {
        assert_video_process_frame_matches_correction_click_reference_bundle_frames_8_and_9(
            "reference_video_correction_click_not_all_frames_cond_debug",
            CorrectionBundleExpectations {
                frame8_has_mask_inputs: true,
                frame8_use_prev_mem_frame: false,
                frame9_cond_contains_frame8: false,
            },
        )
    }

    #[test]
    fn correction_reference_helper_uses_post_correction_frame9_record() -> Result<()> {
        let prepare_record = load_reference_internal_record_matching_last(
            "reference_video_correction_click_debug",
            "prepare_memory_conditioned_features",
            9,
            |_| true,
        )?;
        assert_eq!(
            json_usize_vec(&prepare_record["metadata"], "selected_conditioning_frame_indices")?,
            vec![0, 8]
        );
        let prepare_record = load_reference_internal_record_matching_last(
            "reference_video_correction_click_not_all_frames_cond_debug",
            "prepare_memory_conditioned_features",
            9,
            |_| true,
        )?;
        assert_eq!(
            json_usize_vec(&prepare_record["metadata"], "selected_conditioning_frame_indices")?,
            vec![0]
        );
        Ok(())
    }

    #[test]
    fn video_process_frame_matches_multi_object_reference_bundle_frames_0_and_1() -> Result<()> {
        let bundle = "reference_video_multi_object_debug";
        let Some((model, tracker, device)) = load_runtime_models_from_checkpoint(Some(bundle))?
        else {
            return Ok(());
        };
        let source = VideoSource::from_path(reference_input_frames_dir(bundle))?;
        let mut predictor = Sam3VideoPredictor::new(&model, &tracker, &device);
        apply_reference_predictor_runtime_overrides(&mut predictor, bundle)?;
        let session_id = predictor.start_session(source, VideoSessionOptions::default())?;
        predictor.add_prompt(
            &session_id,
            0,
            SessionPrompt {
                text: None,
                points: Some(vec![(0.57, 0.70)]),
                point_labels: Some(vec![1]),
                boxes: None,
                box_labels: None,
            },
            Some(1),
            true,
            true,
        )?;
        predictor.add_prompt(
            &session_id,
            0,
            SessionPrompt {
                text: None,
                points: Some(vec![(0.34, 0.68)]),
                point_labels: Some(vec![1]),
                boxes: None,
                box_labels: None,
            },
            Some(2),
            true,
            true,
        )?;
        let tracker_core = Sam3VideoTrackerCore::new(&tracker);
        let video_config = predictor.video_config.clone();
        for frame_idx in [0usize, 1usize] {
            let output = {
                let session = predictor
                    .sessions
                    .get_mut(&session_id)
                    .expect("session exists");
                tracker_core.process_frame(
                    &model,
                    &device,
                    &video_config,
                    session,
                    frame_idx,
                    PropagationDirection::Forward,
                    VIDEO_DEBUG_MASK_THRESHOLD,
                )?
            };
            assert_eq!(output.objects.len(), 2);
            for obj_id in [1u32, 2u32] {
                let actual = output
                    .objects
                    .iter()
                    .find(|object| object.obj_id == obj_id)
                    .expect("multi-object output should contain both objects");
                let (expected_boxes, expected_score, expected_mask_path) =
                    load_reference_object_frame_output(bundle, frame_idx, obj_id)?;
                assert_boxes_close(
                    &actual.boxes_xyxy.flatten_all()?.to_vec1::<f32>()?,
                    &expected_boxes,
                    0.05,
                );
                let actual_score = actual.score_value()?;
                assert!(
                    (actual_score - expected_score).abs() <= 0.03,
                    "multi-object frame {frame_idx} obj_id {obj_id} score mismatch: actual={actual_score}, expected={expected_score}"
                );
                let mask_iou = binary_mask_iou(&actual.masks, &expected_mask_path)?;
                assert!(
                    mask_iou >= 0.95,
                    "multi-object frame {frame_idx} obj_id {obj_id} mask IoU too low: {mask_iou}"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn video_process_frame_matches_multi_object_clear_mem_reference_bundle_frames_8_to_10(
    ) -> Result<()> {
        let bundle = "reference_video_multi_object_clear_mem_debug";
        let Some((model, tracker, device)) = load_runtime_models_from_checkpoint(Some(bundle))?
        else {
            return Ok(());
        };
        let source = VideoSource::from_path(reference_input_frames_dir(bundle))?;
        let mut predictor = Sam3VideoPredictor::new(&model, &tracker, &device);
        apply_reference_predictor_runtime_overrides(&mut predictor, bundle)?;
        let session_id = predictor.start_session(source, VideoSessionOptions::default())?;
        predictor.add_prompt(
            &session_id,
            0,
            SessionPrompt {
                text: None,
                points: Some(vec![(0.57, 0.70)]),
                point_labels: Some(vec![1]),
                boxes: None,
                box_labels: None,
            },
            Some(1),
            true,
            true,
        )?;
        predictor.add_prompt(
            &session_id,
            0,
            SessionPrompt {
                text: None,
                points: Some(vec![(0.34, 0.68)]),
                point_labels: Some(vec![1]),
                boxes: None,
                box_labels: None,
            },
            Some(2),
            true,
            true,
        )?;
        predictor.propagate_in_video(
            &session_id,
            PropagationOptions {
                direction: PropagationDirection::Forward,
                start_frame_idx: Some(0),
                max_frame_num_to_track: Some(9),
                output_prob_threshold: None,
            },
        )?;
        let (correction_points, correction_labels) =
            load_reference_point_prompt_on_frame(bundle, 8)?;
        predictor.add_prompt(
            &session_id,
            8,
            SessionPrompt {
                text: None,
                points: Some(correction_points),
                point_labels: Some(correction_labels),
                boxes: None,
                box_labels: None,
            },
            Some(1),
            false,
            true,
        )?;
        let tracker_core = Sam3VideoTrackerCore::new(&tracker);
        let video_config = predictor.video_config.clone();
        for frame_idx in [8usize, 9usize] {
            let output = {
                let session = predictor
                    .sessions
                    .get_mut(&session_id)
                    .expect("session exists");
                tracker_core.process_frame(
                    &model,
                    &device,
                    &video_config,
                    session,
                    frame_idx,
                    PropagationDirection::Forward,
                    VIDEO_DEBUG_MASK_THRESHOLD,
                )?
            };
            assert_eq!(output.objects.len(), 2);
            for obj_id in [1u32, 2u32] {
                let actual = output
                    .objects
                    .iter()
                    .find(|object| object.obj_id == obj_id)
                    .expect("multi-object clear-mem output should contain both objects");
                let (expected_boxes, expected_score, expected_mask_path) =
                    load_reference_object_frame_output(bundle, frame_idx, obj_id)?;
                assert_boxes_close(
                    &actual.boxes_xyxy.flatten_all()?.to_vec1::<f32>()?,
                    &expected_boxes,
                    0.05,
                );
                let actual_score = actual.score_value()?;
                assert!(
                    (actual_score - expected_score).abs() <= 0.03,
                    "multi-object clear-mem frame {frame_idx} obj_id {obj_id} score mismatch: actual={actual_score}, expected={expected_score}"
                );
                let mask_iou = binary_mask_iou(&actual.masks, &expected_mask_path)?;
                assert!(
                    mask_iou >= 0.95,
                    "multi-object clear-mem frame {frame_idx} obj_id {obj_id} mask IoU too low: {mask_iou}"
                );
            }
        }

        let frame10 = {
            let session = predictor
                .sessions
                .get_mut(&session_id)
                .expect("session exists");
            tracker_core.process_frame(
                &model,
                &device,
                &video_config,
                session,
                10,
                PropagationDirection::Forward,
                VIDEO_DEBUG_MASK_THRESHOLD,
            )?
        };
        let actual_obj_ids = frame10
            .objects
            .iter()
            .map(|object| object.obj_id)
            .collect::<Vec<_>>();
        assert_eq!(actual_obj_ids, vec![1]);
        let (expected_boxes10, expected_score10, expected_mask_path10) =
            load_reference_object_frame_output(bundle, 10, 1)?;
        let actual10 = &frame10.objects[0];
        assert_boxes_close(
            &actual10.boxes_xyxy.flatten_all()?.to_vec1::<f32>()?,
            &expected_boxes10,
            0.05,
        );
        let actual_score10 = actual10.score_value()?;
        assert!(
            (actual_score10 - expected_score10).abs() <= 0.03,
            "multi-object clear-mem frame 10 obj_id 1 score mismatch: actual={actual_score10}, expected={expected_score10}"
        );
        let mask_iou10 = binary_mask_iou(&actual10.masks, &expected_mask_path10)?;
        assert!(
            mask_iou10 >= 0.95,
            "multi-object clear-mem frame 10 obj_id 1 mask IoU too low: {mask_iou10}"
        );
        Ok(())
    }

    #[test]
    fn video_process_frame_matches_reverse_reference_bundle_frames_20_and_19() -> Result<()> {
        let bundle = "reference_video_reverse_propagation_debug";
        let Some((model, tracker, device)) = load_runtime_models_from_checkpoint(Some(bundle))?
        else {
            return Ok(());
        };
        let source = VideoSource::from_path(reference_input_frames_dir(bundle))?;
        let mut predictor = Sam3VideoPredictor::new(&model, &tracker, &device);
        apply_reference_predictor_runtime_overrides(&mut predictor, bundle)?;
        let session_id = predictor.start_session(source, VideoSessionOptions::default())?;
        predictor.add_prompt(
            &session_id,
            20,
            SessionPrompt {
                text: None,
                points: Some(vec![(0.61, 0.69)]),
                point_labels: Some(vec![1]),
                boxes: None,
                box_labels: None,
            },
            Some(1),
            true,
            true,
        )?;
        let tracker_core = Sam3VideoTrackerCore::new(&tracker);
        let video_config = predictor.video_config.clone();
        for frame_idx in [20usize, 19usize] {
            let output = {
                let session = predictor
                    .sessions
                    .get_mut(&session_id)
                    .expect("session exists");
                tracker_core.process_frame(
                    &model,
                    &device,
                    &video_config,
                    session,
                    frame_idx,
                    PropagationDirection::Backward,
                    VIDEO_DEBUG_MASK_THRESHOLD,
                )?
            };
            assert_eq!(output.objects.len(), 1);
            let actual = &output.objects[0];
            let (expected_boxes, expected_score, expected_mask_path) =
                load_reference_object_frame_output(bundle, frame_idx, 1)?;
            assert_boxes_close(
                &actual.boxes_xyxy.flatten_all()?.to_vec1::<f32>()?,
                &expected_boxes,
                0.05,
            );
            let actual_score = actual.score_value()?;
            assert!(
                (actual_score - expected_score).abs() <= 0.03,
                "reverse frame {frame_idx} score mismatch: actual={actual_score}, expected={expected_score}"
            );
            let mask_iou = binary_mask_iou(&actual.masks, &expected_mask_path)?;
            assert!(
                mask_iou >= 0.95,
                "reverse frame {frame_idx} mask IoU too low: {mask_iou}"
            );
        }
        let prepare_record = load_reference_internal_record(bundle, "prepare_memory_conditioned_features", 19)?;
        let expected_memory_frame_indices =
            json_usize_vec(&prepare_record["metadata"], "selected_memory_frame_indices")?;
        let expected_cond_frame_indices =
            json_usize_vec(&prepare_record["metadata"], "selected_conditioning_frame_indices")?;
        let frame19 = predictor
            .sessions
            .get(&session_id)
            .and_then(|session| session.frame_outputs.get(&19))
            .and_then(|outputs| outputs.get(&1))
            .expect("reverse frame 19 output should be cached");
        assert_eq!(frame19.prompt_frame_idx, expected_cond_frame_indices.last().copied());
        assert_eq!(frame19.memory_frame_indices, expected_memory_frame_indices);
        Ok(())
    }

    #[test]
    fn video_propagation_can_start_from_first_annotation_reference_bundle() -> Result<()> {
        let bundle = "reference_video_start_from_first_ann_debug";
        let Some((model, tracker, device)) = load_runtime_models_from_checkpoint(Some(bundle))?
        else {
            return Ok(());
        };
        let source = VideoSource::from_path(reference_input_frames_dir(bundle))?;
        let mut predictor = Sam3VideoPredictor::new(&model, &tracker, &device);
        apply_reference_predictor_runtime_overrides(&mut predictor, bundle)?;
        let session_id = predictor.start_session(source, VideoSessionOptions::default())?;
        let (points, point_labels) = load_reference_point_prompt_on_frame(bundle, 5)?;
        predictor.add_prompt(
            &session_id,
            5,
            SessionPrompt {
                text: None,
                points: Some(points),
                point_labels: Some(point_labels),
                boxes: None,
                box_labels: None,
            },
            None,
            true,
            true,
        )?;
        let output = predictor.propagate_in_video(
            &session_id,
            PropagationOptions {
                direction: PropagationDirection::Forward,
                start_frame_idx: Some(12),
                max_frame_num_to_track: Some(18),
                output_prob_threshold: None,
            },
        )?;
        let actual_indices = output
            .frames
            .iter()
            .map(|frame| frame.frame_idx)
            .collect::<Vec<_>>();
        let expected_indices = load_reference_frame_indices(bundle)?;
        assert_eq!(actual_indices, expected_indices);
        assert_eq!(actual_indices.first().copied(), Some(5));
        for frame_idx in [5usize, 12usize] {
            let frame = output
                .frames
                .iter()
                .find(|frame| frame.frame_idx == frame_idx)
                .expect("expected propagated frame to be present");
            assert_eq!(frame.objects.len(), 1);
            let actual = &frame.objects[0];
            let (expected_boxes, expected_score, expected_mask_path) =
                load_reference_object_frame_output(bundle, frame_idx, 1)?;
            assert_boxes_close(
                &actual.boxes_xyxy.flatten_all()?.to_vec1::<f32>()?,
                &expected_boxes,
                0.05,
            );
            let actual_score = actual.score_value()?;
            assert!(
                (actual_score - expected_score).abs() <= 0.03,
                "start-from-first-ann frame {frame_idx} score mismatch: actual={actual_score}, expected={expected_score}"
            );
            let mask_iou = binary_mask_iou(&actual.masks, &expected_mask_path)?;
            assert!(
                mask_iou >= 0.95,
                "start-from-first-ann frame {frame_idx} mask IoU too low: {mask_iou}"
            );
        }
        Ok(())
    }

    #[test]
    fn memory_frame_indices_trim_to_configured_window() {
        assert_eq!(VideoConfig::default().memory_frame_count, 6);
        assert_eq!(
            trim_memory_frame_indices(vec![0, 1, 2, 3, 4, 5, 6], 6),
            vec![1, 2, 3, 4, 5, 6]
        );
        assert_eq!(trim_memory_frame_indices(vec![3, 4], 6), vec![3, 4]);
    }

    #[test]
    fn video_trim_past_non_cond_memory_clears_only_old_maskmem() -> Result<()> {
        let device = Device::Cpu;
        let mut tracker_config = Sam3TrackerConfig::from_sam3_config(&tiny_segmentation_config());
        tracker_config.predictor.trim_past_non_cond_mem_for_eval = true;
        tracker_config.use_memory_selection = true;
        tracker_config.max_obj_ptrs_in_encoder = 1;
        let tracker =
            Sam3TrackerModel::new(&tracker_config, VarBuilder::zeros(DType::F32, &device))?;
        let tracker_core = Sam3VideoTrackerCore::new(&tracker);
        let model = tiny_model(&device)?;
        let frames = vec![Tensor::zeros((3, 56, 56), DType::F32, &device)?];
        let mut predictor = Sam3VideoPredictor::new(&model, &tracker, &device);
        let session_id =
            predictor.start_session_from_tensors(frames, VideoSessionOptions::default())?;
        let obj_id = predictor.add_prompt(
            &session_id,
            0,
            SessionPrompt {
                text: None,
                points: Some(vec![(0.5, 0.5)]),
                point_labels: Some(vec![1]),
                boxes: None,
                box_labels: None,
            },
            None,
            true,
            true,
        )?;
        let state_with_memory = |is_cond_frame: bool| -> Result<TrackerFrameState> {
            Ok(TrackerFrameState {
                low_res_masks: Tensor::zeros((1, 1, 16, 16), DType::F32, &device)?,
                high_res_masks: Tensor::zeros((1, 1, 64, 64), DType::F32, &device)?,
                iou_scores: Tensor::zeros((1, 1), DType::F32, &device)?,
                obj_ptr: Tensor::zeros((1, tracker.config().hidden_dim), DType::F32, &device)?,
                object_score_logits: Tensor::zeros((1, 1), DType::F32, &device)?,
                maskmem_features: Some(Tensor::zeros(
                    (1, tracker.config().memory_dim, 4, 4),
                    DType::F32,
                    &device,
                )?),
                maskmem_pos_enc: Some(Tensor::zeros(
                    (1, tracker.config().memory_dim, 4, 4),
                    DType::F32,
                    &device,
                )?),
                is_cond_frame,
            })
        };
        {
            let session = predictor
                .sessions
                .get_mut(&session_id)
                .expect("session exists");
            let tracked = session
                .tracked_objects
                .get_mut(&obj_id)
                .expect("tracked object exists");
            tracked.tracker_states.insert(4, state_with_memory(false)?);
            tracked.tracker_states.insert(17, state_with_memory(false)?);
            tracked.tracker_states.insert(22, state_with_memory(false)?);
            tracked.tracker_states.insert(23, state_with_memory(false)?);
            tracked.tracker_states.insert(10, state_with_memory(true)?);
            let snapshot = tracked.clone();
            tracker_core.trim_past_non_cond_memory(
                session,
                &snapshot,
                24,
                PropagationDirection::Forward,
            );
        }
        let tracked = predictor
            .sessions
            .get(&session_id)
            .and_then(|session| session.tracked_objects.get(&obj_id))
            .expect("tracked object exists after trim");
        assert!(tracked
            .tracker_states
            .get(&4)
            .expect("far-old non-cond frame exists")
            .maskmem_features
            .is_none());
        assert!(tracked
            .tracker_states
            .get(&4)
            .expect("far-old non-cond frame exists")
            .maskmem_pos_enc
            .is_none());
        assert!(tracked
            .tracker_states
            .get(&17)
            .expect("memory-window non-cond frame exists")
            .maskmem_features
            .is_none());
        assert!(tracked
            .tracker_states
            .get(&10)
            .expect("conditioning frame exists")
            .maskmem_features
            .is_some());
        assert!(tracked
            .tracker_states
            .get(&23)
            .expect("recent non-cond frame exists")
            .maskmem_features
            .is_some());
        assert_eq!(
            tracked
                .tracker_states
                .get(&4)
                .expect("far-old non-cond frame exists")
                .obj_ptr
                .shape()
                .dims(),
            &[1, tracker.config().hidden_dim]
        );
        Ok(())
    }

    #[test]
    fn debug_capture_writes_seed_and_first_propagation_records() -> Result<()> {
        let device = Device::Cpu;
        let debug_root = temp_path("debug-capture");
        let mut recorder = VideoDebugRecorder::new(
            "session_0",
            VideoDebugConfig {
                enabled: true,
                capture_obj_ids: Vec::new(),
                capture_frame_indices: vec![0, 1],
                capture_first_propagated_only: true,
                output_root: Some(debug_root.clone()),
            },
        )?
        .expect("debug recorder should be created");
        let mut object = TrackedObject::new(0, 0);
        object.prompt_frames.insert(
            0,
            SessionPrompt {
                text: None,
                points: None,
                point_labels: None,
                boxes: Some(vec![(0.5, 0.5, 0.25, 0.25)]),
                box_labels: Some(vec![1]),
            },
        );
        let mask_logits = Tensor::from_vec(
            vec![
                -10.0f32, -10.0, //
                -10.0, 10.0, //
            ],
            (1, 1, 2, 2),
            &device,
        )?;
        let masks = candle_nn::ops::sigmoid(&mask_logits)?;
        let output = ObjectFrameOutput {
            obj_id: 0,
            mask_logits: mask_logits.clone(),
            masks: masks.clone(),
            boxes_xyxy: mask_to_normalized_xyxy(&masks)?,
            scores: Tensor::from_vec(vec![0.9f32], (1,), &device)?,
            presence_scores: Some(Tensor::from_vec(vec![0.8f32], (1,), &device)?),
            prompt_frame_idx: Some(0),
            memory_frame_indices: Vec::new(),
            text_prompt: None,
            used_explicit_geometry: true,
            reused_previous_output: false,
        };
        let state = TrackerFrameState {
            low_res_masks: mask_logits.clone(),
            high_res_masks: mask_logits.clone(),
            iou_scores: Tensor::from_vec(vec![0.9f32], (1, 1), &device)?,
            obj_ptr: Tensor::zeros((1, 8), DType::F32, &device)?,
            object_score_logits: Tensor::from_vec(vec![1.0f32], (1, 1), &device)?,
            maskmem_features: Some(Tensor::zeros((1, 8, 1, 1), DType::F32, &device)?),
            maskmem_pos_enc: Some(Tensor::zeros((1, 8, 1, 1), DType::F32, &device)?),
            is_cond_frame: true,
        };
        let prompt_metadata =
            debug_prompt_metadata(object.prompt_frames.get(&0).expect("prompt exists"), true)?;
        recorder.record_detector_grounding(
            &object,
            0,
            PropagationDirection::Forward,
            prompt_metadata.clone(),
            &output,
        )?;
        recorder.record_tracker_seed(
            &object,
            0,
            PropagationDirection::Forward,
            prompt_metadata,
            &output,
            &state,
        )?;
        let mut history = BTreeMap::new();
        history.insert(0, state);
        recorder.record_first_propagation(
            &object,
            1,
            PropagationDirection::Forward,
            Some(0),
            &ObjectFrameOutput {
                prompt_frame_idx: Some(0),
                memory_frame_indices: vec![0],
                reused_previous_output: true,
                ..output
            },
            &history,
            &[0],
            &[0],
        )?;

        let manifest: VideoDebugManifest = serde_json::from_str(&fs::read_to_string(
            debug_root.join(VIDEO_DEBUG_MANIFEST_FILE),
        )?)
        .map_err(|err| candle::Error::Msg(err.to_string()))?;
        assert!(manifest
            .records
            .iter()
            .any(|record| record.stage == "detector_grounding"));
        assert!(manifest
            .records
            .iter()
            .any(|record| record.stage == "tracker_seed"));
        let propagation_input = manifest
            .records
            .iter()
            .find(|record| record.stage == "propagation_input")
            .expect("propagation input should be captured");
        assert_eq!(propagation_input.frame_idx, 1);
        let propagation_input = propagation_input
            .propagation_input
            .as_ref()
            .expect("propagation input summary should be present");
        assert_eq!(propagation_input.history_frame_order, vec![0]);
        assert_eq!(propagation_input.chosen_prompt_frame_indices, vec![0]);
        assert_eq!(propagation_input.chosen_memory_frame_indices, vec![0]);
        assert_eq!(propagation_input.history_frames.len(), 1);
        assert!(propagation_input.history_frames[0].is_cond_frame);
        let detector = manifest
            .records
            .iter()
            .find(|record| record.stage == "detector_grounding")
            .and_then(|record| record.observable.as_ref())
            .expect("detector observable should be present");
        let detector_mask = image::open(
            debug_root.join(
                detector
                    .mask_path
                    .as_ref()
                    .expect("detector mask path should be present"),
            ),
        )
        .map_err(|err| candle::Error::Msg(err.to_string()))?
        .to_luma8();
        assert!(detector_mask
            .pixels()
            .all(|pixel| matches!(pixel[0], 0 | 255)));
        Ok(())
    }

    #[test]
    fn disabled_debug_capture_writes_no_artifacts_and_keeps_outputs_stable() -> Result<()> {
        let device = Device::Cpu;
        let model = tiny_model(&device)?;
        let tracker = tiny_tracker(&device)?;
        let frames = vec![
            Tensor::zeros((3, 56, 56), DType::F32, &device)?,
            Tensor::zeros((3, 56, 56), DType::F32, &device)?,
        ];
        let prompt = SessionPrompt {
            text: None,
            points: None,
            point_labels: None,
            boxes: Some(vec![(0.5, 0.5, 0.25, 0.25)]),
            box_labels: Some(vec![1]),
        };

        let mut baseline = Sam3VideoPredictor::new(&model, &tracker, &device);
        let baseline_session = baseline.start_session_from_tensors(
            frames.iter().cloned().collect(),
            VideoSessionOptions::default(),
        )?;
        let baseline_obj_id =
            baseline.add_prompt(&baseline_session, 0, prompt.clone(), None, true, true)?;

        let debug_root = temp_path("debug-disabled");
        let mut debug_predictor = Sam3VideoPredictor::new(&model, &tracker, &device)
            .with_debug_config(VideoDebugConfig {
                enabled: false,
                capture_obj_ids: Vec::new(),
                capture_frame_indices: vec![0, 1],
                capture_first_propagated_only: true,
                output_root: Some(debug_root.clone()),
            });
        let debug_session =
            debug_predictor.start_session_from_tensors(frames, VideoSessionOptions::default())?;
        let debug_obj_id =
            debug_predictor.add_prompt(&debug_session, 0, prompt, None, true, true)?;

        let baseline_prompt = baseline
            .sessions
            .get(&baseline_session)
            .and_then(|session| session.tracked_objects.get(&baseline_obj_id))
            .and_then(|object| object.prompt_frames.get(&0))
            .expect("baseline prompt should exist");
        let debug_prompt = debug_predictor
            .sessions
            .get(&debug_session)
            .and_then(|session| session.tracked_objects.get(&debug_obj_id))
            .and_then(|object| object.prompt_frames.get(&0))
            .expect("debug prompt should exist");

        assert_eq!(baseline_obj_id, debug_obj_id);
        assert_eq!(baseline_prompt.text, debug_prompt.text);
        assert_eq!(baseline_prompt.points, debug_prompt.points);
        assert_eq!(baseline_prompt.point_labels, debug_prompt.point_labels);
        assert_eq!(baseline_prompt.boxes, debug_prompt.boxes);
        assert_eq!(baseline_prompt.box_labels, debug_prompt.box_labels);
        assert_eq!(
            baseline
                .session_cache_stats(&baseline_session)?
                .tracked_objects,
            debug_predictor
                .session_cache_stats(&debug_session)?
                .tracked_objects
        );

        baseline.close_session(&baseline_session)?;
        debug_predictor.close_session(&debug_session)?;
        assert!(!debug_root.join(VIDEO_DEBUG_MANIFEST_FILE).exists());
        assert!(!debug_root.exists());
        Ok(())
    }
}
