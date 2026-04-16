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
    Config, Sam3ImageModel, Sam3TrackerModel, TrackerFrameState,
};

const CLIP_EOT_TOKEN: &str = "<|endoftext|>";
const VIDEO_DEBUG_MANIFEST_FILE: &str = "debug_manifest.json";
const VIDEO_DEBUG_MASK_THRESHOLD: f32 = 0.5;

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
            "detector_grounding" | "tracker_seed" => object.prompt_frames.contains_key(&frame_idx),
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
    pub has_inference_history: bool,
    pub prompt_frames: BTreeMap<usize, SessionPrompt>,
    pub frame_outputs: BTreeMap<usize, ObjectFrameOutput>,
    pub tracker_states: BTreeMap<usize, TrackerFrameState>,
}

impl TrackedObject {
    fn new(obj_id: u32, creation_frame: usize) -> Self {
        Self {
            obj_id,
            creation_frame,
            last_updated_frame: creation_frame,
            has_inference_history: false,
            prompt_frames: BTreeMap::new(),
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
        self.last_updated_frame = frame_idx;
        self.frame_outputs.clear();
        self.tracker_states.clear();
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
            }
            PropagationDirection::Backward => {
                self.prompt_frames.range(frame_idx..).next().is_some()
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
            .flat_map(|object| object.prompt_frames.keys().copied())
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

        let obj_id = match obj_id {
            Some(obj_id) => obj_id,
            None => {
                if self.tracked_objects.len() >= max_objects {
                    candle::bail!(
                        "cannot allocate another tracked object because max_objects={} was reached",
                        max_objects
                    )
                }
                self.allocate_object(frame_idx)
            }
        };

        let tracked = self
            .tracked_objects
            .get_mut(&obj_id)
            .ok_or_else(|| candle::Error::Msg(format!("unknown obj_id {}", obj_id)))?;
        tracked.add_prompt(frame_idx, prompt, clear_old_points, clear_old_boxes);
        self.invalidate_object_outputs(obj_id);
        Ok(obj_id)
    }

    fn invalidate_object_outputs(&mut self, obj_id: u32) {
        if let Some(object) = self.tracked_objects.get_mut(&obj_id) {
            object.frame_outputs.clear();
            object.tracker_states.clear();
        }
        let mut empty_frames = Vec::new();
        for (frame_idx, frame_outputs) in self.frame_outputs.iter_mut() {
            frame_outputs.remove(&obj_id);
            if frame_outputs.is_empty() {
                empty_frames.push(*frame_idx);
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
        self.invalidate_object_outputs(obj_id);
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
    device: &'a Device,
    video_config: VideoConfig,
    debug_config: VideoDebugConfig,
    sessions: HashMap<String, Sam3VideoSession>,
    next_session_id: usize,
    backend: Box<dyn VideoTrackerBackend + 'a>,
}

impl<'a> Sam3VideoPredictor<'a> {
    pub fn new(model: &'a Sam3ImageModel, device: &'a Device) -> Self {
        Self {
            model,
            device,
            video_config: VideoConfig::default(),
            debug_config: VideoDebugConfig::default(),
            sessions: HashMap::new(),
            next_session_id: 0,
            backend: Box::new(HeuristicVideoTrackerBackend::default()),
        }
    }

    pub fn with_config(mut self, config: VideoConfig) -> Self {
        self.video_config = config;
        self
    }

    pub fn with_backend(mut self, backend: Box<dyn VideoTrackerBackend + 'a>) -> Self {
        self.backend = backend;
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
        )?;
        let output_threshold = options
            .output_prob_threshold
            .unwrap_or(self.video_config.score_threshold);
        for frame_idx in processing_order {
            session.prefetch_for_frame(frame_idx, options.direction)?;
            let output = self.backend.process_frame(
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

pub trait VideoTrackerBackend {
    fn process_frame(
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

#[derive(Debug)]
pub struct Sam3MemoryAttentionVideoTrackerBackend<'a> {
    tracker: &'a Sam3TrackerModel,
}

impl<'a> Sam3MemoryAttentionVideoTrackerBackend<'a> {
    pub fn new(tracker: &'a Sam3TrackerModel) -> Self {
        Self { tracker }
    }

    fn history_on_device(
        &self,
        object: &TrackedObject,
        frame_idx: usize,
        direction: PropagationDirection,
        compute_device: &Device,
    ) -> Result<BTreeMap<usize, TrackerFrameState>> {
        object
            .tracker_history(frame_idx, direction)
            .into_iter()
            .map(|(idx, state)| Ok((idx, state.to_storage_device(compute_device)?)))
            .collect()
    }

    fn store_object_result(
        &self,
        session: &mut Sam3VideoSession,
        frame_idx: usize,
        output: ObjectFrameOutput,
        state: TrackerFrameState,
    ) -> Result<ObjectFrameOutput> {
        let stored_output = output.to_storage_device(session.storage_device())?;
        let stored_state = state.to_storage_device(session.storage_device())?;
        let tracked = session
            .tracked_objects
            .get_mut(&stored_output.obj_id)
            .ok_or_else(|| {
                candle::Error::Msg(format!("unknown obj_id {}", stored_output.obj_id))
            })?;
        tracked
            .frame_outputs
            .insert(frame_idx, stored_output.clone());
        tracked.tracker_states.insert(frame_idx, stored_state);
        tracked.last_updated_frame = frame_idx;
        tracked.has_inference_history = true;
        session
            .frame_outputs
            .entry(frame_idx)
            .or_default()
            .insert(stored_output.obj_id, stored_output.clone());
        Ok(stored_output)
    }

    fn seed_with_detector(
        &self,
        model: &Sam3ImageModel,
        compute_device: &Device,
        config: &VideoConfig,
        session: &mut Sam3VideoSession,
        object: &TrackedObject,
        frame_idx: usize,
        direction: PropagationDirection,
        prompt: &SessionPrompt,
    ) -> Result<ObjectFrameOutput> {
        let visual = session.get_visual_features(model, compute_device, frame_idx)?;
        let use_visual_box_prompt = uses_initial_visual_box_prompt(object, frame_idx, prompt);
        let text_encoding = match prompt.text.as_deref() {
            Some(text_prompt) => {
                Some(session.cached_text_encoding(model, text_prompt, compute_device)?)
            }
            None if use_visual_box_prompt && session.tokenizer.is_some() => {
                Some(session.cached_text_encoding(model, "visual", compute_device)?)
            }
            None => None,
        };
        let geometry_encoding = if prompt.has_geometry() {
            let geometry_prompt = session_prompt_to_geometry(prompt, compute_device)?;
            Some(model.encode_geometry_prompt(&geometry_prompt, &visual)?)
        } else {
            None
        };
        let encoded_prompt =
            combine_encoded_prompts(text_encoding.as_ref(), geometry_encoding.as_ref())?
                .ok_or_else(|| {
                    candle::Error::Msg(
                        "detector seeding requires a text or geometry prompt".to_owned(),
                    )
                })?;
        let grounding = ground_from_encoded_prompt(model, &visual, &encoded_prompt)?;
        let detector_output = grounding_to_object_output(
            object.obj_id,
            &grounding,
            Some(frame_idx),
            Vec::new(),
            prompt.text.clone(),
            prompt.has_geometry(),
            false,
            session.video_size(),
        )?;
        let prompt_metadata =
            debug_prompt_metadata(prompt, prompt.text.is_none() && use_visual_box_prompt)?;
        if let Some(recorder) = session.debug_recorder_mut() {
            recorder.record_detector_grounding(
                object,
                frame_idx,
                direction,
                prompt_metadata.clone(),
                &detector_output,
            )?;
        }
        let history = self.history_on_device(object, frame_idx, direction, compute_device)?;
        // Upstream video inference seeds tracker memory from detector masks on the prompt
        // frame, including the initial box-as-visual prompt case.
        let mask_input = grounding.mask_logits.gt(0f32)?.to_dtype(DType::F32)?;
        let step = self.tracker.track_frame(
            &visual,
            frame_idx,
            session.num_frames(),
            None,
            None,
            None,
            Some(&mask_input),
            &history,
            true,
            matches!(direction, PropagationDirection::Backward),
            false,
        )?;
        let trimmed_memory_frame_indices =
            trim_memory_frame_indices(step.memory_frame_indices.clone(), config.memory_frame_count);
        let output = ObjectFrameOutput {
            memory_frame_indices: trimmed_memory_frame_indices.clone(),
            ..detector_output.clone()
        };
        let video_size = session.video_size();
        if let Some(recorder) = session.debug_recorder_mut() {
            let tracker_seed_output = tracker_state_to_object_output(
                object.obj_id,
                &step.state,
                Some(frame_idx),
                trimmed_memory_frame_indices,
                prompt.text.clone(),
                prompt.has_geometry(),
                false,
                video_size,
            )?;
            recorder.record_tracker_seed(
                object,
                frame_idx,
                direction,
                prompt_metadata,
                &tracker_seed_output,
                &step.state,
            )?;
        }
        self.store_object_result(session, frame_idx, output, step.state)
    }

    fn seed_with_tracker_points(
        &self,
        model: &Sam3ImageModel,
        compute_device: &Device,
        config: &VideoConfig,
        session: &mut Sam3VideoSession,
        object: &TrackedObject,
        frame_idx: usize,
        direction: PropagationDirection,
        prompt: &SessionPrompt,
    ) -> Result<ObjectFrameOutput> {
        let visual = session.get_visual_features(model, compute_device, frame_idx)?;
        let history = self.history_on_device(object, frame_idx, direction, compute_device)?;
        let image_size = model.input_size();
        let point_inputs = session_prompt_to_tracker_points(prompt, image_size, compute_device)?;
        let boxes_xyxy = session_prompt_to_tracker_box(prompt, image_size, compute_device)?;
        let (point_coords, point_labels) = match point_inputs {
            Some((coords, labels)) => (Some(coords), Some(labels)),
            None => (None, None),
        };
        let step = self.tracker.track_frame(
            &visual,
            frame_idx,
            session.num_frames(),
            point_coords.as_ref(),
            point_labels.as_ref(),
            boxes_xyxy.as_ref(),
            None,
            &history,
            true,
            matches!(direction, PropagationDirection::Backward),
            false,
        )?;
        let output = tracker_state_to_object_output(
            object.obj_id,
            &step.state,
            Some(frame_idx),
            trim_memory_frame_indices(step.memory_frame_indices, config.memory_frame_count),
            prompt.text.clone(),
            prompt.has_geometry(),
            false,
            session.video_size(),
        )?;
        self.store_object_result(session, frame_idx, output, step.state)
    }

    fn propagate_with_tracker(
        &self,
        model: &Sam3ImageModel,
        compute_device: &Device,
        config: &VideoConfig,
        session: &mut Sam3VideoSession,
        object: &TrackedObject,
        frame_idx: usize,
        direction: PropagationDirection,
    ) -> Result<Option<ObjectFrameOutput>> {
        let history = self.history_on_device(object, frame_idx, direction, compute_device)?;
        if history.is_empty() {
            return Ok(None);
        }
        let nearest_prompt_frame_idx = object
            .nearest_prompt(frame_idx, direction)
            .map(|(idx, _)| idx);
        let visual = session.get_visual_features(model, compute_device, frame_idx)?;
        let step = self.tracker.track_frame(
            &visual,
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
        )?;
        let prompt_frame_idx = match direction {
            PropagationDirection::Forward | PropagationDirection::Both => {
                step.prompt_frame_indices.last().copied()
            }
            PropagationDirection::Backward => step.prompt_frame_indices.first().copied(),
        };
        let text_prompt = object
            .latest_text_prompt(frame_idx, direction)
            .map(|(_, text)| text);
        let used_explicit_geometry = prompt_frame_idx
            .and_then(|idx| object.prompt_frames.get(&idx))
            .map(|prompt| prompt.has_geometry())
            .unwrap_or(false);
        let output = tracker_state_to_object_output(
            object.obj_id,
            &step.state,
            prompt_frame_idx,
            trim_memory_frame_indices(step.memory_frame_indices, config.memory_frame_count),
            text_prompt,
            used_explicit_geometry,
            true,
            session.video_size(),
        )?;
        let trimmed_memory_frame_indices = output.memory_frame_indices.clone();
        if let Some(recorder) = session.debug_recorder_mut() {
            recorder.record_first_propagation(
                object,
                frame_idx,
                direction,
                prompt_frame_idx.or(nearest_prompt_frame_idx),
                &output,
                &history,
                &step.prompt_frame_indices,
                &trimmed_memory_frame_indices,
            )?;
        }
        Ok(Some(self.store_object_result(
            session, frame_idx, output, step.state,
        )?))
    }

    fn infer_object_on_frame(
        &self,
        model: &Sam3ImageModel,
        compute_device: &Device,
        config: &VideoConfig,
        session: &mut Sam3VideoSession,
        object: TrackedObject,
        frame_idx: usize,
        direction: PropagationDirection,
    ) -> Result<Option<ObjectFrameOutput>> {
        if let Some(cached) = object.frame_outputs.get(&frame_idx) {
            return Ok(Some(cached.clone()));
        }

        if let Some(prompt) = object.prompt_frames.get(&frame_idx) {
            if prompt.text.is_some() || prompt.boxes.is_some() {
                return self
                    .seed_with_detector(
                        model,
                        compute_device,
                        config,
                        session,
                        &object,
                        frame_idx,
                        direction,
                        prompt,
                    )
                    .map(Some);
            }
            if prompt.points.is_some() {
                return self
                    .seed_with_tracker_points(
                        model,
                        compute_device,
                        config,
                        session,
                        &object,
                        frame_idx,
                        direction,
                        prompt,
                    )
                    .map(Some);
            }
        }

        self.propagate_with_tracker(
            model,
            compute_device,
            config,
            session,
            &object,
            frame_idx,
            direction,
        )
    }
}

impl VideoTrackerBackend for Sam3MemoryAttentionVideoTrackerBackend<'_> {
    fn process_frame(
        &self,
        model: &Sam3ImageModel,
        compute_device: &Device,
        config: &VideoConfig,
        session: &mut Sam3VideoSession,
        frame_idx: usize,
        direction: PropagationDirection,
        _output_threshold: f32,
    ) -> Result<VideoFrameOutput> {
        let object_ids = session
            .tracked_objects
            .values()
            .filter(|object| object.is_active_for_frame(frame_idx, direction))
            .map(|object| object.obj_id)
            .collect::<Vec<_>>();

        let mut objects = Vec::new();
        for obj_id in object_ids {
            let object = session
                .tracked_objects
                .get(&obj_id)
                .cloned()
                .ok_or_else(|| candle::Error::Msg(format!("unknown obj_id {}", obj_id)))?;
            if let Some(output) = self.infer_object_on_frame(
                model,
                compute_device,
                config,
                session,
                object,
                frame_idx,
                direction,
            )? {
                objects.push(output);
            }
        }

        Ok(VideoFrameOutput { frame_idx, objects })
    }
}

#[derive(Debug, Default)]
struct HeuristicVideoTrackerBackend;

impl HeuristicVideoTrackerBackend {
    fn infer_object_on_frame(
        &self,
        model: &Sam3ImageModel,
        compute_device: &Device,
        config: &VideoConfig,
        session: &mut Sam3VideoSession,
        object: TrackedObject,
        frame_idx: usize,
        direction: PropagationDirection,
        output_threshold: f32,
    ) -> Result<Option<ObjectFrameOutput>> {
        if let Some(cached) = object.frame_outputs.get(&frame_idx) {
            return Ok(Some(cached.clone()));
        }

        let (effective_prompt, prompt_frame_idx, memory_frame_indices, used_explicit_geometry) =
            build_effective_prompt(&object, frame_idx, direction, config)?;
        if effective_prompt.is_empty() {
            return Ok(None);
        }

        let visual = session.get_visual_features(model, compute_device, frame_idx)?;
        let text_encoding = match effective_prompt.text.as_deref() {
            Some(text_prompt) => {
                Some(session.cached_text_encoding(model, text_prompt, compute_device)?)
            }
            None => None,
        };
        let geometry_encoding = if effective_prompt.has_geometry() {
            let geometry_prompt = session_prompt_to_geometry(&effective_prompt, compute_device)?;
            Some(model.encode_geometry_prompt(&geometry_prompt, &visual)?)
        } else {
            None
        };
        let prompt = combine_encoded_prompts(text_encoding.as_ref(), geometry_encoding.as_ref())?
            .ok_or_else(|| {
            candle::Error::Msg("effective prompt unexpectedly empty".to_owned())
        })?;
        let grounding = ground_from_encoded_prompt(model, &visual, &prompt)?;
        let mut output = ObjectFrameOutput::from_grounding(
            object.obj_id,
            grounding,
            prompt_frame_idx,
            memory_frame_indices.clone(),
            effective_prompt.text.clone(),
            used_explicit_geometry,
            false,
        );

        if output.score_value()? < output_threshold {
            if let Some(previous_frame_idx) = memory_frame_indices.first().copied() {
                if let Some(previous) = object.frame_outputs.get(&previous_frame_idx) {
                    output = previous.clone();
                    output.prompt_frame_idx = prompt_frame_idx;
                    output.memory_frame_indices = memory_frame_indices;
                    output.reused_previous_output = true;
                    output.text_prompt = effective_prompt.text.clone();
                    output.used_explicit_geometry = used_explicit_geometry;
                }
            }
        }

        let stored = output.to_storage_device(session.storage_device())?;
        session
            .tracked_objects
            .get_mut(&object.obj_id)
            .ok_or_else(|| candle::Error::Msg(format!("unknown obj_id {}", object.obj_id)))?
            .frame_outputs
            .insert(frame_idx, stored.clone());
        session
            .tracked_objects
            .get_mut(&object.obj_id)
            .expect("tracked object exists")
            .last_updated_frame = frame_idx;
        session
            .frame_outputs
            .entry(frame_idx)
            .or_default()
            .insert(object.obj_id, stored.clone());
        Ok(Some(stored))
    }
}

impl VideoTrackerBackend for HeuristicVideoTrackerBackend {
    fn process_frame(
        &self,
        model: &Sam3ImageModel,
        compute_device: &Device,
        config: &VideoConfig,
        session: &mut Sam3VideoSession,
        frame_idx: usize,
        direction: PropagationDirection,
        output_threshold: f32,
    ) -> Result<VideoFrameOutput> {
        let object_ids = session
            .tracked_objects
            .values()
            .filter(|object| object.is_active_for_frame(frame_idx, direction))
            .map(|object| object.obj_id)
            .collect::<Vec<_>>();

        let mut objects = Vec::new();
        for obj_id in object_ids {
            let object = session
                .tracked_objects
                .get(&obj_id)
                .cloned()
                .ok_or_else(|| candle::Error::Msg(format!("unknown obj_id {}", obj_id)))?;
            if let Some(output) = self.infer_object_on_frame(
                model,
                compute_device,
                config,
                session,
                object,
                frame_idx,
                direction,
                output_threshold,
            )? {
                objects.push(output);
            }
        }

        Ok(VideoFrameOutput { frame_idx, objects })
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

fn frame_blob_from_rgb_image(
    image: image::RgbImage,
    image_size: usize,
    image_mean: [f32; 3],
    image_std: [f32; 3],
    expected_video_size: ImageSize,
    source_label: &str,
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

    let image = Tensor::from_vec(
        image.into_raw(),
        (expected_video_size.height, expected_video_size.width, 3),
        &Device::Cpu,
    )?
    .permute((2, 0, 1))?;
    let resized = resize_image_exact_for_sam3(&image, image_size)?;
    let normalized = normalize_image_for_sam3(&resized, image_mean, image_std)?.squeeze(0)?;
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

fn resize_image_exact_for_sam3(image_chw: &Tensor, image_size: usize) -> Result<Tensor> {
    let image = match image_chw.rank() {
        3 => image_chw.unsqueeze(0)?,
        4 => image_chw.clone(),
        rank => candle::bail!(
            "sam3 exact resize expects CHW or BCHW image, got rank {}",
            rank
        ),
    };
    let image = image
        .to_dtype(DType::F32)?
        .upsample_bilinear2d(image_size, image_size, false)?;
    image / 255.
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

fn build_effective_prompt(
    object: &TrackedObject,
    frame_idx: usize,
    direction: PropagationDirection,
    config: &VideoConfig,
) -> Result<(SessionPrompt, Option<usize>, Vec<usize>, bool)> {
    let (prompt_frame_idx, mut prompt) = object
        .nearest_prompt(frame_idx, direction)
        .map(|(idx, prompt)| (Some(idx), prompt))
        .unwrap_or((
            None,
            SessionPrompt {
                text: None,
                points: None,
                point_labels: None,
                boxes: None,
                box_labels: None,
            },
        ));
    if prompt.text.is_none() {
        prompt.text = object
            .latest_text_prompt(frame_idx, direction)
            .map(|(_, text)| text);
    }
    let used_explicit_geometry = prompt.has_geometry();
    let memory_frame_indices =
        object.recent_output_frame_indices(frame_idx, direction, config.memory_frame_count);

    if !memory_frame_indices.is_empty() {
        let mut derived_boxes = Vec::new();
        for output_frame_idx in memory_frame_indices.iter().take(config.max_memory_boxes) {
            let output = object.frame_outputs.get(output_frame_idx).ok_or_else(|| {
                candle::Error::Msg(format!(
                    "missing cached output for obj_id {} on frame {}",
                    object.obj_id, output_frame_idx
                ))
            })?;
            derived_boxes.push(xyxy_to_cxcywh(first_box_xyxy(&output.boxes_xyxy)?));
        }

        if !derived_boxes.is_empty() {
            let mut boxes = prompt.boxes.clone().unwrap_or_default();
            boxes.extend(derived_boxes.iter().copied());
            prompt.boxes = Some(boxes);

            let mut labels = prompt.box_labels.clone().unwrap_or_default();
            labels.extend(std::iter::repeat(1).take(derived_boxes.len()));
            prompt.box_labels = Some(labels);
        }

        if config.derive_mask_centroid_points && prompt.points.is_none() {
            if let Some(output) = object.frame_outputs.get(
                memory_frame_indices
                    .first()
                    .expect("memory indices checked"),
            ) {
                if let Some(point) = mask_centroid(&output.masks)? {
                    prompt.points = Some(vec![point]);
                    prompt.point_labels = Some(vec![1]);
                }
            }
        }
    }

    Ok((
        prompt.with_default_labels()?,
        prompt_frame_idx,
        memory_frame_indices,
        used_explicit_geometry,
    ))
}

fn uses_initial_visual_box_prompt(
    object: &TrackedObject,
    frame_idx: usize,
    prompt: &SessionPrompt,
) -> bool {
    !object.has_inference_history
        && object.creation_frame == frame_idx
        && prompt.text.is_none()
        && prompt
            .points
            .as_ref()
            .map(|points| points.is_empty())
            .unwrap_or(true)
        && matches!(prompt.boxes.as_ref(), Some(boxes) if boxes.len() == 1)
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

fn first_box_xyxy(boxes_xyxy: &Tensor) -> Result<[f32; 4]> {
    let values = boxes_xyxy.flatten_all()?.to_vec1::<f32>()?;
    if values.len() < 4 {
        candle::bail!("expected at least 4 box coordinates, got {}", values.len())
    }
    Ok([values[0], values[1], values[2], values[3]])
}

fn tensor_to_mask_probs_2d(tensor: &Tensor) -> Result<Vec<Vec<f32>>> {
    let tensor = match tensor.rank() {
        2 => tensor.clone(),
        3 => tensor.i(0)?,
        4 => tensor.i((0, 0))?,
        rank => candle::bail!("expected mask tensor rank 2/3/4, got {rank}"),
    };
    tensor.to_vec2::<f32>()
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

fn xyxy_to_cxcywh(box_xyxy: [f32; 4]) -> (f32, f32, f32, f32) {
    let width = (box_xyxy[2] - box_xyxy[0]).max(0.0);
    let height = (box_xyxy[3] - box_xyxy[1]).max(0.0);
    (
        box_xyxy[0] + width * 0.5,
        box_xyxy[1] + height * 0.5,
        width,
        height,
    )
}

fn mask_centroid(mask: &Tensor) -> Result<Option<(f32, f32)>> {
    let mask = match mask.rank() {
        3 => mask.i(0)?,
        2 => mask.clone(),
        rank => candle::bail!("expected mask rank 2 or 3, got {}", rank),
    };
    let values = mask.to_dtype(DType::F32)?.to_vec2::<f32>()?;
    if values.is_empty() || values[0].is_empty() {
        return Ok(None);
    }
    let height = values.len();
    let width = values[0].len();
    let mut total_weight = 0.0f32;
    let mut x_sum = 0.0f32;
    let mut y_sum = 0.0f32;
    for (y, row) in values.iter().enumerate() {
        for (x, value) in row.iter().enumerate() {
            if *value >= 0.5 {
                total_weight += *value;
                x_sum += x as f32 * *value;
                y_sum += y as f32 * *value;
            }
        }
    }
    if total_weight <= 0.0 {
        return Ok(None);
    }
    let x = (x_sum / total_weight) / width.max(1) as f32;
    let y = (y_sum / total_weight) / height.max(1) as f32;
    Ok(Some((x.clamp(0.0, 1.0), y.clamp(0.0, 1.0))))
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

fn session_prompt_to_tracker_points(
    prompt: &SessionPrompt,
    image_size: ImageSize,
    device: &Device,
) -> Result<Option<(Tensor, Tensor)>> {
    let Some(points) = prompt.points.as_ref() else {
        return Ok(None);
    };
    let mut coords = Vec::with_capacity(points.len() * 2);
    for (x, y) in points {
        coords.push(x.clamp(0.0, 1.0) * image_size.width as f32);
        coords.push(y.clamp(0.0, 1.0) * image_size.height as f32);
    }
    let labels = prompt
        .point_labels
        .as_ref()
        .ok_or_else(|| candle::Error::Msg("tracker point prompts require point labels".to_owned()))?
        .iter()
        .map(|label| *label as f32)
        .collect::<Vec<_>>();
    Ok(Some((
        Tensor::from_vec(coords, (1, points.len(), 2), device)?,
        Tensor::from_vec(labels, (1, points.len()), device)?,
    )))
}

fn session_prompt_to_tracker_box(
    prompt: &SessionPrompt,
    image_size: ImageSize,
    device: &Device,
) -> Result<Option<Tensor>> {
    let Some(boxes) = prompt.boxes.as_ref() else {
        return Ok(None);
    };
    if boxes.is_empty() {
        return Ok(None);
    }
    let mut x0 = f32::INFINITY;
    let mut y0 = f32::INFINITY;
    let mut x1 = f32::NEG_INFINITY;
    let mut y1 = f32::NEG_INFINITY;
    for (cx, cy, width, height) in boxes {
        let half_w = width * 0.5;
        let half_h = height * 0.5;
        x0 = x0.min((cx - half_w).clamp(0.0, 1.0) * image_size.width as f32);
        y0 = y0.min((cy - half_h).clamp(0.0, 1.0) * image_size.height as f32);
        x1 = x1.max((cx + half_w).clamp(0.0, 1.0) * image_size.width as f32);
        y1 = y1.max((cy + half_h).clamp(0.0, 1.0) * image_size.height as f32);
    }
    Tensor::from_vec(vec![x0, y0, x1, y1], (1, 4), device).map(Some)
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

fn canonicalize_score_tensor(scores: &Tensor) -> Result<Tensor> {
    let values = scores.flatten_all()?.to_vec1::<f32>()?;
    Tensor::from_vec(values, (scores.elem_count(),), scores.device())
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

fn tracker_state_to_object_output(
    obj_id: u32,
    state: &TrackerFrameState,
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
        scores: canonicalize_score_tensor(&state.iou_scores)?,
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
) -> Result<Vec<usize>> {
    let seed_frames = session.prompt_frames();
    if seed_frames.is_empty() {
        candle::bail!("no prompts added to session");
    }
    let num_frames = session.num_frames();
    let start_frame_idx = match start_frame_idx {
        Some(frame_idx) => frame_idx,
        None => match direction {
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
    use image::{ImageBuffer, Rgb, RgbImage};

    use crate::models::sam3::{
        Config, DecoderConfig, EncoderConfig, GeometryConfig, ImageConfig, NeckConfig,
        SegmentationConfig, TextConfig, VisionConfig,
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
                d_model: 8,
                width: 16,
                heads: 2,
                layers: 1,
                context_length: 4,
                vocab_size: 64,
            },
            neck: NeckConfig {
                d_model: 8,
                scale_factors: [1.0, 0.5, 0.5, 0.5],
                scalp: 3,
                add_sam2_neck: false,
            },
            geometry: GeometryConfig {
                d_model: 8,
                num_layers: 1,
                num_heads: 1,
                dim_feedforward: 16,
                roi_size: 2,
                add_cls: true,
                add_post_encode_proj: true,
            },
            encoder: EncoderConfig {
                d_model: 8,
                num_layers: 1,
                num_feature_levels: 1,
                num_heads: 1,
                dim_feedforward: 16,
                add_pooled_text_to_image: false,
                pool_text_with_mask: true,
            },
            decoder: DecoderConfig {
                d_model: 8,
                num_layers: 1,
                num_queries: 2,
                num_heads: 1,
                dim_feedforward: 16,
                presence_token: true,
                use_text_cross_attention: true,
                box_rpb_mode: "none".to_owned(),
                box_rpb_resolution: 56,
                box_rpb_stride: 14,
                clamp_presence_logit_max: 10.0,
            },
            segmentation: SegmentationConfig {
                enabled: true,
                hidden_dim: 8,
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

    fn temp_path(name: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time is after epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("sam3-video-tests-{}-{}", name, unique))
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
        let frames = vec![
            Tensor::zeros((3, 56, 56), DType::F32, &device)?,
            Tensor::zeros((3, 56, 56), DType::F32, &device)?,
        ];
        let mut predictor = Sam3VideoPredictor::new(&model, &device);
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
    fn propagation_emits_directional_frames_and_stays_lazy() -> Result<()> {
        let device = Device::Cpu;
        let model = tiny_model(&device)?;
        let frames = vec![
            Tensor::zeros((3, 56, 56), DType::F32, &device)?,
            Tensor::zeros((3, 56, 56), DType::F32, &device)?,
            Tensor::zeros((3, 56, 56), DType::F32, &device)?,
            Tensor::zeros((3, 56, 56), DType::F32, &device)?,
        ];
        let mut predictor = Sam3VideoPredictor::new(&model, &device);
        let session_id =
            predictor.start_session_from_tensors(frames, VideoSessionOptions::default())?;
        let obj_id = predictor.add_prompt(
            &session_id,
            1,
            SessionPrompt {
                text: None,
                points: Some(vec![(0.5, 0.5)]),
                point_labels: None,
                boxes: None,
                box_labels: None,
            },
            None,
            true,
            true,
        )?;

        let forward = predictor.propagate_in_video(
            &session_id,
            PropagationOptions {
                direction: PropagationDirection::Forward,
                start_frame_idx: None,
                max_frame_num_to_track: None,
                output_prob_threshold: None,
            },
        )?;
        assert_eq!(
            forward
                .frames
                .iter()
                .map(|frame| frame.frame_idx)
                .collect::<Vec<_>>(),
            vec![1, 2, 3]
        );
        assert!(forward
            .frames
            .iter()
            .all(|frame| { frame.objects.iter().any(|object| object.obj_id == obj_id) }));

        let backward = predictor.propagate_in_video(
            &session_id,
            PropagationOptions {
                direction: PropagationDirection::Backward,
                start_frame_idx: Some(1),
                max_frame_num_to_track: Some(2),
                output_prob_threshold: None,
            },
        )?;
        assert_eq!(
            backward
                .frames
                .iter()
                .map(|frame| frame.frame_idx)
                .collect::<Vec<_>>(),
            vec![1, 0]
        );

        let stats = predictor.session_cache_stats(&session_id)?;
        assert!(stats.cached_feature_entries <= 2);
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
        let source = VideoSource::from_path(&dir)?;
        let mut predictor = Sam3VideoPredictor::new(&model, &device);
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
                points: Some(vec![(0.5, 0.5)]),
                point_labels: None,
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
                start_frame_idx: None,
                max_frame_num_to_track: None,
                output_prob_threshold: None,
            },
        )?;

        let session = predictor
            .sessions
            .get_mut(&session_id)
            .expect("session should exist");
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
        let frames = vec![
            Tensor::zeros((3, 56, 56), DType::F32, &device)?,
            Tensor::zeros((3, 56, 56), DType::F32, &device)?,
        ];
        let mut predictor = Sam3VideoPredictor::new(&model, &device);
        let session_id =
            predictor.start_session_from_tensors(frames, VideoSessionOptions::default())?;
        let obj_id = predictor.add_prompt(
            &session_id,
            0,
            SessionPrompt {
                text: None,
                points: Some(vec![(0.5, 0.5)]),
                point_labels: None,
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
                start_frame_idx: None,
                max_frame_num_to_track: None,
                output_prob_threshold: None,
            },
        )?;
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
        let frames = vec![Tensor::zeros((3, 56, 56), DType::F32, &device)?];
        let mut predictor = Sam3VideoPredictor::new(&model, &device);
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
    fn initial_single_box_prompt_is_treated_as_visual_prompt() {
        let mut object = TrackedObject::new(4, 0);
        object.prompt_frames.insert(
            0,
            SessionPrompt {
                text: None,
                points: None,
                point_labels: None,
                boxes: Some(vec![(0.5, 0.5, 0.2, 0.3)]),
                box_labels: Some(vec![1]),
            },
        );
        let prompt = object
            .prompt_frames
            .get(&0)
            .expect("prompt should be present");
        assert!(uses_initial_visual_box_prompt(&object, 0, prompt));
    }

    #[test]
    fn later_box_prompt_is_not_treated_as_initial_visual_prompt() {
        let mut object = TrackedObject::new(4, 0);
        object.has_inference_history = true;
        object.prompt_frames.insert(
            0,
            SessionPrompt {
                text: None,
                points: None,
                point_labels: None,
                boxes: Some(vec![(0.5, 0.5, 0.2, 0.3)]),
                box_labels: Some(vec![1]),
            },
        );
        let prompt = object
            .prompt_frames
            .get(&0)
            .expect("prompt should be present");
        assert!(!uses_initial_visual_box_prompt(&object, 0, prompt));
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

        let mut baseline = Sam3VideoPredictor::new(&model, &device);
        let baseline_session = baseline.start_session_from_tensors(
            frames.iter().cloned().collect(),
            VideoSessionOptions::default(),
        )?;
        baseline.add_prompt(&baseline_session, 0, prompt.clone(), None, true, true)?;
        let baseline_output = baseline.propagate_in_video(
            &baseline_session,
            PropagationOptions {
                direction: PropagationDirection::Forward,
                start_frame_idx: None,
                max_frame_num_to_track: Some(2),
                output_prob_threshold: None,
            },
        )?;

        let debug_root = temp_path("debug-disabled");
        let mut debug_predictor =
            Sam3VideoPredictor::new(&model, &device).with_debug_config(VideoDebugConfig {
                enabled: false,
                capture_obj_ids: Vec::new(),
                capture_frame_indices: vec![0, 1],
                capture_first_propagated_only: true,
                output_root: Some(debug_root.clone()),
            });
        let debug_session =
            debug_predictor.start_session_from_tensors(frames, VideoSessionOptions::default())?;
        debug_predictor.add_prompt(&debug_session, 0, prompt, None, true, true)?;
        let debug_output = debug_predictor.propagate_in_video(
            &debug_session,
            PropagationOptions {
                direction: PropagationDirection::Forward,
                start_frame_idx: None,
                max_frame_num_to_track: Some(2),
                output_prob_threshold: None,
            },
        )?;

        assert_eq!(baseline_output.frames.len(), debug_output.frames.len());
        for (baseline_frame, debug_frame) in baseline_output
            .frames
            .iter()
            .zip(debug_output.frames.iter())
        {
            assert_eq!(baseline_frame.frame_idx, debug_frame.frame_idx);
            assert_eq!(baseline_frame.objects.len(), debug_frame.objects.len());
            for (lhs, rhs) in baseline_frame
                .objects
                .iter()
                .zip(debug_frame.objects.iter())
            {
                assert_eq!(lhs.obj_id, rhs.obj_id);
                assert_eq!(
                    lhs.boxes_xyxy.to_vec2::<f32>()?,
                    rhs.boxes_xyxy.to_vec2::<f32>()?
                );
                assert_eq!(
                    lhs.scores.flatten_all()?.to_vec1::<f32>()?,
                    rhs.scores.flatten_all()?.to_vec1::<f32>()?
                );
            }
        }
        assert!(!debug_root.exists());
        Ok(())
    }
}
