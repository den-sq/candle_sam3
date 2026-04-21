use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

use candle::{DType, IndexOp, Result, Tensor};
use image::{GrayImage, Luma};
use serde::{Deserialize, Serialize};

use super::{
    ObjectFrameOutput, PropagationDirection, SessionPrompt, TrackedObject, TrackerFrameState,
    VideoDebugConfig, VIDEO_DEBUG_MANIFEST_FILE, VIDEO_DEBUG_MASK_THRESHOLD,
};

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
pub(super) struct VideoDebugPromptMetadata {
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
pub(super) struct VideoDebugRecorder {
    config: VideoDebugConfig,
    output_root: PathBuf,
    manifest: VideoDebugManifest,
}

impl VideoDebugRecorder {
    pub(super) fn new(session_id: &str, config: VideoDebugConfig) -> Result<Option<Self>> {
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

    pub(super) fn record_detector_grounding(
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

    pub(super) fn record_tracker_seed(
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

    pub(super) fn record_first_propagation(
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

    pub(super) fn flush_manifest(&self) -> Result<()> {
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

pub(super) fn debug_prompt_metadata(
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

pub(super) fn tensor_to_mask_probs_2d(tensor: &Tensor) -> Result<Vec<Vec<f32>>> {
    let tensor = match tensor.rank() {
        2 => tensor.clone(),
        3 => tensor.i(0)?,
        4 => tensor.i((0, 0))?,
        rank => candle::bail!("expected mask tensor rank 2/3/4, got {rank}"),
    };
    tensor.to_dtype(DType::F32)?.to_vec2::<f32>()
}

pub(super) fn count_foreground_pixels(mask_probs: &[Vec<f32>], threshold: f32) -> usize {
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
