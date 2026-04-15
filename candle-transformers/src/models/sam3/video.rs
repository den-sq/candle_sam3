// Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::fs;
use std::path::{Path, PathBuf};
#[cfg(test)]
use std::time::{SystemTime, UNIX_EPOCH};

use candle::{DType, Device, IndexOp, Result, Tensor};
use image::ImageReader;
use tokenizers::{PaddingDirection, PaddingParams, Tokenizer, TruncationParams};

use super::{
    geometry::{EncodedPrompt, GeometryPrompt},
    image::{GroundingOutput, ImageSize},
    neck::VisualBackboneOutput,
    text::TextEncoding,
    Config, Sam3ImageModel,
};

const CLIP_EOT_TOKEN: &str = "<|endoftext|>";

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
            memory_frame_count: 2,
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
    pub prompt_frames: BTreeMap<usize, SessionPrompt>,
    pub frame_outputs: BTreeMap<usize, ObjectFrameOutput>,
}

impl TrackedObject {
    fn new(obj_id: u32, creation_frame: usize) -> Self {
        Self {
            obj_id,
            creation_frame,
            last_updated_frame: creation_frame,
            prompt_frames: BTreeMap::new(),
            frame_outputs: BTreeMap::new(),
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
            Self::VideoFile(path) => candle::bail!(
                "video file loading is not wired into the Rust predictor yet; extract frames into an image directory first: {}",
                path.display()
            ),
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
            session_id,
            frame_source,
            session_options,
            tokenizer,
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
        self.frame_source.close();
        self.feature_cache.clear();
        self.feature_cache_order.clear();
        self.frame_outputs.clear();
        self.tracked_objects.clear();
        self.text_cache.clear();
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
    let (width, height) = image.dimensions();
    let current_size = ImageSize::new(height as usize, width as usize);
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

fn first_box_xyxy(boxes_xyxy: &Tensor) -> Result<[f32; 4]> {
    let values = boxes_xyxy.flatten_all()?.to_vec1::<f32>()?;
    if values.len() < 4 {
        candle::bail!("expected at least 4 box coordinates, got {}", values.len())
    }
    Ok([values[0], values[1], values[2], values[3]])
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
}
