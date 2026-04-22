use super::*;
use crate::models::sam3::tracker::PackedPromptHistory;

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
    pub confirmation_consecutive_frames: usize,
    pub confirmation_confirmed: bool,
    pub prompt_frames: BTreeMap<usize, SessionPrompt>,
    pub mask_prompt_frames: BTreeMap<usize, Tensor>,
    pub frame_outputs: BTreeMap<usize, ObjectFrameOutput>,
    pub tracker_states: BTreeMap<usize, TrackerFrameState>,
    pub prompt_history_cache: PackedPromptHistory,
}

impl TrackedObject {
    pub(crate) fn new(obj_id: u32, creation_frame: usize) -> Self {
        Self {
            obj_id,
            creation_frame,
            last_updated_frame: creation_frame,
            display_score: None,
            has_inference_history: false,
            confirmation_consecutive_frames: 0,
            confirmation_confirmed: false,
            prompt_frames: BTreeMap::new(),
            mask_prompt_frames: BTreeMap::new(),
            frame_outputs: BTreeMap::new(),
            tracker_states: BTreeMap::new(),
            prompt_history_cache: PackedPromptHistory::default(),
        }
    }

    pub(crate) fn add_prompt(
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

    pub(crate) fn add_mask_prompt(&mut self, frame_idx: usize, mask: Tensor) {
        self.mask_prompt_frames.insert(frame_idx, mask);
        self.prompt_frames.remove(&frame_idx);
        self.last_updated_frame = frame_idx;
    }

    pub(crate) fn has_prompt_on_frame(&self, frame_idx: usize) -> bool {
        self.prompt_frames.contains_key(&frame_idx)
            || self.mask_prompt_frames.contains_key(&frame_idx)
    }

    pub(crate) fn nearest_prompt(
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

    pub(crate) fn nearest_input_frame_idx(
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

    pub(crate) fn nearest_input_uses_explicit_geometry(
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

    pub(crate) fn latest_text_prompt(
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

    pub(crate) fn recent_output_frame_indices(
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

    pub(crate) fn is_active_for_frame(
        &self,
        frame_idx: usize,
        direction: PropagationDirection,
    ) -> bool {
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

    pub(crate) fn tracker_history(
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

    pub(crate) fn clear_prompt_history_cache(&mut self) {
        self.prompt_history_cache.clear();
    }

    pub(crate) fn ensure_prompt_history_cache(&mut self) -> Result<()> {
        self.prompt_history_cache.ensure_built(&self.tracker_states)
    }

    pub(crate) fn maybe_append_prompt_history_cache(
        &mut self,
        frame_idx: usize,
        state: &TrackerFrameState,
    ) -> Result<()> {
        if self.prompt_history_cache.is_initialized() || self.tracker_states.len() == 1 {
            self.prompt_history_cache.append_state(frame_idx, state)?;
        }
        Ok(())
    }

    pub(crate) fn record_confirmation_activity(
        &mut self,
        has_detectable_output: bool,
        threshold: usize,
    ) -> bool {
        if self.confirmation_confirmed {
            if !has_detectable_output {
                self.confirmation_consecutive_frames = 0;
            }
            return true;
        }
        self.confirmation_consecutive_frames = if has_detectable_output {
            self.confirmation_consecutive_frames.saturating_add(1)
        } else {
            0
        };
        if self.confirmation_consecutive_frames >= threshold.max(1) {
            self.confirmation_confirmed = true;
        }
        self.confirmation_confirmed
    }
}

pub struct Sam3VideoSession {
    session_id: String,
    frame_source: Box<dyn FrameSource>,
    session_options: VideoSessionOptions,
    tokenizer: Option<Tokenizer>,
    debug_recorder: Option<VideoDebugRecorder>,
    storage_device: Device,
    pub(super) tracked_objects: BTreeMap<u32, TrackedObject>,
    next_obj_id: u32,
    pub(super) frame_outputs: BTreeMap<usize, BTreeMap<u32, ObjectFrameOutput>>,
    pub(super) temporal_disambiguation_metadata:
        BTreeMap<usize, TemporalDisambiguationFrameMetadata>,
    feature_cache: HashMap<usize, VisualBackboneOutput>,
    feature_cache_order: VecDeque<usize>,
    text_cache: HashMap<String, CachedTextPrompt>,
}

impl Sam3VideoSession {
    pub(crate) fn new(
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
            temporal_disambiguation_metadata: BTreeMap::new(),
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

    pub(crate) fn prompt_frames(&self) -> BTreeSet<usize> {
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

    pub(crate) fn storage_device(&self) -> &Device {
        &self.storage_device
    }

    pub(super) fn debug_recorder_mut(&mut self) -> Option<&mut VideoDebugRecorder> {
        self.debug_recorder.as_mut()
    }

    pub(crate) fn clear_temporal_disambiguation_metadata(&mut self) {
        self.temporal_disambiguation_metadata.clear();
    }

    fn allocate_object(&mut self, creation_frame: usize) -> u32 {
        let obj_id = self.next_obj_id;
        self.next_obj_id += 1;
        self.tracked_objects
            .insert(obj_id, TrackedObject::new(obj_id, creation_frame));
        obj_id
    }

    pub(crate) fn ensure_object(
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

    pub(crate) fn add_prompt(
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

    pub(crate) fn add_mask_prompt(
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

    pub(crate) fn invalidate_object_outputs_from(&mut self, obj_id: u32, frame_idx: usize) {
        if let Some(object) = self.tracked_objects.get_mut(&obj_id) {
            object.frame_outputs.retain(|idx, _| *idx <= frame_idx);
            object.tracker_states.retain(|idx, _| *idx <= frame_idx);
            object.clear_prompt_history_cache();
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

    pub(crate) fn remove_object(&mut self, obj_id: u32) -> Result<()> {
        self.tracked_objects
            .remove(&obj_id)
            .ok_or_else(|| candle::Error::Msg(format!("unknown obj_id {}", obj_id)))?;
        self.invalidate_object_outputs_from(obj_id, 0);
        Ok(())
    }

    pub(crate) fn reset(&mut self) {
        self.tracked_objects.clear();
        self.next_obj_id = 0;
        self.frame_outputs.clear();
        self.temporal_disambiguation_metadata.clear();
        self.feature_cache.clear();
        self.feature_cache_order.clear();
        self.text_cache.clear();
        self.frame_source.close();
    }

    pub(crate) fn close(&mut self) {
        if let Some(recorder) = self.debug_recorder.as_ref() {
            let _ = recorder.flush_manifest();
        }
        self.frame_source.close();
        self.feature_cache.clear();
        self.feature_cache_order.clear();
        self.frame_outputs.clear();
        self.temporal_disambiguation_metadata.clear();
        self.tracked_objects.clear();
        self.text_cache.clear();
        self.debug_recorder = None;
    }

    pub(crate) fn get_frame(&mut self, frame_idx: usize, target_device: &Device) -> Result<Tensor> {
        if frame_idx >= self.num_frames() {
            candle::bail!("frame_idx {} out of bounds", frame_idx);
        }
        self.frame_source.get_frame(frame_idx, target_device)
    }

    pub(crate) fn prefetch_for_frame(
        &mut self,
        frame_idx: usize,
        direction: PropagationDirection,
    ) -> Result<()> {
        let keep = self.prefetch_window(frame_idx, direction);
        let indices = keep.iter().copied().collect::<Vec<_>>();
        self.frame_source.prefetch(&indices)
    }

    pub(crate) fn evict_for_frame(&mut self, frame_idx: usize, direction: PropagationDirection) {
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

    pub(crate) fn get_visual_features(
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

    pub(crate) fn cached_text_encoding(
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

#[derive(Debug, Clone)]
pub(crate) struct CachedTextPrompt {
    attention_mask: Tensor,
    memory: Tensor,
    input_embeddings: Tensor,
}

impl CachedTextPrompt {
    fn from_encoding(encoding: &TextEncoding, storage_device: &Device) -> Result<Self> {
        Ok(Self {
            attention_mask: if encoding.attention_mask.device().same_device(storage_device) {
                encoding.attention_mask.clone()
            } else {
                encoding.attention_mask.to_device(storage_device)?
            },
            memory: if encoding.memory.device().same_device(storage_device) {
                encoding.memory.clone()
            } else {
                encoding.memory.to_device(storage_device)?
            },
            input_embeddings: if encoding.input_embeddings.device().same_device(storage_device) {
                encoding.input_embeddings.clone()
            } else {
                encoding.input_embeddings.to_device(storage_device)?
            },
        })
    }

    fn to_text_encoding(&self, compute_device: &Device) -> Result<TextEncoding> {
        Ok(TextEncoding {
            attention_mask: if self.attention_mask.device().same_device(compute_device) {
                self.attention_mask.clone()
            } else {
                self.attention_mask.to_device(compute_device)?
            },
            memory: if self.memory.device().same_device(compute_device) {
                self.memory.clone()
            } else {
                self.memory.to_device(compute_device)?
            },
            input_embeddings: if self.input_embeddings.device().same_device(compute_device) {
                self.input_embeddings.clone()
            } else {
                self.input_embeddings.to_device(compute_device)?
            },
        })
    }
}
