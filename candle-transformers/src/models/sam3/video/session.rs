use super::*;
use crate::models::sam3::tracker::{add_tensor_memory, PackedPromptHistory};

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SessionCacheByteBreakdown {
    pub frames: usize,
    pub visual_features: usize,
    pub tracker_states: usize,
    pub packed_prompt_history: usize,
    pub text_cache: usize,
    pub cached_outputs: usize,
}

impl SessionCacheByteBreakdown {
    pub fn total(&self) -> usize {
        self.frames
            .saturating_add(self.visual_features)
            .saturating_add(self.tracker_states)
            .saturating_add(self.packed_prompt_history)
            .saturating_add(self.text_cache)
            .saturating_add(self.cached_outputs)
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SessionCacheStats {
    pub loaded_frame_count: usize,
    pub cached_feature_entries: usize,
    pub cached_output_frames: usize,
    pub tracked_objects: usize,
    pub retained_tracker_states: usize,
    pub retained_non_cond_tracker_states: usize,
    pub retained_output_frame_indices: usize,
    pub cpu_low_res_mask_bytes: usize,
    pub device_low_res_mask_bytes: usize,
    pub hotstart_buffered_frames: usize,
    pub hotstart_buffered_cpu_bytes: usize,
    pub hotstart_buffered_device_bytes: usize,
    pub peak_hotstart_buffered_frames: usize,
    pub peak_hotstart_buffered_cpu_bytes: usize,
    pub peak_hotstart_buffered_device_bytes: usize,
    pub cpu_bytes: SessionCacheByteBreakdown,
    pub device_bytes: SessionCacheByteBreakdown,
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
    pub frame_outputs: BTreeSet<usize>,
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
            frame_outputs: BTreeSet::new(),
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
    ) -> Option<(usize, TextPromptTokens)> {
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
                .copied()
                .collect(),
            PropagationDirection::Backward => self
                .frame_outputs
                .range((frame_idx + 1)..)
                .take(limit)
                .copied()
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

    pub(crate) fn retain_output_frames_up_to(&mut self, frame_idx: usize) {
        self.frame_outputs.retain(|idx| *idx <= frame_idx);
    }

    pub(crate) fn record_output_frame(&mut self, frame_idx: usize) {
        self.frame_outputs.insert(frame_idx);
    }

    pub(crate) fn remove_output_frame(&mut self, frame_idx: usize) {
        self.frame_outputs.remove(&frame_idx);
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
    debug_recorder: Option<VideoDebugRecorder>,
    storage_device: Device,
    pub(super) tracked_objects: BTreeMap<u32, TrackedObject>,
    next_obj_id: u32,
    pub(super) frame_outputs: BTreeMap<usize, BTreeMap<u32, ObjectFrameOutput>>,
    pub(super) temporal_disambiguation_metadata:
        BTreeMap<usize, TemporalDisambiguationFrameMetadata>,
    feature_cache: HashMap<usize, VisualBackboneOutput>,
    feature_cache_order: VecDeque<usize>,
    text_cache: HashMap<TextTokenCacheKey, CachedTextPrompt>,
    hotstart_buffered_frames: usize,
    hotstart_buffered_cpu_bytes: usize,
    hotstart_buffered_device_bytes: usize,
    peak_hotstart_buffered_frames: usize,
    peak_hotstart_buffered_cpu_bytes: usize,
    peak_hotstart_buffered_device_bytes: usize,
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
        if let Some(tokens) = session_options.visual_prompt_tokens.as_ref() {
            validate_text_tokens(tokens, model.config().text.context_length)?;
        }
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
            debug_recorder: VideoDebugRecorder::new(&session_id, debug_config)?,
            storage_device,
            tracked_objects: BTreeMap::new(),
            next_obj_id: 0,
            frame_outputs: BTreeMap::new(),
            temporal_disambiguation_metadata: BTreeMap::new(),
            feature_cache: HashMap::new(),
            feature_cache_order: VecDeque::new(),
            text_cache: HashMap::new(),
            hotstart_buffered_frames: 0,
            hotstart_buffered_cpu_bytes: 0,
            hotstart_buffered_device_bytes: 0,
            peak_hotstart_buffered_frames: 0,
            peak_hotstart_buffered_cpu_bytes: 0,
            peak_hotstart_buffered_device_bytes: 0,
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
        let (frame_cpu_bytes, frame_device_bytes) = self.frame_source.memory_bytes();
        let mut cpu_bytes = SessionCacheByteBreakdown {
            frames: frame_cpu_bytes,
            ..Default::default()
        };
        let mut device_bytes = SessionCacheByteBreakdown {
            frames: frame_device_bytes,
            ..Default::default()
        };
        let mut cpu_low_res_mask_bytes = 0usize;
        let mut device_low_res_mask_bytes = 0usize;

        for visual in self.feature_cache.values() {
            let (cpu, device) = visual.memory_bytes();
            cpu_bytes.visual_features = cpu_bytes.visual_features.saturating_add(cpu);
            device_bytes.visual_features = device_bytes.visual_features.saturating_add(device);
        }
        for output_by_object in self.frame_outputs.values() {
            for output in output_by_object.values() {
                let (cpu, device) = output.memory_bytes();
                cpu_bytes.cached_outputs = cpu_bytes.cached_outputs.saturating_add(cpu);
                device_bytes.cached_outputs = device_bytes.cached_outputs.saturating_add(device);
            }
        }
        for object in self.tracked_objects.values() {
            for mask in object.mask_prompt_frames.values() {
                add_tensor_memory(
                    mask,
                    &mut cpu_bytes.tracker_states,
                    &mut device_bytes.tracker_states,
                );
            }
            for state in object.tracker_states.values() {
                let (cpu, device) = state.memory_bytes();
                cpu_bytes.tracker_states = cpu_bytes.tracker_states.saturating_add(cpu);
                device_bytes.tracker_states = device_bytes.tracker_states.saturating_add(device);
                add_tensor_memory(
                    &state.low_res_masks,
                    &mut cpu_low_res_mask_bytes,
                    &mut device_low_res_mask_bytes,
                );
            }
            let (cpu, device) = object.prompt_history_cache.memory_bytes();
            cpu_bytes.packed_prompt_history = cpu_bytes.packed_prompt_history.saturating_add(cpu);
            device_bytes.packed_prompt_history =
                device_bytes.packed_prompt_history.saturating_add(device);
        }
        for cached in self.text_cache.values() {
            let (cpu, device) = cached.memory_bytes();
            cpu_bytes.text_cache = cpu_bytes.text_cache.saturating_add(cpu);
            device_bytes.text_cache = device_bytes.text_cache.saturating_add(device);
        }

        let retained_tracker_states = self
            .tracked_objects
            .values()
            .map(|object| object.tracker_states.len())
            .sum();
        let retained_non_cond_tracker_states = self
            .tracked_objects
            .values()
            .map(|object| {
                object
                    .tracker_states
                    .values()
                    .filter(|state| !state.is_cond_frame)
                    .count()
            })
            .sum();
        let retained_output_frame_indices = self
            .tracked_objects
            .values()
            .map(|object| object.frame_outputs.len())
            .sum();

        SessionCacheStats {
            loaded_frame_count: self.frame_source.loaded_frame_count(),
            cached_feature_entries: self.feature_cache.len(),
            cached_output_frames: self.frame_outputs.len(),
            tracked_objects: self.tracked_objects.len(),
            retained_tracker_states,
            retained_non_cond_tracker_states,
            retained_output_frame_indices,
            cpu_low_res_mask_bytes,
            device_low_res_mask_bytes,
            hotstart_buffered_frames: self.hotstart_buffered_frames,
            hotstart_buffered_cpu_bytes: self.hotstart_buffered_cpu_bytes,
            hotstart_buffered_device_bytes: self.hotstart_buffered_device_bytes,
            peak_hotstart_buffered_frames: self.peak_hotstart_buffered_frames,
            peak_hotstart_buffered_cpu_bytes: self.peak_hotstart_buffered_cpu_bytes,
            peak_hotstart_buffered_device_bytes: self.peak_hotstart_buffered_device_bytes,
            cpu_bytes,
            device_bytes,
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

    pub(crate) fn memory_profile(&self) -> &VideoMemoryProfile {
        &self.session_options.memory_profile
    }

    pub(crate) fn visual_prompt_tokens(&self) -> Option<&TextPromptTokens> {
        self.session_options.visual_prompt_tokens.as_ref()
    }

    pub(crate) fn low_memory_mode(&self) -> bool {
        matches!(
            self.session_options.memory_profile,
            VideoMemoryProfile::LowMemory
        )
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
            object.retain_output_frames_up_to(frame_idx);
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
        self.clear_hotstart_stats();
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
        self.clear_hotstart_stats();
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
        let mut keep = self.prefetch_window(frame_idx, direction);
        if !self.low_memory_mode() {
            keep.extend(self.prompt_frames());
        }
        self.frame_source.evict_except(&keep);
    }

    pub(crate) fn evict_cached_output_frame(&mut self, frame_idx: usize) {
        self.frame_outputs.remove(&frame_idx);
    }

    pub(crate) fn record_hotstart_buffer(&mut self, buffer: &VecDeque<VideoFrameOutput>) {
        let mut cpu_bytes = 0usize;
        let mut device_bytes = 0usize;
        for frame in buffer {
            for output in &frame.objects {
                let (cpu, device) = output.memory_bytes();
                cpu_bytes = cpu_bytes.saturating_add(cpu);
                device_bytes = device_bytes.saturating_add(device);
            }
        }
        self.hotstart_buffered_frames = buffer.len();
        self.hotstart_buffered_cpu_bytes = cpu_bytes;
        self.hotstart_buffered_device_bytes = device_bytes;
        self.peak_hotstart_buffered_frames = self.peak_hotstart_buffered_frames.max(buffer.len());
        self.peak_hotstart_buffered_cpu_bytes =
            self.peak_hotstart_buffered_cpu_bytes.max(cpu_bytes);
        self.peak_hotstart_buffered_device_bytes =
            self.peak_hotstart_buffered_device_bytes.max(device_bytes);
    }

    pub(crate) fn evict_bounded_tracker_history(&mut self, direction: PropagationDirection) {
        let Some(limit) = self.session_options.max_non_cond_tracker_states else {
            return;
        };
        let mut evicted = Vec::new();
        for (obj_id, object) in self.tracked_objects.iter_mut() {
            let protected_prompt_frames = object
                .prompt_frames
                .keys()
                .chain(object.mask_prompt_frames.keys())
                .copied()
                .collect::<BTreeSet<_>>();
            let mut candidates = object
                .tracker_states
                .iter()
                .filter_map(|(idx, state)| {
                    (!state.is_cond_frame && !protected_prompt_frames.contains(idx)).then_some(*idx)
                })
                .collect::<Vec<_>>();
            if candidates.len() <= limit {
                continue;
            }
            candidates.sort_unstable();
            let remove_count = candidates.len() - limit;
            let remove = match direction {
                PropagationDirection::Forward | PropagationDirection::Both => candidates
                    .into_iter()
                    .take(remove_count)
                    .collect::<Vec<_>>(),
                PropagationDirection::Backward => candidates
                    .into_iter()
                    .rev()
                    .take(remove_count)
                    .collect::<Vec<_>>(),
            };
            for evicted_frame_idx in remove {
                object.tracker_states.remove(&evicted_frame_idx);
                object.remove_output_frame(evicted_frame_idx);
                evicted.push((*obj_id, evicted_frame_idx));
            }
            object.clear_prompt_history_cache();
        }
        for (obj_id, evicted_frame_idx) in evicted {
            let remove_frame = if let Some(outputs) = self.frame_outputs.get_mut(&evicted_frame_idx)
            {
                outputs.remove(&obj_id);
                outputs.is_empty()
            } else {
                false
            };
            if remove_frame {
                self.frame_outputs.remove(&evicted_frame_idx);
            }
        }
    }

    fn clear_hotstart_stats(&mut self) {
        self.hotstart_buffered_frames = 0;
        self.hotstart_buffered_cpu_bytes = 0;
        self.hotstart_buffered_device_bytes = 0;
        self.peak_hotstart_buffered_frames = 0;
        self.peak_hotstart_buffered_cpu_bytes = 0;
        self.peak_hotstart_buffered_device_bytes = 0;
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
            let mut visual = move_visual_output(cached, compute_device)?;
            if self.low_memory_mode() {
                visual.ensure_tracker_sequences()?;
            }
            self.touch_feature_cache_entry(frame_idx);
            return Ok(visual);
        }

        let image = self.get_frame(frame_idx, compute_device)?;
        let visual = model.encode_image_features(&image)?;
        let mut stored = move_visual_output(&visual, self.storage_device())?;
        if self.low_memory_mode() {
            stored.strip_tracker_sequences();
        }
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
        text_prompt: &TextPromptTokens,
        compute_device: &Device,
    ) -> Result<TextEncoding> {
        validate_text_tokens(text_prompt, model.config().text.context_length)?;
        let cache_key = TextTokenCacheKey::from(text_prompt);
        if let Some(cached) = self.text_cache.get(&cache_key) {
            return cached.to_text_encoding(compute_device);
        }
        let input_ids = Tensor::new(vec![text_prompt.input_ids.clone()], compute_device)?;
        let attention_mask = Tensor::new(vec![text_prompt.attention_mask.clone()], compute_device)?;
        let encoding = model.encode_text_tokens(&input_ids, &attention_mask)?;
        let cached = CachedTextPrompt::from_encoding(&encoding, self.storage_device())?;
        self.text_cache.insert(cache_key, cached);
        Ok(encoding)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct TextTokenCacheKey {
    input_ids: Vec<u32>,
    attention_mask: Vec<u32>,
}

impl From<&TextPromptTokens> for TextTokenCacheKey {
    fn from(value: &TextPromptTokens) -> Self {
        Self {
            input_ids: value.input_ids.clone(),
            attention_mask: value.attention_mask.clone(),
        }
    }
}

fn validate_text_tokens(tokens: &TextPromptTokens, context_length: usize) -> Result<()> {
    tokens.validate()?;
    if tokens.input_ids.len() != context_length {
        candle::bail!(
            "text prompt has {} tokens but SAM3 requires exactly {}",
            tokens.input_ids.len(),
            context_length
        )
    }
    Ok(())
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
            input_embeddings: if encoding
                .input_embeddings
                .device()
                .same_device(storage_device)
            {
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

    fn memory_bytes(&self) -> (usize, usize) {
        let mut cpu = 0usize;
        let mut device = 0usize;
        add_tensor_memory(&self.attention_mask, &mut cpu, &mut device);
        add_tensor_memory(&self.memory, &mut cpu, &mut device);
        add_tensor_memory(&self.input_embeddings, &mut cpu, &mut device);
        (cpu, device)
    }
}

#[cfg(feature = "sam3-parity-support")]
impl Sam3VideoSessionParityExt for Sam3VideoSession {
    fn parity_tracked_objects(&self) -> &BTreeMap<u32, TrackedObject> {
        &self.tracked_objects
    }

    fn parity_tracked_objects_mut(&mut self) -> &mut BTreeMap<u32, TrackedObject> {
        &mut self.tracked_objects
    }

    fn parity_frame_outputs(&self) -> &BTreeMap<usize, BTreeMap<u32, ObjectFrameOutput>> {
        &self.frame_outputs
    }

    fn parity_frame_outputs_mut(
        &mut self,
    ) -> &mut BTreeMap<usize, BTreeMap<u32, ObjectFrameOutput>> {
        &mut self.frame_outputs
    }

    fn parity_temporal_disambiguation_metadata(
        &self,
    ) -> BTreeMap<usize, ParityTemporalDisambiguationFrameMetadata> {
        self.temporal_disambiguation_metadata
            .iter()
            .map(|(frame_idx, metadata)| {
                (
                    *frame_idx,
                    ParityTemporalDisambiguationFrameMetadata {
                        removed_obj_ids: metadata.removed_obj_ids.clone(),
                        suppressed_obj_ids: metadata.suppressed_obj_ids.clone(),
                        unconfirmed_obj_ids: metadata.unconfirmed_obj_ids.clone(),
                        matched_obj_ids: metadata.matched_obj_ids.clone(),
                        unmatched_obj_ids: metadata.unmatched_obj_ids.clone(),
                    },
                )
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::super::postprocess::persist_visible_frame_output;
    use super::*;

    struct DummyFrameSource {
        frame: Tensor,
    }

    impl DummyFrameSource {
        fn new() -> Result<Self> {
            Ok(Self {
                frame: Tensor::zeros((3, 2, 2), DType::F32, &Device::Cpu)?,
            })
        }
    }

    impl FrameSource for DummyFrameSource {
        fn frame_count(&self) -> usize {
            8
        }

        fn video_size(&self) -> ImageSize {
            ImageSize::new(2, 2)
        }

        fn get_frame(&mut self, frame_idx: usize, target_device: &Device) -> Result<Tensor> {
            if frame_idx >= self.frame_count() {
                candle::bail!("frame_idx {} out of bounds", frame_idx);
            }
            self.frame.to_device(target_device)
        }

        fn prefetch(&mut self, _frame_indices: &[usize]) -> Result<()> {
            Ok(())
        }

        fn evict_except(&mut self, _keep_frame_indices: &BTreeSet<usize>) {}

        fn loaded_frame_count(&self) -> usize {
            1
        }

        fn memory_bytes(&self) -> (usize, usize) {
            (self.frame.elem_count() * std::mem::size_of::<f32>(), 0)
        }

        fn close(&mut self) {}
    }

    fn test_session(memory_profile: VideoMemoryProfile) -> Result<Sam3VideoSession> {
        Ok(Sam3VideoSession {
            session_id: "test".to_owned(),
            frame_source: Box::new(DummyFrameSource::new()?),
            session_options: VideoSessionOptions {
                memory_profile,
                ..VideoSessionOptions::default()
            },
            debug_recorder: None,
            storage_device: Device::Cpu,
            tracked_objects: BTreeMap::new(),
            next_obj_id: 1,
            frame_outputs: BTreeMap::new(),
            temporal_disambiguation_metadata: BTreeMap::new(),
            feature_cache: HashMap::new(),
            feature_cache_order: VecDeque::new(),
            text_cache: HashMap::new(),
            hotstart_buffered_frames: 0,
            hotstart_buffered_cpu_bytes: 0,
            hotstart_buffered_device_bytes: 0,
            peak_hotstart_buffered_frames: 0,
            peak_hotstart_buffered_cpu_bytes: 0,
            peak_hotstart_buffered_device_bytes: 0,
        })
    }

    fn test_tracker_state() -> Result<TrackerFrameState> {
        Ok(TrackerFrameState {
            low_res_masks: Tensor::zeros((1, 1, 1, 1), DType::F32, &Device::Cpu)?,
            high_res_masks: Tensor::zeros((1, 1, 2, 2), DType::F32, &Device::Cpu)?,
            iou_scores: Tensor::ones((1, 1), DType::F32, &Device::Cpu)?,
            memory_selection_score: None,
            obj_ptr: Tensor::zeros((1, 4), DType::F32, &Device::Cpu)?,
            object_score_logits: Tensor::zeros((1, 1), DType::F32, &Device::Cpu)?,
            maskmem_features: None,
            maskmem_pos_enc: None,
            maskmem_prompt_features: None,
            maskmem_prompt_pos_enc: None,
            is_cond_frame: false,
        })
    }

    fn test_output(obj_id: u32) -> Result<ObjectFrameOutput> {
        let mask_logits = Tensor::zeros((1, 1, 2, 2), DType::F32, &Device::Cpu)?;
        let masks = candle_nn::ops::sigmoid(&mask_logits)?;
        Ok(ObjectFrameOutput {
            obj_id,
            mask_logits,
            masks: masks.clone(),
            boxes_xyxy: mask_to_normalized_xyxy(&masks)?,
            scores: Tensor::ones((1,), DType::F32, &Device::Cpu)?,
            presence_scores: Some(Tensor::ones((1,), DType::F32, &Device::Cpu)?),
            prompt_frame_idx: Some(0),
            memory_frame_indices: vec![0],
            text_prompt: None,
            used_explicit_geometry: false,
            reused_previous_output: false,
        })
    }

    #[test]
    fn low_memory_persist_stores_compact_outputs_and_keeps_output_index() -> Result<()> {
        let mut session = test_session(VideoMemoryProfile::LowMemory)?;
        let mut object = TrackedObject::new(7, 0);
        object.tracker_states.insert(0, test_tracker_state()?);
        session.tracked_objects.insert(7, object);
        let frame_output = VideoFrameOutput {
            frame_idx: 0,
            objects: vec![test_output(7)?],
        };

        persist_visible_frame_output(&mut session, &frame_output)?;

        let stored = session
            .frame_outputs
            .get(&0)
            .and_then(|outputs| outputs.get(&7))
            .expect("stored output should exist");
        assert_eq!(stored.masks.elem_count(), 0);
        assert_eq!(stored.boxes_xyxy.elem_count(), 0);
        assert!(session
            .tracked_objects
            .get(&7)
            .expect("tracked object should exist")
            .frame_outputs
            .contains(&0));
        Ok(())
    }

    #[test]
    fn evict_cached_output_frame_preserves_lightweight_output_metadata() -> Result<()> {
        let mut session = test_session(VideoMemoryProfile::LowMemory)?;
        let mut object = TrackedObject::new(7, 0);
        object.record_output_frame(0);
        session.tracked_objects.insert(7, object);
        let mut outputs = BTreeMap::new();
        outputs.insert(7, test_output(7)?);
        session.frame_outputs.insert(0, outputs);

        session.evict_cached_output_frame(0);

        assert!(!session.frame_outputs.contains_key(&0));
        assert!(session
            .tracked_objects
            .get(&7)
            .expect("tracked object should exist")
            .frame_outputs
            .contains(&0));
        Ok(())
    }

    fn bounded_history_session() -> Result<Sam3VideoSession> {
        let mut session = test_session(VideoMemoryProfile::LowMemory)?;
        session.session_options.max_non_cond_tracker_states = Some(2);
        let mut object = TrackedObject::new(7, 0);
        for frame_idx in 0..6 {
            let mut state = test_tracker_state()?;
            state.is_cond_frame = frame_idx == 0;
            object.tracker_states.insert(frame_idx, state);
            object.record_output_frame(frame_idx);
            session
                .frame_outputs
                .entry(frame_idx)
                .or_default()
                .insert(7, test_output(7)?);
        }
        session.tracked_objects.insert(7, object);
        Ok(session)
    }

    #[test]
    fn bounded_history_retains_conditioning_and_directional_windows() -> Result<()> {
        let mut forward = bounded_history_session()?;
        forward.evict_bounded_tracker_history(PropagationDirection::Forward);
        let object = forward.tracked_objects.get(&7).expect("object");
        assert_eq!(
            object.tracker_states.keys().copied().collect::<Vec<_>>(),
            vec![0, 4, 5]
        );
        assert_eq!(
            object.frame_outputs.iter().copied().collect::<Vec<_>>(),
            vec![0, 4, 5]
        );
        assert_eq!(
            forward.frame_outputs.keys().copied().collect::<Vec<_>>(),
            vec![0, 4, 5]
        );
        let stats = forward.cache_stats();
        assert_eq!(stats.retained_tracker_states, 3);
        assert_eq!(stats.retained_non_cond_tracker_states, 2);
        assert_eq!(stats.retained_output_frame_indices, 3);
        assert_eq!(stats.cpu_low_res_mask_bytes, 3 * std::mem::size_of::<f32>());
        assert_eq!(stats.device_low_res_mask_bytes, 0);

        let mut backward = bounded_history_session()?;
        backward.evict_bounded_tracker_history(PropagationDirection::Backward);
        let object = backward.tracked_objects.get(&7).expect("object");
        assert_eq!(
            object.tracker_states.keys().copied().collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        Ok(())
    }

    #[test]
    fn later_prompt_correction_survives_old_state_eviction() -> Result<()> {
        let mut session = bounded_history_session()?;
        session.evict_bounded_tracker_history(PropagationDirection::Forward);
        assert!(!session
            .tracked_objects
            .get(&7)
            .expect("object")
            .tracker_states
            .contains_key(&1));

        let returned = session.add_prompt(
            1,
            SessionPrompt {
                text: None,
                points: Some(vec![(0.5, 0.5)]),
                point_labels: Some(vec![1]),
                boxes: None,
                box_labels: None,
            },
            Some(7),
            true,
            false,
            8,
        )?;
        assert_eq!(returned, 7);
        let object = session.tracked_objects.get(&7).expect("object");
        assert!(object.prompt_frames.contains_key(&1));
        assert!(object.tracker_states.keys().all(|idx| *idx <= 1));
        Ok(())
    }

    #[test]
    fn hotstart_buffer_stats_are_separate_bounded_and_resettable() -> Result<()> {
        let mut session = test_session(VideoMemoryProfile::LowMemory)?;
        let mut buffer = VecDeque::new();
        buffer.push_back(VideoFrameOutput {
            frame_idx: 0,
            objects: vec![test_output(7)?],
        });
        buffer.push_back(VideoFrameOutput {
            frame_idx: 1,
            objects: vec![test_output(7)?],
        });
        session.record_hotstart_buffer(&buffer);
        let buffered = session.cache_stats();
        assert_eq!(buffered.hotstart_buffered_frames, 2);
        assert_eq!(buffered.peak_hotstart_buffered_frames, 2);
        assert!(buffered.hotstart_buffered_cpu_bytes > 0);

        buffer.clear();
        session.record_hotstart_buffer(&buffer);
        let drained = session.cache_stats();
        assert_eq!(drained.hotstart_buffered_frames, 0);
        assert_eq!(drained.peak_hotstart_buffered_frames, 2);
        session.reset();
        assert_eq!(session.cache_stats().peak_hotstart_buffered_frames, 0);
        Ok(())
    }

    #[test]
    fn reset_and_close_release_bounded_history_and_output_state() -> Result<()> {
        let mut reset = bounded_history_session()?;
        reset.reset();
        let reset_stats = reset.cache_stats();
        assert_eq!(reset_stats.retained_tracker_states, 0);
        assert_eq!(reset_stats.retained_output_frame_indices, 0);
        assert_eq!(reset_stats.cached_output_frames, 0);
        assert_eq!(reset_stats.cpu_low_res_mask_bytes, 0);

        let mut close = bounded_history_session()?;
        close.close();
        let close_stats = close.cache_stats();
        assert_eq!(close_stats.retained_tracker_states, 0);
        assert_eq!(close_stats.retained_output_frame_indices, 0);
        assert_eq!(close_stats.cached_output_frames, 0);
        assert_eq!(close_stats.cpu_low_res_mask_bytes, 0);
        Ok(())
    }
}
#[test]
fn text_cache_key_depends_only_on_token_ids_and_attention_mask() {
    let first = TextPromptTokens::new(vec![1, 2, 3], vec![1, 1, 1]).with_display_text("person");
    let renamed =
        TextPromptTokens::new(vec![1, 2, 3], vec![1, 1, 1]).with_display_text("diagnostic alias");
    let changed_mask = TextPromptTokens::new(vec![1, 2, 3], vec![1, 1, 0]);

    assert_eq!(
        TextTokenCacheKey::from(&first),
        TextTokenCacheKey::from(&renamed)
    );
    assert_ne!(
        TextTokenCacheKey::from(&first),
        TextTokenCacheKey::from(&changed_mask)
    );
}

#[test]
fn text_token_contract_rejects_empty_mismatched_and_wrong_context_inputs() {
    assert!(TextPromptTokens::new(Vec::new(), Vec::new())
        .validate()
        .is_err());
    assert!(TextPromptTokens::new(vec![1, 2], vec![1])
        .validate()
        .is_err());
    assert!(validate_text_tokens(&TextPromptTokens::new(vec![1], vec![1]), 2).is_err());
}
