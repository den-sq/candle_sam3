use super::*;
use crate::models::sam3::torch_ops::tensor::first_scalar_f32;

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

    pub(super) fn with_default_labels(mut self) -> Result<Self> {
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

    pub(super) fn merge_from(
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
    pub(super) fn from_grounding(
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

    pub(super) fn score_value(&self) -> Result<f32> {
        first_scalar_f32(&self.scores)
    }

    pub(super) fn to_storage_device(&self, storage_device: &Device) -> Result<Self> {
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

    pub(super) fn grounding(&self) -> GroundingOutput {
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
        let predictor_config = &self.tracker_core.tracker.config().predictor;
        let hotstart_delay = predictor_config.hotstart_delay;
        let hotstart_unmatch_thresh = predictor_config.hotstart_unmatch_thresh;
        let recent_occlusion_suppression_threshold =
            predictor_config.suppress_overlapping_based_on_recent_occlusion_threshold;
        let confirmation_threshold = if predictor_config.masklet_confirmation_enable {
            predictor_config.masklet_confirmation_consecutive_det_thresh
        } else {
            0
        };
        let mut temporal_disambiguation_state = TemporalDisambiguationState::default();
        session.clear_temporal_disambiguation_metadata();
        if let Some(start_frame_idx) = processing_order.first().copied() {
            temporal_disambiguation_state.seed_prompt_frame_confirmation(
                session,
                start_frame_idx,
                options.direction,
                confirmation_threshold,
            );
        }
        let mut hotstart_buffer = VecDeque::new();
        for (order_idx, frame_idx) in processing_order.iter().copied().enumerate() {
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
            let mut matched_obj_ids = output
                .objects
                .iter()
                .filter(|object| {
                    session
                        .tracked_objects
                        .get(&object.obj_id)
                        .map(|tracked| tracked.has_prompt_on_frame(frame_idx))
                        .unwrap_or(false)
                })
                .map(|object| object.obj_id)
                .collect::<BTreeSet<_>>();
            if confirmation_threshold > 0 {
                matched_obj_ids.extend(self.tracker_core.collect_text_detector_matched_obj_ids(
                    self.model,
                    self.device,
                    session,
                    frame_idx,
                    options.direction,
                    &output,
                )?);
            }
            temporal_disambiguation_state.record_confirmation_status(
                session,
                frame_idx,
                options.direction,
                &matched_obj_ids,
                confirmation_threshold,
            );
            let unmatched_obj_ids = output
                .objects
                .iter()
                .map(|object| object.obj_id)
                .filter(|obj_id| !matched_obj_ids.contains(obj_id))
                .collect::<BTreeSet<_>>();
            let previous_removed_obj_ids = temporal_disambiguation_state.hidden_obj_ids().clone();
            let is_last_frame = order_idx + 1 == processing_order.len();
            let mut yield_list = Vec::new();
            if hotstart_delay > 0 {
                temporal_disambiguation_state.record_unmatched_outputs(
                    session,
                    frame_idx,
                    &unmatched_obj_ids,
                    options.direction,
                    hotstart_delay,
                    hotstart_unmatch_thresh,
                );
                let frame_has_prompt_input = session.prompt_frames().contains(&frame_idx);
                if frame_has_prompt_input {
                    yield_list.push(output);
                } else {
                    hotstart_buffer.push_back(output);
                    if is_last_frame {
                        yield_list.extend(hotstart_buffer.drain(..));
                    } else if hotstart_buffer.len() >= hotstart_delay {
                        if let Some(oldest) = hotstart_buffer.pop_front() {
                            yield_list.push(oldest);
                        }
                    }
                }
            } else {
                yield_list.push(output);
            }
            let current_removed_obj_ids = temporal_disambiguation_state.hidden_obj_ids().clone();
            let newly_removed_obj_ids = current_removed_obj_ids
                .difference(&previous_removed_obj_ids)
                .copied()
                .collect::<BTreeSet<_>>();
            let current_suppressed_obj_ids = temporal_disambiguation_state
                .record_recent_occlusion_suppression(
                    yield_list.last().unwrap_or_else(|| {
                        hotstart_buffer
                            .back()
                            .expect("output should be in yield list or hotstart buffer")
                    }),
                    frame_idx,
                    options.direction,
                    recent_occlusion_suppression_threshold,
                    &newly_removed_obj_ids,
                )?;
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
                    suppressed_obj_ids: current_suppressed_obj_ids,
                    unconfirmed_obj_ids: current_unconfirmed_obj_ids,
                    matched_obj_ids: matched_obj_ids.clone(),
                    unmatched_obj_ids: unmatched_obj_ids.clone(),
                },
            );
            for yielded in yield_list {
                let is_prompt_frame = session.prompt_frames().contains(&yielded.frame_idx);
                if hotstart_delay > 0 && is_prompt_frame {
                    on_frame(&yielded)?;
                    continue;
                }
                let mut hidden_obj_ids = temporal_disambiguation_state.hidden_obj_ids().clone();
                hidden_obj_ids.extend(
                    temporal_disambiguation_state.unconfirmed_obj_ids_for_yield_frame(
                        yielded.frame_idx,
                        options.direction,
                        session.num_frames(),
                        confirmation_threshold,
                    ),
                );
                if let Some(frame_metadata) = session
                    .temporal_disambiguation_metadata
                    .get(&yielded.frame_idx)
                {
                    hidden_obj_ids.extend(frame_metadata.suppressed_obj_ids.iter().copied());
                }
                let filtered = filter_video_frame_output(&yielded, &hidden_obj_ids);
                persist_visible_frame_output(session, &filtered)?;
                on_frame(&filtered)?;
            }
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
    pub(super) tracker: &'a Sam3TrackerModel,
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

    fn collect_text_detector_matched_obj_ids(
        &self,
        model: &Sam3ImageModel,
        compute_device: &Device,
        session: &mut Sam3VideoSession,
        frame_idx: usize,
        direction: PropagationDirection,
        frame_output: &VideoFrameOutput,
    ) -> Result<BTreeSet<u32>> {
        const ASSOCIATION_IOU_THRESHOLD: f32 = 0.5;
        let mut matched = BTreeSet::new();
        if frame_output.objects.is_empty() {
            return Ok(matched);
        }
        let visual_features = session.get_visual_features(model, compute_device, frame_idx)?;
        for output in frame_output.objects.iter() {
            let Some(object) = session.tracked_objects.get(&output.obj_id) else {
                continue;
            };
            let Some((_prompt_frame_idx, text_prompt)) =
                object.latest_text_prompt(frame_idx, direction)
            else {
                continue;
            };
            let text_encoding =
                session.cached_text_encoding(model, &text_prompt, compute_device)?;
            let encoded_prompt =
                combine_encoded_prompts(Some(&text_encoding), None)?.ok_or_else(|| {
                    candle::Error::Msg("text detector path produced no encoded prompt".to_owned())
                })?;
            let grounding = ground_from_encoded_prompt(model, &visual_features, &encoded_prompt)?;
            let detector_output = grounding_to_object_output(
                output.obj_id,
                &grounding,
                Some(frame_idx),
                Vec::new(),
                Some(text_prompt),
                false,
                false,
                session.video_size(),
            )?;
            let detector_plane =
                mask_to_bool_plane(&detector_output.masks, VIDEO_DEBUG_MASK_THRESHOLD)?;
            let tracker_plane = mask_to_bool_plane(&output.masks, VIDEO_DEBUG_MASK_THRESHOLD)?;
            if binary_planes_iou(&detector_plane, &tracker_plane) >= ASSOCIATION_IOU_THRESHOLD {
                matched.insert(output.obj_id);
            }
        }
        Ok(matched)
    }

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
        let obj_ids: Vec<u32> = session.tracked_objects.keys().copied().collect();
        let predictor_config = &self.tracker.config().predictor;
        let latest_session_input_frame = session
            .tracked_objects
            .values()
            .filter_map(|object| object.nearest_input_frame_idx(frame_idx, direction))
            .max();
        let mut pending_results = Vec::new();
        for obj_id in obj_ids {
            if let Some(cached) = session
                .frame_outputs
                .get(&frame_idx)
                .and_then(|frame_outputs| frame_outputs.get(&obj_id))
                .cloned()
            {
                pending_results.push((
                    obj_id,
                    cached.to_storage_device(compute_device)?,
                    session
                        .tracked_objects
                        .get(&obj_id)
                        .and_then(|object| object.tracker_states.get(&frame_idx))
                        .cloned()
                        .ok_or_else(|| {
                            candle::Error::Msg(format!(
                                "cached output for obj_id {} on frame {} missing tracker state",
                                obj_id, frame_idx
                            ))
                        })?,
                    session
                        .tracked_objects
                        .get(&obj_id)
                        .and_then(|object| object.display_score),
                ));
                continue;
            }
            if session
                .tracked_objects
                .get(&obj_id)
                .map(|object| {
                    object.tracker_states.contains_key(&frame_idx)
                        && !object.frame_outputs.contains_key(&frame_idx)
                })
                .unwrap_or(false)
            {
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
                        config,
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
                        config,
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
                        config,
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
                        config,
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
                        config,
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
                let own_latest_input_frame =
                    object_snapshot.nearest_input_frame_idx(frame_idx, direction);
                let can_extend_uninteracted_multi_object = predictor_config
                    .clear_non_cond_mem_for_multi_obj
                    && session.tracked_objects.len() > 1
                    && latest_session_input_frame
                        .zip(own_latest_input_frame)
                        .map_or(false, |(session_latest, own_latest)| {
                            session_latest > own_latest
                        })
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
                            config,
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
            pending_results.push((obj_id, output, tracker_state, display_score));
        }

        let frame_objects = self.postprocess_output(
            config,
            session,
            &pending_results,
            output_threshold,
            None,
            None,
            None,
        )?;
        let mut stored_outputs = BTreeMap::new();
        for output in frame_objects.iter() {
            stored_outputs.insert(
                output.obj_id,
                output.to_storage_device(session.storage_device())?,
            );
        }
        if stored_outputs.is_empty() {
            session.frame_outputs.remove(&frame_idx);
        } else {
            session
                .frame_outputs
                .insert(frame_idx, stored_outputs.clone());
        }

        for (obj_id, _output, tracker_state, display_score) in pending_results {
            let tracker_storage = move_tracker_state(&tracker_state, session.storage_device())?;
            if let Some(object) = session.tracked_objects.get_mut(&obj_id) {
                object.tracker_states.insert(frame_idx, tracker_storage);
                object.has_inference_history = true;
                object.last_updated_frame = frame_idx;
                if let Some(display_score) = display_score {
                    object.display_score = Some(display_score);
                }
                if let Some(stored_output) = stored_outputs.get(&obj_id) {
                    object
                        .frame_outputs
                        .insert(frame_idx, stored_output.clone());
                } else {
                    object.frame_outputs.remove(&frame_idx);
                }
            }
        }
        Ok(VideoFrameOutput {
            frame_idx,
            objects: frame_objects,
        })
    }
}

pub(super) fn move_visual_output(
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

pub(super) fn tracker_visual_output(output: &VisualBackboneOutput) -> VisualBackboneOutput {
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

pub(super) fn normalize_video_mask_prompt(mask: &Tensor, device: &Device) -> Result<Tensor> {
    let mask = mask.to_device(device)?.to_dtype(DType::F32)?;
    match mask.rank() {
        2 => mask.unsqueeze(0)?.unsqueeze(0),
        3 => mask.unsqueeze(1),
        4 => Ok(mask),
        rank => candle::bail!("expected video mask prompt rank 2/3/4, got {rank}"),
    }
}

pub(super) fn move_tracker_state(
    state: &TrackerFrameState,
    device: &Device,
) -> Result<TrackerFrameState> {
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

pub(super) fn resize_mask_prompt_to_video(
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

pub(super) fn resize_mask_prompt_to_tracker_input(
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

pub(super) fn build_processing_order(
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
