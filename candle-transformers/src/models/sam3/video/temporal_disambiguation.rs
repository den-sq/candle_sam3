use super::*;

#[derive(Debug, Clone)]
pub(super) struct TemporalDisambiguationObjectState {
    pub(super) first_frame_idx: usize,
    pub(super) unmatched_frame_indices: Vec<usize>,
    pub(super) consecutive_det_num: usize,
    pub(super) confirmed: bool,
    pub(super) removed: bool,
    pub(super) last_occluded_frame: Option<usize>,
}

#[derive(Debug, Clone, Default)]
pub(super) struct TemporalDisambiguationState {
    pub(super) hotstart_removed_obj_ids: BTreeSet<u32>,
    pub(super) unconfirmed_obj_ids_per_frame: BTreeMap<usize, BTreeSet<u32>>,
    pub(super) object_states: BTreeMap<u32, TemporalDisambiguationObjectState>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(super) struct TemporalDisambiguationFrameMetadata {
    pub(super) removed_obj_ids: BTreeSet<u32>,
    pub(super) suppressed_obj_ids: BTreeSet<u32>,
    pub(super) unconfirmed_obj_ids: BTreeSet<u32>,
    pub(super) matched_obj_ids: BTreeSet<u32>,
    pub(super) unmatched_obj_ids: BTreeSet<u32>,
}

impl TemporalDisambiguationState {
    pub(super) fn ensure_object(&mut self, object: &TrackedObject) {
        self.object_states.entry(object.obj_id).or_insert_with(|| {
            TemporalDisambiguationObjectState {
                first_frame_idx: object.creation_frame,
                unmatched_frame_indices: Vec::new(),
                consecutive_det_num: 0,
                confirmed: false,
                removed: false,
                last_occluded_frame: None,
            }
        });
    }

    pub(super) fn ensure_object_id(&mut self, obj_id: u32, first_frame_idx: usize) {
        self.object_states
            .entry(obj_id)
            .or_insert_with(|| TemporalDisambiguationObjectState {
                first_frame_idx,
                unmatched_frame_indices: Vec::new(),
                consecutive_det_num: 0,
                confirmed: false,
                removed: false,
                last_occluded_frame: None,
            });
    }

    pub(super) fn seed_prompt_frame_confirmation(
        &mut self,
        session: &Sam3VideoSession,
        frame_idx: usize,
        direction: PropagationDirection,
        confirmation_threshold: usize,
    ) {
        if confirmation_threshold == 0 {
            return;
        }
        for object in session.tracked_objects.values() {
            if !object.is_active_for_frame(frame_idx, direction)
                || !object.has_prompt_on_frame(frame_idx)
            {
                continue;
            }
            self.ensure_object(object);
            if let Some(state) = self.object_states.get_mut(&object.obj_id) {
                state.consecutive_det_num = state.consecutive_det_num.max(1);
                if state.consecutive_det_num >= confirmation_threshold {
                    state.confirmed = true;
                }
            }
        }
    }

    pub(super) fn record_unmatched_outputs(
        &mut self,
        session: &Sam3VideoSession,
        frame_idx: usize,
        unmatched_obj_ids: &BTreeSet<u32>,
        direction: PropagationDirection,
        hotstart_delay: usize,
        hotstart_unmatch_thresh: usize,
    ) {
        if hotstart_delay == 0 || hotstart_unmatch_thresh == 0 {
            return;
        }
        for object in session.tracked_objects.values() {
            if object.is_active_for_frame(frame_idx, direction) {
                self.ensure_object(object);
            }
        }
        for obj_id in unmatched_obj_ids.iter().copied() {
            let Some(state) = self.object_states.get_mut(&obj_id) else {
                continue;
            };
            if state.removed {
                continue;
            }
            state.unmatched_frame_indices.push(frame_idx);
            if state.unmatched_frame_indices.len() < hotstart_unmatch_thresh {
                continue;
            }
            let hotstart_diff = match direction {
                PropagationDirection::Backward => frame_idx as isize + hotstart_delay as isize,
                PropagationDirection::Forward | PropagationDirection::Both => {
                    frame_idx as isize - hotstart_delay as isize
                }
            };
            let is_within_hotstart = match direction {
                PropagationDirection::Backward => state.first_frame_idx as isize <= hotstart_diff,
                PropagationDirection::Forward | PropagationDirection::Both => {
                    state.first_frame_idx as isize > hotstart_diff
                }
            };
            if is_within_hotstart {
                state.removed = true;
                self.hotstart_removed_obj_ids.insert(obj_id);
            }
        }
    }

    pub(super) fn hidden_obj_ids(&self) -> &BTreeSet<u32> {
        &self.hotstart_removed_obj_ids
    }

    pub(super) fn record_confirmation_status(
        &mut self,
        session: &Sam3VideoSession,
        frame_idx: usize,
        direction: PropagationDirection,
        matched_obj_ids: &BTreeSet<u32>,
        confirmation_threshold: usize,
    ) {
        if confirmation_threshold == 0 {
            return;
        }
        let mut unconfirmed_obj_ids = BTreeSet::new();
        for object in session.tracked_objects.values() {
            if !object.is_active_for_frame(frame_idx, direction) {
                continue;
            }
            self.ensure_object(object);
            let Some(state) = self.object_states.get_mut(&object.obj_id) else {
                continue;
            };
            if state.removed {
                continue;
            }
            if matched_obj_ids.contains(&object.obj_id) {
                state.consecutive_det_num = state.consecutive_det_num.saturating_add(1);
            } else {
                state.consecutive_det_num = 0;
            }
            if state.consecutive_det_num >= confirmation_threshold {
                state.confirmed = true;
            }
            if !state.confirmed {
                unconfirmed_obj_ids.insert(object.obj_id);
            }
        }
        self.unconfirmed_obj_ids_per_frame
            .insert(frame_idx, unconfirmed_obj_ids);
    }

    pub(super) fn unconfirmed_obj_ids_for_yield_frame(
        &self,
        yield_frame_idx: usize,
        direction: PropagationDirection,
        num_frames: usize,
        confirmation_threshold: usize,
    ) -> BTreeSet<u32> {
        if confirmation_threshold <= 1 || num_frames == 0 {
            return BTreeSet::new();
        }
        let delay = confirmation_threshold - 1;
        let status_frame_idx = match direction {
            PropagationDirection::Backward => yield_frame_idx.saturating_sub(delay),
            PropagationDirection::Forward | PropagationDirection::Both => {
                (yield_frame_idx + delay).min(num_frames.saturating_sub(1))
            }
        };
        self.unconfirmed_obj_ids_per_frame
            .get(&status_frame_idx)
            .cloned()
            .unwrap_or_default()
    }

    pub(super) fn record_recent_occlusion_suppression(
        &mut self,
        output: &VideoFrameOutput,
        frame_idx: usize,
        direction: PropagationDirection,
        overlap_threshold: f32,
        newly_removed_obj_ids: &BTreeSet<u32>,
    ) -> Result<BTreeSet<u32>> {
        if output.objects.is_empty() {
            return Ok(BTreeSet::new());
        }
        let mut suppressed_obj_ids = BTreeSet::new();
        if overlap_threshold > 0.0 && output.objects.len() > 1 {
            let object_ids = output
                .objects
                .iter()
                .map(|object| object.obj_id)
                .collect::<Vec<_>>();
            let mask_planes = output
                .objects
                .iter()
                .map(|object| mask_to_bool_plane(&object.mask_logits, 0.0))
                .collect::<Result<Vec<_>>>()?;
            let last_occluded = object_ids
                .iter()
                .map(|obj_id| {
                    if newly_removed_obj_ids.contains(obj_id) {
                        Some(100_000usize)
                    } else {
                        self.object_states
                            .get(obj_id)
                            .and_then(|state| state.last_occluded_frame)
                    }
                })
                .collect::<Vec<_>>();
            let reverse = matches!(direction, PropagationDirection::Backward);
            for i in 0..object_ids.len() {
                for j in (i + 1)..object_ids.len() {
                    let iou = binary_planes_iou(&mask_planes[i], &mask_planes[j]);
                    if iou < overlap_threshold {
                        continue;
                    }
                    let last_i = last_occluded[i];
                    let last_j = last_occluded[j];
                    let suppress_i = if reverse {
                        match (last_i, last_j) {
                            (Some(i_occ), Some(j_occ)) => i_occ < j_occ,
                            _ => false,
                        }
                    } else {
                        match (last_i, last_j) {
                            (Some(i_occ), Some(j_occ)) => i_occ > j_occ,
                            _ => false,
                        }
                    };
                    let suppress_j = if reverse {
                        match (last_i, last_j) {
                            (Some(i_occ), Some(j_occ)) => j_occ < i_occ,
                            _ => false,
                        }
                    } else {
                        match (last_i, last_j) {
                            (Some(i_occ), Some(j_occ)) => j_occ > i_occ,
                            _ => false,
                        }
                    };
                    if suppress_i {
                        suppressed_obj_ids.insert(object_ids[i]);
                    }
                    if suppress_j {
                        suppressed_obj_ids.insert(object_ids[j]);
                    }
                }
            }
            for object in &output.objects {
                self.ensure_object_id(object.obj_id, frame_idx);
                if let Some(state) = self.object_states.get_mut(&object.obj_id) {
                    let is_occluded = !mask_has_foreground(&object.mask_logits, 0.0)?;
                    if is_occluded || suppressed_obj_ids.contains(&object.obj_id) {
                        state.last_occluded_frame = Some(frame_idx);
                    }
                }
            }
        } else {
            for object in &output.objects {
                self.ensure_object_id(object.obj_id, frame_idx);
                if let Some(state) = self.object_states.get_mut(&object.obj_id) {
                    let is_occluded = !mask_has_foreground(&object.mask_logits, 0.0)?;
                    if is_occluded {
                        state.last_occluded_frame = Some(frame_idx);
                    }
                }
            }
        }
        Ok(suppressed_obj_ids)
    }
}
