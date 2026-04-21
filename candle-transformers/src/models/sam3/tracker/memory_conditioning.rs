use super::*;
use crate::models::sam3::torch_ops::tensor::{first_scalar_f32, repeat_interleave};

#[derive(Debug)]
pub(super) struct PreparedMemoryConditioning {
    pub(super) pix_feat_with_mem: Tensor,
    pub(super) selected_conditioning_frame_indices: Vec<usize>,
    pub(super) selected_memory_frame_indices: Vec<usize>,
    pub(super) selected_object_pointer_frame_indices: Vec<usize>,
}

#[derive(Debug)]
struct PreparedMemoryPrompt {
    prompt: Option<Tensor>,
    prompt_pos: Option<Tensor>,
    num_obj_ptr_tokens: usize,
    selected_conditioning_frame_indices: Vec<usize>,
    selected_memory_frame_indices: Vec<usize>,
    selected_object_pointer_frame_indices: Vec<usize>,
}

fn state_maskmem_prompt_tensors(
    state: &TrackerFrameState,
    device: &Device,
    dtype: DType,
) -> Result<(Tensor, Tensor)> {
    match (&state.maskmem_prompt_features, &state.maskmem_prompt_pos_enc) {
        (Some(maskmem_features), Some(maskmem_pos_enc)) => Ok((
            maybe_to_device_dtype(maskmem_features, device, dtype)?,
            maybe_to_device_dtype(maskmem_pos_enc, device, dtype)?,
        )),
        _ => {
            let Some(maskmem_features) = &state.maskmem_features else {
                candle::bail!("tracker memory conditioning is missing maskmem_features")
            };
            let Some(maskmem_pos_enc) = &state.maskmem_pos_enc else {
                candle::bail!("tracker memory conditioning is missing maskmem_pos_enc")
            };
            let maskmem_features =
                maybe_to_device_dtype(maskmem_features, device, dtype)?;
            let maskmem_pos_enc = maybe_to_device_dtype(maskmem_pos_enc, device, dtype)?;
            prepare_maskmem_prompt_tensors(&maskmem_features, &maskmem_pos_enc)
        }
    }
}

impl Sam3TrackerModel {
    pub(super) fn cal_mem_score(
        &self,
        object_score_logits: &Tensor,
        iou_score: &Tensor,
    ) -> Result<f32> {
        if object_score_logits.dim(1)? > 1 || iou_score.dim(1)? > 1 {
            candle::bail!("tracker frame filter only supports one object");
        }
        let object_score = first_scalar_f32(&candle_nn::ops::sigmoid(object_score_logits)?)?;
        let iou_score = first_scalar_f32(iou_score)?;
        Ok(object_score * iou_score)
    }

    pub(super) fn frame_filter(
        &self,
        history: &BTreeMap<usize, TrackerFrameState>,
        track_in_reverse: bool,
        frame_idx: usize,
        num_frames: usize,
        r: usize,
    ) -> Result<Vec<usize>> {
        let max_num = self.config.num_maskmem;
        let mut must_include = if !track_in_reverse {
            frame_idx.saturating_sub(1)
        } else {
            (frame_idx + 1).min(num_frames.saturating_sub(1))
        };
        if must_include >= num_frames && num_frames > 0 {
            must_include = num_frames - 1;
        }
        let (start, end, step) = if !track_in_reverse {
            (
                must_include as isize,
                (frame_idx as isize - (max_num as isize * r as isize)).max(-1),
                -(r as isize),
            )
        } else {
            (
                must_include as isize,
                (frame_idx + max_num * r).min(num_frames) as isize,
                r as isize,
            )
        };

        let mut valid_indices = Vec::new();
        let mut i = start;
        while if !track_in_reverse { i >= end } else { i < end } {
            let frame = i as usize;
            let Some(state) = history.get(&frame) else {
                i += step;
                continue;
            };
            if state.is_cond_frame {
                i += step;
                continue;
            }
            let iou_score = state.iou_scores.max(D::Minus1)?;
            let score_per_frame = self.cal_mem_score(&state.object_score_logits, &iou_score)?;
            if score_per_frame > self.config.mf_threshold {
                valid_indices.insert(0, frame);
            }
            if valid_indices.len() >= max_num.saturating_sub(1) {
                break;
            }
            i += step;
        }
        if !valid_indices.contains(&must_include) {
            valid_indices.push(must_include);
        }
        Ok(valid_indices)
    }

    fn select_closest_cond_frame_indices(
        &self,
        frame_idx: usize,
        cond_frame_outputs: &BTreeMap<usize, &TrackerFrameState>,
    ) -> (Vec<usize>, Vec<usize>) {
        if self.config.max_cond_frames_in_attn == usize::MAX {
            return (cond_frame_outputs.keys().copied().collect(), Vec::new());
        }

        let mut selected = Vec::new();
        let push_unique = |items: &mut Vec<usize>, value: Option<usize>| {
            if let Some(value) = value {
                if !items.contains(&value) {
                    items.push(value);
                }
            }
        };
        if self.config.keep_first_cond_frame {
            let idx_first = cond_frame_outputs
                .keys()
                .copied()
                .filter(|t| *t < frame_idx)
                .min()
                .or_else(|| {
                    cond_frame_outputs
                        .keys()
                        .copied()
                        .filter(|t| *t > frame_idx)
                        .max()
                });
            push_unique(&mut selected, idx_first);
        }
        let idx_before = cond_frame_outputs
            .keys()
            .copied()
            .filter(|t| *t < frame_idx)
            .max();
        push_unique(&mut selected, idx_before);
        let idx_after = cond_frame_outputs
            .keys()
            .copied()
            .filter(|t| *t >= frame_idx)
            .min();
        push_unique(&mut selected, idx_after);

        let num_remain = self
            .config
            .max_cond_frames_in_attn
            .saturating_sub(selected.len());
        let mut remaining = cond_frame_outputs
            .keys()
            .copied()
            .filter(|t| !selected.contains(t))
            .collect::<Vec<_>>();
        remaining.sort_by_key(|t| {
            let abs = if *t >= frame_idx {
                *t - frame_idx
            } else {
                frame_idx - *t
            };
            (abs, *t)
        });
        selected.extend(remaining.into_iter().take(num_remain));

        let selected_set = selected.clone();
        let unselected = cond_frame_outputs
            .keys()
            .copied()
            .filter(|t| !selected_set.contains(t))
            .collect::<Vec<_>>();
        (selected, unselected)
    }

    pub(super) fn prepare_memory_conditioned_features(
        &self,
        frame_idx: usize,
        is_init_cond_frame: bool,
        current_vision_feats: &[Tensor],
        current_vision_pos_embeds: &[Tensor],
        feat_sizes: &[(usize, usize)],
        history: &BTreeMap<usize, TrackerFrameState>,
        num_frames: usize,
        track_in_reverse: bool,
        use_prev_mem_frame: bool,
    ) -> Result<PreparedMemoryConditioning> {
        let batch_size = current_vision_feats
            .last()
            .ok_or_else(|| {
                candle::Error::Msg("tracker requires at least one current vision feature".into())
            })?
            .dim(1)?;
        let channels = self.config.hidden_dim;
        let (height, width) = *feat_sizes.last().ok_or_else(|| {
            candle::Error::Msg("tracker requires at least one feature size".into())
        })?;
        if self.config.num_maskmem == 0 || is_init_cond_frame || !use_prev_mem_frame {
            let pix_feat_with_mem = current_vision_feats
                .last()
                .expect("checked above")
                .broadcast_add(&self.no_mem_embed)?
                .permute((1, 2, 0))?
                .reshape((batch_size, channels, height, width))?;
            return Ok(PreparedMemoryConditioning {
                pix_feat_with_mem,
                selected_conditioning_frame_indices: Vec::new(),
                selected_memory_frame_indices: Vec::new(),
                selected_object_pointer_frame_indices: Vec::new(),
            });
        }

        let cond_frame_outputs = history
            .iter()
            .filter_map(|(frame, state)| state.is_cond_frame.then_some((*frame, state)))
            .collect::<BTreeMap<_, _>>();
        if cond_frame_outputs.is_empty() {
            candle::bail!("tracker memory conditioning expected at least one conditioning frame");
        }
        let prepared_prompt = self.build_memory_conditioning_prompt(
            frame_idx,
            history,
            num_frames,
            track_in_reverse,
            &cond_frame_outputs,
        )?;
        let selected_conditioning_frame_indices =
            prepared_prompt.selected_conditioning_frame_indices.clone();
        let selected_object_pointer_frame_indices = prepared_prompt
            .selected_object_pointer_frame_indices
            .clone();
        let selected_memory_frame_indices = prepared_prompt.selected_memory_frame_indices.clone();

        let Some(prompt) = prepared_prompt.prompt else {
            let pix_feat_with_mem = current_vision_feats
                .last()
                .expect("checked above")
                .broadcast_add(&self.no_mem_embed)?
                .permute((1, 2, 0))?
                .reshape((batch_size, channels, height, width))?;
            return Ok(PreparedMemoryConditioning {
                pix_feat_with_mem,
                selected_conditioning_frame_indices,
                selected_memory_frame_indices,
                selected_object_pointer_frame_indices,
            });
        };
        let prompt_pos = prepared_prompt
            .prompt_pos
            .expect("prompt position encoding must exist whenever prompt exists");
        let num_obj_ptr_tokens = prepared_prompt.num_obj_ptr_tokens;
        let src = current_vision_feats
            .last()
            .expect("checked above")
            .transpose(0, 1)?
            .contiguous()?;
        let src_pos = current_vision_pos_embeds
            .last()
            .expect("checked above")
            .transpose(0, 1)?
            .contiguous()?;
        let prompt = prompt.transpose(0, 1)?.contiguous()?;
        let prompt_pos = prompt_pos.transpose(0, 1)?.contiguous()?;
        let encoded = self.memory_transformer.forward(
            &src,
            &prompt,
            Some(&src_pos),
            Some(&prompt_pos),
            num_obj_ptr_tokens,
        )?;
        let pix_feat_with_mem = encoded
            .transpose(1, 2)?
            .reshape((batch_size, channels, height, width))?;
        Ok(PreparedMemoryConditioning {
            pix_feat_with_mem,
            selected_conditioning_frame_indices,
            selected_memory_frame_indices,
            selected_object_pointer_frame_indices,
        })
    }

    fn build_memory_conditioning_prompt(
        &self,
        frame_idx: usize,
        history: &BTreeMap<usize, TrackerFrameState>,
        num_frames: usize,
        track_in_reverse: bool,
        cond_frame_outputs: &BTreeMap<usize, &TrackerFrameState>,
    ) -> Result<PreparedMemoryPrompt> {
        let device = cond_frame_outputs
            .values()
            .next()
            .map(|state| state.obj_ptr.device())
            .ok_or_else(|| {
                candle::Error::Msg(
                    "tracker memory conditioning expected at least one conditioning frame".into(),
                )
            })?;
        let batch_size = cond_frame_outputs
            .values()
            .next()
            .map(|state| state.obj_ptr.dim(0))
            .transpose()?
            .ok_or_else(|| {
                candle::Error::Msg(
                    "tracker memory conditioning expected at least one conditioning frame".into(),
                )
            })?;
        let channels = self.config.hidden_dim;
        let (selected_cond_ordered, unselected_cond_indices) =
            self.select_closest_cond_frame_indices(frame_idx, cond_frame_outputs);
        let mut selected_conditioning_frame_indices = selected_cond_ordered.clone();
        selected_conditioning_frame_indices.sort_unstable();
        let unselected_cond_outputs = unselected_cond_indices
            .iter()
            .filter_map(|frame| cond_frame_outputs.get(frame).map(|state| (*frame, *state)))
            .collect::<BTreeMap<_, _>>();

        let tpos_sign_mul: i64 = if track_in_reverse { -1 } else { 1 };
        let mut prompt_parts = Vec::new();
        let mut prompt_pos_parts = Vec::new();
        let mut selected_memory_frame_indices_ordered = Vec::new();
        let mut selected_object_pointer_frame_indices = Vec::new();

        for &selected_frame in selected_cond_ordered.iter() {
            let prev = cond_frame_outputs
                .get(&selected_frame)
                .expect("selected conditioning frame missing from history");
            if prev.maskmem_features.is_none() || prev.maskmem_pos_enc.is_none() {
                candle::bail!(
                    "conditioning frame {selected_frame} is missing maskmem tensors required for tracker memory conditioning"
                );
            }
            let (maskmem_features, maskmem_pos_enc) =
                state_maskmem_prompt_tensors(prev, device, self.no_obj_ptr.dtype())?;
            prompt_parts.push(maskmem_features);
            let pos = maskmem_pos_enc;
            let pos = pos.broadcast_add(&self.maskmem_tpos_enc.i(self.config.num_maskmem - 1)?)?;
            prompt_pos_parts.push(pos);
        }

        let r = self.config.memory_temporal_stride_for_eval.max(1);
        let valid_indices = if self.config.use_memory_selection {
            Some(self.frame_filter(history, track_in_reverse, frame_idx, num_frames, r)?)
        } else {
            None
        };
        for t_pos in 1..self.config.num_maskmem {
            let t_rel = self.config.num_maskmem - t_pos;
            let prev_frame_idx = if let Some(valid_indices) = valid_indices.as_ref() {
                if t_rel > valid_indices.len() {
                    continue;
                }
                valid_indices[valid_indices.len() - t_rel]
            } else if t_rel == 1 {
                if !track_in_reverse {
                    frame_idx.saturating_sub(t_rel)
                } else {
                    frame_idx + t_rel
                }
            } else if !track_in_reverse {
                let nearest = ((frame_idx.saturating_sub(2)) / r) * r;
                nearest.saturating_sub((t_rel - 2) * r)
            } else {
                let nearest = (frame_idx + 1).div_ceil(r) * r;
                nearest + (t_rel - 2) * r
            };
            let prev = history
                .get(&prev_frame_idx)
                .filter(|state| !state.is_cond_frame)
                .or_else(|| unselected_cond_outputs.get(&prev_frame_idx).copied());
            let Some(prev) = prev else {
                continue;
            };
            if prev.maskmem_features.is_none() || prev.maskmem_pos_enc.is_none() {
                continue;
            }
            let (maskmem_features, maskmem_pos_enc) =
                state_maskmem_prompt_tensors(prev, device, self.no_obj_ptr.dtype())?;
            prompt_parts.push(maskmem_features);
            let pos = maskmem_pos_enc;
            let pos = pos.broadcast_add(
                &self
                    .maskmem_tpos_enc
                    .i(self.config.num_maskmem - t_pos - 1)?,
            )?;
            prompt_pos_parts.push(pos);
            selected_memory_frame_indices_ordered.push(prev_frame_idx);
        }

        let max_obj_ptrs_in_encoder = self.config.max_obj_ptrs_in_encoder.min(num_frames);
        let ptr_cond_frames = if !track_in_reverse {
            selected_cond_ordered
                .iter()
                .copied()
                .filter(|t| *t <= frame_idx)
                .collect::<Vec<_>>()
        } else {
            selected_cond_ordered
                .iter()
                .copied()
                .filter(|t| *t >= frame_idx)
                .collect::<Vec<_>>()
        };
        let mut obj_ptr_tensors = Vec::new();
        let mut obj_ptr_offsets = Vec::new();
        for selected_frame in ptr_cond_frames.iter().copied() {
            selected_object_pointer_frame_indices.push(selected_frame);
            obj_ptr_offsets
                .push(((frame_idx as i64 - selected_frame as i64) * tpos_sign_mul) as i64);
            obj_ptr_tensors.push(
                cond_frame_outputs
                    .get(&selected_frame)
                    .expect("conditioning frame missing obj_ptr source")
                    .obj_ptr
                    .clone(),
            );
        }
        for t_diff in 1..max_obj_ptrs_in_encoder {
            let frame = if let Some(valid_indices) = valid_indices.as_ref() {
                if t_diff > valid_indices.len().saturating_sub(1) {
                    break;
                }
                valid_indices[valid_indices.len() - t_diff]
            } else if !track_in_reverse {
                let frame = frame_idx.saturating_sub(t_diff);
                if frame_idx < t_diff {
                    break;
                }
                frame
            } else {
                let frame = frame_idx + t_diff;
                if frame >= num_frames {
                    break;
                }
                frame
            };
            let prev = history
                .get(&frame)
                .filter(|state| !state.is_cond_frame)
                .or_else(|| unselected_cond_outputs.get(&frame).copied());
            if let Some(prev) = prev {
                selected_object_pointer_frame_indices.push(frame);
                obj_ptr_offsets.push(t_diff as i64);
                obj_ptr_tensors.push(prev.obj_ptr.clone());
            }
        }

        let mut num_obj_ptr_tokens = 0usize;
        if !obj_ptr_tensors.is_empty() {
            let mut obj_ptrs = Tensor::stack(obj_ptr_tensors.as_slice(), 0)?;
            let mut obj_pos = self.get_tpos_enc(
                obj_ptr_offsets.as_slice(),
                device,
                Some(max_obj_ptrs_in_encoder),
                false,
            )?;
            obj_pos = obj_pos.unsqueeze(1)?.expand((
                obj_ptr_offsets.len(),
                batch_size,
                self.config.memory_dim,
            ))?;
            if self.config.memory_dim < channels {
                let split = channels / self.config.memory_dim;
                obj_ptrs = obj_ptrs
                    .reshape((obj_ptrs.dim(0)?, batch_size, split, self.config.memory_dim))?
                    .permute((0, 2, 1, 3))?
                    .flatten(0, 1)?;
                obj_pos = repeat_interleave(&obj_pos, split, 0)?;
            }
            num_obj_ptr_tokens = obj_ptrs.dim(0)?;
            prompt_parts.push(obj_ptrs);
            prompt_pos_parts.push(obj_pos);
        }

        let mut selected_memory_frame_indices = selected_memory_frame_indices_ordered;
        selected_memory_frame_indices.sort_unstable();
        if prompt_parts.is_empty() {
            return Ok(PreparedMemoryPrompt {
                prompt: None,
                prompt_pos: None,
                num_obj_ptr_tokens,
                selected_conditioning_frame_indices,
                selected_memory_frame_indices,
                selected_object_pointer_frame_indices,
            });
        }

        let prompt_refs = prompt_parts.iter().collect::<Vec<_>>();
        let prompt_pos_refs = prompt_pos_parts.iter().collect::<Vec<_>>();
        Ok(PreparedMemoryPrompt {
            prompt: Some(Tensor::cat(prompt_refs.as_slice(), 0)?),
            prompt_pos: Some(Tensor::cat(prompt_pos_refs.as_slice(), 0)?),
            num_obj_ptr_tokens,
            selected_conditioning_frame_indices,
            selected_memory_frame_indices,
            selected_object_pointer_frame_indices,
        })
    }
}
