use super::*;
use crate::models::sam3::torch_ops::tensor::{first_scalar_f32, flatten_all_contiguous};

impl<'a> Sam3VideoTrackerCore<'a> {
    pub(super) fn postprocess_output(
        &self,
        config: &VideoConfig,
        session: &mut Sam3VideoSession,
        results: &[(u32, ObjectFrameOutput, TrackerFrameState, Option<f32>)],
        output_threshold: f32,
        removed_obj_ids: Option<&[u32]>,
        suppressed_obj_ids: Option<&[u32]>,
        unconfirmed_obj_ids: Option<&[u32]>,
    ) -> Result<Vec<ObjectFrameOutput>> {
        if results.is_empty() {
            return Ok(Vec::new());
        }

        let postprocess_device = session.storage_device().clone();
        let postprocess_on_cpu = matches!(postprocess_device, Device::Cpu);
        let predictor_config = &self.tracker.config().predictor;
        let confirmation_threshold = predictor_config.masklet_confirmation_consecutive_det_thresh;
        let mut hidden_obj_ids = BTreeSet::new();
        if let Some(obj_ids) = removed_obj_ids {
            hidden_obj_ids.extend(obj_ids.iter().copied());
        }
        if let Some(obj_ids) = suppressed_obj_ids {
            hidden_obj_ids.extend(obj_ids.iter().copied());
        }
        if let Some(obj_ids) = unconfirmed_obj_ids {
            hidden_obj_ids.extend(obj_ids.iter().copied());
        }
        let mut visible_outputs = Vec::new();
        let use_local_confirmation_gate =
            predictor_config.masklet_confirmation_enable && config.hotstart_delay == 0;

        for (obj_id, raw_output, _, _) in results.iter() {
            let output = if postprocess_on_cpu
                && !raw_output.masks.device().same_device(&postprocess_device)
            {
                raw_output.to_storage_device(&postprocess_device)?
            } else {
                raw_output.clone()
            };
            if hidden_obj_ids.contains(obj_id) {
                continue;
            }
            let has_detectable_output = object_has_detectable_output(&output, output_threshold)?;
            let is_confirmed = if use_local_confirmation_gate {
                let object = session.tracked_objects.get_mut(obj_id).ok_or_else(|| {
                    candle::Error::Msg(format!(
                        "unknown obj_id {} while updating postprocess confirmation state",
                        obj_id
                    ))
                })?;
                object.record_confirmation_activity(has_detectable_output, confirmation_threshold)
            } else {
                true
            };
            if !mask_has_foreground(&output.masks, output_threshold)? {
                continue;
            }
            if !is_confirmed {
                hidden_obj_ids.insert(*obj_id);
                continue;
            }
            visible_outputs.push(output.clone());
        }

        let visible_outputs = if config.non_overlap_masks_for_output {
            apply_object_wise_non_overlapping_constraints(&visible_outputs)?
        } else {
            visible_outputs
                .into_iter()
                .map(|output| {
                    let plane = mask_to_bool_plane(&output.masks, output_threshold)?;
                    let height = plane.len();
                    let width = plane.first().map(Vec::len).unwrap_or(0);
                    let data = plane
                        .iter()
                        .flat_map(|row| {
                            row.iter().map(|value| if *value { 1.0f32 } else { 0.0f32 })
                        })
                        .collect::<Vec<_>>();
                    let binary_mask =
                        Tensor::from_vec(data, (1, height, width), output.masks.device())?;
                    rebuild_object_output_from_binary_mask(&output, &binary_mask)
                })
                .collect::<Result<Vec<_>>>()?
        };

        let mut final_outputs = Vec::new();
        for mut output in visible_outputs {
            if mask_has_foreground(&output.masks, output_threshold)? {
                output.boxes_xyxy = mask_to_normalized_xyxy(&output.masks)?;
                final_outputs.push(output);
            } else {
                hidden_obj_ids.insert(output.obj_id);
            }
        }

        Ok(final_outputs)
    }
}

pub(super) fn mask_prompt_to_object_output(
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

pub(super) fn apply_prompt_frame_output_postprocess(
    output: &mut ObjectFrameOutput,
    config: &VideoConfig,
    postprocess_device: Option<&Device>,
) -> Result<()> {
    if config.fill_hole_area > 0 {
        if let Some(postprocess_device) = postprocess_device {
            if matches!(postprocess_device, Device::Cpu)
                && !output.mask_logits.device().same_device(postprocess_device)
            {
                *output = output.to_storage_device(postprocess_device)?;
            }
        }
        output.mask_logits =
            postprocess_low_res_mask_logits_for_video(&output.mask_logits, config.fill_hole_area)?;
        output.masks = candle_nn::ops::sigmoid(&output.mask_logits)?;
    }
    output.boxes_xyxy = mask_to_normalized_xyxy(&output.masks)?;
    Ok(())
}

pub(super) fn object_presence_score(output: &ObjectFrameOutput) -> Result<f32> {
    if let Some(presence_scores) = output.presence_scores.as_ref() {
        first_scalar_f32(presence_scores)
    } else {
        output.score_value()
    }
}

pub(super) fn mask_has_foreground(mask: &Tensor, threshold: f32) -> Result<bool> {
    Ok(mask
        .ge(threshold as f64)?
        .to_dtype(DType::F32)?
        .max_all()?
        .to_scalar::<f32>()?
        > 0.0)
}

fn object_has_detectable_output(output: &ObjectFrameOutput, threshold: f32) -> Result<bool> {
    Ok(mask_has_foreground(&output.masks, threshold)? && object_presence_score(output)? > 0.0)
}

fn binary_mask_logits(mask: &Tensor) -> Result<Tensor> {
    let mask = mask.to_dtype(DType::F32)?;
    let binary = mask.ge(0.5f32)?.to_dtype(DType::F32)?;
    binary.affine(2048.0, -1024.0)
}

fn canonicalize_single_score_tensor(tensor: &Tensor) -> Result<Tensor> {
    flatten_all_contiguous(tensor)?.reshape((tensor.elem_count(),))
}

fn rebuild_object_output_from_binary_mask(
    output: &ObjectFrameOutput,
    binary_mask: &Tensor,
) -> Result<ObjectFrameOutput> {
    let binary_mask = match binary_mask.rank() {
        4 => binary_mask.clone(),
        3 => binary_mask.unsqueeze(0)?,
        2 => binary_mask.unsqueeze(0)?.unsqueeze(0)?,
        rank => candle::bail!("expected binary mask rank 2, 3, or 4, got {}", rank),
    }
    .to_dtype(DType::F32)?;
    Ok(ObjectFrameOutput {
        obj_id: output.obj_id,
        mask_logits: binary_mask.clone(),
        masks: binary_mask.clone(),
        boxes_xyxy: mask_to_normalized_xyxy(&binary_mask)?,
        scores: output.scores.clone(),
        presence_scores: output.presence_scores.clone(),
        prompt_frame_idx: output.prompt_frame_idx,
        memory_frame_indices: output.memory_frame_indices.clone(),
        text_prompt: output.text_prompt.clone(),
        used_explicit_geometry: output.used_explicit_geometry,
        reused_previous_output: output.reused_previous_output,
    })
}

fn apply_object_wise_non_overlapping_constraints(
    outputs: &[ObjectFrameOutput],
) -> Result<Vec<ObjectFrameOutput>> {
    if outputs.len() <= 1 {
        return Ok(outputs.to_vec());
    }
    let device = outputs[0].mask_logits.device();
    let mut mask_logit_tensors = Vec::with_capacity(outputs.len());
    for output in outputs {
        let mask_logits = match output.mask_logits.rank() {
            4 => output.mask_logits.clone(),
            3 => output.mask_logits.unsqueeze(0)?,
            2 => output.mask_logits.unsqueeze(0)?.unsqueeze(0)?,
            rank => candle::bail!("expected mask logits rank 2, 3, or 4, got {}", rank),
        }
        .to_dtype(DType::F32)?;
        mask_logit_tensors.push(mask_logits);
    }
    let mask_logit_refs = mask_logit_tensors.iter().collect::<Vec<_>>();
    let mask_logits = Tensor::cat(&mask_logit_refs, 0)?.contiguous()?;
    let (batch_size, channels, height, width) = mask_logits.dims4()?;
    let winner_idx = mask_logits.argmax_keepdim(0)?;
    let object_idx =
        Tensor::arange(0u32, batch_size as u32, device)?.reshape((batch_size, 1, 1, 1))?;
    let keep = object_idx
        .broadcast_eq(&winner_idx.broadcast_as((batch_size, channels, height, width))?)?;
    let neg_ten = Tensor::full(-10f32, mask_logits.shape(), device)?;
    let suppressed_logits = mask_logits.le(-10f64)?.where_cond(&mask_logits, &neg_ten)?;
    let non_overlapping_logits = keep.where_cond(&mask_logits, &suppressed_logits)?;
    outputs
        .iter()
        .enumerate()
        .map(|(idx, output)| {
            rebuild_object_output_from_mask_logits(output, &non_overlapping_logits.i(idx)?)
        })
        .collect()
}

fn rebuild_object_output_from_mask_logits(
    output: &ObjectFrameOutput,
    mask_logits: &Tensor,
) -> Result<ObjectFrameOutput> {
    let mask_logits = match mask_logits.rank() {
        4 => mask_logits.clone(),
        3 => mask_logits.unsqueeze(0)?,
        2 => mask_logits.unsqueeze(0)?.unsqueeze(0)?,
        rank => candle::bail!("expected mask logits rank 2, 3, or 4, got {}", rank),
    }
    .to_dtype(DType::F32)?;
    let masks = candle_nn::ops::sigmoid(&mask_logits)?;
    Ok(ObjectFrameOutput {
        obj_id: output.obj_id,
        mask_logits,
        boxes_xyxy: mask_to_normalized_xyxy(&masks)?,
        masks,
        scores: output.scores.clone(),
        presence_scores: output.presence_scores.clone(),
        prompt_frame_idx: output.prompt_frame_idx,
        memory_frame_indices: output.memory_frame_indices.clone(),
        text_prompt: output.text_prompt.clone(),
        used_explicit_geometry: output.used_explicit_geometry,
        reused_previous_output: output.reused_previous_output,
    })
}

pub(super) fn mask_to_normalized_xyxy(mask: &Tensor) -> Result<Tensor> {
    let mask = match mask.rank() {
        4 => mask.i((0, 0))?,
        3 => mask.i(0)?,
        2 => mask.clone(),
        rank => candle::bail!("expected mask rank 2, 3, or 4, got {}", rank),
    };
    let device = mask.device();
    let (height, width) = mask.dims2()?;
    if height == 0 || width == 0 {
        return Tensor::zeros((1, 4), DType::F32, device);
    }

    let mask_present = mask.ge(0.5f64)?;
    let shape = (height, width);
    let x_coords = Tensor::arange(0u32, width as u32, device)?
        .to_dtype(DType::F32)?
        .reshape((1, width))?
        .broadcast_as(shape)?;
    let y_coords = Tensor::arange(0u32, height as u32, device)?
        .to_dtype(DType::F32)?
        .reshape((height, 1))?
        .broadcast_as(shape)?;
    let hidden_x = Tensor::full(width as f32, shape, device)?;
    let hidden_y = Tensor::full(height as f32, shape, device)?;
    let empty_coord = Tensor::full(-1f32, shape, device)?;

    let min_x = mask_present.where_cond(&x_coords, &hidden_x)?.min_all()?;
    let min_y = mask_present.where_cond(&y_coords, &hidden_y)?.min_all()?;
    let max_x = mask_present
        .where_cond(&x_coords, &empty_coord)?
        .max_all()?;
    let max_y = mask_present
        .where_cond(&y_coords, &empty_coord)?
        .max_all()?;
    let width_scale = width.max(1) as f64;
    let height_scale = height.max(1) as f64;
    let boxes = Tensor::stack(
        &[
            min_x.affine(1.0 / width_scale, 0.0)?,
            min_y.affine(1.0 / height_scale, 0.0)?,
            max_x.affine(1.0 / width_scale, 0.0)?,
            max_y.affine(1.0 / height_scale, 0.0)?,
        ],
        0,
    )?
    .reshape((1, 4))?;

    let has_foreground = mask_present.to_dtype(DType::F32)?.max_all()?.gt(0f64)?;
    has_foreground
        .broadcast_as((1, 4))?
        .where_cond(&boxes, &Tensor::zeros((1, 4), DType::F32, device)?)
}

pub(super) fn resize_mask_logits_to_video(
    mask_logits: &Tensor,
    video_size: ImageSize,
) -> Result<Tensor> {
    let mask_logits = match mask_logits.rank() {
        2 => mask_logits.unsqueeze(0)?.unsqueeze(0)?,
        3 => mask_logits.unsqueeze(0)?,
        4 => mask_logits.clone(),
        rank => candle::bail!("expected mask logits rank 2, 3, or 4, got {}", rank),
    };
    mask_logits.upsample_bilinear2d(video_size.height, video_size.width, false)
}

pub(super) fn resize_mask_probs(
    mask_probs: &Tensor,
    height: usize,
    width: usize,
) -> Result<Tensor> {
    let mask_probs = match mask_probs.rank() {
        2 => mask_probs.unsqueeze(0)?.unsqueeze(0)?,
        3 => mask_probs.unsqueeze(0)?,
        4 => mask_probs.clone(),
        rank => candle::bail!("expected mask probabilities rank 2, 3, or 4, got {}", rank),
    };
    mask_probs.upsample_bilinear2d(height, width, false)
}

fn canonicalize_score_tensor(scores: &Tensor) -> Result<Tensor> {
    flatten_all_contiguous(scores)?.reshape((scores.elem_count(),))
}

fn score_tensor_from_value(score: f32, device: &Device) -> Result<Tensor> {
    Tensor::from_vec(vec![score], (1,), device)
}

pub(super) fn trim_memory_frame_indices(
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

pub(super) fn postprocess_low_res_mask_logits_for_video(
    mask_logits: &Tensor,
    max_area: usize,
) -> Result<Tensor> {
    if max_area == 0 {
        return Ok(mask_logits.clone());
    }
    if matches!(mask_logits.device(), Device::Cpu) {
        return postprocess_low_res_mask_logits_on_cpu(mask_logits, max_area);
    }
    if matches!(mask_logits.device(), Device::Cuda(_))
        && max_area <= candle_nn::ops::CLEANUP_MASK_LOGITS_SMALL_COMPONENTS_2D_CUDA_MAX_AREA
    {
        return candle_nn::ops::cleanup_mask_logits_small_components_2d(
            mask_logits,
            max_area,
            VIDEO_PROPAGATION_HOLE_FILL_LOGIT,
            VIDEO_PROPAGATION_SPRINKLE_REMOVE_LOGIT,
        );
    }
    let device = mask_logits.device().clone();
    let mask_logits = mask_logits.to_device(&Device::Cpu)?;
    postprocess_low_res_mask_logits_on_cpu(&mask_logits, max_area)?.to_device(&device)
}

fn postprocess_low_res_mask_logits_on_cpu(mask_logits: &Tensor, max_area: usize) -> Result<Tensor> {
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

    Tensor::from_vec(processed, (batch, channel, height, width), &Device::Cpu)
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

pub(super) fn tracker_state_to_object_output(
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

pub(super) fn grounding_to_object_output(
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

pub(super) fn filter_video_frame_output(
    frame_output: &VideoFrameOutput,
    hidden_obj_ids: &BTreeSet<u32>,
) -> VideoFrameOutput {
    if hidden_obj_ids.is_empty() {
        return frame_output.clone();
    }
    VideoFrameOutput {
        frame_idx: frame_output.frame_idx,
        objects: frame_output
            .objects
            .iter()
            .filter(|object| !hidden_obj_ids.contains(&object.obj_id))
            .cloned()
            .collect(),
    }
}

pub(super) fn persist_visible_frame_output(
    session: &mut Sam3VideoSession,
    frame_output: &VideoFrameOutput,
) -> Result<()> {
    let mut stored_outputs = BTreeMap::new();
    for output in frame_output.objects.iter() {
        let stored_output = if session.low_memory_mode() {
            output.to_storage_device_compact(session.storage_device())?
        } else {
            output.to_storage_device(session.storage_device())?
        };
        stored_outputs.insert(output.obj_id, stored_output);
    }
    if stored_outputs.is_empty() {
        session.frame_outputs.remove(&frame_output.frame_idx);
    } else {
        session
            .frame_outputs
            .insert(frame_output.frame_idx, stored_outputs.clone());
    }
    for object in session.tracked_objects.values_mut() {
        if object.tracker_states.contains_key(&frame_output.frame_idx) {
            if stored_outputs.contains_key(&object.obj_id) {
                object.record_output_frame(frame_output.frame_idx);
            } else {
                object.remove_output_frame(frame_output.frame_idx);
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mask_to_normalized_xyxy_uses_last_foreground_index_for_max_corner() -> Result<()> {
        let device = Device::Cpu;
        let mask = Tensor::from_vec(
            vec![
                0.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, //
                0.0, 0.0, 1.0, 1.0, 1.0, 0.0, //
                0.0, 0.0, 1.0, 1.0, 1.0, 0.0, //
                0.0, 0.0, 1.0, 1.0, 1.0, 0.0, //
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
            ],
            (5, 6),
            &device,
        )?;

        let actual = mask_to_normalized_xyxy(&mask)?
            .flatten_all()?
            .to_vec1::<f32>()?;

        let expected = [2.0f32 / 6.0, 1.0 / 5.0, 4.0 / 6.0, 3.0 / 5.0];
        assert!(actual
            .iter()
            .zip(expected)
            .all(|(actual, expected)| (actual - expected).abs() <= 1e-6));
        Ok(())
    }

    #[test]
    fn mask_to_normalized_xyxy_keeps_single_pixel_boxes_zero_area() -> Result<()> {
        let device = Device::Cpu;
        let mask = Tensor::from_vec(
            vec![
                0.0f32, 0.0, 0.0, 0.0, //
                0.0, 0.0, 1.0, 0.0, //
                0.0, 0.0, 0.0, 0.0, //
            ],
            (3, 4),
            &device,
        )?;

        let actual = mask_to_normalized_xyxy(&mask)?
            .flatten_all()?
            .to_vec1::<f32>()?;

        let expected = [2.0f32 / 4.0, 1.0 / 3.0, 2.0 / 4.0, 1.0 / 3.0];
        assert!(actual
            .iter()
            .zip(expected)
            .all(|(actual, expected)| (actual - expected).abs() <= 1e-6));
        Ok(())
    }

    #[test]
    fn mask_to_normalized_xyxy_returns_zero_box_for_empty_masks() -> Result<()> {
        let device = Device::Cpu;
        let mask = Tensor::zeros((3, 4), DType::F32, &device)?;

        let actual = mask_to_normalized_xyxy(&mask)?
            .flatten_all()?
            .to_vec1::<f32>()?;

        assert_eq!(actual, vec![0.0f32; 4]);
        Ok(())
    }

    #[test]
    fn prompt_frame_postprocess_recomputes_boxes_without_hole_filling() -> Result<()> {
        let device = Device::Cpu;
        let masks = Tensor::from_vec(
            vec![
                0.0f32, 0.0, 0.0, 0.0, 0.0, //
                0.0, 1.0, 1.0, 1.0, 0.0, //
                0.0, 1.0, 1.0, 1.0, 0.0, //
                0.0, 0.0, 0.0, 0.0, 0.0, //
            ],
            (1, 1, 4, 5),
            &device,
        )?;
        let mut output = ObjectFrameOutput {
            obj_id: 7,
            mask_logits: masks.clone(),
            masks,
            boxes_xyxy: Tensor::zeros((1, 4), DType::F32, &device)?,
            scores: Tensor::ones((1,), DType::F32, &device)?,
            presence_scores: None,
            prompt_frame_idx: Some(0),
            memory_frame_indices: Vec::new(),
            text_prompt: Some("person".to_owned()),
            used_explicit_geometry: false,
            reused_previous_output: false,
        };
        let mut config = VideoConfig::default();
        config.fill_hole_area = 0;

        apply_prompt_frame_output_postprocess(&mut output, &config, Some(&device))?;

        let actual = output.boxes_xyxy.flatten_all()?.to_vec1::<f32>()?;
        let expected = [1.0f32 / 5.0, 1.0 / 4.0, 3.0 / 5.0, 2.0 / 4.0];
        assert!(actual
            .iter()
            .zip(expected)
            .all(|(actual, expected)| (actual - expected).abs() <= 1e-6));
        Ok(())
    }

    #[test]
    fn output_non_overlap_uses_per_pixel_mask_logits() -> Result<()> {
        let device = Device::Cpu;
        let make_output = |obj_id, logit| -> Result<ObjectFrameOutput> {
            let mask_logits = Tensor::from_vec(vec![logit; 6], (1, 1, 2, 3), &device)?;
            let masks = candle_nn::ops::sigmoid(&mask_logits)?;
            Ok(ObjectFrameOutput {
                obj_id,
                mask_logits,
                boxes_xyxy: mask_to_normalized_xyxy(&masks)?,
                masks,
                scores: Tensor::ones((1,), DType::F32, &device)?,
                presence_scores: None,
                prompt_frame_idx: Some(0),
                memory_frame_indices: Vec::new(),
                text_prompt: None,
                used_explicit_geometry: true,
                reused_previous_output: false,
            })
        };
        let outputs = vec![make_output(1, 2.0)?, make_output(2, 3.0)?];

        let actual = apply_object_wise_non_overlapping_constraints(&outputs)?;

        assert_eq!(actual.len(), 2);
        assert_eq!(actual[0].obj_id, 1);
        assert_eq!(actual[1].obj_id, 2);
        assert!(!mask_has_foreground(&actual[0].masks, 0.5)?);
        assert!(mask_has_foreground(&actual[1].masks, 0.5)?);
        assert!(actual[0].mask_logits.max_all()?.to_scalar::<f32>()? <= -10.0);
        let winning_box = actual[1].boxes_xyxy.flatten_all()?.to_vec1::<f32>()?;
        let expected_box = [0.0f32, 0.0, 2.0 / 3.0, 1.0 / 2.0];
        assert!(winning_box
            .iter()
            .zip(expected_box)
            .all(|(actual, expected)| (actual - expected).abs() <= 1e-6));
        Ok(())
    }

    #[test]
    fn filter_video_frame_output_excludes_hidden_object_ids() -> Result<()> {
        let mask_logits = Tensor::zeros((1, 1, 2, 2), DType::F32, &Device::Cpu)?;
        let masks = candle_nn::ops::sigmoid(&mask_logits)?;
        let scores = Tensor::ones((1,), DType::F32, &Device::Cpu)?;
        let make_output = |obj_id| -> Result<ObjectFrameOutput> {
            Ok(ObjectFrameOutput {
                obj_id,
                mask_logits: mask_logits.clone(),
                masks: masks.clone(),
                boxes_xyxy: mask_to_normalized_xyxy(&masks)?,
                scores: scores.clone(),
                presence_scores: None,
                prompt_frame_idx: Some(0),
                memory_frame_indices: Vec::new(),
                text_prompt: None,
                used_explicit_geometry: false,
                reused_previous_output: false,
            })
        };
        let frame_output = VideoFrameOutput {
            frame_idx: 7,
            objects: vec![make_output(1)?, make_output(2)?, make_output(3)?],
        };
        let filtered = filter_video_frame_output(&frame_output, &BTreeSet::from([1u32, 3u32]));

        assert_eq!(filtered.frame_idx, 7);
        assert_eq!(filtered.objects.len(), 1);
        assert_eq!(filtered.objects[0].obj_id, 2);
        Ok(())
    }
}
