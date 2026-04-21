use super::*;

fn select_best_grounding_query(
    scores: &Tensor,
    boxes_xyxy: &Tensor,
    mask_logits: &Tensor,
    presence_scores: Option<&Tensor>,
) -> Result<(Tensor, Tensor, Tensor, Option<Tensor>)> {
    let best_idx = scores.argmax(1)?;
    let batch_size = scores.dim(0)?;
    let num_box_coords = boxes_xyxy.dim(2)?;
    let mask_height = mask_logits.dim(2)?;
    let mask_width = mask_logits.dim(3)?;
    let score_index = best_idx
        .unsqueeze(2)?
        .broadcast_as((batch_size, 1, scores.dim(2)?))?
        .contiguous()?;
    let box_index = best_idx
        .unsqueeze(2)?
        .broadcast_as((batch_size, 1, num_box_coords))?
        .contiguous()?;
    let mask_index = best_idx
        .unsqueeze(2)?
        .unsqueeze(3)?
        .broadcast_as((batch_size, 1, mask_height, mask_width))?
        .contiguous()?;
    let best_score = scores.contiguous()?.gather(&score_index, 1)?.squeeze(1)?;
    let best_box = boxes_xyxy.contiguous()?.gather(&box_index, 1)?.squeeze(1)?;
    let best_mask_logits = mask_logits
        .contiguous()?
        .gather(&mask_index, 1)?
        .squeeze(1)?;
    let best_presence = presence_scores
        .map(|tensor| {
            tensor
                .contiguous()?
                .gather(&best_idx.contiguous()?, 1)?
                .squeeze(1)
        })
        .transpose()?;
    Ok((best_score, best_box, best_mask_logits, best_presence))
}

pub(super) fn load_tokenizer(path: &Path, context_length: usize) -> Result<Tokenizer> {
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

pub(super) fn tokenize_prompt(
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

pub(super) fn combine_encoded_prompts(
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

pub(super) fn ground_from_encoded_prompt(
    model: &Sam3ImageModel,
    visual_features: &VisualBackboneOutput,
    prompt: &EncodedPrompt,
) -> Result<GroundingOutput> {
    let fused = model.encode_fused_prompt(visual_features, prompt)?;
    let decoder = model.decode_grounding(&fused, prompt)?;
    let segmentation = model.segment_grounding(visual_features, &decoder, &fused, prompt)?;
    let scores = model.text_detection_scores(&decoder)?;
    let (best_score, best_box, mask_logits, best_presence) = select_best_grounding_query(
        &scores,
        &decoder.pred_boxes_xyxy,
        &segmentation.mask_logits,
        segmentation.presence_logits.as_ref(),
    )?;
    let mask = candle_nn::ops::sigmoid(&mask_logits)?;
    Ok(GroundingOutput {
        mask_logits,
        masks: mask,
        boxes_xyxy: best_box,
        scores: best_score,
        presence_scores: best_presence,
    })
}

pub(super) fn boxes_cxcywh_to_xyxy_tensor(
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

pub(super) fn truncate_prompt_for_encoder(
    prompt: &SessionPrompt,
    max_points: usize,
) -> SessionPrompt {
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

pub(super) fn session_prompt_to_geometry(
    prompt: &SessionPrompt,
    device: &Device,
) -> Result<GeometryPrompt> {
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
