use super::*;

fn normalize_point_coords(coords: &Tensor, device: &Device) -> Result<Tensor> {
    let coords = maybe_to_device_dtype(coords, device, DType::F32)?;
    match coords.rank() {
        2 => coords.unsqueeze(0),
        3 => Ok(coords),
        rank => candle::bail!("tracker point coords must have rank 2 or 3, got {rank}"),
    }
}

fn normalize_point_labels(labels: &Tensor, device: &Device) -> Result<Tensor> {
    let labels = maybe_to_device_dtype(labels, device, DType::F32)?;
    match labels.rank() {
        1 => labels.unsqueeze(0),
        2 => Ok(labels),
        rank => candle::bail!("tracker point labels must have rank 1 or 2, got {rank}"),
    }
}

fn normalize_boxes_as_points(boxes_xyxy: &Tensor, device: &Device) -> Result<Tensor> {
    let boxes_xyxy = maybe_to_device_dtype(boxes_xyxy, device, DType::F32)?;
    match boxes_xyxy.rank() {
        1 => boxes_xyxy.reshape((1, 2, 2)),
        2 => boxes_xyxy.reshape((boxes_xyxy.dim(0)?, 2, 2)),
        3 => Ok(boxes_xyxy),
        rank => candle::bail!("tracker boxes must have rank 1, 2, or 3, got {rank}"),
    }
}

pub(super) fn normalize_mask_prompt(mask: &Tensor, device: &Device) -> Result<Tensor> {
    let mask = maybe_to_device_dtype(mask, device, DType::F32)?;
    match mask.rank() {
        2 => mask.unsqueeze(0)?.unsqueeze(0),
        3 => mask.unsqueeze(1),
        4 => Ok(mask),
        rank => candle::bail!("tracker mask input must have rank 2, 3, or 4, got {rank}"),
    }
}

impl Sam3TrackerModel {
    pub(super) fn prepare_point_prompt(
        &self,
        point_coords: Option<&Tensor>,
        point_labels: Option<&Tensor>,
        boxes_xyxy: Option<&Tensor>,
        device: &Device,
    ) -> Result<Option<(Tensor, Tensor)>> {
        let point_coords = match point_coords {
            Some(coords) => normalize_point_coords(coords, device)?,
            None => Tensor::zeros((1, 0, 2), DType::F32, device)?,
        };
        let point_labels = match point_labels {
            Some(labels) => normalize_point_labels(labels, device)?,
            None => Tensor::zeros((1, 0), DType::F32, device)?,
        };
        let (point_coords, point_labels) = if let Some(boxes_xyxy) = boxes_xyxy {
            let box_coords = normalize_boxes_as_points(boxes_xyxy, device)?;
            let batch_size = box_coords.dim(0)?;
            let box_labels =
                Tensor::from_vec(vec![2f32, 3f32].repeat(batch_size), (batch_size, 2), device)?;
            (
                Tensor::cat(&[&box_coords, &point_coords], 1)?,
                Tensor::cat(&[&box_labels, &point_labels], 1)?,
            )
        } else {
            (point_coords, point_labels)
        };

        if point_coords.dim(1)? == 0 {
            Ok(None)
        } else {
            Ok(Some((point_coords, point_labels)))
        }
    }

    pub(super) fn use_multimask(&self, is_init_cond_frame: bool, point_count: usize) -> bool {
        self.config.multimask_output_in_sam
            && (is_init_cond_frame || self.config.multimask_output_for_tracking)
            && (self.config.multimask_min_pt_num..=self.config.multimask_max_pt_num)
                .contains(&point_count)
    }

    pub(super) fn prepare_high_res_features(
        &self,
        high_res_features: &[Tensor],
    ) -> Result<Vec<Tensor>> {
        if high_res_features.len() < 2 {
            candle::bail!(
                "tracker expected at least two high-resolution feature levels, got {}",
                high_res_features.len()
            );
        }
        let feat_s0 = &high_res_features[0];
        let feat_s1 = &high_res_features[1];
        let compute_dtype = self.no_obj_ptr.dtype();
        let projected_s0 = self.config.mask_decoder.transformer_dim / 8;
        let projected_s1 = self.config.mask_decoder.transformer_dim / 4;
        let (_, channels_s0, _, _) = feat_s0.dims4()?;
        let (_, channels_s1, _, _) = feat_s1.dims4()?;
        if channels_s0 == projected_s0 && channels_s1 == projected_s1 {
            return Ok(vec![
                maybe_to_dtype(feat_s0, compute_dtype)?,
                maybe_to_dtype(feat_s1, compute_dtype)?,
            ]);
        }
        if channels_s0 == self.config.hidden_dim && channels_s1 == self.config.hidden_dim {
            let conv_s0 = self.sam_mask_decoder.conv_s0.as_ref().ok_or_else(|| {
                candle::Error::Msg("tracker high-res projection conv_s0 missing".into())
            })?;
            let conv_s1 = self.sam_mask_decoder.conv_s1.as_ref().ok_or_else(|| {
                candle::Error::Msg("tracker high-res projection conv_s1 missing".into())
            })?;
            return Ok(vec![
                maybe_to_dtype(&feat_s0.apply(conv_s0)?, compute_dtype)?,
                maybe_to_dtype(&feat_s1.apply(conv_s1)?, compute_dtype)?,
            ]);
        }
        candle::bail!(
            "unexpected tracker high-res feature channel contract: s0={}, s1={}, expected projected [{projected_s0}, {projected_s1}] or hidden_dim {}",
            channels_s0,
            channels_s1,
            self.config.hidden_dim
        );
    }

    #[cfg(test)]
    pub(crate) fn prepare_high_res_features_for_test(
        &self,
        high_res_features: &[Tensor],
    ) -> Result<Vec<Tensor>> {
        self.prepare_high_res_features(high_res_features)
    }
}
