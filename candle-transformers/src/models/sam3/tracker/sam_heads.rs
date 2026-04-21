use super::*;
use crate::models::sam3::{
    torch_ops::tensor::repeat_interleave, tracker::prompt_inputs::normalize_mask_prompt,
};

#[derive(Debug)]
pub(super) struct TrackerMlp {
    layers: Vec<Linear>,
    sigmoid_output: bool,
}

impl TrackerMlp {
    pub(super) fn new(
        input_dim: usize,
        hidden_dim: usize,
        output_dim: usize,
        num_layers: usize,
        sigmoid_output: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let mut layers = Vec::with_capacity(num_layers);
        let vb = vb.pp("layers");
        for i in 0..num_layers {
            let in_dim = if i == 0 { input_dim } else { hidden_dim };
            let out_dim = if i + 1 == num_layers {
                output_dim
            } else {
                hidden_dim
            };
            layers.push(linear(vb.pp(i), in_dim, out_dim, true)?);
        }
        Ok(Self {
            layers,
            sigmoid_output,
        })
    }
}

impl Module for TrackerMlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut xs = xs.clone();
        for (index, layer) in self.layers.iter().enumerate() {
            xs = layer.forward(&xs)?;
            if index + 1 < self.layers.len() {
                xs = xs.relu()?;
            }
        }
        if self.sigmoid_output {
            candle_nn::ops::sigmoid(&xs)
        } else {
            Ok(xs)
        }
    }
}

#[derive(Debug)]
enum PredObjScoreHead {
    Linear(Linear),
    Mlp(TrackerMlp),
}

impl PredObjScoreHead {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Linear(layer) => layer.forward(xs),
            Self::Mlp(layer) => layer.forward(xs),
        }
    }
}

#[derive(Debug)]
pub(super) struct Sam3TrackerMaskDecoder {
    transformer_dim: usize,
    transformer: TwoWayTransformer,
    iou_token: Embedding,
    mask_tokens: Embedding,
    obj_score_token: Option<Embedding>,
    output_tokens: Tensor,
    output_token_count: usize,
    output_upscaling_conv1: ConvTranspose2d,
    output_upscaling_ln: LayerNorm2d,
    output_upscaling_conv2: ConvTranspose2d,
    pub(super) conv_s0: Option<Conv2d>,
    pub(super) conv_s1: Option<Conv2d>,
    output_hypernetworks_mlps: Vec<TrackerMlp>,
    iou_prediction_head: TrackerMlp,
    pred_obj_score_head: Option<PredObjScoreHead>,
    num_mask_tokens: usize,
    use_high_res_features: bool,
    use_multimask_token_for_obj_ptr: bool,
    dynamic_multimask_via_stability: bool,
    dynamic_multimask_stability_delta: f32,
    dynamic_multimask_stability_thresh: f32,
}

impl Sam3TrackerMaskDecoder {
    pub(super) fn new(config: &Sam3TrackerMaskDecoderConfig, vb: VarBuilder) -> Result<Self> {
        let num_mask_tokens = config.num_multimask_outputs + 1;
        let transformer = TwoWayTransformer::new(
            config.transformer_depth,
            config.transformer_embedding_dim,
            config.transformer_num_heads,
            config.transformer_mlp_dim,
            vb.pp("transformer"),
        )?;
        let iou_token = candle_nn::embedding(1, config.transformer_dim, vb.pp("iou_token"))?;
        let mask_tokens = candle_nn::embedding(
            num_mask_tokens,
            config.transformer_dim,
            vb.pp("mask_tokens"),
        )?;
        let obj_score_token = if config.pred_obj_scores {
            Some(candle_nn::embedding(
                1,
                config.transformer_dim,
                vb.pp("obj_score_token"),
            )?)
        } else {
            None
        };
        let output_tokens = {
            let mut tokens = vec![iou_token.embeddings().i(0)?];
            if let Some(obj_score_token) = &obj_score_token {
                tokens.push(obj_score_token.embeddings().i(0)?);
            }
            for index in 0..num_mask_tokens {
                tokens.push(mask_tokens.embeddings().i(index)?);
            }
            Tensor::stack(tokens.as_slice(), 0)?
        };
        let output_token_count = output_tokens.dim(0)?;
        let deconv_cfg = ConvTranspose2dConfig {
            stride: 2,
            ..Default::default()
        };
        let output_upscaling_conv1 = candle_nn::conv_transpose2d(
            config.transformer_dim,
            config.transformer_dim / 4,
            2,
            deconv_cfg,
            vb.pp("output_upscaling.0"),
        )?;
        let output_upscaling_ln = LayerNorm2d::new(
            config.transformer_dim / 4,
            1e-6,
            vb.pp("output_upscaling.1"),
        )?;
        let output_upscaling_conv2 = candle_nn::conv_transpose2d(
            config.transformer_dim / 4,
            config.transformer_dim / 8,
            2,
            deconv_cfg,
            vb.pp("output_upscaling.3"),
        )?;
        let (conv_s0, conv_s1) = if config.use_high_res_features {
            (
                Some(candle_nn::conv2d(
                    config.transformer_dim,
                    config.transformer_dim / 8,
                    1,
                    Default::default(),
                    vb.pp("conv_s0"),
                )?),
                Some(candle_nn::conv2d(
                    config.transformer_dim,
                    config.transformer_dim / 4,
                    1,
                    Default::default(),
                    vb.pp("conv_s1"),
                )?),
            )
        } else {
            (None, None)
        };
        let mut output_hypernetworks_mlps = Vec::with_capacity(num_mask_tokens);
        let output_hypernetworks_vb = vb.pp("output_hypernetworks_mlps");
        for index in 0..num_mask_tokens {
            output_hypernetworks_mlps.push(TrackerMlp::new(
                config.transformer_dim,
                config.transformer_dim,
                config.transformer_dim / 8,
                3,
                false,
                output_hypernetworks_vb.pp(index),
            )?);
        }
        let iou_prediction_head = TrackerMlp::new(
            config.transformer_dim,
            config.iou_head_hidden_dim,
            num_mask_tokens,
            config.iou_head_depth,
            config.iou_prediction_use_sigmoid,
            vb.pp("iou_prediction_head"),
        )?;
        let pred_obj_score_head = if config.pred_obj_scores {
            if config.pred_obj_scores_mlp {
                Some(PredObjScoreHead::Mlp(TrackerMlp::new(
                    config.transformer_dim,
                    config.transformer_dim,
                    1,
                    3,
                    false,
                    vb.pp("pred_obj_score_head"),
                )?))
            } else {
                Some(PredObjScoreHead::Linear(linear(
                    vb.pp("pred_obj_score_head"),
                    config.transformer_dim,
                    1,
                    true,
                )?))
            }
        } else {
            None
        };
        Ok(Self {
            transformer_dim: config.transformer_dim,
            transformer,
            iou_token,
            mask_tokens,
            obj_score_token,
            output_tokens,
            output_token_count,
            output_upscaling_conv1,
            output_upscaling_ln,
            output_upscaling_conv2,
            conv_s0,
            conv_s1,
            output_hypernetworks_mlps,
            iou_prediction_head,
            pred_obj_score_head,
            num_mask_tokens,
            use_high_res_features: config.use_high_res_features,
            use_multimask_token_for_obj_ptr: config.use_multimask_token_for_obj_ptr,
            dynamic_multimask_via_stability: config.dynamic_multimask_via_stability,
            dynamic_multimask_stability_delta: config.dynamic_multimask_stability_delta,
            dynamic_multimask_stability_thresh: config.dynamic_multimask_stability_thresh,
        })
    }

    fn forward(
        &self,
        image_embeddings: &Tensor,
        image_pe: &Tensor,
        sparse_prompt_embeddings: &Tensor,
        dense_prompt_embeddings: &Tensor,
        multimask_output: bool,
        repeat_image: bool,
        high_res_features: Option<&[Tensor]>,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        let (all_masks, all_iou_pred, mask_tokens_out, object_score_logits) = self.predict_masks(
            image_embeddings,
            image_pe,
            sparse_prompt_embeddings,
            dense_prompt_embeddings,
            repeat_image,
            high_res_features,
        )?;
        let (mask_slice, iou_slice) = if multimask_output {
            (1..self.num_mask_tokens, 1..self.num_mask_tokens)
        } else {
            (0..1, 0..1)
        };
        let masks = all_masks.i((.., mask_slice, .., ..))?;
        let iou_pred = all_iou_pred.i((.., iou_slice))?;
        let sam_tokens = if multimask_output {
            if self.use_multimask_token_for_obj_ptr {
                mask_tokens_out.i((.., 1..self.num_mask_tokens, ..))?
            } else {
                mask_tokens_out.i((.., 0..1, ..))?
            }
        } else {
            mask_tokens_out.i((.., 0..1, ..))?
        };
        let sam_tokens = if sam_tokens.rank() == 2 {
            sam_tokens.unsqueeze(1)?
        } else {
            sam_tokens
        };
        let object_score_logits = object_score_logits.i((.., 0..1))?;
        Ok((masks, iou_pred, sam_tokens, object_score_logits))
    }

    fn predict_masks(
        &self,
        image_embeddings: &Tensor,
        image_pe: &Tensor,
        sparse_prompt_embeddings: &Tensor,
        dense_prompt_embeddings: &Tensor,
        repeat_image: bool,
        high_res_features: Option<&[Tensor]>,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        let batch_size = sparse_prompt_embeddings.dim(0)?;
        let output_tokens = self.output_tokens.unsqueeze(0)?.expand((
            batch_size,
            self.output_token_count,
            self.transformer_dim,
        ))?;
        let tokens = Tensor::cat(&[&output_tokens, sparse_prompt_embeddings], 1)?;

        let src = if repeat_image {
            repeat_interleave(image_embeddings, batch_size, 0)?
        } else if image_embeddings.dim(0)? == batch_size {
            image_embeddings.clone()
        } else {
            candle::bail!(
                "tracker mask decoder expected image embeddings batch {} to match prompt batch {batch_size}",
                image_embeddings.dim(0)?
            );
        };
        let src = src.broadcast_add(dense_prompt_embeddings)?;
        let pos_src = if repeat_image {
            repeat_interleave(image_pe, batch_size, 0)?
        } else {
            image_pe.broadcast_as(src.shape())?
        };

        let (hs, src) = self.transformer.forward(&src, &pos_src, &tokens)?;
        let iou_token_out = hs.i((.., 0, ..))?;
        let obj_score_token_out = if self.pred_obj_score_head.is_some() {
            Some(hs.i((.., 1, ..))?)
        } else {
            None
        };
        let mask_tokens_out = hs.i((.., hs.dim(1)? - self.num_mask_tokens.., ..))?;
        let (_, channels, height, width) = image_embeddings.dims4()?;
        let src = src
            .transpose(1, 2)?
            .reshape((batch_size, channels, height, width))?;
        let upscaled_embedding = match (self.use_high_res_features, high_res_features) {
            (true, Some(high_res_features)) => {
                if high_res_features.len() < 2 {
                    candle::bail!(
                        "tracker mask decoder expected two high-resolution feature levels, got {}",
                        high_res_features.len()
                    );
                }
                let feat_s0 = &high_res_features[0];
                let feat_s1 = &high_res_features[1];
                let x = src.apply(&self.output_upscaling_conv1)?;
                let x = x.broadcast_add(feat_s1)?;
                let x = self.output_upscaling_ln.forward(&x)?.gelu_erf()?;
                let x = x.apply(&self.output_upscaling_conv2)?;
                x.broadcast_add(feat_s0)?.gelu_erf()?
            }
            _ => {
                let x = src.apply(&self.output_upscaling_conv1)?;
                let x = self.output_upscaling_ln.forward(&x)?.gelu_erf()?;
                x.apply(&self.output_upscaling_conv2)?.gelu_erf()?
            }
        };
        let (_, upscaled_dim, upscaled_height, upscaled_width) = upscaled_embedding.dims4()?;
        let mut hyper_in_list = Vec::with_capacity(self.num_mask_tokens);
        for index in 0..self.num_mask_tokens {
            hyper_in_list.push(
                self.output_hypernetworks_mlps[index].forward(&mask_tokens_out.i((
                    ..,
                    index,
                    ..,
                ))?)?,
            );
        }
        let hyper_in = Tensor::stack(hyper_in_list.as_slice(), 1)?;
        let masks = hyper_in
            .matmul(&upscaled_embedding.reshape((
                batch_size,
                upscaled_dim,
                upscaled_height * upscaled_width,
            ))?)?
            .reshape((
                batch_size,
                self.num_mask_tokens,
                upscaled_height,
                upscaled_width,
            ))?;
        let iou_pred = self.iou_prediction_head.forward(&iou_token_out)?;
        let object_score_logits = match (&self.pred_obj_score_head, obj_score_token_out.as_ref()) {
            (Some(head), Some(token)) => head.forward(token)?,
            _ => Tensor::ones((batch_size, 1), DType::F32, image_embeddings.device())?,
        };
        Ok((masks, iou_pred, mask_tokens_out, object_score_logits))
    }
}

impl Sam3TrackerModel {
    pub(super) fn forward_sam_heads(
        &self,
        backbone_features: &Tensor,
        point_prompt: Option<&(Tensor, Tensor)>,
        mask_inputs: Option<&Tensor>,
        high_res_features: Option<&[Tensor]>,
        multimask_output: bool,
        is_cond_frame: bool,
    ) -> Result<TrackerFrameState> {
        let batch_size = backbone_features.dim(0)?;
        let device = backbone_features.device();
        let (sam_point_coords, sam_point_labels) = match point_prompt {
            Some((coords, labels)) => (coords.clone(), labels.clone()),
            None => (
                Tensor::zeros((batch_size, 1, 2), DType::F32, device)?,
                Tensor::full(-1f32, (batch_size, 1), device)?,
            ),
        };
        let sam_mask_prompt = match mask_inputs {
            Some(mask_inputs) => {
                let mask_inputs = normalize_mask_prompt(mask_inputs, device)?;
                let (_, _, height, width) = mask_inputs.dims4()?;
                if [height, width] != self.config.prompt_encoder.mask_input_size {
                    Some(mask_inputs.upsample_bilinear2d(
                        self.config.prompt_encoder.mask_input_size[0],
                        self.config.prompt_encoder.mask_input_size[1],
                        false,
                    )?)
                } else {
                    Some(mask_inputs)
                }
            }
            None => None,
        };
        let (sparse_embeddings, dense_embeddings) = self.sam_prompt_encoder.forward(
            Some((&sam_point_coords, &sam_point_labels)),
            None,
            sam_mask_prompt.as_ref(),
        )?;
        let backbone_dtype = backbone_features.dtype();
        let sparse_embeddings = sparse_embeddings.to_dtype(backbone_dtype)?;
        let dense_embeddings = if sam_mask_prompt.is_none() {
            self.sam_prompt_encoder
                .no_mask_dense_embedding_with_dtype(backbone_dtype)?
        } else {
            dense_embeddings.to_dtype(backbone_dtype)?
        };
        let image_pe = self
            .sam_prompt_encoder
            .get_dense_pe_with_dtype(backbone_dtype)?;
        let (low_res_multimasks, ious, sam_output_tokens, object_score_logits) =
            self.sam_mask_decoder.forward(
                backbone_features,
                &image_pe,
                &sparse_embeddings,
                &dense_embeddings,
                multimask_output,
                false,
                high_res_features,
            )?;
        let object_present = object_score_logits.gt(0f64)?;
        let (low_res_masks, high_res_masks, sam_output_token) = if multimask_output {
            let best_iou_indices = ious.argmax(1)?.contiguous()?;
            let (_, _, low_res_height, low_res_width) = low_res_multimasks.dims4()?;
            let low_res_index = best_iou_indices
                .unsqueeze(1)?
                .unsqueeze(2)?
                .unsqueeze(3)?
                .broadcast_as((batch_size, 1, low_res_height, low_res_width))?
                .contiguous()?;
            let low_res_masks = low_res_multimasks.contiguous()?.gather(&low_res_index, 1)?;
            let sam_output_token = if sam_output_tokens.dim(1)? > 1 {
                let token_width = sam_output_tokens.dim(2)?;
                let token_index = best_iou_indices
                    .unsqueeze(1)?
                    .unsqueeze(2)?
                    .broadcast_as((batch_size, 1, token_width))?
                    .contiguous()?;
                sam_output_tokens
                    .contiguous()?
                    .gather(&token_index, 1)?
                    .squeeze(1)?
            } else {
                sam_output_tokens.i((.., 0, ..))?
            };
            let low_res_masks = gate_selected_masks(&low_res_masks, &object_present, device)?;
            let high_res_masks = low_res_masks.upsample_bilinear2d(
                self.config.image_size,
                self.config.image_size,
                false,
            )?;
            (low_res_masks, high_res_masks, sam_output_token)
        } else {
            let low_res_masks = gate_selected_masks(&low_res_multimasks, &object_present, device)?;
            let high_res_masks = low_res_masks.upsample_bilinear2d(
                self.config.image_size,
                self.config.image_size,
                false,
            )?;
            (low_res_masks, high_res_masks, sam_output_tokens.i((.., 0, ..))?)
        };
        let obj_ptr = self.obj_ptr_proj.forward(&sam_output_token)?;
        let object_present_for_ptr = object_present
            .broadcast_as(obj_ptr.shape())?
            .where_cond(&obj_ptr, &self.no_obj_ptr.broadcast_as(obj_ptr.shape())?)?;
        Ok(TrackerFrameState {
            low_res_masks,
            high_res_masks,
            iou_scores: ious,
            obj_ptr: object_present_for_ptr,
            object_score_logits,
            maskmem_features: None,
            maskmem_pos_enc: None,
            maskmem_prompt_features: None,
            maskmem_prompt_pos_enc: None,
            is_cond_frame,
        })
    }

    pub(super) fn use_mask_as_output(
        &self,
        backbone_features: &Tensor,
        high_res_features: Option<&[Tensor]>,
        mask_inputs: &Tensor,
        is_cond_frame: bool,
    ) -> Result<TrackerFrameState> {
        let device = backbone_features.device();
        let mask_inputs = normalize_mask_prompt(mask_inputs, device)?;
        let mask_inputs_float = mask_inputs.to_dtype(DType::F32)?;
        let high_res_masks = mask_inputs_float.affine(20.0, -10.0)?;
        let mask_input_low_res_size = (self.input_mask_size() / self.config.backbone_stride) * 4;
        let low_res_masks = resize_bilinear2d_antialias(
            &high_res_masks,
            mask_input_low_res_size,
            mask_input_low_res_size,
        )?;
        let iou_scores = Tensor::ones((mask_inputs_float.dim(0)?, 1), DType::F32, device)?;
        let mask_prompt = self.mask_downsample.forward(&mask_inputs_float)?;
        let prepared_high_res_features = match high_res_features {
            Some(high_res_features) => Some(self.prepare_high_res_features(high_res_features)?),
            None => None,
        };
        let state = self.forward_sam_heads(
            backbone_features,
            None,
            Some(&mask_prompt),
            prepared_high_res_features.as_deref(),
            false,
            is_cond_frame,
        )?;
        let object_present = mask_inputs_float
            .flatten(1, 3)?
            .gt(0f64)?
            .to_dtype(DType::F32)?
            .sum(1)?
            .gt(0f64)?
            .unsqueeze(1)?;
        let object_score_logits = object_present.to_dtype(DType::F32)?.affine(20.0, -10.0)?;
        let obj_ptr = object_present
            .broadcast_as(state.obj_ptr.shape())?
            .where_cond(
                &state.obj_ptr,
                &self.no_obj_ptr.broadcast_as(state.obj_ptr.shape())?,
            )?;
        Ok(TrackerFrameState {
            low_res_masks,
            high_res_masks,
            iou_scores,
            obj_ptr,
            object_score_logits,
            maskmem_features: None,
            maskmem_pos_enc: None,
            maskmem_prompt_features: None,
            maskmem_prompt_pos_enc: None,
            is_cond_frame,
        })
    }
}

fn gate_selected_masks(low_res_masks: &Tensor, object_present: &Tensor, device: &Device) -> Result<Tensor> {
    object_present
        .reshape((object_present.dim(0)?, 1, 1, 1))?
        .broadcast_as(low_res_masks.shape())?
        .where_cond(
            low_res_masks,
            &Tensor::full(NO_OBJ_SCORE as f32, low_res_masks.shape(), device)?,
        )
}
