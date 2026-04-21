use std::f64::consts::PI;

use candle::{DType, IndexOp, Result, Tensor};
use candle_nn::{LayerNorm, Linear, Module, VarBuilder};

use super::config::DecoderConfig;
use super::debug;
use super::encoder::FusionEncoderOutput;
use super::torch_ops::position::get_interleaved_1d_sine_pe;

#[derive(Debug)]
pub struct DecoderOutput {
    pub queries: Tensor,
    pub reference_boxes: Tensor,
    pub pred_logits: Tensor,
    pub pred_boxes: Tensor,
    pub pred_boxes_xyxy: Tensor,
    pub presence_logits: Option<Tensor>,
}

#[derive(Debug)]
struct DecoderAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl DecoderAttention {
    fn new(config: &DecoderConfig, vb: VarBuilder) -> Result<Self> {
        let in_proj_weight = vb.get((3 * config.d_model, config.d_model), "in_proj_weight")?;
        let in_proj_bias = vb.get(3 * config.d_model, "in_proj_bias")?;
        let split_weights = in_proj_weight.chunk(3, 0)?;
        let split_biases = in_proj_bias.chunk(3, 0)?;
        let q_proj = Linear::new(split_weights[0].clone(), Some(split_biases[0].clone()));
        let k_proj = Linear::new(split_weights[1].clone(), Some(split_biases[1].clone()));
        let v_proj = Linear::new(split_weights[2].clone(), Some(split_biases[2].clone()));
        let out_proj = candle_nn::linear(config.d_model, config.d_model, vb.pp("out_proj"))?;
        let head_dim = config.d_model / config.num_heads;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            num_heads: config.num_heads,
            head_dim,
            scale: (head_dim as f64).powf(-0.5),
        })
    }

    fn project(
        &self,
        xs: &Tensor,
        linear: &Linear,
        batch_size: usize,
        seq_len: usize,
    ) -> Result<Tensor> {
        linear
            .forward(&xs.transpose(0, 1)?.contiguous()?)?
            .reshape((batch_size, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .reshape((batch_size * self.num_heads, seq_len, self.head_dim))?
            .to_dtype(DType::F32)
    }

    fn forward(
        &self,
        query: &Tensor,
        key_value: &Tensor,
        key_padding_mask: Option<&Tensor>,
        query_pos: Option<&Tensor>,
        key_pos: Option<&Tensor>,
        attn_bias: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (tgt_len, batch_size, hidden_size) = query.dims3()?;
        let src_len = key_value.dim(0)?;
        let query = match query_pos {
            Some(query_pos) => query.broadcast_add(query_pos)?,
            None => query.clone(),
        };
        let key = match key_pos {
            Some(key_pos) => key_value.broadcast_add(key_pos)?,
            None => key_value.clone(),
        };
        let q = (self.project(&query, &self.q_proj, batch_size, tgt_len)? * self.scale)?;
        let k = self.project(&key, &self.k_proj, batch_size, src_len)?;
        let v = self.project(key_value, &self.v_proj, batch_size, src_len)?;
        let mut attn = q.matmul(&k.transpose(1, 2)?)?;
        if let Some(attn_bias) = attn_bias {
            let attn_bias =
                normalize_attn_bias(attn_bias, batch_size, self.num_heads, tgt_len, src_len)?;
            attn = attn.broadcast_add(&attn_bias)?;
        }
        if let Some(key_padding_mask) = key_padding_mask {
            let key_padding_mask = normalize_padding_mask(key_padding_mask, batch_size, src_len)?;
            let additive_mask = (key_padding_mask.to_dtype(DType::F32)? * -1e9f64)?
                .reshape((batch_size, 1, 1, src_len))?
                .repeat((1, self.num_heads, tgt_len, 1))?
                .reshape((batch_size * self.num_heads, tgt_len, src_len))?;
            attn = attn.broadcast_add(&additive_mask)?;
        }
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        let hidden_states = attn
            .matmul(&v)?
            .reshape((batch_size, self.num_heads, tgt_len, self.head_dim))?
            .transpose(1, 2)?
            .reshape((batch_size, tgt_len, hidden_size))?;
        self.out_proj
            .forward(&hidden_states)?
            .transpose(0, 1)?
            .contiguous()
    }
}

#[derive(Debug)]
struct MlpHead {
    layers: Vec<Linear>,
    residual: bool,
    out_norm: Option<LayerNorm>,
}

impl MlpHead {
    fn new(dims: &[usize], residual: bool, out_norm: bool, vb: VarBuilder) -> Result<Self> {
        Self::new_with_out_norm_eps(dims, residual, out_norm, 1e-6, vb)
    }

    fn new_with_out_norm_eps(
        dims: &[usize],
        residual: bool,
        out_norm: bool,
        out_norm_eps: f64,
        vb: VarBuilder,
    ) -> Result<Self> {
        if dims.len() < 2 {
            candle::bail!("sam3 mlp head requires at least two dimensions")
        }
        let mut layers = Vec::with_capacity(dims.len() - 1);
        for layer_idx in 0..(dims.len() - 1) {
            layers.push(candle_nn::linear(
                dims[layer_idx],
                dims[layer_idx + 1],
                vb.pp("layers").pp(layer_idx),
            )?);
        }
        let out_norm = if out_norm {
            Some(candle_nn::layer_norm(
                *dims.last().unwrap(),
                out_norm_eps,
                vb.pp("out_norm"),
            )?)
        } else {
            None
        };
        Ok(Self {
            layers,
            residual,
            out_norm,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let residual = if self.residual {
            Some(xs.clone())
        } else {
            None
        };
        let mut hidden_states = xs.clone();
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            hidden_states = layer.forward(&hidden_states)?;
            if layer_idx + 1 != self.layers.len() {
                hidden_states = hidden_states.relu()?;
            }
        }
        if let Some(residual) = residual {
            hidden_states = (hidden_states + residual)?;
        }
        if let Some(out_norm) = &self.out_norm {
            hidden_states = out_norm.forward(&hidden_states)?;
        }
        Ok(hidden_states)
    }
}

#[derive(Debug)]
struct DotProductScoringHead {
    prompt_mlp: Option<MlpHead>,
    prompt_proj: Linear,
    hs_proj: Linear,
    scale: f64,
}

impl DotProductScoringHead {
    fn new(config: &DecoderConfig, vb: VarBuilder) -> Result<Self> {
        let prompt_mlp = if vb.contains_tensor("prompt_mlp.layers.0.weight") {
            Some(MlpHead::new_with_out_norm_eps(
                &[config.d_model, config.dim_feedforward, config.d_model],
                true,
                true,
                1e-5,
                vb.pp("prompt_mlp"),
            )?)
        } else {
            None
        };
        let prompt_proj = candle_nn::linear(config.d_model, config.d_model, vb.pp("prompt_proj"))?;
        let hs_proj = candle_nn::linear(config.d_model, config.d_model, vb.pp("hs_proj"))?;
        Ok(Self {
            prompt_mlp,
            prompt_proj,
            hs_proj,
            scale: (config.d_model as f64).powf(-0.5),
        })
    }

    fn forward(&self, queries: &Tensor, prompt: &Tensor, prompt_mask: &Tensor) -> Result<Tensor> {
        let (_, batch_size, hidden_size) = prompt.dims3()?;
        let prompt = match &self.prompt_mlp {
            Some(prompt_mlp) => prompt_mlp.forward(prompt)?,
            None => prompt.clone(),
        };
        debug::capture_tensor("decoder.dotprod.prompt_after_mlp", &prompt)?;
        let pooled_prompt = mean_pool_prompt(&prompt, prompt_mask)?;
        debug::capture_tensor("decoder.dotprod.pooled_prompt", &pooled_prompt)?;
        let pooled_prompt = self.prompt_proj.forward(&pooled_prompt)?;
        debug::capture_tensor("decoder.dotprod.prompt_proj", &pooled_prompt)?;
        let queries = self.hs_proj.forward(queries)?;
        debug::capture_tensor("decoder.dotprod.query_proj", &queries)?;
        let scores = queries.matmul(&pooled_prompt.reshape((batch_size, hidden_size, 1))?)?;
        let scores = (scores * self.scale)?;
        debug::capture_tensor("decoder.dotprod.scores_pre_clamp", &scores)?;
        let scores = scores.clamp(-12.0, 12.0)?;
        debug::capture_tensor("decoder.dotprod.scores", &scores)?;
        Ok(scores)
    }
}

#[derive(Debug)]
struct DecoderLayer {
    cross_attn_image: DecoderAttention,
    norm1: LayerNorm,
    text_cross_attn: Option<DecoderAttention>,
    catext_norm: Option<LayerNorm>,
    self_attn: DecoderAttention,
    norm2: LayerNorm,
    linear1: Linear,
    linear2: Linear,
    norm3: LayerNorm,
}

impl DecoderLayer {
    fn new(config: &DecoderConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            cross_attn_image: DecoderAttention::new(config, vb.pp("cross_attn"))?,
            norm1: candle_nn::layer_norm(config.d_model, 1e-5, vb.pp("norm1"))?,
            text_cross_attn: if config.use_text_cross_attention {
                Some(DecoderAttention::new(config, vb.pp("ca_text"))?)
            } else {
                None
            },
            catext_norm: if config.use_text_cross_attention {
                Some(candle_nn::layer_norm(
                    config.d_model,
                    1e-6,
                    vb.pp("catext_norm"),
                )?)
            } else {
                None
            },
            self_attn: DecoderAttention::new(config, vb.pp("self_attn"))?,
            norm2: candle_nn::layer_norm(config.d_model, 1e-5, vb.pp("norm2"))?,
            linear1: candle_nn::linear(config.d_model, config.dim_feedforward, vb.pp("linear1"))?,
            linear2: candle_nn::linear(config.dim_feedforward, config.d_model, vb.pp("linear2"))?,
            norm3: candle_nn::layer_norm(config.d_model, 1e-5, vb.pp("norm3"))?,
        })
    }

    fn forward(
        &self,
        layer_idx: usize,
        tgt: &Tensor,
        tgt_query_pos: &Tensor,
        memory_text: &Tensor,
        text_attention_mask: &Tensor,
        memory: &Tensor,
        memory_key_padding_mask: &Tensor,
        memory_pos: &Tensor,
        cross_attn_mask: Option<&Tensor>,
        presence_token: Option<&Tensor>,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let (mut tgt, tgt_query_pos) = match presence_token {
            Some(presence_token) => (
                Tensor::cat(&[presence_token, tgt], 0)?,
                Tensor::cat(&[&Tensor::zeros_like(presence_token)?, tgt_query_pos], 0)?,
            ),
            None => (tgt.clone(), tgt_query_pos.clone()),
        };
        debug::capture_tensor(
            &format!("decoder.layer.{layer_idx}.input_with_presence"),
            &tgt,
        )?;
        debug::capture_tensor(
            &format!("decoder.layer.{layer_idx}.query_pos_with_presence"),
            &tgt_query_pos,
        )?;
        let self_attn_out = self.self_attn.forward(
            &tgt,
            &tgt,
            None,
            Some(&tgt_query_pos),
            Some(&tgt_query_pos),
            None,
        )?;
        debug::capture_tensor(
            &format!("decoder.layer.{layer_idx}.self_attn_output"),
            &self_attn_out,
        )?;
        tgt = self.norm2.forward(&(tgt + self_attn_out)?)?;
        debug::capture_tensor(&format!("decoder.layer.{layer_idx}.post_self_attn"), &tgt)?;

        if let (Some(text_cross_attn), Some(catext_norm)) =
            (&self.text_cross_attn, &self.catext_norm)
        {
            let text_attn_out = text_cross_attn.forward(
                &tgt,
                memory_text,
                Some(text_attention_mask),
                Some(&tgt_query_pos),
                None,
                None,
            )?;
            debug::capture_tensor(
                &format!("decoder.layer.{layer_idx}.text_attn_output"),
                &text_attn_out,
            )?;
            tgt = catext_norm.forward(&(tgt + text_attn_out)?)?;
            debug::capture_tensor(&format!("decoder.layer.{layer_idx}.post_text_cross"), &tgt)?;
        }

        let cross_attn_mask = match (presence_token, cross_attn_mask) {
            (Some(_), Some(cross_attn_mask)) => {
                let bias_shape = cross_attn_mask.dims3()?;
                let prefix = Tensor::zeros(
                    (bias_shape.0, 1, bias_shape.2),
                    cross_attn_mask.dtype(),
                    cross_attn_mask.device(),
                )?;
                Some(Tensor::cat(&[&prefix, cross_attn_mask], 1)?)
            }
            (_, Some(cross_attn_mask)) => Some(cross_attn_mask.clone()),
            _ => None,
        };
        if let Some(mask) = &cross_attn_mask {
            debug::capture_tensor(&format!("decoder.layer.{layer_idx}.cross_attn_mask"), mask)?;
        }

        let image_attn_out = self.cross_attn_image.forward(
            &tgt,
            memory,
            Some(memory_key_padding_mask),
            Some(&tgt_query_pos),
            Some(memory_pos),
            cross_attn_mask.as_ref(),
        )?;
        debug::capture_tensor(
            &format!("decoder.layer.{layer_idx}.image_attn_output"),
            &image_attn_out,
        )?;
        tgt = self.norm1.forward(&(tgt + image_attn_out)?)?;
        debug::capture_tensor(&format!("decoder.layer.{layer_idx}.post_image_cross"), &tgt)?;

        let ffn_hidden = self.linear1.forward(&tgt)?.relu()?;
        debug::capture_tensor(
            &format!("decoder.layer.{layer_idx}.ffn_hidden"),
            &ffn_hidden,
        )?;
        let ffn_out = self.linear2.forward(&ffn_hidden)?;
        debug::capture_tensor(&format!("decoder.layer.{layer_idx}.ffn_output"), &ffn_out)?;
        tgt = self.norm3.forward(&(tgt + ffn_out)?)?;
        debug::capture_tensor(&format!("decoder.layer.{layer_idx}.output"), &tgt)?;

        if presence_token.is_some() {
            let presence_out = tgt.i(0)?.unsqueeze(0)?;
            let queries = tgt.i(1..)?.contiguous()?;
            debug::capture_tensor(
                &format!("decoder.layer.{layer_idx}.queries_output"),
                &queries,
            )?;
            debug::capture_tensor(
                &format!("decoder.layer.{layer_idx}.presence_output"),
                &presence_out,
            )?;
            Ok((queries, Some(presence_out)))
        } else {
            debug::capture_tensor(&format!("decoder.layer.{layer_idx}.queries_output"), &tgt)?;
            Ok((tgt, None))
        }
    }
}

#[derive(Debug)]
pub struct Sam3TransformerDecoder {
    config: DecoderConfig,
    layers: Vec<DecoderLayer>,
    output_norm: LayerNorm,
    query_embed: Tensor,
    reference_points: Tensor,
    bbox_embed: MlpHead,
    ref_point_head: MlpHead,
    presence_token: Option<Tensor>,
    presence_head: Option<MlpHead>,
    presence_out_norm: Option<LayerNorm>,
    box_rpb_embed_x: Option<MlpHead>,
    box_rpb_embed_y: Option<MlpHead>,
    dot_prod_scoring: DotProductScoringHead,
}

impl Sam3TransformerDecoder {
    pub fn new(config: &DecoderConfig, vb: VarBuilder, score_vb: VarBuilder) -> Result<Self> {
        if config.num_heads == 0 || config.d_model % config.num_heads != 0 {
            candle::bail!(
                "sam3 decoder d_model ({}) must be divisible by num_heads ({})",
                config.d_model,
                config.num_heads
            )
        }
        let mut layers = Vec::with_capacity(config.num_layers);
        let layers_vb = vb.pp("layers");
        for layer_idx in 0..config.num_layers {
            layers.push(DecoderLayer::new(config, layers_vb.pp(layer_idx))?);
        }
        let output_norm = candle_nn::layer_norm(config.d_model, 1e-5, vb.pp("norm"))?;
        let query_embed = vb
            .pp("query_embed")
            .get((config.num_queries, config.d_model), "weight")?;
        let reference_points = vb
            .pp("reference_points")
            .get((config.num_queries, 4), "weight")?;
        let bbox_embed = MlpHead::new(
            &[config.d_model, config.d_model, config.d_model, 4],
            false,
            false,
            vb.pp("bbox_embed"),
        )?;
        let ref_point_head = MlpHead::new(
            &[2 * config.d_model, config.d_model, config.d_model],
            false,
            false,
            vb.pp("ref_point_head"),
        )?;
        let presence_token = if config.presence_token && vb.contains_tensor("presence_token.weight")
        {
            Some(vb.pp("presence_token").get((1, config.d_model), "weight")?)
        } else {
            None
        };
        let presence_head = if presence_token.is_some()
            && vb.contains_tensor("presence_token_head.layers.0.weight")
        {
            Some(MlpHead::new(
                &[config.d_model, config.d_model, config.d_model, 1],
                false,
                false,
                vb.pp("presence_token_head"),
            )?)
        } else {
            None
        };
        let presence_out_norm =
            if presence_token.is_some() && vb.contains_tensor("presence_token_out_norm.weight") {
                Some(candle_nn::layer_norm(
                    config.d_model,
                    1e-6,
                    vb.pp("presence_token_out_norm"),
                )?)
            } else {
                None
            };
        let (box_rpb_embed_x, box_rpb_embed_y) = match config.box_rpb_mode.as_str() {
            "none" => (None, None),
            "log" | "linear" | "both" => {
                let rpb_input_dim = if config.box_rpb_mode == "both" { 4 } else { 2 };
                (
                    Some(MlpHead::new(
                        &[rpb_input_dim, config.d_model, config.num_heads],
                        false,
                        false,
                        vb.pp("boxRPB_embed_x"),
                    )?),
                    Some(MlpHead::new(
                        &[rpb_input_dim, config.d_model, config.num_heads],
                        false,
                        false,
                        vb.pp("boxRPB_embed_y"),
                    )?),
                )
            }
            mode => candle::bail!("unsupported sam3 decoder box_rpb_mode `{mode}`"),
        };
        Ok(Self {
            config: config.clone(),
            layers,
            output_norm,
            query_embed,
            reference_points,
            bbox_embed,
            ref_point_head,
            presence_token,
            presence_head,
            presence_out_norm,
            box_rpb_embed_x,
            box_rpb_embed_y,
            dot_prod_scoring: DotProductScoringHead::new(config, score_vb)?,
        })
    }

    pub fn config(&self) -> &DecoderConfig {
        &self.config
    }

    pub fn forward(
        &self,
        encoder_out: &FusionEncoderOutput,
        prompt_features: &Tensor,
        prompt_mask: &Tensor,
    ) -> Result<DecoderOutput> {
        let batch_size = encoder_out.memory.dim(1)?;
        let mut queries = self
            .query_embed
            .reshape((self.config.num_queries, 1, self.config.d_model))?
            .repeat((1, batch_size, 1))?;
        debug::capture_tensor("decoder.initial_queries", &queries)?;
        let mut reference_boxes = self
            .reference_points
            .reshape((self.config.num_queries, 1, 4))?
            .repeat((1, batch_size, 1))?
            .apply(&candle_nn::ops::sigmoid)?;
        debug::capture_tensor("decoder.initial_reference_boxes", &reference_boxes)?;
        let valid_ratios_twice =
            Tensor::cat(&[&encoder_out.valid_ratios, &encoder_out.valid_ratios], 2)?;
        debug::capture_tensor("decoder.valid_ratios_twice", &valid_ratios_twice)?;
        let mut presence_state = match &self.presence_token {
            Some(presence_token) => Some(
                presence_token
                    .reshape((1, 1, self.config.d_model))?
                    .repeat((1, batch_size, 1))?,
            ),
            None => None,
        };
        if let Some(presence_state) = &presence_state {
            debug::capture_tensor("decoder.initial_presence_state", presence_state)?;
        }
        let mut final_queries = None;
        let mut final_reference_boxes = None;
        let mut final_pred_boxes = None;
        let mut final_presence_logits = None;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let layer_reference_boxes = reference_boxes.clone();
            debug::capture_tensor(
                &format!("decoder.layer.{layer_idx}.reference_boxes"),
                &layer_reference_boxes,
            )?;
            let reference_points_input = layer_reference_boxes
                .unsqueeze(2)?
                .broadcast_mul(&valid_ratios_twice.unsqueeze(0)?)?;
            debug::capture_tensor(
                &format!("decoder.layer.{layer_idx}.reference_points_input"),
                &reference_points_input,
            )?;
            let query_sine_embed = gen_sineembed_for_position(
                &reference_points_input.i((.., .., 0, ..))?,
                self.config.d_model,
            )?;
            debug::capture_tensor(
                &format!("decoder.layer.{layer_idx}.query_sine_embed"),
                &query_sine_embed,
            )?;
            let query_pos = self.ref_point_head.forward(&query_sine_embed)?;
            debug::capture_tensor(&format!("decoder.layer.{layer_idx}.query_pos"), &query_pos)?;
            let cross_attn_mask = self.build_box_relative_position_bias(
                &layer_reference_boxes,
                &encoder_out.spatial_shapes,
            )?;
            if let Some(cross_attn_mask) = &cross_attn_mask {
                debug::capture_tensor(
                    &format!("decoder.layer.{layer_idx}.cross_attn_mask_pre_presence"),
                    cross_attn_mask,
                )?;
            }
            let (next_queries, next_presence_state) = layer.forward(
                layer_idx,
                &queries,
                &query_pos,
                prompt_features,
                prompt_mask,
                &encoder_out.memory,
                &encoder_out.padding_mask,
                &encoder_out.pos_embed,
                cross_attn_mask.as_ref(),
                presence_state.as_ref(),
            )?;
            queries = next_queries;
            presence_state = next_presence_state;

            let normed_queries = self.output_norm.forward(&queries)?;
            debug::capture_tensor(
                &format!("decoder.layer.{layer_idx}.normed_queries"),
                &normed_queries,
            )?;
            let box_delta = self.bbox_embed.forward(&normed_queries)?;
            debug::capture_tensor(&format!("decoder.layer.{layer_idx}.box_delta"), &box_delta)?;
            let pred_boxes = (inverse_sigmoid(&layer_reference_boxes)? + box_delta)?
                .apply(&candle_nn::ops::sigmoid)?;
            debug::capture_tensor(
                &format!("decoder.layer.{layer_idx}.pred_boxes"),
                &pred_boxes,
            )?;
            if layer_idx + 1 == self.layers.len() {
                final_queries = Some(normed_queries.transpose(0, 1)?.contiguous()?);
                final_reference_boxes = Some(layer_reference_boxes.transpose(0, 1)?.contiguous()?);
                final_pred_boxes = Some(pred_boxes.transpose(0, 1)?.contiguous()?);
                if let (Some(presence_state), Some(presence_out_norm), Some(presence_head)) = (
                    presence_state.as_ref(),
                    self.presence_out_norm.as_ref(),
                    self.presence_head.as_ref(),
                ) {
                    let presence_logits = presence_head
                        .forward(&presence_out_norm.forward(presence_state)?)?
                        .squeeze(0)?;
                    debug::capture_tensor(
                        &format!("decoder.layer.{layer_idx}.presence_logits"),
                        &presence_logits,
                    )?;
                    final_presence_logits = Some(presence_logits);
                }
            } else {
                reference_boxes = pred_boxes;
            }
        }

        let queries = final_queries.expect("sam3 decoder requires at least one layer");
        let reference_boxes = final_reference_boxes
            .expect("sam3 decoder requires at least one layer reference state");
        let pred_boxes =
            final_pred_boxes.expect("sam3 decoder requires at least one refined box state");
        let pred_boxes_xyxy = box_cxcywh_to_xyxy(&pred_boxes)?;
        debug::capture_tensor("decoder.final_queries", &queries)?;
        debug::capture_tensor("decoder.final_reference_boxes", &reference_boxes)?;
        debug::capture_tensor("decoder.final_pred_boxes", &pred_boxes)?;
        debug::capture_tensor("decoder.final_pred_boxes_xyxy", &pred_boxes_xyxy)?;
        if let Some(presence_logits) = &final_presence_logits {
            debug::capture_tensor("decoder.final_presence_logits", presence_logits)?;
        }
        let pred_logits = self
            .dot_prod_scoring
            .forward(&queries, prompt_features, prompt_mask)?;
        debug::capture_tensor("decoder.pred_logits", &pred_logits)?;
        Ok(DecoderOutput {
            queries,
            reference_boxes,
            pred_logits,
            pred_boxes,
            pred_boxes_xyxy,
            presence_logits: final_presence_logits,
        })
    }

    fn build_box_relative_position_bias(
        &self,
        reference_boxes: &Tensor,
        spatial_shapes: &Tensor,
    ) -> Result<Option<Tensor>> {
        let (Some(box_rpb_embed_x), Some(box_rpb_embed_y)) =
            (&self.box_rpb_embed_x, &self.box_rpb_embed_y)
        else {
            return Ok(None);
        };
        let spatial_shapes = spatial_shapes.to_vec2::<u32>()?;
        if spatial_shapes.len() != 1 {
            candle::bail!(
                "sam3 decoder box relative position bias currently expects a single feature level, got {}",
                spatial_shapes.len()
            )
        }
        let (height, width) = (spatial_shapes[0][0] as usize, spatial_shapes[0][1] as usize);
        let boxes = reference_boxes.transpose(0, 1)?.to_vec3::<f32>()?;
        let batch_size = boxes.len();
        let num_queries = boxes[0].len();
        let x_coords = (0..width)
            .map(|idx| idx as f32 / width as f32)
            .collect::<Vec<_>>();
        let y_coords = (0..height)
            .map(|idx| idx as f32 / height as f32)
            .collect::<Vec<_>>();
        let rpb_input_dim = if self.config.box_rpb_mode == "both" {
            4
        } else {
            2
        };
        let mut deltas_x = Vec::with_capacity(batch_size * num_queries * width * rpb_input_dim);
        let mut deltas_y = Vec::with_capacity(batch_size * num_queries * height * rpb_input_dim);
        for batch in boxes.iter() {
            for box_coords in batch.iter() {
                let (x0, y0, x1, y1) =
                    cxcywh_to_xyxy(box_coords[0], box_coords[1], box_coords[2], box_coords[3]);
                for &coord in x_coords.iter() {
                    push_box_rpb_delta(
                        &mut deltas_x,
                        coord - x0,
                        coord - x1,
                        &self.config.box_rpb_mode,
                    );
                }
                for &coord in y_coords.iter() {
                    push_box_rpb_delta(
                        &mut deltas_y,
                        coord - y0,
                        coord - y1,
                        &self.config.box_rpb_mode,
                    );
                }
            }
        }
        let device = reference_boxes.device();
        let deltas_x = Tensor::from_vec(
            deltas_x,
            (batch_size, num_queries, width, rpb_input_dim),
            device,
        )?;
        let deltas_y = Tensor::from_vec(
            deltas_y,
            (batch_size, num_queries, height, rpb_input_dim),
            device,
        )?;
        let deltas_x = box_rpb_embed_x.forward(&deltas_x)?;
        let deltas_y = box_rpb_embed_y.forward(&deltas_y)?;
        let bias = deltas_y
            .unsqueeze(3)?
            .broadcast_add(&deltas_x.unsqueeze(2)?)?
            .reshape((
                batch_size,
                num_queries,
                height * width,
                self.config.num_heads,
            ))?
            .permute((0, 3, 1, 2))?
            .reshape((
                batch_size * self.config.num_heads,
                num_queries,
                height * width,
            ))?;
        Ok(Some(bias))
    }
}

fn normalize_padding_mask(mask: &Tensor, batch_size: usize, seq_len: usize) -> Result<Tensor> {
    match mask.dims() {
        [b, s] if *b == batch_size && *s == seq_len => Ok(mask.clone()),
        [s, b] if *s == seq_len && *b == batch_size => Ok(mask.transpose(0, 1)?.contiguous()?),
        shape => candle::bail!(
            "sam3 decoder expected padding mask shape ({batch_size}, {seq_len}) or ({seq_len}, {batch_size}), got {shape:?}"
        ),
    }
}

fn normalize_attn_bias(
    attn_bias: &Tensor,
    batch_size: usize,
    num_heads: usize,
    tgt_len: usize,
    src_len: usize,
) -> Result<Tensor> {
    match attn_bias.dims() {
        [bh, t, s] if *bh == batch_size * num_heads && *t == tgt_len && *s == src_len => {
            Ok(attn_bias.clone())
        }
        [b, h, t, s]
            if *b == batch_size && *h == num_heads && *t == tgt_len && *s == src_len =>
        {
            Ok(attn_bias.reshape((batch_size * num_heads, tgt_len, src_len))?)
        }
        shape => candle::bail!(
            "sam3 decoder expected attention bias shape ({}, {tgt_len}, {src_len}) or ({batch_size}, {num_heads}, {tgt_len}, {src_len}), got {shape:?}",
            batch_size * num_heads
        ),
    }
}

fn mean_pool_prompt(prompt: &Tensor, prompt_mask: &Tensor) -> Result<Tensor> {
    let (seq_len, batch_size, hidden_size) = prompt.dims3()?;
    let prompt_mask = normalize_padding_mask(prompt_mask, batch_size, seq_len)?;
    let is_valid = prompt_mask
        .to_dtype(DType::F32)?
        .affine(-1.0, 1.0)?
        .transpose(0, 1)?
        .reshape((seq_len, batch_size, 1))?;
    let pooled_prompt = prompt
        .to_dtype(DType::F32)?
        .broadcast_mul(&is_valid)?
        .sum(0)?
        .broadcast_div(&is_valid.sum(0)?.clamp(1e-6, f64::MAX)?)?;
    pooled_prompt.reshape((batch_size, hidden_size))
}

fn inverse_sigmoid(xs: &Tensor) -> Result<Tensor> {
    let xs = xs.clamp(0f32, 1f32)?;
    let x1 = xs.clamp(1e-3, 1f64)?;
    let x2 = ((1f64 - &xs)?).clamp(1e-3, 1f64)?;
    x1.broadcast_div(&x2)?.log()
}

fn gen_sineembed_for_position(pos_tensor: &Tensor, d_model: usize) -> Result<Tensor> {
    if d_model % 2 != 0 {
        candle::bail!(
            "sam3 decoder requires even d_model for sine position embedding, got {d_model}"
        )
    }
    let (num_queries, batch_size, coord_dim) = pos_tensor.dims3()?;
    let scale = 2.0 * PI as f32;
    let pos_tensor = pos_tensor.to_dtype(DType::F32)?.affine(scale as f64, 0.0)?;
    let out_dim = match coord_dim {
        2 => d_model,
        4 => d_model * 2,
        dim => candle::bail!("sam3 decoder expected 2D or 4D positions, got {dim}"),
    };
    let num_feats = d_model / 2;
    let y_embed = get_interleaved_1d_sine_pe(&pos_tensor.i((.., .., 1))?, num_feats)?;
    let x_embed = get_interleaved_1d_sine_pe(&pos_tensor.i((.., .., 0))?, num_feats)?;
    let embedding = if coord_dim == 4 {
        let w_embed = get_interleaved_1d_sine_pe(&pos_tensor.i((.., .., 2))?, num_feats)?;
        let h_embed = get_interleaved_1d_sine_pe(&pos_tensor.i((.., .., 3))?, num_feats)?;
        Tensor::cat(&[&y_embed, &x_embed, &w_embed, &h_embed], 2)?
    } else {
        Tensor::cat(&[&y_embed, &x_embed], 2)?
    };
    let embedding_shape = embedding.dims3()?;
    if embedding_shape != (num_queries, batch_size, out_dim) {
        candle::bail!(
            "sam3 decoder sine embedding expected shape ({num_queries}, {batch_size}, {out_dim}), got {embedding_shape:?}"
        )
    }
    Ok(embedding)
}

fn push_box_rpb_delta(out: &mut Vec<f32>, delta0: f32, delta1: f32, mode: &str) {
    match mode {
        "log" => {
            out.push(log_rpb(delta0));
            out.push(log_rpb(delta1));
        }
        "linear" => {
            out.push(delta0);
            out.push(delta1);
        }
        "both" => {
            out.push(delta0);
            out.push(delta1);
            out.push(log_rpb(delta0));
            out.push(log_rpb(delta1));
        }
        _ => {}
    }
}

fn log_rpb(delta: f32) -> f32 {
    let scaled = delta * 8.0;
    scaled.signum() * (scaled.abs() + 1.0).log2() / 8.0f32.log2()
}

fn cxcywh_to_xyxy(cx: f32, cy: f32, w: f32, h: f32) -> (f32, f32, f32, f32) {
    let half_w = w * 0.5;
    let half_h = h * 0.5;
    (cx - half_w, cy - half_h, cx + half_w, cy + half_h)
}

fn box_cxcywh_to_xyxy(boxes: &Tensor) -> Result<Tensor> {
    let (batch_size, num_queries, dims) = boxes.dims3()?;
    if dims != 4 {
        candle::bail!("sam3 decoder expected boxes with last dimension 4, got {dims}")
    }
    let device = boxes.device();
    let boxes = boxes.to_vec3::<f32>()?;
    let mut xyxy = Vec::with_capacity(batch_size * num_queries * 4);
    for batch in boxes.iter() {
        for box_coords in batch.iter() {
            let (x0, y0, x1, y1) =
                cxcywh_to_xyxy(box_coords[0], box_coords[1], box_coords[2], box_coords[3]);
            xyxy.extend([x0, y0, x1, y1]);
        }
    }
    Tensor::from_vec(xyxy, (batch_size, num_queries, 4), device)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use candle::{DType, Device, IndexOp, Result, Tensor};
    use candle_nn::VarBuilder;
    use serde::Deserialize;

    use super::{gen_sineembed_for_position, DecoderOutput, Sam3TransformerDecoder};
    use crate::models::sam3::config::DecoderConfig;
    use crate::models::sam3::debug::{self, DebugExporter};
    use crate::models::sam3::encoder::FusionEncoderOutput;

    #[test]
    fn decoder_returns_final_queries_boxes_and_scores() -> Result<()> {
        let device = Device::Cpu;
        let config = test_config();
        let decoder_vb =
            VarBuilder::from_tensors(decoder_weights(&config, &device)?, DType::F32, &device);
        let score_vb =
            VarBuilder::from_tensors(score_weights(&config, &device)?, DType::F32, &device);
        let decoder = Sam3TransformerDecoder::new(&config, decoder_vb, score_vb)?;
        let encoder_out = encoder_out(&config, &device)?;
        let prompt = Tensor::zeros((3, 1, config.d_model), DType::F32, &device)?;
        let prompt_mask = Tensor::zeros((1, 3), DType::U8, &device)?;
        let output = decoder.forward(&encoder_out, &prompt, &prompt_mask)?;
        assert_decoder_output(output, &config)
    }

    #[derive(Debug, Deserialize)]
    struct DecoderFixtureMetadata {
        d_model: usize,
        num_heads: usize,
        dim_feedforward: usize,
        num_layers: usize,
        num_queries: usize,
        height: usize,
        width: usize,
    }

    #[test]
    fn decoder_fixture_smoke_final_matches_upstream() -> Result<()> {
        let output = run_fixture_decoder_forward()?;
        let expected = load_decoder_fixture_tensors("fixture.safetensors")?;
        assert_tensor_close(
            &output.pred_logits,
            fixture_tensor(&expected, "decoder.pred_logits")?,
            1e-5,
            "decoder.pred_logits",
        )?;
        assert_tensor_close(
            &output.pred_boxes_xyxy,
            fixture_tensor(&expected, "decoder.final_pred_boxes_xyxy")?,
            1e-5,
            "decoder.final_pred_boxes_xyxy",
        )?;
        Ok(())
    }

    #[test]
    #[ignore = "fixture-driven parity investigation"]
    fn decoder_fixture_helper_parity_matches_upstream() -> Result<()> {
        let fixture = load_decoder_fixture_tensors("fixture.safetensors")?;
        let reference_boxes = fixture_tensor(&fixture, "decoder.initial_reference_boxes")?;
        let valid_ratios_twice = fixture_tensor(&fixture, "decoder.valid_ratios_twice")?;
        let reference_points_input = reference_boxes
            .unsqueeze(2)?
            .broadcast_mul(&valid_ratios_twice.unsqueeze(0)?)?;
        assert_tensor_close(
            &gen_sineembed_for_position(
                &reference_points_input.i((.., .., 0, ..))?,
                fixture_metadata()?.d_model,
            )?,
            fixture_tensor(&fixture, "decoder.layer.0.query_sine_embed")?,
            1e-5,
            "decoder.layer.0.query_sine_embed",
        )?;
        Ok(())
    }

    #[test]
    #[ignore = "fixture-driven parity investigation"]
    fn decoder_fixture_layer_parity_matches_upstream() -> Result<()> {
        let (_output, debug_tensors) = run_fixture_decoder()?;
        let expected = load_decoder_fixture_tensors("fixture.safetensors")?;
        let keys = [
            "decoder.layer.0.reference_boxes",
            "decoder.layer.0.query_sine_embed",
            "decoder.layer.0.query_pos",
            "decoder.layer.0.cross_attn_mask_pre_presence",
            "decoder.layer.0.input_with_presence",
            "decoder.layer.0.self_attn_output",
            "decoder.layer.0.post_self_attn",
            "decoder.layer.0.text_attn_output",
            "decoder.layer.0.post_text_cross",
            "decoder.layer.0.cross_attn_mask",
            "decoder.layer.0.image_attn_output",
            "decoder.layer.0.post_image_cross",
            "decoder.layer.0.ffn_hidden",
            "decoder.layer.0.ffn_output",
            "decoder.layer.0.output",
            "decoder.layer.0.queries_output",
            "decoder.layer.0.presence_output",
            "decoder.layer.0.normed_queries",
            "decoder.layer.0.box_delta",
            "decoder.layer.0.pred_boxes",
            "decoder.layer.0.presence_logits",
        ];
        assert_debug_keys_close(&debug_tensors, &expected, &keys, 1e-5)
    }

    #[test]
    #[ignore = "fixture-driven parity investigation"]
    fn decoder_fixture_final_parity_matches_upstream() -> Result<()> {
        let (output, debug_tensors) = run_fixture_decoder()?;
        let expected = load_decoder_fixture_tensors("fixture.safetensors")?;
        let keys = [
            "decoder.dotprod.prompt_after_mlp",
            "decoder.dotprod.pooled_prompt",
            "decoder.dotprod.prompt_proj",
            "decoder.dotprod.query_proj",
            "decoder.dotprod.scores_pre_clamp",
            "decoder.dotprod.scores",
            "decoder.final_queries",
            "decoder.final_reference_boxes",
            "decoder.final_pred_boxes",
            "decoder.final_pred_boxes_xyxy",
            "decoder.final_presence_logits",
            "decoder.pred_logits",
        ];
        assert_debug_keys_close(&debug_tensors, &expected, &keys, 1e-5)?;
        assert_tensor_close(
            &output.queries,
            fixture_tensor(&expected, "decoder.final_queries")?,
            1e-5,
            "decoder.final_queries",
        )?;
        assert_tensor_close(
            &output.reference_boxes,
            fixture_tensor(&expected, "decoder.final_reference_boxes")?,
            1e-5,
            "decoder.final_reference_boxes",
        )?;
        assert_tensor_close(
            &output.pred_boxes,
            fixture_tensor(&expected, "decoder.final_pred_boxes")?,
            1e-5,
            "decoder.final_pred_boxes",
        )?;
        assert_tensor_close(
            &output.pred_boxes_xyxy,
            fixture_tensor(&expected, "decoder.final_pred_boxes_xyxy")?,
            1e-5,
            "decoder.final_pred_boxes_xyxy",
        )?;
        assert_tensor_close(
            output
                .presence_logits
                .as_ref()
                .expect("presence logits should exist"),
            fixture_tensor(&expected, "decoder.final_presence_logits")?,
            1e-5,
            "decoder.final_presence_logits",
        )?;
        assert_tensor_close(
            &output.pred_logits,
            fixture_tensor(&expected, "decoder.pred_logits")?,
            1e-5,
            "decoder.pred_logits",
        )?;
        Ok(())
    }

    fn assert_decoder_output(output: DecoderOutput, config: &DecoderConfig) -> Result<()> {
        assert_eq!(
            output.queries.dims3()?,
            (1, config.num_queries, config.d_model)
        );
        assert_eq!(output.reference_boxes.dims3()?, (1, config.num_queries, 4));
        assert_eq!(output.pred_logits.dims3()?, (1, config.num_queries, 1));
        assert_eq!(output.pred_boxes.dims3()?, (1, config.num_queries, 4));
        assert_eq!(output.pred_boxes_xyxy.dims3()?, (1, config.num_queries, 4));
        assert_eq!(output.presence_logits.unwrap().dims2()?, (1, 1));
        Ok(())
    }

    fn test_config() -> DecoderConfig {
        DecoderConfig {
            d_model: 8,
            num_layers: 1,
            num_queries: 2,
            num_heads: 2,
            dim_feedforward: 16,
            presence_token: true,
            use_text_cross_attention: true,
            box_rpb_mode: "log".to_owned(),
            box_rpb_resolution: 28,
            box_rpb_stride: 14,
            clamp_presence_logit_max: 10.0,
        }
    }

    fn fixture_metadata() -> Result<DecoderFixtureMetadata> {
        let path = decoder_fixture_dir().join("metadata.json");
        let contents = fs::read_to_string(&path).map_err(|err| {
            candle::Error::Msg(format!(
                "failed to read decoder fixture metadata {}: {err}",
                path.display()
            ))
        })?;
        serde_json::from_str(&contents).map_err(|err| {
            candle::Error::Msg(format!(
                "failed to parse decoder fixture metadata {}: {err}",
                path.display()
            ))
        })
    }

    fn fixture_config() -> Result<DecoderConfig> {
        let metadata = fixture_metadata()?;
        Ok(DecoderConfig {
            d_model: metadata.d_model,
            num_layers: metadata.num_layers,
            num_queries: metadata.num_queries,
            num_heads: metadata.num_heads,
            dim_feedforward: metadata.dim_feedforward,
            presence_token: true,
            use_text_cross_attention: true,
            box_rpb_mode: "log".to_owned(),
            box_rpb_resolution: metadata.height,
            box_rpb_stride: 1,
            clamp_presence_logit_max: 10.0,
        })
    }

    fn decoder_fixture_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/sam3_decoder_unit")
    }

    fn load_decoder_fixture_tensors(file_name: &str) -> Result<HashMap<String, Tensor>> {
        let path = decoder_fixture_dir().join(file_name);
        candle::safetensors::load(&path, &Device::Cpu).map_err(|err| {
            candle::Error::Msg(format!(
                "failed to load decoder fixture {}: {err}",
                path.display()
            ))
        })
    }

    fn fixture_tensor<'a>(fixture: &'a HashMap<String, Tensor>, key: &str) -> Result<&'a Tensor> {
        fixture
            .get(key)
            .ok_or_else(|| candle::Error::Msg(format!("decoder fixture is missing tensor `{key}`")))
    }

    fn unique_temp_dir(label: &str) -> Result<PathBuf> {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|err| candle::Error::Msg(format!("system clock error: {err}")))?;
        let path = std::env::temp_dir().join(format!(
            "sam3_decoder_fixture_{label}_{}_{}",
            std::process::id(),
            stamp.as_nanos()
        ));
        fs::create_dir_all(&path).map_err(|err| {
            candle::Error::Msg(format!(
                "failed to create temp debug dir {}: {err}",
                path.display()
            ))
        })?;
        Ok(path)
    }

    fn fixture_encoder_out(
        fixture: &HashMap<String, Tensor>,
        _config: &DecoderConfig,
    ) -> Result<FusionEncoderOutput> {
        Ok(FusionEncoderOutput {
            memory: fixture_tensor(fixture, "inputs/memory")?.clone(),
            pos_embed: fixture_tensor(fixture, "inputs/pos_embed")?.clone(),
            padding_mask: fixture_tensor(fixture, "inputs/padding_mask")?.clone(),
            level_start_index: fixture_tensor(fixture, "inputs/level_start_index")?.clone(),
            spatial_shapes: fixture_tensor(fixture, "inputs/spatial_shapes")?.clone(),
            valid_ratios: fixture_tensor(fixture, "inputs/valid_ratios")?.clone(),
        })
    }

    fn run_fixture_decoder() -> Result<(DecoderOutput, HashMap<String, Tensor>)> {
        let device = Device::Cpu;
        let config = fixture_config()?;
        let decoder_weights = load_decoder_fixture_tensors("decoder_weights.safetensors")?;
        let score_weights = load_decoder_fixture_tensors("score_weights.safetensors")?;
        let fixture = load_decoder_fixture_tensors("fixture.safetensors")?;
        let decoder_vb = VarBuilder::from_tensors(decoder_weights, DType::F32, &device);
        let score_vb = VarBuilder::from_tensors(score_weights, DType::F32, &device);
        let decoder = Sam3TransformerDecoder::new(&config, decoder_vb, score_vb)?;
        let encoder_out = fixture_encoder_out(&fixture, &config)?;
        let prompt = fixture_tensor(&fixture, "inputs/prompt")?.clone();
        let prompt_mask = fixture_tensor(&fixture, "inputs/prompt_mask")?.clone();

        let debug_dir = unique_temp_dir("forward")?;
        debug::set_exporter(Some(DebugExporter::new(&debug_dir)?));
        let output = decoder.forward(&encoder_out, &prompt, &prompt_mask)?;
        debug::finish()?;
        let debug_tensors =
            candle::safetensors::load(debug_dir.join("debug_tensors.safetensors"), &device)
                .map_err(|err| {
                    candle::Error::Msg(format!(
                        "failed to load decoder debug tensors from {}: {err}",
                        debug_dir.display()
                    ))
                })?;
        let _ = fs::remove_dir_all(&debug_dir);
        Ok((output, debug_tensors))
    }

    fn run_fixture_decoder_forward() -> Result<DecoderOutput> {
        let device = Device::Cpu;
        let config = fixture_config()?;
        let decoder_weights = load_decoder_fixture_tensors("decoder_weights.safetensors")?;
        let score_weights = load_decoder_fixture_tensors("score_weights.safetensors")?;
        let fixture = load_decoder_fixture_tensors("fixture.safetensors")?;
        let decoder_vb = VarBuilder::from_tensors(decoder_weights, DType::F32, &device);
        let score_vb = VarBuilder::from_tensors(score_weights, DType::F32, &device);
        let decoder = Sam3TransformerDecoder::new(&config, decoder_vb, score_vb)?;
        let encoder_out = fixture_encoder_out(&fixture, &config)?;
        let prompt = fixture_tensor(&fixture, "inputs/prompt")?.clone();
        let prompt_mask = fixture_tensor(&fixture, "inputs/prompt_mask")?.clone();
        decoder.forward(&encoder_out, &prompt, &prompt_mask)
    }

    fn assert_debug_keys_close(
        actual: &HashMap<String, Tensor>,
        expected: &HashMap<String, Tensor>,
        keys: &[&str],
        atol: f32,
    ) -> Result<()> {
        let mut failures = Vec::new();
        for key in keys {
            let Some(actual_tensor) = actual.get(*key) else {
                failures.push(format!("{key}: missing from Candle debug output"));
                continue;
            };
            let Some(expected_tensor) = expected.get(*key) else {
                failures.push(format!("{key}: missing from fixture"));
                continue;
            };
            if let Err(err) = assert_tensor_close(actual_tensor, expected_tensor, atol, key) {
                failures.push(err.to_string());
            }
        }
        if failures.is_empty() {
            return Ok(());
        }
        candle::bail!("{}", failures.join("\n"));
    }

    fn assert_tensor_close(
        actual: &Tensor,
        expected: &Tensor,
        atol: f32,
        name: &str,
    ) -> Result<()> {
        if actual.dims() != expected.dims() {
            candle::bail!(
                "{name}: shape mismatch actual={:?} expected={:?}",
                actual.dims(),
                expected.dims()
            );
        }
        let actual = actual.to_dtype(DType::F32)?;
        let expected = expected.to_dtype(DType::F32)?;
        let max_abs_diff = actual
            .broadcast_sub(&expected)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        if max_abs_diff > atol {
            candle::bail!("{name}: max_abs_diff={max_abs_diff:.8}");
        }
        Ok(())
    }

    fn encoder_out(config: &DecoderConfig, device: &Device) -> Result<FusionEncoderOutput> {
        Ok(FusionEncoderOutput {
            memory: Tensor::zeros((4, 1, config.d_model), DType::F32, device)?,
            pos_embed: Tensor::zeros((4, 1, config.d_model), DType::F32, device)?,
            padding_mask: Tensor::zeros((4, 1), DType::U8, device)?,
            level_start_index: Tensor::from_vec(vec![0u32], 1, device)?,
            spatial_shapes: Tensor::from_vec(vec![2u32, 2u32], (1, 2), device)?,
            valid_ratios: Tensor::from_vec(vec![1f32, 1f32], (1, 1, 2), device)?,
        })
    }

    fn decoder_weights(config: &DecoderConfig, device: &Device) -> Result<HashMap<String, Tensor>> {
        let mut tensors = HashMap::new();
        for attn_name in ["cross_attn", "ca_text", "self_attn"] {
            tensors.insert(
                format!("layers.0.{attn_name}.in_proj_weight"),
                Tensor::zeros((config.d_model * 3, config.d_model), DType::F32, device)?,
            );
            tensors.insert(
                format!("layers.0.{attn_name}.in_proj_bias"),
                Tensor::zeros(config.d_model * 3, DType::F32, device)?,
            );
            tensors.insert(
                format!("layers.0.{attn_name}.out_proj.weight"),
                Tensor::zeros((config.d_model, config.d_model), DType::F32, device)?,
            );
            tensors.insert(
                format!("layers.0.{attn_name}.out_proj.bias"),
                Tensor::zeros(config.d_model, DType::F32, device)?,
            );
        }
        for norm_name in ["norm", "presence_token_out_norm"] {
            tensors.insert(
                format!("{norm_name}.weight"),
                Tensor::ones(config.d_model, DType::F32, device)?,
            );
            tensors.insert(
                format!("{norm_name}.bias"),
                Tensor::zeros(config.d_model, DType::F32, device)?,
            );
        }
        for norm_name in ["norm1", "catext_norm", "norm2", "norm3"] {
            tensors.insert(
                format!("layers.0.{norm_name}.weight"),
                Tensor::ones(config.d_model, DType::F32, device)?,
            );
            tensors.insert(
                format!("layers.0.{norm_name}.bias"),
                Tensor::zeros(config.d_model, DType::F32, device)?,
            );
        }
        for linear_name in ["linear1", "linear2"] {
            let (out_dim, in_dim) = if linear_name == "linear1" {
                (config.dim_feedforward, config.d_model)
            } else {
                (config.d_model, config.dim_feedforward)
            };
            tensors.insert(
                format!("layers.0.{linear_name}.weight"),
                Tensor::zeros((out_dim, in_dim), DType::F32, device)?,
            );
            tensors.insert(
                format!("layers.0.{linear_name}.bias"),
                Tensor::zeros(out_dim, DType::F32, device)?,
            );
        }
        tensors.insert(
            "query_embed.weight".into(),
            Tensor::zeros((config.num_queries, config.d_model), DType::F32, device)?,
        );
        tensors.insert(
            "reference_points.weight".into(),
            Tensor::zeros((config.num_queries, 4), DType::F32, device)?,
        );
        tensors.insert(
            "presence_token.weight".into(),
            Tensor::zeros((1, config.d_model), DType::F32, device)?,
        );
        add_mlp_weights(
            &mut tensors,
            "bbox_embed",
            &[config.d_model, config.d_model, config.d_model, 4],
            device,
        )?;
        add_mlp_weights(
            &mut tensors,
            "ref_point_head",
            &[2 * config.d_model, config.d_model, config.d_model],
            device,
        )?;
        add_mlp_weights(
            &mut tensors,
            "presence_token_head",
            &[config.d_model, config.d_model, config.d_model, 1],
            device,
        )?;
        add_mlp_weights(
            &mut tensors,
            "boxRPB_embed_x",
            &[2, config.d_model, config.num_heads],
            device,
        )?;
        add_mlp_weights(
            &mut tensors,
            "boxRPB_embed_y",
            &[2, config.d_model, config.num_heads],
            device,
        )?;
        Ok(tensors)
    }

    fn score_weights(config: &DecoderConfig, device: &Device) -> Result<HashMap<String, Tensor>> {
        let mut tensors = HashMap::new();
        add_mlp_weights(
            &mut tensors,
            "prompt_mlp",
            &[config.d_model, config.dim_feedforward, config.d_model],
            device,
        )?;
        tensors.insert(
            "prompt_mlp.out_norm.weight".into(),
            Tensor::ones(config.d_model, DType::F32, device)?,
        );
        tensors.insert(
            "prompt_mlp.out_norm.bias".into(),
            Tensor::zeros(config.d_model, DType::F32, device)?,
        );
        tensors.insert(
            "prompt_proj.weight".into(),
            Tensor::zeros((config.d_model, config.d_model), DType::F32, device)?,
        );
        tensors.insert(
            "prompt_proj.bias".into(),
            Tensor::zeros(config.d_model, DType::F32, device)?,
        );
        tensors.insert(
            "hs_proj.weight".into(),
            Tensor::zeros((config.d_model, config.d_model), DType::F32, device)?,
        );
        tensors.insert(
            "hs_proj.bias".into(),
            Tensor::zeros(config.d_model, DType::F32, device)?,
        );
        Ok(tensors)
    }

    fn add_mlp_weights(
        tensors: &mut HashMap<String, Tensor>,
        prefix: &str,
        dims: &[usize],
        device: &Device,
    ) -> Result<()> {
        for layer_idx in 0..(dims.len() - 1) {
            tensors.insert(
                format!("{prefix}.layers.{layer_idx}.weight"),
                Tensor::zeros((dims[layer_idx + 1], dims[layer_idx]), DType::F32, device)?,
            );
            tensors.insert(
                format!("{prefix}.layers.{layer_idx}.bias"),
                Tensor::zeros(dims[layer_idx + 1], DType::F32, device)?,
            );
        }
        Ok(())
    }
}
