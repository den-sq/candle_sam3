use std::f32::consts::PI;

use candle::{DType, IndexOp, Result, Tensor};
use candle_nn::{Conv2d, Embedding, LayerNorm, Linear, Module, VarBuilder};

use super::config::GeometryConfig;
use super::debug;

#[derive(Debug, Default, Clone)]
pub struct GeometryPrompt {
    pub boxes_cxcywh: Option<Tensor>,
    pub box_labels: Option<Tensor>,
    pub points_xy: Option<Tensor>,
    pub point_labels: Option<Tensor>,
    pub masks: Option<Tensor>,
    pub mask_labels: Option<Tensor>,
}

impl GeometryPrompt {
    pub fn is_empty(&self) -> bool {
        self.boxes_cxcywh.is_none()
            && self.box_labels.is_none()
            && self.points_xy.is_none()
            && self.point_labels.is_none()
            && self.masks.is_none()
            && self.mask_labels.is_none()
    }
}

#[derive(Debug)]
pub struct EncodedPrompt {
    /// Sequence-first prompt features, shape `[seq, batch, d_model]`.
    pub features: Tensor,
    /// SAM3 uses `1` for padding and `0` for valid tokens.
    pub padding_mask: Tensor,
}

#[derive(Debug)]
struct GeometryAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl GeometryAttention {
    fn new(config: &GeometryConfig, vb: VarBuilder) -> Result<Self> {
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
            .contiguous()
    }

    fn forward(
        &self,
        query: &Tensor,
        key_value: &Tensor,
        key_padding_mask: Option<&Tensor>,
        key_pos: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (tgt_len, batch_size, hidden_size) = query.dims3()?;
        let src_len = key_value.dim(0)?;
        let key = match key_pos {
            Some(key_pos) => key_value.broadcast_add(key_pos)?,
            None => key_value.clone(),
        };
        let q = self
            .project(query, &self.q_proj, batch_size, tgt_len)?
            .reshape((batch_size * self.num_heads, tgt_len, self.head_dim))?
            .to_dtype(DType::F32)?;
        let q = (q * self.scale)?;
        let k = self
            .project(&key, &self.k_proj, batch_size, src_len)?
            .reshape((batch_size * self.num_heads, src_len, self.head_dim))?
            .to_dtype(DType::F32)?;
        let v = self
            .project(key_value, &self.v_proj, batch_size, src_len)?
            .reshape((batch_size * self.num_heads, src_len, self.head_dim))?
            .to_dtype(DType::F32)?;
        let mut attn = q.matmul(&k.transpose(1, 2)?)?.reshape((
            batch_size,
            self.num_heads,
            tgt_len,
            src_len,
        ))?;
        if let Some(key_padding_mask) = key_padding_mask {
            let mask_shape = key_padding_mask.dims2()?;
            if mask_shape != (batch_size, src_len) {
                candle::bail!(
                    "sam3 geometry attention expected key padding mask shape ({batch_size}, {src_len}), got {mask_shape:?}"
                )
            }
            let additive_mask = (key_padding_mask.to_dtype(DType::F32)? * -1e9f64)?
                .reshape((batch_size, 1, 1, src_len))?;
            attn = attn.broadcast_add(&additive_mask)?;
        }
        let attn = candle_nn::ops::softmax_last_dim(&attn)?.reshape((
            batch_size * self.num_heads,
            tgt_len,
            src_len,
        ))?;
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
struct GeometryEncoderLayer {
    norm1: LayerNorm,
    self_attn: GeometryAttention,
    norm2: LayerNorm,
    cross_attn_image: GeometryAttention,
    norm3: LayerNorm,
    linear1: Linear,
    linear2: Linear,
}

impl GeometryEncoderLayer {
    fn new(config: &GeometryConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            norm1: candle_nn::layer_norm(config.d_model, 1e-5, vb.pp("norm1"))?,
            self_attn: GeometryAttention::new(config, vb.pp("self_attn"))?,
            norm2: candle_nn::layer_norm(config.d_model, 1e-5, vb.pp("norm2"))?,
            cross_attn_image: GeometryAttention::new(config, vb.pp("cross_attn_image"))?,
            norm3: candle_nn::layer_norm(config.d_model, 1e-5, vb.pp("norm3"))?,
            linear1: candle_nn::linear(config.d_model, config.dim_feedforward, vb.pp("linear1"))?,
            linear2: candle_nn::linear(config.dim_feedforward, config.d_model, vb.pp("linear2"))?,
        })
    }

    fn forward(
        &self,
        prompt_feats: &Tensor,
        vision_feats: &Tensor,
        vision_pos_encoding: &Tensor,
        prompt_padding_mask: &Tensor,
    ) -> Result<Tensor> {
        let residual = prompt_feats;
        let hidden_states = self.norm1.forward(prompt_feats)?;

        let hidden_states = self.self_attn.forward(
            &hidden_states,
            &hidden_states,
            Some(prompt_padding_mask),
            None,
        )?;

        let hidden_states = (hidden_states + residual)?;

        let residual = &hidden_states;
        let hidden_states = self.norm2.forward(&hidden_states)?;

        let hidden_states = self.cross_attn_image.forward(
            &hidden_states,
            vision_feats,
            None,
            Some(vision_pos_encoding),
        )?;

        let hidden_states = (hidden_states + residual)?;

        let residual = &hidden_states;
        let hidden_states = self.norm3.forward(&hidden_states)?;

        let hidden_states = self.linear1.forward(&hidden_states)?.relu()?;

        let hidden_states = self.linear2.forward(&hidden_states)?;

        let result = (hidden_states + residual)?;
        Ok(result)
    }
}

#[derive(Debug)]
pub struct SequenceGeometryEncoder {
    config: GeometryConfig,
    label_embed: Embedding,
    cls_embed: Option<Tensor>,
    points_direct_project: Option<Linear>,
    points_pool_project: Option<Linear>,
    points_pos_enc_project: Option<Linear>,
    boxes_direct_project: Option<Linear>,
    boxes_pool_project: Option<Conv2d>,
    boxes_pos_enc_project: Option<Linear>,
    img_pre_norm: Option<LayerNorm>,
    final_proj: Option<Linear>,
    norm: LayerNorm,
    encode: Vec<GeometryEncoderLayer>,
    encode_norm: Option<LayerNorm>,
}

impl SequenceGeometryEncoder {
    pub fn new(config: &GeometryConfig, vb: VarBuilder) -> Result<Self> {
        if config.num_heads == 0 || config.d_model % config.num_heads != 0 {
            candle::bail!(
                "sam3 geometry d_model ({}) must be divisible by num_heads ({})",
                config.d_model,
                config.num_heads
            )
        }
        let label_embed = candle_nn::embedding(2, config.d_model, vb.pp("label_embed"))?;
        let cls_embed = if config.add_cls && vb.contains_tensor("cls_embed.weight") {
            Some(vb.pp("cls_embed").get((1, config.d_model), "weight")?)
        } else {
            None
        };
        let points_direct_project = if vb.contains_tensor("points_direct_project.weight") {
            Some(candle_nn::linear(
                2,
                config.d_model,
                vb.pp("points_direct_project"),
            )?)
        } else {
            None
        };
        let points_pool_project = if vb.contains_tensor("points_pool_project.weight") {
            Some(candle_nn::linear(
                config.d_model,
                config.d_model,
                vb.pp("points_pool_project"),
            )?)
        } else {
            None
        };
        let points_pos_enc_project = if vb.contains_tensor("points_pos_enc_project.weight") {
            Some(candle_nn::linear(
                config.d_model,
                config.d_model,
                vb.pp("points_pos_enc_project"),
            )?)
        } else {
            None
        };
        let boxes_direct_project = if vb.contains_tensor("boxes_direct_project.weight") {
            Some(candle_nn::linear(
                4,
                config.d_model,
                vb.pp("boxes_direct_project"),
            )?)
        } else {
            None
        };
        let boxes_pool_project = if vb.contains_tensor("boxes_pool_project.weight") {
            Some(candle_nn::conv2d(
                config.d_model,
                config.d_model,
                config.roi_size,
                Default::default(),
                vb.pp("boxes_pool_project"),
            )?)
        } else {
            None
        };
        let boxes_pos_enc_project = if vb.contains_tensor("boxes_pos_enc_project.weight") {
            Some(candle_nn::linear(
                config.d_model + 2,
                config.d_model,
                vb.pp("boxes_pos_enc_project"),
            )?)
        } else {
            None
        };
        let img_pre_norm = if vb.contains_tensor("img_pre_norm.weight") {
            Some(candle_nn::layer_norm(
                config.d_model,
                1e-5,
                vb.pp("img_pre_norm"),
            )?)
        } else {
            None
        };
        let final_proj = if config.add_post_encode_proj && vb.contains_tensor("final_proj.weight") {
            Some(candle_nn::linear(
                config.d_model,
                config.d_model,
                vb.pp("final_proj"),
            )?)
        } else {
            eprintln!(
                "[DEBUG] final_proj NOT loaded (add_post_encode_proj={}, has_weight={})",
                config.add_post_encode_proj,
                vb.contains_tensor("final_proj.weight")
            );
            None
        };
        let norm = candle_nn::layer_norm(config.d_model, 1e-5, vb.pp("norm"))?;
        let mut encode = Vec::with_capacity(config.num_layers);
        let encode_vb = vb.pp("encode");
        for layer_idx in 0..config.num_layers {
            encode.push(GeometryEncoderLayer::new(config, encode_vb.pp(layer_idx))?);
        }
        let encode_norm = if vb.contains_tensor("encode_norm.weight") {
            Some(candle_nn::layer_norm(
                config.d_model,
                1e-5,
                vb.pp("encode_norm"),
            )?)
        } else {
            None
        };
        Ok(Self {
            config: config.clone(),
            label_embed,
            cls_embed,
            points_direct_project,
            points_pool_project,
            points_pos_enc_project,
            boxes_direct_project,
            boxes_pool_project,
            boxes_pos_enc_project,
            img_pre_norm,
            final_proj,
            norm,
            encode,
            encode_norm,
        })
    }

    pub fn config(&self) -> &GeometryConfig {
        &self.config
    }

    pub fn encode(
        &self,
        prompt: &GeometryPrompt,
        image_features: &[Tensor],
        image_pos_embeds: &[Tensor],
    ) -> Result<EncodedPrompt> {
        if prompt.masks.is_some() || prompt.mask_labels.is_some() {
            candle::bail!(
                "sam3 geometry mask prompts are not supported by the loaded image checkpoint"
            )
        }
        if prompt.points_xy.is_none() && prompt.point_labels.is_some() {
            candle::bail!("sam3 geometry point labels were provided without point coordinates")
        }
        if prompt.boxes_cxcywh.is_none() && prompt.box_labels.is_some() {
            candle::bail!("sam3 geometry box labels were provided without box coordinates")
        }

        let (vision_feats, vision_pos_embeds, pooled_vision_feats) =
            self.prepare_image_context(image_features, image_pos_embeds)?;
        let batch_size = vision_feats.dim(1)?;
        let mut feature_parts = Vec::new();
        let mut mask_parts = Vec::new();

        if prompt.points_xy.is_some() {
            let (point_features, point_mask) =
                self.encode_points(prompt, batch_size, &pooled_vision_feats)?;
            feature_parts.push(point_features);
            mask_parts.push(point_mask);
        }
        if prompt.boxes_cxcywh.is_some() {
            let (box_features, box_mask) =
                self.encode_boxes(prompt, batch_size, &pooled_vision_feats)?;
            feature_parts.push(box_features);
            mask_parts.push(box_mask);
        }
        if let Some(cls_embed) = &self.cls_embed {
            let cls_embed = cls_embed
                .reshape((1, 1, self.config.d_model))?
                .repeat((1, batch_size, 1))?;
            feature_parts.push(cls_embed);
            mask_parts.push(Tensor::zeros(
                (batch_size, 1),
                DType::U8,
                image_features[0].device(),
            )?);
        }

        let mut features = concat_seq_tensors(&feature_parts)?;
        let padding_mask = concat_batch_tensors(&mask_parts)?;
        if let Some(final_proj) = &self.final_proj {
            features = final_proj.forward(&features)?;
        }
        features = self.norm.forward(&features)?;
        debug::capture_tensor("geometry/features_initial_norm", &features)?;

        for (layer_idx, layer) in self.encode.iter().enumerate() {
            features =
                layer.forward(&features, &vision_feats, &vision_pos_embeds, &padding_mask)?;
            debug::capture_tensor(
                &format!("geometry/features_after_layer_{}", layer_idx),
                &features,
            )?;
        }

        if let Some(encode_norm) = &self.encode_norm {
            features = encode_norm.forward(&features)?;
        }
        debug::capture_tensor("geometry/features_final", &features)?;
        Ok(EncodedPrompt {
            features,
            padding_mask,
        })
    }

    fn prepare_image_context(
        &self,
        image_features: &[Tensor],
        image_pos_embeds: &[Tensor],
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let Some(vision_feats_bchw) = image_features.last() else {
            candle::bail!("sam3 geometry encoder requires at least one image feature map")
        };
        let vision_pos_bchw = match image_pos_embeds.last() {
            Some(pos) => pos.clone(),
            None => vision_feats_bchw.zeros_like()?,
        };
        let (batch_size, channels, height, width) = vision_feats_bchw.dims4()?;
        if channels != self.config.d_model {
            candle::bail!(
                "sam3 geometry encoder expected image feature width {}, got {channels}",
                self.config.d_model
            )
        }
        let vision_feats = vision_feats_bchw.permute((2, 3, 0, 1))?.reshape((
            height * width,
            batch_size,
            channels,
        ))?;
        let vision_pos_embeds = vision_pos_bchw.permute((2, 3, 0, 1))?.reshape((
            height * width,
            batch_size,
            channels,
        ))?;
        let pooled_vision_feats = match &self.img_pre_norm {
            Some(img_pre_norm) => img_pre_norm
                .forward(&vision_feats)?
                .reshape((height, width, batch_size, channels))?
                .permute((2, 3, 0, 1))?,
            None => vision_feats_bchw.clone(),
        };
        Ok((vision_feats, vision_pos_embeds, pooled_vision_feats))
    }

    fn encode_points(
        &self,
        prompt: &GeometryPrompt,
        batch_size: usize,
        vision_feats: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let points_xy = normalize_prompt_coords(
            prompt
                .points_xy
                .as_ref()
                .expect("points were checked before encoding"),
            batch_size,
            2,
        )?;
        let seq_len = points_xy.dim(0)?;
        let point_labels = normalize_prompt_labels(
            prompt.point_labels.as_ref(),
            batch_size,
            seq_len,
            points_xy.device(),
        )?;
        let label_embed = self.label_embed.forward(&point_labels)?;
        debug::capture_tensor("geometry/point_label_embed", &label_embed)?;
        let mut point_features = label_embed.clone();

        if let Some(points_direct_project) = &self.points_direct_project {
            let direct_proj = points_direct_project.forward(&points_xy)?;
            debug::capture_tensor("geometry/point_direct_proj", &direct_proj)?;
            point_features = point_features.broadcast_add(&direct_proj)?;
        }
        if let Some(points_pool_project) = &self.points_pool_project {
            let pooled_points = sample_points_nearest(vision_feats, &points_xy)?;
            debug::capture_tensor("geometry/point_sampled_raw", &pooled_points)?;
            let pool_proj = points_pool_project.forward(&pooled_points)?;
            debug::capture_tensor("geometry/point_pool_proj", &pool_proj)?;
            point_features = point_features.broadcast_add(&pool_proj)?;
        }
        if let Some(points_pos_enc_project) = &self.points_pos_enc_project {
            let pos_enc = encode_points_position(&points_xy, self.config.d_model)?;
            debug::capture_tensor("geometry/point_pos_enc", &pos_enc)?;
            let pos_enc_proj = points_pos_enc_project.forward(&pos_enc)?;
            debug::capture_tensor("geometry/point_pos_enc_proj", &pos_enc_proj)?;
            point_features = point_features.broadcast_add(&pos_enc_proj)?;
        }
        debug::capture_tensor("geometry/point_features", &point_features)?;
        Ok((
            point_features,
            Tensor::zeros((batch_size, seq_len), DType::U8, points_xy.device())?,
        ))
    }

    fn encode_boxes(
        &self,
        prompt: &GeometryPrompt,
        batch_size: usize,
        vision_feats: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let boxes_cxcywh = normalize_prompt_coords(
            prompt
                .boxes_cxcywh
                .as_ref()
                .expect("boxes were checked before encoding"),
            batch_size,
            4,
        )?;
        let seq_len = boxes_cxcywh.dim(0)?;
        let box_labels = normalize_prompt_labels(
            prompt.box_labels.as_ref(),
            batch_size,
            seq_len,
            boxes_cxcywh.device(),
        )?;
        let label_embed = self.label_embed.forward(&box_labels)?;
        eprintln!("[PHASE2] label_embed shape: {:?}", label_embed.dims());
        let debug_idx = debug_probe_index(self.config.d_model);
        let label_val = label_embed.i((0, 0, debug_idx))?.to_scalar::<f32>()?;
        eprintln!("[PHASE2] label_embed[0,0,{debug_idx}] = {:.6}", label_val);
        debug::capture_tensor("geometry/label_embed", &label_embed)?;
        let mut box_features = label_embed.clone();

        if let Some(boxes_direct_project) = &self.boxes_direct_project {
            let direct_proj = boxes_direct_project.forward(&boxes_cxcywh)?;
            eprintln!("[PHASE2] direct_proj shape: {:?}", direct_proj.dims());
            let direct_val = direct_proj.i((0, 0, debug_idx))?.to_scalar::<f32>()?;
            eprintln!("[PHASE2] direct_proj[0,0,{debug_idx}] = {:.6}", direct_val);
            debug::capture_tensor("geometry/direct_proj", &direct_proj)?;
            box_features = box_features.broadcast_add(&direct_proj)?;
            let combined_val = box_features.i((0, 0, debug_idx))?.to_scalar::<f32>()?;
            eprintln!(
                "[PHASE2] After adding direct_proj: box_features[0,0,{debug_idx}] = {:.6}",
                combined_val
            );
        }
        if let Some(boxes_pool_project) = &self.boxes_pool_project {
            let pooled_boxes =
                sample_boxes_nearest(vision_feats, &boxes_cxcywh, self.config.roi_size)?;
            debug::capture_tensor("geometry/pooled_boxes_raw", &pooled_boxes)?;
            let pooled_boxes = boxes_pool_project.forward(&pooled_boxes)?;
            let pooled_boxes = pooled_boxes.reshape((seq_len, batch_size, self.config.d_model))?;
            eprintln!("[PHASE2] pool_proj shape: {:?}", pooled_boxes.dims());
            let pool_val = pooled_boxes.i((0, 0, debug_idx))?.to_scalar::<f32>()?;
            eprintln!("[PHASE2] pool_proj[0,0,{debug_idx}] = {:.6}", pool_val);
            debug::capture_tensor("geometry/pool_proj", &pooled_boxes)?;
            box_features = box_features.broadcast_add(&pooled_boxes)?;
            let combined_val = box_features.i((0, 0, debug_idx))?.to_scalar::<f32>()?;
            eprintln!(
                "[PHASE2] After adding pool_proj: box_features[0,0,{debug_idx}] = {:.6}",
                combined_val
            );
        }
        if let Some(boxes_pos_enc_project) = &self.boxes_pos_enc_project {
            let pos_enc = encode_boxes_position(&boxes_cxcywh, self.config.d_model)?;
            eprintln!("[PHASE2] pos_enc shape: {:?}", pos_enc.dims());
            let pos_enc_proj = boxes_pos_enc_project.forward(&pos_enc)?;
            eprintln!("[PHASE2] pos_enc_proj shape: {:?}", pos_enc_proj.dims());
            let pos_enc_val = pos_enc_proj.i((0, 0, debug_idx))?.to_scalar::<f32>()?;
            eprintln!(
                "[PHASE2] pos_enc_proj[0,0,{debug_idx}] = {:.6}",
                pos_enc_val
            );
            debug::capture_tensor("geometry/pos_enc_proj", &pos_enc_proj)?;
            box_features = box_features.broadcast_add(&pos_enc_proj)?;
            let combined_val = box_features.i((0, 0, debug_idx))?.to_scalar::<f32>()?;
            eprintln!(
                "[PHASE2] After adding pos_enc_proj: box_features[0,0,{debug_idx}] = {:.6}",
                combined_val
            );
        }

        eprintln!(
            "[PHASE2] FINAL box_features shape: {:?}",
            box_features.dims()
        );
        let final_val = box_features.i((0, 0, debug_idx))?.to_scalar::<f32>()?;
        eprintln!(
            "[PHASE2] FINAL box_features[0,0,{debug_idx}] = {:.6}",
            final_val
        );
        debug::capture_tensor("geometry/box_features", &box_features)?;

        Ok((
            box_features,
            Tensor::zeros((batch_size, seq_len), DType::U8, boxes_cxcywh.device())?,
        ))
    }
}

fn normalize_prompt_coords(tensor: &Tensor, batch_size: usize, last_dim: usize) -> Result<Tensor> {
    match tensor.rank() {
        2 => {
            let dims = tensor.dims2()?;
            if batch_size != 1 || dims.1 != last_dim {
                candle::bail!(
                    "sam3 geometry expected 2D prompt coords with shape [num_items, {last_dim}] for batch size 1, got {dims:?}"
                )
            }
            tensor.reshape((dims.0, 1, last_dim))
        }
        3 => {
            let dims = tensor.dims3()?;
            if dims.2 != last_dim {
                candle::bail!(
                    "sam3 geometry expected trailing prompt coord dimension {last_dim}, got {dims:?}"
                )
            }
            if dims.0 == batch_size {
                tensor.transpose(0, 1)?.contiguous()
            } else if dims.1 == batch_size {
                Ok(tensor.clone())
            } else {
                candle::bail!(
                    "sam3 geometry prompt coords could not infer batch axis from shape {dims:?} and batch size {batch_size}"
                )
            }
        }
        rank => {
            candle::bail!("sam3 geometry expected rank-2 or rank-3 prompt coords, got rank {rank}")
        }
    }
}

fn normalize_prompt_labels(
    labels: Option<&Tensor>,
    batch_size: usize,
    seq_len: usize,
    device: &candle::Device,
) -> Result<Tensor> {
    let Some(labels) = labels else {
        return Tensor::ones((seq_len, batch_size), DType::U32, device);
    };
    match labels.rank() {
        1 => {
            let dims = labels.dims1()?;
            if batch_size != 1 || dims != seq_len {
                candle::bail!(
                    "sam3 geometry expected 1D labels with shape [{seq_len}] for batch size 1, got {dims}"
                )
            }
            labels.reshape((seq_len, 1))?.to_dtype(DType::U32)
        }
        2 => {
            let dims = labels.dims2()?;
            let labels = if dims.0 == batch_size && dims.1 == seq_len {
                labels.transpose(0, 1)?.contiguous()?
            } else if dims.0 == seq_len && dims.1 == batch_size {
                labels.clone()
            } else {
                candle::bail!(
                    "sam3 geometry labels could not infer sequence/batch axes from shape {dims:?}, expected [{batch_size}, {seq_len}] or [{seq_len}, {batch_size}]"
                )
            };
            labels.to_dtype(DType::U32)
        }
        rank => {
            candle::bail!("sam3 geometry expected rank-1 or rank-2 prompt labels, got rank {rank}")
        }
    }
}

fn concat_seq_tensors(parts: &[Tensor]) -> Result<Tensor> {
    match parts.len() {
        0 => candle::bail!("sam3 geometry encode produced no prompt tokens"),
        1 => Ok(parts[0].clone()),
        _ => {
            let refs = parts.iter().collect::<Vec<_>>();
            Tensor::cat(&refs, 0)
        }
    }
}

fn concat_batch_tensors(parts: &[Tensor]) -> Result<Tensor> {
    match parts.len() {
        0 => candle::bail!("sam3 geometry encode produced no prompt masks"),
        1 => Ok(parts[0].clone()),
        _ => {
            let refs = parts.iter().collect::<Vec<_>>();
            Tensor::cat(&refs, 1)
        }
    }
}

fn encode_points_position(points_xy: &Tensor, d_model: usize) -> Result<Tensor> {
    if d_model % 2 != 0 {
        candle::bail!("sam3 geometry point position encoding requires even d_model, got {d_model}")
    }
    let batch_first = points_xy.transpose(0, 1)?.contiguous()?;
    let coords = batch_first.to_vec3::<f32>()?;
    let num_pos_feats = d_model / 2;
    let total = coords.len() * coords[0].len();
    let mut pos_y = vec![0f32; total * num_pos_feats];
    let mut pos_x = vec![0f32; total * num_pos_feats];
    for (batch_idx, batch) in coords.iter().enumerate() {
        for (seq_idx, point) in batch.iter().enumerate() {
            let flat_idx = batch_idx * batch.len() + seq_idx;
            encode_1d_position(
                point[1],
                num_pos_feats,
                &mut pos_y[flat_idx * num_pos_feats..(flat_idx + 1) * num_pos_feats],
            );
            encode_1d_position(
                point[0],
                num_pos_feats,
                &mut pos_x[flat_idx * num_pos_feats..(flat_idx + 1) * num_pos_feats],
            );
        }
    }
    let device = points_xy.device();
    let pos_y = Tensor::from_slice(
        &pos_y,
        (coords.len(), coords[0].len(), num_pos_feats),
        device,
    )?;
    let pos_x = Tensor::from_slice(
        &pos_x,
        (coords.len(), coords[0].len(), num_pos_feats),
        device,
    )?;
    Tensor::cat(&[&pos_x, &pos_y], 2)?
        .transpose(0, 1)?
        .contiguous()?
        .to_dtype(points_xy.dtype())
}

fn encode_boxes_position(boxes_cxcywh: &Tensor, d_model: usize) -> Result<Tensor> {
    if d_model % 2 != 0 {
        candle::bail!("sam3 geometry box position encoding requires even d_model, got {d_model}")
    }
    let batch_first = boxes_cxcywh.transpose(0, 1)?.contiguous()?;
    let boxes = batch_first.to_vec3::<f32>()?;
    let num_pos_feats = d_model / 2;

    // DEBUG: Print input box
    if !boxes.is_empty() && !boxes[0].is_empty() {
        eprintln!(
            "[encode_boxes_position] boxes[0][0] = [{}, {}, {}, {}] (cx, cy, w, h)",
            boxes[0][0][0], boxes[0][0][1], boxes[0][0][2], boxes[0][0][3]
        );
    }

    let total = boxes.len() * boxes[0].len();
    let mut pos_y = vec![0f32; total * num_pos_feats];
    let mut pos_x = vec![0f32; total * num_pos_feats];
    let mut hw = vec![0f32; total * 2];
    for (batch_idx, batch) in boxes.iter().enumerate() {
        for (seq_idx, box_coords) in batch.iter().enumerate() {
            let flat_idx = batch_idx * batch.len() + seq_idx;
            encode_1d_position(
                box_coords[1],
                num_pos_feats,
                &mut pos_y[flat_idx * num_pos_feats..(flat_idx + 1) * num_pos_feats],
            );
            encode_1d_position(
                box_coords[0],
                num_pos_feats,
                &mut pos_x[flat_idx * num_pos_feats..(flat_idx + 1) * num_pos_feats],
            );
            hw[flat_idx * 2] = box_coords[3];
            hw[flat_idx * 2 + 1] = box_coords[2];
        }
    }

    // DEBUG: Print specific position encoding values
    if total > 0 {
        let debug_idx = debug_probe_index(num_pos_feats);
        eprintln!(
            "[encode_boxes_position] pos_y[{debug_idx}]={}, pos_x[{debug_idx}]={}",
            pos_y.get(debug_idx).copied().unwrap_or(-999.0),
            pos_x.get(debug_idx).copied().unwrap_or(-999.0)
        );
    }

    let device = boxes_cxcywh.device();
    let pos_y = Tensor::from_slice(&pos_y, (boxes.len(), boxes[0].len(), num_pos_feats), device)?;
    let pos_x = Tensor::from_slice(&pos_x, (boxes.len(), boxes[0].len(), num_pos_feats), device)?;
    let hw = Tensor::from_slice(&hw, (boxes.len(), boxes[0].len(), 2), device)?;

    // DEBUG: Print tensor shapes and first values
    eprintln!(
        "[encode_boxes_position] pos_y shape: {:?}, pos_x shape: {:?}",
        pos_y.dims(),
        pos_x.dims()
    );

    let result = Tensor::cat(&[&pos_y, &pos_x, &hw], 2)?
        .transpose(0, 1)?
        .contiguous()?
        .to_dtype(boxes_cxcywh.dtype())?;

    // DEBUG: Print final shape
    eprintln!(
        "[encode_boxes_position] Final output shape: {:?}",
        result.dims()
    );

    Ok(result)
}

fn encode_1d_position(coord: f32, num_pos_feats: usize, out: &mut [f32]) {
    let temperature = 10_000f32;
    let coord_scaled = coord * 2.0 * PI;

    // DEBUG: Print first few and specific indices
    let debug_this = coord > 0.64 && coord < 0.66; // cy=0.653 range
    if debug_this {
        eprintln!(
            "[encode_1d] input_coord={}, scaled_coord={}, num_pos_feats={}",
            coord, coord_scaled, num_pos_feats
        );
    }

    for idx in 0..num_pos_feats {
        let exponent = 2.0 * (idx / 2) as f32 / num_pos_feats as f32;
        let angle = coord_scaled / temperature.powf(exponent);
        out[idx] = if idx % 2 == 0 {
            angle.sin()
        } else {
            angle.cos()
        };

        // DEBUG: Print specific indices
        if debug_this && (idx == 26 || idx < 3) {
            eprintln!(
                "[encode_1d] idx={}, exponent={}, angle={}, out[{}]={}",
                idx, exponent, angle, idx, out[idx]
            );
        }
    }
}

fn sample_points_nearest(vision_feats: &Tensor, points_xy: &Tensor) -> Result<Tensor> {
    let (batch_size, channels, height, width) = vision_feats.dims4()?;
    let points = points_xy.to_vec3::<f32>()?;
    let seq_len = points.len();
    let mut samples = Vec::with_capacity(batch_size * seq_len * channels);

    for seq_points in points.iter() {
        for (batch_idx, point) in seq_points.iter().enumerate() {
            // Convert from normalized [0, 1] to pixel coordinates
            let x_pixel = point[0] * width as f32;
            let y_pixel = point[1] * height as f32;

            // Perform bilinear interpolation
            let sample = bilinear_sample_zero_padded(
                vision_feats,
                batch_idx,
                x_pixel,
                y_pixel,
                width,
                height,
                channels,
            )?;
            samples.push(sample);
        }
    }

    let sample_refs = samples.iter().collect::<Vec<_>>();
    Tensor::stack(&sample_refs, 0)?.reshape((seq_len, batch_size, channels))
}

#[derive(Debug, Clone, Copy)]
struct RoiAlignConfig {
    spatial_scale: f32,
    sampling_ratio: i32,
    aligned: bool,
}

#[derive(Debug, Clone, Copy)]
struct RoiAlignBox {
    roi_start_w: f32,
    roi_start_h: f32,
    roi_end_w: f32,
    roi_end_h: f32,
    roi_width: f32,
    roi_height: f32,
    bin_size_w: f32,
    bin_size_h: f32,
    grid_w: usize,
    grid_h: usize,
}

fn roi_align_box(
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    pooled_size: usize,
    config: RoiAlignConfig,
) -> RoiAlignBox {
    let offset = if config.aligned { 0.5 } else { 0.0 };
    let roi_start_w = x0 * config.spatial_scale - offset;
    let roi_start_h = y0 * config.spatial_scale - offset;
    let roi_end_w = x1 * config.spatial_scale - offset;
    let roi_end_h = y1 * config.spatial_scale - offset;

    let mut roi_width = roi_end_w - roi_start_w;
    let mut roi_height = roi_end_h - roi_start_h;
    if !config.aligned {
        roi_width = roi_width.max(1.0);
        roi_height = roi_height.max(1.0);
    }

    let bin_size_w = roi_width / pooled_size as f32;
    let bin_size_h = roi_height / pooled_size as f32;
    let grid_h = if config.sampling_ratio > 0 {
        config.sampling_ratio as usize
    } else {
        ((roi_height / pooled_size as f32).ceil() as usize).max(1)
    };
    let grid_w = if config.sampling_ratio > 0 {
        config.sampling_ratio as usize
    } else {
        ((roi_width / pooled_size as f32).ceil() as usize).max(1)
    };

    RoiAlignBox {
        roi_start_w,
        roi_start_h,
        roi_end_w,
        roi_end_h,
        roi_width,
        roi_height,
        bin_size_w,
        bin_size_h,
        grid_w,
        grid_h,
    }
}

fn box_to_feature_xyxy(
    cx: f32,
    cy: f32,
    w: f32,
    h: f32,
    width: usize,
    height: usize,
) -> (f32, f32, f32, f32) {
    let (x0, y0, x1, y1) = cxcywh_to_xyxy(cx, cy, w, h);
    (
        x0 * width as f32,
        y0 * height as f32,
        x1 * width as f32,
        y1 * height as f32,
    )
}

fn roi_align_sample_coord(
    roi_start: f32,
    bin_size: f32,
    pooled_idx: usize,
    grid_idx: usize,
    grid_size: usize,
) -> f32 {
    roi_start
        + pooled_idx as f32 * bin_size
        + (grid_idx as f32 + 0.5) * (bin_size / grid_size as f32)
}

fn bilinear_sample_border_clamped(
    vision_feats: &Tensor,
    batch_idx: usize,
    x_pixel: f32,
    y_pixel: f32,
    width: usize,
    height: usize,
    channels: usize,
) -> Result<Tensor> {
    let x = x_pixel.max(0.0);
    let y = y_pixel.max(0.0);

    let x_low_raw = x.floor() as i32;
    let y_low_raw = y.floor() as i32;
    let x_at_boundary = x_low_raw >= width as i32 - 1;
    let y_at_boundary = y_low_raw >= height as i32 - 1;

    let x_low = if x_at_boundary {
        width as i32 - 1
    } else {
        x_low_raw
    };
    let y_low = if y_at_boundary {
        height as i32 - 1
    } else {
        y_low_raw
    };
    let x_high = if x_at_boundary { x_low } else { x_low + 1 };
    let y_high = if y_at_boundary { y_low } else { y_low + 1 };

    let lx = x - x_low as f32;
    let ly = y - y_low as f32;
    let hx = 1.0 - lx;
    let hy = 1.0 - ly;

    let load = |xi: i32, yi: i32| -> Result<Vec<f32>> {
        vision_feats
            .i((batch_idx, .., yi as usize, xi as usize))?
            .to_vec1::<f32>()
    };

    let v1 = load(x_low, y_low)?;
    let v2 = load(x_high, y_low)?;
    let v3 = load(x_low, y_high)?;
    let v4 = load(x_high, y_high)?;

    let mut result = vec![0.0f32; channels];
    for c in 0..channels {
        result[c] = hy * hx * v1[c] + hy * lx * v2[c] + ly * hx * v3[c] + ly * lx * v4[c];
    }

    let device = vision_feats.device();
    Tensor::from_vec(result, (channels,), device)?.to_dtype(vision_feats.dtype())
}

fn sample_boxes_nearest(
    vision_feats: &Tensor,
    boxes_cxcywh: &Tensor,
    roi_size: usize,
) -> Result<Tensor> {
    let (batch_size, channels, height, width) = vision_feats.dims4()?;
    let boxes = boxes_cxcywh.to_vec3::<f32>()?;
    let seq_len = boxes.len();
    let mut patches = Vec::with_capacity(batch_size * seq_len);

    let config = RoiAlignConfig {
        spatial_scale: 1.0,
        sampling_ratio: -1,
        aligned: false,
    };

    eprintln!(
        "[ROI_DEBUG] Input: batch_size={}, channels={}, height={}, width={}",
        batch_size, channels, height, width
    );
    eprintln!(
        "[ROI_DEBUG] roi_size={}, sampling_ratio={}, aligned={}",
        roi_size, config.sampling_ratio, config.aligned
    );

    for seq_boxes in boxes.iter() {
        for (batch_idx, box_coords) in seq_boxes.iter().enumerate() {
            let (x0, y0, x1, y1) = box_to_feature_xyxy(
                box_coords[0],
                box_coords[1],
                box_coords[2],
                box_coords[3],
                width,
                height,
            );
            let roi = roi_align_box(x0, y0, x1, y1, roi_size, config);
            eprintln!(
                "[ROI_DEBUG] Processing box[{}]: cx={:.6}, cy={:.6}, w={:.6}, h={:.6}",
                batch_idx, box_coords[0], box_coords[1], box_coords[2], box_coords[3]
            );
            eprintln!(
                "[ROI_DEBUG] Feature-space box: x0={:.6}, y0={:.6}, x1={:.6}, y1={:.6}",
                x0, y0, x1, y1
            );
            eprintln!(
                "[ROI_DEBUG] ROI region: start_w={:.6}, start_h={:.6}, end_w={:.6}, end_h={:.6}",
                roi.roi_start_w, roi.roi_start_h, roi.roi_end_w, roi.roi_end_h
            );
            eprintln!(
                "[ROI_DEBUG] ROI size: width={:.6}, height={:.6}",
                roi.roi_width, roi.roi_height
            );
            eprintln!(
                "[ROI_DEBUG] Bin sizes: w={:.6}, h={:.6}, grid_w={}, grid_h={}",
                roi.bin_size_w, roi.bin_size_h, roi.grid_w, roi.grid_h
            );

            let mut rows = Vec::with_capacity(roi_size);
            for roi_y in 0..roi_size {
                let mut cols = Vec::with_capacity(roi_size);
                for roi_x in 0..roi_size {
                    // Accumulate samples for this output pixel using multiple sample points
                    let mut accumulated = vec![0.0f32; channels];
                    let total_samples = (roi.grid_h * roi.grid_w) as f32;

                    // Sample at multiple points within this bin (like roi_align adaptive sampling)
                    for iy in 0..roi.grid_h {
                        for ix in 0..roi.grid_w {
                            // Calculate sampling point coordinates
                            let sample_y = roi_align_sample_coord(
                                roi.roi_start_h,
                                roi.bin_size_h,
                                roi_y,
                                iy,
                                roi.grid_h,
                            );
                            let sample_x = roi_align_sample_coord(
                                roi.roi_start_w,
                                roi.bin_size_w,
                                roi_x,
                                ix,
                                roi.grid_w,
                            );

                            if roi_y == 0 && roi_x == 0 && iy == 0 && ix == 0 {
                                eprintln!(
                                    "[ROI_DEBUG] Sample[0,0]: pixel_x={:.6}, pixel_y={:.6}",
                                    sample_x, sample_y
                                );
                            }

                            let sample = bilinear_sample_border_clamped(
                                vision_feats,
                                batch_idx,
                                sample_x,
                                sample_y,
                                width,
                                height,
                                channels,
                            )?;
                            let sample_vec = sample.to_vec1::<f32>()?;

                            if roi_y == 0 && roi_x == 0 && iy == 0 && ix == 0 {
                                let debug_idx = debug_probe_index(channels);
                                eprintln!(
                                    "[ROI_DEBUG] Sample[0,0] value[{debug_idx}]={:.6}",
                                    sample_vec[debug_idx]
                                );
                            }

                            for c in 0..channels {
                                accumulated[c] += sample_vec[c] / total_samples;
                            }
                        }
                    }

                    if roi_y == 0 && roi_x == 0 {
                        let debug_idx = debug_probe_index(channels);
                        eprintln!(
                            "[ROI_DEBUG] Output[0,0] accumulated[{debug_idx}]={:.6}",
                            accumulated[debug_idx]
                        );
                    }

                    let device = vision_feats.device();
                    let col_tensor = Tensor::from_vec(accumulated, (channels,), device)?;
                    cols.push(col_tensor);
                }
                let col_refs = cols.iter().collect::<Vec<_>>();
                rows.push(Tensor::stack(&col_refs, 1)?);
            }
            let row_refs = rows.iter().collect::<Vec<_>>();
            let patch = Tensor::stack(&row_refs, 1)?;

            // Debug final patch value
            let debug_idx = debug_probe_index(channels);
            let patch_val = patch.i((debug_idx, 0, 0))?.to_scalar::<f32>()?;
            eprintln!("[ROI_DEBUG] Final patch[{debug_idx},0,0]={:.6}", patch_val);

            patches.push(patch);
        }
    }

    let patch_refs = patches.iter().collect::<Vec<_>>();
    Tensor::stack(&patch_refs, 0)?.reshape((seq_len * batch_size, channels, roi_size, roi_size))
}

fn normalized_to_index(coord: f32, size: usize) -> usize {
    let max_idx = size.saturating_sub(1) as f32;
    (coord * size as f32).clamp(0.0, max_idx).round() as usize
}

fn debug_probe_index(width: usize) -> usize {
    width.saturating_sub(1).min(26)
}

fn bilinear_sample_zero_padded(
    vision_feats: &Tensor,
    batch_idx: usize,
    x_pixel: f32,
    y_pixel: f32,
    width: usize,
    height: usize,
    channels: usize,
) -> Result<Tensor> {
    // Implement bilinear interpolation matching PyTorch grid_sample with align_corners=False
    // For normalized [0, 1] coordinates: pixel = norm_coord * size - 0.5

    let x = x_pixel - 0.5;
    let y = y_pixel - 0.5;

    // Get floor and ceil integer coordinates
    let x0_i32 = x.floor() as i32;
    let y0_i32 = y.floor() as i32;
    let x1_i32 = x0_i32 + 1;
    let y1_i32 = y0_i32 + 1;

    // Get interpolation weights
    let fx = x - x0_i32 as f32;
    let fy = y - y0_i32 as f32;
    let fx = fx.max(0.0).min(1.0);
    let fy = fy.max(0.0).min(1.0);
    let fx_inv = 1.0 - fx;
    let fy_inv = 1.0 - fy;

    if x_pixel < 10.0 && y_pixel < 10.0 {
        // Debug only for reasonable pixel coords
        eprintln!("[BILINEAR] pixel_x={:.6}, pixel_y={:.6}", x_pixel, y_pixel);
        eprintln!(
            "[BILINEAR] x={:.6}, y={:.6}, x0={}, y0={}, x1={}, y1={}",
            x, y, x0_i32, y0_i32, x1_i32, y1_i32
        );
        eprintln!("[BILINEAR] fx={:.6}, fy={:.6}", fx, fy);
    }

    // Helper to safely get values with out-of-bounds handling (zero padding)
    let get_value = |xi: i32, yi: i32| -> Result<Vec<f32>> {
        if xi < 0 || xi >= width as i32 || yi < 0 || yi >= height as i32 {
            // Out of bounds: return zeros
            Ok(vec![0.0f32; channels])
        } else {
            vision_feats
                .i((batch_idx, .., yi as usize, xi as usize))?
                .to_vec1::<f32>()
        }
    };

    // Get the four neighboring values
    let v00 = get_value(x0_i32, y0_i32)?;
    let v01 = get_value(x1_i32, y0_i32)?;
    let v10 = get_value(x0_i32, y1_i32)?;
    let v11 = get_value(x1_i32, y1_i32)?;

    // Perform bilinear interpolation
    let w00 = fx_inv * fy_inv;
    let w01 = fx * fy_inv;
    let w10 = fx_inv * fy;
    let w11 = fx * fy;

    let mut result = vec![0.0f32; channels];
    for c in 0..channels {
        result[c] = v00[c] * w00 + v01[c] * w01 + v10[c] * w10 + v11[c] * w11;
    }

    let device = vision_feats.device();
    let result_tensor = Tensor::from_vec(result, (channels,), device)?;
    result_tensor.to_dtype(vision_feats.dtype())
}

fn cxcywh_to_xyxy(cx: f32, cy: f32, w: f32, h: f32) -> (f32, f32, f32, f32) {
    let half_w = w * 0.5;
    let half_h = h * 0.5;
    (cx - half_w, cy - half_h, cx + half_w, cy + half_h)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use candle::{DType, Device, Result, Tensor};
    use candle_nn::VarBuilder;
    use serde::Deserialize;

    use super::{
        encode_boxes_position, encode_points_position, sample_boxes_nearest, sample_points_nearest,
        GeometryPrompt, SequenceGeometryEncoder,
    };
    use crate::models::sam3::debug::{self, DebugExporter};
    use crate::models::sam3::GeometryConfig;

    #[test]
    fn geometry_encoder_returns_cls_token_for_empty_prompt() -> Result<()> {
        let device = Device::Cpu;
        let config = test_config();
        let vb = VarBuilder::from_tensors(geometry_weights(&config, &device)?, DType::F32, &device);
        let encoder = SequenceGeometryEncoder::new(&config, vb)?;
        let prompt = GeometryPrompt::default();
        let image_features = vec![Tensor::zeros(
            (1, config.d_model, 2, 2),
            DType::F32,
            &device,
        )?];
        let image_pos = vec![Tensor::zeros(
            (1, config.d_model, 2, 2),
            DType::F32,
            &device,
        )?];
        let encoded = encoder.encode(&prompt, &image_features, &image_pos)?;
        assert_eq!(encoded.features.dims3()?, (1, 1, config.d_model));
        assert_eq!(encoded.padding_mask.dims2()?, (1, 1));
        assert_eq!(encoded.padding_mask.to_vec2::<u8>()?, vec![vec![0]]);
        Ok(())
    }

    #[test]
    fn geometry_encoder_encodes_points_and_boxes() -> Result<()> {
        let device = Device::Cpu;
        let config = test_config();
        let vb = VarBuilder::from_tensors(geometry_weights(&config, &device)?, DType::F32, &device);
        let encoder = SequenceGeometryEncoder::new(&config, vb)?;
        let prompt = GeometryPrompt {
            points_xy: Some(Tensor::from_slice(
                &[0.25f32, 0.25, 0.75, 0.75],
                (1, 2, 2),
                &device,
            )?),
            boxes_cxcywh: Some(Tensor::from_slice(
                &[0.5f32, 0.5, 0.25, 0.25],
                (1, 1, 4),
                &device,
            )?),
            ..Default::default()
        };
        let image_features = vec![Tensor::zeros(
            (1, config.d_model, 2, 2),
            DType::F32,
            &device,
        )?];
        let image_pos = vec![Tensor::zeros(
            (1, config.d_model, 2, 2),
            DType::F32,
            &device,
        )?];
        let encoded = encoder.encode(&prompt, &image_features, &image_pos)?;
        assert_eq!(encoded.features.dims3()?, (4, 1, config.d_model));
        assert_eq!(
            encoded.padding_mask.to_vec2::<u8>()?,
            vec![vec![0, 0, 0, 0]]
        );
        Ok(())
    }

    #[test]
    fn geometry_encoder_rejects_mask_prompts_without_mask_encoder() -> Result<()> {
        let device = Device::Cpu;
        let config = test_config();
        let vb = VarBuilder::from_tensors(geometry_weights(&config, &device)?, DType::F32, &device);
        let encoder = SequenceGeometryEncoder::new(&config, vb)?;
        let prompt = GeometryPrompt {
            masks: Some(Tensor::zeros((1, 1, 4, 4), DType::F32, &device)?),
            ..Default::default()
        };
        let image_features = vec![Tensor::zeros(
            (1, config.d_model, 2, 2),
            DType::F32,
            &device,
        )?];
        let image_pos = vec![Tensor::zeros(
            (1, config.d_model, 2, 2),
            DType::F32,
            &device,
        )?];
        let err = encoder
            .encode(&prompt, &image_features, &image_pos)
            .unwrap_err();
        assert!(err.to_string().contains("mask prompts"));
        Ok(())
    }

    #[derive(Debug, Deserialize)]
    struct GeometryFixtureMetadata {
        d_model: usize,
        num_heads: usize,
        dim_feedforward: usize,
        num_layers: usize,
        roi_size: usize,
    }

    #[test]
    fn geometry_fixture_smoke_final_matches_upstream() -> Result<()> {
        let encoded = run_fixture_geometry_forward()?;
        let expected = load_geometry_fixture_tensors("fixture.safetensors")?;
        assert_tensor_close(
            &encoded.features,
            fixture_tensor(&expected, "geometry/returned_features")?,
            1e-5,
            "geometry/returned_features",
        )?;
        assert_tensor_close(
            &encoded.padding_mask.to_dtype(DType::U8)?,
            fixture_tensor(&expected, "geometry/padding_mask")?,
            0.0,
            "geometry/padding_mask",
        )?;
        Ok(())
    }

    #[test]
    #[ignore = "fixture-driven parity investigation"]
    fn geometry_fixture_point_helpers_match_upstream() -> Result<()> {
        let fixture = load_geometry_fixture_tensors("fixture.safetensors")?;
        let points_xy = fixture_tensor(&fixture, "inputs/points_xy")?;
        let pool_image_features = fixture_tensor(&fixture, "inputs/pool_image_features")?;
        assert_tensor_close(
            &encode_points_position(points_xy, fixture_metadata()?.d_model)?,
            fixture_tensor(&fixture, "helper/points_position")?,
            1e-5,
            "helper/points_position",
        )?;
        assert_tensor_close(
            &sample_points_nearest(pool_image_features, points_xy)?,
            fixture_tensor(&fixture, "helper/points_sampled")?,
            1e-5,
            "helper/points_sampled",
        )?;
        Ok(())
    }

    #[test]
    #[ignore = "fixture-driven parity investigation"]
    fn geometry_fixture_box_helpers_match_upstream() -> Result<()> {
        let metadata = fixture_metadata()?;
        let fixture = load_geometry_fixture_tensors("fixture.safetensors")?;
        let boxes = fixture_tensor(&fixture, "inputs/boxes_cxcywh")?;
        let pool_image_features = fixture_tensor(&fixture, "inputs/pool_image_features")?;
        assert_tensor_close(
            &encode_boxes_position(boxes, metadata.d_model)?,
            fixture_tensor(&fixture, "helper/boxes_position")?,
            1e-5,
            "helper/boxes_position",
        )?;
        assert_tensor_close(
            &sample_boxes_nearest(pool_image_features, boxes, metadata.roi_size)?,
            fixture_tensor(&fixture, "helper/boxes_sampled_raw")?,
            1e-5,
            "helper/boxes_sampled_raw",
        )?;
        Ok(())
    }

    #[test]
    #[ignore = "fixture-driven parity investigation"]
    fn geometry_fixture_box_feature_composition_matches_upstream() -> Result<()> {
        let (_encoded, debug_tensors) = run_fixture_geometry_encode()?;
        let expected = load_geometry_fixture_tensors("fixture.safetensors")?;
        let keys = [
            "geometry/label_embed",
            "geometry/direct_proj",
            "geometry/pooled_boxes_raw",
            "geometry/pool_proj",
            "geometry/pos_enc_proj",
            "geometry/box_features",
        ];
        assert_debug_keys_close(&debug_tensors, &expected, &keys, 1e-5)
    }

    #[test]
    #[ignore = "fixture-driven parity investigation"]
    fn geometry_fixture_mini_encoder_matches_upstream() -> Result<()> {
        let (encoded, debug_tensors) = run_fixture_geometry_encode()?;
        let expected = load_geometry_fixture_tensors("fixture.safetensors")?;
        let keys = [
            "geometry/features_initial_norm",
            "geometry/features_after_layer_0",
            "geometry/features_final",
        ];
        assert_debug_keys_close(&debug_tensors, &expected, &keys, 1e-5)?;
        assert_tensor_close(
            &encoded.features,
            fixture_tensor(&expected, "geometry/returned_features")?,
            1e-5,
            "geometry/returned_features",
        )?;
        assert_tensor_close(
            &encoded.padding_mask.to_dtype(DType::U8)?,
            fixture_tensor(&expected, "geometry/padding_mask")?,
            0.0,
            "geometry/padding_mask",
        )?;
        Ok(())
    }

    #[test]
    #[ignore = "fixture-driven parity investigation"]
    fn interactive_geometry_fixture_point_helpers_match_upstream() -> Result<()> {
        let fixture = load_interactive_geometry_fixture_tensors("fixture.safetensors")?;
        let points_xy = fixture_tensor(&fixture, "inputs/points_xy")?;
        let pool_image_features = fixture_tensor(&fixture, "inputs/pool_image_features")?;
        assert_tensor_close(
            &encode_points_position(points_xy, interactive_fixture_metadata()?.d_model)?,
            fixture_tensor(&fixture, "helper/points_position")?,
            1e-5,
            "helper/points_position",
        )?;
        assert_tensor_close(
            &sample_points_nearest(pool_image_features, points_xy)?,
            fixture_tensor(&fixture, "helper/points_sampled")?,
            1e-5,
            "helper/points_sampled",
        )?;
        Ok(())
    }

    #[test]
    #[ignore = "fixture-driven parity investigation"]
    fn interactive_geometry_fixture_point_feature_composition_matches_upstream() -> Result<()> {
        let (_encoded, debug_tensors) = run_interactive_fixture_geometry_encode()?;
        let expected = load_interactive_geometry_fixture_tensors("fixture.safetensors")?;
        let keys = [
            "geometry/point_label_embed",
            "geometry/point_direct_proj",
            "geometry/point_sampled_raw",
            "geometry/point_pool_proj",
            "geometry/point_pos_enc",
            "geometry/point_pos_enc_proj",
            "geometry/point_features",
        ];
        assert_debug_keys_close(&debug_tensors, &expected, &keys, 1e-5)
    }

    #[test]
    #[ignore = "fixture-driven parity investigation"]
    fn interactive_geometry_fixture_encoder_matches_upstream() -> Result<()> {
        let (encoded, debug_tensors) = run_interactive_fixture_geometry_encode()?;
        let expected = load_interactive_geometry_fixture_tensors("fixture.safetensors")?;
        let keys = [
            "geometry/features_initial_norm",
            "geometry/features_after_layer_0",
            "geometry/features_after_layer_1",
            "geometry/features_after_layer_2",
        ];
        assert_debug_keys_close(&debug_tensors, &expected, &keys, 1e-5)?;
        assert_tensor_close(
            &encoded.features,
            fixture_tensor(&expected, "geometry/returned_features")?,
            1e-5,
            "geometry/returned_features",
        )?;
        assert_tensor_close(
            &encoded.padding_mask.to_dtype(DType::U8)?,
            fixture_tensor(&expected, "geometry/padding_mask")?,
            0.0,
            "geometry/padding_mask",
        )?;
        Ok(())
    }

    fn test_config() -> GeometryConfig {
        GeometryConfig {
            d_model: 8,
            num_layers: 1,
            num_heads: 2,
            dim_feedforward: 16,
            roi_size: 1,
            add_cls: true,
            add_post_encode_proj: true,
        }
    }

    fn fixture_metadata() -> Result<GeometryFixtureMetadata> {
        let path = geometry_fixture_dir().join("metadata.json");
        let contents = fs::read_to_string(&path).map_err(|err| {
            candle::Error::Msg(format!(
                "failed to read geometry fixture metadata {}: {err}",
                path.display()
            ))
        })?;
        serde_json::from_str(&contents).map_err(|err| {
            candle::Error::Msg(format!(
                "failed to parse geometry fixture metadata {}: {err}",
                path.display()
            ))
        })
    }

    fn fixture_config() -> Result<GeometryConfig> {
        let metadata = fixture_metadata()?;
        Ok(GeometryConfig {
            d_model: metadata.d_model,
            num_layers: metadata.num_layers,
            num_heads: metadata.num_heads,
            dim_feedforward: metadata.dim_feedforward,
            roi_size: metadata.roi_size,
            add_cls: true,
            add_post_encode_proj: true,
        })
    }

    fn geometry_fixture_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/sam3_geometry_unit")
    }

    fn interactive_geometry_fixture_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/sam3_interactive_geometry_seed")
    }

    fn load_geometry_fixture_tensors(file_name: &str) -> Result<HashMap<String, Tensor>> {
        let path = geometry_fixture_dir().join(file_name);
        candle::safetensors::load(&path, &Device::Cpu).map_err(|err| {
            candle::Error::Msg(format!(
                "failed to load geometry fixture {}: {err}",
                path.display()
            ))
        })
    }

    fn load_interactive_geometry_fixture_tensors(
        file_name: &str,
    ) -> Result<HashMap<String, Tensor>> {
        let path = interactive_geometry_fixture_dir().join(file_name);
        candle::safetensors::load(&path, &Device::Cpu).map_err(|err| {
            candle::Error::Msg(format!(
                "failed to load interactive geometry fixture {}: {err}",
                path.display()
            ))
        })
    }

    fn interactive_fixture_metadata() -> Result<GeometryFixtureMetadata> {
        let path = interactive_geometry_fixture_dir().join("metadata.json");
        let contents = fs::read_to_string(&path).map_err(|err| {
            candle::Error::Msg(format!(
                "failed to read interactive geometry fixture metadata {}: {err}",
                path.display()
            ))
        })?;
        serde_json::from_str(&contents).map_err(|err| {
            candle::Error::Msg(format!(
                "failed to parse interactive geometry fixture metadata {}: {err}",
                path.display()
            ))
        })
    }

    fn interactive_fixture_config() -> Result<GeometryConfig> {
        let metadata = interactive_fixture_metadata()?;
        Ok(GeometryConfig {
            d_model: metadata.d_model,
            num_layers: metadata.num_layers,
            num_heads: metadata.num_heads,
            dim_feedforward: metadata.dim_feedforward,
            roi_size: metadata.roi_size,
            add_cls: true,
            add_post_encode_proj: true,
        })
    }

    fn fixture_tensor<'a>(fixture: &'a HashMap<String, Tensor>, key: &str) -> Result<&'a Tensor> {
        fixture.get(key).ok_or_else(|| {
            candle::Error::Msg(format!("geometry fixture is missing tensor `{key}`"))
        })
    }

    fn unique_temp_dir(label: &str) -> Result<PathBuf> {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|err| candle::Error::Msg(format!("system clock error: {err}")))?;
        let path = std::env::temp_dir().join(format!(
            "sam3_geometry_fixture_{label}_{}_{}",
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

    fn run_fixture_geometry_encode() -> Result<(super::EncodedPrompt, HashMap<String, Tensor>)> {
        let device = Device::Cpu;
        let config = fixture_config()?;
        let weights = load_geometry_fixture_tensors("weights.safetensors")?;
        let fixture = load_geometry_fixture_tensors("fixture.safetensors")?;
        let vb = VarBuilder::from_tensors(weights, DType::F32, &device);
        let encoder = SequenceGeometryEncoder::new(&config, vb)?;
        let prompt = GeometryPrompt {
            boxes_cxcywh: Some(fixture_tensor(&fixture, "inputs/boxes_cxcywh")?.clone()),
            box_labels: Some(fixture_tensor(&fixture, "inputs/box_labels")?.to_dtype(DType::U32)?),
            ..Default::default()
        };
        let image_features = vec![fixture_tensor(&fixture, "inputs/image_features")?.clone()];
        let image_pos = vec![fixture_tensor(&fixture, "inputs/image_pos_embeds")?.clone()];

        let debug_dir = unique_temp_dir("encode")?;
        debug::set_exporter(Some(DebugExporter::new(&debug_dir)?));
        let encoded = encoder.encode(&prompt, &image_features, &image_pos)?;
        debug::finish()?;
        let debug_tensors =
            candle::safetensors::load(debug_dir.join("debug_tensors.safetensors"), &device)
                .map_err(|err| {
                    candle::Error::Msg(format!(
                        "failed to load geometry debug tensors from {}: {err}",
                        debug_dir.display()
                    ))
                })?;
        let _ = fs::remove_dir_all(&debug_dir);
        Ok((encoded, debug_tensors))
    }

    fn run_fixture_geometry_forward() -> Result<super::EncodedPrompt> {
        let device = Device::Cpu;
        let config = fixture_config()?;
        let weights = load_geometry_fixture_tensors("weights.safetensors")?;
        let fixture = load_geometry_fixture_tensors("fixture.safetensors")?;
        let vb = VarBuilder::from_tensors(weights, DType::F32, &device);
        let encoder = SequenceGeometryEncoder::new(&config, vb)?;
        let prompt = GeometryPrompt {
            boxes_cxcywh: Some(fixture_tensor(&fixture, "inputs/boxes_cxcywh")?.clone()),
            box_labels: Some(fixture_tensor(&fixture, "inputs/box_labels")?.to_dtype(DType::U32)?),
            ..Default::default()
        };
        let image_features = vec![fixture_tensor(&fixture, "inputs/image_features")?.clone()];
        let image_pos = vec![fixture_tensor(&fixture, "inputs/image_pos_embeds")?.clone()];
        encoder.encode(&prompt, &image_features, &image_pos)
    }

    fn run_interactive_fixture_geometry_encode(
    ) -> Result<(super::EncodedPrompt, HashMap<String, Tensor>)> {
        let device = Device::Cpu;
        let config = interactive_fixture_config()?;
        let weights = load_interactive_geometry_fixture_tensors("weights.safetensors")?;
        let fixture = load_interactive_geometry_fixture_tensors("fixture.safetensors")?;
        let vb = VarBuilder::from_tensors(weights, DType::F32, &device);
        let encoder = SequenceGeometryEncoder::new(&config, vb)?;
        let prompt = GeometryPrompt {
            points_xy: Some(fixture_tensor(&fixture, "inputs/points_xy")?.clone()),
            point_labels: Some(
                fixture_tensor(&fixture, "inputs/point_labels")?.to_dtype(DType::U32)?,
            ),
            ..Default::default()
        };
        let image_features = vec![fixture_tensor(&fixture, "inputs/image_features")?.clone()];
        let image_pos = vec![fixture_tensor(&fixture, "inputs/image_pos_embeds")?.clone()];

        let debug_dir = unique_temp_dir("interactive_encode")?;
        debug::set_exporter(Some(DebugExporter::new(&debug_dir)?));
        let encoded = encoder.encode(&prompt, &image_features, &image_pos)?;
        debug::finish()?;
        let debug_tensors =
            candle::safetensors::load(debug_dir.join("debug_tensors.safetensors"), &device)
                .map_err(|err| {
                    candle::Error::Msg(format!(
                        "failed to load interactive geometry debug tensors from {}: {err}",
                        debug_dir.display()
                    ))
                })?;
        let _ = fs::remove_dir_all(&debug_dir);
        Ok((encoded, debug_tensors))
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
        if actual.dtype() == DType::U8 || expected.dtype() == DType::U8 {
            let actual = actual.to_dtype(DType::U8)?.flatten_all()?.to_vec1::<u8>()?;
            let expected = expected
                .to_dtype(DType::U8)?
                .flatten_all()?
                .to_vec1::<u8>()?;
            if actual != expected {
                candle::bail!("{name}: byte mismatch actual={actual:?} expected={expected:?}");
            }
            return Ok(());
        }

        let actual = actual
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let expected = expected
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let mut max_abs_diff = 0f32;
        let mut max_idx = 0usize;
        for (idx, (lhs, rhs)) in actual.iter().zip(expected.iter()).enumerate() {
            let diff = (lhs - rhs).abs();
            if diff > max_abs_diff {
                max_abs_diff = diff;
                max_idx = idx;
            }
        }
        if max_abs_diff > atol {
            candle::bail!(
                "{name}: max_abs_diff={max_abs_diff:.8} at index {max_idx} (actual={}, expected={})",
                actual[max_idx],
                expected[max_idx]
            );
        }
        Ok(())
    }

    fn geometry_weights(
        config: &GeometryConfig,
        device: &Device,
    ) -> Result<HashMap<String, Tensor>> {
        let mut tensors = HashMap::new();
        tensors.insert(
            "label_embed.weight".into(),
            Tensor::zeros((2, config.d_model), DType::F32, device)?,
        );
        tensors.insert(
            "cls_embed.weight".into(),
            Tensor::zeros((1, config.d_model), DType::F32, device)?,
        );
        tensors.insert(
            "points_direct_project.weight".into(),
            Tensor::zeros((config.d_model, 2), DType::F32, device)?,
        );
        tensors.insert(
            "points_direct_project.bias".into(),
            Tensor::zeros(config.d_model, DType::F32, device)?,
        );
        tensors.insert(
            "points_pool_project.weight".into(),
            Tensor::zeros((config.d_model, config.d_model), DType::F32, device)?,
        );
        tensors.insert(
            "points_pool_project.bias".into(),
            Tensor::zeros(config.d_model, DType::F32, device)?,
        );
        tensors.insert(
            "points_pos_enc_project.weight".into(),
            Tensor::zeros((config.d_model, config.d_model), DType::F32, device)?,
        );
        tensors.insert(
            "points_pos_enc_project.bias".into(),
            Tensor::zeros(config.d_model, DType::F32, device)?,
        );
        tensors.insert(
            "boxes_direct_project.weight".into(),
            Tensor::zeros((config.d_model, 4), DType::F32, device)?,
        );
        tensors.insert(
            "boxes_direct_project.bias".into(),
            Tensor::zeros(config.d_model, DType::F32, device)?,
        );
        tensors.insert(
            "boxes_pool_project.weight".into(),
            Tensor::zeros(
                (
                    config.d_model,
                    config.d_model,
                    config.roi_size,
                    config.roi_size,
                ),
                DType::F32,
                device,
            )?,
        );
        tensors.insert(
            "boxes_pool_project.bias".into(),
            Tensor::zeros(config.d_model, DType::F32, device)?,
        );
        tensors.insert(
            "boxes_pos_enc_project.weight".into(),
            Tensor::zeros((config.d_model, config.d_model + 2), DType::F32, device)?,
        );
        tensors.insert(
            "boxes_pos_enc_project.bias".into(),
            Tensor::zeros(config.d_model, DType::F32, device)?,
        );
        tensors.insert(
            "img_pre_norm.weight".into(),
            Tensor::ones(config.d_model, DType::F32, device)?,
        );
        tensors.insert(
            "img_pre_norm.bias".into(),
            Tensor::zeros(config.d_model, DType::F32, device)?,
        );
        tensors.insert(
            "final_proj.weight".into(),
            Tensor::zeros((config.d_model, config.d_model), DType::F32, device)?,
        );
        tensors.insert(
            "final_proj.bias".into(),
            Tensor::zeros(config.d_model, DType::F32, device)?,
        );
        tensors.insert(
            "norm.weight".into(),
            Tensor::ones(config.d_model, DType::F32, device)?,
        );
        tensors.insert(
            "norm.bias".into(),
            Tensor::zeros(config.d_model, DType::F32, device)?,
        );
        tensors.insert(
            "encode_norm.weight".into(),
            Tensor::ones(config.d_model, DType::F32, device)?,
        );
        tensors.insert(
            "encode_norm.bias".into(),
            Tensor::zeros(config.d_model, DType::F32, device)?,
        );
        tensors.insert(
            "encode.0.self_attn.in_proj_weight".into(),
            Tensor::zeros((config.d_model * 3, config.d_model), DType::F32, device)?,
        );
        tensors.insert(
            "encode.0.self_attn.in_proj_bias".into(),
            Tensor::zeros(config.d_model * 3, DType::F32, device)?,
        );
        tensors.insert(
            "encode.0.self_attn.out_proj.weight".into(),
            Tensor::zeros((config.d_model, config.d_model), DType::F32, device)?,
        );
        tensors.insert(
            "encode.0.self_attn.out_proj.bias".into(),
            Tensor::zeros(config.d_model, DType::F32, device)?,
        );
        tensors.insert(
            "encode.0.cross_attn_image.in_proj_weight".into(),
            Tensor::zeros((config.d_model * 3, config.d_model), DType::F32, device)?,
        );
        tensors.insert(
            "encode.0.cross_attn_image.in_proj_bias".into(),
            Tensor::zeros(config.d_model * 3, DType::F32, device)?,
        );
        tensors.insert(
            "encode.0.cross_attn_image.out_proj.weight".into(),
            Tensor::zeros((config.d_model, config.d_model), DType::F32, device)?,
        );
        tensors.insert(
            "encode.0.cross_attn_image.out_proj.bias".into(),
            Tensor::zeros(config.d_model, DType::F32, device)?,
        );
        tensors.insert(
            "encode.0.linear1.weight".into(),
            Tensor::zeros((config.dim_feedforward, config.d_model), DType::F32, device)?,
        );
        tensors.insert(
            "encode.0.linear1.bias".into(),
            Tensor::zeros(config.dim_feedforward, DType::F32, device)?,
        );
        tensors.insert(
            "encode.0.linear2.weight".into(),
            Tensor::zeros((config.d_model, config.dim_feedforward), DType::F32, device)?,
        );
        tensors.insert(
            "encode.0.linear2.bias".into(),
            Tensor::zeros(config.d_model, DType::F32, device)?,
        );
        for norm_name in ["norm1", "norm2", "norm3"] {
            tensors.insert(
                format!("encode.0.{norm_name}.weight"),
                Tensor::ones(config.d_model, DType::F32, device)?,
            );
            tensors.insert(
                format!("encode.0.{norm_name}.bias"),
                Tensor::zeros(config.d_model, DType::F32, device)?,
            );
        }
        Ok(tensors)
    }
}
