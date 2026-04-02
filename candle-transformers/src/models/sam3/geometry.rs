use std::f32::consts::PI;

use candle::{DType, IndexOp, Result, Tensor};
use candle_nn::{Conv2d, Embedding, LayerNorm, Linear, Module, VarBuilder};

use super::config::GeometryConfig;

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
        eprintln!("[PHASE4] After norm1");
        
        let hidden_states = self.self_attn.forward(
            &hidden_states,
            &hidden_states,
            Some(prompt_padding_mask),
            None,
        )?;
        eprintln!("[PHASE4] After self_attn");
        
        let hidden_states = (hidden_states + residual)?;
        eprintln!("[PHASE4] After self_attn residual");

        let residual = &hidden_states;
        let hidden_states = self.norm2.forward(&hidden_states)?;
        eprintln!("[PHASE4] After norm2");
        
        let hidden_states = self.cross_attn_image.forward(
            &hidden_states,
            vision_feats,
            None,
            Some(vision_pos_encoding),
        )?;
        eprintln!("[PHASE4] After cross_attn");
        
        let hidden_states = (hidden_states + residual)?;
        eprintln!("[PHASE4] After cross_attn residual");

        let residual = &hidden_states;
        let hidden_states = self.norm3.forward(&hidden_states)?;
        eprintln!("[PHASE4] After norm3");
        
        let hidden_states = self.linear1.forward(&hidden_states)?.relu()?;
        eprintln!("[PHASE4] After linear1+relu");
        
        let hidden_states = self.linear2.forward(&hidden_states)?;
        eprintln!("[PHASE4] After linear2");
        
        let result = (hidden_states + residual)?;
        eprintln!("[PHASE4] After FFN residual");
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
        eprintln!("[PHASE3] After initial norm, features shape: {:?}", features.dims());
        
        for (layer_idx, layer) in self.encode.iter().enumerate() {
            eprintln!("[PHASE3] Before layer {}", layer_idx);
            features =
                layer.forward(&features, &vision_feats, &vision_pos_embeds, &padding_mask)?;
            eprintln!("[PHASE3] After layer {}, features shape: {:?}", layer_idx, features.dims());
        }
        
        if let Some(encode_norm) = &self.encode_norm {
            features = encode_norm.forward(&features)?;
            eprintln!("[PHASE3] After encode_norm");
        }
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
        let mut point_features = label_embed;

        if let Some(points_direct_project) = &self.points_direct_project {
            point_features =
                point_features.broadcast_add(&points_direct_project.forward(&points_xy)?)?;
        }
        if let Some(points_pool_project) = &self.points_pool_project {
            let pooled_points = sample_points_nearest(vision_feats, &points_xy)?;
            point_features =
                point_features.broadcast_add(&points_pool_project.forward(&pooled_points)?)?;
        }
        if let Some(points_pos_enc_project) = &self.points_pos_enc_project {
            let pos_enc = encode_points_position(&points_xy, self.config.d_model)?;
            point_features =
                point_features.broadcast_add(&points_pos_enc_project.forward(&pos_enc)?)?;
        }
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
        let mut box_features = label_embed.clone();

        if let Some(boxes_direct_project) = &self.boxes_direct_project {
            let direct_proj = boxes_direct_project.forward(&boxes_cxcywh)?;
            eprintln!("[PHASE2] direct_proj shape: {:?}", direct_proj.dims());
            box_features = box_features.broadcast_add(&direct_proj)?;
            eprintln!("[PHASE2] After adding direct_proj");
        }
        if let Some(boxes_pool_project) = &self.boxes_pool_project {
            let pooled_boxes =
                sample_boxes_nearest(vision_feats, &boxes_cxcywh, self.config.roi_size)?;
            let pooled_boxes = boxes_pool_project.forward(&pooled_boxes)?;
            let pooled_boxes = pooled_boxes.reshape((seq_len, batch_size, self.config.d_model))?;
            eprintln!("[PHASE2] pool_proj shape: {:?}", pooled_boxes.dims());
            box_features = box_features.broadcast_add(&pooled_boxes)?;
            eprintln!("[PHASE2] After adding pool_proj");
        }
        if let Some(boxes_pos_enc_project) = &self.boxes_pos_enc_project {
            let pos_enc = encode_boxes_position(&boxes_cxcywh, self.config.d_model)?;
            eprintln!("[PHASE2] pos_enc shape: {:?}", pos_enc.dims());
            let pos_enc_proj = boxes_pos_enc_project.forward(&pos_enc)?;
            eprintln!("[PHASE2] pos_enc_proj shape: {:?}", pos_enc_proj.dims());
            box_features = box_features.broadcast_add(&pos_enc_proj)?;
            eprintln!("[PHASE2] After adding pos_enc_proj");
        }
        
        eprintln!("[PHASE2] FINAL box_features shape: {:?}", box_features.dims());
        
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
    Tensor::cat(&[&pos_y, &pos_x], 2)?
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
        eprintln!("[encode_boxes_position] boxes[0][0] = [{}, {}, {}, {}] (cx, cy, w, h)", 
            boxes[0][0][0], boxes[0][0][1], boxes[0][0][2], boxes[0][0][3]);
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
        eprintln!("[encode_boxes_position] pos_y[26]={}, pos_x[26]={}", 
            pos_y.get(26).copied().unwrap_or(-999.0),
            pos_x.get(26).copied().unwrap_or(-999.0));
    }
    
    let device = boxes_cxcywh.device();
    let pos_y = Tensor::from_slice(&pos_y, (boxes.len(), boxes[0].len(), num_pos_feats), device)?;
    let pos_x = Tensor::from_slice(&pos_x, (boxes.len(), boxes[0].len(), num_pos_feats), device)?;
    let hw = Tensor::from_slice(&hw, (boxes.len(), boxes[0].len(), 2), device)?;
    
    // DEBUG: Print tensor shapes and first values
    eprintln!("[encode_boxes_position] pos_y shape: {:?}, pos_x shape: {:?}", pos_y.dims(), pos_x.dims());
    
    let result = Tensor::cat(&[&pos_y, &pos_x, &hw], 2)?
        .transpose(0, 1)?
        .contiguous()?
        .to_dtype(boxes_cxcywh.dtype())?;
    
    // DEBUG: Print final shape
    eprintln!("[encode_boxes_position] Final output shape: {:?}", result.dims());
    
    Ok(result)
}

fn encode_1d_position(coord: f32, num_pos_feats: usize, out: &mut [f32]) {
    let temperature = 10_000f32;
    let coord_scaled = coord * 2.0 * PI;
    
    // DEBUG: Print first few and specific indices
    let debug_this = coord > 0.64 && coord < 0.66; // cy=0.653 range
    if debug_this {
        eprintln!("[encode_1d] input_coord={}, scaled_coord={}, num_pos_feats={}", coord, coord_scaled, num_pos_feats);
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
        if debug_this && (idx == 26 || (idx >= 0 && idx < 3)) {
            eprintln!("[encode_1d] idx={}, exponent={}, angle={}, out[{}]={}", 
                idx, exponent, angle, idx, out[idx]);
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
            let sample = bilinear_sample(
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

fn sample_boxes_nearest(
    vision_feats: &Tensor,
    boxes_cxcywh: &Tensor,
    roi_size: usize,
) -> Result<Tensor> {
    let (batch_size, channels, height, width) = vision_feats.dims4()?;
    let boxes = boxes_cxcywh.to_vec3::<f32>()?;
    let seq_len = boxes.len();
    let mut patches = Vec::with_capacity(batch_size * seq_len);
    for seq_boxes in boxes.iter() {
        for (batch_idx, box_coords) in seq_boxes.iter().enumerate() {
            let (x0, y0, x1, y1) =
                cxcywh_to_xyxy(box_coords[0], box_coords[1], box_coords[2], box_coords[3]);
            let x0 = x0 * width as f32;
            let y0 = y0 * height as f32;
            let x1 = x1 * width as f32;
            let y1 = y1 * height as f32;
            let box_w = (x1 - x0).max(1e-6);
            let box_h = (y1 - y0).max(1e-6);
            let mut rows = Vec::with_capacity(roi_size);
            for roi_y in 0..roi_size {
                let sample_y = y0 + (roi_y as f32 + 0.5) * box_h / roi_size as f32;
                let mut cols = Vec::with_capacity(roi_size);
                for roi_x in 0..roi_size {
                    let sample_x = x0 + (roi_x as f32 + 0.5) * box_w / roi_size as f32;
                    let sample = bilinear_sample(
                        vision_feats,
                        batch_idx,
                        sample_x,
                        sample_y,
                        width,
                        height,
                        channels,
                    )?;
                    cols.push(sample);
                }
                let col_refs = cols.iter().collect::<Vec<_>>();
                rows.push(Tensor::stack(&col_refs, 1)?);
            }
            let row_refs = rows.iter().collect::<Vec<_>>();
            patches.push(Tensor::stack(&row_refs, 2)?);
        }
    }
    let patch_refs = patches.iter().collect::<Vec<_>>();
    Tensor::stack(&patch_refs, 0)?.reshape((seq_len * batch_size, channels, roi_size, roi_size))
}

fn normalized_to_index(coord: f32, size: usize) -> usize {
    let max_idx = size.saturating_sub(1) as f32;
    (coord * size as f32).clamp(0.0, max_idx).round() as usize
}

fn bilinear_sample(
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

    use candle::{DType, Device, Result, Tensor};
    use candle_nn::VarBuilder;

    use super::{GeometryPrompt, SequenceGeometryEncoder};
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
