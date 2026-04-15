use std::collections::BTreeMap;

use candle::{DType, IndexOp, Result, Tensor};
use candle_nn::{
    conv2d, conv_transpose2d, layer_norm, Conv2d, Conv2dConfig, ConvTranspose2d,
    ConvTranspose2dConfig, LayerNorm, Linear, Module, VarBuilder,
};

use crate::models::segment_anything::{
    self, prompt_encoder::PromptEncoder, transformer::TwoWayTransformer, LayerNorm2d,
};

use super::{checkpoint::Sam3CheckpointSource, neck::VisualBackboneOutput, Config};

const NO_OBJ_SCORE: f32 = -1024.0;

#[derive(Debug, Clone)]
pub struct Sam3TrackerConfig {
    pub image_size: usize,
    pub hidden_dim: usize,
    pub memory_dim: usize,
    pub backbone_stride: usize,
    pub num_maskmem: usize,
    pub max_cond_frames_in_attn: usize,
    pub max_obj_ptrs_in_encoder: usize,
    pub tracker_num_heads: usize,
    pub tracker_num_layers: usize,
    pub tracker_feedforward_dim: usize,
    pub maskmem_interpol_size: usize,
    pub multimask_min_pt_num: usize,
    pub multimask_max_pt_num: usize,
}

impl Default for Sam3TrackerConfig {
    fn default() -> Self {
        Self {
            image_size: 1008,
            hidden_dim: 256,
            memory_dim: 64,
            backbone_stride: 14,
            num_maskmem: 7,
            max_cond_frames_in_attn: 4,
            max_obj_ptrs_in_encoder: 16,
            tracker_num_heads: 1,
            tracker_num_layers: 4,
            tracker_feedforward_dim: 2048,
            maskmem_interpol_size: 1152,
            multimask_min_pt_num: 1,
            multimask_max_pt_num: 1,
        }
    }
}

impl Sam3TrackerConfig {
    pub fn from_sam3_config(config: &Config) -> Self {
        Self {
            image_size: config.image.image_size,
            hidden_dim: config.neck.d_model,
            ..Self::default()
        }
    }

    fn image_embedding_size(&self) -> usize {
        self.image_size / self.backbone_stride
    }

    fn low_res_mask_size(&self) -> usize {
        self.image_embedding_size() * 4
    }
}

#[derive(Debug, Clone)]
pub struct TrackerFrameState {
    pub low_res_masks: Tensor,
    pub high_res_masks: Tensor,
    pub iou_scores: Tensor,
    pub obj_ptr: Tensor,
    pub object_score_logits: Tensor,
    pub maskmem_features: Option<Tensor>,
    pub maskmem_pos_enc: Option<Tensor>,
    pub is_cond_frame: bool,
}

impl TrackerFrameState {
    pub fn to_storage_device(&self, device: &candle::Device) -> Result<Self> {
        Ok(Self {
            low_res_masks: self.low_res_masks.to_device(device)?,
            high_res_masks: self.high_res_masks.to_device(device)?,
            iou_scores: self.iou_scores.to_device(device)?,
            obj_ptr: self.obj_ptr.to_device(device)?,
            object_score_logits: self.object_score_logits.to_device(device)?,
            maskmem_features: self
                .maskmem_features
                .as_ref()
                .map(|tensor| tensor.to_device(device))
                .transpose()?,
            maskmem_pos_enc: self
                .maskmem_pos_enc
                .as_ref()
                .map(|tensor| tensor.to_device(device))
                .transpose()?,
            is_cond_frame: self.is_cond_frame,
        })
    }
}

#[derive(Debug, Clone)]
pub struct TrackerStepOutput {
    pub state: TrackerFrameState,
    pub prompt_frame_indices: Vec<usize>,
    pub memory_frame_indices: Vec<usize>,
}

#[derive(Debug)]
struct TrackerMlp {
    layers: Vec<segment_anything::Linear>,
    sigmoid_output: bool,
}

impl TrackerMlp {
    fn new(
        input_dim: usize,
        hidden_dim: usize,
        output_dim: usize,
        num_layers: usize,
        sigmoid_output: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let mut layers = Vec::with_capacity(num_layers);
        let vb = vb.pp("layers");
        for idx in 0..num_layers {
            let in_dim = if idx == 0 { input_dim } else { hidden_dim };
            let out_dim = if idx + 1 == num_layers {
                output_dim
            } else {
                hidden_dim
            };
            layers.push(segment_anything::linear(vb.pp(idx), in_dim, out_dim, true)?);
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
        for (idx, layer) in self.layers.iter().enumerate() {
            xs = layer.forward(&xs.contiguous()?)?;
            if idx + 1 < self.layers.len() {
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
struct Sam3TrackerMaskDecoder {
    transformer_dim: usize,
    transformer: TwoWayTransformer,
    num_multimask_outputs: usize,
    iou_token: candle_nn::Embedding,
    num_mask_tokens: usize,
    mask_tokens: candle_nn::Embedding,
    obj_score_token: candle_nn::Embedding,
    output_upscaling_conv1: ConvTranspose2d,
    output_upscaling_ln1: LayerNorm2d,
    output_upscaling_conv2: ConvTranspose2d,
    conv_s0: Conv2d,
    conv_s1: Conv2d,
    output_hypernetworks_mlps: Vec<TrackerMlp>,
    iou_prediction_head: TrackerMlp,
    pred_obj_score_head: TrackerMlp,
    use_multimask_token_for_obj_ptr: bool,
}

#[derive(Debug)]
struct Sam3TrackerMaskDecoderOutput {
    low_res_multimasks: Tensor,
    high_res_multimasks: Tensor,
    iou_pred: Tensor,
    sam_tokens_out: Tensor,
    object_score_logits: Tensor,
}

impl Sam3TrackerMaskDecoder {
    fn new(config: &Sam3TrackerConfig, vb: VarBuilder) -> Result<Self> {
        let transformer_dim = config.hidden_dim;
        let num_multimask_outputs = 3;
        let num_mask_tokens = num_multimask_outputs + 1;
        let iou_token = candle_nn::embedding(1, transformer_dim, vb.pp("iou_token"))?;
        let mask_tokens =
            candle_nn::embedding(num_mask_tokens, transformer_dim, vb.pp("mask_tokens"))?;
        let obj_score_token = candle_nn::embedding(1, transformer_dim, vb.pp("obj_score_token"))?;

        let cfg = ConvTranspose2dConfig {
            stride: 2,
            ..Default::default()
        };
        let output_upscaling_conv1 = conv_transpose2d(
            transformer_dim,
            transformer_dim / 4,
            2,
            cfg,
            vb.pp("output_upscaling.0"),
        )?;
        let output_upscaling_ln1 =
            LayerNorm2d::new(transformer_dim / 4, 1e-6, vb.pp("output_upscaling.1"))?;
        let output_upscaling_conv2 = conv_transpose2d(
            transformer_dim / 4,
            transformer_dim / 8,
            2,
            cfg,
            vb.pp("output_upscaling.3"),
        )?;
        let conv_s0 = conv2d(
            transformer_dim,
            transformer_dim / 8,
            1,
            Default::default(),
            vb.pp("conv_s0"),
        )?;
        let conv_s1 = conv2d(
            transformer_dim,
            transformer_dim / 4,
            1,
            Default::default(),
            vb.pp("conv_s1"),
        )?;
        let mut output_hypernetworks_mlps = Vec::with_capacity(num_mask_tokens);
        let vb_h = vb.pp("output_hypernetworks_mlps");
        for idx in 0..num_mask_tokens {
            output_hypernetworks_mlps.push(TrackerMlp::new(
                transformer_dim,
                transformer_dim,
                transformer_dim / 8,
                3,
                false,
                vb_h.pp(idx),
            )?);
        }
        let iou_prediction_head = TrackerMlp::new(
            transformer_dim,
            transformer_dim,
            num_mask_tokens,
            3,
            true,
            vb.pp("iou_prediction_head"),
        )?;
        let pred_obj_score_head = TrackerMlp::new(
            transformer_dim,
            transformer_dim,
            1,
            3,
            false,
            vb.pp("pred_obj_score_head"),
        )?;
        let transformer =
            TwoWayTransformer::new(2, transformer_dim, 8, 2048, vb.pp("transformer"))?;
        Ok(Self {
            transformer_dim,
            transformer,
            num_multimask_outputs,
            iou_token,
            num_mask_tokens,
            mask_tokens,
            obj_score_token,
            output_upscaling_conv1,
            output_upscaling_ln1,
            output_upscaling_conv2,
            conv_s0,
            conv_s1,
            output_hypernetworks_mlps,
            iou_prediction_head,
            pred_obj_score_head,
            use_multimask_token_for_obj_ptr: true,
        })
    }

    fn forward(
        &self,
        image_embeddings: &Tensor,
        image_pe: &Tensor,
        sparse_prompt_embeddings: &Tensor,
        dense_prompt_embeddings: &Tensor,
        multimask_output: bool,
        high_res_features: Option<(&Tensor, &Tensor)>,
    ) -> Result<Sam3TrackerMaskDecoderOutput> {
        let output_tokens = Tensor::cat(
            &[
                self.obj_score_token.embeddings(),
                self.iou_token.embeddings(),
                self.mask_tokens.embeddings(),
            ],
            0,
        )?;
        let (token_count, token_dim) = output_tokens.dims2()?;
        let output_tokens = output_tokens.unsqueeze(0)?.expand((
            sparse_prompt_embeddings.dim(0)?,
            token_count,
            token_dim,
        ))?;
        let tokens = Tensor::cat(&[&output_tokens, sparse_prompt_embeddings], 1)?;

        let src = image_embeddings.broadcast_add(dense_prompt_embeddings)?;
        let pos_src = repeat_interleave(image_pe, tokens.dim(0)?, 0)?;
        let (batch_size, channels, height, width) = src.dims4()?;
        let (hs, src) = self.transformer.forward(&src, &pos_src, &tokens)?;
        let iou_token_out = hs.i((.., 1))?;
        let mask_tokens_out = hs.i((.., 2..2 + self.num_mask_tokens))?;
        let object_score_logits = self.pred_obj_score_head.forward(&hs.i((.., 0))?)?;

        let src = src
            .transpose(1, 2)?
            .reshape((batch_size, channels, height, width))?;
        let upscaled_embedding = match high_res_features {
            Some((feat_s0, feat_s1)) => {
                let up1 = self.output_upscaling_conv1.forward(&src)?;
                let feat_s1 = self.conv_s1.forward(feat_s1)?;
                let up1 = up1.broadcast_add(&feat_s1)?;
                let up1 = self.output_upscaling_ln1.forward(&up1)?.gelu()?;
                let up2 = self.output_upscaling_conv2.forward(&up1)?;
                let feat_s0 = self.conv_s0.forward(feat_s0)?;
                up2.broadcast_add(&feat_s0)?.gelu()?
            }
            None => self
                .output_upscaling_conv1
                .forward(&src)?
                .apply(&self.output_upscaling_ln1)?
                .gelu()?
                .apply(&self.output_upscaling_conv2)?
                .gelu()?,
        };

        let mut hyper_in = Vec::with_capacity(self.num_mask_tokens);
        for idx in 0..self.num_mask_tokens {
            hyper_in
                .push(self.output_hypernetworks_mlps[idx].forward(&mask_tokens_out.i((.., idx))?)?);
        }
        let hyper_in = Tensor::stack(&hyper_in.iter().collect::<Vec<_>>(), 1)?.contiguous()?;
        let (batch_size, channels, height, width) = upscaled_embedding.dims4()?;
        let low_res_multimasks = hyper_in.matmul(&upscaled_embedding.reshape((
            batch_size,
            channels,
            height * width,
        ))?)?;
        let low_res_multimasks = low_res_multimasks.reshape((batch_size, (), height, width))?;
        let high_res_multimasks =
            low_res_multimasks.upsample_bilinear2d(height * 4, width * 4, false)?;
        let iou_pred = self.iou_prediction_head.forward(&iou_token_out)?;
        let sam_tokens_out = if multimask_output && self.use_multimask_token_for_obj_ptr {
            mask_tokens_out.i((.., 1..))?
        } else {
            mask_tokens_out.i((.., 0..1))?
        };

        Ok(Sam3TrackerMaskDecoderOutput {
            low_res_multimasks,
            high_res_multimasks,
            iou_pred,
            sam_tokens_out,
            object_score_logits,
        })
    }
}

#[derive(Debug)]
struct SimpleMaskDownSampler {
    layers: Vec<SimpleMaskDownSamplerLayer>,
    out_proj: Conv2d,
    interpol_size: usize,
}

#[derive(Debug)]
struct SimpleMaskDownSamplerLayer {
    conv: Conv2d,
    norm: LayerNorm2d,
}

impl SimpleMaskDownSampler {
    fn new(config: &Sam3TrackerConfig, vb: VarBuilder) -> Result<Self> {
        let mut layers = Vec::with_capacity(4);
        let mut in_channels = 1;
        let mut out_channels = 1;
        let conv_cfg = Conv2dConfig {
            stride: 2,
            padding: 1,
            ..Default::default()
        };
        for (idx, out_mult) in [4usize, 16, 64, 256].iter().copied().enumerate() {
            out_channels = out_mult;
            let layer_vb = vb.pp("encoder").pp(idx * 3);
            let norm_vb = vb.pp("encoder").pp(idx * 3 + 1);
            layers.push(SimpleMaskDownSamplerLayer {
                conv: conv2d(in_channels, out_channels, 3, conv_cfg, layer_vb)?,
                norm: LayerNorm2d::new(out_channels, 1e-6, norm_vb)?,
            });
            in_channels = out_channels;
        }
        let out_proj = conv2d(
            out_channels,
            config.hidden_dim,
            1,
            Default::default(),
            vb.pp("encoder").pp(12),
        )?;
        Ok(Self {
            layers,
            out_proj,
            interpol_size: config.maskmem_interpol_size,
        })
    }
}

impl Module for SimpleMaskDownSampler {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut xs = xs.clone();
        let (_, _, height, width) = xs.dims4()?;
        if height != self.interpol_size || width != self.interpol_size {
            xs = xs.upsample_bilinear2d(self.interpol_size, self.interpol_size, false)?;
        }
        for layer in self.layers.iter() {
            xs = layer.conv.forward(&xs)?;
            xs = layer.norm.forward(&xs)?;
            xs = xs.gelu()?;
        }
        self.out_proj.forward(&xs)
    }
}

#[derive(Debug)]
struct CxBlock {
    dwconv: Conv2d,
    norm: LayerNorm2d,
    pwconv1: segment_anything::Linear,
    pwconv2: segment_anything::Linear,
    gamma: Tensor,
}

impl CxBlock {
    fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        let dw_cfg = Conv2dConfig {
            padding: 3,
            groups: dim,
            ..Default::default()
        };
        Ok(Self {
            dwconv: conv2d(dim, dim, 7, dw_cfg, vb.pp("dwconv"))?,
            norm: LayerNorm2d::new(dim, 1e-6, vb.pp("norm"))?,
            pwconv1: segment_anything::linear(vb.pp("pwconv1"), dim, dim * 4, true)?,
            pwconv2: segment_anything::linear(vb.pp("pwconv2"), dim * 4, dim, true)?,
            gamma: vb.get(dim, "gamma")?,
        })
    }
}

impl Module for CxBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let residual = xs.clone();
        let xs = self.dwconv.forward(xs)?;
        let xs = self.norm.forward(&xs)?;
        let xs = xs.permute((0, 2, 3, 1))?;
        let xs = self.pwconv1.forward(&xs.contiguous()?)?.gelu()?;
        let xs = self.pwconv2.forward(&xs.contiguous()?)?;
        let gamma = self.gamma.reshape((1, 1, 1, ()))?;
        let xs = xs.broadcast_mul(&gamma)?;
        let xs = xs.permute((0, 3, 1, 2))?;
        residual.broadcast_add(&xs)
    }
}

#[derive(Debug)]
struct SimpleFuser {
    layers: Vec<CxBlock>,
}

impl SimpleFuser {
    fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            layers: vec![
                CxBlock::new(dim, vb.pp("layers").pp(0))?,
                CxBlock::new(dim, vb.pp("layers").pp(1))?,
            ],
        })
    }
}

impl Module for SimpleFuser {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut xs = xs.clone();
        for layer in self.layers.iter() {
            xs = layer.forward(&xs)?;
        }
        Ok(xs)
    }
}

#[derive(Debug)]
struct SimpleMaskEncoder {
    mask_downsampler: SimpleMaskDownSampler,
    pix_feat_proj: Conv2d,
    fuser: SimpleFuser,
    out_proj: Conv2d,
    out_dim: usize,
}

impl SimpleMaskEncoder {
    fn new(config: &Sam3TrackerConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            mask_downsampler: SimpleMaskDownSampler::new(config, vb.pp("mask_downsampler"))?,
            pix_feat_proj: conv2d(
                config.hidden_dim,
                config.hidden_dim,
                1,
                Default::default(),
                vb.pp("pix_feat_proj"),
            )?,
            fuser: SimpleFuser::new(config.hidden_dim, vb.pp("fuser"))?,
            out_proj: conv2d(
                config.hidden_dim,
                config.memory_dim,
                1,
                Default::default(),
                vb.pp("out_proj"),
            )?,
            out_dim: config.memory_dim,
        })
    }

    fn forward(&self, pix_feat: &Tensor, masks: &Tensor) -> Result<(Tensor, Tensor)> {
        let masks = self.mask_downsampler.forward(masks)?;
        let mut xs = self.pix_feat_proj.forward(pix_feat)?;
        xs = xs.broadcast_add(&masks)?;
        xs = self.fuser.forward(&xs)?;
        xs = self.out_proj.forward(&xs)?;
        let pos = build_2d_sine_position_encoding(&xs, self.out_dim)?;
        Ok((xs, pos))
    }
}

#[derive(Debug)]
struct TrackerAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
    internal_dim: usize,
}

impl TrackerAttention {
    fn new(
        embedding_dim: usize,
        num_heads: usize,
        downsample_rate: usize,
        kv_in_dim: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let internal_dim = embedding_dim / downsample_rate;
        Ok(Self {
            q_proj: candle_nn::linear_b(embedding_dim, internal_dim, true, vb.pp("q_proj"))?,
            k_proj: candle_nn::linear_b(kv_in_dim, internal_dim, true, vb.pp("k_proj"))?,
            v_proj: candle_nn::linear_b(kv_in_dim, internal_dim, true, vb.pp("v_proj"))?,
            out_proj: candle_nn::linear_b(internal_dim, embedding_dim, true, vb.pp("out_proj"))?,
            num_heads,
            internal_dim,
        })
    }

    fn separate_heads(&self, xs: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_len, channels) = xs.dims3()?;
        xs.reshape((
            batch_size,
            seq_len,
            self.num_heads,
            channels / self.num_heads,
        ))?
        .transpose(1, 2)?
        .contiguous()
    }

    fn recombine_heads(&self, xs: &Tensor) -> Result<Tensor> {
        let (batch_size, num_heads, seq_len, head_dim) = xs.dims4()?;
        xs.transpose(1, 2)?
            .reshape((batch_size, seq_len, num_heads * head_dim))
    }

    fn project_qkv(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let q = self.q_proj.forward(&q.contiguous()?)?;
        let k = self.k_proj.forward(&k.contiguous()?)?;
        let v = self.v_proj.forward(&v.contiguous()?)?;
        Ok((q, k, v))
    }

    fn forward_projected(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        let q = self.separate_heads(q)?;
        let k = self.separate_heads(k)?;
        let v = self.separate_heads(v)?;
        let head_dim = self.internal_dim / self.num_heads;
        let scale = Tensor::new((head_dim as f32).powf(-0.5), q.device())?;
        let q = q.to_dtype(DType::F32)?.broadcast_mul(&scale)?;
        let k = k.to_dtype(DType::F32)?;
        let v = v.to_dtype(DType::F32)?;
        let attn = q.matmul(&k.transpose(2, 3)?)?;
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        let out = attn.matmul(&v)?;
        self.out_proj.forward(
            &self
                .recombine_heads(&out)?
                .to_dtype(q.dtype())?
                .contiguous()?,
        )
    }
}

#[derive(Debug)]
struct AxialRotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
}

impl AxialRotaryEmbedding {
    fn new(
        head_dim: usize,
        end_x: usize,
        end_y: usize,
        theta: f64,
        device: &candle::Device,
    ) -> Result<Self> {
        if head_dim % 4 != 0 {
            candle::bail!(
                "tracker RoPE head_dim must be divisible by 4, got {}",
                head_dim
            );
        }
        let rotary_dim = head_dim / 4;
        let seq_len = end_x * end_y;
        let mut cos = vec![0f32; seq_len * head_dim];
        let mut sin = vec![0f32; seq_len * head_dim];
        let inv_freqs: Vec<f32> = (0..rotary_dim)
            .map(|idx| 1f32 / (theta as f32).powf((4 * idx) as f32 / head_dim as f32))
            .collect();
        for flat_idx in 0..seq_len {
            let x_pos = (flat_idx % end_x) as f32;
            let y_pos = (flat_idx / end_x) as f32;
            let row = &mut cos[flat_idx * head_dim..(flat_idx + 1) * head_dim];
            let row_sin = &mut sin[flat_idx * head_dim..(flat_idx + 1) * head_dim];
            for (idx, inv_freq) in inv_freqs.iter().copied().enumerate() {
                let x_freq = x_pos * inv_freq;
                let y_freq = y_pos * inv_freq;
                let x_offset = 2 * idx;
                row[x_offset] = x_freq.cos();
                row[x_offset + 1] = x_freq.cos();
                row_sin[x_offset] = x_freq.sin();
                row_sin[x_offset + 1] = x_freq.sin();

                let y_offset = 2 * rotary_dim + 2 * idx;
                row[y_offset] = y_freq.cos();
                row[y_offset + 1] = y_freq.cos();
                row_sin[y_offset] = y_freq.sin();
                row_sin[y_offset + 1] = y_freq.sin();
            }
        }
        Ok(Self {
            cos: Tensor::from_slice(&cos, (seq_len, head_dim), device)?,
            sin: Tensor::from_slice(&sin, (seq_len, head_dim), device)?,
        })
    }

    fn apply(
        &self,
        q: &Tensor,
        k: &Tensor,
        repeat_freqs_k: bool,
        num_k_exclude_rope: usize,
    ) -> Result<(Tensor, Tensor)> {
        let (_, _, q_len, head_dim) = q.dims4()?;
        let (_, _, k_len, _) = k.dims4()?;
        let cos = self
            .cos
            .narrow(0, 0, q_len)?
            .reshape((1, 1, q_len, head_dim))?;
        let sin = self
            .sin
            .narrow(0, 0, q_len)?
            .reshape((1, 1, q_len, head_dim))?;
        let q_dtype = q.dtype();
        let k_dtype = k.dtype();
        let q = q.to_dtype(DType::F32)?;
        let mut k = k.to_dtype(DType::F32)?;
        let q_rot = rotate_pairwise(&q)?;
        let q = (q.broadcast_mul(&cos)? + q_rot.broadcast_mul(&sin)?)?;

        let num_k_rope = k_len.saturating_sub(num_k_exclude_rope);
        if num_k_rope > 0 {
            let k_rope = k.i((.., .., ..num_k_rope, ..))?;
            let (cos_k, sin_k) = if repeat_freqs_k && num_k_rope > q_len {
                let repeat = num_k_rope / q_len;
                (
                    repeat_interleave(&cos, repeat, 2)?,
                    repeat_interleave(&sin, repeat, 2)?,
                )
            } else {
                (cos.clone(), sin.clone())
            };
            let k_rot = rotate_pairwise(&k_rope)?;
            let k_rope = (k_rope.broadcast_mul(&cos_k)? + k_rot.broadcast_mul(&sin_k)?)?;
            if num_k_exclude_rope == 0 {
                k = k_rope;
            } else {
                let k_tail = k.i((.., .., num_k_rope.., ..))?;
                k = Tensor::cat(&[&k_rope, &k_tail], 2)?;
            }
        }

        Ok((q.to_dtype(q_dtype)?, k.to_dtype(k_dtype)?))
    }
}

fn rotate_pairwise(xs: &Tensor) -> Result<Tensor> {
    let (batch_size, num_heads, seq_len, head_dim) = xs.dims4()?;
    let xs = xs.reshape((batch_size, num_heads, seq_len, head_dim / 2, 2))?;
    let even = xs.i((.., .., .., .., 0))?;
    let odd = xs.i((.., .., .., .., 1))?;
    Tensor::stack(&[&odd.neg()?, &even], 4)?.reshape((batch_size, num_heads, seq_len, head_dim))
}

#[derive(Debug)]
struct TrackerRopeAttention {
    attn: TrackerAttention,
    rope: AxialRotaryEmbedding,
    repeat_freqs_k: bool,
}

impl TrackerRopeAttention {
    fn new(
        embedding_dim: usize,
        num_heads: usize,
        downsample_rate: usize,
        kv_in_dim: usize,
        rope_theta: f64,
        feat_size: usize,
        repeat_freqs_k: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let device = vb.device().clone();
        let attn = TrackerAttention::new(embedding_dim, num_heads, downsample_rate, kv_in_dim, vb)?;
        let rope = AxialRotaryEmbedding::new(
            attn.internal_dim / num_heads,
            feat_size,
            feat_size,
            rope_theta,
            &device,
        )?;
        Ok(Self {
            attn,
            rope,
            repeat_freqs_k,
        })
    }

    fn forward(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        num_k_exclude_rope: usize,
    ) -> Result<Tensor> {
        let (q, k, v) = self.attn.project_qkv(q, k, v)?;
        let q_heads = self.attn.separate_heads(&q)?;
        let k_heads = self.attn.separate_heads(&k)?;
        let (q_heads, k_heads) =
            self.rope
                .apply(&q_heads, &k_heads, self.repeat_freqs_k, num_k_exclude_rope)?;
        let v_heads = self.attn.separate_heads(&v)?;
        let head_dim = self.attn.internal_dim / self.attn.num_heads;
        let scale = Tensor::new((head_dim as f32).powf(-0.5), q_heads.device())?;
        let q_heads = q_heads.to_dtype(DType::F32)?.broadcast_mul(&scale)?;
        let k_heads = k_heads.to_dtype(DType::F32)?;
        let v_heads = v_heads.to_dtype(DType::F32)?;
        let attn = q_heads.matmul(&k_heads.transpose(2, 3)?)?;
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        let out = attn.matmul(&v_heads)?;
        self.attn.out_proj.forward(
            &self
                .attn
                .recombine_heads(&out)?
                .to_dtype(q.dtype())?
                .contiguous()?,
        )
    }
}

#[derive(Debug)]
struct TrackerEncoderLayer {
    self_attn: TrackerRopeAttention,
    cross_attn_image: TrackerRopeAttention,
    linear1: Linear,
    linear2: Linear,
    norm1: LayerNorm,
    norm2: LayerNorm,
    norm3: LayerNorm,
}

impl TrackerEncoderLayer {
    fn new(config: &Sam3TrackerConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            self_attn: TrackerRopeAttention::new(
                config.hidden_dim,
                config.tracker_num_heads,
                1,
                config.hidden_dim,
                10_000.0,
                config.image_embedding_size(),
                false,
                vb.pp("self_attn"),
            )?,
            cross_attn_image: TrackerRopeAttention::new(
                config.hidden_dim,
                config.tracker_num_heads,
                1,
                config.memory_dim,
                10_000.0,
                config.image_embedding_size(),
                true,
                vb.pp("cross_attn_image"),
            )?,
            linear1: candle_nn::linear_b(
                config.hidden_dim,
                config.tracker_feedforward_dim,
                true,
                vb.pp("linear1"),
            )?,
            linear2: candle_nn::linear_b(
                config.tracker_feedforward_dim,
                config.hidden_dim,
                true,
                vb.pp("linear2"),
            )?,
            norm1: layer_norm(config.hidden_dim, 1e-5, vb.pp("norm1"))?,
            norm2: layer_norm(config.hidden_dim, 1e-5, vb.pp("norm2"))?,
            norm3: layer_norm(config.hidden_dim, 1e-5, vb.pp("norm3"))?,
        })
    }

    fn forward(
        &self,
        tgt: &Tensor,
        memory: &Tensor,
        query_pos: &Tensor,
        pos: &Tensor,
        num_k_exclude_rope: usize,
    ) -> Result<Tensor> {
        let tgt2 = self.norm1.forward(tgt)?;
        let q = (&tgt2 + query_pos)?;
        let tgt_sa = self.self_attn.forward(&q, &q, &tgt2, 0)?;
        let tgt = tgt.broadcast_add(&tgt_sa)?;

        let tgt2 = self.norm2.forward(&tgt)?;
        let q = (&tgt2 + query_pos)?;
        let k = (memory + pos)?;
        let tgt_ca = self
            .cross_attn_image
            .forward(&q, &k, memory, num_k_exclude_rope)?;
        let tgt = tgt.broadcast_add(&tgt_ca)?;

        let tgt2 = self.norm3.forward(&tgt)?;
        let tgt2 = self.linear1.forward(&tgt2.contiguous()?)?.relu()?;
        let tgt2 = self.linear2.forward(&tgt2.contiguous()?)?;
        tgt.broadcast_add(&tgt2)
    }
}

#[derive(Debug)]
struct TrackerMemoryAttentionEncoder {
    layers: Vec<TrackerEncoderLayer>,
    norm: LayerNorm,
}

impl TrackerMemoryAttentionEncoder {
    fn new(config: &Sam3TrackerConfig, vb: VarBuilder) -> Result<Self> {
        let mut layers = Vec::with_capacity(config.tracker_num_layers);
        let vb_layers = vb.pp("layers");
        for idx in 0..config.tracker_num_layers {
            layers.push(TrackerEncoderLayer::new(config, vb_layers.pp(idx))?);
        }
        Ok(Self {
            layers,
            norm: layer_norm(config.hidden_dim, 1e-5, vb.pp("norm"))?,
        })
    }

    fn forward(
        &self,
        src: &Tensor,
        src_pos: &Tensor,
        prompt: &Tensor,
        prompt_pos: &Tensor,
        num_obj_ptr_tokens: usize,
    ) -> Result<Tensor> {
        let mut output = src.broadcast_add(&src_pos.affine(0.1, 0.0)?)?;
        for layer in self.layers.iter() {
            output = layer.forward(&output, prompt, src_pos, prompt_pos, num_obj_ptr_tokens)?;
        }
        self.norm.forward(&output)
    }
}

#[derive(Debug)]
pub struct Sam3TrackerModel {
    config: Sam3TrackerConfig,
    prompt_encoder: PromptEncoder,
    mask_decoder: Sam3TrackerMaskDecoder,
    mask_downsample: Conv2d,
    transformer: TrackerMemoryAttentionEncoder,
    maskmem_backbone: SimpleMaskEncoder,
    maskmem_tpos_enc: Tensor,
    no_mem_embed: Tensor,
    no_mem_pos_enc: Tensor,
    no_obj_ptr: Tensor,
    no_obj_embed_spatial: Tensor,
    obj_ptr_proj: TrackerMlp,
    obj_ptr_tpos_proj: Linear,
}

impl Sam3TrackerModel {
    pub fn new(config: &Sam3TrackerConfig, vb: VarBuilder) -> Result<Self> {
        let prompt_encoder = PromptEncoder::new(
            config.hidden_dim,
            (config.image_embedding_size(), config.image_embedding_size()),
            (config.image_size, config.image_size),
            16,
            vb.pp("sam_prompt_encoder"),
        )?;
        let mask_decoder = Sam3TrackerMaskDecoder::new(config, vb.pp("sam_mask_decoder"))?;
        let mask_downsample = conv2d(
            1,
            1,
            4,
            Conv2dConfig {
                stride: 4,
                ..Default::default()
            },
            vb.pp("mask_downsample"),
        )?;
        let transformer =
            TrackerMemoryAttentionEncoder::new(config, vb.pp("transformer").pp("encoder"))?;
        let maskmem_backbone = SimpleMaskEncoder::new(config, vb.pp("maskmem_backbone"))?;
        Ok(Self {
            config: config.clone(),
            prompt_encoder,
            mask_decoder,
            mask_downsample,
            transformer,
            maskmem_backbone,
            maskmem_tpos_enc: vb.get(
                (config.num_maskmem, 1, 1, config.memory_dim),
                "maskmem_tpos_enc",
            )?,
            no_mem_embed: vb.get((1, 1, config.hidden_dim), "no_mem_embed")?,
            no_mem_pos_enc: vb.get((1, 1, config.hidden_dim), "no_mem_pos_enc")?,
            no_obj_ptr: vb.get((1, config.hidden_dim), "no_obj_ptr")?,
            no_obj_embed_spatial: vb.get((1, config.memory_dim), "no_obj_embed_spatial")?,
            obj_ptr_proj: TrackerMlp::new(
                config.hidden_dim,
                config.hidden_dim,
                config.hidden_dim,
                3,
                false,
                vb.pp("obj_ptr_proj"),
            )?,
            obj_ptr_tpos_proj: candle_nn::linear_b(
                config.hidden_dim,
                config.memory_dim,
                true,
                vb.pp("obj_ptr_tpos_proj"),
            )?,
        })
    }

    pub fn from_checkpoint_source(
        sam3_config: &Config,
        checkpoint: &Sam3CheckpointSource,
        dtype: DType,
        device: &candle::Device,
    ) -> Result<Self> {
        let tracker_config = Sam3TrackerConfig::from_sam3_config(sam3_config);
        let vb = checkpoint.load_tracker_var_builder(dtype, device)?;
        Self::new(&tracker_config, vb)
    }

    pub fn config(&self) -> &Sam3TrackerConfig {
        &self.config
    }

    pub fn track_frame(
        &self,
        visual: &VisualBackboneOutput,
        frame_idx: usize,
        num_frames: usize,
        point_coords: Option<&Tensor>,
        point_labels: Option<&Tensor>,
        boxes_xyxy: Option<&Tensor>,
        mask_input: Option<&Tensor>,
        history: &BTreeMap<usize, TrackerFrameState>,
        is_conditioning_frame: bool,
        reverse: bool,
        use_prev_mem_frame: bool,
    ) -> Result<TrackerStepOutput> {
        let (backbone_fpn, vision_pos_enc) = match (&visual.sam2_backbone_fpn, &visual.sam2_pos_enc)
        {
            (Some(backbone_fpn), Some(vision_pos_enc)) => (backbone_fpn, vision_pos_enc),
            _ => (&visual.backbone_fpn, &visual.vision_pos_enc),
        };
        let low_res_feature = backbone_fpn.last().ok_or_else(|| {
            candle::Error::Msg("tracker requires a low-resolution FPN feature".to_owned())
        })?;
        let low_res_pos = vision_pos_enc.last().ok_or_else(|| {
            candle::Error::Msg("tracker requires a low-resolution positional encoding".to_owned())
        })?;
        let high_res_features = if backbone_fpn.len() >= 2 {
            Some((&backbone_fpn[0], &backbone_fpn[1]))
        } else {
            None
        };
        let current_vision_feats = low_res_feature
            .flatten_from(2)?
            .permute((0, 2, 1))?
            .contiguous()?;
        let current_vision_pos = low_res_pos
            .flatten_from(2)?
            .permute((0, 2, 1))?
            .contiguous()?;

        let is_init_cond_frame = history.is_empty() || !use_prev_mem_frame;
        let (pix_feat_with_mem, prompt_frame_indices, memory_frame_indices, num_obj_ptr_tokens) =
            if mask_input.is_none() {
                self.prepare_memory_conditioned_features(
                    frame_idx,
                    num_frames,
                    is_init_cond_frame,
                    &current_vision_feats,
                    &current_vision_pos,
                    history,
                    reverse,
                    use_prev_mem_frame,
                )?
            } else {
                (
                    current_vision_feats.permute((0, 2, 1))?.reshape((
                        low_res_feature.dim(0)?,
                        self.config.hidden_dim,
                        self.config.image_embedding_size(),
                        self.config.image_embedding_size(),
                    ))?,
                    Vec::new(),
                    Vec::new(),
                    0,
                )
            };

        let sam_outputs = if let Some(mask_input) = mask_input {
            self.use_mask_as_output(&pix_feat_with_mem, high_res_features, mask_input)?
        } else {
            let (sparse_embeddings, dense_embeddings) =
                self.prompt_encoder
                    .forward(point_coords.zip(point_labels), boxes_xyxy, None)?;
            let image_pe = self.prompt_encoder.get_dense_pe()?;
            let point_count = point_coords
                .map(|coords| coords.dim(1))
                .transpose()?
                .unwrap_or(0);
            let multimask_output = is_conditioning_frame
                && boxes_xyxy.is_none()
                && point_count >= self.config.multimask_min_pt_num
                && point_count <= self.config.multimask_max_pt_num;
            self.forward_sam_heads(
                &pix_feat_with_mem,
                &image_pe,
                &sparse_embeddings,
                &dense_embeddings,
                multimask_output,
                high_res_features,
            )?
        };
        let (maskmem_features, maskmem_pos_enc) = self.encode_new_memory(
            low_res_feature,
            &sam_outputs.state.high_res_masks,
            &sam_outputs.state.object_score_logits,
            point_coords.is_some(),
        )?;

        let state = TrackerFrameState {
            low_res_masks: sam_outputs.state.low_res_masks,
            high_res_masks: sam_outputs.state.high_res_masks,
            iou_scores: sam_outputs.state.iou_scores,
            obj_ptr: sam_outputs.state.obj_ptr,
            object_score_logits: sam_outputs.state.object_score_logits,
            maskmem_features: Some(maskmem_features),
            maskmem_pos_enc: Some(maskmem_pos_enc),
            is_cond_frame: is_conditioning_frame,
        };
        let _ = num_obj_ptr_tokens;
        Ok(TrackerStepOutput {
            state,
            prompt_frame_indices,
            memory_frame_indices,
        })
    }

    fn use_mask_as_output(
        &self,
        backbone_features: &Tensor,
        high_res_features: Option<(&Tensor, &Tensor)>,
        mask_input: &Tensor,
    ) -> Result<TrackerStepOutput> {
        let mask_input = if mask_input.rank() == 3 {
            mask_input.unsqueeze(1)?
        } else {
            mask_input.clone()
        };
        let high_res_masks = mask_input.to_dtype(DType::F32)?.affine(20.0, -10.0)?;
        let low_res_masks = high_res_masks.upsample_bilinear2d(
            self.config.low_res_mask_size(),
            self.config.low_res_mask_size(),
            false,
        )?;
        let mask_prompt = self
            .mask_downsample
            .forward(&mask_input.to_dtype(DType::F32)?)?;
        let image_pe = self.prompt_encoder.get_dense_pe()?;
        let empty_sparse = Tensor::zeros(
            (mask_input.dim(0)?, 0, self.config.hidden_dim),
            DType::F32,
            mask_input.device(),
        )?;
        let dense_embeddings = if mask_prompt.rank() == 4 {
            mask_prompt
        } else {
            mask_prompt.unsqueeze(1)?
        };
        let mut sam = self.forward_sam_heads(
            backbone_features,
            &image_pe,
            &empty_sparse,
            &dense_embeddings,
            false,
            high_res_features,
        )?;
        let is_obj_appearing = mask_input
            .flatten_from(1)?
            .to_dtype(DType::F32)?
            .sum_keepdim(1)?
            .gt(0f32)?;
        let lambda = is_obj_appearing.to_dtype(DType::F32)?;
        sam.state.obj_ptr = lambda.broadcast_mul(&sam.state.obj_ptr)?.broadcast_add(
            &(lambda
                .affine(-1.0, 1.0)?
                .broadcast_mul(&self.no_obj_ptr.broadcast_as(sam.state.obj_ptr.shape())?)?),
        )?;
        sam.state.low_res_masks = low_res_masks;
        sam.state.high_res_masks = high_res_masks;
        sam.state.object_score_logits = lambda.affine(20.0, -10.0)?;
        Ok(sam)
    }

    fn forward_sam_heads(
        &self,
        backbone_features: &Tensor,
        image_pe: &Tensor,
        sparse_prompt_embeddings: &Tensor,
        dense_prompt_embeddings: &Tensor,
        multimask_output: bool,
        high_res_features: Option<(&Tensor, &Tensor)>,
    ) -> Result<TrackerStepOutput> {
        let decoded = self.mask_decoder.forward(
            backbone_features,
            image_pe,
            sparse_prompt_embeddings,
            dense_prompt_embeddings,
            multimask_output,
            high_res_features,
        )?;
        let object_score_logits = decoded.object_score_logits;
        let is_obj_appearing = object_score_logits.gt(0f32)?;
        let low_res_multimasks = is_obj_appearing
            .reshape((object_score_logits.dim(0)?, 1, 1, 1))?
            .broadcast_as(decoded.low_res_multimasks.shape())?
            .where_cond(
                &decoded.low_res_multimasks,
                &decoded
                    .low_res_multimasks
                    .affine(0.0, NO_OBJ_SCORE as f64)?,
            )?;
        let high_res_multimasks = is_obj_appearing
            .reshape((object_score_logits.dim(0)?, 1, 1, 1))?
            .broadcast_as(decoded.high_res_multimasks.shape())?
            .where_cond(
                &decoded.high_res_multimasks,
                &decoded
                    .high_res_multimasks
                    .affine(0.0, NO_OBJ_SCORE as f64)?,
            )?;

        let (low_res_masks, high_res_masks, iou_scores, sam_output_token) = if multimask_output {
            let best_iou_inds = decoded.iou_pred.argmax(1)?;
            let best_iou_inds = best_iou_inds.flatten_all()?.to_vec1::<u32>()?;
            let mut low = Vec::with_capacity(best_iou_inds.len());
            let mut high = Vec::with_capacity(best_iou_inds.len());
            let mut iou = Vec::with_capacity(best_iou_inds.len());
            let mut token = Vec::with_capacity(best_iou_inds.len());
            for (batch_idx, best_idx) in best_iou_inds.into_iter().enumerate() {
                low.push(low_res_multimasks.i((batch_idx, best_idx as usize))?);
                high.push(high_res_multimasks.i((batch_idx, best_idx as usize))?);
                iou.push(decoded.iou_pred.i((batch_idx, best_idx as usize))?);
                token.push(decoded.sam_tokens_out.i((batch_idx, best_idx as usize))?);
            }
            (
                Tensor::stack(&low.iter().collect::<Vec<_>>(), 0)?.unsqueeze(1)?,
                Tensor::stack(&high.iter().collect::<Vec<_>>(), 0)?.unsqueeze(1)?,
                Tensor::stack(&iou.iter().collect::<Vec<_>>(), 0)?,
                Tensor::stack(&token.iter().collect::<Vec<_>>(), 0)?,
            )
        } else {
            (
                low_res_multimasks.i((.., 0..1))?,
                high_res_multimasks.i((.., 0..1))?,
                decoded.iou_pred.i((.., 0..1))?,
                decoded.sam_tokens_out.i((.., 0))?,
            )
        };
        let lambda = is_obj_appearing.to_dtype(DType::F32)?;
        let obj_ptr = self.obj_ptr_proj.forward(&sam_output_token)?;
        let obj_ptr = lambda.broadcast_mul(&obj_ptr)?.broadcast_add(
            &(lambda
                .affine(-1.0, 1.0)?
                .broadcast_mul(&self.no_obj_ptr.broadcast_as(obj_ptr.shape())?)?),
        )?;

        Ok(TrackerStepOutput {
            state: TrackerFrameState {
                low_res_masks,
                high_res_masks,
                iou_scores,
                obj_ptr,
                object_score_logits,
                maskmem_features: None,
                maskmem_pos_enc: None,
                is_cond_frame: false,
            },
            prompt_frame_indices: Vec::new(),
            memory_frame_indices: Vec::new(),
        })
    }

    fn encode_new_memory(
        &self,
        low_res_feature: &Tensor,
        high_res_masks: &Tensor,
        object_score_logits: &Tensor,
        is_mask_from_points: bool,
    ) -> Result<(Tensor, Tensor)> {
        let mask_for_mem = if is_mask_from_points {
            high_res_masks.gt(0f32)?.to_dtype(DType::F32)?
        } else {
            candle_nn::ops::sigmoid(high_res_masks)?
        };
        let mask_for_mem = mask_for_mem.affine(20.0, -10.0)?;
        let (mut maskmem_features, maskmem_pos_enc) = self
            .maskmem_backbone
            .forward(low_res_feature, &mask_for_mem)?;
        let is_obj_appearing = object_score_logits.gt(0f32)?.to_dtype(DType::F32)?;
        let no_obj = self
            .no_obj_embed_spatial
            .reshape((1, self.config.memory_dim, 1, 1))?
            .broadcast_as(maskmem_features.shape())?;
        let no_obj_bias = is_obj_appearing
            .affine(-1.0, 1.0)?
            .reshape((is_obj_appearing.dim(0)?, 1, 1, 1))?
            .broadcast_mul(&no_obj)?;
        maskmem_features = maskmem_features.broadcast_add(&no_obj_bias)?;
        Ok((maskmem_features, maskmem_pos_enc))
    }

    fn prepare_memory_conditioned_features(
        &self,
        frame_idx: usize,
        num_frames: usize,
        is_init_cond_frame: bool,
        current_vision_feats: &Tensor,
        current_vision_pos: &Tensor,
        history: &BTreeMap<usize, TrackerFrameState>,
        reverse: bool,
        use_prev_mem_frame: bool,
    ) -> Result<(Tensor, Vec<usize>, Vec<usize>, usize)> {
        let batch_size = current_vision_feats.dim(0)?;
        let current_low_res =
            if self.config.num_maskmem == 0 || is_init_cond_frame || !use_prev_mem_frame {
                let pix = current_vision_feats.broadcast_add(
                    &self
                        .no_mem_embed
                        .broadcast_as(current_vision_feats.shape())?,
                )?;
                return Ok((
                    pix.permute((0, 2, 1))?.reshape((
                        batch_size,
                        self.config.hidden_dim,
                        self.config.image_embedding_size(),
                        self.config.image_embedding_size(),
                    ))?,
                    Vec::new(),
                    Vec::new(),
                    0,
                ));
            } else {
                current_vision_feats.clone()
            };
        let cond_outputs: BTreeMap<usize, &TrackerFrameState> = history
            .iter()
            .filter_map(|(idx, state)| state.is_cond_frame.then_some((*idx, state)))
            .collect();
        let (selected_cond_outputs, unselected_cond_outputs) = select_closest_cond_frames(
            frame_idx,
            &cond_outputs,
            self.config.max_cond_frames_in_attn,
        );

        let mut prompt_frame_indices = selected_cond_outputs.keys().copied().collect::<Vec<_>>();
        let mut memory_frame_indices = Vec::new();
        let mut prompt_tokens = Vec::new();
        let mut prompt_pos_tokens = Vec::new();

        let mut t_pos_and_states = selected_cond_outputs
            .iter()
            .map(|(idx, state)| (0usize, *idx, *state, true))
            .collect::<Vec<_>>();

        for t_pos in 1..self.config.num_maskmem {
            let t_rel = self.config.num_maskmem - t_pos;
            let prev_frame_idx = if reverse {
                frame_idx + t_rel
            } else if frame_idx >= t_rel {
                frame_idx - t_rel
            } else {
                continue;
            };
            if prev_frame_idx >= num_frames {
                continue;
            }
            let prev = history
                .get(&prev_frame_idx)
                .or_else(|| unselected_cond_outputs.get(&prev_frame_idx).copied());
            if let Some(prev) = prev {
                t_pos_and_states.push((t_pos, prev_frame_idx, prev, false));
            }
        }

        for (t_pos, prev_frame_idx, prev, _is_selected_cond_frame) in t_pos_and_states.iter() {
            let Some(maskmem_features) = prev.maskmem_features.as_ref() else {
                continue;
            };
            let Some(maskmem_pos_enc) = prev.maskmem_pos_enc.as_ref() else {
                continue;
            };
            prompt_tokens.push(maskmem_features.flatten_from(2)?.transpose(1, 2)?);
            let pos = maskmem_pos_enc.flatten_from(2)?.transpose(1, 2)?;
            let tpos = self
                .maskmem_tpos_enc
                .i(self.config.num_maskmem - *t_pos - 1)?
                .broadcast_as(pos.shape())?;
            prompt_pos_tokens.push(pos.broadcast_add(&tpos)?);
            if *t_pos > 0 {
                memory_frame_indices.push(*prev_frame_idx);
            }
        }

        prompt_frame_indices.sort_unstable();
        memory_frame_indices.sort_unstable();

        let mut pos_and_ptrs = selected_cond_outputs
            .iter()
            .filter(|(idx, _)| {
                if reverse {
                    **idx >= frame_idx
                } else {
                    **idx <= frame_idx
                }
            })
            .map(|(idx, state)| (frame_distance(frame_idx, *idx), state.obj_ptr.clone()))
            .collect::<Vec<_>>();
        for t_diff in 1..self.config.max_obj_ptrs_in_encoder.min(num_frames) {
            let prev_frame_idx = if reverse {
                frame_idx + t_diff
            } else if frame_idx >= t_diff {
                frame_idx - t_diff
            } else {
                break;
            };
            if prev_frame_idx >= num_frames {
                break;
            }
            if let Some(prev) = history
                .get(&prev_frame_idx)
                .or_else(|| unselected_cond_outputs.get(&prev_frame_idx).copied())
            {
                pos_and_ptrs.push((t_diff, prev.obj_ptr.clone()));
            }
        }

        let mut num_obj_ptr_tokens = 0usize;
        if !pos_and_ptrs.is_empty() {
            let mut pos_values = Vec::with_capacity(pos_and_ptrs.len());
            let mut ptrs = Vec::with_capacity(pos_and_ptrs.len());
            for (pos, ptr) in pos_and_ptrs.iter() {
                pos_values.push(*pos as f32);
                ptrs.push(ptr.clone());
            }
            let obj_ptrs = Tensor::stack(&ptrs.iter().collect::<Vec<_>>(), 1)?;
            let obj_pos = self.get_obj_ptr_tpos(&pos_values, obj_ptrs.device())?;
            if self.config.memory_dim < self.config.hidden_dim {
                let splits = self.config.hidden_dim / self.config.memory_dim;
                let obj_ptrs = obj_ptrs.reshape((
                    obj_ptrs.dim(0)?,
                    obj_ptrs.dim(1)?,
                    splits,
                    self.config.memory_dim,
                ))?;
                let obj_ptrs = obj_ptrs.flatten(1, 2)?;
                let obj_pos = repeat_interleave(&obj_pos, splits, 1)?;
                num_obj_ptr_tokens = obj_ptrs.dim(1)?;
                prompt_tokens.push(obj_ptrs);
                prompt_pos_tokens.push(obj_pos);
            } else {
                num_obj_ptr_tokens = obj_ptrs.dim(1)?;
                prompt_tokens.push(obj_ptrs);
                prompt_pos_tokens.push(obj_pos);
            }
        }

        let prompt = if prompt_tokens.is_empty() {
            self.no_mem_embed
                .expand((batch_size, 1, self.config.hidden_dim))?
        } else {
            Tensor::cat(&prompt_tokens.iter().collect::<Vec<_>>(), 1)?
        };
        let prompt_pos = if prompt_pos_tokens.is_empty() {
            self.no_mem_pos_enc
                .expand((batch_size, 1, self.config.hidden_dim))?
        } else {
            Tensor::cat(&prompt_pos_tokens.iter().collect::<Vec<_>>(), 1)?
        };
        let current = self.transformer.forward(
            &current_low_res,
            current_vision_pos,
            &prompt,
            &prompt_pos,
            num_obj_ptr_tokens,
        )?;
        Ok((
            current.permute((0, 2, 1))?.reshape((
                batch_size,
                self.config.hidden_dim,
                self.config.image_embedding_size(),
                self.config.image_embedding_size(),
            ))?,
            prompt_frame_indices,
            memory_frame_indices,
            num_obj_ptr_tokens,
        ))
    }

    fn get_obj_ptr_tpos(&self, positions: &[f32], device: &candle::Device) -> Result<Tensor> {
        let pe = get_1d_sine_pe(positions, self.config.hidden_dim, device)?;
        let pe = self.obj_ptr_tpos_proj.forward(&pe.contiguous()?)?;
        pe.unsqueeze(0)
    }
}

fn frame_distance(current: usize, other: usize) -> usize {
    current.abs_diff(other)
}

fn build_2d_sine_position_encoding(feature: &Tensor, d_model: usize) -> Result<Tensor> {
    let (batch_size, channels, height, width) = feature.dims4()?;
    if channels != d_model {
        candle::bail!(
            "tracker position encoding expected width {}, got {}",
            d_model,
            channels
        );
    }
    if d_model % 4 != 0 {
        candle::bail!(
            "tracker position encoding requires width divisible by 4, got {}",
            d_model
        );
    }
    let num_pos_feats = d_model / 2;
    let device = feature.device();
    let dtype = feature.dtype();
    let temperature = 10_000f32;
    let scale = 2.0 * std::f32::consts::PI;
    let eps = 1e-6f32;
    let mut dim_t = Vec::with_capacity(num_pos_feats);
    for idx in 0..num_pos_feats {
        let exponent = 2.0 * (idx / 2) as f32 / num_pos_feats as f32;
        dim_t.push(temperature.powf(exponent));
    }
    let mut encoding = vec![0f32; d_model * height * width];
    for y in 0..height {
        let y_pos = ((y + 1) as f32 / (height as f32 + eps)) * scale;
        for x in 0..width {
            let x_pos = ((x + 1) as f32 / (width as f32 + eps)) * scale;
            for idx in 0..num_pos_feats {
                let div = dim_t[idx];
                let y_value = if idx % 2 == 0 {
                    (y_pos / div).sin()
                } else {
                    (y_pos / div).cos()
                };
                let x_value = if idx % 2 == 0 {
                    (x_pos / div).sin()
                } else {
                    (x_pos / div).cos()
                };
                let spatial_index = y * width + x;
                encoding[idx * height * width + spatial_index] = y_value;
                encoding[(num_pos_feats + idx) * height * width + spatial_index] = x_value;
            }
        }
    }
    let encoding = Tensor::from_slice(&encoding, (1, d_model, height, width), device)?;
    encoding.repeat((batch_size, 1, 1, 1))?.to_dtype(dtype)
}

fn select_closest_cond_frames<'a>(
    frame_idx: usize,
    cond_frame_outputs: &'a BTreeMap<usize, &'a TrackerFrameState>,
    max_cond_frame_num: usize,
) -> (
    BTreeMap<usize, &'a TrackerFrameState>,
    BTreeMap<usize, &'a TrackerFrameState>,
) {
    if cond_frame_outputs.len() <= max_cond_frame_num {
        return (cond_frame_outputs.clone(), BTreeMap::new());
    }
    let mut selected = BTreeMap::new();
    if let Some(idx_before) = cond_frame_outputs
        .keys()
        .copied()
        .filter(|idx| *idx < frame_idx)
        .max()
    {
        if let Some(state) = cond_frame_outputs.get(&idx_before) {
            selected.insert(idx_before, *state);
        }
    }
    if let Some(idx_after) = cond_frame_outputs
        .keys()
        .copied()
        .filter(|idx| *idx >= frame_idx)
        .min()
    {
        if let Some(state) = cond_frame_outputs.get(&idx_after) {
            selected.insert(idx_after, *state);
        }
    }
    let num_remaining = max_cond_frame_num.saturating_sub(selected.len());
    let mut remaining = cond_frame_outputs
        .keys()
        .copied()
        .filter(|idx| !selected.contains_key(idx))
        .collect::<Vec<_>>();
    remaining.sort_by_key(|idx| frame_distance(frame_idx, *idx));
    for idx in remaining.into_iter().take(num_remaining) {
        if let Some(state) = cond_frame_outputs.get(&idx) {
            selected.insert(idx, *state);
        }
    }
    let unselected = cond_frame_outputs
        .iter()
        .filter_map(|(idx, state)| (!selected.contains_key(idx)).then_some((*idx, *state)))
        .collect();
    (selected, unselected)
}

fn get_1d_sine_pe(positions: &[f32], dim: usize, device: &candle::Device) -> Result<Tensor> {
    let pe_dim = dim / 2;
    let temperature = 10_000f32;
    let mut dim_t = Vec::with_capacity(pe_dim);
    for idx in 0..pe_dim {
        dim_t.push(temperature.powf(2.0 * (idx / 2) as f32 / pe_dim as f32));
    }
    let mut values = vec![0f32; positions.len() * dim];
    for (row_idx, pos) in positions.iter().copied().enumerate() {
        for col_idx in 0..pe_dim {
            let value = pos / dim_t[col_idx];
            values[row_idx * dim + col_idx] = value.sin();
            values[row_idx * dim + pe_dim + col_idx] = value.cos();
        }
    }
    Tensor::from_slice(&values, (positions.len(), dim), device)
}

fn repeat_interleave(xs: &Tensor, repeats: usize, dim: usize) -> Result<Tensor> {
    let xs = xs.unsqueeze(dim + 1)?;
    let mut dims = xs.dims().to_vec();
    dims[dim + 1] = repeats;
    xs.broadcast_as(dims)?.flatten(dim, dim + 1)
}
