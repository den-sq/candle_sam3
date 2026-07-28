use std::collections::BTreeMap;

use candle::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, LayerNorm, Linear, Module, VarBuilder};

use super::{
    config::VisionConfig,
    torch_ops::window::{window_partition_nhwc, window_unpartition_nhwc},
};

#[derive(Debug)]
pub struct ViTDetTrunkOutput {
    /// Final spatial feature map from the ViT trunk in `[batch, height, width, channels]` layout.
    pub stage_features: Vec<Tensor>,
}

#[derive(Debug)]
struct PatchEmbed {
    proj: Conv2d,
}

impl PatchEmbed {
    fn new(config: &VisionConfig, vb: VarBuilder) -> Result<Self> {
        let conv_cfg = Conv2dConfig {
            stride: config.patch_size,
            ..Default::default()
        };
        let proj = candle_nn::conv2d_no_bias(
            3,
            config.embed_dim,
            config.patch_size,
            conv_cfg,
            vb.pp("proj"),
        )?;
        Ok(Self { proj })
    }

    fn forward(&self, images: &Tensor) -> Result<Tensor> {
        self.proj.forward(images)?.permute((0, 2, 3, 1))
    }
}

#[derive(Debug)]
struct VisionRotaryEmbedding {
    freqs_real: Tensor,
    freqs_imag: Tensor,
}

impl VisionRotaryEmbedding {
    fn new(
        config: &VisionConfig,
        end_x: usize,
        end_y: usize,
        scale: f32,
        device: &candle::Device,
    ) -> Result<Self> {
        let head_dim = config.embed_dim / config.num_heads;
        if head_dim % 4 != 0 {
            candle::bail!("sam3 vision head dim must be divisible by 4, got {head_dim}")
        }
        let rotary_dim = head_dim / 4;
        let seq_len = end_x * end_y;
        let inv_freqs: Vec<f32> = (0..rotary_dim)
            .map(|i| 1f32 / (config.rope_theta as f32).powf((4 * i) as f32 / head_dim as f32))
            .collect();
        let mut freqs_real = vec![0f32; seq_len * (head_dim / 2)];
        let mut freqs_imag = vec![0f32; seq_len * (head_dim / 2)];
        for flat_idx in 0..seq_len {
            let x_pos = (flat_idx % end_x) as f32 * scale;
            let y_pos = (flat_idx / end_x) as f32 * scale;
            let row_real =
                &mut freqs_real[flat_idx * (head_dim / 2)..(flat_idx + 1) * (head_dim / 2)];
            let row_imag =
                &mut freqs_imag[flat_idx * (head_dim / 2)..(flat_idx + 1) * (head_dim / 2)];
            for (i, inv_freq) in inv_freqs.iter().copied().enumerate() {
                let x_freq = x_pos * inv_freq;
                let y_freq = y_pos * inv_freq;
                row_real[i] = x_freq.cos();
                row_imag[i] = x_freq.sin();
                row_real[rotary_dim + i] = y_freq.cos();
                row_imag[rotary_dim + i] = y_freq.sin();
            }
        }
        Ok(Self {
            freqs_real: Tensor::from_slice(&freqs_real, (seq_len, head_dim / 2), device)?,
            freqs_imag: Tensor::from_slice(&freqs_imag, (seq_len, head_dim / 2), device)?,
        })
    }

    fn apply(&self, q: &Tensor, k: &Tensor) -> Result<(Tensor, Tensor)> {
        let (_, _, seq_len, head_dim) = q.dims4()?;
        let freqs_real =
            self.freqs_real
                .narrow(0, 0, seq_len)?
                .reshape((1, 1, seq_len, head_dim / 2))?;
        let freqs_imag =
            self.freqs_imag
                .narrow(0, 0, seq_len)?
                .reshape((1, 1, seq_len, head_dim / 2))?;
        Ok((
            apply_rotary_enc_real(q, &freqs_real, &freqs_imag)?,
            apply_rotary_enc_real(k, &freqs_real, &freqs_imag)?,
        ))
    }
}

fn apply_rotary_enc_real(xs: &Tensor, freqs_real: &Tensor, freqs_imag: &Tensor) -> Result<Tensor> {
    let (batch_size, num_heads, seq_len, head_dim) = xs.dims4()?;
    let xs_dtype = xs.dtype();
    let xs = xs
        .to_dtype(DType::F32)?
        .reshape((batch_size, num_heads, seq_len, head_dim / 2, 2))?;
    let xs_real = xs.i((.., .., .., .., 0))?;
    let xs_imag = xs.i((.., .., .., .., 1))?;
    let real = (xs_real.broadcast_mul(freqs_real)? - xs_imag.broadcast_mul(freqs_imag)?)?;
    let imag = (xs_real.broadcast_mul(freqs_imag)? + xs_imag.broadcast_mul(freqs_real)?)?;
    Tensor::stack(&[&real, &imag], 4)?
        .reshape((batch_size, num_heads, seq_len, head_dim))?
        .to_dtype(xs_dtype)
}

#[derive(Debug)]
struct Sam3VisionAttention {
    qkv: Linear,
    proj: Linear,
    num_heads: usize,
    head_dim: usize,
    scale: Tensor,
    scale_value: f32,
    rotary_emb: VisionRotaryEmbedding,
}

fn f32_sm75_sdpa_supported(_device: &Device) -> bool {
    #[cfg(feature = "cuda")]
    if let Device::Cuda(device) = _device {
        use candle::cuda_backend::cudarc::driver::{result, sys};

        let cuda_device = device.cuda_stream().context().cu_device();
        let attribute =
            |attribute| unsafe { result::device::get_attribute(cuda_device, attribute) };
        let major =
            attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR);
        let minor =
            attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR);
        return matches!((major, minor), (Ok(7), Ok(5)));
    }

    false
}

impl Sam3VisionAttention {
    fn new(
        config: &VisionConfig,
        rotary_emb: VisionRotaryEmbedding,
        vb: VarBuilder,
    ) -> Result<Self> {
        let qkv = candle_nn::linear_b(config.embed_dim, config.embed_dim * 3, true, vb.pp("qkv"))?;
        let proj = candle_nn::linear_b(config.embed_dim, config.embed_dim, true, vb.pp("proj"))?;
        let head_dim = config.embed_dim / config.num_heads;
        let scale_value = (head_dim as f32).powf(-0.5);
        Ok(Self {
            qkv,
            proj,
            num_heads: config.num_heads,
            head_dim,
            scale: Tensor::new(scale_value, vb.device())?,
            scale_value,
            rotary_emb,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let in_dtype = hidden_states.dtype();
        let (batch_size, height, width, channels) = hidden_states.dims4()?;
        let seq_len = height * width;
        let qkv = self
            .qkv
            .forward(&hidden_states.contiguous()?)?
            .reshape((batch_size, seq_len, 3, self.num_heads, self.head_dim))?
            .permute((2, 0, 3, 1, 4))?
            .contiguous()?;
        let q = qkv.i(0)?;
        let k = qkv.i(1)?;
        let v = qkv.i(2)?;
        let (q, k) = self.rotary_emb.apply(&q, &k)?;
        let q = q.to_dtype(DType::F32)?;
        let k = k.to_dtype(DType::F32)?;
        let v = v.to_dtype(DType::F32)?;
        let hidden_states = if in_dtype == DType::F32
            && f32_sm75_sdpa_supported(q.device())
            && self.head_dim == 64
            && q.is_contiguous()
            && k.is_contiguous()
            && v.is_contiguous()
        {
            candle_nn::ops::sdpa(&q, &k, &v, None, false, self.scale_value, 1.)?
        } else {
            let q = q.broadcast_mul(&self.scale)?;
            let attn = q.matmul(&k.transpose(2, 3)?)?;
            candle_nn::ops::softmax_last_dim(&attn)?.matmul(&v)?
        };
        let hidden_states = hidden_states
            .to_dtype(in_dtype)?
            .transpose(1, 2)?
            .reshape((batch_size, height, width, channels))?;
        self.proj.forward(&hidden_states)
    }
}

#[derive(Debug)]
struct Sam3VisionMlp {
    fc1: Linear,
    fc2: Linear,
}

impl Sam3VisionMlp {
    fn new(config: &VisionConfig, vb: VarBuilder) -> Result<Self> {
        let hidden_dim = ((config.embed_dim as f64) * config.mlp_ratio) as usize;
        Ok(Self {
            fc1: candle_nn::linear_b(config.embed_dim, hidden_dim, true, vb.pp("fc1"))?,
            fc2: candle_nn::linear_b(hidden_dim, config.embed_dim, true, vb.pp("fc2"))?,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let hidden_states = self.fc1.forward(&hidden_states.contiguous()?)?.gelu_erf()?;
        self.fc2.forward(&hidden_states.contiguous()?)
    }

    fn forward_with_debug(&self, hidden_states: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let fc1 = self.fc1.forward(&hidden_states.contiguous()?)?;
        let gelu = fc1.gelu_erf()?;
        let output = self.fc2.forward(&gelu.contiguous()?)?;
        Ok((fc1, gelu, output))
    }
}

#[derive(Debug)]
struct Sam3VisionBlock {
    norm1: LayerNorm,
    attn: Sam3VisionAttention,
    norm2: LayerNorm,
    mlp: Sam3VisionMlp,
    window_size: usize,
}

impl Sam3VisionBlock {
    fn new(
        config: &VisionConfig,
        window_size: usize,
        input_size: (usize, usize),
        vb: VarBuilder,
    ) -> Result<Self> {
        let rotary_input_size = if window_size == 0 {
            input_size
        } else {
            (window_size, window_size)
        };
        let rotary_scale = config.rope_pt_size as f32 / rotary_input_size.0 as f32;
        Ok(Self {
            norm1: candle_nn::layer_norm(config.embed_dim, 1e-5, vb.pp("norm1"))?,
            attn: Sam3VisionAttention::new(
                config,
                VisionRotaryEmbedding::new(
                    config,
                    rotary_input_size.0,
                    rotary_input_size.1,
                    rotary_scale,
                    vb.device(),
                )?,
                vb.pp("attn"),
            )?,
            norm2: candle_nn::layer_norm(config.embed_dim, 1e-5, vb.pp("norm2"))?,
            mlp: Sam3VisionMlp::new(config, vb.pp("mlp"))?,
            window_size,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let residual = hidden_states;
        let hidden_states = self.norm1.forward(hidden_states)?;
        let original_hw = (hidden_states.dim(1)?, hidden_states.dim(2)?);
        let (hidden_states, padded_hw) = if self.window_size > 0 {
            window_partition_nhwc(&hidden_states, self.window_size)?
        } else {
            (hidden_states, original_hw)
        };
        let hidden_states = self.attn.forward(&hidden_states)?;
        let hidden_states = if self.window_size > 0 {
            window_unpartition_nhwc(&hidden_states, self.window_size, padded_hw, original_hw)?
        } else {
            hidden_states
        };
        let hidden_states = (residual + hidden_states)?.contiguous()?;
        let residual = &hidden_states;
        let hidden_states = self.norm2.forward(&hidden_states)?;
        let hidden_states = self.mlp.forward(&hidden_states)?;
        (residual + hidden_states)?.contiguous()
    }

    fn forward_windowed(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let residual = hidden_states;
        let hidden_states = self.norm1.forward(hidden_states)?;
        let hidden_states = self.attn.forward(&hidden_states)?;
        let hidden_states = (residual + hidden_states)?.contiguous()?;
        let residual = &hidden_states;
        let hidden_states = self.norm2.forward(&hidden_states)?;
        let hidden_states = self.mlp.forward(&hidden_states)?;
        (residual + hidden_states)?.contiguous()
    }

    fn forward_with_debug(
        &self,
        hidden_states: &Tensor,
        block_index: usize,
    ) -> Result<(Tensor, BTreeMap<String, Tensor>)> {
        let mut debug = BTreeMap::new();
        debug.insert(
            format!("vision.block_debug.{block_index}.input"),
            hidden_states.clone(),
        );

        let residual = hidden_states;
        let hidden_states = self.norm1.forward(hidden_states)?;
        debug.insert(
            format!("vision.block_debug.{block_index}.norm1"),
            hidden_states.clone(),
        );

        let original_hw = (hidden_states.dim(1)?, hidden_states.dim(2)?);
        let (hidden_states, padded_hw) = if self.window_size > 0 {
            window_partition_nhwc(&hidden_states, self.window_size)?
        } else {
            (hidden_states, original_hw)
        };
        let hidden_states = self.attn.forward(&hidden_states)?;
        let hidden_states = if self.window_size > 0 {
            window_unpartition_nhwc(&hidden_states, self.window_size, padded_hw, original_hw)?
        } else {
            hidden_states
        };
        debug.insert(
            format!("vision.block_debug.{block_index}.attn_output"),
            hidden_states.clone(),
        );

        let hidden_states = (residual + hidden_states)?.contiguous()?;
        debug.insert(
            format!("vision.block_debug.{block_index}.post_attn"),
            hidden_states.clone(),
        );

        let residual = &hidden_states;
        let hidden_states = self.norm2.forward(&hidden_states)?;
        debug.insert(
            format!("vision.block_debug.{block_index}.norm2"),
            hidden_states.clone(),
        );

        let (mlp_fc1, mlp_gelu, mlp_output) = self.mlp.forward_with_debug(&hidden_states)?;
        debug.insert(
            format!("vision.block_debug.{block_index}.mlp_fc1"),
            mlp_fc1.clone(),
        );
        debug.insert(
            format!("vision.block_debug.{block_index}.mlp_gelu"),
            mlp_gelu.clone(),
        );
        debug.insert(
            format!("vision.block_debug.{block_index}.mlp_output"),
            mlp_output.clone(),
        );

        let hidden_states = (residual + mlp_output)?.contiguous()?;
        debug.insert(
            format!("vision.block_debug.{block_index}.output"),
            hidden_states.clone(),
        );
        Ok((hidden_states, debug))
    }
}

fn load_pos_embed(config: &VisionConfig, vb: VarBuilder) -> Result<Tensor> {
    let pretrain_grid = config.pretrain_image_size / config.patch_size;
    let no_cls_shape = (1, pretrain_grid * pretrain_grid, config.embed_dim);
    let with_cls_shape = (1, pretrain_grid * pretrain_grid + 1, config.embed_dim);
    match vb.get(with_cls_shape, "pos_embed") {
        Ok(pos_embed) => Ok(pos_embed),
        Err(_) => vb.get(no_cls_shape, "pos_embed"),
    }
}

fn strip_cls_position_embedding(pos_embed: &Tensor) -> Result<Tensor> {
    let (batch_size, tokens, hidden_size) = pos_embed.dims3()?;
    if batch_size != 1 {
        candle::bail!("sam3 vision pos_embed expected batch dimension 1, got {batch_size}")
    }
    let square = (tokens as f64).sqrt() as usize;
    if square * square == tokens {
        return Ok(pos_embed.clone());
    }
    let square_without_cls = ((tokens - 1) as f64).sqrt() as usize;
    if square_without_cls * square_without_cls == tokens - 1 {
        return pos_embed.narrow(1, 1, tokens - 1);
    }
    candle::bail!(
        "sam3 vision pos_embed length {tokens} is neither square nor square-plus-cls for hidden size {hidden_size}"
    )
}

fn tile_position_embeddings(
    pos_embed: &Tensor,
    target_height: usize,
    target_width: usize,
) -> Result<Tensor> {
    let pos_embed = strip_cls_position_embedding(pos_embed)?;
    let (_, tokens, hidden_size) = pos_embed.dims3()?;
    let pretrain_size = (tokens as f64).sqrt() as usize;
    if pretrain_size * pretrain_size != tokens {
        candle::bail!("sam3 vision pos_embed length {tokens} is not square after cls stripping")
    }
    let pos_embed = pos_embed
        .reshape((1, pretrain_size, pretrain_size, hidden_size))?
        .permute((0, 3, 1, 2))?;
    let repeat_h = target_height.div_ceil(pretrain_size);
    let repeat_w = target_width.div_ceil(pretrain_size);
    let pos_embed = if repeat_h == 1 && repeat_w == 1 {
        pos_embed
    } else {
        pos_embed.repeat((1, 1, repeat_h, repeat_w))?
    };
    pos_embed
        .narrow(2, 0, target_height)?
        .narrow(3, 0, target_width)?
        .permute((0, 2, 3, 1))
}

#[derive(Debug)]
struct Sam3TiledPositionEmbeddings {
    source: Tensor,
    cached: Tensor,
    cached_hw: (usize, usize),
}

impl Sam3TiledPositionEmbeddings {
    fn new(source: Tensor, cached_hw: (usize, usize)) -> Result<Self> {
        let cached = tile_position_embeddings(&source, cached_hw.0, cached_hw.1)?;
        Ok(Self {
            source,
            cached,
            cached_hw,
        })
    }

    fn get(&self, target_height: usize, target_width: usize) -> Result<Tensor> {
        if (target_height, target_width) == self.cached_hw {
            Ok(self.cached.clone())
        } else {
            tile_position_embeddings(&self.source, target_height, target_width)
        }
    }
}

#[derive(Debug)]
pub struct Sam3ViTDetTrunk {
    config: VisionConfig,
    patch_embed: PatchEmbed,
    pos_embed: Option<Sam3TiledPositionEmbeddings>,
    blocks: Vec<Sam3VisionBlock>,
    pre_layer_norm: Option<LayerNorm>,
}

impl Sam3ViTDetTrunk {
    pub fn new(config: &VisionConfig, vb: VarBuilder) -> Result<Self> {
        let patch_embed = PatchEmbed::new(config, vb.pp("patch_embed"))?;
        let input_size = (
            config.image_size / config.patch_size,
            config.image_size / config.patch_size,
        );
        let pos_embed = if config.use_abs_pos {
            Some(Sam3TiledPositionEmbeddings::new(
                load_pos_embed(config, vb.clone())?,
                input_size,
            )?)
        } else {
            None
        };
        let pre_layer_norm = if config.ln_pre && vb.contains_tensor("ln_pre.weight") {
            Some(candle_nn::layer_norm(
                config.embed_dim,
                1e-5,
                vb.pp("ln_pre"),
            )?)
        } else {
            None
        };
        let block_vb = vb.pp("blocks");
        let mut blocks = Vec::with_capacity(config.depth);
        for layer_idx in 0..config.depth {
            let window_size = if config.global_attn_blocks.contains(&layer_idx) {
                0
            } else {
                config.window_size
            };
            blocks.push(Sam3VisionBlock::new(
                config,
                window_size,
                input_size,
                block_vb.pp(layer_idx),
            )?);
        }
        Ok(Self {
            config: config.clone(),
            patch_embed,
            pos_embed,
            blocks,
            pre_layer_norm,
        })
    }

    pub fn config(&self) -> &VisionConfig {
        &self.config
    }

    pub fn forward(&self, images: &Tensor) -> Result<ViTDetTrunkOutput> {
        let (output, _, _) = self.forward_impl(images, false, &[])?;
        Ok(output)
    }

    pub fn forward_with_block_outputs(
        &self,
        images: &Tensor,
    ) -> Result<(ViTDetTrunkOutput, Vec<Tensor>)> {
        let (output, block_outputs, _) = self.forward_impl(images, true, &[])?;
        Ok((output, block_outputs.unwrap_or_default()))
    }

    pub fn forward_with_debug_blocks(
        &self,
        images: &Tensor,
        debug_blocks: &[usize],
    ) -> Result<(ViTDetTrunkOutput, Vec<Tensor>, BTreeMap<String, Tensor>)> {
        let (output, block_outputs, debug_tensors) =
            self.forward_impl(images, true, debug_blocks)?;
        Ok((output, block_outputs.unwrap_or_default(), debug_tensors))
    }

    pub(crate) fn forward_block_with_debug_from_hidden_states(
        &self,
        hidden_states: &Tensor,
        block_index: usize,
    ) -> Result<(Tensor, BTreeMap<String, Tensor>)> {
        let block = self.blocks.get(block_index).ok_or_else(|| {
            candle::Error::Msg(format!("sam3 vision block {block_index} is out of range"))
        })?;
        block.forward_with_debug(hidden_states, block_index)
    }

    fn forward_blocks_fast(&self, mut hidden_states: Tensor) -> Result<Tensor> {
        let mut block_index = 0;
        while block_index < self.blocks.len() {
            let block = &self.blocks[block_index];
            if block.window_size == 0 {
                hidden_states = block.forward(&hidden_states)?;
                block_index += 1;
                continue;
            }

            let window_size = block.window_size;
            let original_hw = (hidden_states.dim(1)?, hidden_states.dim(2)?);
            // Keep tensors partitioned across consecutive windowed blocks so we only
            // pay the NHWC<->window layout materialization once per run.
            let (mut windowed_states, padded_hw) =
                window_partition_nhwc(&hidden_states, window_size)?;
            while block_index < self.blocks.len() {
                let block = &self.blocks[block_index];
                if block.window_size != window_size {
                    break;
                }
                windowed_states = block.forward_windowed(&windowed_states)?;
                block_index += 1;
            }
            hidden_states =
                window_unpartition_nhwc(&windowed_states, window_size, padded_hw, original_hw)?;
        }
        Ok(hidden_states)
    }

    fn forward_impl(
        &self,
        images: &Tensor,
        collect_block_outputs: bool,
        debug_blocks: &[usize],
    ) -> Result<(
        ViTDetTrunkOutput,
        Option<Vec<Tensor>>,
        BTreeMap<String, Tensor>,
    )> {
        let (_, _, image_height, image_width) = images.dims4()?;
        if image_height % self.config.patch_size != 0 || image_width % self.config.patch_size != 0 {
            candle::bail!(
                "sam3 vision trunk expects image sizes divisible by patch size {}, got {image_height}x{image_width}",
                self.config.patch_size
            )
        }
        let patch_height = image_height / self.config.patch_size;
        let patch_width = image_width / self.config.patch_size;
        let mut debug_tensors = BTreeMap::new();
        let mut hidden_states = self.patch_embed.forward(images)?;
        if !debug_blocks.is_empty() {
            debug_tensors.insert(
                "vision.pre_block.patch_embed".to_owned(),
                hidden_states.clone(),
            );
        }
        if let Some(pos_embed) = &self.pos_embed {
            let pos_embed = pos_embed.get(patch_height, patch_width)?;
            hidden_states = hidden_states.broadcast_add(&pos_embed)?;
        }
        if !debug_blocks.is_empty() {
            debug_tensors.insert(
                "vision.pre_block.pos_embed_added".to_owned(),
                hidden_states.clone(),
            );
        }
        if let Some(pre_layer_norm) = &self.pre_layer_norm {
            hidden_states = pre_layer_norm.forward(&hidden_states)?;
        }
        if !debug_blocks.is_empty() {
            debug_tensors.insert("vision.pre_block.ln_pre".to_owned(), hidden_states.clone());
        }
        let mut block_outputs =
            collect_block_outputs.then(|| Vec::with_capacity(self.blocks.len()));
        if collect_block_outputs || !debug_blocks.is_empty() {
            for (block_index, block) in self.blocks.iter().enumerate() {
                if debug_blocks.contains(&block_index) {
                    let (next_hidden_states, block_debug) =
                        block.forward_with_debug(&hidden_states, block_index)?;
                    hidden_states = next_hidden_states;
                    debug_tensors.extend(block_debug);
                } else {
                    hidden_states = block.forward(&hidden_states)?;
                }

                if let Some(block_outputs) = block_outputs.as_mut() {
                    block_outputs.push(hidden_states.clone());
                }
            }
        } else {
            hidden_states = self.forward_blocks_fast(hidden_states)?;
        }
        Ok((
            ViTDetTrunkOutput {
                stage_features: vec![hidden_states],
            },
            block_outputs,
            debug_tensors,
        ))
    }
}

#[cfg(all(test, feature = "cuda"))]
mod cuda_attention_tests {
    use super::{VisionConfig, VisionRotaryEmbedding};
    use candle::{Device, Result, Tensor};

    fn explicit_sdpa(q: &Tensor, k: &Tensor, v: &Tensor, scale: f64) -> Result<Tensor> {
        let scores = (q * scale)?.matmul(&k.transpose(2, 3)?)?;
        candle_nn::ops::softmax_last_dim(&scores)?.matmul(v)
    }

    fn check_sam3_shape(
        batch: usize,
        sequence: usize,
        grid: usize,
        rotary_scale: f32,
    ) -> Result<()> {
        const HEADS: usize = 16;
        const HEAD_DIM: usize = 64;

        let device = Device::new_cuda(0)?;
        let config = VisionConfig::default();
        let rotary = VisionRotaryEmbedding::new(&config, grid, grid, rotary_scale, &device)?;
        let shape = (batch, HEADS, sequence, HEAD_DIM);
        let q = Tensor::randn(0f32, 0.2, shape, &device)?;
        let k = Tensor::randn(0f32, 0.2, shape, &device)?;
        let v = Tensor::randn(0f32, 0.2, shape, &device)?;
        let (q, k) = rotary.apply(&q, &k)?;
        let scale = (HEAD_DIM as f64).sqrt().recip();

        // Exercise the exact full SAM3 launch shape. The explicit comparison
        // uses one representative batch/head slice to avoid materializing the
        // global shape's one-gigabyte score tensor in a correctness test.
        let actual = candle_nn::ops::sdpa(&q, &k, &v, None, false, scale as f32, 1.)?;
        let q_ref = q.narrow(0, 0, 1)?.narrow(1, 0, 1)?;
        let k_ref = k.narrow(0, 0, 1)?.narrow(1, 0, 1)?;
        let v_ref = v.narrow(0, 0, 1)?.narrow(1, 0, 1)?;
        let expected = explicit_sdpa(&q_ref, &k_ref, &v_ref, scale)?;
        let actual = actual.narrow(0, 0, 1)?.narrow(1, 0, 1)?;
        device.synchronize()?;

        let max_abs = (&actual - &expected)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert!(max_abs <= 2e-5, "max absolute error {max_abs}");
        Ok(())
    }

    #[test]
    fn fused_sdpa_matches_local_attention_with_rotary() -> Result<()> {
        check_sam3_shape(9, 24 * 24, 24, 1.)
    }

    #[test]
    fn fused_sdpa_matches_global_attention_with_rotary() -> Result<()> {
        check_sam3_shape(1, 72 * 72, 72, 24. / 72.)
    }
}
