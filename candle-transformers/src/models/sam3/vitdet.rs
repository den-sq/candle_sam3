use std::collections::BTreeMap;

use candle::{DType, IndexOp, Result, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, LayerNorm, Linear, Module, VarBuilder};

use super::config::VisionConfig;

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
    scale: f64,
    rotary_emb: VisionRotaryEmbedding,
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
        Ok(Self {
            qkv,
            proj,
            num_heads: config.num_heads,
            head_dim,
            scale: (head_dim as f64).powf(-0.5),
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
            .permute((2, 0, 3, 1, 4))?;
        let q = qkv.i(0)?.contiguous()?;
        let k = qkv.i(1)?.contiguous()?;
        let v = qkv.i(2)?.contiguous()?;
        let (q, k) = self.rotary_emb.apply(&q, &k)?;
        let scale = Tensor::new(self.scale as f32, q.device())?;
        let q = q
            .to_dtype(DType::F32)?
            .broadcast_mul(&scale)?
            .contiguous()?;
        let k = k.to_dtype(DType::F32)?.contiguous()?;
        let v = v.to_dtype(DType::F32)?.contiguous()?;
        let attn = q.matmul(&k.transpose(2, 3)?)?;
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        let hidden_states = attn
            .matmul(&v)?
            .to_dtype(in_dtype)?
            .transpose(1, 2)?
            .reshape((batch_size, height, width, channels))?
            .contiguous()?;
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
            window_partition(&hidden_states, self.window_size)?
        } else {
            (hidden_states, original_hw)
        };
        let hidden_states = self.attn.forward(&hidden_states)?;
        let hidden_states = if self.window_size > 0 {
            window_unpartition(&hidden_states, self.window_size, padded_hw, original_hw)?
        } else {
            hidden_states
        };
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
            window_partition(&hidden_states, self.window_size)?
        } else {
            (hidden_states, original_hw)
        };
        let hidden_states = self.attn.forward(&hidden_states)?;
        let hidden_states = if self.window_size > 0 {
            window_unpartition(&hidden_states, self.window_size, padded_hw, original_hw)?
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

fn window_partition(
    hidden_states: &Tensor,
    window_size: usize,
) -> Result<(Tensor, (usize, usize))> {
    let hidden_states = hidden_states.contiguous()?;
    let (batch_size, height, width, channels) = hidden_states.dims4()?;
    let pad_height = (window_size - height % window_size) % window_size;
    let pad_width = (window_size - width % window_size) % window_size;
    let hidden_states = if pad_height > 0 {
        hidden_states.pad_with_zeros(1, 0, pad_height)?
    } else {
        hidden_states
    };
    let hidden_states = if pad_width > 0 {
        hidden_states.pad_with_zeros(2, 0, pad_width)?
    } else {
        hidden_states
    };
    let padded_height = height + pad_height;
    let padded_width = width + pad_width;
    let windows = hidden_states
        .reshape((
            batch_size,
            padded_height / window_size,
            window_size,
            padded_width / window_size,
            window_size,
            channels,
        ))?
        .permute((0, 1, 3, 2, 4, 5))?
        .reshape((
            batch_size * (padded_height / window_size) * (padded_width / window_size),
            window_size,
            window_size,
            channels,
        ))?;
    Ok((windows, (padded_height, padded_width)))
}

fn window_unpartition(
    windows: &Tensor,
    window_size: usize,
    padded_hw: (usize, usize),
    original_hw: (usize, usize),
) -> Result<Tensor> {
    let (padded_height, padded_width) = padded_hw;
    let (height, width) = original_hw;
    let num_windows_per_image = padded_height * padded_width / window_size / window_size;
    let batch_size = windows.dim(0)? / num_windows_per_image;
    let hidden_states = windows
        .reshape((
            batch_size,
            padded_height / window_size,
            padded_width / window_size,
            window_size,
            window_size,
            windows.dim(3)?,
        ))?
        .permute((0, 1, 3, 2, 4, 5))?
        .reshape((batch_size, padded_height, padded_width, windows.dim(3)?))?;
    let hidden_states = if padded_height > height {
        hidden_states.narrow(1, 0, height)?
    } else {
        hidden_states
    };
    let hidden_states = if padded_width > width {
        hidden_states.narrow(2, 0, width)?
    } else {
        hidden_states
    };
    hidden_states.contiguous()
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
    let repeat_h = target_height / pretrain_size + 1;
    let repeat_w = target_width / pretrain_size + 1;
    let pos_embed = pos_embed.repeat((1, 1, repeat_h, repeat_w))?;
    pos_embed
        .narrow(2, 0, target_height)?
        .narrow(3, 0, target_width)?
        .permute((0, 2, 3, 1))
}

#[derive(Debug)]
pub struct Sam3ViTDetTrunk {
    config: VisionConfig,
    patch_embed: PatchEmbed,
    pos_embed: Option<Tensor>,
    blocks: Vec<Sam3VisionBlock>,
    pre_layer_norm: Option<LayerNorm>,
}

impl Sam3ViTDetTrunk {
    pub fn new(config: &VisionConfig, vb: VarBuilder) -> Result<Self> {
        let patch_embed = PatchEmbed::new(config, vb.pp("patch_embed"))?;
        let pos_embed = if config.use_abs_pos {
            Some(load_pos_embed(config, vb.clone())?)
        } else {
            None
        };
        let input_size = (
            config.image_size / config.patch_size,
            config.image_size / config.patch_size,
        );
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
            let pos_embed = tile_position_embeddings(pos_embed, patch_height, patch_width)?;
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
        Ok((
            ViTDetTrunkOutput {
                stage_features: vec![hidden_states],
            },
            block_outputs,
            debug_tensors,
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use std::collections::HashMap;

    use candle::{DType, Device, IndexOp, Result, Tensor};
    use candle_nn::{Conv2d, Conv2dConfig, Module, VarBuilder};

    use super::Sam3ViTDetTrunk;
    use crate::models::sam3::VisionConfig;

    fn small_config() -> VisionConfig {
        VisionConfig {
            image_size: 56,
            pretrain_image_size: 28,
            patch_size: 14,
            embed_dim: 8,
            depth: 0,
            num_heads: 2,
            mlp_ratio: 4.0,
            window_size: 2,
            global_attn_blocks: vec![],
            use_abs_pos: true,
            tile_abs_pos: true,
            use_rope: true,
            use_interp_rope: true,
            rope_theta: 10_000.0,
            rope_pt_size: 2,
            retain_cls_token: false,
            ln_pre: false,
        }
    }

    fn test_device() -> Result<Device> {
        #[cfg(feature = "cuda")]
        {
            Device::new_cuda(0)
        }
        #[cfg(all(not(feature = "cuda"), feature = "metal"))]
        {
            Device::new_metal(0)
        }
        #[cfg(not(any(feature = "cuda", feature = "metal")))]
        {
            Ok(Device::Cpu)
        }
    }

    #[test]
    fn vitdet_trunk_tiles_position_embeddings_and_strips_cls_token() -> Result<()> {
        let device = test_device()?;
        let config = small_config();
        let mut tensors = HashMap::new();
        tensors.insert(
            "patch_embed.proj.weight".to_string(),
            Tensor::zeros((config.embed_dim, 3, 14, 14), DType::F32, &device)?,
        );
        let mut pos = vec![0f32; 5 * config.embed_dim];
        for token_idx in 1..5 {
            pos[token_idx * config.embed_dim] = token_idx as f32;
        }
        tensors.insert(
            "pos_embed".to_string(),
            Tensor::from_slice(&pos, (1, 5, config.embed_dim), &device)?,
        );
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
        let trunk = Sam3ViTDetTrunk::new(&config, vb)?;
        let images = Tensor::zeros((1, 3, 56, 56), DType::F32, &device)?;
        let output = trunk.forward(&images)?;
        let feature = output.stage_features[0].i((0, .., .., 0))?;
        assert_eq!(output.stage_features[0].dims4()?, (1, 4, 4, 8));
        assert_eq!(
            feature.to_vec2::<f32>()?,
            vec![
                vec![1.0, 2.0, 1.0, 2.0],
                vec![3.0, 4.0, 3.0, 4.0],
                vec![1.0, 2.0, 1.0, 2.0],
                vec![3.0, 4.0, 3.0, 4.0],
            ]
        );
        Ok(())
    }

    #[test]
    #[ignore = "fixture-driven parity investigation"]
    fn interactive_visual_fixture_trunk_matches_upstream() -> Result<()> {
        let device = test_device()?;
        let weights = load_interactive_visual_fixture_tensors(
            "vision_backbone_weights.safetensors",
            &device,
        )?;
        let fixture = load_interactive_visual_fixture_tensors("fixture.safetensors", &device)?;
        let vb = VarBuilder::from_tensors(weights, DType::F32, &device);
        let trunk = Sam3ViTDetTrunk::new(&VisionConfig::default(), vb.pp("trunk"))?;
        let image = fixture_tensor(&fixture, "inputs.image_preprocessed")?.clone();
        let output = trunk.forward(&image)?;
        let actual = output
            .stage_features
            .last()
            .expect("trunk should emit one stage feature")
            .permute((0, 3, 1, 2))?;
        assert_tensor_close(
            &actual,
            fixture_tensor(&fixture, "vision.trunk.last")?,
            1e-5,
            "vision.trunk.last",
        )
    }

    #[test]
    #[ignore = "fixture-driven parity investigation"]
    fn interactive_visual_fixture_trunk_block_outputs_cover_fixture() -> Result<()> {
        let device = test_device()?;
        let weights = load_interactive_visual_fixture_tensors(
            "vision_backbone_weights.safetensors",
            &device,
        )?;
        let fixture = load_interactive_visual_fixture_tensors("fixture.safetensors", &device)?;
        let vb = VarBuilder::from_tensors(weights, DType::F32, &device);
        let trunk = Sam3ViTDetTrunk::new(&VisionConfig::default(), vb.pp("trunk"))?;
        let image = fixture_tensor(&fixture, "inputs.image_preprocessed")?.clone();
        let (_, block_outputs) = trunk.forward_with_block_outputs(&image)?;
        let expected_block_count = fixture
            .keys()
            .filter(|key| key.starts_with("vision.block."))
            .count();
        if block_outputs.len() != expected_block_count {
            candle::bail!(
                "expected {expected_block_count} trunk block outputs, got {}",
                block_outputs.len()
            );
        }

        for (block_idx, block_output) in block_outputs.iter().enumerate() {
            let actual = block_output.permute((0, 3, 1, 2))?;
            let name = format!("vision.block.{block_idx}");
            if actual.dims() != fixture_tensor(&fixture, &name)?.dims() {
                candle::bail!(
                    "{name}: shape mismatch actual={:?} expected={:?}",
                    actual.dims(),
                    fixture_tensor(&fixture, &name)?.dims()
                );
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "fixture-driven parity investigation"]
    fn interactive_visual_fixture_trunk_first_diverging_block_is_0() -> Result<()> {
        let device = test_device()?;
        let weights = load_interactive_visual_fixture_tensors(
            "vision_backbone_weights.safetensors",
            &device,
        )?;
        let fixture = load_interactive_visual_fixture_tensors("fixture.safetensors", &device)?;
        let vb = VarBuilder::from_tensors(weights, DType::F32, &device);
        let trunk = Sam3ViTDetTrunk::new(&VisionConfig::default(), vb.pp("trunk"))?;
        let image = fixture_tensor(&fixture, "inputs.image_preprocessed")?.clone();
        let (_, block_outputs) = trunk.forward_with_block_outputs(&image)?;

        let (first_block, max_abs_diff) = first_diverging_block(&block_outputs, &fixture, 1e-5)?
            .ok_or_else(|| {
                candle::Error::Msg(
                    "expected at least one diverging trunk block in the interactive visual fixture"
                        .to_owned(),
                )
            })?;
        if first_block != 0 {
            candle::bail!(
                "expected first diverging trunk block to be 0, got {first_block} (max_abs_diff={max_abs_diff:.8})"
            );
        }
        Ok(())
    }

    #[test]
    #[ignore = "fixture-driven parity investigation"]
    fn interactive_visual_fixture_block0_substages_match_external_reference() -> Result<()> {
        let fixture_path = std::env::var("SAM3_BLOCK0_DEBUG_FIXTURE").map_err(|_| {
            candle::Error::Msg(
                "set SAM3_BLOCK0_DEBUG_FIXTURE to a reference.safetensors block-debug export"
                    .to_owned(),
            )
        })?;
        let device = test_device()?;
        let fixture =
            candle::safetensors::load(PathBuf::from(&fixture_path), &device).map_err(|err| {
                candle::Error::Msg(format!(
                    "failed to load block0 debug fixture {}: {err}",
                    fixture_path
                ))
            })?;
        let mut weights = load_interactive_visual_fixture_tensors(
            "vision_backbone_weights.safetensors",
            &device,
        )?;
        if let Some(exported_weights) = fixture_trunk_weights(&fixture)? {
            weights.extend(exported_weights);
        }
        let vb = VarBuilder::from_tensors(weights, DType::F32, &device);
        let trunk = Sam3ViTDetTrunk::new(&VisionConfig::default(), vb.pp("trunk"))?;
        let image = fixture_tensor(&fixture, "inputs.image")?.clone();
        let (_, _, debug_tensors) = trunk.forward_with_debug_blocks(&image, &[0])?;

        let mut failures = Vec::new();
        for name in [
            "vision.pre_block.patch_embed",
            "vision.pre_block.pos_embed_added",
            "vision.pre_block.ln_pre",
            "vision.block_debug.0.input",
            "vision.block_debug.0.norm1",
            "vision.block_debug.0.attn_output",
            "vision.block_debug.0.post_attn",
            "vision.block_debug.0.norm2",
            "vision.block_debug.0.mlp_fc1",
            "vision.block_debug.0.mlp_gelu",
            "vision.block_debug.0.mlp_output",
            "vision.block_debug.0.output",
        ] {
            let actual = debug_tensors.get(name).ok_or_else(|| {
                candle::Error::Msg(format!("missing debug tensor `{name}` from Candle trunk"))
            })?;
            let actual = actual.permute((0, 3, 1, 2))?;
            let expected = fixture_tensor(&fixture, name)?;
            let max_abs_diff = tensor_max_abs_diff(&actual, expected)?;
            println!("{name}: max_abs_diff={max_abs_diff:.8}");
            if max_abs_diff > 1e-5 {
                failures.push((name, max_abs_diff));
            }
        }
        if let Some((name, max_abs_diff)) = failures.first() {
            candle::bail!("{name}: max_abs_diff={max_abs_diff:.8} exceeded atol=0.00001000");
        }
        Ok(())
    }

    #[test]
    #[ignore = "fixture-driven parity investigation"]
    fn interactive_visual_fixture_patch_embed_manual_diagnostic() -> Result<()> {
        let fixture_path = std::env::var("SAM3_BLOCK0_DEBUG_FIXTURE").map_err(|_| {
            candle::Error::Msg(
                "set SAM3_BLOCK0_DEBUG_FIXTURE to a reference.safetensors block-debug export"
                    .to_owned(),
            )
        })?;
        let device = test_device()?;
        let fixture =
            candle::safetensors::load(PathBuf::from(&fixture_path), &device).map_err(|err| {
                candle::Error::Msg(format!(
                    "failed to load block0 debug fixture {}: {err}",
                    fixture_path
                ))
            })?;
        let mut weights = load_interactive_visual_fixture_tensors(
            "vision_backbone_weights.safetensors",
            &device,
        )?;
        if let Some(exported_weights) = fixture_trunk_weights(&fixture)? {
            weights.extend(exported_weights);
        }
        let vb = VarBuilder::from_tensors(weights.clone(), DType::F32, &device);
        let trunk = Sam3ViTDetTrunk::new(&VisionConfig::default(), vb.pp("trunk"))?;
        let image = fixture_tensor(&fixture, "inputs.image")?.clone();
        let (_, _, debug_tensors) = trunk.forward_with_debug_blocks(&image, &[0])?;
        let conv_patch = debug_tensors
            .get("vision.pre_block.patch_embed")
            .ok_or_else(|| {
                candle::Error::Msg("missing Candle patch_embed debug tensor".to_owned())
            })?
            .permute((0, 3, 1, 2))?;
        let expected_patch = fixture_tensor(&fixture, "vision.pre_block.patch_embed")?;

        let patch_weight = weights
            .get("trunk.patch_embed.proj.weight")
            .ok_or_else(|| {
                candle::Error::Msg(
                    "interactive visual weights missing trunk.patch_embed.proj.weight".to_owned(),
                )
            })?
            .clone();
        let manual_patch = manual_patch_embed_no_bias(&image, &patch_weight, 14)?;

        let conv_diff = tensor_max_abs_diff(&conv_patch, expected_patch)?;
        let manual_diff = tensor_max_abs_diff(&manual_patch, expected_patch)?;
        let conv_vs_manual = tensor_max_abs_diff(&conv_patch, &manual_patch)?;
        println!("patch_embed conv vs upstream: {conv_diff:.8}");
        println!("patch_embed manual vs upstream: {manual_diff:.8}");
        println!("patch_embed conv vs manual: {conv_vs_manual:.8}");
        if matches!(device, Device::Cuda(_)) {
            let image_bf16 = image.to_dtype(DType::BF16)?;
            let weight_bf16 = patch_weight.to_dtype(DType::BF16)?;
            match manual_patch_embed_no_bias(&image_bf16, &weight_bf16, 14) {
                Ok(manual_bf16) => {
                    let manual_bf16_diff = tensor_max_abs_diff(&manual_bf16, expected_patch)?;
                    let manual_bf16_vs_manual = tensor_max_abs_diff(&manual_bf16, &manual_patch)?;
                    println!("patch_embed manual_bf16 vs upstream: {manual_bf16_diff:.8}");
                    println!("patch_embed manual_bf16 vs manual_f32: {manual_bf16_vs_manual:.8}");
                }
                Err(err) => {
                    println!("patch_embed manual_bf16 unavailable: {err}");
                }
            }
            let conv_bf16 = Conv2d::new(
                weight_bf16,
                None,
                Conv2dConfig {
                    stride: 14,
                    ..Default::default()
                },
            );
            match conv_bf16.forward(&image_bf16) {
                Ok(conv_bf16_out) => {
                    let conv_bf16_diff = tensor_max_abs_diff(&conv_bf16_out, expected_patch)?;
                    let conv_bf16_vs_f32 = tensor_max_abs_diff(&conv_bf16_out, &conv_patch)?;
                    println!("patch_embed conv_bf16 vs upstream: {conv_bf16_diff:.8}");
                    println!("patch_embed conv_bf16 vs conv_f32: {conv_bf16_vs_f32:.8}");
                }
                Err(err) => {
                    println!("patch_embed conv_bf16 unavailable: {err}");
                }
            }
        }
        Ok(())
    }

    fn interactive_visual_fixture_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/sam3_interactive_visual_seed")
    }

    fn load_interactive_visual_fixture_tensors(
        file_name: &str,
        device: &Device,
    ) -> Result<HashMap<String, Tensor>> {
        let path = interactive_visual_fixture_dir().join(file_name);
        candle::safetensors::load(&path, device).map_err(|err| {
            candle::Error::Msg(format!(
                "failed to load interactive visual fixture {}: {err}",
                path.display()
            ))
        })
    }

    fn fixture_tensor<'a>(fixture: &'a HashMap<String, Tensor>, key: &str) -> Result<&'a Tensor> {
        fixture.get(key).ok_or_else(|| {
            candle::Error::Msg(format!(
                "interactive visual fixture is missing tensor `{key}`"
            ))
        })
    }

    fn first_diverging_block(
        block_outputs: &[Tensor],
        fixture: &HashMap<String, Tensor>,
        atol: f32,
    ) -> Result<Option<(usize, f32)>> {
        for (block_idx, block_output) in block_outputs.iter().enumerate() {
            let name = format!("vision.block.{block_idx}");
            let actual = block_output.permute((0, 3, 1, 2))?;
            let max_abs_diff = tensor_max_abs_diff(&actual, fixture_tensor(fixture, &name)?)?;
            if max_abs_diff > atol {
                return Ok(Some((block_idx, max_abs_diff)));
            }
        }
        Ok(None)
    }

    fn fixture_trunk_weights(
        fixture: &HashMap<String, Tensor>,
    ) -> Result<Option<HashMap<String, Tensor>>> {
        let mut weights = HashMap::new();
        for (fixture_key, tensor) in fixture.iter() {
            let Some(weight_key) = fixture_key.strip_prefix("vision.weights.") else {
                continue;
            };
            weights.insert(format!("trunk.{weight_key}"), tensor.clone());
        }
        if weights.is_empty() {
            Ok(None)
        } else {
            Ok(Some(weights))
        }
    }

    fn manual_patch_embed_no_bias(
        image: &Tensor,
        weight: &Tensor,
        patch_size: usize,
    ) -> Result<Tensor> {
        let (batch, channels, height, width) = image.dims4()?;
        let (out_channels, in_channels, kernel_h, kernel_w) = weight.dims4()?;
        if in_channels != channels || kernel_h != patch_size || kernel_w != patch_size {
            candle::bail!(
                "manual patch embed shape mismatch image={:?} weight={:?} patch_size={patch_size}",
                image.dims(),
                weight.dims()
            );
        }
        if height % patch_size != 0 || width % patch_size != 0 {
            candle::bail!(
                "manual patch embed expects image divisible by patch size {patch_size}, got {height}x{width}"
            );
        }
        let patch_h = height / patch_size;
        let patch_w = width / patch_size;
        let image = image.reshape((batch, channels, patch_h, patch_size, patch_w, patch_size))?;
        let image = image.permute((0, 2, 4, 1, 3, 5))?.reshape((
            batch * patch_h * patch_w,
            channels * patch_size * patch_size,
        ))?;
        let weight = weight.reshape((out_channels, in_channels * patch_size * patch_size))?;
        let output = image.matmul(&weight.transpose(0, 1)?)?;
        output
            .reshape((batch, patch_h, patch_w, out_channels))?
            .permute((0, 3, 1, 2))
    }

    fn assert_tensor_close(
        actual: &Tensor,
        expected: &Tensor,
        atol: f32,
        name: &str,
    ) -> Result<()> {
        let max_abs_diff = tensor_max_abs_diff(actual, expected)?;
        if max_abs_diff > atol {
            candle::bail!("{name}: max_abs_diff={max_abs_diff:.8} exceeded atol={atol:.8}");
        }
        Ok(())
    }

    fn tensor_max_abs_diff(actual: &Tensor, expected: &Tensor) -> Result<f32> {
        if actual.dims() != expected.dims() {
            candle::bail!(
                "shape mismatch actual={:?} expected={:?}",
                actual.dims(),
                expected.dims()
            );
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
        for (lhs, rhs) in actual.iter().zip(expected.iter()) {
            max_abs_diff = max_abs_diff.max((lhs - rhs).abs());
        }
        Ok(max_abs_diff)
    }
}
