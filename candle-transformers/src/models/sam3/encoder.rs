use candle::{DType, Result, Tensor};
use candle_nn::{LayerNorm, Linear, Module, VarBuilder};

use super::config::EncoderConfig;
use super::geometry::EncodedPrompt;

#[derive(Debug)]
pub struct FusionEncoderOutput {
    pub memory: Tensor,
    pub pos_embed: Tensor,
    pub padding_mask: Tensor,
    pub level_start_index: Tensor,
    pub spatial_shapes: Tensor,
    pub valid_ratios: Tensor,
}

#[derive(Debug)]
struct FusionAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl FusionAttention {
    fn new(config: &EncoderConfig, vb: VarBuilder) -> Result<Self> {
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
struct FusionEncoderLayer {
    norm1: LayerNorm,
    self_attn: FusionAttention,
    norm2: LayerNorm,
    cross_attn_prompt: FusionAttention,
    norm3: LayerNorm,
    linear1: Linear,
    linear2: Linear,
}

impl FusionEncoderLayer {
    fn new(config: &EncoderConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            norm1: candle_nn::layer_norm(config.d_model, 1e-5, vb.pp("norm1"))?,
            self_attn: FusionAttention::new(config, vb.pp("self_attn"))?,
            norm2: candle_nn::layer_norm(config.d_model, 1e-5, vb.pp("norm2"))?,
            cross_attn_prompt: FusionAttention::new(config, vb.pp("cross_attn_image"))?,
            norm3: candle_nn::layer_norm(config.d_model, 1e-5, vb.pp("norm3"))?,
            linear1: candle_nn::linear(config.d_model, config.dim_feedforward, vb.pp("linear1"))?,
            linear2: candle_nn::linear(config.dim_feedforward, config.d_model, vb.pp("linear2"))?,
        })
    }

    fn forward(
        &self,
        tgt: &Tensor,
        prompt: &Tensor,
        tgt_padding_mask: &Tensor,
        prompt_padding_mask: &Tensor,
        query_pos: &Tensor,
    ) -> Result<Tensor> {
        let residual = tgt;
        let hidden_states = self.norm1.forward(tgt)?;
        let hidden_states = self.self_attn.forward(
            &hidden_states,
            &hidden_states,
            Some(tgt_padding_mask),
            Some(query_pos),
            Some(query_pos),
        )?;
        let hidden_states = (hidden_states + residual)?;

        let residual = &hidden_states;
        let hidden_states = self.norm2.forward(&hidden_states)?;
        let hidden_states = self.cross_attn_prompt.forward(
            &hidden_states,
            prompt,
            Some(prompt_padding_mask),
            None,
            None,
        )?;
        let hidden_states = (hidden_states + residual)?;

        let residual = &hidden_states;
        let hidden_states = self.norm3.forward(&hidden_states)?;
        let hidden_states = self.linear1.forward(&hidden_states)?.relu()?;
        let hidden_states = self.linear2.forward(&hidden_states)?;
        hidden_states + residual
    }
}

#[derive(Debug)]
pub struct Sam3FusionEncoder {
    config: EncoderConfig,
    layers: Vec<FusionEncoderLayer>,
}

impl Sam3FusionEncoder {
    pub fn new(config: &EncoderConfig, vb: VarBuilder) -> Result<Self> {
        if config.num_heads == 0 || config.d_model % config.num_heads != 0 {
            candle::bail!(
                "sam3 fusion encoder d_model ({}) must be divisible by num_heads ({})",
                config.d_model,
                config.num_heads
            )
        }
        let mut layers = Vec::with_capacity(config.num_layers);
        let layers_vb = vb.pp("layers");
        for layer_idx in 0..config.num_layers {
            layers.push(FusionEncoderLayer::new(config, layers_vb.pp(layer_idx))?);
        }
        Ok(Self {
            config: config.clone(),
            layers,
        })
    }

    pub fn config(&self) -> &EncoderConfig {
        &self.config
    }

    pub fn forward(
        &self,
        visual_features: &[Tensor],
        visual_pos_embeds: &[Tensor],
        prompt: &EncodedPrompt,
    ) -> Result<FusionEncoderOutput> {
        let (selected_features, selected_pos) = select_feature_levels(
            visual_features,
            visual_pos_embeds,
            self.config.num_feature_levels,
        )?;
        let pooled_prompt = if self.config.add_pooled_text_to_image {
            Some(pool_prompt_feat(
                &prompt.features,
                &prompt.padding_mask,
                self.config.pool_text_with_mask,
            )?)
        } else {
            None
        };
        let (memory_parts, pos_parts, spatial_shapes, level_start_index, valid_ratios, batch_size) =
            prepare_multilevel_features(&selected_features, &selected_pos, pooled_prompt.as_ref())?;
        let memory_refs: Vec<&Tensor> = memory_parts.iter().collect();
        let pos_refs: Vec<&Tensor> = pos_parts.iter().collect();
        let mut memory = Tensor::cat(&memory_refs, 0)?;
        let pos_embed = Tensor::cat(&pos_refs, 0)?;
        let padding_mask = Tensor::zeros((memory.dim(0)?, batch_size), DType::U8, memory.device())?;
        for layer in self.layers.iter() {
            memory = layer.forward(
                &memory,
                &prompt.features,
                &padding_mask,
                &prompt.padding_mask,
                &pos_embed,
            )?;
        }
        Ok(FusionEncoderOutput {
            memory,
            pos_embed,
            padding_mask,
            level_start_index,
            spatial_shapes,
            valid_ratios,
        })
    }
}

fn normalize_padding_mask(mask: &Tensor, batch_size: usize, seq_len: usize) -> Result<Tensor> {
    match mask.dims() {
        [b, s] if *b == batch_size && *s == seq_len => Ok(mask.clone()),
        [s, b] if *s == seq_len && *b == batch_size => Ok(mask.transpose(0, 1)?.contiguous()?),
        shape => candle::bail!(
            "sam3 fusion encoder expected padding mask shape ({batch_size}, {seq_len}) or ({seq_len}, {batch_size}), got {shape:?}"
        ),
    }
}

fn select_feature_levels<'a>(
    visual_features: &'a [Tensor],
    visual_pos_embeds: &'a [Tensor],
    num_feature_levels: usize,
) -> Result<(Vec<&'a Tensor>, Vec<&'a Tensor>)> {
    if visual_features.len() != visual_pos_embeds.len() {
        candle::bail!(
            "sam3 fusion encoder expected the same number of visual features and position encodings, got {} and {}",
            visual_features.len(),
            visual_pos_embeds.len(),
        )
    }
    if num_feature_levels == 0 || num_feature_levels > visual_features.len() {
        candle::bail!(
            "sam3 fusion encoder expected 1..={} feature levels, got {}",
            visual_features.len(),
            num_feature_levels
        )
    }
    let start = visual_features.len() - num_feature_levels;
    Ok((
        visual_features[start..].iter().collect(),
        visual_pos_embeds[start..].iter().collect(),
    ))
}

fn prepare_multilevel_features(
    visual_features: &[&Tensor],
    visual_pos_embeds: &[&Tensor],
    pooled_prompt: Option<&Tensor>,
) -> Result<(Vec<Tensor>, Vec<Tensor>, Tensor, Tensor, Tensor, usize)> {
    let mut memory_parts = Vec::with_capacity(visual_features.len());
    let mut pos_parts = Vec::with_capacity(visual_pos_embeds.len());
    let mut spatial_shapes_vec = Vec::with_capacity(visual_features.len() * 2);
    let mut level_start_index_vec = Vec::with_capacity(visual_features.len());
    let mut valid_ratios_vec = Vec::new();
    let mut current_offset = 0u32;
    let mut batch_size = None;

    for (feature_map, pos_embed) in visual_features.iter().zip(visual_pos_embeds.iter()) {
        let (batch, channels, height, width) = feature_map.dims4()?;
        let pos_shape = pos_embed.dims4()?;
        if pos_shape != (batch, channels, height, width) {
            candle::bail!(
                "sam3 fusion encoder expected matching feature and pos shapes, got ({batch}, {channels}, {height}, {width}) and {pos_shape:?}"
            )
        }
        if let Some(prev_batch) = batch_size {
            if prev_batch != batch {
                candle::bail!(
                    "sam3 fusion encoder expected consistent batch size across levels, got {prev_batch} and {batch}"
                )
            }
        } else {
            batch_size = Some(batch);
            valid_ratios_vec = vec![1f32; batch * visual_features.len() * 2];
        }
        let feature_map = match pooled_prompt {
            Some(pooled_prompt) => feature_map.broadcast_add(
                &pooled_prompt
                    .reshape((batch, channels, 1, 1))?
                    .contiguous()?,
            )?,
            None => (*feature_map).clone(),
        };
        memory_parts.push(feature_map.permute((2, 3, 0, 1))?.reshape((
            height * width,
            batch,
            channels,
        ))?);
        pos_parts.push(pos_embed.permute((2, 3, 0, 1))?.reshape((
            height * width,
            batch,
            channels,
        ))?);
        level_start_index_vec.push(current_offset);
        current_offset += (height * width) as u32;
        spatial_shapes_vec.push(height as u32);
        spatial_shapes_vec.push(width as u32);
    }

    let batch_size = batch_size.unwrap_or(0);
    let device = visual_features[0].device();
    let spatial_shapes = Tensor::from_vec(spatial_shapes_vec, (visual_features.len(), 2), device)?;
    let level_start_index = Tensor::from_vec(level_start_index_vec, visual_features.len(), device)?;
    let valid_ratios = Tensor::from_vec(
        valid_ratios_vec,
        (batch_size, visual_features.len(), 2),
        device,
    )?;
    Ok((
        memory_parts,
        pos_parts,
        spatial_shapes,
        level_start_index,
        valid_ratios,
        batch_size,
    ))
}

fn pool_prompt_feat(prompt: &Tensor, prompt_mask: &Tensor, pool_with_mask: bool) -> Result<Tensor> {
    if !pool_with_mask {
        return prompt.mean(0);
    }
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

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::fs;
    use std::path::PathBuf;

    use candle::{DType, Device, Result, Tensor};
    use candle_nn::VarBuilder;

    use super::{FusionEncoderOutput, Sam3FusionEncoder};
    use crate::models::sam3::config::EncoderConfig;
    use crate::models::sam3::geometry::EncodedPrompt;

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
    fn fusion_encoder_flattens_last_feature_level() -> Result<()> {
        let device = test_device()?;
        let config = test_config();
        let vb = VarBuilder::from_tensors(encoder_weights(&config, &device)?, DType::F32, &device);
        let encoder = Sam3FusionEncoder::new(&config, vb)?;
        let visual_features = vec![
            Tensor::zeros((1, config.d_model, 8, 8), DType::F32, &device)?,
            Tensor::zeros((1, config.d_model, 4, 4), DType::F32, &device)?,
            Tensor::zeros((1, config.d_model, 2, 2), DType::F32, &device)?,
        ];
        let visual_pos = visual_features
            .iter()
            .map(Tensor::zeros_like)
            .collect::<Result<Vec<_>>>()?;
        let prompt = EncodedPrompt {
            features: Tensor::zeros((3, 1, config.d_model), DType::F32, &device)?,
            padding_mask: Tensor::zeros((1, 3), DType::U8, &device)?,
        };
        let output = encoder.forward(&visual_features, &visual_pos, &prompt)?;
        assert_fusion_output(output, 4, 1, &config)
    }

    #[test]
    fn fusion_encoder_can_pool_prompt_into_image_features() -> Result<()> {
        let device = test_device()?;
        let mut config = test_config();
        config.add_pooled_text_to_image = true;
        let vb = VarBuilder::from_tensors(encoder_weights(&config, &device)?, DType::F32, &device);
        let encoder = Sam3FusionEncoder::new(&config, vb)?;
        let visual_features = vec![Tensor::zeros(
            (1, config.d_model, 2, 2),
            DType::F32,
            &device,
        )?];
        let visual_pos = vec![Tensor::zeros(
            (1, config.d_model, 2, 2),
            DType::F32,
            &device,
        )?];
        let prompt = EncodedPrompt {
            features: Tensor::ones((2, 1, config.d_model), DType::F32, &device)?,
            padding_mask: Tensor::zeros((1, 2), DType::U8, &device)?,
        };
        let output = encoder.forward(&visual_features, &visual_pos, &prompt)?;
        assert_fusion_output(output, 4, 1, &config)
    }

    #[test]
    fn fusion_encoder_smoke_fixture_matches_expected_values() -> Result<()> {
        let device = test_device()?;
        let config = test_config();
        let vb = VarBuilder::from_tensors(encoder_weights(&config, &device)?, DType::F32, &device);
        let encoder = Sam3FusionEncoder::new(&config, vb)?;
        let values = (0..(config.d_model * 2 * 2))
            .map(|value| value as f32)
            .collect::<Vec<_>>();
        let pos_values = (0..(config.d_model * 2 * 2))
            .map(|value| value as f32 + 100.0)
            .collect::<Vec<_>>();
        let visual_features = vec![Tensor::from_vec(
            values,
            (1, config.d_model, 2, 2),
            &device,
        )?];
        let visual_pos = vec![Tensor::from_vec(
            pos_values,
            (1, config.d_model, 2, 2),
            &device,
        )?];
        let prompt = EncodedPrompt {
            features: Tensor::zeros((2, 1, config.d_model), DType::F32, &device)?,
            padding_mask: Tensor::zeros((1, 2), DType::U8, &device)?,
        };

        let output = encoder.forward(&visual_features, &visual_pos, &prompt)?;
        let expected_memory =
            visual_features[0]
                .permute((2, 3, 0, 1))?
                .reshape((4, 1, config.d_model))?;
        let expected_pos = visual_pos[0]
            .permute((2, 3, 0, 1))?
            .reshape((4, 1, config.d_model))?;

        assert_tensor_close(&output.memory, &expected_memory, 0.0, "fusion.smoke.memory")?;
        assert_tensor_close(
            &output.pos_embed,
            &expected_pos,
            0.0,
            "fusion.smoke.pos_embed",
        )?;
        assert_eq!(
            output.padding_mask.to_vec2::<u8>()?,
            vec![vec![0], vec![0], vec![0], vec![0]]
        );
        assert_eq!(output.spatial_shapes.to_vec2::<u32>()?, vec![vec![2, 2]]);
        assert_eq!(output.level_start_index.to_vec1::<u32>()?, vec![0]);
        assert_eq!(
            output.valid_ratios.to_vec3::<f32>()?,
            vec![vec![vec![1.0, 1.0]]]
        );
        Ok(())
    }

    #[test]
    #[ignore = "fixture-driven parity investigation"]
    fn fusion_fixture_memory_matches_upstream() -> Result<()> {
        let device = test_device()?;
        let output = run_fixture_fusion(&device)?;
        let expected = load_fusion_fixture_tensors("fixture.safetensors", &device)?;
        assert_tensor_close(
            &output.memory,
            fixture_tensor(&expected, "fusion.memory")?,
            1e-5,
            "fusion.memory",
        )?;
        assert_tensor_close(
            &output.pos_embed,
            fixture_tensor(&expected, "fusion.pos_embed")?,
            1e-5,
            "fusion.pos_embed",
        )?;
        assert_tensor_close(
            &output.valid_ratios,
            fixture_tensor(&expected, "fusion.valid_ratios")?,
            1e-5,
            "fusion.valid_ratios",
        )?;
        Ok(())
    }

    #[test]
    #[ignore = "fixture-driven parity investigation"]
    fn fusion_fixture_metadata_matches_upstream() -> Result<()> {
        let device = test_device()?;
        let output = run_fixture_fusion(&device)?;
        let expected = load_fusion_fixture_tensors("fixture.safetensors", &device)?;
        if let Some(expected_padding_mask) = expected.get("fusion.padding_mask") {
            assert_tensor_close(
                &output.padding_mask.to_dtype(DType::U8)?,
                expected_padding_mask,
                0.0,
                "fusion.padding_mask",
            )?;
        } else {
            let zero_mask = Tensor::zeros_like(&output.padding_mask)?;
            assert_tensor_close(&output.padding_mask, &zero_mask, 0.0, "fusion.padding_mask")?;
        }
        assert_tensor_close(
            &output.spatial_shapes,
            fixture_tensor(&expected, "fusion.spatial_shapes")?,
            0.0,
            "fusion.spatial_shapes",
        )?;
        assert_tensor_close(
            &output.level_start_index,
            fixture_tensor(&expected, "fusion.level_start_index")?,
            0.0,
            "fusion.level_start_index",
        )?;
        Ok(())
    }

    fn assert_fusion_output(
        output: FusionEncoderOutput,
        seq_len: usize,
        batch_size: usize,
        config: &EncoderConfig,
    ) -> Result<()> {
        assert_eq!(
            output.memory.dims3()?,
            (seq_len, batch_size, config.d_model)
        );
        assert_eq!(
            output.pos_embed.dims3()?,
            (seq_len, batch_size, config.d_model)
        );
        assert_eq!(output.padding_mask.dims2()?, (seq_len, batch_size));
        assert_eq!(output.spatial_shapes.to_vec2::<u32>()?, vec![vec![2, 2]]);
        assert_eq!(output.level_start_index.to_vec1::<u32>()?, vec![0]);
        assert_eq!(
            output.valid_ratios.dims3()?,
            (batch_size, config.num_feature_levels, 2)
        );
        Ok(())
    }

    fn test_config() -> EncoderConfig {
        EncoderConfig {
            d_model: 8,
            num_layers: 1,
            num_feature_levels: 1,
            num_heads: 2,
            dim_feedforward: 16,
            add_pooled_text_to_image: false,
            pool_text_with_mask: true,
        }
    }

    fn encoder_weights(config: &EncoderConfig, device: &Device) -> Result<HashMap<String, Tensor>> {
        let mut tensors = HashMap::new();
        tensors.insert(
            "layers.0.self_attn.in_proj_weight".into(),
            Tensor::zeros((config.d_model * 3, config.d_model), DType::F32, device)?,
        );
        tensors.insert(
            "layers.0.self_attn.in_proj_bias".into(),
            Tensor::zeros(config.d_model * 3, DType::F32, device)?,
        );
        tensors.insert(
            "layers.0.self_attn.out_proj.weight".into(),
            Tensor::zeros((config.d_model, config.d_model), DType::F32, device)?,
        );
        tensors.insert(
            "layers.0.self_attn.out_proj.bias".into(),
            Tensor::zeros(config.d_model, DType::F32, device)?,
        );
        tensors.insert(
            "layers.0.cross_attn_image.in_proj_weight".into(),
            Tensor::zeros((config.d_model * 3, config.d_model), DType::F32, device)?,
        );
        tensors.insert(
            "layers.0.cross_attn_image.in_proj_bias".into(),
            Tensor::zeros(config.d_model * 3, DType::F32, device)?,
        );
        tensors.insert(
            "layers.0.cross_attn_image.out_proj.weight".into(),
            Tensor::zeros((config.d_model, config.d_model), DType::F32, device)?,
        );
        tensors.insert(
            "layers.0.cross_attn_image.out_proj.bias".into(),
            Tensor::zeros(config.d_model, DType::F32, device)?,
        );
        tensors.insert(
            "layers.0.linear1.weight".into(),
            Tensor::zeros((config.dim_feedforward, config.d_model), DType::F32, device)?,
        );
        tensors.insert(
            "layers.0.linear1.bias".into(),
            Tensor::zeros(config.dim_feedforward, DType::F32, device)?,
        );
        tensors.insert(
            "layers.0.linear2.weight".into(),
            Tensor::zeros((config.d_model, config.dim_feedforward), DType::F32, device)?,
        );
        tensors.insert(
            "layers.0.linear2.bias".into(),
            Tensor::zeros(config.d_model, DType::F32, device)?,
        );
        for norm_name in ["norm1", "norm2", "norm3"] {
            tensors.insert(
                format!("layers.0.{norm_name}.weight"),
                Tensor::ones(config.d_model, DType::F32, device)?,
            );
            tensors.insert(
                format!("layers.0.{norm_name}.bias"),
                Tensor::zeros(config.d_model, DType::F32, device)?,
            );
        }
        Ok(tensors)
    }

    fn run_fixture_fusion(device: &Device) -> Result<FusionEncoderOutput> {
        let config = fusion_fixture_config()?;
        let weights = load_fusion_fixture_tensors("encoder_weights.safetensors", device)?;
        let fixture = load_fusion_fixture_tensors("fixture.safetensors", device)?;
        let vb = VarBuilder::from_tensors(weights, DType::F32, &device);
        let encoder = Sam3FusionEncoder::new(&config, vb)?;
        let visual_features = fixture_feature_levels(&fixture, "inputs.backbone_fpn");
        let visual_pos = fixture_feature_levels(&fixture, "inputs.vision_pos_enc");
        let prompt = EncodedPrompt {
            features: fixture_tensor(&fixture, "inputs.prompt")?.clone(),
            padding_mask: fixture_tensor(&fixture, "inputs.prompt_mask")?.to_dtype(DType::U8)?,
        };
        encoder.forward(&visual_features, &visual_pos, &prompt)
    }

    fn fusion_fixture_config() -> Result<EncoderConfig> {
        let path = fusion_fixture_dir().join("metadata.json");
        let contents = fs::read_to_string(&path).map_err(|err| {
            candle::Error::Msg(format!(
                "failed to read fusion fixture metadata {}: {err}",
                path.display()
            ))
        })?;
        let metadata: serde_json::Value = serde_json::from_str(&contents).map_err(|err| {
            candle::Error::Msg(format!(
                "failed to parse fusion fixture metadata {}: {err}",
                path.display()
            ))
        })?;
        Ok(EncoderConfig {
            d_model: metadata["d_model"]
                .as_u64()
                .expect("fusion fixture d_model should be an integer")
                as usize,
            num_layers: metadata["num_layers"]
                .as_u64()
                .expect("fusion fixture num_layers should be an integer")
                as usize,
            num_feature_levels: metadata["num_feature_levels"]
                .as_u64()
                .expect("fusion fixture num_feature_levels should be an integer")
                as usize,
            num_heads: metadata["num_heads"]
                .as_u64()
                .expect("fusion fixture num_heads should be an integer")
                as usize,
            dim_feedforward: metadata["dim_feedforward"]
                .as_u64()
                .expect("fusion fixture dim_feedforward should be an integer")
                as usize,
            add_pooled_text_to_image: false,
            pool_text_with_mask: true,
        })
    }

    fn fusion_fixture_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/sam3_fusion_unit")
    }

    fn load_fusion_fixture_tensors(
        file_name: &str,
        device: &Device,
    ) -> Result<HashMap<String, Tensor>> {
        let path = fusion_fixture_dir().join(file_name);
        candle::safetensors::load(&path, device).map_err(|err| {
            candle::Error::Msg(format!(
                "failed to load fusion fixture {}: {err}",
                path.display()
            ))
        })
    }

    fn fixture_feature_levels(fixture: &HashMap<String, Tensor>, prefix: &str) -> Vec<Tensor> {
        let mut levels = Vec::new();
        for idx in 0.. {
            let key = format!("{prefix}.{idx}");
            let Some(level) = fixture.get(&key) else {
                break;
            };
            levels.push(level.clone());
        }
        levels
    }

    fn fixture_tensor<'a>(fixture: &'a HashMap<String, Tensor>, key: &str) -> Result<&'a Tensor> {
        fixture
            .get(key)
            .ok_or_else(|| candle::Error::Msg(format!("fusion fixture is missing tensor `{key}`")))
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
        if max_abs_diff > atol {
            candle::bail!("{name}: max_abs_diff={max_abs_diff:.8} exceeded atol={atol:.8}");
        }
        Ok(())
    }
}
