use candle::{DType, Result, Tensor, D};
use candle_nn::{Embedding, LayerNorm, Linear, Module, VarBuilder};

use super::config::TextConfig;

#[derive(Debug)]
pub struct TextEncoding {
    /// SAM3 uses a key-padding mask with `1` for padding and `0` for valid tokens.
    pub attention_mask: Tensor,
    /// Sequence-first resized text memory, shape `[seq, batch, d_model]`.
    pub memory: Tensor,
    /// Sequence-first token embeddings before the transformer, shape `[seq, batch, width]`.
    pub input_embeddings: Tensor,
}

#[derive(Debug)]
struct Sam3TextAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl Sam3TextAttention {
    fn new(config: &TextConfig, vb: VarBuilder) -> Result<Self> {
        let width = config.width;
        let num_heads = config.heads;
        let in_proj_weight = vb.get((3 * width, width), "in_proj_weight")?;
        let in_proj_bias = vb.get(3 * width, "in_proj_bias")?;
        let split_weights = in_proj_weight.chunk(3, 0)?;
        let split_biases = in_proj_bias.chunk(3, 0)?;
        let q_proj = Linear::new(split_weights[0].clone(), Some(split_biases[0].clone()));
        let k_proj = Linear::new(split_weights[1].clone(), Some(split_biases[1].clone()));
        let v_proj = Linear::new(split_weights[2].clone(), Some(split_biases[2].clone()));
        let out_proj = candle_nn::linear(width, width, vb.pp("out_proj"))?;
        let head_dim = width / num_heads;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            num_heads,
            head_dim,
            scale: (head_dim as f64).powf(-0.5),
        })
    }

    fn shape(&self, xs: &Tensor, batch_size: usize, seq_len: usize) -> Result<Tensor> {
        xs.reshape((batch_size, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()
    }

    fn forward(&self, xs: &Tensor, causal_mask: &Tensor) -> Result<Tensor> {
        let in_dtype = xs.dtype();
        let (batch_size, seq_len, width) = xs.dims3()?;
        let projected_shape = (batch_size * self.num_heads, seq_len, self.head_dim);
        let q = (self.q_proj.forward(xs)? * self.scale)?
            .reshape((batch_size, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .reshape(projected_shape)?
            .to_dtype(DType::F32)?;
        let k = self
            .shape(&self.k_proj.forward(xs)?, batch_size, seq_len)?
            .reshape(projected_shape)?
            .to_dtype(DType::F32)?;
        let v = self
            .shape(&self.v_proj.forward(xs)?, batch_size, seq_len)?
            .reshape(projected_shape)?
            .to_dtype(DType::F32)?;
        let attn = q.matmul(&k.transpose(1, 2)?)?;
        let attn = attn
            .reshape((batch_size, self.num_heads, seq_len, seq_len))?
            .broadcast_add(causal_mask)?
            .reshape((batch_size * self.num_heads, seq_len, seq_len))?;
        let attn = candle_nn::ops::softmax(&attn, D::Minus1)?;
        let output = attn.matmul(&v)?.to_dtype(in_dtype)?;
        let output = output
            .reshape((batch_size, self.num_heads, seq_len, self.head_dim))?
            .transpose(1, 2)?
            .reshape((batch_size, seq_len, width))?;
        self.out_proj.forward(&output)
    }
}

#[derive(Debug)]
struct Sam3TextMlp {
    c_fc: Linear,
    c_proj: Linear,
}

impl Sam3TextMlp {
    fn new(config: &TextConfig, vb: VarBuilder) -> Result<Self> {
        let hidden_width = config.width * 4;
        Ok(Self {
            c_fc: candle_nn::linear(config.width, hidden_width, vb.pp("c_fc"))?,
            c_proj: candle_nn::linear(hidden_width, config.width, vb.pp("c_proj"))?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.c_proj.forward(&self.c_fc.forward(xs)?.gelu_erf()?)
    }
}

#[derive(Debug)]
struct Sam3ResidualAttentionBlock {
    attn: Sam3TextAttention,
    ln_1: LayerNorm,
    mlp: Sam3TextMlp,
    ln_2: LayerNorm,
}

impl Sam3ResidualAttentionBlock {
    fn new(config: &TextConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            attn: Sam3TextAttention::new(config, vb.pp("attn"))?,
            ln_1: candle_nn::layer_norm(config.width, 1e-5, vb.pp("ln_1"))?,
            mlp: Sam3TextMlp::new(config, vb.pp("mlp"))?,
            ln_2: candle_nn::layer_norm(config.width, 1e-5, vb.pp("ln_2"))?,
        })
    }

    fn forward(&self, xs: &Tensor, causal_mask: &Tensor) -> Result<Tensor> {
        let residual = xs;
        let xs = self.ln_1.forward(xs)?;
        let xs = self.attn.forward(&xs, causal_mask)?;
        let xs = (xs + residual)?;

        let residual = &xs;
        let xs = self.ln_2.forward(&xs)?;
        let xs = self.mlp.forward(&xs)?;
        xs + residual
    }
}

#[derive(Debug)]
struct Sam3TextTransformer {
    resblocks: Vec<Sam3ResidualAttentionBlock>,
}

impl Sam3TextTransformer {
    fn new(config: &TextConfig, vb: VarBuilder) -> Result<Self> {
        let mut resblocks = Vec::with_capacity(config.layers);
        let vb = vb.pp("resblocks");
        for layer_idx in 0..config.layers {
            resblocks.push(Sam3ResidualAttentionBlock::new(
                config,
                vb.pp(layer_idx.to_string()),
            )?);
        }
        Ok(Self { resblocks })
    }

    fn forward(&self, xs: &Tensor, causal_mask: &Tensor) -> Result<Tensor> {
        let mut xs = xs.clone();
        for block in self.resblocks.iter() {
            xs = block.forward(&xs, causal_mask)?;
        }
        Ok(xs)
    }
}

#[derive(Debug)]
pub struct Sam3TextEncoder {
    config: TextConfig,
    token_embedding: Embedding,
    positional_embedding: Tensor,
    transformer: Sam3TextTransformer,
    ln_final: LayerNorm,
    resizer: Linear,
    causal_attention_mask: Tensor,
}

impl Sam3TextEncoder {
    pub fn new(config: &TextConfig, vb: VarBuilder) -> Result<Self> {
        if config.heads == 0 || config.width % config.heads != 0 {
            candle::bail!(
                "sam3 text width ({}) must be divisible by heads ({})",
                config.width,
                config.heads
            )
        }
        let encoder_vb = vb.pp("encoder");
        let token_embedding = candle_nn::embedding(
            config.vocab_size,
            config.width,
            encoder_vb.pp("token_embedding"),
        )?;
        let positional_embedding = encoder_vb.get(
            (config.context_length, config.width),
            "positional_embedding",
        )?;
        let transformer = Sam3TextTransformer::new(config, encoder_vb.pp("transformer"))?;
        let ln_final = candle_nn::layer_norm(config.width, 1e-5, encoder_vb.pp("ln_final"))?;
        let resizer = candle_nn::linear(config.width, config.d_model, vb.pp("resizer"))?;
        let causal_attention_mask =
            Self::build_causal_attention_mask(config.context_length, vb.device())?;
        Ok(Self {
            config: config.clone(),
            token_embedding,
            positional_embedding,
            transformer,
            ln_final,
            resizer,
            causal_attention_mask,
        })
    }

    fn build_causal_attention_mask(seq_len: usize, device: &candle::Device) -> Result<Tensor> {
        let mut mask = vec![0f32; seq_len * seq_len];
        for row in 0..seq_len {
            for col in (row + 1)..seq_len {
                mask[row * seq_len + col] = f32::NEG_INFINITY;
            }
        }
        Tensor::from_slice(&mask, (seq_len, seq_len), device)
    }

    pub fn config(&self) -> &TextConfig {
        &self.config
    }

    pub fn forward(&self, input_ids: &Tensor, attention_mask: &Tensor) -> Result<TextEncoding> {
        let (batch_size, seq_len) = input_ids.dims2()?;
        let attention_mask_shape = attention_mask.dims2()?;
        if attention_mask_shape != (batch_size, seq_len) {
            candle::bail!(
                "sam3 text attention mask shape mismatch, expected ({batch_size}, {seq_len}), got {attention_mask_shape:?}"
            )
        }
        if seq_len > self.config.context_length {
            candle::bail!(
                "sam3 text sequence length {seq_len} exceeds configured context length {}",
                self.config.context_length
            )
        }

        let inputs_embeds = self.token_embedding.forward(input_ids)?;
        let positional_embedding = self.positional_embedding.narrow(0, 0, seq_len)?;
        let hidden_states = inputs_embeds.broadcast_add(&positional_embedding)?;
        let causal_mask = self
            .causal_attention_mask
            .narrow(0, 0, seq_len)?
            .narrow(1, 0, seq_len)?
            .reshape((1, 1, seq_len, seq_len))?;
        let hidden_states = self.transformer.forward(&hidden_states, &causal_mask)?;
        let hidden_states = self.ln_final.forward(&hidden_states)?;

        let input_embeddings = inputs_embeds.transpose(0, 1)?.contiguous()?;
        let memory = hidden_states.transpose(0, 1)?.contiguous()?;
        let memory = self.resizer.forward(&memory)?;
        let attention_mask = attention_mask.to_dtype(DType::U8)?.eq(0u8)?;

        Ok(TextEncoding {
            attention_mask,
            memory,
            input_embeddings,
        })
    }
}

#[cfg(test)]
mod tests {
    use candle::{Device, Result, Tensor};
    use candle_nn::VarBuilder;

    use super::Sam3TextEncoder;
    use crate::models::sam3::TextConfig;

    #[test]
    fn text_encoder_returns_sequence_first_outputs() -> Result<()> {
        let device = Device::Cpu;
        let config = TextConfig {
            d_model: 3,
            width: 4,
            heads: 2,
            layers: 1,
            context_length: 5,
            vocab_size: 16,
        };
        let encoder =
            Sam3TextEncoder::new(&config, VarBuilder::zeros(candle::DType::F32, &device))?;
        let input_ids = Tensor::new(&[[1u32, 2, 3, 0], [4, 5, 0, 0]], &device)?;
        let attention_mask = Tensor::new(&[[1u8, 1, 1, 0], [1, 1, 0, 0]], &device)?;
        let encoding = encoder.forward(&input_ids, &attention_mask)?;

        assert_eq!(
            encoding.attention_mask.to_vec2::<u8>()?,
            [[0, 0, 0, 1], [0, 0, 1, 1]]
        );
        assert_eq!(encoding.input_embeddings.dims3()?, (4, 2, 4));
        assert_eq!(encoding.memory.dims3()?, (4, 2, 3));
        Ok(())
    }

    #[test]
    fn text_encoder_rejects_sequences_longer_than_context() -> Result<()> {
        let device = Device::Cpu;
        let config = TextConfig {
            context_length: 3,
            ..TextConfig::default()
        };
        let encoder =
            Sam3TextEncoder::new(&config, VarBuilder::zeros(candle::DType::F32, &device))?;
        let input_ids = Tensor::new(&[[1u32, 2, 3, 4]], &device)?;
        let attention_mask = Tensor::new(&[[1u8, 1, 1, 1]], &device)?;
        let err = encoder.forward(&input_ids, &attention_mask).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("exceeds configured context length"));
        Ok(())
    }
}
