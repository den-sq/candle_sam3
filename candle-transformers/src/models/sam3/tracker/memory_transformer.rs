use super::*;

#[derive(Debug)]
struct TrackerRoPEAttention {
    num_heads: usize,
    head_dim: usize,
    scale: f64,
    freqs_real: Tensor,
    freqs_imag: Tensor,
    rope_k_repeat: bool,
}

impl TrackerRoPEAttention {
    fn new(config: &Sam3TrackerAttentionConfig, device: &Device) -> Result<Self> {
        if config.embedding_dim % config.num_heads != 0 {
            candle::bail!(
                "tracker attention embedding_dim {} must be divisible by num_heads {}",
                config.embedding_dim,
                config.num_heads
            );
        }
        let head_dim = config.embedding_dim / config.num_heads;
        if head_dim % 4 != 0 {
            candle::bail!("tracker attention head_dim must be divisible by 4, got {head_dim}");
        }
        let (freqs_real, freqs_imag) = compute_tracker_axial_freqs(
            head_dim,
            config.feat_sizes[0],
            config.feat_sizes[1],
            config.rope_theta,
            device,
        )?;
        Ok(Self {
            num_heads: config.num_heads,
            head_dim,
            scale: (head_dim as f64).powf(-0.5),
            freqs_real,
            freqs_imag,
            rope_k_repeat: config.rope_k_repeat,
        })
    }

    fn forward(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        num_k_exclude_rope: usize,
    ) -> Result<Tensor> {
        let in_dtype = q.dtype();
        let (batch_size, q_seq_len, q_dim) = q.dims3()?;
        let (_, k_seq_len, k_dim) = k.dims3()?;
        let (_, v_seq_len, v_dim) = v.dims3()?;
        if k_seq_len != v_seq_len {
            candle::bail!(
                "tracker attention expected key/value sequence lengths to match, got {k_seq_len} and {v_seq_len}"
            );
        }
        if q_dim != self.num_heads * self.head_dim
            || k_dim != self.num_heads * self.head_dim
            || v_dim != self.num_heads * self.head_dim
        {
            candle::bail!(
                "tracker attention expected q/k/v dims to match num_heads*head_dim={}, got q={}, k={}, v={}",
                self.num_heads * self.head_dim,
                q_dim,
                k_dim,
                v_dim
            );
        }
        let q = q
            .reshape((batch_size, q_seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .to_dtype(DType::F32)?;
        let k = k
            .reshape((batch_size, k_seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .to_dtype(DType::F32)?;
        let v = v
            .reshape((batch_size, v_seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .to_dtype(DType::F32)?;
        let (q, k) = apply_tracker_axial_rotary(
            &q,
            &k,
            &self.freqs_real,
            &self.freqs_imag,
            self.rope_k_repeat,
            num_k_exclude_rope,
        )?;
        let q = (q * self.scale)?;
        let attn = q.matmul(&k.transpose(2, 3)?)?;
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        let out = attn.matmul(&v)?;
        out.transpose(1, 2)?
            .reshape((batch_size, q_seq_len, self.num_heads * self.head_dim))?
            .to_dtype(in_dtype)
    }
}

#[derive(Debug)]
struct TrackerTransformerLayer {
    self_attn_q_proj: Linear,
    self_attn_k_proj: Linear,
    self_attn_v_proj: Linear,
    self_attn_out_proj: Linear,
    cross_attn_q_proj: Linear,
    cross_attn_k_proj: Linear,
    cross_attn_v_proj: Linear,
    cross_attn_out_proj: Linear,
    self_attention_rope: TrackerRoPEAttention,
    cross_attention_rope: TrackerRoPEAttention,
    linear1: Linear,
    linear2: Linear,
    norm1: LayerNorm,
    norm2: LayerNorm,
    norm3: LayerNorm,
    pos_enc_at_attn: bool,
    pos_enc_at_cross_attn_queries: bool,
    pos_enc_at_cross_attn_keys: bool,
    cross_attention_first: bool,
}

impl TrackerTransformerLayer {
    fn new(config: &Sam3TrackerTransformerConfig, vb: VarBuilder) -> Result<Self> {
        let d_model = config.d_model;
        let cross_kv_dim = config.cross_attention.kv_in_dim.unwrap_or(d_model);
        let self_attn_q_proj = linear(vb.pp("self_attn").pp("q_proj"), d_model, d_model, true)?;
        let self_attn_k_proj = linear(vb.pp("self_attn").pp("k_proj"), d_model, d_model, true)?;
        let self_attn_v_proj = linear(vb.pp("self_attn").pp("v_proj"), d_model, d_model, true)?;
        let self_attn_out_proj = linear(vb.pp("self_attn").pp("out_proj"), d_model, d_model, true)?;
        let cross_attn_q_proj = linear(
            vb.pp("cross_attn_image").pp("q_proj"),
            d_model,
            d_model,
            true,
        )?;
        let cross_attn_k_proj = linear(
            vb.pp("cross_attn_image").pp("k_proj"),
            cross_kv_dim,
            d_model,
            true,
        )?;
        let cross_attn_v_proj = linear(
            vb.pp("cross_attn_image").pp("v_proj"),
            cross_kv_dim,
            d_model,
            true,
        )?;
        let cross_attn_out_proj = linear(
            vb.pp("cross_attn_image").pp("out_proj"),
            d_model,
            d_model,
            true,
        )?;
        Ok(Self {
            self_attn_q_proj,
            self_attn_k_proj,
            self_attn_v_proj,
            self_attn_out_proj,
            cross_attn_q_proj,
            cross_attn_k_proj,
            cross_attn_v_proj,
            cross_attn_out_proj,
            self_attention_rope: TrackerRoPEAttention::new(&config.self_attention, vb.device())?,
            cross_attention_rope: TrackerRoPEAttention::new(&config.cross_attention, vb.device())?,
            linear1: linear(
                vb.pp("linear1"),
                d_model,
                config.layer.dim_feedforward,
                true,
            )?,
            linear2: linear(
                vb.pp("linear2"),
                config.layer.dim_feedforward,
                d_model,
                true,
            )?,
            norm1: candle_nn::layer_norm(d_model, 1e-5, vb.pp("norm1"))?,
            norm2: candle_nn::layer_norm(d_model, 1e-5, vb.pp("norm2"))?,
            norm3: candle_nn::layer_norm(d_model, 1e-5, vb.pp("norm3"))?,
            pos_enc_at_attn: config.layer.pos_enc_at_attn,
            pos_enc_at_cross_attn_queries: config.layer.pos_enc_at_cross_attn_queries,
            pos_enc_at_cross_attn_keys: config.layer.pos_enc_at_cross_attn_keys,
            cross_attention_first: config.layer.cross_attention_first,
        })
    }

    fn forward_self_attention(&self, tgt: &Tensor, query_pos: Option<&Tensor>) -> Result<Tensor> {
        let tgt2 = self.norm1.forward(tgt)?;
        let qk = match (self.pos_enc_at_attn, query_pos) {
            (true, Some(query_pos)) => tgt2.broadcast_add(query_pos)?,
            _ => tgt2.clone(),
        };
        let q = self.self_attn_q_proj.forward(&qk)?;
        let k = self.self_attn_k_proj.forward(&qk)?;
        let v = self.self_attn_v_proj.forward(&tgt2)?;
        let out = self.self_attention_rope.forward(&q, &k, &v, 0)?;
        let tgt2 = self.self_attn_out_proj.forward(&out)?;
        tgt.broadcast_add(&tgt2)
    }

    fn forward_cross_attention(
        &self,
        tgt: &Tensor,
        memory: &Tensor,
        query_pos: Option<&Tensor>,
        memory_pos: Option<&Tensor>,
        num_k_exclude_rope: usize,
    ) -> Result<Tensor> {
        let tgt2 = self.norm2.forward(tgt)?;
        let mut q_input = tgt2.clone();
        if self.pos_enc_at_cross_attn_queries {
            if let Some(query_pos) = query_pos {
                q_input = q_input.broadcast_add(query_pos)?;
            }
        }
        let q = self.cross_attn_q_proj.forward(&q_input)?;
        let mut k_input = memory.clone();
        if self.pos_enc_at_cross_attn_keys {
            if let Some(memory_pos) = memory_pos {
                k_input = k_input.broadcast_add(memory_pos)?;
            }
        }
        let k = self.cross_attn_k_proj.forward(&k_input)?;
        let v = self.cross_attn_v_proj.forward(memory)?;
        let out = self
            .cross_attention_rope
            .forward(&q, &k, &v, num_k_exclude_rope)?;
        let tgt2 = self.cross_attn_out_proj.forward(&out)?;
        tgt.broadcast_add(&tgt2)
    }

    fn forward(
        &self,
        tgt: &Tensor,
        memory: &Tensor,
        query_pos: Option<&Tensor>,
        memory_pos: Option<&Tensor>,
        num_k_exclude_rope: usize,
    ) -> Result<Tensor> {
        let tgt = if self.cross_attention_first {
            let tgt = self.forward_cross_attention(
                tgt,
                memory,
                query_pos,
                memory_pos,
                num_k_exclude_rope,
            )?;
            self.forward_self_attention(&tgt, query_pos)?
        } else {
            let tgt = self.forward_self_attention(tgt, query_pos)?;
            self.forward_cross_attention(&tgt, memory, query_pos, memory_pos, num_k_exclude_rope)?
        };
        let tgt2 = self.norm3.forward(&tgt)?;
        let tgt2 = self
            .linear2
            .forward(&self.linear1.forward(&tgt2)?.relu()?)?;
        tgt.broadcast_add(&tgt2)
    }
}

#[derive(Debug)]
pub(super) struct TrackerMemoryTransformer {
    layers: Vec<TrackerTransformerLayer>,
    norm: LayerNorm,
    pos_enc_at_input: bool,
}

impl TrackerMemoryTransformer {
    pub(super) fn new(config: &Sam3TrackerTransformerConfig, vb: VarBuilder) -> Result<Self> {
        let layers_vb = vb.pp("layers");
        let mut layers = Vec::with_capacity(config.encoder.num_layers);
        for layer_idx in 0..config.encoder.num_layers {
            layers.push(TrackerTransformerLayer::new(
                config,
                layers_vb.pp(layer_idx),
            )?);
        }
        Ok(Self {
            layers,
            norm: candle_nn::layer_norm(config.d_model, 1e-5, vb.pp("norm"))?,
            pos_enc_at_input: config.encoder.pos_enc_at_input,
        })
    }

    pub(super) fn forward(
        &self,
        src: &Tensor,
        memory: &Tensor,
        src_pos: Option<&Tensor>,
        memory_pos: Option<&Tensor>,
        num_obj_ptr_tokens: usize,
    ) -> Result<Tensor> {
        let mut output = if self.pos_enc_at_input {
            match src_pos {
                Some(src_pos) => src.broadcast_add(&(src_pos * 0.1f64)?)?,
                None => src.clone(),
            }
        } else {
            src.clone()
        };
        for layer in self.layers.iter() {
            output = layer.forward(&output, memory, src_pos, memory_pos, num_obj_ptr_tokens)?;
        }
        self.norm.forward(&output)
    }
}

fn compute_tracker_axial_freqs(
    dim: usize,
    end_x: usize,
    end_y: usize,
    theta: f32,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    if dim % 4 != 0 {
        candle::bail!("tracker rotary dim must be divisible by 4, got {dim}");
    }
    let rotary_dim = dim / 4;
    let seq_len = end_x * end_y;
    let inv_freqs: Vec<f32> = (0..rotary_dim)
        .map(|i| 1f32 / theta.powf((4 * i) as f32 / dim as f32))
        .collect();
    let mut freqs_real = vec![0f32; seq_len * (dim / 2)];
    let mut freqs_imag = vec![0f32; seq_len * (dim / 2)];
    for flat_idx in 0..seq_len {
        let x_pos = (flat_idx % end_x) as f32;
        let y_pos = (flat_idx / end_x) as f32;
        let row_real = &mut freqs_real[flat_idx * (dim / 2)..(flat_idx + 1) * (dim / 2)];
        let row_imag = &mut freqs_imag[flat_idx * (dim / 2)..(flat_idx + 1) * (dim / 2)];
        for (i, inv_freq) in inv_freqs.iter().copied().enumerate() {
            let x_freq = x_pos * inv_freq;
            let y_freq = y_pos * inv_freq;
            row_real[i] = x_freq.cos();
            row_imag[i] = x_freq.sin();
            row_real[rotary_dim + i] = y_freq.cos();
            row_imag[rotary_dim + i] = y_freq.sin();
        }
    }
    Ok((
        Tensor::from_slice(&freqs_real, (seq_len, dim / 2), device)?,
        Tensor::from_slice(&freqs_imag, (seq_len, dim / 2), device)?,
    ))
}

fn apply_tracker_rotary_single(
    xs: &Tensor,
    freqs_real: &Tensor,
    freqs_imag: &Tensor,
) -> Result<Tensor> {
    let (batch_size, num_heads, seq_len, head_dim) = xs.dims4()?;
    let xs_dtype = xs.dtype();
    let xs = xs
        .to_dtype(DType::F32)?
        .reshape((batch_size, num_heads, seq_len, head_dim / 2, 2))?;
    let xs_real = xs.i((.., .., .., .., 0))?;
    let xs_imag = xs.i((.., .., .., .., 1))?;
    let freqs_real = freqs_real.reshape((1, 1, seq_len, head_dim / 2))?;
    let freqs_imag = freqs_imag.reshape((1, 1, seq_len, head_dim / 2))?;
    let real = (xs_real.broadcast_mul(&freqs_real)? - xs_imag.broadcast_mul(&freqs_imag)?)?;
    let imag = (xs_real.broadcast_mul(&freqs_imag)? + xs_imag.broadcast_mul(&freqs_real)?)?;
    Tensor::stack(&[&real, &imag], 4)?
        .reshape((batch_size, num_heads, seq_len, head_dim))?
        .to_dtype(xs_dtype)
}

fn apply_tracker_axial_rotary(
    q: &Tensor,
    k: &Tensor,
    freqs_real: &Tensor,
    freqs_imag: &Tensor,
    repeat_freqs_k: bool,
    num_k_exclude_rope: usize,
) -> Result<(Tensor, Tensor)> {
    let q_seq_len = q.dim(2)?;
    let k_seq_len = k.dim(2)?;
    let q_freqs_real = freqs_real.narrow(0, 0, q_seq_len)?;
    let q_freqs_imag = freqs_imag.narrow(0, 0, q_seq_len)?;
    let q = apply_tracker_rotary_single(q, &q_freqs_real, &q_freqs_imag)?;

    let num_k_rope = k_seq_len.saturating_sub(num_k_exclude_rope);
    if num_k_rope == 0 {
        return Ok((q, k.clone()));
    }
    let mut k_freqs_real = q_freqs_real.clone();
    let mut k_freqs_imag = q_freqs_imag.clone();
    if num_k_rope != q_seq_len {
        if !repeat_freqs_k || num_k_rope % q_seq_len != 0 {
            candle::bail!(
                "tracker rotary expected key rope length {num_k_rope} to equal query length {q_seq_len} or be a whole-number repeat"
            );
        }
        let repeat_factor = num_k_rope / q_seq_len;
        k_freqs_real = k_freqs_real.repeat((repeat_factor, 1))?;
        k_freqs_imag = k_freqs_imag.repeat((repeat_factor, 1))?;
    }
    let k_rope = k.narrow(2, 0, num_k_rope)?;
    let k_rope = apply_tracker_rotary_single(&k_rope, &k_freqs_real, &k_freqs_imag)?;
    let k = if num_k_rope < k_seq_len {
        let k_tail = k.narrow(2, num_k_rope, k_seq_len - num_k_rope)?;
        Tensor::cat(&[&k_rope, &k_tail], 2)?
    } else {
        k_rope
    };
    Ok((q, k))
}
