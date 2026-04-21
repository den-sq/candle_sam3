use candle::{Result, Tensor};

pub(crate) fn get_1d_sine_pe(pos_inds: &Tensor, dim: usize) -> Result<Tensor> {
    if dim % 2 != 0 {
        candle::bail!("tracker temporal position encoding requires even dim, got {dim}");
    }
    let device = pos_inds.device();
    let dtype = pos_inds.dtype();
    let pe_dim = dim / 2;
    let mut dim_t = Vec::with_capacity(pe_dim);
    for idx in 0..pe_dim {
        let exponent = 2.0 * (idx / 2) as f32 / pe_dim as f32;
        dim_t.push(10_000f32.powf(exponent));
    }
    let dim_t = Tensor::from_vec(dim_t, pe_dim, device)?.to_dtype(dtype)?;
    let pos_embed = pos_inds.unsqueeze(1)?.broadcast_div(&dim_t)?;
    let sin = pos_embed.sin()?;
    let cos = pos_embed.cos()?;
    Tensor::cat(&[&sin, &cos], 1)
}

pub(crate) fn build_2d_sine_position_encoding(
    feature: &Tensor,
    num_pos_feats_total: usize,
    normalize: bool,
    scale: f32,
    temperature: f32,
) -> Result<Tensor> {
    let (batch_size, channels, height, width) = feature.dims4()?;
    if channels != num_pos_feats_total {
        candle::bail!(
            "2d sine position encoding expected feature width {num_pos_feats_total}, got {channels}"
        )
    }
    if num_pos_feats_total % 2 != 0 {
        candle::bail!("2d sine position encoding requires even width, got {num_pos_feats_total}")
    }
    let num_pos_feats = num_pos_feats_total / 2;
    let device = feature.device();
    let dtype = feature.dtype();
    let eps = 1e-6f32;
    let mut dim_t = Vec::with_capacity(num_pos_feats);
    for idx in 0..num_pos_feats {
        let exponent = 2.0 * (idx / 2) as f32 / num_pos_feats as f32;
        dim_t.push(temperature.powf(exponent));
    }
    let mut encoding = vec![0f32; num_pos_feats_total * height * width];
    for y in 0..height {
        let mut y_embed = (y + 1) as f32;
        if normalize {
            y_embed = y_embed / (height as f32 + eps) * scale;
        }
        for x in 0..width {
            let mut x_embed = (x + 1) as f32;
            if normalize {
                x_embed = x_embed / (width as f32 + eps) * scale;
            }
            for pair_idx in 0..(num_pos_feats / 2) {
                let even_idx = pair_idx * 2;
                let odd_idx = even_idx + 1;
                let y_even = y_embed / dim_t[even_idx];
                let y_odd = y_embed / dim_t[odd_idx];
                let x_even = x_embed / dim_t[even_idx];
                let x_odd = x_embed / dim_t[odd_idx];
                let spatial_index = y * width + x;
                encoding[even_idx * height * width + spatial_index] = y_even.sin();
                encoding[odd_idx * height * width + spatial_index] = y_odd.cos();
                encoding[(num_pos_feats + even_idx) * height * width + spatial_index] =
                    x_even.sin();
                encoding[(num_pos_feats + odd_idx) * height * width + spatial_index] = x_odd.cos();
            }
        }
    }
    let encoding = Tensor::from_slice(&encoding, (1, num_pos_feats_total, height, width), device)?;
    encoding.repeat((batch_size, 1, 1, 1))?.to_dtype(dtype)
}
