use candle::{DType, Device, Result, Tensor};

pub(crate) fn resize_bilinear2d_antialias(
    input: &Tensor,
    out_h: usize,
    out_w: usize,
) -> Result<Tensor> {
    let input_cpu = input.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
    let (batch, channels, in_h, in_w) = input_cpu.dims4()?;
    let input_vec = input_cpu.flatten_all()?.to_vec1::<f32>()?;
    let width_weights = antialias_linear_weights(in_w, out_w);
    let height_weights = antialias_linear_weights(in_h, out_h);
    let mut horizontal = vec![0.0f32; batch * channels * in_h * out_w];
    let mut output = vec![0.0f32; batch * channels * out_h * out_w];
    let input_stride_c = in_h * in_w;
    let input_stride_b = channels * input_stride_c;
    let horizontal_stride_c = in_h * out_w;
    let horizontal_stride_b = channels * horizontal_stride_c;
    let output_stride_c = out_h * out_w;
    let output_stride_b = channels * output_stride_c;

    for b in 0..batch {
        for c in 0..channels {
            let input_base = b * input_stride_b + c * input_stride_c;
            let horizontal_base = b * horizontal_stride_b + c * horizontal_stride_c;
            let output_base = b * output_stride_b + c * output_stride_c;
            for y in 0..in_h {
                let row_offset = input_base + y * in_w;
                let horizontal_row_offset = horizontal_base + y * out_w;
                for (out_x, weights) in width_weights.iter().enumerate() {
                    let mut value = 0.0f32;
                    for (src_x, weight) in weights {
                        value += input_vec[row_offset + *src_x] * *weight;
                    }
                    horizontal[horizontal_row_offset + out_x] = value;
                }
            }
            for (out_y, weights) in height_weights.iter().enumerate() {
                let output_row_offset = output_base + out_y * out_w;
                for out_x in 0..out_w {
                    let mut value = 0.0f32;
                    for (src_y, weight) in weights {
                        value += horizontal[horizontal_base + *src_y * out_w + out_x] * *weight;
                    }
                    output[output_row_offset + out_x] = value;
                }
            }
        }
    }

    Tensor::from_vec(output, (batch, channels, out_h, out_w), &Device::Cpu)?
        .to_device(input.device())
}

fn antialias_linear_weights(input_size: usize, output_size: usize) -> Vec<Vec<(usize, f32)>> {
    let scale = input_size as f32 / output_size as f32;
    let support = scale.max(1.0);
    let radius = support;
    let mut all_weights = Vec::with_capacity(output_size);
    for out_idx in 0..output_size {
        let center = scale * (out_idx as f32 + 0.5) - 0.5;
        let xmin = (center - radius).floor() as isize;
        let xmax = (center + radius).ceil() as isize;
        let mut weights = Vec::new();
        let mut weight_sum = 0.0f32;
        for src_idx in xmin..=xmax {
            let distance = (src_idx as f32 - center) / support;
            let weight = (1.0 - distance.abs()).max(0.0) / support;
            if weight == 0.0 {
                continue;
            }
            let clamped = src_idx.clamp(0, input_size.saturating_sub(1) as isize) as usize;
            weights.push((clamped, weight));
            weight_sum += weight;
        }
        if weight_sum > 0.0 {
            for (_, weight) in weights.iter_mut() {
                *weight /= weight_sum;
            }
        }
        all_weights.push(weights);
    }
    all_weights
}
