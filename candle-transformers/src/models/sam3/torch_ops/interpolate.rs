use candle::{DType, Device, Result, Tensor, WithDType};
use half::{bf16, f16};

pub(crate) fn resize_bilinear2d_antialias(
    input: &Tensor,
    out_h: usize,
    out_w: usize,
) -> Result<Tensor> {
    let input_dtype = input.dtype();
    let compute_dtype = match input_dtype {
        // Keep model compute tensors native; callers accept the reduced accumulation precision.
        // If long-history drift becomes material, give F16 the same F32 boundary roundtrip as BF16.
        DType::F16 | DType::F32 => input_dtype,
        DType::BF16 if bf16_resize_supported(input.device()) => DType::BF16,
        _ => DType::F32,
    };
    let input = input.to_dtype(compute_dtype)?;
    let output = resize_bilinear2d_antialias_native(&input, out_h, out_w)?;
    if input_dtype == DType::BF16 && compute_dtype == DType::F32 {
        output.to_dtype(DType::BF16)
    } else {
        Ok(output)
    }
}

fn resize_bilinear2d_antialias_native(
    input: &Tensor,
    out_h: usize,
    out_w: usize,
) -> Result<Tensor> {
    let (batch, channels, in_h, in_w) = input.dims4()?;
    if in_h == out_h && in_w == out_w {
        return Ok(input.clone());
    }
    if out_h >= in_h && out_w >= in_w {
        return input.upsample_bilinear2d(out_h, out_w, false);
    }
    if in_h % out_h == 0 && in_w % out_w == 0 {
        let stride_h = in_h / out_h;
        let stride_w = in_w / out_w;
        if stride_h > 0 && stride_w > 0 {
            return input.avg_pool2d_with_stride((stride_h, stride_w), (stride_h, stride_w));
        }
    }

    let width_weights = antialias_linear_weights(in_w, out_w);
    let height_weights = antialias_linear_weights(in_h, out_h);
    match input.dtype() {
        DType::F16 => resize_bilinear2d_antialias_cpu::<f16>(
            input,
            (batch, channels, in_h, in_w),
            out_h,
            out_w,
            &width_weights,
            &height_weights,
        ),
        DType::BF16 => resize_bilinear2d_antialias_cpu::<bf16>(
            input,
            (batch, channels, in_h, in_w),
            out_h,
            out_w,
            &width_weights,
            &height_weights,
        ),
        DType::F32 => resize_bilinear2d_antialias_cpu::<f32>(
            input,
            (batch, channels, in_h, in_w),
            out_h,
            out_w,
            &width_weights,
            &height_weights,
        ),
        dtype => candle::bail!("unsupported dtype {dtype:?} for bilinear antialias resize"),
    }
}

trait AntialiasValue: WithDType + Copy {
    fn zero_value() -> Self;
    fn weighted_add(sum: Self, value: Self, weight: f32) -> Self;
}

impl AntialiasValue for f32 {
    fn zero_value() -> Self {
        0.0
    }

    fn weighted_add(sum: Self, value: Self, weight: f32) -> Self {
        sum + value * weight
    }
}

impl AntialiasValue for f16 {
    fn zero_value() -> Self {
        Self::ZERO
    }

    fn weighted_add(sum: Self, value: Self, weight: f32) -> Self {
        sum + value * Self::from_f32(weight)
    }
}

impl AntialiasValue for bf16 {
    fn zero_value() -> Self {
        Self::ZERO
    }

    fn weighted_add(sum: Self, value: Self, weight: f32) -> Self {
        sum + value * Self::from_f32(weight)
    }
}

fn resize_bilinear2d_antialias_cpu<T: AntialiasValue>(
    input: &Tensor,
    (batch, channels, in_h, in_w): (usize, usize, usize, usize),
    out_h: usize,
    out_w: usize,
    width_weights: &[Vec<(usize, f32)>],
    height_weights: &[Vec<(usize, f32)>],
) -> Result<Tensor> {
    let input_vec = input
        .to_device(&Device::Cpu)?
        .flatten_all()?
        .to_vec1::<T>()?;
    let mut horizontal = vec![T::zero_value(); batch * channels * in_h * out_w];
    let mut output = vec![T::zero_value(); batch * channels * out_h * out_w];
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
                    let mut value = T::zero_value();
                    for (src_x, weight) in weights {
                        value = T::weighted_add(value, input_vec[row_offset + *src_x], *weight);
                    }
                    horizontal[horizontal_row_offset + out_x] = value;
                }
            }
            for (out_y, weights) in height_weights.iter().enumerate() {
                let output_row_offset = output_base + out_y * out_w;
                for out_x in 0..out_w {
                    let mut value = T::zero_value();
                    for (src_y, weight) in weights {
                        value = T::weighted_add(
                            value,
                            horizontal[horizontal_base + *src_y * out_w + out_x],
                            *weight,
                        );
                    }
                    output[output_row_offset + out_x] = value;
                }
            }
        }
    }

    Tensor::from_vec(output, (batch, channels, out_h, out_w), &Device::Cpu)?
        .to_device(input.device())
}

fn bf16_resize_supported(_device: &Device) -> bool {
    #[cfg(feature = "cuda")]
    if let Device::Cuda(device) = _device {
        use candle::cuda_backend::cudarc::driver::{result, sys};

        // Candle's native BF16 CUDA resize/pooling kernels require Ampere (SM 8.0).
        let cuda_device = device.cuda_stream().context().cu_device();
        let major = unsafe {
            result::device::get_attribute(
                cuda_device,
                sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
            )
        }
        .unwrap_or(0);
        return major >= 8;
    }

    true
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

#[cfg(test)]
mod tests {
    use super::*;

    fn input(height: usize, width: usize) -> Result<Tensor> {
        let values = (0..2 * height * width)
            .map(|index| {
                let index = index as f32;
                (index * 0.173).sin() * 1.7 + (index * 0.071).cos() * 0.3
            })
            .collect::<Vec<_>>();
        Tensor::from_vec(values, (1, 2, height, width), &Device::Cpu)
    }

    fn max_abs_diff(lhs: &Tensor, rhs: &Tensor) -> Result<f32> {
        let lhs = lhs.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let rhs = rhs.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        Ok(lhs
            .iter()
            .zip(rhs.iter())
            .map(|(lhs, rhs)| (lhs - rhs).abs())
            .fold(0.0f32, f32::max))
    }

    #[test]
    fn preserves_native_float_dtype_across_all_resize_paths() -> Result<()> {
        let cases = [(3, 5, 3, 5), (3, 5, 7, 11), (8, 12, 4, 3), (7, 11, 4, 6)];
        for (in_h, in_w, out_h, out_w) in cases {
            let input_f32 = input(in_h, in_w)?;
            let reference = resize_bilinear2d_antialias(&input_f32, out_h, out_w)?;
            for (dtype, tolerance) in [
                (DType::F16, 0.003f32),
                (DType::BF16, 0.02f32),
                (DType::F32, 0.0f32),
            ] {
                let input = input_f32.to_dtype(dtype)?;
                let output = resize_bilinear2d_antialias(&input, out_h, out_w)?;
                assert_eq!(output.dtype(), dtype);
                assert!(
                    max_abs_diff(&reference, &output)? <= tolerance,
                    "{dtype:?} resize {in_h}x{in_w} -> {out_h}x{out_w} exceeded {tolerance}"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn non_float_input_retains_legacy_f32_output() -> Result<()> {
        let input = Tensor::zeros((1, 1, 3, 5), DType::U8, &Device::Cpu)?;
        let output = resize_bilinear2d_antialias(&input, 7, 11)?;
        assert_eq!(output.dtype(), DType::F32);
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_preserves_f16_and_compatibly_restores_bf16() -> Result<()> {
        if !candle::utils::cuda_is_available() {
            return Ok(());
        }
        let device = Device::new_cuda(0)?;
        let input = input(3, 5)?.to_device(&device)?;
        for dtype in [DType::F16, DType::BF16] {
            let output = resize_bilinear2d_antialias(&input.to_dtype(dtype)?, 7, 11)?;
            device.synchronize()?;
            assert_eq!(output.dtype(), dtype);
            assert_eq!(output.dims4()?, (1, 2, 7, 11));
        }
        Ok(())
    }
}
