"""INT8 scaled matrix-multiplication kernel for the NPU.

SGLang's ``w8a8_int8`` linear layer runs on the NPU through the
``per_token_quant_int8`` + ``int8_scaled_mm`` path. This module implements
``int8_scaled_mm`` on top of the native CANN int8 quantized GEMM operator
``torch.ops.npu.npu_quant_matmul``:

  - activation: per-token symmetric int8 quantization, scale = max(|x|) / 127
  - weight:     static per-channel symmetric int8 quantization
  - compute:    out = (A_q @ W_q^T) * scale_a * scale_w + bias, bf16/fp16 out

The compute runs on the AI Core int8 cube unit, which keeps both precision
and performance.
"""

import torch
import torch_npu  # noqa: F401  makes torch.ops.npu available


def per_token_quant_int8(x, scale_dtype=torch.float32):
    """Per-token symmetric int8 quantization of an activation tensor.

    Delegates to the native CANN per-token dynamic quant operator
    ``torch.ops.npu.npu_dynamic_quant``, which fuses the per-token
    ``max(|x|)`` reduction and the quantization into a single op:

        scale = max(|x|) / 127
        x_q   = round(x / scale)

    Args:
        x: [..., K] bf16/fp16 tensor.
        scale_dtype: dtype of the returned scale (default float32).

    Returns:
        (x_q, scale): x_q is an int8 tensor with the same shape as x;
        scale is [..., 1] with scale = max(|x|) / 127.
    """
    x_q, scale = torch.ops.npu.npu_dynamic_quant(x)
    if scale_dtype != torch.float32:
        scale = scale.to(scale_dtype)
    return x_q, scale.unsqueeze(-1)


def _npu_quant_matmul_2d(x_q, weight, scale_b, pertoken_scale, bias, out_dtype):
    """Core int8 scaled GEMM backed by the native CANN ``npu_quant_matmul``.

    Args:
        x_q: [M, K] int8 quantized activation (row-major).
        weight: int8 weight in [K, N] shape, i.e. the column-major view sglang
            produces via ``layer.weight = weight.t()`` on non-CPU platforms.
        scale_b: [N] float32 per-channel weight scale.
        pertoken_scale: [M] float32 per-token activation scale.
        bias: [N] optional bias.
        out_dtype: output dtype, bf16 or fp16.

    Returns:
        [M, N] tensor: out = (x_q @ weight) * pertoken_scale * scale_b + bias.
    """
    k = x_q.shape[1]
    if weight.shape[0] != k:
        raise ValueError(
            f"weight shape {tuple(weight.shape)} incompatible with K={k} "
            f"(expect [K, N])"
        )
    weight_nk = weight.contiguous()
    if bias is not None:
        # npu_quant_matmul only supports int32/fp16/fp32 bias (not bf16).
        bias = bias.contiguous().to(torch.float32)
    return torch.ops.npu.npu_quant_matmul(
        x_q,
        weight_nk,
        scale_b,
        offset=None,
        pertoken_scale=pertoken_scale,
        bias=bias,
        output_dtype=out_dtype,
    )


def int8_scaled_mm(mat_a, mat_b, scales_a, scales_b, out_dtype, bias=None):
    """int8 scaled GEMM: out = (A @ B) * scales_a * scales_b + bias.

    Args:
        mat_a: [M, K] int8 (quantized activation, row-major).
        mat_b: int8 weight in [K, N] shape, i.e. the column-major view sglang
            produces via ``layer.weight = weight.t()`` (matching the CUDA
            ``sgl_kernel`` semantics).
        scales_a: [M] or [M, 1] float32 per-token activation scale.
        scales_b: [N] or [N, 1] float32 per-channel weight scale.
        out_dtype: output dtype, bf16 or fp16.
        bias: [N] optional bias.

    Returns:
        [M, N] tensor.
    """
    scale_a = scales_a.reshape(-1)
    scale_b = scales_b.reshape(-1)
    return _npu_quant_matmul_2d(
        mat_a, mat_b, scale_b, scale_a, bias=bias, out_dtype=out_dtype
    )
