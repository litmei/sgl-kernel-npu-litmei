"""Correctness tests for ``int8_scaled_mm`` on the NPU.

Mirrors the sglang int8 GEMM test design
(``python/sglang/kernels/aot/tests/test_int8_gemm.py``): inputs are built
directly as int8 tensors with a generic ``to_int8`` helper and the result is
compared against a plain float32 matmul reference, swept over shapes, bias and
output dtypes.

``per_token_quant_int8`` is checked against mathematical invariants (bounds,
scale reduction, dequantization error) instead of a re-derived formula, so the
tests cannot silently pass a bug shared with the implementation.
"""

import pytest
import sgl_kernel_npu  # noqa: F401  makes torch.ops.npu available
import torch
import torch_npu  # noqa: F401
from sgl_kernel_npu.gemm.int8_scaled_mm import int8_scaled_mm, per_token_quant_int8
from utils import require_npu_op

pytestmark = require_npu_op("npu_quant_matmul")


def to_int8(tensor: torch.Tensor) -> torch.Tensor:
    """Generic int8 rounding helper used to build quantized inputs."""
    return torch.round(tensor.clamp(min=-128, max=127)).to(dtype=torch.int8)


def torch_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias):
    """Float32 reference: out = (a @ b) * scale_a * scale_b (+ bias)."""
    o = torch.matmul(a.to(torch.float32), b.to(torch.float32))
    o = o * scale_a.view(-1, 1) * scale_b.view(1, -1)
    if bias is not None:
        o = o + bias.to(torch.float32)
    return o.to(out_dtype)


def _test_accuracy_once(M, N, K, with_bias, out_dtype, device):
    a = to_int8(torch.randn((M, K), device="cpu") * 5).to(device)
    # Weight as a [K, N] column-major view (the convention sglang produces via
    # layer.weight = weight.t() on non-CPU platforms, including the NPU).
    b = to_int8(torch.randn((N, K), device="cpu").t() * 5).to(device)
    scale_a = torch.randn((M,), dtype=torch.float32)
    scale_b = torch.randn((N,), dtype=torch.float32)
    if with_bias:
        bias = torch.randn((N,), dtype=out_dtype) * 10
    else:
        bias = None
    o = int8_scaled_mm(
        a,
        b,
        scale_a.to(device),
        scale_b.to(device),
        out_dtype,
        bias.to(device) if bias is not None else None,
    )
    o1 = torch_scaled_mm(
        a,
        b,
        scale_a.to(device),
        scale_b.to(device),
        out_dtype,
        bias.to(device) if bias is not None else None,
    )
    torch.testing.assert_close(o, o1)


# ---------------------------------------------------------------------------
# int8_scaled_mm accuracy sweep (mirrors sglang test_int8_gemm.py)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("M", [1, 16, 128, 512])
@pytest.mark.parametrize("N", [128, 512, 7168])
@pytest.mark.parametrize("K", [128, 512, 7168])
@pytest.mark.parametrize("with_bias", [True, False])
@pytest.mark.parametrize("out_dtype", [torch.float16, torch.bfloat16])
def test_accuracy(M, N, K, with_bias, out_dtype):
    _test_accuracy_once(M, N, K, with_bias, out_dtype, "npu")


# ---------------------------------------------------------------------------
# per_token_quant_int8 (mathematical invariants, no re-derived formula)
# ---------------------------------------------------------------------------
def test_per_token_quant_int8():
    x = torch.randn(16, 128, dtype=torch.bfloat16, device="npu") * 0.8
    x_q, scale = per_token_quant_int8(x)
    x_fp32 = x.to(torch.float32)

    assert x_q.dtype == torch.int8
    assert x_q.shape == x.shape
    assert scale.shape == x.shape[:-1] + (1,)

    # scale must equal max(|x|) / 127 (computed via an independent reduction)
    absmax = x_fp32.abs().max(dim=-1, keepdim=True).values
    assert torch.allclose(scale, absmax / 127.0, rtol=1e-6, atol=1e-6)

    # dequantization error is bounded by half a quantization step
    dequant = x_q.to(torch.float32) * scale
    step = scale.squeeze(-1).max().item()
    assert (x_fp32 - dequant).abs().max().item() <= 0.5 * step + 1e-6

    # values lie in the symmetric int8 range
    assert x_q.min().item() >= -127
    assert x_q.max().item() <= 127
