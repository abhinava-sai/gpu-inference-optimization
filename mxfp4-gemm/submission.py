import torch
from task import input_t, output_t
import aiter
from aiter import dtypes
from aiter.ops.triton.quant import dynamic_mxfp4_quant
from aiter.utility.fp4_utils import e8m0_shuffle

_quant = dynamic_mxfp4_quant
_shuf = e8m0_shuffle
_gemm = aiter.gemm_a4w4

_fp4 = dtypes.fp4x2
_fp8 = dtypes.fp8_e8m0
_bf16 = dtypes.bf16

def custom_kernel(data: input_t) -> output_t:
    A = data[0].contiguous()
    B_shuffle = data[3]
    B_scale_sh = data[4]

    x_fp4, bs_e8m0 = _quant(A)
    bs_e8m0_sh = _shuf(bs_e8m0)

    return _gemm(
        x_fp4.view(_fp4),
        B_shuffle,
        bs_e8m0_sh.view(_fp8),
        B_scale_sh,
        dtype=_bf16,
        bpreshuffle=True
    )