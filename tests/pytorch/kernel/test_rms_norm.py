import pytest
import torch

from lmdeploy.utils import is_bf16_supported


def _bf16_mark():
    return pytest.mark.skipif(not is_bf16_supported(), reason='bf16 not supported.')


class TestRMSNorm:

    @pytest.fixture(scope='class')
    def dtype(self, request):
        yield request.param

    @pytest.fixture(scope='class')
    def input(self, dtype):
        yield torch.rand(4, 8, dtype=dtype, device='cuda')

    @pytest.fixture(scope='class')
    def weight(self, dtype):
        yield torch.rand(8, dtype=dtype, device='cuda')

    @pytest.fixture(scope='class')
    def eps(self):
        yield 1e-6

    @pytest.fixture(scope='class')
    def gt(self, input, weight, eps):
        input_dtype = input.dtype
        input = input.to(torch.float32)
        variance = input.pow(2).mean(-1, keepdim=True)
        input = input * torch.rsqrt(variance + eps)
        return weight * input.to(input_dtype)

    @pytest.mark.parametrize('dtype', [pytest.param(torch.bfloat16, marks=_bf16_mark()), torch.float16, torch.float32],
                             indirect=True)
    def test_rms_norm(self, input, weight, eps, gt):
        from lmdeploy.pytorch.kernels import rms_norm

        out = rms_norm(input, weight, eps)
        torch.testing.assert_close(out, gt)
