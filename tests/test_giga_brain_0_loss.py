import importlib.util
from pathlib import Path

import pytest
import torch

LOSS_MODULE_PATH = Path(__file__).resolve().parents[1] / 'giga_brain_0' / 'giga_brain_0_loss.py'
LOSS_MODULE_SPEC = importlib.util.spec_from_file_location(
    'giga_brain_0_loss',
    LOSS_MODULE_PATH,
)
LOSS_MODULE = importlib.util.module_from_spec(LOSS_MODULE_SPEC)
LOSS_MODULE_SPEC.loader.exec_module(LOSS_MODULE)
GigaBrain07Loss = LOSS_MODULE.GigaBrain07Loss


@pytest.mark.parametrize(('alpha', 'beta'), ((1.5, 1.0), (2.0, 3.0)))
def test_sample_beta_matches_theoretical_moments(alpha, beta):
    torch.manual_seed(2026)
    samples = GigaBrain07Loss()._sample_beta(
        alpha,
        beta,
        bsize=100_000,
        device=torch.device('cpu'),
    )

    expected_mean = alpha / (alpha + beta)
    expected_variance = alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1))

    assert samples.shape == (100_000,)
    assert samples.dtype == torch.float32
    assert samples.device.type == 'cpu'
    assert samples.mean().item() == pytest.approx(expected_mean, abs=0.005)
    assert samples.var(unbiased=False).item() == pytest.approx(
        expected_variance,
        abs=0.005,
    )


def test_sample_time_preserves_configured_range_and_beta_bias():
    torch.manual_seed(2026)
    times = GigaBrain07Loss().sample_time(100_000, torch.device('cpu'))

    expected_mean = 0.001 + 0.999 * (1.5 / (1.5 + 1.0))
    high_noise_threshold = 0.001 + 0.999 * 0.8
    expected_high_noise_probability = 1.0 - 0.8**1.5
    assert torch.all((times >= 0.001) & (times <= 1.0))
    assert times.mean().item() == pytest.approx(expected_mean, abs=0.005)
    assert (times > high_noise_threshold).float().mean().item() == pytest.approx(
        expected_high_noise_probability,
        abs=0.01,
    )
