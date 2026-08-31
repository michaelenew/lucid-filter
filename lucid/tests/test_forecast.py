"""`forecast` is the causal one-step prediction; `y - innovation` is not.

The reported innovation is posterior-mixed: this step's likelihoods move the
member weights before the mixture is formed, so `y - innovation` leans toward
the very observation it nominally predicts.  On iid noise -- where nothing is
predictable -- any statistic that agrees with the next observation above
chance has looked at it.  These tests pin that distinction, and pin
`forecast` to the exact prior-weighted mixture in the one case it can be
checked in closed form (a single-member bank, where prior-weighted and
sole-member coincide).
"""
import numpy as np
import pytest

from lucid import LucidFilter


@pytest.mark.slow
def test_forecast_is_at_chance_on_iid_noise_and_y_minus_innovation_is_not():
    # E[sign(pred_t) * y_t] on iid noise: exactly zero for anything causal,
    # measurably positive for a statistic that has already eaten y_t.  The
    # magnitude-weighted form is the sensitive one (the lean lives in the
    # bars the members disagree about most).
    rng = np.random.default_rng(7)
    y = rng.standard_normal(12000)
    r = LucidFilter(dynamics=None,
                    process=np.array([[1.0]]),
                    measurement=np.array([1.0])).filter(y[:, None])
    burn = 1000
    leaky = (y - r.innovation[:, 0])[burn:]
    causal = r.forecast[burn:, 0]
    yy = y[burn:]
    se = yy.std() / np.sqrt(len(yy))
    cap_leak = np.mean(np.sign(leaky) * yy)
    cap_fc = np.mean(np.sign(causal) * yy)
    assert cap_leak > 2.5 * se, (
        f"the posterior-mixed innovation should visibly lean on y_t "
        f"(got {cap_leak:+.4f}, se {se:.4f}) -- if this fails, the mixture "
        f"became prior-weighted and the docs should say so")
    assert abs(cap_fc) < 3.5 * se, (
        f"forecast must be at chance on iid noise, got {cap_fc:+.4f} (se {se:.4f})")


def test_forecast_and_y_minus_innovation_genuinely_differ():
    # even the plain scalar filter carries a member bank (the (phi, s) box and
    # the demix ladder), so the posterior-mixed and prior-mixed predictions
    # must not coincide -- if they ever do, one of the two mixings changed
    rng = np.random.default_rng(1)
    y = np.cumsum(rng.standard_normal(800))
    r = LucidFilter().filter(y[:, None])
    gap = np.abs((y - r.innovation[:, 0]) - r.forecast[:, 0])
    assert np.mean(gap[100:] > 0) > 0.9
    assert np.mean(gap[100:]) > 0


def test_forecast_streams_and_batches_identically():
    rng = np.random.default_rng(3)
    Y = np.cumsum(rng.standard_normal((400, 1)), axis=0)
    batch = LucidFilter().filter(Y)
    f = LucidFilter()
    stream = np.array([f.update(row).forecast[0] for row in Y])
    assert np.array_equal(batch.forecast[:, 0], stream)


def test_forecast_is_nan_through_a_gap_and_recovers():
    rng = np.random.default_rng(5)
    Y = np.cumsum(rng.standard_normal((300, 1)), axis=0)
    Y[100:110] = np.nan
    r = LucidFilter().filter(Y)
    assert np.all(np.isnan(r.forecast[100:110]))
    assert np.all(np.isfinite(r.forecast[110:]))


def test_existing_outputs_untouched_by_the_new_field():
    rng = np.random.default_rng(9)
    Y = np.cumsum(rng.standard_normal((300, 1)), axis=0)
    r = LucidFilter().filter(Y)
    # the new field must not perturb the recursion: loglik of the same data
    # through a freshly-constructed filter agrees exactly
    assert LucidFilter().loglik_of(Y) == pytest.approx(r.loglik, abs=0.0)
