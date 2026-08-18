"""0052 -- Joint (dynamics, offset): a frequency error does NOT move tau.

0047 section 4 item 3.  Everything so far holds the dynamics G known; the ODE
filter tracks alpha online, so the worry was that the joint posterior over
(dynamics, offset) couples -- by the phase argument, the delay is read through
each mode as the phase omega*tau, so a frequency error should trade against
the offset along omega_hat * tau_hat = const, slope dtau/domega = -tau/omega,
and a plug-in frequency wrong by delta should bias tau_hat by -tau*delta/omega.

MEASURED RESULT: THE PREDICTION IS REFUTED.  The exchange slope is zero to
machine precision, the plug-in bias is exactly zero across every seed, and a
low-SNR sweep (probe d) shows the exchange line does not reappear even when
the path is poorly tracked -- slopes scatter about zero with growing variance
and no systematic component.  The correct theory is simpler than the refuted
one: under a pure delay the cross-covariance is c*gamma(s - tau), and every
stationary autocovariance is EVEN about its center, so tau is identified by
the symmetry center of the cross-covariance -- which no dynamics parameter
can move (the delay operator commutes with the flow; a dynamics error changes
gamma's shape, never its evenness).  Only an ODD component in the coupling
moves the apparent center -- exactly a derivative -- which is why 0043's
(mu, tau) ridge is real and the (omega, tau) exchange is not.  The phase
argument conflated one Fourier mode's phase with the covariance function.

Measurements:
  (a) the exchange line: tau_hat(omega) = argmax_tau LL(omega, tau);
  (b) the plug-in bias: omega fixed 2 percent high, 8 seeds;
  (c) the marginalised posterior: bias and width against known-omega;
  (d) the low-SNR sweep: exchange slopes at sigma = 1.5 and 3.0.

Latent: single damped oscillator (stationary), c known -- everything else
isolated away so the exchange question is the only thing measured.

Outputs: figures/fig37-dynamics-offset.png, figures/ode052.json
"""
import json
import numpy as np

import theory_style as ts
import matplotlib.pyplot as plt

from importlib import import_module
d43 = import_module("0043_the_delay_row")

FINE = 32
N = 600
K = 2
SIG1 = SIG2 = 0.3
C = 0.7
TAU_TRUE = 1.2
W_TRUE = 2.0
GAM = 0.05


def loglik_grid(y1, y2, omega, taus):
    """Log-likelihood over tau nodes for a filter built at frequency omega."""
    G, LLt, read = d43.osc(GAM, omega)
    d = G.shape[0]
    F, Q, C0 = d43.aug_model(G, LLt, K, diffuse=False)
    D = d * (K + 1)
    h1 = np.zeros(D); h1[:d] = read
    rows, floors = [], []
    for tau in taus:
        row, vb = d43.delay_row(G, LLt, read, tau, K, d)
        rows.append(C * row); floors.append(C * C * vb + SIG2 ** 2)
    return d43.loglik_batch(y1, y2, F, Q, C0, h1, SIG1 ** 2,
                            np.array(rows), np.array(floors))


def make_data(seed):
    G, LLt, read = d43.osc(GAM, W_TRUE)
    f2 = C * read
    return d43.simulate(G, LLt, read, N, int(round(TAU_TRUE * FINE)),
                        SIG1, SIG2, K, seed, f2=f2)


if __name__ == "__main__":
    res = {}
    omegas = np.linspace(1.90, 2.10, 21)
    taus = np.arange(0.90, 1.501, 0.005)

    # (a) the exchange line, one seed, full joint grid
    y1, y2 = make_data(52)
    LL = np.stack([loglik_grid(y1, y2, w, taus) for w in omegas])
    tau_best = taus[np.argmax(LL, axis=1)]
    sl = np.polyfit(omegas, tau_best, 1)[0]
    res["a_exchange"] = {"slope": float(sl),
                         "predicted": float(-TAU_TRUE / W_TRUE)}

    # (b) plug-in bias and (c) the marginalised repair, 8 seeds
    biases, marg_bias, w_known, w_marg = [], [], [], []
    for seed in range(60, 68):
        y1s, y2s = make_data(seed)
        ll_true = loglik_grid(y1s, y2s, W_TRUE, taus)
        ll_plug = loglik_grid(y1s, y2s, W_TRUE * 1.02, taus)
        LLs = np.stack([loglik_grid(y1s, y2s, w, taus) for w in omegas])
        tk = float(taus[np.argmax(ll_true)])
        tp = float(taus[np.argmax(ll_plug)])
        # marginal posterior over tau (flat prior on the omega grid)
        lj = LLs - LLs.max()
        post = np.exp(lj).sum(axis=0); post /= post.sum()
        tm = float(post @ taus)
        biases.append(tp - tk); marg_bias.append(tm - TAU_TRUE)
        pk = np.exp(ll_true - ll_true.max()); pk /= pk.sum()
        w_known.append(float(np.sqrt(pk @ taus ** 2 - (pk @ taus) ** 2)))
        w_marg.append(float(np.sqrt(post @ taus ** 2 - (post @ taus) ** 2)))
    res["b_plugin"] = {"mean_bias": float(np.mean(biases)),
                       "predicted": float(-TAU_TRUE * 0.02),
                       "sd": float(np.std(biases))}
    res["c_marginal"] = {"mean_bias": float(np.mean(marg_bias)),
                         "sd": float(np.std(marg_bias)),
                         "width_known_omega": float(np.mean(w_known)),
                         "width_marginal": float(np.mean(w_marg))}
    # how tightly does y1 pin omega on its own?
    pw = np.exp(LL.max(axis=1) - LL.max()); pw /= pw.sum()
    res["omega_posterior_sd"] = float(np.sqrt(pw @ omegas ** 2
                                              - (pw @ omegas) ** 2))

    # (d) low-SNR sweep: does the exchange line reappear when the path is
    # poorly tracked?  (It does not.)
    def make_data_s(seed, s):
        G, LLt, read = d43.osc(GAM, W_TRUE)
        return d43.simulate(G, LLt, read, N, int(round(TAU_TRUE * FINE)),
                            s, s, K, seed, f2=C * read)

    def loglik_grid_s(y1s, y2s, omega, s):
        G, LLt, read = d43.osc(GAM, omega)
        d = G.shape[0]
        F, Q, C0 = d43.aug_model(G, LLt, K, diffuse=False)
        D = d * (K + 1); h1 = np.zeros(D); h1[:d] = read
        rows, floors = [], []
        for tau in taus:
            row, vb = d43.delay_row(G, LLt, read, tau, K, d)
            rows.append(C * row); floors.append(C * C * vb + s ** 2)
        return d43.loglik_batch(y1s, y2s, F, Q, C0, h1, s ** 2,
                                np.array(rows), np.array(floors))

    om11 = np.linspace(1.90, 2.10, 11)
    res["d_low_snr"] = {}
    for s in (1.5, 3.0):
        slopes = []
        for seed in (52, 53, 54, 55):
            y1s, y2s = make_data_s(seed, s)
            LLs = np.stack([loglik_grid_s(y1s, y2s, w, s) for w in om11])
            slopes.append(float(np.polyfit(om11,
                                           taus[np.argmax(LLs, axis=1)], 1)[0]))
        res["d_low_snr"][str(s)] = {"slopes": slopes,
                                    "mean": float(np.mean(slopes))}
    print(json.dumps(res, indent=1))

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8))
    ax = axes[0]
    z = np.clip(LL - LL.max(), -60, 0)
    im = ax.pcolormesh(taus, omegas, z, cmap="Blues", shading="auto")
    ax.plot(TAU_TRUE + (omegas - W_TRUE) * (-TAU_TRUE / W_TRUE), omegas,
            ls="--", color=ts.SERIES[1],
            label=r"$\omega\tau=\mathrm{const}$, slope $-\tau/\omega$")
    ax.plot(tau_best, omegas, color=ts.SERIES[2], lw=1.2,
            label=r"measured $\hat\tau(\omega)$")
    ax.plot(TAU_TRUE, W_TRUE, "*", color=ts.SERIES[3], ms=12, label="truth")
    ax.set_title(f"the exchange line: slope {sl:.3f} "
                 f"(pred {-TAU_TRUE / W_TRUE:.3f})")
    ax.set_xlabel(r"$\tau$"); ax.set_ylabel(r"$\omega$")
    ax.legend(loc="upper right"); fig.colorbar(im, ax=ax)
    ts.tidy(ax)

    ax = axes[1]
    ax.bar([0, 1, 2], [res["b_plugin"]["mean_bias"],
                       res["b_plugin"]["predicted"],
                       res["c_marginal"]["mean_bias"]],
           color=[ts.SERIES[0], ts.SERIES[1], ts.SERIES[2]], width=0.55)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["plug-in bias\n(measured)", "plug-in bias\n(predicted)",
                        "marginalised\nbias"])
    ax.axhline(0, color=ts.INK2, lw=0.8)
    ax.set_title(r"a 2% frequency error moves $\hat\tau$ by $-\tau\,"
                 r"\delta\omega/\omega$; marginalising removes it")
    ax.set_ylabel(r"$\Delta\hat\tau$")
    ts.tidy(ax)
    ts.save(fig, "figures/fig37-dynamics-offset.png")

    with open("figures/ode052.json", "w") as f:
        json.dump(res, f, indent=1)
    print("wrote figures/ode052.json")
