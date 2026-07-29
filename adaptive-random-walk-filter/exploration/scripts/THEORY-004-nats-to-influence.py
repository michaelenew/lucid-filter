"""THEORY-004: turning nats into trust, and trust into influence.

Three conversions, each derived rather than posited.

1.  nats -> trust.     Evidence adds in log-odds, so trust approaches 1 as
    1 - trust = exp(-Lambda).  The "exponential confirmation loop" is not an
    analogy: it is what log-odds accumulation *is*.  The noisy-OR form
    1 - prod(1 - p_i) is a different (and miscalibrated) rule; both are shown.

2.  nats -> influence.  Verified exactly in THEORY-002: incremental information
    decays as (1-K)^{2k} while optimal influence decays as (1-K)^k.  Influence
    is an amplitude, information is an energy:  influence  ~  sqrt(nats),
    normalised to sum to one.

3.  information -> TRUSTWORTHY information.  Nats are always relative to an
    alternative.  Define the trustworthy evidence for a mode as the WORST CASE
    over the alternative set -- the row-minimum of the pairwise LLR matrix,
    including H0.  This is the e-value / GRO reading, and it converts
    THEORY-003's matrix directly into an actionable threshold.
"""
import json
import numpy as np
from theory_style import plt, tidy, save, SERIES

OUT = "figures"
S2, Q = 1.0, 0.05
MODES = ["PA", "MA", "PR", "MR"]
LABEL = {"PA": "process anomaly (jump)", "MA": "measurement anomaly (outlier)",
         "PR": "process regime ($Q$ change)", "MR": "measurement regime ($\\sigma^2$ change)"}
ALTS = ["H0"] + MODES


def gain(q):
    return (-q + np.sqrt(q * q + 4 * q)) / 2.0


K = gain(Q / S2)
VINC = Q + 2 * S2                       # increment variance under H0


def sigma0(m):
    return (np.diag(np.full(m, VINC)) + np.diag(np.full(m - 1, -S2), 1)
            + np.diag(np.full(m - 1, -S2), -1))


def hypothesis(mode, m, delta, rho):
    mu, S = np.zeros(m), sigma0(m)
    if mode == "PA":
        mu = np.zeros(m); mu[0] = delta
    elif mode == "MA":
        mu = np.zeros(m); mu[0] = delta
        if m > 1:
            mu[1] = -delta
    elif mode == "PR":
        S = S + (rho - 1.0) * Q * np.eye(m)
    elif mode == "MR":
        T = (np.diag(np.full(m, 2.0)) + np.diag(np.full(m - 1, -1.0), 1)
             + np.diag(np.full(m - 1, -1.0), -1))
        T[0, 0] = 1.0
        S = S + (rho - 1.0) * S2 * T
    return mu, S


def kl(m1, S1, m2, S2_):
    n = len(m1)
    S2i = np.linalg.inv(S2_)
    dm = m2 - m1
    _, ld1 = np.linalg.slogdet(S1)
    _, ld2 = np.linalg.slogdet(S2_)
    return 0.5 * (np.trace(S2i @ S1) - n + dm @ S2i @ dm + ld2 - ld1)


def robust_nats(mode, m, delta, rho):
    """Worst-case expected evidence for `mode` against every alternative incl. H0."""
    ma, Sa = hypothesis(mode, m, delta, rho)
    return min(kl(ma, Sa, *hypothesis(b, m, delta, rho))
               for b in ALTS if b != mode)


# ------------------------------------------- 1. trustworthy nats vs confirmations
MS = np.arange(1, 61)
SIZES = [2.0, 3.0, 4.0, 6.0]            # location: multiples of increment SD
RHOS = [1.5, 2.0, 3.0, 6.0]             # scale: fold change
traj = {}
for si, (s, r) in enumerate(zip(SIZES, RHOS)):
    d = s * np.sqrt(VINC)
    for mode in MODES:
        traj[(mode, si)] = np.array([robust_nats(mode, m, d, r) for m in MS])

THRESH = np.log(99.0)                   # 99:1 posterior odds = 4.595 nats
mstar = {}
print(f"confirmations needed for {THRESH:.2f} nats (99:1) of WORST-CASE evidence")
print(f"{'mode':>5} " + " ".join(f"{s:>6.1f}sd/{r:g}x" for s, r in zip(SIZES, RHOS)))
for mode in MODES:
    row = []
    for si in range(len(SIZES)):
        ok = np.where(traj[(mode, si)] >= THRESH)[0]
        row.append(int(MS[ok[0]]) if len(ok) else None)
    mstar[mode] = row
    print(f"{mode:>5} " + " ".join(f"{str(v):>11}" for v in row))

# ------------------------------------ 2. nats -> trust, and the noisy-OR error
lam = np.linspace(0, 10, 400)
trust_logodds = 1.0 / (1.0 + np.exp(-lam))
# a noisy-OR built from the same per-step evidence, 1 nat per confirmation
n_conf = np.arange(0, 11)
per_step_p = 1.0 / (1.0 + np.exp(-1.0))
noisy_or = 1.0 - (1.0 - per_step_p) ** n_conf
logodds_seq = 1.0 / (1.0 + np.exp(-n_conf * 1.0))

# -------------------------------------- 3. influence allocation after a jump
def evidence_shift(m, delta, tau=None):
    """Expected nats of evidence that a level shift happened at t=0, vs H0.

    tau=None  -> ORACLE: the jump size is known, evidence = KL = delta^2 c / 2.
    tau given -> non-oracle: delta is marginalised over N(0, tau^2), so an Occam
    factor log(1+tau^2 c)/2 is deducted.  Sherman-Morrison gives, in closed form,
        E[log BF] = 1/2 [ t c/(1+t c) + delta^2 t c^2/(1+t c) - log(1+t c) ],  t=tau^2
    """
    u = np.zeros(m); u[0] = 1.0
    c = u @ np.linalg.inv(sigma0(m)) @ u
    if tau is None:
        return 0.5 * delta ** 2 * c
    t = tau ** 2
    return 0.5 * (t * c / (1 + t * c) + delta ** 2 * t * c ** 2 / (1 + t * c)
                  - np.log(1 + t * c))


TAU = 4.0 * np.sqrt(VINC)          # broad prior on jump size: SD of 4 increment-SD


def allocation(m, delta, tau=None):
    """Share of the posterior mean's sensitivity sitting on post-jump points.

    Bayes model averaging over {shift at 0, no shift}: with posterior weight pi
    on the shift, the mean is pi*(post-shift-only estimate) + (1-pi)*(plain KF),
    and the plain KF has itself already moved 1-(1-K)^m of the way.
    """
    pi = 1.0 / (1.0 + np.exp(-evidence_shift(m, delta, tau)))
    kf = 1.0 - (1.0 - K) ** m
    return pi + (1 - pi) * kf, pi, kf


alloc = {}
for s_ in SIZES:
    d = s_ * np.sqrt(VINC)
    a, p, k_ = zip(*[allocation(m, d) for m in MS])
    am = np.array([allocation(m, d, TAU)[0] for m in MS])
    alloc[s_] = dict(a=np.array(a), pi=np.array(p), kf=np.array(k_), marg=am)

print("\noracle vs non-oracle evidence for a level shift (nats, m=2)")
for s_ in SIZES:
    d = s_ * np.sqrt(VINC)
    print(f"  {s_:g} SD: oracle {evidence_shift(2, d):6.2f}   "
          f"delta marginalised {evidence_shift(2, d, TAU):6.2f}   "
          f"Occam cost {evidence_shift(2, d) - evidence_shift(2, d, TAU):5.2f}")

json.dump(dict(mstar=mstar, sizes=SIZES, rhos=RHOS, K=K, thresh=THRESH,
               alloc={str(s_): alloc[s_]["a"].tolist() for s_ in SIZES},
               alloc_marginalised={str(s_): alloc[s_]["marg"].tolist() for s_ in SIZES}),
          open("figures/theory004.json", "w"), indent=1)
print(f"\nK={K:.4f}; plain KF reaches 90% allocation at m="
      f"{int(np.ceil(np.log(0.1)/np.log(1-K)))}; "
      f"Bayes at 4sd reaches 90% at m={int(MS[np.argmax(alloc[4.0]['a']>0.9)])}")


# --------------------------------------------------------------------- figures
# Fig 11: trustworthy evidence accumulating, per mode
fig, axes = plt.subplots(1, 4, figsize=(12.4, 3.6), sharey=True)
for ax, mode in zip(axes, MODES):
    tidy(ax)
    for si, (s, r) in enumerate(zip(SIZES, RHOS)):
        y = traj[(mode, si)]
        lab = f"{s:g}$\\,$SD" if mode in ("PA", "MA") else f"{r:g}$\\times$"
        ax.plot(MS, y, color=SERIES[si], label=lab)
        # realised evidence fluctuates: SD ~ sqrt(2*E) for a Gaussian LLR
        ax.fill_between(MS, y - np.sqrt(2 * np.maximum(y, 0)),
                        y + np.sqrt(2 * np.maximum(y, 0)),
                        color=SERIES[si], alpha=0.10, lw=0)
    ax.axhline(THRESH, color="#d03b3b", lw=1.2, ls="--")
    ax.set_ylim(0, 25)
    ax.set_xlim(1, 40)
    ax.set_title(f"{mode} -- {LABEL[mode]}", fontsize=9.5)
    ax.set_xlabel("m  (confirmations)")
axes[0].set_ylabel("worst-case evidence (nats)")
axes[0].text(2, THRESH + 0.7, "99:1", fontsize=8.5, color="#d03b3b")
axes[0].legend(loc="lower right", ncol=2)
fig.suptitle("Trustworthy information = evidence that survives the worst alternative.\n"
             "Location events clear 99:1 in 2-3 points; scale events need tens.",
             fontsize=11, color="#0b0b0b", y=1.10)
save(fig, f"{OUT}/fig11-trustworthy-nats.png")

# Fig 12: nats -> trust, and the noisy-OR discrepancy
fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.9))
ax = tidy(axes[0])
ax.semilogy(lam, 1 - trust_logodds, color=SERIES[0], label="$1-\\mathrm{trust}=\\sigma(-\\Lambda)$")
ax.semilogy(lam, np.exp(-lam), color=SERIES[1], ls="--", label="$e^{-\\Lambda}$ (large-$\\Lambda$ limit)")
for t, n in [(np.log(9), "90%"), (np.log(99), "99%"), (np.log(999), "99.9%")]:
    ax.axvline(t, color="#c9c8c3", lw=1.0)
    ax.text(t + 0.08, 3e-1, n, fontsize=8, color="#52514e")
ax.set_xlabel("$\\Lambda$  (accumulated evidence, nats)")
ax.set_ylabel("residual doubt  $1-\\mathrm{trust}$")
ax.set_title("Nats are the natural units of trust:\ndoubt decays by $e$ per nat")
ax.legend(loc="lower left")

ax = tidy(axes[1])
ax.plot(n_conf, logodds_seq, color=SERIES[0], marker="o", label="log-odds sum (correct)")
ax.plot(n_conf, noisy_or, color=SERIES[1], marker="o",
        label="$1-\\prod(1-p_i)$ (noisy-OR)")
ax.set_ylim(0.5, 1.02)
ax.set_xlabel("number of confirmations, 1 nat each")
ax.set_ylabel("posterior belief")
ax.set_title("The two confirmation rules are not the same\n(noisy-OR is over-confident early, then saturates)")
ax.legend(loc="lower right")
save(fig, f"{OUT}/fig12-nats-to-trust.png")

# Fig 13: influence allocation to post-jump data
fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9))
ax = tidy(axes[0])
for i, s_ in enumerate([2.0, 4.0]):
    ax.plot(MS, alloc[s_]["a"], color=SERIES[i], label=f"oracle ($\\delta$ known), {s_:g} SD")
    ax.plot(MS, alloc[s_]["marg"], color=SERIES[i], ls=":", lw=1.8,
            label=f"$\\delta$ marginalised, {s_:g} SD")
ax.plot(MS, alloc[SIZES[0]]["kf"], color="#8a8880", ls="--", lw=1.5,
        label=f"plain Kalman, $K$={K:.2f}")
ax.axhline(0.99, color="#d03b3b", lw=1.1, ls=":")
ax.set_xlim(1, 40)
ax.set_ylim(0, 1.03)
ax.set_xlabel("m  (points since the jump)")
ax.set_ylabel("share of the mean carried by post-jump data")
ax.set_title("What the oracle wants (99% at once), what evidence permits,\nand what it costs not to know the jump size")
ax.legend(loc="lower right")

ax = tidy(axes[1])
kk = np.arange(0, 60)
for i, q in enumerate([0.005, 0.05, 0.5]):
    Ki = gain(q)
    nats = (1 - Ki) ** (2 * kk)
    infl = Ki * (1 - Ki) ** kk
    ax.plot(np.sqrt(nats), infl / Ki, color=SERIES[i], marker="o", ms=3,
            ls="none", label=f"q={q}")
ax.plot([0, 1], [0, 1], color="#8a8880", ls="--", lw=1.2, label="identity")
ax.set_xlabel("$\\sqrt{\\mathrm{incremental\\ nats}}$ (normalised)")
ax.set_ylabel("optimal influence (normalised)")
ax.set_title("The conversion law: influence $=\\sqrt{\\mathrm{nats}}$\n"
             "information is energy, influence is amplitude")
ax.legend(loc="upper left")
save(fig, f"{OUT}/fig13-influence-allocation.png")
