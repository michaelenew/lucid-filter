# 0005 — predictions, recorded before the run

> **AI-generated, not peer-reviewed.** Written before
> [`0005_the_table_from_theory.py`](0005_the_table_from_theory.py) was first executed.

**The object.** `0001`–`0004` read a table `KL[i, j]`, the divergence rate between bank
members, measured by simulation. It has a definition — the Kullback–Leibler rate between
two processes — and an approximate closed form: in `y = log e²` a member is
`y_t = lam_t + xi_t` with `xi = log z²` (variance `pi²/2`), so with `xi` treated as
Gaussian the rate is Whittle's

    D(A || B) = (1/4pi) ∫ [ f_A/f_B − 1 − ln(f_A/f_B) ] dω,
    f(ω) = s²(1 − phi²) / (1 + phi² − 2 phi cos ω) + pi²/2.

**T1.** For grid neighbours (the only entries the rate rules use) the formula is within a
factor 2 of the measured rate for `s ≤ 0.8`, and too small by more than that at
`s ≥ 1.6`, where the non-Gaussian tail of the data carries evidence the formula cannot see.

**T2.** Rank agreement over all 210 ordered pairs is high (Spearman > 0.9).

**T3.** The walking star run on the formula's rates is within 10% of the star run on the
measured rates in every scenario — the rate direction is flat (`0002`: 7× in rate, ~2× in
result), and the star finds the fraction itself, so a table wrong by a common factor is
absorbed.
