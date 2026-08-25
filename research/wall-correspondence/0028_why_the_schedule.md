# 0028 — Why a filter reads on the exponential schedule: stationarity selects it

> **AI-generated, not peer-reviewed.** 0025's residue, closed. Code:
> `0028_why_the_schedule.py`.

The answer is the most filter-native requirement there is: **a
recursive filter needs a stationary record.** One model, one
transfer, one innovation law reused at every step presumes the
statistics don't depend on when you look; a filter whose record is
non-stationary has no time-invariant model to run at all.

- **Stationarity is a geometric condition.** Reading a
  Lorentz-invariant vacuum along a worldline gives a covariance
  W_ij = f(interval_ij), which is Toeplitz *exactly* when the
  sampled points lie on a symmetry orbit. Measured stationarity
  defect: inertial (time-translation orbit) **8e−16**, uniformly
  accelerated (boost orbit) **1e−13**, generic timelike worldline
  **0.37**.
- **The boost orbit is the exponential schedule.** Uniform proper
  time along it gives t = sinh(aτ)/a — precisely 0025's stretched
  schedule. So the schedule is not chosen: it is the *unique
  non-inertial way to read a record and still have a stationary
  model*.
- **The consequence.** A filter that insists on a time-invariant
  model has exactly two options in 1+1D, and one of them is hot:
  inertial (T = 0) or accelerated (T = a/2π). **The Unruh
  temperature is the price of the second option.**

(Stated for 1+1D, where the timelike stationary families are just
these two. In 3+1D there are six — Letaw's classification — and the
filter question becomes which of those a bank can run: a sharper
open question than the one it replaces. A numerical note: the naive
interval suffers catastrophic cancellation once aτ ≫ 1, so the test
range is kept modest to measure geometry rather than floating point.)
