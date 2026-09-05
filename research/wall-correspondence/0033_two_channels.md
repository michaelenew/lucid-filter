# 0033 — Composition has two channels, and only one is the source ledger

> **AI-generated, not peer-reviewed.** A correction to 0031. Code:
> `0033_two_channels.py`.

> **⚖️ ATTRIBUTION —** _A clean self-correction separating two distinct effects: classical non-abelian composition carries order information but no interference (0009), while amplitude addition carries interference but is order-blind. Restates that interference specifically requires amplitudes; the two "records" just instantiate the textbook classical/quantum split._ Prior art: quantum interference vs classical mixing; non-commutativity (0009). Status: RECOMBINATION (measured separation) + SPECULATIVE (the "source ledger" physics framing).

0031 concluded that the source ledger is "a statement about
composition" and pointed at the order channel (0009). **That was
sloppy.** 0009 composes *classical* group elements — no amplitudes
anywhere. So composition has two independent layers, and two records
separate them:

| record | order channel | interference |
|---|---|---|
| **A** classical nonabelian | **0.0545 nats/triple** | empty (0005) |
| **B** quantum abelian | **0 (exact)** | **0.3020 nats/trial** |

**Record A**: three SU(2) elements composed in a random order, the
composite read through noise. Order information is positive; the
generator is classical by construction, so by 0005 no amplitude
model can beat the correct classical one.

**Record B**: two paths whose amplitudes add, with a known relative
phase varied across trials. The coherent model predicts the fringe
and the incoherent one cannot (0.391 vs 0.693 nats/trial). Swapping
the paths leaves every outcome probability **bitwise identical** —
addition commutes, so there is no order to read.

Each record fires exactly one detector: **the two channels are
independent.**

> **The source ledger's observable content is interference, not
> composition in general.** The order channel is a *record-ledger*
> phenomenon that happens to require nonabelian structure — which is
> why 0009 could measure it with no amplitudes in sight.

0031's phrasing is corrected in place. The chain now reads: the
static factorisation phase is gauge (0031); the observable
source-ledger content is *interference specifically*; and
composition order is a separate, classical channel.
