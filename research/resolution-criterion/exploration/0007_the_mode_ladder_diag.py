import sys, time, importlib.util, numpy as np
S="/tmp/claude-0/-home-user-lucid-filter/56bbf4e5-3fe2-5e9c-93f7-d50a0493cfbc/scratchpad/"
src=open(S+"probe_modeladder.py").read(); head=src[:src.index('A = load("A"); C = load("C")')]
sys.argv=["x","1","arm"]; ns={}; exec(compile(head,"ml","exec"),ns); C=ns["load"]("C")
sys.path.insert(0,"research/multivariate-statfilter/scripts"); import arm5dof as AR
spec=importlib.util.spec_from_file_location("p54","research/multivariate-statfilter/exploration/0054_physical_sensors.py"); m54=importlib.util.module_from_spec(spec); spec.loader.exec_module(m54)
jstd,pot,acc=m54.schedule(); U,S_,Y=AR.simulate(0,jstd,pot,acc); np.seterr(all="ignore")
f=C.LucidFilter(dynamics=AR.F, control=AR.B, H=AR.measure, process=AR.Q0, measurement=AR.R0)
nb=len(f.mode_star); nphi, ns_=3,5
def cell(c): bi=c%nb; r=c//nb; return (r//ns_, r%ns_, f.mode_star[bi])
print("members", len(f._members), "star", nb, flush=True)
for t in range(400):
    st=f.update(Y[t],U[t])
    if t%40==39 or t<3:
        w=np.exp(f._logw-f._logw.max()); w/=w.sum(); top=np.argsort(w)[::-1][:3]
        mags=np.array([np.abs(e._m).max() for e in f._members]); Pt=np.array([np.trace(e._P) for e in f._members])
        print(f"t={t:4d} |truth|max {np.abs(S_[t]).max():7.2f} |mix|max {np.abs(st.mean).max():9.2f} trP(mix) {np.trace(st.var):9.2e} | "
              f"top w {w[top[0]]:.3f} {cell(top[0])} ; {w[top[1]]:.3f} {cell(top[1])} | n(w>1e-3) {int((w>1e-3).sum())} | "
              f"members: max|m| {mags.max():9.2e} (cell {cell(int(mags.argmax()))}) max trP {Pt.max():9.2e} ; centre-cells max|m| {mags[[c for c in range(len(mags)) if cell(c)[2] is None]].max():8.2f}", flush=True)
