# E3.0: per-coordinate SNR of forward-gradient update rules, g fixed, v ~ N(0, I_P)
# raw: (v.g) v;  tangent-normalized: sign(v.g) v;  sign: sign(v.g) sign(v)
import numpy as np, json
rng = np.random.default_rng(0)
def run(P, kind, K=200000):
    if kind=="dense":  g = rng.standard_normal(P)
    if kind=="sparse":
        g = np.zeros(P); idx = rng.choice(P, max(1,P//20), replace=False); g[idx]=rng.standard_normal(len(idx))
    if kind=="heavy":  g = rng.standard_t(2, P)
    g /= np.linalg.norm(g)
    M1={k:np.zeros(P) for k in "rns"}; M2={k:np.zeros(P) for k in "rns"}; cos={k:0. for k in "rns"}
    B=2000
    for _ in range(K//B):
        v=rng.standard_normal((B,P)); a=v@g; sa=np.sign(a)
        U={"r":a[:,None]*v,"n":sa[:,None]*v,"s":sa[:,None]*np.sign(v)}
        for k,u in U.items():
            M1[k]+=u.sum(0); M2[k]+=(u*u).sum(0)
            cos[k]+=((u@g)/np.linalg.norm(u,axis=1)).sum()
    N=(K//B)*B; out={}
    for k in "rns":
        m=M1[k]/N; var=M2[k]/N-m*m
        # SNR-weighted summary: slope of per-coordinate mean on g, in units of noise std
        slope=(m@g)/(g@g); out[k]=dict(slope=float(slope), noise_std=float(np.sqrt(var.mean())), snr_per_g=float(slope/np.sqrt(var.mean())), cos=float(cos[k]/N*np.sqrt(P)))
    return out
pred=dict(r=1.0,n=float(np.sqrt(2/np.pi)),s=float(2/np.pi))
print("predicted snr_per_g  raw 1.000  tangent-norm %.3f  sign %.3f (small g_i limit)"%(pred["n"],pred["s"]))
print("predicted cos*sqrt(P) raw %.3f  tangent-norm %.3f  sign %.3f"%(np.sqrt(2/np.pi),np.sqrt(2/np.pi),2/np.pi))
res={}
for P in [100,2000]:
    for kind in ["dense","sparse","heavy"]:
        o=run(P,kind); res[f"{P}_{kind}"]=o
        print(P,kind," ".join(f"{k}: snr={o[k]['snr_per_g']:.3f} cos*rtP={o[k]['cos']:.3f}" for k in "rns"))
print("E3_0_JSON="+json.dumps(res))
