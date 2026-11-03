# E1.0: Monte Carlo check of beta* = (mu - sqrt((mu^2 - 2 s^2 L)_+)) / s^2, L = log n + kappa
import numpy as np, json
from scipy.special import logsumexp
rng = np.random.default_rng(0)
s = 1.0
def beta_star(mu, s, L):
    return (mu - np.sqrt(np.maximum(mu**2 - 2*s*s*L, 0.0))) / (s*s)
def draw(kind, size):
    if kind == "gauss":  return rng.standard_normal(size)
    if kind == "t3":     return rng.standard_t(3, size) / np.sqrt(3.0)   # unit variance
    if kind == "laplace":return rng.laplace(0, 1/np.sqrt(2), size)       # unit variance
def needle_mass(n, mu, beta, kind, reps):
    ms = np.empty(reps)
    for r in range(reps):
        z = draw(kind, n-1)
        d = logsumexp(beta*z)
        ms[r] = 1/(1+np.exp(d - beta*mu))
    return ms.mean(), np.quantile(ms,.1), np.quantile(ms,.9)
kappa = np.log(9); target = 0.9
print(f"target mass {target}, kappa={kappa:.3f}")
print("kind    log2n  ratio mu/thr   beta*   beta_c  mean_mass  q10    q90")
rows=[]
for kind in ["gauss","laplace","t3"]:
    for lg in [6,10,14,18]:
        n=2**lg; L=np.log(n)+kappa; thr=s*np.sqrt(2*L)
        for ratio in [1.05,1.25,1.5,2.0,4.0]:
            mu=ratio*thr; b=beta_star(mu,s,L); reps=300 if lg<=14 else 40
            m,q1,q9=needle_mass(n,mu,b,kind,reps)
            rows.append(dict(kind=kind,log2n=lg,ratio=ratio,beta=float(b),mass=float(m),q10=float(q1),q90=float(q9)))
            print(f"{kind:7s} {lg:4d} {ratio:8.2f}   {b:8.3f} {np.sqrt(2*np.log(n))/s:7.3f}  {m:8.3f} {q1:6.3f} {q9:6.3f}")
# limit checks
n=2**14; L=np.log(n)+kappa
print("high-signal limit beta* vs L/mu:", [(mu, round(float(beta_star(mu,1,L)),4), round(float(L/mu),4)) for mu in [50,200,1000]])
thr=np.sqrt(2*L); print("continuity at threshold:", beta_star(thr*0.999,1,L), beta_star(thr*1.001,1,L), "thr", thr)
print("scale equivariance:", beta_star(30.,1.,L)*1, beta_star(60.,2.,L)*2)
print("E1_0_JSON="+json.dumps(rows))
