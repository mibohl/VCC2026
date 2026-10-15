import h5py,numpy as np,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
rng=np.random.default_rng(0)
for c in sys.argv[1:]:
    with h5py.File(ROOT/f'data/{c}.h5') as h:
        T=h['pert/half_delta_A'].shape[0]; idx=np.sort(rng.choice(T,min(300,T),replace=False))
        A=h['pert/half_delta_A'][idx].astype(float); B=h['pert/half_delta_B'][idx].astype(float)
        gated=h['control/mean_cpm'][:]>5; meas=h['measured'][:].astype(bool); m=gated&meas
        cols=np.where(m)[0] if A.shape[1]==m.size else np.arange(A.shape[1])
        A=A[:,cols]; B=B[:,cols]
        rows=np.isfinite(A).mean(1)>0.9; A=A[rows]; B=B[rows]
        ok=np.isfinite(A).all(0)&np.isfinite(B).all(0); A=A[:,ok]; B=B[:,ok]
        print(c, 'rows', rows.sum(), 'genes', ok.sum())
        keep=(A.std(1)>0)&(B.std(1)>0); A=A[keep]; B=B[keep]
        z=lambda X:(X-X.mean(1,keepdims=True))/X.std(1,keepdims=True)
        a,b=z(A),z(B); C=a@b.T/a.shape[1]
        same=np.diag(C); off=C[~np.eye(len(C),dtype=bool)]
        mean=A.mean(0); ac=A-mean; bc=B-B.mean(0)
        sc=np.array([np.corrcoef(ac[i],bc[i])[0,1] for i in range(len(ac))])
        print(f'{c}: same-target r {np.median(same):.3f}, other-target r {np.median(off):.3f}, same-target r after removing mean response {np.median(sc):.3f}, |mean resp|/|effect| {np.linalg.norm(mean)/np.median(np.linalg.norm(A,axis=1)):.2f}')
