"""Original reference BCa function; 20000 paired resampling indices per r."""
import math
from statistics import NormalDist
import numpy as np
NORMAL=NormalDist()

def bca_mean(values,indices):
    x=np.asarray(values,dtype=float)
    mean=float(x.mean())
    result=dict(mean=mean,std=float(x.std(ddof=1)),bca_low=math.nan,bca_high=math.nan,
                z0=math.nan,acceleration=math.nan,status='degenerate')
    if np.all(x==x[0]):return result
    boot=x[indices].mean(axis=1)
    rank=(np.count_nonzero(boot<mean)+np.count_nonzero(boot<=mean))/(2*len(boot))
    if not 0<rank<1:raise ValueError('BCa bias correction is infinite')
    z0=NORMAL.inv_cdf(rank)
    jack=np.array([np.delete(x,i).mean() for i in range(len(x))])
    delta=jack.mean()-jack
    acceleration=float(np.sum(delta**3)/(6*np.sum(delta**2)**1.5))
    # Independent mean-specific acceleration check, not used to replace jackknife.
    centered=x-x.mean()
    analytic=float(np.sum(centered**3)/(6*np.sum(centered**2)**1.5))
    assert math.isclose(acceleration,analytic,abs_tol=2e-8)
    adjusted=[]
    for q in (.025,.975):
        z=NORMAL.inv_cdf(q)
        adjusted.append(NORMAL.cdf(z0+(z0+z)/(1-acceleration*(z0+z))))
    assert 0<=adjusted[0]<adjusted[1]<=1
    low,high=np.quantile(boot,adjusted,method='linear')
    assert x.min()<=low<=high<=x.max()
    result.update(bca_low=float(low),bca_high=float(high),z0=z0,acceleration=acceleration,
                  adjusted_low=adjusted[0],adjusted_high=adjusted[1],status='BCa')
    return result
