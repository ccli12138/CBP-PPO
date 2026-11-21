"""Uniform financing settlement and exact unilateral C-minus-D payoff maps."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import numpy as np

def neighbor_sum(array):
    """Periodic von Neumann group sum, including the focal site."""
    return (array + np.roll(array, 1, axis=0) + np.roll(array, -1, axis=0)
            + np.roll(array, 1, axis=1) + np.roll(array, -1, axis=1))

NEIGHBORS = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))


def at_offset(array, offset):
    return np.roll(array, (-offset[0], -offset[1]), axis=(0, 1))


def settle(state, r, eta, beta):
    s = np.asarray(state, dtype=np.float64)
    nc = neighbor_sum(s)
    d = 5-nc
    base = neighbor_sum(r*nc/5)-5*s
    m = neighbor_sum((d > 0).astype(float))
    budget = eta*s*np.maximum(base, 0)
    allocation = np.divide(budget, m, out=np.zeros_like(s), where=m > 0)
    fund = (d > 0)*neighbor_sum(allocation)
    paid = budget*(m > 0)
    group_fine = np.divide(beta*fund, d, out=np.zeros_like(s), where=d > 0)
    fine = (1-s)*neighbor_sum(group_fine)
    return dict(base=base, budget=budget, demand=d, active_groups=m,
                fund=fund, paid=paid, fine=fine, reward=base-paid-fine)


def incentive_map(state, r, eta, beta):
    """Each focal site switches alone; all affected budgets and active counts change.

    Shared group centers and contributor offsets come from the inherited five-site
    neighborhood. These are derived memberships, not a chosen spatial parameter.
    """
    s = np.asarray(state, dtype=np.float64)
    nc = neighbor_sum(s)
    d = 5-nc
    base = neighbor_sum(r*nc/5)-5*s
    active = neighbor_sum((d > 0).astype(float))
    base_C = base+(1-s)*(r-5)
    m_C = sum(at_offset(d, g) > (1-s) for g in NEIGHBORS)
    paid_C = eta*np.maximum(base_C, 0)*(m_C > 0)
    contributors = { (g[0]+j[0],g[1]+j[1])
                     for g in NEIGHBORS for j in NEIGHBORS } - {(0,0)}
    fine_D = np.zeros_like(s)
    for offset in sorted(contributors):
        shared = [g for g in NEIGHBORS
                  if (g[0]-offset[0],g[1]-offset[1]) in NEIGHBORS]
        neighbor_s = at_offset(s, offset)
        neighbor_base_D = at_offset(base, offset)-s*r*len(shared)/5
        neighbor_active_D = at_offset(active, offset)+s*sum(at_offset(d,g) == 0 for g in shared)
        budget_D = eta*neighbor_s*np.maximum(neighbor_base_D, 0)
        allocation_D = np.divide(budget_D, neighbor_active_D,
                                 out=np.zeros_like(s), where=neighbor_active_D > 0)
        incidence = np.zeros_like(s)
        for g in shared:
            d_D = at_offset(d,g)+s
            incidence += np.divide(1, d_D, out=np.zeros_like(s), where=d_D > 0)
        fine_D += beta*allocation_D*incidence
    return r-5-paid_C+fine_D


def explicit_settle(state, r, eta, beta):
    """Independent group enumeration for the mathematical verification only."""
    size = len(state)
    sites = list(np.ndindex(state.shape))
    groups = {g:[((g[0]+dy)%size,(g[1]+dx)%size) for dy,dx in NEIGHBORS] for g in sites}
    joined = {i:[g for g in sites if i in groups[g]] for i in sites}
    nc = {g:sum(state[i] for i in members) for g,members in groups.items()}
    d = {g:5-nc[g] for g in sites}
    base = {i:sum(r*nc[g]/5-state[i] for g in joined[i]) for i in sites}
    m = {i:sum(d[g] > 0 for g in joined[i]) for i in sites}
    b = {i:eta*state[i]*max(base[i],0) for i in sites}
    p = {(i,g):b[i]/m[i] if m[i] and d[g] > 0 else 0 for g in sites for i in groups[g]}
    funds = {g:sum(p[i,g] for i in groups[g]) for g in sites}
    reward = np.zeros_like(state)
    for i in sites:
        reward[i] = base[i]-sum(p[i,g] for g in joined[i])
        reward[i] -= (1-state[i])*sum(beta*funds[g]/d[g] if d[g] else 0 for g in joined[i])
    return reward


def check(output=None):
    rng = np.random.default_rng(20261012)
    cases = 0
    error = 0
    for r in (1.5,3,4,5.5):
        for eta,beta in ((0,4),(.3,4),(1,0)):
            for state in (np.zeros((7,7)),np.ones((7,7)),rng.integers(0,2,(7,7)).astype(float)):
                current = settle(state,r,eta,beta)
                np.testing.assert_allclose(current['reward'],explicit_settle(state,r,eta,beta),atol=2e-12,rtol=0)
                np.testing.assert_allclose(current['fund'].sum(),current['paid'].sum(),atol=2e-12,rtol=0)
                delta = incentive_map(state,r,eta,beta)
                for site in ((0,0),(3,3),(6,6)):
                    c,d = state.copy(),state.copy()
                    c[site] = 1
                    d[site] = 0
                    direct = explicit_settle(c,r,eta,beta)[site]-explicit_settle(d,r,eta,beta)[site]
                    error = max(error,abs(float(delta[site])-direct))
                    np.testing.assert_allclose(delta[site],direct,atol=2e-12,rtol=0)
                    cases += 1
    record = dict(status='passed',unilateral_cases=cases,max_absolute_error=error,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  training_runs=0,new_design_parameters=0,
                  meaning='Fixed-state C-minus-D payoff with recomputed uniform budgets, active groups and defector counts.')
    path = output or (Path(__file__).resolve().parents[2]/'output'/datetime.now().strftime('incentive_check_%Y%m%d_%H%M%S.json'))
    path.parent.mkdir(parents=True,exist_ok=True)
    assert not path.exists(), 'Preserve completed check'
    path.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description='Verify fixed-state payoffs by independent group enumeration')
    parser.add_argument('--output',type=Path)
    check(parser.parse_args().output)
