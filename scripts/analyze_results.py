"""Recompute from the original2513 trajectories; split learning files do not alter the archived data format."""
import argparse,csv,hashlib,json,math
from datetime import datetime
from pathlib import Path
import numpy as np
from statistics_helpers import bca_mean
CODE=Path(__file__).resolve().parents[1];ROOT=CODE.parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def csvwrite(p,rows):
    with p.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path);a=p.parse_args()
    plan=json.loads((CODE/'config/experiments.json').read_text(encoding='utf-8'))
    missing=[r['archived_run_path'] for r in plan['runs'] if not (ROOT/r['archived_run_path']/'trajectory.npz').is_file() or getattr((ROOT/r['archived_run_path']/'trajectory.npz').stat(),'st_flags',0)&0x40000000]
    if missing:raise SystemExit(f'{len(missing)} archived trajectories are unavailable locally. Restore the complete data before recomputing; see provenance/missing_files.csv if present.')
    out=(a.output_dir or ROOT/'output'/datetime.now().strftime('recomputed_%Y%m%d_%H%M%S')).resolve()
    assert ROOT/'output' in out.parents and not out.exists(),'Use a fresh directory inside output'
    rows=[]
    for item in plan['runs']:
        run=ROOT/item['archived_run_path'];file=run/'trajectory.npz';assert sha(file)==item['data_sha256']
        meta=json.loads((run/'provenance.json').read_text(encoding='utf-8'));assert meta['status']=='complete'
        assert meta['source_sha256']=={n:plan['submitted_source_sha256'][n] for n in ('PPO.py','main_PPO.py')}
        cfg={**plan['shared_settings'],**item['config']};assert all(meta['config'][k]==v for k,v in cfg.items())
        with np.load(file,allow_pickle=False) as z:
            c=z['cooperation'];assert c.shape==(1001,) and np.isfinite(c).all()
            assert np.allclose(c+z['defection'],1,atol=1e-7,rtol=0)
            for name in z.files:assert hashlib.sha256(z[name].tobytes()).hexdigest()==meta['array_sha256'][name]
            window=c[-101:];exposure=float(z['defector_exposure'].sum())
            row={'run_id':item['id'],'actual_run_id':item['actual_run_id'],'run_path':item['archived_run_path'],**item['config'],'model':plan['model_display'][cfg['variant']],'endpoint':float(c[-1]),'window_mean':float(window.mean()),'window_min':float(window.min()),'stable_full':bool((window>=.999).all()),'payment_per_player_round':float(z['payment_total'].sum()/(200**2*1000)),'fine_activity':float((z['fine_total']>0).mean()),'fine_per_defector_incidence':float(z['fine_total'].sum()/exposure) if exposure else None,'data_sha256':item['data_sha256']}
        rows.append(row)
    out.mkdir(parents=True);csvwrite(out/'run_metrics.csv',rows);byid={r['run_id']:r for r in rows}
    for category,entry in plan['categories'].items():csvwrite(out/(category+'_metrics.csv'),[byid[i] for i in entry['run_ids']])
    stat=[byid[i] for i in plan['categories']['statistics']['run_ids']];summary=[];paired=[]
    for r in plan['r_grid']:
        groups={};indices=np.random.default_rng(20260909+round(10*r)).integers(0,20,(20000,20))
        for variant in ('original','UA'):
            group=sorted([x for x in stat if x['r']==r and x['variant']==variant],key=lambda x:x['seed']);assert [x['seed'] for x in group]==plan['statistical_seeds'];groups[variant]=group
            result=bca_mean([x['window_mean'] for x in group],indices)
            summary.append({'r':r,'model':plan['model_display'][variant],'n':20,'stable_count':sum(x['stable_full'] for x in group),**result})
        for ppo,cbp in zip(groups['original'],groups['UA']):paired.append({'r':r,'seed':ppo['seed'],'difference':cbp['window_mean']-ppo['window_mean']})
    csvwrite(out/'statistics_summary.csv',summary);csvwrite(out/'paired_metrics.csv',paired)
    thresholds={}
    for model in ('PPO','CBP-PPO'):
        selected=[x for x in summary if x['model']==model]
        thresholds[model]=next((x['r'] for k,x in enumerate(selected) if all(y['stable_count']==20 for y in selected[k:])),None)
    report={'status':'raw_data_recomputed','unique_runs':len(rows),'thresholds':thresholds,'bootstrap_resamples':20000,'source_config_sha256':sha(CODE/'config/experiments.json'),'archived_results_overwritten':False,'training_started':False}
    (out/'analysis_report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
