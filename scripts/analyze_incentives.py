"""Recompute fixed-state incentive evidence from the four archived spatial runs."""
import argparse,csv,hashlib,json
from datetime import datetime
from pathlib import Path
import numpy as np
from uniform_incentives import incentive_map,settle

CODE=Path(__file__).resolve().parents[1]
ROOT=CODE.parent

def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args()
    out=(args.output_dir or ROOT/'output'/datetime.now().strftime('incentives_%Y%m%d_%H%M%S')).resolve()
    assert ROOT/'output' in out.parents and not out.exists(),'Use a fresh directory inside output'
    plan=json.loads((CODE/'config/experiments.json').read_text())
    ids=set(plan['categories']['spatial']['run_ids'])
    archived=ROOT/'output/results/seed9202_r35_analysis_20261015'
    rows=[];sites=[];arrays={};checks=0;max_error=0.
    for item in plan['runs']:
        if item['id'] not in ids:continue
        cfg=item['config'];path=ROOT/item['archived_run_path']/'trajectory.npz'
        assert hashlib.sha256(path.read_bytes()).hexdigest()==item['data_sha256']
        arrays[item['id']]={}
        with np.load(path,allow_pickle=False) as raw,np.load(archived/(item['id']+'_spatial_diagnostics.npz'),allow_pickle=False) as old:
            for epoch,state in zip(raw['snapshot_epochs'],raw['strategy_snapshots']):
                state=state.astype(np.float64);epoch=int(epoch)
                values=settle(state,cfg['r'],cfg['eta'],cfg['beta'])
                delta=incentive_map(state,cfg['r'],cfg['eta'],cfg['beta']) if cfg['variant']!='original' else np.full_like(state,cfg['r']-5)
                adjustment=values['reward']-values['base']
                for key,array in [(f'delta_{epoch}',delta),(f'net_adjustment_{epoch}',adjustment)]:
                    error=float(np.max(np.abs(array-old[key])));max_error=max(max_error,error)
                    np.testing.assert_allclose(array,old[key],rtol=0,atol=2e-12,err_msg=item['id']+':'+key)
                    arrays[item['id']][key]=array;checks+=1
                positive=0
                for row in (25,75,125,175):
                    for column in (25,75,125,175):
                        positive+=int(delta[row,column]>0)
                        sites.append(dict(run_id=item['id'],model=plan['model_display'][cfg['variant']],r=cfg['r'],epoch=epoch,row=row,column=column,strategy=int(state[row,column]),delta=float(delta[row,column])))
                c=state==1;d=~c
                rows.append(dict(run_id=item['id'],model=plan['model_display'][cfg['variant']],r=cfg['r'],epoch=epoch,cooperation=float(state.mean()),mean_net_C=float(adjustment[c].mean()) if c.any() else None,mean_net_D=float(adjustment[d].mean()) if d.any() else None,total_payment=float(values['paid'].sum()),total_fine=float(values['fine'].sum()),min_delta=float(delta.min()),max_delta=float(delta.max()),mean_delta=float(delta.mean()),sampled_positive=positive))
    assert len(arrays)==4 and len(rows)==20 and len(sites)==320
    out.mkdir(parents=True)
    for ident,data in arrays.items():np.savez_compressed(out/(ident+'_spatial_diagnostics.npz'),**data)
    write_csv(out/'spatial_state_metrics.csv',rows);write_csv(out/'fixed_site_incentives.csv',sites)
    report={'status':'passed','spatial_runs':4,'snapshots':20,'array_comparisons':checks,'focal_site_values':len(sites),'max_absolute_array_difference':max_error,'array_absolute_tolerance':2e-12,'archived_data_overwritten':False,'training_started':False,'source_sha256':{name:hashlib.sha256((CODE/'scripts'/name).read_bytes()).hexdigest() for name in ['analyze_incentives.py','uniform_incentives.py']}}
    (out/'incentive_reproduction_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))

if __name__=='__main__':main()
