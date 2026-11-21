"""Route PPO and CBP-PPO to independent entry points; keep archived results immutable."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

CODE = Path(__file__).resolve().parents[1]
ROOT = CODE.parent
CATEGORIES = ('parameters','mechanisms','statistics','across_r','spatial','initialization')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def write_json(path,value):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    temp.replace(path)

def command(item,common,output,deadline):
    values={**common,**item['config']}
    argv=[sys.executable,'-B',str(CODE/item['entrypoint'])]
    for k,v in values.items():
        flag=('--' if k in ('eta','beta','variant','threads') else '-')+k
        argv.extend([flag,str(v)])
    return argv+['--output-dir',str(output),'--experiment','portable_replay','--deadline',str(deadline)]

def worker(item,common,output,deadline,batch,verify_archived=False):
    recorddir=ROOT/'output/provenance/reruns'/batch
    source_snapshots=CODE/'provenance/reruns'/batch/'executed_sources'
    log=recorddir/'logs'/(item['id']+'.txt');log.parent.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy();env.setdefault('MPLBACKEND','Agg');env.setdefault('MPLCONFIGDIR',str(recorddir/'mpl_cache'));env.setdefault('KMP_USE_SHM','0')
    expected=item['replay_run_id']
    run=output/'runs'/expected
    try:
        with log.open('x',encoding='utf-8') as f:
            subprocess.run(command(item,common,output,deadline),cwd=CODE,stdout=f,stderr=subprocess.STDOUT,env=env,check=True)
    finally:
        # The unmodified original program saves executed-source copies beside data.
        # Move only those .py copies to code, keeping output numeric/provenance only.
        for q in run.glob('*.py'):
            dest=source_snapshots/q.parent.name/q.name
            dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.exists():
                if sha(dest)!=sha(q):raise RuntimeError('Executed source copy mismatch')
                q.unlink()
            else:q.replace(dest)
        for q in (run/'checkpoint').glob('*.pth'):
            dest=recorddir/'checkpoints'/expected/q.name
            dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.exists():
                if sha(dest)!=sha(q):raise RuntimeError('Checkpoint copy mismatch')
                q.unlink()
            else:q.replace(dest)
        if (run/'checkpoint').exists() and not list((run/'checkpoint').iterdir()):(run/'checkpoint').rmdir()
    meta=json.loads((run/'provenance.json').read_text(encoding='utf-8'))
    assert meta['status']=='complete' and sha(run/'trajectory.npz')==meta['data_sha256']
    assert meta['source_sha256']==item['replay_source_sha256']
    meta['archived_run_id']=item['actual_run_id']
    meta['archived_data_sha256']=item['data_sha256']
    meta['entrypoint']=item['entrypoint']
    comparison=None
    if verify_archived:
        # Numeric libraries stay in a short-lived process: loading their OpenMP
        # runtime in the scheduler prevents later child starts on some macOS setups.
        checker="from pathlib import Path; import sys,json; from experiment_runner import compare_trajectories; print(json.dumps(compare_trajectories(Path(sys.argv[1]),Path(sys.argv[2]))))"
        checked=subprocess.run([sys.executable,'-B','-c',checker,
            str(ROOT/item['archived_run_path']/'trajectory.npz'),str(run/'trajectory.npz')],
            cwd=CODE/'scripts',env=env,check=True,capture_output=True,text=True)
        comparison=json.loads(checked.stdout)
        meta['archived_comparison']=comparison
    write_json(run/'provenance.json',meta)
    return {'id':item['id'],'run_path':run.relative_to(ROOT).as_posix(),'data_sha256':meta['data_sha256'],
            'replay_run_id':expected,'archived_run_id':item['actual_run_id'],'comparison':comparison}

def compare_trajectories(archived,replayed):
    import numpy as np
    differences=[]
    with np.load(archived,allow_pickle=False) as a,np.load(replayed,allow_pickle=False) as b:
        assert set(a.files)==set(b.files),'Array-key mismatch'
        for name in a.files:
            if a[name].shape!=b[name].shape or a[name].dtype!=b[name].dtype:
                differences.append({'array':name,'reason':'shape/dtype'})
            elif a[name].tobytes()!=b[name].tobytes():
                differences.append({'array':name,'reason':'numeric difference','max_abs_difference':float(np.max(np.abs(a[name].astype(float)-b[name].astype(float))))})
        result={'status':'all_arrays_identical' if not differences else 'different','array_count':len(a.files),'differences':differences}
    if differences:raise AssertionError(result)
    return result

def validate_active_sources(plan):
    for name,digest in plan['source_sha256'].items():assert sha(CODE/name)==digest,('Active source drift',name)
    archived=ROOT/plan['submitted_source_directory']
    for name in ('PPO.py','main_PPO.py'):assert sha(archived/name)==plan['submitted_source_sha256'][name],('Submitted source drift',name)


def run_category(category,common,experiment,argv=None):
    p=argparse.ArgumentParser(description='Preview or explicitly reproduce one frozen experiment family.')
    mode=p.add_mutually_exclusive_group();mode.add_argument('--dry-run',action='store_true');mode.add_argument('--run',action='store_true')
    p.add_argument('--batch');p.add_argument('--workers',type=int,default=1,choices=range(1,5));p.add_argument('--max-hours',type=float,default=24)
    p.add_argument('--resume',action='store_true');p.add_argument('--verify-archived',action='store_true',help='Require every replayed array to match the archived run exactly')
    a=p.parse_args(argv)
    plan=json.loads((CODE/'config/experiments.json').read_text(encoding='utf-8'))
    specs=json.loads((CODE/'config/category_settings.json').read_text(encoding='utf-8'))
    assert category in CATEGORIES and common==plan['shared_settings'] and experiment==specs[category]
    selected=set(plan['categories'][category]['run_ids']);items=[r for r in plan['runs'] if r['id'] in selected]
    assert len(items)==experiment['unique_runs']
    if not a.run:
        print(json.dumps({'category':category,'unique_configurations':len(items),'common_settings':common,'experiment_settings':experiment,'entrypoints':plan['entrypoints'],'training_started':False},indent=2));return
    validate_active_sources(plan)
    batch=a.batch or datetime.now().strftime('replay_%Y%m%d_%H%M%S')
    if not batch or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in batch):p.error('Use letters, digits, underscores and hyphens for --batch')
    if a.max_hours<=0:p.error('--max-hours must be positive')
    output=ROOT/'output/reruns'/batch;output.mkdir(parents=True,exist_ok=True)
    lock=output/'.run.lock'
    try:fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:p.error('This batch has an active or unresolved lock. Inspect it before restarting.')
    os.close(fd)
    try:
        ledgerfile=output/'state.json'
        if ledgerfile.exists():
            ledger=json.loads(ledgerfile.read_text(encoding='utf-8'))
            assert ledger['config_sha256']==sha(CODE/'config/experiments.json')
            assert not ledger['failures'] and {r['id'] for r in ledger['submitted']}=={r['id'] for r in ledger['completed']},'Inspect failed/unresolved attempts; no automatic retry.'
            if category in ledger['started_categories'] and not a.resume:p.error('Category previously started. Inspect then use --resume for untouched configurations.')
        else:
            ledger={'config_sha256':sha(CODE/'config/experiments.json'),'deadline_unix':time.time()+a.max_hours*3600,'submitted':[],'completed':[],'failures':[],'started_categories':[]}
        for row in ledger['completed']:assert sha(ROOT/row['run_path']/'trajectory.npz')==row['data_sha256']
        completed={r['id'] for r in ledger['completed']};remaining=[r for r in items if r['id'] not in completed]
        ledger['started_categories'].append(category);write_json(ledgerfile,ledger)
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            pending={};iterator=iter(remaining);failed=False
            def launch():
                if time.time()>=ledger['deadline_unix']:raise TimeoutError('Batch wall-clock ceiling reached')
                try:item=next(iterator)
                except StopIteration:return False
                ledger['submitted'].append({'id':item['id'],'category':category});write_json(ledgerfile,ledger)
                pending[pool.submit(worker,item,common,output,ledger['deadline_unix'],batch,a.verify_archived)]=item
                return True
            for _ in range(a.workers):
                if not launch():break
            while pending:
                done=next(as_completed(pending));item=pending.pop(done)
                try:ledger['completed'].append(done.result())
                except BaseException as exc:
                    failed=True;ledger['failures'].append({'id':item['id'],'type':type(exc).__name__,'error':str(exc)})
                write_json(ledgerfile,ledger)
                print(json.dumps({'category':category,'completed_in_batch':len(ledger['completed']),'failures':len(ledger['failures'])}),flush=True)
                if not failed:launch()
        if ledger['failures']:raise RuntimeError('Failed run preserved; inspect before further execution.')
        print(json.dumps({'category':category,'status':'complete','batch':batch,'new_runs':len(remaining)}))
    finally:lock.unlink(missing_ok=True)
