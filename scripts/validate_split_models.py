"""Finite equivalence validation; --run explicitly permits 56 complete replays.
All configured payoff conditions and common-method ASTs are checked. This does
not retrain all 2513 configurations. Selection is fixed before source edits.
"""
import argparse,ast,hashlib,json,sys,time,importlib.util,platform,subprocess,os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from collections import Counter
from experiment_runner import worker,validate_active_sources,write_json
CODE=Path(__file__).resolve().parents[1];ROOT=CODE.parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load_model(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def common_functions(path):
    tree=ast.parse(path.read_text());return {f'{c.name}.{n.name}':ast.dump(n,include_attributes=False) for c in tree.body if isinstance(c,ast.ClassDef) for n in c.body if isinstance(n,ast.FunctionDef)}

def validate_payoffs(plan,report_dir):
    import numpy as np,torch
    torch.set_num_threads(1)
    old=load_model('submitted_joint_ppo',ROOT/plan['submitted_source_directory']/'PPO.py')
    modules={'original':load_model('separated_ppo',CODE/'PPO.py'),'UA':load_model('separated_cbp',CODE/'CBP_PPO.py')};modules['GF']=modules['UA']
    shared=['ActorCritic.__init__','ActorCritic.forward','SPGG.ppo_update','SPGG._make_batch','SPGG.run','SPGG.choose_action','SPGG.encode_state','SPGG._reset_buffer']
    reference=common_functions(ROOT/plan['submitted_source_directory']/'PPO.py')
    for name in ['PPO.py','CBP_PPO.py']:
        current=common_functions(CODE/name);assert all(reference[k]==current[k] for k in shared),(name,'Common learning routine changed')
    conditions=sorted({(x['config']['variant'],x['config']['r'],x['config']['eta'],x['config']['beta']) for x in plan['runs']})
    def models(L):
        result={}
        for variant in ['original','UA','GF']:
            values={'L_num':L,'device':torch.device('cpu'),'alpha':.001,'gamma':.99,'clip_epsilon':.2,'r':4.1,'epochs':1,'now_time':'equivalence','question':2,'ppo_epochs':1,'batch_size':1,'gae_lambda':.95,'output_path':str(report_dir/'unused_io'),'delta':.5,'rho':.01,'eta':0 if variant=='original' else .2,'beta':0 if variant=='original' else 3.,'variant':variant}
            result[variant]=(old.SPGG(**values),modules[variant].SPGG(**values))
        return result
    checks=0;coverage=[]
    for L,grid in [(8,conditions),(200,[(v,r,0. if v=='original' else .2,0. if v=='original' else 3.) for v in ['original','UA','GF'] for r in [3.5,4.1]])]:
        instances=models(L);rng=np.random.default_rng(20261116);half=np.zeros((L,L),dtype=np.float32);half[L//2:]=1
        states=[torch.zeros((L,L)),torch.ones((L,L)),torch.tensor(half),torch.tensor((rng.random((L,L))<.5).astype('float32'))]
        for variant,r,eta,beta in grid:
            a,b=instances[variant]
            for obj in [a,b]:obj.r=r;obj.eta=eta;obj.beta=beta
            for state in states:
                assert torch.equal(a.calculate_reward(state),b.calculate_reward(state)),('Settlement mismatch',variant,r,eta,beta,L)
                assert a.last_diagnostics==b.last_diagnostics,('Accounting mismatch',variant,r,eta,beta,L)
                checks+=1
        coverage.append({'L':L,'conditions':len(grid),'states_per_condition':len(states)})
    result={'status':'all_settlement_arrays_and_diagnostics_identical','checks':checks,'coverage':coverage,'common_learning_ASTs_identical':True,'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__}
    write_json(report_dir/'payoff_and_learning_equivalence.json',result);return result

def main():
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group();g.add_argument('--dry-run',action='store_true');g.add_argument('--run',action='store_true');p.add_argument('--workers',type=int,choices=range(1,5),default=4);p.add_argument('--max-hours',type=float,default=2);p.add_argument('--batch',default='split_validation_20261116');p.add_argument('--payoffs-only',type=Path,help=argparse.SUPPRESS);a=p.parse_args()
    plan=json.loads((CODE/'config/experiments.json').read_text());selection=json.loads((CODE/'config/split_validation_selection.json').read_text());ids=set(selection['run_ids']);items=[x for x in plan['runs'] if x['id'] in ids];assert len(items)==selection['count']==56
    validate_active_sources(plan)
    if a.payoffs_only is not None:
        validate_payoffs(plan,a.payoffs_only);return
    if not a.run:
        print(json.dumps({'status':'preview','full_replays':len(items),'L':200,'epochs':1000,'models':dict(Counter(x['config']['variant'] for x in items)),'training_started':False},indent=2));return
    assert a.max_hours>0 and a.batch and all(c.isalnum() or c in '_-' for c in a.batch)
    report_dir=ROOT/'output/provenance/reproduction_validation'/a.batch;report_dir.mkdir(parents=True,exist_ok=False)
    output=ROOT/'output/reproduction_validation'/a.batch;output.mkdir(parents=True,exist_ok=False)
    deadline=time.time()+a.max_hours*3600;started=time.monotonic()
    # Keep PyTorch out of this scheduling process before launching training children.
    subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'--payoffs-only',str(report_dir)],check=True,env=os.environ.copy())
    payoff=json.loads((report_dir/'payoff_and_learning_equivalence.json').read_text())
    state={'status':'running','selection_sha256':sha(CODE/'config/split_validation_selection.json'),'configuration_sha256':sha(CODE/'config/experiments.json'),'deadline_unix':deadline,'selected':len(items),'completed':[],'failures':[],'training_started':True,'all_2513_retrained':False,'payoff_checks':payoff['checks']};write_json(report_dir/'progress.json',state)
    pending={};iterator=iter(items);stopped=False
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        def launch():
            if time.time()>=deadline:raise TimeoutError('Finite validation deadline reached')
            try:item=next(iterator)
            except StopIteration:return False
            pending[pool.submit(worker,item,plan['shared_settings'],output,deadline,a.batch,True)]=item;return True
        for _ in range(a.workers):
            if not launch():break
        while pending:
            done=next(as_completed(pending));item=pending.pop(done)
            try:state['completed'].append(done.result())
            except BaseException as exc:state['failures'].append({'id':item['id'],'error_type':type(exc).__name__,'error':str(exc)});stopped=True
            write_json(report_dir/'progress.json',state)
            print(json.dumps({'completed':len(state['completed']),'selected':len(items),'failures':len(state['failures']),'elapsed_minutes':round((time.monotonic()-started)/60,2)}),flush=True)
            if not stopped:launch()
    state['elapsed_seconds']=time.monotonic()-started;state['status']='passed' if len(state['completed'])==len(items) and not state['failures'] else 'failed';state['models']=dict(Counter(plan['model_display'][x['config']['variant']] for x in items));state['arrays_compared']=sum(x['comparison']['array_count'] for x in state['completed']);state['source_sha256']=plan['source_sha256'];write_json(report_dir/'report.json',state);write_json(report_dir/'progress.json',state)
    print(json.dumps({'status':state['status'],'full_replays':len(state['completed']),'arrays_compared':state['arrays_compared'],'report':str(report_dir/'report.json')},indent=2),flush=True)
    if state['status']!='passed':raise SystemExit(1)
if __name__=='__main__':main()
