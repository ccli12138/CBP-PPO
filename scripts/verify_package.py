"""Read-only checks of split active sources, original execution provenance and numeric data."""
import argparse,ast,csv,hashlib,json,re
from pathlib import Path
CODE=Path(__file__).resolve().parents[1];ROOT=CODE.parent

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--structure-only',action='store_true');a=p.parse_args()
    folders={q.name for q in ROOT.iterdir() if q.is_dir()}
    assert {'paper','code','output'} <= folders <= {'paper','code','output','paper_clear'}
    assert not (CODE/'output').exists()
    assert [q.name for q in (ROOT/'paper').glob('*.tex')]==['main.tex']
    prohibited={'.png','.pdf','.svg','.jpg','.jpeg','.gif','.eps','.py','.pyc'}
    assert not [q for q in (ROOT/'output').rglob('*') if q.is_file() and q.suffix.lower() in prohibited]
    text=(ROOT/'paper/main.tex').read_text(encoding='utf-8')
    assets=set(re.findall(r'\\(?:includegraphics(?:\[[^]]*\])?|snapshot)\{([^}]+\.(?:pdf|png|eps|jpg))\}',text));assert len(assets)==88
    assert all((ROOT/'paper'/n).is_file() for n in assets)
    plan=json.loads((CODE/'config/experiments.json').read_text());assert plan['code_layout_version']==2 and len(plan['runs'])==2513
    original=ROOT/plan['submitted_source_directory'];old_sources={n:sha(original/n) for n in ['PPO.py','main_PPO.py']}
    assert old_sources=={n:plan['submitted_source_sha256'][n] for n in old_sources}
    for name,digest in plan['source_sha256'].items():assert sha(CODE/name)==digest,('Active source drift',name)
    for file in [*CODE.glob('*.py'),*(CODE/'scripts').glob('*.py')]:ast.parse(file.read_text())
    oldtree=ast.parse((original/'PPO.py').read_text());oldmethods={f'{c.name}.{f.name}':ast.dump(f,include_attributes=False) for c in oldtree.body if isinstance(c,ast.ClassDef) for f in c.body if isinstance(f,ast.FunctionDef)}
    for name in ['PPO.py','CBP_PPO.py']:
        tree=ast.parse((CODE/name).read_text());methods={f'{c.name}.{f.name}':ast.dump(f,include_attributes=False) for c in tree.body if isinstance(c,ast.ClassDef) for f in c.body if isinstance(f,ast.FunctionDef)}
        for key in ['ActorCritic.__init__','ActorCritic.forward','SPGG.ppo_update','SPGG._make_batch','SPGG.run','SPGG.choose_action','SPGG.encode_state','SPGG._reset_buffer']:assert methods[key]==oldmethods[key],(name,key)
        if name=='PPO.py':assert 'SPGG._settle_intervention' not in methods
    checks=0;replay_ids=set()
    for row in plan['runs']:
        run=ROOT/row['archived_run_path'];data=run/'trajectory.npz';assert data.is_file(),str(data)
        meta=json.loads((run/'provenance.json').read_text());assert meta['status']=='complete' and meta['data_sha256']==row['data_sha256']
        assert meta['source_sha256']==old_sources
        cfg={**plan['shared_settings'],**row['config']};assert meta['config']==cfg
        oldid=hashlib.sha256(json.dumps({'config':cfg,'sources':old_sources},sort_keys=True).encode()).hexdigest()[:20];assert oldid==row['actual_run_id']
        route=plan['entrypoints'][cfg['variant']];assert (row['entrypoint'],row['model_file'])==(route['main'],route['model_file'])
        sources={name:sha(CODE/name) for name in [route['main'],route['model_file']]};assert sources==row['replay_source_sha256']
        replayid=hashlib.sha256(json.dumps({'config':cfg,'sources':sources},sort_keys=True).encode()).hexdigest()[:20];assert replayid==row['replay_run_id'];replay_ids.add(replayid)
        if not a.structure_only:assert sha(data)==row['data_sha256'],str(data)
        checks+=1
    assert len(replay_ids)==2513
    if not a.structure_only:
        inv=CODE/'provenance/package_inventory.csv';assert inv.exists()
        for row in csv.DictReader(inv.open(encoding='utf-8')):
            q=ROOT/row['path'];assert q.is_file() and q.stat().st_size==int(row['bytes']) and sha(q)==row['sha256'],row['path'];checks+=1
    print(json.dumps({'status':'passed','checks':checks,'archived_raw_runs':2513,'separate_replay_ids':len(replay_ids),'manuscript_assets':len(assets),'learning_AST_equivalence':True,'training_started':False,'mode':'structure' if a.structure_only else 'full_hash_inventory'}))
if __name__=='__main__':main()
