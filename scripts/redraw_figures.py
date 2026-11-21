"""Draw the original six-category data; PPO/CBP-PPO training files are independent. Figures go only under paper."""
import argparse,csv,hashlib,json,os,re,shutil,subprocess,sys
from datetime import datetime
from pathlib import Path
CODE=Path(__file__).resolve().parents[1];ROOT=CODE.parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path);a=p.parse_args()
    data=ROOT/'output/results/seed9202_r35_analysis_20261015'
    required=['parameter_surface.csv','mechanisms_metrics.csv','r_threshold_run_metrics.csv','initialization_metrics.csv','plot_reuse_map.json','analysis_status.json','r_threshold_report.json']
    missing=[str(data/n) for n in required if not (data/n).is_file() or getattr((data/n).stat(),'st_flags',0)&0x40000000]
    if missing:raise SystemExit('Required data are missing; restore full data before redrawing: '+', '.join(missing))
    plan=json.loads((CODE/'config/experiments.json').read_text());ids=set(plan['categories']['spatial']['run_ids']+plan['categories']['initialization']['run_ids'])
    for r in plan['runs']:
        if r['id'] in ids:
            q=ROOT/r['archived_run_path']/'trajectory.npz';assert q.is_file() and sha(q)==r['data_sha256'],str(q)
    out=(a.output_dir or ROOT/'paper'/datetime.now().strftime('redrawn_%Y%m%d_%H%M%S')).resolve()
    assert ROOT/'paper' in out.parents and not out.exists(),'Use a fresh directory under paper'
    out.mkdir(parents=True);env=os.environ.copy();env['CBP_DRAW_ROOT']=str(out);env.setdefault('MPLBACKEND','Agg')
    # Historical adapters read this tiny drawing-control JSON. Keep the original data directory read-only.
    scratch=out/'control';scratch.mkdir()
    (scratch/'plot_completion.json').write_text(json.dumps({'report_sha256':sha(data/'r_threshold_report.json')})+'\n');env['CBP_DRAW_CONTROL']=str(scratch)
    flat=out/'figures/seed9202_r35_20261015/flat';flat.mkdir(parents=True)
    for name in ('parameters','gf','statistics','r_threshold','spatial','initializations'):
        subprocess.run([sys.executable,'-B',str(CODE/'scripts'/f'_seed9202_r35_plot_{name}.py')],check=True,cwd=CODE,env=env)
    figures=out/'figures/seed9202_r35_20261015'
    for folder,mapping in [('statistics',{'PPO_window_violin.pdf':'fig06_01_window_cooperation_violinplot.pdf','CBP-PPO_window_violin.pdf':'fig06_02_window_cooperation_violinplot.pdf'}),('across_r',{'PPO_ordinary_across_r.pdf':'fig07_01_C_D_r.pdf','CBP-PPO_ordinary_across_r.pdf':'fig07_02_C_D_r.pdf'})]:
        for source,target in mapping.items():shutil.copy2(figures/folder/source,flat/target)
    for q in (figures/'gf_only/flat').glob('*.pdf'):shutil.copy2(q,flat/q.name)
    subprocess.run([sys.executable,'-B',str(CODE/'scripts/plot_uniform_model_diagram_20261012.py')],check=True,cwd=CODE,env=env)
    for q in (out/'flat').glob('*.pdf'):shutil.copy2(q,flat/q.name)
    text=(ROOT/'paper/main.tex').read_text();needed=set(re.findall(r'\\(?:includegraphics(?:\[[^]]*\])?|snapshot)\{([^}]+\.pdf)\}',text));assert len(needed)==88
    assert all((flat/n).exists() for n in needed),sorted(n for n in needed if not(flat/n).exists())
    (out/'drawing_report.json').write_text(json.dumps({'status':'redrawn_pending_actual_PDF_visual_review','flat_figures':len(needed),'paper_assets_unchanged':True,'archived_data_unchanged':True,'training_started':False},indent=2)+'\n')
    print(str(flat))
if __name__=='__main__':main()
