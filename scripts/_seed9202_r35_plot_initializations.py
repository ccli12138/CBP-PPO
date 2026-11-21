"""Original initialization trajectories and snapshots, with the NPZ key adapted."""
import csv
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path(os.environ.get('TMPDIR', '/tmp')) / 'cbp_mpl'))
os.environ.setdefault('MPLBACKEND', 'Agg')
ROOT = Path(__file__).resolve().parents[1]


def main():
    data = ROOT.parent/'output/results/seed9202_r35_analysis_20261015'
    status = json.loads((data / 'analysis_status.json').read_text())
    assert status['complete'] and status['audit_sha256'] == hashlib.sha256((ROOT.parent/'output' / status['audit']).read_bytes()).hexdigest()
    source = ROOT / 'reference/LMI-PPO_CSF_20260928/code/archive/experiments/plot_caption_matched_20260906_v2.py'
    spec = importlib.util.spec_from_file_location('cbp_initialization_reference', source)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    module.Figure.savefig = module.save_matched
    original = inspect.getsource(module.old.trajectory_and_snapshots)
    adapted = original.replace('a["snapshots"]', 'a["strategy_snapshots"]')
    assert original != adapted
    exec(compile(adapted, 'original_drawing::NPZ_key_adapter', 'exec'), module.old.__dict__)
    out = Path(os.environ['CBP_DRAW_ROOT']) / 'figures/seed9202_r35_20261015/initialization'; out.mkdir(parents=True, exist_ok=True)
    with (data / 'initialization_metrics.csv').open() as handle: rows = list(csv.DictReader(handle))
    assert len(rows) == 8
    records = []
    for row in rows:
        mapped = dict(run_id=row['run_id'], artifact=str(Path(row['run_path']) / 'trajectory.npz'), r=row['r'])
        assert hashlib.sha256((ROOT.parent/'output' / mapped['artifact']).read_bytes()).hexdigest() == row['data_sha256']
        module.old.trajectory_and_snapshots(mapped, ROOT.parent/'output', out)
        src = out / 'spatial' / row['run_id']
        figure = 10 if row['question'] == '1' else 11
        prefix = f'fig{figure}_r{float(row["r"]):g}_{"PPO" if row["variant"] == "original" else "CBP"}'
        files = {'trajectory': src / f'strategy_evolution_r{float(row["r"])}_run0.pdf'}
        files.update({f't{epoch}': src / f't={epoch}.pdf' for epoch in [0, 1, 10, 100, 1000]})
        for key, path in files.items():
            target = Path(os.environ['CBP_DRAW_ROOT']) / 'figures/seed9202_r35_20261015/flat' / f'{prefix}_{key}.pdf'
            target.write_bytes(path.read_bytes())
            records.append(dict(run_id=row['run_id'], figure=target.name,
                                figure_sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
    (out / 'drawing_provenance.json').write_text(json.dumps(dict(
        status='drawn_pending_actual_pdf_review', records=records,
        interface_change='Only snapshots key becomes strategy_snapshots; data are read from audited new-run paths.',
        original_adapter_sha256=hashlib.sha256(original.encode()).hexdigest(),
        executed_adapter_sha256=hashlib.sha256(adapted.encode()).hexdigest(),
        font_calibration=module.RECORDS, blocks=module.old.BLOCKS), indent=2) + '\n')
    print('Eight initialization runs drawn using original blocks; 48 flat paper figures.')


if __name__ == '__main__': main()
