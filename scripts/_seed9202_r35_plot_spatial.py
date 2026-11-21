"""Execute original trajectory/snapshot drawing blocks with new NPZ interfaces."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import textwrap
from types import SimpleNamespace

os.environ.setdefault('MPLCONFIGDIR', str(Path(os.environ.get('TMPDIR', '/tmp')) / 'cbp_mpl'))
os.environ.setdefault('MPLBACKEND', 'Agg')
ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    import numpy as np
    source = ROOT / 'reference/LMI-PPO_CSF_20260928/code/archive/experiments/plot_caption_matched_20260906_v2.py'
    spec = importlib.util.spec_from_file_location('cbp_spatial_reference', source)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    module.Figure.savefig = module.save_matched
    manifest = json.loads((ROOT / 'config/plot_manifest_20261015.json').read_text())
    reuse = json.loads((ROOT.parent/'output/results/seed9202_r35_analysis_20261015/plot_reuse_map.json').read_text())
    mapping = {row['id']: row for row in reuse['reused_runs']}
    out = Path(os.environ['CBP_DRAW_ROOT']) / 'figures/seed9202_r35_20261015/spatial'; out.mkdir(parents=True, exist_ok=True)
    paper = Path(os.environ['CBP_DRAW_ROOT']) / 'figures/seed9202_r35_20261015/flat'
    records = []
    for index, logical_id in enumerate(manifest['categories']['spatial']['run_ids']):
        row = mapping[logical_id]; cfg = row['planned_config']
        dest = out / logical_id; dest.mkdir(parents=True, exist_ok=True)
        data = ROOT.parent/'output' / row['run_path'] / 'trajectory.npz'
        assert digest(data) == row['data_sha256']
        with np.load(data) as archive:
            cooperation = archive['cooperation'].copy()
            epochs = archive['snapshot_epochs'].copy()
            states = archive['strategy_snapshots'].copy()
        block = module.old.source('plot_r_data.py').split('# ==================== 绘图部分 ====================', 1)[1]
        module.old.execute('plot_r_data.py::drawing', block,
                           dict(C_Y=cooperation, D_Y=1-cooperation, r=cfg['r'], num=0,
                                output_path=str(dest), fontsize=24, xticks=[0, 1, 10, 100, 1000],
                                fra_yticks=[0., .2, .4, .5, .6, .8, 1.]))
        raw = module.old.source('PPO.py')
        start = raw.index('        fig1 = plt.figure(figsize=(8, 8))')
        end = raw.index('        plt.close(fig1)', start) + len('        plt.close(fig1)')
        block = textwrap.dedent(raw[start:end])
        for epoch, state in zip(epochs, states):
            module.old.execute('PPO.py::strategy_snapshot', block,
                               dict(type_t_matrix=np.asarray(state), self=SimpleNamespace(L_num=200),
                                    img_dir=str(dest), epoch=int(epoch)))
        # Flat paper files, with no image written into a results directory.
        prefix = f'fig{8 if cfg["r"] == manifest["spatial_r"][0] else 9:02d}_{"PPO" if cfg["variant"] == "original" else "CBP"}'
        files = {'trajectory': dest / f'strategy_evolution_r{cfg["r"]}_run0.pdf'}
        files.update({f't{epoch}': dest / f't={epoch}.pdf' for epoch in epochs})
        for key, path in files.items():
            target = paper / f'{prefix}_{key}.pdf'; target.write_bytes(path.read_bytes())
            records.append(dict(logical_id=logical_id, trajectory_sha256=digest(data),
                                figure=str(target.relative_to(ROOT.parent)), figure_sha256=digest(target)))
    (out / 'drawing_provenance.json').write_text(json.dumps(
        dict(status='drawn_pending_actual_pdf_review', records=records, source_sha256=digest(source),
             reuse='Original trajectory and black/white snapshot drawing blocks; NPZ key adapter and physical font matching.',
             font_calibration=module.RECORDS, blocks=module.old.BLOCKS), indent=2) + '\n')
    print(json.dumps(dict(spatial_runs=4, flat_paper_figures=len(records), output=str(out))))


if __name__ == '__main__':
    main()
