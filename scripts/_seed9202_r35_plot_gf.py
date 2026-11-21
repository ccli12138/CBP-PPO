"""Inherited line drawing blocks, with author-approved GF-only financing comparison.

Requires all formal data to pass the unified audit. No training or reselection.
"""
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path(os.environ.get('TMPDIR', '/tmp')) / 'cbp_mpl'))
os.environ.setdefault('MPLBACKEND', 'Agg')
ROOT = Path(__file__).resolve().parents[1]


def read(path):
    with path.open() as handle: return list(csv.DictReader(handle))


def main():
    import numpy as np
    data = ROOT.parent/'output/results/seed9202_r35_analysis_20261015'
    status = json.loads((data / 'analysis_status.json').read_text())
    assert status['complete'] and status['audit_sha256'] == hashlib.sha256((ROOT.parent/'output' / status['audit']).read_bytes()).hexdigest()
    source = ROOT / 'reference/LMI-PPO_CSF_20260928/code/archive/experiments/plot_caption_matched_20260906_v2.py'
    spec = importlib.util.spec_from_file_location('cbp_formal_lines', source)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    module.Figure.savefig = module.save_matched
    out = Path(os.environ['CBP_DRAW_ROOT']) / 'figures/seed9202_r35_20261015/gf_only/formal'; out.mkdir(parents=True, exist_ok=True)
    raw = module.old.source('plot_density_para.py')
    original = raw.split('# 定义不同的标记样式', 1)[1].split('else:\n    print', 1)[0]
    paper = Path(os.environ['CBP_DRAW_ROOT']) / 'figures/seed9202_r35_20261015/gf_only/flat'
    paper.mkdir(parents=True, exist_ok=True)
    plots = []
    def draw(curves, path, ylabel, named=False, stable=None):
        path.mkdir(parents=True, exist_ok=True)
        block = original
        # All21 observations remain plotted; spaced labels keep the physical
        # 10pt caption-matched type readable at the inherited panel width.
        block = block.replace('plt.xticks(fontsize=16)',
                              "plt.xticks([3.0,3.5,4.0,4.5,5.0], ['3.0','3.5','4.0','4.5','5.0'], fontsize=16)")
        block = block.replace('plt.ylabel("Fractions", fontsize=16)', f'plt.ylabel({ylabel!r}, fontsize=16)')
        if named:
            block = block.replace("label=fr'$\\{legend} = {gl_value}$'", 'label=str(gl_value)')
            # The same variant has the same marker/color in both diagnostics.
            block = block.replace('marker=marker,', "marker=STYLE[str(gl_value)][1], color=STYLE[str(gl_value)][0],")
        if ylabel not in ('Fractions', 'Fine activity'):
            block = block.replace('plt.ylim(-0.1, 1.1)', 'plt.ylim(bottom=-0.05 * max(plt.ylim()[1], 1e-12))')
        # Keep the original fraction margins: boundary markers remain visible.
        if stable is not None:
            # Inherit the original filled/open stability diagnostic, adapting
            # only the newly audited Boolean values. Preserve endpoint heights.
            statement = "plt.plot(x_values, y_values, marker=STYLE[str(gl_value)][1], color=STYLE[str(gl_value)][0], markersize=10, markerfacecolor='none', linestyle=\"-\", label=str(gl_value))"
            assert statement in block
            block = block.replace(statement, statement + "\n        for x,y,ok in zip(x_values,y_values,STABLE[str(gl_value)]):\n            if ok: plt.plot(x,y,marker=STYLE[str(gl_value)][1],color=STYLE[str(gl_value)][0],markersize=10,linestyle='none')")
        variables = dict(data_dict=curves, root_path=str(path), legend='variant' if named else 'eta',
                         STYLE={'CBP-PPO': ('#1f77b4', 'o'), 'WA': ('#ff7f0e', 's'),
                                'GF': ('#2ca02c', '^'), 'NC': ('#d62728', 'D')}, STABLE=stable)
        module.old.execute('plot_density_para.py::approved_interface', block, variables)
        return path / ('C_D_r_variant.pdf' if named else 'C_D_r_eta.pdf')
    manifest = json.loads((ROOT / 'config/plot_manifest_20261015.json').read_text())
    rows = read(data / 'mechanisms_metrics.csv')
    metrics = [('endpoint', 'Fractions'), ('payment_per_player_round', 'Mean payment'),
               ('fine_activity', 'Fine activity'), ('fine_per_defector_incidence', 'Fine per D incidence')]
    for figure_id, variants, grid in [(5, ['UA', 'GF'], manifest['component_r'])]:
        for index, (metric, ylabel) in enumerate(metrics, 1):
            curves, stable = {}, {}
            for variant in variants:
                name = {'UA':'CBP-PPO','full':'WA','GF':'GF','UNC':'NC'}[variant]
                group = sorted([row for row in rows if row['variant'] == variant and float(row['r']) in grid], key=lambda row: float(row['r']))
                assert len(group) == len(grid)
                assert all(row[metric] != '' for row in group), 'Undefined metric must not be replaced by zero'
                curves[name] = ([float(row['r']) for row in group], [float(row[metric]) for row in group])
                stable[name] = [row['stable_full'] == 'True' for row in group]
            figure = draw(curves, out / f'fig{figure_id:02d}_{metric}', ylabel, named=True,
                          stable=stable if figure_id == 5 and metric == 'endpoint' else None)
            target = paper / f'fig{figure_id:02d}_{index:02d}_diagnostic.pdf'; target.write_bytes(figure.read_bytes())
            plots.append(dict(figure=target.name, metric=metric, curves=curves, stable=stable))
    (out / 'drawing_provenance.json').write_text(json.dumps(dict(
        status='drawn_pending_actual_pdf_review', source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        raw_block_sha256=hashlib.sha256(original.encode()).hexdigest(), plots=plots,
        scientific_data='New frozen formal runs only; exact eta=0 copies are declared in source CSV.',
        adaptations='New parameter/metric labels, variant abbreviations, unbounded cost axes, stable-marker interface; original drawing blocks and physical font matching.',
        font_calibration=module.RECORDS, blocks=module.old.BLOCKS), indent=2) + '\n')
    print('Four inherited GF-only financing panels drawn; no training.')


if __name__ == '__main__': main()
