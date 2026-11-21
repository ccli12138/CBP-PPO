"""Redraw completed paired observations using the reference violin drawing block."""
import ast
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil

os.environ.setdefault('MPLCONFIGDIR', str(Path(os.environ.get('TMPDIR', '/tmp')) / 'cbp_mpl'))
os.environ.setdefault('MPLBACKEND', 'Agg')
ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    data = ROOT.parent/'output/results/seed9202_r35_analysis_20261015'
    completion = json.loads((Path(os.environ['CBP_DRAW_CONTROL']) / 'plot_completion.json').read_text())
    assert sha(data / 'r_threshold_report.json') == completion['report_sha256']
    source = ROOT / 'reference/LMI-PPO_CSF_20260928/code/archive/experiments/plot_caption_matched_20260906_v2.py'
    spec = importlib.util.spec_from_file_location('cbp_statistics_reference', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Reference TeX stacks these panels at full linewidth. Calibrate to that actual width.
    text = source.read_text()
    node = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == 'save_matched')
    block = ast.get_source_segment(text, node).replace("('architecture','mechanism_flow')", "('architecture','mechanism_flow','violin')")
    exec(compile(block, 'reference_fullwidth_violin_font_adapter', 'exec'), module.__dict__)
    module.Figure.savefig = module.save_matched
    module.old.METHODS = ('original', 'full')
    raw = module.old.source('main_PPO_stat_box_save.py')
    node = next(n for n in ast.parse(raw).body if isinstance(n, ast.FunctionDef) and n.name == 'plot_cooperation_violinplot')
    draw = ast.get_source_segment(raw, node).replace('Final Cooperation Fractions', 'Window-mean\ncooperation')
    # The reference function contains Python string literals: preserve the escaped newline.
    draw = draw.replace('Window-mean\ncooperation', 'Window-mean\\ncooperation')
    draw = draw.replace('figsize=(12, 6)', 'figsize=(16, 6)')
    draw = draw.replace("loc='lower center'", "loc='best'")
    draw = draw.replace('bbox_to_anchor=(0.5, 1.01),', '')
    draw = draw.replace('ncol=3', 'ncol=1')
    draw = draw.replace('plt.tight_layout(rect=[0, 0, 1, 0.9])', 'plt.tight_layout()')
    draw = draw.replace('plt.yticks(fontsize=fontsize)', 'plt.xticks(rotation=90)\n    plt.yticks(fontsize=fontsize)')
    rows = list(csv.DictReader((data / 'r_threshold_run_metrics.csv').open()))
    out = Path(os.environ['CBP_DRAW_ROOT']) / 'figures/seed9202_r35_20261015/statistics'
    out.mkdir(parents=True, exist_ok=True)
    for variant, model in [('original', 'PPO'), ('UA', 'CBP-PPO')]:
        selected = [row for row in rows if row['category'] == 'statistics' and row['model'] == model]
        assert len(selected) == 720
        values = {}
        for row in selected:
            values.setdefault(float(row['r']), []).append(float(row['window_mean']))
        assert len(values) == 36 and all(len(v) == 20 for v in values.values())
        dest = out / variant
        dest.mkdir(exist_ok=True)
        ns = module.old.execute('main_PPO_stat_box_save.py::window_mean_data_adapter', draw, {})
        ns['plot_cooperation_violinplot']({r: {'C': {'raw_data': v}} for r, v in sorted(values.items())}, str(dest), fontsize=16)
        for ext in ('pdf', 'png'):
            candidates = list(dest.glob('*violin*.' + ext))
            assert len(candidates) == 1
            shutil.copy2(candidates[0], out / (model + '_window_violin.' + ext))
    audit = dict(status='drawn_pending_visual_review', scientific_source='1440 new paired runs; final 101-state means',
                 reference_source=str(source.relative_to(ROOT)), reference_source_sha256=sha(source),
                 data_csv_sha256=sha(data / 'r_threshold_run_metrics.csv'),
                 report_sha256=completion['report_sha256'],
                 changes=['Window-mean data interface and ylabel', '36 tested r positions; labels every 0.5, vertical ticks',
                          'User-required inside loc=best legend',
                          'Full-linewidth calibration consistent with stacked reference TeX'],
                 font_calibration=module.RECORDS, manuscript_pdf_verified=False)
    (out / 'drawing_provenance.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps({'output': str(out), 'models': ['PPO', 'CBP-PPO'], 'runs': 1440}))


if __name__ == '__main__':
    main()
