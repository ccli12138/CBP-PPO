"""Reuse the frozen reference's ordinary across-r drawing block on new data."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil

os.environ.setdefault('MPLCONFIGDIR', str(Path(os.environ.get('TMPDIR', '/tmp')) / 'cbp_mpl'))
os.environ.setdefault('MPLBACKEND', 'Agg')
ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    code_root = ROOT
    import csv
    assert code_root == ROOT
    data = ROOT.parent/'output/results/seed9202_r35_analysis_20261015'
    report = json.loads((data / 'r_threshold_report.json').read_text())
    assert report['status'] == 'complete_audited'
    with (data / 'r_threshold_run_metrics.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    ordinary = [dict(variant='original' if row['model'] == 'PPO' else 'full',
                     r=row['r'], final_cooperation=row['endpoint'])
                for row in rows if row['category'] == 'ordinary']
    assert len(ordinary) == 72
    source = ROOT / 'reference/LMI-PPO_CSF_20260928/code/archive/experiments/plot_caption_matched_20260906_v2.py'
    spec = importlib.util.spec_from_file_location('cbp_reference_drawing', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = Path(os.environ['CBP_DRAW_ROOT']) / 'figures/seed9202_r35_20261015/across_r'
    out.mkdir(parents=True, exist_ok=True)
    module.old.METHODS = ('original', 'full')
    original_save_matched = module.save_matched
    def margin_save(fig, *args, **kwargs):
        for ax in fig.axes: ax.set_ylim(-0.1, 1.1)
        return original_save_matched(fig, *args, **kwargs)
    module.Figure.savefig = margin_save
    module.old.across_r(ordinary, out)
    for method, name in [('original', 'PPO'), ('full', 'CBP-PPO')]:
        pdf = out / 'across_r' / method / 'C_D_r.pdf'
        shutil.copy2(pdf, out / (name + '_ordinary_across_r.pdf'))
    # PNG previews use the exact same original block, with only export format adapted.
    # PDF figures above remain vector exports using the inherited font calibration.
    import numpy as np
    import textwrap
    raw = module.old.source('plot_C_D_r.py')
    block = textwrap.dedent(raw.split('    # 5. 绘制折线图', 1)[1].split('else:', 1)[0])
    png_block = block.replace('C_D_r.pdf', 'C_D_r.png').replace("format='pdf'", "format='png'").replace('format="pdf"', 'format="png"')
    for method, name in [('original', 'PPO'), ('full', 'CBP-PPO')]:
        selected = sorted([row for row in ordinary if row['variant'] == method], key=lambda row: float(row['r']))
        dest = out / 'across_r' / method
        module.old.execute('plot_C_D_r.py::PNG_export_only', png_block,
                           dict(x_values=[float(row['r']) for row in selected],
                                y_values=np.asarray([float(row['final_cooperation']) for row in selected]),
                                font_size=24, output_path=str(dest)))
        shutil.copy2(dest / 'C_D_r.png', out / (name + '_ordinary_across_r.png'))
    audit = dict(status='drawn_pending_visual_review', scientific_data='new CBP/PPO ordinary runs only; not old LMI results',
                 metric='f_C(T) and f_D(T); statistical success counts are in report, not these ordinary endpoint panels',
                 source=str(source.relative_to(ROOT)), source_sha256=digest(source),
                 raw_source_sha256=digest(module.old.ROOT / 'raw/source/plot_C_D_r.py'),
                 report_sha256=digest(data / 'r_threshold_report.json'),
                 reuse='Original drawing block and physical font matching; only data adapter and PNG export format differ.',
                 font_calibration=module.RECORDS)
    (out / 'drawing_provenance.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps({'figures': str(out), 'models': ['PPO', 'CBP-PPO'], 'metric': audit['metric']}))


if __name__ == '__main__':
    main()
