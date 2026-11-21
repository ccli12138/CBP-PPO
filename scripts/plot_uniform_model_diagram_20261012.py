"""Redraw publication-scale vector diagrams for the uniform CBP-PPO model.

Author-authorized 2026-10-12 whole-manuscript update. The inherited rounded-box and
arrow helpers are retained. No training, numerical results or equations change.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path(os.environ.get('TMPDIR', '/tmp')) / 'cbp_mpl'))
os.environ.setdefault('MPLBACKEND', 'Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.path import Path as MplPath

ROOT = Path(__file__).resolve().parents[1]
PAPER = Path(os.environ['CBP_DRAW_ROOT']) / 'flat'
ARCHIVE = ROOT / 'reference/LMI-PPO_CSF_20260928/code/archive/experiments'
OUT = Path(os.environ['CBP_DRAW_ROOT']) / 'model_diagrams'
WIDTH = 390 / 72.27  # exactly the manuscript's full text width, inches
BASE = 10 * 72 / 72.27  # match a 10 TeX pt caption at insertion
INK = '#24364B'
MUTED = '#586779'
LINE = '#7C8A9A'
BLUE = '#416B94'
PURPLE = '#6A6597'
TEAL = '#2E7B78'
AMBER = '#A77A39'


def inherited_helper(path, name):
    source = path.read_text()
    node = next(n for n in ast.parse(source).body
                if isinstance(n, ast.FunctionDef) and n.name == name)
    return ast.get_source_segment(source, node)


_helpers = dict(FancyBboxPatch=FancyBboxPatch, FancyArrowPatch=FancyArrowPatch)
_source = ARCHIVE / 'draw_mechanism_flow_audited_20260905_v2.py'
for _name in ['add_box', 'add_arrow']:
    exec(inherited_helper(_source, _name), _helpers)


def canvas(height):
    fig = plt.figure(figsize=(WIDTH, height), facecolor='white')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, WIDTH), ylim=(0, height), aspect='equal')
    ax.axis('off')
    return fig, ax


def label(ax, x, y, text, size=BASE, color=INK, weight='normal', **kw):
    return ax.text(x, y, text, ha='center', va='center', fontsize=size,
                   color=color, fontweight=weight, zorder=5, **kw)


def box(ax, x, y, w, h, edge, fill, title=None, body=None,
        subtitle=None, title_y=0.76, body_y=0.42, subtitle_y=0.16):
    # Original shape helper, with author-approved restrained visual styling.
    _helpers['add_box'](ax, x, y, w, h, '', edge, fill, size=BASE)
    patch = ax.patches[-1]
    patch.set_linewidth(0.8)
    patch.set_boxstyle('round,pad=0.0,rounding_size=0.045')
    if title:
        label(ax, x+w/2, y+h*title_y, title, color=edge, weight='semibold')
    if body:
        label(ax, x+w/2, y+h*body_y, body)
    if subtitle:
        label(ax, x+w/2, y+h*subtitle_y, subtitle, color=MUTED,
              size=BASE*0.94)
    return patch


def arrow(ax, start, end, color=LINE):
    _helpers['add_arrow'](ax, start, end, color=color)
    patch = ax.patches[-1]
    patch.set_linewidth(0.85)
    patch.set_mutation_scale(8.4)
    patch.set_zorder(4)
    return patch


def elbow(ax, points, color=LINE):
    path = MplPath(points, [MplPath.MOVETO]+[MplPath.LINETO]*(len(points)-1))
    patch = FancyArrowPatch(path=path, arrowstyle='-|>', linewidth=0.85,
                            mutation_scale=8.4, color=color, zorder=4)
    ax.add_patch(patch)
    return patch


def save(fig, stem):
    # Keep the native page width; no tight-bbox rescaling or rasterized text.
    for suffix in ['pdf', 'svg', 'png']:
        options = dict(facecolor='white', pad_inches=0)
        if suffix == 'png':
            options['dpi'] = 320
        fig.savefig(OUT / f'{stem}.{suffix}', **options)
    plt.close(fig)


def architecture():
    fig, ax = canvas(1.58)
    # Observation, two genuinely shared feature layers, then two distinct heads.
    box(ax, .035, .55, .975, .48, BLUE, '#F1F5F9',
        'Input', title_y=.50)
    box(ax, 1.215, .37, 1.845, .97, '#D8D4E5', '#F8F7FB')
    label(ax, 2.1375, 1.16, 'Shared encoder', color=PURPLE,
          weight='semibold')
    box(ax, 1.34, .55, .70, .48, PURPLE, 'white', 'FC 32', title_y=.50)
    box(ax, 2.225, .55, .70, .48, PURPLE, 'white', 'FC 32', title_y=.50)
    box(ax, 3.40, 1.005, .765, .48, TEAL, '#EFF7F5', 'Actor', title_y=.50)
    box(ax, 3.40, .095, .765, .48, AMBER, '#FBF6EE', 'Critic', title_y=.50)
    box(ax, 4.40, 1.005, .955, .48, TEAL, 'white', 'Policy', title_y=.50)
    box(ax, 4.40, .095, .955, .48, AMBER, 'white', 'Baseline', title_y=.50)
    arrow(ax, (1.01, .79), (1.34, .79))
    arrow(ax, (2.04, .79), (2.225, .79))
    ax.plot([2.925,3.19],[.79,.79],color=LINE,linewidth=.85,zorder=4)
    elbow(ax, [(3.19,.79),(3.19,1.245),(3.40,1.245)], TEAL)
    elbow(ax, [(3.19,.79),(3.19,.335),(3.40,.335)], AMBER)
    arrow(ax, (4.165,1.245), (4.40,1.245), TEAL)
    arrow(ax, (4.165,.335), (4.40,.335), AMBER)
    save(fig, 'architecture')


def mechanism():
    fig, ax = canvas(2.00)
    xs = [.18, 1.515, 2.85, 4.185]
    w, h, top, bottom = 1.175, .48, 1.265, .53
    # Labels describe established accounting roles, not extra model states.
    for x, text, color in [(xs[0]+w/2,'Local state',BLUE),
                            (xs[1]+w/2,'Cooperator',PURPLE),
                            (xs[2]+w/2,'Allocation',TEAL),
                            (xs[3]+w/2,'Enforcement',AMBER)]:
        label(ax,x,1.89,text,color=color,weight='semibold')
    box(ax, xs[0],top,w,h,BLUE,'#F1F5F9','Earnings',title_y=.50)
    box(ax, xs[0],bottom,w,h,BLUE,'#F1F5F9','Defectors',title_y=.50)
    box(ax, xs[1],top,w,h,PURPLE,'#F5F3FA','Budget',title_y=.50)
    box(ax, xs[1],bottom,w,h,PURPLE,'#F5F3FA','Active groups',title_y=.50)
    box(ax, xs[2],top,w,h,TEAL,'#EFF7F5','Payment',title_y=.50)
    box(ax, xs[3],top,w,h,AMBER,'#FBF6EE','Group fund',title_y=.50)
    box(ax, xs[3],bottom,w,h,AMBER,'#FBF6EE','Defector fine',title_y=.50)
    box(ax, xs[2],bottom,w,h,TEAL,'#EFF7F5','Settlement',title_y=.50)
    for a,b in [(0,1),(1,2),(2,3)]:
        arrow(ax,(xs[a]+w,top+h/2),(xs[b],top+h/2))
    arrow(ax,(xs[0]+w,bottom+h/2),(xs[1],bottom+h/2))
    # Budget and active-group count converge in the actual normalized allocation.
    elbow(ax,[(xs[1]+w,bottom+h/2),(xs[2]-.08,bottom+h/2),
              (xs[2]-.08,top+.10),(xs[2],top+.10)],TEAL)
    arrow(ax,(xs[3]+w/2,top),(xs[3]+w/2,bottom+h),AMBER)
    arrow(ax,(xs[3],bottom+h/2),(xs[2]+w,bottom+h/2),AMBER)
    arrow(ax,(xs[2]+w/2,top),(xs[2]+w/2,bottom+h),TEAL)
    box(ax,xs[2],.035,xs[3]+w-xs[2],.335,TEAL,'#F8FBFA')
    label(ax,(xs[2]+xs[3]+w)/2,.20,'PPO reward',weight='semibold',color=TEAL)
    arrow(ax,(xs[2]+w/2,bottom),(xs[2]+w/2,.370),TEAL)
    # Original earnings enter settlement independently of the enforcement budget.
    # Route outside the lower state boxes to avoid crossings or implied junctions.
    elbow(ax,[(xs[0]+.165,top),(xs[0]+.165,1.14),(.035,1.14),(.035,.425),
              (xs[2]+.24,.425),(xs[2]+.24,bottom)],BLUE)
    save(fig, 'mechanism_flow')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    PAPER.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':BASE,
                         'mathtext.fontset':'dejavusans','pdf.fonttype':42,
                         'ps.fonttype':42,'svg.fonttype':'none',
                         'axes.unicode_minus':False})
    architecture()
    mechanism()
    assets = {}
    for source,target in [('architecture.pdf','fig01_actor_critic.pdf'),('mechanism_flow.pdf','fig02_mechanism_flow.pdf')]:
        (PAPER/target).write_bytes((OUT/source).read_bytes())
        assets[target]=hashlib.sha256((PAPER/target).read_bytes()).hexdigest()
    record = {
        'status':'drawn_pending_pdf_visual_review',
        'authorization':'workflow/author_review/FIG2_EARNINGS_INPUT_20261019/applied/review.json',
        'diagram_labels':'Core labels only in each box. No explanatory text, formulas or mathematical symbols; FC 32 is the inherited encoder size.',
        'latest_author_instruction':'workflow/author_review/diagrams_minimal_labels_20261008/instruction.json',
        'style':'Restrained vector cards, coherent semantic palette, shared-stage enclosure, aligned branches and chronological accounting; no shadows or data-like decoration.',
        'physical_width_inches':WIDTH,'insertion_fraction':1.0,
        'base_font_pdf_bp':BASE,'effective_font_tex_pt':BASE*72.27/72,
        'secondary_font_tex_pt':BASE*.94*72.27/72,
        'architecture':{'encoder':[32,32],'actor_outputs':2,'critic_outputs':1,
                         'shared_network':True,'critic_meaning':'learned baseline'},
        'flow':{'budget':'eta*s_i*positive(U_i)','allocation':'b_i/m_i for active groups if m_i>0, zero otherwise',
                'budget_and_active_group_count_parallel':True,
                'earnings_directly_enter_settlement':True,
                'left_rail_clearance_inches':0.145,
                'fines':'beta*F_g/d_g per defector for d_g>0, zero otherwise',
                'settlement':'U_i-S_i-h_i','reward':'settled payoff to common PPO'},
        'inherited_shape_source':str(_source.relative_to(ROOT)),
        'inherited_shape_source_sha256':hashlib.sha256(_source.read_bytes()).hexdigest(),
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'paper_assets_sha256':assets,'new_training_runs':0}
    (OUT/'drawing_provenance.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'status':record['status'],'assets':assets,'output':str(OUT)},indent=2))


if __name__=='__main__':
    main()
