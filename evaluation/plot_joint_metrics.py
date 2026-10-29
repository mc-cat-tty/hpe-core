"""
Visualize a per-joint metric (PCK or MPJPE) and its std dev on a 13-joint skeleton.

Each joint gets a filled circle sized after the metric and a lighter, symmetric
annulus spanning [metric - std, metric + std] around the circle edge.

 - MPJPE is expressed in pixels: scale is 1:1 (radius = error in px on a skeleton
   laid out in pixels, see --skeleton-height).
 - PCK is unitless, so sizes are autoscaled such that the largest (metric + std)
   gets --max-radius pixels. With --pck-scale area (default) the circle area is
   proportional to the value, which is how readers perceive size; with
   --pck-scale radius the radius is proportional instead (exaggerates differences).
   The band edges are the same mapping applied to metric -/+ std.

Metrics files are the ones produced by evaluate_hpe.py (tabulate format), e.g.
evaluations/MoveENet_EROS_first_run/{pck_0.2,pck_0.2_std,mpjpe,mpjpe_std}.txt

Example usage:
    python evaluations/plot_joint_metrics.py \
        --metrics evaluations/MoveENet_EROS_first_run/pck_0.2.txt \
        --std evaluations/MoveENet_EROS_first_run/pck_0.2_std.txt \
        --title "PCK@0.2 per joint" --pck --dark \
        --output evaluations/MoveENet_EROS_first_run/pck_0.2_skeleton.png

    python evaluations/plot_joint_metrics.py \
        --metrics evaluations/MoveENet_EROS_first_run/mpjpe.txt \
        --std evaluations/MoveENet_EROS_first_run/mpjpe_std.txt \
        --title "MPJPE per joint [px]" --mpjpe --color "#f4a6c0"
"""

import argparse
from pathlib import Path

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Wedge
from matplotlib.ticker import MultipleLocator

# Joint positions on a unit-height canvas: (x, y) with x in [-0.5, 0.5] and y in [0, 1], y pointing down
# (image convention). The subject faces the viewer, so the subject's right is on the viewer's left.
JOINTS = {
    'head':           (0.00, 0.06),
    'shoulder_right': (-0.14, 0.22),
    'shoulder_left':  (0.14, 0.22),
    'elbow_right':    (-0.22, 0.42),
    'elbow_left':     (0.22, 0.42),
    'wrist_right':    (-0.25, 0.60),
    'wrist_left':     (0.25, 0.60),
    'hip_right':      (-0.09, 0.58),
    'hip_left':       (0.09, 0.58),
    'knee_right':     (-0.10, 0.78),
    'knee_left':      (0.10, 0.78),
    'ankle_right':    (-0.11, 0.97),
    'ankle_left':     (0.11, 0.97),
}

BONES = [
    ('head', 'neck'),  # 'neck' is not a joint: it is the midpoint between the shoulders
    ('shoulder_right', 'shoulder_left'),
    ('shoulder_right', 'elbow_right'), ('elbow_right', 'wrist_right'),
    ('shoulder_left', 'elbow_left'), ('elbow_left', 'wrist_left'),
    ('shoulder_right', 'hip_right'), ('shoulder_left', 'hip_left'),
    ('hip_right', 'hip_left'),
    ('hip_right', 'knee_right'), ('knee_right', 'ankle_right'),
    ('hip_left', 'knee_left'), ('knee_left', 'ankle_left'),
]

DEFAULT_COLOR = '#8ecae6'  # pastel blue
LABEL_STYLE = dict(fontsize=8, fontweight='bold', zorder=5)
EDGE_WIDTH = 0.5  # [pt] outline of the mean circles


def read_metrics(path):
    """Parse a tabulate-style metrics file into {joint: value}; the 'avg' row is ignored."""
    values = {}
    for line in Path(path).read_text().splitlines():
        tokens = line.split()
        if len(tokens) < 2 or tokens[0] not in JOINTS:
            continue
        values[tokens[0]] = float(tokens[-1])

    missing = [j for j in JOINTS if j not in values]
    if missing:
        raise ValueError(f'{path}: missing joints {missing}')
    return values


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--metrics', required=True, help='metrics file (pck_*.txt or mpjpe.txt)')
    parser.add_argument('--std', default=None, help='optional metrics std dev file')
    parser.add_argument('--title', required=True, help='illustration title')
    parser.add_argument('--output', default=None,
                        help='output image path (default: <metrics file stem>_skeleton.png next to the metrics file)')
    parser.add_argument('--show', action='store_true', help='also open an interactive window')
    parser.add_argument('--dark', action='store_true', help='black background and white skeleton')
    parser.add_argument('--color', default=DEFAULT_COLOR,
                        help=f'color of the solid circles, any matplotlib color (default: {DEFAULT_COLOR}); '
                             'the std annulus uses the same color with lower opacity')
    parser.add_argument('--skeleton-height', type=float, default=260.,
                        help='skeleton height in pixels, i.e. the 1:1 reference for MPJPE (default: 260)')
    parser.add_argument('--max-radius', type=float, default=20.,
                        help='PCK only: pixel radius of the largest (metric + std) after autoscaling (default: 20)')
    parser.add_argument('--pck-scale', choices=['area', 'radius'], default='area',
                        help='PCK only: make circle area (default) or circle radius proportional to the value')
    parser.add_argument('--grid-step', type=float, default=5.,
                        help='background grid spacing in pixels, 0 to disable (default: 2)')
    parser.add_argument('--dpi', type=int, default=400)

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--pck', action='store_true', help='metrics are PCK values (autoscaled)')
    group.add_argument('--mpjpe', action='store_true', help='metrics are MPJPE values in pixels (1:1 scale)')
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.show:
        matplotlib.use('Agg')

    mean = read_metrics(args.metrics)
    std = read_metrics(args.std) if args.std else {j: 0. for j in JOINTS}

    # value -> radius [px]
    top = max(mean[j] + std[j] for j in JOINTS)
    if args.mpjpe:
        scale = 1.
        radius = lambda v: v
    elif args.pck_scale == 'area':
        scale = args.max_radius / top ** 0.5
        radius = lambda v: scale * v ** 0.5
    else:
        scale = args.max_radius / top
        radius = lambda v: scale * v

    bg, fg = ('black', 'white') if args.dark else ('white', '#3a3a3a')
    h = args.skeleton_height
    pos = {j: (x * h, y * h) for j, (x, y) in JOINTS.items()}

    pos['neck'] = tuple((a + b) / 2 for a, b in zip(pos['shoulder_right'], pos['shoulder_left']))
    grid_color = '#2b2b2b' if args.dark else '#e3e3e3'

    fig, ax = plt.subplots(figsize=(4.5, 6), facecolor=bg)
    ax.set_facecolor(bg)

    for a, b in BONES:
        ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]],
                color=fg, linewidth=1.8, solid_capstyle='round', zorder=1)

    rgb = mcolors.to_rgb(args.color)
    edge = tuple(0.5 * c + 0.5 for c in rgb)  # lighter tint of the circle color
    for j in JOINTS:
        r = radius(mean[j])
        r_in, r_out = radius(max(mean[j] - std[j], 0.)), radius(mean[j] + std[j])
        if r_out > r_in:
            ax.add_patch(Wedge(pos[j], r_out, 0, 360, width=r_out - r_in, facecolor=rgb, alpha=0.35, linewidth=0,
                               zorder=2))
        # slightly transparent so the inner half of the band stays visible through the circle
        ax.add_patch(Circle(pos[j], r, facecolor=rgb, alpha=0.8, edgecolor=edge, linewidth=EDGE_WIDTH, zorder=3))
        ax.plot(*pos[j], marker='o', markersize=2.5, color='black', zorder=4)

        # value label on the outer side of the joint, same color as the circles
        value = f'{mean[j]:.2f}' if args.pck else f'{mean[j]:.1f}'
        x, y = pos[j]
        gap = 0.012 * h
        side = 1 if x > 0 else -1
        if x == 0:  # head
            ax.text(x, y - r_out - gap, value, color=rgb, ha='center', va='bottom', **LABEL_STYLE)
        elif j.startswith('hip'):  # outer side is taken by the wrist circles
            ax.text(x - side * gap, y - r_out - 0.5 * gap, value, color=rgb, ha='left' if -side > 0 else 'right',
                    va='bottom', **LABEL_STYLE)
        else:
            ax.text(x + side * (r_out + gap), y, value, color=rgb, ha='left' if side > 0 else 'right',
                    va='center', **LABEL_STYLE)

    # legend: handles use the same color/opacity as the drawn patches
    handles = [Line2D([], [], marker='o', linestyle='', markersize=9, markerfacecolor=rgb,
                      markeredgecolor=edge, markeredgewidth=EDGE_WIDTH, label='mean')]
    if args.std:
        handles.append(Line2D([], [], marker='o', linestyle='', markersize=12, markerfacecolor=rgb,
                              markeredgecolor='none', alpha=0.35, label='± std dev'))
    if args.grid_step > 0 and args.mpjpe:
        handles.append(Line2D([], [], color=grid_color, linewidth=1.5, label=f'grid: {args.grid_step:g} px'))
    legend = ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.03), ncol=len(handles),
                       frameon=False)
    for text in legend.get_texts():
        text.set_color(fg)

    if args.mpjpe:
        unit_note = 'radius in pixels (1:1 scale)'
    elif args.pck_scale == 'area':
        unit_note = 'area proportional to PCK (autoscaled)'
    else:
        unit_note = f'radius proportional to PCK (autoscaled, 1 unit = {scale:.1f} px)'
    ax.text(0.5, -0.115, unit_note, transform=ax.transAxes, ha='center', va='top', color=fg, fontsize=8, alpha=0.7)

    margin = (args.max_radius if args.pck else top) + 0.02 * h
    half_width = max(abs(x) for x, _ in pos.values())
    label_room = 0.1 * h  # space for the value labels beside the outermost joints
    ax.set_xlim(-half_width - margin - label_room, half_width + margin + label_room)
    ax.set_ylim(max(y for _, y in pos.values()) + margin, min(y for _, y in pos.values()) - margin - 0.04 * h)  # y down

    # the subject faces the viewer: subject's right is on the viewer's left
    for text, x, ha in (('R', 0.02, 'left'), ('L', 0.98, 'right')):
        ax.text(x, 0.99, text, transform=ax.transAxes, ha=ha, va='top', color=fg, fontsize=16, fontweight='bold',
                alpha=0.8)
    ax.set_aspect('equal')
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0, labelbottom=False, labelleft=False)
    if args.grid_step > 0:
        ax.xaxis.set_major_locator(MultipleLocator(args.grid_step))
        ax.yaxis.set_major_locator(MultipleLocator(args.grid_step))
        ax.set_axisbelow(True)
        ax.grid(True, color=grid_color, linewidth=0.3)
    ax.set_title(args.title, color=fg, fontsize=14, pad=10)

    out = Path(args.output) if args.output else Path(args.metrics).with_name(Path(args.metrics).stem + '_skeleton.png')
    fig.savefig(out, dpi=args.dpi, facecolor=bg, bbox_inches='tight')
    print(f'saved {out}')
    if args.show:
        plt.show()


if __name__ == '__main__':
    main()
