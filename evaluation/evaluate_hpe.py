import argparse
import io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import os
import seaborn as sns
import math
from tqdm import tqdm

from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stdout
from functools import partial
from pathlib import Path
from scipy import interpolate
from tabulate import tabulate
from typing import Optional

from datasets.utils import constants as ds_constants, parsing as ds_parsing
from evaluation.utils import metrics as metrics_utils, plots as plots_utils
from evaluation.utils.plots import plot_poses


# TODO:
#   - compute oks
#     - for a set of thresholds
#     - for each joint
#     - global
#   - randomly sample good and bad predictions and plot joints

# class PredictionsPlot:
#
#     def __init__(self, output_folder_path, timestamps, joints_gt):
#
#         assert 1 <= joints_gt.shape[2] <= 3, 'coordinates must be either 2D or 3D'
#
#         self.output_folder_path = output_folder_path
#         self.timestamps = timestamps
#         self.joints_gt = joints_gt
#
#         # iterate on each joint
#         for joint_key, joint_ind in ds_constants.HPECoreSkeleton.KEYPOINTS_MAP.items():
#
#             for coord_ind in range(joints_gt.shape[2]):
#
#                 if coord_ind == 0:
#                     lbl_coord = 'x'
#                 if coord_ind == 1:
#                     lbl_coord = 'y'
#                 if coord_ind == 2:
#                     lbl_coord = 'z'
#
#                 lbl_legend_gt = lbl_coord + r'$_{GT}$'
#                 lbl_legend_pred = lbl_coord + r'$_{predicted}$'
#                 lbl_x_axis = 'time [sec]'
#                 lbl_y_axis = f'{lbl_coord} coordinate [px]'
#                 lbl_fig = f'Joint \'{joint_key}\', {lbl_coord} coordinate'
#
#                 # create plot
#                 my_dpi = 96
#                 fig = plt.figure(figsize=(2048 / my_dpi, 900 / my_dpi), dpi=96)
#                 ax = fig.add_subplot(111)
#                 fig.tight_layout(pad=5)
#
#                 # plot ground-truth
#                 coord_gt = joints_gt[:, joint_ind, coord_ind]
#                 ax.plot(timestamps, coord_gt, color='tab:green', alpha=0.3, label=lbl_legend_gt)
#
#                 y_lim_min = math.inf
#                 y_lim_max = 0
#                 for predictions_algo in joints_predicted:
#                     # plot predictions
#                     coord_pred = predictions_algo[:, joint_ind, coord_ind]
#                     ax.plot(timestamps, coord_pred, color=np.random.rand(3, ), marker=".", label=lbl_legend_pred,
#                             linestyle='None', alpha=1.0)
#
#                     y_lim_min = min(y_lim_min, coord_pred)
#                     y_lim_max = max(y_lim_max, coord_pred)
#
#                 # set axis limits
#                 ax.set_xlim([timestamps[0], timestamps[-1]])
#                 ax.set_ylim([y_lim_min * 0.6, y_lim_max * 1.4])
#
#                 # labels and title
#                 plt.xlabel(lbl_x_axis, fontsize=22, labelpad=5)
#                 plt.ylabel(lbl_y_axis, fontsize=22, labelpad=5)
#                 fig.suptitle(lbl_fig, fontsize=28, y=0.97)
#                 plt.tick_params(axis='both', which='major', labelsize=18)
#                 ax.legend(fontsize=16, loc='upper right')


def plot_pck_over_thresholds(pck_thresholds_results, output_folder_path, ds_name=None, pck_thresholds_std=None):
    """
    Plots the average PCK of each algorithm over the PCK thresholds; if pck_thresholds_std is given (std over sequences
    of the average PCK, indexed like pck_thresholds_results), a +-1 std band is drawn around each line
    """
    # sort thresholds
    thresholds = sorted(pck_thresholds_results.keys())

    # collect results as long-form data, as expected by seaborn
    rows = list()
    for th in thresholds:
        for (algo_name, metric) in pck_thresholds_results[th].items():
            std = pck_thresholds_std[th][algo_name].get_value()[1] * 100 if pck_thresholds_std else .0
            rows.append({'algo': plots_utils.RENAMING.get(algo_name, algo_name), 'algo_key': algo_name,
                         'threshold': th, 'pck': metric.get_value()[1] * 100, 'std': std})
    data = pd.DataFrame(rows)

    # colors and line styles of each algorithm, moveEnet is always drawn with a solid line
    names = list(dict.fromkeys(data['algo']))
    colors = dict()
    dashes = dict()
    # copy, so that popping colors does not consume the module-level list across calls
    extra_colors = list(plots_utils.OTHER_COLORS)
    other_dashes = [(4, 2), (1, 2), (4, 2, 1, 2), (6, 2, 1, 2, 1, 2)]
    for name, algo_name in data[['algo', 'algo_key']].drop_duplicates().itertuples(index=False):
        try:
            colors[name] = plots_utils.ALGO_COLORS[algo_name]
        except KeyError:
            if not extra_colors:
                extra_colors = list(plots_utils.OTHER_COLORS)
            colors[name] = extra_colors.pop()
        dashes[name] = '' if name == 'moveEnet' else (other_dashes.pop(0) if other_dashes else (4, 2))

    # setup the figure
    my_dpi = 96
    if ds_name:
        figure_name = f'Dataset {ds_name}, Average PCK'
    else:
        figure_name = f'Global Average PCK'
    with sns.axes_style('whitegrid'):
        fig = plt.figure(num=figure_name,
                         figsize=(2048 / my_dpi, 900 / my_dpi),
                         dpi=96)
        ax = fig.add_subplot(111)
    fig.tight_layout(pad=10)

    # plot pcks
    sns.lineplot(data=data, x='threshold', y='pck', hue='algo', style='algo', hue_order=names, style_order=names,
                 palette=colors, dashes=dashes, markers=['o'] * len(names), markersize=10, linewidth=4, ax=ax)

    # plot std bands
    if pck_thresholds_std:
        for name, algo_data in data.groupby('algo', sort=False):
            ax.fill_between(algo_data['threshold'], np.clip(algo_data['pck'] - algo_data['std'], 0, 100),
                            np.clip(algo_data['pck'] + algo_data['std'], 0, 100), color=colors[name], alpha=0.15,
                            linewidth=0)

    # one tick for each requested threshold
    ax.set_xticks(thresholds)
    ax.set_xticklabels([f'{th:g}' for th in thresholds], rotation=45 if len(thresholds) > 10 else 0)

    # set axis limits
    # small margin, so that the markers of the first and last thresholds are not clipped
    x_margin = 0.02 * (thresholds[-1] - thresholds[0]) or 0.05
    ax.set_xlim([thresholds[0] - x_margin, thresholds[-1] + x_margin])
    ax.set_ylim([0, 100])

    # labels and title
    ax.set_xlabel('PCK thresholds', fontsize=32)
    ax.set_ylabel(f'Average PCK %', fontsize=32, labelpad=5)
    # fig.suptitle(figure_name, fontsize=30, y=0.97)
    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.legend(fontsize=24, loc='lower right', title='± 1 std over sequences' if pck_thresholds_std else None,
              title_fontsize=20)

    # save plot
    if ds_name:
        fig_path = output_folder_path / f'{ds_name}_pck.png'
    else:
        fig_path = output_folder_path / f'global_pck.png'
    plt.savefig(str(fig_path.resolve()))


def plot_boxplot(algo_metrics: dict, descr: Optional[str], file_path: Path) -> None:
    # create figure
    fig, ax = plt.subplots()
    fig.set_size_inches(4,3)
    all_values = list()
    tick_labels = [' ']
    for (algo_name, metric) in sorted(algo_metrics.items(), reverse=True):
        values = metric.get_value()
        joints_metric_values = values[0]

        if isinstance(metric, metrics_utils.PCK):
            all_values.append(joints_metric_values)

            tick_labels.append(algo_name)
            ax.set_xlabel('Joints PCK')

        elif isinstance(metric, metrics_utils.RMSE):
            # all_values.append(joints_metric_values[::2])  # add metric values for x coordinates
            # all_values.append(joints_metric_values[1::2])  # add metric values for y coordinates
            all_values.append(np.maximum(joints_metric_values[::2], joints_metric_values[1::2]))

            tick_labels.extend([f'{algo_name}'])
            ax.set_xlabel('Joints RMSE')

        elif isinstance(metric, metrics_utils.MPJPE):
            all_values.append(joints_metric_values)

            tick_labels.extend([f'{algo_name}'])
            ax.set_xlabel('Joints MPJPE')
            
    # plot values
    y_ticks = np.arange(len(tick_labels))
    meanlineprops = dict(linestyle='-', linewidth=2.0, color='tab:blue')
    ax.boxplot(all_values, vert=False, showfliers=False, showmeans=True, meanline=True, meanprops=meanlineprops)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(tick_labels)
    ax.set_title(descr)

    # Save the figure and show
    plt.tight_layout()
    plt.savefig(str(file_path.resolve()))


def plot_predictions(output_folder_path, ds_name, timestamps, joints_gt, algo_names, timestamps_predicted,
                     joints_predicted):
    assert 1 <= joints_gt.shape[2] <= 3, 'coordinates must be either 2D or 3D'

    # define a color for each algorithm and each coordinate
    # algo_colors = [[np.random.rand(3,) for _ in range(joints_gt.shape[2])] for _ in algo_names] # random colors
    algo_colors = [['tab:green', 'tab:orange'], ['blue', 'red'], ['violet', 'chocolate'], ['purple', 'sienna'],
                   ['lime', 'gold'], ['aqua', 'olive'], ['tab:blue', 'green'], ['gold', 'lime']]  # fixed colors

    # iterate on each joint
    for joint_key, joint_ind in ds_constants.HPECoreSkeleton.KEYPOINTS_MAP.items():
        # create plot
        my_dpi = 96
        fig = plt.figure(num=f'Dataset {ds_name}, Joint \'{joint_key}\'',
                         figsize=(2048 / my_dpi, 900 / my_dpi),
                         dpi=96)
        ax = fig.add_subplot(111)
        fig.tight_layout(pad=5)

        y_lim_min = np.inf
        y_lim_max = 0

        for coord_ind in range(joints_gt.shape[2]):

            # plot ground-truth
            coord_gt = joints_gt[:, joint_ind, coord_ind]
            if coord_ind == 0:
                lbl_coord = 'x'
                ax.plot(timestamps, coord_gt, color='tab:blue', alpha=0.2, label=f'GT {lbl_coord}')
            if coord_ind == 1:
                lbl_coord = 'y'
                ax.plot(timestamps, coord_gt, color='tab:orange', alpha=0.2, label=f'GT {lbl_coord}')
            if coord_ind == 2:
                lbl_coord = 'z'
                ax.plot(timestamps, coord_gt, color='tab:green', alpha=0.2, label=f'GT {lbl_coord}')

            y_lim_min = min(y_lim_min, min(coord_gt))
            y_lim_max = max(y_lim_max, max(coord_gt))

            for pi, predictions_algo in enumerate(joints_predicted):
                # plot predictions
                coord_pred = predictions_algo[:, joint_ind, coord_ind]
                ax.plot(timestamps_predicted[pi], coord_pred, color=algo_colors[pi][coord_ind], marker=".",
                        label=f'{algo_names[pi][:]} {lbl_coord}', linestyle='-', alpha=1.0, markersize=12)
                y_lim_min = min(y_lim_min, min(coord_pred))
                y_lim_max = max(y_lim_max, max(coord_pred))

            # set axis limits
            ax.set_xlim([timestamps[0], timestamps[-1]])
            ax.set_ylim([y_lim_min * 0.6, y_lim_max * 1.4])

            # labels and title
            plt.xlabel('time [sec]', fontsize=22, labelpad=5)
            plt.ylabel('coordinates [px]', fontsize=22, labelpad=5)
            fig.suptitle(f'dataset {ds_name}, joint \'{joint_key}\' coordinates', fontsize=28, y=0.97)
            plt.tick_params(axis='both', which='major', labelsize=18)
            ax.legend(fontsize=16, loc='upper right')

            # save plot
            fig_path = output_folder_path / f'{ds_name}_{joint_key}_predictions.png'
            # plt.savefig(str(fig_path.resolve()))


def plot_predictions_all_joints(output_folder_path, ds_name, timestamps, joints_gt, algo_names, timestamps_predicted,
                                joints_predicted):
    assert 1 <= joints_gt.shape[2] <= 3, 'coordinates must be either 2D or 3D'

    for pi, predictions_algo in enumerate(joints_predicted):
        # create plot
        my_dpi = 96
        fig = plt.figure(num=f'Dataset {ds_name} - all joints - Algorithm: {algo_names[pi]}',
                         figsize=(2048 / my_dpi, 900 / my_dpi),
                         dpi=96)
        ax = fig.add_subplot(111)
        fig.tight_layout(pad=5)

        y_lim_min = np.inf
        y_lim_max = 0

        # iterate on each joint
        for joint_key, joint_ind in ds_constants.HPECoreSkeleton.KEYPOINTS_MAP.items():
            for coord_ind in range(joints_gt.shape[2]):

                # plot ground-truth
                coord_gt = joints_gt[:, joint_ind, coord_ind]
                if coord_ind == 0:
                    lbl_coord = 'x'
                    if (joint_ind == 0):
                        ax.plot(timestamps, coord_gt, color='tab:green', alpha=0.8, label=f'GT {lbl_coord}')
                    ax.plot(timestamps, coord_gt, color='tab:green', alpha=0.8)
                if coord_ind == 1:
                    lbl_coord = 'y'
                    if (joint_ind == 0):
                        ax.plot(timestamps, coord_gt, color='red', alpha=0.8, label=f'GT {lbl_coord}')
                    ax.plot(timestamps, coord_gt, color='red', alpha=0.8)
                if coord_ind == 2:
                    lbl_coord = 'z'
                    ax.plot(timestamps, coord_gt, color='tab:green', alpha=0.2, label=f'GT {lbl_coord}')

                y_lim_min = min(y_lim_min, min(coord_gt))
                y_lim_max = max(y_lim_max, max(coord_gt))

                # plot predictions
                coord_pred = predictions_algo[:, joint_ind, coord_ind]
                if (coord_ind == 0):  # x component
                    if (joint_ind == 0):
                        ax.plot(timestamps_predicted[pi], coord_pred, color='tab:blue', marker=".", linestyle='-', alpha=1.0,
                                label=f'Tracked {lbl_coord}')
                    ax.plot(timestamps_predicted[pi], coord_pred, color='tab:blue', marker=".", linestyle='-', alpha=1.0)
                if (coord_ind == 1):  # y component
                    if (joint_ind == 0):
                        ax.plot(timestamps_predicted[pi], coord_pred, color='tab:orange', marker=".", linestyle='-', alpha=1.0,
                                label=f'Tracked {lbl_coord}')
                    ax.plot(timestamps_predicted[pi], coord_pred, color='tab:orange', marker=".", linestyle='-', alpha=1.0)

                y_lim_min = min(y_lim_min, min(coord_pred))
                y_lim_max = max(y_lim_max, max(coord_pred))

        # set axis limits
        ax.set_xlim([timestamps[0], timestamps[-1]])
        ax.set_ylim([y_lim_min * 0.6, y_lim_max * 1.4])

        # labels and title
        plt.xlabel('time [sec]', fontsize=22, labelpad=5)
        plt.ylabel('coordinates [px]', fontsize=22, labelpad=5)
        fig.suptitle(f'Dataset {ds_name} - all joints - Algorithm: {algo_names[pi]}', fontsize=28, y=0.97)
        plt.tick_params(axis='both', which='major', labelsize=18)
        ax.legend(fontsize=16, loc='upper right')

        # save plot
        fig_path = output_folder_path / f'{ds_name}_{joint_key}_predictions.png'
        # plt.savefig(str(fig_path.resolve()))


def plot_latency(output_folder_path, ds_name, timestamps, algo_names, latencies):
    # define a color for each algorithm and each coordinate
    # algo_colors = [[np.random.rand(3,) for _ in range(joints_gt.shape[2])] for _ in algo_names] # random colors
    algo_colors = ['tab:green', 'tab:orange', 'blue', 'red', 'violet', 'chocolate', 'purple', 'sienna', 'lime', 'gold',
                   'aqua', 'olive']  # fixed colors

    # create plot
    my_dpi = 96
    fig = plt.figure(num=f'Latencies: Dataset {ds_name}',
                     figsize=(2048 / my_dpi, 900 / my_dpi),
                     dpi=96)
    ax = fig.add_subplot(111)
    fig.tight_layout(pad=5)

    y_lim_min = np.inf
    y_lim_max = 0

    for pi, latency in enumerate(latencies):
        # plot predictions
        coord_pred = latency
        t_axis = timestamps[pi]
        # ax.plot(timestamps[pi], coord_pred, color=algo_colors[pi], marker="None", label=f'{algo_names[pi]}', linestyle='-', alpha=1.0)
        ax.plot(timestamps[pi], coord_pred, marker="None", label=f'{algo_names[pi]}', linestyle='-', alpha=1.0)

        y_lim_min = min(y_lim_min, min(latency))
        y_lim_max = max(y_lim_max, max(latency))
    # ax.plot(timestamps, latencies, marker=".", linestyle='-', alpha=1.0)

    # set axis limits
    ax.set_xlim([min([min(r) for r in timestamps]), max([max(r) for r in timestamps])])
    ax.set_ylim([y_lim_min * 0.6, y_lim_max * 1.4])

    # labels and title
    plt.xlabel('time [sec]', fontsize=22, labelpad=5)
    plt.ylabel('Latency of detection [msec]', fontsize=22, labelpad=5)
    fig.suptitle(f'Latencies for dataset {ds_name}', fontsize=28, y=0.97)
    plt.tick_params(axis='both', which='major', labelsize=18)
    ax.legend(fontsize=16, loc='upper right')

    # save plot
    # fig_path = output_folder_path / f'{ds_name}_{joint_key}_predictions.png'
    # plt.savefig(str(fig_path.resolve()))


def tabulate_metric_over_algorithms(algo_metrics: dict, headers: list, descr: Optional[str], to_latex: bool = False,
                                    file_path: Optional[Path] = None, file_stem: str = None) -> None:
    # create results table
    table = list()
    header = list()
    header.append("Joint")
    table.append(headers)
    for (algo_name, metric) in sorted(algo_metrics.items()):
        values = metric.get_value()
        if (len(values) > 2):
            joints_values = np.around(values[2], decimals=2)
        else:
            joints_values = np.around(values[0], decimals=2)
        # joints_values = values[0]
        # avg_value = np.max(values[1])
        if isinstance(metric, metrics_utils.SequencesWeightedStd):
            # the std of the average over joints is not the average of the stds of the joints
            avg_value = np.around(values[1], decimals=2)
        else:
            avg_value = np.around(np.mean(joints_values), decimals=2)
        # joints_values = values[0] # this mean matches the mean in the data in boxplot (green line), not the median (orange line)
        table_row = joints_values.tolist()
        if isinstance(avg_value, np.ndarray):
            table_row.extend(avg_value.tolist())
        else:
            table_row.append(avg_value)
        table.append(table_row)
        header.append(algo_name)
    # transpose list of lists
    table = list(map(list, zip(*table)))

    # set table format
    fmt = 'simple'
    ext = '.txt'
    if to_latex:
        fmt = 'latex'
        ext = '.tex'

    if file_path and file_stem:

        full_path = file_path / (file_stem + ext)
        with open(str(full_path.resolve()), 'w') as f:

            # create latex string
            if to_latex:
                file_content = ''
                file_content += r"""\documentclass[varwidth]{standalone}
\usepackage[utf8]{inputenc}
\begin{document}
\begin{table}
\centering"""
                file_content += tabulate(table, headers=header, tablefmt=fmt)
                if descr:
                    descr_tex = descr.replace('_', '\_')
                    file_content += r'\caption{' \
                                    f'{descr_tex}' \
                                    r'}'
                file_content += r"""
\end{table}
\end{document}"""

            else:
                file_content = tabulate(table, headers=header, tablefmt=fmt)
                print(file_content)

            f.write(file_content)

    else:
        if descr:
            print(descr)
        print(tabulate(table, headers=header, tablefmt=fmt))


def tabulate_latency_over_algorithms(algo_metrics: dict, headers: list, descr: Optional[str], to_latex: bool = False,
                                     file_path: Optional[Path] = None, file_stem: str = None) -> None:
    # create results table
    table = list()
    header = list()
    header.append("")
    table.append(headers)
    for (algo_name, metric) in sorted(algo_metrics.items()):
        table_row = []
        table_row.append(metric.tolist())
        table.append(table_row)
        header.append(algo_name)
    # transpose list of lists
    table = list(map(list, zip(*table)))

    # set table format
    fmt = 'simple'
    ext = '.txt'
    if to_latex:
        fmt = 'latex'
        ext = '.tex'

    if file_path and file_stem:

        full_path = file_path / (file_stem + ext)
        with open(str(full_path.resolve()), 'w') as f:

            # create latex string
            if to_latex:
                file_content = ''
                file_content += r"""\documentclass[varwidth]{standalone}
\usepackage[utf8]{inputenc}
\begin{document}
\begin{table}
\centering"""
                file_content += tabulate(table, headers=header, tablefmt=fmt)
                if descr:
                    descr_tex = descr.replace('_', '\_')
                    file_content += r'\caption{' \
                                    f'{descr_tex}' \
                                    r'}'
                file_content += r"""
\end{table}
\end{document}"""

            else:
                file_content = tabulate(table, headers=header, tablefmt=fmt)
                print(file_content)

            f.write(file_content)

    else:
        if descr:
            print(descr)
        print(tabulate(table, headers=header, tablefmt=fmt))

def create_pred_GT_pairing(predictions_folder,datasets_path,multi_channel):
    data_samples = {}
    pred_samples = os.listdir(predictions_folder)
    if multi_channel:
        for subsample in pred_samples:
            sample = '_'.join(subsample.split('_')[1:])
            channel = re.findall('[0-9]+', subsample.split('_')[0])[0]
            for folder in (datasets_path / sample).iterdir():
                if channel in folder.name and 'skeleton' in folder.name:
                    data_samples[subsample] = datasets_path / folder / 'data.log'
                    continue
    else:
        for sample in pred_samples:
            yarp_path_dir = datasets_path / sample
            yarp_path = [x for x in yarp_path_dir.iterdir() if 'skeleton' in x.name][0] / 'data.log'
            data_samples[sample] = yarp_path
    return data_samples


def report_dataset(ds_name: str, metrics: dict, output_folder_path: Path, to_latex: bool) -> str:
    """
    Saves tables and plots of the metrics computed for a single dataset.
    Return:
        str: the text of the report, which is captured instead of being printed, so that reports created in
        separate processes can be printed in order
    """
    open_figures = set(plt.get_fignums())

    with io.StringIO() as report, redirect_stdout(report):
        if (not (to_latex)):
            print("\n==================================================")
            print("Dataset: ", ds_name)
            print("==================================================")
        else:
            print("Dataset ", ds_name, " ✓")

        output_ds_folder_path = output_folder_path / ds_name
        output_ds_folder_path.mkdir(parents=True, exist_ok=True)

        for (metric_name, metric_results) in metrics.items():
            if metric_name == 'pck':

                plot_pck_over_thresholds(metric_results, output_ds_folder_path, ds_name)

                # create table
                header = [key for key in ds_constants.HPECoreSkeleton.KEYPOINTS_MAP.keys()]
                header.append('avg PCK')

                for (th, algos) in metric_results.items():
                    if (not (to_latex)):
                        print("\n-= PCK threshold = ", th, " =-")
                    tabulate_metric_over_algorithms(algos, header,
                                                    descr=f'PCK results for dataset {ds_name}, threshold {th}',
                                                    to_latex=to_latex,
                                                    file_path=output_ds_folder_path,
                                                    file_stem=f'pck_{th}_{ds_name}')

            elif metric_name == 'rmse':
                if (not (to_latex)):
                    print("\n-= RMSE =-")
                # header = [header_str for key in ds_constants.HPECoreSkeleton.KEYPOINTS_MAP.keys() for header_str in [f'{key} x', f'{key} y']]
                # header.append('avg RMSE x')
                # header.append('avg RMSE y')
                header = [header_str for key in ds_constants.HPECoreSkeleton.KEYPOINTS_MAP.keys() for header_str in
                          [f'{key}']]
                header.append('avg RMSE')

                tabulate_metric_over_algorithms(metric_results, header,
                                                descr=f'RMSE results for dataset {ds_name}',
                                                to_latex=to_latex,
                                                file_path=output_ds_folder_path,
                                                file_stem=f'rmse_{ds_name}')

                plot_boxplot(metric_results,
                             descr=f'RMSE results for dataset {ds_name}',
                             file_path=output_ds_folder_path / f'rmse_{ds_name}.png')

            if metric_name == 'latency':
                print("\n-= LATENCY =-")
                header = ['avg latency [ms]']
                tabulate_latency_over_algorithms(metric_results, header,
                                                 descr=f'Latency results for dataset {ds_name}',
                                                 to_latex=to_latex,
                                                 file_path=output_ds_folder_path,
                                                 file_stem=f'latency_{ds_name}')

            if metric_name == 'mpjpe':
                print("\n-= MPJPE =-")
                header = [header_str for key in ds_constants.HPECoreSkeleton.KEYPOINTS_MAP.keys() for header_str in
                          [f'{key}']]
                header.append('MPJPE')

                tabulate_metric_over_algorithms(metric_results, header,
                                                descr=f'MPJPE results for dataset {ds_name}',
                                                to_latex=to_latex,
                                                file_path=output_ds_folder_path,
                                                file_stem=f'mpjpe_{ds_name}')
                plot_boxplot(metric_results,
                             descr=f'MPJPE results for dataset {ds_name}',
                             file_path=output_ds_folder_path / f'mpjpe_{ds_name}.png')

            if metric_name == 'missed':
                print("\n-= MISSED DETECTIONS [%] =-")
                header = [key for key in ds_constants.HPECoreSkeleton.KEYPOINTS_MAP.keys()]
                header.append('avg missed [%]')

                tabulate_metric_over_algorithms(metric_results, header,
                                                descr=f'Missed detections for dataset {ds_name}',
                                                to_latex=to_latex,
                                                file_path=output_ds_folder_path,
                                                file_stem=f'missed_{ds_name}')

        report_text = report.getvalue()

    # figures are saved to file; closing them avoids accumulating hundreds of open figures
    for fig_num in set(plt.get_fignums()) - open_figures:
        plt.close(fig_num)

    return report_text


def evaluate_sample(sample, yarp_path, predictions_folder, output_folder_path, args):
    """
    Evaluates the predictions of a single sample against its GT. Samples are independent of each other, so that they
    can be evaluated in separate processes.
    Return:
        None if the sample is skipped, otherwise a dict with the dataset metrics, the evaluated samples used for
        updating the global metrics and the data used for plotting
    """
    predictions_path = predictions_folder / sample
    # yarp_path_dir = datasets_path / sample ## This should have the next level folder list as well.

    if yarp_path.exists() == False:
        print('\x1b[1;33;20m' + "Skipping " + str(sample) + " as no GT available at " + str(yarp_path) + '\x1b[0m')
        return None

    # yarp_path = [x for x in yarp_path_dir.iterdir() if 'skeleton' in x.name][0] / 'data.log'

    # ground truth in yarp format is supposed to be stored in folders name <dataset_name>/ch<channel_id>[frequency_info]skeleton
    # find the channel id
    numbers = re.findall('[0-9]+', yarp_path.parent.name)
    channel_id = numbers[0]
    parent_folder_prefix = yarp_path.parent.name.split(channel_id)[0]
    channel_folder = f'{parent_folder_prefix}{channel_id}'
    dataset_name = yarp_path.parent.parent.name
    results_key = f'{dataset_name}_{channel_folder}'

    # predictions_path = Path(args.predictions_path) / dataset_name
    predictions_file_path = list(predictions_path.glob('**/*.csv'))
    # predictions_file_path.sort()
    if len(predictions_file_path) == 0:
        print('\x1b[1;33;20m' + "Skipping " + str(dataset_name) + " as no results exist in" + str(
            predictions_path) + '\x1b[0m')
        return None

    predictions_file_path.sort(reverse=True)
    data = ds_parsing.import_yarp_skeleton_data(yarp_path, args.multi_channel)

    ts_gt = np.concatenate(([.0], data['ts'], [data['ts'][-1] + 1]))

    # TODO: use numpy.interp
    # TODO: make it GeneratorExit()eral for 2d/3d coordinates

    # interpolate ground truth joints so that they can be compared with the high frequency predictions
    for k_map in ds_constants.HPECoreSkeleton.KEYPOINTS_MAP.items():
        x_interpolation = interpolate.interp1d(ts_gt, np.concatenate(
            ([data[k_map[0]][0, 0]], data[k_map[0]][:, 0], [data[k_map[0]][-1, 0]])))  # , fill_value="extrapolate"
        y_interpolation = interpolate.interp1d(ts_gt, np.concatenate(
            ([data[k_map[0]][0, 1]], data[k_map[0]][:, 1], [data[k_map[0]][-1, 1]])))
        data[k_map[0]] = dict()
        data[k_map[0]]['x'] = x_interpolation
        data[k_map[0]]['y'] = y_interpolation

    # GT contains the size of the torso
    if data['head_sizes'][0] == -1:
        pck_sizes_gt_interp = interpolate.interp1d(ts_gt, np.concatenate(
            ([data['torso_sizes'][0]], data['torso_sizes'], [data['torso_sizes'][-1]])))
    # GT contains the size of the head
    else:
        pck_sizes_gt_interp = interpolate.interp1d(ts_gt, np.concatenate(
            ([data['head_sizes'][0]], data['head_sizes'], [data['head_sizes'][-1]])))

    output_ds_folder_path = output_folder_path / results_key
    output_ds_folder_path.mkdir(parents=True, exist_ok=True)

    ds_results = dict()
    # evaluated samples of each algorithm, used by the main process for updating the global metrics
    global_samples = list()

    # GT samples within the window used to filter the predictions; the GT is never interpolated
    t_window = data['ts'][np.logical_and(data['ts'] > 0.1 * ts_gt[-1], data['ts'] < ts_gt[-1])]
    skeletons_gt_window = np.zeros((len(t_window), len(ds_constants.HPECoreSkeleton.KEYPOINTS_MAP), 2))
    for k_map in ds_constants.HPECoreSkeleton.KEYPOINTS_MAP.items():
        skeletons_gt_window[:, k_map[1], 0] = data[k_map[0]]['x'](t_window)
        skeletons_gt_window[:, k_map[1], 1] = data[k_map[0]]['y'](t_window)

    algorithm_names = []
    skeletons_predictions = []
    skeletons_predictions_eval = []
    timestamps_eval = []
    timestamps = []
    latency = []

    # parse predictions
    for pred_path in predictions_file_path:

        algo_name = pred_path.stem
        if algo_name in args.exclude: # exclude algos as requested in command line.
            continue
        try:
            predictions_old = np.loadtxt(str(pred_path.resolve()), dtype=float)
        except ValueError:
            with open(str(pred_path.resolve())) as f:
                content = f.readlines()
            for l, line in enumerate(content):
                predictions_old[l,:] = np.asarray(line.split(','))
        predictions_old = predictions_old[predictions_old[:, 0].argsort()]
        # ts_pred = predictions[:, 0]
        idx = np.where(np.logical_and(predictions_old[:, 0] > 0.1 * ts_gt[-1], predictions_old[:, 0] < ts_gt[-1]))
        # idx = np.where(predictions_old[:, 0]<ts_gt[-1])
        predictions = predictions_old[idx[0], :]
        # ts_pred = predictions[idx[0], 0]
        ts_pred = predictions[:, 0]
        if len(ts_pred) == 0:
            print(f"Looks like prediction file {pred_path} is mostly empty.")
            continue
        timestamps.append(ts_pred)
        try:
            skeletons_pred = predictions[:, 2:].reshape(len(predictions),
                                                    len(ds_constants.HPECoreSkeleton.KEYPOINTS_MAP), -1)
        except ValueError:
            print(sample)
            print(pred_path)
            print(f"Looks like prediction file {pred_path} is mostly empty.")
            continue

        # evaluate only at the GT timestamps covered by the predictions, so that the first/last prediction is
        # never held frozen before the algorithm starts or after it stops
        in_pred_span = np.logical_and(t_window >= ts_pred[0], t_window <= ts_pred[-1])
        t_eval = t_window[in_pred_span]
        skeletons_gt_eval = skeletons_gt_window[in_pred_span]
        if len(t_eval) == 0:
            print(f"No GT samples within the time span of predictions {pred_path}, skipping.")
            timestamps.pop()
            continue
        # zero-order hold: take the latest prediction available at each GT timestamp (no look-ahead)
        pred_idx = np.searchsorted(ts_pred, t_eval, side='right') - 1
        pred_eval = predictions[pred_idx, :].copy()
        pred_eval[:, 0] = t_eval

        # predictions
        skeletons_pred = predictions[:, 2:].reshape(len(predictions),
                                                    len(ds_constants.HPECoreSkeleton.KEYPOINTS_MAP), -1)

        if algo_name == 'movenet' or algo_name == 'gl-hpe':
            skeletons_pred = skeletons_pred[:,:,1::-1]
        skeletons_pred_eval = pred_eval[:, 2:].reshape(len(pred_eval), len(ds_constants.HPECoreSkeleton.KEYPOINTS_MAP), -1)
        if algo_name == 'movenet' or algo_name == 'gl-hpe':
            skeletons_pred_eval = skeletons_pred_eval[:,:,1::-1]

        if args.multi_channel: ############################################ This is only done in case of DHP19
            # todo: Figure out the source of this error.
            skeletons_pred[:, :, [1, 0]] = skeletons_pred[:, :, [0, 1]]
            skeletons_pred_eval[:, :, [1, 0]] = skeletons_pred_eval[:, :, [0, 1]]

        algorithm_names.append(algo_name)
        skeletons_predictions.append(skeletons_pred)
        skeletons_predictions_eval.append(skeletons_pred_eval)
        timestamps_eval.append(t_eval)

        # for i in range(len(skeletons_pred)):
        #     plot_poses(np.zeros((400,400,3)),(skeletons_pred[i,:,:].squeeze().astype(int)),(skeletons_gt[i,:,:].squeeze().astype(int)))
        # compute PCK
        if len(args.pck) != 0:

            if 'pck' not in ds_results.keys():
                ds_results['pck'] = dict()

            pck_sizes_eval = pck_sizes_gt_interp(t_eval)
            for thi, th in enumerate(args.pck):

                # update dataset metric
                pck = metrics_utils.PCK(threshold=th)
                pck.update_samples(skeletons_pred_eval, skeletons_gt_eval, pck_sizes_eval)

                if th not in ds_results['pck'].keys():
                    ds_results['pck'][th] = dict()

                ds_results['pck'][th][algo_name] = pck
        else:
            pck_sizes_eval = None

        # compute RMSE
        if args.rmse:

            if 'rmse' not in ds_results.keys():
                ds_results['rmse'] = dict()

            rmse = metrics_utils.RMSE()
            rmse.update_samples(skeletons_pred_eval, skeletons_gt_eval)

            algo_name = pred_path.stem
            ds_results['rmse'][algo_name] = rmse

        # compute latency
        if args.lat:
            if 'latency' not in ds_results.keys():
                ds_results['latency'] = dict()

            latency.append(predictions[:, 1])
            ds_results['latency'][algo_name] = np.mean(predictions[:, 1])
            # latency = np.mean(predictions[:, 1])
            # ds_results['latency'][algo_name] = np.arange(1)
            # ds_results['latency'][algo_name][0] = np.mean(latency)

        # compute MPJPE
        if args.mpjpe:

            if 'mpjpe' not in ds_results.keys():
                ds_results['mpjpe'] = dict()

            mpjpe = metrics_utils.MPJPE()
            mpjpe.update_samples(skeletons_pred_eval, skeletons_gt_eval)

            algo_name = pred_path.stem
            ds_results['mpjpe'][algo_name] = mpjpe

        # compute missed detections, which are excluded from MPJPE and counted as incorrect by PCK
        if len(args.pck) != 0 or args.mpjpe:

            if 'missed' not in ds_results.keys():
                ds_results['missed'] = dict()

            missed = metrics_utils.MissedDetections()
            missed.update_samples(skeletons_pred_eval, skeletons_gt_eval)
            ds_results['missed'][algo_name] = missed

        global_samples.append((algo_name, skeletons_pred_eval, skeletons_gt_eval, pck_sizes_eval))

    report = report_dataset(results_key, ds_results, output_folder_path, args.latex) if args.sample_plots else None

    return {
        'results_key': results_key,
        'ds_results': ds_results,
        'report': report,
        'global_samples': global_samples,
        'output_ds_folder_path': output_ds_folder_path,
        't_window': t_window,
        'skeletons_gt_window': skeletons_gt_window,
        'algorithm_names': algorithm_names,
        'timestamps_eval': timestamps_eval,
        'skeletons_predictions_eval': skeletons_predictions_eval,
        'timestamps': timestamps,
        'latency': latency,
    }


def update_global_metrics(global_results: dict, global_samples: list, args) -> None:
    for algo_name, skeletons_pred_eval, skeletons_gt_eval, pck_sizes_eval in global_samples:

        if len(args.pck) != 0:
            if 'pck' not in global_results.keys():
                global_results['pck'] = dict()

            for th in args.pck:
                if th not in global_results['pck'].keys():
                    global_results['pck'][th] = dict()

                if algo_name not in global_results['pck'][th].keys():
                    global_results['pck'][th][algo_name] = metrics_utils.PCK(threshold=th)

                global_results['pck'][th][algo_name].update_samples(skeletons_pred_eval, skeletons_gt_eval,
                                                                    pck_sizes_eval)

        metrics_classes = list()
        if args.rmse:
            metrics_classes.append(('rmse', metrics_utils.RMSE))
        if args.mpjpe:
            metrics_classes.append(('mpjpe', metrics_utils.MPJPE))
        if len(args.pck) != 0 or args.mpjpe:
            metrics_classes.append(('missed', metrics_utils.MissedDetections))

        for metric_name, metric_class in metrics_classes:
            if metric_name not in global_results.keys():
                global_results[metric_name] = dict()

            if algo_name not in global_results[metric_name].keys():
                global_results[metric_name][algo_name] = metric_class()

            global_results[metric_name][algo_name].update_samples(skeletons_pred_eval, skeletons_gt_eval)


def update_sequences_std(sequences_std: dict, ds_results: dict) -> None:
    for metric_name in ('pck', 'mpjpe', 'missed'):
        if metric_name not in ds_results.keys():
            continue

        # PCK results are further indexed by threshold
        if metric_name == 'pck':
            metric_groups = ds_results['pck'].items()
        else:
            metric_groups = [(None, ds_results[metric_name])]

        for group_key, algos in metric_groups:
            std_algos = sequences_std.setdefault(metric_name, dict()).setdefault(group_key, dict())
            for algo_name, metric in algos.items():
                if algo_name not in std_algos.keys():
                    std_algos[algo_name] = metrics_utils.SequencesWeightedStd()

                std_algos[algo_name].update_samples(metric.get_value()[0], metric.get_weights())


def main(args):
    plt.close('all')

    output_folder_path = Path(args.output_folder).resolve()
    output_folder_path.mkdir(parents=True, exist_ok=True)

    predictions_folder = Path(args.predictions_path)

    to_latex = args.latex

    results = dict()
    results['datasets'] = dict()
    results['global'] = dict()
    # weighted std over sequences of the global metrics, indexed by metric name and PCK threshold (None otherwise)
    results['global_std'] = dict()
    # import GT from yarp
    datasets_path = Path(args.datasets_path)

    data_samples = create_pred_GT_pairing(predictions_folder,datasets_path,args.multi_channel)
    # yarp_file_paths = list(datasets_path.glob('**/data.log'))
    count = 0
    reports = dict()

    evaluate = partial(evaluate_sample, predictions_folder=predictions_folder, output_folder_path=output_folder_path,
                       args=args)
    # workers only save figures to file, so they do not need an interactive backend
    executor = ProcessPoolExecutor(max_workers=args.workers, initializer=plt.switch_backend,
                                   initargs=('Agg',)) if args.workers > 1 else None
    sample_results = (executor.map if executor else map)(evaluate, data_samples.keys(), data_samples.values())

    try:
        # results are merged in the order of data_samples, so that global metrics do not depend on the scheduling
        for r in tqdm(sample_results, total=len(data_samples)):
            if r is None:
                continue

            results['datasets'][r['results_key']] = r['ds_results']
            reports[r['results_key']] = r['report']
            update_global_metrics(results['global'], r['global_samples'], args)
            update_sequences_std(results['global_std'], r['ds_results'])

            # plots are created in the main process, so that they are shown by plt.show()
            if (args.plot_traj):
                plot_predictions(r['output_ds_folder_path'], r['results_key'], r['t_window'], r['skeletons_gt_window'],
                                 r['algorithm_names'], r['timestamps_eval'], r['skeletons_predictions_eval'])
            if (args.plot_sklt):
                plot_predictions_all_joints(r['output_ds_folder_path'], r['results_key'], r['t_window'],
                                            r['skeletons_gt_window'], r['algorithm_names'], r['timestamps_eval'],
                                            r['skeletons_predictions_eval'])
            if (args.lat):
                plot_latency(r['output_ds_folder_path'], r['results_key'], r['timestamps'], r['algorithm_names'],
                             r['latency'])
    finally:
        if executor:
            executor.shutdown()

    if args.sample_plots:
        # print the reports created while evaluating the samples
        for report in reports.values():
            print(report, end='')


    # iterate over global metrics and print results
    for (metric_name, metric_results) in results['global'].items():
        if (not (to_latex)):
            print("\n==================================================")
            print("Global results: ")
            print("==================================================")
        else:
            print("Global results ✓")

        if metric_name == 'pck':

            plot_pck_over_thresholds(metric_results, output_folder_path,
                                     pck_thresholds_std=results['global_std']['pck'])

            # create table
            header = [key for key in ds_constants.HPECoreSkeleton.KEYPOINTS_MAP.keys()]
            header.append('avg PCK')

            for (th, algos) in metric_results.items():
                if (not (to_latex)):
                    print("\n-= PCK threshold = ", th, " =-")
                tabulate_metric_over_algorithms(algos, header,
                                                descr=f'Global PCK results for threshold {th}',
                                                to_latex=to_latex,
                                                file_path=output_folder_path,
                                                file_stem=f'pck_{th}')
                tabulate_metric_over_algorithms(results['global_std']['pck'][th], header,
                                                descr=f'Global PCK std over sequences for threshold {th}',
                                                to_latex=to_latex,
                                                file_path=output_folder_path,
                                                file_stem=f'pck_{th}_std')

        elif metric_name == 'rmse':
            if (not (to_latex)):
                print("\n-= RMSE =-")
            # create table
            # header = [header_str for key in ds_constants.HPECoreSkeleton.KEYPOINTS_MAP.keys() for header_str in [f'{key} x', f'{key} y']]
            # header.append('avg RMSE x')
            # header.append('avg RMSE y')
            header = [header_str for key in ds_constants.HPECoreSkeleton.KEYPOINTS_MAP.keys() for header_str in
                      [f'{key}']]
            header.append('avg RMSE')

            tabulate_metric_over_algorithms(metric_results, header,
                                            descr=f'Global RMSE results',
                                            to_latex=to_latex,
                                            file_path=output_folder_path,
                                            file_stem='rmse')

            plot_boxplot(metric_results,
                         descr=f'Global RMSE results',
                         file_path=output_folder_path / f'rmse.png')
            
        elif metric_name == 'mpjpe':
            if (not (to_latex)):
                print("\n-= MPJPE =-")
            header = [header_str for key in ds_constants.HPECoreSkeleton.KEYPOINTS_MAP.keys() for header_str in
                      [f'{key}']]
            header.append('avg MPJPE')

            tabulate_metric_over_algorithms(metric_results, header,
                                            descr=f'Global MPJPE results',
                                            to_latex=to_latex,
                                            file_path=output_folder_path,
                                            file_stem='mpjpe')
            tabulate_metric_over_algorithms(results['global_std']['mpjpe'][None], header,
                                            descr=f'Global MPJPE std over sequences',
                                            to_latex=to_latex,
                                            file_path=output_folder_path,
                                            file_stem='mpjpe_std')

            plot_boxplot(metric_results,
                         descr=f'Global MPJPE results',
                         file_path=output_folder_path / f'mpjpe.png')

        elif metric_name == 'missed':
            if (not (to_latex)):
                print("\n-= MISSED DETECTIONS [%] =-")
            header = [key for key in ds_constants.HPECoreSkeleton.KEYPOINTS_MAP.keys()]
            header.append('avg missed [%]')

            tabulate_metric_over_algorithms(metric_results, header,
                                            descr=f'Global missed detections',
                                            to_latex=to_latex,
                                            file_path=output_folder_path,
                                            file_stem='missed')
            tabulate_metric_over_algorithms(results['global_std']['missed'][None], header,
                                            descr=f'Global missed detections std over sequences',
                                            to_latex=to_latex,
                                            file_path=output_folder_path,
                                            file_stem='missed_std')
            
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='...')
    parser.add_argument('-d', '--datasets_path', help='Path to the folders containing data saved in Yarp format',
                        required=True)
    parser.add_argument('-p', '--predictions_path',
                        help='Path to the predictions folder containing subfolders with results in .csv format.',
                        required=True)
    parser.add_argument('-i', '--images_folder', help='Path to the folder containing the image frames')
    parser.add_argument('-o', '--output_folder', help='Path to the folder where evaluation results will be saved',
                        required=True)
    parser.add_argument('-pck', help='List of thresholds for computing metric PCK; specifies that PCK must be computed',
                        type=float, nargs='+', default=[])
    parser.add_argument('-rmse', help='flag specifying that the metric RMSE must be computed', dest='rmse',
                        action='store_true')
    parser.set_defaults(rmse=False)
    parser.add_argument('-lat', help='flag specifying that the latency must be computed', dest='lat',
                        action='store_true')
    parser.set_defaults(lat=False)
    parser.add_argument('-latex', help='flag specifying that table results should saved to latex files', dest='latex',
                        action='store_true')
    parser.set_defaults(latex=False)
    parser.add_argument('-pt', help='flag specifying to plot trajectories', dest='plot_traj', action='store_true')
    parser.set_defaults(plot_traj=False)
    parser.add_argument('-ps', help='flag specifying to plot full skeleton trajectories in a single figure',
                        dest='plot_sklt', action='store_true')
    parser.set_defaults(plot_sklt=False)
    parser.add_argument('-splts', help='flag specifying to plot separate plots for each sample ',
                        dest='sample_plots', action='store_true')
    parser.set_defaults(sample_plots=False)
    parser.add_argument('-mul', help='flag specifying the input has multichannel nomenclature (Use for DHP19) ',
                        dest='multi_channel', action='store_true')
    parser.set_defaults(multi_channel=False)
    parser.add_argument('-e', '--exclude', action='append', default=[],
                        help='Exclude specific algorithms from results. Add a new -e for each algo.', required=False)
    parser.add_argument('-mpjpe', help='flag specifying that the metric MPJPE must be computed', dest='mpjpe',
                        action='store_true')
    parser.set_defaults(mpjpe=False)
    parser.add_argument('-j', '--workers', type=int, default=os.cpu_count(),
                        help='Number of processes evaluating samples in parallel; 1 disables multiprocessing')

    args, unknown = parser.parse_known_args()
    if (unknown):
        print('\x1b[1;31;20m' + 'Unknown argument/s: ' + ' '.join(unknown) + '\x1b[0m')

    main(args)
