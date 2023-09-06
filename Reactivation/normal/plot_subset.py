import warnings
import preprocess
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from os import path
from matplotlib import gridspec
warnings.filterwarnings('ignore')


def reactivation_bias(y_pred, behavior):
    y_pred_cs_2 = y_pred[:, 1]
    y_pred_cs_1 = y_pred[:, 0]
    y_pred_cs_1_bias = y_pred_cs_1 - y_pred_cs_2
    y_pred_cs_2_bias = y_pred_cs_2 - y_pred_cs_1
    y_pred_bias = np.zeros(len(y_pred_cs_1_bias))
    y_pred_rate = y_pred_cs_1 + y_pred_cs_2
    for i in range(0, len(behavior['onsets']) - 1):
        start = int(behavior['onsets'][i])
        end = int(behavior['onsets'][i+1])
        if behavior['cue_codes'][i][0] == behavior['cs_1_code']:
            y_pred_bias[start:end] = y_pred_cs_1_bias[start:end]
        if behavior['cue_codes'][i][0] == behavior['cs_2_code']:
            y_pred_bias[start:end] = y_pred_cs_2_bias[start:end]
    y_pred_binned_norm = []
    x_label = []
    factor = 2
    idx = .5
    for i in range(0, behavior['task_runs']):
        trials_per_run = int(int(len(behavior['onsets'])) / int(behavior['task_runs']))
        start = int(behavior['onsets'][i*trials_per_run])
        end = int(behavior['offsets'][((i+1)*trials_per_run) - 1])
        step = round((end-start)/factor)+1
        for j in range(0, factor):
            y_pred_binned_norm.append(np.sum(y_pred_bias[start + (j * step):start + ((j + 1) * step)]) /
                                      np.sum(y_pred_rate[start + (j * step):start + ((j + 1) * step)]))
            x_label.append((step/behavior['framerate']/60/60 * idx)[0][0])
            idx = idx + 1
    return y_pred_binned_norm


def save_reactivation_bias(y_pred_binned_norm, sa, paths, day, days):
    days_path = paths['base_path'] + paths['mouse'] + '/data_across_days/'
    if days:
        if path.isfile(days_path + 'y_pred_bias_binned_subset_' + str(sa) + '.npy') == 0 or day == 0:
            y_pred_bias_binned_across_days = [list(range(0, days))]
            y_pred_bias_binned_across_days[0][day] = y_pred_binned_norm
            np.save(days_path + 'y_pred_bias_binned_subset_' + str(sa), y_pred_bias_binned_across_days)
        else:
            y_pred_bias_binned_across_days = np.load(days_path + 'y_pred_bias_binned_subset_' + str(sa) + '.npy',
                                                     allow_pickle=True)
            y_pred_bias_binned_across_days[0][day] = y_pred_binned_norm
            np.save(days_path + 'y_pred_bias_binned_subset_' + str(sa), y_pred_bias_binned_across_days)


def reactivation_difference(y_pred_original, y_pred_subset, behavior, times_considered):
    reactivation_cs_1 = y_pred_subset[:, 0].copy()
    reactivation_cs_2 = y_pred_subset[:, 1].copy()
    p_threshold = .75
    cs_1_peak = 0
    cs_2_peak = 0
    i = 0
    reactivation_subset_frames = np.zeros(len(reactivation_cs_1))
    next_r = 0
    while i < len(reactivation_cs_1) - 1:
        i += 1
        if reactivation_cs_1[i] > 0 or reactivation_cs_2[i] > 0:
            if next_r == 0:
                r_start = i
                next_r = 1
            if reactivation_cs_1[i] > cs_1_peak:
                cs_1_peak = reactivation_cs_1[i]
            if reactivation_cs_2[i] > cs_2_peak:
                cs_2_peak = reactivation_cs_2[i]
            if reactivation_cs_1[i + 1] == 0 and reactivation_cs_2[i + 1] == 0:
                r_end = i + 1
                next_r = 0
                if cs_1_peak > p_threshold:
                    reactivation_subset_frames[r_start:r_end] = 1
                if cs_2_peak > p_threshold:
                    reactivation_subset_frames[r_start:r_end] = 1
                i = r_end
                cs_1_peak = 0
                cs_2_peak = 0
    y_pred_subset[reactivation_subset_frames == 0, 0] = 0
    y_pred_subset[reactivation_subset_frames == 0, 1] = 0

    all_original = y_pred_original[:, 0] + y_pred_original[:, 1]
    all_subset = y_pred_subset[:, 0] + y_pred_subset[:, 1]
    sum_original = np.sum(all_original) / (np.sum(times_considered) / int(behavior['framerate']))
    difference = all_subset - all_original
    false_positive = np.sum(difference[difference > 0]) / (np.sum(times_considered) / int(behavior['framerate']))
    false_negative = np.abs(np.sum(difference[difference < 0])) / (np.sum(times_considered) / int(behavior['framerate']))
    return [sum_original, false_positive, false_negative]


def save_reactivation_difference(sum_reactivation_original, false_positive, false_negative, sa, paths, day, days):
    days_path = paths['base_path'] + paths['mouse'] + '/data_across_days/'
    if days:
        if path.isfile(days_path + 'reactivation_difference_subset_continuous_' + str(sa) + '.npy') == 0 or day == 0:
            reactivation_difference_across_days = [list(range(0, days)), list(range(0, days)), list(range(0, days))]
            reactivation_difference_across_days[0][day] = sum_reactivation_original
            reactivation_difference_across_days[1][day] = false_positive
            reactivation_difference_across_days[2][day] = false_negative
            np.save(days_path + 'reactivation_difference_subset_continuous_' + str(sa), reactivation_difference_across_days)
        else:
            reactivation_difference_across_days = np.load(days_path + 'reactivation_difference_subset_continuous_' + str(sa) + '.npy',
                                                          allow_pickle=True)
            reactivation_difference_across_days[0][day] = sum_reactivation_original
            reactivation_difference_across_days[1][day] = false_positive
            reactivation_difference_across_days[2][day] = false_negative
            np.save(days_path + 'reactivation_difference_subset_continuous_' + str(sa), reactivation_difference_across_days)


def reactivation_raster(behavior, activity, y_pred, y_pred_original, idx_1, idx_2, both_idx, paths, session):
    """
    makes heatmap of reactivations
    :param behavior: behavior
    :param activity: deconvolved matrix
    :param y_pred: classifier output
    :param idx_1: cs 1 index
    :param idx_2: cs 2 index
    :param both_idx: index for both
    :param paths: path to data
    :param session: which session to plot, if [] then plot all
    :return: saves heatmap
    """
    if len(session) == 0:
        session = list(range(1, behavior['task_runs'] + behavior['dark_runs'] + 1))
    for i in range(0, len(session)):
        fig = plt.figure(figsize=(307, 20))
        gs0 = gridspec.GridSpec(3, 1, height_ratios=[1, .68, .68])
        gs1 = gridspec.GridSpecFromSubplotSpec(nrows=4, ncols=1, height_ratios=[.75, 6, 6, 25],
                                               subplot_spec=gs0[0])
        gs2 = gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=gs0[1])
        gs3 = gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=gs0[2])
        session_num = int(session[i])
        variables = {'session': session_num, 'num_neurons': 150, 'labels': 1}
        plot_reactivation(behavior, activity, both_idx, variables, gs1, fig, y_pred, y_pred_original, 'All cue')
        variables = {'session': session_num, 'num_neurons': 150, 'labels': 2}
        plot_reactivation(behavior, activity, idx_1, variables, gs2, fig, y_pred, y_pred_original, 'CS 1')
        variables = {'session': session_num, 'num_neurons': 150, 'labels': 2}
        plot_reactivation(behavior, activity, idx_2, variables, gs3, fig, y_pred, y_pred_original, 'CS 2')
        plt.savefig(paths['save_path'] + 'plots/' + paths['mouse'] + '_' + paths['date'] + '_' +
                    'reactivation_heatmap_run_' + str(session_num) + '.png',  bbox_inches='tight')
        plt.close(fig)


def plot_reactivation(behavior, norm_moving_deconvolved, idx, variables, gs, fig, y_pred, y_pred_original, label):
    """
    plots behavior plus cue sorted deconvolved dark
    :param behavior: dict of behavior
    :param norm_moving_deconvolved: normalized activity vector
    :param idx: sort of 1
    :param variables: which dark session to plot, num neurons
    :param gs: grid spec
    :param fig: figure handle
    :param y_pred: classifier output
    :param label: label
    :return: heat map of sorted dark
    """
    gs_num = 0
    session = variables['session']
    norm_moving_deconvolved = pd.DataFrame(norm_moving_deconvolved)
    sorted_deconvolved = norm_moving_deconvolved.reindex(idx.index[0:variables['num_neurons']])
    frames_per_run = int(behavior['frames_per_run'])
    sns.set(font_scale=1)
    if variables['labels'] == 1:
        fig.add_subplot(gs[gs_num])
        cue_idx = preprocess.cue_times(behavior, 0, 0)
        if session > behavior['dark_runs']:
            ax = sns.heatmap(cue_idx[:, (session - 1) * frames_per_run:session * frames_per_run],
                             cmap=['white', 'mediumseagreen', 'salmon'], cbar=0)
            ax.set(xticklabels=[])
        gs_num = gs_num + 1
        fig.add_subplot(gs[gs_num])
        plt.plot(y_pred_original[(session - 1) * frames_per_run:session * frames_per_run, 1], color='salmon',
                 lw=1.5)
        plt.plot(y_pred_original[(session - 1) * frames_per_run:session * frames_per_run, 0],
                 color='mediumseagreen', lw=1.5)
        plt.plot([0, frames_per_run], [1, 1], 'k--', lw=.5)
        plt.text(-220, .3, 'Reactivation', color='k', fontsize=14)
        plt.text(-220, .1, 'probability all neurons', color='k', fontsize=14)
        plt.xlim((0, frames_per_run))
        plt.ylim((0, 1))
        plt.axis('off')
        gs_num = gs_num + 1
        fig.add_subplot(gs[gs_num])
        plt.plot(y_pred[(session - 1) * frames_per_run:session * frames_per_run, 1], color='salmon',
                 lw=1.5)
        plt.plot(y_pred[(session - 1) * frames_per_run:session * frames_per_run, 0],
                 color='mediumseagreen', lw=1.5)
        plt.plot([0, frames_per_run], [1, 1], 'k--', lw=.5)
        plt.text(-220, .3, 'Reactivation', color='k', fontsize=14)
        plt.text(-220, .1, 'probability 10% neurons', color='k', fontsize=14)
        plt.xlim((0, frames_per_run))
        plt.ylim((0, 1))
        plt.axis('off')
        gs_num = gs_num + 1
        fig.add_subplot(gs[gs_num])
    if variables['labels'] == 0 or variables['labels'] == 2:
        fig.add_subplot(gs[0])
    ax = sns.heatmap(sorted_deconvolved.iloc[:, (session - 1) * frames_per_run:session * frames_per_run], vmin=0,
                     vmax=.75, cmap='Greys', cbar=0)
    ax.set_yticks(range(0, len(sorted_deconvolved) + 1, 50))
    ax.set_ylim(len(sorted_deconvolved), 0)
    if variables['labels'] == 1 or variables['labels'] == 2:
        ax.set(xticklabels=[])
    if variables['labels'] == 0:
        ax.set(xlabel='Frame')
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.025)
    if label == 'All cue':
        plt.text(-175, 100, label + ' neurons', color='k', fontsize=18, rotation=90)
    if label == 'CS 1':
        plt.text(-175, 100, label + ' neurons', color='mediumseagreen', fontsize=18, rotation=90)
    if label == 'CS 2':
        plt.text(-175, 100, label + ' neurons', color='salmon', fontsize=18, rotation=90)


def sample_reactivation_raster(behavior, activity, activity_original, y_pred, y_pred_original, idx_1, idx_2, idx_1_original,
                               idx_2_original, paths, start, end):
    num_neurons = 20
    num_neurons_original = 200
    fig = plt.figure(figsize=(10, 13.18))
    gs0 = gridspec.GridSpec(4, 1, height_ratios=[1, .76923, .08547, .08547])
    gs1 = gridspec.GridSpecFromSubplotSpec(nrows=4, ncols=1, height_ratios=[1.25, 3, 3, 25],
                                           subplot_spec=gs0[0])
    gs2 = gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=gs0[1])
    gs3 = gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=gs0[2])
    gs4 = gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=gs0[3])
    variables = {'num_neurons': num_neurons_original, 'labels': 1}
    sample_plot_reactivation(behavior, activity_original, idx_1_original, variables, gs1, fig, y_pred, y_pred_original, start, end)
    variables = {'num_neurons': num_neurons_original, 'labels': 2}
    sample_plot_reactivation(behavior, activity_original, idx_2_original, variables, gs2, fig, y_pred, y_pred_original, start, end)
    variables = {'num_neurons': num_neurons, 'labels': 2}
    sample_plot_reactivation(behavior, activity, idx_1, variables, gs3, fig, y_pred, y_pred_original, start, end)
    variables = {'num_neurons': num_neurons, 'labels': 2}
    sample_plot_reactivation(behavior, activity, idx_2, variables, gs4, fig, y_pred, y_pred_original, start, end)
    plt.savefig(paths['save_path'] + 'plots/' + paths['mouse'] + '_' + paths['date'] + '_' +
                'sample_reactivation_heatmap_' + str(start) + '.png',  bbox_inches='tight', dpi=500)
    plt.close()


def sample_plot_reactivation(behavior, norm_moving_deconvolved, idx, variables, gs, fig, y_pred, y_pred_original, start, end):
    gs_num = 0
    norm_moving_deconvolved = pd.DataFrame(norm_moving_deconvolved)
    sorted_deconvolved = norm_moving_deconvolved.reindex(idx.index[0:variables['num_neurons']])
    frames_per_run = int(behavior['frames_per_run'])
    sns.set(font_scale=1)
    if variables['labels'] == 1:
        fig.add_subplot(gs[gs_num])
        cue_idx = preprocess.cue_times(behavior, 0, 0)
        ax = sns.heatmap(cue_idx[:, start:end],
                         cmap=['white', 'mediumseagreen', 'mediumseagreen'], cbar=0)
        ax.set(xticklabels=[])
        gs_num = gs_num + 1
        plt.axis('off')
        fig.add_subplot(gs[gs_num])
        plt.plot(y_pred_original[start:end, 0],
                 color='mediumseagreen', lw=1.5)
        plt.plot(y_pred_original[start:end, 1], color='salmon',
                 lw=1.5)
        plt.plot([0, frames_per_run], [1, 1], 'k--', lw=.5)
        plt.xlim((0, end-start))
        plt.ylim((0, 1))
        plt.axis('off')
        gs_num = gs_num + 1
        fig.add_subplot(gs[gs_num])
        plt.plot(y_pred[start:end, 0],
                 color='mediumseagreen', lw=1.5)
        plt.plot(y_pred[start:end, 1], color='salmon',
                 lw=1.5)
        plt.plot([0, frames_per_run], [1, 1], 'k--', lw=.5)
        plt.xlim((0, end - start))
        plt.ylim((0, 1))
        plt.axis('off')
        gs_num = gs_num + 1
        fig.add_subplot(gs[gs_num])
    if variables['labels'] == 0 or variables['labels'] == 2:
        fig.add_subplot(gs[0])
    ax = sns.heatmap(sorted_deconvolved.iloc[:, start:end], vmin=0, vmax=.75, cmap='Greys', cbar=0)
    ax.set_yticks(range(0, len(sorted_deconvolved) + 1, 50))
    ax.set_ylim(len(sorted_deconvolved), 0)
    if variables['labels'] == 1 or variables['labels'] == 2:
        ax.set(xticklabels=[])
    if variables['labels'] == 0:
        ax.set(xlabel='Frame')
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.025)



























