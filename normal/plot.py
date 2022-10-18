import math
import random
import classify
import warnings
import preprocess
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from os import path
from scipy import stats
from scipy.io import loadmat
from matplotlib import gridspec
from numpy.linalg import norm
from scipy.spatial import distance
from sklearn import decomposition
from matplotlib.colors import LinearSegmentedColormap
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
warnings.filterwarnings('ignore')


def sorted_map(behavior, responses_cs_1, responses_cs_2, cs_1_idx, cs_2_idx, neuron_number, paths):
    """
    plot heatmap
    :param behavior: behavior
    :param responses_cs_1: cs 1 activity
    :param responses_cs_2: cs 2 activity
    :param cs_1_idx: cs 1 index
    :param cs_2_idx: cs 2 index
    :param neuron_number: number of neurons
    :param paths: path to data
    :return: saved heatmap
    """
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4, figsize=(10, 10))
    make_sorted_map(behavior, responses_cs_1, cs_1_idx, 'cs_1', neuron_number, ax1, 1)
    make_sorted_map(behavior, responses_cs_2, cs_1_idx, 'cs_2', neuron_number, ax2, 0)
    make_sorted_map(behavior, responses_cs_2, cs_2_idx, 'cs_2', neuron_number, ax3, 0)
    make_sorted_map(behavior, responses_cs_1, cs_2_idx, 'cs_1', neuron_number, ax4, 0)
    plt.subplots_adjust(right=1)
    plt.savefig(paths['save_path'] + 'plots/' + paths['mouse'] + '_' + paths['date'] + '_' + 'cue_heatmap.png',
                bbox_inches='tight', dpi=500)
    plt.close(fig)


def make_sorted_map(behavior, mean_responses, idx, trial_type, num_neurons, axes, label):
    """
    makes heatmap
    :param behavior: dict of behavior
    :param mean_responses: fluorescence
    :param idx: sorted index
    :param trial_type: trial type
    :param num_neurons: number of neurons to plot
    :param axes: axes handle
    :param label: label or not
    :return: heatmap plot
    """
    mean_responses = mean_responses.reindex(idx.index)
    frames_before = behavior['frames_before']
    sns.set(font_scale=2)
    ax = sns.heatmap(mean_responses, vmin=0, vmax=.12, cmap='Greys', cbar=0, yticklabels=False, ax=axes)
    # ax.set_xticks([frames_before, frames_before * 2])
    # ax.set_xticklabels(['0', '2'], rotation=0)
    ax.set_xticks([])
    # if label == 1:
    #     ax.set_yticks([0, 500, 1000, 1500])
    #     ax.set_yticklabels(['0', '500', '1000', '1500'], rotation=0)
    ax.set_ylim(num_neurons, 0)
    if trial_type == 'cs_1':
        ax.axvline(x=frames_before + .25, color='mediumseagreen', linestyle='-', linewidth=5, snap=False)
        ax.axvline(x=frames_before * 2 + .25, color='mediumseagreen', linestyle='-', linewidth=5, snap=False)
    if trial_type == 'cs_2':
        ax.axvline(x=frames_before + .25, color='salmon', linestyle='-', linewidth=5, snap=False)
        ax.axvline(x=frames_before * 2 + .25, color='salmon', linestyle='-', linewidth=5, snap=False)
    #ax.set(xlabel='Stimulus onset (s)')
    #sns.set_style("ticks")
    ax.tick_params(axis=u'both', which=u'both', length=0)
    ax.set_xlim(frames_before * .75, frames_before * 2.5)


def reactivation_raster(behavior, activity, y_pred, idx_1, idx_2, both_idx, paths, session):
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
        gs1 = gridspec.GridSpecFromSubplotSpec(nrows=7, ncols=1, height_ratios=[1.25, 1.25, 1.25, 1.25, .75, 6, 25],
                                               subplot_spec=gs0[0])
        gs2 = gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=gs0[1])
        gs3 = gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=gs0[2])
        session_num = int(session[i])
        variables = {'session': session_num, 'num_neurons': 150, 'labels': 1}
        plot_reactivation(behavior, activity, both_idx, variables, gs1, fig, y_pred, 'All cue')
        variables = {'session': session_num, 'num_neurons': 150, 'labels': 2}
        plot_reactivation(behavior, activity, idx_1, variables, gs2, fig, y_pred, 'CS 1')
        variables = {'session': session_num, 'num_neurons': 150, 'labels': 2}
        plot_reactivation(behavior, activity, idx_2, variables, gs3, fig, y_pred, 'CS 2')
        plt.savefig(paths['save_path'] + 'plots/' + paths['mouse'] + '_' + paths['date'] + '_' +
                    'reactivation_heatmap_run_' + str(session_num) + '.png',  bbox_inches='tight')
        plt.close(fig)


def plot_reactivation(behavior, norm_moving_deconvolved, idx, variables, gs, fig, y_pred, label):
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
        plt.plot(behavior['pupil'][(session - 1) * frames_per_run:session * frames_per_run], lw=1.5)
        plt.plot([0, frames_per_run], [behavior['pupil_max'].mean(), behavior['pupil_max'].mean()],
                 'k--', lw=.75)
        plt.xlim((0, frames_per_run))
        plt.axis('off')
        plt.text(-220, np.mean(behavior['pupil'][(session - 1) * frames_per_run:session * frames_per_run]),
                 'Pupil area (a.u.)', color='b', fontsize=14)
        gs_num = gs_num + 1
        fig.add_subplot(gs[gs_num])
        plt.plot(behavior['brain_motion'][(session - 1) * frames_per_run:session * frames_per_run],
                 color='darkgoldenrod', lw=1.5)
        plt.xlim((0, frames_per_run))
        plt.axis('off')
        plt.text(-220, np.mean(behavior['brain_motion'][(session - 1) * frames_per_run:session * frames_per_run]),
                 'Brain motion (μm)', color='darkgoldenrod', fontsize=14)
        gs_num = gs_num + 1
        fig.add_subplot(gs[gs_num])
        plt.plot(behavior['running'][(session - 1) * frames_per_run:session * frames_per_run], color='teal', lw=1.5)
        plt.xlim((0, frames_per_run))
        plt.axis('off')
        plt.text(-220, 0, 'Running (cm/s)', color='teal', fontsize=14)
        gs_num = gs_num + 1
        fig.add_subplot(gs[gs_num])
        plt.plot(behavior['licking'][(session - 1) * frames_per_run:session * frames_per_run], color='dimgray',
                 lw=1.5)
        plt.xlim((0, frames_per_run))
        plt.axis('off')
        plt.text(-220, 0, 'Licking', color='dimgray', fontsize=14)
        gs_num = gs_num + 1
        fig.add_subplot(gs[gs_num])
        cue_idx = preprocess.cue_times(behavior, 0, 0)
        if session > behavior['dark_runs']:
            ax = sns.heatmap(cue_idx[:, (session - 1) * frames_per_run:session * frames_per_run],
                             cmap=['white', 'mediumseagreen', 'salmon'], cbar=0)
            ax.set(xticklabels=[])
        gs_num = gs_num + 1
        fig.add_subplot(gs[gs_num])
        plt.plot(y_pred[(session - 1) * frames_per_run:session * frames_per_run, 1], color='salmon',
                 lw=1.5)
        plt.plot(y_pred[(session - 1) * frames_per_run:session * frames_per_run, 0],
                 color='mediumseagreen', lw=1.5)
        plt.plot([0, frames_per_run], [1, 1], 'k--', lw=.5)
        plt.text(-220, .3, 'Reactivation', color='k', fontsize=14)
        plt.text(-220, .1, 'probability', color='k', fontsize=14)
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
    # ax.set_yticklabels(['0', '50', '100', '150', '200', '250', '300'], rotation=0)
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


def sample_reactivation_raster(behavior, activity, norm_moving_deconvolved_filtered, y_pred, idx_full, idx_1, idx_2,
                               paths, start, end):
    num_neurons = 200
    fig = plt.figure(figsize=(10, 12)) #5
    gs0 = gridspec.GridSpec(2, 1, height_ratios=[1, .8547])
    gs1 = gridspec.GridSpecFromSubplotSpec(nrows=3, ncols=1, height_ratios=[1.25, 3, 25],
                                           subplot_spec=gs0[0])
    gs2 = gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=gs0[1])
    variables = {'num_neurons': num_neurons, 'labels': 1}
    sample_plot_reactivation(behavior, activity, norm_moving_deconvolved_filtered, idx_full, idx_1, variables, gs1, fig, y_pred,
                             start, end, paths)
    variables = {'num_neurons': num_neurons, 'labels': 2}
    sample_plot_reactivation(behavior, activity, norm_moving_deconvolved_filtered, idx_full, idx_2, variables, gs2, fig, y_pred,
                             start, end, paths)
    plt.savefig(paths['save_path'] + 'plots/' + paths['mouse'] + '_' + paths['date'] + '_' +
                'sample_reactivation_heatmap.png',  bbox_inches='tight', dpi=500)
    #plt.close(fig)


def sample_plot_reactivation(behavior, activity, norm_moving_deconvolved_filtered, idx_full, idx, variables, gs, fig, y_pred,
                             start, end, paths):
    gs_num = 0

    # activity = activity.loc[idx.index, :]
    # activity = activity.to_numpy()
    # sig_cells = preprocess.sig_reactivated_cells(activity, norm_moving_deconvolved_filtered, idx_full, y_pred, behavior,
    #                                              paths, 2)
    # activity = activity[sig_cells > 0, :]
    # sorted_deconvolved = activity[0:variables['num_neurons']]
    # sorted_deconvolved = pd.DataFrame(sorted_deconvolved)

    activity = pd.DataFrame(activity)
    sorted_deconvolved = activity.reindex(idx.index[0:variables['num_neurons']])

    frames_per_run = int(behavior['frames_per_run'])
    text_place = -250
    sns.set(font_scale=1)
    if variables['labels'] == 1:

        # fig.add_subplot(gs[gs_num])
        # plt.plot(behavior['pupil'][start:end], lw=1.5)
        # plt.xlim((0, end-start))
        # plt.plot([0, frames_per_run*5], [behavior['pupil_max'].mean(), behavior['pupil_max'].mean()], 'k--', lw=1.5)
        # plt.axis('off')
        # # plt.text(text_place, np.mean(behavior['pupil'][start:end]), 'Pupil area (a.u.)', color='b', fontsize=17)
        # gs_num = gs_num + 1
        # fig.add_subplot(gs[gs_num])
        #
        # plt.plot(behavior['brain_motion'][start:end], color='darkgoldenrod', lw=1.5)
        # plt.xlim((0, end-start))
        # plt.axis('off')
        # # plt.text(text_place, 0, 'Brain motion (μm)', color='darkgoldenrod', fontsize=17)
        # gs_num = gs_num + 1
        # fig.add_subplot(gs[gs_num])
        fig.add_subplot(gs[gs_num])
        cue_idx = preprocess.cue_times(behavior, 0, 0)
        ax = sns.heatmap(cue_idx[:, start:end],
                         cmap=['white', 'mediumseagreen', 'mediumseagreen'], cbar=0)
        ax.set(xticklabels=[])
        gs_num = gs_num + 1
        plt.axis('off')
        fig.add_subplot(gs[gs_num])


        plt.plot(y_pred[start:end, 0],
                 color='mediumseagreen', lw=1.5)
        plt.plot(y_pred[start:end, 1], color='salmon',
                 lw=1.5)
        plt.plot([0, frames_per_run], [1, 1], 'k--', lw=.5)
        # plt.text(text_place, .5, 'Reactivation', color='k', fontsize=17)
        # plt.text(text_place, .1, 'probability', color='k', fontsize=17)
        plt.xlim((0, end-start))
        plt.ylim((0, 1))
        plt.axis('off')
        gs_num = gs_num + 1
        fig.add_subplot(gs[gs_num])

    if variables['labels'] == 0 or variables['labels'] == 2:
        fig.add_subplot(gs[0])
    ax = sns.heatmap(sorted_deconvolved.iloc[:, start:end], vmin=0, vmax=.75, cmap='Greys', cbar=0)
    ax.set_yticks(range(0, len(sorted_deconvolved) + 1, 50))
    # ax.set_yticklabels(['0', '200'], rotation=0)
    ax.set_ylim(len(sorted_deconvolved), 0)
    if variables['labels'] == 1 or variables['labels'] == 2:
        ax.set(xticklabels=[])
    if variables['labels'] == 0:
        ax.set(xlabel='Frame')
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.025)
    # for i in range(0, len(behavior['onsets'])):
    #     if behavior['cue_codes'][i] == behavior['cs_1_code']:
    #         ax.axvline(x=behavior['onsets'][i]-start, color='mediumseagreen', linestyle='-', linewidth=2, snap=False)
    #         ax.axvline(x=behavior['offsets'][i]-start, color='mediumseagreen', linestyle='-', linewidth=2, snap=False)
    #     if behavior['cue_codes'][i] == behavior['cs_2_code']:
    #         ax.axvline(x=start-behavior['onsets'][i], color='salmon', linestyle='-', linewidth=2, snap=False)
    #         ax.axvline(x=start-behavior['offsets'][i], color='salmon', linestyle='-', linewidth=2, snap=False)


def reactivation_rate(y_pred, behavior, paths, day):
    """
    plot all reactivations
    :param y_pred: y pred
    :param behavior: behavior
    :param paths: path to data
    :param day: which day
    :return: plot
    """
    y_pred = true_reactivations(y_pred)
    frames_per_run = int(behavior['frames_per_run'])
    y_pred_all = y_pred[:, 0] + y_pred[:, 1]
    times_considered = preprocess.get_times_considered(y_pred, behavior)
    y_pred_binned = []
    x_label = []
    factor = 2
    idx = .5
    for i in range(0, behavior['dark_runs']+behavior['task_runs']):
        if i == 0:
            y_pred_binned.append(np.sum(y_pred_all[(i * frames_per_run):(i + 1) * frames_per_run]) /
                                 (np.sum(times_considered[(i * frames_per_run):(i + 1) * frames_per_run]) /
                                  int(behavior['framerate'])))
            x_label.append((-int(frames_per_run/factor)/behavior['framerate']/60/60)[0][0])
        else:
            trials_per_run = int(int(len(behavior['onsets'])) / int(behavior['task_runs']))
            start = int(behavior['onsets'][(i-1)*trials_per_run])
            end = int(behavior['offsets'][(i*trials_per_run) - 1])
            step = round((end-start)/factor)+1
            for j in range(0, factor):
                y_pred_binned.append(np.sum(y_pred_all[start + (j * step):start + ((j+1) * step)]) /
                                     (np.sum(times_considered[start + (j * step):start + ((j+1) * step)]) /
                                      int(behavior['framerate'])))
                x_label.append((step/behavior['framerate']/60/60 * idx)[0][0])
                idx = idx + 1
    sns.set(font_scale=1.5)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(8, 5))
    plt.plot(x_label[0:2], y_pred_binned[0:2], '--ok')
    plt.plot(x_label[1:len(x_label)], y_pred_binned[1:len(y_pred_binned)], '-ok')
    plt.axvspan(-.5, 0, alpha=.25, color='gray')
    plt.ylabel('Mean reactivation probability (/s)')
    plt.xlabel('Time from first cue presentation (hours)')
    sns.despine()
    plt.savefig(paths['save_path'] + 'plots/' + paths['mouse'] + '_' + paths['date'] + '_' +
                'mean_reactivation_binned.png', bbox_inches='tight', dpi=150)
    plt.close()

    days_path = paths['base_path'] + paths['mouse'] + '/data_across_days/'
    if path.isfile(days_path + 'y_pred_binned.npy') == 0 or day == 0:
        np.save(days_path + 'y_pred_binned', [])
    y_pred_binned_across_days = list(np.load(days_path + 'y_pred_binned.npy'))
    y_pred_binned_across_days.append(y_pred_binned)
    np.save(days_path + 'y_pred_binned', y_pred_binned_across_days)


def reactivation_bias(y_pred, behavior, paths, day, days):
    """
    plot all reactivation bias over time
    :param y_pred: y pred
    :param behavior: behavior
    :param paths: path to data
    :param day: which day
    :param days: total days
    :return: plot
    """
    y_pred = true_reactivations(y_pred)
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
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    fig = plt.figure(figsize=(8, 5))
    y_pred_binned_norm = np.nan_to_num(y_pred_binned_norm, copy=True, nan=0.0, posinf=None, neginf=None)
    plt.plot(x_label, y_pred_binned_norm, '-ok')
    plt.ylabel('Mean reactivation bias')
    plt.xlabel('Time from first cue presentation (hours)')
    sns.despine()
    plt.savefig(paths['save_path'] + 'plots/' + paths['mouse'] + '_' + paths['date'] + '_' +
                'mean_reactivation_bias_binned.png', bbox_inches='tight', dpi=150)
    plt.close(fig)

    days_path = paths['base_path'] + paths['mouse'] + '/data_across_days/'
    if days:
        if path.isfile(days_path + 'y_pred_bias_binned.npy') == 0 or day == 0:
            y_pred_bias_binned_across_days = [list(range(0, days))]
            y_pred_bias_binned_across_days[0][day] = y_pred_binned_norm
            np.save(days_path + 'y_pred_bias_binned', y_pred_bias_binned_across_days)
        else:
            y_pred_bias_binned_across_days = np.load(days_path + 'y_pred_bias_binned.npy', allow_pickle=True)
            y_pred_bias_binned_across_days[0][day] = y_pred_binned_norm
            np.save(days_path + 'y_pred_bias_binned', y_pred_bias_binned_across_days)


def rate_within_trial(y_pred, behavior, paths, day, days):
    """
    trial reactivation
    :param y_pred: y pred
    :param behavior: behavior
    :param trial_type: trial type
    :param paths: path
    :param day: day
    :param days: days
    :return: trial reactivation
    """
    y_pred = true_reactivations(y_pred)
    times_considered = preprocess.get_times_considered(y_pred, behavior)
    fr = behavior['framerate']
    duration = int(fr * (behavior['iti'] + 5)) + 1
    end_trials = behavior['end_trials']
    rate_norm = np.empty((len(behavior['cue_codes']), duration)) * np.nan
    times_considered_norm = np.empty((len(behavior['cue_codes']), duration)) * np.nan

    for i in range(0, len(behavior['cue_codes'])):
        if i not in end_trials:
            idx_curr = int(behavior['onsets'][i])
            rate_norm[i, :] = y_pred[:, 0][idx_curr:idx_curr+duration] + y_pred[:, 1][idx_curr:idx_curr+duration]
            times_considered_norm[i, :] = times_considered[idx_curr:idx_curr+duration]

    rate_norm_binned = []
    x_binned = []
    factor = 5
    duration = int(fr * 8)
    step = int(((len(rate_norm[0])) - duration) / factor)
    for i in range(0, factor):
        rate_norm_binned.append(
            np.nansum(rate_norm[:, duration + (i * step):duration + (i + 1) * step]) *
            int(fr) / np.nansum(times_considered_norm[:, duration + (i * step):duration + (i + 1) * step]))
        x_binned.append(duration+(i + .5) * step)

    days_path = paths['base_path'] + paths['mouse'] + '/data_across_days/'
    if days:
        if path.isfile(days_path + 'rate_within_trial.npy') == 0 or day == 0:
            rate_within_trial_across_days = [list(range(0, days)), list(range(0, days)), list(range(0, days))]
            rate_within_trial_across_days[0][day] = rate_norm_binned
            rate_within_trial_across_days[1][day] = x_binned
            np.save(days_path + 'rate_within_trial', rate_within_trial_across_days)
        else:
            rate_within_trial_across_days = np.load(days_path + 'rate_within_trial.npy', allow_pickle=True)
            rate_within_trial_across_days[0][day] = rate_norm_binned
            rate_within_trial_across_days[1][day] = x_binned
            np.save(days_path + 'rate_within_trial', rate_within_trial_across_days)


def bias_within_trial(y_pred, behavior, paths, day, days):
    """
    trial reactivation
    :param y_pred: y pred
    :param behavior: behavior
    :param trial_type: trial type
    :param paths: path
    :param day: day
    :param days: days
    :return: trial reactivation
    """
    y_pred = true_reactivations(y_pred)
    fr = behavior['framerate']
    duration = int(fr * (behavior['iti'] + 5)) + 1
    end_trials = behavior['end_trials']
    bias_norm = np.empty((len(behavior['cue_codes']), duration)) * np.nan
    total_norm = np.empty((len(behavior['cue_codes']), duration)) * np.nan

    for i in range(0, len(behavior['cue_codes'])):
        if i not in end_trials:
            idx_curr = int(behavior['onsets'][i])
            if behavior['cue_codes'][i] == behavior['cs_1_code']:
                bias_norm[i, :] = y_pred[:, 0][idx_curr:idx_curr+duration] - y_pred[:, 1][idx_curr:idx_curr+duration]
                total_norm[i, :] = y_pred[:, 0][idx_curr:idx_curr+duration] + y_pred[:, 1][idx_curr:idx_curr+duration]
            if behavior['cue_codes'][i] == behavior['cs_2_code']:
                bias_norm[i, :] = y_pred[:, 1][idx_curr:idx_curr+duration] - y_pred[:, 0][idx_curr:idx_curr+duration]
                total_norm[i, :] = y_pred[:, 0][idx_curr:idx_curr+duration] + y_pred[:, 1][idx_curr:idx_curr+duration]

    bias_norm_binned = []
    x_binned = []
    factor = 5
    duration = int(fr * 8)
    step = int(((len(bias_norm[0])) - duration) / factor)
    for i in range(0, factor):
        bias_norm_binned.append(
            np.nansum(bias_norm[:, duration + (i * step):duration + (i + 1) * step]) /
            np.nansum(total_norm[:, duration + (i * step):duration + (i + 1) * step]))
        x_binned.append(duration+(i + .5) * step)

    days_path = paths['base_path'] + paths['mouse'] + '/data_across_days/'
    if days:
        if path.isfile(days_path + 'bias_within_trial.npy') == 0 or day == 0:
            bias_within_trial_across_days = [list(range(0, days)), list(range(0, days)), list(range(0, days))]
            bias_within_trial_across_days[0][day] = bias_norm_binned
            bias_within_trial_across_days[1][day] = x_binned
            np.save(days_path + 'bias_within_trial', bias_within_trial_across_days)
        else:
            bias_within_trial_across_days = np.load(days_path + 'bias_within_trial.npy', allow_pickle=True)
            bias_within_trial_across_days[0][day] = bias_norm_binned
            bias_within_trial_across_days[1][day] = x_binned
            np.save(days_path + 'bias_within_trial', bias_within_trial_across_days)


def iti_activity_across_trials(norm_deconvolved, y_pred, idx, behavior, paths, day):
    """
    gets iti activity across session
    :param y_pred: pupil
    :param behavior: behavior
    :param paths: path
    :param day: day
    :return: pupil binned
    """
    activity = norm_deconvolved.to_numpy()
    activity = activity[idx['both'].index]
    activity = np.mean(activity, axis=0)
    times_considered = preprocess.get_times_considered(y_pred, behavior)
    frames_per_run = int(behavior['frames_per_run'])
    activity[times_considered == 0] = np.nan
    activity_binned = []
    factor = 2
    for i in range(0, behavior['dark_runs']+behavior['task_runs']):
        if i == 0:
            activity_binned.append(np.nanmean(activity[(i * frames_per_run):(i + 1) * frames_per_run]))
        else:
            trials_per_run = int(int(len(behavior['onsets'])) / int(behavior['task_runs']))
            start = int(behavior['onsets'][(i-1)*trials_per_run])
            end = int(behavior['offsets'][(i*trials_per_run) - 1])
            step = round((end-start)/factor)+1
            for j in range(0, factor):
                activity_binned.append(np.nanmean(activity[start + (j * step):start + ((j+1) * step)]))
    days_path = paths['base_path'] + paths['mouse'] + '/data_across_days/'
    if path.isfile(days_path + 'activity_binned.npy') == 0 or day == 0:
        np.save(days_path + 'activity_binned', [])
    activity_binned_across_days = list(np.load(days_path + 'activity_binned.npy'))
    activity_binned_across_days.append(activity_binned)
    np.save(days_path + 'activity_binned', activity_binned_across_days)


def iti_activity_within_trial(norm_deconvolved, y_pred, idx, behavior, paths, day, days):
    """
    gets iti activity across session
    :param y_pred: pupil
    :param behavior: behavior
    :param paths: path
    :param day: day
    :return: pupil binned
    """
    activity = norm_deconvolved.to_numpy()
    activity = activity[idx['both'].index]
    activity = np.mean(activity, axis=0)
    times_considered = preprocess.get_times_considered(y_pred, behavior)
    activity[times_considered == 0] = np.nan
    fr = behavior['framerate']
    duration = int(fr * (behavior['iti'] + 5)) + 1
    end_trials = behavior['end_trials']

    activity_vec = np.empty((len(behavior['cue_codes']), duration)) * np.nan
    for i in range(0, len(behavior['cue_codes'])):
        if i not in end_trials:
            idx_curr = int(behavior['onsets'][i])
            activity_vec[i, :] = activity[idx_curr:idx_curr+duration]
    activity_binned = []
    x_binned = []
    factor = 5
    duration = int(fr * 8)
    step = int(((len(activity_vec[0])) - duration) / factor)
    for i in range(0, factor):
        activity_binned.append(
            np.nanmean(activity_vec[:, duration + (i * step):duration + (i + 1) * step]))
        x_binned.append(duration + (i + .5) * step)

    days_path = paths['base_path'] + paths['mouse'] + '/data_across_days/'
    if days:
        if path.isfile(days_path + 'activity_within_trial.npy') == 0 or day == 0:
            activity_within_trial_across_days = [list(range(0, days)), list(range(0, days))]
            activity_within_trial_across_days[0][day] = activity_binned
            activity_within_trial_across_days[1][day] = x_binned
            np.save(days_path + 'activity_within_trial', activity_within_trial_across_days)
        else:
            activity_within_trial_across_days = np.load(days_path + 'activity_within_trial.npy', allow_pickle=True)
            activity_within_trial_across_days[0][day] = activity_binned
            activity_within_trial_across_days[1][day] = x_binned
            np.save(days_path + 'activity_within_trial', activity_within_trial_across_days)


def pupil_across_trials(y_pred, behavior, paths, day):
    """
    gets pupil across session
    :param y_pred: pupil
    :param behavior: behavior
    :param paths: path
    :param day: day
    :return: pupil binned
    """
    times_considered = preprocess.get_times_considered(y_pred, behavior)
    frames_per_run = int(behavior['frames_per_run'])
    pupil = behavior['pupil']
    pupil[times_considered == 0] = np.nan
    pupil_binned = []
    factor = 2
    for i in range(0, behavior['dark_runs']+behavior['task_runs']):
        if i == 0:
            pupil_binned.append(np.nanmean(pupil[(i * frames_per_run):(i + 1) * frames_per_run])
                                / behavior['pupil_max'])
        else:
            trials_per_run = int(int(len(behavior['onsets'])) / int(behavior['task_runs']))
            start = int(behavior['onsets'][(i-1)*trials_per_run])
            end = int(behavior['offsets'][(i*trials_per_run) - 1])
            step = round((end-start)/factor)+1
            for j in range(0, factor):
                pupil_binned.append(np.nanmean(pupil[start + (j * step):start + ((j+1) * step)])
                                    / behavior['pupil_max'])
    days_path = paths['base_path'] + paths['mouse'] + '/data_across_days/'
    if path.isfile(days_path + 'pupil_binned.npy') == 0 or day == 0:
        np.save(days_path + 'pupil_binned', [])
    pupil_binned_across_days = list(np.load(days_path + 'pupil_binned.npy'))
    pupil_binned_across_days.append(pupil_binned)
    np.save(days_path + 'pupil_binned', pupil_binned_across_days)


def pupil_within_trial(y_pred, behavior, paths, day, days):
    """
    gets pupil across session
    :param y_pred: pupil
    :param behavior: behavior
    :param paths: path
    :param day: day
    :return: pupil binned
    """
    times_considered = preprocess.get_times_considered(y_pred, behavior)
    pupil = behavior['pupil']
    pupil[times_considered == 0] = np.nan
    fr = behavior['framerate']
    duration = int(fr * (behavior['iti'] + 5)) + 1
    end_trials = behavior['end_trials']

    pupil_vec = np.empty((len(behavior['cue_codes']), duration)) * np.nan
    for i in range(0, len(behavior['cue_codes'])):
        if i not in end_trials:
            idx_curr = int(behavior['onsets'][i])
            pupil_vec[i, :] = pupil[idx_curr:idx_curr + duration] / behavior['pupil_max']
    pupil_binned = []
    x_binned = []
    factor = 5
    duration = int(fr * 8)
    step = int(((len(pupil_vec[0])) - duration) / factor)
    for i in range(0, factor):
        pupil_binned.append(
            np.nanmean(pupil_vec[:, duration + (i * step):duration + (i + 1) * step]))
        x_binned.append(duration + (i + .5) * step)

    days_path = paths['base_path'] + paths['mouse'] + '/data_across_days/'
    if days:
        if path.isfile(days_path + 'pupil_within_trial.npy') == 0 or day == 0:
            pupil_within_trial_across_days = [list(range(0, days)), list(range(0, days))]
            pupil_within_trial_across_days[0][day] = pupil_binned
            pupil_within_trial_across_days[1][day] = x_binned
            np.save(days_path + 'pupil_within_trial', pupil_within_trial_across_days)
        else:
            pupil_within_trial_across_days = np.load(days_path + 'pupil_within_trial.npy', allow_pickle=True)
            pupil_within_trial_across_days[0][day] = pupil_binned
            pupil_within_trial_across_days[1][day] = x_binned
            np.save(days_path + 'pupil_within_trial', pupil_within_trial_across_days)


def trial_reactivations(y_pred, behavior, paths):
    """
    plot trial evoked reactivation
    :param y_pred: y pred
    :param behavior: behavior
    :param paths: path
    :return: plot of trial evoked reactivations
    """
    y_pred = true_reactivations(y_pred)
    fr = behavior['framerate']
    duration = int(fr * (behavior['iti'] + 5)) + 1
    end_trials = behavior['end_trials']
    cs_1_r_1 = np.empty((len(behavior['cue_codes']), duration)) * np.nan
    cs_1_r_2 = np.empty((len(behavior['cue_codes']), duration)) * np.nan
    cs_2_r_2 = np.empty((len(behavior['cue_codes']), duration)) * np.nan
    cs_2_r_1 = np.empty((len(behavior['cue_codes']), duration)) * np.nan

    for i in range(0, len(behavior['cue_codes'])):
        if i not in end_trials:
            idx_curr = int(behavior['onsets'][i])
            if behavior['cue_codes'][i] == behavior['cs_1_code']:
                cs_1_r_1[i, :] = y_pred[:, 0][idx_curr:idx_curr + duration]
                cs_1_r_2[i, :] = y_pred[:, 1][idx_curr:idx_curr + duration]
            if behavior['cue_codes'][i] == behavior['cs_2_code']:
                cs_2_r_1[i, :] = y_pred[:, 0][idx_curr:idx_curr + duration]
                cs_2_r_2[i, :] = y_pred[:, 1][idx_curr:idx_curr + duration]

    cs_1_r_1 = cs_1_r_1[~np.isnan(cs_1_r_1).any(axis=1)]
    cs_1_r_2 = cs_1_r_2[~np.isnan(cs_1_r_2).any(axis=1)]
    cs_2_r_2 = cs_2_r_2[~np.isnan(cs_2_r_2).any(axis=1)]
    cs_2_r_1 = cs_2_r_1[~np.isnan(cs_2_r_1).any(axis=1)]

    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(9, 3))
    plt.subplots_adjust(hspace=.5)
    plt.subplot(2, 2, 1)
    sns.heatmap(cs_1_r_1, vmin=0, vmax=.75,
                cmap=LinearSegmentedColormap.from_list('mycmap', ['white', 'white', 'mediumseagreen']), cbar=0)

    plt.axvspan(0, int(fr * 2), alpha=.75, color='mediumseagreen', lw=0)
    plt.axvspan(int(fr * 2), int(fr * 8), alpha=.1, color='gray', lw=0)
    plt.ylabel('Trial number')
    plt.xticks([int(fr * 0), int(fr * 10), int(fr * 20), int(fr * 30), int(fr * 40), int(fr * 50), int(fr * 60)],
               ['0', '10', '20', '30', '40', '50', '60'], rotation=0)
    plt.yticks([0, 24, 49, 74],
               ['1', '25', '50', '75'])
    plt.ylim((len(cs_1_r_1), 0))
    plt.xlabel('Time relative to stimulus onset (s)')
    plt.subplot(2, 2, 2)
    sns.heatmap(cs_1_r_2, vmin=0, vmax=.75,
                cmap=LinearSegmentedColormap.from_list('mycmap', ['white', 'white', 'salmon']), cbar=0)

    plt.axvspan(0, int(fr * 2), alpha=.75, color='mediumseagreen', lw=0)
    plt.axvspan(int(fr * 2), int(fr * 8), alpha=.1, color='gray', lw=0)
    plt.xticks([int(fr * 0), int(fr * 10), int(fr * 20), int(fr * 30), int(fr * 40), int(fr * 50), int(fr * 60)],
               ['0', '10', '20', '30', '40', '50', '60'], rotation=0)
    plt.yticks([0, 24, 49, 74],
               ['1', '25', '50', '75'])
    plt.ylim((len(cs_1_r_2), 0))
    plt.xlabel('Time relative to stimulus onset (s)')
    plt.subplot(2, 2, 3)
    sns.heatmap(cs_2_r_1, vmin=0, vmax=.75,
                cmap=LinearSegmentedColormap.from_list('mycmap', ['white', 'white', 'mediumseagreen']), cbar=0)

    plt.axvspan(0, int(fr * 2), alpha=.75, color='salmon', lw=0)
    plt.axvspan(int(fr * 2), int(fr * 8), alpha=.1, color='gray', lw=0)
    plt.ylabel('Trial number')
    plt.xlabel('Time relative to stimulus onset (s)')
    plt.xticks([int(fr * 0), int(fr * 10), int(fr * 20), int(fr * 30), int(fr * 40), int(fr * 50), int(fr * 60)],
               ['0', '10', '20', '30', '40', '50', '60'], rotation=0)
    plt.yticks([0, 24, 49, 74],
               ['1', '25', '50', '75'])
    plt.ylim((len(cs_2_r_1), 0))
    plt.subplot(2, 2, 4)
    sns.heatmap(cs_2_r_2, vmin=0, vmax=.75,
                cmap=LinearSegmentedColormap.from_list('mycmap', ['white', 'white', 'salmon']), cbar=0)

    plt.axvspan(0, int(fr * 2), alpha=.75, color='salmon', lw=0)
    plt.axvspan(int(fr * 2), int(fr * 8), alpha=.1, color='gray', lw=0)
    plt.xlabel('Time relative to stimulus onset (s)')
    plt.xticks([int(fr * 0), int(fr * 10), int(fr * 20), int(fr * 30), int(fr * 40), int(fr * 50), int(fr * 60)],
               ['0', '10', '20', '30', '40', '50', '60'], rotation=0)
    plt.yticks([0, 24, 49, 74],
               ['1', '25', '50', '75'])
    plt.ylim((len(cs_2_r_2), 0))
    sns.despine()
    plt.savefig(paths['save_path'] + 'plots/' + paths['mouse'] + '_' + paths['date'] + '_' +
                'trial_averaged_reactivation.pdf', bbox_inches='tight', dpi=200, transparent=True)
    #plt.close()


def trial_history(norm_deconvolved, idx, y_pred, behavior, paths, num_prev, day, days):
    """
    plots trial history modulation
    :param norm_deconvolved: activity
    :param idx: index
    :param y_pred: y pred
    :param behavior: behavior
    :param paths: path to data
    :param num_prev: how many trials to look previous
    :param day: day
    :param days: days
    :return: trial history modulation
    """

    num_neurons = len(idx['both'])
    norm_deconvolved = pd.DataFrame(norm_deconvolved)
    sorted_deconvolved = norm_deconvolved.reindex(idx['both'].index[0:num_neurons]).mean()

    y_pred = true_reactivations(y_pred)
    y_pred_all = y_pred[:, 0] + y_pred[:, 1]
    times_considered = preprocess.get_times_considered(y_pred, behavior)

    fr = behavior['framerate']
    duration = int(fr * (behavior['iti'] + 5)) + 1
    before = int(fr * 5)

    reactivation_same = []
    reactivation_diff = []
    bias_same = []
    bias_diff = []
    pupil_same = []
    pupil_diff = []
    cue_evoked_same = np.empty((1000, duration + before)) * np.nan
    cue_evoked_different = np.empty((1000, duration + before)) * np.nan

    end_trials = behavior['end_trials']

    if num_prev == 1:
        for i in range(1, len(behavior['cue_codes'])):
            if i not in end_trials:
                prev_cue = behavior['cue_codes'][i-1]
                idx = int(behavior['onsets'][i])

                if prev_cue == behavior['cue_codes'][i]:
                    reactivation_same.append(fr * np.sum(y_pred_all[idx:idx + duration]) /
                                             np.sum(times_considered[idx:idx + duration]))
                    pupil_same.append((np.mean(behavior['pupil'][idx:idx + int(fr * 5)]) -
                                       np.mean(behavior['pupil'][idx - before:idx])) /
                                      np.mean(behavior['pupil'][idx - before:idx]))
                    cue_evoked_same[i, :] = sorted_deconvolved[idx - before:idx + duration]
                    if behavior['cue_codes'][i] == behavior['cs_1_code']:
                        bias_same.append(
                            (np.sum(y_pred[idx:idx + duration, 0]) - np.sum(y_pred[idx:idx + duration, 1]))
                            / (np.sum(y_pred[idx:idx + duration, 0]) + np.sum(y_pred[idx:idx + duration, 1])))
                    if behavior['cue_codes'][i] == behavior['cs_2_code']:
                        bias_same.append(
                            (np.sum(y_pred[idx:idx + duration, 1]) - np.sum(y_pred[idx:idx + duration, 0]))
                            / (np.sum(y_pred[idx:idx + duration, 0]) + np.sum(y_pred[idx:idx + duration, 1])))

                if prev_cue != behavior['cue_codes'][i]:
                    reactivation_diff.append(fr * np.sum(y_pred_all[idx:idx + duration]) /
                                             np.sum(times_considered[idx:idx + duration]))
                    pupil_diff.append((np.mean(behavior['pupil'][idx:idx + int(fr * 5)]) -
                                       np.mean(behavior['pupil'][idx - before:idx])) /
                                      np.mean(behavior['pupil'][idx - before:idx]))
                    cue_evoked_different[i, :] = sorted_deconvolved[idx - before:idx + duration]
                    if behavior['cue_codes'][i] == behavior['cs_1_code']:
                        bias_diff.append((np.sum(y_pred[idx:idx + duration, 0]) - np.sum(y_pred[idx:idx + duration, 1]))
                                         / (np.sum(y_pred[idx:idx + duration, 0]) + np.sum(
                            y_pred[idx:idx + duration, 1])))
                    if behavior['cue_codes'][i] == behavior['cs_2_code']:
                        bias_diff.append(
                            (np.sum(y_pred[idx:idx + duration, 1]) - np.sum(y_pred[idx:idx + duration, 0]))
                            / (np.sum(y_pred[idx:idx + duration, 0]) + np.sum(y_pred[idx:idx + duration, 1])))

    days_path = paths['base_path'] + paths['mouse'] + '/data_across_days/'
    if days:
        if path.isfile(days_path + 'trial_history_' + str(num_prev) + '.npy') == 0 or day == 0:
            trial_history_across_days = [list(range(0, days)), list(range(0, days)), list(range(0, days)),
                                         list(range(0, days)), list(range(0, days)), list(range(0, days)),
                                         list(range(0, days)), list(range(0, days))]
            trial_history_across_days[0][day] = np.mean(reactivation_same)
            trial_history_across_days[1][day] = np.mean(reactivation_diff)
            trial_history_across_days[2][day] = np.mean(pupil_same)
            trial_history_across_days[3][day] = np.mean(pupil_diff)
            trial_history_across_days[4][day] = np.nanmean(cue_evoked_same, axis=0)
            trial_history_across_days[5][day] = np.nanmean(cue_evoked_different, axis=0)
            trial_history_across_days[6][day] = np.nanmean(bias_same)
            trial_history_across_days[7][day] = np.nanmean(bias_diff)
            np.save(days_path + 'trial_history_' + str(num_prev), trial_history_across_days)
        else:
            trial_history_across_days = np.load(days_path + 'trial_history_' + str(num_prev) + '.npy',
                                                allow_pickle=True)
            trial_history_across_days[0][day] = np.mean(reactivation_same)
            trial_history_across_days[1][day] = np.mean(reactivation_diff)
            trial_history_across_days[2][day] = np.mean(pupil_same)
            trial_history_across_days[3][day] = np.mean(pupil_diff)
            trial_history_across_days[4][day] = np.nanmean(cue_evoked_same, axis=0)
            trial_history_across_days[5][day] = np.nanmean(cue_evoked_different, axis=0)
            trial_history_across_days[6][day] = np.nanmean(bias_same)
            trial_history_across_days[7][day] = np.nanmean(bias_diff)
            np.save(days_path + 'trial_history_' + str(num_prev), trial_history_across_days)


def pupil_reactivation_modulation(behavior, y_pred, paths, day, days):
    """
    get reactivation rate during high or low pupil
    :param behavior: behavior
    :param y_pred: y pred
    :param paths: path
    :param day: day
    :param days: days
    :return: rates based on pupil
    """
    all_y_pred = y_pred[:, 0] + y_pred[:, 1]
    cue_start = behavior['onsets']
    cue_end = behavior['offsets']
    fr = behavior['framerate']
    duration = int(fr * (behavior['iti'] + 5)) + 1
    pupil_vec = behavior['pupil']
    reactivation_prob = []
    pupil_pre_cue = []
    pupil_cue = []
    pupil_post_cue = []
    pupil_iti = []
    pupil_cue_evoked = []
    pupil_post_cue_evoked = []
    end_trials = behavior['end_trials']
    times_considered = preprocess.get_times_considered(y_pred, behavior)
    for i in range(0, len(cue_start)):

        pre_pupil = np.mean(pupil_vec[int(cue_start[i]) - int(behavior['framerate'] * 4):int(cue_start[i])])
        temp_pupil_cue = np.mean(pupil_vec[int(cue_start[i]):int(cue_end[i]) + 1])
        temp_pupil_cue_post = np.mean(
            pupil_vec[int(cue_start[i]) + 1:int(cue_start[i]) + 1 + int(behavior['framerate'] * 8)])
        temp_pupil_cue_evoked = (temp_pupil_cue - pre_pupil) / pre_pupil
        temp_pupil_post_evoked = (temp_pupil_cue_post - pre_pupil) / pre_pupil
        temp_pupil_iti = np.mean(
            pupil_vec[int(cue_start[i]) + int(behavior['framerate'] * 8):int(cue_start[i]) + duration])
        temp_reactivation = np.sum(
            all_y_pred[int(cue_start[i]) + int(behavior['framerate'] * 8):int(cue_start[i]) + duration])
        temp_times_considered_iti = np.sum(
            times_considered[int(cue_start[i]) + int(behavior['framerate'] * 8):int(cue_start[i]) + duration])
        temp_reactivation = (temp_reactivation / temp_times_considered_iti) * behavior['framerate']

        if i not in end_trials:
            pupil_pre_cue.append(pre_pupil)
            pupil_cue.append(temp_pupil_cue)
            pupil_post_cue.append(temp_pupil_cue_post)
            pupil_cue_evoked.append(temp_pupil_cue_evoked)
            pupil_post_cue_evoked.append(temp_pupil_post_evoked)
            pupil_iti.append(temp_pupil_iti)
            reactivation_prob.append(temp_reactivation[0][0])

    num_trials_total = len(pupil_cue)
    num_trials = round(len(pupil_cue) / 10)
    all_data = pd.DataFrame({'pupil_pre_cue': pupil_pre_cue, 'pupil_cue': pupil_cue, 'pupil_post_cue': pupil_post_cue,
                             'pupil_cue_evoked': pupil_cue_evoked, 'pupil_post_cue_evoked': pupil_post_cue_evoked,
                             'pupil_iti': pupil_iti, 'reactivation_prob': reactivation_prob})
    all_data = all_data.sort_values(by='pupil_pre_cue')
    pupil_pre_cue_vec = [all_data['reactivation_prob'][0:num_trials].mean(),
                         all_data['reactivation_prob'][num_trials_total - num_trials:num_trials_total].mean()]
    all_data = all_data.sort_values(by='pupil_cue')
    pupil_cue_vec = [all_data['reactivation_prob'][0:num_trials].mean(),
                     all_data['reactivation_prob'][num_trials_total - num_trials:num_trials_total].mean()]
    all_data = all_data.sort_values(by='pupil_post_cue')
    pupil_post_cue_vec = [all_data['reactivation_prob'][0:num_trials].mean(),
                          all_data['reactivation_prob'][num_trials_total - num_trials:num_trials_total].mean()]
    all_data = all_data.sort_values(by='pupil_cue_evoked')
    pupil_cue_evoked_vec = [all_data['reactivation_prob'][0:num_trials].mean(),
                            all_data['reactivation_prob'][num_trials_total - num_trials:num_trials_total].mean()]
    all_data = all_data.sort_values(by='pupil_post_cue_evoked')
    pupil_post_cue_evoked_vec = [all_data['reactivation_prob'][0:num_trials].mean(),
                                 all_data['reactivation_prob'][num_trials_total - num_trials:num_trials_total].mean()]
    all_data = all_data.sort_values(by='pupil_iti')
    pupil_iti_vec = [all_data['reactivation_prob'][0:num_trials].mean(),
                     all_data['reactivation_prob'][num_trials_total - num_trials:num_trials_total].mean()]

    days_path = paths['base_path'] + paths['mouse'] + '/data_across_days/'
    if days:
        if path.isfile(days_path + 'pupil_reactivation.npy') == 0 or day == 0:
            pupil_reactivation_across_days = [list(range(0, days)), list(range(0, days)), list(range(0, days)),
                                              list(range(0, days)), list(range(0, days)), list(range(0, days))]
            pupil_reactivation_across_days[0][day] = pupil_pre_cue_vec
            pupil_reactivation_across_days[1][day] = pupil_cue_vec
            pupil_reactivation_across_days[2][day] = pupil_post_cue_vec
            pupil_reactivation_across_days[3][day] = pupil_cue_evoked_vec
            pupil_reactivation_across_days[4][day] = pupil_post_cue_evoked_vec
            pupil_reactivation_across_days[5][day] = pupil_iti_vec
            np.save(days_path + 'pupil_reactivation', pupil_reactivation_across_days)
        else:
            pupil_reactivation_across_days = np.load(days_path + 'pupil_reactivation.npy', allow_pickle=True)
            pupil_reactivation_across_days[0][day] = pupil_pre_cue_vec
            pupil_reactivation_across_days[1][day] = pupil_cue_vec
            pupil_reactivation_across_days[2][day] = pupil_post_cue_vec
            pupil_reactivation_across_days[3][day] = pupil_cue_evoked_vec
            pupil_reactivation_across_days[4][day] = pupil_post_cue_evoked_vec
            pupil_reactivation_across_days[5][day] = pupil_iti_vec
            np.save(days_path + 'pupil_reactivation', pupil_reactivation_across_days)


def reactivation_physical(y_pred, behavior, paths, day, days):
    """
    makes plot of physical evoked reactivations
    :param y_pred: y pred
    :param behavior: behavior
    :param paths: path to data
    :param day: day
    :param days: days
    :return: plot
    """

    p_threshold = .75
    reactivation_cs_1 = y_pred[:, 0]
    reactivation_cs_2 = y_pred[:, 1]
    cs_1_peak = 0
    cs_2_peak = 0
    i = 0
    next_r = 0
    r_start = []
    r_times = []
    while i < len(reactivation_cs_1):
        if reactivation_cs_1[i] > 0 or reactivation_cs_2[i] > 0:
            if next_r == 0:
                r_start = i
                next_r = 1
            if reactivation_cs_1[i] > cs_1_peak:
                cs_1_peak = reactivation_cs_1[i]
            if reactivation_cs_2[i] > cs_2_peak:
                cs_2_peak = reactivation_cs_2[i]
            if reactivation_cs_1[i + 1] == 0 and reactivation_cs_2[i + 1] == 0:
                next_r = 0
                if cs_1_peak > p_threshold or cs_2_peak > p_threshold:
                    r_times.append(r_start)
                cs_1_peak = 0
                cs_2_peak = 0
        i += 1

    plt.figure(figsize=(12, 10))
    sns.set(font_scale=1.5)
    sns.set_style("white")
    mpl.rcParams['axes.edgecolor'] = 'white'
    mean_pupil = make_reactivation_physical('pupil', r_times, behavior, 1, 'blue',
                                            'Normalized pupil area (a.u.)')
    mean_pupil_movement = make_reactivation_physical('pupil_movement', r_times, behavior, 3, 'red',
                                                     'Pupil movement')
    mean_brain_motion = make_reactivation_physical('brain_motion', r_times, behavior, 5, 'darkgoldenrod',
                                                   'Brain motion')
    plt.savefig(paths['save_path'] + 'plots/' + paths['mouse'] + '_' + paths['date'] + '_'
                + 'reactivation_physical.png', bbox_inches='tight', dpi=150)
    plt.close()

    days_path = paths['base_path'] + paths['mouse'] + '/data_across_days/'
    if days:
        if path.isfile(days_path + 'reactivation_physical.npy') == 0 or day == 0:
            reactivation_physical_across_days = [list(range(0, days)), list(range(0, days)), list(range(0, days))]
            reactivation_physical_across_days[0][day] = mean_pupil
            reactivation_physical_across_days[1][day] = mean_pupil_movement
            reactivation_physical_across_days[2][day] = mean_brain_motion
            np.save(days_path + 'reactivation_physical', reactivation_physical_across_days)
        else:
            reactivation_physical_across_days = np.load(days_path + 'reactivation_physical.npy', allow_pickle=True)
            reactivation_physical_across_days[0][day] = mean_pupil
            reactivation_physical_across_days[1][day] = mean_pupil_movement
            reactivation_physical_across_days[2][day] = mean_brain_motion
            np.save(days_path + 'reactivation_physical', reactivation_physical_across_days)


def make_reactivation_physical(physical_type, reactivation_times, behavior, idx, c, y_label):
    """
    makes physical plot
    :param physical_type: type
    :param reactivation_times: reactivation times
    :param behavior: behavior
    :param idx: which subplot
    :param c: color
    :param y_label: y label
    :return: plot
    """
    time_window = int(behavior['framerate'] * 20)
    physical_data = np.zeros((len(reactivation_times), (time_window * 2) + 1))
    for i in range(0, len(reactivation_times)):
        time = reactivation_times[i]
        if time_window + 1 < time < len(behavior[physical_type]) - (time_window+1):
            if physical_type == 'pupil':
                temp_physical = behavior[physical_type][time - time_window:time + time_window + 1] / behavior['pupil_max']
                physical_data[i, :] = temp_physical
            elif physical_type == 'pupil_movement':
                temp_physical = behavior[physical_type][time - time_window:time + time_window + 1] / behavior[
                    'pupil_movement_max']
                physical_data[i, :] = temp_physical
            else:
                physical_data[i, :] = behavior[physical_type][time - time_window:time + time_window + 1]

    color_map = LinearSegmentedColormap.from_list('mycmap', ['royalblue', 'white', 'crimson'])
    plt.subplot(3, 2, idx)
    ax = sns.heatmap(physical_data, cmap=color_map, cbar_kws={'label': ''})
    ax.set_xticks([int(behavior['framerate'] * 0), int(behavior['framerate'] * 10), int(behavior['framerate'] * 20),
                   int(behavior['framerate'] * 30), int(behavior['framerate'] * 40)])
    ax.set_xticklabels(['-20', '-10', '0', '10', '20'], rotation=0)
    ax.set_yticks([0, 20, 40, 60, 80, 100, 120])
    ax.set_yticklabels(['0', '20', '40', '60', '80', '100', '120'], rotation=0)
    ax.set_ylim(len(physical_data), 0)
    ax.set(ylabel='Reactivation number')
    if idx == 5:
        ax.set(xlabel='Time relative to reactivation (s)')
    plt.axvline(x=time_window, color='k', linestyle='--')
    mpl.rcParams['axes.edgecolor'] = 'black'
    plt.subplot(3, 2, idx+1)
    mean_physical = np.mean(physical_data, axis=0)
    plt.plot(mean_physical, c=c)
    sem_plus = mean_physical + stats.sem(physical_data, axis=0, nan_policy='omit')
    sem_minus = mean_physical - stats.sem(physical_data, axis=0, nan_policy='omit')
    plt.fill_between(np.arange(mean_physical.shape[0]), sem_plus, sem_minus, alpha=0.2, color=c)
    plt.xticks([int(behavior['framerate'] * 0), int(behavior['framerate'] * 10), int(behavior['framerate'] * 20),
                int(behavior['framerate'] * 30), int(behavior['framerate'] * 40)], ['-20', '-10', '0', '10', '20'])
    plt.ylabel(y_label)
    if idx == 5:
        plt.xlabel('Time relative to reactivation (s)')
    plt.axvline(x=time_window, color='k', linestyle='--')
    mpl.rcParams['axes.edgecolor'] = 'black'
    sns.despine()
    plt.subplots_adjust(right=1.2)
    return mean_physical


def activity_across_trials(norm_deconvolved, behavior, y_pred, idx, paths, day, days):
    """
    make correlation similarity and activity across trials
    :param norm_deconvolved: activity
    :param behavior: behavior
    :param y_pred: reactivation probabilities
    :param idx: index of sorted cells
    :param paths: path to save
    :param day: day
    :param days: days
    :return: correlation and activity and reactivations per trial
    """
    activity = norm_deconvolved.to_numpy()
    activity = activity[idx['both'].index]

    sig_cells = preprocess.sig_reactivated_cells([], [], [], [], [], paths, 0)
    activity = activity[sig_cells > 0, :]

    times_considered = preprocess.get_times_considered(y_pred, behavior)
    y_pred = true_reactivations(y_pred)
    y_pred_cs_1 = y_pred[:, 0]
    y_pred_cs_2 = y_pred[:, 1]
    all_y_pred = y_pred_cs_1 + y_pred_cs_2
    cue_start = behavior['onsets']
    cue_end = behavior['offsets']
    end_trials = behavior['end_trials']
    reactivation_prob = []
    correlation = []
    cue_activity = []

    start = max(list(behavior['cue_codes']).index(behavior['cs_1_code']), list(behavior['cue_codes']).
                index(behavior['cs_2_code']))
    past_cs_1_type = behavior['cue_codes'][start - 1]
    past_cs_2_type = behavior['cue_codes'][start]
    past_cs_1_mean = np.mean(activity[:, int(cue_start[start - 1]):int(cue_end[start - 1]) + 1], axis=1)
    past_cs_2_mean = np.mean(activity[:, int(cue_start[start]):int(cue_end[start]) + 1], axis=1)
    for i in range(start+1, len(cue_start)):
        current_cs_type = behavior['cue_codes'][i]
        if current_cs_type == past_cs_1_type:
            current_temp_mean = np.mean(activity[:, int(cue_start[i]):int(cue_end[i]) + 1], axis=1)
            if i not in end_trials:
                correlation.append(np.corrcoef(current_temp_mean, past_cs_2_mean)[0][1])
            past_cs_1_mean = current_temp_mean
        if current_cs_type == past_cs_2_type:
            current_temp_mean = np.mean(activity[:, int(cue_start[i]):int(cue_end[i]) + 1], axis=1)
            if i not in end_trials:
                correlation.append(np.corrcoef(current_temp_mean, past_cs_1_mean)[0][1])
            past_cs_2_mean = current_temp_mean
        temp_sum_reactivation = np.sum(all_y_pred[int(cue_start[i]):int(cue_start[i]) + int((behavior['framerate'] *
                                                                                             (behavior['iti'] + 6)))]) / np.sum(times_considered[int(cue_start[i]):int(cue_start[i]) + int((behavior['framerate'] *
                                                                                             (behavior['iti'] + 6)))]) * int(behavior['framerate'])
        temp_cue_activity = np.mean(activity[:, int(cue_start[i]):int(cue_end[i]) + 1])
        if i not in end_trials:
            reactivation_prob.append(temp_sum_reactivation)
            cue_activity.append(temp_cue_activity)

    reactivation_prob_cs_1 = []
    reactivation_prob_cs_2 = []
    correlation_cs_1 = []
    correlation_cs_2 = []
    start_cs_1 = list(behavior['cue_codes']).index(behavior['cs_1_code'])
    start_cs_2 = list(behavior['cue_codes']).index(behavior['cs_2_code'])
    past_cs_1_mean = np.mean(activity[:, int(cue_start[start_cs_1]):int(cue_end[start_cs_1]) + 1], axis=1)
    past_cs_2_mean = np.mean(activity[:, int(cue_start[start_cs_2]):int(cue_end[start_cs_2]) + 1], axis=1)
    for i in range(0, len(cue_start)):
        if i not in end_trials:
            if behavior['cue_codes'][i] == behavior['cs_1_code'] and i > start_cs_1:
                current_temp_mean = np.mean(activity[:, int(cue_start[i]):int(cue_end[i]) + 1], axis=1)
                correlation_cs_1.append(np.corrcoef(current_temp_mean, past_cs_1_mean)[0][1])
                reactivation_prob_cs_1.append(
                    np.sum(y_pred_cs_1[int(cue_start[i]):int(cue_start[i]) + int((behavior['framerate'] *
                                                                                 (behavior['iti'] + 6)))]))
                # past_cs_1_mean = current_temp_mean
            if behavior['cue_codes'][i] == behavior['cs_2_code'] and i > start_cs_2:
                current_temp_mean = np.mean(activity[:, int(cue_start[i]):int(cue_end[i]) + 1], axis=1)
                correlation_cs_2.append(np.corrcoef(current_temp_mean, past_cs_2_mean)[0][1])
                reactivation_prob_cs_2.append(
                    np.sum(y_pred_cs_2[int(cue_start[i]):int(cue_start[i]) + int((behavior['framerate'] *
                                                                                  (behavior['iti'] + 6)))]))
                # past_cs_2_mean = current_temp_mean
    days_path = paths['base_path'] + paths['mouse'] + '/data_across_days/'
    if days:
        if path.isfile(days_path + 'activity.npy') == 0 or day == 0:
            correlation_across_days = [list(range(0, days)), list(range(0, days)), list(range(0, days)),
                                       list(range(0, days)), list(range(0, days)), list(range(0, days)),
                                       list(range(0, days))]
            correlation_across_days[0][day] = correlation
            correlation_across_days[1][day] = reactivation_prob
            correlation_across_days[2][day] = cue_activity
            correlation_across_days[3][day] = correlation_cs_1
            correlation_across_days[4][day] = correlation_cs_2
            correlation_across_days[5][day] = reactivation_prob_cs_1
            correlation_across_days[6][day] = reactivation_prob_cs_2
            np.save(days_path + 'activity', correlation_across_days)
        else:
            correlation_across_days = np.load(days_path + 'activity.npy', allow_pickle=True)
            correlation_across_days[0][day] = correlation
            correlation_across_days[1][day] = reactivation_prob
            correlation_across_days[2][day] = cue_activity
            correlation_across_days[3][day] = correlation_cs_1
            correlation_across_days[4][day] = correlation_cs_2
            correlation_across_days[5][day] = reactivation_prob_cs_1
            correlation_across_days[6][day] = reactivation_prob_cs_2
            np.save(days_path + 'activity', correlation_across_days)


def activity_across_trials_grouped(norm_deconvolved, behavior, idx, paths, day, days):
    """
    make correlation similarity and activity across trials
    :param norm_deconvolved: activity
    :param behavior: behavior
    :param idx: index of sorted cells
    :param paths: path to save
    :param day: day
    :param days: days
    :return: correlation and activity and reactivations per trial
    """
    activity = norm_deconvolved.to_numpy()
    activity = activity[idx['both'].index]
    sig_cells = preprocess.sig_reactivated_cells([], [], [], [], [], paths, 0)

    activity = activity[sig_cells > 0, :]
    [no_change_cells_cs_1, increase_sig_cells_cs_1, decrease_sig_cells_cs_1] = preprocess.group_neurons(activity, behavior, 'cs_1')

    [no_change_cells_cs_2, increase_sig_cells_cs_2, decrease_sig_cells_cs_2] = preprocess.group_neurons(activity, behavior, 'cs_2')

    all_correlation = []
    all_cue_activity = []
    for g in range(0, 3):
        activity = norm_deconvolved.to_numpy()
        activity = activity[idx['both'].index]
        sig_cells = preprocess.sig_reactivated_cells([], [], [], [], [], paths, 0)

        activity = activity[sig_cells > 0, :]
        if g == 0:
            cells_to_use = no_change_cells_cs_1 + no_change_cells_cs_2
            activity = activity[cells_to_use == 2, :]
            # cells_to_use = decrease_sig_cells_cs_1 + decrease_sig_cells_cs_2 + increase_sig_cells_cs_1 + increase_sig_cells_cs_2
            # activity = activity[cells_to_use > 0, :]
        if g == 1:
            cells_to_use = increase_sig_cells_cs_1 + increase_sig_cells_cs_2
            activity = activity[cells_to_use > 0, :]
            # cells_to_use_1 = decrease_sig_cells_cs_1 + decrease_sig_cells_cs_2
            # cells_to_use_1[cells_to_use_1 > 0] = 1
            # cells_to_use_2 = no_change_cells_cs_1 + no_change_cells_cs_2
            # cells_to_use_2[cells_to_use_2 != 2] = 0
            # cells_to_use_2[cells_to_use_2 == 2] = 1
            # cells_to_use = cells_to_use_1 + cells_to_use_2
            # activity = activity[cells_to_use > 0, :]
        if g == 2:
            cells_to_use = decrease_sig_cells_cs_1 + decrease_sig_cells_cs_2
            activity = activity[cells_to_use > 0, :]
            # cells_to_use_1 = increase_sig_cells_cs_1 + increase_sig_cells_cs_2
            # cells_to_use_1[cells_to_use_1 > 0] = 1
            # cells_to_use_2 = no_change_cells_cs_1 + no_change_cells_cs_2
            # cells_to_use_2[cells_to_use_2 != 2] = 0
            # cells_to_use_2[cells_to_use_2 == 2] = 1
            # cells_to_use = cells_to_use_1 + cells_to_use_2
            # activity = activity[cells_to_use > 0, :]

        cue_start = behavior['onsets']
        cue_end = behavior['offsets']
        end_trials = behavior['end_trials']
        correlation = []
        cue_activity = []

        start = max(list(behavior['cue_codes']).index(behavior['cs_1_code']), list(behavior['cue_codes']).
                    index(behavior['cs_2_code']))
        past_cs_1_type = behavior['cue_codes'][start - 1]
        past_cs_2_type = behavior['cue_codes'][start]
        past_cs_1_mean = np.mean(activity[:, int(cue_start[start - 1]):int(cue_end[start - 1]) + 1], axis=1)
        past_cs_2_mean = np.mean(activity[:, int(cue_start[start]):int(cue_end[start]) + 1], axis=1)
        for i in range(start+1, len(cue_start)):
            current_cs_type = behavior['cue_codes'][i]
            if current_cs_type == past_cs_1_type:
                current_temp_mean = np.mean(activity[:, int(cue_start[i]):int(cue_end[i]) + 1], axis=1)
                if i not in end_trials:
                    correlation.append(np.corrcoef(current_temp_mean, past_cs_2_mean)[0][1])
                past_cs_1_mean = current_temp_mean
            if current_cs_type == past_cs_2_type:
                current_temp_mean = np.mean(activity[:, int(cue_start[i]):int(cue_end[i]) + 1], axis=1)
                if i not in end_trials:
                    correlation.append(np.corrcoef(current_temp_mean, past_cs_1_mean)[0][1])
                past_cs_2_mean = current_temp_mean
            temp_cue_activity = np.mean(activity[:, int(cue_start[i]):int(cue_end[i]) + 1])
            if i not in end_trials:
                cue_activity.append(temp_cue_activity)
        all_correlation.append(correlation)
        all_cue_activity.append(cue_activity)
    days_path = paths['base_path'] + paths['mouse'] + '/data_across_days/'
    if days:
        if path.isfile(days_path + 'activity_grouped.npy') == 0 or day == 0:
            correlation_across_days = [list(range(0, days)), list(range(0, days))]
            correlation_across_days[0][day] = all_correlation
            correlation_across_days[1][day] = all_cue_activity
            np.save(days_path + 'activity_grouped', correlation_across_days)
        else:
            correlation_across_days = np.load(days_path + 'activity_grouped.npy', allow_pickle=True)
            correlation_across_days[0][day] = all_correlation
            correlation_across_days[1][day] = all_cue_activity
            np.save(days_path + 'activity_grouped', correlation_across_days)


def activity_across_trials_grouped_decrease(norm_deconvolved, behavior, idx, paths, day, days):
    """
    make correlation similarity and activity across trials
    :param norm_deconvolved: activity
    :param behavior: behavior
    :param idx: index of sorted cells
    :param paths: path to save
    :param day: day
    :param days: days
    :return: correlation and activity and reactivations per trial
    """
    activity = norm_deconvolved.to_numpy()
    activity = activity[idx['both'].index]
    sig_cells = preprocess.sig_reactivated_cells([], [], [], [], [], paths, 0)

    activity = activity[sig_cells > 0, :]
    [no_change_cells_cs_1, increase_sig_cells_cs_1, decrease_sig_cells_cs_1] = preprocess.group_neurons(activity, behavior, 'cs_1')

    [no_change_cells_cs_2, increase_sig_cells_cs_2, decrease_sig_cells_cs_2] = preprocess.group_neurons(activity, behavior, 'cs_2')

    cs1d_cs1 = []
    cs1d_cs2 = []
    cs2d_cs1 = []
    cs2d_cs2 = []
    activity = norm_deconvolved.to_numpy()
    activity = activity[idx['both'].index]
    sig_cells = preprocess.sig_reactivated_cells([], [], [], [], [], paths, 0)
    activity = activity[sig_cells > 0, :]

    cells_to_use_1 = no_change_cells_cs_1
    cells_to_use_2 = no_change_cells_cs_2

    activity_1 = activity[cells_to_use_1 > 0, :]
    activity_2 = activity[cells_to_use_2 > 0, :]

    cue_start = behavior['onsets']
    cue_end = behavior['offsets']
    end_trials = behavior['end_trials']
    start = max(list(behavior['cue_codes']).index(behavior['cs_1_code']), list(behavior['cue_codes']).
                index(behavior['cs_2_code']))
    for i in range(start+1, len(cue_start)):
        if i not in end_trials:
            if behavior['cue_codes'][i] == behavior['cs_1_code']:
                cs1d_cs1.append(np.mean(activity_1[:, int(cue_start[i]):int(cue_end[i]) + 1]))
                cs2d_cs1.append(np.mean(activity_2[:, int(cue_start[i]):int(cue_end[i]) + 1]))
            if behavior['cue_codes'][i] == behavior['cs_2_code']:
                cs1d_cs2.append(np.mean(activity_1[:, int(cue_start[i]):int(cue_end[i]) + 1]))
                cs2d_cs2.append(np.mean(activity_2[:, int(cue_start[i]):int(cue_end[i]) + 1]))
    days_path = paths['base_path'] + paths['mouse'] + '/data_across_days/'
    if days:
        if path.isfile(days_path + 'activity_grouped_no_change.npy') == 0 or day == 0:
            across_days = [list(range(0, days)), list(range(0, days)), list(range(0, days)),
                           list(range(0, days))]
            across_days[0][day] = cs1d_cs1
            across_days[1][day] = cs1d_cs2
            across_days[2][day] = cs2d_cs2
            across_days[3][day] = cs2d_cs1
            np.save(days_path + 'activity_grouped_no_change', across_days)
        else:
            across_days = np.load(days_path + 'activity_grouped_no_change.npy', allow_pickle=True)
            across_days[0][day] = cs1d_cs1
            across_days[1][day] = cs1d_cs2
            across_days[2][day] = cs2d_cs2
            across_days[3][day] = cs2d_cs1
            np.save(days_path + 'activity_grouped_no_change', across_days)


def cue_reactivation_activity(norm_deconvolved, behavior, y_pred, idx, paths, day, days):
    """
    make correlation similarity and activity across trials
    :param norm_deconvolved: activity
    :param behavior: behavior
    :param y_pred: reactivation probabilities
    :param idx: index of sorted cells
    :param paths: path to save
    :param day: day
    :param days: days
    :return: correlation and activity and reactivations per trial
    """
    activity = norm_deconvolved.to_numpy()
    activity = activity[idx['both'].index]
    sig_cells = preprocess.sig_reactivated_cells([], [], [], [], [], paths, 0)
    activity = activity[sig_cells > 0, :]

    reactivation_cs_1 = y_pred[:, 0].copy()
    reactivation_cs_2 = y_pred[:, 1].copy()
    p_threshold = .75
    cs_1_peak = 0
    cs_2_peak = 0
    i = 0
    reactivation_times_cs_1 = np.zeros(len(reactivation_cs_1))
    reactivation_times_cs_2 = np.zeros(len(reactivation_cs_1))
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
                if cs_1_peak > p_threshold and r_start > int(behavior['onsets'][0]):
                    reactivation_times_cs_1[r_start:r_end] = 1
                if cs_2_peak > p_threshold and r_start > int(behavior['onsets'][0]):
                    reactivation_times_cs_2[r_start:r_end] = 1
                i = r_end
                cs_1_peak = 0
                cs_2_peak = 0
    cue_time = preprocess.cue_times(behavior, 0, 0)

    activity_cs_1_r = np.mean(activity[:, reactivation_times_cs_1 == 1], axis=1) * behavior['framerate']
    activity_cs_2_r = np.mean(activity[:, reactivation_times_cs_2 == 1], axis=1) * behavior['framerate']
    activity_cs_1 = np.mean(activity[:, cue_time[0] == 1], axis=1) * behavior['framerate']
    activity_cs_2 = np.mean(activity[:, cue_time[0] == 2], axis=1) * behavior['framerate']

    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(9, 6))
    plt.subplots_adjust(wspace=.35)
    m_size = 10
    w_size = len(activity_cs_1[0])
    plt.subplot(2, 2, 1)
    xx = activity_cs_1[0]
    yy = activity_cs_1_r[0]
    a, b = np.polyfit(xx, yy, 1)
    x_plot_s1 = []
    y_plot_s1 = []
    for x in range(0, 250):
        x = x / 100
        y = (a*x) + b
        x_plot_s1.append(x)
        y_plot_s1.append(y)
    plt.plot(x_plot_s1, y_plot_s1, c='mediumseagreen', linewidth=3)
    plt.scatter(x=activity_cs_1[0], y=activity_cs_1_r[0], color='mediumseagreen', s=m_size)
    plt.xlim(0, 2.5)
    plt.xlabel('Stimulus 1 activity\n(Normalized $\mathregular{Ca^{2+}}$transient $\mathregular{s^{-1}}$)')
    plt.ylabel('Stimulus 1 reactivation activity\n(Normalized $\mathregular{Ca^{2+}}$transient $\mathregular{s^{-1}}$)')
    plt.subplot(2, 2, 2)
    xx = activity_cs_2[0]
    yy = activity_cs_2_r[0]
    a, b = np.polyfit(xx, yy, 1)
    x_plot_s2 = []
    y_plot_s2 = []
    for x in range(0, 250):
        x = x / 100
        y = (a*x) + b
        x_plot_s2.append(x)
        y_plot_s2.append(y)
    plt.plot(x_plot_s2, y_plot_s2, c='salmon', linewidth=3)
    plt.scatter(x=activity_cs_2[0], y=activity_cs_2_r[0], color='salmon', s=m_size)
    plt.xlim(0, 2.5)
    plt.xlabel('Stimulus 2 activity\n(Normalized $\mathregular{Ca^{2+}}$transient $\mathregular{s^{-1}}$)')
    plt.ylabel('Stimulus 2 reactivation activity\n(Normalized $\mathregular{Ca^{2+}}$transient $\mathregular{s^{-1}}$)')
    sns.despine()
    plt.savefig(paths['save_path'] + 'plots/' + paths['mouse'] + '_' + paths['date'] + '_' +
                'cue_reactivation_activity.png', bbox_inches='tight', dpi=200)
    plt.savefig(paths['save_path'] + 'plots/' + paths['mouse'] + '_' + paths['date'] + '_' +
                'cue_reactivation_activity.pdf', bbox_inches='tight', dpi=200, transparent=True)
    # plt.close()

    days_path = paths['base_path'] + paths['mouse'] + '/data_across_days/'
    if days:
        if path.isfile(days_path + 'cue_reactivation_activity.npy') == 0 or day == 0:
            correlation_across_days = [list(range(0, days)), list(range(0, days)), list(range(0, days)),
                                       list(range(0, days))]
            correlation_across_days[0][day] = y_plot_s1
            correlation_across_days[1][day] = y_plot_s2
            correlation_across_days[2][day] = x_plot_s1
            correlation_across_days[3][day] = x_plot_s2
            np.save(days_path + 'cue_reactivation_activity', correlation_across_days)
        else:
            correlation_across_days = np.load(days_path + 'cue_reactivation_activity.npy', allow_pickle=True)
            correlation_across_days[0][day] = y_plot_s1
            correlation_across_days[1][day] = y_plot_s2
            correlation_across_days[2][day] = x_plot_s1
            correlation_across_days[3][day] = x_plot_s2
            np.save(days_path + 'cue_reactivation_activity', correlation_across_days)


def iti_activity_bias(norm_deconvolved, norm_moving_deconvolved_filtered, behavior, y_pred, idx, paths, day, days):
    """
    plot bias in raw activity
    :param norm_deconvolved: activity
    :param norm_moving_deconvolved_filtered: filtered activity
    :param behavior: behavior
    :param y_pred: y pred
    :param idx: index
    :param paths: path
    :param day: day
    :param days: days
    :return: raw activity cs bias
    """
    activity_cs_1 = norm_deconvolved.reindex(idx['cs_1'].index[0:int(len(idx['cs_1'])/10)])
    activity_cs_2 = norm_deconvolved.reindex(idx['cs_2'].index[0:int(len(idx['cs_2'])/10)])
    activity_cs_1 = activity_cs_1.to_numpy()
    activity_cs_2 = activity_cs_2.to_numpy()
    iti_cs_1_activity_low = []
    iti_cs_2_activity_low = []
    iti_cs_1_activity_norm = []
    iti_cs_2_activity_norm = []
    cue_type = []
    cue_start = behavior['onsets']
    end_trials = behavior['end_trials']
    times_considered = preprocess.get_times_considered(y_pred, behavior)
    prior_low = classify.prior(norm_moving_deconvolved_filtered, idx['cs_1'], idx['cs_2'], behavior, 1)
    prior_norm = classify.prior(norm_moving_deconvolved_filtered, idx['cs_1'], idx['cs_2'], behavior, [])
    duration = int(behavior['framerate'] * (behavior['iti'] + 5)) + 1

    for i in range(0, len(cue_start)):
        temp_times_considered_iti = times_considered[int(cue_start[i]):int(cue_start[i])+duration]
        temp_prior_low_considered_iti = prior_low[int(cue_start[i]):int(cue_start[i])+duration]
        temp_prior_norm_considered_iti = prior_norm[int(cue_start[i]):int(cue_start[i])+duration]
        temp_iti_cs_1_activity = activity_cs_1[:, int(cue_start[i]):int(cue_start[i])+duration]
        temp_iti_cs_2_activity = activity_cs_2[:, int(cue_start[i]):int(cue_start[i])+duration]
        temp_times_considered_iti[temp_prior_low_considered_iti == 1] = 0
        if i not in end_trials:
            temp_iti_cs_1_activity_low = np.mean(temp_iti_cs_1_activity[:, temp_times_considered_iti == 1])
            temp_iti_cs_2_activity_low = np.mean(temp_iti_cs_2_activity[:, temp_times_considered_iti == 1])
            temp_iti_cs_1_activity_norm = np.mean(temp_iti_cs_1_activity[:, temp_prior_norm_considered_iti == 1])
            temp_iti_cs_2_activity_norm = np.mean(temp_iti_cs_2_activity[:, temp_prior_norm_considered_iti == 1])
            iti_cs_1_activity_low.append(temp_iti_cs_1_activity_low)
            iti_cs_2_activity_low.append(temp_iti_cs_2_activity_low)
            iti_cs_1_activity_norm.append(temp_iti_cs_1_activity_norm)
            iti_cs_2_activity_norm.append(temp_iti_cs_2_activity_norm)
            cue_type.append(behavior['cue_codes'][i])

    days_path = paths['base_path'] + paths['mouse'] + '/data_across_days/'
    if days:
        if path.isfile(days_path + 'iti_bias.npy') == 0 or day == 0:
            iti_bias_across_days = [list(range(0, days)), list(range(0, days)), list(range(0, days)),
                                    list(range(0, days)), list(range(0, days))]
            iti_bias_across_days[0][day] = iti_cs_1_activity_low
            iti_bias_across_days[1][day] = iti_cs_2_activity_low
            iti_bias_across_days[2][day] = iti_cs_1_activity_norm
            iti_bias_across_days[3][day] = iti_cs_2_activity_norm
            iti_bias_across_days[4][day] = cue_type
            np.save(days_path + 'iti_bias', iti_bias_across_days)
        else:
            iti_bias_across_days = np.load(days_path + 'iti_bias.npy', allow_pickle=True)
            iti_bias_across_days[0][day] = iti_cs_1_activity_low
            iti_bias_across_days[1][day] = iti_cs_2_activity_low
            iti_bias_across_days[2][day] = iti_cs_1_activity_norm
            iti_bias_across_days[3][day] = iti_cs_2_activity_norm
            iti_bias_across_days[4][day] = cue_type
            np.save(days_path + 'iti_bias', iti_bias_across_days)


def reactivation_duration(y_pred, behavior, paths, day, days):
    """
    gets length of reactivation
    :param y_pred: y pred
    :param behavior: behavior
    :param paths: path
    :param day: day
    :param days: days
    :return: length of reactivation pre vs post cues
    """
    p_threshold = .75
    reactivation_cs_1 = y_pred[:, 0]
    reactivation_cs_2 = y_pred[:, 1]
    cs_1_peak = 0
    cs_2_peak = 0
    i = 0
    next_r = 0
    r_start = []
    r_length_before = []
    r_length = []
    while i < len(reactivation_cs_1):
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
                if cs_1_peak > p_threshold or cs_2_peak > p_threshold:
                    if r_end < int(behavior['frames_per_run']):
                        r_length_before.append(r_end - r_start)
                    if r_start > int(behavior['frames_per_run']):
                        r_length.append(r_end - r_start)
                cs_1_peak = 0
                cs_2_peak = 0
        i += 1

    days_path = paths['base_path'] + paths['mouse'] + '/data_across_days/'
    if days:
        if path.isfile(days_path + 'reactivation_length.npy') == 0 or day == 0:
            length_across_days = [list(range(0, days)), list(range(0, days))]
            length_across_days[0][day] = np.mean(r_length)
            length_across_days[1][day] = np.mean(r_length_before)
            np.save(days_path + 'reactivation_length', length_across_days)
        else:
            length_across_days = np.load(days_path + 'reactivation_length.npy', allow_pickle=True)
            length_across_days[0][day] = np.mean(r_length)
            length_across_days[1][day] = np.mean(r_length_before)
            np.save(days_path + 'reactivation_length', length_across_days)


def reactivation_top_bottom_activity(norm_deconvolved, idx, y_pred, behavior, both_poscells, paths, day, days):
    """
    get mean activity during cue and reactivation of top vs bottom cue cues
    :param norm_deconvolved: activity
    :param idx: index
    :param y_pred: y pred
    :param behavior: behavior
    :param both_poscells: pos mod cells
    :param paths: path
    :param day: day
    :param days: days
    :return: mean activity vec of top and bottom cue cells
    """

    p_threshold = .75
    reactivation_cs_1 = y_pred[:, 0]
    reactivation_cs_2 = y_pred[:, 1]
    cs_1_peak = 0
    cs_2_peak = 0
    i = 0
    next_r = 0
    r_start = []
    r_times_cs_1 = np.zeros(len(reactivation_cs_1))
    r_times_cs_2 = np.zeros(len(reactivation_cs_2))
    while i < len(reactivation_cs_1):
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
                    r_times_cs_1[r_start:r_start+(r_end-r_start)] = 1
                if cs_2_peak > p_threshold:
                    r_times_cs_2[r_start:r_start+(r_end-r_start)] = 1
                cs_1_peak = 0
                cs_2_peak = 0
        i += 1

    cue_idx = preprocess.cue_times(behavior, 0, 0)
    cs_1_times = cue_idx[0] == 1
    cs_2_times = cue_idx[0] == 2

    baseline = preprocess.get_times_considered(y_pred, behavior)
    baseline[int(behavior['frames_per_run']):len(baseline)] = 0

    top_activity_cs_1 = norm_deconvolved.reindex(idx['cs_1'].index[0:int(len(idx['cs_1']) / 20)])
    top_activity_cs_2 = norm_deconvolved.reindex(idx['cs_2'].index[0:int(len(idx['cs_2']) / 20)])
    bottom_activity_cs_1 = norm_deconvolved.reindex(idx['cs_1'].index[int(len(idx['cs_1']) / 20):len(idx['cs_1'].index)])
    bottom_activity_cs_2 = norm_deconvolved.reindex(idx['cs_2'].index[int(len(idx['cs_2']) / 20):len(idx['cs_2'].index)])
    top_activity_cs_1 = top_activity_cs_1.to_numpy()
    top_activity_cs_2 = top_activity_cs_2.to_numpy()
    bottom_activity_cs_1 = bottom_activity_cs_1.to_numpy()
    bottom_activity_cs_2 = bottom_activity_cs_2.to_numpy()

    activity_other = norm_deconvolved.to_numpy()
    activity_other = activity_other[both_poscells == 0, :]

    top_cs_1_cs_1r = np.mean(top_activity_cs_1[:, r_times_cs_1 == 1])
    top_cs_2_cs_1r = np.mean(top_activity_cs_2[:, r_times_cs_1 == 1])
    bottom_cs_1_cs_1r = np.mean(bottom_activity_cs_1[:, r_times_cs_1 == 1])
    bottom_cs_2_cs_1r = np.mean(bottom_activity_cs_2[:, r_times_cs_1 == 1])
    all_other_cs_1r = np.mean(activity_other[:, r_times_cs_1 == 1])

    top_cs_1_cs_1c = np.mean(top_activity_cs_1[:, cs_1_times])
    top_cs_2_cs_1c = np.mean(top_activity_cs_2[:, cs_1_times])
    bottom_cs_1_cs_1c = np.mean(bottom_activity_cs_1[:, cs_1_times])
    bottom_cs_2_cs_1c = np.mean(bottom_activity_cs_2[:, cs_1_times])
    all_other_cs_1c = np.mean(activity_other[:, cs_1_times])

    top_cs_1_cs_2r = np.mean(top_activity_cs_1[:, r_times_cs_2 == 1])
    top_cs_2_cs_2r = np.mean(top_activity_cs_2[:, r_times_cs_2 == 1])
    bottom_cs_2_cs_2r = np.mean(bottom_activity_cs_2[:, r_times_cs_2 == 1])
    bottom_cs_1_cs_2r = np.mean(bottom_activity_cs_1[:, r_times_cs_2 == 1])
    all_other_cs_2r = np.mean(activity_other[:, r_times_cs_2 == 1])

    top_cs_1_cs_2c = np.mean(top_activity_cs_1[:, cs_2_times])
    top_cs_2_cs_2c = np.mean(top_activity_cs_2[:, cs_2_times])
    bottom_cs_2_cs_2c = np.mean(bottom_activity_cs_2[:, cs_2_times])
    bottom_cs_1_cs_2c = np.mean(bottom_activity_cs_1[:, cs_2_times])
    all_other_cs_2c = np.mean(activity_other[:, cs_2_times])

    top_cs_1_b = np.mean(top_activity_cs_1[:, baseline == 1])
    bottom_cs_1_b = np.mean(bottom_activity_cs_1[:, baseline == 1])
    all_other_cs_1_b = np.mean(activity_other[:, baseline == 1])
    top_cs_2_b = np.mean(top_activity_cs_2[:, baseline == 1])
    bottom_cs_2_b = np.mean(bottom_activity_cs_2[:, baseline == 1])
    all_other_cs_2_b = np.mean(activity_other[:, baseline == 1])

    days_path = paths['base_path'] + paths['mouse'] + '/data_across_days/'
    if days:
        if path.isfile(days_path + 'reactivation_top_bottom_activity.npy') == 0 or day == 0:
            reactivation_top_bottom = [list(range(0, days)), list(range(0, days)), list(range(0, days)),
                                       list(range(0, days)), list(range(0, days)), list(range(0, days)),
                                       list(range(0, days)), list(range(0, days)), list(range(0, days)),
                                       list(range(0, days)), list(range(0, days)), list(range(0, days)),
                                       list(range(0, days)), list(range(0, days)), list(range(0, days)),
                                       list(range(0, days)), list(range(0, days)), list(range(0, days)),
                                       list(range(0, days)), list(range(0, days)), list(range(0, days)),
                                       list(range(0, days)), list(range(0, days)), list(range(0, days)),
                                       list(range(0, days)), list(range(0, days))]
            reactivation_top_bottom[0][day] = top_cs_1_cs_1r
            reactivation_top_bottom[1][day] = top_cs_2_cs_1r
            reactivation_top_bottom[2][day] = bottom_cs_1_cs_1r
            reactivation_top_bottom[3][day] = bottom_cs_2_cs_1r
            reactivation_top_bottom[4][day] = all_other_cs_1r

            reactivation_top_bottom[5][day] = top_cs_1_cs_1c
            reactivation_top_bottom[6][day] = top_cs_2_cs_1c
            reactivation_top_bottom[7][day] = bottom_cs_1_cs_1c
            reactivation_top_bottom[8][day] = bottom_cs_2_cs_1c
            reactivation_top_bottom[9][day] = all_other_cs_1c

            reactivation_top_bottom[10][day] = top_cs_1_b
            reactivation_top_bottom[11][day] = bottom_cs_1_b
            reactivation_top_bottom[12][day] = all_other_cs_1_b

            reactivation_top_bottom[13][day] = top_cs_1_cs_2r
            reactivation_top_bottom[14][day] = top_cs_2_cs_2r
            reactivation_top_bottom[15][day] = bottom_cs_1_cs_2r
            reactivation_top_bottom[16][day] = bottom_cs_2_cs_2r
            reactivation_top_bottom[17][day] = all_other_cs_2r

            reactivation_top_bottom[18][day] = top_cs_1_cs_2c
            reactivation_top_bottom[19][day] = top_cs_2_cs_2c
            reactivation_top_bottom[20][day] = bottom_cs_1_cs_2c
            reactivation_top_bottom[21][day] = bottom_cs_2_cs_2c
            reactivation_top_bottom[22][day] = all_other_cs_2c

            reactivation_top_bottom[23][day] = top_cs_2_b
            reactivation_top_bottom[24][day] = bottom_cs_2_b
            reactivation_top_bottom[25][day] = all_other_cs_2_b
            np.save(days_path + 'reactivation_top_bottom_activity', reactivation_top_bottom)
        else:
            reactivation_top_bottom = np.load(days_path + 'reactivation_top_bottom_activity.npy', allow_pickle=True)
            reactivation_top_bottom[0][day] = top_cs_1_cs_1r
            reactivation_top_bottom[1][day] = top_cs_2_cs_1r
            reactivation_top_bottom[2][day] = bottom_cs_1_cs_1r
            reactivation_top_bottom[3][day] = bottom_cs_2_cs_1r
            reactivation_top_bottom[4][day] = all_other_cs_1r

            reactivation_top_bottom[5][day] = top_cs_1_cs_1c
            reactivation_top_bottom[6][day] = top_cs_2_cs_1c
            reactivation_top_bottom[7][day] = bottom_cs_1_cs_1c
            reactivation_top_bottom[8][day] = bottom_cs_2_cs_1c
            reactivation_top_bottom[9][day] = all_other_cs_1c

            reactivation_top_bottom[10][day] = top_cs_1_b
            reactivation_top_bottom[11][day] = bottom_cs_1_b
            reactivation_top_bottom[12][day] = all_other_cs_1_b

            reactivation_top_bottom[13][day] = top_cs_1_cs_2r
            reactivation_top_bottom[14][day] = top_cs_2_cs_2r
            reactivation_top_bottom[15][day] = bottom_cs_1_cs_2r
            reactivation_top_bottom[16][day] = bottom_cs_2_cs_2r
            reactivation_top_bottom[17][day] = all_other_cs_2r

            reactivation_top_bottom[18][day] = top_cs_1_cs_2c
            reactivation_top_bottom[19][day] = top_cs_2_cs_2c
            reactivation_top_bottom[20][day] = bottom_cs_1_cs_2c
            reactivation_top_bottom[21][day] = bottom_cs_2_cs_2c
            reactivation_top_bottom[22][day] = all_other_cs_2c

            reactivation_top_bottom[23][day] = top_cs_2_b
            reactivation_top_bottom[24][day] = bottom_cs_2_b
            reactivation_top_bottom[25][day] = all_other_cs_2_b
            np.save(days_path + 'reactivation_top_bottom_activity', reactivation_top_bottom)


def reactivation_cue_correlation(norm_deconvolved, idx, y_pred, behavior, paths, day, days):
    """
    get correlation between cues and reactivation
    :param norm_deconvolved: activity
    :param idx: index
    :param y_pred: y pred
    :param behavior: behavior
    :param paths: path
    :param day: day
    :param days: days
    :return: correlation vec
    """
    activity = norm_deconvolved.to_numpy()
    activity = activity[idx['both'].index]

    sig_cells = preprocess.sig_reactivated_cells([], [], [], [], [], paths, 0)
    activity = activity[sig_cells > 0, :]

    mean_activity_cs_1 = []
    mean_activity_cs_2 = []
    onsets_cs_1 = []
    onsets_cs_2 = []
    for i in range(0, len(behavior['onsets'])):
        temp_activity = np.mean(activity[:, int(behavior['onsets'][i]):int(behavior['offsets'][i]) + 1], axis=1)
        if behavior['cue_codes'][i] == behavior['cs_1_code']:
            mean_activity_cs_1.append(temp_activity)
            onsets_cs_1.append(behavior['onsets'][i])
        if behavior['cue_codes'][i] == behavior['cs_2_code']:
            mean_activity_cs_2.append(temp_activity)
            onsets_cs_2.append(behavior['onsets'][i])

    mean_activity_cs_1 = np.array(
        pd.DataFrame(mean_activity_cs_1).rolling(10, min_periods=1, center=True, axis=1).mean())
    mean_activity_cs_2 = np.array(
        pd.DataFrame(mean_activity_cs_2).rolling(10, min_periods=1, center=True, axis=1).mean())

    duration = int(behavior['framerate'] * (behavior['iti'] + 5)) + 1
    p_threshold = .75
    trials = 20
    reactivation_cs_1 = y_pred[:, 0].copy()
    reactivation_cs_2 = y_pred[:, 1].copy()
    cs_1c_cs_1r = np.empty((1000, trials*2 + 1)) * np.nan
    cs_2c_cs_2r = np.empty((1000, trials * 2 + 1)) * np.nan
    r_num = 0
    next_r = 0
    cs_1_peak = 0
    cs_2_peak = 0
    i = 0
    r_start = []
    while i < len(reactivation_cs_1):
        if reactivation_cs_1[i] > 0 or reactivation_cs_2[i] > 0:
            if next_r == 0:
                r_start = i
                next_r = 1
            if reactivation_cs_1[i] > cs_1_peak:
                cs_1_peak = reactivation_cs_1[i]
            if reactivation_cs_2[i] > cs_2_peak:
                cs_2_peak = reactivation_cs_2[i]
            if reactivation_cs_1[i + 1] == 0 and reactivation_cs_2[i + 1] == 0:
                r_end = i+1
                if cs_1_peak > p_threshold:
                    for j in range(0, len(onsets_cs_1)):
                        if r_start < onsets_cs_1[j]:
                            if onsets_cs_1[j - 1] < r_start < onsets_cs_1[j - 1] + duration:
                                corr = np.empty(trials*2 + 1) * np.nan
                                mean_r = np.mean(activity[:, r_start:r_end], axis=1)
                                for c in range(-trials, trials+1):
                                    if 0 <= j + c < len(onsets_cs_1):
                                        temp_corr = np.corrcoef(mean_r, mean_activity_cs_1[j+c])[0][1]
                                        corr[c+trials] = temp_corr
                                cs_1c_cs_1r[r_num, :] = corr
                                r_num += 1
                                break
                if cs_2_peak > p_threshold:
                    for j in range(0, len(onsets_cs_2)):
                        if r_start < onsets_cs_2[j]:
                            if onsets_cs_2[j - 1] < r_start < onsets_cs_2[j - 1] + duration:
                                corr = np.empty(trials*2 + 1) * np.nan
                                mean_r = np.mean(activity[:, r_start:r_end], axis=1)
                                for c in range(-trials, trials+1):
                                    if 0 <= j + c < len(onsets_cs_2):
                                        temp_corr = np.corrcoef(mean_r, mean_activity_cs_2[j+c])[0][1]
                                        corr[c+trials] = temp_corr
                                cs_2c_cs_2r[r_num, :] = corr
                                r_num += 1
                                break
                cs_1_peak = 0
                cs_2_peak = 0
                next_r = 0
        i += 1

    days_path = paths['base_path'] + paths['mouse'] + '/data_across_days/'
    if days:
        if path.isfile(days_path + 'reactivation_cue_corr_smooth.npy') == 0 or day == 0:
            reactivation_cue_corr = [list(range(0, days)), list(range(0, days))]
            reactivation_cue_corr[0][day] = np.nanmean(cs_1c_cs_1r, axis=0)
            reactivation_cue_corr[1][day] = np.nanmean(cs_2c_cs_2r, axis=0)
            np.save(days_path + 'reactivation_cue_corr_smooth', reactivation_cue_corr)
        else:
            reactivation_cue_corr = np.load(days_path + 'reactivation_cue_corr_smooth.npy', allow_pickle=True)
            reactivation_cue_corr[0][day] = np.nanmean(cs_1c_cs_1r, axis=0)
            reactivation_cue_corr[1][day] = np.nanmean(cs_2c_cs_2r, axis=0)
            np.save(days_path + 'reactivation_cue_corr_smooth', reactivation_cue_corr)


def reactivation_cue_correlation_shuffle(norm_deconvolved, idx, y_pred, behavior, paths, day, days):
    """
    get correlation between cues and reactivation shuffled
    :param norm_deconvolved: activity
    :param idx: index
    :param y_pred: y pred
    :param behavior: behavior
    :param paths: path
    :param day: day
    :param days: days
    :return: correlation vec
    """
    activity = norm_deconvolved.to_numpy()
    activity = activity[idx['both'].index]

    sig_cells = preprocess.sig_reactivated_cells([], [], [], [], [], paths, 0)
    activity = activity[sig_cells > 0, :]

    mean_activity_cs_1 = []
    mean_activity_cs_2 = []
    onsets_cs_1 = []
    onsets_cs_2 = []
    for i in range(0, len(behavior['onsets'])):
        temp_activity = np.mean(activity[:, int(behavior['onsets'][i]):int(behavior['offsets'][i]) + 1], axis=1)
        if behavior['cue_codes'][i] == behavior['cs_1_code']:
            mean_activity_cs_1.append(temp_activity)
            onsets_cs_1.append(behavior['onsets'][i])
        if behavior['cue_codes'][i] == behavior['cs_2_code']:
            mean_activity_cs_2.append(temp_activity)
            onsets_cs_2.append(behavior['onsets'][i])

    duration = int(behavior['framerate'] * (behavior['iti'] + 5)) + 1
    p_threshold = .75
    reactivation_cs_1 = y_pred[:, 0].copy()
    reactivation_cs_2 = y_pred[:, 1].copy()
    cs_1_peak = 0
    cs_2_peak = 0
    i = 0
    r_start_cs_1_s = np.empty(1000) * np.nan
    r_end_cs_1_s = np.empty(1000) * np.nan
    r_start_cs_2_s = np.empty(1000) * np.nan
    r_end_cs_2_s = np.empty(1000) * np.nan
    num_r_cs_1 = 0
    num_r_cs_2 = 0
    next_r = 0
    r_start = []
    while i < len(reactivation_cs_1):
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
                if cs_1_peak > p_threshold:
                    for j in range(0, len(onsets_cs_1)):
                        if r_start < onsets_cs_1[j]:
                            if onsets_cs_1[j - 1] < r_start < onsets_cs_1[j - 1] + duration:
                                r_start_cs_1_s[num_r_cs_1] = r_start
                                r_end_cs_1_s[num_r_cs_1] = r_end
                                num_r_cs_1 += 1
                                break
                if cs_2_peak > p_threshold:
                    for j in range(0, len(onsets_cs_2)):
                        if r_start < onsets_cs_2[j]:
                            if onsets_cs_2[j - 1] < r_start < onsets_cs_2[j - 1] + duration:
                                r_start_cs_2_s[num_r_cs_2] = r_start
                                r_end_cs_2_s[num_r_cs_2] = r_end
                                num_r_cs_2 += 1
                                break
                cs_1_peak = 0
                cs_2_peak = 0
                next_r = 0
        i += 1

    iterations = 10
    trials = 20
    cs_1c_cs_1r_all = np.empty((iterations, trials * 2 + 1)) * np.nan
    cs_2c_cs_2r_all = np.empty((iterations, trials * 2 + 1)) * np.nan
    for it in range(0, iterations):

        r_start_cs_1_s = r_start_cs_1_s[~np.isnan(r_start_cs_1_s)]
        r_end_cs_1_s = r_end_cs_1_s[~np.isnan(r_end_cs_1_s)]
        r_start_cs_2_s = r_start_cs_2_s[~np.isnan(r_start_cs_2_s)]
        r_end_cs_2_s = r_end_cs_2_s[~np.isnan(r_end_cs_2_s)]

        rand_p_cs_1 = np.random.permutation(len(r_end_cs_1_s))
        r_start_cs_1_s = r_start_cs_1_s[rand_p_cs_1]
        r_end_cs_1_s = r_end_cs_1_s[rand_p_cs_1]
        rand_p_cs_2 = np.random.permutation(len(r_end_cs_2_s))
        r_start_cs_2_s = r_start_cs_2_s[rand_p_cs_2]
        r_end_cs_2_s = r_end_cs_2_s[rand_p_cs_2]

        num_r_cs_1 = 0
        num_r_cs_2 = 0
        r_num = 0
        i = 0
        cs_1_peak = 0
        cs_2_peak = 0
        cs_1c_cs_1r = np.empty((1000, trials * 2 + 1)) * np.nan
        cs_2c_cs_2r = np.empty((1000, trials * 2 + 1)) * np.nan
        next_r = 0
        while i < len(reactivation_cs_1):
            if reactivation_cs_1[i] > 0 or reactivation_cs_2[i] > 0:
                if next_r == 0:
                    r_start = i
                    next_r = 1
                if reactivation_cs_1[i] > cs_1_peak:
                    cs_1_peak = reactivation_cs_1[i]
                if reactivation_cs_2[i] > cs_2_peak:
                    cs_2_peak = reactivation_cs_2[i]
                if reactivation_cs_1[i + 1] == 0 and reactivation_cs_2[i + 1] == 0:
                    if cs_1_peak > p_threshold:
                        for j in range(0, len(onsets_cs_1)):
                            if r_start < onsets_cs_1[j]:
                                if onsets_cs_1[j - 1] < r_start < onsets_cs_1[j - 1] + duration:
                                    corr = np.empty(trials*2 + 1) * np.nan
                                    mean_r = np.mean(activity[:, int(r_start_cs_1_s[num_r_cs_1]):int(r_end_cs_1_s[num_r_cs_1])], axis=1)
                                    for c in range(-trials, trials+1):
                                        if 0 <= j+c < len(onsets_cs_1):
                                            temp_corr = np.corrcoef(mean_r, mean_activity_cs_1[j+c])[0][1]
                                            corr[c+trials] = temp_corr
                                    cs_1c_cs_1r[r_num, :] = corr
                                    r_num += 1
                                    num_r_cs_1 += 1
                                    break
                    if cs_2_peak > p_threshold:
                        for j in range(0, len(onsets_cs_2)):
                            if r_start < onsets_cs_2[j]:
                                if onsets_cs_2[j - 1] < r_start < onsets_cs_2[j - 1] + duration:
                                    corr = np.empty(trials*2 + 1) * np.nan
                                    mean_r = np.mean(activity[:, int(r_start_cs_2_s[num_r_cs_2]):int(r_end_cs_2_s[num_r_cs_2])], axis=1)
                                    for c in range(-trials, trials+1):
                                        if 0 <= j + c < len(onsets_cs_2):
                                            temp_corr = np.corrcoef(mean_r, mean_activity_cs_2[j+c])[0][1]
                                            corr[c+trials] = temp_corr
                                    cs_2c_cs_2r[r_num, :] = corr
                                    r_num += 1
                                    num_r_cs_2 += 1
                                    break
                    cs_1_peak = 0
                    cs_2_peak = 0
                    next_r = 0
            i += 1

        cs_1c_cs_1r = np.nanmean(cs_1c_cs_1r, axis=0)
        cs_2c_cs_2r = np.nanmean(cs_2c_cs_2r, axis=0)
        cs_1c_cs_1r_all[it, :] = cs_1c_cs_1r
        cs_2c_cs_2r_all[it, :] = cs_2c_cs_2r

    days_path = paths['base_path'] + paths['mouse'] + '/data_across_days/'
    if days:
        if path.isfile(days_path + 'reactivation_cue_corr_shuffle.npy') == 0 or day == 0:
            reactivation_cue_corr = [list(range(0, days)), list(range(0, days))]
            reactivation_cue_corr[0][day] = np.nanmean(cs_1c_cs_1r_all, axis=0)
            reactivation_cue_corr[1][day] = np.nanmean(cs_2c_cs_2r_all, axis=0)
            np.save(days_path + 'reactivation_cue_corr_shuffle', reactivation_cue_corr)
        else:
            reactivation_cue_corr = np.load(days_path + 'reactivation_cue_corr_shuffle.npy', allow_pickle=True)
            reactivation_cue_corr[0][day] = np.nanmean(cs_1c_cs_1r_all, axis=0)
            reactivation_cue_corr[1][day] = np.nanmean(cs_2c_cs_2r_all, axis=0)
            np.save(days_path + 'reactivation_cue_corr_shuffle', reactivation_cue_corr)


def cue_cue_correlation(norm_deconvolved, idx, y_pred, behavior, paths, day, days):
    """
    get correlation between cues
    :param norm_deconvolved: activity
    :param idx: index
    :param y_pred: y pred
    :param behavior: behavior
    :param paths: path
    :param day: day
    :param days: days
    :return: correlation vec
    """
    activity = norm_deconvolved.to_numpy()
    activity = activity[idx['both'].index]

    sig_cells = preprocess.sig_reactivated_cells([], [], [], [], [], paths, 0)
    activity = activity[sig_cells > 0, :]

    mean_activity_cs_1 = []
    mean_activity_cs_2 = []
    for i in range(0, len(behavior['onsets'])):
        temp_activity = np.mean(activity[:, int(behavior['onsets'][i]):int(behavior['offsets'][i]) + 1], axis=1)
        if behavior['cue_codes'][i] == behavior['cs_1_code']:
            mean_activity_cs_1.append(temp_activity)
        if behavior['cue_codes'][i] == behavior['cs_2_code']:
            mean_activity_cs_2.append(temp_activity)

    trials = 20
    cs_1c_cs_1c = np.empty((1000, trials*2 + 1)) * np.nan
    cs_2c_cs_2c = np.empty((1000, trials * 2 + 1)) * np.nan

    for j in range(0, len(mean_activity_cs_1)):
        corr = np.empty(trials*2 + 1) * np.nan
        for c in range(-trials, trials+1):
            if 0 <= j+c < len(mean_activity_cs_1):
                temp_corr = np.corrcoef(mean_activity_cs_1[j+c], mean_activity_cs_1[j])[0][1]
                corr[c+trials] = temp_corr
        cs_1c_cs_1c[j, :] = corr
    for j in range(0, len(mean_activity_cs_2)):
        corr = np.empty(trials * 2 + 1) * np.nan
        for c in range(-trials, trials + 1):
            if 0 <= j + c < len(mean_activity_cs_2):
                temp_corr = np.corrcoef(mean_activity_cs_2[j + c], mean_activity_cs_2[j])[0][1]
                corr[c + trials] = temp_corr
        cs_2c_cs_2c[j, :] = corr

    days_path = paths['base_path'] + paths['mouse'] + '/data_across_days/'
    if days:
        if path.isfile(days_path + 'cue_cue_corr.npy') == 0 or day == 0:
            reactivation_cue_corr = [list(range(0, days)), list(range(0, days))]
            reactivation_cue_corr[0][day] = np.nanmean(cs_1c_cs_1c, axis=0)
            reactivation_cue_corr[1][day] = np.nanmean(cs_2c_cs_2c, axis=0)
            np.save(days_path + 'cue_cue_corr', reactivation_cue_corr)
        else:
            reactivation_cue_corr = np.load(days_path + 'cue_cue_corr.npy', allow_pickle=True)
            reactivation_cue_corr[0][day] = np.nanmean(cs_1c_cs_1c, axis=0)
            reactivation_cue_corr[1][day] = np.nanmean(cs_2c_cs_2c, axis=0)
            np.save(days_path + 'cue_cue_corr', reactivation_cue_corr)


def cue_cue_correlation_shuffle(norm_deconvolved, idx, y_pred, behavior, paths, day, days):
    """
    get correlation between cues shuffled
    :param norm_deconvolved: activity
    :param idx: index
    :param y_pred: y pred
    :param behavior: behavior
    :param paths: path
    :param day: day
    :param days: days
    :return: correlation vec
    """
    activity = norm_deconvolved.to_numpy()
    activity = activity[idx['both'].index]

    sig_cells = preprocess.sig_reactivated_cells([], [], [], [], [], paths, 0)
    activity = activity[sig_cells > 0, :]

    mean_activity_cs_1 = []
    mean_activity_cs_2 = []
    for i in range(0, len(behavior['onsets'])):
        temp_activity = np.mean(activity[:, int(behavior['onsets'][i]):int(behavior['offsets'][i]) + 1], axis=1)
        if behavior['cue_codes'][i] == behavior['cs_1_code']:
            mean_activity_cs_1.append(temp_activity)
        if behavior['cue_codes'][i] == behavior['cs_2_code']:
            mean_activity_cs_2.append(temp_activity)

    iterations = 10
    trials = 20
    cs_1c_cs_1c_all = np.empty((iterations, trials * 2 + 1)) * np.nan
    cs_2c_cs_2c_all = np.empty((iterations, trials * 2 + 1)) * np.nan
    for it in range(0, iterations):
        rand_perm_cs_1 = np.random.permutation(len(mean_activity_cs_1))
        rand_perm_cs_2 = np.random.permutation(len(mean_activity_cs_2))

        cs_1c_cs_1c = np.empty((1000, trials*2 + 1)) * np.nan
        cs_2c_cs_2c = np.empty((1000, trials * 2 + 1)) * np.nan

        for j in range(0, len(mean_activity_cs_1)):
            corr = np.empty(trials*2 + 1) * np.nan
            for c in range(-trials, trials+1):
                if 0 <= j+c < len(mean_activity_cs_1):
                    temp_corr = np.corrcoef(mean_activity_cs_1[j+c], mean_activity_cs_1[rand_perm_cs_1[j]])[0][1]
                    corr[c+trials] = temp_corr
            cs_1c_cs_1c[j, :] = corr
        for j in range(0, len(mean_activity_cs_2)):
            corr = np.empty(trials * 2 + 1) * np.nan
            for c in range(-trials, trials + 1):
                if 0 <= j + c < len(mean_activity_cs_2):
                    temp_corr = np.corrcoef(mean_activity_cs_2[j + c], mean_activity_cs_2[rand_perm_cs_2[j]])[0][1]
                    corr[c + trials] = temp_corr
            cs_2c_cs_2c[j, :] = corr
        cs_1c_cs_1c = np.nanmean(cs_1c_cs_1c, axis=0)
        cs_2c_cs_2c = np.nanmean(cs_2c_cs_2c, axis=0)
        cs_1c_cs_1c_all[it, :] = cs_1c_cs_1c
        cs_2c_cs_2c_all[it, :] = cs_2c_cs_2c
    days_path = paths['base_path'] + paths['mouse'] + '/data_across_days/'
    if days:
        if path.isfile(days_path + 'cue_cue_corr_shuffle.npy') == 0 or day == 0:
            reactivation_cue_corr = [list(range(0, days)), list(range(0, days))]
            reactivation_cue_corr[0][day] = np.nanmean(cs_1c_cs_1c_all, axis=0)
            reactivation_cue_corr[1][day] = np.nanmean(cs_2c_cs_2c_all, axis=0)
            np.save(days_path + 'cue_cue_corr_shuffle', reactivation_cue_corr)
        else:
            reactivation_cue_corr = np.load(days_path + 'cue_cue_corr_shuffle.npy', allow_pickle=True)
            reactivation_cue_corr[0][day] = np.nanmean(cs_1c_cs_1c_all, axis=0)
            reactivation_cue_corr[1][day] = np.nanmean(cs_2c_cs_2c_all, axis=0)
            np.save(days_path + 'cue_cue_corr_shuffle', reactivation_cue_corr)


def reactivation_spatial(norm_deconvolved, y_pred, behavior, planes, idx, paths, day, days):
    stat = []
    for plane in range(1, planes + 1):
        plane_path = paths['base_path'] + paths['mouse'] + '/' + paths['date'] + '_' + paths['mouse'] + \
                     '/suite2p_plane_' + str(plane) + '/suite2p/plane0/'
        accepted_cells = np.load(plane_path + 'iscell.npy')[:, 0]
        fluorescence = np.load(plane_path + 'F.npy').sum(axis=1)
        accepted_cells[fluorescence == 0] = 0
        accepted_cells = accepted_cells == 1
        stat_plane = np.load(plane_path + 'stat.npy', allow_pickle=True)
        if plane == 1:
            stat = stat_plane[accepted_cells]
        else:
            stat_plane = stat_plane[accepted_cells]
            to_delete = preprocess.cells_to_delete(paths, plane, plane_path, 0)
            stat_plane = stat_plane[to_delete == 1]
            stat = np.concatenate((stat, stat_plane))
    stat = stat[idx['both'].index]

    activity = norm_deconvolved.to_numpy()
    activity = activity[idx['both'].index]
    cue_time = preprocess.cue_times(behavior, 0, 0)
    activity_cue = np.mean(activity[:, cue_time[0] > 0], axis=1) * behavior['framerate'][0][0]

    retinotopy = loadmat(paths['base_path'] + paths['mouse'] + '/' + 'retinotopy' + '/' + 'retinotopy.mat')
    retinotopy_mouse = loadmat(paths['save_path'] + 'saved_data/retinotopy_day.mat')

    area_li_cue = []
    area_por_cue = []
    area_p_cue = []
    area_lm_cue = []
    for n in range(0, len(stat)):
        ypix = int(np.mean(stat[n]['ypix'][~stat[n]['overlap']]))
        xpix = int(np.mean(stat[n]['xpix'][~stat[n]['overlap']]))
        diff_x = retinotopy_mouse['base_c'][0][0] - retinotopy_mouse['imaging_c'][0][0]
        diff_y = retinotopy_mouse['base_c'][0][1] - retinotopy_mouse['imaging_c'][0][1]
        true_xpix = xpix + diff_x
        true_ypix = ypix + diff_y
        if retinotopy['LI'][true_ypix][true_xpix] == 1:
            area_li_cue.append(activity_cue[n])
        if retinotopy['POR'][true_ypix][true_xpix] == 1:
            area_por_cue.append(activity_cue[n])
        if retinotopy['P'][true_ypix][true_xpix] == 1:
            area_p_cue.append(activity_cue[n])
        if retinotopy['LM'][true_ypix][true_xpix] == 1:
            area_lm_cue.append(activity_cue[n])

    reactivation_cs_1 = y_pred[:, 0].copy()
    reactivation_cs_2 = y_pred[:, 1].copy()
    p_threshold = .75
    cs_1_peak = 0
    cs_2_peak = 0
    i = 0
    reactivation_frames = np.zeros(len(reactivation_cs_1))
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
                if cs_1_peak > p_threshold and r_start > int(behavior['onsets'][0]):
                    reactivation_frames[r_start:r_end] = 1
                if cs_2_peak > p_threshold and r_start > int(behavior['onsets'][0]):
                    reactivation_frames[r_start:r_end] = 1
                i = r_end
                cs_1_peak = 0
                cs_2_peak = 0
    activity_reactivation = np.mean(activity[:, reactivation_frames > 0], axis=1) * behavior['framerate'][0][0]

    area_li_reactivation = []
    area_por_reactivation = []
    area_p_reactivation = []
    area_lm_reactivation = []
    for n in range(0, len(stat)):
        ypix = int(np.mean(stat[n]['ypix'][~stat[n]['overlap']]))
        xpix = int(np.mean(stat[n]['xpix'][~stat[n]['overlap']]))
        diff_x = retinotopy_mouse['base_c'][0][0] - retinotopy_mouse['imaging_c'][0][0]
        diff_y = retinotopy_mouse['base_c'][0][1] - retinotopy_mouse['imaging_c'][0][1]
        true_xpix = xpix + diff_x
        true_ypix = ypix + diff_y
        if retinotopy['LI'][true_ypix][true_xpix] == 1:
            area_li_reactivation.append(activity_reactivation[n])
        if retinotopy['POR'][true_ypix][true_xpix] == 1:
            area_por_reactivation.append(activity_reactivation[n])
        if retinotopy['P'][true_ypix][true_xpix] == 1:
            area_p_reactivation.append(activity_reactivation[n])
        if retinotopy['LM'][true_ypix][true_xpix] == 1:
            area_lm_reactivation.append(activity_reactivation[n])

    im = np.zeros((len(retinotopy['LI']), len(retinotopy['LI'][0])))
    for n in range(0, len(stat)):
        ypix = stat[n]['ypix'][~stat[n]['overlap']]
        xpix = stat[n]['xpix'][~stat[n]['overlap']]
        diff_x = retinotopy_mouse['base_c'][0][0] - retinotopy_mouse['imaging_c'][0][0]
        diff_y = retinotopy_mouse['base_c'][0][1] - retinotopy_mouse['imaging_c'][0][1]
        true_xpix = xpix + diff_x
        true_ypix = ypix + diff_y
        im[true_ypix, true_xpix] = activity_cue[n]
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(11.66, 7.5))
    sns.heatmap(im, cbar=0, cmap=['white', [.8, .8, .8], 'red'], vmin=-.1, vmax=.8)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(paths['save_path'] + 'plots/' + paths['mouse'] + '_' + paths['date'] + '_' +
                'reactivation_spatial_cue.png', bbox_inches='tight', dpi=500)
    #plt.close()

    im = np.zeros((len(retinotopy['LI']), len(retinotopy['LI'][0])))
    for n in range(0, len(stat)):
        ypix = stat[n]['ypix'][~stat[n]['overlap']]
        xpix = stat[n]['xpix'][~stat[n]['overlap']]
        diff_x = retinotopy_mouse['base_c'][0][0] - retinotopy_mouse['imaging_c'][0][0]
        diff_y = retinotopy_mouse['base_c'][0][1] - retinotopy_mouse['imaging_c'][0][1]
        true_xpix = xpix + diff_x
        true_ypix = ypix + diff_y
        im[true_ypix, true_xpix] = activity_reactivation[n]
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(11.66, 7.5))
    sns.heatmap(im, cbar=0, cmap=['white', [.8, .8, .8], 'red'], vmin=-.1, vmax=.8)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(paths['save_path'] + 'plots/' + paths['mouse'] + '_' + paths['date'] + '_' +
                'reactivation_spatial_reactivation.png', bbox_inches='tight', dpi=500)
    #plt.close()
    ggg
    im = retinotopy['POR'] + retinotopy['P'] + retinotopy['LI'] + retinotopy['LM']
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(11.66, 7.5))
    sns.heatmap(im, cbar=0)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(paths['save_path'] + 'plots/' + paths['mouse'] + '_' + paths['date'] + '_' +
                'reactivation_spatial_retinotopy.png', bbox_inches='tight', dpi=500)
    plt.close()

    days_path = paths['base_path'] + paths['mouse'] + '/data_across_days/'
    if days:
        if path.isfile(days_path + 'reactivation_spatial.npy') == 0 or day == 0:
            reactivation_spatial_days = [list(range(0, days)), list(range(0, days)), list(range(0, days)),
                                         list(range(0, days)), list(range(0, days)), list(range(0, days)),
                                         list(range(0, days)), list(range(0, days))]
            reactivation_spatial_days[0][day] = np.mean(area_li_cue)
            reactivation_spatial_days[1][day] = np.mean(area_por_cue)
            reactivation_spatial_days[2][day] = np.mean(area_p_cue)
            reactivation_spatial_days[3][day] = np.mean(area_lm_cue)
            reactivation_spatial_days[4][day] = np.mean(area_li_reactivation)
            reactivation_spatial_days[5][day] = np.mean(area_por_reactivation)
            reactivation_spatial_days[6][day] = np.mean(area_p_reactivation)
            reactivation_spatial_days[7][day] = np.mean(area_lm_reactivation)
            np.save(days_path + 'reactivation_spatial', reactivation_spatial_days)
        else:
            reactivation_spatial_days = np.load(days_path + 'reactivation_spatial.npy', allow_pickle=True)
            reactivation_spatial_days[0][day] = np.mean(area_li_cue)
            reactivation_spatial_days[1][day] = np.mean(area_por_cue)
            reactivation_spatial_days[2][day] = np.mean(area_p_cue)
            reactivation_spatial_days[3][day] = np.mean(area_lm_cue)
            reactivation_spatial_days[4][day] = np.mean(area_li_reactivation)
            reactivation_spatial_days[5][day] = np.mean(area_por_reactivation)
            reactivation_spatial_days[6][day] = np.mean(area_p_reactivation)
            reactivation_spatial_days[7][day] = np.mean(area_lm_reactivation)
            np.save(days_path + 'reactivation_spatial', reactivation_spatial_days)


def reactivation_spatial_percent(norm_deconvolved, y_pred, behavior, planes, idx, paths, day, days):
    stat = []
    for plane in range(1, planes + 1):
        plane_path = paths['base_path'] + paths['mouse'] + '/' + paths['date'] + '_' + paths['mouse'] + \
                     '/suite2p_plane_' + str(plane) + '/suite2p/plane0/'
        accepted_cells = np.load(plane_path + 'iscell.npy')[:, 0]
        fluorescence = np.load(plane_path + 'F.npy').sum(axis=1)
        accepted_cells[fluorescence == 0] = 0
        accepted_cells = accepted_cells == 1
        stat_plane = np.load(plane_path + 'stat.npy', allow_pickle=True)
        if plane == 1:
            stat = stat_plane[accepted_cells]
        else:
            stat_plane = stat_plane[accepted_cells]
            to_delete = preprocess.cells_to_delete(paths, plane, plane_path, 0)
            stat_plane = stat_plane[to_delete == 1]
            stat = np.concatenate((stat, stat_plane))
    stat = stat[idx['both'].index]

    retinotopy = loadmat(paths['base_path'] + paths['mouse'] + '/' + 'retinotopy' + '/' + 'retinotopy.mat')
    retinotopy_mouse = loadmat(paths['save_path'] + 'saved_data/retinotopy_day.mat')

    area_li_cue = 0
    area_por_cue = 0
    area_p_cue = 0
    area_lm_cue = 0
    for n in range(0, len(stat)):
        ypix = int(np.mean(stat[n]['ypix'][~stat[n]['overlap']]))
        xpix = int(np.mean(stat[n]['xpix'][~stat[n]['overlap']]))
        diff_x = retinotopy_mouse['base_c'][0][0] - retinotopy_mouse['imaging_c'][0][0]
        diff_y = retinotopy_mouse['base_c'][0][1] - retinotopy_mouse['imaging_c'][0][1]
        true_xpix = xpix + diff_x
        true_ypix = ypix + diff_y
        if retinotopy['LI'][true_ypix][true_xpix] == 1:
            area_li_cue += 1
        if retinotopy['POR'][true_ypix][true_xpix] == 1:
            area_por_cue += 1
        if retinotopy['P'][true_ypix][true_xpix] == 1:
            area_p_cue += 1
        if retinotopy['LM'][true_ypix][true_xpix] == 1:
            area_lm_cue += 1

    sig_cells = preprocess.sig_reactivated_cells([], [], [], [], [], paths, 0)
    sig_cells[sig_cells > 0] = 1

    area_li_reactivation = 0
    area_por_reactivation = 0
    area_p_reactivation = 0
    area_lm_reactivation = 0
    for n in range(0, len(stat)):
        ypix = int(np.mean(stat[n]['ypix'][~stat[n]['overlap']]))
        xpix = int(np.mean(stat[n]['xpix'][~stat[n]['overlap']]))
        diff_x = retinotopy_mouse['base_c'][0][0] - retinotopy_mouse['imaging_c'][0][0]
        diff_y = retinotopy_mouse['base_c'][0][1] - retinotopy_mouse['imaging_c'][0][1]
        true_xpix = xpix + diff_x
        true_ypix = ypix + diff_y
        if retinotopy['LI'][true_ypix][true_xpix] == 1:
            area_li_reactivation += sig_cells[n]
        if retinotopy['POR'][true_ypix][true_xpix] == 1:
            area_por_reactivation += sig_cells[n]
        if retinotopy['P'][true_ypix][true_xpix] == 1:
            area_p_reactivation += sig_cells[n]
        if retinotopy['LM'][true_ypix][true_xpix] == 1:
            area_lm_reactivation += sig_cells[n]

    days_path = paths['base_path'] + paths['mouse'] + '/data_across_days/'
    if days:
        if path.isfile(days_path + 'reactivation_spatial_percent.npy') == 0 or day == 0:
            reactivation_spatial_days = [list(range(0, days)), list(range(0, days)), list(range(0, days)),
                                         list(range(0, days))]
            reactivation_spatial_days[0][day] = area_li_reactivation/area_li_cue
            reactivation_spatial_days[1][day] = area_por_reactivation/area_por_cue
            if area_p_cue > 0:
                reactivation_spatial_days[2][day] = area_p_reactivation / area_p_cue
            if area_p_cue == 0:
                reactivation_spatial_days[2][day] = np.nan
            reactivation_spatial_days[3][day] = area_lm_reactivation/area_lm_cue
            np.save(days_path + 'reactivation_spatial_percent', reactivation_spatial_days)
        else:
            reactivation_spatial_days = np.load(days_path + 'reactivation_spatial_percent.npy', allow_pickle=True)
            reactivation_spatial_days[0][day] = area_li_reactivation / area_li_cue
            reactivation_spatial_days[1][day] = area_por_reactivation / area_por_cue
            if area_p_cue > 0:
                reactivation_spatial_days[2][day] = area_p_reactivation / area_p_cue
            if area_p_cue == 0:
                reactivation_spatial_days[2][day] = np.nan
            reactivation_spatial_days[3][day] = area_lm_reactivation / area_lm_cue
            np.save(days_path + 'reactivation_spatial_percent', reactivation_spatial_days)


def true_reactivations(y_pred):
    reactivation_cs_1 = y_pred[:, 0].copy()
    reactivation_cs_2 = y_pred[:, 1].copy()
    p_threshold = .75
    cs_1_peak = 0
    cs_2_peak = 0
    i = 0
    reactivation_frames = np.zeros(len(reactivation_cs_1))
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
                    reactivation_frames[r_start:r_end] = 1
                if cs_2_peak > p_threshold:
                    reactivation_frames[r_start:r_end] = 1
                i = r_end
                cs_1_peak = 0
                cs_2_peak = 0
    y_pred[reactivation_frames == 0, 0] = 0
    y_pred[reactivation_frames == 0, 1] = 0
    return y_pred


def tricubic(x):
    y = np.zeros_like(x)
    idx = (x >= -1) & (x <= 1)
    y[idx] = np.power(1.0 - np.power(np.abs(x[idx]), 3), 3)
    return y


class Loess(object):

    @staticmethod
    def normalize_array(array):
        min_val = np.min(array)
        max_val = np.max(array)
        return (array - min_val) / (max_val - min_val), min_val, max_val

    def __init__(self, xx, yy, degree=1):
        self.n_xx, self.min_xx, self.max_xx = self.normalize_array(xx)
        self.n_yy, self.min_yy, self.max_yy = self.normalize_array(yy)
        self.degree = degree

    @staticmethod
    def get_min_range(distances, window):
        min_idx = np.argmin(distances)
        n = len(distances)
        if min_idx == 0:
            return np.arange(0, window)
        if min_idx == n - 1:
            return np.arange(n - window, n)

        min_range = [min_idx]
        while len(min_range) < window:
            i0 = min_range[0]
            i1 = min_range[-1]
            if i0 == 0:
                min_range.append(i1 + 1)
            elif i1 == n - 1:
                min_range.insert(0, i0 - 1)
            elif distances[i0 - 1] < distances[i1 + 1]:
                min_range.insert(0, i0 - 1)
            else:
                min_range.append(i1 + 1)
        return np.array(min_range)

    @staticmethod
    def get_weights(distances, min_range):
        max_distance = np.max(distances[min_range])
        weights = tricubic(distances[min_range] / max_distance)
        return weights

    def normalize_x(self, value):
        return (value - self.min_xx) / (self.max_xx - self.min_xx)

    def denormalize_y(self, value):
        return value * (self.max_yy - self.min_yy) + self.min_yy

    def estimate(self, x, window, use_matrix=False, degree=1):
        n_x = self.normalize_x(x)
        distances = np.abs(self.n_xx - n_x)
        min_range = self.get_min_range(distances, window)
        weights = self.get_weights(distances, min_range)

        if use_matrix or degree > 1:
            wm = np.multiply(np.eye(window), weights)
            xm = np.ones((window, degree + 1))

            xp = np.array([[math.pow(n_x, p)] for p in range(degree + 1)])
            for i in range(1, degree + 1):
                xm[:, i] = np.power(self.n_xx[min_range], i)

            ym = self.n_yy[min_range]
            xmt_wm = np.transpose(xm) @ wm
            beta = np.linalg.pinv(xmt_wm @ xm) @ xmt_wm @ ym
            y = (beta @ xp)[0]
        else:
            xx = self.n_xx[min_range]
            yy = self.n_yy[min_range]
            sum_weight = np.sum(weights)
            sum_weight_x = np.dot(xx, weights)
            sum_weight_y = np.dot(yy, weights)
            sum_weight_x2 = np.dot(np.multiply(xx, xx), weights)
            sum_weight_xy = np.dot(np.multiply(xx, yy), weights)

            mean_x = sum_weight_x / sum_weight
            mean_y = sum_weight_y / sum_weight

            b = (sum_weight_xy - mean_x * mean_y * sum_weight) / \
                (sum_weight_x2 - mean_x * mean_x * sum_weight)
            a = mean_y - b * mean_x
            y = a + b * n_x
        return self.denormalize_y(y)


def reactivation_cue_vector(norm_deconvolved, idx, y_pred, behavior, paths, day, days):
    activity = norm_deconvolved.to_numpy()
    activity = activity[idx['both'].index]

    sig_cells = preprocess.sig_reactivated_cells([], [], [], [], [], paths, 0)
    activity = activity[sig_cells > 0, :]

    mean_activity_cs_1 = []
    mean_activity_cs_2 = []
    onsets_cs_1 = []
    onsets_cs_2 = []
    for i in range(0, len(behavior['onsets'])):
        temp_activity = np.mean(activity[:, int(behavior['onsets'][i]):int(behavior['offsets'][i]) + 1], axis=1)
        if behavior['cue_codes'][i] == behavior['cs_1_code']:
            mean_activity_cs_1.append(temp_activity)
            onsets_cs_1.append(behavior['onsets'][i])
        if behavior['cue_codes'][i] == behavior['cs_2_code']:
            mean_activity_cs_2.append(temp_activity)
            onsets_cs_2.append(behavior['onsets'][i])

    mean_activity_cs_1_mean = [np.mean(mean_activity_cs_1[0:3], axis=0), np.mean(mean_activity_cs_1[len(mean_activity_cs_1) - 3:len(mean_activity_cs_1)], axis=0)]
    mean_activity_cs_2_mean = [np.mean(mean_activity_cs_2[0:3], axis=0), np.mean(mean_activity_cs_2[len(mean_activity_cs_2) - 3:len(mean_activity_cs_2)], axis=0)]

    mean_activity_cs_1_vec = mean_activity_cs_1_mean[1] - mean_activity_cs_1_mean[0]
    mean_activity_cs_2_vec = mean_activity_cs_2_mean[1] - mean_activity_cs_2_mean[0]

    for i in range(0, len(mean_activity_cs_1)):
        mean_activity_cs_1[i] = np.dot(mean_activity_cs_1[i], mean_activity_cs_1_vec) / np.linalg.norm(mean_activity_cs_1_vec)
    for i in range(0, len(mean_activity_cs_2)):
        mean_activity_cs_2[i] = np.dot(mean_activity_cs_2[i], mean_activity_cs_2_vec) / np.linalg.norm(mean_activity_cs_2_vec)

    mean_activity_r_1 = []
    mean_activity_r_2 = []
    trial_r_1 = []
    trial_r_2 = []
    reactivation_cs_1 = y_pred[:, 0].copy()
    reactivation_cs_2 = y_pred[:, 1].copy()
    p_threshold = .75
    cs_1_peak = 0
    cs_2_peak = 0
    i = 0
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
                    for j in range(0, len(onsets_cs_1)):
                        if r_start < onsets_cs_1[j] and r_start > onsets_cs_1[j-1] and r_start < onsets_cs_1[j-1] + int(behavior['framerate']*61):
                            mean_activity_r_1.append(np.mean(activity[:, r_start:r_end], axis=1))
                            trial_r_1.append(j-1)
                            break
                if cs_2_peak > p_threshold:
                    for j in range(0, len(onsets_cs_2)):
                        if r_start < onsets_cs_2[j] and r_start > onsets_cs_2[j-1] and r_start < onsets_cs_2[j-1] + int(behavior['framerate']*61):
                            mean_activity_r_2.append(np.mean(activity[:, r_start:r_end], axis=1))
                            trial_r_2.append(j-1)
                            break
                i = r_end
                cs_1_peak = 0
                cs_2_peak = 0

    for i in range(0, len(mean_activity_r_1)):
        mean_activity_r_1[i] = np.dot(mean_activity_r_1[i], mean_activity_cs_1_vec) / np.linalg.norm(mean_activity_cs_1_vec)
    for i in range(0, len(mean_activity_r_2)):
        mean_activity_r_2[i] = np.dot(mean_activity_r_2[i], mean_activity_cs_2_vec) / np.linalg.norm(mean_activity_cs_2_vec)

    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(9, 6))
    plt.subplots_adjust(wspace=.3)
    m_size = 10
    plt.subplot(2, 2, 1)
    plt.ylim(-1.7, .1)
    plt.gca().invert_yaxis()
    plt.scatter(x=list(range(0, len(mean_activity_cs_1))), y=mean_activity_cs_1, color='darkgreen', s=m_size)
    plt.scatter(x=trial_r_1, y=mean_activity_r_1, color='lime', s=m_size)
    plt.xlim(-2, 60)
    plt.xticks([0, 19, 39, 59], ['1', '20', '40', '60'])
    plt.ylabel('Similarity to early vs. late\n S1 response pattern')
    plt.xlabel('Trial number')
    label_1 = mlines.Line2D([], [], color='darkgreen', linewidth=2, label='S1')
    label_2 = mlines.Line2D([], [], color='lime', linewidth=2, label='S1 reactivation')
    label_3 = mlines.Line2D([], [], color='k', linewidth=2, label='S1 reactivation baseline')
    plt.legend(handles=[label_1, label_2, label_3], frameon=False)
    plt.subplot(2, 2, 2)
    plt.ylim(-1.7, .1)
    plt.gca().invert_yaxis()
    plt.scatter(x=list(range(0, len(mean_activity_cs_2))), y=mean_activity_cs_2, color='darkred', s=m_size)
    plt.scatter(x=trial_r_2, y=mean_activity_r_2, color='hotpink', s=m_size)
    plt.xlim(-2, 60)
    plt.xticks([0, 19, 39, 59], ['1', '20', '40', '60'])
    plt.ylabel('Similarity to early vs. late\n S2 response pattern')
    plt.xlabel('Trial number')
    label_1 = mlines.Line2D([], [], color='darkred', linewidth=2, label='S2')
    label_2 = mlines.Line2D([], [], color='hotpink', linewidth=2, label='S2 reactivation')
    label_3 = mlines.Line2D([], [], color='k', linewidth=2, label='S2 reactivation baseline')
    plt.legend(handles=[label_1, label_2, label_3], frameon=False)
    sns.despine()

    min_r_num = np.min([len(mean_activity_r_1), len(mean_activity_r_2)])
    w_size = min_r_num
    plt.subplot(2, 2, 1)
    xx = list(range(0, len(mean_activity_cs_1)))
    yy = mean_activity_cs_1
    loess = Loess(xx, yy)
    x_plot_s1 = []
    y_plot_s1 = []
    for x in range(0, 60):
        y = loess.estimate(x, window=w_size, use_matrix=False, degree=1)
        x_plot_s1.append(x)
        y_plot_s1.append(y)
    plt.plot(x_plot_s1, y_plot_s1, c='darkgreen', linewidth=3)
    xx = trial_r_1
    yy = mean_activity_r_1
    loess = Loess(xx, yy)
    x_plot_s1r = []
    y_plot_s1r = []
    for x in range(0, 60):
        y = loess.estimate(x, window=w_size, use_matrix=False, degree=1)
        x_plot_s1r.append(x)
        y_plot_s1r.append(y)
    plt.plot(x_plot_s1r, y_plot_s1r, c='lime', linewidth=3)
    plt.subplot(2, 2, 2)
    xx = list(range(0, len(mean_activity_cs_2)))
    yy = mean_activity_cs_2
    loess = Loess(xx, yy)
    x_plot_s2 = []
    y_plot_s2 = []
    for x in range(0, 60):
        y = loess.estimate(x, window=w_size, use_matrix=False, degree=1)
        x_plot_s2.append(x)
        y_plot_s2.append(y)
    plt.plot(x_plot_s2, y_plot_s2, c='darkred', linewidth=3)
    xx = trial_r_2
    yy = mean_activity_r_2
    loess = Loess(xx, yy)
    x_plot_s2r = []
    y_plot_s2r = []
    for x in range(0, 60):
        y = loess.estimate(x, window=w_size, use_matrix=False, degree=1)
        x_plot_s2r.append(x)
        y_plot_s2r.append(y)
    plt.plot(x_plot_s2r, y_plot_s2r, c='hotpink', linewidth=3)

    mean_activity_r_1_before = []
    mean_activity_r_2_before = []
    cs_1_peak = 0
    cs_2_peak = 0
    i = 0
    next_r = 0
    while i < int(behavior['frames_per_run']):
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
                    mean_activity_r_1_before.append(np.mean(activity[:, r_start:r_end], axis=1))
                if cs_2_peak > p_threshold:
                    mean_activity_r_2_before.append(np.mean(activity[:, r_start:r_end], axis=1))
                i = r_end
                cs_1_peak = 0
                cs_2_peak = 0
    if len(mean_activity_r_1_before) > 0:
        for i in range(0, len(mean_activity_r_1_before)):
            mean_activity_r_1_before[i] = np.dot(mean_activity_r_1_before[i], mean_activity_cs_1_vec) / np.linalg.norm(mean_activity_cs_1_vec)
        plt.subplot(2, 2, 1)
        plt.scatter(x=np.ones(len(mean_activity_r_1_before)) - 3.5, y=mean_activity_r_1_before, color='k', s=m_size)
        #plt.axvspan(-5, 0, alpha=.1, color='gray', zorder=0)
    if len(mean_activity_r_2_before) > 0:
        for i in range(0, len(mean_activity_r_2_before)):
            mean_activity_r_2_before[i] = np.dot(mean_activity_r_2_before[i], mean_activity_cs_2_vec) / np.linalg.norm(mean_activity_cs_2_vec)
        plt.subplot(2, 2, 2)
        plt.scatter(x=np.ones(len(mean_activity_r_2_before)) - 3.5, y=mean_activity_r_2_before, color='k', s=m_size)
        #plt.axvspan(-5, 0, alpha=.1, color='gray', zorder=0)

    #plt.close()
    plt.savefig(paths['save_path'] + 'plots/' + paths['mouse'] + '_' + paths['date'] + '_' +
                'reactivation_cue_vector.png', bbox_inches='tight', dpi=200)
    plt.savefig(paths['save_path'] + 'plots/' + paths['mouse'] + '_' + paths['date'] + '_' +
                'reactivation_cue_vector.pdf', bbox_inches='tight', dpi=200, transparent=True)

    days_path = paths['base_path'] + paths['mouse'] + '/data_across_days/'
    if days:
        if path.isfile(days_path + 'reactivation_cue_vector.npy') == 0 or day == 0:
            reactivation_cue_pca_vec = [list(range(0, days)), list(range(0, days)), list(range(0, days)),
                                        list(range(0, days)), list(range(0, days)), list(range(0, days))]
            reactivation_cue_pca_vec[0][day] = y_plot_s1
            reactivation_cue_pca_vec[1][day] = y_plot_s1r
            reactivation_cue_pca_vec[2][day] = y_plot_s2
            reactivation_cue_pca_vec[3][day] = y_plot_s2r
            if len(mean_activity_r_1_before) > 0:
                reactivation_cue_pca_vec[4][day] = np.mean(mean_activity_r_1_before)
            else:
                reactivation_cue_pca_vec[4][day] = np.nan
            if len(mean_activity_r_2_before) > 0:
                reactivation_cue_pca_vec[5][day] = np.mean(mean_activity_r_2_before)
            else:
                reactivation_cue_pca_vec[5][day] = np.nan
            np.save(days_path + 'reactivation_cue_vector', reactivation_cue_pca_vec)
        else:
            reactivation_cue_pca_vec = np.load(days_path + 'reactivation_cue_vector.npy', allow_pickle=True)
            reactivation_cue_pca_vec[0][day] = y_plot_s1
            reactivation_cue_pca_vec[1][day] = y_plot_s1r
            reactivation_cue_pca_vec[2][day] = y_plot_s2
            reactivation_cue_pca_vec[3][day] = y_plot_s2r
            if len(mean_activity_r_1_before) > 0:
                reactivation_cue_pca_vec[4][day] = np.mean(mean_activity_r_1_before)
            else:
                reactivation_cue_pca_vec[4][day] = np.nan
            if len(mean_activity_r_2_before) > 0:
                reactivation_cue_pca_vec[5][day] = np.mean(mean_activity_r_2_before)
            else:
                reactivation_cue_pca_vec[5][day] = np.nan
            np.save(days_path + 'reactivation_cue_vector', reactivation_cue_pca_vec)


def reactivation_cue_vector_evolve(norm_deconvolved, idx, y_pred, behavior, return_s, p_value, paths, day, days):
    activity = norm_deconvolved.to_numpy()
    activity = activity[idx['both'].index]

    sig_cells = preprocess.sig_reactivated_cells([], [], [], [], [], paths, 0)
    activity = activity[sig_cells > 0, :]

    mean_activity_cs_1 = []
    mean_activity_cs_2 = []
    onsets_cs_1 = []
    onsets_cs_2 = []
    for i in range(0, len(behavior['onsets'])):
        temp_activity = np.mean(activity[:, int(behavior['onsets'][i]):int(behavior['offsets'][i]) + 1], axis=1)
        if behavior['cue_codes'][i] == behavior['cs_1_code']:
            mean_activity_cs_1.append(temp_activity)
            onsets_cs_1.append(behavior['onsets'][i])
        if behavior['cue_codes'][i] == behavior['cs_2_code']:
            mean_activity_cs_2.append(temp_activity)
            onsets_cs_2.append(behavior['onsets'][i])

    mean_activity_cs_1_mean = [np.mean(mean_activity_cs_1[0:3], axis=0),
                               np.mean(mean_activity_cs_1[len(mean_activity_cs_1) - 3:len(mean_activity_cs_1)],
                                       axis=0)]
    mean_activity_cs_2_mean = [np.mean(mean_activity_cs_2[0:3], axis=0),
                               np.mean(mean_activity_cs_2[len(mean_activity_cs_2) - 3:len(mean_activity_cs_2)],
                                       axis=0)]

    mean_activity_cs_1_rt = []
    mean_activity_cs_2_rt = []
    mean_activity_cs_1_rt.append(mean_activity_cs_1[0])
    mean_activity_cs_2_rt.append(mean_activity_cs_2[0])

    mean_activity_cs_1_vec = mean_activity_cs_1_mean[1] - mean_activity_cs_1_mean[0]
    mean_activity_cs_2_vec = mean_activity_cs_2_mean[1] - mean_activity_cs_2_mean[0]

    for i in range(0, len(mean_activity_cs_1)):
        mean_activity_cs_1[i] = np.dot(mean_activity_cs_1[i], mean_activity_cs_1_vec) / np.linalg.norm(mean_activity_cs_1_vec)
    for i in range(0, len(mean_activity_cs_2)):
        mean_activity_cs_2[i] = np.dot(mean_activity_cs_2[i], mean_activity_cs_2_vec) / np.linalg.norm(mean_activity_cs_2_vec)

    reactivation_cs_1 = y_pred[:, 0].copy()
    reactivation_cs_2 = y_pred[:, 1].copy()
    p_threshold = .75
    cs_1_peak = 0
    cs_2_peak = 0
    i = 0
    r_start_cs_1 = np.empty(1000) * np.nan
    r_end_cs_1 = np.empty(1000) * np.nan
    r_start_cs_2 = np.empty(1000) * np.nan
    r_end_cs_2 = np.empty(1000) * np.nan
    num_r_cs_1 = 0
    num_r_cs_2 = 0
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
                    for j in range(0, len(onsets_cs_1)):
                        if r_start < onsets_cs_1[j] and r_start > onsets_cs_1[j-1] and r_start < onsets_cs_1[j-1] + int(behavior['framerate']*61):
                            r_start_cs_1[num_r_cs_1] = r_start
                            r_end_cs_1[num_r_cs_1] = r_end
                            num_r_cs_1 += 1
                            break
                if cs_2_peak > p_threshold:
                    for j in range(0, len(onsets_cs_2)):
                        if r_start < onsets_cs_2[j] and r_start > onsets_cs_2[j-1] and r_start < onsets_cs_2[j-1] + int(behavior['framerate']*61):
                            r_start_cs_2[num_r_cs_2] = r_start
                            r_end_cs_2[num_r_cs_2] = r_end
                            num_r_cs_2 += 1
                            break
                i = r_end
                cs_1_peak = 0
                cs_2_peak = 0
    r_start_cs_1 = r_start_cs_1[~np.isnan(r_start_cs_1)]
    r_start_cs_2 = r_start_cs_2[~np.isnan(r_start_cs_2)]
    r_end_cs_1 = r_end_cs_1[~np.isnan(r_end_cs_1)]
    r_end_cs_2 = r_end_cs_2[~np.isnan(r_end_cs_2)]

    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(9, 6))
    plt.subplots_adjust(wspace=.3)
    m_size = 10
    min_r_num = np.min([len(r_start_cs_1), len(r_start_cs_2)])
    w_size = min_r_num
    plt.subplot(2, 2, 1)
    #plt.ylim(-2.25, -.55)
    plt.gca().invert_yaxis()
    xx = list(range(0, len(mean_activity_cs_1)))
    yy = mean_activity_cs_1
    loess = Loess(xx, yy)
    x_plot_s1 = []
    y_plot_s1_t = []
    for x in range(0, 60):
        y = loess.estimate(x, window=w_size, use_matrix=False, degree=1)
        x_plot_s1.append(x)
        y_plot_s1_t.append(y)
    plt.plot(x_plot_s1, y_plot_s1_t, c='darkgreen', linewidth=3)
    plt.scatter(x=list(range(0, len(mean_activity_cs_1))), y=mean_activity_cs_1, color='darkgreen', s=m_size)
    plt.xlim(0, 60)
    plt.xticks([0, 19, 39, 59], ['1', '20', '40', '60'])
    plt.ylabel('Similarity to early vs. late\n S1 response pattern')
    plt.xlabel('Trial number')
    label_1 = mlines.Line2D([], [], color='darkgreen', linewidth=2, label='S1')
    label_2 = mlines.Line2D([], [], color='lime', linewidth=2, label='S1 modeled')
    plt.legend(handles=[label_1, label_2], frameon=False)
    plt.subplot(2, 2, 2)
    #plt.ylim(-2.25, -.55)
    plt.gca().invert_yaxis()
    xx = list(range(0, len(mean_activity_cs_2)))
    yy = mean_activity_cs_2
    loess = Loess(xx, yy)
    x_plot_s2 = []
    y_plot_s2_t = []
    for x in range(0, 60):
        y = loess.estimate(x, window=w_size, use_matrix=False, degree=1)
        x_plot_s2.append(x)
        y_plot_s2_t.append(y)
    plt.plot(x_plot_s2, y_plot_s2_t, c='darkred', linewidth=3)
    plt.scatter(x=list(range(0, len(mean_activity_cs_2))), y=mean_activity_cs_2, color='darkred', s=m_size)
    plt.xlim(0, 60)
    plt.xticks([0, 19, 39, 59], ['1', '20', '40', '60'])
    plt.ylabel('Similarity to early vs. late\n S2 response pattern')
    plt.xlabel('Trial number')
    label_1 = mlines.Line2D([], [], color='darkred', linewidth=2, label='S2')
    label_2 = mlines.Line2D([], [], color='hotpink', linewidth=2, label='S2 modeled')
    plt.legend(handles=[label_1, label_2], frameon=False)

    scale_factor = 1.4059200220209798 # get_reactivation_cue_scale(paths)
    p_value = .2
    for i in range(0, len(mean_activity_cs_1)-1):
        curr_cs = mean_activity_cs_1_rt[i].copy()
        for j in range(0, len(r_start_cs_1)):
            if onsets_cs_1[i] < r_start_cs_1[j] < onsets_cs_1[i + 1]:
                mean_r = np.mean(activity[:, int(r_start_cs_1[j]):int(r_end_cs_1[j])], axis=1)

                curr_cs = curr_cs + p_value*((mean_r*scale_factor)-curr_cs)


        mean_activity_cs_1_rt.append(curr_cs)


    for i in range(0, len(mean_activity_cs_2)-1):
        curr_cs = mean_activity_cs_2_rt[i].copy()
        for j in range(0, len(r_start_cs_2)):
            if onsets_cs_2[i] < r_start_cs_2[j] < onsets_cs_2[i + 1]:
                mean_r = np.mean(activity[:, int(r_start_cs_2[j]):int(r_end_cs_2[j])], axis=1)

                curr_cs = curr_cs + p_value*((mean_r*scale_factor)-curr_cs)


        mean_activity_cs_2_rt.append(curr_cs)

    if return_s == 1:
        mean_activity_all = []
        num_s1 = 0
        num_s2 = 0
        for i in range(0, len(behavior['onsets'])):
            if behavior['cue_codes'][i] == behavior['cs_1_code']:
                mean_activity_all.append(mean_activity_cs_1_rt[num_s1])
                num_s1 += 1
            if behavior['cue_codes'][i] == behavior['cs_2_code']:
                mean_activity_all.append(mean_activity_cs_2_rt[num_s2])
                num_s2 += 1
        plt.close()
        return mean_activity_all

    for i in range(0, len(mean_activity_cs_1_rt)):
        mean_activity_cs_1_rt[i] = np.dot(mean_activity_cs_1_rt[i], mean_activity_cs_1_vec) / np.linalg.norm(mean_activity_cs_1_vec)
    for i in range(0, len(mean_activity_cs_2_rt)):
        mean_activity_cs_2_rt[i] = np.dot(mean_activity_cs_2_rt[i], mean_activity_cs_2_vec) / np.linalg.norm(mean_activity_cs_2_vec)

    plt.subplot(2, 2, 1)
    xx = list(range(0, len(mean_activity_cs_1_rt)))
    yy = mean_activity_cs_1_rt
    loess = Loess(xx, yy)
    x_plot_s1 = []
    y_plot_s1 = []
    for x in range(0, 60):
        y = loess.estimate(x, window=w_size, use_matrix=False, degree=1)
        x_plot_s1.append(x)
        y_plot_s1.append(y)
    plt.plot(x_plot_s1, y_plot_s1, c='lime', linewidth=3)
    plt.scatter(x=list(range(0, len(mean_activity_cs_1_rt))), y=mean_activity_cs_1_rt, color='lime', s=m_size)
    plt.subplot(2, 2, 2)
    xx = list(range(0, len(mean_activity_cs_2_rt)))
    yy = mean_activity_cs_2_rt
    loess = Loess(xx, yy)
    x_plot_s2 = []
    y_plot_s2 = []
    for x in range(0, 60):
        y = loess.estimate(x, window=w_size, use_matrix=False, degree=1)
        x_plot_s2.append(x)
        y_plot_s2.append(y)
    plt.plot(x_plot_s2, y_plot_s2, c='hotpink', linewidth=3)
    plt.scatter(x=list(range(0, len(mean_activity_cs_2_rt))), y=mean_activity_cs_2_rt, color='hotpink', s=m_size)
    sns.despine()
    # plt.close()

    if return_s == 2:
        return [y_plot_s1, y_plot_s2, y_plot_s1_t, y_plot_s2_t]

    plt.savefig(paths['save_path'] + 'plots/' + paths['mouse'] + '_' + paths['date'] + '_' +
                'reactivation_cue_vector_evolve.png', bbox_inches='tight', dpi=200)
    plt.savefig(paths['save_path'] + 'plots/' + paths['mouse'] + '_' + paths['date'] + '_' +
                'reactivation_cue_vector_evolve.pdf', bbox_inches='tight', dpi=200, transparent=True)


    days_path = paths['base_path'] + paths['mouse'] + '/data_across_days/'
    if days:
        if path.isfile(days_path + 'reactivation_cue_vector_evolve.npy') == 0 or day == 0:
            reactivation_cue_pca_vec = [list(range(0, days)), list(range(0, days)), list(range(0, days)),
                                        list(range(0, days))]
            reactivation_cue_pca_vec[0][day] = y_plot_s1
            reactivation_cue_pca_vec[1][day] = y_plot_s2
            reactivation_cue_pca_vec[2][day] = y_plot_s1_t
            reactivation_cue_pca_vec[3][day] = y_plot_s2_t
            np.save(days_path + 'reactivation_cue_vector_evolve', reactivation_cue_pca_vec)
        else:
            reactivation_cue_pca_vec = np.load(days_path + 'reactivation_cue_vector_evolve.npy', allow_pickle=True)
            reactivation_cue_pca_vec[0][day] = y_plot_s1
            reactivation_cue_pca_vec[1][day] = y_plot_s2
            reactivation_cue_pca_vec[2][day] = y_plot_s1_t
            reactivation_cue_pca_vec[3][day] = y_plot_s2_t
            np.save(days_path + 'reactivation_cue_vector_evolve', reactivation_cue_pca_vec)


def activity_across_trials_evolve(norm_deconvolved, behavior, y_pred, idx, paths, day, days):
    """
    make correlation similarity and activity across trials
    :param norm_deconvolved: activity
    :param behavior: behavior
    :param y_pred: reactivation probabilities
    :param idx: index of sorted cells
    :param paths: path to save
    :param day: day
    :param days: days
    :return: correlation and activity and reactivations per trial
    """
    activity = reactivation_cue_vector_evolve(norm_deconvolved, idx, y_pred, behavior, 1, [], paths, day, days)

    y_pred_cs_1 = y_pred[:, 0]
    y_pred_cs_2 = y_pred[:, 1]
    all_y_pred = y_pred_cs_1 + y_pred_cs_2
    cue_start = behavior['onsets']
    end_trials = behavior['end_trials']
    reactivation_prob = []
    correlation = []
    cue_activity = []

    start = max(list(behavior['cue_codes']).index(behavior['cs_1_code']), list(behavior['cue_codes']).
                index(behavior['cs_2_code']))
    past_cs_1_type = behavior['cue_codes'][start - 1]
    past_cs_2_type = behavior['cue_codes'][start]
    past_cs_1_mean = activity[start - 1]
    past_cs_2_mean = activity[start]
    for i in range(start+1, len(cue_start)):
        current_cs_type = behavior['cue_codes'][i]
        if current_cs_type == past_cs_1_type:
            current_temp_mean = activity[i]
            if i not in end_trials:
                correlation.append(np.corrcoef(current_temp_mean, past_cs_2_mean)[0][1])
            past_cs_1_mean = current_temp_mean
        if current_cs_type == past_cs_2_type:
            current_temp_mean = activity[i]
            if i not in end_trials:
                correlation.append(np.corrcoef(current_temp_mean, past_cs_1_mean)[0][1])
            past_cs_2_mean = current_temp_mean
        temp_sum_reactivation = np.sum(all_y_pred[int(cue_start[i]):int(cue_start[i]) + int((behavior['framerate'] *
                                                                                             (behavior['iti'] + 6)))])
        temp_cue_activity = np.mean(activity[i])
        if i not in end_trials:
            reactivation_prob.append(temp_sum_reactivation)
            cue_activity.append(temp_cue_activity)

    reactivation_prob_cs_1 = []
    reactivation_prob_cs_2 = []
    correlation_cs_1 = []
    correlation_cs_2 = []
    start_cs_1 = list(behavior['cue_codes']).index(behavior['cs_1_code'])
    start_cs_2 = list(behavior['cue_codes']).index(behavior['cs_2_code'])
    past_cs_1_mean = activity[start_cs_1]
    past_cs_2_mean = activity[start_cs_2]
    for i in range(0, len(cue_start)):
        if i not in end_trials:
            if behavior['cue_codes'][i] == behavior['cs_1_code'] and i > start_cs_1:
                current_temp_mean = activity[i]
                correlation_cs_1.append(np.corrcoef(current_temp_mean, past_cs_1_mean)[0][1])
                reactivation_prob_cs_1.append(
                    np.sum(y_pred_cs_1[int(cue_start[i]):int(cue_start[i]) + int((behavior['framerate'] *
                                                                                 (behavior['iti'] + 6)))]))
                # past_cs_1_mean = current_temp_mean
            if behavior['cue_codes'][i] == behavior['cs_2_code'] and i > start_cs_2:
                current_temp_mean = activity[i]
                correlation_cs_2.append(np.corrcoef(current_temp_mean, past_cs_2_mean)[0][1])
                reactivation_prob_cs_2.append(
                    np.sum(y_pred_cs_2[int(cue_start[i]):int(cue_start[i]) + int((behavior['framerate'] *
                                                                                  (behavior['iti'] + 6)))]))
                # past_cs_2_mean = current_temp_mean

    days_path = paths['base_path'] + paths['mouse'] + '/data_across_days/'
    if days:
        if path.isfile(days_path + 'activity_evolve.npy') == 0 or day == 0:
            correlation_across_days = [list(range(0, days)), list(range(0, days)), list(range(0, days)),
                                       list(range(0, days)), list(range(0, days)), list(range(0, days)),
                                       list(range(0, days))]
            correlation_across_days[0][day] = correlation
            correlation_across_days[1][day] = reactivation_prob
            correlation_across_days[2][day] = cue_activity
            correlation_across_days[3][day] = correlation_cs_1
            correlation_across_days[4][day] = correlation_cs_2
            correlation_across_days[5][day] = reactivation_prob_cs_1
            correlation_across_days[6][day] = reactivation_prob_cs_2
            np.save(days_path + 'activity_evolve', correlation_across_days)
        else:
            correlation_across_days = np.load(days_path + 'activity_evolve.npy', allow_pickle=True)
            correlation_across_days[0][day] = correlation
            correlation_across_days[1][day] = reactivation_prob
            correlation_across_days[2][day] = cue_activity
            correlation_across_days[3][day] = correlation_cs_1
            correlation_across_days[4][day] = correlation_cs_2
            correlation_across_days[5][day] = reactivation_prob_cs_1
            correlation_across_days[6][day] = reactivation_prob_cs_2
            np.save(days_path + 'activity_evolve', correlation_across_days)


def activity_across_trials_evolve_grouped(norm_deconvolved, behavior, y_pred, idx, paths, day, days):
    """
    make correlation similarity and activity across trials
    :param norm_deconvolved: activity
    :param behavior: behavior
    :param idx: index of sorted cells
    :param paths: path to save
    :param day: day
    :param days: days
    :return: correlation and activity and reactivations per trial
    """
    activity = norm_deconvolved.to_numpy()
    activity = activity[idx['both'].index]
    sig_cells = preprocess.sig_reactivated_cells([], [], [], [], [], paths, 0)

    activity = activity[sig_cells > 0, :]
    [no_change_cells_cs_1, increase_sig_cells_cs_1, decrease_sig_cells_cs_1] = preprocess.group_neurons(activity, behavior, 'cs_1')

    [no_change_cells_cs_2, increase_sig_cells_cs_2, decrease_sig_cells_cs_2] = preprocess.group_neurons(activity, behavior, 'cs_2')

    all_correlation = []
    all_cue_activity = []
    for g in range(0, 3):
        activity = reactivation_cue_vector_evolve(norm_deconvolved, idx, y_pred, behavior, 1, [], paths, day, days)
        if g == 0:
            cells_to_use = no_change_cells_cs_1 + no_change_cells_cs_2
            # activity = activity[cells_to_use == 2, :]
            # cells_to_use = decrease_sig_cells_cs_1 + decrease_sig_cells_cs_2 + increase_sig_cells_cs_1 + increase_sig_cells_cs_2
            # activity = activity[cells_to_use > 0, :]
        if g == 1:
            cells_to_use = increase_sig_cells_cs_1 + increase_sig_cells_cs_2
            # activity = activity[cells_to_use > 0]
            # cells_to_use_1 = decrease_sig_cells_cs_1 + decrease_sig_cells_cs_2
            # cells_to_use_1[cells_to_use_1 > 0] = 1
            # cells_to_use_2 = no_change_cells_cs_1 + no_change_cells_cs_2
            # cells_to_use_2[cells_to_use_2 != 2] = 0
            # cells_to_use_2[cells_to_use_2 == 2] = 1
            # cells_to_use = cells_to_use_1 + cells_to_use_2
            # activity = activity[cells_to_use > 0, :]
        if g == 2:
            cells_to_use = decrease_sig_cells_cs_1 + decrease_sig_cells_cs_2
            # activity = activity[cells_to_use > 0]
            # cells_to_use_1 = increase_sig_cells_cs_1 + increase_sig_cells_cs_2
            # cells_to_use_1[cells_to_use_1 > 0] = 1
            # cells_to_use_2 = no_change_cells_cs_1 + no_change_cells_cs_2
            # cells_to_use_2[cells_to_use_2 != 2] = 0
            # cells_to_use_2[cells_to_use_2 == 2] = 1
            # cells_to_use = cells_to_use_1 + cells_to_use_2
            # activity = activity[cells_to_use > 0, :]

        cue_start = behavior['onsets']
        end_trials = behavior['end_trials']
        correlation = []
        cue_activity = []

        start = max(list(behavior['cue_codes']).index(behavior['cs_1_code']), list(behavior['cue_codes']).
                    index(behavior['cs_2_code']))
        past_cs_1_type = behavior['cue_codes'][start - 1]
        past_cs_2_type = behavior['cue_codes'][start]
        if g == 0:
            past_cs_1_mean = activity[start - 1][cells_to_use == 2]
            past_cs_2_mean = activity[start][cells_to_use == 2]
        if g == 1 or g == 2:
            past_cs_1_mean = activity[start - 1][cells_to_use > 0]
            past_cs_2_mean = activity[start][cells_to_use > 0]
        for i in range(start + 1, len(cue_start)):
            current_cs_type = behavior['cue_codes'][i]
            if current_cs_type == past_cs_1_type:
                if g == 0:
                    current_temp_mean = activity[i][cells_to_use == 2]
                if g == 1 or g == 2:
                    current_temp_mean = activity[i][cells_to_use > 0]
                if i not in end_trials:
                    correlation.append(np.corrcoef(current_temp_mean, past_cs_2_mean)[0][1])
                past_cs_1_mean = current_temp_mean
            if current_cs_type == past_cs_2_type:
                if g == 0:
                    current_temp_mean = activity[i][cells_to_use == 2]
                if g == 1 or g == 2:
                    current_temp_mean = activity[i][cells_to_use > 0]
                if i not in end_trials:
                    correlation.append(np.corrcoef(current_temp_mean, past_cs_1_mean)[0][1])
                past_cs_2_mean = current_temp_mean
            if g == 0:
                temp_cue_activity = np.mean(activity[i][cells_to_use == 2])
            if g == 1 or g == 2:
                temp_cue_activity = np.mean(activity[i][cells_to_use > 0])
            if i not in end_trials:
                cue_activity.append(temp_cue_activity)
        all_correlation.append(correlation)
        all_cue_activity.append(cue_activity)
    days_path = paths['base_path'] + paths['mouse'] + '/data_across_days/'
    if days:
        if path.isfile(days_path + 'activity_evolve_grouped.npy') == 0 or day == 0:
            correlation_across_days = [list(range(0, days)), list(range(0, days))]
            correlation_across_days[0][day] = all_correlation
            correlation_across_days[1][day] = all_cue_activity
            np.save(days_path + 'activity_evolve_grouped', correlation_across_days)
        else:
            correlation_across_days = np.load(days_path + 'activity_evolve_grouped.npy', allow_pickle=True)
            correlation_across_days[0][day] = all_correlation
            correlation_across_days[1][day] = all_cue_activity
            np.save(days_path + 'activity_evolve_grouped', correlation_across_days)


def activity_across_trials_evolve_grouped_decrease(norm_deconvolved, behavior, y_pred, idx, paths, day, days):
    """
    make correlation similarity and activity across trials
    :param norm_deconvolved: activity
    :param behavior: behavior
    :param idx: index of sorted cells
    :param paths: path to save
    :param day: day
    :param days: days
    :return: correlation and activity and reactivations per trial
    """
    activity = norm_deconvolved.to_numpy()
    activity = activity[idx['both'].index]
    sig_cells = preprocess.sig_reactivated_cells([], [], [], [], [], paths, 0)

    activity = activity[sig_cells > 0, :]
    [no_change_cells_cs_1, increase_sig_cells_cs_1, decrease_sig_cells_cs_1] = preprocess.group_neurons(activity, behavior, 'cs_1')

    [no_change_cells_cs_2, increase_sig_cells_cs_2, decrease_sig_cells_cs_2] = preprocess.group_neurons(activity, behavior, 'cs_2')

    cs1d_cs1 = []
    cs1d_cs2 = []
    cs2d_cs1 = []
    cs2d_cs2 = []
    activity = norm_deconvolved.to_numpy()
    activity = activity[idx['both'].index]
    sig_cells = preprocess.sig_reactivated_cells([], [], [], [], [], paths, 0)
    activity = activity[sig_cells > 0, :]

    cells_to_use_1 = no_change_cells_cs_1
    cells_to_use_2 = no_change_cells_cs_2

    activity = reactivation_cue_vector_evolve(norm_deconvolved, idx, y_pred, behavior, 1, [], paths, day, days)

    cue_start = behavior['onsets']
    cue_end = behavior['offsets']
    end_trials = behavior['end_trials']
    start = max(list(behavior['cue_codes']).index(behavior['cs_1_code']), list(behavior['cue_codes']).
                index(behavior['cs_2_code']))
    for i in range(start+1, len(cue_start)):
        if i not in end_trials:
            if behavior['cue_codes'][i] == behavior['cs_1_code']:
                cs1d_cs1.append(np.mean(activity[i][cells_to_use_1 > 0]))
                cs2d_cs1.append(np.mean(activity[i][cells_to_use_2 > 0]))
            if behavior['cue_codes'][i] == behavior['cs_2_code']:
                cs1d_cs2.append(np.mean(activity[i][cells_to_use_1 > 0]))
                cs2d_cs2.append(np.mean(activity[i][cells_to_use_2 > 0]))
    days_path = paths['base_path'] + paths['mouse'] + '/data_across_days/'
    if days:
        if path.isfile(days_path + 'activity_grouped_no_change_evolve.npy') == 0 or day == 0:
            across_days = [list(range(0, days)), list(range(0, days)), list(range(0, days)),
                           list(range(0, days))]
            across_days[0][day] = cs1d_cs1
            across_days[1][day] = cs1d_cs2
            across_days[2][day] = cs2d_cs2
            across_days[3][day] = cs2d_cs1
            np.save(days_path + 'activity_grouped_no_change_evolve', across_days)
        else:
            across_days = np.load(days_path + 'activity_grouped_no_change_evolve.npy', allow_pickle=True)
            across_days[0][day] = cs1d_cs1
            across_days[1][day] = cs1d_cs2
            across_days[2][day] = cs2d_cs2
            across_days[3][day] = cs2d_cs1
            np.save(days_path + 'activity_grouped_no_change_evolve', across_days)


def reactivation_cue_vector_evolve_parametric(norm_deconvolved, idx, y_pred, behavior, paths, day, days):
    all_y_plot_s1 = []
    all_y_plot_s2 = []
    all_y_plot_s1_t = []
    all_y_plot_s2_t = []
    p_values = np.array(range(0, 105, 5))/100
    for i in p_values:
        [y_plot_s1, y_plot_s2, y_plot_s1_t, y_plot_s2_t] = reactivation_cue_vector_evolve(norm_deconvolved, idx, y_pred,
                                                                                       behavior, 2, i, paths, day, days)
        all_y_plot_s1.append(y_plot_s1)
        all_y_plot_s2.append(y_plot_s2)
        all_y_plot_s1_t.append(y_plot_s1_t)
        all_y_plot_s2_t.append(y_plot_s2_t)

    days_path = paths['base_path'] + paths['mouse'] + '/data_across_days/'
    if days:
        if path.isfile(days_path + 'reactivation_cue_vector_evolve_parametric.npy') == 0 or day == 0:
            reactivation_cue_pca_vec = [list(range(0, days)), list(range(0, days)), list(range(0, days)),
                                        list(range(0, days))]
            reactivation_cue_pca_vec[0][day] = all_y_plot_s1
            reactivation_cue_pca_vec[1][day] = all_y_plot_s2
            reactivation_cue_pca_vec[2][day] = all_y_plot_s1_t
            reactivation_cue_pca_vec[3][day] = all_y_plot_s2_t
            np.save(days_path + 'reactivation_cue_vector_evolve_parametric', reactivation_cue_pca_vec)
        else:
            reactivation_cue_pca_vec = np.load(days_path + 'reactivation_cue_vector_evolve_parametric.npy',
                                               allow_pickle=True)
            reactivation_cue_pca_vec[0][day] = all_y_plot_s1
            reactivation_cue_pca_vec[1][day] = all_y_plot_s2
            reactivation_cue_pca_vec[2][day] = all_y_plot_s1_t
            reactivation_cue_pca_vec[3][day] = all_y_plot_s2_t
            np.save(days_path + 'reactivation_cue_vector_evolve_parametric', reactivation_cue_pca_vec)


def reactivation_cue_scale(norm_deconvolved, idx, y_pred, behavior, paths, day, days):
    activity = norm_deconvolved.to_numpy()
    activity = activity[idx['both'].index]

    sig_cells = preprocess.sig_reactivated_cells([], [], [], [], [], paths, 0)
    activity = activity[sig_cells > 0, :]

    mean_activity_cs = []
    for i in range(0, 20):
        temp_activity = np.mean(activity[:, int(behavior['onsets'][i]):int(behavior['offsets'][i]) + 1], axis=1)
        mean_activity_cs.append(temp_activity)
    mean_activity_cs = np.mean(mean_activity_cs)

    reactivation_cs_1 = y_pred[:, 0].copy()
    reactivation_cs_2 = y_pred[:, 1].copy()
    p_threshold = .75
    cs_1_peak = 0
    cs_2_peak = 0
    i = 0
    reactivation_response = []
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
                if cs_1_peak > p_threshold and int(behavior['onsets'][20]) > r_start > int(behavior['onsets'][0]):
                    temp_activity = np.mean(activity[:, r_start:r_end], axis=1)
                    reactivation_response.append(temp_activity)
                if cs_2_peak > p_threshold and int(behavior['onsets'][20]) > r_start > int(behavior['onsets'][0]):
                    temp_activity = np.mean(activity[:, r_start:r_end], axis=1)
                    reactivation_response.append(temp_activity)
                i = r_end
                cs_1_peak = 0
                cs_2_peak = 0
    reactivation_response = np.mean(reactivation_response)

    days_path = paths['base_path'] + paths['mouse'] + '/data_across_days/'
    if days:
        if path.isfile(days_path + 'reactivation_cue_scale.npy') == 0 or day == 0:
            across_days_vec = [list(range(0, days)), list(range(0, days))]
            across_days_vec[0][day] = mean_activity_cs
            across_days_vec[1][day] = reactivation_response
            np.save(days_path + 'reactivation_cue_scale', across_days_vec)
        else:
            across_days_vec = np.load(days_path + 'reactivation_cue_scale.npy', allow_pickle=True)
            across_days_vec[0][day] = mean_activity_cs
            across_days_vec[1][day] = reactivation_response
            np.save(days_path + 'reactivation_cue_scale', across_days_vec)


def reactivation_influence(norm_deconvolved, behavior, y_pred, idx, paths, day, days):
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(6, 6))
    plt.subplots_adjust(wspace=.35)

    [y1_cs_1, y2_cs_1, y3_cs_1] = reactivation_influence_helper(norm_deconvolved, behavior, y_pred, idx, 'cs_1', 1, paths)
    [y1_cs_2, y2_cs_2, y3_cs_2] = reactivation_influence_helper(norm_deconvolved, behavior, y_pred, idx, 'cs_2', 2, paths)

    days_path = paths['base_path'] + paths['mouse'] + '/data_across_days/'
    if days:
        if path.isfile(days_path + 'reactivation_influence.npy') == 0 or day == 0:
            across_days = [list(range(0, days)), list(range(0, days)), list(range(0, days)), list(range(0, days)),
                           list(range(0, days)), list(range(0, days))]
            across_days[0][day] = y1_cs_1
            across_days[1][day] = y2_cs_1
            across_days[2][day] = y3_cs_1
            across_days[3][day] = y1_cs_2
            across_days[4][day] = y2_cs_2
            across_days[5][day] = y3_cs_2
            np.save(days_path + 'reactivation_influence', across_days)
        else:
            across_days = np.load(days_path + 'reactivation_influence.npy', allow_pickle=True)
            across_days[0][day] = y1_cs_1
            across_days[1][day] = y2_cs_1
            across_days[2][day] = y3_cs_1
            across_days[3][day] = y1_cs_2
            across_days[4][day] = y2_cs_2
            across_days[5][day] = y3_cs_2
            np.save(days_path + 'reactivation_influence', across_days)


def reactivation_influence_helper(norm_deconvolved, behavior, y_pred, idx, trial_type, plt_num, paths):
    activity = norm_deconvolved.to_numpy()
    activity = activity[idx['both'].index]
    sig_cells = preprocess.sig_reactivated_cells([], [], [], [], [], paths, 0)
    activity = activity[sig_cells > 0, :]

    index_frames_start = []
    index_frames_end = []
    num_trials_total = 10
    num_trials = 0
    trial_times = behavior['onsets'][behavior['cue_codes'] == behavior[trial_type + '_code']]
    for i in trial_times:
        if i > 0 and num_trials < num_trials_total:
            for j in range(0, behavior['frames_before']):
                index_frames_start.append(int(i) + j)
            num_trials = num_trials + 1
    num_trials = 0
    for i in reversed(trial_times):
        if i > 0 and num_trials < num_trials_total:
            for j in range(0, behavior['frames_before']):
                index_frames_end.append(int(i) + j)
            num_trials = num_trials + 1

    activity_start = activity[:, index_frames_start]
    activity_start = np.reshape(activity_start, (activity_start.shape[0], num_trials_total, behavior['frames_before']))
    activity_end = activity[:, index_frames_end]
    activity_end = np.reshape(activity_end, (activity_end.shape[0], num_trials_total, behavior['frames_before']))

    increase_sig_cells = np.zeros(len(activity_start))
    decrease_sig_cells = np.zeros(len(activity_start))
    no_change_cells = np.zeros(len(activity_start))

    dist = np.zeros(len(activity_start))
    for j in range(len(activity_start)):
        before = np.reshape(activity_start[j, :, 0:behavior['frames_before']], (num_trials * behavior['frames_before']))
        after = np.reshape(activity_end[j, :, 0:behavior['frames_before']], (num_trials * behavior['frames_before']))
        dist[j] = (np.mean(after) - np.mean(before)) / np.mean(before)

    for j in range(len(activity_start)):
        if dist[j] > np.mean(dist) + np.std(dist):
            increase_sig_cells[j] = 1
        if dist[j] < np.mean(dist) - np.std(dist):
            decrease_sig_cells[j] = 1
        if np.mean(dist) - (np.std(dist) / 2) < dist[j] < np.mean(dist) + (np.std(dist) / 2):
            no_change_cells[j] = 1

    reactivation_cs_1 = y_pred[:, 0].copy()
    reactivation_cs_2 = y_pred[:, 1].copy()
    p_threshold = .75
    cs_1_peak = 0
    cs_2_peak = 0
    i = 0
    reactivation_times = np.zeros(len(reactivation_cs_1))
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
                if cs_1_peak > p_threshold and index_frames_start[len(index_frames_start)-1] + int(behavior['framerate']*61) > r_start > int(behavior['onsets'][0]) and trial_type == 'cs_1':
                    reactivation_times[r_start:r_end] = 1
                if cs_2_peak > p_threshold and index_frames_start[len(index_frames_start)-1] + int(behavior['framerate']*61) > r_start > int(behavior['onsets'][0]) and trial_type == 'cs_2':
                    reactivation_times[r_start:r_end] = 1
                i = r_end
                cs_1_peak = 0
                cs_2_peak = 0

    scale_factor = 1.4059200220209798 # get_reactivation_cue_scale(paths)

    dist_r = np.zeros(len(activity_start))
    for j in range(len(activity_start)):
        cue_s = np.reshape(activity_start[j, :, 0:behavior['frames_before']], (num_trials * behavior['frames_before']))
        reactivation_s = activity[j, reactivation_times == 1]
        dist_r[j] = ((np.mean(reactivation_s)*scale_factor) - np.mean(cue_s))
    dist_r_decrease = dist_r[decrease_sig_cells == 1]
    dist_r_increase = dist_r[increase_sig_cells == 1]
    dist_r_no_change = dist_r[no_change_cells == 1]

    plt.subplot(2, 2, plt_num)
    if trial_type == 'cs_1':
        c = 'mediumseagreen'
        plt.ylabel('S1 early - S1R early')
    c = 'mediumseagreen'
    if trial_type == 'cs_2':
        c = 'salmon'
        plt.ylabel('S2 early - S2R early')
    x = np.array(list(range(0, len(dist_r_decrease))))/len(dist_r_decrease)
    rand = np.random.permutation(len(dist_r_decrease))
    plt.scatter(x[rand]+.25, dist_r_decrease, s=10, c=c)

    x = np.array(list(range(0, len(dist_r_no_change)))) / len(dist_r_no_change)
    rand = np.random.permutation(len(dist_r_no_change))
    plt.scatter(x[rand]+2, dist_r_no_change, s=10, c=c)

    x = np.array(list(range(0, len(dist_r_increase)))) / len(dist_r_increase)
    rand = np.random.permutation(len(dist_r_increase))
    plt.scatter(x[rand]+1.75, dist_r_increase, s=10, c=c)
    sns.despine()

    plt.xticks([.75, 2.25, 3.75], ['Decrease\ncells', 'No change\ncells', 'Increase\ncells'])

    y1 = np.mean(dist_r_decrease)
    y1_err = stats.sem(dist_r_decrease)
    plt.errorbar(.75, y1, yerr=y1_err, c='k', linewidth=2, marker='o', mfc='k', mec='k',
                 ms=7, mew=0, zorder=100)
    y2 = np.mean(dist_r_no_change)
    y2_err = stats.sem(dist_r_no_change)
    plt.errorbar(2.25, y2, yerr=y2_err, c='k', linewidth=2, marker='o', mfc='k', mec='k',
                 ms=7, mew=0, zorder=100)
    y3 = np.mean(dist_r_increase)
    y3_err = stats.sem(dist_r_increase)
    plt.errorbar(3.75, y3, yerr=y3_err, c='k', linewidth=2, marker='o', mfc='k', mec='k',
                 ms=7, mew=0, zorder=100)
    plt.ylim(-0.25, .2)
    plt.xlim([0, 4])
    plt.savefig(paths['save_path'] + 'plots/' + paths['mouse'] + '_' + paths['date'] + '_' +
                'reactivation_influence.png', bbox_inches='tight', dpi=200)
    plt.savefig(paths['save_path'] + 'plots/' + paths['mouse'] + '_' + paths['date'] + '_' +
                'reactivation_influence.pdf', bbox_inches='tight', dpi=200, transparent=True)

    return [y1, y2, y3]


def grouped_count(norm_deconvolved, behavior, idx, paths, day, days):

    activity = norm_deconvolved.to_numpy()
    activity = activity[idx['both'].index]
    sig_cells = preprocess.sig_reactivated_cells([], [], [], [], [], paths, 0)

    activity = activity[sig_cells > 0, :]
    [no_change_cells_cs_1, increase_sig_cells_cs_1, decrease_sig_cells_cs_1] = preprocess.group_neurons(activity, behavior, 'cs_1')

    [no_change_cells_cs_2, increase_sig_cells_cs_2, decrease_sig_cells_cs_2] = preprocess.group_neurons(activity, behavior, 'cs_2')

    no_change_cells = no_change_cells_cs_1 + no_change_cells_cs_2
    increase_sig_cells = increase_sig_cells_cs_1 + increase_sig_cells_cs_2
    decrease_sig_cells = decrease_sig_cells_cs_1 + decrease_sig_cells_cs_2

    days_path = paths['base_path'] + paths['mouse'] + '/data_across_days/'
    if days:
        if path.isfile(days_path + 'activity_grouped_count.npy') == 0 or day == 0:
            num_across_days = [list(range(0, days)), list(range(0, days)), list(range(0, days)), list(range(0, days))]
            num_across_days[0][day] = sum(no_change_cells == 2)
            num_across_days[1][day] = sum(increase_sig_cells > 0)
            num_across_days[2][day] = sum(decrease_sig_cells > 0)
            num_across_days[3][day] = len(activity)
            np.save(days_path + 'activity_grouped_count', num_across_days)
        else:
            num_across_days = np.load(days_path + 'activity_grouped_count.npy', allow_pickle=True)
            num_across_days[0][day] = sum(no_change_cells == 2)
            num_across_days[1][day] = sum(increase_sig_cells > 0)
            num_across_days[2][day] = sum(decrease_sig_cells > 0)
            num_across_days[3][day] = len(activity)
            np.save(days_path + 'activity_grouped_count', num_across_days)


def activity_difference_grouped(norm_deconvolved, behavior, y_pred, idx, paths, day, days):
    """
    make correlation similarity and activity across trials
    :param norm_deconvolved: activity
    :param behavior: behavior
    :param idx: index of sorted cells
    :param paths: path to save
    :param day: day
    :param days: days
    :return: correlation and activity and reactivations per trial
    """
    activity = norm_deconvolved.to_numpy()
    activity = activity[idx['both'].index]
    sig_cells = preprocess.sig_reactivated_cells([], [], [], [], [], paths, 0)

    activity = activity[sig_cells > 0, :]
    [no_change_cells_cs_1, increase_sig_cells_cs_1, decrease_sig_cells_cs_1] = preprocess.group_neurons(activity, behavior, 'cs_1')

    [no_change_cells_cs_2, increase_sig_cells_cs_2, decrease_sig_cells_cs_2] = preprocess.group_neurons(activity, behavior, 'cs_2')

    cs1c_nc = []
    cs1c_d = []
    cs1c_i = []
    cs1r_nc = []
    cs1r_d = []
    cs1r_i = []
    cs2c_nc = []
    cs2c_d = []
    cs2c_i = []
    cs2r_nc = []
    cs2r_d = []
    cs2r_i = []

    activity = norm_deconvolved.to_numpy()
    activity = activity[idx['both'].index]
    sig_cells = preprocess.sig_reactivated_cells([], [], [], [], [], paths, 0)
    activity = activity[sig_cells > 0, :]

    cue_start = behavior['onsets']
    cue_end = behavior['offsets']
    end_trials = behavior['end_trials']
    for i in range(0, 20):
        if i not in end_trials:
            if behavior['cue_codes'][i] == behavior['cs_1_code']:
                cs1c_nc.append(np.nanmean(activity[no_change_cells_cs_1 > 0, int(cue_start[i]):int(cue_end[i]) + 1]))
                cs1c_d.append(np.nanmean(activity[decrease_sig_cells_cs_1 > 0, int(cue_start[i]):int(cue_end[i]) + 1]))
                cs1c_i.append(np.nanmean(activity[increase_sig_cells_cs_1 > 0, int(cue_start[i]):int(cue_end[i]) + 1]))
            if behavior['cue_codes'][i] == behavior['cs_2_code']:
                cs2c_nc.append(np.nanmean(activity[no_change_cells_cs_2 > 0, int(cue_start[i]):int(cue_end[i]) + 1]))
                cs2c_d.append(np.nanmean(activity[decrease_sig_cells_cs_2 > 0, int(cue_start[i]):int(cue_end[i]) + 1]))
                cs2c_i.append(np.nanmean(activity[increase_sig_cells_cs_2 > 0, int(cue_start[i]):int(cue_end[i]) + 1]))
    cs1c_nc = np.nanmean(cs1c_nc)
    cs1c_d = np.nanmean(cs1c_d)
    cs1c_i = np.nanmean(cs1c_i)
    cs2c_nc = np.nanmean(cs2c_nc)
    cs2c_d = np.nanmean(cs2c_d)
    cs2c_i = np.nanmean(cs2c_i)

    reactivation_cs_1 = y_pred[:, 0].copy()
    reactivation_cs_2 = y_pred[:, 1].copy()
    p_threshold = .75
    cs_1_peak = 0
    cs_2_peak = 0
    i = 0
    reactivation_times_cs_1 = np.zeros(len(reactivation_cs_1))
    reactivation_times_cs_2 = np.zeros(len(reactivation_cs_1))
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
                if cs_1_peak > p_threshold and int(behavior['onsets'][20]) > r_start > int(behavior['onsets'][0]):
                    reactivation_times_cs_1[r_start:r_end] = 1
                if cs_2_peak > p_threshold and int(behavior['onsets'][20]) > r_start > int(behavior['onsets'][0]):
                    reactivation_times_cs_2[r_start:r_end] = 1
                i = r_end
                cs_1_peak = 0
                cs_2_peak = 0

    cs1r_nc = np.nanmean(activity[no_change_cells_cs_1 > 0, :][:, reactivation_times_cs_1 == 1])
    cs1r_d = np.nanmean(activity[decrease_sig_cells_cs_1 > 0, :][:, reactivation_times_cs_1 == 1])
    cs1r_i = np.nanmean(activity[increase_sig_cells_cs_1 > 0, :][:, reactivation_times_cs_1 == 1])
    cs2r_nc = np.nanmean(activity[no_change_cells_cs_2 > 0, :][:, reactivation_times_cs_2 == 1])
    cs2r_d = np.nanmean(activity[decrease_sig_cells_cs_2 > 0, :][:, reactivation_times_cs_2 == 1])
    cs2r_i = np.nanmean(activity[increase_sig_cells_cs_2 > 0, :][:, reactivation_times_cs_2 == 1])

    days_path = paths['base_path'] + paths['mouse'] + '/data_across_days/'
    if days:
        if path.isfile(days_path + 'activity_difference_grouped.npy') == 0 or day == 0:
            across_days = [list(range(0, days)), list(range(0, days)), list(range(0, days)),
                           list(range(0, days)), list(range(0, days)), list(range(0, days)), list(range(0, days)),
                           list(range(0, days))]
            across_days[0][day] = cs1c_d / cs1c_nc
            across_days[1][day] = cs2c_d / cs2c_nc
            across_days[2][day] = cs1c_i / cs1c_nc
            across_days[3][day] = cs2c_i / cs2c_nc
            across_days[4][day] = cs1r_d / cs1r_nc
            across_days[5][day] = cs2r_d / cs2r_nc
            across_days[6][day] = cs1r_i / cs1r_nc
            across_days[7][day] = cs2r_i / cs2r_nc
            np.save(days_path + 'activity_difference_grouped', across_days)
        else:
            across_days = np.load(days_path + 'activity_difference_grouped.npy', allow_pickle=True)
            across_days[0][day] = cs1c_d / cs1c_nc
            across_days[1][day] = cs2c_d / cs2c_nc
            across_days[2][day] = cs1c_i / cs1c_nc
            across_days[3][day] = cs2c_i / cs2c_nc
            across_days[4][day] = cs1r_d / cs1r_nc
            across_days[5][day] = cs2r_d / cs2r_nc
            across_days[6][day] = cs1r_i / cs1r_nc
            across_days[7][day] = cs2r_i / cs2r_nc
            np.save(days_path + 'activity_difference_grouped', across_days)































