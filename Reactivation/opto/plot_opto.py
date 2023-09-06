import math
import scipy
import warnings
import preprocess_opto
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from os import path
from scipy.io import loadmat
from scipy import stats
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
warnings.filterwarnings('ignore')


def sorted_map(behavior, responses_cs_1, responses_cs_2, responses_cs_1_opto, responses_cs_2_opto, cs_1_idx, cs_2_idx,
               neuron_number, paths):
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
    opto_cells_to_remove = preprocess_opto.opto_cells(3, paths)
    cs_1_idx_no_opto = cs_1_idx.index[opto_cells_to_remove[cs_1_idx.index] == 0]
    cs_2_idx_no_opto = cs_2_idx.index[opto_cells_to_remove[cs_2_idx.index] == 0]
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4, figsize=(10, 10))
    make_sorted_map(behavior, responses_cs_1, cs_1_idx_no_opto, neuron_number, ax1, 1, 'mediumseagreen', 'CS 1')
    make_sorted_map(behavior, responses_cs_1_opto, cs_1_idx_no_opto, neuron_number, ax2, 0, 'mediumseagreen', 'CS 1 opto')
    make_sorted_map(behavior, responses_cs_2, cs_2_idx_no_opto, neuron_number, ax3, 0, 'salmon', 'CS 2')
    make_sorted_map(behavior, responses_cs_2_opto, cs_2_idx_no_opto, neuron_number, ax4, 0, 'salmon', 'CS 2 opto')
    plt.subplots_adjust(right=1)
    plt.savefig(paths['save_path'] + 'plots/' + paths['mouse'] + '_' + paths['date'] + '_' + 'cue_heatmap.png',
                bbox_inches='tight', dpi=500)
    plt.close(fig)


def make_sorted_map(behavior, mean_responses, idx, num_neurons, axes, label, color, title):
    """
    makes heatmap
    :param behavior: dict of behavior
    :param mean_responses: fluorescence
    :param idx: sorted index
    :param num_neurons: number of neurons to plot
    :param axes: axes handle
    :param label: label or not
    :return: heatmap plot
    """
    mean_responses = mean_responses.reindex(idx)
    frames_before = behavior['frames_before']
    sns.set(font_scale=1)
    ax = sns.heatmap(mean_responses, vmin=0, vmax=.2, cmap='Greys', cbar=0, yticklabels=False, ax=axes)
    # ax.set_xticks([0, frames_before, frames_before * 2, frames_before * 3, frames_before * 4,
    #                frames_before * 5])
    ax.set_xticks([])
    # ax.set_xticklabels(['-2', '0', '2', '4', '6', '8'], rotation=0)
    # if label == 1:
        # ax.set_yticks([0, 75, 150, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600])
        # ax.set_yticklabels(['0', '75', '150', '200', '400', '600', '800', '1000', '1200', '1400', '1600',
        #                     '1800', '2000', '2200', '2400', '2600'], rotation=0)
        # ax.set(ylabel='Neuron number sorted by cue 1')
    ax.set_ylim(num_neurons, 0)
    ax.axvline(x=frames_before + .25, color=color, linestyle='-', linewidth=5, snap=False)
    ax.axvline(x=frames_before * 2 + .25, color=color, linestyle='-', linewidth=5, snap=False)
    # ax.set(xlabel='Time relative to stimulus onset (s)')
    # sns.set_style("ticks")
    ax.tick_params(axis=u'both', which=u'both', length=0)
    ax.set_xlim(0, frames_before * 3)
    # ax.set_title(title, c=color)


def reactivation_raster(behavior, activity, y_pred, idx_1, idx_2, both_idx, paths, session):
    """
    makes heatmap
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
        cue_idx = preprocess_opto.cue_times(behavior, 0, 0)
        if session > behavior['dark_runs']:
            ax = sns.heatmap(cue_idx[:, (session - 1) * frames_per_run:session * frames_per_run],
                             cmap=['white', 'green', 'firebrick', 'lime', 'hotpink'], cbar=0)
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


def sample_reactivation_raster(behavior, activity, y_pred, idx_1, idx_2, paths, start, end):
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
    num_neurons = 200
    fig = plt.figure(figsize=(10, 12))
    gs0 = gridspec.GridSpec(2, 1, height_ratios=[1, .7874])
    gs1 = gridspec.GridSpecFromSubplotSpec(nrows=5, ncols=1, height_ratios=[1.25, 1.25, 1.25, 3, 25],
                                           subplot_spec=gs0[0])
    gs2 = gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=gs0[1])
    variables = {'num_neurons': num_neurons, 'labels': 1}
    sample_plot_reactivation(behavior, activity, idx_1, variables, gs1, fig, y_pred, 'CS 1', start, end)
    variables = {'num_neurons': num_neurons, 'labels': 2}
    sample_plot_reactivation(behavior, activity, idx_2, variables, gs2, fig, y_pred, 'CS 2', start, end)
    plt.savefig(paths['save_path'] + 'plots/' + paths['mouse'] + '_' + paths['date'] + '_' +
                'sample_reactivation_heatmap.png',  bbox_inches='tight', dpi=500)
    #plt.close(fig)


def sample_plot_reactivation(behavior, norm_moving_deconvolved, idx, variables, gs, fig, y_pred, start, end):
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
    norm_moving_deconvolved = pd.DataFrame(norm_moving_deconvolved)
    sorted_deconvolved = norm_moving_deconvolved.reindex(idx.index[0:variables['num_neurons']])
    frames_per_run = int(behavior['frames_per_run'])
    text_place = -250
    sns.set(font_scale=1)
    if variables['labels'] == 1:

        fig.add_subplot(gs[gs_num])
        plt.plot(behavior['pupil'][start:end], lw=1.5)
        plt.xlim((0, end-start))
        plt.plot([0, frames_per_run*5], [behavior['pupil_max'].mean(), behavior['pupil_max'].mean()], 'k--', lw=1.5)
        plt.axis('off')
        plt.text(text_place, np.mean(behavior['pupil'][start:end]), 'Pupil area (a.u.)', color='b', fontsize=17)
        gs_num = gs_num + 1
        fig.add_subplot(gs[gs_num])

        plt.plot(behavior['brain_motion'][start:end], color='darkgoldenrod', lw=1.5)
        plt.xlim((0, end-start))
        plt.axis('off')
        plt.text(text_place, 0, 'Brain motion (μm)', color='darkgoldenrod', fontsize=17)
        gs_num = gs_num + 1
        fig.add_subplot(gs[gs_num])
        cue_idx = preprocess_opto.cue_times(behavior, 0, 0)
        ax = sns.heatmap(cue_idx[:, start:end],
                         cmap=['white', 'lime', 'mediumseagreen', 'hotpink', 'hotpink'], cbar=0)
        ax.set(xticklabels=[])
        gs_num = gs_num + 1
        plt.axis('off')
        fig.add_subplot(gs[gs_num])

        plt.plot(y_pred[start:end, 1], color='salmon',
                 lw=1.5)
        plt.plot(y_pred[start:end, 0],
                 color='mediumseagreen', lw=1.5)
        plt.plot([0, frames_per_run], [1, 1], 'k--', lw=.5)
        plt.text(text_place, .5, 'Reactivation', color='k', fontsize=17)
        plt.text(text_place, .1, 'probability', color='k', fontsize=17)
        plt.xlim((0, end-start))
        plt.ylim((0, 1))
        plt.axis('off')
        gs_num = gs_num + 1
        fig.add_subplot(gs[gs_num])

    if variables['labels'] == 0 or variables['labels'] == 2:
        fig.add_subplot(gs[0])
    ax = sns.heatmap(sorted_deconvolved.iloc[:, start:end], vmin=0, vmax=.75, cmap='Greys', cbar=0)
    ax.set_yticks(range(0, len(sorted_deconvolved) + 1, 50))
    #ax.set_yticklabels(['0', '50', '100', '150', '200', '250', '300'], rotation=0)
    ax.set_ylim(len(sorted_deconvolved), 0)
    if variables['labels'] == 1 or variables['labels'] == 2:
        ax.set(xticklabels=[])
    if variables['labels'] == 0:
        ax.set(xlabel='Frame')
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.025)


def reactivation_rate(y_pred, behavior, paths, day, days):
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
    y_pred_cs_2 = y_pred[:, 1]
    y_pred_cs_1 = y_pred[:, 0]
    y_pred_all = y_pred_cs_1 + y_pred_cs_2
    times_considered = preprocess_opto.get_times_considered(y_pred, behavior)
    y_pred_all_opto = y_pred_cs_1 + y_pred_cs_2
    times_considered_opto = preprocess_opto.get_times_considered(y_pred, behavior)
    y_pred_all_total = y_pred_cs_1 + y_pred_cs_2
    times_considered_total = preprocess_opto.get_times_considered(y_pred, behavior)
    for i in range(0, len(behavior['onsets'])-1):
        cue_onset = int(behavior['onsets'][i])
        next_cue_onset = int(behavior['onsets'][i+1])
        if behavior['cue_codes'][i] == behavior['cs_1_code']:
            y_pred_all_opto[cue_onset:next_cue_onset] = 0
            times_considered_opto[cue_onset:next_cue_onset] = 0
        if behavior['cue_codes'][i] == behavior['cs_2_code']:
            y_pred_all_opto[cue_onset:next_cue_onset] = 0
            times_considered_opto[cue_onset:next_cue_onset] = 0
        if behavior['cue_codes'][i] == behavior['cs_1_opto_code']:
            y_pred_all[cue_onset:next_cue_onset] = 0
            times_considered[cue_onset:next_cue_onset] = 0
        if behavior['cue_codes'][i] == behavior['cs_2_opto_code']:
            y_pred_all[cue_onset:next_cue_onset] = 0
            times_considered[cue_onset:next_cue_onset] = 0

    y_pred_binned = []
    y_pred_binned_total = []
    y_pred_binned_opto = []
    x_label = []
    factor = 2
    idx = .5
    for i in range(0, behavior['dark_runs']+behavior['task_runs']):
        if i == 0:
            y_pred_binned.append(np.sum(y_pred_all[(i * frames_per_run):(i + 1) * frames_per_run]) /
                                 (np.sum(times_considered[(i * frames_per_run):(i + 1) * frames_per_run]) /
                                  int(behavior['framerate'])))
            y_pred_binned_opto.append(np.sum(y_pred_all_opto[(i * frames_per_run):(i + 1) * frames_per_run]) /
                                 (np.sum(times_considered_opto[(i * frames_per_run):(i + 1) * frames_per_run]) /
                                  int(behavior['framerate'])))
            y_pred_binned_total.append(np.sum(y_pred_all_total[(i * frames_per_run):(i + 1) * frames_per_run]) /
                                 (np.sum(times_considered_total[(i * frames_per_run):(i + 1) * frames_per_run]) /
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
                y_pred_binned_opto.append(np.sum(y_pred_all_opto[start + (j * step):start + ((j + 1) * step)]) /
                                     (np.sum(times_considered_opto[start + (j * step):start + ((j + 1) * step)]) /
                                      int(behavior['framerate'])))
                y_pred_binned_total.append(np.sum(y_pred_all_total[start + (j * step):start + ((j + 1) * step)]) /
                                     (np.sum(times_considered_total[start + (j * step):start + ((j + 1) * step)]) /
                                      int(behavior['framerate'])))
                x_label.append((step/behavior['framerate']/60/60 * idx)[0][0])
                idx = idx + 1
    sns.set(font_scale=1.5)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(8, 5))
    plt.plot(x_label[0:2], y_pred_binned[0:2], '--ok')
    plt.plot(x_label[1:len(x_label)], y_pred_binned[1:len(y_pred_binned)], '-ok')
    plt.plot(x_label[0:2], y_pred_binned_opto[0:2], '--or')
    plt.plot(x_label[1:len(x_label)], y_pred_binned_opto[1:len(y_pred_binned_opto)], '-or')
    plt.axvspan(-.5, 0, alpha=.25, color='gray')
    plt.ylabel('Mean reactivation probability (/s)')
    plt.xlabel('Time from first cue presentation (hours)')
    sns.despine()
    plt.savefig(paths['save_path'] + 'plots/' + paths['mouse'] + '_' + paths['date'] + '_' +
                'mean_reactivation_binned.png', bbox_inches='tight', dpi=150)
    plt.close()

    days_path = paths['base_path'] + paths['mouse'] + '/data_across_days/'
    if days:
        if path.isfile(days_path + 'y_pred_binned.npy') == 0 or day == 0:
            y_pred_binned_across_days = [list(range(0, days)), list(range(0, days)), list(range(0, days)),
                                         list(range(0, days))]
            y_pred_binned_across_days[0][day] = y_pred_binned
            y_pred_binned_across_days[1][day] = y_pred_binned_opto
            y_pred_binned_across_days[2][day] = x_label
            y_pred_binned_across_days[3][day] = y_pred_binned_total
            np.save(days_path + 'y_pred_binned', y_pred_binned_across_days)
        else:
            y_pred_binned_across_days = np.load(days_path + 'y_pred_binned.npy', allow_pickle=True)
            y_pred_binned_across_days[0][day] = y_pred_binned
            y_pred_binned_across_days[1][day] = y_pred_binned_opto
            y_pred_binned_across_days[2][day] = x_label
            y_pred_binned_across_days[3][day] = y_pred_binned_total
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
    y_pred_rate =y_pred_cs_1 + y_pred_cs_2
    y_pred_rate_norm = np.zeros(len(y_pred_cs_1_bias))
    y_pred_bias_opto = np.zeros(len(y_pred_cs_1_bias))
    y_pred_rate_opto = np.zeros(len(y_pred_cs_1_bias))
    for i in range(0, len(behavior['onsets']) - 1):
        start = int(behavior['onsets'][i])
        end = int(behavior['onsets'][i+1])
        if behavior['cue_codes'][i][0] == behavior['cs_1_code']:
            y_pred_bias[start:end] = y_pred_cs_1_bias[start:end]
            y_pred_rate_norm[start:end] = y_pred_rate[start:end]
        if behavior['cue_codes'][i][0] == behavior['cs_2_code']:
            y_pred_bias[start:end] = y_pred_cs_2_bias[start:end]
            y_pred_rate_norm[start:end] = y_pred_rate[start:end]
        if behavior['cue_codes'][i][0] == behavior['cs_1_opto_code']:
            y_pred_bias_opto[start:end] = y_pred_cs_1_bias[start:end]
            y_pred_rate_opto[start:end] = y_pred_rate[start:end]
        if behavior['cue_codes'][i][0] == behavior['cs_2_opto_code']:
            y_pred_bias_opto[start:end] = y_pred_cs_2_bias[start:end]
            y_pred_rate_opto[start:end] = y_pred_rate[start:end]
    y_pred_binned = []
    y_pred_binned_opto = []
    x_label = []
    factor = 2
    idx = .5
    for i in range(0, behavior['task_runs']):
        trials_per_run = int(int(len(behavior['onsets'])) / int(behavior['task_runs']))
        start = int(behavior['onsets'][i*trials_per_run])
        end = int(behavior['offsets'][((i+1)*trials_per_run) - 1])
        step = round((end-start)/factor)+1
        for j in range(0, factor):
            y_pred_binned.append(np.sum(y_pred_bias[start + (j * step):start + ((j + 1) * step)]) /
                                        np.sum(y_pred_rate_norm[start + (j * step):start + ((j + 1) * step)]))
            y_pred_binned_opto.append(np.sum(y_pred_bias_opto[start + (j * step):start + ((j + 1) * step)]) /
                                      np.sum(y_pred_rate_opto[start + (j * step):start + ((j + 1) * step)]))
            x_label.append((step/behavior['framerate']/60/60 * idx)[0][0])
            idx = idx + 1

    days_path = paths['base_path'] + paths['mouse'] + '/data_across_days/'
    if days:
        if path.isfile(days_path + 'y_pred_bias_binned.npy') == 0 or day == 0:
            y_pred_bias_binned_across_days = [list(range(0, days)), list(range(0, days)), list(range(0, days))]
            y_pred_bias_binned_across_days[0][day] = y_pred_binned
            y_pred_bias_binned_across_days[1][day] = y_pred_binned_opto
            y_pred_bias_binned_across_days[2][day] = x_label
            np.save(days_path + 'y_pred_bias_binned', y_pred_bias_binned_across_days)
        else:
            y_pred_bias_binned_across_days = np.load(days_path + 'y_pred_bias_binned.npy', allow_pickle=True)
            y_pred_bias_binned_across_days[0][day] = y_pred_binned
            y_pred_bias_binned_across_days[1][day] = y_pred_binned_opto
            y_pred_bias_binned_across_days[2][day] = x_label
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
    times_considered = preprocess_opto.get_times_considered(y_pred, behavior)
    fr = behavior['framerate']
    duration = int(fr * (behavior['iti'] + 5)) + 1
    end_trials = behavior['end_trials']
    rate_norm = np.empty((len(behavior['cue_codes']), duration)) * np.nan
    rate_opto = np.empty((len(behavior['cue_codes']), duration)) * np.nan
    times_considered_norm = np.empty((len(behavior['cue_codes']), duration)) * np.nan
    times_considered_opto = np.empty((len(behavior['cue_codes']), duration)) * np.nan

    for i in range(0, len(behavior['cue_codes'])):
        if i not in end_trials:
            if (behavior['cue_codes'][i] == behavior['cs_1_code'] or behavior['cue_codes'][i] == behavior['cs_2_code']):
                idx_curr = int(behavior['onsets'][i])
                rate_norm[i, :] = y_pred[:, 0][idx_curr:idx_curr+duration] + y_pred[:, 1][idx_curr:idx_curr+duration]
                times_considered_norm[i, :] = times_considered[idx_curr:idx_curr+duration]
            if (behavior['cue_codes'][i] == behavior['cs_1_opto_code'] or behavior['cue_codes'][i] == behavior['cs_2_opto_code']):
                idx_curr = int(behavior['onsets'][i])
                rate_opto[i, :] = y_pred[:, 0][idx_curr:idx_curr+duration] + y_pred[:, 1][idx_curr:idx_curr+duration]
                times_considered_opto[i, :] = times_considered[idx_curr:idx_curr+duration]
    rate_norm_binned = []
    rate_opto_binned = []
    x_binned = []
    factor = 5
    duration = int(fr * 8)
    step = int(((len(rate_norm[0])) - duration) / factor)
    for i in range(0, factor):
        rate_norm_binned.append(
            np.nansum(rate_norm[:, duration + (i * step):duration + (i + 1) * step]) *
            int(fr) / np.nansum(times_considered_norm[:, duration + (i * step):duration + (i + 1) * step]))
        rate_opto_binned.append(
            np.nansum(rate_opto[:, duration + (i * step):duration + (i + 1) * step]) *
            int(fr) / np.nansum(times_considered_opto[:, duration + (i * step):duration + (i + 1) * step]))
        x_binned.append(duration+(i + .5) * step)

    days_path = paths['base_path'] + paths['mouse'] + '/data_across_days/'
    if days:
        if path.isfile(days_path + 'rate_within_trial.npy') == 0 or day == 0:
            rate_within_trial_across_days = [list(range(0, days)), list(range(0, days)), list(range(0, days))]
            rate_within_trial_across_days[0][day] = rate_norm_binned
            rate_within_trial_across_days[1][day] = rate_opto_binned
            rate_within_trial_across_days[2][day] = x_binned
            np.save(days_path + 'rate_within_trial', rate_within_trial_across_days)
        else:
            rate_within_trial_across_days = np.load(days_path + 'rate_within_trial.npy', allow_pickle=True)
            rate_within_trial_across_days[0][day] = rate_norm_binned
            rate_within_trial_across_days[1][day] = rate_opto_binned
            rate_within_trial_across_days[2][day] = x_binned
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
    bias_opto = np.empty((len(behavior['cue_codes']), duration)) * np.nan
    total_norm = np.empty((len(behavior['cue_codes']), duration)) * np.nan
    total_opto = np.empty((len(behavior['cue_codes']), duration)) * np.nan

    for i in range(0, len(behavior['cue_codes'])):
        if i not in end_trials:
            idx_curr = int(behavior['onsets'][i])
            if behavior['cue_codes'][i] == behavior['cs_1_code']:
                bias_norm[i, :] = y_pred[:, 0][idx_curr:idx_curr+duration] - y_pred[:, 1][idx_curr:idx_curr+duration]
                total_norm[i, :] = y_pred[:, 0][idx_curr:idx_curr+duration] + y_pred[:, 1][idx_curr:idx_curr+duration]
            if behavior['cue_codes'][i] == behavior['cs_2_code']:
                bias_norm[i, :] = y_pred[:, 1][idx_curr:idx_curr+duration] - y_pred[:, 0][idx_curr:idx_curr+duration]
                total_norm[i, :] = y_pred[:, 0][idx_curr:idx_curr+duration] + y_pred[:, 1][idx_curr:idx_curr+duration]
            if behavior['cue_codes'][i] == behavior['cs_1_opto_code']:
                bias_opto[i, :] = y_pred[:, 0][idx_curr:idx_curr+duration] - y_pred[:, 1][idx_curr:idx_curr+duration]
                total_opto[i, :] = y_pred[:, 0][idx_curr:idx_curr+duration] + y_pred[:, 1][idx_curr:idx_curr+duration]
            if behavior['cue_codes'][i] == behavior['cs_2_opto_code']:
                bias_opto[i, :] = y_pred[:, 1][idx_curr:idx_curr+duration] - y_pred[:, 0][idx_curr:idx_curr+duration]
                total_opto[i, :] = y_pred[:, 0][idx_curr:idx_curr+duration] + y_pred[:, 1][idx_curr:idx_curr+duration]

    bias_norm_binned = []
    bias_opto_binned = []
    x_binned = []
    factor = 5
    duration = int(fr * 8)
    step = int(((len(bias_norm[0])) - duration) / factor)
    for i in range(0, factor):
        bias_norm_binned.append(
            np.nansum(bias_norm[:, duration + (i * step):duration + (i + 1) * step]) /
            np.nansum(total_norm[:, duration + (i * step):duration + (i + 1) * step]))
        bias_opto_binned.append(
            np.nansum(bias_opto[:, duration + (i * step):duration + (i + 1) * step]) /
            np.nansum(total_opto[:, duration + (i * step):duration + (i + 1) * step]))
        x_binned.append(duration+(i + .5) * step)

    days_path = paths['base_path'] + paths['mouse'] + '/data_across_days/'
    if days:
        if path.isfile(days_path + 'bias_within_trial.npy') == 0 or day == 0:
            bias_within_trial_across_days = [list(range(0, days)), list(range(0, days)), list(range(0, days))]
            bias_within_trial_across_days[0][day] = bias_norm_binned
            bias_within_trial_across_days[1][day] = bias_opto_binned
            bias_within_trial_across_days[2][day] = x_binned
            np.save(days_path + 'bias_within_trial', bias_within_trial_across_days)
        else:
            bias_within_trial_across_days = np.load(days_path + 'bias_within_trial.npy', allow_pickle=True)
            bias_within_trial_across_days[0][day] = bias_norm_binned
            bias_within_trial_across_days[1][day] = bias_opto_binned
            bias_within_trial_across_days[2][day] = x_binned
            np.save(days_path + 'bias_within_trial', bias_within_trial_across_days)


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
    p_threshold = 0.75
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
                r_end = i + 1
                i = r_end
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
    mean_phase_correlation = make_reactivation_physical('phase_correlation', r_times, behavior, 5, 'darkgoldenrod',
                                                        'Phase correlation')
    plt.savefig(paths['save_path'] + 'plots/' + paths['mouse'] + '_' + paths['date'] + '_'
                + 'reactivation_physical.png', bbox_inches='tight', dpi=150)
    plt.close()

    days_path = paths['base_path'] + paths['mouse'] + '/data_across_days/'
    if days:
        if path.isfile(days_path + 'reactivation_physical.npy') == 0 or day == 0:
            reactivation_physical_across_days = [list(range(0, days)), list(range(0, days)), list(range(0, days)),
                                                 list(range(0, days))]
            reactivation_physical_across_days[0][day] = mean_pupil
            reactivation_physical_across_days[1][day] = mean_pupil_movement
            reactivation_physical_across_days[2][day] = mean_brain_motion
            reactivation_physical_across_days[3][day] = mean_phase_correlation
            np.save(days_path + 'reactivation_physical', reactivation_physical_across_days)
        else:
            reactivation_physical_across_days = np.load(days_path + 'reactivation_physical.npy', allow_pickle=True)
            reactivation_physical_across_days[0][day] = mean_pupil
            reactivation_physical_across_days[1][day] = mean_pupil_movement
            reactivation_physical_across_days[2][day] = mean_brain_motion
            reactivation_physical_across_days[3][day] = mean_phase_correlation
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
        if time_window < time < len(behavior[physical_type]) - (time_window+1):
            if physical_type == 'pupil':
                temp_physical = behavior[physical_type][time - time_window:time + time_window + 1] / behavior['pupil_max']
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


def pupil_control(behavior, paths, day, days):
    """
    plot pupil on control vs opto trials
    :param behavior: behavior
    :param paths: path
    :param day: day
    :param days: days
    :return: pupil
    """
    fr = behavior['framerate']
    frames_before = int(behavior['framerate'] * 5)
    frames_after = int(fr * (behavior['iti'] + 5)) + 1
    trial_types = ['cs_1', 'cs_2', 'cs_1_opto', 'cs_2_opto']
    pupil_cs_1 = []
    pupil_cs_2 = []
    pupil_cs_1_opto = []
    pupil_cs_2_opto = []
    for trial_type in trial_types:
        pupil_vec = np.zeros((frames_before + frames_after, behavior[trial_type + '_trials']))
        num_trials = 0
        trial_number = 0
        for i in behavior['onsets']:
            temp_pupil = behavior['pupil'][int(i) - frames_before:int(i) + frames_after] / behavior['pupil_max']
            if behavior['cue_codes'][trial_number] == behavior[trial_type + '_code']:
                pupil_vec[:, num_trials] = temp_pupil
                num_trials += 1
            trial_number += 1
        if trial_type == 'cs_1':
            pupil_cs_1 = np.mean(pupil_vec, axis=1)
        if trial_type == 'cs_2':
            pupil_cs_2 = np.mean(pupil_vec, axis=1)
        if trial_type == 'cs_1_opto':
            pupil_cs_1_opto = np.mean(pupil_vec, axis=1)
        if trial_type == 'cs_2_opto':
            pupil_cs_2_opto = np.mean(pupil_vec, axis=1)

    days_path = paths['base_path'] + paths['mouse'] + '/data_across_days/'
    if days:
        if path.isfile(days_path + 'pupil.npy') == 0 or day == 0:
            pupil_across_days = [list(range(0, days)), list(range(0, days)), list(range(0, days)),
                                 list(range(0, days))]
            pupil_across_days[0][day] = pupil_cs_1
            pupil_across_days[1][day] = pupil_cs_2
            pupil_across_days[2][day] = pupil_cs_1_opto
            pupil_across_days[3][day] = pupil_cs_2_opto
            np.save(days_path + 'pupil', pupil_across_days)
        else:
            pupil_across_days = np.load(days_path + 'pupil.npy', allow_pickle=True)
            pupil_across_days[0][day] = pupil_cs_1
            pupil_across_days[1][day] = pupil_cs_2
            pupil_across_days[2][day] = pupil_cs_1_opto
            pupil_across_days[3][day] = pupil_cs_2_opto
            np.save(days_path + 'pupil', pupil_across_days)


def activity_control(norm_deconvolved, behavior, idx, paths, day, days):
    """
    plot cs activity on control vs opto trials
    :param norm_deconvolved: activity
    :param behavior: behavior
    :param idx: index
    :param paths: path
    :param day: day
    :param days: days
    :return: cs activity
    """
    activity = norm_deconvolved.to_numpy()
    opto_cells_to_remove = preprocess_opto.opto_cells(3, paths)
    cell_index_no_opto = idx['both'].index[opto_cells_to_remove[idx['both'].index] == 0]
    activity = activity[cell_index_no_opto]
    cue_start = behavior['onsets']
    fr = behavior['framerate']
    iti_start = int(behavior['framerate'] * 5)
    iti_end = int(fr * (behavior['iti'] + 5)) + 1
    iti_activity = np.empty((len(cue_start), iti_end+iti_start)) * np.nan
    iti_activity_opto = np.empty((len(cue_start), iti_end+iti_start)) * np.nan
    end_trials = behavior['end_trials']
    for i in range(0, len(cue_start)):
        if i not in end_trials and (
                behavior['cue_codes'][i] == behavior['cs_1_code'] or behavior['cue_codes'][i] == behavior['cs_2_code']):
            temp_iti_activity = np.mean(activity[:, int(cue_start[i]) - iti_start:int(cue_start[i]) + iti_end], axis=0)
            iti_activity[i, :] = temp_iti_activity
        if i not in end_trials and (
                behavior['cue_codes'][i] == behavior['cs_1_opto_code'] or behavior['cue_codes'][i] == behavior[
            'cs_2_opto_code']):
            temp_iti_activity = np.mean(activity[:, int(cue_start[i]) - iti_start:int(cue_start[i]) + iti_end], axis=0)
            iti_activity_opto[i, :] = temp_iti_activity
    iti_activity = np.nanmean(iti_activity, axis=0)
    iti_activity_opto = np.nanmean(iti_activity_opto, axis=0)
    days_path = paths['base_path'] + paths['mouse'] + '/data_across_days/'
    if days:
        if path.isfile(days_path + 'activity_control.npy') == 0 or day == 0:
            activity_control_across_days = [list(range(0, days)), list(range(0, days))]
            activity_control_across_days[0][day] = iti_activity
            activity_control_across_days[1][day] = iti_activity_opto
            np.save(days_path + 'activity_control', activity_control_across_days)
        else:
            activity_control_across_days = np.load(days_path + 'activity_control.npy', allow_pickle=True)
            activity_control_across_days[0][day] = iti_activity
            activity_control_across_days[1][day] = iti_activity_opto
            np.save(days_path + 'activity_control', activity_control_across_days)


def activity_difference(norm_deconvolved, behavior, idx, paths, day, days):
    """
    plot cs activity on control vs opto trials
    :param norm_deconvolved: activity
    :param behavior: behavior
    :param idx: index
    :param paths: path
    :param day: day
    :param days: days
    :return: cs activity
    """
    activity = norm_deconvolved.to_numpy()
    opto_cells_to_remove = preprocess_opto.opto_cells(3, paths)
    cell_index_no_opto = idx['both'].index[opto_cells_to_remove[idx['both'].index] == 0]
    activity = activity[cell_index_no_opto]
    cue_start = behavior['onsets']
    fr = behavior['framerate']
    iti_start = int(behavior['framerate'] * 5)
    iti_end = int(fr * (behavior['iti'] + 5)) + 1
    iti_activity = np.empty((len(cue_start), iti_end+iti_start)) * np.nan
    iti_activity_opto = np.empty((len(cue_start), iti_end+iti_start)) * np.nan
    end_trials = behavior['end_trials']
    for i in range(0, len(cue_start)):
        if i not in end_trials and (
                behavior['cue_codes'][i] == behavior['cs_1_code'] or behavior['cue_codes'][i] == behavior['cs_2_code']):
            temp_iti_activity = np.mean(activity[:, int(cue_start[i]) - iti_start:int(cue_start[i]) + iti_end], axis=0)
            iti_activity[i, :] = temp_iti_activity
        if i not in end_trials and (
                behavior['cue_codes'][i] == behavior['cs_1_opto_code'] or behavior['cue_codes'][i] == behavior[
            'cs_2_opto_code']):
            temp_iti_activity = np.mean(activity[:, int(cue_start[i]) - iti_start:int(cue_start[i]) + iti_end], axis=0)
            iti_activity_opto[i, :] = temp_iti_activity
    iti_activity = np.nanmean(iti_activity, axis=0)
    iti_activity_opto = np.nanmean(iti_activity_opto, axis=0)
    days_path = paths['base_path'] + paths['mouse'] + '/data_across_days/'
    if days:
        if path.isfile(days_path + 'activity_control.npy') == 0 or day == 0:
            activity_control_across_days = [list(range(0, days)), list(range(0, days))]
            activity_control_across_days[0][day] = iti_activity
            activity_control_across_days[1][day] = iti_activity_opto
            np.save(days_path + 'activity_control', activity_control_across_days)
        else:
            activity_control_across_days = np.load(days_path + 'activity_control.npy', allow_pickle=True)
            activity_control_across_days[0][day] = iti_activity
            activity_control_across_days[1][day] = iti_activity_opto
            np.save(days_path + 'activity_control', activity_control_across_days)


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


def neuron_count_R2(norm_deconvolved, behavior, idx, paths, day, days):

    activity = norm_deconvolved.to_numpy()
    total_num = len(activity)

    S1_cells = idx['cs_1'].index
    S2_cells = idx['cs_2'].index
    S_overlap = len(np.intersect1d(S1_cells, S2_cells))
    S1_only = len(np.setdiff1d(S1_cells, S2_cells))
    S2_only = len(np.setdiff1d(S2_cells, S1_cells))

    activity = activity[idx['both'].index]
    sig_cells = preprocess_opto.sig_reactivated_cells([], [], [], [], [], paths, 0)
    activity = activity[sig_cells > 0, :]
    [no_change_cells_cs_1, increase_sig_cells_cs_1, decrease_sig_cells_cs_1] = preprocess_opto.group_neurons(activity, behavior, 'cs_1')
    [no_change_cells_cs_2, increase_sig_cells_cs_2, decrease_sig_cells_cs_2] = preprocess_opto.group_neurons(activity, behavior, 'cs_2')
    no_change_cells = no_change_cells_cs_1 + no_change_cells_cs_2
    increase_sig_cells = increase_sig_cells_cs_1 + increase_sig_cells_cs_2
    decrease_sig_cells = decrease_sig_cells_cs_1 + decrease_sig_cells_cs_2

    no_change_num = sum(no_change_cells == 2)
    increase_num = sum(increase_sig_cells > 0)
    decrease_num = sum(decrease_sig_cells > 0)

    # get sig cells
    [cs_1_poscells, cs_1_negcells] = preprocess_opto.sig_test_R2(norm_deconvolved, behavior, 'cs_1', 'start')
    [cs_2_poscells, cs_2_negcells] = preprocess_opto.sig_test_R2(norm_deconvolved, behavior, 'cs_2', 'start')
    [both_poscells, _] = preprocess_opto.combine_sig(cs_1_poscells, cs_1_negcells, cs_2_poscells, cs_2_negcells)
    S_overlap_start = cs_1_poscells + cs_2_poscells
    S_overlap_start = sum(S_overlap_start == 2)
    S1_only_s = both_poscells - cs_2_poscells
    S1_only_s = sum(S1_only_s == 1)
    S2_only_s = both_poscells - cs_1_poscells
    S2_only_s = sum(S2_only_s == 1)

    [cs_1_poscells, cs_1_negcells] = preprocess_opto.sig_test_R2(norm_deconvolved, behavior, 'cs_1', 'end')
    [cs_2_poscells, cs_2_negcells] = preprocess_opto.sig_test_R2(norm_deconvolved, behavior, 'cs_2', 'end')
    [both_poscells, _] = preprocess_opto.combine_sig(cs_1_poscells, cs_1_negcells, cs_2_poscells, cs_2_negcells)
    S_overlap_end = cs_1_poscells + cs_2_poscells
    S_overlap_end = sum(S_overlap_end == 2)
    S1_only_e = both_poscells - cs_2_poscells
    S1_only_e = sum(S1_only_e == 1)
    S2_only_e = both_poscells - cs_1_poscells
    S2_only_e = sum(S2_only_e == 1)


    days_path = paths['base_path'] + paths['mouse'] + '/data_across_days/'
    if days:
        if path.isfile(days_path + 'grouped_count.npy') == 0 or day == 0:
            num_across_days = [list(range(0, days)), list(range(0, days)), list(range(0, days)), list(range(0, days)),
                               list(range(0, days)), list(range(0, days)), list(range(0, days)), list(range(0, days)),
                               list(range(0, days)), list(range(0, days)), list(range(0, days)), list(range(0, days)),
                               list(range(0, days))]
            num_across_days[0][day] = total_num
            num_across_days[1][day] = S_overlap
            num_across_days[2][day] = S1_only
            num_across_days[3][day] = S2_only
            num_across_days[4][day] = S_overlap_start
            num_across_days[5][day] = S1_only_s
            num_across_days[6][day] = S2_only_s
            num_across_days[7][day] = S_overlap_end
            num_across_days[8][day] = S1_only_e
            num_across_days[9][day] = S2_only_e
            num_across_days[10][day] = no_change_num
            num_across_days[11][day] = increase_num
            num_across_days[12][day] = decrease_num
            np.save(days_path + 'grouped_count', num_across_days)
        else:
            num_across_days = np.load(days_path + 'grouped_count.npy', allow_pickle=True)
            num_across_days[0][day] = total_num
            num_across_days[1][day] = S_overlap
            num_across_days[2][day] = S1_only
            num_across_days[3][day] = S2_only
            num_across_days[4][day] = S_overlap_start
            num_across_days[5][day] = S1_only_s
            num_across_days[6][day] = S2_only_s
            num_across_days[7][day] = S_overlap_end
            num_across_days[8][day] = S1_only_e
            num_across_days[9][day] = S2_only_e
            num_across_days[10][day] = no_change_num
            num_across_days[11][day] = increase_num
            num_across_days[12][day] = decrease_num
            np.save(days_path + 'grouped_count', num_across_days)


def neuron_dist_R3(norm_deconvolved, behavior, idx_cell, paths, day, days):

    runs = int(behavior['task_runs']) + int(behavior['dark_runs'])
    frames_per_run = int(behavior['frames_per_run'])
    pre_cue_idx = np.zeros((1, runs * frames_per_run))
    for i in range(len(behavior['onsets'])):
        cue_number = []
        if behavior['cue_codes'][i] == behavior['cs_1_code']:
            cue_number = 1
            cue_onset = int(behavior['onsets'][i])
            pre_cue_time = int(2 * behavior['framerate']) + 1
            for j in range(1, pre_cue_time):
                idx = cue_onset - j
                pre_cue_idx[0, idx] = cue_number
        if behavior['cue_codes'][i] == behavior['cs_2_code']:
            cue_number = 2
            cue_onset = int(behavior['onsets'][i])
            pre_cue_time = int(2 * behavior['framerate']) + 1
            for j in range(1, pre_cue_time):
                idx = cue_onset - j
                pre_cue_idx[0, idx] = cue_number

    cue_idx = np.zeros((1, runs * frames_per_run))
    for i in range(len(behavior['onsets'])):
        cue_number = []
        if behavior['cue_codes'][i] == behavior['cs_1_code']:
            cue_number = 1
            cue_onset = int(behavior['onsets'][i])
            cue_offset = int(behavior['offsets'][i])
            cue_time = cue_offset - cue_onset + 1
            for j in range(0, cue_time):
                idx = cue_onset + j
                cue_idx[0, idx] = cue_number
        if behavior['cue_codes'][i] == behavior['cs_2_code']:
            cue_number = 2
            cue_onset = int(behavior['onsets'][i])
            cue_offset = int(behavior['offsets'][i])
            cue_time = cue_offset - cue_onset + 1
            for j in range(0, cue_time):
                idx = cue_onset + j
                cue_idx[0, idx] = cue_number

    cs_1_times = cue_idx[0] == 1
    cs_2_times = cue_idx[0] == 2
    cs_1_pre_times = pre_cue_idx[0] == 1
    cs_2_pre_times = pre_cue_idx[0] == 2

    activity = norm_deconvolved.reindex(idx_cell['both'].index)
    activity = activity.to_numpy()

    activity_cs_1 = np.mean(activity[:, cs_1_times], axis=1) - np.mean(activity[:, cs_1_pre_times], axis=1)
    activity_cs_2 = np.mean(activity[:, cs_2_times], axis=1) - np.mean(activity[:, cs_2_pre_times], axis=1)

    activity_bias = (activity_cs_1 - activity_cs_2) / (activity_cs_1 + activity_cs_2)
    data_all = sns.kdeplot(data=activity_bias, clip=(-1, 1), common_norm=True).get_lines()
    plt.close()

    days_path = paths['base_path'] + paths['mouse'] + '/data_across_days/'
    if days:
        if path.isfile(days_path + 'overlap_dist.npy') == 0 or day == 0:
            num_across_days = [list(range(0, days)), list(range(0, days))]
            num_across_days[0][day] = data_all[0].get_data()[1]
            num_across_days[1][day] = data_all[0].get_data()[0]
            np.save(days_path + 'overlap_dist', num_across_days)
        else:
            num_across_days = np.load(days_path + 'overlap_dist.npy', allow_pickle=True)
            num_across_days[0][day] = data_all[0].get_data()[1]
            num_across_days[1][day] = data_all[0].get_data()[0]
            np.save(days_path + 'overlap_dist', num_across_days)


def neuron_dist_grouped(norm_deconvolved, behavior, idx_cell, paths, day, days):

    runs = int(behavior['task_runs']) + int(behavior['dark_runs'])
    frames_per_run = int(behavior['frames_per_run'])
    pre_cue_idx_start_cs_1 = np.zeros((1, runs * frames_per_run))
    cue_idx_start_cs_1 = np.zeros((1, runs * frames_per_run))
    pre_cue_idx_end_cs_1 = np.zeros((1, runs * frames_per_run))
    cue_idx_end_cs_1 = np.zeros((1, runs * frames_per_run))
    pre_cue_idx_start_cs_2 = np.zeros((1, runs * frames_per_run))
    cue_idx_start_cs_2 = np.zeros((1, runs * frames_per_run))
    pre_cue_idx_end_cs_2 = np.zeros((1, runs * frames_per_run))
    cue_idx_end_cs_2 = np.zeros((1, runs * frames_per_run))
    trial_times_cs_1 = behavior['onsets'][behavior['cue_codes'] == behavior['cs_1_code']]
    trial_times_cs_2 = behavior['onsets'][behavior['cue_codes'] == behavior['cs_2_code']]
    trial_times_off_cs_1 = behavior['offsets'][behavior['cue_codes'] == behavior['cs_1_code']]
    trial_times_off_cs_2 = behavior['offsets'][behavior['cue_codes'] == behavior['cs_2_code']]
    for i in range(0, 5):
        cue_onset = int(trial_times_cs_1[i])
        pre_cue_time = int(2 * behavior['framerate']) + 1
        for j in range(1, pre_cue_time):
            idx = cue_onset - j
            pre_cue_idx_start_cs_1[0, idx] = 1
        cue_offset = int(trial_times_off_cs_1[i])
        cue_time = cue_offset - cue_onset + 1
        for j in range(0, cue_time):
            idx = cue_onset + j
            cue_idx_start_cs_1[0, idx] = 1
    for i in range(0, 5):
        cue_onset = int(trial_times_cs_2[i])
        pre_cue_time = int(2 * behavior['framerate']) + 1
        for j in range(1, pre_cue_time):
            idx = cue_onset - j
            pre_cue_idx_start_cs_2[0, idx] = 1
        cue_offset = int(trial_times_off_cs_2[i])
        cue_time = cue_offset - cue_onset + 1
        for j in range(0, cue_time):
            idx = cue_onset + j
            cue_idx_start_cs_2[0, idx] = 1
    for i in range(0, 5):
        cue_onset = int(trial_times_cs_1[len(trial_times_cs_1)-1-i])
        pre_cue_time = int(2 * behavior['framerate']) + 1
        for j in range(1, pre_cue_time):
            idx = cue_onset - j
            pre_cue_idx_end_cs_1[0, idx] = 1
        cue_offset = int(trial_times_off_cs_1[len(trial_times_cs_1)-1-i])
        cue_time = cue_offset - cue_onset + 1
        for j in range(0, cue_time):
            idx = cue_onset + j
            cue_idx_end_cs_1[0, idx] = 1
    for i in range(0, 5):
        cue_onset = int(trial_times_cs_2[len(trial_times_cs_2)-1-i])
        pre_cue_time = int(2 * behavior['framerate']) + 1
        for j in range(1, pre_cue_time):
            idx = cue_onset - j
            pre_cue_idx_end_cs_2[0, idx] = 1
        cue_offset = int(trial_times_off_cs_2[len(trial_times_cs_2)-1-i])
        cue_time = cue_offset - cue_onset + 1
        for j in range(0, cue_time):
            idx = cue_onset + j
            cue_idx_end_cs_2[0, idx] = 1

    activity = norm_deconvolved.reindex(idx_cell['both'].index)
    activity = activity.to_numpy()
    [no_change_cells_cs_1, increase_sig_cells_cs_1, decrease_sig_cells_cs_1] = preprocess_opto.group_neurons(activity,
                                                                                                        behavior,
                                                                                                        'cs_1')

    [no_change_cells_cs_2, increase_sig_cells_cs_2, decrease_sig_cells_cs_2] = preprocess_opto.group_neurons(activity,
                                                                                                        behavior,
                                                                                                          'cs_2')
    start_sel = []
    end_sel = []
    for g in range(0, 3):
        activity = norm_deconvolved.reindex(idx_cell['both'].index)
        activity = activity.to_numpy()
        if g == 0:
            cells_to_use = no_change_cells_cs_1 + no_change_cells_cs_2
            activity = activity[cells_to_use == 2, :]
        if g == 1:
            cells_to_use = increase_sig_cells_cs_1 + increase_sig_cells_cs_2
            activity = activity[cells_to_use > 0, :]
        if g == 2:
            cells_to_use = decrease_sig_cells_cs_1 + decrease_sig_cells_cs_2
            activity = activity[cells_to_use > 0, :]
        activity_cs_1_start = np.mean(activity[:, cue_idx_start_cs_1[0] == 1], axis=1) - np.mean(
            activity[:, pre_cue_idx_start_cs_1[0] == 1], axis=1)
        activity_cs_2_start = np.mean(activity[:, cue_idx_start_cs_2[0] == 1], axis=1) - np.mean(
            activity[:, pre_cue_idx_start_cs_2[0] == 1], axis=1)
        activity_bias_start = (activity_cs_1_start - activity_cs_2_start) / (np.absolute(activity_cs_1_start) + np.absolute(activity_cs_2_start))
        activity_cs_1_end = np.mean(activity[:, cue_idx_end_cs_1[0] == 1], axis=1) - np.mean(
            activity[:, pre_cue_idx_end_cs_1[0] == 1], axis=1)
        activity_cs_2_end = np.mean(activity[:, cue_idx_end_cs_2[0] == 1], axis=1) - np.mean(
            activity[:, pre_cue_idx_end_cs_2[0] == 1], axis=1)
        activity_bias_end = (activity_cs_1_end - activity_cs_2_end) / (np.absolute(activity_cs_1_end) + np.absolute(activity_cs_2_end))
        start_sel.append(np.mean(np.absolute(activity_bias_start)))
        end_sel.append(np.mean(np.absolute(activity_bias_end)))

    days_path = paths['base_path'] + paths['mouse'] + '/data_across_days/'
    if days:
        if path.isfile(days_path + 'overlap_dist_grouped.npy') == 0 or day == 0:
            num_across_days = [list(range(0, days)), list(range(0, days))]
            num_across_days[0][day] = start_sel
            num_across_days[1][day] = end_sel
            np.save(days_path + 'overlap_dist_grouped', num_across_days)
        else:
            num_across_days = np.load(days_path + 'overlap_dist_grouped.npy', allow_pickle=True)
            num_across_days[0][day] = start_sel
            num_across_days[1][day] = end_sel
            np.save(days_path + 'overlap_dist_grouped', num_across_days)


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
            to_delete = preprocess_opto.cells_to_delete(paths, plane, plane_path, 0)
            stat_plane = stat_plane[to_delete == 1]
            stat = np.concatenate((stat, stat_plane))
    stat = stat[idx['both'].index]

    activity = norm_deconvolved.to_numpy()
    activity = activity[idx['both'].index]
    cue_time = preprocess_opto.cue_times(behavior, 0, 0)
    cue_time[0][cue_time[0] > 2] = 0
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
    plt.close()
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
    plt.close()
    # im = retinotopy['POR'] + retinotopy['P'] + retinotopy['LI'] + retinotopy['LM']
    # sns.set(font_scale=1)
    # sns.set_style("whitegrid", {'axes.grid': False})
    # sns.set_style("ticks")
    # plt.figure(figsize=(11.66, 7.5))
    # sns.heatmap(im, cbar=0)
    # plt.xticks([])
    # plt.yticks([])
    # plt.savefig(paths['save_path'] + 'plots/' + paths['mouse'] + '_' + paths['date'] + '_' +
    #             'reactivation_spatial_retinotopy.png', bbox_inches='tight', dpi=500)
    # plt.close()

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


def reactivation_layer_R1(norm_deconvolved, upper_layer_cells, lower_layer_cells, y_pred, behavior, idx, paths, day, days):

    upper_layer_idx = np.intersect1d(np.array(idx['both'].index), upper_layer_cells).astype(int)
    lower_layer_idx = np.intersect1d(np.array(idx['both'].index), lower_layer_cells).astype(int)

    activity = norm_deconvolved.to_numpy()
    activity_upper = activity[upper_layer_idx]
    activity_lower = activity[lower_layer_idx]
    cue_time = preprocess_opto.cue_times(behavior, 0, 0)
    cue_time[0][cue_time[0] > 2] = 0
    activity_cue_upper = np.mean(activity_upper[:, cue_time[0] > 0], axis=1) * behavior['framerate'][0][0]
    activity_cue_lower = np.mean(activity_lower[:, cue_time[0] > 0], axis=1) * behavior['framerate'][0][0]

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
    activity_reactivation_upper = np.mean(activity_upper[:, reactivation_frames > 0], axis=1) * behavior['framerate'][0][0]
    activity_reactivation_lower = np.mean(activity_lower[:, reactivation_frames > 0], axis=1) * behavior['framerate'][0][0]

    days_path = paths['base_path'] + paths['mouse'] + '/data_across_days/'
    if days:
        if path.isfile(days_path + 'reactivation_layer.npy') == 0 or day == 0:
            reactivation_layer_days = [list(range(0, days)), list(range(0, days)), list(range(0, days)),
                                       list(range(0, days))]
            reactivation_layer_days[0][day] = np.mean(activity_cue_upper)
            reactivation_layer_days[1][day] = np.mean(activity_cue_lower)
            reactivation_layer_days[2][day] = np.mean(activity_reactivation_upper)
            reactivation_layer_days[3][day] = np.mean(activity_reactivation_lower)
            np.save(days_path + 'reactivation_layer', reactivation_layer_days)
        else:
            reactivation_layer_days = np.load(days_path + 'reactivation_layer.npy', allow_pickle=True)
            reactivation_layer_days[0][day] = np.mean(activity_cue_upper)
            reactivation_layer_days[1][day] = np.mean(activity_cue_lower)
            reactivation_layer_days[2][day] = np.mean(activity_reactivation_upper)
            reactivation_layer_days[3][day] = np.mean(activity_reactivation_lower)
            np.save(days_path + 'reactivation_layer', reactivation_layer_days)


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

    cue_idx = preprocess_opto.cue_times(behavior, 0, 0)
    cs_1_times = cue_idx[0] == 1
    cs_2_times = cue_idx[0] == 2

    baseline = preprocess_opto.get_times_considered(y_pred, behavior)
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


def top_activity_stable_R3(norm_deconvolved, idx, behavior, paths, day, days):

    # get sig cells
    [cs_1_poscells, cs_1_negcells] = preprocess_opto.sig_test(norm_deconvolved, behavior, 'cs_1')
    [cs_2_poscells, cs_2_negcells] = preprocess_opto.sig_test(norm_deconvolved, behavior, 'cs_2')
    [both_poscells, both_sigcells] = preprocess_opto.combine_sig(cs_1_poscells, cs_1_negcells, cs_2_poscells, cs_2_negcells)

    # make trial averaged traces and baseline subtract
    mean_cs_1_responses_df_s = preprocess_opto.normalized_trial_averaged_R3(norm_deconvolved, behavior, 'cs_1', 'start')
    mean_cs_2_responses_df_s = preprocess_opto.normalized_trial_averaged_R3(norm_deconvolved, behavior, 'cs_2', 'start')
    mean_cs_1_responses_df_e = preprocess_opto.normalized_trial_averaged_R3(norm_deconvolved, behavior, 'cs_1', 'end')
    mean_cs_2_responses_df_e = preprocess_opto.normalized_trial_averaged_R3(norm_deconvolved, behavior, 'cs_2', 'end')

    # get idx of top cell differences
    idx_early = preprocess_opto.get_index(behavior, mean_cs_1_responses_df_s, mean_cs_2_responses_df_s, cs_1_poscells, cs_2_poscells,
                                     both_poscells, both_sigcells, paths, 1)
    idx_late = preprocess_opto.get_index(behavior, mean_cs_1_responses_df_e, mean_cs_2_responses_df_e, cs_1_poscells,
                                     cs_2_poscells, both_poscells, both_sigcells, paths, 1)

    num_top_cs_1 = int(len(idx['cs_1']) / 20)
    num_top_cs_2 = int(len(idx['cs_2']) / 20)

    idx_cs_1_overlap = len(np.intersect1d(idx_early['cs_1'].index[0:num_top_cs_1], idx_late['cs_1'].index[0:num_top_cs_1])) / num_top_cs_1
    idx_cs_2_overlap = len(np.intersect1d(idx_early['cs_2'].index[0:num_top_cs_2], idx_late['cs_2'].index[0:num_top_cs_2])) / num_top_cs_2

    days_path = paths['base_path'] + paths['mouse'] + '/data_across_days/'
    if days:
        if path.isfile(days_path + 'top_activity_stable.npy') == 0 or day == 0:
            reactivation_top_bottom = [list(range(0, days)), list(range(0, days))]
            reactivation_top_bottom[0][day] = idx_cs_1_overlap
            reactivation_top_bottom[1][day] = idx_cs_2_overlap
            np.save(days_path + 'top_activity_stable', reactivation_top_bottom)
        else:
            reactivation_top_bottom = np.load(days_path + 'top_activity_stable.npy', allow_pickle=True)
            reactivation_top_bottom[0][day] = idx_cs_1_overlap
            reactivation_top_bottom[1][day] = idx_cs_2_overlap
            np.save(days_path + 'top_activity_stable', reactivation_top_bottom)


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
    times_considered = preprocess_opto.get_times_considered(y_pred, behavior)

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

    cue_start = behavior['onsets']
    cue_codes = behavior['cue_codes']
    end_trials = behavior['end_trials']

    if num_prev == 1:
        for i in range(1, len(cue_codes)):
            if i not in end_trials and cue_codes[i] != behavior['cs_1_opto_code'] and cue_codes[i] != behavior['cs_2_opto_code']:
                prev_cue = cue_codes[i-1]
                idx = int(cue_start[i])

                if prev_cue[0:4] == cue_codes[i][0:4]:
                    reactivation_same.append(fr * np.sum(y_pred_all[idx:idx + duration]) /
                                             np.sum(times_considered[idx:idx + duration]))
                    pupil_same.append((np.mean(behavior['pupil'][idx:idx + int(fr * 5)]) -
                                       np.mean(behavior['pupil'][idx - before:idx])) /
                                      np.mean(behavior['pupil'][idx - before:idx]))
                    cue_evoked_same[i, :] = sorted_deconvolved[idx - before:idx + duration]
                    if cue_codes[i] == behavior['cs_1_code']:
                        bias_same.append(
                            (np.sum(y_pred[idx:idx + duration, 0]) - np.sum(y_pred[idx:idx + duration, 1]))
                            / (np.sum(y_pred[idx:idx + duration, 0]) + np.sum(y_pred[idx:idx + duration, 1])))
                    if cue_codes[i] == behavior['cs_2_code']:
                        bias_same.append(
                            (np.sum(y_pred[idx:idx + duration, 1]) - np.sum(y_pred[idx:idx + duration, 0]))
                            / (np.sum(y_pred[idx:idx + duration, 0]) + np.sum(y_pred[idx:idx + duration, 1])))

                if prev_cue[0:4] != cue_codes[i][0:4]:
                    reactivation_diff.append(fr * np.sum(y_pred_all[idx:idx + duration]) /
                                             np.sum(times_considered[idx:idx + duration]))
                    pupil_diff.append((np.mean(behavior['pupil'][idx:idx + int(fr * 5)]) -
                                       np.mean(behavior['pupil'][idx - before:idx])) /
                                      np.mean(behavior['pupil'][idx - before:idx]))
                    cue_evoked_different[i, :] = sorted_deconvolved[idx - before:idx + duration]
                    if cue_codes[i] == behavior['cs_1_code']:
                        bias_diff.append((np.sum(y_pred[idx:idx + duration, 0]) - np.sum(y_pred[idx:idx + duration, 1]))
                                         / (np.sum(y_pred[idx:idx + duration, 0]) + np.sum(
                            y_pred[idx:idx + duration, 1])))
                    if cue_codes[i] == behavior['cs_2_code']:
                        bias_diff.append(
                            (np.sum(y_pred[idx:idx + duration, 1]) - np.sum(y_pred[idx:idx + duration, 0]))
                            / (np.sum(y_pred[idx:idx + duration, 0]) + np.sum(y_pred[idx:idx + duration, 1])))

    days_path = paths['base_path'] + paths['mouse'] + '/data_across_days/'
    if days:
        if path.isfile(days_path + 'trial_history_' + str(num_prev) + '.npy') == 0 or day == 0:
            trial_history_across_days = [list(range(0, days)), list(range(0, days)), list(range(0, days)),
                                         list(range(0, days)), list(range(0, days)), list(range(0, days)),
                                         list(range(0, days)), list(range(0, days))]
            trial_history_across_days[0][day] = np.nanmean(reactivation_same)
            trial_history_across_days[1][day] = np.nanmean(reactivation_diff)
            trial_history_across_days[2][day] = np.nanmean(pupil_same)
            trial_history_across_days[3][day] = np.nanmean(pupil_diff)
            trial_history_across_days[4][day] = np.nanmean(cue_evoked_same, axis=0)
            trial_history_across_days[5][day] = np.nanmean(cue_evoked_different, axis=0)
            trial_history_across_days[6][day] = np.nanmean(bias_same)
            trial_history_across_days[7][day] = np.nanmean(bias_diff)
            np.save(days_path + 'trial_history_' + str(num_prev), trial_history_across_days)
        else:
            trial_history_across_days = np.load(days_path + 'trial_history_' + str(num_prev) + '.npy',
                                                allow_pickle=True)
            trial_history_across_days[0][day] = np.nanmean(reactivation_same)
            trial_history_across_days[1][day] = np.nanmean(reactivation_diff)
            trial_history_across_days[2][day] = np.nanmean(pupil_same)
            trial_history_across_days[3][day] = np.nanmean(pupil_diff)
            trial_history_across_days[4][day] = np.nanmean(cue_evoked_same, axis=0)
            trial_history_across_days[5][day] = np.nanmean(cue_evoked_different, axis=0)
            trial_history_across_days[6][day] = np.nanmean(bias_same)
            trial_history_across_days[7][day] = np.nanmean(bias_diff)
            np.save(days_path + 'trial_history_' + str(num_prev), trial_history_across_days)


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
    times_considered = preprocess_opto.get_times_considered(y_pred, behavior)
    frames_per_run = int(behavior['frames_per_run'])
    activity[times_considered == 0] = np.nan
    fr = behavior['framerate']
    for i in range(0, len(behavior['onsets'])):
        if behavior['cue_codes'][i] == behavior['cs_1_opto_code'] or behavior['cue_codes'][i] == behavior['cs_2_opto_code']:
            activity[int(behavior['onsets'][i]):int(behavior['onsets'][i])+int(fr * (behavior['iti'] + 5)) + 6] = np.nan
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


def iti_activity_within_trial(norm_deconvolved, idx, behavior, paths, day, days):
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
    cue_start = behavior['onsets']
    fr = behavior['framerate']
    iti_start = int(behavior['framerate'] * 5)
    iti_end = int(fr * (behavior['iti'] + 5)) + 1
    iti_activity = np.empty((len(cue_start), iti_end + iti_start)) * np.nan
    end_trials = behavior['end_trials']
    for i in range(0, len(cue_start)):
        if i not in end_trials and (
                behavior['cue_codes'][i] == behavior['cs_1_code'] or behavior['cue_codes'][i] == behavior['cs_2_code']):
            temp_iti_activity = np.mean(activity[:, int(cue_start[i]) - iti_start:int(cue_start[i]) + iti_end], axis=0)
            iti_activity[i, :] = temp_iti_activity
    iti_activity = np.nanmean(iti_activity, axis=0)

    days_path = paths['base_path'] + paths['mouse'] + '/data_across_days/'
    if days:
        if path.isfile(days_path + 'activity_within_trial.npy') == 0 or day == 0:
            activity_within_trial_across_days = [list(range(0, days))]
            activity_within_trial_across_days[0][day] = iti_activity
            np.save(days_path + 'activity_within_trial', activity_within_trial_across_days)
        else:
            activity_within_trial_across_days = np.load(days_path + 'activity_within_trial.npy', allow_pickle=True)
            activity_within_trial_across_days[0][day] = iti_activity
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
    times_considered = preprocess_opto.get_times_considered(y_pred, behavior)
    frames_per_run = int(behavior['frames_per_run'])
    pupil = behavior['pupil'].copy()
    pupil[times_considered == 0] = np.nan
    fr = behavior['framerate']
    for i in range(0, len(behavior['onsets'])):
        if behavior['cue_codes'][i] == behavior['cs_1_opto_code'] or behavior['cue_codes'][i] == behavior['cs_2_opto_code']:
            pupil[int(behavior['onsets'][i]):int(behavior['onsets'][i])+int(fr * (behavior['iti'] + 5)) + 6] = np.nan
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


def pupil_within_trial(behavior, paths, day, days):
    """
    gets pupil across session
    :param y_pred: pupil
    :param behavior: behavior
    :param paths: path
    :param day: day
    :return: pupil binned
    """
    pupil = behavior['pupil'] / behavior['pupil_max']
    cue_start = behavior['onsets']
    fr = behavior['framerate']
    iti_start = int(behavior['framerate'] * 5)
    iti_end = int(fr * (behavior['iti'] + 5)) + 1
    iti_pupil = np.empty((len(cue_start), iti_end + iti_start)) * np.nan
    end_trials = behavior['end_trials']
    for i in range(0, len(cue_start)):
        if i not in end_trials and (
                behavior['cue_codes'][i] == behavior['cs_1_code'] or behavior['cue_codes'][i] == behavior['cs_2_code']):
            temp_iti_pupil = pupil[int(cue_start[i]) - iti_start:int(cue_start[i]) + iti_end]
            iti_pupil[i, :] = temp_iti_pupil
    iti_pupil = np.nanmean(iti_pupil, axis=0)

    days_path = paths['base_path'] + paths['mouse'] + '/data_across_days/'
    if days:
        if path.isfile(days_path + 'pupil_within_trial.npy') == 0 or day == 0:
            pupil_within_trial_across_days = [list(range(0, days))]
            pupil_within_trial_across_days[0][day] = iti_pupil
            np.save(days_path + 'pupil_within_trial', pupil_within_trial_across_days)
        else:
            pupil_within_trial_across_days = np.load(days_path + 'pupil_within_trial.npy', allow_pickle=True)
            pupil_within_trial_across_days[0][day] = iti_pupil
            np.save(days_path + 'pupil_within_trial', pupil_within_trial_across_days)


def behavior_across_trials_R2(behavior, paths, day, days):

    normal_trials_idx = np.where(np.isin(behavior['cue_codes'], [behavior['cs_1_code'], behavior['cs_2_code']]))[0]
    cue_start = behavior['onsets'][normal_trials_idx]
    cue_end = behavior['offsets'][normal_trials_idx]
    cue_codes = behavior['cue_codes'][normal_trials_idx]
    end_trials = behavior['end_trials']
    norm_pupil = behavior['pupil']/behavior['pupil_max']
    pupil = []
    brain_motion = []
    start = max(list(cue_codes).index(behavior['cs_1_code']), list(cue_codes).index(behavior['cs_2_code']))
    for i in range(start+1, len(cue_start)):
        temp_pupil = np.mean(norm_pupil[int(cue_start[i]):int(cue_end[i]) + 1])
        temp_brain_motion = np.mean(behavior['brain_motion'][int(cue_start[i]):int(cue_end[i]) + 1])
        if normal_trials_idx[i] not in end_trials:
            pupil.append(temp_pupil)
            brain_motion.append(temp_brain_motion)

    days_path = paths['base_path'] + paths['mouse'] + '/data_across_days/'
    if days:
        if path.isfile(days_path + 'behavior_across_days.npy') == 0 or day == 0:
            across_days = [list(range(0, days)), list(range(0, days))]
            across_days[0][day] = brain_motion
            across_days[1][day] = pupil
            np.save(days_path + 'behavior_across_days', across_days)
        else:
            across_days = np.load(days_path + 'behavior_across_days.npy', allow_pickle=True)
            across_days[0][day] = brain_motion
            across_days[1][day] = pupil
            np.save(days_path + 'behavior_across_days', across_days)


def rate_within_trial_controlopto_R1(y_pred, behavior, paths, day, days):
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
    times_considered = preprocess_opto.get_times_considered(y_pred, behavior)
    fr = behavior['framerate']
    duration = (int(fr * (behavior['iti'] + 5)) + 1) * 2
    end_trials = behavior['end_trials']
    rate_norm = np.empty((len(behavior['cue_codes']), duration)) * np.nan
    rate_opto = np.empty((len(behavior['cue_codes']), duration)) * np.nan
    times_considered_norm = np.empty((len(behavior['cue_codes']), duration)) * np.nan
    times_considered_opto = np.empty((len(behavior['cue_codes']), duration)) * np.nan

    for i in range(0, len(behavior['cue_codes'])-1):
        if i not in end_trials:
            if (behavior['cue_codes'][i] == behavior['cs_1_code'] or behavior['cue_codes'][i] == behavior['cs_2_code']):
                if (behavior['cue_codes'][i+1] == behavior['cs_1_opto_code'] or behavior['cue_codes'][i+1] == behavior[
                    'cs_2_opto_code']):
                    idx_curr = int(behavior['onsets'][i])
                    rate_norm[i, :] = y_pred[:, 0][idx_curr:idx_curr+duration] + y_pred[:, 1][idx_curr:idx_curr+duration]
                    times_considered_norm[i, :] = times_considered[idx_curr:idx_curr+duration]
            if (behavior['cue_codes'][i] == behavior['cs_1_opto_code'] or behavior['cue_codes'][i] == behavior['cs_2_opto_code']):
                idx_curr = int(behavior['onsets'][i])
                if (behavior['cue_codes'][i + 1] == behavior['cs_1_opto_code'] or behavior['cue_codes'][i + 1] ==
                        behavior[
                            'cs_2_opto_code']):
                    rate_opto[i, :] = y_pred[:, 0][idx_curr:idx_curr+duration] + y_pred[:, 1][idx_curr:idx_curr+duration]
                    times_considered_opto[i, :] = times_considered[idx_curr:idx_curr+duration]
    rate_norm_binned = []
    rate_opto_binned = []
    x_binned = []
    factor = 5
    duration = int(fr * 8)
    step = int((((int(fr * (behavior['iti'] + 5)) + 1)) - duration) / factor)
    for i in range(0, factor):
        rate_norm_binned.append(
            np.nansum(rate_norm[:, duration + (i * step):duration + (i + 1) * step]) *
            int(fr) / np.nansum(times_considered_norm[:, duration + (i * step):duration + (i + 1) * step]))
        rate_opto_binned.append(
            np.nansum(rate_opto[:, duration + (i * step):duration + (i + 1) * step]) *
            int(fr) / np.nansum(times_considered_opto[:, duration + (i * step):duration + (i + 1) * step]))
        x_binned.append(duration+(i + .5) * step)

    duration = int(fr * 8) + (int(fr * (behavior['iti'] + 5)) + 1)
    step = int(((len(rate_norm[0])) - duration) / factor)
    for i in range(0, factor):
        rate_norm_binned.append(
            np.nansum(rate_norm[:, duration + (i * step):duration + (i + 1) * step]) *
            int(fr) / np.nansum(times_considered_norm[:, duration + (i * step):duration + (i + 1) * step]))
        rate_opto_binned.append(
            np.nansum(rate_opto[:, duration + (i * step):duration + (i + 1) * step]) *
            int(fr) / np.nansum(times_considered_opto[:, duration + (i * step):duration + (i + 1) * step]))
        x_binned.append(duration + (i + .5) * step)

    days_path = paths['base_path'] + paths['mouse'] + '/data_across_days/'
    if days:
        if path.isfile(days_path + 'rate_within_trial_long.npy') == 0 or day == 0:
            rate_within_trial_across_days = [list(range(0, days)), list(range(0, days)), list(range(0, days))]
            rate_within_trial_across_days[0][day] = rate_norm_binned
            rate_within_trial_across_days[1][day] = rate_opto_binned
            rate_within_trial_across_days[2][day] = x_binned
            np.save(days_path + 'rate_within_trial_long', rate_within_trial_across_days)
        else:
            rate_within_trial_across_days = np.load(days_path + 'rate_within_trial_long.npy', allow_pickle=True)
            rate_within_trial_across_days[0][day] = rate_norm_binned
            rate_within_trial_across_days[1][day] = rate_opto_binned
            rate_within_trial_across_days[2][day] = x_binned
            np.save(days_path + 'rate_within_trial_long', rate_within_trial_across_days)


def bias_within_trial_controlopto_R1(y_pred, behavior, paths, day, days):
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
    duration = (int(fr * (behavior['iti'] + 5)) + 1) *2
    end_trials = behavior['end_trials']
    bias_norm = np.empty((len(behavior['cue_codes']), duration)) * np.nan
    bias_opto = np.empty((len(behavior['cue_codes']), duration)) * np.nan
    total_norm = np.empty((len(behavior['cue_codes']), duration)) * np.nan
    total_opto = np.empty((len(behavior['cue_codes']), duration)) * np.nan

    for i in range(1, len(behavior['cue_codes'])-1):
        if i not in end_trials:
            idx_curr = int(behavior['onsets'][i])
            if behavior['cue_codes'][i] == behavior['cs_1_code']:
                if (behavior['cue_codes'][i + 1] == behavior['cs_1_opto_code'] or behavior['cue_codes'][i + 1] ==
                        behavior[
                            'cs_2_opto_code']):
                    bias_norm[i, :] = y_pred[:, 0][idx_curr:idx_curr+duration] - y_pred[:, 1][idx_curr:idx_curr+duration]
                    total_norm[i, :] = y_pred[:, 0][idx_curr:idx_curr+duration] + y_pred[:, 1][idx_curr:idx_curr+duration]
            if behavior['cue_codes'][i] == behavior['cs_2_code']:
                if (behavior['cue_codes'][i + 1] == behavior['cs_1_opto_code'] or behavior['cue_codes'][i + 1] ==
                        behavior[
                            'cs_2_opto_code']):
                    bias_norm[i, :] = y_pred[:, 1][idx_curr:idx_curr+duration] - y_pred[:, 0][idx_curr:idx_curr+duration]
                    total_norm[i, :] = y_pred[:, 0][idx_curr:idx_curr+duration] + y_pred[:, 1][idx_curr:idx_curr+duration]
            if behavior['cue_codes'][i] == behavior['cs_1_opto_code']:
                if (behavior['cue_codes'][i + 1] == behavior['cs_1_opto_code'] or behavior['cue_codes'][i + 1] ==
                        behavior[
                            'cs_2_opto_code']):
                    bias_opto[i, :] = y_pred[:, 0][idx_curr:idx_curr+duration] - y_pred[:, 1][idx_curr:idx_curr+duration]
                    total_opto[i, :] = y_pred[:, 0][idx_curr:idx_curr+duration] + y_pred[:, 1][idx_curr:idx_curr+duration]
            if behavior['cue_codes'][i] == behavior['cs_2_opto_code']:
                if (behavior['cue_codes'][i + 1] == behavior['cs_1_opto_code'] or behavior['cue_codes'][i + 1] ==
                        behavior[
                            'cs_2_opto_code']):
                    bias_opto[i, :] = y_pred[:, 1][idx_curr:idx_curr+duration] - y_pred[:, 0][idx_curr:idx_curr+duration]
                    total_opto[i, :] = y_pred[:, 0][idx_curr:idx_curr+duration] + y_pred[:, 1][idx_curr:idx_curr+duration]

    bias_norm_binned = []
    bias_opto_binned = []
    x_binned = []
    factor = 5
    duration = int(fr * 8)
    step = int((((int(fr * (behavior['iti'] + 5)) + 1)) - duration) / factor)
    for i in range(0, factor):
        bias_norm_binned.append(
            np.nansum(bias_norm[:, duration + (i * step):duration + (i + 1) * step]) /
            np.nansum(total_norm[:, duration + (i * step):duration + (i + 1) * step]))
        bias_opto_binned.append(
            np.nansum(bias_opto[:, duration + (i * step):duration + (i + 1) * step]) /
            np.nansum(total_opto[:, duration + (i * step):duration + (i + 1) * step]))
        x_binned.append(duration+(i + .5) * step)

    duration = int(fr * 8) + (int(fr * (behavior['iti'] + 5)) + 1)
    step = int(((len(bias_norm[0])) - duration) / factor)
    for i in range(0, factor):
        bias_norm_binned.append(
            np.nansum(bias_norm[:, duration + (i * step):duration + (i + 1) * step]) /
            np.nansum(total_norm[:, duration + (i * step):duration + (i + 1) * step]))
        bias_opto_binned.append(
            np.nansum(bias_opto[:, duration + (i * step):duration + (i + 1) * step]) /
            np.nansum(total_opto[:, duration + (i * step):duration + (i + 1) * step]))
        x_binned.append(duration + (i + .5) * step)

    days_path = paths['base_path'] + paths['mouse'] + '/data_across_days/'
    if days:
        if path.isfile(days_path + 'bias_within_trial_long.npy') == 0 or day == 0:
            bias_within_trial_across_days = [list(range(0, days)), list(range(0, days)), list(range(0, days))]
            bias_within_trial_across_days[0][day] = bias_norm_binned
            bias_within_trial_across_days[1][day] = bias_opto_binned
            bias_within_trial_across_days[2][day] = x_binned
            np.save(days_path + 'bias_within_trial_long', bias_within_trial_across_days)
        else:
            bias_within_trial_across_days = np.load(days_path + 'bias_within_trial_long.npy', allow_pickle=True)
            bias_within_trial_across_days[0][day] = bias_norm_binned
            bias_within_trial_across_days[1][day] = bias_opto_binned
            bias_within_trial_across_days[2][day] = x_binned
            np.save(days_path + 'bias_within_trial_long', bias_within_trial_across_days)


def reactivation_rate_pupil_control_baseline_R1(y_pred, behavior, paths, day, days):
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
    y_pred_cs_2 = y_pred[:, 1]
    y_pred_cs_1 = y_pred[:, 0]
    y_pred_all_opto = y_pred_cs_1 + y_pred_cs_2
    times_considered_opto = preprocess_opto.get_times_considered(y_pred, behavior)

    for i in range(0, len(behavior['onsets'])-1):
        cue_onset = int(behavior['onsets'][i])
        next_cue_onset = int(behavior['onsets'][i+1])
        if behavior['cue_codes'][i] == behavior['cs_1_code']:
            y_pred_all_opto[cue_onset:next_cue_onset] = 0
            times_considered_opto[cue_onset:next_cue_onset] = 0
        if behavior['cue_codes'][i] == behavior['cs_2_code']:
            y_pred_all_opto[cue_onset:next_cue_onset] = 0
            times_considered_opto[cue_onset:next_cue_onset] = 0

    pupil = behavior['pupil']
    pupil_task_opto = pupil.copy()
    pupil_task_opto[0:frames_per_run] = np.nan
    pupil_task_opto[times_considered_opto == 0] = np.nan
    pupil_task_opto = np.nanmean(pupil_task_opto)

    pupil_baseline = pupil[0:frames_per_run]
    sorted_index = np.argsort(pupil_baseline)
    mean_pupil_baseline = []
    mean_pupil_baseline_idx = []
    for i in sorted_index:
        mean_pupil_baseline.append(pupil_baseline[i])
        mean_pupil_baseline_idx.append(i)
        if np.mean(mean_pupil_baseline) > pupil_task_opto:
            break

    y_pred_binned_opto_baseline = np.sum(y_pred_all_opto[mean_pupil_baseline_idx]) / (np.sum(times_considered_opto[mean_pupil_baseline_idx]) / int(behavior['framerate']))
    y_pred_binned_opto_task = np.sum(y_pred_all_opto[frames_per_run:len(y_pred_all_opto)]) / (np.sum(times_considered_opto[frames_per_run:len(y_pred_all_opto)]) / int(behavior['framerate']))


    days_path = paths['base_path'] + paths['mouse'] + '/data_across_days/'
    if days:
        if path.isfile(days_path + 'y_pred_binned_pupil_baseline.npy') == 0 or day == 0:
            y_pred_binned_across_days = [list(range(0, days)), list(range(0, days))]
            y_pred_binned_across_days[0][day] = y_pred_binned_opto_baseline
            y_pred_binned_across_days[1][day] = y_pred_binned_opto_task
            np.save(days_path + 'y_pred_binned_pupil_baseline', y_pred_binned_across_days)
        else:
            y_pred_binned_across_days = np.load(days_path + 'y_pred_binned_pupil_baseline.npy', allow_pickle=True)
            y_pred_binned_across_days[0][day] = y_pred_binned_opto_baseline
            y_pred_binned_across_days[1][day] = y_pred_binned_opto_task
            np.save(days_path + 'y_pred_binned_pupil_baseline', y_pred_binned_across_days)


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

    # sig_cells = preprocess_opto.sig_reactivated_cells([], [], [], [], [], paths, 0)
    # activity = activity[sig_cells > 0, :]

    normal_trials_idx = np.where(np.isin(behavior['cue_codes'], [behavior['cs_1_code'], behavior['cs_2_code']]))[0]

    times_considered = preprocess_opto.get_times_considered(y_pred, behavior)
    y_pred = true_reactivations(y_pred)
    y_pred_cs_1 = y_pred[:, 0]
    y_pred_cs_2 = y_pred[:, 1]
    all_y_pred = y_pred_cs_1 + y_pred_cs_2
    cue_start = behavior['onsets'][normal_trials_idx]
    cue_end = behavior['offsets'][normal_trials_idx]
    cue_codes = behavior['cue_codes'][normal_trials_idx]
    end_trials = behavior['end_trials']
    reactivation_prob = []
    correlation = []
    cue_activity = []

    start = max(list(cue_codes).index(behavior['cs_1_code']), list(cue_codes).index(behavior['cs_2_code']))
    past_cs_1_type = cue_codes[start - 1]
    past_cs_2_type = cue_codes[start]
    past_cs_1_mean = np.mean(activity[:, int(cue_start[start - 1]):int(cue_end[start - 1]) + 1], axis=1)
    past_cs_2_mean = np.mean(activity[:, int(cue_start[start]):int(cue_end[start]) + 1], axis=1)
    for i in range(start+1, len(cue_start)):
        current_cs_type = cue_codes[i]
        if current_cs_type == past_cs_1_type:
            current_temp_mean = np.mean(activity[:, int(cue_start[i]):int(cue_end[i]) + 1], axis=1)
            if normal_trials_idx[i] not in end_trials:
                correlation.append(np.corrcoef(current_temp_mean, past_cs_2_mean)[0][1])
            past_cs_1_mean = current_temp_mean
        if current_cs_type == past_cs_2_type:
            current_temp_mean = np.mean(activity[:, int(cue_start[i]):int(cue_end[i]) + 1], axis=1)
            if normal_trials_idx[i] not in end_trials:
                correlation.append(np.corrcoef(current_temp_mean, past_cs_1_mean)[0][1])
            past_cs_2_mean = current_temp_mean
        temp_sum_reactivation = np.sum(all_y_pred[int(cue_start[i]):int(cue_start[i]) + int((behavior['framerate'] *
                                                                                             (behavior['iti'] + 6)))]) / np.sum(times_considered[int(cue_start[i]):int(cue_start[i]) + int((behavior['framerate'] *
                                                                                             (behavior['iti'] + 6)))]) * int(behavior['framerate'])
        temp_cue_activity = np.mean(activity[:, int(cue_start[i]):int(cue_end[i]) + 1])
        if normal_trials_idx[i] not in end_trials:
            reactivation_prob.append(temp_sum_reactivation)
            cue_activity.append(temp_cue_activity)

    days_path = paths['base_path'] + paths['mouse'] + '/data_across_days/'
    if days:
        if path.isfile(days_path + 'activity.npy') == 0 or day == 0:
            correlation_across_days = [list(range(0, days)), list(range(0, days)), list(range(0, days))]
            correlation_across_days[0][day] = correlation
            correlation_across_days[1][day] = reactivation_prob
            correlation_across_days[2][day] = cue_activity
            np.save(days_path + 'activity', correlation_across_days)
        else:
            correlation_across_days = np.load(days_path + 'activity.npy', allow_pickle=True)
            correlation_across_days[0][day] = correlation
            correlation_across_days[1][day] = reactivation_prob
            correlation_across_days[2][day] = cue_activity
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
    # sig_cells = preprocess_opto.sig_reactivated_cells([], [], [], [], [], paths, 0)
    # activity = activity[sig_cells > 0, :]
    [no_change_cells_cs_1, increase_sig_cells_cs_1, decrease_sig_cells_cs_1] = preprocess_opto.group_neurons(activity, behavior, 'cs_1')

    [no_change_cells_cs_2, increase_sig_cells_cs_2, decrease_sig_cells_cs_2] = preprocess_opto.group_neurons(activity, behavior, 'cs_2')

    all_correlation = []
    all_cue_activity = []
    for g in range(0, 3):
        activity = norm_deconvolved.to_numpy()
        activity = activity[idx['both'].index]
        # sig_cells = preprocess_opto.sig_reactivated_cells([], [], [], [], [], paths, 0)
        # activity = activity[sig_cells > 0, :]
        if g == 0:
            cells_to_use = no_change_cells_cs_1 + no_change_cells_cs_2
            activity = activity[cells_to_use == 2, :]
        if g == 1:
            cells_to_use = increase_sig_cells_cs_1 + increase_sig_cells_cs_2
            activity = activity[cells_to_use > 0, :]
        if g == 2:
            cells_to_use = decrease_sig_cells_cs_1 + decrease_sig_cells_cs_2
            activity = activity[cells_to_use > 0, :]

        normal_trials_idx = np.where(np.isin(behavior['cue_codes'], [behavior['cs_1_code'], behavior['cs_2_code']]))[0]

        cue_start = behavior['onsets'][normal_trials_idx]
        cue_end = behavior['offsets'][normal_trials_idx]
        cue_codes = behavior['cue_codes'][normal_trials_idx]
        end_trials = behavior['end_trials']
        correlation = []
        cue_activity = []

        start = max(list(cue_codes).index(behavior['cs_1_code']), list(cue_codes).
                    index(behavior['cs_2_code']))
        past_cs_1_type = cue_codes[start - 1]
        past_cs_2_type = cue_codes[start]
        past_cs_1_mean = np.mean(activity[:, int(cue_start[start - 1]):int(cue_end[start - 1]) + 1], axis=1)
        past_cs_2_mean = np.mean(activity[:, int(cue_start[start]):int(cue_end[start]) + 1], axis=1)
        for i in range(start+1, len(cue_start)):
            current_cs_type = cue_codes[i]
            if current_cs_type == past_cs_1_type:
                current_temp_mean = np.mean(activity[:, int(cue_start[i]):int(cue_end[i]) + 1], axis=1)
                if normal_trials_idx[i] not in end_trials:
                    correlation.append(np.corrcoef(current_temp_mean, past_cs_2_mean)[0][1])
                past_cs_1_mean = current_temp_mean
            if current_cs_type == past_cs_2_type:
                current_temp_mean = np.mean(activity[:, int(cue_start[i]):int(cue_end[i]) + 1], axis=1)
                if normal_trials_idx[i] not in end_trials:
                    correlation.append(np.corrcoef(current_temp_mean, past_cs_1_mean)[0][1])
                past_cs_2_mean = current_temp_mean
            temp_cue_activity = np.mean(activity[:, int(cue_start[i]):int(cue_end[i]) + 1])
            if normal_trials_idx[i] not in end_trials:
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


def activity_across_trials_grouped_omit(norm_deconvolved, behavior, idx, paths, day, days):
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
    # sig_cells = preprocess_opto.sig_reactivated_cells([], [], [], [], [], paths, 0)
    # activity = activity[sig_cells > 0, :]
    [no_change_cells_cs_1, increase_sig_cells_cs_1, decrease_sig_cells_cs_1] = preprocess_opto.group_neurons(activity, behavior, 'cs_1')

    [no_change_cells_cs_2, increase_sig_cells_cs_2, decrease_sig_cells_cs_2] = preprocess_opto.group_neurons(activity, behavior, 'cs_2')

    all_correlation = []
    all_cue_activity = []
    for g in range(0, 3):
        activity = norm_deconvolved.to_numpy()
        activity = activity[idx['both'].index]
        # sig_cells = preprocess_opto.sig_reactivated_cells([], [], [], [], [], paths, 0)
        # activity = activity[sig_cells > 0, :]
        if g == 0:
            cells_to_use = decrease_sig_cells_cs_1 + decrease_sig_cells_cs_2 + increase_sig_cells_cs_1 + increase_sig_cells_cs_2
            activity = activity[cells_to_use > 0, :]
        if g == 1:
            cells_to_use_1 = decrease_sig_cells_cs_1 + decrease_sig_cells_cs_2
            cells_to_use_1[cells_to_use_1 > 0] = 1
            cells_to_use_2 = no_change_cells_cs_1 + no_change_cells_cs_2
            cells_to_use_2[cells_to_use_2 != 2] = 0
            cells_to_use_2[cells_to_use_2 == 2] = 1
            cells_to_use = cells_to_use_1 + cells_to_use_2
            activity = activity[cells_to_use > 0, :]
        if g == 2:
            cells_to_use_1 = increase_sig_cells_cs_1 + increase_sig_cells_cs_2
            cells_to_use_1[cells_to_use_1 > 0] = 1
            cells_to_use_2 = no_change_cells_cs_1 + no_change_cells_cs_2
            cells_to_use_2[cells_to_use_2 != 2] = 0
            cells_to_use_2[cells_to_use_2 == 2] = 1
            cells_to_use = cells_to_use_1 + cells_to_use_2
            activity = activity[cells_to_use > 0, :]

        normal_trials_idx = np.where(np.isin(behavior['cue_codes'], [behavior['cs_1_code'], behavior['cs_2_code']]))[0]

        cue_start = behavior['onsets'][normal_trials_idx]
        cue_end = behavior['offsets'][normal_trials_idx]
        cue_codes = behavior['cue_codes'][normal_trials_idx]
        end_trials = behavior['end_trials']
        correlation = []
        cue_activity = []

        start = max(list(cue_codes).index(behavior['cs_1_code']), list(cue_codes).
                    index(behavior['cs_2_code']))
        past_cs_1_type = cue_codes[start - 1]
        past_cs_2_type = cue_codes[start]
        past_cs_1_mean = np.mean(activity[:, int(cue_start[start - 1]):int(cue_end[start - 1]) + 1], axis=1)
        past_cs_2_mean = np.mean(activity[:, int(cue_start[start]):int(cue_end[start]) + 1], axis=1)
        for i in range(start+1, len(cue_start)):
            current_cs_type = cue_codes[i]
            if current_cs_type == past_cs_1_type:
                current_temp_mean = np.mean(activity[:, int(cue_start[i]):int(cue_end[i]) + 1], axis=1)
                if normal_trials_idx[i] not in end_trials:
                    correlation.append(np.corrcoef(current_temp_mean, past_cs_2_mean)[0][1])
                past_cs_1_mean = current_temp_mean
            if current_cs_type == past_cs_2_type:
                current_temp_mean = np.mean(activity[:, int(cue_start[i]):int(cue_end[i]) + 1], axis=1)
                if normal_trials_idx[i] not in end_trials:
                    correlation.append(np.corrcoef(current_temp_mean, past_cs_1_mean)[0][1])
                past_cs_2_mean = current_temp_mean
            temp_cue_activity = np.mean(activity[:, int(cue_start[i]):int(cue_end[i]) + 1])
            if normal_trials_idx[i] not in end_trials:
                cue_activity.append(temp_cue_activity)
        all_correlation.append(correlation)
        all_cue_activity.append(cue_activity)
    days_path = paths['base_path'] + paths['mouse'] + '/data_across_days/'
    if days:
        if path.isfile(days_path + 'activity_grouped_omit.npy') == 0 or day == 0:
            correlation_across_days = [list(range(0, days)), list(range(0, days))]
            correlation_across_days[0][day] = all_correlation
            correlation_across_days[1][day] = all_cue_activity
            np.save(days_path + 'activity_grouped_omit', correlation_across_days)
        else:
            correlation_across_days = np.load(days_path + 'activity_grouped_omit.npy', allow_pickle=True)
            correlation_across_days[0][day] = all_correlation
            correlation_across_days[1][day] = all_cue_activity
            np.save(days_path + 'activity_grouped_omit', correlation_across_days)


def activity_across_trials_across_days_R1(norm_deconvolved, behavior, y_pred, paths, day, days):
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

    days_path = paths['base_path'] + paths['mouse'] + '/data_across_days/'
    align_vec = np.load(days_path + 'alignment_across_days_intersect.npy', allow_pickle=True)
    align_vec = align_vec[0][day]

    activity = norm_deconvolved.to_numpy()
    activity = activity[align_vec >= 0, :]
    print(len(activity))

    normal_trials_idx = np.where(np.isin(behavior['cue_codes'], [behavior['cs_1_code'], behavior['cs_2_code']]))[0]

    times_considered = preprocess_opto.get_times_considered(y_pred, behavior)
    y_pred = true_reactivations(y_pred)
    y_pred_cs_1 = y_pred[:, 0]
    y_pred_cs_2 = y_pred[:, 1]
    all_y_pred = y_pred_cs_1 + y_pred_cs_2
    cue_start = behavior['onsets'][normal_trials_idx]
    cue_end = behavior['offsets'][normal_trials_idx]
    cue_codes = behavior['cue_codes'][normal_trials_idx]
    end_trials = behavior['end_trials']
    reactivation_prob = []
    correlation = []
    cue_activity = []

    start = max(list(cue_codes).index(behavior['cs_1_code']), list(cue_codes).index(behavior['cs_2_code']))
    past_cs_1_type = cue_codes[start - 1]
    past_cs_2_type = cue_codes[start]
    past_cs_1_mean = np.mean(activity[:, int(cue_start[start - 1]):int(cue_end[start - 1]) + 1], axis=1)
    past_cs_2_mean = np.mean(activity[:, int(cue_start[start]):int(cue_end[start]) + 1], axis=1)
    for i in range(start + 1, len(cue_start)):
        current_cs_type = cue_codes[i]
        if current_cs_type == past_cs_1_type:
            current_temp_mean = np.mean(activity[:, int(cue_start[i]):int(cue_end[i]) + 1], axis=1)
            if normal_trials_idx[i] not in end_trials:
                correlation.append(np.corrcoef(current_temp_mean, past_cs_2_mean)[0][1])
            past_cs_1_mean = current_temp_mean
        if current_cs_type == past_cs_2_type:
            current_temp_mean = np.mean(activity[:, int(cue_start[i]):int(cue_end[i]) + 1], axis=1)
            if normal_trials_idx[i] not in end_trials:
                correlation.append(np.corrcoef(current_temp_mean, past_cs_1_mean)[0][1])
            past_cs_2_mean = current_temp_mean
        temp_sum_reactivation = np.sum(all_y_pred[int(cue_start[i]):int(cue_start[i]) + int((behavior['framerate'] *
                                                                                             (behavior[
                                                                                                  'iti'] + 6)))]) / np.sum(
            times_considered[int(cue_start[i]):int(cue_start[i]) + int((behavior['framerate'] *
                                                                        (behavior['iti'] + 6)))]) * int(
            behavior['framerate'])
        temp_cue_activity = np.mean(activity[:, int(cue_start[i]):int(cue_end[i]) + 1])
        if normal_trials_idx[i] not in end_trials:
            reactivation_prob.append(temp_sum_reactivation)
            cue_activity.append(temp_cue_activity)

    days_path = paths['base_path'] + paths['mouse'] + '/data_across_days/'
    if days:
        if path.isfile(days_path + 'activity_across_days.npy') == 0 or day == 0:
            correlation_across_days = [list(range(0, days)), list(range(0, days)), list(range(0, days)),
                                       list(range(0, days)), list(range(0, days)), list(range(0, days)),
                                       list(range(0, days))]
            correlation_across_days[0][day] = correlation
            correlation_across_days[1][day] = reactivation_prob
            correlation_across_days[2][day] = cue_activity
            np.save(days_path + 'activity_across_days', correlation_across_days)
        else:
            correlation_across_days = np.load(days_path + 'activity_across_days.npy', allow_pickle=True)
            correlation_across_days[0][day] = correlation
            correlation_across_days[1][day] = reactivation_prob
            correlation_across_days[2][day] = cue_activity
            np.save(days_path + 'activity_across_days', correlation_across_days)


def reactivation_spatial_drift_R3(norm_deconvolved, y_pred, behavior, planes, idx, paths, day, days):
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
            to_delete = preprocess_opto.cells_to_delete(paths, plane, plane_path, 0)
            stat_plane = stat_plane[to_delete == 1]
            stat = np.concatenate((stat, stat_plane))

    activity = norm_deconvolved.to_numpy()
    activity = activity[idx['both'].index]
    # sig_cells = preprocess_opto.sig_reactivated_cells([], [], [], [], [], paths, 0)
    # activity = activity[sig_cells > 0, :]
    stat = stat[idx['both'].index]
    # stat = stat[sig_cells > 0]

    drift_all_cs_1_x = []
    drift_all_cs_2_x = []
    drift_all_cs_1_y = []
    drift_all_cs_2_y = []
    x_label = []
    factor = 2
    idx = .5
    for i in range(0, behavior['task_runs']):
        trials_per_run = int(int(len(behavior['onsets'])) / int(behavior['task_runs']))
        start = int(behavior['onsets'][i * trials_per_run])
        end = int(behavior['offsets'][((i + 1) * trials_per_run) - 1])
        step = round((end - start) / factor) + 1
        for j in range(0, factor):
            reactivation_cs_1 = y_pred[:, 0][start + (j * step):start + ((j + 1) * step)]
            reactivation_cs_2 = y_pred[:, 1][start + (j * step):start + ((j + 1) * step)]
            p_threshold = .75
            cs_1_peak = 0
            cs_2_peak = 0
            i = 0
            reactivation_frames_cs_1 = np.zeros(len(reactivation_cs_1))
            reactivation_frames_cs_2 = np.zeros(len(reactivation_cs_1))
            next_r = 0
            while i < len(reactivation_cs_1) - 2:
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
                            reactivation_frames_cs_1[r_start:r_end] = 1
                        if cs_2_peak > p_threshold:
                            reactivation_frames_cs_2[r_start:r_end] = 1
                        i = r_end
                        cs_1_peak = 0
                        cs_2_peak = 0
            activity_reactivation = activity[:, start + (j * step):start + ((j + 1) * step)]
            activity_reactivation_cs_1 = np.mean(activity_reactivation[:, reactivation_frames_cs_1 > 0], axis=1) * behavior['framerate'][0][0]
            activity_reactivation_cs_2 = np.mean(activity_reactivation[:, reactivation_frames_cs_2 > 0], axis=1) * behavior['framerate'][0][0]

            pos_vec_cs_1_x = []
            pos_vec_cs_1_y = []
            pos_vec_cs_2_x = []
            pos_vec_cs_2_y = []
            for n in range(0, len(stat)):
                ypix = int(np.mean(stat[n]['ypix'][~stat[n]['overlap']]))
                xpix = int(np.mean(stat[n]['xpix'][~stat[n]['overlap']]))
                r_activity_cs_1 = activity_reactivation_cs_1[n]
                r_activity_cs_2 = activity_reactivation_cs_2[n]
                pos_vec_cs_1_x.append(xpix * r_activity_cs_1)
                pos_vec_cs_2_x.append(xpix * r_activity_cs_2)
                pos_vec_cs_1_y.append(ypix * r_activity_cs_1)
                pos_vec_cs_2_y.append(ypix * r_activity_cs_2)

            drift_cs_1_x = np.nanmean(pos_vec_cs_1_x)
            drift_cs_2_x = np.nanmean(pos_vec_cs_2_x)
            drift_cs_1_y = np.nanmean(pos_vec_cs_1_y)
            drift_cs_2_y = np.nanmean(pos_vec_cs_2_y)
            x_label.append((step / behavior['framerate'] / 60 / 60 * idx)[0][0])
            idx = idx + 1
            drift_all_cs_1_x.append(drift_cs_1_x)
            drift_all_cs_2_x.append(drift_cs_2_x)
            drift_all_cs_1_y.append(drift_cs_1_y)
            drift_all_cs_2_y.append(drift_cs_2_y)

    days_path = paths['base_path'] + paths['mouse'] + '/data_across_days/'
    if days:
        if path.isfile(days_path + 'reactivation_spatial_drift.npy') == 0 or day == 0:
            reactivation_spatial_days = [list(range(0, days)), list(range(0, days)), list(range(0, days)),
                                         list(range(0, days))]
            reactivation_spatial_days[0][day] = drift_all_cs_1_x
            reactivation_spatial_days[1][day] = drift_all_cs_2_x
            reactivation_spatial_days[2][day] = drift_all_cs_1_y
            reactivation_spatial_days[3][day] = drift_all_cs_2_y
            np.save(days_path + 'reactivation_spatial_drift', reactivation_spatial_days)
        else:
            reactivation_spatial_days = np.load(days_path + 'reactivation_spatial_drift.npy', allow_pickle=True)
            reactivation_spatial_days[0][day] = drift_all_cs_1_x
            reactivation_spatial_days[1][day] = drift_all_cs_2_x
            reactivation_spatial_days[2][day] = drift_all_cs_1_y
            reactivation_spatial_days[3][day] = drift_all_cs_2_y
            np.save(days_path + 'reactivation_spatial_drift', reactivation_spatial_days)


def activity_across_trials_layer_R1(norm_deconvolved, upper_layer_cells, lower_layer_cells, idx, y_pred, behavior,
                                     paths, day, days):

    # sig_cells = preprocess_opto.sig_reactivated_cells([], [], [], [], [], paths, 0)
    idx_both = np.array(idx['both'].index.copy())
    # idx_both[sig_cells == 0] = -1

    upper_layer_idx = np.intersect1d(np.array(idx_both), upper_layer_cells).astype(int)
    lower_layer_idx = np.intersect1d(np.array(idx_both), lower_layer_cells).astype(int)

    activity = norm_deconvolved.to_numpy()
    activity_upper = activity[upper_layer_idx]
    activity_lower = activity[lower_layer_idx]

    correlation_upper = activity_across_trials_layer_R1_helper(activity_upper, y_pred, behavior)
    correlation_lower = activity_across_trials_layer_R1_helper(activity_lower, y_pred, behavior)

    days_path = paths['base_path'] + paths['mouse'] + '/data_across_days/'
    if days:
        if path.isfile(days_path + 'activity_layer.npy') == 0 or day == 0:
            activity_across_trials = [list(range(0, days)), list(range(0, days))]
            activity_across_trials[0][day] = correlation_upper
            activity_across_trials[1][day] = correlation_lower
            np.save(days_path + 'activity_layer', activity_across_trials)
        else:
            activity_across_trials = np.load(days_path + 'activity_layer.npy', allow_pickle=True)
            activity_across_trials[0][day] = correlation_upper
            activity_across_trials[1][day] = correlation_lower
            np.save(days_path + 'activity_layer', activity_across_trials)


def activity_across_trials_layer_R1_helper(activity, y_pred, behavior):
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

    normal_trials_idx = np.where(np.isin(behavior['cue_codes'], [behavior['cs_1_code'], behavior['cs_2_code']]))[0]

    times_considered = preprocess_opto.get_times_considered(y_pred, behavior)
    y_pred = true_reactivations(y_pred)
    y_pred_cs_1 = y_pred[:, 0]
    y_pred_cs_2 = y_pred[:, 1]
    all_y_pred = y_pred_cs_1 + y_pred_cs_2
    cue_start = behavior['onsets'][normal_trials_idx]
    cue_end = behavior['offsets'][normal_trials_idx]
    cue_codes = behavior['cue_codes'][normal_trials_idx]
    end_trials = behavior['end_trials']
    reactivation_prob = []
    correlation = []
    cue_activity = []

    start = max(list(cue_codes).index(behavior['cs_1_code']), list(cue_codes).index(behavior['cs_2_code']))
    past_cs_1_type = cue_codes[start - 1]
    past_cs_2_type = cue_codes[start]
    past_cs_1_mean = np.mean(activity[:, int(cue_start[start - 1]):int(cue_end[start - 1]) + 1], axis=1)
    past_cs_2_mean = np.mean(activity[:, int(cue_start[start]):int(cue_end[start]) + 1], axis=1)
    for i in range(start + 1, len(cue_start)):
        current_cs_type = cue_codes[i]
        if current_cs_type == past_cs_1_type:
            current_temp_mean = np.mean(activity[:, int(cue_start[i]):int(cue_end[i]) + 1], axis=1)
            if normal_trials_idx[i] not in end_trials:
                correlation.append(np.corrcoef(current_temp_mean, past_cs_2_mean)[0][1])
            past_cs_1_mean = current_temp_mean
        if current_cs_type == past_cs_2_type:
            current_temp_mean = np.mean(activity[:, int(cue_start[i]):int(cue_end[i]) + 1], axis=1)
            if normal_trials_idx[i] not in end_trials:
                correlation.append(np.corrcoef(current_temp_mean, past_cs_1_mean)[0][1])
            past_cs_2_mean = current_temp_mean
        temp_sum_reactivation = np.sum(all_y_pred[int(cue_start[i]):int(cue_start[i]) + int((behavior['framerate'] *
                                                                                             (behavior[
                                                                                                  'iti'] + 6)))]) / np.sum(
            times_considered[int(cue_start[i]):int(cue_start[i]) + int((behavior['framerate'] *
                                                                        (behavior['iti'] + 6)))]) * int(
            behavior['framerate'])
        temp_cue_activity = np.mean(activity[:, int(cue_start[i]):int(cue_end[i]) + 1])
        if normal_trials_idx[i] not in end_trials:
            reactivation_prob.append(temp_sum_reactivation)
            cue_activity.append(temp_cue_activity)

    return correlation


def activity_across_trials_grouped_baseline_R3(norm_deconvolved, behavior, idx, paths, day, days):
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
    # sig_cells = preprocess_opto.sig_reactivated_cells([], [], [], [], [], paths, 0)
    # activity = activity[sig_cells > 0, :]

    [no_change_cells_cs_1, increase_sig_cells_cs_1, decrease_sig_cells_cs_1] = preprocess_opto.group_neurons(activity, behavior, 'cs_1')

    [no_change_cells_cs_2, increase_sig_cells_cs_2, decrease_sig_cells_cs_2] = preprocess_opto.group_neurons(activity, behavior, 'cs_2')

    all_baseline_activity = []
    for g in range(0, 3):
        activity = norm_deconvolved.to_numpy()
        activity = activity[idx['both'].index]
        # sig_cells = preprocess_opto.sig_reactivated_cells([], [], [], [], [], paths, 0)
        # activity = activity[sig_cells > 0, :]

        if g == 0:
            cells_to_use = no_change_cells_cs_1 + no_change_cells_cs_2
            activity = activity[cells_to_use == 2, :]
        if g == 1:
            cells_to_use = increase_sig_cells_cs_1 + increase_sig_cells_cs_2
            activity = activity[cells_to_use > 0, :]
        if g == 2:
            cells_to_use = decrease_sig_cells_cs_1 + decrease_sig_cells_cs_2
            activity = activity[cells_to_use > 0, :]

        baseline_activity = np.mean(activity[:, 0:behavior['frames_per_run']])
        all_baseline_activity.append(baseline_activity)
    days_path = paths['base_path'] + paths['mouse'] + '/data_across_days/'
    if days:
        if path.isfile(days_path + 'baseline_activity_grouped.npy') == 0 or day == 0:
            across_days = [list(range(0, days))]
            across_days[0][day] = all_baseline_activity
            np.save(days_path + 'baseline_activity_grouped', across_days)
        else:
            across_days = np.load(days_path + 'baseline_activity_grouped.npy', allow_pickle=True)
            across_days[0][day] = all_baseline_activity
            np.save(days_path + 'baseline_activity_grouped', across_days)


def neuron_count_grouped_layer_R3(norm_deconvolved, behavior, upper_layer_cells, lower_layer_cells, idx, paths, day, days):

    activity = norm_deconvolved.to_numpy()
    activity = activity[idx['both'].index]
    # sig_cells = preprocess_opto.sig_reactivated_cells([], [], [], [], [], paths, 0)
    # activity = activity[sig_cells > 0, :]
    [no_change_cells_cs_1, increase_sig_cells_cs_1, decrease_sig_cells_cs_1] = preprocess_opto.group_neurons(activity,
                                                                                                        behavior,
                                                                                                        'cs_1')
    [no_change_cells_cs_2, increase_sig_cells_cs_2, decrease_sig_cells_cs_2] = preprocess_opto.group_neurons(activity,
                                                                                                        behavior,
                                                                                                        'cs_2')
    no_change_cells = no_change_cells_cs_1 + no_change_cells_cs_2
    increase_sig_cells = increase_sig_cells_cs_1 + increase_sig_cells_cs_2
    decrease_sig_cells = decrease_sig_cells_cs_1 + decrease_sig_cells_cs_2

    idx_both = idx['both'].index
    # idx_both = idx_both[sig_cells > 0]

    idx_both_nc = idx_both[no_change_cells == 2]
    idx_both_inc = idx_both[increase_sig_cells > 0]
    idx_both_dec = idx_both[decrease_sig_cells > 0]

    upper_layer_idx_nc = np.intersect1d(np.array(idx_both_nc), upper_layer_cells).astype(int)
    lower_layer_idx_nc = np.intersect1d(np.array(idx_both_nc), lower_layer_cells).astype(int)
    upper_layer_idx_inc = np.intersect1d(np.array(idx_both_inc), upper_layer_cells).astype(int)
    lower_layer_idx_inc = np.intersect1d(np.array(idx_both_inc), lower_layer_cells).astype(int)
    upper_layer_idx_dec = np.intersect1d(np.array(idx_both_dec), upper_layer_cells).astype(int)
    lower_layer_idx_dec = np.intersect1d(np.array(idx_both_dec), lower_layer_cells).astype(int)

    days_path = paths['base_path'] + paths['mouse'] + '/data_across_days/'
    if days:
        if path.isfile(days_path + 'activity_grouped_count_layer.npy') == 0 or day == 0:
            num_across_days = [list(range(0, days)), list(range(0, days)), list(range(0, days)), list(range(0, days)),
                               list(range(0, days)), list(range(0, days))]
            num_across_days[0][day] = len(upper_layer_idx_nc) / len(upper_layer_cells)
            num_across_days[1][day] = len(lower_layer_idx_nc) / len(lower_layer_cells)
            num_across_days[2][day] = len(upper_layer_idx_inc) / len(upper_layer_cells)
            num_across_days[3][day] = len(lower_layer_idx_inc) / len(lower_layer_cells)
            num_across_days[4][day] = len(upper_layer_idx_dec) / len(upper_layer_cells)
            num_across_days[5][day] = len(lower_layer_idx_dec) / len(lower_layer_cells)
            np.save(days_path + 'activity_grouped_count_layer', num_across_days)
        else:
            num_across_days = np.load(days_path + 'activity_grouped_count_layer.npy', allow_pickle=True)
            num_across_days[0][day] = len(upper_layer_idx_nc) / len(upper_layer_cells)
            num_across_days[1][day] = len(lower_layer_idx_nc) / len(lower_layer_cells)
            num_across_days[2][day] = len(upper_layer_idx_inc) / len(upper_layer_cells)
            num_across_days[3][day] = len(lower_layer_idx_inc) / len(lower_layer_cells)
            num_across_days[4][day] = len(upper_layer_idx_dec) / len(upper_layer_cells)
            num_across_days[5][day] = len(lower_layer_idx_dec) / len(lower_layer_cells)
            np.save(days_path + 'activity_grouped_count_layer', num_across_days)


def reactivation_spatial_grouped_R3(norm_deconvolved, behavior, planes, idx, paths, day, days):
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
            to_delete = preprocess_opto.cells_to_delete(paths, plane, plane_path, 0)
            stat_plane = stat_plane[to_delete == 1]
            stat = np.concatenate((stat, stat_plane))

    activity = norm_deconvolved.to_numpy()
    activity = activity[idx['both'].index]
    # sig_cells = preprocess_opto.sig_reactivated_cells([], [], [], [], [], paths, 0)
    # activity = activity[sig_cells > 0, :]
    [no_change_cells_cs_1, increase_sig_cells_cs_1, decrease_sig_cells_cs_1] = preprocess_opto.group_neurons(activity,
                                                                                                        behavior,
                                                                                                        'cs_1')
    [no_change_cells_cs_2, increase_sig_cells_cs_2, decrease_sig_cells_cs_2] = preprocess_opto.group_neurons(activity,
                                                                                                        behavior,
                                                                                                        'cs_2')
    no_change_cells = no_change_cells_cs_1 + no_change_cells_cs_2
    increase_sig_cells = increase_sig_cells_cs_1 + increase_sig_cells_cs_2
    decrease_sig_cells = decrease_sig_cells_cs_1 + decrease_sig_cells_cs_2

    idx_both = idx['both'].index
    # idx_both = idx_both[sig_cells > 0]
    idx_both_nc = idx_both[no_change_cells == 2]
    idx_both_inc = idx_both[increase_sig_cells > 0]
    idx_both_dec = idx_both[decrease_sig_cells > 0]

    retinotopy = loadmat(paths['base_path'] + paths['mouse'] + '/' + 'retinotopy' + '/' + 'retinotopy.mat')
    retinotopy_mouse = loadmat(paths['save_path'] + 'saved_data/retinotopy_day.mat')

    area_li_count_nc = 0
    area_li_count_inc = 0
    area_li_count_dec = 0
    area_por_count_nc = 0
    area_por_count_inc = 0
    area_por_count_dec = 0
    area_lm_count_nc = 0
    area_lm_count_inc = 0
    area_lm_count_dec = 0
    area_p_count_nc = 0
    area_p_count_inc = 0
    area_p_count_dec = 0
    total_li = 0
    total_p = 0
    total_lm = 0
    total_por = 0

    for n in range(0, len(stat)):
        ypix = int(np.mean(stat[n]['ypix'][~stat[n]['overlap']]))
        xpix = int(np.mean(stat[n]['xpix'][~stat[n]['overlap']]))
        diff_x = retinotopy_mouse['base_c'][0][0] - retinotopy_mouse['imaging_c'][0][0]
        diff_y = retinotopy_mouse['base_c'][0][1] - retinotopy_mouse['imaging_c'][0][1]
        true_xpix = xpix + diff_x
        true_ypix = ypix + diff_y
        if retinotopy['LI'][true_ypix][true_xpix] == 1:
           total_li += 1
        if retinotopy['POR'][true_ypix][true_xpix] == 1:
            total_por += 1
        if retinotopy['P'][true_ypix][true_xpix] == 1:
            total_p += 1
        if retinotopy['LM'][true_ypix][true_xpix] == 1:
            total_lm += 1
        if n in idx_both_nc:
            if retinotopy['LI'][true_ypix][true_xpix] == 1:
                area_li_count_nc += 1
            if retinotopy['POR'][true_ypix][true_xpix] == 1:
                area_por_count_nc += 1
            if retinotopy['P'][true_ypix][true_xpix] == 1:
                area_p_count_nc += 1
            if retinotopy['LM'][true_ypix][true_xpix] == 1:
                area_lm_count_nc += 1
        if n in idx_both_inc:
            if retinotopy['LI'][true_ypix][true_xpix] == 1:
                area_li_count_inc += 1
            if retinotopy['POR'][true_ypix][true_xpix] == 1:
                area_por_count_inc += 1
            if retinotopy['P'][true_ypix][true_xpix] == 1:
                area_p_count_inc += 1
            if retinotopy['LM'][true_ypix][true_xpix] == 1:
                area_lm_count_inc += 1
        if n in idx_both_dec:
            if retinotopy['LI'][true_ypix][true_xpix] == 1:
                area_li_count_dec += 1
            if retinotopy['POR'][true_ypix][true_xpix] == 1:
                area_por_count_dec += 1
            if retinotopy['P'][true_ypix][true_xpix] == 1:
                area_p_count_dec += 1
            if retinotopy['LM'][true_ypix][true_xpix] == 1:
                area_lm_count_dec += 1

    days_path = paths['base_path'] + paths['mouse'] + '/data_across_days/'
    if days:
        if path.isfile(days_path + 'reactivation_spatial_grouped.npy') == 0 or day == 0:
            reactivation_spatial_days = [list(range(0, days)), list(range(0, days)), list(range(0, days)),
                                         list(range(0, days)), list(range(0, days)), list(range(0, days)),
                                         list(range(0, days)), list(range(0, days)), list(range(0, days)),
                                         list(range(0, days)), list(range(0, days)), list(range(0, days)),
                                         list(range(0, days)), list(range(0, days)), list(range(0, days)),
                                         list(range(0, days))]
            reactivation_spatial_days[0][day] = area_li_count_nc
            reactivation_spatial_days[1][day] = area_por_count_nc
            reactivation_spatial_days[2][day] = area_p_count_nc
            reactivation_spatial_days[3][day] = area_lm_count_nc
            reactivation_spatial_days[4][day] = area_li_count_inc
            reactivation_spatial_days[5][day] = area_por_count_inc
            reactivation_spatial_days[6][day] = area_p_count_inc
            reactivation_spatial_days[7][day] = area_lm_count_inc
            reactivation_spatial_days[8][day] = area_li_count_dec
            reactivation_spatial_days[9][day] = area_por_count_dec
            reactivation_spatial_days[10][day] = area_p_count_dec
            reactivation_spatial_days[11][day] = area_lm_count_dec
            reactivation_spatial_days[12][day] = total_li
            reactivation_spatial_days[13][day] = total_por
            reactivation_spatial_days[14][day] = total_p
            reactivation_spatial_days[15][day] = total_lm
            np.save(days_path + 'reactivation_spatial_grouped', reactivation_spatial_days)
        else:
            reactivation_spatial_days = np.load(days_path + 'reactivation_spatial_grouped.npy', allow_pickle=True)
            reactivation_spatial_days[0][day] = area_li_count_nc
            reactivation_spatial_days[1][day] = area_por_count_nc
            reactivation_spatial_days[2][day] = area_p_count_nc
            reactivation_spatial_days[3][day] = area_lm_count_nc
            reactivation_spatial_days[4][day] = area_li_count_inc
            reactivation_spatial_days[5][day] = area_por_count_inc
            reactivation_spatial_days[6][day] = area_p_count_inc
            reactivation_spatial_days[7][day] = area_lm_count_inc
            reactivation_spatial_days[8][day] = area_li_count_dec
            reactivation_spatial_days[9][day] = area_por_count_dec
            reactivation_spatial_days[10][day] = area_p_count_dec
            reactivation_spatial_days[11][day] = area_lm_count_dec
            reactivation_spatial_days[12][day] = total_li
            reactivation_spatial_days[13][day] = total_por
            reactivation_spatial_days[14][day] = total_p
            reactivation_spatial_days[15][day] = total_lm
            np.save(days_path + 'reactivation_spatial_grouped', reactivation_spatial_days)


def noise_correlation_grouped(norm_deconvolved, behavior, idx, paths, day, days):
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
    # sig_cells = preprocess_opto.sig_reactivated_cells([], [], [], [], [], paths, 0)
    # activity = activity[sig_cells > 0, :]

    [no_change_cells_cs_1, increase_sig_cells_cs_1, decrease_sig_cells_cs_1] = preprocess_opto.group_neurons(activity, behavior, 'cs_1')

    [no_change_cells_cs_2, increase_sig_cells_cs_2, decrease_sig_cells_cs_2] = preprocess_opto.group_neurons(activity, behavior, 'cs_2')

    all_noise_corr = []
    for g in range(0, 3):
        activity = norm_deconvolved.to_numpy()
        activity = activity[idx['both'].index]
        # sig_cells = preprocess_opto.sig_reactivated_cells([], [], [], [], [], paths, 0)
        # activity = activity[sig_cells > 0, :]

        if g == 0:
            cells_to_use = no_change_cells_cs_1 + no_change_cells_cs_2
            activity = activity[cells_to_use == 2, :]
        if g == 1:
            cells_to_use = increase_sig_cells_cs_1 + increase_sig_cells_cs_2
            activity = activity[cells_to_use > 0, :]
        if g == 2:
            cells_to_use = decrease_sig_cells_cs_1 + decrease_sig_cells_cs_2
            activity = activity[cells_to_use > 0, :]

        normal_trials_idx = np.where(np.isin(behavior['cue_codes'], [behavior['cs_1_code'], behavior['cs_2_code']]))[0]
        cue_start = behavior['onsets'][normal_trials_idx]
        cue_codes = behavior['cue_codes'][normal_trials_idx]
        start = max(list(cue_codes).index(behavior['cs_1_code']), list(cue_codes).index(behavior['cs_2_code']))
        frames = []
        for i in range(start+1, len(cue_start)):
            frames.append(np.arange(int(cue_start[i]), cue_start[i] + 21))
        frames = np.concatenate(frames, axis=0)

        noise_corr_all = []
        noise_corr_all_shift = []
        for i in range(0, len(activity)):
            noise_corr = []
            noise_corr_shift = []
            for j in range(0, len(activity)):
                if i != j:
                    noise_corr.append(np.corrcoef(activity[i, frames], activity[j, frames])[0][1])
                    noise_corr_shift.append(np.corrcoef(activity[i, frames[0:len(frames)-21]], activity[j, frames[21:len(frames)]])[0][1])
            noise_corr_all.append(np.mean(noise_corr))
            noise_corr_all_shift.append(np.mean(noise_corr_shift))
        noise_corr_total = np.mean(noise_corr_all) - np.mean(noise_corr_all_shift)
        all_noise_corr.append(noise_corr_total)

    days_path = paths['base_path'] + paths['mouse'] + '/data_across_days/'
    if days:
        if path.isfile(days_path + 'noise_grouped.npy') == 0 or day == 0:
            correlation_across_days = [list(range(0, days))]
            correlation_across_days[0][day] = all_noise_corr
            np.save(days_path + 'noise_grouped', correlation_across_days)
        else:
            correlation_across_days = np.load(days_path + 'noise_grouped.npy', allow_pickle=True)
            correlation_across_days[0][day] = all_noise_corr
            np.save(days_path + 'noise_grouped', correlation_across_days)


def activity_across_trials_grouped_separate(norm_deconvolved, behavior, idx, paths, day, days):
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
    # sig_cells = preprocess_opto.sig_reactivated_cells([], [], [], [], [], paths, 0)
    # activity = activity[sig_cells > 0, :]
    [no_change_cells_cs_1, increase_sig_cells_cs_1, decrease_sig_cells_cs_1] = preprocess_opto.group_neurons(activity, behavior, 'cs_1')

    [no_change_cells_cs_2, increase_sig_cells_cs_2, decrease_sig_cells_cs_2] = preprocess_opto.group_neurons(activity, behavior, 'cs_2')

    cs1d_cs1 = []
    cs1d_cs2 = []
    cs2d_cs1 = []
    cs2d_cs2 = []
    activity = norm_deconvolved.to_numpy()
    activity = activity[idx['both'].index]
    # sig_cells = preprocess_opto.sig_reactivated_cells([], [], [], [], [], paths, 0)
    # activity = activity[sig_cells > 0, :]

    cells_to_use_1 = increase_sig_cells_cs_1
    cells_to_use_2 = increase_sig_cells_cs_2

    activity_1 = activity[cells_to_use_1 > 0, :]
    activity_2 = activity[cells_to_use_2 > 0, :]

    normal_trials_idx = np.where(np.isin(behavior['cue_codes'], [behavior['cs_1_code'], behavior['cs_2_code']]))[0]

    cue_start = behavior['onsets'][normal_trials_idx]
    cue_end = behavior['offsets'][normal_trials_idx]
    cue_codes = behavior['cue_codes'][normal_trials_idx]
    end_trials = behavior['end_trials']

    start = max(list(cue_codes).index(behavior['cs_1_code']), list(cue_codes).index(behavior['cs_2_code']))
    for i in range(start+1, len(cue_start)):
        if normal_trials_idx[i] not in end_trials:
            if cue_codes[i] == behavior['cs_1_code']:
                cs1d_cs1.append(np.mean(activity_1[:, int(cue_start[i]):int(cue_end[i]) + 1]))
                cs2d_cs1.append(np.mean(activity_2[:, int(cue_start[i]):int(cue_end[i]) + 1]))
            if cue_codes[i] == behavior['cs_2_code']:
                cs1d_cs2.append(np.mean(activity_1[:, int(cue_start[i]):int(cue_end[i]) + 1]))
                cs2d_cs2.append(np.mean(activity_2[:, int(cue_start[i]):int(cue_end[i]) + 1]))
    days_path = paths['base_path'] + paths['mouse'] + '/data_across_days/'
    if days:
        if path.isfile(days_path + 'activity_grouped_increase.npy') == 0 or day == 0:
            across_days = [list(range(0, days)), list(range(0, days)), list(range(0, days)),
                           list(range(0, days))]
            across_days[0][day] = cs1d_cs1
            across_days[1][day] = cs1d_cs2
            across_days[2][day] = cs2d_cs2
            across_days[3][day] = cs2d_cs1
            np.save(days_path + 'activity_grouped_increase', across_days)
        else:
            across_days = np.load(days_path + 'activity_grouped_increase.npy', allow_pickle=True)
            across_days[0][day] = cs1d_cs1
            across_days[1][day] = cs1d_cs2
            across_days[2][day] = cs2d_cs2
            across_days[3][day] = cs2d_cs1
            np.save(days_path + 'activity_grouped_increase', across_days)

    cs1d_cs1 = []
    cs1d_cs2 = []
    cs2d_cs1 = []
    cs2d_cs2 = []
    cells_to_use_1 = decrease_sig_cells_cs_1
    cells_to_use_2 = decrease_sig_cells_cs_2

    activity_1 = activity[cells_to_use_1 > 0, :]
    activity_2 = activity[cells_to_use_2 > 0, :]

    normal_trials_idx = np.where(np.isin(behavior['cue_codes'], [behavior['cs_1_code'], behavior['cs_2_code']]))[0]

    cue_start = behavior['onsets'][normal_trials_idx]
    cue_end = behavior['offsets'][normal_trials_idx]
    cue_codes = behavior['cue_codes'][normal_trials_idx]
    end_trials = behavior['end_trials']

    start = max(list(cue_codes).index(behavior['cs_1_code']), list(cue_codes).index(behavior['cs_2_code']))
    for i in range(start + 1, len(cue_start)):
        if normal_trials_idx[i] not in end_trials:
            if cue_codes[i] == behavior['cs_1_code']:
                cs1d_cs1.append(np.mean(activity_1[:, int(cue_start[i]):int(cue_end[i]) + 1]))
                cs2d_cs1.append(np.mean(activity_2[:, int(cue_start[i]):int(cue_end[i]) + 1]))
            if cue_codes[i] == behavior['cs_2_code']:
                cs1d_cs2.append(np.mean(activity_1[:, int(cue_start[i]):int(cue_end[i]) + 1]))
                cs2d_cs2.append(np.mean(activity_2[:, int(cue_start[i]):int(cue_end[i]) + 1]))
    days_path = paths['base_path'] + paths['mouse'] + '/data_across_days/'
    if days:
        if path.isfile(days_path + 'activity_grouped_decrease.npy') == 0 or day == 0:
            across_days = [list(range(0, days)), list(range(0, days)), list(range(0, days)),
                           list(range(0, days))]
            across_days[0][day] = cs1d_cs1
            across_days[1][day] = cs1d_cs2
            across_days[2][day] = cs2d_cs2
            across_days[3][day] = cs2d_cs1
            np.save(days_path + 'activity_grouped_decrease', across_days)
        else:
            across_days = np.load(days_path + 'activity_grouped_decrease.npy', allow_pickle=True)
            across_days[0][day] = cs1d_cs1
            across_days[1][day] = cs1d_cs2
            across_days[2][day] = cs2d_cs2
            across_days[3][day] = cs2d_cs1
            np.save(days_path + 'activity_grouped_decrease', across_days)

    cs1d_cs1 = []
    cs1d_cs2 = []
    cs2d_cs1 = []
    cs2d_cs2 = []
    cells_to_use_1 = no_change_cells_cs_1
    cells_to_use_2 = no_change_cells_cs_2

    activity_1 = activity[cells_to_use_1 > 0, :]
    activity_2 = activity[cells_to_use_2 > 0, :]

    normal_trials_idx = np.where(np.isin(behavior['cue_codes'], [behavior['cs_1_code'], behavior['cs_2_code']]))[0]

    cue_start = behavior['onsets'][normal_trials_idx]
    cue_end = behavior['offsets'][normal_trials_idx]
    cue_codes = behavior['cue_codes'][normal_trials_idx]
    end_trials = behavior['end_trials']

    start = max(list(cue_codes).index(behavior['cs_1_code']), list(cue_codes).index(behavior['cs_2_code']))
    for i in range(start + 1, len(cue_start)):
        if normal_trials_idx[i] not in end_trials:
            if cue_codes[i] == behavior['cs_1_code']:
                cs1d_cs1.append(np.mean(activity_1[:, int(cue_start[i]):int(cue_end[i]) + 1]))
                cs2d_cs1.append(np.mean(activity_2[:, int(cue_start[i]):int(cue_end[i]) + 1]))
            if cue_codes[i] == behavior['cs_2_code']:
                cs1d_cs2.append(np.mean(activity_1[:, int(cue_start[i]):int(cue_end[i]) + 1]))
                cs2d_cs2.append(np.mean(activity_2[:, int(cue_start[i]):int(cue_end[i]) + 1]))
    days_path = paths['base_path'] + paths['mouse'] + '/data_across_days/'
    if days:
        if path.isfile(days_path + 'activity_grouped_nochange.npy') == 0 or day == 0:
            across_days = [list(range(0, days)), list(range(0, days)), list(range(0, days)),
                           list(range(0, days))]
            across_days[0][day] = cs1d_cs1
            across_days[1][day] = cs1d_cs2
            across_days[2][day] = cs2d_cs2
            across_days[3][day] = cs2d_cs1
            np.save(days_path + 'activity_grouped_nochange', across_days)
        else:
            across_days = np.load(days_path + 'activity_grouped_nochange.npy', allow_pickle=True)
            across_days[0][day] = cs1d_cs1
            across_days[1][day] = cs1d_cs2
            across_days[2][day] = cs2d_cs2
            across_days[3][day] = cs2d_cs1
            np.save(days_path + 'activity_grouped_nochange', across_days)


def activity_across_trials_grouped_decrease_novelty_R1(norm_deconvolved, behavior, idx, paths, day, days):
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
    # sig_cells = preprocess_opto.sig_reactivated_cells([], [], [], [], [], paths, 0)
    # activity = activity[sig_cells > 0, :]

    [no_change_cells_cs_1, increase_sig_cells_cs_1, decrease_sig_cells_cs_1] = preprocess_opto.group_neurons(activity, behavior, 'cs_1')

    [no_change_cells_cs_2, increase_sig_cells_cs_2, decrease_sig_cells_cs_2] = preprocess_opto.group_neurons(activity, behavior, 'cs_2')

    decrease_sig_cells = decrease_sig_cells_cs_1 + decrease_sig_cells_cs_2
    decrease_sig_cells[decrease_sig_cells < 2] = 0
    decrease_sig_cells[decrease_sig_cells == 2] = 1

    no_change_decrease_cells = preprocess_opto.no_change_decrease_neurons_novelty_R1(activity, behavior, decrease_sig_cells)
    decrease_sig_cells = decrease_sig_cells_cs_1 + decrease_sig_cells_cs_2
    decrease_sig_cells[decrease_sig_cells > 1] = 1

    cs1d_cs1 = []
    cs1d_cs2 = []
    cs2d_cs1 = []
    cs2d_cs2 = []
    activity = norm_deconvolved.to_numpy()
    activity = activity[idx['both'].index]
    # sig_cells = preprocess_opto.sig_reactivated_cells([], [], [], [], [], paths, 0)
    # activity = activity[sig_cells > 0, :]

    cells_to_use_1 = no_change_decrease_cells
    cells_to_use_2 = no_change_decrease_cells

    activity_1 = activity[cells_to_use_1 > 0, :]
    activity_2 = activity[cells_to_use_2 > 0, :]

    normal_trials_idx = np.where(np.isin(behavior['cue_codes'], [behavior['cs_1_code'], behavior['cs_2_code']]))[0]

    cue_start = behavior['onsets'][normal_trials_idx]
    cue_end = behavior['offsets'][normal_trials_idx]
    cue_codes = behavior['cue_codes'][normal_trials_idx]
    end_trials = behavior['end_trials']

    start = max(list(cue_codes).index(behavior['cs_1_code']), list(cue_codes).index(behavior['cs_2_code']))
    for i in range(start + 1, len(cue_start)):
        if normal_trials_idx[i] not in end_trials:
            if cue_codes[i] == behavior['cs_1_code']:
                cs1d_cs1.append(np.mean(activity_1[:, int(cue_start[i]):int(cue_end[i]) + 1]))
                cs2d_cs1.append(np.mean(activity_2[:, int(cue_start[i]):int(cue_end[i]) + 1]))
            if cue_codes[i] == behavior['cs_2_code']:
                cs1d_cs2.append(np.mean(activity_1[:, int(cue_start[i]):int(cue_end[i]) + 1]))
                cs2d_cs2.append(np.mean(activity_2[:, int(cue_start[i]):int(cue_end[i]) + 1]))
    days_path = paths['base_path'] + paths['mouse'] + '/data_across_days/'
    if days:
        if path.isfile(days_path + 'activity_grouped_no_change_decrease.npy') == 0 or day == 0:
            across_days = [list(range(0, days)), list(range(0, days)), list(range(0, days)),
                           list(range(0, days)), list(range(0, days))]
            across_days[0][day] = cs1d_cs1
            across_days[1][day] = cs1d_cs2
            across_days[2][day] = cs2d_cs2
            across_days[3][day] = cs2d_cs1
            across_days[4][day] = sum(no_change_decrease_cells)/sum(decrease_sig_cells)
            np.save(days_path + 'activity_grouped_no_change_decrease', across_days)
        else:
            across_days = np.load(days_path + 'activity_grouped_no_change_decrease.npy', allow_pickle=True)
            across_days[0][day] = cs1d_cs1
            across_days[1][day] = cs1d_cs2
            across_days[2][day] = cs2d_cs2
            across_days[3][day] = cs2d_cs1
            across_days[4][day] = sum(no_change_decrease_cells)/sum(decrease_sig_cells)
            np.save(days_path + 'activity_grouped_no_change_decrease', across_days)


def activity_across_trials_novelty_R1(norm_deconvolved, behavior, y_pred, idx, paths, day, days):
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
    # sig_cells = preprocess_opto.sig_reactivated_cells([], [], [], [], [], paths, 0)
    # activity = activity[sig_cells > 0, :]

    [no_change_cells_cs_1, increase_sig_cells_cs_1, decrease_sig_cells_cs_1] = preprocess_opto.group_neurons(activity,
                                                                                                        behavior,
                                                                                                        'cs_1')

    [no_change_cells_cs_2, increase_sig_cells_cs_2, decrease_sig_cells_cs_2] = preprocess_opto.group_neurons(activity,
                                                                                                        behavior,
                                                                                                        'cs_2')

    decrease_sig_cells = decrease_sig_cells_cs_1 + decrease_sig_cells_cs_2
    decrease_sig_cells[decrease_sig_cells < 2] = 0
    decrease_sig_cells[decrease_sig_cells == 2] = 1
    no_change_decrease_cells = preprocess_opto.no_change_decrease_neurons_novelty_R1(activity, behavior, decrease_sig_cells)
    activity = activity[no_change_decrease_cells == 0, :]

    normal_trials_idx = np.where(np.isin(behavior['cue_codes'], [behavior['cs_1_code'], behavior['cs_2_code']]))[0]

    times_considered = preprocess_opto.get_times_considered(y_pred, behavior)
    y_pred = true_reactivations(y_pred)
    y_pred_cs_1 = y_pred[:, 0]
    y_pred_cs_2 = y_pred[:, 1]
    all_y_pred = y_pred_cs_1 + y_pred_cs_2
    cue_start = behavior['onsets'][normal_trials_idx]
    cue_end = behavior['offsets'][normal_trials_idx]
    cue_codes = behavior['cue_codes'][normal_trials_idx]
    end_trials = behavior['end_trials']
    reactivation_prob = []
    correlation = []
    cue_activity = []

    start = max(list(cue_codes).index(behavior['cs_1_code']), list(cue_codes).index(behavior['cs_2_code']))
    past_cs_1_type = cue_codes[start - 1]
    past_cs_2_type = cue_codes[start]
    past_cs_1_mean = np.mean(activity[:, int(cue_start[start - 1]):int(cue_end[start - 1]) + 1], axis=1)
    past_cs_2_mean = np.mean(activity[:, int(cue_start[start]):int(cue_end[start]) + 1], axis=1)
    for i in range(start + 1, len(cue_start)):
        current_cs_type = cue_codes[i]
        if current_cs_type == past_cs_1_type:
            current_temp_mean = np.mean(activity[:, int(cue_start[i]):int(cue_end[i]) + 1], axis=1)
            if normal_trials_idx[i] not in end_trials:
                correlation.append(np.corrcoef(current_temp_mean, past_cs_2_mean)[0][1])
            past_cs_1_mean = current_temp_mean
        if current_cs_type == past_cs_2_type:
            current_temp_mean = np.mean(activity[:, int(cue_start[i]):int(cue_end[i]) + 1], axis=1)
            if normal_trials_idx[i] not in end_trials:
                correlation.append(np.corrcoef(current_temp_mean, past_cs_1_mean)[0][1])
            past_cs_2_mean = current_temp_mean
        temp_sum_reactivation = np.sum(all_y_pred[int(cue_start[i]):int(cue_start[i]) + int((behavior['framerate'] *
                                                                                             (behavior[
                                                                                                  'iti'] + 6)))]) / np.sum(
            times_considered[int(cue_start[i]):int(cue_start[i]) + int((behavior['framerate'] *
                                                                        (behavior['iti'] + 6)))]) * int(
            behavior['framerate'])
        temp_cue_activity = np.mean(activity[:, int(cue_start[i]):int(cue_end[i]) + 1])
        if normal_trials_idx[i] not in end_trials:
            reactivation_prob.append(temp_sum_reactivation)
            cue_activity.append(temp_cue_activity)

    days_path = paths['base_path'] + paths['mouse'] + '/data_across_days/'
    if days:
        if path.isfile(days_path + 'activity_no_novelty.npy') == 0 or day == 0:
            correlation_across_days = [list(range(0, days)), list(range(0, days)), list(range(0, days)),
                                       list(range(0, days)), list(range(0, days)), list(range(0, days)),
                                       list(range(0, days))]
            correlation_across_days[0][day] = correlation
            correlation_across_days[1][day] = reactivation_prob
            correlation_across_days[2][day] = cue_activity
            np.save(days_path + 'activity_no_novelty', correlation_across_days)
        else:
            correlation_across_days = np.load(days_path + 'activity_no_novelty.npy', allow_pickle=True)
            correlation_across_days[0][day] = correlation
            correlation_across_days[1][day] = reactivation_prob
            correlation_across_days[2][day] = cue_activity
            np.save(days_path + 'activity_no_novelty', correlation_across_days)


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

    # sig_cells = preprocess_opto.sig_reactivated_cells([], [], [], [], [], paths, 0)
    # activity = activity[sig_cells > 0, :]

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
    #plt.ylim(-1.7, .1)
    plt.gca().invert_yaxis()
    plt.scatter(x=list(range(0, len(mean_activity_cs_1))), y=mean_activity_cs_1, color='darkgreen', s=m_size)
    plt.scatter(x=trial_r_1, y=mean_activity_r_1, color='lime', s=m_size)
    plt.xlim(-2, 32)
    # plt.xticks([0, 19, 39, 59], ['1', '20', '40', '60'])
    plt.ylabel('Similarity to early vs. late\n S1 response pattern')
    plt.xlabel('Trial number')

    plt.subplot(2, 2, 2)
    #plt.ylim(-1.7, .1)
    plt.gca().invert_yaxis()
    plt.scatter(x=list(range(0, len(mean_activity_cs_2))), y=mean_activity_cs_2, color='darkred', s=m_size)
    plt.scatter(x=trial_r_2, y=mean_activity_r_2, color='hotpink', s=m_size)
    plt.xlim(-2, 32)
    # plt.xticks([0, 19, 39, 59], ['1', '20', '40', '60'])
    plt.ylabel('Similarity to early vs. late\n S2 response pattern')
    plt.xlabel('Trial number')

    sns.despine()

    plt.subplot(2, 2, 1)

    xx = list(range(0, len(mean_activity_cs_1)))
    yy = mean_activity_cs_1
    loess = Loess(xx, yy)
    x_plot_s1 = []
    y_plot_s1 = []
    for x in range(0, 30):
        y = loess.estimate(x, window=np.max([20, int(len(mean_activity_cs_1)/3)]), use_matrix=False, degree=1)
        x_plot_s1.append(x)
        y_plot_s1.append(y)
    plt.plot(x_plot_s1, y_plot_s1, c='darkgreen', linewidth=3)
    xx = trial_r_1
    yy = mean_activity_r_1
    loess = Loess(xx, yy)
    x_plot_s1r = []
    y_plot_s1r = []
    for x in range(0, 30):
        y = loess.estimate(x, window=np.min([np.max([20, int(len(mean_activity_r_1)/3)]), len(mean_activity_r_1)]), use_matrix=False, degree=1)
        x_plot_s1r.append(x)
        y_plot_s1r.append(y)
    if np.max(trial_r_1) < len(y_plot_s1r):
        for i in range(np.max(trial_r_1)+1, len(y_plot_s1r)):
            y_plot_s1r[i] = np.nan
    plt.plot(x_plot_s1r, y_plot_s1r, c='lime', linewidth=3)
    plt.subplot(2, 2, 2)
    xx = list(range(0, len(mean_activity_cs_2)))
    yy = mean_activity_cs_2
    loess = Loess(xx, yy)
    x_plot_s2 = []
    y_plot_s2 = []
    for x in range(0, 30):
        y = loess.estimate(x, window=np.max([20, int(len(mean_activity_cs_2)/3)]), use_matrix=False, degree=1)
        x_plot_s2.append(x)
        y_plot_s2.append(y)
    plt.plot(x_plot_s2, y_plot_s2, c='darkred', linewidth=3)
    xx = trial_r_2
    yy = mean_activity_r_2
    loess = Loess(xx, yy)
    x_plot_s2r = []
    y_plot_s2r = []
    for x in range(0, 30):
        y = loess.estimate(x, window=np.min([np.max([20, int(len(mean_activity_r_2)/3)]), len(mean_activity_r_2)]), use_matrix=False, degree=1)
        x_plot_s2r.append(x)
        y_plot_s2r.append(y)
    if np.max(trial_r_2) < len(y_plot_s2r):
        for i in range(np.max(trial_r_2) + 1, len(y_plot_s2r)):
            y_plot_s2r[i] = np.nan
    plt.plot(x_plot_s2r, y_plot_s2r, c='hotpink', linewidth=3)
    plt.show()

    days_path = paths['base_path'] + paths['mouse'] + '/data_across_days/'
    if days:
        if path.isfile(days_path + 'reactivation_cue_vector.npy') == 0 or day == 0:
            reactivation_cue_pca_vec = [list(range(0, days)), list(range(0, days)), list(range(0, days)),
                                        list(range(0, days))]
            reactivation_cue_pca_vec[0][day] = y_plot_s1
            reactivation_cue_pca_vec[1][day] = y_plot_s1r
            reactivation_cue_pca_vec[2][day] = y_plot_s2
            reactivation_cue_pca_vec[3][day] = y_plot_s2r
            np.save(days_path + 'reactivation_cue_vector', reactivation_cue_pca_vec)
        else:
            reactivation_cue_pca_vec = np.load(days_path + 'reactivation_cue_vector.npy', allow_pickle=True)
            reactivation_cue_pca_vec[0][day] = y_plot_s1
            reactivation_cue_pca_vec[1][day] = y_plot_s1r
            reactivation_cue_pca_vec[2][day] = y_plot_s2
            reactivation_cue_pca_vec[3][day] = y_plot_s2r
            np.save(days_path + 'reactivation_cue_vector', reactivation_cue_pca_vec)


def reactivation_cue_scale(norm_deconvolved, idx, y_pred, behavior, paths, day, days):
    activity = norm_deconvolved.to_numpy()
    activity = activity[idx['both'].index]

    # sig_cells = preprocess_opto.sig_reactivated_cells([], [], [], [], [], paths, 0)
    # activity = activity[sig_cells > 0, :]

    normal_trials_idx = np.where(np.isin(behavior['cue_codes'], [behavior['cs_1_code'], behavior['cs_2_code']]))[0]
    cue_start = behavior['onsets'][normal_trials_idx]
    cue_end = behavior['offsets'][normal_trials_idx]
    mean_activity_cs = []
    for i in range(0, len(cue_start)):
        temp_activity = np.mean(activity[:, int(cue_start[i]):int(cue_end[i]) + 1], axis=1)
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
                if cs_1_peak > p_threshold and int(cue_start[len(cue_start)-1]) > r_start > int(cue_start[0]):
                    temp_activity = np.mean(activity[:, r_start:r_end], axis=1)
                    reactivation_response.append(temp_activity)
                if cs_2_peak > p_threshold and int(cue_start[len(cue_start)-1]) > r_start > int(cue_start[0]):
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


def reactivation_cue_pattern_difference(norm_deconvolved, behavior, y_pred, idx, paths, day, days):
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
    # sig_cells = preprocess_opto.sig_reactivated_cells([], [], [], [], [], paths, 0)
    # activity = activity[sig_cells > 0, :]

    [no_change_cells_cs_1, increase_sig_cells_cs_1, decrease_sig_cells_cs_1] = preprocess_opto.group_neurons(activity, behavior, 'cs_1')

    [no_change_cells_cs_2, increase_sig_cells_cs_2, decrease_sig_cells_cs_2] = preprocess_opto.group_neurons(activity, behavior, 'cs_2')

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
    # sig_cells = preprocess_opto.sig_reactivated_cells([], [], [], [], [], paths, 0)
    # activity = activity[sig_cells > 0, :]

    normal_trials_idx = np.where(np.isin(behavior['cue_codes'], [behavior['cs_1_code'], behavior['cs_2_code']]))[0]
    cue_start = behavior['onsets'][normal_trials_idx]
    cue_end = behavior['offsets'][normal_trials_idx]
    cue_codes = behavior['cue_codes'][normal_trials_idx]
    end_trials = behavior['end_trials']
    for i in range(0, 10):
        if normal_trials_idx[i] not in end_trials:
            if cue_codes[i] == behavior['cs_1_code']:
                cs1c_nc.append(np.nanmean(activity[no_change_cells_cs_1 > 0, int(cue_start[i]):int(cue_end[i]) + 1]))
                cs1c_d.append(np.nanmean(activity[decrease_sig_cells_cs_1 > 0, int(cue_start[i]):int(cue_end[i]) + 1]))
                cs1c_i.append(np.nanmean(activity[increase_sig_cells_cs_1 > 0, int(cue_start[i]):int(cue_end[i]) + 1]))
            if cue_codes[i] == behavior['cs_2_code']:
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
                if cs_1_peak > p_threshold and int(behavior['onsets'][normal_trials_idx][10]) > r_start > int(behavior['onsets'][normal_trials_idx][0]):
                    reactivation_times_cs_1[r_start:r_end] = 1
                if cs_2_peak > p_threshold and int(behavior['onsets'][normal_trials_idx][10]) > r_start > int(behavior['onsets'][normal_trials_idx][0]):
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


def reactivation_cue_difference(norm_deconvolved, behavior, y_pred, idx, paths, day, days):
    activity = norm_deconvolved.to_numpy()
    activity = activity[idx['both'].index]
    # sig_cells = preprocess.sig_reactivated_cells([], [], [], [], [], paths, 0)
    # activity = activity[sig_cells > 0, :]

    [no_change_cells_cs_1, increase_sig_cells_cs_1, decrease_sig_cells_cs_1] = preprocess_opto.group_neurons(activity,
                                                                                                        behavior,
                                                                                                        'cs_1')
    [no_change_cells_cs_2, increase_sig_cells_cs_2, decrease_sig_cells_cs_2] = preprocess_opto.group_neurons(activity,
                                                                                                        behavior,
                                                                                                        'cs_2')

    index_frames_start_cs_1 = []
    index_frames_start_cs_2 = []
    normal_trials_idx = np.where(np.isin(behavior['cue_codes'], [behavior['cs_1_code'], behavior['cs_2_code']]))[0]
    cue_start = behavior['onsets'][normal_trials_idx]
    cue_codes = behavior['cue_codes'][normal_trials_idx]
    end_trials = behavior['end_trials']
    for i in range(0, 10):
        if normal_trials_idx[i] not in end_trials:
            if cue_codes[i] == behavior['cs_1_code']:
                for j in range(0, behavior['frames_before']):
                    index_frames_start_cs_1.append(int(cue_start[i]) + j)
            if cue_codes[i] == behavior['cs_2_code']:
                for j in range(0, behavior['frames_before']):
                    index_frames_start_cs_2.append(int(cue_start[i]) + j)

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
                if cs_1_peak > p_threshold and int(cue_start[10]) > r_start > int(cue_start[0]):
                    reactivation_times_cs_1[r_start:r_end] = 1
                if cs_2_peak > p_threshold and int(cue_start[10]) > r_start > int(cue_start[0]):
                    reactivation_times_cs_2[r_start:r_end] = 1
                i = r_end
                cs_1_peak = 0
                cs_2_peak = 0

    scale_factor = 1.3071117795875629  # get_reactivation_cue_scale(paths)

    diff_cs_1 = np.zeros(len(activity))
    diff_cs_2 = np.zeros(len(activity))
    for j in range(len(activity)):
        cue_1 = activity[j, index_frames_start_cs_1]
        reactivation_1 = activity[j, reactivation_times_cs_1 == 1]
        diff_cs_1[j] = ((np.mean(reactivation_1) * scale_factor) - np.mean(cue_1))
        cue_2 = activity[j, index_frames_start_cs_2]
        reactivation_2 = activity[j, reactivation_times_cs_2 == 1]
        diff_cs_2[j] = ((np.mean(reactivation_2) * scale_factor) - np.mean(cue_2))

    y1_cs_1 = np.mean(diff_cs_1[no_change_cells_cs_1 == 1])
    y2_cs_1 = np.mean(diff_cs_1[increase_sig_cells_cs_1 == 1])
    y3_cs_1 = np.mean(diff_cs_1[decrease_sig_cells_cs_1 == 1])
    y1_cs_2 = np.mean(diff_cs_2[no_change_cells_cs_2 == 1])
    y2_cs_2 = np.mean(diff_cs_2[increase_sig_cells_cs_2 == 1])
    y3_cs_2 = np.mean(diff_cs_2[decrease_sig_cells_cs_2 == 1])

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


def reactivation_cue_vector_novelty_R1(norm_deconvolved, idx, y_pred, behavior, paths, day, days):
    activity = norm_deconvolved.to_numpy()
    activity = activity[idx['both'].index]
    # sig_cells = preprocess_opto.sig_reactivated_cells([], [], [], [], [], paths, 0)
    # activity = activity[sig_cells > 0, :]

    [no_change_cells_cs_1, increase_sig_cells_cs_1, decrease_sig_cells_cs_1] = preprocess_opto.group_neurons(activity,
                                                                                                             behavior,
                                                                                                             'cs_1')

    [no_change_cells_cs_2, increase_sig_cells_cs_2, decrease_sig_cells_cs_2] = preprocess_opto.group_neurons(activity,
                                                                                                             behavior,
                                                                                                             'cs_2')

    decrease_sig_cells = decrease_sig_cells_cs_1 + decrease_sig_cells_cs_2
    decrease_sig_cells[decrease_sig_cells < 2] = 0
    decrease_sig_cells[decrease_sig_cells == 2] = 1
    no_change_decrease_cells = preprocess_opto.no_change_decrease_neurons_novelty_R1(activity, behavior,
                                                                                     decrease_sig_cells)
    activity = activity[no_change_decrease_cells == 0, :]

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

    xx = list(range(0, len(mean_activity_cs_1)))
    yy = mean_activity_cs_1
    loess = Loess(xx, yy)
    x_plot_s1 = []
    y_plot_s1 = []
    for x in range(0, 30):
        y = loess.estimate(x, window=np.max([20, int(len(mean_activity_cs_1) / 3)]), use_matrix=False, degree=1)
        x_plot_s1.append(x)
        y_plot_s1.append(y)
    xx = trial_r_1
    yy = mean_activity_r_1
    loess = Loess(xx, yy)
    x_plot_s1r = []
    y_plot_s1r = []
    for x in range(0, 30):
        y = loess.estimate(x, window=np.min([np.max([20, int(len(mean_activity_r_1) / 3)]), len(mean_activity_r_1)]),
                           use_matrix=False, degree=1)
        x_plot_s1r.append(x)
        y_plot_s1r.append(y)
    if np.max(trial_r_1) < len(y_plot_s1r):
        for i in range(np.max(trial_r_1) + 1, len(y_plot_s1r)):
            y_plot_s1r[i] = np.nan
    xx = list(range(0, len(mean_activity_cs_2)))
    yy = mean_activity_cs_2
    loess = Loess(xx, yy)
    x_plot_s2 = []
    y_plot_s2 = []
    for x in range(0, 30):
        y = loess.estimate(x, window=np.max([20, int(len(mean_activity_cs_2) / 3)]), use_matrix=False, degree=1)
        x_plot_s2.append(x)
        y_plot_s2.append(y)
    xx = trial_r_2
    yy = mean_activity_r_2
    loess = Loess(xx, yy)
    x_plot_s2r = []
    y_plot_s2r = []
    for x in range(0, 30):
        y = loess.estimate(x, window=np.min([np.max([20, int(len(mean_activity_r_2) / 3)]), len(mean_activity_r_2)]),
                           use_matrix=False, degree=1)
        x_plot_s2r.append(x)
        y_plot_s2r.append(y)
    if np.max(trial_r_2) < len(y_plot_s2r):
        for i in range(np.max(trial_r_2) + 1, len(y_plot_s2r)):
            y_plot_s2r[i] = np.nan

    days_path = paths['base_path'] + paths['mouse'] + '/data_across_days/'
    if days:
        if path.isfile(days_path + 'reactivation_cue_vector_novelty.npy') == 0 or day == 0:
            reactivation_cue_pca_vec = [list(range(0, days)), list(range(0, days)), list(range(0, days)),
                                        list(range(0, days))]
            reactivation_cue_pca_vec[0][day] = y_plot_s1
            reactivation_cue_pca_vec[1][day] = y_plot_s1r
            reactivation_cue_pca_vec[2][day] = y_plot_s2
            reactivation_cue_pca_vec[3][day] = y_plot_s2r
            np.save(days_path + 'reactivation_cue_vector_novelty', reactivation_cue_pca_vec)
        else:
            reactivation_cue_pca_vec = np.load(days_path + 'reactivation_cue_vector_novelty.npy', allow_pickle=True)
            reactivation_cue_pca_vec[0][day] = y_plot_s1
            reactivation_cue_pca_vec[1][day] = y_plot_s1r
            reactivation_cue_pca_vec[2][day] = y_plot_s2
            reactivation_cue_pca_vec[3][day] = y_plot_s2r
            np.save(days_path + 'reactivation_cue_vector_novelty', reactivation_cue_pca_vec)


def reactivation_cue_vector_layer_R1(norm_deconvolved, upper_layer_cells, lower_layer_cells, idx, y_pred, behavior,
                                     paths, day, days):
    # sig_cells = preprocess_opto.sig_reactivated_cells([], [], [], [], [], paths, 0)
    idx_both = np.array(idx['both'].index.copy())
    # idx_both[sig_cells == 0] = -1

    upper_layer_idx = np.intersect1d(np.array(idx_both), upper_layer_cells).astype(int)
    lower_layer_idx = np.intersect1d(np.array(idx_both), lower_layer_cells).astype(int)

    activity = norm_deconvolved.to_numpy()
    activity_upper = activity[upper_layer_idx]
    activity_lower = activity[lower_layer_idx]

    [y_plot_s1_u, y_plot_s1r_u, y_plot_s2_u, y_plot_s2r_u] = reactivation_cue_vector_layer_R1_helper(activity_upper, y_pred, behavior)
    [y_plot_s1_l, y_plot_s1r_l, y_plot_s2_l, y_plot_s2r_l] = reactivation_cue_vector_layer_R1_helper(activity_lower, y_pred, behavior)

    days_path = paths['base_path'] + paths['mouse'] + '/data_across_days/'
    if days:
        if path.isfile(days_path + 'reactivation_cue_vector_layer.npy') == 0 or day == 0:
            reactivation_cue_pca_vec = [list(range(0, days)), list(range(0, days)), list(range(0, days)),
                                        list(range(0, days)), list(range(0, days)), list(range(0, days)),
                                        list(range(0, days)), list(range(0, days)), list(range(0, days)),
                                        list(range(0, days)), list(range(0, days)), list(range(0, days))]
            reactivation_cue_pca_vec[0][day] = y_plot_s1_u
            reactivation_cue_pca_vec[1][day] = y_plot_s1r_u
            reactivation_cue_pca_vec[2][day] = y_plot_s2_u
            reactivation_cue_pca_vec[3][day] = y_plot_s2r_u
            reactivation_cue_pca_vec[6][day] = y_plot_s1_l
            reactivation_cue_pca_vec[7][day] = y_plot_s1r_l
            reactivation_cue_pca_vec[8][day] = y_plot_s2_l
            reactivation_cue_pca_vec[9][day] = y_plot_s2r_l
            np.save(days_path + 'reactivation_cue_vector_layer', reactivation_cue_pca_vec)
        else:
            reactivation_cue_pca_vec = np.load(days_path + 'reactivation_cue_vector_layer.npy', allow_pickle=True)
            reactivation_cue_pca_vec[0][day] = y_plot_s1_u
            reactivation_cue_pca_vec[1][day] = y_plot_s1r_u
            reactivation_cue_pca_vec[2][day] = y_plot_s2_u
            reactivation_cue_pca_vec[3][day] = y_plot_s2r_u
            reactivation_cue_pca_vec[6][day] = y_plot_s1_l
            reactivation_cue_pca_vec[7][day] = y_plot_s1r_l
            reactivation_cue_pca_vec[8][day] = y_plot_s2_l
            reactivation_cue_pca_vec[9][day] = y_plot_s2r_l
            np.save(days_path + 'reactivation_cue_vector_layer', reactivation_cue_pca_vec)


def reactivation_cue_vector_layer_R1_helper(activity, y_pred, behavior):
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
                               np.mean(mean_activity_cs_1[len(mean_activity_cs_1) - 3:len(mean_activity_cs_1)], axis=0)]
    mean_activity_cs_2_mean = [np.mean(mean_activity_cs_2[0:3], axis=0),
                               np.mean(mean_activity_cs_2[len(mean_activity_cs_2) - 3:len(mean_activity_cs_2)], axis=0)]

    mean_activity_cs_1_vec = mean_activity_cs_1_mean[1] - mean_activity_cs_1_mean[0]
    mean_activity_cs_2_vec = mean_activity_cs_2_mean[1] - mean_activity_cs_2_mean[0]

    for i in range(0, len(mean_activity_cs_1)):
        mean_activity_cs_1[i] = np.dot(mean_activity_cs_1[i], mean_activity_cs_1_vec) / np.linalg.norm(
            mean_activity_cs_1_vec)
    for i in range(0, len(mean_activity_cs_2)):
        mean_activity_cs_2[i] = np.dot(mean_activity_cs_2[i], mean_activity_cs_2_vec) / np.linalg.norm(
            mean_activity_cs_2_vec)

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
                        if r_start < onsets_cs_1[j] and r_start > onsets_cs_1[j - 1] and r_start < onsets_cs_1[
                            j - 1] + int(behavior['framerate'] * 61):
                            mean_activity_r_1.append(np.mean(activity[:, r_start:r_end], axis=1))
                            trial_r_1.append(j - 1)
                            break
                if cs_2_peak > p_threshold:
                    for j in range(0, len(onsets_cs_2)):
                        if r_start < onsets_cs_2[j] and r_start > onsets_cs_2[j - 1] and r_start < onsets_cs_2[
                            j - 1] + int(behavior['framerate'] * 61):
                            mean_activity_r_2.append(np.mean(activity[:, r_start:r_end], axis=1))
                            trial_r_2.append(j - 1)
                            break
                i = r_end
                cs_1_peak = 0
                cs_2_peak = 0

    for i in range(0, len(mean_activity_r_1)):
        mean_activity_r_1[i] = np.dot(mean_activity_r_1[i], mean_activity_cs_1_vec) / np.linalg.norm(
            mean_activity_cs_1_vec)
    for i in range(0, len(mean_activity_r_2)):
        mean_activity_r_2[i] = np.dot(mean_activity_r_2[i], mean_activity_cs_2_vec) / np.linalg.norm(
            mean_activity_cs_2_vec)

    xx = list(range(0, len(mean_activity_cs_1)))
    yy = mean_activity_cs_1
    loess = Loess(xx, yy)
    x_plot_s1 = []
    y_plot_s1 = []
    for x in range(0, 30):
        y = loess.estimate(x, window=np.max([20, int(len(mean_activity_cs_1) / 3)]), use_matrix=False, degree=1)
        x_plot_s1.append(x)
        y_plot_s1.append(y)
    xx = trial_r_1
    yy = mean_activity_r_1
    loess = Loess(xx, yy)
    x_plot_s1r = []
    y_plot_s1r = []
    for x in range(0, 30):
        y = loess.estimate(x, window=np.min([np.max([20, int(len(mean_activity_r_1) / 3)]), len(mean_activity_r_1)]),
                           use_matrix=False, degree=1)
        x_plot_s1r.append(x)
        y_plot_s1r.append(y)
    if np.max(trial_r_1) < len(y_plot_s1r):
        for i in range(np.max(trial_r_1) + 1, len(y_plot_s1r)):
            y_plot_s1r[i] = np.nan
    xx = list(range(0, len(mean_activity_cs_2)))
    yy = mean_activity_cs_2
    loess = Loess(xx, yy)
    x_plot_s2 = []
    y_plot_s2 = []
    for x in range(0, 30):
        y = loess.estimate(x, window=np.max([20, int(len(mean_activity_cs_2) / 3)]), use_matrix=False, degree=1)
        x_plot_s2.append(x)
        y_plot_s2.append(y)
    xx = trial_r_2
    yy = mean_activity_r_2
    loess = Loess(xx, yy)
    x_plot_s2r = []
    y_plot_s2r = []
    for x in range(0, 30):
        y = loess.estimate(x, window=np.min([np.max([20, int(len(mean_activity_r_2) / 3)]), len(mean_activity_r_2)]),
                           use_matrix=False, degree=1)
        x_plot_s2r.append(x)
        y_plot_s2r.append(y)
    if np.max(trial_r_2) < len(y_plot_s2r):
        for i in range(np.max(trial_r_2) + 1, len(y_plot_s2r)):
            y_plot_s2r[i] = np.nan

    return [y_plot_s1, y_plot_s1r, y_plot_s2, y_plot_s2r]


def reactivation_cue_vector_evolve(norm_deconvolved, idx, y_pred, behavior, return_s, p_value, paths, day, days):
    activity = norm_deconvolved.to_numpy()
    activity = activity[idx['both'].index]

    # sig_cells = preprocess_opto.sig_reactivated_cells([], [], [], [], [], paths, 0)
    # activity = activity[sig_cells > 0, :]

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
                    r_start_cs_1[num_r_cs_1] = r_start
                    r_end_cs_1[num_r_cs_1] = r_end
                    num_r_cs_1 += 1
                if cs_2_peak > p_threshold:
                    r_start_cs_2[num_r_cs_2] = r_start
                    r_end_cs_2[num_r_cs_2] = r_end
                    num_r_cs_2 += 1
                i = r_end
                cs_1_peak = 0
                cs_2_peak = 0
    r_start_cs_1 = r_start_cs_1[~np.isnan(r_start_cs_1)]
    r_start_cs_2 = r_start_cs_2[~np.isnan(r_start_cs_2)]
    r_end_cs_1 = r_end_cs_1[~np.isnan(r_end_cs_1)]
    r_end_cs_2 = r_end_cs_2[~np.isnan(r_end_cs_2)]

    xx = list(range(0, len(mean_activity_cs_1)))
    yy = mean_activity_cs_1
    loess = Loess(xx, yy)
    x_plot_s1 = []
    y_plot_s1_t = []
    for x in range(0, 30):
        y = loess.estimate(x, window=np.max([20, int(len(mean_activity_cs_1)/3)]), use_matrix=False, degree=1)
        x_plot_s1.append(x)
        y_plot_s1_t.append(y)
    xx = list(range(0, len(mean_activity_cs_2)))
    yy = mean_activity_cs_2
    loess = Loess(xx, yy)
    x_plot_s2 = []
    y_plot_s2_t = []
    for x in range(0, 30):
        y = loess.estimate(x, window=np.max([20, int(len(mean_activity_cs_2)/3)]), use_matrix=False, degree=1)
        x_plot_s2.append(x)
        y_plot_s2_t.append(y)

    if return_s != 2:
        sns.set(font_scale=1)
        sns.set_style("whitegrid", {'axes.grid': False})
        sns.set_style("ticks")
        plt.figure(figsize=(9, 6))
        plt.subplots_adjust(wspace=.3)
        m_size = 10
        plt.subplot(2, 2, 1)
        # plt.ylim(-2.1, .1)
        #plt.gca().invert_yaxis()
        plt.scatter(x=list(range(0, len(mean_activity_cs_1))), y=mean_activity_cs_1, color='darkgreen', s=m_size)
        plt.xlim(-1, 30)
        plt.xticks([0, 29], ['1', '30'])
        plt.ylabel('Similarity to early vs. late\n S1 response pattern')
        plt.xlabel('Trial number')
        plt.subplot(2, 2, 2)
        # plt.ylim(-2.1, .1)
        #plt.gca().invert_yaxis()
        plt.scatter(x=list(range(0, len(mean_activity_cs_2))), y=mean_activity_cs_2, color='darkred', s=m_size)
        plt.xlim(-1, 30)
        plt.xticks([0, 29], ['1', '30'])
        plt.ylabel('Similarity to early vs. late\n S2 response pattern')
        plt.xlabel('Trial number')
        plt.subplot(2, 2, 1)
        plt.plot(x_plot_s1, y_plot_s1_t, c='darkgreen', linewidth=3)
        plt.subplot(2, 2, 2)
        plt.plot(x_plot_s2, y_plot_s2_t, c='darkred', linewidth=3)
        sns.despine()

    scale_factor = 1.3071117795875629 # get_reactivation_cue_scale(paths)
    p_value = .2
    for i in range(0, len(mean_activity_cs_1)-1):
        curr_cs = mean_activity_cs_1_rt[i].copy()
        for j in range(0, len(r_start_cs_1)):
            if onsets_cs_1[i] < r_start_cs_1[j] < onsets_cs_1[i] + int(behavior['framerate']*61):
                mean_r = np.mean(activity[:, int(r_start_cs_1[j]):int(r_end_cs_1[j])], axis=1)
                curr_cs = curr_cs + p_value*((mean_r*scale_factor)-curr_cs)
        mean_activity_cs_1_rt.append(curr_cs)

    for i in range(0, len(mean_activity_cs_2)-1):
        curr_cs = mean_activity_cs_2_rt[i].copy()
        for j in range(0, len(r_start_cs_2)):
            if onsets_cs_2[i] < r_start_cs_2[j] < onsets_cs_2[i] + int(behavior['framerate']*61):
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

    xx = list(range(0, len(mean_activity_cs_1_rt)))
    yy = mean_activity_cs_1_rt
    loess = Loess(xx, yy)
    x_plot_s1 = []
    y_plot_s1 = []
    for x in range(0, 30):
        y = loess.estimate(x, window=np.max([20, int(len(mean_activity_cs_1_rt)/3)]), use_matrix=False, degree=1)
        x_plot_s1.append(x)
        y_plot_s1.append(y)
    xx = list(range(0, len(mean_activity_cs_2_rt)))
    yy = mean_activity_cs_2_rt
    loess = Loess(xx, yy)
    x_plot_s2 = []
    y_plot_s2 = []
    for x in range(0, 30):
        y = loess.estimate(x, window=np.max([20, int(len(mean_activity_cs_2_rt)/3)]), use_matrix=False, degree=1)
        x_plot_s2.append(x)
        y_plot_s2.append(y)

    if return_s != 2:
        plt.subplot(2, 2, 1)
        # plt.ylim(-2.1, .1)
        plt.gca().invert_yaxis()
        plt.scatter(x=list(range(0, len(mean_activity_cs_1_rt))), y=mean_activity_cs_1_rt, color='lime', s=m_size)
        plt.xlim(-1, 30)
        plt.xticks([0, 29], ['1', '30'])
        plt.ylabel('Similarity to early vs. late\n S1 response pattern')
        plt.xlabel('Trial number')
        plt.subplot(2, 2, 2)
        # plt.ylim(-2.1, .1)
        plt.gca().invert_yaxis()
        plt.scatter(x=list(range(0, len(mean_activity_cs_2_rt))), y=mean_activity_cs_2_rt, color='hotpink', s=m_size)
        plt.xlim(-1, 30)
        plt.xticks([0, 29], ['1', '30'])
        plt.ylabel('Similarity to early vs. late\n S2 response pattern')
        plt.xlabel('Trial number')
        plt.subplot(2, 2, 1)
        plt.plot(x_plot_s1, y_plot_s1, c='lime', linewidth=3)
        plt.subplot(2, 2, 2)
        plt.plot(x_plot_s2, y_plot_s2, c='hotpink', linewidth=3)
        sns.despine()
        plt.savefig(paths['save_path'] + 'plots/' + paths['mouse'] + '_' + paths['date'] + '_' +
                    'reactivation_cue_vector_evolve.png', bbox_inches='tight', dpi=200)
        plt.savefig(paths['save_path'] + 'plots/' + paths['mouse'] + '_' + paths['date'] + '_' +
                    'reactivation_cue_vector_evolve.pdf', bbox_inches='tight', dpi=200, transparent=True)
        plt.show()

    if return_s == 2:
        return [y_plot_s1, y_plot_s2, y_plot_s1_t, y_plot_s2_t]

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


def reactivation_cue_vector_evolve_low_reactivation(norm_deconvolved, idx, y_pred, behavior, return_s, p_value, paths, day, days):
    activity = norm_deconvolved.to_numpy()
    activity = activity[idx['both'].index]

    sig_cells = preprocess_opto.sig_reactivated_cells([], [], [], [], [], paths, 0)
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
                    r_start_cs_1[num_r_cs_1] = r_start
                    r_end_cs_1[num_r_cs_1] = r_end
                    num_r_cs_1 += 1
                if cs_2_peak > p_threshold:
                    r_start_cs_2[num_r_cs_2] = r_start
                    r_end_cs_2[num_r_cs_2] = r_end
                    num_r_cs_2 += 1
                i = r_end
                cs_1_peak = 0
                cs_2_peak = 0
    r_start_cs_1 = r_start_cs_1[~np.isnan(r_start_cs_1)]
    r_start_cs_2 = r_start_cs_2[~np.isnan(r_start_cs_2)]
    r_end_cs_1 = r_end_cs_1[~np.isnan(r_end_cs_1)]
    r_end_cs_2 = r_end_cs_2[~np.isnan(r_end_cs_2)]

    xx = list(range(0, len(mean_activity_cs_1)))
    yy = mean_activity_cs_1
    loess = Loess(xx, yy)
    x_plot_s1 = []
    y_plot_s1_t = []
    for x in range(0, 30):
        y = loess.estimate(x, window=np.max([20, int(len(mean_activity_cs_1)/3)]), use_matrix=False, degree=1)
        x_plot_s1.append(x)
        y_plot_s1_t.append(y)
    xx = list(range(0, len(mean_activity_cs_2)))
    yy = mean_activity_cs_2
    loess = Loess(xx, yy)
    x_plot_s2 = []
    y_plot_s2_t = []
    for x in range(0, 30):
        y = loess.estimate(x, window=np.max([20, int(len(mean_activity_cs_2)/3)]), use_matrix=False, degree=1)
        x_plot_s2.append(x)
        y_plot_s2_t.append(y)

    scale_factor = 1.3612373436223617 # get_reactivation_cue_scale(paths)
    p_value = .2

    inc = 0
    for i in range(0, len(mean_activity_cs_1) - 1):
        curr_cs = mean_activity_cs_1_rt[i].copy()
        for j in range(0, len(r_start_cs_1)):
            if onsets_cs_1[i] < r_start_cs_1[j] < onsets_cs_1[i] + int(behavior['framerate'] * 61):
                if inc % 4 == 3:
                    mean_r = np.mean(activity[:, int(r_start_cs_1[j]):int(r_end_cs_1[j])], axis=1)
                    curr_cs = curr_cs + p_value * ((mean_r * scale_factor) - curr_cs)
                inc += 1
        mean_activity_cs_1_rt.append(curr_cs)

    inc = 0
    for i in range(0, len(mean_activity_cs_2) - 1):
        curr_cs = mean_activity_cs_2_rt[i].copy()
        for j in range(0, len(r_start_cs_2)):
            if onsets_cs_2[i] < r_start_cs_2[j] < onsets_cs_2[i] + + int(behavior['framerate'] * 61):
                if inc % 4 == 3:
                    mean_r = np.mean(activity[:, int(r_start_cs_2[j]):int(r_end_cs_2[j])], axis=1)
                    curr_cs = curr_cs + p_value * ((mean_r * scale_factor) - curr_cs)
                inc += 1
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

    xx = list(range(0, len(mean_activity_cs_1_rt)))
    yy = mean_activity_cs_1_rt
    loess = Loess(xx, yy)
    x_plot_s1 = []
    y_plot_s1 = []
    for x in range(0, 30):
        y = loess.estimate(x, window=np.max([20, int(len(mean_activity_cs_1_rt)/3)]), use_matrix=False, degree=1)
        x_plot_s1.append(x)
        y_plot_s1.append(y)
    xx = list(range(0, len(mean_activity_cs_2_rt)))
    yy = mean_activity_cs_2_rt
    loess = Loess(xx, yy)
    x_plot_s2 = []
    y_plot_s2 = []
    for x in range(0, 30):
        y = loess.estimate(x, window=np.max([20, int(len(mean_activity_cs_2_rt)/3)]), use_matrix=False, degree=1)
        x_plot_s2.append(x)
        y_plot_s2.append(y)

    if return_s == 2:
        return [y_plot_s1, y_plot_s2, y_plot_s1_t, y_plot_s2_t]

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

    normal_trials_idx = np.where(np.isin(behavior['cue_codes'], [behavior['cs_1_code'], behavior['cs_2_code']]))[0]
    cue_start = behavior['onsets'][normal_trials_idx]
    cue_codes = behavior['cue_codes'][normal_trials_idx]
    end_trials = behavior['end_trials']
    correlation = []
    cue_activity = []

    start = max(list(cue_codes).index(behavior['cs_1_code']), list(cue_codes).index(behavior['cs_2_code']))
    past_cs_1_type = cue_codes[start - 1]
    past_cs_2_type = cue_codes[start]
    past_cs_1_mean = activity[start - 1]
    past_cs_2_mean = activity[start]
    for i in range(start+1, len(cue_start)):
        current_cs_type = cue_codes[i]
        if current_cs_type == past_cs_1_type:
            current_temp_mean = activity[i]
            if normal_trials_idx[i] not in end_trials:
                correlation.append(np.corrcoef(current_temp_mean, past_cs_2_mean)[0][1])
            past_cs_1_mean = current_temp_mean
        if current_cs_type == past_cs_2_type:
            current_temp_mean = activity[i]
            if normal_trials_idx[i] not in end_trials:
                correlation.append(np.corrcoef(current_temp_mean, past_cs_1_mean)[0][1])
            past_cs_2_mean = current_temp_mean

        temp_cue_activity = np.mean(activity[i])
        if normal_trials_idx[i] not in end_trials:
            cue_activity.append(temp_cue_activity)

    days_path = paths['base_path'] + paths['mouse'] + '/data_across_days/'
    if days:
        if path.isfile(days_path + 'activity_evolve.npy') == 0 or day == 0:
            correlation_across_days = [list(range(0, days)), list(range(0, days)), list(range(0, days)),
                                       list(range(0, days)), list(range(0, days)), list(range(0, days)),
                                       list(range(0, days))]
            correlation_across_days[0][day] = correlation
            correlation_across_days[1][day] = cue_activity
            np.save(days_path + 'activity_evolve', correlation_across_days)
        else:
            correlation_across_days = np.load(days_path + 'activity_evolve.npy', allow_pickle=True)
            correlation_across_days[0][day] = correlation
            correlation_across_days[1][day] = cue_activity
            np.save(days_path + 'activity_evolve', correlation_across_days)


def activity_across_trials_evolve_low_reactivation(norm_deconvolved, behavior, y_pred, idx, paths, day, days):
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
    activity = reactivation_cue_vector_evolve_low_reactivation(norm_deconvolved, idx, y_pred, behavior, 1, [], paths, day, days)

    normal_trials_idx = np.where(np.isin(behavior['cue_codes'], [behavior['cs_1_code'], behavior['cs_2_code']]))[0]
    cue_start = behavior['onsets'][normal_trials_idx]
    cue_codes = behavior['cue_codes'][normal_trials_idx]
    end_trials = behavior['end_trials']
    correlation = []
    cue_activity = []

    start = max(list(cue_codes).index(behavior['cs_1_code']), list(cue_codes).index(behavior['cs_2_code']))
    past_cs_1_type = cue_codes[start - 1]
    past_cs_2_type = cue_codes[start]
    past_cs_1_mean = activity[start - 1]
    past_cs_2_mean = activity[start]
    for i in range(start+1, len(cue_start)):
        current_cs_type = cue_codes[i]
        if current_cs_type == past_cs_1_type:
            current_temp_mean = activity[i]
            if normal_trials_idx[i] not in end_trials:
                correlation.append(np.corrcoef(current_temp_mean, past_cs_2_mean)[0][1])
            past_cs_1_mean = current_temp_mean
        if current_cs_type == past_cs_2_type:
            current_temp_mean = activity[i]
            if normal_trials_idx[i] not in end_trials:
                correlation.append(np.corrcoef(current_temp_mean, past_cs_1_mean)[0][1])
            past_cs_2_mean = current_temp_mean

        temp_cue_activity = np.mean(activity[i])
        if normal_trials_idx[i] not in end_trials:
            cue_activity.append(temp_cue_activity)

    days_path = paths['base_path'] + paths['mouse'] + '/data_across_days/'
    if days:
        if path.isfile(days_path + 'activity_evolve_low_reactivation.npy') == 0 or day == 0:
            correlation_across_days = [list(range(0, days)), list(range(0, days)), list(range(0, days)),
                                       list(range(0, days)), list(range(0, days)), list(range(0, days)),
                                       list(range(0, days))]
            correlation_across_days[0][day] = correlation
            correlation_across_days[1][day] = cue_activity
            np.save(days_path + 'activity_evolve_low_reactivation', correlation_across_days)
        else:
            correlation_across_days = np.load(days_path + 'activity_evolve_low_reactivation.npy', allow_pickle=True)
            correlation_across_days[0][day] = correlation
            correlation_across_days[1][day] = cue_activity
            np.save(days_path + 'activity_evolve_low_reactivation', correlation_across_days)


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
    # sig_cells = preprocess_opto.sig_reactivated_cells([], [], [], [], [], paths, 0)
    # activity = activity[sig_cells > 0, :]

    [no_change_cells_cs_1, increase_sig_cells_cs_1, decrease_sig_cells_cs_1] = preprocess_opto.group_neurons(activity, behavior, 'cs_1')

    [no_change_cells_cs_2, increase_sig_cells_cs_2, decrease_sig_cells_cs_2] = preprocess_opto.group_neurons(activity, behavior, 'cs_2')

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
            # activity = activity[cells_to_use > 0, :

        normal_trials_idx = np.where(np.isin(behavior['cue_codes'], [behavior['cs_1_code'], behavior['cs_2_code']]))[0]
        cue_start = behavior['onsets'][normal_trials_idx]
        cue_codes = behavior['cue_codes'][normal_trials_idx]
        end_trials = behavior['end_trials']
        correlation = []
        cue_activity = []

        start = max(list(cue_codes).index(behavior['cs_1_code']), list(cue_codes).index(behavior['cs_2_code']))
        past_cs_1_type = cue_codes[start - 1]
        past_cs_2_type = cue_codes[start]
        if g == 0:
            past_cs_1_mean = activity[start - 1][cells_to_use == 2]
            past_cs_2_mean = activity[start][cells_to_use == 2]
        if g == 1 or g == 2:
            past_cs_1_mean = activity[start - 1][cells_to_use > 0]
            past_cs_2_mean = activity[start][cells_to_use > 0]
        for i in range(start + 1, len(cue_start)):
            current_cs_type = cue_codes[i]
            if current_cs_type == past_cs_1_type:
                if g == 0:
                    current_temp_mean = activity[i][cells_to_use == 2]
                if g == 1 or g == 2:
                    current_temp_mean = activity[i][cells_to_use > 0]
                if normal_trials_idx[i] not in end_trials:
                    correlation.append(np.corrcoef(current_temp_mean, past_cs_2_mean)[0][1])
                past_cs_1_mean = current_temp_mean
            if current_cs_type == past_cs_2_type:
                if g == 0:
                    current_temp_mean = activity[i][cells_to_use == 2]
                if g == 1 or g == 2:
                    current_temp_mean = activity[i][cells_to_use > 0]
                if normal_trials_idx[i] not in end_trials:
                    correlation.append(np.corrcoef(current_temp_mean, past_cs_1_mean)[0][1])
                past_cs_2_mean = current_temp_mean
            if g == 0:
                temp_cue_activity = np.mean(activity[i][cells_to_use == 2])
            if g == 1 or g == 2:
                temp_cue_activity = np.mean(activity[i][cells_to_use > 0])
            if normal_trials_idx[i] not in end_trials:
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


def activity_across_trials_evolve_grouped_separate(norm_deconvolved, behavior, y_pred, idx, paths, day, days):
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
    # sig_cells = preprocess_opto.sig_reactivated_cells([], [], [], [], [], paths, 0)
    # activity = activity[sig_cells > 0, :]

    [no_change_cells_cs_1, increase_sig_cells_cs_1, decrease_sig_cells_cs_1] = preprocess_opto.group_neurons(activity, behavior, 'cs_1')

    [no_change_cells_cs_2, increase_sig_cells_cs_2, decrease_sig_cells_cs_2] = preprocess_opto.group_neurons(activity, behavior, 'cs_2')

    activity = reactivation_cue_vector_evolve(norm_deconvolved, idx, y_pred, behavior, 1, [], paths, day, days)

    normal_trials_idx = np.where(np.isin(behavior['cue_codes'], [behavior['cs_1_code'], behavior['cs_2_code']]))[0]
    cue_start = behavior['onsets'][normal_trials_idx]
    cue_codes = behavior['cue_codes'][normal_trials_idx]
    end_trials = behavior['end_trials']
    start = max(list(cue_codes).index(behavior['cs_1_code']), list(cue_codes).index(behavior['cs_2_code']))

    cs1d_cs1 = []
    cs1d_cs2 = []
    cs2d_cs1 = []
    cs2d_cs2 = []
    cells_to_use_1 = decrease_sig_cells_cs_1
    cells_to_use_2 = decrease_sig_cells_cs_2
    for i in range(start+1, len(cue_start)):
        if normal_trials_idx[i] not in end_trials:
            if cue_codes[i] == behavior['cs_1_code']:
                cs1d_cs1.append(np.mean(activity[i][cells_to_use_1 > 0]))
                cs2d_cs1.append(np.mean(activity[i][cells_to_use_2 > 0]))
            if cue_codes[i] == behavior['cs_2_code']:
                cs1d_cs2.append(np.mean(activity[i][cells_to_use_1 > 0]))
                cs2d_cs2.append(np.mean(activity[i][cells_to_use_2 > 0]))
    days_path = paths['base_path'] + paths['mouse'] + '/data_across_days/'
    if days:
        if path.isfile(days_path + 'activity_grouped_decrease_evolve.npy') == 0 or day == 0:
            across_days = [list(range(0, days)), list(range(0, days)), list(range(0, days)),
                           list(range(0, days))]
            across_days[0][day] = cs1d_cs1
            across_days[1][day] = cs1d_cs2
            across_days[2][day] = cs2d_cs2
            across_days[3][day] = cs2d_cs1
            np.save(days_path + 'activity_grouped_decrease_evolve', across_days)
        else:
            across_days = np.load(days_path + 'activity_grouped_decrease_evolve.npy', allow_pickle=True)
            across_days[0][day] = cs1d_cs1
            across_days[1][day] = cs1d_cs2
            across_days[2][day] = cs2d_cs2
            across_days[3][day] = cs2d_cs1
            np.save(days_path + 'activity_grouped_decrease_evolve', across_days)

    cs1d_cs1 = []
    cs1d_cs2 = []
    cs2d_cs1 = []
    cs2d_cs2 = []
    cells_to_use_1 = increase_sig_cells_cs_1
    cells_to_use_2 = increase_sig_cells_cs_2
    for i in range(start + 1, len(cue_start)):
        if normal_trials_idx[i] not in end_trials:
            if cue_codes[i] == behavior['cs_1_code']:
                cs1d_cs1.append(np.mean(activity[i][cells_to_use_1 > 0]))
                cs2d_cs1.append(np.mean(activity[i][cells_to_use_2 > 0]))
            if cue_codes[i] == behavior['cs_2_code']:
                cs1d_cs2.append(np.mean(activity[i][cells_to_use_1 > 0]))
                cs2d_cs2.append(np.mean(activity[i][cells_to_use_2 > 0]))
    days_path = paths['base_path'] + paths['mouse'] + '/data_across_days/'
    if days:
        if path.isfile(days_path + 'activity_grouped_increase_evolve.npy') == 0 or day == 0:
            across_days = [list(range(0, days)), list(range(0, days)), list(range(0, days)),
                           list(range(0, days))]
            across_days[0][day] = cs1d_cs1
            across_days[1][day] = cs1d_cs2
            across_days[2][day] = cs2d_cs2
            across_days[3][day] = cs2d_cs1
            np.save(days_path + 'activity_grouped_increase_evolve', across_days)
        else:
            across_days = np.load(days_path + 'activity_grouped_increase_evolve.npy', allow_pickle=True)
            across_days[0][day] = cs1d_cs1
            across_days[1][day] = cs1d_cs2
            across_days[2][day] = cs2d_cs2
            across_days[3][day] = cs2d_cs1
            np.save(days_path + 'activity_grouped_increase_evolve', across_days)

    cs1d_cs1 = []
    cs1d_cs2 = []
    cs2d_cs1 = []
    cs2d_cs2 = []
    cells_to_use_1 = no_change_cells_cs_1
    cells_to_use_2 = no_change_cells_cs_2
    for i in range(start + 1, len(cue_start)):
        if normal_trials_idx[i] not in end_trials:
            if cue_codes[i] == behavior['cs_1_code']:
                cs1d_cs1.append(np.mean(activity[i][cells_to_use_1 == 2]))
                cs2d_cs1.append(np.mean(activity[i][cells_to_use_2 == 2]))
            if cue_codes[i] == behavior['cs_2_code']:
                cs1d_cs2.append(np.mean(activity[i][cells_to_use_1 == 2]))
                cs2d_cs2.append(np.mean(activity[i][cells_to_use_2 == 2]))
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


def prior_R1(norm_deconvolved, norm_moving_deconvolved_filtered, behavior, y_pred, idx, paths):
    """
    get cue prior
    :param norm_moving_deconvolved_filtered: processed activity
    :param cs_1_idx: cs 1 index
    :param cs_2_idx: cs 2 index
    :param behavior: behavior
    :param threshold: threshold
    :return: prior
    """

    norm_deconvolved = norm_deconvolved.to_numpy()
    y_pred = true_reactivations(y_pred)
    y_pred_all = y_pred[:, 0] + y_pred[:, 1]
    y_pred_all = y_pred_all[100:behavior['frames_per_run']]
    threshold = 1
    prior_vec = norm_moving_deconvolved_filtered
    prior_vec[prior_vec < 0] = 0
    prior_vec = pd.DataFrame(prior_vec[:, 100:behavior['frames_per_run']])
    prior_vec = prior_vec.mean()
    prior_vec = (prior_vec - prior_vec.mean()) / prior_vec.std()
    prior_vec[prior_vec < threshold] = 0
    prior_vec[prior_vec > 0] = 1
    cluster_vec = np.empty((1000, len(norm_deconvolved))) * np.nan
    i = 0
    next_s = 0
    num = 0
    while i < len(prior_vec)-1:
        if prior_vec[i] > 0 and y_pred_all[i] == 0:
            if next_s == 0:
                s_start = i
                next_s = 1
            if prior_vec[i + 1] == 0 or y_pred_all[i+1] > 0:
                s_end = i + 1
                next_s = 0
                cluster_vec[num, :] = np.mean(norm_deconvolved[:, 100+s_start:100+s_end], axis=1)
                num += 1
                i = s_end
        i += 1
    cluster_vec = cluster_vec[0:num, :]
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(22, 5))
    Z = scipy.cluster.hierarchy.linkage(cluster_vec, 'single')
    R = scipy.cluster.hierarchy.dendrogram(Z, no_plot=True)
    cluster_vec = cluster_vec[R['leaves'], :]
    plt.subplot(1, 3, 1)
    corr_matrix = pd.DataFrame(cluster_vec).transpose().corr()
    sns.heatmap(corr_matrix, vmin=0, vmax=.3, cmap='Greys')
    plt.ylabel('Clustered synchronous\nevent number')
    plt.xlabel('Clustered synchronous\nevent number')

    reactivation_cs_1 = y_pred[:, 0].copy()
    reactivation_cs_2 = y_pred[:, 1].copy()
    p_threshold = .75
    cs_1_peak = 0
    cs_2_peak = 0
    i = 0
    next_r = 0
    cluster_vec = np.empty((1000, len(norm_deconvolved))) * np.nan
    num = 0
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
                if cs_1_peak > p_threshold and r_start > behavior['frames_per_run']:
                    cluster_vec[num, :] = np.mean(norm_deconvolved[:, r_start:r_end], axis=1)
                    num += 1
                if cs_2_peak > p_threshold and r_start > behavior['frames_per_run']:
                    cluster_vec[num, :] = np.mean(norm_deconvolved[:, r_start:r_end], axis=1)
                    num += 1
                i = r_end
                cs_1_peak = 0
                cs_2_peak = 0
    cluster_vec = cluster_vec[0:num, :]
    Z = scipy.cluster.hierarchy.linkage(cluster_vec, 'single')
    R = scipy.cluster.hierarchy.dendrogram(Z, no_plot=True)
    cluster_vec = cluster_vec[R['leaves'], :]
    plt.subplot(1, 3, 3)
    corr_matrix = pd.DataFrame(cluster_vec).transpose().corr()
    sns.heatmap(corr_matrix, vmin=0, vmax=.3, cmap='Greys')
    plt.ylabel('Clustered synchronous\nevent number')
    plt.xlabel('Clustered synchronous\nevent number')


    norm_deconvolved = norm_deconvolved[idx['both'].index]
    sig_cells = preprocess_opto.sig_reactivated_cells([], [], [], [], [], paths, 0)
    norm_deconvolved = norm_deconvolved[sig_cells > 0, :]
    reactivation_cs_1 = y_pred[:, 0].copy()
    reactivation_cs_2 = y_pred[:, 1].copy()
    p_threshold = .75
    cs_1_peak = 0
    cs_2_peak = 0
    i = 0
    next_r = 0
    cluster_vec = np.empty((1000, len(norm_deconvolved))) * np.nan
    num = 0
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
                if cs_1_peak > p_threshold and r_start > behavior['frames_per_run']:
                    cluster_vec[num, :] = np.mean(norm_deconvolved[:, r_start:r_end], axis=1)
                    num += 1
                if cs_2_peak > p_threshold and r_start > behavior['frames_per_run']:
                    cluster_vec[num, :] = np.mean(norm_deconvolved[:, r_start:r_end], axis=1)
                    num += 1
                i = r_end
                cs_1_peak = 0
                cs_2_peak = 0
    cluster_vec = cluster_vec[0:num, :]
    Z = scipy.cluster.hierarchy.linkage(cluster_vec, 'single')
    R = scipy.cluster.hierarchy.dendrogram(Z, no_plot=True)
    cluster_vec = cluster_vec[R['leaves'], :]
    plt.subplot(1, 3, 2)
    corr_matrix = pd.DataFrame(cluster_vec).transpose().corr()
    sns.heatmap(corr_matrix, vmin=0, vmax=.3, cmap='Greys')
    plt.ylabel('Clustered synchronous\nevent number')
    plt.xlabel('Clustered synchronous\nevent number')

    sns.despine()
    plt.savefig(paths['save_path'] + 'plots/' + paths['mouse'] + '_' + paths['date'] + '_' +
                'prior_R1.png', bbox_inches='tight', dpi=200)
    plt.savefig(paths['save_path'] + 'plots/' + paths['mouse'] + '_' + paths['date'] + '_' +
                'prior_R1.pdf', bbox_inches='tight', dpi=150)
    plt.show()


def pupil_activity_reactivation_modulation(norm_deconvolved, behavior, y_pred, idx, paths, day, days):
    """
    get reactivation rate during high or low pupil
    :param behavior: behavior
    :param y_pred: y pred
    :param paths: path
    :param day: day
    :param days: days
    :return: rates based on pupil
    """
    activity = norm_deconvolved.to_numpy()
    activity = activity[idx['both'].index]
    sig_cells = preprocess_opto.sig_reactivated_cells([], [], [], [], [], paths, 0)
    activity = activity[sig_cells > 0, :]

    normal_trials_idx = np.where(np.isin(behavior['cue_codes'], [behavior['cs_1_code'], behavior['cs_2_code']]))[0]
    y_pred = true_reactivations(y_pred)
    y_pred_cs_1 = y_pred[:, 0]
    y_pred_cs_2 = y_pred[:, 1]
    all_y_pred = y_pred_cs_1 + y_pred_cs_2
    cue_start = behavior['onsets'][normal_trials_idx]
    cue_end = behavior['offsets'][normal_trials_idx]
    end_trials = behavior['end_trials']
    fr = behavior['framerate']
    duration = int(fr * (behavior['iti'] + 5)) + 1
    pupil_vec = behavior['pupil'] / behavior['pupil_max']
    reactivation_prob = []
    pupil_cue = []
    cue_activity = []
    times_considered = preprocess_opto.get_times_considered(y_pred, behavior)

    for i in range(0, len(cue_start)):
        temp_pupil_cue = np.mean(pupil_vec[int(cue_start[i]):int(cue_end[i]) + 1])
        temp_cue_activity = np.mean(activity[:, int(cue_start[i]):int(cue_end[i]) + 1]) * behavior['framerate']
        # temp_pupil_cue_evoked = (temp_pupil_cue - pre_pupil) / pre_pupil
        temp_reactivation = np.sum(
            all_y_pred[int(cue_start[i]):int(cue_start[i]) + duration])
        temp_times_considered_iti = np.sum(
            times_considered[int(cue_start[i]):int(cue_start[i]) + duration])
        temp_reactivation = (temp_reactivation / temp_times_considered_iti) * behavior['framerate']
        if normal_trials_idx[i] not in end_trials:
            pupil_cue.append(temp_pupil_cue)
            reactivation_prob.append(temp_reactivation[0][0])
            cue_activity.append(temp_cue_activity[0][0])

    # sns.set(font_scale=1)
    # sns.set_style("whitegrid", {'axes.grid': False})
    # sns.set_style("ticks")
    # plt.figure(figsize=(9, 6))
    # plt.subplot(2, 2, 1)
    # sns.regplot(x=reactivation_prob, y=pupil_cue, color='k')
    # plt.xlabel('Reactivation probability')
    # plt.ylabel('Max. normalized\npupil area')
    # plt.xlim(-.01, np.max(reactivation_prob) + .01)
    # plt.subplot(2, 2, 2)
    # sns.regplot(x=reactivation_prob, y=cue_activity, color='k')
    # plt.xlabel('Reactivation prob')
    # plt.ylabel('Stimulus activity')
    # plt.xlim(-.01, np.max(reactivation_prob) + .01)
    # sns.despine()
    # plt.show()
    # plt.savefig(paths['save_path'] + 'plots/' + paths['mouse'] + '_' + paths['date'] + '_' +
    #             'pupil_activity_reactivation_modulation.pdf', bbox_inches='tight', dpi=200)

    # num_trials_total = len(pupil_cue)
    # num_trials = round(len(pupil_cue) / 10)
    # all_data = pd.DataFrame({'pupil_cue': pupil_cue, 'reactivation_prob': reactivation_prob,
    #                          'cue_activity': cue_activity})
    #
    # all_data = all_data.sort_values(by='pupil_cue')
    # pupil_cue_vec = [all_data['reactivation_prob'][0:num_trials].mean(),
    #                  all_data['reactivation_prob'][num_trials_total - num_trials:num_trials_total].mean()]
    #
    # all_data = all_data.sort_values(by='cue_activity')
    # cue_activity_vec = [all_data['reactivation_prob'][0:num_trials].mean(),
    #                  all_data['reactivation_prob'][num_trials_total - num_trials:num_trials_total].mean()]

    days_path = paths['base_path'] + paths['mouse'] + '/data_across_days/'
    if days:
        if path.isfile(days_path + 'pupil_activity_reactivation.npy') == 0 or day == 0:
            pupil_reactivation_across_days = [list(range(0, days)), list(range(0, days))]
            pupil_reactivation_across_days[0][day] = np.corrcoef(pupil_cue, reactivation_prob)[0][1]
            pupil_reactivation_across_days[1][day] = np.corrcoef(cue_activity, reactivation_prob)[0][1]
            np.save(days_path + 'pupil_activity_reactivation', pupil_reactivation_across_days)
        else:
            pupil_reactivation_across_days = np.load(days_path + 'pupil_activity_reactivation.npy', allow_pickle=True)
            pupil_reactivation_across_days[0][day] = np.corrcoef(pupil_cue, reactivation_prob)[0][1]
            pupil_reactivation_across_days[1][day] = np.corrcoef(cue_activity, reactivation_prob)[0][1]
            np.save(days_path + 'pupil_activity_reactivation', pupil_reactivation_across_days)


def num_selective_grouped(norm_deconvolved, behavior, idx, paths, day, days):
    activity = norm_deconvolved.to_numpy()
    activity = activity[idx['both'].index]

    [no_change_cells_cs_1, increase_sig_cells_cs_1, decrease_sig_cells_cs_1] = preprocess_opto.group_neurons(activity, behavior, 'cs_1')

    [no_change_cells_cs_2, increase_sig_cells_cs_2, decrease_sig_cells_cs_2] = preprocess_opto.group_neurons(activity, behavior, 'cs_2')

    decrease_sig_cells = decrease_sig_cells_cs_1 + decrease_sig_cells_cs_2
    decrease_sig_cells[decrease_sig_cells > 0] = 1

    increase_sig_cells = increase_sig_cells_cs_1 + increase_sig_cells_cs_2
    increase_sig_cells[increase_sig_cells > 0] = 1

    no_change_sig_cells = no_change_cells_cs_1 + no_change_cells_cs_2
    no_change_sig_cells[no_change_sig_cells < 2] = 0
    no_change_sig_cells[no_change_sig_cells == 2] = 1

    decrease_cells_selectivity = preprocess_opto.selectivity_grouped(activity, behavior, decrease_sig_cells)
    increase_cells_selectivity = preprocess_opto.selectivity_grouped(activity, behavior, increase_sig_cells)
    no_change_cells_selectivity = preprocess_opto.selectivity_grouped(activity, behavior, no_change_sig_cells)

    days_path = paths['base_path'] + paths['mouse'] + '/data_across_days/'
    if days:
        if path.isfile(days_path + 'num_selective_grouped.npy') == 0 or day == 0:
            across_days = [list(range(0, days)), list(range(0, days)), list(range(0, days)),
                           list(range(0, days)), list(range(0, days)), list(range(0, days)),
                           list(range(0, days)), list(range(0, days)), list(range(0, days))]
            across_days[0][day] = sum(decrease_cells_selectivity == 1) / sum(decrease_sig_cells)
            across_days[1][day] = sum(decrease_cells_selectivity == 2) / sum(decrease_sig_cells)
            across_days[2][day] = sum(decrease_cells_selectivity == 3) / sum(decrease_sig_cells)

            across_days[3][day] = sum(increase_cells_selectivity == 1) / sum(increase_sig_cells)
            across_days[4][day] = sum(increase_cells_selectivity == 2) / sum(increase_sig_cells)
            across_days[5][day] = sum(increase_cells_selectivity == 3) / sum(increase_sig_cells)

            across_days[6][day] = sum(no_change_cells_selectivity == 1) / sum(no_change_sig_cells)
            across_days[7][day] = sum(no_change_cells_selectivity == 2) / sum(no_change_sig_cells)
            across_days[8][day] = sum(no_change_cells_selectivity == 3) / sum(no_change_sig_cells)
            np.save(days_path + 'num_selective_grouped', across_days)
        else:
            across_days = np.load(days_path + 'num_selective_grouped.npy', allow_pickle=True)
            across_days[0][day] = sum(decrease_cells_selectivity == 1) / sum(decrease_sig_cells)
            across_days[1][day] = sum(decrease_cells_selectivity == 2) / sum(decrease_sig_cells)
            across_days[2][day] = sum(decrease_cells_selectivity == 3) / sum(decrease_sig_cells)

            across_days[3][day] = sum(increase_cells_selectivity == 1) / sum(increase_sig_cells)
            across_days[4][day] = sum(increase_cells_selectivity == 2) / sum(increase_sig_cells)
            across_days[5][day] = sum(increase_cells_selectivity == 3) / sum(increase_sig_cells)

            across_days[6][day] = sum(no_change_cells_selectivity == 1) / sum(no_change_sig_cells)
            across_days[7][day] = sum(no_change_cells_selectivity == 2) / sum(no_change_sig_cells)
            across_days[8][day] = sum(no_change_cells_selectivity == 3) / sum(no_change_sig_cells)
            np.save(days_path + 'num_selective_grouped', across_days)


def reactivation_difference_tunedflip_R2(norm_deconvolved, behavior, y_pred, idx, paths, day, days):
    activity = norm_deconvolved.to_numpy()
    activity = activity[idx['both'].index]

    [no_change_cells_cs_1, increase_sig_cells_cs_1, decrease_sig_cells_cs_1] = preprocess_opto.group_neurons(activity,
                                                                                                        behavior,
                                                                                                        'cs_1')
    [no_change_cells_cs_2, increase_sig_cells_cs_2, decrease_sig_cells_cs_2] = preprocess_opto.group_neurons(activity,
                                                                                                        behavior,
                                                                                                        'cs_2')

    index_frames_start_cs_1 = []
    index_frames_start_cs_2 = []
    normal_trials_idx = np.where(np.isin(behavior['cue_codes'], [behavior['cs_1_code'], behavior['cs_2_code']]))[0]
    cue_start = behavior['onsets'][normal_trials_idx]
    cue_codes = behavior['cue_codes'][normal_trials_idx]
    end_trials = behavior['end_trials']
    for i in range(0, 10):
        if normal_trials_idx[i] not in end_trials:
            if cue_codes[i] == behavior['cs_1_code']:
                for j in range(0, behavior['frames_before']):
                    index_frames_start_cs_1.append(int(cue_start[i]) + j)
            if cue_codes[i] == behavior['cs_2_code']:
                for j in range(0, behavior['frames_before']):
                    index_frames_start_cs_2.append(int(cue_start[i]) + j)

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
                if cs_1_peak > p_threshold and int(cue_start[10]) > r_start > int(cue_start[0]):
                    reactivation_times_cs_1[r_start:r_end] = 1
                if cs_2_peak > p_threshold and int(cue_start[10]) > r_start > int(cue_start[0]):
                    reactivation_times_cs_2[r_start:r_end] = 1
                i = r_end
                cs_1_peak = 0
                cs_2_peak = 0

    scale_factor = 1.3071117795875629  # get_reactivation_cue_scale(paths)

    diff_cs_1 = np.zeros(len(activity))
    diff_cs_2 = np.zeros(len(activity))
    for j in range(len(activity)):
        cue_1 = activity[j, index_frames_start_cs_1]
        reactivation_1 = activity[j, reactivation_times_cs_1 == 1]
        diff_cs_1[j] = ((np.mean(reactivation_1) * scale_factor) - np.mean(cue_1))
        cue_2 = activity[j, index_frames_start_cs_2]
        reactivation_2 = activity[j, reactivation_times_cs_2 == 1]
        diff_cs_2[j] = ((np.mean(reactivation_2) * scale_factor) - np.mean(cue_2))

    [cs_1_poscells, cs_1_negcells] = preprocess_opto.sig_test_R2(pd.DataFrame(activity), behavior, 'cs_1', 'start')
    [cs_2_poscells, cs_2_negcells] = preprocess_opto.sig_test_R2(pd.DataFrame(activity), behavior, 'cs_2', 'start')
    [both_poscells, _] = preprocess_opto.combine_sig(cs_1_poscells, cs_1_negcells, cs_2_poscells, cs_2_negcells)
    S1_only_s = both_poscells - cs_2_poscells
    S2_only_s = both_poscells - cs_1_poscells

    [cs_1_poscells, cs_1_negcells] = preprocess_opto.sig_test_R2(pd.DataFrame(activity), behavior, 'cs_1', 'end')
    [cs_2_poscells, cs_2_negcells] = preprocess_opto.sig_test_R2(pd.DataFrame(activity), behavior, 'cs_2', 'end')
    [both_poscells, _] = preprocess_opto.combine_sig(cs_1_poscells, cs_1_negcells, cs_2_poscells, cs_2_negcells)
    S1_only_e = both_poscells - cs_2_poscells
    S2_only_e = both_poscells - cs_1_poscells

    S1_S2 = S1_only_s + S2_only_e
    S2_S1 = S1_only_e + S2_only_s
    S1_S1 = S1_only_s + S1_only_e
    S2_S2 = S2_only_e + S2_only_s

    S1_S2_1 = np.mean(diff_cs_1[S1_S2 == 2])
    S2_S1_1 = np.mean(diff_cs_1[S2_S1 == 2])
    S1_S2_2 = np.mean(diff_cs_2[S1_S2 == 2])
    S2_S1_2 = np.mean(diff_cs_2[S2_S1 == 2])

    S1_S1_1 = np.mean(diff_cs_1[S1_S1 == 2])
    S1_S1_2 = np.mean(diff_cs_2[S1_S1 == 2])
    S2_S2_1 = np.mean(diff_cs_1[S2_S2 == 2])
    S2_S2_2 = np.mean(diff_cs_2[S2_S2 == 2])

    days_path = paths['base_path'] + paths['mouse'] + '/data_across_days/'
    if days:
        if path.isfile(days_path + 'reactivation_influence_flip.npy') == 0 or day == 0:
            across_days = [list(range(0, days)), list(range(0, days)), list(range(0, days)), list(range(0, days)),
                           list(range(0, days)), list(range(0, days)), list(range(0, days)), list(range(0, days))]
            across_days[0][day] = S1_S2_1
            across_days[1][day] = S1_S2_2
            across_days[2][day] = S2_S1_2
            across_days[3][day] = S2_S1_1
            across_days[4][day] = S1_S1_1
            across_days[5][day] = S1_S1_2
            across_days[6][day] = S2_S2_1
            across_days[7][day] = S2_S2_2
            np.save(days_path + 'reactivation_influence_flip', across_days)
        else:
            across_days = np.load(days_path + 'reactivation_influence_flip.npy', allow_pickle=True)
            across_days[0][day] = S1_S2_1
            across_days[1][day] = S1_S2_2
            across_days[2][day] = S2_S1_2
            across_days[3][day] = S2_S1_1
            across_days[4][day] = S1_S1_1
            across_days[5][day] = S1_S1_2
            across_days[6][day] = S2_S2_1
            across_days[7][day] = S2_S2_2
            np.save(days_path + 'reactivation_influence_flip', across_days)


def group_neurons_range(norm_deconvolved, idx, behavior, paths, day, days):
    activity = norm_deconvolved.to_numpy()
    activity = activity[idx['both'].index]

    index_frames_start = []
    index_frames_end = []
    num_trials_total = 5
    num_trials = 0
    trial_times = behavior['onsets'][behavior['cue_codes'] == behavior['cs_1_code']]
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

    dist = np.zeros(len(activity_start))
    for j in range(len(activity_start)):
        before = np.reshape(activity_start[j, :, 0:behavior['frames_before']], (num_trials * behavior['frames_before']))
        after = np.reshape(activity_end[j, :, 0:behavior['frames_before']], (num_trials * behavior['frames_before']))
        dist[j] = (np.mean(after) - np.mean(before)) / np.mean(before)

    inc_1 = np.mean(dist) + np.std(dist)
    dec_1 = np.mean(dist) - np.std(dist)
    nc_l_1 = np.mean(dist) - (np.std(dist) / 2)
    nc_h_1 = np.mean(dist) + (np.std(dist) / 2)

    index_frames_start = []
    index_frames_end = []
    num_trials_total = 5
    num_trials = 0
    trial_times = behavior['onsets'][behavior['cue_codes'] == behavior['cs_2_code']]
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

    dist = np.zeros(len(activity_start))
    for j in range(len(activity_start)):
        before = np.reshape(activity_start[j, :, 0:behavior['frames_before']], (num_trials * behavior['frames_before']))
        after = np.reshape(activity_end[j, :, 0:behavior['frames_before']], (num_trials * behavior['frames_before']))
        dist[j] = (np.mean(after) - np.mean(before)) / np.mean(before)

    inc_2 = np.mean(dist) + np.std(dist)
    dec_2 = np.mean(dist) - np.std(dist)
    nc_l_2 = np.mean(dist) - (np.std(dist) / 2)
    nc_h_2 = np.mean(dist) + (np.std(dist) / 2)

    days_path = paths['base_path'] + paths['mouse'] + '/data_across_days/'
    if days:
        if path.isfile(days_path + 'group_neurons_dist.npy') == 0 or day == 0:
            reactivation_cue_pca_vec = [list(range(0, days)), list(range(0, days)), list(range(0, days)),
                                        list(range(0, days)), list(range(0, days)), list(range(0, days)),
                                        list(range(0, days)), list(range(0, days))]
            reactivation_cue_pca_vec[0][day] = inc_1
            reactivation_cue_pca_vec[1][day] = inc_2
            reactivation_cue_pca_vec[2][day] = dec_1
            reactivation_cue_pca_vec[3][day] = dec_2
            reactivation_cue_pca_vec[4][day] = nc_l_1
            reactivation_cue_pca_vec[5][day] = nc_l_2
            reactivation_cue_pca_vec[6][day] = nc_h_1
            reactivation_cue_pca_vec[7][day] = nc_h_2
            np.save(days_path + 'group_neurons_dist', reactivation_cue_pca_vec)
        else:
            reactivation_cue_pca_vec = np.load(days_path + 'group_neurons_dist.npy', allow_pickle=True)
            reactivation_cue_pca_vec[0][day] = inc_1
            reactivation_cue_pca_vec[1][day] = inc_2
            reactivation_cue_pca_vec[2][day] = dec_1
            reactivation_cue_pca_vec[3][day] = dec_2
            reactivation_cue_pca_vec[4][day] = nc_l_1
            reactivation_cue_pca_vec[5][day] = nc_l_2
            reactivation_cue_pca_vec[6][day] = nc_h_1
            reactivation_cue_pca_vec[7][day] = nc_h_2
            np.save(days_path + 'group_neurons_dist', reactivation_cue_pca_vec)















































