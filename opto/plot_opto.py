import warnings
import preprocess_opto
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from os import path
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
            y_pred_binned_across_days = [list(range(0, days)), list(range(0, days)), list(range(0, days))]
            y_pred_binned_across_days[0][day] = y_pred_binned
            y_pred_binned_across_days[1][day] = y_pred_binned_opto
            y_pred_binned_across_days[2][day] = x_label
            np.save(days_path + 'y_pred_binned', y_pred_binned_across_days)
        else:
            y_pred_binned_across_days = np.load(days_path + 'y_pred_binned.npy', allow_pickle=True)
            y_pred_binned_across_days[0][day] = y_pred_binned
            y_pred_binned_across_days[1][day] = y_pred_binned_opto
            y_pred_binned_across_days[2][day] = x_label
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









