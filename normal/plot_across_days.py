from scipy import stats
from scipy import signal
import preprocess
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import warnings
warnings.filterwarnings('ignore')


def within_day(paths):
    y_pred_binned_across_days = list(np.load(paths['base_path'] + paths['mouse'] +
                                             '/data_across_days/y_pred_binned.npy'))
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(10, 7))
    within_day_helper(y_pred_binned_across_days, paths, 'Reactivation probability ($\mathregular{s^{-1}}$)', '', 1)
    within_day_helper(y_pred_binned_across_days, paths, 'Reactivation probability (/s) norm.', '_normalized', 3)
    plt.savefig(paths['base_path'] + paths['mouse'] + '/data_across_days/plots/' + paths['mouse'] + '_' +
                'mean_within_day.png', bbox_inches='tight', dpi=150)
    plt.close()


def within_day_helper(binned_vec, paths, y_label, norm, idx):
    """
    plot within day
    :param binned_vec: vector of pupil or reactivation
    :param paths: path to data
    :param y_label: y label
    :param norm: to normalize or not
    :return: plot
    """
    session_data = preprocess.load_data(paths)
    task_runs = int(session_data['task_runs'])
    dark_runs = int(session_data['dark_runs'])
    x_label = np.zeros(len(binned_vec[0]))
    for i in range(0, dark_runs):
        x_label[dark_runs-i-1] = (-17 + (i * -17)) / 60
    for i in range(dark_runs, dark_runs + (task_runs*2)):
        x_label[i] = (15.4 + (15.4 * (i-dark_runs))) / 60
    plt.subplot(2, 2, idx)
    mean_reactivation = np.zeros((len(binned_vec), len(binned_vec[0])))
    for i in range(0, len(binned_vec)):
        if norm != '_normalized':
            plt.plot(x_label, binned_vec[i], '-o', c=[.5*(.97 - (1/(len(binned_vec)+1) * i)),
                                                      .9*(.97 - (1/(len(binned_vec)+1) * i)),
                                                      .9*(.97 - (1/(len(binned_vec)+1) * i))], ms=5)
            mean_reactivation[i, :] = binned_vec[i]
        else:
            plt.plot(x_label, binned_vec[i]/binned_vec[i][dark_runs], '-o', c=[.5*(.97 - (1/(len(binned_vec)+1) * i)),
                                                                               .9*(.97 - (1/(len(binned_vec)+1) * i)),
                                                                               .9*(.97 - (1/(len(binned_vec)+1) * i))],
                     ms=5)
            mean_reactivation[i, :] = binned_vec[i]/binned_vec[i][dark_runs]
    plt.axvspan(-.5, 0, alpha=.25, color='gray')
    plt.ylabel(y_label)
    plt.xlabel('Time from first cue (hours)')
    sns.despine()
    plt.subplot(2, 2, idx+1)
    mean = mean_reactivation.mean(axis=0)
    sem_plus = mean + stats.sem(mean_reactivation, axis=0)
    sem_minus = mean - stats.sem(mean_reactivation, axis=0)
    plt.plot(x_label, mean, '-o', c='cadetblue', linewidth=2, ms=7)
    plt.fill_between(x_label, sem_plus, sem_minus, alpha=0.1, color='cadetblue')
    plt.axvspan(-.5, 0, alpha=.25, color='gray')
    plt.ylabel(y_label)
    plt.xlabel('Time from first cue (hours)')
    sns.despine()


def across_day(paths):
    """
    plot across day
    :param paths: path to data
    :return: plot
    """
    y_pred_binned_across_days = list(np.load(paths['base_path'] + paths['mouse'] +
                                             '/data_across_days/y_pred_binned.npy'))
    session_data = preprocess.load_data(paths)
    task_runs = int(session_data['task_runs'])
    dark_runs = int(session_data['dark_runs'])
    mean_vec = []
    for i in range(0, len(y_pred_binned_across_days)):
        mean_vec.append(np.mean(y_pred_binned_across_days[i][dark_runs:dark_runs+(task_runs*2)]))
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    fig = plt.figure(figsize=(5, 2.5))
    plt.plot(range(1, len(np.mean(y_pred_binned_across_days, axis=1))+1), mean_vec, '-o', color='cadetblue',
             linewidth=2, ms=7)
    plt.ylabel('Reactivation probability (/s)')
    plt.xlabel('Day')
    plt.xlim(.5, len(np.mean(y_pred_binned_across_days, axis=1))+.5)
    sns.despine()
    plt.savefig(paths['base_path'] + paths['mouse'] + '/data_across_days/plots/' + paths['mouse'] + '_'
                + 'mean_across_day.png', bbox_inches='tight', dpi=150)
    plt.close(fig)


def cue_duration(paths):
    """
    plots cue selectivity
    :param paths: path to data
    :return: plot
    """
    trial_reactivation = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/trial_reactivation_all.npy',
                                 allow_pickle=True)
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(10, 7))
    cue_duration_helper(trial_reactivation, paths, 'Reactivation probability ($\mathregular{s^{-1}}$)', '', 1)
    cue_duration_helper(trial_reactivation, paths, 'Reactivation probability (/s) norm.', '_normalized', 3)
    plt.savefig(paths['base_path'] + paths['mouse'] + '/data_across_days/plots/' + paths['mouse'] + '_' +
                'cue_duration.png', bbox_inches='tight', dpi=150)
    plt.close()


def cue_duration_helper(trial_reactivation, paths, y_label, norm, idx):
    """
    helper to cue selectivity
    :param trial_reactivation: data from all days
    :param paths: path to data
    :param y_label: y label
    :param norm: to normalize
    :return: plot
    """
    session_data = preprocess.load_data(paths)
    fr = session_data['framerate']
    plt.subplot(2, 2, idx)
    duration = int(fr * 8) + 1
    factor = 5
    step = int(((len(trial_reactivation[0][0][0])) - duration) / factor)
    mean_reactivation = np.zeros((len(trial_reactivation[0]), factor))
    x_binned = []
    for day in range(0, len(trial_reactivation[0])):
        temp_reactivation_data = trial_reactivation[0][day]
        reactivation_times_considered = trial_reactivation[1][day]
        reactivation_data_binned = []
        x_binned = []
        for i in range(0, factor):
            reactivation_data_binned.append(
                np.sum(temp_reactivation_data[:, duration + (i * step):duration + (i + 1) * step]) *
                int(fr) / np.sum(reactivation_times_considered[:, duration + (i * step):duration + (i + 1) * step]))
            x_binned.append(duration + (i + .5) * step)
        if norm != '_normalized':
            plt.plot(x_binned, reactivation_data_binned, '-o',
                     c=[.5 * (.97 - (1 / (len(trial_reactivation[0]) + 1) * day)),
                        .9 * (.97 - (1 / (len(trial_reactivation[0]) + 1) * day)),
                        .9 * (.97 - (1 / (len(trial_reactivation[0]) + 1) * day))], ms=5)
            mean_reactivation[day, :] = reactivation_data_binned
        else:
            plt.plot(x_binned, reactivation_data_binned/reactivation_data_binned[0], '-o',
                     c=[.5 * (.97 - (1 / (len(trial_reactivation[0]) + 1) * day)),
                        .9 * (.97 - (1 / (len(trial_reactivation[0]) + 1) * day)),
                        .9 * (.97 - (1 / (len(trial_reactivation[0]) + 1) * day))], ms=5)
            mean_reactivation[day, :] = reactivation_data_binned/reactivation_data_binned[0]
    plt.axvspan(0, int(fr * 1), alpha=.75, color='mediumseagreen')
    plt.axvspan(int(fr * 1), int(fr * 2), alpha=.75, color='salmon')
    plt.axvspan(int(fr * 2), int(fr * 8), alpha=.2, color='gray')
    plt.ylabel(y_label)
    plt.xlabel('Time relative to cue onset (s)')
    plt.xticks([int(fr * 0), int(fr * 10), int(fr * 20), int(fr * 30), int(fr * 40), int(fr * 50), int(fr * 60)],
               ['0', '10', '20', '30', '40', '50', '60'])
    plt.xlim((0, int(fr * 61)))
    sns.despine()
    plt.subplot(2, 2, idx+1)
    mean = mean_reactivation.mean(axis=0)
    sem_plus = mean + stats.sem(mean_reactivation, axis=0)
    sem_minus = mean - stats.sem(mean_reactivation, axis=0)
    plt.plot(x_binned, mean, '-o', c='cadetblue', linewidth=2, ms=7)
    plt.fill_between(x_binned, sem_plus, sem_minus, alpha=0.1, color='cadetblue')
    plt.axvspan(0, int(fr * 1), alpha=.75, color='mediumseagreen')
    plt.axvspan(int(fr * 1), int(fr * 2), alpha=.75, color='salmon')
    plt.axvspan(int(fr * 2), int(fr * 8), alpha=.2, color='gray')
    plt.ylabel(y_label)
    plt.xlabel('Time relative to cue onset (s)')
    plt.xticks([int(fr * 0), int(fr * 10), int(fr * 20), int(fr * 30), int(fr * 40), int(fr * 50), int(fr * 60)],
               ['0', '10', '20', '30', '40', '50', '60'])
    plt.xlim((0, int(fr * 61)))
    sns.despine()


def mean_cue_selectivity(paths):
    cs_pref = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/cs_pref.npy', allow_pickle=True)
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(8.5, 7))
    mean_cue_selectivity_helper(cs_pref, 1, 'Reactivation cue bias (odds)')
    mean_cue_selectivity_helper(cs_pref, 3, 'Reactivation cue bias')
    plt.subplots_adjust(wspace=.3)
    plt.savefig(paths['base_path'] + paths['mouse'] + '/data_across_days/plots/' + paths['mouse'] + '_'
                + 'cue_bias.png', bbox_inches='tight', dpi=500)
    plt.close()


def mean_cue_selectivity_helper(cs_pref, idx, y_label):
    plt.subplot(2, 2, idx)
    x = [1, 2]
    cs_1_bias = []
    cs_2_bias = []
    for i in range(0, len(cs_pref[0])):
        if y_label == 'Reactivation cue bias (odds)':
            cs_1 = cs_pref[0][i][0] / cs_pref[0][i][1]
            cs_2 = cs_pref[1][i][1] / cs_pref[1][i][0]
            plt.plot(x[0], cs_1, 'o', c=[.5 * (1 - (1 / len(cs_pref[0]) * i)),
                                         1 - (1 / len(cs_pref[0]) * i),
                                         .7 * (1 - (1 / len(cs_pref[0]) * i))],
                     markersize=6, zorder=100 / (i + 1))
            plt.plot(x[1], cs_2, 'o', c=[1 - (1 / len(cs_pref[0]) * i),
                                         .5 * (1 - (1 / len(cs_pref[0]) * i)),
                                         .5 * (1 - (1 / len(cs_pref[0]) * i))],
                     markersize=6, zorder=100 / (i + 1))
            plt.errorbar(x, [cs_1, cs_2], yerr=0,
                         c=[.5, .5, .5],
                         linewidth=1, linestyle='--')
            cs_1_bias.append(cs_1)
            cs_2_bias.append(cs_2)
        if y_label == 'Reactivation cue bias':
            cs_1 = (cs_pref[0][i][0] - cs_pref[0][i][1]) / (cs_pref[0][i][0] + cs_pref[0][i][1])
            cs_2 = (cs_pref[1][i][1] - cs_pref[1][i][0]) / (cs_pref[1][i][1] + cs_pref[1][i][0])
            plt.plot(x[0], cs_1, 'o', c=[.5 * (1 - (1 / len(cs_pref[0]) * i)),
                                         1 - (1 / len(cs_pref[0]) * i),
                                         .7 * (1 - (1 / len(cs_pref[0]) * i))],
                     markersize=6, zorder=100 / (i + 1))
            plt.plot(x[1], cs_2, 'o', c=[1 - (1 / len(cs_pref[0]) * i),
                                         .5 * (1 - (1 / len(cs_pref[0]) * i)),
                                         .5 * (1 - (1 / len(cs_pref[0]) * i))],
                     markersize=6, zorder=100 / (i + 1))
            plt.errorbar(x, [cs_1, cs_2], yerr=0,
                         c=[.5, .5, .5],
                         linewidth=1, linestyle='--')
            cs_1_bias.append(cs_1)
            cs_2_bias.append(cs_2)
    plt.xlim((.5, 2.5))
    plt.xticks([1, 2], ['Following Cue 1', 'Following Cue 2'])
    plt.gca().get_xticklabels()[1].set_color('salmon')
    plt.gca().get_xticklabels()[0].set_color('mediumseagreen')
    plt.ylabel(y_label)
    if y_label == 'Reactivation cue bias (odds)':
        plt.axhline(y=1, color='black', linestyle='--', linewidth=2, snap=False)
        plt.ylim(0, np.max([cs_1_bias, cs_2_bias]) + 1)
    if y_label == 'Reactivation cue bias':
        plt.axhline(y=0, color='black', linestyle='--', linewidth=2, snap=False)
        plt.ylim(-1, 1)
    y1 = np.mean(cs_1_bias)
    y2 = np.mean(cs_2_bias)
    y1_err = stats.sem(cs_1_bias)
    y2_err = stats.sem(cs_2_bias)
    plt.errorbar(x[0]-.1, y1, yerr=y1_err, c='mediumseagreen', linewidth=2, marker='o', mfc='mediumseagreen', mec='k',
                 ms=10, mew=0)
    plt.errorbar(x[1]+.1, y2, yerr=y2_err, c='salmon', linewidth=2, marker='o', mfc='salmon', mec='black', ms=10, mew=0)
    sns.despine()


def cue_selectivity_trial(paths):
    trial_reactivation_cs_1 = np.load(paths['base_path'] + paths['mouse'] +
                                      '/data_across_days/trial_reactivation_cs_1.npy', allow_pickle=True)
    trial_reactivation_cs_2 = np.load(paths['base_path'] + paths['mouse'] +
                                      '/data_across_days/trial_reactivation_cs_2.npy', allow_pickle=True)
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(10, 7))
    cue_selectivity_trial_helper(paths, trial_reactivation_cs_1, trial_reactivation_cs_2, 1,
                                 'Reactivation cue bias (odds)')
    cue_selectivity_trial_helper(paths, trial_reactivation_cs_1, trial_reactivation_cs_2, 3, 'Reactivation cue bias')
    plt.savefig(paths['base_path'] + paths['mouse'] + '/data_across_days/plots/' + paths['mouse'] + '_'
                + 'cue_duration_bias.png', bbox_inches='tight', dpi=500)
    plt.close()


def cue_selectivity_trial_helper(paths, trial_reactivation_cs_1, trial_reactivation_cs_2, idx, y_label):
    session_data = preprocess.load_data(paths)
    fr = session_data['framerate']
    trial_reactivation = [trial_reactivation_cs_1, trial_reactivation_cs_2]
    weight_cs_1 = len(trial_reactivation_cs_1[0][1]) / (len(trial_reactivation_cs_1[0][1]) +
                                                        len(trial_reactivation_cs_2[0][1]))
    weight_cs_2 = 1 - weight_cs_1
    duration = int(fr * 8) + 1
    factor = 5
    step = int(((len(trial_reactivation[0][0][0][0])) - duration) / factor)
    x_binned = []
    mean_reactivation = np.zeros((len(trial_reactivation[0][0]), factor))
    for cue_type in range(0, len(trial_reactivation)):
        temp_trial_reactivation = trial_reactivation[cue_type]
        for day in range(0, len(temp_trial_reactivation[0])):
            temp_reactivation_data_1 = temp_trial_reactivation[0][day]
            temp_reactivation_data_2 = temp_trial_reactivation[1][day]
            reactivation_data_binned = []
            x_binned = []
            for i in range(0, factor):
                if y_label == 'Reactivation cue bias (odds)':
                    if cue_type == 0:
                        reactivation_data_binned.append(
                            (np.sum(temp_reactivation_data_1[:, duration + (i * step):duration + (i + 1) * step]) /
                             np.sum(temp_reactivation_data_2[:, duration + (i * step):duration + (i + 1) * step])))
                    if cue_type == 1:
                        reactivation_data_binned.append(
                            (np.sum(temp_reactivation_data_2[:, duration + (i * step):duration + (i + 1) * step]) /
                             np.sum(temp_reactivation_data_1[:, duration + (i * step):duration + (i + 1) * step])))
                if y_label == 'Reactivation cue bias':
                    if cue_type == 0:
                        reactivation_data_binned.append(
                            (np.sum(temp_reactivation_data_1[:, duration + (i * step):duration + (i + 1) * step]) -
                             np.sum(temp_reactivation_data_2[:, duration + (i * step):duration + (i + 1) * step])) /
                            (np.sum(temp_reactivation_data_1[:, duration + (i * step):duration + (i + 1) * step]) +
                             np.sum(temp_reactivation_data_2[:, duration + (i * step):duration + (i + 1) * step])))
                    if cue_type == 1:
                        reactivation_data_binned.append(
                            (np.sum(temp_reactivation_data_1[:, duration + (i * step):duration + (i + 1) * step]) -
                             np.sum(temp_reactivation_data_2[:, duration + (i * step):duration + (i + 1) * step])) /
                            (np.sum(temp_reactivation_data_1[:, duration + (i * step):duration + (i + 1) * step]) +
                             np.sum(temp_reactivation_data_2[:, duration + (i * step):duration + (i + 1) * step])) * -1)
                x_binned.append(duration + (i + .5) * step)
            reactivation_data_binned = np.nan_to_num(reactivation_data_binned, copy=True, nan=0)
            if cue_type == 0:
                mean_reactivation[day, :] += np.array(reactivation_data_binned) * weight_cs_1
            if cue_type == 1:
                mean_reactivation[day, :] += np.array(reactivation_data_binned) * weight_cs_2
    plt.subplot(2, 2, idx)
    for i in range(len(mean_reactivation)):
        plt.plot(x_binned, mean_reactivation[i, :], '-o',
                 c=[.5 * (.97 - (1 / (len(trial_reactivation[0][0]) + 1) * i)),
                    .9 * (.97 - (1 / (len(trial_reactivation[0][0]) + 1) * i)),
                    .9 * (.97 - (1 / (len(trial_reactivation[0][0]) + 1) * i))], ms=5)
    plt.axvspan(0, int(fr * 1), alpha=.75, color='mediumseagreen')
    plt.axvspan(int(fr * 1), int(fr * 2), alpha=.75, color='salmon')
    plt.axvspan(int(fr * 2), int(fr * 8), alpha=.2, color='gray')
    plt.ylabel(y_label)
    plt.xlabel('Time relative to cue onset (s)')
    plt.xticks([int(fr * 0), int(fr * 10), int(fr * 20), int(fr * 30), int(fr * 40), int(fr * 50), int(fr * 60)],
               ['0', '10', '20', '30', '40', '50', '60'])
    plt.xlim((0, int(fr * 61)))
    if y_label == 'Reactivation cue bias (odds)':
        plt.axhline(y=1, color='black', linestyle='--', linewidth=2, snap=False)
    if y_label == 'Reactivation cue bias':
        plt.axhline(y=0, color='black', linestyle='--', linewidth=2, snap=False)
        plt.ylim(-.5, .75)
    sns.despine()
    plt.subplot(2, 2, idx+1)
    mean = mean_reactivation.mean(axis=0)
    sem_plus = mean + stats.sem(mean_reactivation, axis=0)
    sem_minus = mean - stats.sem(mean_reactivation, axis=0)
    plt.plot(x_binned, mean, '-o', c='cadetblue', linewidth=2, ms=7)
    plt.fill_between(x_binned, sem_plus, sem_minus, alpha=0.1, color='cadetblue')
    plt.axvspan(0, int(fr * 1), alpha=.75, color='mediumseagreen')
    plt.axvspan(int(fr * 1), int(fr * 2), alpha=.75, color='salmon')
    plt.axvspan(int(fr * 2), int(fr * 8), alpha=.2, color='gray')
    plt.ylabel(y_label)
    plt.xlabel('Time relative to cue onset (s)')
    plt.xticks([int(fr * 0), int(fr * 10), int(fr * 20), int(fr * 30), int(fr * 40), int(fr * 50), int(fr * 60)],
               ['0', '10', '20', '30', '40', '50', '60'])
    plt.xlim((0, int(fr * 61)))
    if y_label == 'Reactivation cue bias (odds)':
        plt.axhline(y=1, color='black', linestyle='--', linewidth=2, snap=False)
    if y_label == 'Reactivation cue bias':
        plt.axhline(y=0, color='black', linestyle='--', linewidth=2, snap=False)
        plt.ylim(-.5, .75)
    sns.despine()


def cue_selectivity_day(paths):
    y_pred_bias_binned = list(np.load(paths['base_path'] + paths['mouse'] +
                                             '/data_across_days/y_pred_bias_binned.npy', allow_pickle=True))
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(10, 7))
    cue_selectivity_day_helper(y_pred_bias_binned[2], paths, 'Reactivation cue bias (odds)', 1)
    cue_selectivity_day_helper(y_pred_bias_binned[1], paths, 'Reactivation cue bias', 3)
    plt.savefig(paths['base_path'] + paths['mouse'] + '/data_across_days/plots/' + paths['mouse'] + '_' +
                'within_day_bias.png', bbox_inches='tight', dpi=500)
    plt.close()


def cue_selectivity_day_helper(binned_vec, paths, y_label, idx):
    session_data = preprocess.load_data(paths)
    task_runs = int(session_data['task_runs'])
    x_label = np.zeros(len(binned_vec[0]))
    for i in range(0, task_runs*2):
        x_label[i] = (15.4 + (15.4 * i)) / 60
    plt.subplot(2, 2, idx)
    mean_reactivation = np.zeros((len(binned_vec), len(binned_vec[0])))
    for i in range(0, len(binned_vec)):
        plt.plot(x_label, binned_vec[i], '-o', c=[.5*(.97 - (1/(len(binned_vec)+1) * i)),
                                                  .9*(.97 - (1/(len(binned_vec)+1) * i)),
                                                  .9*(.97 - (1/(len(binned_vec)+1) * i))], ms=5)
        mean_reactivation[i, :] = binned_vec[i]
    plt.ylabel(y_label)
    plt.xlabel('Time from first cue (hours)')
    if y_label == 'Reactivation cue bias (odds)':
        plt.axhline(y=1, color='black', linestyle='--', linewidth=1, snap=False)
    if y_label == 'Reactivation cue bias':
        plt.ylim(-1, 1)
        plt.axhline(y=0, color='black', linestyle='--', linewidth=1, snap=False)
    sns.despine()
    plt.subplot(2, 2, idx+1)
    mean = mean_reactivation.mean(axis=0)
    sem_plus = mean + stats.sem(mean_reactivation, axis=0)
    sem_minus = mean - stats.sem(mean_reactivation, axis=0)
    plt.plot(x_label, mean, '-o', c='cadetblue', linewidth=2, ms=7)
    plt.fill_between(x_label, sem_plus, sem_minus, alpha=0.1, color='cadetblue')
    plt.ylabel(y_label)
    plt.xlabel('Time from first cue (hours)')
    if y_label == 'Reactivation cue bias (odds)':
        plt.axhline(y=1, color='black', linestyle='--', linewidth=1, snap=False)
    if y_label == 'Reactivation cue bias':
        plt.ylim(-1, 1)
        plt.axhline(y=0, color='black', linestyle='--', linewidth=1, snap=False)
    sns.despine()


def reactivation_physical(paths):
    """
    plot reactivation physical
    :param paths: path
    :return: plot
    """
    reactivation_physical_across_days = list(np.load(paths['base_path'] + paths['mouse'] +
                                      '/data_across_days/reactivation_physical.npy', allow_pickle=True))
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(5, 10.5))
    reactivation_physical_helper(reactivation_physical_across_days[0], paths, 'Normalized pupil area (a.u.)', 'blue',
                                 1, 0, .4)
    reactivation_physical_helper(reactivation_physical_across_days[1], paths, 'Pupil movement (a.u.)', 'red', 2, 0, 3)
    reactivation_physical_helper(reactivation_physical_across_days[2], paths, 'Brain motion (abs, a.u.)',
                                 'darkgoldenrod', 3, 0, 2)

    plt.savefig(paths['base_path'] + paths['mouse'] + '/data_across_days/plots/' + paths['mouse'] + '_' +
                'reactivation_physical.png', bbox_inches='tight', dpi=150)
    plt.close()


def reactivation_physical_helper(vec, paths, y_label, c, idx, lim1, lim2):
    """
    reactivation physical helper
    :param vec: vec of data
    :param paths: path
    :param y_label: y label
    :param c: color
    :param idx: which subplot
    :param lim1: y lim
    :param lim2: y lim
    :return: plot
    """
    session_data = preprocess.load_data(paths)
    framerate = session_data['framerate']
    plt.subplot(3, 1, idx)
    mean_vec = np.zeros((len(vec), len(vec[0])))
    for i in range(0, len(vec)):
        mean_vec[i, :] = vec[i]
    mean = mean_vec.mean(axis=0)
    sem_plus = mean + stats.sem(mean_vec, axis=0)
    sem_minus = mean - stats.sem(mean_vec, axis=0)
    plt.plot(mean, c=c, linewidth=2, ms=7)
    plt.fill_between(np.arange(mean.shape[0]), sem_plus, sem_minus, alpha=0.1, color=c)
    plt.xticks([int(framerate * 0), int(framerate * 10), int(framerate * 20),
                int(framerate * 30), int(framerate * 40)], ['-20', '-10', '0', '10', '20'])
    plt.ylabel(y_label)
    plt.ylim(lim1, lim2)
    plt.xlabel('Time relative to reactivation (s)')
    sns.despine()


def activity_learning(paths):
    activity_all = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/activity.npy', allow_pickle=True)
    session_data = preprocess.load_data(paths)
    fr = session_data['framerate']
    num_trials = len(session_data['onsets'])

    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(10, 14))

    plt.subplot(4, 2, 1)
    plt.xlabel('Trial number')
    plt.ylabel("Correlation between cues")
    corr_all = np.empty((len(activity_all[0]), num_trials)) * np.nan
    corr_all_norm = np.empty((len(activity_all[0]), num_trials)) * np.nan
    for i in range(0, len(activity_all[0])):
        smoothed = np.array(pd.DataFrame(activity_all[0][i]).rolling(int(fr * 2), min_periods=1, center=True).mean())
        x = range(0, len(smoothed))
        plt.plot(x, smoothed, c=[0, 1 - (1 / len(activity_all[0]) * i), 1 - (1 / len(activity_all[0]) * i)])
        corr_all[i, 0:len(smoothed)] = np.concatenate(smoothed, axis=0)
        corr_all_norm[i, 0:len(smoothed)] = np.concatenate(smoothed, axis=0) - smoothed[0]
    sns.despine()

    plt.subplot(4, 2, 2)
    plt.xlabel('Day')
    plt.ylabel("Initial correlation between cues")
    corr_initial = []
    day = []
    for i in range(0, len(activity_all[0])):
        initial_corr = np.mean(activity_all[0][i][0:20])
        corr_initial.append(initial_corr)
        day.append(i+1)
    ax = sns.regplot(x=day, y=corr_initial, color=[0, .6, .6])
    [r, p] = stats.pearsonr(day, corr_initial)
    plt.text(.8, .9, 'r=' + str(round(r, 2)) + ', p=' + str(round(p, 5)), horizontalalignment='center',
             verticalalignment='center', transform=ax.transAxes)
    plt.xlim(0, day[len(day)-1] + 1)
    sns.despine()

    plt.subplot(4, 2, 3)
    plt.xlabel('Day')
    plt.ylabel("Δ Correlation between cues")
    corr_diff = []
    day = []
    for i in range(0, len(activity_all[0])):
        diff_corr = np.mean(activity_all[0][i][len(activity_all[0][i])-20:len(activity_all[0][i])]) - \
                       np.mean(activity_all[0][i][0:20])
        corr_diff.append(diff_corr)
        day.append(i + 1)
    ax = sns.regplot(x=day, y=corr_diff, color=[0, .6, .6])
    [r, p] = stats.pearsonr(day, corr_diff)
    plt.text(.8, .9, 'r=' + str(round(r, 2)) + ', p=' + str(round(p, 5)), horizontalalignment='center',
             verticalalignment='center', transform=ax.transAxes)
    plt.xlim(0, day[len(day) - 1] + 1)
    sns.despine()

    plt.subplot(4, 2, 4)
    plt.xlabel('Day')
    plt.ylabel("Δ Correlation between cues")
    corr_diff = []
    for i in range(0, len(activity_all[0])):
        diff_corr = np.mean(activity_all[0][i][len(activity_all[0][i]) - 20:len(activity_all[0][i])]) - \
                    np.mean(activity_all[0][i][0:20])
        corr_diff.append(diff_corr)
        plt.plot(1, diff_corr, 'o-', ms=5, mfc='none', c=[0, 1 - (1 / len(activity_all[0]) * i),
                                                          1 - (1 / len(activity_all[0]) * i)])
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1, snap=False)
    y1 = np.mean(corr_diff)
    y1_err = stats.sem(corr_diff)
    plt.errorbar(1.1, y1, yerr=y1_err, c=[0, 0, 0], linewidth=2, marker='o', mfc=[0, 0, 0], mec=[0, 0, 0], ms=7,
                 mew=0, zorder=100)
    plt.xlim(.5, 4)
    plt.ylim(-.3, .3)
    plt.xticks([1], ['Within day'])
    sns.despine()

    plt.subplot(4, 2, 5)
    plt.xlabel('Trial number')
    plt.ylabel("Cue activity ($\mathregular{Ca^{2+}}$transient $\mathregular{s^{-1}}$)")
    corr_all = np.empty((len(activity_all[3]), num_trials)) * np.nan
    corr_all_norm = np.empty((len(activity_all[3]), num_trials)) * np.nan
    for i in range(0, len(activity_all[3])):
        smoothed = np.array(pd.DataFrame(activity_all[3][i]).rolling(int(fr * 2), min_periods=1, center=True).mean())
        x = range(0, len(smoothed))
        plt.plot(x, smoothed*fr, c=[0, 1 - (1 / len(activity_all[3]) * i), 1 - (1 / len(activity_all[3]) * i)])
        corr_all[i, 0:len(smoothed)] = np.concatenate(smoothed, axis=0)
        corr_all_norm[i, 0:len(smoothed)] = np.concatenate(smoothed, axis=0) - smoothed[0]
    sns.despine()

    plt.subplot(4, 2, 6)
    plt.xlabel('Day')
    plt.ylabel("Initial cue activity")
    cue_initial = []
    day = []
    for i in range(0, len(activity_all[3])):
        initial_cue = np.mean(activity_all[3][i][0:20]) * fr
        cue_initial.append(initial_cue[0][0])
        day.append(i + 1)
    ax = sns.regplot(x=day, y=cue_initial, color=[0, .6, .6])
    [r, p] = stats.pearsonr(day, cue_initial)
    plt.text(.8, .9, 'r=' + str(round(r, 2)) + ', p=' + str(round(p, 5)), horizontalalignment='center',
             verticalalignment='center', transform=ax.transAxes)
    plt.xlim(0, day[len(day) - 1] + 1)
    sns.despine()

    plt.subplot(4, 2, 7)
    plt.xlabel('Day')
    plt.ylabel("Δ Cue activity")
    cue_diff = []
    day = []
    for i in range(0, len(activity_all[3])):
        diff_cue = (np.mean(activity_all[3][i][len(activity_all[3][i]) - 20:len(activity_all[3][i])]) - \
                       np.mean(activity_all[3][i][0:20])) * fr
        cue_diff.append(diff_cue[0][0])
        day.append(i + 1)
    ax = sns.regplot(x=day, y=cue_diff, color=[0, .6, .6])
    [r, p] = stats.pearsonr(day, cue_diff)
    plt.text(.8, .9, 'r=' + str(round(r, 2)) + ', p=' + str(round(p, 5)), horizontalalignment='center',
             verticalalignment='center', transform=ax.transAxes)
    plt.xlim(0, day[len(day) - 1] + 1)
    sns.despine()
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=.3, hspace=None)

    plt.subplot(4, 2, 8)
    plt.xlabel('Day')
    plt.ylabel("Δ Cue activity")
    cue_diff = []
    for i in range(0, len(activity_all[0])):
        diff_cue = (np.mean(activity_all[3][i][len(activity_all[3][i]) - 20:len(activity_all[3][i])]) - \
                   np.mean(activity_all[3][i][0:20])) * fr
        cue_diff.append(diff_cue[0][0])
        plt.plot(1, diff_cue, 'o-', ms=5, mfc='none', c=[0, 1 - (1 / len(activity_all[0]) * i),
                                                          1 - (1 / len(activity_all[0]) * i)])
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1, snap=False)
    y1 = np.mean(cue_diff)
    y1_err = stats.sem(cue_diff)
    plt.errorbar(1.1, y1, yerr=y1_err, c=[0, 0, 0], linewidth=2, marker='o', mfc=[0, 0, 0], mec=[0, 0, 0], ms=7,
                 mew=0, zorder=100)
    plt.xlim(.5, 4)
    plt.ylim(-.3, .3)
    plt.xticks([1], ['Within day'])
    sns.despine()

    plt.savefig(paths['base_path'] + paths['mouse'] + '/data_across_days/plots/' + paths['mouse'] + '_'
                + 'activity_learning.png', bbox_inches='tight', dpi=500)
    plt.close()


def activity_reactivation(paths):
    activity_all = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/activity.npy', allow_pickle=True)
    session_data = preprocess.load_data(paths)
    fr = session_data['framerate']

    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(10, 14))
    plt.subplot(4, 2, 1)
    corr_all = []
    cue_all = []
    pupil_all = []
    for i in range(0, len(activity_all[0])):
        smoothed_correlation = np.array(pd.DataFrame(activity_all[0][i]).rolling(int(fr * 2), min_periods=1,
                                                                                 center=True).mean())
        smoothed_pupil = np.array(pd.DataFrame(activity_all[2][i]).rolling(int(fr * 2), min_periods=1,
                                                                           center=True).mean())
        smoothed_cue = np.array(pd.DataFrame(activity_all[3][i]).rolling(int(fr * 2), min_periods=1,
                                                                         center=True).mean())
        smoothed_reactivation_prob = np.array(pd.DataFrame(activity_all[1][i]).rolling(int(fr * 2), min_periods=1,
                                                                                       center=True).mean())
        # fig, (ax1) = plt.subplots(1, 1, figsize=(5, 3.5))
        # ax1.spines['left'].set_color('k')
        # ax1.plot(smoothed_correlation, color=[0, .3, .3])
        # ax1.set_ylabel("Correlation between cues", c=[0, .3, .3])
        # ax1.set_xlabel('Trial number')
        # ax1.tick_params(axis='y', colors=[0, .3, .3])
        # ax1.spines['top'].set_visible(False)
        # ax2 = ax1.twinx()
        # ax2.plot(smoothed_reactivation_prob, c='firebrick')
        # ax2.set_ylabel('Sum reactivation probability', rotation=270, va='bottom', c='firebrick')
        # ax2.set_xlabel('Trial number')
        # ax2.spines['right'].set_color('firebrick')
        # ax2.tick_params(axis='y', colors='firebrick')
        # ax2.spines['top'].set_visible(False)
        # plt.savefig(paths['base_path'] + paths['mouse'] + '/data_across_days/plots/' + paths['mouse'] + '_'
        #             + 'activity_reactivation.png', bbox_inches='tight', dpi=150)
        # return 0

        corr_temp = np.corrcoef(np.concatenate(smoothed_correlation, axis=0),
                                np.concatenate(smoothed_reactivation_prob, axis=0))[0][1]
        pupil_temp = np.corrcoef(np.concatenate(smoothed_pupil, axis=0),
                                 np.concatenate(smoothed_reactivation_prob, axis=0))[0][1]
        cue_temp = np.corrcoef(np.concatenate(smoothed_cue, axis=0),
                               np.concatenate(smoothed_reactivation_prob, axis=0))[0][1]
        plt.plot(3, pupil_temp, 'o-', ms=5, mfc='none',
                 c=[0, 1 - (1 / len(activity_all[0]) * i), 1 - (1 / len(activity_all[0]) * i)])
        plt.plot(1, corr_temp, 'o-', ms=5, mfc='none',
                 c=[0, 1 - (1 / len(activity_all[0]) * i), 1 - (1 / len(activity_all[0]) * i)])
        plt.plot(2, cue_temp, 'o-', ms=5, mfc='none',
                 c=[0, 1 - (1 / len(activity_all[0]) * i), 1 - (1 / len(activity_all[0]) * i)])
        corr_all.append(corr_temp)
        cue_all.append(cue_temp)
        pupil_all.append(pupil_temp)
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1, snap=False)
    y1 = np.mean(corr_all)
    y1_err = stats.sem(corr_all)
    plt.errorbar(1.1, y1, yerr=y1_err, c=[0, 0, 0], linewidth=2, marker='o', mfc=[0, 0, 0], mec=[0, 0, 0], ms=7,
                 mew=0, zorder=100)
    y2 = np.mean(cue_all)
    y2_err = stats.sem(cue_all)
    plt.errorbar(2.1, y2, yerr=y2_err, c=[0, 0, 0], linewidth=2, marker='o', mfc=[0, 0, 0], mec=[0, 0, 0], ms=7,
                 mew=0, zorder=100)
    y4 = np.mean(pupil_all)
    y4_err = stats.sem(pupil_all)
    plt.errorbar(3.1, y4, yerr=y4_err, c=[0, 0, 0], linewidth=2, marker='o', mfc=[0, 0, 0], mec=[0, 0, 0],
                 ms=7, mew=0, zorder=100)
    plt.ylabel('Correlation with reactivation probability (r)')
    plt.xlim(.5, 3.5)
    plt.ylim(-1, 1)
    plt.xticks([1, 2, 3], ['Cue correlation', 'Cue activity', 'Pupil size'])
    sns.despine()

    plt.subplot(4, 2, 2)
    maxlags = 20
    xcorr_correlation = np.zeros((len(activity_all[0]), (maxlags * 2) + 1))
    for i in range(0, len(activity_all[0])):
        smoothed_correlation = np.array(pd.DataFrame(activity_all[0][i]).rolling(int(fr * 2), min_periods=1,
                                                                                 center=True).mean())
        smoothed_reactivation_prob = np.array(pd.DataFrame(activity_all[1][i]).rolling(int(fr * 2), min_periods=1,
                                                                                       center=True).mean())
        smoothed_correlation = np.concatenate(smoothed_correlation, axis=0)
        smoothed_reactivation_prob = np.concatenate(smoothed_reactivation_prob, axis=0)
        temp_corr = []
        for j in range(-maxlags, maxlags + 1):
            shift_1 = smoothed_correlation[maxlags:len(smoothed_correlation) - maxlags]
            shift_2 = smoothed_reactivation_prob[maxlags + j:len(smoothed_reactivation_prob) - maxlags + j]
            temp_corr.append(np.corrcoef(shift_1, shift_2)[0][1])
        temp_corr /= np.max(np.abs(temp_corr))
        plt.plot(range(-maxlags, maxlags + 1), temp_corr, c=[0, 1 - (1 / len(activity_all[0]) * i),
                                                             1 - (1 / len(activity_all[0]) * i)], alpha=.5)
        xcorr_correlation[i, :] = temp_corr
    sns.despine()
    mean = xcorr_correlation.mean(axis=0)
    sem_plus = mean + stats.sem(xcorr_correlation, axis=0)
    sem_minus = mean - stats.sem(xcorr_correlation, axis=0)
    plt.plot(range(-maxlags, maxlags + 1), mean, '-o', c='k', linewidth=2, ms=4)
    plt.fill_between(range(-maxlags, maxlags + 1), sem_plus, sem_minus, alpha=0.1, color='k')
    plt.ylabel('Correlation between cue \n similarity and ITI reactivation')
    plt.xlabel('Shift (trial)')
    sns.despine()

    y_pred_binned_across_days = list(np.load(paths['base_path'] + paths['mouse'] +
                                             '/data_across_days/y_pred_binned.npy'))
    task_runs = int(session_data['task_runs'])
    dark_runs = int(session_data['dark_runs'])
    plt.subplot(4, 2, 4)
    plt.xlabel('Sum reactivation probability')
    plt.ylabel("Initial correlation between cues")
    corr_initial = []
    reac_prob = []
    for i in range(0, len(activity_all[0])):
        initial_corr = np.mean(activity_all[0][i][0:20])
        corr_initial.append(initial_corr)
        prob_reac = np.sum(activity_all[1][i])
        reac_prob.append(prob_reac)
    ax = sns.regplot(x=reac_prob, y=corr_initial, color=[0, .6, .6])
    [r, p] = stats.pearsonr(reac_prob, corr_initial)
    plt.text(.8, .9, 'r=' + str(round(r, 2)) + ', p=' + str(round(p, 5)), horizontalalignment='center',
             verticalalignment='center', transform=ax.transAxes)
    plt.xlim(np.min(reac_prob)-10, np.max(reac_prob)+10)
    sns.despine()

    plt.subplot(4, 2, 3)
    plt.xlabel('Sum reactivation probability')
    plt.ylabel("Δ Correlation between cues")
    corr_diff = []
    reac_prob = []
    for i in range(0, len(activity_all[0])):
        diff_corr = np.mean(activity_all[0][i][len(activity_all[0][i]) - 20:len(activity_all[0][i])]) - \
                    np.mean(activity_all[0][i][0:20])
        corr_diff.append(diff_corr)
        prob_reac = np.sum(activity_all[1][i])
        reac_prob.append(prob_reac)
    ax = sns.regplot(x=reac_prob, y=corr_diff, color=[0, .6, .6])
    [r, p] = stats.pearsonr(reac_prob, corr_diff)
    plt.text(.8, .9, 'r=' + str(round(r, 2)) + ', p=' + str(round(p, 5)), horizontalalignment='center',
             verticalalignment='center', transform=ax.transAxes)
    plt.xlim(np.min(reac_prob) - 10, np.max(reac_prob) + 10)
    sns.despine()

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=.3, hspace=None)
    plt.savefig(paths['base_path'] + paths['mouse'] + '/data_across_days/plots/' + paths['mouse'] + '_'
                + 'activity_reactivation.png', bbox_inches='tight', dpi=150)
    plt.close()


def iti_bias(paths):
    activity_all = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/activity.npy', allow_pickle=True)
    session_data = preprocess.load_data(paths)
    fr = session_data['framerate']
    cs_1_code = int(session_data['CS_1_code'])
    cs_2_code = int(session_data['CS_2_code'])

    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(10, 7))
    plt.subplot(2, 2, 1)

    x = [1, 2]
    cs_1_low = []
    cs_2_low = []
    cs_1_norm = []
    cs_2_norm = []
    for i in range(0, len(activity_all[0])):
        iti_cs_1_activity_low = np.array(activity_all[4][i])
        iti_cs_2_activity_low = np.array(activity_all[5][i])
        iti_cs_1_activity_norm = np.array(activity_all[6][i])
        iti_cs_2_activity_norm = np.array(activity_all[7][i])
        cue_type = list(np.concatenate(activity_all[8][i], axis=0))
        cs_1_trials = []
        cs_2_trials = []
        for j in range(0, len(cue_type)):
            if cue_type[j] == cs_1_code:
                cs_1_trials.append(j)
            if cue_type[j] == cs_2_code:
                cs_2_trials.append(j)

        iti_cs_1_cs_1_activity_low = np.nanmean(iti_cs_1_activity_low[cs_1_trials])
        iti_cs_2_cs_1_activity_low = np.nanmean(iti_cs_2_activity_low[cs_1_trials])
        iti_cs_2_cs_2_activity_low = np.nanmean(iti_cs_2_activity_low[cs_2_trials])
        iti_cs_1_cs_2_activity_low = np.nanmean(iti_cs_1_activity_low[cs_2_trials])
        cs_1_low_bias = (iti_cs_1_cs_1_activity_low - iti_cs_1_cs_2_activity_low) * fr
        cs_2_low_bias = (iti_cs_2_cs_2_activity_low - iti_cs_2_cs_1_activity_low) * fr
        cs_1_low.append(cs_1_low_bias[0][0])
        cs_2_low.append(cs_2_low_bias[0][0])

        iti_cs_1_cs_1_activity_norm = np.nanmean(iti_cs_1_activity_norm[cs_1_trials])
        iti_cs_2_cs_1_activity_norm = np.nanmean(iti_cs_2_activity_norm[cs_1_trials])
        iti_cs_2_cs_2_activity_norm = np.nanmean(iti_cs_2_activity_norm[cs_2_trials])
        iti_cs_1_cs_2_activity_norm = np.nanmean(iti_cs_1_activity_norm[cs_2_trials])
        cs_1_norm_bias = (iti_cs_1_cs_1_activity_norm - iti_cs_1_cs_2_activity_norm) * fr
        cs_2_norm_bias = (iti_cs_2_cs_2_activity_norm - iti_cs_2_cs_1_activity_norm) * fr
        cs_1_norm.append(cs_1_norm_bias[0][0])
        cs_2_norm.append(cs_2_norm_bias[0][0])

        plt.errorbar(x, [cs_1_low_bias, cs_2_low_bias], yerr=0, c='k', linewidth=.5, linestyle='--',
                     zorder=0, alpha=0.5, label='_nolegend_')
        plt.plot(x[0], cs_1_low_bias, 'o', color='k', markersize=5, alpha=1, zorder=50, label='_nolegend_')
        plt.plot(x[1], cs_2_low_bias, 'o', color='k', markersize=5, alpha=1, zorder=50, label='_nolegend_')

        plt.errorbar(x, [cs_1_norm_bias, cs_2_norm_bias], yerr=0, c='b', linewidth=.5, linestyle='--', zorder=0,
                     alpha=0.5, label='_nolegend_')
        plt.plot(x[0], cs_1_norm_bias, 'o', color='b', markersize=5, alpha=1, zorder=50, label='_nolegend_')
        plt.plot(x[1], cs_2_norm_bias, 'o', color='b', markersize=5, alpha=1, zorder=50, label='_nolegend_')

    y1 = np.mean(cs_1_low)
    y1_err = stats.sem(cs_1_low)
    plt.errorbar(.9, y1, yerr=y1_err, c=[0, 0, 0], linewidth=2, marker='o', mfc=[0, 0, 0], mec=[0, 0, 0], ms=10,
                 mew=0, zorder=100, label='_nolegend_')
    y2 = np.mean(cs_2_low)
    y2_err = stats.sem(cs_2_low)
    plt.errorbar(2.1, y2, yerr=y2_err, c=[0, 0, 0], linewidth=2, marker='o', mfc=[0, 0, 0], mec=[0, 0, 0], ms=10,
                 mew=0, zorder=100, label='_nolegend_')
    y4 = np.mean(cs_1_norm)
    y4_err = stats.sem(cs_1_norm)
    plt.errorbar(.9, y4, yerr=y4_err, c='b', linewidth=2, marker='o', mfc='b', mec=[0, 0, 0],
                 ms=10, mew=0, zorder=100, label='_nolegend_')
    y3 = np.mean(cs_2_norm)
    y3_err = stats.sem(cs_2_norm)
    plt.errorbar(2.1, y3, yerr=y3_err, c='b', linewidth=2, marker='o', mfc='b', mec=[0, 0, 0],
                 ms=10, mew=0, zorder=100, label='_nolegend_')
    plt.ylabel('ITI activity bias ($\mathregular{Ca^{2+}}$transient $\mathregular{s^{-1}}$)')
    plt.xlim(.5, 2.5)
    plt.xticks([1, 2], ['Following Cue 1', 'Following Cue 2'])
    plt.gca().get_xticklabels()[1].set_color('salmon')
    plt.gca().get_xticklabels()[0].set_color('mediumseagreen')

    label_1 = mlines.Line2D([], [], color='b', marker='o', linestyle='None', markersize=5,
                            label='Classified synchronous')
    label_2 = mlines.Line2D([], [], color='k', marker='o', linestyle='None', markersize=5, label='Non-synchronous')
    plt.legend(handles=[label_2, label_1], frameon=False)

    sns.despine()
    plt.savefig(paths['base_path'] + paths['mouse'] + '/data_across_days/plots/' + paths['mouse'] + '_'
                + 'ITI_bias.png', bbox_inches='tight', dpi=500)
    plt.close()


def trial_history(num_prev, paths):
    """
    plots trial history
    :param num_prev: number of previous trials
    :param paths: path to data
    :return:
    """
    trial_hist = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/trial_history_' +
                            str(num_prev) + '.npy', allow_pickle=True)

    pupil_history = [trial_hist[2][np.isnan(trial_hist)[2] == 0],
                     trial_hist[3][np.isnan(trial_hist)[3] == 0]]
    ypred_history = [trial_hist[0][np.isnan(trial_hist)[2] == 0],
                     trial_hist[1][np.isnan(trial_hist)[3] == 0]]
    activity_history = [trial_hist[4][np.isnan(trial_hist)[2] == 0],
                        trial_hist[5][np.isnan(trial_hist)[3] == 0]]

    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    fig = plt.figure(figsize=(10, 7))
    plt.subplot(2, 2, 1)
    x = [1, 2]
    for i in range(0, len(ypred_history[0])):
        plt.errorbar(x, [ypred_history[0][i], ypred_history[1][i]], yerr=0, c='k', linewidth=.5, linestyle='--',
                     zorder=0, alpha=0.5)
        plt.plot(x[0], ypred_history[0][i], 'o', color='k', markersize=5, alpha=0.2, mew=0, zorder=50)
        plt.plot(x[1], ypred_history[1][i], 'o', color='b', markersize=5, alpha=0.2, mew=0, zorder=50)
    y0 = np.mean(ypred_history[0])
    y0_err = stats.sem(ypred_history[0])
    y1 = np.mean(ypred_history[1])
    y1_err = stats.sem(ypred_history[1])
    plt.errorbar(x, [y0, y1], yerr=0, c='k', linewidth=1.5, linestyle='--', zorder=0)
    plt.errorbar(x[0], y0, yerr=y0_err, c=[0, 0, 0], linewidth=2, marker='o', mfc=[0, 0, 0], mec=[0, 0, 0], ms=7,
                 mew=0, zorder=100)
    plt.errorbar(x[1], y1, yerr=y1_err, c=[0, 0, 1], linewidth=2, marker='o', mfc=[0, 0, 1], mec=[0, 0, 1], ms=7,
                 mew=0, zorder=100)
    plt.xlim((.5, 2.5))
    plt.xticks([1, 2], ['Same cue (' + str(num_prev) + ')', 'Different cue (' + str(num_prev) + ')'])
    plt.gca().get_xticklabels()[1].set_color('blue')
    plt.gca().get_xticklabels()[0].set_color('k')
    plt.ylabel('Reactivation probability (/s)')
    sns.despine()
    plt.subplot(2, 2, 2)
    x = [1, 2]
    for i in range(0, len(pupil_history[0])):
        plt.errorbar(x, [pupil_history[0][i], pupil_history[1][i]], yerr=0, c='k', linewidth=.5, linestyle='--',
                     zorder=0, alpha=0.5)
        plt.plot(x[0], pupil_history[0][i], 'o', color='k', markersize=5, alpha=0.2, mew=0, zorder=50)
        plt.plot(x[1], pupil_history[1][i], 'o', color='b', markersize=5, alpha=0.2, mew=0, zorder=50)
    y0 = np.mean(pupil_history[0])
    y0_err = stats.sem(pupil_history[0])
    y1 = np.mean(pupil_history[1])
    y1_err = stats.sem(pupil_history[1])
    plt.errorbar(x, [y0, y1], yerr=0, c='k', linewidth=1.5, linestyle='--', zorder=0)
    plt.errorbar(x[0], y0, yerr=y0_err, c=[0, 0, 0], linewidth=2, marker='o', mfc=[0, 0, 0], mec=[0, 0, 0], ms=7,
                 mew=0, zorder=100)
    plt.errorbar(x[1], y1, yerr=y1_err, c=[0, 0, 1], linewidth=2, marker='o', mfc=[0, 0, 1], mec=[0, 0, 1], ms=7,
                 mew=0, zorder=100)
    plt.xlim((.5, 2.5))
    plt.xticks([1, 2], ['Same cue (' + str(num_prev) + ')', 'Different cue (' + str(num_prev) + ')'])
    plt.gca().get_xticklabels()[1].set_color('blue')
    plt.gca().get_xticklabels()[0].set_color('k')
    plt.ylabel('Pupil area (ΔA/A)')
    sns.despine()
    plt.subplot(2, 2, 3)
    x = [1, 2]
    for i in range(0, len(activity_history[0])):
        plt.errorbar(x, [activity_history[0][i], activity_history[1][i]], yerr=0, c='k', linewidth=.5, linestyle='--',
                     zorder=0, alpha=0.5)
        plt.plot(x[0], activity_history[0][i], 'o', color='k', markersize=5, alpha=0.2, mew=0, zorder=50)
        plt.plot(x[1], activity_history[1][i], 'o', color='b', markersize=5, alpha=0.2, mew=0, zorder=50)
    y0 = np.mean(activity_history[0])
    y0_err = stats.sem(activity_history[0])
    y1 = np.mean(activity_history[1])
    y1_err = stats.sem(activity_history[1])
    plt.errorbar(x, [y0, y1], yerr=0, c='k', linewidth=1.5, linestyle='--', zorder=0)
    plt.errorbar(x[0], y0, yerr=y0_err, c=[0, 0, 0], linewidth=2, marker='o', mfc=[0, 0, 0], mec=[0, 0, 0], ms=7,
                 mew=0, zorder=100)
    plt.errorbar(x[1], y1, yerr=y1_err, c=[0, 0, 1], linewidth=2, marker='o', mfc=[0, 0, 1], mec=[0, 0, 1], ms=7,
                 mew=0, zorder=100)
    plt.xlim((.5, 2.5))
    plt.xticks([1, 2], ['Same cue (' + str(num_prev) + ')', 'Different cue (' + str(num_prev) + ')'])
    plt.gca().get_xticklabels()[1].set_color('blue')
    plt.gca().get_xticklabels()[0].set_color('k')
    plt.ylabel('Cue activity')
    sns.despine()
    plt.subplot(2, 2, 4)
    sns.regplot(x=pupil_history[1] + pupil_history[0], y=ypred_history[1] - ypred_history[0], color="k")
    # print(stats.pearsonr((pupil_history[1] - pupil_history[0]) / pupil_history[0], ypred_history[1]/ypred_history[0]))
    plt.ylabel('Reactivation history bias')
    plt.xlabel('Pupil area (ΔA/A)')
    sns.despine()
    plt.savefig(paths['base_path'] + paths['mouse'] + '/data_across_days/plots/' + paths['mouse'] + '_'
                + 'trial_history_' + str(num_prev) + '.png', bbox_inches='tight', dpi=150)
    plt.close(fig)


def trial_effect(paths):
    trial_effect_diff = np.load(paths['base_path'] + paths['mouse'] +
                                      '/data_across_days/trial_effect_diff.npy', allow_pickle=True)
    trial_effect_same = np.load(paths['base_path'] + paths['mouse'] +
                                '/data_across_days/trial_effect_same.npy', allow_pickle=True)
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(15, 7))
    trial_effect_helper(paths, trial_effect_diff, trial_effect_same, 1,
                        'Reactivation probability ($\mathregular{s^{-1}}$)')
    trial_effect_helper(paths, trial_effect_diff, trial_effect_same, 3,
                        'Reactivation probability ($\mathregular{s^{-1}}$)')
    plt.savefig(paths['base_path'] + paths['mouse'] + '/data_across_days/plots/' + paths['mouse'] + '_'
                + 'trial_effect_rate.png', bbox_inches='tight', dpi=500)
    plt.close()

    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(15, 7))
    trial_effect_helper(paths, trial_effect_diff, trial_effect_same, 1, 'Reactivation cue bias')
    trial_effect_helper(paths, trial_effect_diff, trial_effect_same, 3, 'Reactivation cue bias')
    plt.savefig(paths['base_path'] + paths['mouse'] + '/data_across_days/plots/' + paths['mouse'] + '_'
                + 'trial_effect_bias.png', bbox_inches='tight', dpi=500)
    plt.close()


def trial_effect_helper(paths, trial_effect_diff, trial_effect_same, idx, y_label):
    session_data = preprocess.load_data(paths)
    fr = session_data['framerate']
    factor = 5
    duration = int(fr * 8) + 1
    if y_label == 'Reactivation probability ($\mathregular{s^{-1}}$)':
        idx_1 = 0
        idx_2 = 1
    if y_label == 'Reactivation cue bias':
        idx_1 = 2
        idx_2 = 3
    if idx == 1:
        trial_effect_all = trial_effect_diff
    if idx == 3:
        trial_effect_all = trial_effect_same
    mean_prev = np.empty((len(trial_effect_all[0]), factor)) * np.nan
    mean_curr = np.empty((len(trial_effect_all[0]), factor)) * np.nan
    plt.subplot(2, 2, idx)
    for i in range(0, len(trial_effect_all[0])):
        plt.plot(trial_effect_all[4][i], trial_effect_all[idx_1][i], '-o',
                 c=[.5 * (.97 - (1 / (len(trial_effect_all[0]) + 1) * i)),
                 .9 * (.97 - (1 / (len(trial_effect_all[0]) + 1) * i)),
                 .9 * (.97 - (1 / (len(trial_effect_all[0]) + 1) * i))], ms=5)
        plt.plot(trial_effect_all[5][i], trial_effect_all[idx_2][i], '-o',
                 c=[.5 * (.97 - (1 / (len(trial_effect_all[0]) + 1) * i)),
                    .9 * (.97 - (1 / (len(trial_effect_all[0]) + 1) * i)),
                    .9 * (.97 - (1 / (len(trial_effect_all[0]) + 1) * i))], ms=5)
        plt.plot([trial_effect_all[4][i][factor - 1], trial_effect_all[5][i][0]],
                 [trial_effect_all[idx_1][i][factor - 1], trial_effect_all[idx_2][i][0]], '--', c=[.3, .3, .3], zorder=1)
        mean_prev[i, :] = trial_effect_all[idx_1][i]
        mean_curr[i, :] = trial_effect_all[idx_2][i]
    plt.axvspan(0, int(fr * 1), alpha=.75, color='mediumseagreen')
    plt.axvspan(int(fr * 1), int(fr * 2), alpha=.75, color='salmon')
    plt.axvspan(int(fr * 2), int(fr * 8), alpha=.2, color='gray')
    time_extra = trial_effect_all[5][0][0] - duration - .5*108
    if idx == 1:
        plt.axvspan(time_extra, int(fr * 1) + time_extra, alpha=.75, color='salmon', zorder=0)
        plt.axvspan(int(fr * 1) + time_extra, int(fr * 2) + time_extra, alpha=.75, color='mediumseagreen', zorder=0)
    if idx == 3:
        plt.axvspan(time_extra, int(fr * 1) + time_extra, alpha=.75, color='mediumseagreen', zorder=0)
        plt.axvspan(int(fr * 1) + time_extra, int(fr * 2) + time_extra, alpha=.75, color='salmon', zorder=0)
    plt.axvspan(int(fr * 2) + time_extra, int(fr * 8) + time_extra, alpha=.2, color='gray', zorder=0)
    plt.ylabel(y_label)
    plt.xlabel('Time relative to first cue onset (s)')
    plt.xticks([int(fr * 0), int(fr * 10), int(fr * 20), int(fr * 30), int(fr * 40), int(fr * 50), int(fr * 60),
                int(fr * 70), int(fr * 80), int(fr * 90), int(fr * 100), int(fr * 110), int(fr * 120)],
               ['0', '10', '20', '30', '40', '50', '60', '70', '80', '90', '100', '110', '120'])
    plt.xlim((-int(fr * 2), int(fr * 61 * 2)))
    sns.despine()

    plt.subplot(2, 2, idx+1)
    mean_prev_all = np.nanmean(mean_prev, axis=0)
    sem_plus = mean_prev_all + stats.sem(mean_prev, axis=0, nan_policy='omit')
    sem_minus = mean_prev_all - stats.sem(mean_prev, axis=0, nan_policy='omit')
    plt.plot(trial_effect_all[4][0], mean_prev_all, '-o', c='cadetblue', linewidth=2, ms=7)
    plt.fill_between(trial_effect_all[4][0], sem_plus, sem_minus, alpha=0.1, color='cadetblue')
    mean_curr_all = np.nanmean(mean_curr, axis=0)
    sem_plus = mean_curr_all + stats.sem(mean_curr, axis=0, nan_policy='omit')
    sem_minus = mean_curr_all - stats.sem(mean_curr, axis=0, nan_policy='omit')
    plt.plot(trial_effect_all[5][0], mean_curr_all, '-o', c='cadetblue', linewidth=2, ms=7)
    plt.fill_between(trial_effect_all[5][0], sem_plus, sem_minus, alpha=0.1, color='cadetblue')
    plt.plot([trial_effect_all[4][0][factor - 1], trial_effect_all[5][0][0]],
             [mean_prev_all[factor - 1], mean_curr_all[0]], '--', c=[.3, .3, .3], zorder=1)
    plt.axvspan(0, int(fr * 1), alpha=.75, color='mediumseagreen')
    plt.axvspan(int(fr * 1), int(fr * 2), alpha=.75, color='salmon')
    plt.axvspan(int(fr * 2), int(fr * 8), alpha=.2, color='gray')
    time_extra = trial_effect_all[5][0][0] - duration - .5 * 108
    if idx == 1:
        plt.axvspan(time_extra, int(fr * 1) + time_extra, alpha=.75, color='salmon', zorder=0)
        plt.axvspan(int(fr * 1) + time_extra, int(fr * 2) + time_extra, alpha=.75, color='mediumseagreen', zorder=0)
    if idx == 3:
        plt.axvspan(time_extra, int(fr * 1) + time_extra, alpha=.75, color='mediumseagreen', zorder=0)
        plt.axvspan(int(fr * 1) + time_extra, int(fr * 2) + time_extra, alpha=.75, color='salmon', zorder=0)
    plt.axvspan(int(fr * 2) + time_extra, int(fr * 8) + time_extra, alpha=.2, color='gray', zorder=0)
    plt.ylabel(y_label)
    plt.xlabel('Time relative to first cue onset (s)')
    plt.xticks([int(fr * 0), int(fr * 10), int(fr * 20), int(fr * 30), int(fr * 40), int(fr * 50), int(fr * 60),
                int(fr * 70), int(fr * 80), int(fr * 90), int(fr * 100), int(fr * 110), int(fr * 120)],
               ['0', '10', '20', '30', '40', '50', '60', '70', '80', '90', '100', '110', '120'])
    plt.xlim((-int(fr * 2), int(fr * 61 * 2)))
    sns.despine()


def reactivation_length(paths):
    length_all = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/reactivation_length.npy',
                        allow_pickle=True)
    session_data = preprocess.load_data(paths)
    fr = session_data['framerate']
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(10, 7))
    plt.subplot(2, 2, 1)
    x = [1, 2]
    length_control = []
    length_cue = []
    for i in range(0, len(length_all[0])):
        control = length_all[0][i] / fr[0]
        cue = length_all[1][i] / fr[0]
        plt.plot(x[0], control, 'o', c='k', markersize=6, zorder=100)
        plt.plot(x[1], cue, 'o', c='b', markersize=6, zorder=100)
        plt.errorbar(x, [control, cue], yerr=0, c=[.5, .5, .5], linewidth=1, linestyle='--')
        length_control.append(control)
        length_cue.append(cue)
    plt.xlim((.5, 2.5))
    plt.xticks([1, 2], ['Pre-cue', 'During cue'])
    plt.gca().get_xticklabels()[1].set_color('b')
    plt.gca().get_xticklabels()[0].set_color('k')
    plt.ylabel('Reactivation duration (s)')
    y1 = np.mean(length_control)
    y2 = np.mean(length_cue)
    y1_err = stats.sem(length_control)
    y2_err = stats.sem(length_cue)
    plt.errorbar(x[0]-.1, y1, yerr=y1_err, c='k', linewidth=2, marker='o', mfc='k', mec='k',
                 ms=10, mew=0)
    plt.errorbar(x[1]+.1, y2, yerr=y2_err, c='b', linewidth=2, marker='o', mfc='b', mec='black', ms=10, mew=0)
    sns.despine()
    plt.savefig(paths['base_path'] + paths['mouse'] + '/data_across_days/plots/' + paths['mouse'] + '_'
                + 'reactivation_length.png', bbox_inches='tight', dpi=150)
    plt.close()











