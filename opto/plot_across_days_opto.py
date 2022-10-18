from scipy import stats
import preprocess_opto
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


def mean_cue_selectivity(paths):
    cs_pref = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/cs_pref.npy', allow_pickle=True)
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(10, 7))
    mean_cue_selectivity_helper(cs_pref, 1, 'Reactivation cue bias (odds)')
    mean_cue_selectivity_helper(cs_pref, 3, 'Reactivation cue bias')
    plt.subplots_adjust(wspace=.3)
    plt.savefig(paths['base_path'] + paths['mouse'] + '/data_across_days/plots/' + paths['mouse'] + '_'
                + 'cue_bias.png', bbox_inches='tight', dpi=150)
    plt.close()


def mean_cue_selectivity_helper(cs_pref, idx, y_label):
    plt.subplot(2, 2, idx)
    x = [1, 2, 3, 4]
    cs_1_bias = []
    cs_2_bias = []
    cs_1_bias_opto = []
    cs_2_bias_opto = []
    for i in range(0, len(cs_pref[0])):
        if y_label == 'Reactivation cue bias (odds)':
            cs_1 = cs_pref[0][i][0] / cs_pref[0][i][1]
            cs_2 = cs_pref[1][i][1] / cs_pref[1][i][0]
            cs_1_opto = cs_pref[2][i][0] / cs_pref[2][i][1]
            cs_2_opto = cs_pref[3][i][1] / cs_pref[3][i][0]
            plt.plot(x[0], cs_1, 'o', c='green', markersize=5, zorder=100 / (i + 1))
            plt.plot(x[1], cs_1_opto, 'o', c='lime', markersize=5, zorder=100 / (i + 1))
            plt.errorbar(x[0:2], [cs_1, cs_1_opto], yerr=0, c=[.5, .5, .5], linewidth=1, linestyle='--')
            plt.plot(x[2], cs_2, 'o', c='firebrick', markersize=5, zorder=100 / (i + 1))
            plt.plot(x[3], cs_2_opto, 'o', c='hotpink', markersize=5, zorder=100 / (i + 1))
            plt.errorbar(x[2:4], [cs_2, cs_2_opto], yerr=0, c=[.5, .5, .5], linewidth=1, linestyle='--')
            cs_1_bias.append(cs_1)
            cs_2_bias.append(cs_2)
            cs_1_bias_opto.append(cs_1_opto)
            cs_2_bias_opto.append(cs_2_opto)
        if y_label == 'Reactivation cue bias':
            cs_1 = (cs_pref[0][i][0] - cs_pref[0][i][1]) / (cs_pref[0][i][0] + cs_pref[0][i][1])
            cs_2 = (cs_pref[1][i][1] - cs_pref[1][i][0]) / (cs_pref[1][i][1] + cs_pref[1][i][0])
            cs_1_opto = (cs_pref[2][i][0] - cs_pref[2][i][1]) / (cs_pref[2][i][0] + cs_pref[2][i][1])
            cs_2_opto = (cs_pref[3][i][1] - cs_pref[3][i][0]) / (cs_pref[3][i][1] + cs_pref[3][i][0])
            plt.plot(x[0], cs_1, 'o', c='green', markersize=5, zorder=100 / (i + 1))
            plt.plot(x[1], cs_1_opto, 'o', c='lime', markersize=5, zorder=100 / (i + 1))
            plt.errorbar(x[0:2], [cs_1, cs_1_opto], yerr=0, c=[.5, .5, .5], linewidth=1, linestyle='--')
            plt.plot(x[2], cs_2, 'o', c='firebrick', markersize=5, zorder=100 / (i + 1))
            plt.plot(x[3], cs_2_opto, 'o', c='hotpink', markersize=5, zorder=100 / (i + 1))
            plt.errorbar(x[2:4], [cs_2, cs_2_opto], yerr=0, c=[.5, .5, .5], linewidth=1, linestyle='--')
            cs_1_bias.append(cs_1)
            cs_2_bias.append(cs_2)
            cs_1_bias_opto.append(cs_1_opto)
            cs_2_bias_opto.append(cs_2_opto)
    plt.xlim((.5, 4.5))
    plt.xticks([1, 2, 3, 4], ['F. Cue 1', 'F. Cue 1 opto', 'F. Cue 2', 'F Cue 2 opto'])
    plt.gca().get_xticklabels()[0].set_color('green')
    plt.gca().get_xticklabels()[1].set_color('lime')
    plt.gca().get_xticklabels()[2].set_color('firebrick')
    plt.gca().get_xticklabels()[3].set_color('hotpink')
    plt.ylabel(y_label)
    if y_label == 'Reactivation cue bias (odds)':
        plt.axhline(y=1, color='black', linestyle='--', linewidth=2, snap=False)
        plt.ylim(0, np.max([cs_1_bias, cs_2_bias]) + 1)
    if y_label == 'Reactivation cue bias':
        plt.axhline(y=0, color='black', linestyle='--', linewidth=2, snap=False)
        plt.ylim(-.5, .5)
    y1 = np.mean(cs_1_bias)
    y2 = np.mean(cs_2_bias)
    y1_err = stats.sem(cs_1_bias)
    y2_err = stats.sem(cs_2_bias)
    y1_opto = np.mean(cs_1_bias_opto)
    y2_opto = np.mean(cs_2_bias_opto)
    y1_err_opto = stats.sem(cs_1_bias_opto)
    y2_err_opto = stats.sem(cs_2_bias_opto)
    plt.errorbar(x[0]-.2, y1, yerr=y1_err, c='green', linewidth=2, marker='o', mfc='green', mec='k',
                 ms=8, mew=0)
    plt.errorbar(x[2]-.2, y2, yerr=y2_err, c='firebrick', linewidth=2, marker='o', mfc='firebrick', mec='black',
                 ms=8, mew=0)
    plt.errorbar(x[1] + .2, y1_opto, yerr=y1_err_opto, c='lime', linewidth=2, marker='o', mfc='lime', mec='k',
                 ms=8, mew=0)
    plt.errorbar(x[3] + .2, y2_opto, yerr=y2_err_opto, c='hotpink', linewidth=2, marker='o', mfc='hotpink', mec='black',
                 ms=8, mew=0)
    sns.despine()


def within_day(paths):
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(10, 7))
    plt.subplot(2, 2, 1)
    y_pred_binned_across_days = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/y_pred_binned.npy',
                                        allow_pickle=True)
    binned_reactivation = np.zeros((len(y_pred_binned_across_days[0]), len(y_pred_binned_across_days[0][0])))
    binned_reactivation_opto = np.zeros((len(y_pred_binned_across_days[0]), len(y_pred_binned_across_days[0][0])))
    for i in range(0, len(y_pred_binned_across_days[0])):
        plt.plot(y_pred_binned_across_days[2][0][0:2], y_pred_binned_across_days[0][i][0:2], '--ok')
        plt.plot(y_pred_binned_across_days[2][0][1:len(y_pred_binned_across_days[2][0])],
                 y_pred_binned_across_days[0][i][1:len(y_pred_binned_across_days[0][i])], '-ok')
        plt.plot(y_pred_binned_across_days[2][0][0:2], y_pred_binned_across_days[1][i][0:2], '--or')
        plt.plot(y_pred_binned_across_days[2][0][1:len(y_pred_binned_across_days[2][0])],
                 y_pred_binned_across_days[1][i][1:len(y_pred_binned_across_days[1][i])], '-or')
        binned_reactivation[i, :] = y_pred_binned_across_days[0][i]
        binned_reactivation_opto[i, :] = y_pred_binned_across_days[1][i]
    plt.axvspan(-.5, 0, alpha=.25, color='gray')
    plt.ylabel('Reactivation probability ($\mathregular{s^{-1}}$)')
    plt.xlabel('Time from first cue presentation (hours)')
    sns.despine()
    plt.subplot(2, 2, 2)
    mean = binned_reactivation.mean(axis=0)
    sem_plus = mean + stats.sem(binned_reactivation, axis=0)
    sem_minus = mean - stats.sem(binned_reactivation, axis=0)
    mean_opto = binned_reactivation_opto.mean(axis=0)
    sem_plus_opto = mean_opto + stats.sem(binned_reactivation_opto, axis=0)
    sem_minus_opto = mean_opto - stats.sem(binned_reactivation_opto, axis=0)
    plt.plot(y_pred_binned_across_days[2][0][0:2], mean[0:2], '--ok', linewidth=2, ms=7)
    plt.plot(y_pred_binned_across_days[2][0][1:len(y_pred_binned_across_days[2][0])], mean[1:len(mean)], '-ok',
             linewidth=2, ms=7)
    plt.plot(y_pred_binned_across_days[2][0][0:2], mean_opto[0:2], '--or', linewidth=2, ms=7)
    plt.plot(y_pred_binned_across_days[2][0][1:len(y_pred_binned_across_days[2][0])], mean_opto[1:len(mean_opto)],
             '-or', linewidth=2, ms=7)
    plt.fill_between(y_pred_binned_across_days[2][0][0:2], sem_plus[0:2], sem_minus[0:2], alpha=0.1, color='k')
    plt.fill_between(y_pred_binned_across_days[2][0][1:len(y_pred_binned_across_days[2][0])], sem_plus[1:len(sem_plus)],
                     sem_minus[1:len(sem_minus)], alpha=0.1, color='k')
    plt.fill_between(y_pred_binned_across_days[2][0][0:2], sem_plus_opto[0:2], sem_minus_opto[0:2], alpha=0.1,
                     color='r')
    plt.fill_between(y_pred_binned_across_days[2][0][1:len(y_pred_binned_across_days[2][0])],
                     sem_plus_opto[1:len(sem_plus_opto)],
                     sem_minus_opto[1:len(sem_minus_opto)], alpha=0.1, color='r')
    plt.axvspan(-.5, 0, alpha=.25, color='gray')
    plt.ylabel('Reactivation probability ($\mathregular{s^{-1}}$)')
    plt.xlabel('Time from first cue presentation (hours)')
    sns.despine()
    plt.savefig(paths['base_path'] + paths['mouse'] + '/data_across_days/plots/' + paths['mouse'] + '_' +
                'mean_within_day.png', bbox_inches='tight', dpi=150)
    plt.close()


def trial_effect(paths):
    trial_effect_opto = np.load(paths['base_path'] + paths['mouse'] +
                                      '/data_across_days/trial_effect_opto.npy', allow_pickle=True)
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(15, 7))
    plt.subplot(2, 2, 1)
    trial_effect_helper(paths, trial_effect_opto, 1, 'opto', 'Reactivation probability ($\mathregular{s^{-1}}$)')
    trial_effect_helper(paths, trial_effect_opto, 3, 'opto', 'Reactivation cue bias')
    plt.savefig(paths['base_path'] + paths['mouse'] + '/data_across_days/plots/' + paths['mouse'] + '_'
                + 'trial_effect_opto.png', bbox_inches='tight', dpi=500)
    plt.close()

    trial_effect_normal = np.load(paths['base_path'] + paths['mouse'] +
                                '/data_across_days/trial_effect_normal.npy', allow_pickle=True)
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(15, 7))
    plt.subplot(2, 2, 1)
    trial_effect_helper(paths, trial_effect_normal, 1, 'normal', 'Reactivation probability ($\mathregular{s^{-1}}$)')
    trial_effect_helper(paths, trial_effect_normal, 3, 'normal', 'Reactivation cue bias')
    plt.savefig(paths['base_path'] + paths['mouse'] + '/data_across_days/plots/' + paths['mouse'] + '_'
                + 'trial_effect_normal.png', bbox_inches='tight', dpi=500)
    plt.close()


def trial_effect_helper(paths, trial_effect_all, idx, type, y_label):
    behavior = preprocess_opto.process_behavior(paths)
    fr = behavior['framerate']
    factor = 5
    duration = int(fr * 8) + 1
    if y_label == 'Reactivation probability ($\mathregular{s^{-1}}$)':
        idx_1 = 0
        idx_2 = 1
    if y_label == 'Reactivation cue bias':
        idx_1 = 2
        idx_2 = 3
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
    plt.axvspan(0, int(fr * 1), alpha=.75, color='green')
    plt.axvspan(int(fr * 1), int(fr * 2), alpha=.75, color='firebrick')
    plt.axvspan(int(fr * 2), int(fr * 8), alpha=.2, color='gray')
    time_extra = trial_effect_all[5][0][0] - duration - .5*108
    if type == 'opto':
        plt.axvspan(time_extra, int(fr * 1) + time_extra, alpha=.75, color='lime', zorder=0)
        plt.axvspan(int(fr * 1) + time_extra, int(fr * 2) + time_extra, alpha=.75, color='hotpink', zorder=0)
    if type == 'normal':
        plt.axvspan(time_extra, int(fr * 1) + time_extra, alpha=.75, color='green', zorder=0)
        plt.axvspan(int(fr * 1) + time_extra, int(fr * 2) + time_extra, alpha=.75, color='firebrick', zorder=0)
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
    plt.axvspan(0, int(fr * 1), alpha=.75, color='green')
    plt.axvspan(int(fr * 1), int(fr * 2), alpha=.75, color='firebrick')
    plt.axvspan(int(fr * 2), int(fr * 8), alpha=.2, color='gray')
    time_extra = trial_effect_all[5][0][0] - duration - .5 * 108
    if type == 'opto':
        plt.axvspan(time_extra, int(fr * 1) + time_extra, alpha=.75, color='lime', zorder=0)
        plt.axvspan(int(fr * 1) + time_extra, int(fr * 2) + time_extra, alpha=.75, color='hotpink', zorder=0)
    if type == 'normal':
        plt.axvspan(time_extra, int(fr * 1) + time_extra, alpha=.75, color='green', zorder=0)
        plt.axvspan(int(fr * 1) + time_extra, int(fr * 2) + time_extra, alpha=.75, color='firebrick', zorder=0)
    plt.axvspan(int(fr * 2) + time_extra, int(fr * 8) + time_extra, alpha=.2, color='gray', zorder=0)
    plt.ylabel(y_label)
    plt.xlabel('Time relative to first cue onset (s)')
    plt.xticks([int(fr * 0), int(fr * 10), int(fr * 20), int(fr * 30), int(fr * 40), int(fr * 50), int(fr * 60),
                int(fr * 70), int(fr * 80), int(fr * 90), int(fr * 100), int(fr * 110), int(fr * 120)],
               ['0', '10', '20', '30', '40', '50', '60', '70', '80', '90', '100', '110', '120'])
    plt.xlim((-int(fr * 2), int(fr * 61 * 2)))
    if y_label == 'Reactivation probability ($\mathregular{s^{-1}}$)':
        plt.ylim((.04, .13))
    if y_label == 'Reactivation cue bias':
        plt.ylim((-.5, .6))
    sns.despine()


def activity_control(paths):
    behavior = preprocess_opto.process_behavior(paths)
    fr = behavior['framerate']
    activity_control_all = np.load(paths['base_path'] + paths['mouse'] +
                                '/data_across_days/activity_control.npy', allow_pickle=True)
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(15, 7))
    plt.subplot(2, 2, 1)
    activity_control_norm = np.zeros((len(activity_control_all[0]), len(activity_control_all[0][0])))
    activity_control_opto = np.zeros((len(activity_control_all[0]), len(activity_control_all[0][0])))
    for i in range(0, len(activity_control_all[0])):
        activity_control_norm[i, :] = activity_control_all[0][i]
        activity_control_opto[i, :] = activity_control_all[1][i]
        plt.plot(activity_control_all[0][i], c='k')
        plt.plot(activity_control_all[1][i], c='r')
    # mean = activity_control_norm.mean(axis=0)
    # sem_plus = mean + stats.sem(activity_control_norm, axis=0)
    # sem_minus = mean - stats.sem(activity_control_norm, axis=0)
    # mean_opto = activity_control_opto.mean(axis=0)
    # sem_plus_opto = mean_opto + stats.sem(activity_control_opto, axis=0)
    # sem_minus_opto = mean_opto - stats.sem(activity_control_opto, axis=0)
    # plt.plot(mean, linewidth=2)
    # plt.fill_between(range(0, len(mean)), sem_plus, sem_minus, alpha=0.1, color='k')
    # plt.plot(mean_opto, linewidth=2)
    # plt.fill_between(range(0, len(mean_opto)), sem_plus_opto, sem_minus_opto, alpha=0.1, color='k')
    plt.xticks([int(fr * 3), int(fr * 5), int(fr * 15), int(fr * 25), int(fr * 35), int(fr * 45), int(fr * 55),
                int(fr * 65)], ['', '0', '10', '20', '30', '40', '50', '60'])
    plt.xlim((int(fr * 3), int(fr * 62)))
    plt.ylabel('Mean activity ($\mathregular{Ca^{2+}}$transient $\mathregular{s^{-1}}$)')
    plt.xlabel('Time relative to cue onset (s)')
    plt.axvspan(int(fr * 4), int(fr * 8), alpha=.2, color='r', zorder=0)
    plt.axvspan(int(fr * 8), int(fr * 13), alpha=.2, color='gray', zorder=0)
    sns.despine()
    plt.savefig(paths['base_path'] + paths['mouse'] + '/data_across_days/plots/' + paths['mouse'] + '_'
                + 'activity_control.png', bbox_inches='tight', dpi=500)
    plt.close()
    
    
def activity_learning(paths):
    activity_all = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/activity.npy', allow_pickle=True)
    behavior = preprocess_opto.process_behavior(paths)
    fr = behavior['framerate']
    num_trials = len(behavior['onsets'])

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
    corr_all = np.empty((len(activity_all[2]), num_trials)) * np.nan
    corr_all_norm = np.empty((len(activity_all[2]), num_trials)) * np.nan
    for i in range(0, len(activity_all[2])):
        smoothed = np.array(pd.DataFrame(activity_all[2][i]).rolling(int(fr * 2), min_periods=1, center=True).mean())
        x = range(0, len(smoothed))
        plt.plot(x, smoothed*fr, c=[0, 1 - (1 / len(activity_all[2]) * i), 1 - (1 / len(activity_all[2]) * i)])
        corr_all[i, 0:len(smoothed)] = np.concatenate(smoothed, axis=0)
        corr_all_norm[i, 0:len(smoothed)] = np.concatenate(smoothed, axis=0) - smoothed[0]
    sns.despine()

    plt.subplot(4, 2, 6)
    plt.xlabel('Day')
    plt.ylabel("Initial cue activity")
    cue_initial = []
    day = []
    for i in range(0, len(activity_all[2])):
        initial_cue = np.mean(activity_all[2][i][0:20]) * fr
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
    for i in range(0, len(activity_all[2])):
        diff_cue = (np.mean(activity_all[2][i][len(activity_all[2][i]) - 20:len(activity_all[2][i])]) - \
                       np.mean(activity_all[2][i][0:20])) * fr
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
        diff_cue = (np.mean(activity_all[2][i][len(activity_all[2][i]) - 20:len(activity_all[2][i])]) - \
                   np.mean(activity_all[2][i][0:20])) * fr
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


def pupil_control(paths):
    behavior = preprocess_opto.process_behavior(paths)
    fr = behavior['framerate']
    pupil_control_all = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/pupil.npy', allow_pickle=True)
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(15, 7))
    plt.subplot(2, 2, 1)
    pupil_control_norm = np.zeros((len(pupil_control_all[0]), len(pupil_control_all[0][0])))
    pupil_control_opto = np.zeros((len(pupil_control_all[0]), len(pupil_control_all[0][0])))
    for i in range(0, len(pupil_control_all[0])):
        pupil_control_norm[i, :] = (pupil_control_all[0][i] + pupil_control_all[1][i]) / 2
        pupil_control_opto[i, :] =(pupil_control_all[2][i] + pupil_control_all[3][i]) / 2
        # plt.plot(pupil_control_all[0][i], c='k')
        # plt.plot(pupil_control_all[1][i], c='k')
        # plt.plot(pupil_control_all[2][i], c='r')
        # plt.plot(pupil_control_all[3][i], c='r')
    mean = pupil_control_norm.mean(axis=0)
    sem_plus = mean + stats.sem(pupil_control_norm, axis=0)
    sem_minus = mean - stats.sem(pupil_control_norm, axis=0)
    mean_opto = pupil_control_opto.mean(axis=0)
    sem_plus_opto = mean_opto + stats.sem(pupil_control_opto, axis=0)
    sem_minus_opto = mean_opto - stats.sem(pupil_control_opto, axis=0)
    plt.plot(mean, linewidth=2, c='k')
    plt.fill_between(range(0, len(mean)), sem_plus, sem_minus, alpha=0.1, color='k')
    plt.plot(mean_opto, linewidth=2, c='r')
    plt.fill_between(range(0, len(mean_opto)), sem_plus_opto, sem_minus_opto, alpha=0.1, color='r')
    plt.xticks([int(fr * 0), int(fr * 6), int(fr * 16), int(fr * 26), int(fr * 36), int(fr * 46), int(fr * 56),
                int(fr * 66)], ['-6', '0', '10', '20', '30', '40', '50', '60'])
    plt.xlim((int(fr * 0), int(fr * 67)))
    plt.ylabel('Max normalized pupil area')
    plt.xlabel('Time relative to cue onset (s)')
    plt.axvspan(int(fr * 6), int(fr * 8), alpha=.2, color='r', zorder=0)
    plt.axvspan(int(fr * 8), int(fr * 14), alpha=.2, color='gray', zorder=0)
    sns.despine()
    plt.savefig(paths['base_path'] + paths['mouse'] + '/data_across_days/plots/' + paths['mouse'] + '_'
                + 'pupil_control.png', bbox_inches='tight', dpi=500)
    plt.close()
