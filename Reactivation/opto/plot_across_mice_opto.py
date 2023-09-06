import warnings
import matplotlib
import preprocess_opto
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from scipy import stats
from statsmodels.stats.multitest import multipletests
warnings.filterwarnings('ignore')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def reactivation_rate_day(mice, sample_dates):
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(9, 6))
    plt.subplot(2, 2, 1)
    paths = preprocess_opto.create_folders(mice[0], sample_dates[0])
    y_pred_binned_across_days = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/y_pred_binned.npy',
                                        allow_pickle=True)
    binned_reactivation_all = np.zeros((len(mice), len(y_pred_binned_across_days[0][0])))
    binned_reactivation_opto_all = np.zeros((len(mice), len(y_pred_binned_across_days[0][0])))
    behavior = preprocess_opto.process_behavior(paths)
    task_runs = behavior['task_runs']
    dark_runs = behavior['dark_runs']
    x_label = np.zeros(len(y_pred_binned_across_days[0][0]))
    hours_per_run = 64000 / 31.25 / 60 / 60
    for i in range(0, dark_runs):
        x_label[dark_runs - i - 1] = - hours_per_run / 2
    for i in range(dark_runs, dark_runs + (task_runs * 2)):
        x_label[i] = hours_per_run / 4 + (hours_per_run / 2 * (i - dark_runs))
    for mouse in range(0, len(mice)):
        paths = preprocess_opto.create_folders(mice[mouse], sample_dates[mouse])
        y_pred_binned_across_days = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/y_pred_binned.npy',
                                            allow_pickle=True)
        binned_reactivation = np.zeros((len(y_pred_binned_across_days[0]), len(y_pred_binned_across_days[0][0])))
        binned_reactivation_opto = np.zeros((len(y_pred_binned_across_days[0]), len(y_pred_binned_across_days[0][0])))
        for i in range(0, len(y_pred_binned_across_days[0])):
            binned_reactivation[i, :] = y_pred_binned_across_days[0][i]
            binned_reactivation_opto[i, :] = y_pred_binned_across_days[1][i]
        binned_reactivation_all[mouse, :] = binned_reactivation.mean(axis=0)
        binned_reactivation_opto_all[mouse, :] = binned_reactivation_opto.mean(axis=0)
        plt.plot(x_label, binned_reactivation.mean(axis=0), 'k', alpha=.2, linewidth=2)
        plt.plot(x_label, binned_reactivation_opto.mean(axis=0), 'r', alpha=.2, linewidth=2)

    [_, p_value] = stats.ttest_rel(np.mean(binned_reactivation_all[:, 1:9], axis=1),
                          np.mean(binned_reactivation_opto_all[:, 1:9], axis=1))

    [_, s_p_value] = stats.shapiro(binned_reactivation_all[:, 1])
    print(s_p_value)


    anova_results = []
    for i in range(1, 9):
        anova_results.append(
            stats.ttest_rel(binned_reactivation_all[:, i], binned_reactivation_all[:, 0], alternative='greater')[1])
    for i in range(1, 9):
        anova_results.append(
            stats.ttest_rel(binned_reactivation_opto_all[:, i], binned_reactivation_opto_all[:, 0], alternative='greater')[1])
    anova_results.append(p_value)
    anova_results_corrected = multipletests(anova_results, alpha=.05, method='holm')


    print([anova_results, anova_results_corrected[1]])


    mean = binned_reactivation_all.mean(axis=0)
    sem_plus = mean + stats.sem(binned_reactivation_all, axis=0)
    sem_minus = mean - stats.sem(binned_reactivation_all, axis=0)
    mean_opto = binned_reactivation_opto_all.mean(axis=0)
    sem_plus_opto = mean_opto + stats.sem(binned_reactivation_opto_all, axis=0)
    sem_minus_opto = mean_opto - stats.sem(binned_reactivation_opto_all, axis=0)
    plt.plot(x_label, mean[0:len(mean)], 'k', linewidth=3, ms=0)
    plt.plot(x_label, mean_opto[0:len(mean_opto)], 'r', linewidth=3, ms=0)
    plt.fill_between(x_label, sem_plus[0:len(sem_plus)], sem_minus[0:len(sem_minus)], alpha=0.2, color='k', lw=0)
    plt.fill_between(x_label, sem_plus_opto[0:len(sem_plus_opto)], sem_minus_opto[0:len(sem_minus_opto)], alpha=0.2,
                     color='r', lw=0)
    plt.axvspan(-hours_per_run, 0, alpha=.1, color='gray', lw=0)
    plt.ylabel('Reactivation rate (probability $\mathregular{s^{-1}}$)')
    plt.xlabel('Time relative to stimulus onset (h)')
    plt.xlim(-hours_per_run, x_label[len(x_label) - 1] + hours_per_run / 4)
    plt.xticks([-.5, 0, .5, 1, 1.5, 2])
    label_1 = mlines.Line2D([], [], color='k', linestyle='-', label='Control', linewidth=3)
    label_2 = mlines.Line2D([], [], color='r', linestyle='-', label='Inhibition', linewidth=3)
    plt.legend(handles=[label_1, label_2], frameon=False, prop={'size': 8}, labelspacing=.1)
    plt.ylim((0, .15))
    plt.yticks([0, .05, .1, .15])
    sns.despine()
    plt.savefig(paths['base_path'] + '/NN_opto/plots/reactivation_rate_day.png', bbox_inches='tight', dpi=200)
    plt.savefig(paths['base_path'] + '/NN_opto/plots/reactivation_rate_day.pdf', bbox_inches='tight', dpi=200, Transparent=True)
    plt.close()


def reactivation_bias_day(mice, sample_dates):
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(9, 6))
    plt.subplot(2, 2, 1)
    paths = preprocess_opto.create_folders(mice[0], sample_dates[0])
    y_pred_binned_across_days = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/y_pred_bias_binned.npy',
                                        allow_pickle=True)
    binned_reactivation_all = np.zeros((len(mice), len(y_pred_binned_across_days[0][0])))
    binned_reactivation_opto_all = np.zeros((len(mice), len(y_pred_binned_across_days[0][0])))
    behavior = preprocess_opto.process_behavior(paths)
    task_runs = behavior['task_runs']
    dark_runs = behavior['dark_runs']
    x_label = np.zeros((task_runs * 2))
    hours_per_run = 64000 / 31.25 / 60 / 60
    for i in range(0, (task_runs * 2)):
        x_label[i] = hours_per_run / 4 + (hours_per_run / 2 * i)
    for mouse in range(0, len(mice)):
        paths = preprocess_opto.create_folders(mice[mouse], sample_dates[mouse])
        y_pred_binned_across_days = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/y_pred_bias_binned.npy',
                                            allow_pickle=True)
        binned_reactivation = np.zeros((len(y_pred_binned_across_days[0]), len(y_pred_binned_across_days[0][0])))
        binned_reactivation_opto = np.zeros((len(y_pred_binned_across_days[0]), len(y_pred_binned_across_days[0][0])))
        for i in range(0, len(y_pred_binned_across_days[0])):
            binned_reactivation[i, :] = y_pred_binned_across_days[0][i]
            binned_reactivation_opto[i, :] = y_pred_binned_across_days[1][i]
        binned_reactivation_all[mouse, :] = np.nanmean(binned_reactivation, axis=0)
        binned_reactivation_opto_all[mouse, :] = np.nanmean(binned_reactivation_opto, axis=0)
        plt.plot(x_label, np.nanmean(binned_reactivation, axis=0), 'k', alpha=.2, linewidth=2)
        plt.plot(x_label, np.nanmean(binned_reactivation_opto, axis=0), 'r', alpha=.2, linewidth=2)

    [_, p_value] = stats.ttest_rel(np.mean(binned_reactivation_all, axis=1), np.mean(binned_reactivation_opto_all, axis=1))
    print(p_value)

    mean = binned_reactivation_all.mean(axis=0)
    sem_plus = mean + stats.sem(binned_reactivation_all, axis=0)
    sem_minus = mean - stats.sem(binned_reactivation_all, axis=0)
    mean_opto = binned_reactivation_opto_all.mean(axis=0)
    sem_plus_opto = mean_opto + stats.sem(binned_reactivation_opto_all, axis=0)
    sem_minus_opto = mean_opto - stats.sem(binned_reactivation_opto_all, axis=0)
    plt.plot(x_label, mean[0:len(mean)], '-k', linewidth=3, ms=0)
    plt.plot(x_label, mean_opto[0:len(mean_opto)], '-3r', linewidth=3, ms=0)
    plt.fill_between(x_label, sem_plus[0:len(sem_plus)], sem_minus[0:len(sem_minus)], alpha=0.2, color='k', lw=0)
    plt.fill_between(x_label, sem_plus_opto[0:len(sem_plus_opto)], sem_minus_opto[0:len(sem_minus_opto)], alpha=0.2,
                     color='r', lw=0)
    plt.ylabel('Bias in reactivation rate\n toward the previous stimulus')
    plt.xlabel('Time relative to stimulus onset (h)')
    plt.ylim((-1, 1))
    plt.xlim(0, x_label[len(x_label)-1] + hours_per_run / 4)
    plt.xticks([0, .5, 1, 1.5, 2])
    label_1 = mlines.Line2D([], [], color='k', linestyle='-', label='Control', linewidth=3)
    label_2 = mlines.Line2D([], [], color='r', linestyle='-', label='Inhibition', linewidth=3)
    plt.legend(handles=[label_1, label_2], frameon=False, prop={'size': 8}, labelspacing=.1)
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1, snap=False)
    sns.despine()
    plt.savefig(paths['base_path'] + '/NN_opto/plots/reactivation_bias_day.png', bbox_inches='tight', dpi=200)
    plt.savefig(paths['base_path'] + '/NN_opto/plots/reactivation_bias_day.pdf', bbox_inches='tight', dpi=200, transparent=True)

    plt.close()


def reactivation_rate_trial(mice, sample_dates):
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(9, 6))
    plt.subplot(2, 2, 1)
    mean_norm_mice = []
    mean_opto_mice = []
    for mouse in range(0, len(mice)):
        paths = preprocess_opto.create_folders(mice[mouse], sample_dates[mouse])
        rate_within_trial_all = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/rate_within_trial.npy',
                                        allow_pickle=True)
        mean_norm = []
        mean_opto = []
        for i in range(0, len(rate_within_trial_all[0])):
            mean_norm.append(rate_within_trial_all[0][i])
            mean_opto.append(rate_within_trial_all[1][i])
            x = rate_within_trial_all[2][i]
        mean_norm = np.nanmean(mean_norm, axis=0)
        mean_opto = np.nanmean(mean_opto, axis=0)
        mean_norm_mice.append(mean_norm)
        mean_opto_mice.append(mean_opto)
        plt.plot(x, mean_norm, '-', c='k', ms=0, alpha=.2, linewidth=2)
        plt.plot(x, mean_opto, '-', c='r', ms=0, alpha=.2, linewidth=2)

    [_, p_value] = stats.ttest_rel(np.mean(mean_norm_mice, axis=1), np.mean(mean_opto_mice, axis=1))

    [_, s_p_value] = stats.shapiro(mean_norm_mice[0])
    print(s_p_value)

    paths = preprocess_opto.create_folders(mice[0], sample_dates[0])
    y_pred_binned_across_days = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/y_pred_binned.npy',
                                        allow_pickle=True)
    binned_reactivation_all = np.zeros((len(mice), len(y_pred_binned_across_days[0][0])))
    binned_reactivation_opto_all = np.zeros((len(mice), len(y_pred_binned_across_days[0][0])))
    behavior = preprocess_opto.process_behavior(paths)
    task_runs = behavior['task_runs']
    dark_runs = behavior['dark_runs']
    x_label = np.zeros(len(y_pred_binned_across_days[0][0]))
    hours_per_run = 64000 / 31.25 / 60 / 60
    for i in range(0, dark_runs):
        x_label[dark_runs - i - 1] = - hours_per_run / 2
    for i in range(dark_runs, dark_runs + (task_runs * 2)):
        x_label[i] = hours_per_run / 4 + (hours_per_run / 2 * (i - dark_runs))
    for mouse in range(0, len(mice)):
        paths = preprocess_opto.create_folders(mice[mouse], sample_dates[mouse])
        y_pred_binned_across_days = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/y_pred_binned.npy',
                                            allow_pickle=True)
        binned_reactivation = np.zeros((len(y_pred_binned_across_days[0]), len(y_pred_binned_across_days[0][0])))
        binned_reactivation_opto = np.zeros((len(y_pred_binned_across_days[0]), len(y_pred_binned_across_days[0][0])))
        for i in range(0, len(y_pred_binned_across_days[0])):
            binned_reactivation[i, :] = y_pred_binned_across_days[0][i]
            binned_reactivation_opto[i, :] = y_pred_binned_across_days[1][i]
        binned_reactivation_all[mouse, :] = binned_reactivation.mean(axis=0)
        binned_reactivation_opto_all[mouse, :] = binned_reactivation_opto.mean(axis=0)
    anova_results = []
    for i in range(0, len(mean_norm_mice[0])):
        vec = []
        for j in range(0, len(mice)):
            vec.append(mean_norm_mice[j][i])
        anova_results.append(
            stats.ttest_rel(vec, binned_reactivation_all[:, 0], alternative='greater')[1])
    for i in range(0, len(mean_opto_mice[0])):
        vec = []
        for j in range(0, len(mice)):
            vec.append(mean_opto_mice[j][i])
        anova_results.append(
            stats.ttest_rel(vec, binned_reactivation_opto_all[:, 0], alternative='greater')[1])
    anova_results.append(p_value)
    anova_results_corrected = multipletests(anova_results, alpha=.05, method='holm')
    print([anova_results, anova_results_corrected[1]])


    mean_norm_mice_all = np.nanmean(mean_norm_mice, axis=0)
    y0_err = stats.sem(mean_norm_mice, axis=0, nan_policy='omit')
    mean_opto_mice_all = np.nanmean(mean_opto_mice, axis=0)
    y1_err = stats.sem(mean_opto_mice, axis=0, nan_policy='omit')

    sem_plus_norm = mean_norm_mice_all + y0_err
    sem_minus_norm = mean_norm_mice_all - y0_err
    sem_plus_opto = mean_opto_mice_all + y1_err
    sem_minus_opto = mean_opto_mice_all - y1_err

    plt.plot(x, mean_norm_mice_all, '-o', c='k', linewidth=3, ms=0)
    plt.fill_between(x, sem_plus_norm, sem_minus_norm, alpha=0.2, color='k', lw=0)
    plt.plot(x, mean_opto_mice_all, '-o', c='r', linewidth=3, ms=0)
    plt.fill_between(x, sem_plus_opto, sem_minus_opto, alpha=0.2, color='r', lw=0)

    paths = preprocess_opto.create_folders(mice[0], sample_dates[0])
    behavior = preprocess_opto.process_behavior(paths)
    fr = behavior['framerate']
    # plt.axvspan(0, int(fr * 1), alpha=.75, color='mediumseagreen', lw=0)
    # plt.axvspan(int(fr * 1), int(fr * 2), alpha=.75, color='salmon', lw=0)
    plt.axvspan(0, int(fr * 2), alpha=1, color='k', lw=0)
    plt.axvspan(int(fr * 2), int(fr * 8), alpha=.1, color='gray', lw=0)
    plt.ylabel('Reactivation rate (probability $\mathregular{s^{-1}}$)')
    plt.xlabel('Time relative to stimulus onset (s)')
    plt.xticks([int(fr * 0), int(fr * 10), int(fr * 20), int(fr * 30), int(fr * 40), int(fr * 50), int(fr * 60)],
               ['0', '10', '20', '30', '40', '50', '60'])
    plt.xlim((0, int(fr * 60)))
    plt.ylim(0, .15)
    plt.yticks([0, .05, .1, .15])
    label_1 = mlines.Line2D([], [], color='k', linestyle='-', label='Control', linewidth=3)
    label_2 = mlines.Line2D([], [], color='r', linestyle='-', label='Inhibition', linewidth=3)
    plt.legend(handles=[label_1, label_2], frameon=False, prop={'size': 8}, labelspacing=.1)
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1, snap=False)
    sns.despine()
    plt.savefig(paths['base_path'] + '/NN_opto/plots/reactivation_rate_trial.png', bbox_inches='tight', dpi=200)
    plt.savefig(paths['base_path'] + '/NN_opto/plots/reactivation_rate_trial.pdf', bbox_inches='tight', dpi=200, Transparent=True)

    plt.close()


def reactivation_bias_trial(mice, sample_dates):
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(9, 6))
    plt.subplot(2, 2, 1)
    mean_norm_mice = []
    mean_opto_mice = []
    for mouse in range(0, len(mice)):
        paths = preprocess_opto.create_folders(mice[mouse], sample_dates[mouse])
        bias_within_trial_all = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/bias_within_trial.npy',
                                        allow_pickle=True)
        mean_norm = []
        mean_opto = []
        for i in range(0, len(bias_within_trial_all[0])):
            mean_norm.append(bias_within_trial_all[0][i])
            mean_opto.append(bias_within_trial_all[1][i])
            x = bias_within_trial_all[2][i]
        mean_norm = np.nanmean(mean_norm, axis=0)
        mean_opto = np.nanmean(mean_opto, axis=0)
        mean_norm_mice.append(mean_norm)
        mean_opto_mice.append(mean_opto)
        plt.plot(x, mean_norm, '-', c='k', ms=0, alpha=.2, linewidth=2)
        plt.plot(x, mean_opto, '-', c='r', ms=0, alpha=.2, linewidth=2)

    [_, s_p_value] = stats.shapiro(mean_norm_mice[0])
    print(s_p_value)

    anova_results = []
    for i in range(0, len(mean_norm_mice[0])):
        vec = []
        for j in range(0, len(mice)):
            vec.append(mean_norm_mice[j][i])
        anova_results.append(
            stats.ttest_1samp(vec, 0)[1])
    for i in range(0, len(mean_opto_mice[0])):
        vec = []
        for j in range(0, len(mice)):
            vec.append(mean_opto_mice[j][i])
        anova_results.append(
            stats.ttest_1samp(vec, 0)[1])
    [_, p_value] = stats.ttest_rel(np.mean(mean_norm_mice, axis=1), np.mean(mean_opto_mice, axis=1))
    anova_results.append(p_value)
    anova_results_corrected = multipletests(anova_results, alpha=.05, method='holm')
    print(anova_results)
    print(anova_results_corrected[1])

    mean_norm_mice_all = np.nanmean(mean_norm_mice, axis=0)
    y0_err = stats.sem(mean_norm_mice, axis=0, nan_policy='omit')
    mean_opto_mice_all = np.nanmean(mean_opto_mice, axis=0)
    y1_err = stats.sem(mean_opto_mice, axis=0, nan_policy='omit')

    sem_plus_norm = mean_norm_mice_all + y0_err
    sem_minus_norm = mean_norm_mice_all - y0_err
    sem_plus_opto = mean_opto_mice_all + y1_err
    sem_minus_opto = mean_opto_mice_all - y1_err

    plt.plot(x, mean_norm_mice_all, '-o', c='k', linewidth=3, ms=0)
    plt.fill_between(x, sem_plus_norm, sem_minus_norm, alpha=0.2, color='k', lw=0)
    plt.plot(x, mean_opto_mice_all, '-o', c='r', linewidth=3, ms=0)
    plt.fill_between(x, sem_plus_opto, sem_minus_opto, alpha=0.2, color='r', lw=0)

    paths = preprocess_opto.create_folders(mice[0], sample_dates[0])
    behavior = preprocess_opto.process_behavior(paths)
    fr = behavior['framerate']
    # plt.axvspan(0, int(fr * 1), alpha=.75, color='mediumseagreen', lw=0)
    # plt.axvspan(int(fr * 1), int(fr * 2), alpha=.75, color='salmon', lw=0)
    plt.axvspan(0, int(fr * 2), alpha=1, color='k', lw=0)
    plt.axvspan(int(fr * 2), int(fr * 8), alpha=.1, color='gray', lw=0)
    plt.ylabel('Bias in reactivation rate\n toward the previous stimulus')
    plt.xlabel('Time relative to stimulus onset (s)')
    plt.xticks([int(fr * 0), int(fr * 10), int(fr * 20), int(fr * 30), int(fr * 40), int(fr * 50), int(fr * 60)],
               ['0', '10', '20', '30', '40', '50', '60'])
    plt.xlim((0, int(fr * 61)))
    plt.ylim(-1, 1)
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1, snap=False)
    label_1 = mlines.Line2D([], [], color='k', linestyle='-', label='Control', linewidth=3)
    label_2 = mlines.Line2D([], [], color='r', linestyle='-', label='Inhibition', linewidth=3)
    plt.legend(handles=[label_1, label_2], frameon=False, prop={'size': 8}, labelspacing=.1)
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1, snap=False)
    sns.despine()
    plt.savefig(paths['base_path'] + '/NN_opto/plots/reactivation_bias_trial.png', bbox_inches='tight', dpi=200)
    plt.savefig(paths['base_path'] + '/NN_opto/plots/reactivation_bias_trial.pdf', bbox_inches='tight', dpi=200, transparent=0)
    plt.close()


def activity_control(mice, sample_dates):
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(9, 6))
    plt.subplot(2, 2, 1)
    paths = preprocess_opto.create_folders(mice[0], sample_dates[0])
    activity_control_all = np.load(paths['base_path'] + paths['mouse'] +
                                   '/data_across_days/activity_control.npy', allow_pickle=True)
    activity_control_norm_all = np.zeros((len(mice), len(activity_control_all[0][0])))
    activity_control_opto_all = np.zeros((len(mice), len(activity_control_all[0][0])))
    for mouse in range(0, len(mice)):
        paths = preprocess_opto.create_folders(mice[mouse], sample_dates[mouse])
        behavior = preprocess_opto.process_behavior(paths)
        fr = behavior['framerate']
        activity_control_all = np.load(paths['base_path'] + paths['mouse'] +
                                    '/data_across_days/activity_control.npy', allow_pickle=True)
        activity_control_norm = np.zeros((len(activity_control_all[0]), len(activity_control_all[0][0])))
        activity_control_opto = np.zeros((len(activity_control_all[0]), len(activity_control_all[0][0])))
        for i in range(0, len(activity_control_all[0])):
            activity_control_norm[i, :] = activity_control_all[0][i] * fr[0][0]
            activity_control_opto[i, :] = activity_control_all[1][i] * fr[0][0]
        y1 = activity_control_norm.mean(axis=0)
        y2 = activity_control_opto.mean(axis=0)
        activity_control_norm_all[mouse, :] = y1
        activity_control_opto_all[mouse, :] = y2
        plt.plot(y1, c='k', alpha=.2, linewidth=2)
        plt.plot(y2, c='r', alpha=.2, linewidth=2)

    print(stats.ttest_rel(np.mean(activity_control_norm_all[:, int(fr * 13):len(activity_control_norm_all[0])], axis=1),
                          np.mean(activity_control_opto_all[:, int(fr * 13):len(activity_control_opto_all[0])], axis=1)))

    mean = activity_control_norm_all.mean(axis=0)
    sem_plus = mean + stats.sem(activity_control_norm_all, axis=0)
    sem_minus = mean - stats.sem(activity_control_norm_all, axis=0)
    mean_opto = activity_control_opto_all.mean(axis=0)
    sem_plus_opto = mean_opto + stats.sem(activity_control_opto_all, axis=0)
    sem_minus_opto = mean_opto - stats.sem(activity_control_opto_all, axis=0)
    plt.plot(mean, linewidth=3, c='k')
    plt.fill_between(range(0, len(mean)), sem_plus, sem_minus, alpha=0.2, color='k', lw=0)
    plt.plot(mean_opto, linewidth=3, c='r')
    plt.fill_between(range(0, len(mean_opto)), sem_plus_opto, sem_minus_opto, alpha=0.2, color='r', lw=0)
    plt.xticks([int(fr * 3), int(fr * 5), int(fr * 15), int(fr * 25), int(fr * 35), int(fr * 45), int(fr * 55),
                int(fr * 65)], ['', '0', '10', '20', '30', '40', '50', '60'])
    plt.xlim((int(fr * 3), int(fr * 65)))
    plt.ylabel('Activity\n(Normalized $\mathregular{Ca^{2+}}$transient $\mathregular{s^{-1}}$)')
    plt.xlabel('Time relative to stimulus onset (s)')
    plt.axvspan(int(fr * 5), int(fr * 7), alpha=.2, color='k', zorder=0, lw=0)
    plt.axvspan(int(fr * 7), int(fr * 13), alpha=.1, color='gray', zorder=0, lw=0)
    label_1 = mlines.Line2D([], [], color='k', linestyle='-', label='Control', linewidth=3)
    label_2 = mlines.Line2D([], [], color='r', linestyle='-', label='Inhibition', linewidth=3)
    plt.legend(handles=[label_1, label_2], frameon=False, prop={'size': 8}, labelspacing=.1)
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1, snap=False)
    sns.despine()
    plt.ylim((0, 2))
    plt.savefig(paths['base_path'] + '/NN_opto/plots/activity_control.png', bbox_inches='tight', dpi=200)
    plt.savefig(paths['base_path'] + '/NN_opto/plots/activity_control.pdf', bbox_inches='tight', dpi=200, transparent=0)
    plt.close()


def pupil_control(mice, sample_dates):
    paths = preprocess_opto.create_folders(mice[0], sample_dates[0])
    behavior = preprocess_opto.process_behavior(paths)
    fr = behavior['framerate']
    pupil_control_all = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/pupil.npy', allow_pickle=True)
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(9, 6))
    plt.subplot(2, 2, 1)
    pupil_control_norm_all = np.zeros((len(mice), len(pupil_control_all[0][0])))
    pupil_control_opto_all = np.zeros((len(mice), len(pupil_control_all[0][0])))
    for mouse in range(0, len(mice)):
        paths = preprocess_opto.create_folders(mice[mouse], sample_dates[mouse])
        pupil_control_all = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/pupil.npy',
                                    allow_pickle=True)
        pupil_control_norm = np.zeros((len(pupil_control_all[0]), len(pupil_control_all[0][0])))
        pupil_control_opto = np.zeros((len(pupil_control_all[0]), len(pupil_control_all[0][0])))
        for i in range(0, len(pupil_control_all[0])):
            pupil_control_norm[i, :] = (pupil_control_all[0][i] + pupil_control_all[1][i]) / 2
            pupil_control_opto[i, :] =(pupil_control_all[2][i] + pupil_control_all[3][i]) / 2
        y1 = pupil_control_norm.mean(axis=0)
        y2 = pupil_control_opto.mean(axis=0)
        pupil_control_norm_all[mouse, :] = y1
        pupil_control_opto_all[mouse, :] = y2
        plt.plot(y1, c='k', alpha=.2, linewidth=2)
        plt.plot(y2, c='r', alpha=.2, linewidth=2)

    print(stats.ttest_rel(np.mean(pupil_control_norm_all[:, int(fr * 13):len(pupil_control_norm_all[0])], axis=1),
                          np.mean(pupil_control_opto_all[:, int(fr * 13):len(pupil_control_opto_all[0])],
                                  axis=1)))
    print(stats.ttest_rel(np.mean(pupil_control_norm_all[:, int(fr * 5):int(fr * 13)], axis=1),
                          np.mean(pupil_control_opto_all[:, int(fr * 5):int(fr * 13)],
                                  axis=1)))

    mean = pupil_control_norm_all.mean(axis=0)
    sem_plus = mean + stats.sem(pupil_control_norm_all, axis=0)
    sem_minus = mean - stats.sem(pupil_control_norm_all, axis=0)
    mean_opto = pupil_control_opto_all.mean(axis=0)
    sem_plus_opto = mean_opto + stats.sem(pupil_control_opto_all, axis=0)
    sem_minus_opto = mean_opto - stats.sem(pupil_control_opto_all, axis=0)
    plt.plot(mean, linewidth=3, c='k')
    plt.fill_between(range(0, len(mean)), sem_plus, sem_minus, alpha=0.2, color='k', lw=0)
    plt.plot(mean_opto, linewidth=3, c='r')
    plt.fill_between(range(0, len(mean_opto)), sem_plus_opto, sem_minus_opto, alpha=0.2, color='r', lw=0)
    plt.xticks([int(fr * 3), int(fr * 5), int(fr * 15), int(fr * 25), int(fr * 35), int(fr * 45), int(fr * 55),
                int(fr * 65)], ['', '0', '10', '20', '30', '40', '50', '60'])
    plt.xlim((int(fr * 3), int(fr * 63)))
    plt.ylabel('Max normalized pupil area')
    plt.xlabel('Time relative to stimulus onset (s)')
    # plt.axvspan(int(fr * 4), int(fr * 8), alpha=.2, color='r', zorder=0, lw=0)
    plt.axvspan(int(fr * 5), int(fr * 7), alpha=.2, color='k', zorder=0, lw=0)
    plt.axvspan(int(fr * 7), int(fr * 13), alpha=.1, color='gray', zorder=0, lw=0)
    plt.ylim((0, 1))
    label_1 = mlines.Line2D([], [], color='k', linestyle='-', label='Control', linewidth=3)
    label_2 = mlines.Line2D([], [], color='r', linestyle='-', label='Inhibition', linewidth=3)
    plt.legend(handles=[label_1, label_2], frameon=False, prop={'size': 8}, labelspacing=.1)
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1, snap=False)
    sns.despine()
    plt.savefig(paths['base_path'] + '/NN_opto/plots/pupil_control.png', bbox_inches='tight', dpi=200)
    plt.savefig(paths['base_path'] + '/NN_opto/plots/pupil_control.pdf', bbox_inches='tight', dpi=200, transparent=True)
    plt.close()


def activity_difference(mice, sample_dates):
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(6, 6))
    plt.subplots_adjust(wspace=.4)
    x = [1, 2]
    opto_all = []
    norm_all = []
    diff_all = []
    for mouse in range(0, len(mice)):
        paths = preprocess_opto.create_folders(mice[mouse], sample_dates[mouse])
        activity_all = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/activity_control.npy',
                               allow_pickle=True)
        behavior = preprocess_opto.process_behavior(paths)
        fr = behavior['framerate']
        opto = []
        norm = []
        diff = []
        for i in range(0, len(activity_all[0])):
            opto.append(np.mean(activity_all[1][i][int(fr * 5):int(fr * 7)]) * fr[0][0])
            norm.append(np.mean(activity_all[0][i][int(fr * 5):int(fr * 7)]) * fr[0][0])
            diff.append(100 * ((np.mean(activity_all[1][i][int(fr * 5):int(fr * 7)]) - np.mean(activity_all[0][i][int(fr * 5):int(fr * 7)])) / np.mean(activity_all[0][i][int(fr * 5):int(fr * 7)])))
        opto_all.append(np.mean(opto))
        norm_all.append(np.mean(norm))
        diff_all.append(np.mean(diff))
        plt.subplot(2, 2, 1)
        plt.plot(x, [np.mean(norm), np.mean(opto)], color='k', alpha=.2, linewidth=2)
        plt.subplot(2, 2, 2)
        plt.errorbar(1, np.mean(diff), yerr=0, c='k', marker='o', mfc='none',mec='k', ms=5, alpha=.3)

    print(stats.ttest_rel(opto_all, norm_all))

    print(stats.ttest_1samp(diff_all, 0))

    opto_all_mice = np.mean(opto_all)
    y0_err = stats.sem(opto_all)
    norm_all_mice = np.mean(norm_all)
    y1_err = stats.sem(norm_all)
    plt.subplot(2, 2, 1)
    plt.plot(x, [norm_all_mice, opto_all_mice], '-', c='k', linewidth=3)
    plt.fill_between(x, [norm_all_mice + y1_err, opto_all_mice + y0_err],
                     [norm_all_mice - y1_err, opto_all_mice - y0_err], alpha=0.2,
                     color='k', lw=0)
    plt.xlim((.5, 2.5))
    plt.xticks([1, 2], ['Control', 'Inhibition'])
    plt.ylabel('Stimulus activity\n(Normalized $\mathregular{Ca^{2+}}$transient $\mathregular{s^{-1}}$)')
    plt.subplot(2, 2, 2)
    y1 = np.mean(diff_all)
    y1_err = stats.sem(diff_all)
    plt.errorbar(1.2, y1, yerr=y1_err, c='k', linewidth=2, marker='o', mfc='k', mec='k',
                 ms=7, mew=0, zorder=100)
    plt.xlim(.5, 4.5)
    plt.ylim(-100, 0)
    plt.xticks([])
    plt.ylabel('Percent change\nin stimulus activity')
    sns.despine()
    plt.savefig(paths['base_path'] + '/NN_opto/plots/activity_difference.png', bbox_inches='tight', dpi=200)
    plt.savefig(paths['base_path'] + '/NN_opto/plots/activity_difference.pdf', bbox_inches='tight', dpi=200, transparent=True)
    plt.close()


def reactivation_rate_trial_controlopto_R1(mice, sample_dates):
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(15, 6))
    plt.subplot(2, 2, 1)
    mean_norm_mice = []
    mean_opto_mice = []
    for mouse in range(0, len(mice)):
        paths = preprocess_opto.create_folders(mice[mouse], sample_dates[mouse])
        rate_within_trial_all = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/rate_within_trial_long.npy',
                                        allow_pickle=True)
        mean_norm = []
        mean_opto = []
        for i in range(0, len(rate_within_trial_all[0])):
            mean_norm.append(rate_within_trial_all[0][i])
            mean_opto.append(rate_within_trial_all[1][i])
            x = rate_within_trial_all[2][i]
        mean_norm = np.nanmean(mean_norm, axis=0)
        mean_opto = np.nanmean(mean_opto, axis=0)
        mean_norm_mice.append(mean_norm)
        mean_opto_mice.append(mean_opto)
        plt.plot(x[0:5], mean_norm[0:5], '-', c='k', ms=0, alpha=.2, linewidth=2)
        # plt.plot(x[0:5], mean_opto[0:5], '-', c='r', ms=0, alpha=.2, linewidth=2)
        plt.plot(x[5:10], mean_norm[5:10], '-', c='r', ms=0, alpha=.2, linewidth=2)
        # plt.plot(x[5:10], mean_opto[5:10], '-', c='r', ms=0, alpha=.2, linewidth=2)
        plt.plot([x[4], x[5]], [mean_norm[4], mean_norm[5]], linestyle='dotted', linewidth=2, c='darkred', alpha=.2)

    slope, intercept, r_value, p_value, std_err = stats.linregress(
        np.concatenate([x[0:5], x[0:5], x[0:5]]),
        np.concatenate([mean_norm_mice[0][0:5], mean_norm_mice[1][0:5], mean_norm_mice[2][0:5]]))
    print(r_value)
    print(p_value)

    slope, intercept, r_value, p_value_o, std_err = stats.linregress(
        np.concatenate([x[5:10], x[5:10], x[5:10]]),
        np.concatenate([mean_norm_mice[0][5:10], mean_norm_mice[1][5:10], mean_norm_mice[2][5:10]]))
    print(r_value)
    print(p_value_o)

    print(multipletests([p_value, p_value_o], alpha=.05, method='holm')[1])




    mean_norm_mice_all = np.nanmean(mean_norm_mice, axis=0)
    y0_err = stats.sem(mean_norm_mice, axis=0, nan_policy='omit')
    mean_opto_mice_all = np.nanmean(mean_opto_mice, axis=0)
    y1_err = stats.sem(mean_opto_mice, axis=0, nan_policy='omit')

    sem_plus_norm = mean_norm_mice_all + y0_err
    sem_minus_norm = mean_norm_mice_all - y0_err
    sem_plus_opto = mean_opto_mice_all + y1_err
    sem_minus_opto = mean_opto_mice_all - y1_err

    plt.plot(x[0:5], mean_norm_mice_all[0:5], '-o', c='k', linewidth=3, ms=0)
    plt.fill_between(x[0:5], sem_plus_norm[0:5], sem_minus_norm[0:5], alpha=0.2, color='k', lw=0)
    # plt.plot(x[0:5], mean_opto_mice_all[0:5], '-o', c='r', linewidth=3, ms=0)
    # plt.fill_between(x[0:5], sem_plus_opto[0:5], sem_minus_opto[0:5], alpha=0.2, color='r', lw=0)
    plt.plot(x[5:10], mean_norm_mice_all[5:10], '-o', c='r', linewidth=3, ms=0)
    plt.fill_between(x[5:10], sem_plus_norm[5:10], sem_minus_norm[5:10], alpha=0.2, color='r', lw=0)
    # plt.plot(x[5:10], mean_opto_mice_all[5:10], '-o', c='r', linewidth=3, ms=0)
    # plt.fill_between(x[5:10], sem_plus_opto[5:10], sem_minus_opto[5:10], alpha=0.2, color='r', lw=0)
    plt.plot([x[4], x[5]], [mean_norm_mice_all[4], mean_norm_mice_all[5]], linestyle='dotted', linewidth=3, c='darkred')

    paths = preprocess_opto.create_folders(mice[0], sample_dates[0])
    behavior = preprocess_opto.process_behavior(paths)
    fr = behavior['framerate']
    plt.axvspan(int(fr * 60), int(fr * 62), alpha=.75, color='r', lw=0)
    plt.axvspan(int(fr * 62), int(fr * 68), alpha=.1, color='gray', lw=0)
    plt.axvspan(0, int(fr * 2), alpha=1, color='k', lw=0)
    plt.axvspan(int(fr * 2), int(fr * 8), alpha=.1, color='gray', lw=0)
    plt.ylabel('Reactivation rate (probability $\mathregular{s^{-1}}$)')
    plt.xlabel('Time relative to stimulus onset (s)')
    plt.xticks([int(fr * 0), int(fr * 10), int(fr * 20), int(fr * 30), int(fr * 40), int(fr * 50), int(fr * 60),
                int(fr * 70), int(fr * 80), int(fr * 90), int(fr * 100), int(fr * 110), int(fr * 120)],
               ['0', '10', '20', '30', '40', '50', '60', '70', '80', '90', '100', '110', '120'])
    plt.xlim((0, int(fr * 120)))
    plt.ylim(0, .15)
    plt.yticks([0, .05, .1, .15])
    # label_1 = mlines.Line2D([], [], color='k', linestyle='-', label='Control', linewidth=3)
    # label_2 = mlines.Line2D([], [], color='r', linestyle='-', label='Inhibition', linewidth=3)
    # plt.legend(handles=[label_1, label_2], frameon=False, prop={'size': 8}, labelspacing=.1)
    plt.axhline(y=0.013425233333333333, color='black', linestyle='--', linewidth=2, snap=False)
    sns.despine()
    plt.savefig(paths['base_path'] + '/NN_opto/plots/reactivation_rate_trial_long.png', bbox_inches='tight', dpi=200)
    plt.savefig(paths['base_path'] + '/NN_opto/plots/reactivation_rate_trial_long.pdf', bbox_inches='tight', dpi=200, Transparent=True)

    plt.close()


def reactivation_bias_trial_controlopto_R1(mice, sample_dates):
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(15, 6))
    plt.subplot(2, 2, 1)
    mean_norm_mice = []
    mean_opto_mice = []
    for mouse in range(0, len(mice)):
        paths = preprocess_opto.create_folders(mice[mouse], sample_dates[mouse])
        rate_within_trial_all = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/bias_within_trial_long.npy',
                                        allow_pickle=True)
        mean_norm = []
        mean_opto = []
        for i in range(0, len(rate_within_trial_all[0])):
            mean_norm.append(rate_within_trial_all[0][i])
            mean_opto.append(rate_within_trial_all[1][i])
            x = rate_within_trial_all[2][i]
        mean_norm = np.nanmean(mean_norm, axis=0)
        mean_opto = np.nanmean(mean_opto, axis=0)
        mean_norm_mice.append(mean_norm)
        mean_opto_mice.append(mean_opto)
        plt.plot(x[0:5], mean_norm[0:5], '-', c='k', ms=0, alpha=.2, linewidth=2)
        # plt.plot(x[0:5], mean_opto[0:5], '-', c='r', ms=0, alpha=.2, linewidth=2)
        plt.plot(x[5:10], mean_norm[5:10], '-', c='r', ms=0, alpha=.2, linewidth=2)
        # plt.plot(x[5:10], mean_opto[5:10], '-', c='r', ms=0, alpha=.2, linewidth=2)
        plt.plot([x[4], x[5]], [mean_norm[4], mean_norm[5]], linestyle='dotted', linewidth=2, c='darkred', alpha=.2)

    slope, intercept, r_value, p_value, std_err = stats.linregress(
        np.concatenate([x[0:5], x[0:5], x[0:5]]),
        np.concatenate([mean_norm_mice[0][0:5], mean_norm_mice[1][0:5], mean_norm_mice[2][0:5]]))
    print(r_value)
    print(p_value)

    slope, intercept, r_value, p_value_o, std_err = stats.linregress(
        np.concatenate([x[5:10], x[5:10], x[5:10]]),
        np.concatenate([mean_norm_mice[0][5:10], mean_norm_mice[1][5:10], mean_norm_mice[2][5:10]]))
    print(r_value)
    print(p_value_o)

    print(multipletests([p_value, p_value_o], alpha=.05, method='holm')[1])

    # print(stats.ttest_rel(np.mean(mean_norm_mice, axis=1), np.mean(mean_opto_mice, axis=1)))
    mean_norm_mice_all = np.nanmean(mean_norm_mice, axis=0)
    y0_err = stats.sem(mean_norm_mice, axis=0, nan_policy='omit')
    mean_opto_mice_all = np.nanmean(mean_opto_mice, axis=0)
    y1_err = stats.sem(mean_opto_mice, axis=0, nan_policy='omit')

    sem_plus_norm = mean_norm_mice_all + y0_err
    sem_minus_norm = mean_norm_mice_all - y0_err
    sem_plus_opto = mean_opto_mice_all + y1_err
    sem_minus_opto = mean_opto_mice_all - y1_err

    plt.plot(x[0:5], mean_norm_mice_all[0:5], '-o', c='k', linewidth=3, ms=0)
    plt.fill_between(x[0:5], sem_plus_norm[0:5], sem_minus_norm[0:5], alpha=0.2, color='k', lw=0)
    # plt.plot(x[0:5], mean_opto_mice_all[0:5], '-o', c='r', linewidth=3, ms=0)
    # plt.fill_between(x[0:5], sem_plus_opto[0:5], sem_minus_opto[0:5], alpha=0.2, color='r', lw=0)
    plt.plot(x[5:10], mean_norm_mice_all[5:10], '-o', c='r', linewidth=3, ms=0)
    plt.fill_between(x[5:10], sem_plus_norm[5:10], sem_minus_norm[5:10], alpha=0.2, color='r', lw=0)
    # plt.plot(x[5:10], mean_opto_mice_all[5:10], '-o', c='r', linewidth=3, ms=0)
    # plt.fill_between(x[5:10], sem_plus_opto[5:10], sem_minus_opto[5:10], alpha=0.2, color='r', lw=0)
    plt.plot([x[4], x[5]], [mean_norm_mice_all[4], mean_norm_mice_all[5]], linestyle='dotted', linewidth=3, c='darkred')

    paths = preprocess_opto.create_folders(mice[0], sample_dates[0])
    behavior = preprocess_opto.process_behavior(paths)
    fr = behavior['framerate']
    plt.axvspan(int(fr * 60), int(fr * 62), alpha=.75, color='r', lw=0)
    #plt.axvspan(int(fr * 61), int(fr * 62), alpha=.75, color='hotpink', lw=0)
    plt.axvspan(int(fr * 62), int(fr * 68), alpha=.1, color='gray', lw=0)
    plt.axvspan(0, int(fr * 2), alpha=1, color='k', lw=0)
    plt.axvspan(int(fr * 2), int(fr * 8), alpha=.1, color='gray', lw=0)
    plt.ylabel('Reactivation bias')
    plt.xlabel('Time relative to stimulus onset (s)')
    plt.xticks([int(fr * 0), int(fr * 10), int(fr * 20), int(fr * 30), int(fr * 40), int(fr * 50), int(fr * 60),
                int(fr * 70), int(fr * 80), int(fr * 90), int(fr * 100), int(fr * 110), int(fr * 120)],
               ['0', '10', '20', '30', '40', '50', '60', '70', '80', '90', '100', '110', '120'])
    plt.xlim((0, int(fr * 120)))
    plt.ylim(-1, 1)
    #plt.yticks([0, .05, .1, .15])
    # label_1 = mlines.Line2D([], [], color='k', linestyle='-', label='Control', linewidth=3)
    # label_2 = mlines.Line2D([], [], color='r', linestyle='-', label='Inhibition', linewidth=3)
    # plt.legend(handles=[label_1, label_2], frameon=False, prop={'size': 8}, labelspacing=.1)
    plt.axhline(y=0.013425233333333333, color='black', linestyle='--', linewidth=2, snap=False, zorder=0)
    sns.despine()
    plt.savefig(paths['base_path'] + '/NN_opto/plots/reactivation_bias_trial_long.png', bbox_inches='tight', dpi=200)
    plt.savefig(paths['base_path'] + '/NN_opto/plots/reactivation_bias_trial_long.pdf', bbox_inches='tight', dpi=200, Transparent=True)

    plt.close()


def reactivation_rate_pupil_control_baseline_R1(mice, sample_dates):
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(6, 6))
    plt.subplot(2, 2, 1)
    reactivation_opto_all = np.zeros(len(mice))
    reactivation_opto_all_baseline = np.zeros(len(mice))
    for mouse in range(0, len(mice)):
        paths = preprocess_opto.create_folders(mice[mouse], sample_dates[mouse])
        y_pred_binned_across_days = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/y_pred_binned_pupil_baseline.npy',
                                            allow_pickle=True)
        reactivation = np.zeros(len(y_pred_binned_across_days[0]))
        reactivation_baseline = np.zeros(len(y_pred_binned_across_days[0]))
        for i in range(0, len(y_pred_binned_across_days[0])):
            reactivation_baseline[i] = np.mean(y_pred_binned_across_days[0][i])
            reactivation[i] = np.mean(y_pred_binned_across_days[1][i])
        reactivation_opto_all[mouse] = reactivation.mean()
        reactivation_opto_all_baseline[mouse] = reactivation_baseline.mean()
        plt.subplot(2, 2, 1)
        plt.plot([1, 2], [reactivation_baseline.mean(), reactivation.mean()], c='k', linewidth=2, linestyle='-', alpha=.2)

    print(stats.ttest_rel(reactivation_opto_all, reactivation_opto_all_baseline))

    plt.subplot(2, 2, 1)
    y0 = np.mean(reactivation_opto_all_baseline)
    y0_err = stats.sem(reactivation_opto_all_baseline)
    y1 = np.mean(reactivation_opto_all)
    y1_err = stats.sem(reactivation_opto_all)
    plt.plot([1, 2], [y0, y1], '-', c='k', linewidth=3)
    plt.fill_between([1, 2], [y0 + y0_err, y1 + y1_err], [y0 - y0_err, y1 - y1_err], alpha=0.2, color='k', lw=0)
    plt.xlim(.5, 2.5)
    plt.ylim(0, .1)
    plt.xticks([1, 2], ['Baseline\nperiod', 'Following\ninhibiton trials'])
    plt.ylabel('Pupil matched\nreactivation rate (probability $\mathregular{s^{-1}}$)')

    sns.despine()
    plt.savefig(paths['base_path'] + '/NN_opto/plots/reactivation_rate_pupil_baseline_comparison.png', bbox_inches='tight', dpi=200)
    plt.savefig(paths['base_path'] + '/NN_opto/plots/reactivation_rate_pupil_baseline_comparison.pdf', bbox_inches='tight', dpi=200, Transparent=True)
    plt.close()

























def activity_across_trials(mice, sample_dates):
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(9, 6))
    plt.subplots_adjust(wspace=.33)
    m_colors = ['yellowgreen', 'cadetblue', 'plum']

    plt.subplot(2, 2, 1)
    activity_all = np.empty((len(mice), 64)) * np.nan
    mean_activity_mice_heatmap = []
    days = []
    for mouse in range(0, len(mice)):
        paths = preprocess_opto.create_folders(mice[mouse], sample_dates[mouse])
        activity_data = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/activity.npy',
                               allow_pickle=True)
        activity = np.empty((len(activity_data[0]), 64)) * np.nan
        for i in range(0, len(activity_data[0])):
            smoothed_activity = np.array(pd.DataFrame(activity_data[0][i]).rolling(3, min_periods=1, center=True).mean())
            smoothed_activity = np.concatenate(smoothed_activity, axis=0)
            activity[i, 0:len(smoothed_activity)] = smoothed_activity
            # mean_activity_mice_heatmap.append(smoothed_activity[0:118]/np.max(smoothed_activity[0:118]))
        activity = np.nanmean(activity, axis=0)
        activity_all[mouse, :] = activity
        x = range(1, len(activity)+1)
        plt.plot(x, activity, c=m_colors[mouse], alpha=.2)
        days.append(len(activity_data[0]))

    # print(stats.ttest_rel(np.mean(activity_all[:, 0:3], axis=0), np.mean(activity_all[:, 118:121], axis=0)))
    # mask = ~np.isnan(np.concatenate([x, x, x, x])) & ~np.isnan(np.concatenate(activity_all))
    # slope, intercept, r_value, p_value, std_err = stats.linregress(
    #     np.concatenate([x, x, x, x])[mask], np.concatenate(activity_all)[mask])
    # print(r_value)
    # print(p_value)

    mean = np.nanmean(activity_all, axis=0)
    sem_plus = mean + stats.sem(activity_all, axis=0, nan_policy='omit')
    sem_minus = mean - stats.sem(activity_all, axis=0, nan_policy='omit')
    plt.plot(x, mean, c='k', linewidth=3)
    plt.fill_between(x, sem_plus, sem_minus, alpha=0.2, color='k', lw=0)
    plt.ylabel('Correlation between\nstimulus 1 and stimulus 2')
    plt.xlabel('Trial number')
    # plt.ylim(-.1, .6)
    plt.xlim(1, 60)
    plt.xticks([1, 20, 40, 60])
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
    sns.despine()

    # sns.set(font_scale=.7)
    # sns.set_style("whitegrid", {'axes.grid': False})
    # sns.set_style("ticks")
    # plt.figure(figsize=(4, 6))
    # plt.subplot(2, 2, 1)
    # sns.heatmap(mean_activity_mice_heatmap, vmin=.3, vmax=1, cmap="Reds", cbar=0)
    # y_idx = 0
    # for mouse in range(0, len(mice)):
    #     plt.plot([-5, -5], [y_idx, y_idx + days[mouse]], color=m_colors[mouse], linewidth=7, snap=False,
    #              solid_capstyle='butt')
    #     y_idx += days[mouse]
    # plt.ylim(len(mean_activity_mice_heatmap) + 3, -3)
    # plt.xlim(-5, 120)
    # plt.axis('off')
    # plt.savefig(paths['base_path'] + '/NN/plots/activity_across_trials_correlation_heatmap.png', bbox_inches='tight', dpi=500,
    #             transparent=True)

    plt.subplot(2, 2, 2)
    activity_all = np.empty((len(mice), 64)) * np.nan
    mean_activity_mice_heatmap = []
    days = []
    for mouse in range(0, len(mice)):
        paths = preprocess_opto.create_folders(mice[mouse], sample_dates[mouse])
        activity_data = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/activity.npy',
                                allow_pickle=True)
        activity = np.empty((len(activity_data[0]), 64)) * np.nan
        for i in range(0, len(activity_data[0])):
            smoothed_activity = np.array(
                pd.DataFrame(activity_data[2][i]).rolling(3, min_periods=1, center=True).mean())
            smoothed_activity = np.concatenate(smoothed_activity, axis=0)
            activity[i, 0:len(smoothed_activity)] = smoothed_activity
            # mean_activity_mice_heatmap.append(smoothed_activity[0:118])
        activity = np.nanmean(activity, axis=0)
        activity_all[mouse, :] = activity
        x = range(1, len(activity) + 1)
        plt.plot(x, activity, c=m_colors[mouse], alpha=.2)
        days.append(len(activity_data[0]))

    # mask = ~np.isnan(np.concatenate([x, x, x, x])) & ~np.isnan(np.concatenate(activity_all))
    # slope, intercept, r_value, p_value, std_err = stats.linregress(
    #     np.concatenate([x, x, x, x])[mask], np.concatenate(activity_all)[mask])
    # print(r_value)
    # print(p_value)
    #print(stats.ttest_rel(np.mean(activity_all[:, 0:3], axis=0), np.mean(activity_all[:, 118:121], axis=0)))

    # def monoExp(x, m, t, b):
    #     return m * np.exp(-t * x) + b
    # # def linear(x, m, b):
    # #     return (m * x) + b
    #
    # xs = range(0, 120)
    # ys = np.nanmean(activity_all[:, 0:120], axis=0)
    #
    # # perform the fit
    # p0 = (2000, .1, 50)  # start with values near those we expect
    # params, cv = scipy.optimize.curve_fit(monoExp, xs, ys, p0)
    # m, t, b = params
    # sampleRate = 20_000  # Hz
    # tauSec = (1 / t) / sampleRate
    #
    # # determine quality of the fit
    # squaredDiffs = np.square(ys - monoExp(xs, m, t, b))
    # squaredDiffsFromMean = np.square(ys - np.mean(ys))
    # rSquared = 1 - np.sum(squaredDiffs) / np.sum(squaredDiffsFromMean)
    # print(f"R² = {rSquared}")
    #
    # plt.figure()
    # # plot the results
    # plt.plot(xs, ys, '.', label="data")
    # plt.plot(xs, monoExp(xs, m, t, b), '--', label="fitted")
    # plt.title("Fitted Exponential Curve")
    # plt.ylim(0, .07)
    # ggg

    mean = np.nanmean(activity_all, axis=0)
    sem_plus = mean + stats.sem(activity_all, axis=0, nan_policy='omit')
    sem_minus = mean - stats.sem(activity_all, axis=0, nan_policy='omit')
    plt.plot(x, mean, c='k', linewidth=3)
    plt.fill_between(x, sem_plus, sem_minus, alpha=0.2, color='k', lw=0)
    plt.ylabel('Stimulus activity\n(Normalized $\mathregular{Ca^{2+}}$transient $\mathregular{s^{-1}}$)')
    plt.xlabel('Trial number')
    plt.ylim(0, .1)
    plt.xlim(1, 60)
    plt.xticks([1, 20, 40, 60])
    sns.despine()

    # sns.set(font_scale=.7)
    # sns.set_style("whitegrid", {'axes.grid': False})
    # sns.set_style("ticks")
    # plt.figure(figsize=(4, 6))
    # plt.subplot(2, 2, 1)
    # sns.heatmap(mean_activity_mice_heatmap, vmin=.03, vmax=.08, cmap="Reds", cbar=0)
    # y_idx = 0
    # for mouse in range(0, len(mice)):
    #     plt.plot([-5, -5], [y_idx, y_idx + days[mouse]], color=m_colors[mouse], linewidth=7, snap=False,
    #              solid_capstyle='butt')
    #     y_idx += days[mouse]
    # plt.ylim(len(mean_activity_mice_heatmap) + 3, -3)
    # plt.xlim(-5, 120)
    # plt.axis('off')
    # plt.savefig(paths['base_path'] + '/NN/plots/activity_across_trials_activity_heatmap.png', bbox_inches='tight',
    #             dpi=500,
    #             transparent=True)



    plt.savefig(paths['base_path'] + '/NN_opto/plots/activity_across_trials.png', bbox_inches='tight', dpi=200)
    plt.savefig(paths['base_path'] + '/NN_opto/plots/activity_across_trials.pdf', bbox_inches='tight', dpi=200, transparent=True)
    plt.close()


def activity_across_trials_grouped(mice, sample_dates):
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(9, 6))
    plt.subplots_adjust(wspace=.35)

    plt.subplot(2, 2, 1)
    activity_same_all = np.empty((len(mice), 128)) * np.nan
    activity_increase_all = np.empty((len(mice), 128)) * np.nan
    activity_decrease_all = np.empty((len(mice), 128)) * np.nan
    for mouse in range(0, len(mice)):
        paths = preprocess_opto.create_folders(mice[mouse], sample_dates[mouse])
        activity_data = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/activity_grouped.npy',
                               allow_pickle=True)
        activity_same = np.empty((len(activity_data[0]), 128)) * np.nan
        activity_increase = np.empty((len(activity_data[0]), 128)) * np.nan
        activity_decrease = np.empty((len(activity_data[0]), 128)) * np.nan
        for i in range(0, len(activity_data[0])):
            smoothed_activity_same = np.array(
                pd.DataFrame(activity_data[0][i][0]).rolling(3, min_periods=1, center=True).mean())
            smoothed_activity_same = np.concatenate(smoothed_activity_same, axis=0)
            smoothed_activity_increase = np.array(
                pd.DataFrame(activity_data[0][i][1]).rolling(3, min_periods=1, center=True).mean())
            smoothed_activity_increase = np.concatenate(smoothed_activity_increase, axis=0)
            smoothed_activity_decrease = np.array(
                pd.DataFrame(activity_data[0][i][2]).rolling(3, min_periods=1, center=True).mean())
            smoothed_activity_decrease = np.concatenate(smoothed_activity_decrease, axis=0)
            activity_same[i, 0:len(smoothed_activity_same)] = smoothed_activity_same
            activity_increase[i, 0:len(smoothed_activity_increase)] = smoothed_activity_increase
            activity_decrease[i, 0:len(smoothed_activity_decrease)] = smoothed_activity_decrease
        activity_same = np.nanmean(activity_same, axis=0)
        activity_same_all[mouse, :] = activity_same
        activity_increase = np.nanmean(activity_increase, axis=0)
        activity_increase_all[mouse, :] = activity_increase
        activity_decrease = np.nanmean(activity_decrease, axis=0)
        activity_decrease_all[mouse, :] = activity_decrease
        x = range(1, len(activity_same)+1)

    # print(stats.ttest_rel(
    #     np.nanmean(activity_increase_all[:, 0:2], axis=1) - np.nanmean(activity_increase_all[:, 119:121], axis=1),
    #     np.nanmean(activity_decrease_all[:, 0:3], axis=1) - np.nanmean(activity_decrease_all[:, 119:121], axis=1))[1])

    # mask = ~np.isnan(np.concatenate([x, x, x, x])) & ~np.isnan(np.concatenate(activity_decrease_all))
    # slope, intercept, r_value, p_value_dec, std_err = stats.linregress(
    #     np.concatenate([x, x, x, x])[mask], np.concatenate(activity_decrease_all)[mask])
    # print(r_value)
    # slope, intercept, r_value, p_value_inc, std_err = stats.linregress(
    #     np.concatenate([x, x, x, x])[mask], np.concatenate(activity_increase_all)[mask])
    # print(r_value)
    # anova_results = []
    # anova_results.append(p_value_dec)
    # anova_results.append(p_value_inc)
    # anova_results_corrected = multipletests(anova_results, alpha=.05, method='holm')
    # return [anova_results, anova_results_corrected[1]]

    # mean = np.nanmean(activity_same_all, axis=0)
    # sem_plus = mean + stats.sem(activity_same_all, axis=0, nan_policy='omit')
    # sem_minus = mean - stats.sem(activity_same_all, axis=0, nan_policy='omit')
    # plt.plot(x, mean, c='k', linewidth=3)
    # plt.fill_between(x, sem_plus, sem_minus, alpha=0.2, color='k', lw=0)
    mean = np.nanmean(activity_increase_all, axis=0)
    sem_plus = mean + stats.sem(activity_increase_all, axis=0, nan_policy='omit')
    sem_minus = mean - stats.sem(activity_increase_all, axis=0, nan_policy='omit')
    plt.plot(x, mean, c='darkred', linewidth=3)
    plt.fill_between(x, sem_plus, sem_minus, alpha=0.2, color='darkred', lw=0)
    mean = np.nanmean(activity_decrease_all, axis=0)
    sem_plus = mean + stats.sem(activity_decrease_all, axis=0, nan_policy='omit')
    sem_minus = mean - stats.sem(activity_decrease_all, axis=0, nan_policy='omit')
    plt.plot(x, mean, c='darkblue', linewidth=3)
    plt.fill_between(x, sem_plus, sem_minus, alpha=0.2, color='darkblue', lw=0)
    plt.ylabel('Response similarity\n(correlation between\nS1 and S2)')
    plt.xlabel('Trial number')
    # plt.ylim(-.1, .6)
    plt.xlim(1, 60)
    plt.xticks([1, 20, 40, 60])
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
    sns.despine()

    plt.subplot(2, 2, 2)
    activity_same_all = np.empty((len(mice), 128)) * np.nan
    activity_increase_all = np.empty((len(mice), 128)) * np.nan
    activity_decrease_all = np.empty((len(mice), 128)) * np.nan
    for mouse in range(0, len(mice)):
        paths = preprocess_opto.create_folders(mice[mouse], sample_dates[mouse])
        activity_data = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/activity_grouped.npy',
                                allow_pickle=True)
        activity_same = np.empty((len(activity_data[0]), 128)) * np.nan
        activity_increase = np.empty((len(activity_data[0]), 128)) * np.nan
        activity_decrease = np.empty((len(activity_data[0]), 128)) * np.nan
        for i in range(0, len(activity_data[0])):
            smoothed_activity_same = np.array(
                pd.DataFrame(activity_data[1][i][0]).rolling(3, min_periods=1, center=True).mean())
            smoothed_activity_same = np.concatenate(smoothed_activity_same, axis=0)
            smoothed_activity_increase = np.array(
                pd.DataFrame(activity_data[1][i][1]).rolling(3, min_periods=1, center=True).mean())
            smoothed_activity_increase = np.concatenate(smoothed_activity_increase, axis=0)
            smoothed_activity_decrease = np.array(
                pd.DataFrame(activity_data[1][i][2]).rolling(3, min_periods=1, center=True).mean())
            smoothed_activity_decrease = np.concatenate(smoothed_activity_decrease, axis=0)
            activity_same[i, 0:len(smoothed_activity_same)] = smoothed_activity_same
            activity_increase[i, 0:len(smoothed_activity_increase)] = smoothed_activity_increase
            activity_decrease[i, 0:len(smoothed_activity_decrease)] = smoothed_activity_decrease
        activity_same = np.nanmean(activity_same, axis=0)
        activity_same_all[mouse, :] = activity_same
        activity_increase = np.nanmean(activity_increase, axis=0)
        activity_increase_all[mouse, :] = activity_increase
        activity_decrease = np.nanmean(activity_decrease, axis=0)
        activity_decrease_all[mouse, :] = activity_decrease
        x = range(1, len(activity_same) + 1)

    # anova_results = []
    # anova_results.append(stats.ttest_rel(np.concatenate(activity_same_all[:, 0:3]),
    #                                      np.concatenate(activity_same_all[:, 118:121]))[1])
    # anova_results.append(stats.ttest_rel(np.concatenate(activity_increase_all[:, 0:3]),
    #                                      np.concatenate(activity_increase_all[:, 118:121]))[1])
    # anova_results.append(stats.ttest_rel(np.concatenate(activity_decrease_all[:, 0:3]),
    #                                      np.concatenate(activity_decrease_all[:, 118:121]))[1])
    # anova_results_corrected = multipletests(anova_results, alpha=.05, method='holm')
    # return [anova_results, anova_results_corrected[1]]

    # mask = ~np.isnan(np.concatenate([x, x, x, x])) & ~np.isnan(np.concatenate(activity_decrease_all))
    # slope, intercept, r_value, p_value_dec, std_err = stats.linregress(
    #     np.concatenate([x, x, x, x])[mask], np.concatenate(activity_decrease_all)[mask])
    # print(r_value)
    # slope, intercept, r_value, p_value_n, std_err = stats.linregress(
    #     np.concatenate([x, x, x, x])[mask], np.concatenate(activity_same_all)[mask])
    # print(r_value)
    # slope, intercept, r_value, p_value_inc, std_err = stats.linregress(
    #     np.concatenate([x, x, x, x])[mask], np.concatenate(activity_increase_all)[mask])
    # print(r_value)
    # anova_results = []
    # anova_results.append(p_value_dec)
    # anova_results.append(p_value_n)
    # anova_results.append(p_value_inc)
    # anova_results_corrected = multipletests(anova_results, alpha=.05, method='holm')
    # return [anova_results, anova_results_corrected[1]]

    mean = np.nanmean(activity_same_all, axis=0)
    sem_plus = mean + stats.sem(activity_same_all, axis=0, nan_policy='omit')
    sem_minus = mean - stats.sem(activity_same_all, axis=0, nan_policy='omit')
    plt.plot(x, mean, c='k', linewidth=3)
    plt.fill_between(x, sem_plus, sem_minus, alpha=0.2, color='k', lw=0)
    mean = np.nanmean(activity_increase_all, axis=0)
    sem_plus = mean + stats.sem(activity_increase_all, axis=0, nan_policy='omit')
    sem_minus = mean - stats.sem(activity_increase_all, axis=0, nan_policy='omit')
    plt.plot(x, mean, c='darkred', linewidth=3)
    plt.fill_between(x, sem_plus, sem_minus, alpha=0.2, color='darkred', lw=0)
    mean = np.nanmean(activity_decrease_all, axis=0)
    sem_plus = mean + stats.sem(activity_decrease_all, axis=0, nan_policy='omit')
    sem_minus = mean - stats.sem(activity_decrease_all, axis=0, nan_policy='omit')
    plt.plot(x, mean, c='darkblue', linewidth=3)
    plt.fill_between(x, sem_plus, sem_minus, alpha=0.2, color='darkblue', lw=0)
    plt.ylabel('Stimulus activity\n(Normalized $\mathregular{Ca^{2+}}$transient $\mathregular{s^{-1}}$)')
    plt.xlabel('Trial number')
    # plt.ylim(0, .07)
    plt.xlim(1, 60)
    plt.xticks([1, 20, 40, 60])
    plt.ylim(0, .12)
    label_1 = mlines.Line2D([], [], color='k', linewidth=2, label='No change cells')
    label_2 = mlines.Line2D([], [], color='darkred', linewidth=2, label='Increase cells')
    label_3 = mlines.Line2D([], [], color='darkblue', linewidth=2, label='Decrease cells')
    plt.legend(handles=[label_1, label_2, label_3], frameon=False)
    sns.despine()

    plt.savefig(paths['base_path'] + '/NN_opto/plots/activity_across_trials_grouped.png', bbox_inches='tight', dpi=200)
    plt.savefig(paths['base_path'] + '/NN_opto/plots/activity_across_trials_grouped.pdf', bbox_inches='tight', dpi=200, transparent=True)
    plt.close()


def activity_across_trials_grouped_decrease(mice, sample_dates):
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(9, 6))
    plt.subplots_adjust(wspace=.35)

    plt.subplot(2, 2, 1)
    cs1d_cs1_all = np.empty((len(mice), 128)) * np.nan
    cs1d_cs2_all = np.empty((len(mice), 128)) * np.nan
    cs2d_cs2_all = np.empty((len(mice), 128)) * np.nan
    cs2d_cs1_all = np.empty((len(mice), 128)) * np.nan
    for mouse in range(0, len(mice)):
        paths = preprocess_opto.create_folders(mice[mouse], sample_dates[mouse])
        activity_data = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/activity_grouped_decrease.npy',
                               allow_pickle=True)
        cs1d_cs1 = np.empty((len(activity_data[0]), 128)) * np.nan
        cs1d_cs2 = np.empty((len(activity_data[0]), 128)) * np.nan
        cs2d_cs2 = np.empty((len(activity_data[0]), 128)) * np.nan
        cs2d_cs1 = np.empty((len(activity_data[0]), 128)) * np.nan
        for i in range(0, len(activity_data[0])):
            cs1d_cs1[i, 0:len(activity_data[0][i])] = activity_data[0][i]
            cs1d_cs2[i, 0:len(activity_data[1][i])] = activity_data[1][i]
            cs2d_cs2[i, 0:len(activity_data[2][i])] = activity_data[2][i]
            cs2d_cs1[i, 0:len(activity_data[3][i])] = activity_data[3][i]
        cs1d_cs1 = np.nanmean(cs1d_cs1, axis=0)
        cs1d_cs1_all[mouse, :] = cs1d_cs1
        cs1d_cs2 = np.nanmean(cs1d_cs2, axis=0)
        cs1d_cs2_all[mouse, :] = cs1d_cs2
        cs2d_cs2 = np.nanmean(cs2d_cs2, axis=0)
        cs2d_cs2_all[mouse, :] = cs2d_cs2
        cs2d_cs1 = np.nanmean(cs2d_cs1, axis=0)
        cs2d_cs1_all[mouse, :] = cs2d_cs1
        x = range(1, len(cs1d_cs1)+1)

    # anova_results = []
    #
    # anova_results.append(stats.ttest_rel(np.mean(cs1d_cs1_all[:, 0:3], axis=0),
    #                                      np.mean(cs1d_cs2_all[:, 0:3], axis=0))[1])
    # anova_results.append(stats.ttest_rel(np.mean(cs2d_cs2_all[:, 0:3], axis=0),
    #                                      np.mean(cs2d_cs1_all[:, 0:3], axis=0))[1])
    # anova_results.append(stats.ttest_rel(np.mean(cs1d_cs1_all[:, 58:61], axis=0),
    #                                      np.mean(cs1d_cs2_all[:, 58:61], axis=0))[1])
    # anova_results.append(stats.ttest_rel(np.mean(cs2d_cs2_all[:, 58:61], axis=0),
    #                                      np.mean(cs2d_cs1_all[:, 58:61], axis=0))[1])
    #
    # anova_results_corrected = multipletests(anova_results, alpha=.05, method='holm')
    # return [anova_results, anova_results_corrected[1]]


    mean = np.nanmean(cs1d_cs1_all, axis=0)
    sem_plus = mean + stats.sem(cs1d_cs1_all, axis=0, nan_policy='omit')
    sem_minus = mean - stats.sem(cs1d_cs1_all, axis=0, nan_policy='omit')
    plt.plot(x, mean, c='mediumseagreen', linewidth=3)
    plt.fill_between(x, sem_plus, sem_minus, alpha=0.2, color='mediumseagreen', lw=0)
    mean = np.nanmean(cs1d_cs2_all, axis=0)
    sem_plus = mean + stats.sem(cs1d_cs2_all, axis=0, nan_policy='omit')
    sem_minus = mean - stats.sem(cs1d_cs2_all, axis=0, nan_policy='omit')
    plt.plot(x, mean, c='darkgreen', linewidth=3)
    plt.fill_between(x, sem_plus, sem_minus, alpha=0.2, color='darkgreen', lw=0)
    mean = np.nanmean(cs2d_cs2_all, axis=0)
    sem_plus = mean + stats.sem(cs2d_cs2_all, axis=0, nan_policy='omit')
    sem_minus = mean - stats.sem(cs2d_cs2_all, axis=0, nan_policy='omit')
    plt.plot(x, mean, c='salmon', linewidth=3)
    plt.fill_between(x, sem_plus, sem_minus, alpha=0.2, color='salmon', lw=0)
    mean = np.nanmean(cs2d_cs1_all, axis=0)
    sem_plus = mean + stats.sem(cs2d_cs1_all, axis=0, nan_policy='omit')
    sem_minus = mean - stats.sem(cs2d_cs1_all, axis=0, nan_policy='omit')
    plt.plot(x, mean, c='darkred', linewidth=3)
    plt.fill_between(x, sem_plus, sem_minus, alpha=0.2, color='darkred', lw=0)
    plt.ylabel('Stimulus activity\n(Normalized $\mathregular{Ca^{2+}}$transient $\mathregular{s^{-1}}$)')
    plt.xlabel('Trial number')
    plt.ylim(0, .13)
    plt.xlim(1, 30)
    plt.xticks([1, 10, 20, 30])
    plt.yticks([0, .05, .1])
    sns.despine()
    # label_1 = mlines.Line2D([], [], color='mediumseagreen', linewidth=2, label='S1 no change cells, S1 trials')
    # label_2 = mlines.Line2D([], [], color='darkgreen', linewidth=2, label='S1 no change cells, S2 trials')
    # label_3 = mlines.Line2D([], [], color='salmon', linewidth=2, label='S2 no change cells, S2 trials')
    # label_4 = mlines.Line2D([], [], color='darkred', linewidth=2, label='S2 no change cells, S1 trials')
    # plt.legend(handles=[label_1, label_2, label_3, label_4], frameon=False)

    plt.savefig(paths['base_path'] + '/NN_opto/plots/activity_across_trials_grouped_decrease.png', bbox_inches='tight', dpi=200)
    plt.savefig(paths['base_path'] + '/NN_opto/plots/activity_across_trials_grouped_decrease.pdf', bbox_inches='tight', dpi=200, transparent=True)
    plt.close()


def reactivation_cue_vector(mice, sample_dates):
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(9, 6))
    plt.subplots_adjust(wspace=.3)
    x_label = list(range(0, 30))
    paths = preprocess_opto.create_folders(mice[0], sample_dates[0])
    reactivation_cue_pca_vec = np.load(paths['base_path'] + paths['mouse'] +
                                    '/data_across_days/reactivation_cue_vector.npy', allow_pickle=True)
    all_s1 = np.zeros((len(mice), len(reactivation_cue_pca_vec[0][0])))
    all_s1r = np.zeros((len(mice), len(reactivation_cue_pca_vec[0][0])))
    all_s2 = np.zeros((len(mice), len(reactivation_cue_pca_vec[0][0])))
    all_s2r = np.zeros((len(mice), len(reactivation_cue_pca_vec[0][0])))
    for mouse in range(0, len(mice)):
        paths = preprocess_opto.create_folders(mice[mouse], sample_dates[mouse])
        reactivation_cue_pca_vec = np.load(paths['base_path'] + paths['mouse'] +
                                           '/data_across_days/reactivation_cue_vector.npy', allow_pickle=True)
        temp_s1 = np.zeros((len(reactivation_cue_pca_vec[0]), len(reactivation_cue_pca_vec[0][0])))
        temp_s1r = np.zeros((len(reactivation_cue_pca_vec[0]), len(reactivation_cue_pca_vec[0][0])))
        temp_s2 = np.zeros((len(reactivation_cue_pca_vec[0]), len(reactivation_cue_pca_vec[0][0])))
        temp_s2r = np.zeros((len(reactivation_cue_pca_vec[0]), len(reactivation_cue_pca_vec[0][0])))
        for i in range(0, len(reactivation_cue_pca_vec[0])):
            temp_s1[i, :] = reactivation_cue_pca_vec[0][i]
            temp_s1r[i, :] = reactivation_cue_pca_vec[1][i]
            temp_s2[i, :] = reactivation_cue_pca_vec[2][i]
            temp_s2r[i, :] = reactivation_cue_pca_vec[3][i]
        y_s1 = np.nanmean(temp_s1, axis=0)
        y_s1r = np.nanmean(temp_s1r, axis=0)
        y_s2 = np.nanmean(temp_s2, axis=0)
        y_s2r = np.nanmean(temp_s2r, axis=0)
        # plt.subplot(2, 2, 1)
        # plt.plot(x_label, y_s1, '-', c='darkgreen', alpha=.3, linewidth=2)
        # plt.plot(x_label, y_s1r, '-', c='lime', alpha=.3, linewidth=2)
        # plt.subplot(2, 2, 2)
        # plt.plot(x_label, y_s2, '-', c='darkred', alpha=.3, linewidth=2)
        # plt.plot(x_label, y_s2r, '-', c='hotpink', alpha=.3, linewidth=2)
        all_s1[mouse, :] = y_s1
        all_s1r[mouse, :] = y_s1r
        all_s2[mouse, :] = y_s2
        all_s2r[mouse, :] = y_s2r


    #print(stats.ttest_rel(all_s2[:, 0:1], all_s2r[:, 59:60])[1])

    plt.subplot(2, 2, 1)
    plt.ylim(-1.2, .2)
    plt.gca().invert_yaxis()
    mean = all_s1.mean(axis=0)
    sem_plus = mean + stats.sem(all_s1, axis=0)
    sem_minus = mean - stats.sem(all_s1, axis=0)
    plt.plot(x_label, mean, '-', c='darkgreen', linewidth=3)
    plt.fill_between(x_label, sem_plus, sem_minus, alpha=0.2, color='darkgreen', lw=0)
    mean = all_s1r.mean(axis=0)
    sem_plus = mean + stats.sem(all_s1r, axis=0)
    sem_minus = mean - stats.sem(all_s1r, axis=0)
    plt.plot(x_label, mean, '-', c='lime', linewidth=3)
    plt.fill_between(x_label, sem_plus, sem_minus, alpha=0.2, color='lime', lw=0)
    plt.ylabel('Similarity to early vs. late\n S1 response pattern')
    plt.xlabel('Trial number')
    label_1 = mlines.Line2D([], [], color='darkgreen', linewidth=2, label='S1')
    label_2 = mlines.Line2D([], [], color='lime', linewidth=2, label='S1 reactivation')
    # label_3 = mlines.Line2D([], [], color='k', linewidth=2, label='S1 reactivation baseline')
    # plt.legend(handles=[label_1, label_2, label_3], frameon=False)
    # plt.errorbar(-2.5, np.mean(all_s1b), yerr=stats.sem(all_s1b), c='k', linewidth=2, marker='o', mfc='k', mec='k',
    #              ms=5, mew=0, zorder=100)
    # plt.axvspan(-5, 0, alpha=.1, color='gray')
    plt.xlim(-1, 30)
    plt.xticks([0, 9, 19, 29], ['1', '10', '20', '30'])
    plt.subplot(2, 2, 2)
    plt.ylim(-1.2, .2)
    plt.gca().invert_yaxis()
    mean = all_s2.mean(axis=0)
    sem_plus = mean + stats.sem(all_s2, axis=0)
    sem_minus = mean - stats.sem(all_s2, axis=0)
    plt.plot(x_label, mean, '-', c='darkred', linewidth=3)
    plt.fill_between(x_label, sem_plus, sem_minus, alpha=0.2, color='darkred', lw=0)
    mean = all_s2r.mean(axis=0)
    sem_plus = mean + stats.sem(all_s2r, axis=0)
    sem_minus = mean - stats.sem(all_s2r, axis=0)
    plt.plot(x_label, mean, '-', c='hotpink', linewidth=3)
    plt.fill_between(x_label, sem_plus, sem_minus, alpha=0.2, color='hotpink', lw=0)
    plt.ylabel('Similarity to early vs. late\n S2 response pattern)')
    plt.xlabel('Trial number')
    label_1 = mlines.Line2D([], [], color='darkred', linewidth=2, label='S2')
    label_2 = mlines.Line2D([], [], color='hotpink', linewidth=2, label='S2 reactivation')
    # label_3 = mlines.Line2D([], [], color='k', linewidth=2, label='S2 reactivation baseline')
    # plt.legend(handles=[label_1, label_2, label_3], frameon=False)
    # plt.axvspan(-5, 0, alpha=.1, color='gray')
    # plt.errorbar(-2.5, np.mean(all_s2b), yerr=stats.sem(all_s2b), c='k', linewidth=2, marker='o', mfc='k', mec='k',
    #              ms=5, mew=0, zorder=100)
    plt.xlim(-1, 30)
    plt.xticks([0, 9, 19, 29], ['1', '10', '20', '30'])
    sns.despine()
    plt.savefig(paths['base_path'] + '/NN_opto/plots/reactivation_cue_vector.png', bbox_inches='tight', dpi=200)
    plt.savefig(paths['base_path'] + '/NN_opto/plots/reactivation_cue_vector.pdf', bbox_inches='tight', dpi=200, transparent=True)
    plt.close()


def reactivation_influence(mice, sample_dates):
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(7, 6))
    m_colors = ['yellowgreen', 'cadetblue', 'plum']
    x = [1, 2, 3, 4]
    decrease_cs_1_all = []
    increase_cs_1_all = []
    no_change_cs_1_all = []
    decrease_cs_2_all = []
    increase_cs_2_all = []
    no_change_cs_2_all = []
    for mouse in range(0, len(mice)):
        paths = preprocess_opto.create_folders(mice[mouse], sample_dates[mouse])
        reactivation_influence = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/reactivation_influence.npy', allow_pickle=True)
        decrease_cs_1 = []
        increase_cs_1 = []
        no_change_cs_1 = []
        decrease_cs_2 = []
        increase_cs_2 = []
        no_change_cs_2 = []
        for i in range(0, len(reactivation_influence[0])):
            decrease_cs_1.append(reactivation_influence[0][i])
            increase_cs_1.append(reactivation_influence[2][i])
            no_change_cs_1.append(reactivation_influence[1][i])
            decrease_cs_2.append(reactivation_influence[3][i])
            increase_cs_2.append(reactivation_influence[5][i])
            no_change_cs_2.append(reactivation_influence[4][i])
        decrease_cs_1_all.append(np.nanmean(decrease_cs_1))
        increase_cs_1_all.append(np.nanmean(increase_cs_1))
        no_change_cs_1_all.append(np.nanmean(no_change_cs_1))
        decrease_cs_2_all.append(np.nanmean(decrease_cs_2))
        increase_cs_2_all.append(np.nanmean(increase_cs_2))
        no_change_cs_2_all.append(np.nanmean(no_change_cs_2))
        plt.subplot(2, 2, 1)
        plt.errorbar(x[0], np.nanmean(decrease_cs_1), yerr=0, c=m_colors[mouse], marker='o', mfc='none', mec=m_colors[mouse],
                     ms=5, alpha=.3)
        plt.errorbar(x[1], np.nanmean(no_change_cs_1), yerr=0, c=m_colors[mouse], marker='o', mfc='none', mec=m_colors[mouse],
                     ms=5, alpha=.3)
        plt.errorbar(x[2], np.nanmean(increase_cs_1), yerr=0, c=m_colors[mouse], marker='o', mfc='none', mec=m_colors[mouse],
                     ms=5, alpha=.3)
        plt.subplot(2, 2, 2)
        plt.errorbar(x[0], np.nanmean(decrease_cs_2), yerr=0, c=m_colors[mouse], marker='o', mfc='none', mec=m_colors[mouse],
                     ms=5, alpha=.3)
        plt.errorbar(x[1], np.nanmean(no_change_cs_2), yerr=0, c=m_colors[mouse], marker='o', mfc='none',
                     mec=m_colors[mouse],
                     ms=5, alpha=.3)
        plt.errorbar(x[2], np.nanmean(increase_cs_2), yerr=0, c=m_colors[mouse], marker='o', mfc='none', mec=m_colors[mouse],
                     ms=5, alpha=.3)

    # print(stats.f_oneway(decrease_cs_1_all, no_change_cs_1_all, increase_cs_1_all))
    # print(
    #     pairwise_tukeyhsd(np.concatenate([decrease_cs_1_all, no_change_cs_1_all, increase_cs_1_all], axis=0),
    #                       ['d', 'd', 'd', 'd', 'd', 'n', 'n', 'n', 'n', 'n', 'i', 'i', 'i', 'i', 'i'], alpha=0.05))
    # print(stats.f_oneway(decrease_cs_2_all, no_change_cs_2_all, increase_cs_2_all))
    # print(
    #     pairwise_tukeyhsd(np.concatenate([decrease_cs_2_all, no_change_cs_2_all, increase_cs_2_all], axis=0),
    #                       ['d', 'd', 'd', 'd', 'd', 'n', 'n', 'n', 'n', 'n', 'i', 'i', 'i', 'i', 'i'], alpha=0.05))

    plt.subplot(2, 2, 1)
    y1 = np.mean(decrease_cs_1_all)
    y1_err = stats.sem(decrease_cs_1_all)
    plt.errorbar(1.2, y1, yerr=y1_err, c='mediumseagreen', linewidth=2, marker='o', mfc='mediumseagreen', mec='mediumseagreen',
                 ms=7, mew=0, zorder=100)
    y2 = np.mean(no_change_cs_1_all)
    y2_err = stats.sem(no_change_cs_1_all)
    plt.errorbar(2.2, y2, yerr=y2_err, c='mediumseagreen', linewidth=2, marker='o', mfc='mediumseagreen', mec='mediumseagreen',
                 ms=7, mew=0, zorder=100)
    y3 = np.mean(increase_cs_1_all)
    y3_err = stats.sem(increase_cs_1_all)
    plt.errorbar(3.2, y3, yerr=y3_err, c='mediumseagreen', linewidth=2, marker='o', mfc='mediumseagreen', mec='mediumseagreen',
                 ms=7, mew=0, zorder=100)
    plt.xlim(.5, 3.5)
    plt.ylim(-0.04, .04)
    plt.xticks([1, 2, 3], ['Decrease', 'No change', 'Increase'])
    # plt.ylabel('Stimulus activity\n(normalized deconvolved\n$\mathregular{Ca^{2+}}$transient $\mathregular{s^{-1}}$)')

    plt.subplot(2, 2, 2)
    y1 = np.mean(decrease_cs_2_all)
    y1_err = stats.sem(decrease_cs_2_all)
    plt.errorbar(1.2, y1, yerr=y1_err, c='salmon', linewidth=2, marker='o', mfc='salmon', mec='salmon',
                 ms=7, mew=0, zorder=100)
    y2 = np.mean(no_change_cs_2_all)
    y2_err = stats.sem(no_change_cs_2_all)
    plt.errorbar(2.2, y2, yerr=y2_err, c='salmon', linewidth=2, marker='o', mfc='salmon', mec='salmon',
                 ms=7, mew=0, zorder=100)
    y3 = np.mean(increase_cs_2_all)
    y3_err = stats.sem(increase_cs_2_all)
    plt.errorbar(3.2, y3, yerr=y3_err, c='salmon', linewidth=2, marker='o', mfc='salmon', mec='salmon',
                 ms=7, mew=0, zorder=100)
    plt.xlim(.5, 3.5)
    plt.ylim(-0.04, .04)
    plt.xticks([1, 2, 3], ['Decrease', 'No change', 'Increase'])
    # plt.ylabel('Stimulus activity\n(normalized deconvolved\n$\mathregular{Ca^{2+}}$transient $\mathregular{s^{-1}}$)')
    sns.despine()
    plt.savefig(paths['base_path'] + '/NN_opto/plots/reactivation_influence.png', bbox_inches='tight', dpi=200)
    plt.savefig(paths['base_path'] + '/NN_opto/plots/reactivation_influence.pdf', bbox_inches='tight', dpi=200,
                transparent=True)
    plt.close()


def activity_difference_grouped(mice, sample_dates):
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(5, 6))
    m_colors = ['b', 'purple', 'darkorange', 'green', 'darkred']

    plt.subplot(2, 2, 1)
    cs1c_d_all = []
    cs2c_d_all = []
    cs1c_i_all = []
    cs2c_i_all = []
    cs1r_d_all = []
    cs2r_d_all = []
    cs1r_i_all = []
    cs2r_i_all = []
    x = [1, 2, 3, 4]
    for mouse in range(0, len(mice)):
        paths = preprocess_opto.create_folders(mice[mouse], sample_dates[mouse])
        diff_all = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/activity_difference_grouped.npy',
                             allow_pickle=True)
        cs1c_d = []
        cs2c_d = []
        cs1c_i = []
        cs2c_i = []
        cs1r_d = []
        cs2r_d = []
        cs1r_i = []
        cs2r_i = []
        for i in range(0, len(diff_all[0])):
            cs1c_d.append(diff_all[0][i])
            cs2c_d.append(diff_all[1][i])
            cs1c_i.append(diff_all[2][i])
            cs2c_i.append(diff_all[3][i])
            cs1r_d.append(diff_all[4][i])
            cs2r_d.append(diff_all[5][i])
            cs1r_i.append(diff_all[6][i])
            cs2r_i.append(diff_all[7][i])
        cs1c_d = np.nanmean(cs1c_d)
        cs2c_d = np.nanmean(cs2c_d)
        cs1c_i = np.nanmean(cs1c_i)
        cs2c_i = np.nanmean(cs2c_i)
        cs1r_d = np.nanmean(cs1r_d)
        cs2r_d = np.nanmean(cs2r_d)
        cs1r_i = np.nanmean(cs1r_i)
        cs2r_i = np.nanmean(cs2r_i)
        plt.subplot(2, 2, 1)
        plt.plot(x[0:2], [cs1c_d, cs1r_d], c=m_colors[mouse], linewidth=2, linestyle='-', alpha=.2)
        plt.plot(x[2:4], [cs2c_d, cs2r_d], c=m_colors[mouse], linewidth=2, linestyle='-', alpha=.2)
        plt.subplot(2, 2, 2)
        plt.plot(x[0:2], [cs1c_i, cs1r_i], c=m_colors[mouse], linewidth=2, linestyle='-', alpha=.2)
        plt.plot(x[2:4], [cs2c_i, cs2r_i], c=m_colors[mouse], linewidth=2, linestyle='-', alpha=.2)
        cs1c_d_all.append(cs1c_d)
        cs2c_d_all.append(cs2c_d)
        cs1c_i_all.append(cs1c_i)
        cs2c_i_all.append(cs2c_i)
        cs1r_d_all.append(cs1r_d)
        cs2r_d_all.append(cs2r_d)
        cs1r_i_all.append(cs1r_i)
        cs2r_i_all.append(cs2r_i)

    # anova_results = []
    # anova_results.append(stats.ttest_rel(cs1c_d_all, cs1r_d_all)[1])
    # anova_results.append(stats.ttest_rel(cs2c_d_all, cs2r_d_all)[1])
    # anova_results_corrected = multipletests(anova_results, alpha=.05, method='holm')
    # return [anova_results, anova_results_corrected[1]]

    y0 = np.mean(cs1c_d_all)
    y1 = np.mean(cs2c_d_all)
    y2 = np.mean(cs1c_i_all)
    y3 = np.mean(cs2c_i_all)
    y4 = np.mean(cs1r_d_all)
    y5 = np.mean(cs2r_d_all)
    y6 = np.mean(cs1r_i_all)
    y7 = np.mean(cs2r_i_all)
    y0_err = stats.sem(cs1c_d_all)
    y1_err = stats.sem(cs2c_d_all)
    y2_err = stats.sem(cs1c_i_all)
    y3_err = stats.sem(cs2c_i_all)
    y4_err = stats.sem(cs1r_d_all)
    y5_err = stats.sem(cs2r_d_all)
    y6_err = stats.sem(cs1r_i_all)
    y7_err = stats.sem(cs2r_i_all)


    plt.subplot(2, 2, 1)
    plt.plot(x[0:2], [y0, y4], '-', c='mediumseagreen', linewidth=3)
    plt.fill_between(x[0:2], [y0 + y0_err, y4 + y4_err], [y0 - y0_err, y4 - y4_err], alpha=0.2, color='mediumseagreen', lw=0)
    plt.plot(x[2:4], [y1, y5], '-', c='salmon', linewidth=3)
    plt.fill_between(x[2:4], [y1 + y1_err, y5 + y5_err], [y1 - y1_err, y5 - y5_err], alpha=0.2, color='salmon', lw=0)
    plt.xticks([1, 2, 3, 4], ['S1', 'S1R', 'S2', 'S2R'])
    plt.ylabel('Ratio of decrease /\nno change neuron activity')
    plt.ylim((0, 2))
    plt.xlim((.5, 4.5))
    plt.subplot(2, 2, 2)
    plt.plot(x[0:2], [y2, y6], '-', c='mediumseagreen', linewidth=3)
    plt.fill_between(x[0:2], [y2 + y2_err, y6 + y6_err], [y2 - y2_err, y6 - y6_err], alpha=0.2, color='mediumseagreen',
                     lw=0)
    plt.plot(x[2:4], [y3, y7], '-', c='salmon', linewidth=3)
    plt.fill_between(x[2:4], [y3 + y3_err, y7 + y7_err], [y3 - y3_err, y7 - y7_err], alpha=0.2, color='salmon',
                     lw=0)
    plt.xlim((.5, 4.5))
    plt.ylim((0, 2))
    plt.xticks([1, 2, 3, 4], ['S1', 'S1R', 'S2', 'S2R'])
    plt.ylabel('Ratio of increase /\nno change neuron activity')
    sns.despine()
    plt.savefig(paths['base_path'] + '/NN_opto/plots/activity_difference_grouped.png', bbox_inches='tight', dpi=200)
    plt.savefig(paths['base_path'] + '/NN_opto/plots/activity_difference_grouped.pdf', bbox_inches='tight', dpi=200, transparent=True)
    plt.close()


def reactivation_cue_vector_evolve(mice, sample_dates):
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(9, 6))
    plt.subplots_adjust(wspace=.3)
    x_label = list(range(0, 30))
    paths = preprocess_opto.create_folders(mice[0], sample_dates[0])
    reactivation_cue_pca_vec = np.load(paths['base_path'] + paths['mouse'] +
                                    '/data_across_days/reactivation_cue_vector_evolve.npy', allow_pickle=True)
    all_s1 = np.zeros((len(mice), len(reactivation_cue_pca_vec[0][0])))
    all_s1r = np.zeros((len(mice), len(reactivation_cue_pca_vec[0][0])))
    all_s2 = np.zeros((len(mice), len(reactivation_cue_pca_vec[0][0])))
    all_s2r = np.zeros((len(mice), len(reactivation_cue_pca_vec[0][0])))
    for mouse in range(0, len(mice)):
        paths = preprocess_opto.create_folders(mice[mouse], sample_dates[mouse])
        reactivation_cue_pca_vec = np.load(paths['base_path'] + paths['mouse'] +
                                           '/data_across_days/reactivation_cue_vector_evolve.npy', allow_pickle=True)
        temp_s1 = np.zeros((len(reactivation_cue_pca_vec[0]), len(reactivation_cue_pca_vec[0][0])))
        temp_s1r = np.zeros((len(reactivation_cue_pca_vec[0]), len(reactivation_cue_pca_vec[0][0])))
        temp_s2 = np.zeros((len(reactivation_cue_pca_vec[0]), len(reactivation_cue_pca_vec[0][0])))
        temp_s2r = np.zeros((len(reactivation_cue_pca_vec[0]), len(reactivation_cue_pca_vec[0][0])))
        for i in range(0, len(reactivation_cue_pca_vec[0])):
            temp_s1[i, :] = reactivation_cue_pca_vec[2][i]
            temp_s1r[i, :] = reactivation_cue_pca_vec[0][i]
            temp_s2[i, :] = reactivation_cue_pca_vec[3][i]
            temp_s2r[i, :] = reactivation_cue_pca_vec[1][i]
        y_s1 = np.nanmean(temp_s1, axis=0)
        y_s1r = np.nanmean(temp_s1r, axis=0)
        y_s2 = np.nanmean(temp_s2, axis=0)
        y_s2r = np.nanmean(temp_s2r, axis=0)
        all_s1[mouse, :] = y_s1
        all_s1r[mouse, :] = y_s1r
        all_s2[mouse, :] = y_s2
        all_s2r[mouse, :] = y_s2r

    # print(stats.ttest_rel(np.mean(all_s1, axis=1), np.mean(all_s1r, axis=1))[1])
    # print(stats.ttest_rel(np.mean(all_s2, axis=1), np.mean(all_s2r, axis=1))[1])

    plt.subplot(2, 2, 1)
    plt.ylim(-1.2, .2)
    plt.gca().invert_yaxis()
    mean = all_s1.mean(axis=0)
    sem_plus = mean + stats.sem(all_s1, axis=0)
    sem_minus = mean - stats.sem(all_s1, axis=0)
    plt.plot(x_label, mean, '-', c='darkgreen', linewidth=3)
    plt.fill_between(x_label, sem_plus, sem_minus, alpha=0.2, color='darkgreen', lw=0)
    mean = all_s1r.mean(axis=0)
    sem_plus = mean + stats.sem(all_s1r, axis=0)
    sem_minus = mean - stats.sem(all_s1r, axis=0)
    plt.plot(x_label, mean, '-', c='lime', linewidth=3)
    plt.fill_between(x_label, sem_plus, sem_minus, alpha=0.2, color='lime', lw=0)
    plt.ylabel('Similarity to early vs. late\n S1 response pattern')
    plt.xlabel('Trial number')
    label_1 = mlines.Line2D([], [], color='darkgreen', linewidth=2, label='S1')
    label_2 = mlines.Line2D([], [], color='lime', linewidth=2, label='S1 modeled')
    plt.legend(handles=[label_1, label_2], frameon=False)
    plt.xlim(0, 30)
    plt.xticks([0, 9, 19, 29], ['1', '10', '20', '30'])
    plt.subplot(2, 2, 2)
    plt.ylim(-1.2, .2)
    plt.gca().invert_yaxis()
    mean = all_s2.mean(axis=0)
    sem_plus = mean + stats.sem(all_s2, axis=0)
    sem_minus = mean - stats.sem(all_s2, axis=0)
    plt.plot(x_label, mean, '-', c='darkred', linewidth=3)
    plt.fill_between(x_label, sem_plus, sem_minus, alpha=0.2, color='darkred', lw=0)
    mean = all_s2r.mean(axis=0)
    sem_plus = mean + stats.sem(all_s2r, axis=0)
    sem_minus = mean - stats.sem(all_s2r, axis=0)
    plt.plot(x_label, mean, '-', c='hotpink', linewidth=3)
    plt.fill_between(x_label, sem_plus, sem_minus, alpha=0.2, color='hotpink', lw=0)
    plt.ylabel('Similarity to early vs. late\n S2 response pattern')
    plt.xlabel('Trial number')
    label_1 = mlines.Line2D([], [], color='darkred', linewidth=2, label='S2')
    label_2 = mlines.Line2D([], [], color='hotpink', linewidth=2, label='S2 modeled')
    plt.legend(handles=[label_1, label_2], frameon=False)
    plt.xlim(0, 30)
    plt.xticks([0, 9, 19, 29], ['1', '10', '20', '30'])
    sns.despine()
    plt.savefig(paths['base_path'] + '/NN_opto/plots/reactivation_cue_vector_evolve.png', bbox_inches='tight', dpi=200)
    plt.savefig(paths['base_path'] + '/NN_opto/plots/reactivation_cue_vector_evolve.pdf', bbox_inches='tight', dpi=200,
                transparent=True)
    plt.close()
























