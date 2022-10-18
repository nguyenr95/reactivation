import math
import warnings
import preprocess
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy import signal
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
warnings.filterwarnings('ignore')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def p_distribution_shuffle(mice, sample_dates):
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(9, 6))
    paths = preprocess.create_folders(mice[0], sample_dates[0])
    p_all = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/p_shuffle.npy',
                    allow_pickle=True)
    p_norm_total = np.zeros((len(mice), len(p_all[0][0])))
    p_shuffle_prior_total = np.zeros((len(mice), len(p_all[0][0])))
    p_shuffle_beta_total = np.zeros((len(mice), len(p_all[0][0])))
    p_shuffle_prior_fold_total = np.zeros((len(mice), len(p_all[0][0])))
    p_shuffle_beta_fold_total = np.zeros((len(mice), len(p_all[0][0])))
    for mouse in range(0, len(mice)):
        paths = preprocess.create_folders(mice[mouse], sample_dates[mouse])
        p_all = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/p_shuffle.npy', allow_pickle=True)
        p_norm = []
        p_shuffle_prior = []
        p_shuffle_beta = []
        for i in range(0, len(p_all[0])):
            p_norm.append(p_all[0][i])
            p_shuffle_prior.append(p_all[1][i])
            p_shuffle_beta.append(p_all[2][i])
        # plt.subplot(2, 2, 1)
        # plt.plot(list(range(0, len(np.mean(p_norm, axis=0)))), np.mean(p_norm, axis=0), c='k', alpha=.2)
        # plt.plot(list(range(0, len(np.mean(p_shuffle_prior, axis=0)))), np.mean(p_shuffle_prior, axis=0), c=[.6, 0, .6],
        #          alpha=.2)
        # plt.plot(list(range(0, len(np.mean(p_shuffle_beta, axis=0)))), np.mean(p_shuffle_beta, axis=0), c='b', alpha=.2)
        # plt.subplot(2, 2, 2)
        # plt.plot(list(range(0, len(np.mean(p_shuffle_prior, axis=0)))),
        #          np.mean(p_norm, axis=0)/np.mean(p_shuffle_prior, axis=0), c=[.6, 0, .6], alpha=.2)
        # plt.plot(list(range(0, len(np.mean(p_shuffle_beta, axis=0)))),
        #          np.mean(p_norm, axis=0)/np.mean(p_shuffle_beta, axis=0), c='b', alpha=.2)
        p_norm_total[mouse, :] = np.mean(p_norm, axis=0)
        p_shuffle_prior_total[mouse, :] = np.mean(p_shuffle_prior, axis=0)
        p_shuffle_beta_total[mouse, :] = np.mean(p_shuffle_beta, axis=0)
        p_shuffle_prior_fold_total[mouse, :] = np.mean(p_norm, axis=0) / np.mean(p_shuffle_prior, axis=0)
        p_shuffle_beta_fold_total[mouse, :] = np.mean(p_norm, axis=0) / np.mean(p_shuffle_beta, axis=0)

    # anova_results = []
    # for i in range(0, len(p_norm_total[0])):
    #     anova_results.append(stats.ttest_ind(p_norm_total[:, i], p_shuffle_prior_total[:, i])[1])
    # for i in range(0, len(p_norm_total[0])):
    #     anova_results.append(stats.ttest_ind(p_norm_total[:, i], p_shuffle_beta_total[:, i])[1])
    # anova_results_corrected = multipletests(anova_results, alpha=.05, method='holm')
    # return anova_results_corrected[0]

    plt.subplot(2, 2, 1)
    mean_norm = p_norm_total.mean(axis=0)
    sem_plus = mean_norm + stats.sem(p_norm_total, axis=0)
    sem_minus = mean_norm - stats.sem(p_norm_total, axis=0)
    plt.plot(list(range(0, len(mean_norm))), mean_norm, c='k', linewidth=2)
    plt.fill_between(list(range(0, len(mean_norm))), sem_plus, sem_minus, alpha=0.2, color='k', lw=0)
    mean_prior = p_shuffle_prior_total.mean(axis=0)
    sem_plus = mean_prior + stats.sem(p_shuffle_prior_total, axis=0)
    sem_minus = mean_prior - stats.sem(p_shuffle_prior_total, axis=0)
    plt.plot(list(range(0, len(mean_prior))), mean_prior, c=[.6, 0, .6], linewidth=2)
    plt.fill_between(list(range(0, len(mean_prior))), sem_plus, sem_minus, alpha=0.2, color=[.6, 0, .6], lw=0)
    mean_beta = p_shuffle_beta_total.mean(axis=0)
    sem_plus = mean_beta + stats.sem(p_shuffle_beta_total, axis=0)
    sem_minus = mean_beta - stats.sem(p_shuffle_beta_total, axis=0)
    plt.plot(list(range(0, len(mean_beta))), mean_beta, c='b', linewidth=2)
    plt.fill_between(list(range(0, len(mean_beta))), sem_plus, sem_minus, alpha=0.2, color='b', lw=0)
    plt.ylim(0, .3)
    plt.ylabel('Density')
    plt.xlabel('Reactivation probability')
    plt.xticks([0, int(len(mean_norm)/4), int(len(mean_norm)/2), int(len(mean_norm)*3/4), len(mean_norm)],
               ['0', '.25', '.5', '.75', '1'])
    label_1 = mlines.Line2D([], [], color='k', linewidth=2, label='Control')
    label_2 = mlines.Line2D([], [], color=[.6, 0, .6], linewidth=2, label='Temporal prior shuffle')
    label_3 = mlines.Line2D([], [], color='b', linewidth=2, label='Cell identity shuffle')
    plt.legend(handles=[label_1, label_2, label_3], frameon=False)
    plt.subplot(2, 2, 2)
    mean_prior = p_shuffle_prior_fold_total.mean(axis=0)
    sem_plus = mean_prior + stats.sem(p_shuffle_prior_fold_total, axis=0)
    sem_minus = mean_prior - stats.sem(p_shuffle_prior_fold_total, axis=0)
    plt.plot(list(range(0, len(mean_prior))), mean_prior, c=[.6, 0, .6], linewidth=2)
    plt.fill_between(list(range(0, len(mean_prior))), sem_plus, sem_minus, alpha=0.2, color=[.6, 0, .6], lw=0)
    mean_beta = p_shuffle_beta_fold_total.mean(axis=0)
    sem_plus = mean_beta + stats.sem(p_shuffle_beta_fold_total, axis=0)
    sem_minus = mean_beta - stats.sem(p_shuffle_beta_fold_total, axis=0)
    plt.plot(list(range(0, len(mean_beta))), mean_beta, c='b', linewidth=2)
    plt.fill_between(list(range(0, len(mean_beta))), sem_plus, sem_minus, alpha=0.2, color='b', lw=0)
    plt.ylabel('Fold change vs. shuffle')
    plt.xlabel('Reactivation probability')
    plt.xticks([0, int(len(mean_norm) / 4), int(len(mean_norm) / 2), int(len(mean_norm) * 3 / 4), len(mean_norm)],
               ['0', '.25', '.5', '.75', '1'])
    plt.yticks([0, 3, 6, 9])
    label_2 = mlines.Line2D([], [], color=[.6, 0, .6], linewidth=2, label='Fold change vs.\ntemporal prior shuffle')
    label_3 = mlines.Line2D([], [], color='b', linewidth=2, label='Fold change vs.\ncell identity shuffle')
    plt.legend(handles=[label_2, label_3], frameon=False)
    plt.axvline(x=int(len(mean_norm) * 3 / 4), color='black', linestyle='--', linewidth=1, snap=False)
    plt.axhline(y=3, color='black', linestyle='--', linewidth=1, snap=False)
    sns.despine()
    plt.savefig(paths['base_path'] + '/NN/plots/reactivation_shuffle.png', bbox_inches='tight', dpi=200)
    plt.savefig(paths['base_path'] + '/NN/plots/reactivation_shuffle.pdf', bbox_inches='tight', dpi=200, transparent=True)

    plt.close()


def reactivation_rate_day(mice, sample_dates):
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(9, 6))
    paths = preprocess.create_folders(mice[0], sample_dates[0])
    session_data = preprocess.load_data(paths)
    task_runs = int(session_data['task_runs'])
    dark_runs = int(session_data['dark_runs'])
    binned_vec = list(np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/y_pred_binned.npy'))
    x_label = np.zeros(len(binned_vec[0]))
    hours_per_run = 64000/31.25/60/60
    for i in range(0, dark_runs):
        x_label[dark_runs - i - 1] = - hours_per_run / 2
    for i in range(dark_runs, dark_runs + (task_runs * 2)):
        x_label[i] = hours_per_run / 4 + (hours_per_run / 2 * (i - dark_runs))
    plt.subplot(2, 2, 1)
    mean_reactivation_mice = np.zeros((len(mice), len(binned_vec[0])))
    mean_reactivation_mice_heatmap = []
    days = []
    m_colors = ['b', 'purple', 'darkorange', 'green']
    for mouse in range(0, len(mice)):
        paths = preprocess.create_folders(mice[mouse], sample_dates[mouse])
        binned_vec = list(np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/y_pred_binned.npy'))
        mean_reactivation = np.zeros((len(binned_vec), len(binned_vec[0])))
        for i in range(0, len(binned_vec)):
            mean_reactivation[i, :] = binned_vec[i]
            mean_reactivation_mice_heatmap.append(binned_vec[i])
            # plt.plot(x_label, binned_vec[i], '-', c='k', ms=0, alpha=.04, linewidth=1.5)
        mean = mean_reactivation.mean(axis=0)
        plt.plot(x_label, mean, '-', c=m_colors[mouse], ms=0, alpha=.2, linewidth=2)
        mean_reactivation_mice[mouse, :] = mean
        days.append(len(binned_vec))

    slope, intercept, r_value, p_value, std_err = stats.linregress(np.concatenate([x_label[1:9], x_label[1:9], x_label[1:9], x_label[1:9]]), np.concatenate(mean_reactivation_mice[:, 1:9]))
    print(r_value)
    anova_results = []
    anova_results.append(stats.ttest_rel(mean_reactivation_mice[:, 1], mean_reactivation_mice[:, 0])[1])
    anova_results.append(p_value)
    anova_results_corrected = multipletests(anova_results, alpha=.05, method='holm')
    return [anova_results, anova_results_corrected[1]]

    mean = mean_reactivation_mice.mean(axis=0)
    sem_plus = mean + stats.sem(mean_reactivation_mice, axis=0)
    sem_minus = mean - stats.sem(mean_reactivation_mice, axis=0)
    plt.plot(x_label, mean, '-o', c='k', linewidth=3, ms=0)
    plt.fill_between(x_label, sem_plus, sem_minus, alpha=0.2, color='k', lw=0)
    plt.axvspan(-hours_per_run, 0, alpha=.1, color='gray', lw=0)
    plt.ylabel('Reactivation rate (probablity $\mathregular{s^{-1}}$)')
    plt.xlabel('Time relative to first stimulus onset (h)')
    plt.ylim(0, .15)
    plt.xlim(-hours_per_run, x_label[len(x_label)-1] + hours_per_run / 4)
    plt.xticks([-.5, 0, .5, 1, 1.5, 2])
    label_1 = mlines.Line2D([], [], color='b', linestyle='-', label='Mouse 1', alpha=.2, linewidth=2)
    label_2 = mlines.Line2D([], [], color='purple', linestyle='-', label='Mouse 2', alpha=.2, linewidth=2)
    label_3 = mlines.Line2D([], [], color='darkorange', linestyle='-', label='Mouse 3', alpha=.2, linewidth=2)
    label_4 = mlines.Line2D([], [], color='green', linestyle='-', label='Mouse 4', alpha=.2, linewidth=2)
    label_5 = mlines.Line2D([], [], color='k', linestyle='-', label='Mean', linewidth=3)
    plt.legend(handles=[label_1, label_2, label_3, label_4, label_5], frameon=False, prop={'size': 8},
               labelspacing=.1)
    sns.despine()
    plt.savefig(paths['base_path'] + '/NN/plots/reactivation_rate_day.png', bbox_inches='tight', dpi=200, transparent=True)
    plt.savefig(paths['base_path'] + '/NN/plots/reactivation_rate_day.pdf', bbox_inches='tight', dpi=200, transparent=True)
    plt.close()

    sns.set(font_scale=.7)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(4, 6))
    plt.subplot(2, 2, 1)
    sns.heatmap(mean_reactivation_mice_heatmap, vmin=0, vmax=.15, cmap="Reds", cbar=0)
    y_idx = 0
    for mouse in range(0, len(mice)):
        plt.plot([-.4, -.4], [y_idx, y_idx+days[mouse]], color=m_colors[mouse], linewidth=7, snap=False,
                 solid_capstyle='butt')
        y_idx += days[mouse]
    plt.ylim(len(mean_reactivation_mice_heatmap)+3, -3)
    plt.xlim(-.4, 9)
    plt.axis('off')
    plt.savefig(paths['base_path'] + '/NN/plots/reactivation_rate_day_heatmap.png', bbox_inches='tight', dpi=500,
                transparent=True)


def reactivation_bias_day(mice, sample_dates):
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(9, 6))
    paths = preprocess.create_folders(mice[0], sample_dates[0])
    session_data = preprocess.load_data(paths)
    task_runs = int(session_data['task_runs'])
    x_label = np.zeros((task_runs * 2))
    hours_per_run = 64000 / 31.25 / 60 / 60
    for i in range(0, (task_runs * 2)):
        x_label[i] = hours_per_run / 4 + (hours_per_run / 2 * i)
    m_colors = ['b', 'purple', 'darkorange', 'green']
    plt.subplot(2, 2, 1)
    paths = preprocess.create_folders(mice[0], sample_dates[0])
    bias_across_days = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/y_pred_bias_binned.npy',
                               allow_pickle=True)
    bias_all = np.zeros((len(mice), len(bias_across_days[0][0])))
    mean_reactivation_mice_heatmap = []
    days = []
    for mouse in range(0, len(mice)):
        paths = preprocess.create_folders(mice[mouse], sample_dates[mouse])
        bias_across_days = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/y_pred_bias_binned.npy',
                                   allow_pickle=True)
        bias = np.zeros((len(bias_across_days[0]), len(bias_across_days[0][0])))
        for i in range(0, len(bias_across_days[0])):
            bias[i, :] = bias_across_days[0][i]
            mean_reactivation_mice_heatmap.append(bias_across_days[0][i])
        bias_all[mouse, :] = bias.mean(axis=0)
        plt.plot(x_label, bias.mean(axis=0), c=m_colors[mouse], alpha=.2, linewidth=2)
        days.append(len(bias_across_days[0]))

    slope, intercept, r_value, p_value, std_err = stats.linregress(
        np.concatenate([x_label, x_label, x_label, x_label]),
        np.concatenate(bias_all))
    print(r_value)
    print(p_value)

    mean = bias_all.mean(axis=0)
    sem_plus = mean + stats.sem(bias_all, axis=0)
    sem_minus = mean - stats.sem(bias_all, axis=0)
    plt.plot(x_label, mean[0:len(mean)], '-k', linewidth=3)
    plt.fill_between(x_label, sem_plus[0:len(sem_plus)], sem_minus[0:len(sem_minus)], alpha=0.2, color='k', lw=0)
    plt.ylabel('Bias in reactivation rate\n toward the previous stimulus')
    plt.xlabel('Time relative to stimulus onset (h)')
    plt.xlim(0, x_label[len(x_label)-1] + hours_per_run / 4)
    plt.ylim(-1, 1)
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1, snap=False)
    label_1 = mlines.Line2D([], [], color='b', linestyle='-', label='Mouse 1', alpha=.2, linewidth=2)
    label_2 = mlines.Line2D([], [], color='purple', linestyle='-', label='Mouse 2', alpha=.2, linewidth=2)
    label_3 = mlines.Line2D([], [], color='darkorange', linestyle='-', label='Mouse 3', alpha=.2, linewidth=2)
    label_4 = mlines.Line2D([], [], color='green', linestyle='-', label='Mouse 4', alpha=.2, linewidth=2)
    label_5 = mlines.Line2D([], [], color='k', linestyle='-', label='Mean', linewidth=3)
    plt.legend(handles=[label_1, label_2, label_3, label_4, label_5], frameon=False, prop={'size': 8},
               labelspacing=.1)
    sns.despine()
    plt.savefig(paths['base_path'] + '/NN/plots/reactivation_bias_day.png', bbox_inches='tight', dpi=200)
    plt.savefig(paths['base_path'] + '/NN/plots/reactivation_bias_day.pdf', bbox_inches='tight', dpi=200, transparent=True)
    plt.close()

    sns.set(font_scale=.7)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(4, 6))
    plt.subplot(2, 2, 1)
    sns.heatmap(mean_reactivation_mice_heatmap, vmin=-.5, vmax=.5, cmap="coolwarm", cbar=0)
    y_idx = 0
    for mouse in range(0, len(mice)):
        plt.plot([-.4, -.4], [y_idx, y_idx + days[mouse]], color=m_colors[mouse], linewidth=7, snap=False,
                 solid_capstyle='butt')
        y_idx += days[mouse]
    plt.ylim(len(mean_reactivation_mice_heatmap) + 3, -3)
    plt.xlim(-.4, 9)
    plt.axis('off')
    plt.savefig(paths['base_path'] + '/NN/plots/reactivation_bias_day_heatmap.png', bbox_inches='tight', dpi=500,
                transparent=True)


def reactivation_rate_trial(mice, sample_dates):
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(9, 6))
    plt.subplot(2, 2, 1)
    m_colors = ['b', 'purple', 'darkorange', 'green']
    mean_norm_mice = []
    mean_reactivation_mice_heatmap = []
    days = []
    for mouse in range(0, len(mice)):
        paths = preprocess.create_folders(mice[mouse], sample_dates[mouse])
        rate_within_trial_all = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/rate_within_trial.npy',
                                        allow_pickle=True)
        session_data = preprocess.load_data(paths)
        fr = session_data['framerate']
        mean_norm = []
        for i in range(0, len(rate_within_trial_all[0])):
            mean_norm.append(rate_within_trial_all[0][i])
            x = rate_within_trial_all[1][i]
            mean_reactivation_mice_heatmap.append(rate_within_trial_all[0][i])
        mean_norm = np.nanmean(mean_norm, axis=0)
        mean_norm_mice.append(mean_norm)
        plt.plot(x, mean_norm, '-', c=m_colors[mouse], ms=0, alpha=.2, linewidth=2)
        days.append(len(rate_within_trial_all[0]))

    slope, intercept, r_value, p_value, std_err = stats.linregress(
        np.concatenate([x, x, x, x]),
        np.concatenate(mean_norm_mice))
    print(r_value)
    print(p_value)

    mean_norm_mice_all = np.nanmean(mean_norm_mice, axis=0)
    y0_err = stats.sem(mean_norm_mice, axis=0, nan_policy='omit')
    sem_plus_norm = mean_norm_mice_all + y0_err
    sem_minus_norm = mean_norm_mice_all - y0_err

    plt.plot(x, mean_norm_mice_all, '-', c='k', linewidth=3)
    plt.fill_between(x, sem_plus_norm, sem_minus_norm, alpha=0.2, color='k', lw=0)

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
    label_1 = mlines.Line2D([], [], color='b', linestyle='-', label='Mouse 1', alpha=.2, linewidth=2)
    label_2 = mlines.Line2D([], [], color='purple', linestyle='-', label='Mouse 2', alpha=.2, linewidth=2)
    label_3 = mlines.Line2D([], [], color='darkorange', linestyle='-', label='Mouse 3', alpha=.2, linewidth=2)
    label_4 = mlines.Line2D([], [], color='green', linestyle='-', label='Mouse 4', alpha=.2, linewidth=2)
    label_5 = mlines.Line2D([], [], color='k', linestyle='-', label='Mean', linewidth=3)
    plt.legend(handles=[label_1, label_2, label_3, label_4, label_5], frameon=False, prop={'size': 8},
               labelspacing=.1)
    sns.despine()
    plt.savefig(paths['base_path'] + '/NN/plots/reactivation_rate_trial.png', bbox_inches='tight', dpi=200)
    plt.savefig(paths['base_path'] + '/NN/plots/reactivation_rate_trial.pdf', bbox_inches='tight', dpi=200, transparent=True)
    plt.close()

    sns.set(font_scale=.7)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(4, 6))
    plt.subplot(2, 2, 1)
    sns.heatmap(mean_reactivation_mice_heatmap, vmin=.03, vmax=.09, cmap="Reds", cbar=0)
    y_idx = 0
    for mouse in range(0, len(mice)):
        plt.plot([-.4, -.4], [y_idx, y_idx + days[mouse]], color=m_colors[mouse], linewidth=7, snap=False,
                 solid_capstyle='butt')
        y_idx += days[mouse]
    plt.ylim(len(mean_reactivation_mice_heatmap) + 3, -3)
    plt.xlim(-.4, 9)
    plt.axis('off')
    plt.savefig(paths['base_path'] + '/NN/plots/reactivation_rate_trial_heatmap.png', bbox_inches='tight', dpi=500,
                transparent=True)


def reactivation_bias_trial(mice, sample_dates):
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(9, 6))
    plt.subplot(2, 2, 1)
    m_colors = ['b', 'purple', 'darkorange', 'green']
    mean_norm_mice = []
    mean_reactivation_mice_heatmap = []
    days = []
    for mouse in range(0, len(mice)):
        paths = preprocess.create_folders(mice[mouse], sample_dates[mouse])
        bias_within_trial_all = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/bias_within_trial.npy',
                                        allow_pickle=True)
        session_data = preprocess.load_data(paths)
        fr = session_data['framerate']
        mean_norm = []
        for i in range(0, len(bias_within_trial_all[0])):
            mean_norm.append(bias_within_trial_all[0][i])
            x = bias_within_trial_all[1][i]
            mean_reactivation_mice_heatmap.append(bias_within_trial_all[0][i])
        mean_norm = np.nanmean(mean_norm, axis=0)
        mean_norm_mice.append(mean_norm)
        plt.plot(x, mean_norm, '-', c=m_colors[mouse], ms=0, alpha=.2, linewidth=2)
        days.append(len(bias_within_trial_all[0]))

    slope, intercept, r_value, p_value, std_err = stats.linregress(
        np.concatenate([x, x, x, x]),
        np.concatenate(mean_norm_mice))
    print(r_value)
    print(p_value)

    mean_norm_mice_all = np.nanmean(mean_norm_mice, axis=0)
    y0_err = stats.sem(mean_norm_mice, axis=0, nan_policy='omit')
    sem_plus_norm = mean_norm_mice_all + y0_err
    sem_minus_norm = mean_norm_mice_all - y0_err

    plt.plot(x, mean_norm_mice_all, '-', c='k', linewidth=3, ms=0)
    plt.fill_between(x, sem_plus_norm, sem_minus_norm, alpha=0.2, color='k', lw=0)

    # plt.axvspan(0, int(fr * 1), alpha=.75, color='mediumseagreen', lw=0)
    # plt.axvspan(int(fr * 1), int(fr * 2), alpha=.75, color='salmon', lw=0)
    plt.axvspan(0, int(fr * 2), alpha=1, color='k', lw=0)
    plt.axvspan(int(fr * 2), int(fr * 8), alpha=.1, color='gray', lw=0)
    plt.ylabel('Bias in reactivation rate\n toward the previous stimulus')
    plt.xlabel('Time relative to stimulus onset (s)')
    plt.xticks([int(fr * 0), int(fr * 10), int(fr * 20), int(fr * 30), int(fr * 40), int(fr * 50), int(fr * 60)],
               ['0', '10', '20', '30', '40', '50', '60'])
    plt.xlim((0, int(fr * 60)))
    plt.ylim(-1, 1)
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1, snap=False)
    label_1 = mlines.Line2D([], [], color='b', linestyle='-', label='Mouse 1', alpha=.2, linewidth=2)
    label_2 = mlines.Line2D([], [], color='purple', linestyle='-', label='Mouse 2', alpha=.2, linewidth=2)
    label_3 = mlines.Line2D([], [], color='darkorange', linestyle='-', label='Mouse 3', alpha=.2, linewidth=2)
    label_4 = mlines.Line2D([], [], color='green', linestyle='-', label='Mouse 4', alpha=.2, linewidth=2)
    label_5 = mlines.Line2D([], [], color='k', linestyle='-', label='Mean', linewidth=3)
    plt.legend(handles=[label_1, label_2, label_3, label_4, label_5], frameon=False, prop={'size': 8},
               labelspacing=.1)
    sns.despine()
    plt.savefig(paths['base_path'] + '/NN/plots/reactivation_bias_trial.png', bbox_inches='tight', dpi=200)
    plt.savefig(paths['base_path'] + '/NN/plots/reactivation_bias_trial.pdf', bbox_inches='tight', dpi=200, Transparent=True)
    plt.close()

    sns.set(font_scale=.7)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(4, 6))
    plt.subplot(2, 2, 1)
    sns.heatmap(mean_reactivation_mice_heatmap, vmin=-.5, vmax=.5, cmap="coolwarm", cbar=0)
    y_idx = 0
    for mouse in range(0, len(mice)):
        plt.plot([-.4, -.4], [y_idx, y_idx + days[mouse]], color=m_colors[mouse], linewidth=7, snap=False,
                 solid_capstyle='butt')
        y_idx += days[mouse]
    plt.ylim(len(mean_reactivation_mice_heatmap) + 3, -3)
    plt.xlim(-.4, 9)
    plt.axis('off')
    plt.savefig(paths['base_path'] + '/NN/plots/reactivation_bias_trial_heatmap.png', bbox_inches='tight', dpi=500,
                transparent=True)


def iti_activity_across_trials(mice, sample_dates):
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(9, 6))
    m_colors = ['b', 'purple', 'darkorange', 'green']
    paths = preprocess.create_folders(mice[0], sample_dates[0])
    session_data = preprocess.load_data(paths)
    fr = session_data['framerate']
    task_runs = int(session_data['task_runs'])
    dark_runs = int(session_data['dark_runs'])
    binned_vec = list(np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/activity_binned.npy'))
    x_label = np.zeros(len(binned_vec[0]))
    hours_per_run = 64000 / 31.25 / 60 / 60
    for i in range(0, dark_runs):
        x_label[dark_runs - i - 1] = - hours_per_run / 2
    for i in range(dark_runs, dark_runs + (task_runs * 2)):
        x_label[i] = hours_per_run / 4 + (hours_per_run / 2 * (i - dark_runs))
    plt.subplot(2, 2, 1)
    mean_iti_mice = np.zeros((len(mice), len(binned_vec[0])))
    for mouse in range(0, len(mice)):
        paths = preprocess.create_folders(mice[mouse], sample_dates[mouse])
        binned_vec = list(np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/activity_binned.npy'))
        mean_iti = np.zeros((len(binned_vec), len(binned_vec[0])))
        for i in range(0, len(binned_vec)):
            mean_iti[i, :] = binned_vec[i] * fr[0][0]
        mean = mean_iti.mean(axis=0)
        plt.plot(x_label, mean, '-', c=m_colors[mouse], alpha=.2, linewidth=2)
        mean_iti_mice[mouse, :] = mean
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        np.concatenate([x_label[1:9], x_label[1:9], x_label[1:9], x_label[1:9]]),
        np.concatenate(mean_iti_mice[:, 1:9]))
    print(r_value)
    anova_results = []
    anova_results.append(stats.ttest_rel(mean_iti_mice[:, 1], mean_iti_mice[:, 0])[1])
    anova_results.append(p_value)
    anova_results_corrected = multipletests(anova_results, alpha=.05, method='holm')
    return [anova_results, anova_results_corrected[1]]

    mean = mean_iti_mice.mean(axis=0)
    sem_plus = mean + stats.sem(mean_iti_mice, axis=0)
    sem_minus = mean - stats.sem(mean_iti_mice, axis=0)
    plt.plot(x_label, mean, '-', c='k', linewidth=3)
    plt.fill_between(x_label, sem_plus, sem_minus, alpha=0.2, color='k', lw=0)
    plt.axvspan(-hours_per_run, 0, alpha=.1, color='gray', lw=0)
    plt.ylabel('Inter-trial-interval activity\n(Normalized $\mathregular{Ca^{2+}}$transient $\mathregular{s^{-1}}$)')
    plt.xlim(-hours_per_run, x_label[len(x_label)-1] + hours_per_run / 4)
    plt.xlabel('Time relative to stimulus onset (h)')
    plt.ylim(0, .3)
    plt.yticks([0, .1, .2, .3])
    sns.despine()
    plt.savefig(paths['base_path'] + '/NN/plots/iti_activity_across_trials.png', bbox_inches='tight', dpi=200)
    plt.savefig(paths['base_path'] + '/NN/plots/iti_activity_across_trials.pdf', bbox_inches='tight', dpi=200, transparent=True)
    plt.close()


def iti_activity_within_trial(mice, sample_dates):
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(9, 6))
    plt.subplot(2, 2, 1)
    m_colors = ['b', 'purple', 'darkorange', 'green']
    mean_activity_mice = []
    for mouse in range(0, len(mice)):
        paths = preprocess.create_folders(mice[mouse], sample_dates[mouse])
        activity_within_trial_all = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/activity_within_trial.npy',
                                        allow_pickle=True)
        session_data = preprocess.load_data(paths)
        fr = session_data['framerate']
        mean_activity = []
        for i in range(0, len(activity_within_trial_all[0])):
            mean_activity.append(activity_within_trial_all[0][i])
            x = activity_within_trial_all[1][i]
        mean_activity = np.nanmean(mean_activity, axis=0) * fr[0][0]
        mean_activity_mice.append(mean_activity)
        plt.plot(x, mean_activity, '-', c=m_colors[mouse], ms=0, alpha=.2, linewidth=2)

    slope, intercept, r_value, p_value, std_err = stats.linregress(
        np.concatenate([x, x, x, x]),
        np.concatenate(mean_activity_mice))
    print(p_value)
    print(r_value)

    mean_activity_mice_all = np.nanmean(mean_activity_mice, axis=0)
    y0_err = stats.sem(mean_activity_mice, axis=0, nan_policy='omit')
    sem_plus = mean_activity_mice_all + y0_err
    sem_minus = mean_activity_mice_all - y0_err

    plt.plot(x, mean_activity_mice_all, '-', c='k', linewidth=3)
    plt.fill_between(x, sem_plus, sem_minus, alpha=0.2, color='k', lw=0)
    plt.axvspan(0, int(fr * 2), alpha=1, color='k', lw=0)
    plt.axvspan(int(fr * 2), int(fr * 8), alpha=.1, color='gray', lw=0)
    plt.ylabel('Inter-trial-interval activity\n(Normalized $\mathregular{Ca^{2+}}$transient $\mathregular{s^{-1}}$)')
    plt.xlabel('Time relative to stimulus onset (s)')
    plt.xticks([int(fr * 0), int(fr * 10), int(fr * 20), int(fr * 30), int(fr * 40), int(fr * 50), int(fr * 60)],
               ['0', '10', '20', '30', '40', '50', '60'])
    plt.xlim((0, int(fr * 60)))
    plt.ylim(0, .3)
    plt.yticks([0, .1, .2, .3])
    sns.despine()
    plt.savefig(paths['base_path'] + '/NN/plots/iti_activity_within_trial.png', bbox_inches='tight', dpi=200)
    plt.savefig(paths['base_path'] + '/NN/plots/iti_activity_within_trial.pdf', bbox_inches='tight', dpi=200, transparent=True)
    plt.close()


def pupil_across_trials(mice, sample_dates):
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(9, 6))
    m_colors = ['b', 'purple', 'darkorange', 'green']
    paths = preprocess.create_folders(mice[0], sample_dates[0])
    session_data = preprocess.load_data(paths)
    task_runs = int(session_data['task_runs'])
    dark_runs = int(session_data['dark_runs'])
    binned_vec = list(np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/pupil_binned.npy'))
    x_label = np.zeros(len(binned_vec[0]))
    hours_per_run = 64000 / 31.25 / 60 / 60
    for i in range(0, dark_runs):
        x_label[dark_runs - i - 1] = - hours_per_run / 2
    for i in range(dark_runs, dark_runs + (task_runs * 2)):
        x_label[i] = hours_per_run / 4 + (hours_per_run / 2 * (i - dark_runs))
    plt.subplot(2, 2, 1)
    mean_pupil_mice = np.zeros((len(mice), len(binned_vec[0])))
    for mouse in range(0, len(mice)):
        paths = preprocess.create_folders(mice[mouse], sample_dates[mouse])
        binned_vec = list(np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/pupil_binned.npy'))
        mean_pupil = np.zeros((len(binned_vec), len(binned_vec[0])))
        for i in range(0, len(binned_vec)):
            mean_pupil[i, :] = binned_vec[i]
        mean = mean_pupil.mean(axis=0)
        plt.plot(x_label, mean, '-', c=m_colors[mouse], alpha=.2, linewidth=2)
        mean_pupil_mice[mouse, :] = mean

    slope, intercept, r_value, p_value, std_err = stats.linregress(
        np.concatenate([x_label[1:9], x_label[1:9], x_label[1:9], x_label[1:9]]),
        np.concatenate(mean_pupil_mice[:, 1:9]))
    print(r_value)

    anova_results = []
    anova_results.append(stats.ttest_rel(mean_pupil_mice[:, 1], mean_pupil_mice[:, 0])[1])
    anova_results.append(p_value)
    anova_results_corrected = multipletests(anova_results, alpha=.05, method='holm')
    return [anova_results, anova_results_corrected[1]]

    mean = mean_pupil_mice.mean(axis=0)
    sem_plus = mean + stats.sem(mean_pupil_mice, axis=0)
    sem_minus = mean - stats.sem(mean_pupil_mice, axis=0)
    plt.plot(x_label, mean, '-', c='k', linewidth=3)
    plt.fill_between(x_label, sem_plus, sem_minus, alpha=0.2, color='k', lw=0)
    plt.axvspan(-hours_per_run, 0, alpha=.1, color='gray', lw=0)
    plt.ylabel('Max. normalized pupil area')
    plt.xlim(-hours_per_run, x_label[len(x_label)-1] + hours_per_run / 4)
    plt.xlabel('Time relative to stimulus onset (h)')
    plt.ylim(0, 1)
    sns.despine()
    plt.savefig(paths['base_path'] + '/NN/plots/pupil_across_trials.png', bbox_inches='tight', dpi=200)
    plt.savefig(paths['base_path'] + '/NN/plots/pupil_across_trials.pdf', bbox_inches='tight', dpi=200, transparent=True)
    plt.close()


def pupil_within_trial(mice, sample_dates):
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(9, 6))
    plt.subplot(2, 2, 1)
    m_colors = ['b', 'purple', 'darkorange', 'green']
    mean_pupil_mice = []
    for mouse in range(0, len(mice)):
        paths = preprocess.create_folders(mice[mouse], sample_dates[mouse])
        pupil_within_trial_all = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/pupil_within_trial.npy',
                                        allow_pickle=True)
        session_data = preprocess.load_data(paths)
        fr = session_data['framerate']
        mean_pupil = []
        for i in range(0, len(pupil_within_trial_all[0])):
            mean_pupil.append(pupil_within_trial_all[0][i])
            x = pupil_within_trial_all[1][i]
        mean_pupil = np.nanmean(mean_pupil, axis=0)
        mean_pupil_mice.append(mean_pupil)
        plt.plot(x, mean_pupil, '-', c=m_colors[mouse], ms=0, alpha=.2, linewidth=2)

    slope, intercept, r_value, p_value, std_err = stats.linregress(
        np.concatenate([x, x, x, x]),
        np.concatenate(mean_pupil_mice))
    print(p_value)
    print(r_value)

    mean_pupil_mice_all = np.nanmean(mean_pupil_mice, axis=0)
    y0_err = stats.sem(mean_pupil_mice, axis=0, nan_policy='omit')
    sem_plus = mean_pupil_mice_all + y0_err
    sem_minus = mean_pupil_mice_all - y0_err

    plt.plot(x, mean_pupil_mice_all, '-', c='k', linewidth=3)
    plt.fill_between(x, sem_plus, sem_minus, alpha=0.2, color='k', lw=0)
    plt.axvspan(0, int(fr * 2), alpha=1, color='k', lw=0)
    plt.axvspan(int(fr * 2), int(fr * 8), alpha=.1, color='gray', lw=0)
    plt.ylabel('Max. normalized pupil area')
    plt.xlabel('Time relative to stimulus onset (s)')
    plt.xticks([int(fr * 0), int(fr * 10), int(fr * 20), int(fr * 30), int(fr * 40), int(fr * 50), int(fr * 60)],
               ['0', '10', '20', '30', '40', '50', '60'])
    plt.xlim((0, int(fr * 60)))
    plt.ylim(0, 1)
    #plt.yticks([0, .05])
    sns.despine()
    plt.savefig(paths['base_path'] + '/NN/plots/pupil_trial.png', bbox_inches='tight', dpi=200)
    plt.savefig(paths['base_path'] + '/NN/plots/pupil_trial.pdf', bbox_inches='tight', dpi=200, transparent=True)
    plt.close()


def reactivation_physical(mice, sample_dates):
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(9, 7))
    # plt.subplots_adjust(hspace=.3)
    paths = preprocess.create_folders(mice[0], sample_dates[0])
    reactivation_physical_helper(0, mice, sample_dates, paths, 'Pupil area (ΔA/A)', 'blue',
                                 1, -.2, .02)
    reactivation_physical_helper(0, mice, sample_dates, paths, 'Max normalized pupil area', 'blue',
                                 2, 0, 1)
    reactivation_physical_helper(1, mice, sample_dates, paths, 'Max normalized pupil movement', 'red', 3, 0, 1)
    reactivation_physical_helper(2, mice, sample_dates, paths, 'Brain motion (μm, abs)',
                                 'darkgoldenrod', 4, 0, 1)
    plt.savefig(paths['base_path'] + '/NN/plots/reactivation_physical.png', bbox_inches='tight', dpi=200)
    plt.savefig(paths['base_path'] + '/NN/plots/reactivation_physical.pdf', bbox_inches='tight', dpi=500, transparent=True)
    plt.close()


def reactivation_physical_helper(vec_idx, mice, sample_dates, paths, y_label, c, idx, lim1, lim2):
    plt.subplot(2, 2, idx)
    vec = list(np.load(paths['base_path'] + paths['mouse'] +
                       '/data_across_days/reactivation_physical.npy', allow_pickle=True))
    mean_mice = np.zeros((len(mice), len(vec[vec_idx][0])))
    for mouse in range(0, len(mice)):
        paths = preprocess.create_folders(mice[mouse], sample_dates[mouse])
        vec = list(np.load(paths['base_path'] +
                           paths['mouse'] + '/data_across_days/reactivation_physical.npy', allow_pickle=True))
        vec = vec[vec_idx]
        session_data = preprocess.load_data(paths)
        framerate = session_data['framerate']
        mean_vec = np.zeros((len(vec), len(vec[0])))
        for i in range(0, len(vec)):
            if y_label == 'Pupil area (ΔA/A)':
                mean_vec[i, :] = (vec[i] - np.mean(vec[i][0:5])) / np.mean(vec[i][0:5])
            if y_label == 'Brain motion (μm, abs)':
                mean_vec[i, :] = vec[i] * 1500/512
            else:
                mean_vec[i, :] = vec[i]
        mean = mean_vec.mean(axis=0)
        mean_mice[mouse, :] = mean
    print(stats.ttest_rel(mean_mice[:, 0], mean_mice[:, int(framerate * 20)]))
    mean = mean_mice.mean(axis=0)
    sem_plus = mean + stats.sem(mean_mice, axis=0)
    sem_minus = mean - stats.sem(mean_mice, axis=0)
    plt.plot(mean, c=c, linewidth=2, ms=7)
    plt.fill_between(np.arange(mean.shape[0]), sem_plus, sem_minus, alpha=0.2, color=c, lw=0)
    plt.xticks([int(framerate * 0), int(framerate * 10), int(framerate * 20),
                int(framerate * 30), int(framerate * 40)], ['-20', '-10', '0', '10', '20'])
    plt.ylabel(y_label)
    plt.ylim(lim1, lim2)
    plt.axvline(x=int(framerate * 20), color='black', linestyle='--', linewidth=1, snap=False)
    plt.xlabel('Time relative to reactivation onset (s)')
    sns.despine()


def trial_history(num_prev, mice, sample_dates):
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(9, 6))
    m_colors = ['b', 'purple', 'darkorange', 'green']
    x = [1, 2]
    rate_same_all = []
    rate_diff_all = []
    bias_same_all = []
    bias_diff_all = []
    for mouse in range(0, len(mice)):
        paths = preprocess.create_folders(mice[mouse], sample_dates[mouse])
        trial_hist = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/trial_history_' +
                             str(num_prev) + '.npy', allow_pickle=True)
        rate_same = np.zeros(len(trial_hist[0]))
        rate_diff = np.zeros(len(trial_hist[0]))
        bias_same = np.zeros(len(trial_hist[0]))
        bias_diff = np.zeros(len(trial_hist[0]))
        for i in range(0, len(trial_hist[0])):
            rate_same[i] = trial_hist[0][i]
            rate_diff[i] = trial_hist[1][i]
            bias_same[i] = trial_hist[6][i]
            bias_diff[i] = trial_hist[7][i]
        rate_same_all.append(np.mean(rate_same))
        rate_diff_all.append(np.mean(rate_diff))
        bias_same_all.append(np.mean(bias_same))
        bias_diff_all.append(np.mean(bias_diff))
        plt.subplot(2, 2, 1)
        plt.plot(x, [rate_same_all[mouse], rate_diff_all[mouse]], '-', c=m_colors[mouse], alpha=.2, linewidth=2)
        plt.subplot(2, 2, 2)
        plt.plot(x, [bias_same_all[mouse], bias_diff_all[mouse]], '-', c=m_colors[mouse], alpha=.2, linewidth=2)

    print(stats.ttest_rel(rate_same_all, rate_diff_all))

    plt.subplot(2, 2, 1)
    y0_err = stats.sem(rate_same_all)
    rate_same_all = np.mean(rate_same_all)
    y1_err = stats.sem(rate_diff_all)
    rate_diff_all = np.mean(rate_diff_all)
    plt.plot(x, [rate_same_all, rate_diff_all], '-', c='k', linewidth=3)
    plt.fill_between(x, [rate_same_all + y0_err, rate_diff_all + y1_err],
                     [rate_same_all - y0_err, rate_diff_all - y1_err], alpha=.2, color='k', lw=0)
    plt.xlim(.5, 2.5)
    plt.ylim(0, .08)
    plt.xticks([1, 2], ['Same stimulus\nas previous', 'Different stimulus\nthan previous'])
    plt.ylabel('Reactivation rate (probability $\mathregular{s^{-1}}$)')
    plt.subplot(2, 2, 2)
    y0_err = stats.sem(bias_same_all)
    bias_same_all = np.mean(bias_same_all)
    y1_err = stats.sem(bias_diff_all)
    bias_diff_all = np.mean(bias_diff_all)
    plt.plot(x, [bias_same_all, bias_diff_all], '-', c='k', linewidth=3)
    plt.fill_between(x, [bias_same_all + y0_err, bias_diff_all + y1_err],
                     [bias_same_all - y0_err, bias_diff_all - y1_err], alpha=.2, color='k', lw=0)
    plt.xlim(.5, 2.5)
    plt.ylim(-1, 1)
    plt.xticks([1, 2], ['Same stimulus \n as previous', 'Different stimulus \n than previous'])
    plt.ylabel('Bias in reactivation rate\n toward the previous stimulus')
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1, snap=False)
    sns.despine()
    plt.savefig(paths['base_path'] + '/NN/plots/trial_history.png', bbox_inches='tight', dpi=200)
    plt.savefig(paths['base_path'] + '/NN/plots/trial_history.pdf', bbox_inches='tight', dpi=200, transparent=True)
    plt.close()


def reactivation_duration(mice, sample_dates):
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(9, 6))
    m_colors = ['b', 'purple', 'darkorange', 'green']

    plt.subplot(2, 2, 1)
    length_control_all = []
    length_cue_all = []
    x = [1, 2]
    for mouse in range(0, len(mice)):
        paths = preprocess.create_folders(mice[mouse], sample_dates[mouse])
        length_all = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/reactivation_length.npy',
                             allow_pickle=True)
        session_data = preprocess.load_data(paths)
        fr = session_data['framerate']
        length_control = []
        length_cue = []
        for i in range(0, len(length_all[0])):
            control = length_all[1][i] / fr[0]
            cue = length_all[0][i] / fr[0]
            length_control.append(control)
            length_cue.append(cue)
        length_control = np.mean(length_control)
        length_cue = np.mean(length_cue)
        plt.plot(x, [length_control, length_cue], c=m_colors[mouse], linewidth=2, linestyle='-', alpha=.2)
        length_control_all.append(length_control)
        length_cue_all.append(length_cue)

    print(stats.ttest_rel(length_control_all, length_cue_all))

    y0 = np.mean(length_control_all)
    y1 = np.mean(length_cue_all)
    y0_err = stats.sem(length_control_all)
    y1_err = stats.sem(length_cue_all)
    plt.plot(x, [y0, y1], '-', c='k', linewidth=3)
    plt.fill_between(x, [y0 + y0_err, y1 + y1_err], [y0 - y0_err, y1 - y1_err], alpha=0.2, color='k', lw=0)
    plt.xlim((.5, 2.5))
    plt.ylim((.25, .4))
    plt.xticks([1, 2], ['Baseline\nperiod', 'Stimulus\npresentations'])
    plt.yticks([.2, .25, .3, .35, .4])
    plt.gca().get_xticklabels()[1].set_color('k')
    plt.gca().get_xticklabels()[0].set_color('k')
    plt.ylabel('Reactivation duration (s)')
    sns.despine()
    plt.savefig(paths['base_path'] + '/NN/plots/reactivation_duration.png', bbox_inches='tight', dpi=200)
    plt.savefig(paths['base_path'] + '/NN/plots/reactivation_duration.pdf', bbox_inches='tight', dpi=200, transparent=True)
    plt.close()


def reactivation_top_bottom_activity(mice, sample_dates):
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(18, 3))
    plt.subplots_adjust(wspace=.3)
    baseline = np.zeros((len(mice), 5))
    idxs = [0, 13]
    colors = ['mediumseagreen', 'salmon']
    cs_1 = np.zeros((len(mice), 5))
    cs_2 = np.zeros((len(mice), 5))
    for cue_type in range(0, 2):
        c = colors[cue_type]
        idx = idxs[cue_type]
        top_cs_1_r_all = []
        top_cs_2_r_all = []
        top_cs_1_c_all = []
        top_cs_2_c_all = []
        bottom_cs_1_r_all = []
        bottom_cs_2_r_all = []
        bottom_cs_1_c_all = []
        bottom_cs_2_c_all = []
        other_r_all = []
        other_c_all = []
        top_b_all = []
        bottom_b_all = []
        other_b_all = []
        x = [1, 2, 3, 4, 5]
        for mouse in range(0, len(mice)):
            paths = preprocess.create_folders(mice[mouse], sample_dates[mouse])
            activity_all = np.load(paths['base_path'] + paths['mouse'] +
                                 '/data_across_days/reactivation_top_bottom_activity.npy', allow_pickle=True)
            session_data = preprocess.load_data(paths)
            fr = session_data['framerate']
            top_cs_1_r = []
            top_cs_2_r = []
            top_cs_1_c = []
            top_cs_2_c = []
            bottom_cs_1_r = []
            bottom_cs_2_r = []
            bottom_cs_1_c = []
            bottom_cs_2_c = []
            other_r = []
            other_c = []
            top_b = []
            bottom_b = []
            other_b = []
            for i in range(0, len(activity_all[0])):
                top_cs_1_r.append(activity_all[idx][i] * fr)
                top_cs_2_r.append(activity_all[idx+1][i] * fr)
                bottom_cs_1_r.append(activity_all[idx+2][i] * fr)
                bottom_cs_2_r.append(activity_all[idx+3][i] * fr)
                other_r.append(activity_all[idx+4][i] * fr)
                top_cs_1_c.append(activity_all[idx+5][i] * fr)
                top_cs_2_c.append(activity_all[idx+6][i] * fr)
                bottom_cs_1_c.append(activity_all[idx+7][i] * fr)
                bottom_cs_2_c.append(activity_all[idx+8][i] * fr)
                other_c.append(activity_all[idx+9][i] * fr)
                top_b.append(activity_all[idx+10][i] * fr)
                bottom_b.append(activity_all[idx+11][i] * fr)
                other_b.append(activity_all[idx+12][i] * fr)
            top_cs_1_r = np.mean(top_cs_1_r)
            top_cs_2_r = np.mean(top_cs_2_r)
            top_cs_1_c = np.mean(top_cs_1_c)
            top_cs_2_c = np.mean(top_cs_2_c)
            bottom_cs_1_r = np.mean(bottom_cs_1_r)
            bottom_cs_2_r = np.mean(bottom_cs_2_r)
            bottom_cs_1_c = np.mean(bottom_cs_1_c)
            bottom_cs_2_c = np.mean(bottom_cs_2_c)
            other_r = np.mean(other_r)
            other_c = np.mean(other_c)
            top_b = np.mean(top_b)
            bottom_b = np.mean(bottom_b)
            other_b = np.mean(other_b)
            plt.subplot(1, 3, 1)
            plt.errorbar(x, [top_cs_1_c, top_cs_2_c, bottom_cs_1_c, bottom_cs_2_c, other_c], yerr=0, c=c, linewidth=2,
                         linestyle='-', zorder=0, alpha=.2)
            plt.subplot(1, 3, 2)
            plt.errorbar(x, [top_cs_1_r, top_cs_2_r, bottom_cs_1_r, bottom_cs_2_r, other_r], yerr=0, c=c, linewidth=2,
                         linestyle='-', zorder=0, alpha=.2)
            if cue_type == 0:
                baseline[mouse, 0] = top_b
                baseline[mouse, 2] = bottom_b
            if cue_type == 1:
                baseline[mouse, 1] = top_b
                baseline[mouse, 3] = bottom_b
                baseline[mouse, 4] = other_b
            if cue_type == 0:
                cs_1[mouse, :] = [top_cs_1_r, top_cs_2_r, bottom_cs_1_r, bottom_cs_2_r, other_r]
            if cue_type == 1:
                cs_2[mouse, :] = [top_cs_1_r, top_cs_2_r, bottom_cs_1_r, bottom_cs_2_r, other_r]
            top_cs_1_r_all.append(top_cs_1_r)
            top_cs_2_r_all.append(top_cs_2_r)
            top_cs_1_c_all.append(top_cs_1_c)
            top_cs_2_c_all.append(top_cs_2_c)
            bottom_cs_1_r_all.append(bottom_cs_1_r)
            bottom_cs_2_r_all.append(bottom_cs_2_r)
            bottom_cs_1_c_all.append(bottom_cs_1_c)
            bottom_cs_2_c_all.append(bottom_cs_2_c)
            other_r_all.append(other_r)
            other_c_all.append(other_c)
            top_b_all.append(top_b)
            bottom_b_all.append(bottom_b)
            other_b_all.append(other_b)
        y0 = np.mean(top_cs_1_r_all)
        y1 = np.mean(top_cs_2_r_all)
        y2 = np.mean(top_cs_1_c_all)
        y3 = np.mean(top_cs_2_c_all)
        y4 = np.mean(bottom_cs_1_r_all)
        y5 = np.mean(bottom_cs_2_r_all)
        y6 = np.mean(bottom_cs_1_c_all)
        y7 = np.mean(bottom_cs_2_c_all)
        y8 = np.mean(other_r_all)
        y9 = np.mean(other_c_all)
        y10 = np.mean(top_b_all)
        y11 = np.mean(bottom_b_all)
        y12 = np.mean(other_b_all)
        y0_err = stats.sem(top_cs_1_r_all)
        y1_err = stats.sem(top_cs_2_r_all)
        y2_err = stats.sem(top_cs_1_c_all)
        y3_err = stats.sem(top_cs_2_c_all)
        y4_err = stats.sem(bottom_cs_1_r_all)
        y5_err = stats.sem(bottom_cs_2_r_all)
        y6_err = stats.sem(bottom_cs_1_c_all)
        y7_err = stats.sem(bottom_cs_2_c_all)
        y8_err = stats.sem(other_r_all)
        y9_err = stats.sem(other_c_all)
        y10_err = stats.sem(top_b_all)
        y11_err = stats.sem(bottom_b_all)
        y12_err = stats.sem(other_b_all)

        plt.subplot(1, 3, 1)
        plt.plot(x, [y2, y3, y6, y7, y9], c=c, linewidth=3, linestyle='-', zorder=0)
        plt.fill_between(x, [y2+y2_err, y3+y3_err, y6+y6_err, y7+y7_err, y9+y9_err],
                         [y2-y2_err, y3-y3_err, y6-y6_err, y7-y7_err, y9-y9_err], alpha=0.2, color=c, lw=0)
        plt.xlim((.5, 5.5))
        plt.ylabel('Stimulus activity\n(normalized\ndeconvolved$\mathregular{Ca^{2+}}$$\mathregular{s^{-1}}$)')
        plt.xticks([1, 2, 3, 4, 5], ['Top\n5% S1', 'Top\n5% S2', 'Bottom\n95% S1', 'Bottom\n95% S2', 'Other'])
        #plt.yticks([0, .05, .1, .15])
        plt.ylim((0, 1.85))
        label_1 = mlines.Line2D([], [], color='mediumseagreen', linewidth=2, label='Stimulus 1')
        label_2 = mlines.Line2D([], [], color='salmon', linewidth=2, label='Stimulus 2')
        plt.legend(handles=[label_1, label_2], frameon=False)
        plt.subplot(1, 3, 2)
        plt.plot(x, [y0, y1, y4, y5, y8], c=c, linewidth=3, linestyle='-', zorder=0)
        plt.fill_between(x, [y0 + y0_err, y1 + y1_err, y4 + y4_err, y5 + y5_err, y8 + y8_err],
                         [y0 - y0_err, y1 - y1_err, y4 - y4_err, y5 - y5_err, y8 - y8_err], alpha=0.2, color=c, lw=0)
        plt.xlim((.5, 5.5))
        plt.ylabel(
            'Reactivation activity\n(normalized\ndeconvolved $\mathregular{Ca^{2+}}$$\mathregular{s^{-1}}$)')
        plt.xticks([1, 2, 3, 4, 5], ['Top\n5% S1', 'Top\n5% S2', 'Bottom\n95% S1', 'Bottom\n95% S2', 'Other'])
        # plt.yticks([0, .05, .1, .15])
        plt.ylim((0, 1.85))
        label_1 = mlines.Line2D([], [], color='mediumseagreen', linewidth=2, label='Stimulus 1 reactivation')
        label_2 = mlines.Line2D([], [], color='salmon', linewidth=2, label='Stimulus 2 reactivation')
        plt.legend(handles=[label_1, label_2], frameon=False)

    # anova_results = []
    # for i in range(0, len(cs_1[0])):
    #     anova_results.append(stats.ttest_ind(cs_1[:, i], cs_2[:, i])[1])
    # anova_results_corrected = multipletests(anova_results, alpha=.05, method='holm')
    # return [anova_results, anova_results_corrected[1]]

    plt.subplot(1, 3, 3)
    for i in range(0, len(mice)):
        plt.errorbar(x, baseline[i], yerr=0, c='k', linewidth=2, linestyle='-', zorder=0, alpha=.2)
    plt.plot(x, np.mean(baseline, axis=0), c='k', linewidth=3, linestyle='-', zorder=0)
    plt.fill_between(x, np.mean(baseline, axis=0) + stats.sem(baseline[:, 0]),
                     np.mean(baseline, axis=0) - stats.sem(baseline[:, 0]), alpha=0.2, color='k', lw=0)
    plt.xlim((.5, 5.5))
    plt.ylabel('Baseline activity\n(normalized\ndeconvolved $\mathregular{Ca^{2+}}$$\mathregular{s^{-1}}$)')
    plt.xticks([1, 2, 3, 4, 5], ['Top\n5% S1', 'Top\n5% S2', 'Bottom\n95% S1', 'Bottom\n95% S2', 'Other'])
    plt.ylim((0, 1.85))
    # plt.yticks([0, .05, .1, .15])
    sns.despine()
    plt.savefig(paths['base_path'] + '/NN/plots/reactivation_top_bottom_activity.png', bbox_inches='tight', dpi=200)
    plt.savefig(paths['base_path'] + '/NN/plots/reactivation_top_bottom_activity.pdf', bbox_inches='tight', dpi=200, transparent=True)
    plt.close()


def cue_cue_correlation(mice, sample_dates):
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(9, 6))
    plt.subplots_adjust(wspace=.4, hspace=.3)
    x_label = list(range(-20, 21))
    m_colors = ['b', 'purple', 'darkorange', 'green']
    paths = preprocess.create_folders(mice[0], sample_dates[0])
    reactivation_cue_corr = np.load(paths['base_path'] + paths['mouse'] +
                                    '/data_across_days/cue_cue_corr.npy', allow_pickle=True)
    all_cs = np.zeros((len(mice), len(reactivation_cue_corr[0][0])))
    all_cs_s = np.zeros((len(mice), len(reactivation_cue_corr[0][0])))
    all_cs_t = np.zeros((len(mice), len(reactivation_cue_corr[0][0])))
    all_cs_r_t = np.zeros((len(mice), len(reactivation_cue_corr[0][0])))
    all_cs_r_s_t = np.zeros((len(mice), len(reactivation_cue_corr[0][0])))
    for mouse in range(0, len(mice)):
        paths = preprocess.create_folders(mice[mouse], sample_dates[mouse])
        reactivation_cue_corr = np.load(paths['base_path'] + paths['mouse'] +
                                        '/data_across_days/cue_cue_corr.npy', allow_pickle=True)
        reactivation_cue_corr_shuffle = np.load(paths['base_path'] + paths['mouse'] +
                                        '/data_across_days/cue_cue_corr_shuffle.npy', allow_pickle=True)
        temp_cs = np.zeros((len(reactivation_cue_corr[0]), len(reactivation_cue_corr[0][0])))
        temp_cs_s = np.zeros((len(reactivation_cue_corr[0]), len(reactivation_cue_corr[0][0])))
        temp_cs_t = np.zeros((len(reactivation_cue_corr[0]), len(reactivation_cue_corr[0][0])))
        for i in range(0, len(reactivation_cue_corr[0])):
            temp_cs[i, :] = (reactivation_cue_corr[0][i] + reactivation_cue_corr[1][i]) / 2
            temp_cs_s[i, :] = (reactivation_cue_corr_shuffle[0][i] + reactivation_cue_corr_shuffle[1][i]) / 2
            temp_cs_t[i, :] = (temp_cs[i, :] - temp_cs_s[i, :]) / temp_cs_s[i, :]

        reactivation_cue_corr = np.load(paths['base_path'] + paths['mouse'] +
                                        '/data_across_days/reactivation_cue_corr.npy', allow_pickle=True)
        reactivation_cue_corr_s = np.load(paths['base_path'] + paths['mouse'] +
                                          '/data_across_days/reactivation_cue_corr_shuffle.npy', allow_pickle=True)
        temp_cs_r_t = np.zeros((len(reactivation_cue_corr[0]), len(reactivation_cue_corr[0][0])))
        for i in range(0, len(reactivation_cue_corr[0])):
            temp_cs_r = (reactivation_cue_corr[0][i] + reactivation_cue_corr[1][i]) / 2
            temp_cs_r_s = (reactivation_cue_corr_s[0][i] + reactivation_cue_corr_s[1][i]) / 2
            temp_cs_r_t[i, :] = temp_cs_r #(temp_cs_r - temp_cs_r_s) / temp_cs_r_s

        temp_cs_r_s_t = np.zeros((len(reactivation_cue_corr[0]), len(reactivation_cue_corr[0][0])))
        for i in range(0, len(reactivation_cue_corr[0])):
            temp_cs_r_s_t[i, :] = temp_cs_r_t[i, :] / temp_cs[i, :]

        y1 = np.nanmean(temp_cs, axis=0)
        y1[20] = np.nan
        all_cs[mouse, :] = y1
        plt.subplot(2, 2, 1)
        plt.plot(x_label, y1, '-', c=m_colors[mouse], alpha=.2, linewidth=2)
        y2 = np.nanmean(temp_cs_s, axis=0)
        y2[20] = np.nan
        all_cs_s[mouse, :] = y2
        plt.subplot(2, 2, 2)
        plt.plot(x_label, y2, '-', c=m_colors[mouse], alpha=.2, linewidth=2)
        y3 = np.nanmean(temp_cs_t, axis=0)
        y3[20] = np.nan
        all_cs_t[mouse, :] = y3
        plt.subplot(2, 2, 3)
        plt.plot(x_label, y3, '-', c=m_colors[mouse], alpha=.2, linewidth=2)
        y4 = np.nanmean(temp_cs_r_t, axis=0)
        y4[20] = np.nan
        all_cs_r_t[mouse, :] = y4
        y5 = np.nanmean(temp_cs_r_s_t, axis=0)
        y5[20] = np.nan
        all_cs_r_s_t[mouse, :] = y5

    anova_results = []
    slope, intercept, r_value, p_value_past, std_err = stats.linregress(
        np.concatenate([x_label[10:19], x_label[10:19], x_label[10:19], x_label[10:19]]),
        np.concatenate(all_cs_t[:, 10:19]))
    slope, intercept, r_value, p_value_future, std_err = stats.linregress(
        np.concatenate([x_label[21:30], x_label[21:30], x_label[21:30], x_label[21:30]]),
        np.concatenate(all_cs_t[:, 21:30]))
    anova_results.append(p_value_past)
    anova_results.append(p_value_future)
    anova_results_corrected = multipletests(anova_results, alpha=.05, method='holm')
    return [anova_results, anova_results_corrected[1]]

    plt.subplot(2, 2, 1)
    mean = all_cs.mean(axis=0)
    sem_plus = mean + stats.sem(all_cs, axis=0)
    sem_minus = mean - stats.sem(all_cs, axis=0)
    plt.plot(x_label, mean, '-', c='k', linewidth=3)
    plt.fill_between(x_label, sem_plus, sem_minus, alpha=0.2, color='k', lw=0)
    plt.ylabel('Correlation of response\n pattern to same stimlus\non trial i vs. trial j')
    plt.xlabel('Trial number (j-i)')
    plt.axvline(0, color='black', linestyle='--', linewidth=1, snap=False)
    plt.xlim(-10, 10)
    plt.ylim(0, .8)
    # plt.xticks([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
    plt.subplot(2, 2, 2)
    mean = all_cs_s.mean(axis=0)
    sem_plus = mean + stats.sem(all_cs_s, axis=0)
    sem_minus = mean - stats.sem(all_cs_s, axis=0)
    plt.plot(x_label, mean, '-', c='k', linewidth=3)
    plt.fill_between(x_label, sem_plus, sem_minus, alpha=0.2, color='k', lw=0)
    plt.ylabel('Correlation of\nshuffledstimuli patterns')
    plt.xlabel('Trial number (relative to trial i)')
    plt.axvline(0, color='black', linestyle='--', linewidth=1, snap=False)
    plt.xlim(-10, 10)
    plt.ylim(0, .8)
    # plt.yticks([.4, .5, .6])
    plt.subplot(2, 2, 3)
    mean = all_cs_t.mean(axis=0)
    sem_plus = mean + stats.sem(all_cs_t, axis=0)
    sem_minus = mean - stats.sem(all_cs_t, axis=0)
    plt.plot(x_label, mean, '-', c='k', linewidth=2)
    plt.fill_between(x_label, sem_plus, sem_minus, alpha=0.2, color='k', lw=0)
    plt.ylabel('Correlation of same\nstimuli patterns\n(% different from shuffle)')
    plt.xlabel('Trial number (relative to trial i)')
    plt.axvline(0, color='black', linestyle='--', linewidth=1, snap=False)
    plt.xlim(-10, 10)
    plt.ylim(-.15, .15)
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1, snap=False)
    sns.despine()
    plt.savefig(paths['base_path'] + '/NN/plots/cue_cue_correlation.png', bbox_inches='tight', dpi=200)
    plt.savefig(paths['base_path'] + '/NN/plots/cue_cue_correlation.pdf', bbox_inches='tight', dpi=200, transparent=200)
    plt.close()


def reactivation_cue_correlation(mice, sample_dates):
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(6, 6))
    plt.subplots_adjust(wspace=.4, hspace=.3)
    x_label = list(range(-19, 22))
    m_colors = ['b', 'purple', 'darkorange', 'green']
    paths = preprocess.create_folders(mice[0], sample_dates[0])
    reactivation_cue_corr = np.load(paths['base_path'] + paths['mouse'] +
                                    '/data_across_days/reactivation_cue_corr.npy', allow_pickle=True)
    all_cs_r = np.zeros((len(mice), len(reactivation_cue_corr[0][0])))
    all_cs_r_s = np.zeros((len(mice), len(reactivation_cue_corr[0][0])))
    all_cs_r_t = np.zeros((len(mice), len(reactivation_cue_corr[0][0])))
    for mouse in range(0, len(mice)):
        paths = preprocess.create_folders(mice[mouse], sample_dates[mouse])
        reactivation_cue_corr = np.load(paths['base_path'] + paths['mouse'] +
                                        '/data_across_days/reactivation_cue_corr.npy', allow_pickle=True)
        reactivation_cue_corr_s = np.load(paths['base_path'] + paths['mouse'] +
                                        '/data_across_days/reactivation_cue_corr_shuffle.npy', allow_pickle=True)
        temp_cs_r = np.zeros((len(reactivation_cue_corr[0]), len(reactivation_cue_corr[0][0])))
        temp_cs_r_s = np.zeros((len(reactivation_cue_corr[0]), len(reactivation_cue_corr[0][0])))
        temp_cs_r_t = np.zeros((len(reactivation_cue_corr[0]), len(reactivation_cue_corr[0][0])))
        for i in range(0, len(reactivation_cue_corr[0])):
            temp_cs_r[i, :] = (reactivation_cue_corr[0][i] + reactivation_cue_corr[1][i]) / 2
            temp_cs_r_s[i, :] = (reactivation_cue_corr_s[0][i] + reactivation_cue_corr_s[1][i]) / 2
            temp_cs_r_t[i, :] = 100 * (temp_cs_r[i, :] - temp_cs_r_s[i, :]) / temp_cs_r_s[i, :]
        y = np.nanmean(temp_cs_r, axis=0)
        y_s = np.nanmean(temp_cs_r_s, axis=0)
        y_t = np.nanmean(temp_cs_r_t, axis=0)
        all_cs_r[mouse, :] = y
        all_cs_r_s[mouse, :] = y_s
        all_cs_r_t[mouse, :] = y_t
        plt.subplot(2, 2, 1)
        plt.plot(x_label, y, '-', c=m_colors[mouse], alpha=.3, linewidth=2)
        plt.subplot(2, 2, 2)
        plt.plot(x_label, y_s, '-', c=m_colors[mouse], alpha=.3, linewidth=2)
        plt.subplot(2, 2, 3)
        plt.plot(x_label[18:21], y_t[18:21], '-', c=m_colors[mouse], alpha=.3, linewidth=2)

    anova_results = []
    slope, intercept, r_value, p_value_past, std_err = stats.linregress(
        np.concatenate([x_label[10:20], x_label[10:20], x_label[10:20], x_label[10:20]]),
        np.concatenate(all_cs_r[:, 10:20]))
    print(r_value)
    slope, intercept, r_value, p_value_future, std_err = stats.linregress(
        np.concatenate([x_label[19:29], x_label[19:29], x_label[19:29], x_label[19:29]]),
        np.concatenate(all_cs_r[:, 19:29]))
    print(r_value)
    anova_results.append(p_value_past)
    anova_results.append(p_value_future)
    anova_results_corrected = multipletests(anova_results, alpha=.05, method='holm')
    return [anova_results, anova_results_corrected[1]]

    plt.subplot(2, 2, 1)
    mean = all_cs_r.mean(axis=0)
    sem_plus = mean + stats.sem(all_cs_r, axis=0)
    sem_minus = mean - stats.sem(all_cs_r, axis=0)
    plt.plot(x_label, mean, '-', c='k', linewidth=3)
    plt.fill_between(x_label, sem_plus, sem_minus, alpha=0.2, color='k', lw=0)
    plt.ylabel('Correlation of reactivation\npattern on trial i with\nstimulus pattern on trial j')
    plt.xlabel('Trial number (j-i)')
    plt.axvline(0, color='black', linestyle='--', linewidth=1, snap=False)
    plt.yticks([0, .05, .1, .15, .2, .25])
    plt.ylim((0, .22))
    plt.xlim(-10, 10)
    plt.subplot(2, 2, 2)
    mean = all_cs_r_s.mean(axis=0)
    sem_plus = mean + stats.sem(all_cs_r_s, axis=0)
    sem_minus = mean - stats.sem(all_cs_r_s, axis=0)
    plt.plot(x_label, mean, '-', c='k', linewidth=3)
    plt.fill_between(x_label, sem_plus, sem_minus, alpha=0.2, color='k', lw=0)
    plt.ylabel('Correlation of reactivation\npattern on trial i with\nstimulus pattern on trial j')
    plt.xlabel('Trial number (j-i)')
    plt.axvline(0, color='black', linestyle='--', linewidth=1, snap=False)
    plt.yticks([0, .05, .1, .15, .2, .25])
    plt.ylim((0, .22))
    plt.xlim(-10, 10)
    plt.subplot(2, 2, 3)
    mean = all_cs_r_t.mean(axis=0)
    sem_plus = mean + stats.sem(all_cs_r_t, axis=0)
    sem_minus = mean - stats.sem(all_cs_r_t, axis=0)
    plt.plot(x_label[18:21], mean[18:21], '-', c='k', linewidth=2)
    plt.fill_between(x_label[18:21], sem_plus[18:21], sem_minus[18:21], alpha=0.2, color='k', lw=0)
    plt.ylabel('Correlation of reactivation\npattern on trial i with\nstimulus pattern on trial j\n(% change from shuffle)')
    plt.xlabel('Trial number (j-i)')
    plt.axvline(0, color='black', linestyle='--', linewidth=1, snap=False)
    plt.ylim((-15, 15))
    plt.xlim(-1.35, 1.35)
    plt.xticks([-1, 0, 1])
    # plt.yticks([-30, -20, -10, 0, 10, 20, 30])
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1, snap=False)
    sns.despine()
    plt.savefig(paths['base_path'] + '/NN/plots/reactivation_cue_correlation.png', bbox_inches='tight', dpi=200)
    plt.savefig(paths['base_path'] + '/NN/plots/reactivation_cue_correlation.pdf', bbox_inches='tight', dpi=200, transparent=True)
    plt.close()


def reactivation_cue_correlation_smooth(mice, sample_dates):
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(9, 6))
    plt.subplots_adjust(wspace=.4, hspace=.3)
    x_label = list(range(-19, 22))
    m_colors = ['b', 'purple', 'darkorange', 'green']
    paths = preprocess.create_folders(mice[0], sample_dates[0])
    reactivation_cue_corr = np.load(paths['base_path'] + paths['mouse'] +
                                    '/data_across_days/reactivation_cue_corr_smooth.npy', allow_pickle=True)
    all_cs_r = np.zeros((len(mice), len(reactivation_cue_corr[0][0])))
    all_cs_r_s = np.zeros((len(mice), len(reactivation_cue_corr[0][0])))
    all_cs_r_t = np.zeros((len(mice), len(reactivation_cue_corr[0][0])))
    all_cs_r_t_s = np.zeros((len(mice), len(reactivation_cue_corr[0][0])))
    for mouse in range(0, len(mice)):
        paths = preprocess.create_folders(mice[mouse], sample_dates[mouse])
        reactivation_cue_corr = np.load(paths['base_path'] + paths['mouse'] +
                                        '/data_across_days/reactivation_cue_corr_smooth.npy', allow_pickle=True)
        reactivation_cue_corr_s = np.load(paths['base_path'] + paths['mouse'] +
                                        '/data_across_days/reactivation_cue_corr_smooth_shuffle.npy', allow_pickle=True)
        reactivation_cue_corr_r = np.load(paths['base_path'] + paths['mouse'] +
                                          '/data_across_days/reactivation_cue_corr.npy', allow_pickle=True)
        reactivation_cue_corr_r_s = np.load(paths['base_path'] + paths['mouse'] +
                                          '/data_across_days/reactivation_cue_corr_shuffle.npy', allow_pickle=True)
        temp_cs_r = np.zeros((len(reactivation_cue_corr[0]), len(reactivation_cue_corr[0][0])))
        temp_cs_r_s = np.zeros((len(reactivation_cue_corr[0]), len(reactivation_cue_corr[0][0])))
        temp_cs_r_t = np.zeros((len(reactivation_cue_corr[0]), len(reactivation_cue_corr[0][0])))
        temp_cs_r_t_s = np.zeros((len(reactivation_cue_corr[0]), len(reactivation_cue_corr[0][0])))
        for i in range(0, len(reactivation_cue_corr[0])):
            temp_cs_r[i, :] = (reactivation_cue_corr[0][i] + reactivation_cue_corr[1][i]) / 2
            temp_cs_r_s[i, :] = (reactivation_cue_corr_s[0][i] + reactivation_cue_corr_s[1][i]) / 2
            temp_cs_r_t[i, :] = 100 * (temp_cs_r[i, :] - temp_cs_r_s[i, :]) / temp_cs_r_s[i, :]
            temp_1 = (reactivation_cue_corr_r[0][i] + reactivation_cue_corr_r[1][i]) / 2
            temp_2 = (reactivation_cue_corr_r_s[0][i] + reactivation_cue_corr_r_s[1][i]) / 2
            temp_cs_r_t_s[i, :] = 100 * (temp_1 - temp_2) / temp_1
        y = np.nanmean(temp_cs_r, axis=0)
        y_s = np.nanmean(temp_cs_r_s, axis=0)
        y_t = np.nanmean(temp_cs_r_t, axis=0)
        y_t_s = np.nanmean(temp_cs_r_t_s, axis=0)
        all_cs_r[mouse, :] = y
        all_cs_r_s[mouse, :] = y_s
        all_cs_r_t[mouse, :] = y_t
        all_cs_r_t_s[mouse, :] = y_t_s
        plt.subplot(2, 2, 1)
        plt.plot(x_label, y, '-', c=m_colors[mouse], alpha=.3, linewidth=2)
        plt.subplot(2, 2, 2)
        plt.plot(x_label, y_s, '-', c=m_colors[mouse], alpha=.3, linewidth=2)
        plt.subplot(2, 2, 3)
        plt.plot(x_label, y_t, '-', c=m_colors[mouse], alpha=.3, linewidth=2)
    plt.subplot(2, 2, 1)
    mean = all_cs_r.mean(axis=0)
    sem_plus = mean + stats.sem(all_cs_r, axis=0)
    sem_minus = mean - stats.sem(all_cs_r, axis=0)
    plt.plot(x_label, mean, '-', c='k', linewidth=3)
    plt.fill_between(x_label, sem_plus, sem_minus, alpha=0.2, color='k')
    plt.ylabel('Correlation of reactivation\n and stimulus response pattern\n on trial i vs. trial j')
    plt.xlabel('Trial number (relative to reactivation i)')
    plt.axvline(0, color='black', linestyle='--', linewidth=1, snap=False)
    # plt.ylim((0, .175))
    plt.yticks([0, .05, .1, .15])
    plt.xlim(-10, 10)
    plt.subplot(2, 2, 2)
    mean = all_cs_r_s.mean(axis=0)
    sem_plus = mean + stats.sem(all_cs_r_s, axis=0)
    sem_minus = mean - stats.sem(all_cs_r_s, axis=0)
    plt.plot(x_label, mean, '-', c='k', linewidth=3)
    plt.fill_between(x_label, sem_plus, sem_minus, alpha=0.2, color='k')
    plt.ylabel('Correlation of reactivation\n and stimulus response pattern\n on trial i vs. trial j')
    plt.xlabel('Trial number (relative to reactivation i)')
    plt.axvline(0, color='black', linestyle='--', linewidth=1, snap=False)
    plt.ylim((0, .175))
    plt.yticks([0, .05, .1, .15])
    plt.xlim(-10, 10)
    plt.subplot(2, 2, 3)
    mean = all_cs_r_t.mean(axis=0)
    sem_plus = mean + stats.sem(all_cs_r_t, axis=0)
    sem_minus = mean - stats.sem(all_cs_r_t, axis=0)
    plt.plot(x_label, mean, '-', c='k', linewidth=2)
    plt.fill_between(x_label, sem_plus, sem_minus, alpha=0.2, color='k')
    plt.ylabel('Correlation of reactivation\n and stimulus response pattern\n on trial i vs. trial j\n (% different from shuffle)')
    plt.xlabel('Trial number (relative to reactivation i)')
    plt.axvline(0, color='black', linestyle='--', linewidth=1, snap=False)
    plt.ylim((-30, 30))
    plt.xlim(-10, 10)
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1, snap=False)
    plt.subplot(2, 2, 4)
    mean = all_cs_r_t_s.mean(axis=0)
    sem_plus = mean + stats.sem(all_cs_r_t_s, axis=0)
    sem_minus = mean - stats.sem(all_cs_r_t_s, axis=0)
    plt.plot(x_label, mean, '-', c=[.6, 0, .6], linewidth=2)
    plt.fill_between(x_label, sem_plus, sem_minus, alpha=0.2, color=[.6, 0, .6])
    mean = all_cs_r_t.mean(axis=0)
    sem_plus = mean + stats.sem(all_cs_r_t, axis=0)
    sem_minus = mean - stats.sem(all_cs_r_t, axis=0)
    plt.plot(x_label, mean, '-', c='k', linewidth=2)
    plt.fill_between(x_label, sem_plus, sem_minus, alpha=0.2, color='k')
    plt.ylabel(
        'Correlation of reactivation\n and stimulus response pattern\n on trial i vs. trial j\n (% different from shuffle)')
    plt.xlabel('Trial number (relative to reactivation i)')
    plt.axvline(0, color='black', linestyle='--', linewidth=1, snap=False)
    plt.ylim((-20, 20))
    plt.xlim(-10, 10)
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1, snap=False)
    label_1 = mlines.Line2D([], [], color='k', linewidth=2, label='Real data')
    label_2 = mlines.Line2D([], [], color=[.6, 0, .6], linewidth=2, label='Smoothed\nstimulus\nresponses')
    plt.legend(handles=[label_1, label_2], frameon=False)
    sns.despine()
    plt.savefig(paths['base_path'] + '/NN/plots/reactivation_cue_correlation_smooth.png', bbox_inches='tight', dpi=200)
    plt.savefig(paths['base_path'] + '/NN/plots/reactivation_cue_correlation_smooth.pdf', bbox_inches='tight', dpi=200, transparent=True)
    plt.close()


def activity_across_trials(mice, sample_dates):
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(9, 6))
    plt.subplots_adjust(wspace=.33)
    m_colors = ['b', 'purple', 'darkorange', 'green']

    plt.subplot(2, 2, 1)
    activity_all = np.empty((len(mice), 128)) * np.nan
    mean_activity_mice_heatmap = []
    days = []
    for mouse in range(0, len(mice)):
        paths = preprocess.create_folders(mice[mouse], sample_dates[mouse])
        activity_data = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/activity.npy',
                               allow_pickle=True)
        activity = np.empty((len(activity_data[0]), 128)) * np.nan
        for i in range(0, len(activity_data[0])):
            smoothed_activity = np.array(pd.DataFrame(activity_data[0][i]).rolling(3, min_periods=1, center=True).mean())
            smoothed_activity = np.concatenate(smoothed_activity, axis=0)
            activity[i, 0:len(smoothed_activity)] = smoothed_activity
            mean_activity_mice_heatmap.append(smoothed_activity[0:118]/np.max(smoothed_activity[0:118]))
        activity = np.nanmean(activity, axis=0)
        activity_all[mouse, :] = activity
        x = range(1, len(activity)+1)
        plt.plot(x, activity, c=m_colors[mouse], alpha=.2)
        days.append(len(activity_data[0]))

    # print(stats.ttest_rel(np.mean(activity_all[:, 0:3], axis=0), np.mean(activity_all[:, 118:121], axis=0)))
    mask = ~np.isnan(np.concatenate([x, x, x, x])) & ~np.isnan(np.concatenate(activity_all))
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        np.concatenate([x, x, x, x])[mask], np.concatenate(activity_all)[mask])
    print(r_value)
    print(p_value)

    mean = np.nanmean(activity_all, axis=0)
    sem_plus = mean + stats.sem(activity_all, axis=0, nan_policy='omit')
    sem_minus = mean - stats.sem(activity_all, axis=0, nan_policy='omit')
    plt.plot(x, mean, c='k', linewidth=3)
    plt.fill_between(x, sem_plus, sem_minus, alpha=0.2, color='k', lw=0)
    plt.ylabel('Correlation between\nstimulus 1 and stimulus 2')
    plt.xlabel('Trial number')
    # plt.ylim(-.1, .6)
    plt.xlim(1, 123)
    plt.xticks([1, 20, 40, 60, 80, 100, 120])
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
    sns.despine()

    sns.set(font_scale=.7)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(4, 6))
    plt.subplot(2, 2, 1)
    sns.heatmap(mean_activity_mice_heatmap, vmin=.3, vmax=1, cmap="Reds", cbar=0)
    y_idx = 0
    for mouse in range(0, len(mice)):
        plt.plot([-5, -5], [y_idx, y_idx + days[mouse]], color=m_colors[mouse], linewidth=7, snap=False,
                 solid_capstyle='butt')
        y_idx += days[mouse]
    plt.ylim(len(mean_activity_mice_heatmap) + 3, -3)
    plt.xlim(-5, 120)
    plt.axis('off')
    plt.savefig(paths['base_path'] + '/NN/plots/activity_across_trials_correlation_heatmap.png', bbox_inches='tight', dpi=500,
                transparent=True)

    plt.subplot(2, 2, 2)
    activity_all = np.empty((len(mice), 128)) * np.nan
    mean_activity_mice_heatmap = []
    days = []
    for mouse in range(0, len(mice)):
        paths = preprocess.create_folders(mice[mouse], sample_dates[mouse])
        activity_data = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/activity.npy',
                                allow_pickle=True)
        activity = np.empty((len(activity_data[0]), 128)) * np.nan
        for i in range(0, len(activity_data[0])):
            smoothed_activity = np.array(
                pd.DataFrame(activity_data[2][i]).rolling(3, min_periods=1, center=True).mean())
            smoothed_activity = np.concatenate(smoothed_activity, axis=0)
            activity[i, 0:len(smoothed_activity)] = smoothed_activity
            mean_activity_mice_heatmap.append(smoothed_activity[0:118])
        activity = np.nanmean(activity, axis=0)
        activity_all[mouse, :] = activity
        x = range(1, len(activity) + 1)
        plt.plot(x, activity, c=m_colors[mouse], alpha=.2)
        days.append(len(activity_data[0]))

    mask = ~np.isnan(np.concatenate([x, x, x, x])) & ~np.isnan(np.concatenate(activity_all))
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        np.concatenate([x, x, x, x])[mask], np.concatenate(activity_all)[mask])
    print(r_value)
    print(p_value)
    #print(stats.ttest_rel(np.mean(activity_all[:, 0:3], axis=0), np.mean(activity_all[:, 118:121], axis=0)))

    mean = np.nanmean(activity_all, axis=0)
    sem_plus = mean + stats.sem(activity_all, axis=0, nan_policy='omit')
    sem_minus = mean - stats.sem(activity_all, axis=0, nan_policy='omit')
    plt.plot(x, mean, c='k', linewidth=3)
    plt.fill_between(x, sem_plus, sem_minus, alpha=0.2, color='k', lw=0)
    plt.ylabel('Stimulus activity\n(Normalized $\mathregular{Ca^{2+}}$transient $\mathregular{s^{-1}}$)')
    plt.xlabel('Trial number')
    # plt.ylim(0, .07)
    plt.xlim(1, 123)
    plt.xticks([1, 20, 40, 60, 80, 100, 120])
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
    sns.despine()

    sns.set(font_scale=.7)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(4, 6))
    plt.subplot(2, 2, 1)
    sns.heatmap(mean_activity_mice_heatmap, vmin=.03, vmax=.08, cmap="Reds", cbar=0)
    y_idx = 0
    for mouse in range(0, len(mice)):
        plt.plot([-5, -5], [y_idx, y_idx + days[mouse]], color=m_colors[mouse], linewidth=7, snap=False,
                 solid_capstyle='butt')
        y_idx += days[mouse]
    plt.ylim(len(mean_activity_mice_heatmap) + 3, -3)
    plt.xlim(-5, 120)
    plt.axis('off')
    plt.savefig(paths['base_path'] + '/NN/plots/activity_across_trials_activity_heatmap.png', bbox_inches='tight',
                dpi=500,
                transparent=True)

    plt.subplot(2, 2, 3)
    activity_all = np.empty((len(mice), 128)) * np.nan
    for mouse in range(0, len(mice)):
        paths = preprocess.create_folders(mice[mouse], sample_dates[mouse])
        activity_data = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/activity.npy',
                                allow_pickle=True)
        activity = np.empty((len(activity_data[0]), 128)) * np.nan
        for i in range(0, len(activity_data[0])):
            smoothed_activity = np.array(
                pd.DataFrame(activity_data[3][i]).rolling(3, min_periods=1, center=True).mean())
            smoothed_activity = np.concatenate(smoothed_activity, axis=0)
            activity[i, 0:len(smoothed_activity)] = smoothed_activity
        activity = np.nanmean(activity, axis=0)
        activity_all[mouse, :] = activity
        x = range(1, len(activity) + 1)
        plt.plot(x, activity, c=m_colors[mouse], alpha=.2)
    mean = np.nanmean(activity_all, axis=0)
    sem_plus = mean + stats.sem(activity_all, axis=0, nan_policy='omit')
    sem_minus = mean - stats.sem(activity_all, axis=0, nan_policy='omit')
    plt.plot(x, mean, c='mediumseagreen', linewidth=3)
    plt.fill_between(x, sem_plus, sem_minus, alpha=0.2, color='mediumseagreen', lw=0)
    plt.ylabel('Correlation between\nstimulus 1 and stimulus 1')
    plt.xlabel('Trial number')
    plt.ylim(.3, .8)
    plt.xlim(1, 60)
    plt.xticks([1, 20, 40, 60])
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
    sns.despine()

    plt.subplot(2, 2, 4)
    activity_all = np.empty((len(mice), 128)) * np.nan
    for mouse in range(0, len(mice)):
        paths = preprocess.create_folders(mice[mouse], sample_dates[mouse])
        activity_data = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/activity.npy',
                                allow_pickle=True)
        activity = np.empty((len(activity_data[0]), 128)) * np.nan
        for i in range(0, len(activity_data[0])):
            smoothed_activity = np.array(
                pd.DataFrame(activity_data[4][i]).rolling(3, min_periods=1, center=True).mean())
            smoothed_activity = np.concatenate(smoothed_activity, axis=0)
            activity[i, 0:len(smoothed_activity)] = smoothed_activity
        activity = np.nanmean(activity, axis=0)
        activity_all[mouse, :] = activity
        x = range(1, len(activity) + 1)
        plt.plot(x, activity, c=m_colors[mouse], alpha=.2)
    mean = np.nanmean(activity_all, axis=0)
    sem_plus = mean + stats.sem(activity_all, axis=0, nan_policy='omit')
    sem_minus = mean - stats.sem(activity_all, axis=0, nan_policy='omit')
    plt.plot(x, mean, c='salmon', linewidth=3)
    plt.fill_between(x, sem_plus, sem_minus, alpha=0.2, color='salmon', lw=0)
    plt.ylabel('Correlation between\nstimulus 2 and stimulus 2')
    plt.xlabel('Trial number')
    plt.ylim(.3, .8)
    plt.xlim(1, 60)
    plt.xticks([1, 20, 40, 60])
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
    sns.despine()

    plt.savefig(paths['base_path'] + '/NN/plots/activity_across_trials.png', bbox_inches='tight', dpi=200)
    plt.savefig(paths['base_path'] + '/NN/plots/activity_across_trials.pdf', bbox_inches='tight', dpi=200, transparent=True)
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
        paths = preprocess.create_folders(mice[mouse], sample_dates[mouse])
        activity_data = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/activity_grouped_remove.npy',
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

    print(stats.ttest_rel(
        np.nanmean(activity_increase_all[:, 0:2], axis=1) - np.nanmean(activity_increase_all[:, 119:121], axis=1),
        np.nanmean(activity_decrease_all[:, 0:3], axis=1) - np.nanmean(activity_decrease_all[:, 119:121], axis=1))[1])

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
    plt.ylabel('Response similarity\n(correlation between\nS1 and S2)')
    plt.xlabel('Trial number')
    # plt.ylim(-.1, .6)
    plt.xlim(1, 123)
    plt.xticks([1, 20, 40, 60, 80, 100, 120])
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
    sns.despine()

    plt.subplot(2, 2, 2)
    activity_same_all = np.empty((len(mice), 128)) * np.nan
    activity_increase_all = np.empty((len(mice), 128)) * np.nan
    activity_decrease_all = np.empty((len(mice), 128)) * np.nan
    for mouse in range(0, len(mice)):
        paths = preprocess.create_folders(mice[mouse], sample_dates[mouse])
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
    plt.xlim(1, 123)
    plt.xticks([1, 20, 40, 60, 80, 100, 120])
    plt.ylim(0, .12)
    label_1 = mlines.Line2D([], [], color='k', linewidth=2, label='No change cells')
    label_2 = mlines.Line2D([], [], color='darkred', linewidth=2, label='Increase cells')
    label_3 = mlines.Line2D([], [], color='darkblue', linewidth=2, label='Decrease cells')
    plt.legend(handles=[label_1, label_2, label_3], frameon=False)
    sns.despine()

    plt.savefig(paths['base_path'] + '/NN/plots/activity_across_trials_grouped.png', bbox_inches='tight', dpi=200)
    plt.savefig(paths['base_path'] + '/NN/plots/activity_across_trials_grouped.pdf', bbox_inches='tight', dpi=200, transparent=True)
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
        paths = preprocess.create_folders(mice[mouse], sample_dates[mouse])
        activity_data = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/activity_grouped_no_change.npy',
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

    anova_results = []

    anova_results.append(stats.ttest_rel(np.mean(cs1d_cs1_all[:, 0:3], axis=0),
                                         np.mean(cs1d_cs2_all[:, 0:3], axis=0))[1])
    anova_results.append(stats.ttest_rel(np.mean(cs2d_cs2_all[:, 0:3], axis=0),
                                         np.mean(cs2d_cs1_all[:, 0:3], axis=0))[1])
    anova_results.append(stats.ttest_rel(np.mean(cs1d_cs1_all[:, 58:61], axis=0),
                                         np.mean(cs1d_cs2_all[:, 58:61], axis=0))[1])
    anova_results.append(stats.ttest_rel(np.mean(cs2d_cs2_all[:, 58:61], axis=0),
                                         np.mean(cs2d_cs1_all[:, 58:61], axis=0))[1])

    anova_results_corrected = multipletests(anova_results, alpha=.05, method='holm')
    return [anova_results, anova_results_corrected[1]]


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
    plt.xlim(1, 60)
    plt.xticks([1, 20, 40, 60])
    plt.yticks([0, .05, .1])
    sns.despine()
    label_1 = mlines.Line2D([], [], color='mediumseagreen', linewidth=2, label='S1 no change cells, S1 trials')
    label_2 = mlines.Line2D([], [], color='darkgreen', linewidth=2, label='S1 no change cells, S2 trials')
    label_3 = mlines.Line2D([], [], color='salmon', linewidth=2, label='S2 no change cells, S2 trials')
    label_4 = mlines.Line2D([], [], color='darkred', linewidth=2, label='S2 no change cells, S1 trials')
    plt.legend(handles=[label_1, label_2, label_3, label_4], frameon=False)

    plt.savefig(paths['base_path'] + '/NN/plots/activity_across_trials_grouped_no_change.png', bbox_inches='tight', dpi=200)
    plt.savefig(paths['base_path'] + '/NN/plots/activity_across_trials_grouped_no_change.pdf', bbox_inches='tight', dpi=200, transparent=True)
    plt.close()


def cue_reactivation_activity(mice, sample_dates):
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(9, 6))
    plt.subplots_adjust(wspace=.4)
    paths = preprocess.create_folders(mice[0], sample_dates[0])
    activity_vec = np.load(paths['base_path'] + paths['mouse'] +
                                    '/data_across_days/cue_reactivation_activity.npy', allow_pickle=True)
    all_s1 = np.zeros((len(mice), len(activity_vec[0][0])))
    all_s2 = np.zeros((len(mice), len(activity_vec[0][0])))
    for mouse in range(0, len(mice)):
        paths = preprocess.create_folders(mice[mouse], sample_dates[mouse])
        activity_vec = np.load(paths['base_path'] + paths['mouse'] +
                                           '/data_across_days/cue_reactivation_activity.npy', allow_pickle=True)
        temp_s1 = np.zeros((len(activity_vec[0]), len(activity_vec[0][0])))
        temp_s2 = np.zeros((len(activity_vec[0]), len(activity_vec[0][0])))
        for i in range(0, len(activity_vec[0])):
            temp_s1[i, :] = np.array(activity_vec[0][i])
            temp_s2[i, :] = np.array(activity_vec[1][i])
        y_s1 = np.mean(temp_s1, axis=0)
        y_s2 = np.mean(temp_s2, axis=0)
        all_s1[mouse, :] = y_s1
        all_s2[mouse, :] = y_s2
    x_label = activity_vec[2][0]
    plt.subplot(2, 2, 1)
    mean = all_s1.mean(axis=0)
    sem_plus = mean + stats.sem(all_s1, axis=0)
    sem_minus = mean - stats.sem(all_s1, axis=0)
    plt.plot(x_label, mean, '-', c='mediumseagreen', linewidth=3)

    # a, b = np.polyfit(mean, x_label, 1)
    # print(a)

    plt.fill_between(x_label, sem_plus, sem_minus, alpha=0.2, color='mediumseagreen', lw=0)
    plt.xlim(0, 2.5)
    plt.xlabel('Stimulus 1 activity\n(Normalized $\mathregular{Ca^{2+}}$transient $\mathregular{s^{-1}}$)')
    plt.ylabel('Stimulus 1 reactivation activity\n(Normalized $\mathregular{Ca^{2+}}$transient $\mathregular{s^{-1}}$)')
    plt.subplot(2, 2, 2)
    mean = all_s2.mean(axis=0)
    sem_plus = mean + stats.sem(all_s2, axis=0)
    sem_minus = mean - stats.sem(all_s2, axis=0)
    plt.plot(x_label, mean, '-', c='salmon', linewidth=3)

    # a, b = np.polyfit(mean, x_label, 1)
    # print(a)

    plt.fill_between(x_label, sem_plus, sem_minus, alpha=0.2, color='salmon', lw=0)
    plt.xlim(0, 2.5)
    plt.xlabel('Stimulus 2 activity\n(Normalized $\mathregular{Ca^{2+}}$transient $\mathregular{s^{-1}}$)')
    plt.ylabel('Stimulus 2 reactivation activity\n(Normalized $\mathregular{Ca^{2+}}$transient $\mathregular{s^{-1}}$)')

    sns.despine()
    plt.savefig(paths['base_path'] + '/NN/plots/cue_reactivation_activity.png', bbox_inches='tight', dpi=200)
    plt.savefig(paths['base_path'] + '/NN/plots/cue_reactivation_activity.pdf', bbox_inches='tight', dpi=200,
                transparent=True)
    plt.close()


def activity_reactivation_correlation(mice, sample_dates):
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(13.5, 6))
    plt.subplots_adjust(wspace=.3)
    m_colors = ['b', 'purple', 'darkorange', 'green']

    corr_all_mice = []
    plt.subplot(2, 3, 1)
    for mouse in range(0, len(mice)):
        paths = preprocess.create_folders(mice[mouse], sample_dates[mouse])
        activity_all = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/activity.npy', allow_pickle=True)
        corr_all = []
        for i in range(0, len(activity_all[0])):
            smoothed_correlation = np.array(pd.DataFrame(activity_all[0][i]).rolling(8, min_periods=1,
                                                                                        center=True).mean())
            smoothed_reactivation_prob = np.array(pd.DataFrame(activity_all[1][i]).rolling(8, min_periods=1,
                                                                                              center=True).mean())

            # sos = signal.butter(2, .1 / 60 / 2, btype='highpass', output='sos', fs=1 / 60)
            # smoothed_correlation = signal.sosfilt(sos, smoothed_correlation)
            # smoothed_reactivation_prob = signal.sosfilt(sos, smoothed_reactivation_prob)
            # smoothed_correlation = smoothed_correlation[5:len(smoothed_correlation)]
            # smoothed_reactivation_prob = smoothed_reactivation_prob[5:len(smoothed_reactivation_prob)]

            corr_temp = np.corrcoef(np.concatenate(smoothed_correlation, axis=0),
                                    np.concatenate(smoothed_reactivation_prob, axis=0))[0][1]
            corr_all.append(corr_temp)

            # if mouse == 2 and i == 2:
            #     sns.set(font_scale=1)
            #     sns.set_style("whitegrid", {'axes.grid': False})
            #     sns.set_style("ticks")
            #     plt.figure(figsize=(6, 6))
            #     x = range(1, 121)
            #     ax1 = plt.subplot(2, 2, 1)
            #     ax1.spines['top'].set_visible(False)
            #     ax1.plot(x, smoothed_correlation, c='k')
            #     plt.ylabel('Response similarity\n(correlation between\nS1 and S2)')
            #     plt.xlabel('Trial number')
            #     ax2 = ax1.twinx()
            #     ax2.spines['top'].set_visible(False)
            #     ax2.plot(x, smoothed_reactivation_prob, c='darkorange')
            #     ax2.spines['right'].set_color('darkorange')
            #     ax2.tick_params(axis='y', colors='darkorange')
            #     plt.ylabel('Reactivation rate (probablity $\mathregular{s^{-1}}$)', rotation=270, c='darkorange', labelpad=15)
            #     plt.xlim(1, 120)
            #     plt.xticks([1, 20, 40, 60, 80, 100, 120])
            #     plt.savefig(paths['base_path'] + '/NN/plots/activity_reactivation_correlation_example.png',
            #                 bbox_inches='tight', dpi=200)
            #     plt.savefig(paths['base_path'] + '/NN/plots/activity_reactivation_correlation_example.pdf',
            #                 bbox_inches='tight', dpi=200, transparent=True)
            #     plt.close()

        plt.axhline(y=0, color='black', linestyle='--', linewidth=1, snap=False)
        y1 = np.mean(corr_all)
        plt.errorbar(1, y1, yerr=0, c=m_colors[mouse], marker='o', mfc='none', mec=m_colors[mouse], ms=5, alpha=.3)
        corr_all_mice.append(y1)

    print(stats.ttest_1samp(corr_all_mice, 0))

    # anova_results = []
    # anova_results.append(0.01378381384301423)
    # anova_results.append(0.01681435334458543)
    # anova_results_corrected = multipletests(anova_results, alpha=.05, method='holm')
    # return [anova_results, anova_results_corrected[1]]

    y1 = np.mean(corr_all_mice)
    y1_err = stats.sem(corr_all_mice)
    plt.errorbar(1.2, y1, yerr=y1_err, c='k', linewidth=2, marker='o', mfc='k', mec='k',
                 ms=7, mew=0, zorder=100)
    plt.ylabel('Correlation between\nresponse similarity\nand reactivation probability')
    plt.xlim(.5, 4.5)
    plt.ylim(-.62, .62)
    plt.yticks([-.6, -.4, -.2, 0, .2, .4, .6])
    plt.xticks([])
    sns.despine()

    corr_all_mice = []
    plt.subplot(2, 3, 2)
    for mouse in range(0, len(mice)):
        paths = preprocess.create_folders(mice[mouse], sample_dates[mouse])
        activity_all = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/activity.npy',
                               allow_pickle=True)
        corr_all = []
        for i in range(0, len(activity_all[3])):
            smoothed_correlation = np.array(pd.DataFrame(activity_all[3][i]).rolling(4, min_periods=1,
                                                                                     center=True).mean())
            smoothed_reactivation_prob = np.array(pd.DataFrame(activity_all[5][i]).rolling(4, min_periods=1,
                                                                                           center=True).mean())
            corr_temp = np.corrcoef(np.concatenate(smoothed_correlation, axis=0),
                                    np.concatenate(smoothed_reactivation_prob, axis=0))[0][1]
            corr_all.append(corr_temp)
        plt.axhline(y=0, color='black', linestyle='--', linewidth=1, snap=False)
        y1 = np.mean(corr_all)
        plt.errorbar(1, y1, yerr=0, c=m_colors[mouse], marker='o', mfc='none', mec=m_colors[mouse], ms=5, alpha=.3)
        corr_all_mice.append(y1)
    y1 = np.mean(corr_all_mice)
    y1_err = stats.sem(corr_all_mice)
    plt.errorbar(1.1, y1, yerr=y1_err, c='k', linewidth=2, marker='o', mfc='k', mec='k',
                 ms=7, mew=0, zorder=100)
    plt.ylabel('Correlation between S1\n and S1 reactivation prob.')
    plt.xlim(.5, 1.5)
    plt.ylim(-.6, .6)
    plt.xticks([])
    sns.despine()

    corr_all_mice = []
    plt.subplot(2, 3, 3)
    for mouse in range(0, len(mice)):
        paths = preprocess.create_folders(mice[mouse], sample_dates[mouse])
        activity_all = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/activity.npy',
                               allow_pickle=True)
        corr_all = []
        for i in range(0, len(activity_all[3])):
            smoothed_correlation = np.array(pd.DataFrame(activity_all[4][i]).rolling(4, min_periods=1,
                                                                                     center=True).mean())
            smoothed_reactivation_prob = np.array(pd.DataFrame(activity_all[6][i]).rolling(4, min_periods=1,
                                                                                           center=True).mean())
            corr_temp = np.corrcoef(np.concatenate(smoothed_correlation, axis=0),
                                    np.concatenate(smoothed_reactivation_prob, axis=0))[0][1]
            corr_all.append(corr_temp)
        plt.axhline(y=0, color='black', linestyle='--', linewidth=1, snap=False)
        y1 = np.mean(corr_all)
        plt.errorbar(1, y1, yerr=0, c=m_colors[mouse], marker='o', mfc='none', mec=m_colors[mouse], ms=5, alpha=.3)
        corr_all_mice.append(y1)
    y1 = np.mean(corr_all_mice)
    y1_err = stats.sem(corr_all_mice)
    plt.errorbar(1.1, y1, yerr=y1_err, c='k', linewidth=2, marker='o', mfc='k', mec='k',
                 ms=7, mew=0, zorder=100)
    plt.ylabel('Correlation between S2\n and S2 reactivation prob.')
    plt.xlim(.5, 1.5)
    plt.ylim(-.6, .6)
    plt.xticks([])

    plt.subplot(2, 3, 4)
    maxlags = 10
    xcorr_correlation_all = np.zeros((len(mice), (maxlags * 2) + 1))
    for mouse in range(0, len(mice)):
        paths = preprocess.create_folders(mice[mouse], sample_dates[mouse])
        activity_all = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/activity.npy',
                               allow_pickle=True)
        xcorr_correlation = np.zeros((len(activity_all[0]), (maxlags * 2) + 1))
        for i in range(0, len(activity_all[0])):
            smoothed_correlation = np.array(pd.DataFrame(activity_all[0][i]).rolling(8, min_periods=1,
                                                                                     center=True).mean())
            smoothed_reactivation_prob = np.array(pd.DataFrame(activity_all[1][i]).rolling(8, min_periods=1,
                                                                                           center=True).mean())
            smoothed_correlation = np.concatenate(smoothed_correlation, axis=0)
            smoothed_reactivation_prob = np.concatenate(smoothed_reactivation_prob, axis=0)

            sos = signal.butter(2, .1/60/2, btype='highpass', output='sos', fs=1/60)
            smoothed_correlation = signal.sosfilt(sos, smoothed_correlation)
            smoothed_reactivation_prob = signal.sosfilt(sos, smoothed_reactivation_prob)
            smoothed_correlation = smoothed_correlation[5:len(smoothed_correlation)]
            smoothed_reactivation_prob = smoothed_reactivation_prob[5:len(smoothed_reactivation_prob)]
            smoothed_correlation = smoothed_correlation - np.mean(smoothed_correlation)
            smoothed_reactivation_prob = smoothed_reactivation_prob - np.mean(smoothed_reactivation_prob)

            # smoothed_correlation = smoothed_correlation-np.mean(smoothed_correlation)
            # smoothed_reactivation_prob = smoothed_reactivation_prob-np.mean(smoothed_reactivation_prob)

            corr = signal.correlate(smoothed_correlation, smoothed_reactivation_prob)
            corr = corr/np.max(corr)
            num_trials = int((len(corr)-1)/2)
            xcorr_correlation[i, :] = corr[num_trials-maxlags:num_trials+maxlags+1]

        mean = xcorr_correlation.mean(axis=0)
        plt.plot(range(-maxlags, maxlags + 1), mean, '-', c=m_colors[mouse], alpha=.2, linewidth=2)
        xcorr_correlation_all[mouse, :] = mean
    mean = xcorr_correlation_all.mean(axis=0)
    sem_plus = mean + stats.sem(xcorr_correlation_all, axis=0)
    sem_minus = mean - stats.sem(xcorr_correlation_all, axis=0)
    plt.plot(range(-maxlags, maxlags + 1), mean, '-', c='k', linewidth=3)
    plt.fill_between(range(-maxlags, maxlags + 1), sem_plus, sem_minus, alpha=0.2, color='k', lw=0)
    plt.axvline(0, color='black', linestyle='--', linewidth=1, snap=False)
    plt.ylabel('Cross correlation between\nstimulus orthogonalization\nand reactivation probability')
    plt.xlabel('Shift in trial')
    plt.xticks([-10, -5, 0, 5, 10])
    plt.ylim(0, 1)
    sns.despine()

    plt.savefig(paths['base_path'] + '/NN/plots/activity_reactivation_correlation.png', bbox_inches='tight', dpi=200)
    plt.savefig(paths['base_path'] + '/NN/plots/activity_reactivation_correlation.pdf', bbox_inches='tight', dpi=200, transparent=True)
    plt.close()


def reactivation_spatial(mice, sample_dates):
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(wspace=.45)
    m_colors = ['b', 'purple', 'darkorange', 'green']
    x = [1, 2, 3, 4]
    area_li_cue_all = []
    area_por_cue_all = []
    area_p_cue_all = []
    area_lm_cue_all = []
    area_li_reactivation_all = []
    area_por_reactivation_all = []
    area_p_reactivation_all = []
    area_lm_reactivation_all = []
    for mouse in range(0, len(mice)):
        paths = preprocess.create_folders(mice[mouse], sample_dates[mouse])
        reactivation_spatial = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/reactivation_spatial.npy', allow_pickle=True)
        area_li_cue = []
        area_por_cue = []
        area_p_cue = []
        area_lm_cue = []
        area_li_reactivation = []
        area_por_reactivation = []
        area_p_reactivation = []
        area_lm_reactivation = []
        for i in range(0, len(reactivation_spatial[0])):
            area_li_cue.append(reactivation_spatial[0][i])
            area_por_cue.append(reactivation_spatial[1][i])
            area_p_cue.append(reactivation_spatial[2][i])
            area_lm_cue.append(reactivation_spatial[3][i])
            area_li_reactivation.append(reactivation_spatial[4][i])
            area_por_reactivation.append(reactivation_spatial[5][i])
            area_p_reactivation.append(reactivation_spatial[6][i])
            area_lm_reactivation.append(reactivation_spatial[7][i])
        area_li_cue_all.append(np.nanmean(area_li_cue))
        area_por_cue_all.append(np.nanmean(area_por_cue))
        area_p_cue_all.append(np.nanmean(area_p_cue))
        area_lm_cue_all.append(np.nanmean(area_lm_cue))
        area_li_reactivation_all.append(np.nanmean(area_li_reactivation))
        area_por_reactivation_all.append(np.nanmean(area_por_reactivation))
        area_p_reactivation_all.append(np.nanmean(area_p_reactivation))
        area_lm_reactivation_all.append(np.nanmean(area_lm_reactivation))
        plt.subplot(2, 2, 1)
        plt.errorbar(x[0], np.nanmean(area_li_cue), yerr=0, c=m_colors[mouse], marker='o', mfc='none', mec=m_colors[mouse],
                     ms=5, alpha=.3)
        plt.errorbar(x[1], np.nanmean(area_por_cue), yerr=0, c=m_colors[mouse], marker='o', mfc='none', mec=m_colors[mouse],
                     ms=5, alpha=.3)
        plt.errorbar(x[2], np.nanmean(area_p_cue), yerr=0, c=m_colors[mouse], marker='o', mfc='none', mec=m_colors[mouse],
                     ms=5, alpha=.3)
        plt.errorbar(x[3], np.nanmean(area_lm_cue), yerr=0, c=m_colors[mouse], marker='o', mfc='none', mec=m_colors[mouse],
                     ms=5, alpha=.3)
        plt.subplot(2, 2, 2)
        plt.errorbar(x[0], np.nanmean(area_li_reactivation), yerr=0, c=m_colors[mouse], marker='o', mfc='none', mec=m_colors[mouse],
                     ms=5, alpha=.3)
        plt.errorbar(x[1], np.nanmean(area_por_reactivation), yerr=0, c=m_colors[mouse], marker='o', mfc='none',
                     mec=m_colors[mouse],
                     ms=5, alpha=.3)
        plt.errorbar(x[2], np.nanmean(area_p_reactivation), yerr=0, c=m_colors[mouse], marker='o', mfc='none', mec=m_colors[mouse],
                     ms=5, alpha=.3)
        plt.errorbar(x[3], np.nanmean(area_lm_reactivation), yerr=0, c=m_colors[mouse], marker='o', mfc='none', mec=m_colors[mouse],
                     ms=5, alpha=.3)

    print(stats.f_oneway(area_li_cue_all, area_por_cue_all, area_p_cue_all, area_lm_cue_all))
    print(pairwise_tukeyhsd(np.concatenate([area_li_cue_all, area_por_cue_all, area_p_cue_all, area_lm_cue_all], axis=0),
                            ['li', 'li', 'li', 'li', 'por', 'por', 'por', 'por', 'p', 'p', 'p', 'p', 'lm', 'lm', 'lm',
                             'lm'], alpha=0.05))
    print(stats.f_oneway(area_li_reactivation_all, area_por_reactivation_all, area_p_reactivation_all, area_lm_reactivation_all))
    print(
        pairwise_tukeyhsd(np.concatenate([area_li_reactivation_all, area_por_reactivation_all, area_p_reactivation_all, area_lm_reactivation_all], axis=0),
                          ['li', 'li', 'li', 'li', 'por', 'por', 'por', 'por', 'p', 'p', 'p', 'p', 'lm', 'lm', 'lm',
                           'lm'], alpha=0.05))
    plt.subplot(2, 2, 1)
    y1 = np.mean(area_li_cue_all)
    y1_err = stats.sem(area_li_cue_all)
    plt.errorbar(1.2, y1, yerr=y1_err, c='k', linewidth=2, marker='o', mfc='k', mec='k',
                 ms=7, mew=0, zorder=100)
    y2 = np.mean(area_por_cue_all)
    y2_err = stats.sem(area_por_cue_all)
    plt.errorbar(2.2, y2, yerr=y2_err, c='k', linewidth=2, marker='o', mfc='k', mec='k',
                 ms=7, mew=0, zorder=100)
    y3 = np.mean(area_p_cue_all)
    y3_err = stats.sem(area_p_cue_all)
    plt.errorbar(3.2, y3, yerr=y3_err, c='k', linewidth=2, marker='o', mfc='k', mec='k',
                 ms=7, mew=0, zorder=100)
    y4 = np.mean(area_lm_cue_all)
    y4_err = stats.sem(area_lm_cue_all)
    plt.errorbar(4.2, y4, yerr=y4_err, c='k', linewidth=2, marker='o', mfc='k', mec='k',
                 ms=7, mew=0, zorder=100)
    plt.xlim(.5, 4.5)
    plt.ylim(0, .7)
    plt.xticks([1, 2, 3, 4], ['LI', 'POR', 'P', 'LM'])
    plt.ylabel('Stimulus activity\n(normalized deconvolved\n$\mathregular{Ca^{2+}}$transient $\mathregular{s^{-1}}$)')

    plt.subplot(2, 2, 2)
    y1 = np.mean(area_li_reactivation_all)
    y1_err = stats.sem(area_li_reactivation_all)
    plt.errorbar(1.2, y1, yerr=y1_err, c='k', linewidth=2, marker='o', mfc='k', mec='k',
                 ms=7, mew=0, zorder=100)
    y2 = np.mean(area_por_reactivation_all)
    y2_err = stats.sem(area_por_reactivation_all)
    plt.errorbar(2.2, y2, yerr=y2_err, c='k', linewidth=2, marker='o', mfc='k', mec='k',
                 ms=7, mew=0, zorder=100)
    y3 = np.mean(area_p_reactivation_all)
    y3_err = stats.sem(area_p_reactivation_all)
    plt.errorbar(3.2, y3, yerr=y3_err, c='k', linewidth=2, marker='o', mfc='k', mec='k',
                 ms=7, mew=0, zorder=100)
    y4 = np.mean(area_lm_reactivation_all)
    y4_err = stats.sem(area_lm_reactivation_all)
    plt.errorbar(4.2, y4, yerr=y4_err, c='k', linewidth=2, marker='o', mfc='k', mec='k',
                 ms=7, mew=0, zorder=100)
    plt.xlim(.5, 4.5)
    plt.ylim(0, .7)
    plt.xticks([1, 2, 3, 4], ['LI', 'POR', 'P', 'LM'])
    plt.ylabel('Reactivation activity\n(normalized deconvolved\n$\mathregular{Ca^{2+}}$transient $\mathregular{s^{-1}}$)')

    sns.despine()
    plt.savefig(paths['base_path'] + '/NN/plots/reactivation_spatial.png', bbox_inches='tight', dpi=200)
    plt.savefig(paths['base_path'] + '/NN/plots/reactivation_spatial.pdf', bbox_inches='tight', dpi=200,
                transparent=True)
    plt.close()


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


def reactivation_cue_vector(mice, sample_dates):
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(9, 6))
    plt.subplots_adjust(wspace=.3)
    x_label = list(range(0, 60))
    paths = preprocess.create_folders(mice[0], sample_dates[0])
    reactivation_cue_pca_vec = np.load(paths['base_path'] + paths['mouse'] +
                                    '/data_across_days/reactivation_cue_vector.npy', allow_pickle=True)
    all_s1 = np.zeros((len(mice), len(reactivation_cue_pca_vec[0][0])))
    all_s1r = np.zeros((len(mice), len(reactivation_cue_pca_vec[0][0])))
    all_s2 = np.zeros((len(mice), len(reactivation_cue_pca_vec[0][0])))
    all_s2r = np.zeros((len(mice), len(reactivation_cue_pca_vec[0][0])))
    all_s1b = []
    all_s2b = []
    for mouse in range(0, len(mice)):
        paths = preprocess.create_folders(mice[mouse], sample_dates[mouse])
        reactivation_cue_pca_vec = np.load(paths['base_path'] + paths['mouse'] +
                                           '/data_across_days/reactivation_cue_vector.npy', allow_pickle=True)
        temp_s1 = np.zeros((len(reactivation_cue_pca_vec[0]), len(reactivation_cue_pca_vec[0][0])))
        temp_s1r = np.zeros((len(reactivation_cue_pca_vec[0]), len(reactivation_cue_pca_vec[0][0])))
        temp_s2 = np.zeros((len(reactivation_cue_pca_vec[0]), len(reactivation_cue_pca_vec[0][0])))
        temp_s2r = np.zeros((len(reactivation_cue_pca_vec[0]), len(reactivation_cue_pca_vec[0][0])))
        temp_s1b = []
        temp_s2b = []
        for i in range(0, len(reactivation_cue_pca_vec[0])):
            temp_s1[i, :] = reactivation_cue_pca_vec[0][i]
            temp_s1r[i, :] = reactivation_cue_pca_vec[1][i]
            temp_s1b.append(reactivation_cue_pca_vec[4][i])
            temp_s2[i, :] = reactivation_cue_pca_vec[2][i]
            temp_s2r[i, :] = reactivation_cue_pca_vec[3][i]
            temp_s2b.append(reactivation_cue_pca_vec[5][i])
        y_s1 = np.nanmean(temp_s1, axis=0)
        y_s1r = np.nanmean(temp_s1r, axis=0)
        y_s2 = np.nanmean(temp_s2, axis=0)
        y_s2r = np.nanmean(temp_s2r, axis=0)
        y_s1b = np.nanmean(temp_s1b, axis=0)
        y_s2b = np.nanmean(temp_s2b, axis=0)
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
        all_s1b.append(y_s1b)
        all_s2b.append(y_s2b)

    print(stats.ttest_rel(all_s2[:, 0:1], all_s2r[:, 59:60])[1])

    plt.subplot(2, 2, 1)
    plt.ylim(-1.2, .1)
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
    plt.xlim(-1, 60)
    plt.xticks([0, 19, 39, 59], ['1', '20', '40', '60'])
    plt.subplot(2, 2, 2)
    plt.ylim(-1.2, .1)
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
    plt.xlim(-1, 60)
    plt.xticks([0, 19, 39, 59], ['1', '20', '40', '60'])
    sns.despine()
    plt.savefig(paths['base_path'] + '/NN/plots/reactivation_cue_vector.png', bbox_inches='tight', dpi=200)
    plt.savefig(paths['base_path'] + '/NN/plots/reactivation_cue_vector.pdf', bbox_inches='tight', dpi=200, transparent=True)
    plt.close()


def reactivation_cue_vector_evolve(mice, sample_dates):
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(9, 6))
    plt.subplots_adjust(wspace=.3)
    x_label = list(range(0, 60))
    paths = preprocess.create_folders(mice[0], sample_dates[0])
    reactivation_cue_pca_vec = np.load(paths['base_path'] + paths['mouse'] +
                                    '/data_across_days/reactivation_cue_vector_evolve.npy', allow_pickle=True)
    all_s1 = np.zeros((len(mice), len(reactivation_cue_pca_vec[0][0])))
    all_s1r = np.zeros((len(mice), len(reactivation_cue_pca_vec[0][0])))
    all_s2 = np.zeros((len(mice), len(reactivation_cue_pca_vec[0][0])))
    all_s2r = np.zeros((len(mice), len(reactivation_cue_pca_vec[0][0])))
    for mouse in range(0, len(mice)):
        paths = preprocess.create_folders(mice[mouse], sample_dates[mouse])
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

    print(stats.ttest_rel(np.mean(all_s1, axis=1), np.mean(all_s1r, axis=1))[1])
    print(stats.ttest_rel(np.mean(all_s2, axis=1), np.mean(all_s2r, axis=1))[1])

    plt.subplot(2, 2, 1)
    plt.ylim(-1.2, .05)
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
    plt.xlim(0, 60)
    plt.xticks([0, 19, 39, 59], ['1', '20', '40', '60'])
    plt.subplot(2, 2, 2)
    plt.ylim(-1.2, .05)
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
    plt.xlim(0, 60)
    plt.xticks([0, 19, 39, 59], ['1', '20', '40', '60'])
    sns.despine()
    plt.savefig(paths['base_path'] + '/NN/plots/reactivation_cue_vector_evolve.png', bbox_inches='tight', dpi=200)
    plt.savefig(paths['base_path'] + '/NN/plots/reactivation_cue_vector_evolve.pdf', bbox_inches='tight', dpi=200,
                transparent=True)
    plt.close()


def activity_across_trials_evolve(mice, sample_dates):
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(9, 6))
    plt.subplots_adjust(wspace=.33)
    m_colors = ['b', 'purple', 'darkorange', 'green']

    plt.subplot(2, 2, 1)
    activity_all = np.empty((len(mice), 128)) * np.nan
    activity_all_t = np.empty((len(mice), 128)) * np.nan
    for mouse in range(0, len(mice)):
        paths = preprocess.create_folders(mice[mouse], sample_dates[mouse])
        activity_data = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/activity_evolve.npy',
                               allow_pickle=True)
        activity_data_t = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/activity.npy',
                                allow_pickle=True)
        activity = np.empty((len(activity_data[0]), 128)) * np.nan
        activity_t = np.empty((len(activity_data[0]), 128)) * np.nan
        for i in range(0, len(activity_data[0])):
            smoothed_activity = np.array(pd.DataFrame(activity_data[0][i]).rolling(3, min_periods=1, center=True).mean())
            smoothed_activity = np.concatenate(smoothed_activity, axis=0)
            smoothed_activity_t = np.array(
                pd.DataFrame(activity_data_t[0][i]).rolling(3, min_periods=1, center=True).mean())
            smoothed_activity_t = np.concatenate(smoothed_activity_t, axis=0)
            activity[i, 0:len(smoothed_activity)] = smoothed_activity
            activity_t[i, 0:len(smoothed_activity_t)] = smoothed_activity_t
        activity = np.nanmean(activity, axis=0)
        activity_all[mouse, :] = activity
        activity_t = np.nanmean(activity_t, axis=0)
        activity_all_t[mouse, :] = activity_t
        x = range(1, len(activity)+1)
        # plt.plot(x, activity, '--', c=m_colors[mouse], alpha=1)
        # plt.plot(x, activity_t, c=m_colors[mouse], alpha=1)

    print(stats.ttest_rel(np.mean(activity_all_t[:, 0:120], axis=1), np.mean(activity_all[:, 0:120], axis=1))[1])

    mean = np.nanmean(activity_all_t, axis=0)
    sem_plus = mean + stats.sem(activity_all_t, axis=0, nan_policy='omit')
    sem_minus = mean - stats.sem(activity_all_t, axis=0, nan_policy='omit')
    plt.plot(x, mean, c='k', linewidth=3)
    plt.fill_between(x, sem_plus, sem_minus, alpha=0.2, color='k', lw=0)
    mean = np.nanmean(activity_all, axis=0)
    sem_plus = mean + stats.sem(activity_all, axis=0, nan_policy='omit')
    sem_minus = mean - stats.sem(activity_all, axis=0, nan_policy='omit')
    plt.plot(x, mean, c=[.6, 0, .6], linewidth=3)
    plt.fill_between(x, sem_plus, sem_minus, alpha=0.2, color=[.6, 0, .6], lw=0)
    plt.ylabel('Correlation between\nstimulus 1 and stimulus 2')
    plt.xlabel('Trial number')
    # plt.ylim(-.1, .6)
    plt.xlim(1, 120)
    plt.xticks([1, 20, 40, 60, 80, 100, 120])
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
    label_1 = mlines.Line2D([], [], color='k', linewidth=2, label='Real data')
    label_2 = mlines.Line2D([], [], color=[.6, 0, .6], linewidth=2, label='Modeled data')
    plt.legend(handles=[label_1, label_2], frameon=False)
    sns.despine()

    plt.subplot(2, 2, 2)
    activity_all = np.empty((len(mice), 128)) * np.nan
    for mouse in range(0, len(mice)):
        paths = preprocess.create_folders(mice[mouse], sample_dates[mouse])
        activity_data = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/activity_evolve.npy',
                                allow_pickle=True)
        activity = np.empty((len(activity_data[0]), 128)) * np.nan
        for i in range(0, len(activity_data[0])):
            smoothed_activity = np.array(
                pd.DataFrame(activity_data[2][i]).rolling(3, min_periods=1, center=True).mean())
            smoothed_activity = np.concatenate(smoothed_activity, axis=0)
            activity[i, 0:len(smoothed_activity)] = smoothed_activity
        activity = np.nanmean(activity, axis=0)
        activity_all[mouse, :] = activity
        x = range(1, len(activity) + 1)
        plt.plot(x, activity, c=m_colors[mouse], alpha=.2)
    mean = np.nanmean(activity_all, axis=0)
    sem_plus = mean + stats.sem(activity_all, axis=0, nan_policy='omit')
    sem_minus = mean - stats.sem(activity_all, axis=0, nan_policy='omit')
    plt.plot(x, mean, c='k', linewidth=3)
    plt.fill_between(x, sem_plus, sem_minus, alpha=0.2, color='k', lw=0)
    plt.ylabel('Stimulus activity\n(Normalized $\mathregular{Ca^{2+}}$transient $\mathregular{s^{-1}}$)')
    plt.xlabel('Trial number')
    # plt.ylim(0, .07)
    plt.xlim(1, 120)
    plt.xticks([1, 20, 40, 60, 80, 100, 120])
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
    sns.despine()

    plt.subplot(2, 2, 3)
    activity_all = np.empty((len(mice), 128)) * np.nan
    for mouse in range(0, len(mice)):
        paths = preprocess.create_folders(mice[mouse], sample_dates[mouse])
        activity_data = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/activity_evolve.npy',
                                allow_pickle=True)
        activity = np.empty((len(activity_data[0]), 128)) * np.nan
        for i in range(0, len(activity_data[0])):
            smoothed_activity = np.array(
                pd.DataFrame(activity_data[3][i]).rolling(3, min_periods=1, center=True).mean())
            smoothed_activity = np.concatenate(smoothed_activity, axis=0)
            activity[i, 0:len(smoothed_activity)] = smoothed_activity
        activity = np.nanmean(activity, axis=0)
        activity_all[mouse, :] = activity
        x = range(1, len(activity) + 1)
        plt.plot(x, activity, c=m_colors[mouse], alpha=.2)
    mean = np.nanmean(activity_all, axis=0)
    sem_plus = mean + stats.sem(activity_all, axis=0, nan_policy='omit')
    sem_minus = mean - stats.sem(activity_all, axis=0, nan_policy='omit')
    plt.plot(x, mean, c='mediumseagreen', linewidth=3)
    plt.fill_between(x, sem_plus, sem_minus, alpha=0.2, color='mediumseagreen', lw=0)
    plt.ylabel('Correlation between\nstimulus 1 and stimulus 1')
    plt.xlabel('Trial number')
    # plt.ylim(0, .8)
    plt.xlim(1, 62)
    plt.xticks([1, 20, 40, 60])
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
    sns.despine()

    plt.subplot(2, 2, 4)
    activity_all = np.empty((len(mice), 128)) * np.nan
    for mouse in range(0, len(mice)):
        paths = preprocess.create_folders(mice[mouse], sample_dates[mouse])
        activity_data = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/activity_evolve.npy',
                                allow_pickle=True)
        activity = np.empty((len(activity_data[0]), 128)) * np.nan
        for i in range(0, len(activity_data[0])):
            smoothed_activity = np.array(
                pd.DataFrame(activity_data[4][i]).rolling(3, min_periods=1, center=True).mean())
            smoothed_activity = np.concatenate(smoothed_activity, axis=0)
            activity[i, 0:len(smoothed_activity)] = smoothed_activity
        activity = np.nanmean(activity, axis=0)
        activity_all[mouse, :] = activity
        x = range(1, len(activity) + 1)
        plt.plot(x, activity, c=m_colors[mouse], alpha=.2)
    mean = np.nanmean(activity_all, axis=0)
    sem_plus = mean + stats.sem(activity_all, axis=0, nan_policy='omit')
    sem_minus = mean - stats.sem(activity_all, axis=0, nan_policy='omit')
    plt.plot(x, mean, c='salmon', linewidth=3)
    plt.fill_between(x, sem_plus, sem_minus, alpha=0.2, color='salmon', lw=0)
    plt.ylabel('Correlation between\nstimulus 2 and stimulus 2')
    plt.xlabel('Trial number')
    # plt.ylim(0, .8)
    plt.xlim(1, 62)
    plt.xticks([1, 20, 40, 60])
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
    sns.despine()
    plt.savefig(paths['base_path'] + '/NN/plots/activity_across_trials_evolve.png', bbox_inches='tight', dpi=200)
    plt.savefig(paths['base_path'] + '/NN/plots/activity_across_trials_evolve.pdf', bbox_inches='tight', dpi=200, transparent=True)
    plt.close()


def activity_across_trials_evolve_grouped(mice, sample_dates):
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
        paths = preprocess.create_folders(mice[mouse], sample_dates[mouse])
        activity_data = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/activity_evolve_grouped.npy',
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
    # mean = np.nanmean(activity_same_all, axis=0)
    # sem_plus = mean + stats.sem(activity_same_all, axis=0, nan_policy='omit')
    # sem_minus = mean - stats.sem(activity_same_all, axis=0, nan_policy='omit')
    # plt.plot(x, mean, c='k', linewidth=3)
    # plt.fill_between(x, sem_plus, sem_minus, alpha=0.2, color='k', lw=0)

    mask = ~np.isnan(np.concatenate([x, x, x, x])) & ~np.isnan(np.concatenate(activity_decrease_all))
    slope, intercept, r_value, p_value_dec, std_err = stats.linregress(
        np.concatenate([x, x, x, x])[mask], np.concatenate(activity_decrease_all)[mask])
    print(r_value)
    slope, intercept, r_value, p_value_inc, std_err = stats.linregress(
        np.concatenate([x, x, x, x])[mask], np.concatenate(activity_increase_all)[mask])
    print(r_value)
    anova_results = []
    anova_results.append(p_value_dec)
    anova_results.append(p_value_inc)
    anova_results_corrected = multipletests(anova_results, alpha=.05, method='holm')
    return [anova_results, anova_results_corrected[1]]

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
    plt.xlim(1, 120)
    plt.xticks([1, 20, 40, 60, 80, 100, 120])
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
    sns.despine()

    plt.subplot(2, 2, 2)
    activity_same_all = np.empty((len(mice), 128)) * np.nan
    activity_increase_all = np.empty((len(mice), 128)) * np.nan
    activity_decrease_all = np.empty((len(mice), 128)) * np.nan
    for mouse in range(0, len(mice)):
        paths = preprocess.create_folders(mice[mouse], sample_dates[mouse])
        activity_data = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/activity_evolve_grouped.npy',
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
    plt.xlim(1, 123)
    plt.xticks([1, 20, 40, 60, 80, 100, 120])
    plt.ylim(0, .12)
    label_1 = mlines.Line2D([], [], color='k', linewidth=2, label='No change cells')
    label_2 = mlines.Line2D([], [], color='darkred', linewidth=2, label='Increase cells')
    label_3 = mlines.Line2D([], [], color='darkblue', linewidth=2, label='Decrease cells')
    plt.legend(handles=[label_1, label_2, label_3], frameon=False)
    sns.despine()

    plt.savefig(paths['base_path'] + '/NN/plots/activity_across_trials_evolve_grouped.png', bbox_inches='tight', dpi=200)
    plt.savefig(paths['base_path'] + '/NN/plots/activity_across_trials_evolve_grouped.pdf', bbox_inches='tight', dpi=200, transparent=True)
    plt.close()


def activity_across_trials_evolve_grouped_decrease(mice, sample_dates):
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
        paths = preprocess.create_folders(mice[mouse], sample_dates[mouse])
        activity_data = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/activity_grouped_no_change_evolve.npy',
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
    plt.ylabel('Modeled stimulus activity\n(Normalized $\mathregular{Ca^{2+}}$transient $\mathregular{s^{-1}}$)')
    plt.xlabel('Trial number')
    plt.ylim(0, .13)
    plt.xlim(1, 60)
    plt.xticks([1, 20, 40, 60])
    plt.yticks([0, .05, .1])
    sns.despine()
    label_1 = mlines.Line2D([], [], color='mediumseagreen', linewidth=2, label='S1 no change cells, S1 trials')
    label_2 = mlines.Line2D([], [], color='darkgreen', linewidth=2, label='S1  no change cells, S2 trials')
    label_3 = mlines.Line2D([], [], color='salmon', linewidth=2, label='S2  no change cells, S2 trials')
    label_4 = mlines.Line2D([], [], color='darkred', linewidth=2, label='S2  no change cells, S1 trials')
    plt.legend(handles=[label_1, label_2, label_3, label_4], frameon=False)

    plt.savefig(paths['base_path'] + '/NN/plots/activity_across_trials_evolve_grouped_no_change.png', bbox_inches='tight', dpi=200)
    plt.savefig(paths['base_path'] + '/NN/plots/activity_across_trials_evolve_grouped_no_change.pdf', bbox_inches='tight', dpi=200, transparent=True)
    plt.close()


def reactivation_cue_vector_cross_evolve(mice, sample_dates):
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(9, 6))
    plt.subplots_adjust(wspace=.35)
    m_colors = ['b', 'purple', 'darkorange', 'green']
    maxlags = 10
    all_s1 = np.zeros((len(mice), (maxlags * 2) + 1))
    all_s2 = np.zeros((len(mice), (maxlags * 2) + 1))
    for mouse in range(0, len(mice)):
        paths = preprocess.create_folders(mice[mouse], sample_dates[mouse])
        reactivation_cue_pca_vec = np.load(paths['base_path'] + paths['mouse'] +
                                           '/data_across_days/reactivation_cue_vector_evolve.npy', allow_pickle=True)
        temp_s1 = np.zeros((len(reactivation_cue_pca_vec[0]), (maxlags * 2) + 1))
        temp_s2 = np.zeros((len(reactivation_cue_pca_vec[0]), (maxlags * 2) + 1))
        for i in range(0, len(reactivation_cue_pca_vec[0])):

            sos = signal.butter(2, .1 / 60 / 2, btype='highpass', output='sos', fs=1 / 60)
            s1_temp = np.array(reactivation_cue_pca_vec[2][i])
            s1_temp = signal.sosfilt(sos, s1_temp)
            s1_temp = s1_temp[5:len(s1_temp)]
            s1r_temp = np.array(reactivation_cue_pca_vec[0][i])
            s1r_temp = signal.sosfilt(sos, s1r_temp)
            s1r_temp = s1r_temp[5:len(s1r_temp)]
            s2_temp = np.array(reactivation_cue_pca_vec[3][i])
            s2_temp = signal.sosfilt(sos, s2_temp)
            s2_temp = s2_temp[5:len(s2_temp)]
            s2r_temp = np.array(reactivation_cue_pca_vec[1][i])
            s2r_temp = signal.sosfilt(sos, s2r_temp)
            s2r_temp = s2r_temp[5:len(s2r_temp)]
            s1 = s1_temp - np.mean(s1_temp)
            s1r = s1r_temp - np.mean(s1r_temp)
            s2 = s2_temp - np.mean(s2_temp)
            s2r = s2r_temp - np.mean(s2r_temp)

            # s1 = np.array(reactivation_cue_pca_vec[2][i]) - np.mean(np.array(reactivation_cue_pca_vec[2][i]))
            # s1r = np.array(reactivation_cue_pca_vec[0][i]) - np.mean(np.array(reactivation_cue_pca_vec[0][i]))
            # s2 = np.array(reactivation_cue_pca_vec[3][i]) - np.mean(np.array(reactivation_cue_pca_vec[3][i]))
            # s2r = np.array(reactivation_cue_pca_vec[1][i]) - np.mean(np.array(reactivation_cue_pca_vec[1][i]))

            corr_s1 = signal.correlate(s1, s1r)
            corr_s1 = corr_s1 / np.max(corr_s1)
            num_trials = int((len(corr_s1) - 1) / 2)
            temp_s1[i, :] = corr_s1[num_trials - maxlags:num_trials + maxlags + 1]

            corr_s2 = signal.correlate(s2, s2r)
            corr_s2 = corr_s2 / np.max(corr_s2)
            num_trials = int((len(corr_s2) - 1) / 2)
            temp_s2[i, :] = corr_s2[num_trials - maxlags:num_trials + maxlags + 1]
        y_s1 = np.nanmean(temp_s1, axis=0)
        y_s2 = np.nanmean(temp_s2, axis=0)
        plt.subplot(2, 2, 1)
        plt.plot(range(-maxlags, maxlags + 1), y_s1, '-', c=m_colors[mouse], alpha=.2, linewidth=2)
        plt.subplot(2, 2, 2)
        plt.plot(range(-maxlags, maxlags + 1), y_s2, '-', c=m_colors[mouse], alpha=.2, linewidth=2)
        all_s1[mouse, :] = y_s1
        all_s2[mouse, :] = y_s2
    plt.subplot(2, 2, 1)
    mean = all_s1.mean(axis=0)
    sem_plus = mean + stats.sem(all_s1, axis=0)
    sem_minus = mean - stats.sem(all_s1, axis=0)
    plt.plot(range(-maxlags, maxlags + 1), mean, '-', c='mediumseagreen', linewidth=3)
    plt.fill_between(range(-maxlags, maxlags + 1), sem_plus, sem_minus, alpha=0.2, color='mediumseagreen', lw=0)
    plt.axvline(0, color='black', linestyle='--', linewidth=1, snap=False)
    plt.ylabel('Cross correlation\nbetween real and modeled\nstimulus 1 responses')
    plt.xlabel('Shift in trial')
    plt.ylim(-.2, 1.05)
    plt.yticks([0, .2, .4, .6, .8, 1])
    plt.subplot(2, 2, 2)
    mean = all_s2.mean(axis=0)
    sem_plus = mean + stats.sem(all_s2, axis=0)
    sem_minus = mean - stats.sem(all_s2, axis=0)
    plt.plot(range(-maxlags, maxlags + 1), mean, '-', c='salmon', linewidth=3)
    plt.fill_between(range(-maxlags, maxlags + 1), sem_plus, sem_minus, alpha=0.2, color='salmon', lw=0)
    plt.axvline(0, color='black', linestyle='--', linewidth=1, snap=False)
    plt.ylabel('Cross correlation between\nreal and modeled\nstimulus 2 responses')
    plt.xlabel('Shift in trial')
    plt.ylim(-.2, 1.05)
    plt.yticks([0, .2, .4, .6, .8, 1])
    sns.despine()
    plt.savefig(paths['base_path'] + '/NN/plots/reactivation_cue_vector_evolve_cross.png', bbox_inches='tight', dpi=200)
    plt.savefig(paths['base_path'] + '/NN/plots/reactivation_cue_vector_evolve_cross.pdf', bbox_inches='tight', dpi=200,
                transparent=True)
    plt.close()


def reactivation_cue_vector_evolve_parametric(mice, sample_dates):
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(9, 6))
    plt.subplots_adjust(wspace=.3)
    m_colors = ['b', 'purple', 'darkorange', 'green']

    p_values = np.array(range(0, 105, 5))/100
    total_s1 = np.zeros((len(mice), len(p_values)))
    total_s2 = np.zeros((len(mice), len(p_values)))
    for p in range(0, len(p_values)):
        for mouse in range(0, len(mice)):
            paths = preprocess.create_folders(mice[mouse], sample_dates[mouse])
            reactivation_cue_pca_vec = np.load(paths['base_path'] + paths['mouse'] +
                                               '/data_across_days/reactivation_cue_vector_evolve_parametric.npy',
                                               allow_pickle=True)
            temp_s1 = []
            temp_s2 = []
            for i in range(0, len(reactivation_cue_pca_vec[0])):
                temp_s1.append(np.mean(np.absolute(np.array(reactivation_cue_pca_vec[2][i][p]) -
                                                   np.array(reactivation_cue_pca_vec[0][i][p]))))
                temp_s2.append(np.mean(np.absolute(np.array(reactivation_cue_pca_vec[3][i][p]) -
                                                   np.array(reactivation_cue_pca_vec[1][i][p]))))
            total_s1[mouse, p] = np.mean(temp_s1)
            total_s2[mouse, p] = np.mean(temp_s2)
    plt.subplot(2, 2, 1)
    mean = np.mean(total_s1, axis=0)
    sem_plus = mean + stats.sem(total_s1, axis=0)
    sem_minus = mean - stats.sem(total_s1, axis=0)
    plt.plot(p_values, mean, '-', c='mediumseagreen', linewidth=3)
    plt.fill_between(p_values, sem_plus, sem_minus, alpha=0.2, color='mediumseagreen', lw=0)
    for i in range(0, len(mice)):
        plt.plot(p_values, total_s1[i, :], '-', c=m_colors[i], alpha=.2, linewidth=2)
    plt.ylabel('Error in stimulus 1 model\n(mean absolute difference)')
    plt.xlabel('Plasticity value (p)')
    plt.xlim(0, 1)
    plt.xticks([0, .15, .3, .45, .6, .75, .9])
    plt.subplot(2, 2, 2)
    mean = np.mean(total_s2, axis=0)
    sem_plus = mean + stats.sem(total_s2, axis=0)
    sem_minus = mean - stats.sem(total_s2, axis=0)
    plt.plot(p_values, mean, '-', c='salmon', linewidth=3)
    plt.fill_between(p_values, sem_plus, sem_minus, alpha=0.2, color='salmon', lw=0)
    for i in range(0, len(mice)):
        plt.plot(p_values, total_s2[i, :], '-', c=m_colors[i], alpha=.2, linewidth=2)
    plt.ylabel('Error in stimulus 2 model\n(mean absolute difference)')
    plt.xlabel('Plasticity value (p)')
    plt.xlim(0, 1)
    plt.xticks([0, .15, .3, .45, .6, .75, .9])
    sns.despine()
    plt.savefig(paths['base_path'] + '/NN/plots/reactivation_cue_vector_evolve_parametric.png', bbox_inches='tight', dpi=200)
    plt.savefig(paths['base_path'] + '/NN/plots/reactivation_cue_vector_evolve_parametric.pdf', bbox_inches='tight', dpi=200,
                transparent=True)
    plt.close()


def reactivation_cue_scale(mice, sample_dates):
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(3, 6))
    m_colors = ['b', 'purple', 'darkorange', 'green']
    scale_all = []
    for mouse in range(0, len(mice)):
        paths = preprocess.create_folders(mice[mouse], sample_dates[mouse])
        reactivation_scale = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/reactivation_cue_scale.npy', allow_pickle=True)
        scale = []
        for i in range(0, len(reactivation_scale[0])):
            scale.append(reactivation_scale[0][i]/reactivation_scale[1][i])
        scale_all.append(np.mean(scale))
        plt.subplot(2, 2, 1)
        plt.errorbar(1, np.mean(scale), yerr=0, c=m_colors[mouse], marker='o', mfc='none', mec=m_colors[mouse],
                     ms=5, alpha=.3)
    print(np.mean(scale_all))
    plt.subplot(2, 2, 1)
    y1 = np.mean(scale_all)
    y1_err = stats.sem(scale_all)
    plt.errorbar(1.2, y1, yerr=y1_err, c='k', linewidth=2, marker='o', mfc='k', mec='k',
                 ms=7, mew=0, zorder=100)
    plt.xlim(.5, 1.5)
    plt.ylim(0, 2)
    plt.xticks([])
    plt.ylabel('Stimulus / reactivation\nactivity')
    plt.axhline(1, color='black', linestyle='--', linewidth=1, snap=False)
    sns.despine()
    plt.savefig(paths['base_path'] + '/NN/plots/reactivation_cue_scale.png', bbox_inches='tight', dpi=200)
    plt.savefig(paths['base_path'] + '/NN/plots/reactivation_cue_scale.pdf', bbox_inches='tight', dpi=200,
                transparent=True)
    plt.close()


def reactivation_influence(mice, sample_dates):
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(7, 6))
    m_colors = ['b', 'purple', 'darkorange', 'green']
    x = [1, 2, 3, 4]
    decrease_cs_1_all = []
    increase_cs_1_all = []
    no_change_cs_1_all = []
    decrease_cs_2_all = []
    increase_cs_2_all = []
    no_change_cs_2_all = []
    for mouse in range(0, len(mice)):
        paths = preprocess.create_folders(mice[mouse], sample_dates[mouse])
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

    print(stats.f_oneway(decrease_cs_1_all, no_change_cs_1_all, increase_cs_1_all))
    print(
        pairwise_tukeyhsd(np.concatenate([decrease_cs_1_all, no_change_cs_1_all, increase_cs_1_all], axis=0),
                          ['d', 'd', 'd', 'd', 'n', 'n', 'n', 'n', 'i', 'i', 'i', 'i'], alpha=0.05))
    print(stats.f_oneway(decrease_cs_2_all, no_change_cs_2_all, increase_cs_2_all))
    print(
        pairwise_tukeyhsd(np.concatenate([decrease_cs_2_all, no_change_cs_2_all, increase_cs_2_all], axis=0),
                          ['d', 'd', 'd', 'd', 'n', 'n', 'n', 'n', 'i', 'i', 'i', 'i'], alpha=0.05))

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
    plt.savefig(paths['base_path'] + '/NN/plots/reactivation_influence.png', bbox_inches='tight', dpi=200)
    plt.savefig(paths['base_path'] + '/NN/plots/reactivation_influence.pdf', bbox_inches='tight', dpi=200,
                transparent=True)
    plt.close()


def grouped_count(mice, sample_dates):
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(9, 6))
    m_colors = ['b', 'purple', 'darkorange', 'green']
    x = [1, 2, 3]
    no_change_all = []
    increase_all = []
    decrease_all = []
    for mouse in range(0, len(mice)):
        paths = preprocess.create_folders(mice[mouse], sample_dates[mouse])
        grouped_count = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/activity_grouped_count.npy', allow_pickle=True)
        no_change = []
        increase = []
        decrease = []
        for i in range(0, len(grouped_count[0])):
            no_change.append(grouped_count[0][i]/grouped_count[3][i])
            increase.append(grouped_count[1][i]/grouped_count[3][i])
            decrease.append(grouped_count[2][i]/grouped_count[3][i])

        no_change_all.append(np.nanmean(no_change))
        increase_all.append(np.nanmean(increase))
        decrease_all.append(np.nanmean(decrease))

        plt.subplot(2, 2, 1)
        plt.errorbar(x[0], np.nanmean(no_change), yerr=0, c=m_colors[mouse], marker='o', mfc='none', mec=m_colors[mouse],
                     ms=5, alpha=.3)
        plt.errorbar(x[1], np.nanmean(increase), yerr=0, c=m_colors[mouse], marker='o', mfc='none', mec=m_colors[mouse],
                     ms=5, alpha=.3)
        plt.errorbar(x[2], np.nanmean(decrease), yerr=0, c=m_colors[mouse], marker='o', mfc='none', mec=m_colors[mouse],
                     ms=5, alpha=.3)

    # print(stats.f_oneway(area_li_cue_all, area_por_cue_all, area_p_cue_all, area_lm_cue_all))
    # print(pairwise_tukeyhsd(np.concatenate([area_li_cue_all, area_por_cue_all, area_p_cue_all, area_lm_cue_all], axis=0),
    #                         ['li', 'li', 'li', 'li', 'por', 'por', 'por', 'por', 'p', 'p', 'p', 'p', 'lm', 'lm', 'lm',
    #                          'lm'], alpha=0.05))

    plt.subplot(2, 2, 1)
    y1 = np.mean(no_change_all)
    y1_err = stats.sem(no_change_all)
    plt.errorbar(1.2, y1, yerr=y1_err, c='k', linewidth=2, marker='o', mfc='k', mec='k',
                 ms=7, mew=0, zorder=100)
    y2 = np.mean(increase_all)
    y2_err = stats.sem(increase_all)
    plt.errorbar(2.2, y2, yerr=y2_err, c='k', linewidth=2, marker='o', mfc='k', mec='k',
                 ms=7, mew=0, zorder=100)
    y3 = np.mean(decrease_all)
    y3_err = stats.sem(decrease_all)
    plt.errorbar(3.2, y3, yerr=y3_err, c='k', linewidth=2, marker='o', mfc='k', mec='k',
                 ms=7, mew=0, zorder=100)
    plt.xlim(.5, 3.5)
    plt.ylim(0, 1)
    plt.xticks([1, 2, 3], ['No Change', 'Increase', 'Decrease'])
    plt.ylabel('Percent of stimulus and reactivation\nparticipating neurons')

    sns.despine()
    plt.savefig(paths['base_path'] + '/NN/plots/grouped_count.png', bbox_inches='tight', dpi=200)
    plt.savefig(paths['base_path'] + '/NN/plots/grouped_count.pdf', bbox_inches='tight', dpi=200,
                transparent=True)
    plt.close()


def activity_difference_grouped(mice, sample_dates):
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(5, 6))
    m_colors = ['b', 'purple', 'darkorange', 'green']

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
        paths = preprocess.create_folders(mice[mouse], sample_dates[mouse])
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

    anova_results = []
    anova_results.append(stats.ttest_rel(cs1c_d_all, cs1r_d_all)[1])
    anova_results.append(stats.ttest_rel(cs2c_d_all, cs2r_d_all)[1])
    anova_results_corrected = multipletests(anova_results, alpha=.05, method='holm')
    return [anova_results, anova_results_corrected[1]]

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
    plt.savefig(paths['base_path'] + '/NN/plots/activity_difference_grouped.png', bbox_inches='tight', dpi=200)
    plt.savefig(paths['base_path'] + '/NN/plots/activity_difference_grouped.pdf', bbox_inches='tight', dpi=200, transparent=True)
    plt.close()
























































