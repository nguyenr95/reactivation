import os
import math
import scipy
import pickle
import statsmodels
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

    [_, s_p_value] = stats.shapiro(p_norm_total[:, int(len(p_norm_total[0])*2/4)])
    print(s_p_value)
    anova_results = []
    for i in range(0, len(p_norm_total[0])):
        anova_results.append(stats.ttest_ind(p_norm_total[:, i], p_shuffle_prior_total[:, i])[1])
    for i in range(0, len(p_norm_total[0])):
        anova_results.append(stats.ttest_ind(p_norm_total[:, i], p_shuffle_beta_total[:, i])[1])
    anova_results_corrected = multipletests(anova_results, alpha=.05, method='holm')

    print(int(len(p_norm_total[0])))
    print(anova_results_corrected[0][100:200])
    print(anova_results_corrected[0])

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
    m_colors = ['b', 'teal', 'slategray', 'green', 'darkolivegreen']
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

    [_, s_p_value] = stats.shapiro(mean_reactivation_mice[:, 1])
    print(s_p_value)

    slope, intercept, r_value, p_value, std_err = stats.linregress(np.concatenate([x_label[1:9], x_label[1:9], x_label[1:9], x_label[1:9], x_label[1:9]]), np.concatenate(mean_reactivation_mice[:, 1:9]))
    print(r_value)
    anova_results = []
    for i in range(1, 9):
        anova_results.append(stats.ttest_rel(mean_reactivation_mice[:, i], mean_reactivation_mice[:, 0], alternative='greater')[1])
    anova_results.append(p_value)
    anova_results_corrected = multipletests(anova_results, alpha=.05, method='holm')
    print([anova_results, anova_results_corrected[1]])

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
    m_colors = ['b', 'teal', 'slategray', 'green', 'darkolivegreen']
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
        np.concatenate([x_label, x_label, x_label, x_label, x_label]),
        np.concatenate(bias_all))
    print(r_value)
    print(p_value)

    [_, s_p_value] = stats.shapiro(bias_all[0])
    print(s_p_value)

    anova_results = []
    for i in range(0, len(bias_all[0])):
        vec = []
        for j in range(0, len(mice)):
            vec.append(bias_all[j][i])
        anova_results.append(
            stats.ttest_1samp(vec, 0)[1])
    anova_results.append(p_value)
    anova_results_corrected = multipletests(anova_results, alpha=.05, method='holm')
    print([anova_results, anova_results_corrected[1]])

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
    m_colors = ['b', 'teal', 'slategray', 'green', 'darkolivegreen']
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
        np.concatenate([x, x, x, x, x]),
        np.concatenate(mean_norm_mice))
    print(r_value)
    print(p_value)

    [_, s_p_value] = stats.shapiro(mean_norm_mice)
    print(s_p_value)
    mean_reactivation_mice = np.zeros((len(mice), 9))
    for mouse in range(0, len(mice)):
        paths = preprocess.create_folders(mice[mouse], sample_dates[mouse])
        binned_vec = list(np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/y_pred_binned.npy'))
        mean_reactivation = np.zeros((len(binned_vec), len(binned_vec[0])))
        for i in range(0, len(binned_vec)):
            mean_reactivation[i, :] = binned_vec[i]
        mean = mean_reactivation.mean(axis=0)
        mean_reactivation_mice[mouse, :] = mean
    anova_results = []
    for i in range(0, len(mean_norm_mice)):
        vec = []
        for j in range(0, len(mice)):
            vec.append(mean_norm_mice[j][i])
        anova_results.append(
            stats.ttest_rel(vec, mean_reactivation_mice[:, 0], alternative='greater')[1])
    anova_results.append(p_value)
    anova_results_corrected = multipletests(anova_results, alpha=.05, method='holm')
    print([anova_results, anova_results_corrected[1]])

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
    m_colors = ['b', 'teal', 'slategray', 'green', 'darkolivegreen']
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
        np.concatenate([x, x, x, x, x]),
        np.concatenate(mean_norm_mice))
    print(r_value)
    print(p_value)

    [_, s_p_value] = stats.shapiro(mean_norm_mice[0])
    print(s_p_value)

    anova_results = []
    for i in range(0, len(mean_norm_mice[0])):
        vec = []
        for j in range(0, len(mice)):
            vec.append(mean_norm_mice[j][i])
        anova_results.append(
            stats.ttest_1samp(vec, 0)[1])
    anova_results.append(p_value)
    anova_results_corrected = multipletests(anova_results, alpha=.05, method='holm')
    print([anova_results, anova_results_corrected[1]])

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
    m_colors = ['b', 'teal', 'slategray', 'green', 'darkolivegreen', 'darkorange', 'purple', 'darkred']
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
    # slope, intercept, r_value, p_value, std_err = stats.linregress(
    #     np.concatenate([x_label[1:9], x_label[1:9], x_label[1:9], x_label[1:9], x_label[1:9], x_label[1:9], x_label[1:9], x_label[1:9]]),
    #     np.concatenate(mean_iti_mice[:, 1:9]))
    # print(r_value)
    # anova_results = []
    # anova_results.append(stats.ttest_rel(mean_iti_mice[:, 1], mean_iti_mice[:, 0])[1])
    # anova_results.append(p_value)
    # anova_results_corrected = multipletests(anova_results, alpha=.05, method='holm')
    # return [anova_results, anova_results_corrected[1]]

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
    m_colors = ['b', 'teal', 'slategray', 'green', 'darkolivegreen', 'darkorange', 'purple', 'darkred']
    plt.subplot(2, 2, 1)
    paths = preprocess.create_folders(mice[0], sample_dates[0])
    activity_control_all = np.load(paths['base_path'] + paths['mouse'] +
                                   '/data_across_days/activity_within_trial.npy', allow_pickle=True)
    activity_control_norm_all = np.zeros((len(mice), len(activity_control_all[0][0])))
    for mouse in range(0, len(mice)):
        paths = preprocess.create_folders(mice[mouse], sample_dates[mouse])
        session_data = preprocess.load_data(paths)
        fr = session_data['framerate']
        activity_control_all = np.load(paths['base_path'] + paths['mouse'] +
                                       '/data_across_days/activity_within_trial.npy', allow_pickle=True)
        activity_control_norm = np.zeros((len(activity_control_all[0]), len(activity_control_all[0][0])))
        for i in range(0, len(activity_control_all[0])):
            activity_control_norm[i, :] = activity_control_all[0][i] * fr[0][0]
        y1 = activity_control_norm.mean(axis=0)
        activity_control_norm_all[mouse, :] = y1
        plt.plot(y1, '-', c=m_colors[mouse], alpha=.2, linewidth=2)
        x_label = range(0, len(y1))

    slope, intercept, r_value, p_value, std_err = stats.linregress(
        np.concatenate(
            [x_label[int(fr * 13):int(fr * 63)], x_label[int(fr * 13):int(fr * 63)], x_label[int(fr * 13):int(fr * 63)],
             x_label[int(fr * 13):int(fr * 63)], x_label[int(fr * 13):int(fr * 63)], x_label[int(fr * 13):int(fr * 63)],
             x_label[int(fr * 13):int(fr * 63)], x_label[int(fr * 13):int(fr * 63)]]),
        np.concatenate(activity_control_norm_all[:, int(fr * 13):int(fr * 63)]))
    print(r_value)
    print(p_value)

    mean = activity_control_norm_all.mean(axis=0)
    sem_plus = mean + stats.sem(activity_control_norm_all, axis=0)
    sem_minus = mean - stats.sem(activity_control_norm_all, axis=0)
    plt.plot(mean, linewidth=3, c='k')
    plt.fill_between(range(0, len(mean)), sem_plus, sem_minus, alpha=0.2, color='k', lw=0)
    plt.xticks([int(fr * 3), int(fr * 5), int(fr * 15), int(fr * 25), int(fr * 35), int(fr * 45), int(fr * 55),
                int(fr * 65)], ['', '0', '10', '20', '30', '40', '50', '60'])
    plt.xlim((int(fr * 3), int(fr * 63)))
    plt.ylabel('Activity\n(Normalized $\mathregular{Ca^{2+}}$transient $\mathregular{s^{-1}}$)')
    plt.xlabel('Time relative to stimulus onset (s)')
    plt.axvspan(int(fr * 5), int(fr * 7), alpha=.2, color='k', zorder=0, lw=0)
    plt.axvspan(int(fr * 7), int(fr * 13), alpha=.1, color='gray', zorder=0, lw=0)
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1, snap=False)
    sns.despine()
    plt.ylim((0, 2))
    plt.savefig(paths['base_path'] + '/NN/plots/iti_activity_within_trial.png', bbox_inches='tight', dpi=200)
    plt.savefig(paths['base_path'] + '/NN/plots/iti_activity_within_trial.pdf', bbox_inches='tight', dpi=200, transparent=True)
    plt.close()


def pupil_across_trials(mice, sample_dates):
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(9, 6))
    m_colors = ['b', 'teal', 'slategray', 'green', 'darkolivegreen', 'darkorange', 'purple', 'darkred']
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
        np.concatenate([x_label[1:9], x_label[1:9], x_label[1:9], x_label[1:9], x_label[1:9], x_label[1:9], x_label[1:9], x_label[1:9]]),
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
    m_colors = ['b', 'teal', 'slategray', 'green', 'darkolivegreen', 'darkorange', 'purple', 'darkred']
    plt.subplot(2, 2, 1)
    paths = preprocess.create_folders(mice[0], sample_dates[0])
    control_all = np.load(paths['base_path'] + paths['mouse'] +
                                   '/data_across_days/pupil_within_trial.npy', allow_pickle=True)
    control_norm_all = np.zeros((len(mice), len(control_all[0][0])))
    for mouse in range(0, len(mice)):
        paths = preprocess.create_folders(mice[mouse], sample_dates[mouse])
        session_data = preprocess.load_data(paths)
        fr = session_data['framerate']
        control_all = np.load(paths['base_path'] + paths['mouse'] +
                                       '/data_across_days/pupil_within_trial.npy', allow_pickle=True)
        control_norm = np.zeros((len(control_all[0]), len(control_all[0][0])))
        for i in range(0, len(control_all[0])):
            control_norm[i, :] = control_all[0][i]
        y1 = control_norm.mean(axis=0)
        control_norm_all[mouse, :] = y1
        plt.plot(y1, '-', c=m_colors[mouse], alpha=.2, linewidth=2)
        x_label = range(0, len(y1))
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        np.concatenate(
            [x_label[int(fr * 13):int(fr * 63)], x_label[int(fr * 13):int(fr * 63)], x_label[int(fr * 13):int(fr * 63)],
             x_label[int(fr * 13):int(fr * 63)], x_label[int(fr * 13):int(fr * 63)], x_label[int(fr * 13):int(fr * 63)],
             x_label[int(fr * 13):int(fr * 63)], x_label[int(fr * 13):int(fr * 63)]]),
        np.concatenate(control_norm_all[:, int(fr * 13):int(fr * 63)]))
    print(r_value)
    print(p_value)

    print(stats.ttest_rel(control_norm_all[:, int(fr * 13)], control_norm_all[:, int(fr * 63)-1]))
    mean = control_norm_all.mean(axis=0)
    sem_plus = mean + stats.sem(control_norm_all, axis=0)
    sem_minus = mean - stats.sem(control_norm_all, axis=0)
    plt.plot(mean, linewidth=3, c='k')
    plt.fill_between(range(0, len(mean)), sem_plus, sem_minus, alpha=0.2, color='k', lw=0)
    plt.xticks([int(fr * 3), int(fr * 5), int(fr * 15), int(fr * 25), int(fr * 35), int(fr * 45), int(fr * 55),
                int(fr * 65)], ['', '0', '10', '20', '30', '40', '50', '60'])
    plt.xlim((int(fr * 3), int(fr * 63)))
    plt.ylabel('Max. normalized pupil area')
    plt.xlabel('Time relative to stimulus onset (s)')
    plt.axvspan(int(fr * 5), int(fr * 7), alpha=.2, color='k', zorder=0, lw=0)
    plt.axvspan(int(fr * 7), int(fr * 13), alpha=.1, color='gray', zorder=0, lw=0)
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1, snap=False)
    sns.despine()
    plt.ylim((0, 1))
    plt.savefig(paths['base_path'] + '/NN/plots/pupil_trial.png', bbox_inches='tight', dpi=200)
    plt.savefig(paths['base_path'] + '/NN/plots/pupil_trial.pdf', bbox_inches='tight', dpi=200, transparent=True)
    plt.close()


def reactivation_physical(mice, sample_dates):
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(14.5, 7))
    # plt.subplots_adjust(hspace=.3)
    paths = preprocess.create_folders(mice[0], sample_dates[0])
    reactivation_physical_helper(0, mice, sample_dates, paths, 'Pupil area (ΔA/A)', 'blue',
                                 1, -.22, .02)
    reactivation_physical_helper(0, mice, sample_dates, paths, 'Max normalized pupil area', 'blue',
                                 2, 0, 1)
    reactivation_physical_helper(1, mice, sample_dates, paths, 'Max normalized pupil movement', 'red', 3, 0, 1)
    reactivation_physical_helper(2, mice, sample_dates, paths, 'Brain motion (μm, abs)',
                                 'darkgoldenrod', 4, 0, 1)
    reactivation_physical_helper(3, mice, sample_dates, paths, 'Phase correlation to reference frame',
                                 'cadetblue', 5, 0, .1)
    plt.savefig(paths['base_path'] + '/NN/plots/reactivation_physical.png', bbox_inches='tight', dpi=200)
    plt.savefig(paths['base_path'] + '/NN/plots/reactivation_physical.pdf', bbox_inches='tight', dpi=500, transparent=True)
    plt.close()


def reactivation_physical_helper(vec_idx, mice, sample_dates, paths, y_label, c, idx, lim1, lim2):
    plt.subplot(2, 3, idx)
    vec = list(np.load(paths['base_path'] + paths['mouse'] +
                       '/data_across_days/reactivation_physical.npy', allow_pickle=True))
    mean_mice = np.zeros((len(mice), len(vec[vec_idx][0])))
    m_colors = ['b', 'teal', 'slategray', 'green', 'darkolivegreen', 'darkorange', 'purple', 'darkred']
    mean_mice_heatmap = []
    days = []
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
                mean_mice_heatmap.append((vec[i] - np.mean(vec[i][0:5])) / np.mean(vec[i][0:5]))
            if y_label == 'Brain motion (μm, abs)':
                mean_vec[i, :] = vec[i] * 1500/512
            if y_label == 'Max normalized pupil movement':
                mean_vec[i, :] = vec[i]
            if y_label == 'Max normalized pupil area':
                mean_vec[i, :] = vec[i]
            if y_label == 'Phase correlation to reference frame':
                mean_vec[i, :] = vec[i]
        mean = mean_vec.mean(axis=0)
        mean_mice[mouse, :] = mean
        days.append(len(vec))

    [_, s_p_value] = stats.shapiro(mean_mice[:, int(framerate * 20)])
    print(s_p_value)
    print(stats.ttest_rel(np.mean(mean_mice[:, 0:11], axis=1), np.mean(mean_mice[:, int(framerate * 20)-5:int(framerate * 20)+6], axis=1)))

    mean = mean_mice.mean(axis=0)
    sem_plus = mean + stats.sem(mean_mice, axis=0)
    sem_minus = mean - stats.sem(mean_mice, axis=0)
    plt.plot(mean, c=c, linewidth=2, ms=7)
    plt.fill_between(np.arange(mean.shape[0]), sem_plus, sem_minus, alpha=0.2, color=c, lw=0)
    plt.xticks([int(framerate * 0), int(framerate * 10), int(framerate * 20),
                int(framerate * 30), int(framerate * 40)], ['-20', '-10', '0', '10', '20'])
    plt.ylabel(y_label)
    plt.ylim(lim1, lim2)
    if y_label == 'Pupil area (ΔA/A)':
        plt.yticks([-.1, -.2, 0])
    if y_label == 'Brain motion (μm, abs)':
        plt.yticks([0, .5, 1])
    if y_label == 'Max normalized pupil movement':
        plt.yticks([0, .5, 1])
    if y_label == 'Max normalized pupil area':
        plt.yticks([0, .5, 1])
    if y_label == 'Phase correlation to reference frame':
        plt.yticks([0, .05, .1])
    plt.axvline(x=int(framerate * 20), color='black', linestyle='--', linewidth=1, snap=False)
    plt.xlabel('Time relative to reactivation onset (s)')
    sns.despine()

    if y_label == 'Pupil area (ΔA/A)':
        sns.set(font_scale=.7)
        sns.set_style("whitegrid", {'axes.grid': False})
        sns.set_style("ticks")
        plt.figure(figsize=(4, 6))
        plt.subplot(2, 2, 1)
        sns.heatmap(mean_mice_heatmap, vmin=-.2, vmax=.2, cmap="coolwarm", cbar=0)
        y_idx = 0
        for mouse in range(0, 5):
            plt.plot([-5, -5], [y_idx, y_idx + days[mouse]], linewidth=7, snap=False, solid_capstyle='butt')
            y_idx += days[mouse]
        plt.ylim(len(mean_mice_heatmap) + 3, -3)
        plt.xlim(0, len(mean))
        plt.axis('off')
        plt.savefig(paths['base_path'] + '/NN/plots/pupil_heatmap.png', bbox_inches='tight', dpi=500,
                    transparent=True)


def trial_history(num_prev, mice, sample_dates):
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(2.5, 6))
    m_colors = ['b', 'teal', 'slategray', 'green', 'darkolivegreen', 'darkorange', 'purple', 'darkred']
    x = [1, 2]
    rate_same_all = []
    for mouse in range(0, len(mice)):
        paths = preprocess.create_folders(mice[mouse], sample_dates[mouse])
        trial_hist = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/trial_history_' +
                             str(num_prev) + '.npy', allow_pickle=True)
        rate_same = np.zeros(len(trial_hist[0]))
        rate_diff = np.zeros(len(trial_hist[0]))
        for i in range(0, len(trial_hist[0])):
            rate_same[i] = ((trial_hist[1][i] - trial_hist[0][i]) / trial_hist[0][i]) * 100
        rate_same_all.append(np.mean(rate_same))
        plt.subplot(2, 2, 1)
        plt.errorbar(x[0], rate_same_all[mouse], yerr=0, c=m_colors[mouse], marker='o', mfc='none', mec=m_colors[mouse],
                     ms=5, alpha=.3)
    print(stats.ttest_1samp(rate_same_all, 0))
    plt.subplot(2, 2, 1)
    plt.errorbar(x[0] + .2, np.mean(rate_same_all, axis=0), yerr=stats.sem(rate_same_all, axis=0), c='k', linewidth=2, marker='o',
                 mfc='k', mec='k',
                 ms=7, mew=0, zorder=100)
    plt.xlim(.5, 1.5)
    plt.ylim(-50, 50)
    plt.yticks([-50, -25, 0, 25, 50])
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1, snap=False)
    plt.xticks([])
    plt.ylabel('Reactivation rate (probability $\mathregular{s^{-1}}$)')

    sns.despine()
    plt.savefig(paths['base_path'] + '/NN/plots/trial_history.png', bbox_inches='tight', dpi=200)
    plt.savefig(paths['base_path'] + '/NN/plots/trial_history.pdf', bbox_inches='tight', dpi=200, transparent=True)
    plt.close()


def reactivation_duration(mice, sample_dates):
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(5, 6))
    m_colors = ['b', 'teal', 'slategray', 'green', 'darkolivegreen', 'darkorange', 'purple', 'darkred']

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
            control = length_all[1][i] / fr[0] * 1000
            cue = length_all[0][i] / fr[0] * 1000
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
    plt.ylim((0, 400))
    plt.xticks([1, 2], ['Baseline\nperiod', 'Stimulus\npresentations'])
    plt.yticks([0, 100, 200, 300, 400])
    plt.gca().get_xticklabels()[1].set_color('k')
    plt.gca().get_xticklabels()[0].set_color('k')
    plt.ylabel('Reactivation duration (ms)')
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
            # plt.subplot(1, 3, 1)
            # plt.errorbar(x, [top_cs_1_c, top_cs_2_c, bottom_cs_1_c, bottom_cs_2_c, other_c], yerr=0, c=c, linewidth=2,
            #              linestyle='-', zorder=0, alpha=.2)
            # plt.subplot(1, 3, 2)
            # plt.errorbar(x, [top_cs_1_r, top_cs_2_r, bottom_cs_1_r, bottom_cs_2_r, other_r], yerr=0, c=c, linewidth=2,
            #              linestyle='-', zorder=0, alpha=.2)
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

    [_, s_p_value] = stats.shapiro(cs_1[:, 0])
    print(s_p_value)
    anova_results = []
    for i in range(0, len(cs_1[0])):
        anova_results.append(stats.ttest_ind(cs_1[:, i], cs_2[:, i])[1])
    anova_results_corrected = multipletests(anova_results, alpha=.05, method='holm')
    return [anova_results, anova_results_corrected[1]]

    plt.subplot(1, 3, 3)
    # for i in range(0, len(mice)):
    #     plt.errorbar(x, baseline[i], yerr=0, c='k', linewidth=2, linestyle='-', zorder=0, alpha=.2)
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


def activity_across_trials(mice, sample_dates):
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(9, 6))
    plt.subplots_adjust(wspace=.33)
    m_colors = ['b', 'teal', 'slategray', 'green', 'darkolivegreen', 'darkorange', 'purple', 'darkred']

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
            if mouse < 5:
                smoothed_activity = np.concatenate(smoothed_activity, axis=0)
                activity[i, 0:len(smoothed_activity)] = smoothed_activity
                mean_activity_mice_heatmap.append(smoothed_activity[0:112] - np.mean(smoothed_activity[0:3]))
            if mouse > 4:
                smoothed_activity = np.concatenate(smoothed_activity, axis=0)
                smoothed_activity = scipy.signal.resample_poly(smoothed_activity, 2, 1, padtype='smooth')[0:(len(smoothed_activity)*2)]
                activity[i, 0:len(smoothed_activity)] = smoothed_activity
                mean_activity_mice_heatmap.append(smoothed_activity[0:112] - np.mean(smoothed_activity[0:3]))
        activity = np.nanmean(activity, axis=0)
        activity_all[mouse, :] = activity
        x = range(1, len(activity)+1)
        plt.plot(x[0:120], activity[0:120], c=m_colors[mouse], alpha=.2)
        days.append(len(activity_data[0]))

    [_, p_value] = stats.ttest_rel(np.concatenate(activity_all[:, 0:3], axis=0), np.concatenate(activity_all[:, 117:120], axis=0), nan_policy='omit')
    slope, intercept, r_value, p_value_dec, std_err = stats.linregress(
        np.concatenate([x[0:120], x[0:120], x[0:120], x[0:120], x[0:120], x[0:120], x[0:120], x[0:120]]),
        np.concatenate(activity_all[:, 0:120]))
    print(r_value)
    anova_results = []
    anova_results.append(p_value)
    anova_results.append(p_value_dec)
    anova_results_corrected = multipletests(anova_results, alpha=.05, method='holm')
    print(anova_results_corrected[1])

    mean = np.nanmean(activity_all, axis=0)
    sem_plus = mean + stats.sem(activity_all, axis=0, nan_policy='omit')
    sem_minus = mean - stats.sem(activity_all, axis=0, nan_policy='omit')
    plt.plot(x[0:120], mean[0:120], c='k', linewidth=3)
    plt.fill_between(x[0:120], sem_plus[0:120], sem_minus[0:120], alpha=0.2, color='k', lw=0)
    plt.ylabel('Correlation between\nstimulus 1 and stimulus 2')
    plt.ylim(-.05, .6)
    plt.xlim(1, 121)
    plt.xticks([1, 120], ['First trial', 'Last trial'])
    sns.despine()

    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(5, 6))
    m_colors = ['b', 'teal', 'slategray', 'green', 'darkolivegreen', 'darkorange', 'purple', 'darkred']
    r_value_exp = []
    for i in range(0, 8):
        def monoExp(x, m, t, b):
            return m * np.exp(-t * x) + b

        xs = range(0, 115)
        ys = activity_all[i, 0:115]
        # perform the fit
        params, cv = scipy.optimize.curve_fit(monoExp, xs, ys)
        m, t, b = params
        # determine quality of the fit
        squaredDiffs = np.square(ys - monoExp(xs, m, t, b))
        squaredDiffsFromMean = np.square(ys - np.mean(ys))
        rSquared = 1 - np.sum(squaredDiffs) / np.sum(squaredDiffsFromMean)
        r_value_exp.append(rSquared)
        plt.subplot(2, 2, 1)
        # plt.errorbar(1, rSquared, yerr=0, c=m_colors[mouse], marker='o', mfc='none', mec=m_colors[mouse],
        #              ms=5, alpha=.3)
    r_value_linear = []
    for i in range(0, 8):
        def linear(x, m, b):
            return (m * x) + b

        xs = range(0, 115)
        ys = activity_all[i, 0:115]
        # perform the fit
        params, cv = scipy.optimize.curve_fit(linear, xs, ys)
        m, b = params
        # determine quality of the fit
        squaredDiffs = np.square(ys - linear(xs, m, b))
        squaredDiffsFromMean = np.square(ys - np.mean(ys))
        rSquared = 1 - np.sum(squaredDiffs) / np.sum(squaredDiffsFromMean)
        r_value_linear.append(rSquared)
        plt.subplot(2, 2, 1)
        # plt.errorbar(2, rSquared, yerr=0, c=m_colors[mouse], marker='o', mfc='none', mec=m_colors[mouse],
        #              ms=5, alpha=.3)
    plt.subplot(2, 2, 1)
    y1 = np.mean(r_value_linear)
    y1_err = stats.sem(r_value_linear)
    plt.errorbar(1, y1, yerr=y1_err, c='k', linewidth=2, marker='o', mfc='k', mec='k',
                 ms=7, mew=0, zorder=100)
    y1 = np.mean(r_value_exp)
    y1_err = stats.sem(r_value_exp)
    plt.errorbar(2, y1, yerr=y1_err, c='k', linewidth=2, marker='o', mfc='k', mec='k',
                 ms=7, mew=0, zorder=100)
    plt.xlim(.5, 2.5)
    plt.ylim(0, 1)
    plt.xticks([1, 2], ['Linear', 'Exp.'])
    plt.ylabel('Figure 3C\nR-squared value')
    sns.despine()
    plt.savefig(paths['base_path'] + '/NN/plots/rsquared_correlation.png', bbox_inches='tight', dpi=200)
    plt.savefig(paths['base_path'] + '/NN/plots/rsquared_correlation.pdf', bbox_inches='tight', dpi=200,
                transparent=True)
    plt.close()

    plt.subplot(2, 2, 2)
    activity_all = np.empty((len(mice), 128)) * np.nan
    activity_all_opto = np.empty((len(mice), 128)) * np.nan
    for mouse in range(0, len(mice)):
        paths = preprocess.create_folders(mice[mouse], sample_dates[mouse])
        activity_data = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/activity.npy',
                                allow_pickle=True)
        activity = np.empty((len(activity_data[0]), 128)) * np.nan
        activity_opto = np.empty((len(activity_data[0]), 128)) * np.nan
        for i in range(0, len(activity_data[0])):
            smoothed_activity = np.array(
                pd.DataFrame(activity_data[2][i]).rolling(3, min_periods=1, center=True).mean())
            smoothed_activity = np.array(
                pd.DataFrame(activity_data[0][i]).rolling(3, min_periods=1, center=True).mean())
            if mouse < 5:
                smoothed_activity = np.concatenate(smoothed_activity, axis=0)
                activity[i, 0:len(smoothed_activity)] = smoothed_activity
            if mouse > 4:
                smoothed_activity = np.concatenate(smoothed_activity, axis=0)
                smoothed_activity = scipy.signal.resample_poly(smoothed_activity, 2, 1, padtype='smooth')[
                                    0:(len(smoothed_activity) * 2)]
                activity_opto[i, 0:len(smoothed_activity)] = smoothed_activity
        activity = np.nanmean(activity, axis=0)
        activity_all[mouse, :] = activity
        activity_opto = np.nanmean(activity_opto, axis=0)
        activity_all_opto[mouse, :] = activity_opto
        if mouse < 5:
            x = range(1, len(activity) + 1)
            plt.plot(x[0:120], activity[0:120], c='k', alpha=.2)
        if mouse > 4:
            x = range(1, len(activity_opto) + 1)
            plt.plot(x[0:120], activity_opto[0:120], c='r', alpha=.2)

    mean = np.nanmean(activity_all, axis=0)
    sem_plus = mean + stats.sem(activity_all, axis=0, nan_policy='omit')
    sem_minus = mean - stats.sem(activity_all, axis=0, nan_policy='omit')
    plt.plot(x[0:120], mean[0:120], c='k', linewidth=3)
    plt.fill_between(x[0:120], sem_plus[0:120], sem_minus[0:120], alpha=0.2, color='k', lw=0)
    mean = np.nanmean(activity_all_opto, axis=0)
    sem_plus = mean + stats.sem(activity_all_opto, axis=0, nan_policy='omit')
    sem_minus = mean - stats.sem(activity_all_opto, axis=0, nan_policy='omit')
    plt.plot(x[0:120], mean[0:120], c='r', linewidth=3)
    plt.fill_between(x[0:120], sem_plus[0:120], sem_minus[0:120], alpha=0.2, color='r', lw=0)
    plt.ylabel('Correlation between\nstimulus 1 and stimulus 2')
    plt.ylim(-.05, .6)
    plt.xlim(1, 121)
    plt.xticks([1, 120], ['First trial', 'Last trial'])
    sns.despine()

    plt.subplot(2, 2, 3)
    activity_all = np.empty((len(mice), 128)) * np.nan
    activity_all_opto = np.empty((len(mice), 128)) * np.nan
    for mouse in range(0, len(mice)):
        paths = preprocess.create_folders(mice[mouse], sample_dates[mouse])
        activity_data = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/activity.npy',
                                allow_pickle=True)
        activity = np.empty((len(activity_data[0]), 128)) * np.nan
        activity_opto = np.empty((len(activity_data[0]), 128)) * np.nan
        for i in range(0, len(activity_data[0])):
            smoothed_activity = np.array(
                pd.DataFrame(activity_data[2][i]).rolling(3, min_periods=1, center=True).mean())
            smoothed_activity = np.array(
                pd.DataFrame(activity_data[0][i]).rolling(3, min_periods=1, center=True).mean())
            if mouse < 5:
                smoothed_activity = np.concatenate(smoothed_activity, axis=0)
                activity[i, 0:len(smoothed_activity)] = smoothed_activity - np.mean(smoothed_activity[0:3])
            if mouse > 4:
                smoothed_activity = np.concatenate(smoothed_activity, axis=0)
                smoothed_activity = scipy.signal.resample_poly(smoothed_activity, 2, 1, padtype='smooth')[
                                    0:(len(smoothed_activity) * 2)]
                activity_opto[i, 0:len(smoothed_activity)] = smoothed_activity - np.mean(smoothed_activity[0:3])
        activity = np.nanmean(activity, axis=0)
        activity_all[mouse, :] = activity
        activity_opto = np.nanmean(activity_opto, axis=0)
        activity_all_opto[mouse, :] = activity_opto
        if mouse < 5:
            x = range(1, len(activity) + 1)
            plt.plot(x[0:120], activity[0:120], c='k', alpha=.2)
        if mouse > 4:
            x = range(1, len(activity_opto) + 1)
            plt.plot(x[0:120], activity_opto[0:120], c='r', alpha=.2)

    mean = np.nanmean(activity_all, axis=0)
    sem_plus = mean + stats.sem(activity_all, axis=0, nan_policy='omit')
    sem_minus = mean - stats.sem(activity_all, axis=0, nan_policy='omit')
    plt.plot(x[0:120], mean[0:120], c='k', linewidth=3)
    plt.fill_between(x[0:120], sem_plus[0:120], sem_minus[0:120], alpha=0.2, color='k', lw=0)
    mean = np.nanmean(activity_all_opto, axis=0)
    sem_plus = mean + stats.sem(activity_all_opto, axis=0, nan_policy='omit')
    sem_minus = mean - stats.sem(activity_all_opto, axis=0, nan_policy='omit')
    plt.plot(x[0:120], mean[0:120], c='r', linewidth=3)
    plt.fill_between(x[0:120], sem_plus[0:120], sem_minus[0:120], alpha=0.2, color='r', lw=0)
    plt.ylabel('ΔCorrelation between\nstimulus 1 and stimulus 2')
    plt.ylim(-.4, .1)
    plt.xlim(1, 121)
    plt.xticks([1, 120], ['First trial', 'Last trial'])
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
    sns.despine()

    plt.savefig(paths['base_path'] + '/NN/plots/correlation_across_trials.png', bbox_inches='tight', dpi=200)
    plt.savefig(paths['base_path'] + '/NN/plots/correlation_across_trials.pdf', bbox_inches='tight', dpi=200,
                transparent=True)
    plt.close()

    sns.set(font_scale=.7)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(4, 6))
    plt.subplot(2, 2, 1)
    sns.heatmap(mean_activity_mice_heatmap, vmin=-.3, vmax=.3, cmap="coolwarm", cbar=0)
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

    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(9, 6))
    plt.subplots_adjust(wspace=.33)

    plt.subplot(2, 2, 1)
    activity_all = np.empty((len(mice), 128)) * np.nan
    mean_activity_mice_heatmap = []
    days = []
    for mouse in range(0, len(mice)):
        paths = preprocess.create_folders(mice[mouse], sample_dates[mouse])
        session_data = preprocess.load_data(paths)
        fr = session_data['framerate']
        activity_data = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/activity.npy',
                                allow_pickle=True)
        activity = np.empty((len(activity_data[0]), 128)) * np.nan
        for i in range(0, len(activity_data[0])):
            smoothed_activity = np.array(
                pd.DataFrame(activity_data[2][i]).rolling(3, min_periods=1, center=True).mean()) * fr
            if mouse < 5:
                smoothed_activity = np.concatenate(smoothed_activity, axis=0)
                activity[i, 0:len(smoothed_activity)] = smoothed_activity
                mean_activity_mice_heatmap.append(smoothed_activity[0:112])
            if mouse > 4:
                smoothed_activity = np.concatenate(smoothed_activity, axis=0)
                smoothed_activity = scipy.signal.resample_poly(smoothed_activity, 2, 1, padtype='smooth')[
                                    0:(len(smoothed_activity) * 2)]
                activity[i, 0:len(smoothed_activity)] = smoothed_activity
                mean_activity_mice_heatmap.append(smoothed_activity[0:112])
        activity = np.nanmean(activity, axis=0)
        activity_all[mouse, :] = activity
        x = range(1, len(activity) + 1)
        plt.plot(x[0:120], activity[0:120], c=m_colors[mouse], alpha=.2)
        days.append(len(activity_data[0]))

    # mask = ~np.isnan(np.concatenate([x, x, x, x])) & ~np.isnan(np.concatenate(activity_all))
    # slope, intercept, r_value, p_value, std_err = stats.linregress(
    #     np.concatenate([x, x, x, x])[mask], np.concatenate(activity_all)[mask])
    # print(r_value)
    # print(p_value)
    #print(stats.ttest_rel(np.mean(activity_all[:, 0:3], axis=0), np.mean(activity_all[:, 118:121], axis=0)))

    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(5, 6))
    m_colors = ['b', 'teal', 'slategray', 'green', 'darkolivegreen', 'darkorange', 'purple', 'darkred']
    r_value_exp = []
    for i in range(0, 8):
        def monoExp(x, m, t, b):
            return m * np.exp(-t * x) + b
        xs = range(0, 115)
        ys = activity_all[i, 0:115]
        # perform the fit
        params, cv = scipy.optimize.curve_fit(monoExp, xs, ys)
        m, t, b = params
        # determine quality of the fit
        squaredDiffs = np.square(ys - monoExp(xs, m, t, b))
        squaredDiffsFromMean = np.square(ys - np.mean(ys))
        rSquared = 1 - np.sum(squaredDiffs) / np.sum(squaredDiffsFromMean)
        r_value_exp.append(rSquared)
        plt.subplot(2, 2, 1)
        # plt.errorbar(1, rSquared, yerr=0, c=m_colors[mouse], marker='o', mfc='none', mec=m_colors[mouse],
        #              ms=5, alpha=.3)
    r_value_linear = []
    for i in range(0, 8):
        def linear(x, m, b):
            return (m * x) + b
        xs = range(0, 115)
        ys = activity_all[i, 0:115]
        # perform the fit
        params, cv = scipy.optimize.curve_fit(linear, xs, ys)
        m, b = params
        # determine quality of the fit
        squaredDiffs = np.square(ys - linear(xs, m, b))
        squaredDiffsFromMean = np.square(ys - np.mean(ys))
        rSquared = 1 - np.sum(squaredDiffs) / np.sum(squaredDiffsFromMean)
        r_value_linear.append(rSquared)
        plt.subplot(2, 2, 1)
        # plt.errorbar(2, rSquared, yerr=0, c=m_colors[mouse], marker='o', mfc='none', mec=m_colors[mouse],
        #              ms=5, alpha=.3)
    plt.subplot(2, 2, 1)
    y1 = np.mean(r_value_linear)
    y1_err = stats.sem(r_value_linear)
    plt.errorbar(1, y1, yerr=y1_err, c='k', linewidth=2, marker='o', mfc='k', mec='k',
                 ms=7, mew=0, zorder=100)
    y1 = np.mean(r_value_exp)
    y1_err = stats.sem(r_value_exp)
    plt.errorbar(2, y1, yerr=y1_err, c='k', linewidth=2, marker='o', mfc='k', mec='k',
                 ms=7, mew=0, zorder=100)
    plt.xlim(.5, 2.5)
    plt.ylim(0, .4)
    plt.xticks([1, 2], ['Linear', 'Exp.'])
    plt.ylabel('Figure 3A\nR-squared value')
    sns.despine()
    plt.savefig(paths['base_path'] + '/NN/plots/rsquared_activity.png', bbox_inches='tight', dpi=200)
    plt.savefig(paths['base_path'] + '/NN/plots/rsquared_activity.pdf', bbox_inches='tight', dpi=200,
                transparent=True)
    plt.close()

    [_, p_value] = stats.ttest_rel(np.concatenate(activity_all[:, 0:3], axis=0),
                                   np.concatenate(activity_all[:, 117:120], axis=0), nan_policy='omit')
    slope, intercept, r_value, p_value_dec, std_err = stats.linregress(
        np.concatenate([x[0:120], x[0:120], x[0:120], x[0:120], x[0:120], x[0:120], x[0:120], x[0:120]]),
        np.concatenate(activity_all[:, 0:120]))
    print(r_value)
    anova_results = []
    anova_results.append(p_value)
    anova_results.append(p_value_dec)
    anova_results_corrected = multipletests(anova_results, alpha=.05, method='holm')
    print(anova_results_corrected[1])

    mean = np.nanmean(activity_all, axis=0)
    sem_plus = mean + stats.sem(activity_all, axis=0, nan_policy='omit')
    sem_minus = mean - stats.sem(activity_all, axis=0, nan_policy='omit')
    plt.plot(x[0:120], mean[0:120], c='k', linewidth=3)
    plt.fill_between(x[0:120], sem_plus[0:120], sem_minus[0:120], alpha=0.2, color='k', lw=0)
    plt.ylabel('Stimulus activity\n(Normalized $\mathregular{Ca^{2+}}$transient $\mathregular{s^{-1}}$)')
    plt.ylim(0, 1)
    plt.xlim(1, 121)
    plt.xticks([1, 120], ['First trial', 'Last trial'])
    plt.yticks([0, .2, .4, .6, .8, 1])
    sns.despine()

    activity_control_norm_all = []
    for mouse in range(0, len(mice)):
        paths = preprocess.create_folders(mice[mouse], sample_dates[mouse])
        session_data = preprocess.load_data(paths)
        fr = session_data['framerate']
        activity_control_all = np.load(paths['base_path'] + paths['mouse'] +
                                       '/data_across_days/activity_within_trial.npy', allow_pickle=True)
        activity_control_norm = []
        for i in range(0, len(activity_control_all[0])):
            activity_control_norm.append(activity_control_all[0][i] * fr[0][0])
        y1 = np.mean(activity_control_norm)
        activity_control_norm_all.append(y1)
    activity_control_norm_all = np.mean(activity_control_norm_all)
    plt.axhline(y=activity_control_norm_all, color='black', linestyle='--', linewidth=1)

    plt.subplot(2, 2, 2)
    activity_all = np.empty((len(mice), 128)) * np.nan
    activity_all_opto = np.empty((len(mice), 128)) * np.nan
    for mouse in range(0, len(mice)):
        paths = preprocess.create_folders(mice[mouse], sample_dates[mouse])
        activity_data = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/activity.npy',
                                allow_pickle=True)
        activity = np.empty((len(activity_data[0]), 128)) * np.nan
        activity_opto = np.empty((len(activity_data[0]), 128)) * np.nan
        for i in range(0, len(activity_data[0])):
            smoothed_activity = np.array(
                pd.DataFrame(activity_data[2][i]).rolling(3, min_periods=1, center=True).mean()) * fr
            if mouse < 5:
                smoothed_activity = np.concatenate(smoothed_activity, axis=0)
                activity[i, 0:len(smoothed_activity)] = smoothed_activity
            if mouse > 4:
                smoothed_activity = np.concatenate(smoothed_activity, axis=0)
                smoothed_activity = scipy.signal.resample_poly(smoothed_activity, 2, 1, padtype='smooth')[
                                    0:(len(smoothed_activity) * 2)]
                activity_opto[i, 0:len(smoothed_activity)] = smoothed_activity
        activity = np.nanmean(activity, axis=0)
        activity_all[mouse, :] = activity
        activity_opto = np.nanmean(activity_opto, axis=0)
        activity_all_opto[mouse, :] = activity_opto
        if mouse < 5:
            x = range(1, len(activity) + 1)
            plt.plot(x[0:120], activity[0:120], c='k', alpha=.2)
        if mouse > 4:
            x = range(1, len(activity_opto) + 1)
            plt.plot(x[0:120], activity_opto[0:120], c='r', alpha=.2)

    mean = np.nanmean(activity_all, axis=0)
    sem_plus = mean + stats.sem(activity_all, axis=0, nan_policy='omit')
    sem_minus = mean - stats.sem(activity_all, axis=0, nan_policy='omit')
    plt.plot(x[0:120], mean[0:120], c='k', linewidth=3)
    plt.fill_between(x[0:120], sem_plus[0:120], sem_minus[0:120], alpha=0.2, color='k', lw=0)
    mean = np.nanmean(activity_all_opto, axis=0)
    sem_plus = mean + stats.sem(activity_all_opto, axis=0, nan_policy='omit')
    sem_minus = mean - stats.sem(activity_all_opto, axis=0, nan_policy='omit')
    plt.plot(x[0:120], mean[0:120], c='r', linewidth=3)
    plt.fill_between(x[0:120], sem_plus[0:120], sem_minus[0:120], alpha=0.2, color='r', lw=0)
    plt.ylabel('Stimulus activity\n(Normalized $\mathregular{Ca^{2+}}$transient $\mathregular{s^{-1}}$)')
    plt.ylim(0, 1)
    plt.xlim(1, 121)
    plt.xticks([1, 120], ['First trial', 'Last trial'])
    plt.yticks([0, .2, .4, .6, .8, 1])
    sns.despine()

    plt.savefig(paths['base_path'] + '/NN/plots/activity_across_trials.png', bbox_inches='tight', dpi=200)
    plt.savefig(paths['base_path'] + '/NN/plots/activity_across_trials.pdf', bbox_inches='tight', dpi=200,
                transparent=True)
    plt.close()

    sns.set(font_scale=.7)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(4, 6))
    plt.subplot(2, 2, 1)
    sns.heatmap(mean_activity_mice_heatmap, vmin=0, vmax=1.5, cmap="Reds", cbar=0)
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


def activity_across_trials_grouped(mice, sample_dates):
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(13.5, 9))
    plt.subplots_adjust(wspace=.35)

    plt.subplot(3, 3, 1)
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
                pd.DataFrame(activity_data[0][i][0]).rolling(3, min_periods=1, center=True).mean())
            smoothed_activity_same = np.concatenate(smoothed_activity_same, axis=0)
            smoothed_activity_increase = np.array(
                pd.DataFrame(activity_data[0][i][1]).rolling(3, min_periods=1, center=True).mean())
            smoothed_activity_increase = np.concatenate(smoothed_activity_increase, axis=0)
            smoothed_activity_decrease = np.array(
                pd.DataFrame(activity_data[0][i][2]).rolling(3, min_periods=1, center=True).mean())
            smoothed_activity_decrease = np.concatenate(smoothed_activity_decrease, axis=0)
            if mouse < 5:
                activity_same[i, 0:len(smoothed_activity_same)] = smoothed_activity_same
                activity_increase[i, 0:len(smoothed_activity_increase)] = smoothed_activity_increase
                activity_decrease[i, 0:len(smoothed_activity_decrease)] = smoothed_activity_decrease
            if mouse > 4:
                smoothed_activity_same = scipy.signal.resample_poly(smoothed_activity_same, 2, 1, padtype='smooth')[0:(len(smoothed_activity_same)*2)]
                smoothed_activity_increase = scipy.signal.resample_poly(smoothed_activity_increase, 2, 1, padtype='smooth')[0:(len(smoothed_activity_increase)*2)]
                smoothed_activity_decrease = scipy.signal.resample_poly(smoothed_activity_decrease, 2, 1, padtype='smooth')[0:(len(smoothed_activity_decrease)*2)]
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

    slope, intercept, r_value, p_value_dec, std_err = stats.linregress(
        np.concatenate([x[0:120], x[0:120], x[0:120], x[0:120], x[0:120], x[0:120], x[0:120], x[0:120]]), np.concatenate(activity_decrease_all[:, 0:120]))
    print(r_value)
    slope, intercept, r_value, p_value_inc, std_err = stats.linregress(
        np.concatenate([x[0:120], x[0:120], x[0:120], x[0:120], x[0:120], x[0:120], x[0:120], x[0:120]]), np.concatenate(activity_increase_all[:, 0:120]))
    print(r_value)
    anova_results = []
    anova_results.append(p_value_dec)
    anova_results.append(p_value_inc)
    anova_results_corrected = multipletests(anova_results, alpha=.05, method='holm')
    print(anova_results_corrected[1])

    mean = np.nanmean(activity_increase_all, axis=0)
    sem_plus = mean + stats.sem(activity_increase_all, axis=0, nan_policy='omit')
    sem_minus = mean - stats.sem(activity_increase_all, axis=0, nan_policy='omit')
    plt.plot(x[0:120], mean[0:120], c='darkred', linewidth=3)
    plt.fill_between(x[0:120], sem_plus[0:120], sem_minus[0:120], alpha=0.2, color='darkred', lw=0)
    mean = np.nanmean(activity_decrease_all, axis=0)
    sem_plus = mean + stats.sem(activity_decrease_all, axis=0, nan_policy='omit')
    sem_minus = mean - stats.sem(activity_decrease_all, axis=0, nan_policy='omit')
    plt.plot(x[0:120], mean[0:120], c='darkblue', linewidth=3)
    plt.fill_between(x[0:120], sem_plus[0:120], sem_minus[0:120], alpha=0.2, color='darkblue', lw=0)
    plt.ylabel('Response similarity\n(correlation between\nS1 and S2)')
    plt.ylim(-.15, .35)
    plt.yticks([-.1, 0, .1, .2, .3])
    plt.xlim(1, 121)
    plt.xticks([1, 120], ['First trial', 'Last trial'])
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
    sns.despine()

    activity_same_all = np.empty((len(mice), 128)) * np.nan
    activity_increase_all = np.empty((len(mice), 128)) * np.nan
    activity_decrease_all = np.empty((len(mice), 128)) * np.nan
    activity_same_all_opto = np.empty((len(mice), 128)) * np.nan
    activity_increase_all_opto = np.empty((len(mice), 128)) * np.nan
    activity_decrease_all_opto = np.empty((len(mice), 128)) * np.nan
    for mouse in range(0, len(mice)):
        paths = preprocess.create_folders(mice[mouse], sample_dates[mouse])
        activity_data = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/activity_grouped.npy',
                                allow_pickle=True)
        activity_same = np.empty((len(activity_data[0]), 128)) * np.nan
        activity_increase = np.empty((len(activity_data[0]), 128)) * np.nan
        activity_decrease = np.empty((len(activity_data[0]), 128)) * np.nan
        activity_same_opto = np.empty((len(activity_data[0]), 128)) * np.nan
        activity_increase_opto = np.empty((len(activity_data[0]), 128)) * np.nan
        activity_decrease_opto = np.empty((len(activity_data[0]), 128)) * np.nan
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
            if mouse < 5:
                activity_same[i, 0:len(smoothed_activity_same)] = smoothed_activity_same
                activity_increase[i, 0:len(smoothed_activity_increase)] = smoothed_activity_increase
                activity_decrease[i, 0:len(smoothed_activity_decrease)] = smoothed_activity_decrease
            if mouse > 4:
                smoothed_activity_same = scipy.signal.resample_poly(smoothed_activity_same, 2, 1, padtype='smooth')[
                                         0:(len(smoothed_activity_same) * 2)]
                smoothed_activity_increase = scipy.signal.resample_poly(smoothed_activity_increase, 2, 1,
                                                                        padtype='smooth')[
                                             0:(len(smoothed_activity_increase) * 2)]
                smoothed_activity_decrease = scipy.signal.resample_poly(smoothed_activity_decrease, 2, 1,
                                                                        padtype='smooth')[
                                             0:(len(smoothed_activity_decrease) * 2)]
                activity_same_opto[i, 0:len(smoothed_activity_same)] = smoothed_activity_same
                activity_increase_opto[i, 0:len(smoothed_activity_increase)] = smoothed_activity_increase
                activity_decrease_opto[i, 0:len(smoothed_activity_decrease)] = smoothed_activity_decrease
        activity_same = np.nanmean(activity_same, axis=0)
        activity_same_all[mouse, :] = activity_same
        activity_increase = np.nanmean(activity_increase, axis=0)
        activity_increase_all[mouse, :] = activity_increase
        activity_decrease = np.nanmean(activity_decrease, axis=0)
        activity_decrease_all[mouse, :] = activity_decrease
        activity_same_opto = np.nanmean(activity_same_opto, axis=0)
        activity_same_all_opto[mouse, :] = activity_same_opto
        activity_increase_opto = np.nanmean(activity_increase_opto, axis=0)
        activity_increase_all_opto[mouse, :] = activity_increase_opto
        activity_decrease_opto = np.nanmean(activity_decrease_opto, axis=0)
        activity_decrease_all_opto[mouse, :] = activity_decrease_opto
        x = range(1, len(activity_same) + 1)

    slope, intercept, r_value, p_value, std_err = stats.linregress(
        np.concatenate([x[0:120], x[0:120], x[0:120], x[0:120], x[0:120]]), np.concatenate(activity_increase_all[0:5, 0:120]))
    print(r_value)
    slope, intercept, r_value, p_value_opto, std_err = stats.linregress(
        np.concatenate([x[0:120], x[0:120], x[0:120]]), np.concatenate(activity_increase_all_opto[5:8, 0:120]))
    print(r_value)
    anova_results = []
    anova_results.append(p_value)
    anova_results.append(p_value_opto)
    anova_results_corrected = multipletests(anova_results, alpha=.05, method='holm')
    print(anova_results_corrected[1])

    slope, intercept, r_value, p_value, std_err = stats.linregress(
        np.concatenate([x[0:120], x[0:120], x[0:120], x[0:120], x[0:120]]),
        np.concatenate(activity_decrease_all[0:5, 0:120]))
    print(r_value)
    slope, intercept, r_value, p_value_opto, std_err = stats.linregress(
        np.concatenate([x[0:120], x[0:120], x[0:120]]), np.concatenate(activity_decrease_all_opto[5:8, 0:120]))
    print(r_value)
    anova_results = []
    anova_results.append(p_value)
    anova_results.append(p_value_opto)
    anova_results_corrected = multipletests(anova_results, alpha=.05, method='holm')
    print(anova_results_corrected[1])

    plt.subplot(3, 3, 2)
    mean = np.nanmean(activity_increase_all, axis=0)
    sem_plus = mean + stats.sem(activity_increase_all, axis=0, nan_policy='omit')
    sem_minus = mean - stats.sem(activity_increase_all, axis=0, nan_policy='omit')
    plt.plot(x[0:120], mean[0:120], c='darkred', linewidth=3)
    plt.fill_between(x[0:120], sem_plus[0:120], sem_minus[0:120], alpha=0.2, color='darkred', lw=0)
    mean = np.nanmean(activity_increase_all_opto, axis=0)
    sem_plus = mean + stats.sem(activity_increase_all_opto, axis=0, nan_policy='omit')
    sem_minus = mean - stats.sem(activity_increase_all_opto, axis=0, nan_policy='omit')
    plt.plot(x[0:120], mean[0:120], c='r', linewidth=3)
    plt.fill_between(x[0:120], sem_plus[0:120], sem_minus[0:120], alpha=0.2, color='r', lw=0)
    mean = np.nanmean(activity_decrease_all, axis=0)
    sem_plus = mean + stats.sem(activity_decrease_all, axis=0, nan_policy='omit')
    sem_minus = mean - stats.sem(activity_decrease_all, axis=0, nan_policy='omit')
    plt.ylabel('Response similarity\n(correlation between\nS1 and S2)')
    plt.ylim(-.15, .35)
    plt.yticks([-.1, 0, .1, .2, .3])
    plt.xlim(1, 121)
    plt.xticks([1, 120], ['First trial', 'Last trial'])
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
    sns.despine()
    plt.subplot(3, 3, 3)
    plt.plot(x[0:120], mean[0:120], c='darkblue', linewidth=3)
    plt.fill_between(x[0:120], sem_plus[0:120], sem_minus[0:120], alpha=0.2, color='darkblue', lw=0)
    mean = np.nanmean(activity_decrease_all_opto, axis=0)
    sem_plus = mean + stats.sem(activity_decrease_all_opto, axis=0, nan_policy='omit')
    sem_minus = mean - stats.sem(activity_decrease_all_opto, axis=0, nan_policy='omit')
    plt.plot(x[0:120], mean[0:120], c='r', linewidth=3)
    plt.fill_between(x[0:120], sem_plus[0:120], sem_minus[0:120], alpha=0.2, color='r', lw=0)
    plt.ylabel('Response similarity\n(correlation between\nS1 and S2)')
    plt.ylim(-.15, .35)
    plt.yticks([-.1, 0, .1, .2, .3])
    plt.xlim(1, 121)
    plt.xticks([1, 120], ['First trial', 'Last trial'])
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
    sns.despine()





    plt.subplot(3, 3, 4)
    activity_same_all = np.empty((len(mice), 128)) * np.nan
    activity_increase_all = np.empty((len(mice), 128)) * np.nan
    activity_decrease_all = np.empty((len(mice), 128)) * np.nan
    for mouse in range(0, len(mice)):
        paths = preprocess.create_folders(mice[mouse], sample_dates[mouse])
        session_data = preprocess.load_data(paths)
        fr = session_data['framerate']
        activity_data = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/activity_grouped.npy',
                                allow_pickle=True)
        activity_same = np.empty((len(activity_data[0]), 128)) * np.nan
        activity_increase = np.empty((len(activity_data[0]), 128)) * np.nan
        activity_decrease = np.empty((len(activity_data[0]), 128)) * np.nan
        for i in range(0, len(activity_data[0])):
            smoothed_activity_same = np.array(
                pd.DataFrame(activity_data[1][i][0]).rolling(3, min_periods=1, center=True).mean()) * fr
            smoothed_activity_same = np.concatenate(smoothed_activity_same, axis=0)
            smoothed_activity_increase = np.array(
                pd.DataFrame(activity_data[1][i][1]).rolling(3, min_periods=1, center=True).mean()) * fr
            smoothed_activity_increase = np.concatenate(smoothed_activity_increase, axis=0)
            smoothed_activity_decrease = np.array(
                pd.DataFrame(activity_data[1][i][2]).rolling(3, min_periods=1, center=True).mean()) * fr
            smoothed_activity_decrease = np.concatenate(smoothed_activity_decrease, axis=0)
            if mouse < 5:
                activity_same[i, 0:len(smoothed_activity_same)] = smoothed_activity_same
                activity_increase[i, 0:len(smoothed_activity_increase)] = smoothed_activity_increase
                activity_decrease[i, 0:len(smoothed_activity_decrease)] = smoothed_activity_decrease
            if mouse > 4:
                smoothed_activity_same = scipy.signal.resample_poly(smoothed_activity_same, 2, 1, padtype='smooth')[
                                         0:(len(smoothed_activity_same) * 2)]
                smoothed_activity_increase = scipy.signal.resample_poly(smoothed_activity_increase, 2, 1,
                                                                        padtype='smooth')[
                                             0:(len(smoothed_activity_increase) * 2)]
                smoothed_activity_decrease = scipy.signal.resample_poly(smoothed_activity_decrease, 2, 1,
                                                                        padtype='smooth')[
                                             0:(len(smoothed_activity_decrease) * 2)]
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

    anova_results = []
    anova_results.append(stats.ttest_rel(np.concatenate(activity_same_all[:, 0:3]),
                                         np.concatenate(activity_increase_all[:, 0:3]))[1])
    anova_results.append(stats.ttest_rel(np.concatenate(activity_same_all[:, 0:3]),
                                         np.concatenate(activity_decrease_all[:, 0:3]))[1])
    anova_results.append(stats.ttest_rel(np.concatenate(activity_same_all[:, 117:120]),
                                         np.concatenate(activity_increase_all[:, 117:120]))[1])
    anova_results.append(stats.ttest_rel(np.concatenate(activity_same_all[:, 117:120]),
                                         np.concatenate(activity_decrease_all[:, 117:120]))[1])
    anova_results_corrected = multipletests(anova_results, alpha=.05, method='holm')
    print(anova_results_corrected[1])

    mean = np.nanmean(activity_same_all, axis=0)
    sem_plus = mean + stats.sem(activity_same_all, axis=0, nan_policy='omit')
    sem_minus = mean - stats.sem(activity_same_all, axis=0, nan_policy='omit')
    plt.plot(x[0:120], mean[0:120], c='k', linewidth=3)
    plt.fill_between(x[0:120], sem_plus[0:120], sem_minus[0:120], alpha=0.2, color='k', lw=0)
    mean = np.nanmean(activity_increase_all, axis=0)
    sem_plus = mean + stats.sem(activity_increase_all, axis=0, nan_policy='omit')
    sem_minus = mean - stats.sem(activity_increase_all, axis=0, nan_policy='omit')
    plt.plot(x[0:120], mean[0:120], c='darkred', linewidth=3)
    plt.fill_between(x[0:120], sem_plus[0:120], sem_minus[0:120], alpha=0.2, color='darkred', lw=0)
    mean = np.nanmean(activity_decrease_all, axis=0)
    sem_plus = mean + stats.sem(activity_decrease_all, axis=0, nan_policy='omit')
    sem_minus = mean - stats.sem(activity_decrease_all, axis=0, nan_policy='omit')
    plt.plot(x[0:120], mean[0:120], c='darkblue', linewidth=3)
    plt.fill_between(x[0:120], sem_plus[0:120], sem_minus[0:120], alpha=0.2, color='darkblue', lw=0)
    plt.ylabel('Stimulus activity\n(Normalized $\mathregular{Ca^{2+}}$transient $\mathregular{s^{-1}}$)')
    plt.ylim(0, 1)
    plt.xlim(1, 121)
    plt.xticks([1, 120], ['First trial', 'Last trial'])
    sns.despine()

    activity_same_all = np.empty((len(mice), 128)) * np.nan
    activity_increase_all = np.empty((len(mice), 128)) * np.nan
    activity_decrease_all = np.empty((len(mice), 128)) * np.nan
    activity_same_all_opto = np.empty((len(mice), 128)) * np.nan
    activity_increase_all_opto = np.empty((len(mice), 128)) * np.nan
    activity_decrease_all_opto = np.empty((len(mice), 128)) * np.nan
    for mouse in range(0, len(mice)):
        paths = preprocess.create_folders(mice[mouse], sample_dates[mouse])
        activity_data = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/activity_grouped.npy',
                                allow_pickle=True)
        activity_same = np.empty((len(activity_data[0]), 128)) * np.nan
        activity_increase = np.empty((len(activity_data[0]), 128)) * np.nan
        activity_decrease = np.empty((len(activity_data[0]), 128)) * np.nan
        activity_same_opto = np.empty((len(activity_data[0]), 128)) * np.nan
        activity_increase_opto = np.empty((len(activity_data[0]), 128)) * np.nan
        activity_decrease_opto = np.empty((len(activity_data[0]), 128)) * np.nan
        for i in range(0, len(activity_data[0])):
            smoothed_activity_same = np.array(
                pd.DataFrame(activity_data[1][i][0]).rolling(3, min_periods=1, center=True).mean()) * fr
            smoothed_activity_same = np.concatenate(smoothed_activity_same, axis=0)
            smoothed_activity_increase = np.array(
                pd.DataFrame(activity_data[1][i][1]).rolling(3, min_periods=1, center=True).mean()) * fr
            smoothed_activity_increase = np.concatenate(smoothed_activity_increase, axis=0)
            smoothed_activity_decrease = np.array(
                pd.DataFrame(activity_data[1][i][2]).rolling(3, min_periods=1, center=True).mean()) * fr
            smoothed_activity_decrease = np.concatenate(smoothed_activity_decrease, axis=0)
            if mouse < 5:
                activity_same[i, 0:len(smoothed_activity_same)] = smoothed_activity_same
                activity_increase[i, 0:len(smoothed_activity_increase)] = smoothed_activity_increase
                activity_decrease[i, 0:len(smoothed_activity_decrease)] = smoothed_activity_decrease
            if mouse > 4:
                smoothed_activity_same = scipy.signal.resample_poly(smoothed_activity_same, 2, 1, padtype='smooth')[
                                         0:(len(smoothed_activity_same) * 2)]
                smoothed_activity_increase = scipy.signal.resample_poly(smoothed_activity_increase, 2, 1,
                                                                        padtype='smooth')[
                                             0:(len(smoothed_activity_increase) * 2)]
                smoothed_activity_decrease = scipy.signal.resample_poly(smoothed_activity_decrease, 2, 1,
                                                                        padtype='smooth')[
                                             0:(len(smoothed_activity_decrease) * 2)]
                activity_same_opto[i, 0:len(smoothed_activity_same)] = smoothed_activity_same
                activity_increase_opto[i, 0:len(smoothed_activity_increase)] = smoothed_activity_increase
                activity_decrease_opto[i, 0:len(smoothed_activity_decrease)] = smoothed_activity_decrease
        activity_same = np.nanmean(activity_same, axis=0)
        activity_same_all[mouse, :] = activity_same
        activity_increase = np.nanmean(activity_increase, axis=0)
        activity_increase_all[mouse, :] = activity_increase
        activity_decrease = np.nanmean(activity_decrease, axis=0)
        activity_decrease_all[mouse, :] = activity_decrease
        activity_same_opto = np.nanmean(activity_same_opto, axis=0)
        activity_same_all_opto[mouse, :] = activity_same_opto
        activity_increase_opto = np.nanmean(activity_increase_opto, axis=0)
        activity_increase_all_opto[mouse, :] = activity_increase_opto
        activity_decrease_opto = np.nanmean(activity_decrease_opto, axis=0)
        activity_decrease_all_opto[mouse, :] = activity_decrease_opto
        x = range(1, len(activity_same) + 1)

    plt.subplot(3, 3, 5)
    mean = np.nanmean(activity_same_all, axis=0)
    sem_plus = mean + stats.sem(activity_same_all, axis=0, nan_policy='omit')
    sem_minus = mean - stats.sem(activity_same_all, axis=0, nan_policy='omit')
    plt.plot(x[0:120], mean[0:120], c='k', linewidth=3)
    plt.fill_between(x[0:120], sem_plus[0:120], sem_minus[0:120], alpha=0.2, color='k', lw=0)
    mean = np.nanmean(activity_same_all_opto, axis=0)
    sem_plus = mean + stats.sem(activity_same_all_opto, axis=0, nan_policy='omit')
    sem_minus = mean - stats.sem(activity_same_all_opto, axis=0, nan_policy='omit')
    plt.plot(x[0:120], mean[0:120], c='r', linewidth=3)
    plt.fill_between(x[0:120], sem_plus[0:120], sem_minus[0:120], alpha=0.2, color='r', lw=0)
    plt.ylabel('Stimulus activity\n(Normalized $\mathregular{Ca^{2+}}$transient $\mathregular{s^{-1}}$)')
    plt.xlim(1, 121)
    plt.xticks([1, 120], ['First trial', 'Last trial'])
    plt.ylim(0, 1)
    plt.subplot(3, 3, 6)
    mean = np.nanmean(activity_increase_all, axis=0)
    sem_plus = mean + stats.sem(activity_increase_all, axis=0, nan_policy='omit')
    sem_minus = mean - stats.sem(activity_increase_all, axis=0, nan_policy='omit')
    plt.plot(x[0:120], mean[0:120], c='darkred', linewidth=3)
    plt.fill_between(x[0:120], sem_plus[0:120], sem_minus[0:120], alpha=0.2, color='darkred', lw=0)
    mean = np.nanmean(activity_increase_all_opto, axis=0)
    sem_plus = mean + stats.sem(activity_increase_all_opto, axis=0, nan_policy='omit')
    sem_minus = mean - stats.sem(activity_increase_all_opto, axis=0, nan_policy='omit')
    plt.plot(x[0:120], mean[0:120], c='r', linewidth=3)
    plt.fill_between(x[0:120], sem_plus[0:120], sem_minus[0:120], alpha=0.2, color='r', lw=0)
    plt.ylabel('Stimulus activity\n(Normalized $\mathregular{Ca^{2+}}$transient $\mathregular{s^{-1}}$)')
    plt.xlim(1, 121)
    plt.xticks([1, 120], ['First trial', 'Last trial'])
    plt.ylim(0, 1)
    plt.subplot(3, 3, 7)
    mean = np.nanmean(activity_decrease_all, axis=0)
    sem_plus = mean + stats.sem(activity_decrease_all, axis=0, nan_policy='omit')
    sem_minus = mean - stats.sem(activity_decrease_all, axis=0, nan_policy='omit')
    plt.plot(x[0:120], mean[0:120], c='darkblue', linewidth=3)
    plt.fill_between(x[0:120], sem_plus[0:120], sem_minus[0:120], alpha=0.2, color='darkblue', lw=0)
    mean = np.nanmean(activity_decrease_all_opto, axis=0)
    sem_plus = mean + stats.sem(activity_decrease_all_opto, axis=0, nan_policy='omit')
    sem_minus = mean - stats.sem(activity_decrease_all_opto, axis=0, nan_policy='omit')
    plt.plot(x[0:120], mean[0:120], c='r', linewidth=3)
    plt.fill_between(x[0:120], sem_plus[0:120], sem_minus[0:120], alpha=0.2, color='r', lw=0)
    plt.ylabel('Stimulus activity\n(Normalized $\mathregular{Ca^{2+}}$transient $\mathregular{s^{-1}}$)')
    plt.xlim(1, 121)
    plt.xticks([1, 120], ['First trial', 'Last trial'])
    plt.ylim(0, 1)
    sns.despine()

    plt.savefig(paths['base_path'] + '/NN/plots/activity_across_trials_grouped.png', bbox_inches='tight', dpi=200)
    plt.savefig(paths['base_path'] + '/NN/plots/activity_across_trials_grouped.pdf', bbox_inches='tight', dpi=200, transparent=True)
    plt.close()


def activity_across_trials_grouped_omit(mice, sample_dates):
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
        activity_data = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/activity_grouped_omit.npy',
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
            if mouse < 5:
                activity_same[i, 0:len(smoothed_activity_same)] = smoothed_activity_same
                activity_increase[i, 0:len(smoothed_activity_increase)] = smoothed_activity_increase - np.mean(smoothed_activity_increase[0:3])
                activity_decrease[i, 0:len(smoothed_activity_decrease)] = smoothed_activity_decrease - np.mean(smoothed_activity_decrease[0:3])
            if mouse > 4:
                smoothed_activity_same = scipy.signal.resample_poly(smoothed_activity_same, 2, 1, padtype='smooth')[
                                         0:(len(smoothed_activity_same) * 2)]
                smoothed_activity_increase = scipy.signal.resample_poly(smoothed_activity_increase, 2, 1,
                                                                        padtype='smooth')[
                                             0:(len(smoothed_activity_increase) * 2)]
                smoothed_activity_decrease = scipy.signal.resample_poly(smoothed_activity_decrease, 2, 1,
                                                                        padtype='smooth')[
                                             0:(len(smoothed_activity_decrease) * 2)]
                activity_same[i, 0:len(smoothed_activity_same)] = smoothed_activity_same
                activity_increase[i, 0:len(smoothed_activity_increase)] = smoothed_activity_increase - np.mean(smoothed_activity_increase[0:3])
                activity_decrease[i, 0:len(smoothed_activity_decrease)] = smoothed_activity_decrease - np.mean(smoothed_activity_decrease[0:3])
        activity_same = np.nanmean(activity_same, axis=0)
        activity_same_all[mouse, :] = activity_same
        activity_increase = np.nanmean(activity_increase, axis=0)
        activity_increase_all[mouse, :] = activity_increase
        activity_decrease = np.nanmean(activity_decrease, axis=0)
        activity_decrease_all[mouse, :] = activity_decrease
        x = range(1, len(activity_same)+1)

    print(stats.ttest_rel(np.mean(activity_increase_all[:,0:120], axis=1), np.mean(activity_decrease_all[:,0:120], axis=1))[1])


    mean = np.nanmean(activity_increase_all, axis=0)
    sem_plus = mean + stats.sem(activity_increase_all, axis=0, nan_policy='omit')
    sem_minus = mean - stats.sem(activity_increase_all, axis=0, nan_policy='omit')
    plt.plot(x[0:120], mean[0:120], c='darkred', linewidth=3)
    plt.fill_between(x[0:120], sem_plus[0:120], sem_minus[0:120], alpha=0.2, color='darkred', lw=0)
    mean = np.nanmean(activity_decrease_all, axis=0)
    sem_plus = mean + stats.sem(activity_decrease_all, axis=0, nan_policy='omit')
    sem_minus = mean - stats.sem(activity_decrease_all, axis=0, nan_policy='omit')
    plt.plot(x[0:120], mean[0:120], c='darkblue', linewidth=3)
    plt.fill_between(x[0:120], sem_plus[0:120], sem_minus[0:120], alpha=0.2, color='darkblue', lw=0)
    plt.ylabel('Response similarity\n(correlation between\nS1 and S2)')
    plt.ylim(-.2, .05)
    plt.xlim(1, 121)
    plt.xticks([1, 120], ['First trial', 'Last trial'])
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
    sns.despine()

    plt.savefig(paths['base_path'] + '/NN/plots/activity_across_trials_grouped_omit.png', bbox_inches='tight', dpi=200)
    plt.savefig(paths['base_path'] + '/NN/plots/activity_across_trials_grouped_omit.pdf', bbox_inches='tight', dpi=200, transparent=True)
    plt.close()


def activity_across_trials_grouped_separate(mice, sample_dates):
    activity_across_trials_grouped_separate_helper(mice, sample_dates, 'nochange')
    activity_across_trials_grouped_separate_helper(mice, sample_dates, 'increase')
    activity_across_trials_grouped_separate_helper(mice, sample_dates, 'decrease')


def activity_across_trials_grouped_separate_helper(mice, sample_dates, group_type):
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
        session_data = preprocess.load_data(paths)
        fr = session_data['framerate']
        activity_data = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/activity_grouped_' + group_type + '.npy',
                               allow_pickle=True)
        cs1d_cs1 = np.empty((len(activity_data[0]), 128)) * np.nan
        cs1d_cs2 = np.empty((len(activity_data[0]), 128)) * np.nan
        cs2d_cs2 = np.empty((len(activity_data[0]), 128)) * np.nan
        cs2d_cs1 = np.empty((len(activity_data[0]), 128)) * np.nan
        for i in range(0, len(activity_data[0])):
            if mouse < 5:
                cs1d_cs1[i, 0:len(activity_data[0][i])] = activity_data[0][i] * fr
                cs1d_cs2[i, 0:len(activity_data[1][i])] = activity_data[1][i] * fr
                cs2d_cs2[i, 0:len(activity_data[2][i])] = activity_data[2][i] * fr
                cs2d_cs1[i, 0:len(activity_data[3][i])] = activity_data[3][i] * fr
            if mouse > 4:
                vec_1 = scipy.signal.resample_poly(activity_data[0][i], 2, 1, padtype='smooth')[
                        0:(len(activity_data[0][i]) * 2)]
                cs1d_cs1[i, 0:len(vec_1)] = vec_1 * fr
                vec_2 = scipy.signal.resample_poly(activity_data[1][i], 2, 1, padtype='smooth')[
                        0:(len(activity_data[1][i]) * 2)]
                cs1d_cs2[i, 0:len(vec_2)] = vec_2 * fr
                vec_3 = scipy.signal.resample_poly(activity_data[2][i], 2, 1, padtype='smooth')[
                        0:(len(activity_data[2][i]) * 2)]
                cs2d_cs2[i, 0:len(vec_3)] = vec_3 * fr
                vec_4 = scipy.signal.resample_poly(activity_data[3][i], 2, 1, padtype='smooth')[
                        0:(len(activity_data[3][i]) * 2)]
                cs2d_cs1[i, 0:len(vec_4)] = vec_4 * fr
        cs1d_cs1 = np.nanmean(cs1d_cs1, axis=0)
        cs2d_cs2 = np.nanmean(cs2d_cs2, axis=0)
        cs1d_cs1_all[mouse, :] = (cs1d_cs1 + cs2d_cs2) / 2
        # cs1d_cs2 = np.nanmean(cs1d_cs2, axis=0)
        # cs1d_cs2_all[mouse, :] = cs1d_cs2
        # cs2d_cs2 = np.nanmean(cs2d_cs2, axis=0)
        # cs2d_cs2_all[mouse, :] = cs2d_cs2
        cs2d_cs1 = np.nanmean(cs2d_cs1, axis=0)
        cs1d_cs2 = np.nanmean(cs1d_cs2, axis=0)
        cs2d_cs1_all[mouse, :] = (cs2d_cs1 + cs1d_cs2) / 2
        x = range(1, len(cs1d_cs1)+1)

    print(stats.ttest_rel(np.abs(np.mean(cs1d_cs1_all[:, 0:3], axis=1)) - np.abs(np.mean(cs2d_cs1_all[:, 0:3], axis=1)),
                                         np.abs(np.mean(cs2d_cs1_all[:, 57:60], axis=1)) - np.abs(np.mean(cs1d_cs1_all[:, 57:60], axis=1)))[1])

    mean = np.nanmean(cs1d_cs1_all, axis=0)
    sem_plus = mean + stats.sem(cs1d_cs1_all, axis=0, nan_policy='omit')
    sem_minus = mean - stats.sem(cs1d_cs1_all, axis=0, nan_policy='omit')
    plt.plot(x[0:60], mean[0:60], c='darkslategray', linewidth=3)
    plt.fill_between(x[0:60], sem_plus[0:60], sem_minus[0:60], alpha=0.2, color='darkslategray', lw=0)
    # mean = np.nanmean(cs1d_cs2_all, axis=0)
    # sem_plus = mean + stats.sem(cs1d_cs2_all, axis=0, nan_policy='omit')
    # sem_minus = mean - stats.sem(cs1d_cs2_all, axis=0, nan_policy='omit')
    # plt.plot(x, mean, c='darkgreen', linewidth=3)
    # plt.fill_between(x, sem_plus, sem_minus, alpha=0.2, color='darkgreen', lw=0)
    # mean = np.nanmean(cs2d_cs2_all, axis=0)
    # sem_plus = mean + stats.sem(cs2d_cs2_all, axis=0, nan_policy='omit')
    # sem_minus = mean - stats.sem(cs2d_cs2_all, axis=0, nan_policy='omit')
    # plt.plot(x, mean, c='salmon', linewidth=3)
    # plt.fill_between(x, sem_plus, sem_minus, alpha=0.2, color='salmon', lw=0)
    mean = np.nanmean(cs2d_cs1_all, axis=0)
    sem_plus = mean + stats.sem(cs2d_cs1_all, axis=0, nan_policy='omit')
    sem_minus = mean - stats.sem(cs2d_cs1_all, axis=0, nan_policy='omit')
    plt.plot(x[0:60], mean[0:60], c='cadetblue', linewidth=3)
    plt.fill_between(x[0:60], sem_plus[0:60], sem_minus[0:60], alpha=0.2, color='cadetblue', lw=0)
    plt.ylabel('Stimulus activity\n(Normalized $\mathregular{Ca^{2+}}$transient $\mathregular{s^{-1}}$)')
    # plt.xlabel('Trial number')
    plt.ylim(0, 1)
    plt.xlim(1, 61)
    plt.xticks([1, 60], ['First trial', 'Last trial'])
    # plt.yticks([0, .05, .1])
    sns.despine()
    # label_1 = mlines.Line2D([], [], color='mediumseagreen', linewidth=2, label='S1 no change cells, S1 trials')
    # label_2 = mlines.Line2D([], [], color='darkgreen', linewidth=2, label='S1 no change cells, S2 trials')
    # label_3 = mlines.Line2D([], [], color='salmon', linewidth=2, label='S2 no change cells, S2 trials')
    # label_4 = mlines.Line2D([], [], color='darkred', linewidth=2, label='S2 no change cells, S1 trials')
    # plt.legend(handles=[label_1, label_2, label_3, label_4], frameon=False)

    cs1d_cs1_all = np.empty((len(mice), 128)) * np.nan
    cs1d_cs2_all = np.empty((len(mice), 128)) * np.nan
    cs2d_cs2_all = np.empty((len(mice), 128)) * np.nan
    cs2d_cs1_all = np.empty((len(mice), 128)) * np.nan
    cs1d_cs1_all_opto = np.empty((len(mice), 128)) * np.nan
    cs1d_cs2_all_opto = np.empty((len(mice), 128)) * np.nan
    cs2d_cs2_all_opto = np.empty((len(mice), 128)) * np.nan
    cs2d_cs1_all_opto = np.empty((len(mice), 128)) * np.nan
    for mouse in range(0, len(mice)):
        paths = preprocess.create_folders(mice[mouse], sample_dates[mouse])
        activity_data = np.load(
            paths['base_path'] + paths['mouse'] + '/data_across_days/activity_grouped_' + group_type + '.npy',
            allow_pickle=True)
        cs1d_cs1 = np.empty((len(activity_data[0]), 128)) * np.nan
        cs1d_cs2 = np.empty((len(activity_data[0]), 128)) * np.nan
        cs2d_cs2 = np.empty((len(activity_data[0]), 128)) * np.nan
        cs2d_cs1 = np.empty((len(activity_data[0]), 128)) * np.nan
        cs1d_cs1_opto = np.empty((len(activity_data[0]), 128)) * np.nan
        cs1d_cs2_opto = np.empty((len(activity_data[0]), 128)) * np.nan
        cs2d_cs2_opto = np.empty((len(activity_data[0]), 128)) * np.nan
        cs2d_cs1_opto = np.empty((len(activity_data[0]), 128)) * np.nan
        for i in range(0, len(activity_data[0])):
            if mouse < 5:
                cs1d_cs1[i, 0:len(activity_data[0][i])] = activity_data[0][i] * fr
                cs1d_cs2[i, 0:len(activity_data[1][i])] = activity_data[1][i] * fr
                cs2d_cs2[i, 0:len(activity_data[2][i])] = activity_data[2][i] * fr
                cs2d_cs1[i, 0:len(activity_data[3][i])] = activity_data[3][i] * fr
            if mouse > 4:
                vec_1 = scipy.signal.resample_poly(activity_data[0][i], 2, 1, padtype='smooth')[
                        0:(len(activity_data[0][i]) * 2)]
                cs1d_cs1_opto[i, 0:len(vec_1)] = vec_1 * fr
                vec_2 = scipy.signal.resample_poly(activity_data[1][i], 2, 1, padtype='smooth')[
                        0:(len(activity_data[1][i]) * 2)]
                cs1d_cs2_opto[i, 0:len(vec_2)] = vec_2 * fr
                vec_3 = scipy.signal.resample_poly(activity_data[2][i], 2, 1, padtype='smooth')[
                        0:(len(activity_data[2][i]) * 2)]
                cs2d_cs2_opto[i, 0:len(vec_3)] = vec_3 * fr
                vec_4 = scipy.signal.resample_poly(activity_data[3][i], 2, 1, padtype='smooth')[
                        0:(len(activity_data[3][i]) * 2)]
                cs2d_cs1_opto[i, 0:len(vec_4)] = vec_4 * fr
        cs1d_cs1 = np.nanmean(cs1d_cs1, axis=0)
        cs2d_cs2 = np.nanmean(cs2d_cs2, axis=0)
        cs1d_cs1_all[mouse, :] = (cs1d_cs1 + cs2d_cs2) / 2
        cs1d_cs2 = np.nanmean(cs1d_cs2, axis=0)
        # cs1d_cs2_all[mouse, :] = cs1d_cs2
        # cs2d_cs2 = np.nanmean(cs2d_cs2, axis=0)
        # cs2d_cs2_all[mouse, :] = cs2d_cs2
        cs2d_cs1 = np.nanmean(cs2d_cs1, axis=0)
        cs2d_cs1_all[mouse, :] = (cs2d_cs1 + cs1d_cs2) / 2
        cs1d_cs1_opto = np.nanmean(cs1d_cs1_opto, axis=0)
        cs2d_cs2_opto = np.nanmean(cs2d_cs2_opto, axis=0)
        cs1d_cs1_all_opto[mouse, :] = (cs1d_cs1_opto + cs2d_cs2_opto) / 2
        cs1d_cs2_opto = np.nanmean(cs1d_cs2_opto, axis=0)
        # cs1d_cs2_all_opto[mouse, :] = cs1d_cs2_opto
        # cs2d_cs2_opto = np.nanmean(cs2d_cs2_opto, axis=0)
        # cs2d_cs2_all_opto[mouse, :] = cs2d_cs2_opto
        cs2d_cs1_opto = np.nanmean(cs2d_cs1_opto, axis=0)
        cs2d_cs1_all_opto[mouse, :] = (cs2d_cs1_opto + cs1d_cs2_opto) / 2
        x = range(1, len(cs1d_cs1) + 1)

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

    plt.subplot(2, 2, 2)
    mean = np.nanmean(cs1d_cs1_all, axis=0)
    sem_plus = mean + stats.sem(cs1d_cs1_all, axis=0, nan_policy='omit')
    sem_minus = mean - stats.sem(cs1d_cs1_all, axis=0, nan_policy='omit')
    plt.plot(x[0:60], mean[0:60], c='darkslategray', linewidth=3)
    plt.fill_between(x[0:60], sem_plus[0:60], sem_minus[0:60], alpha=0.2, color='darkslategray', lw=0)
    # mean = np.nanmean(cs1d_cs2_all, axis=0)
    # sem_plus = mean + stats.sem(cs1d_cs2_all, axis=0, nan_policy='omit')
    # sem_minus = mean - stats.sem(cs1d_cs2_all, axis=0, nan_policy='omit')
    # plt.plot(x, mean, c='darkgreen', linewidth=3)
    # plt.fill_between(x, sem_plus, sem_minus, alpha=0.2, color='darkgreen', lw=0)
    mean = np.nanmean(cs1d_cs1_all_opto, axis=0)
    sem_plus = mean + stats.sem(cs1d_cs1_all_opto, axis=0, nan_policy='omit')
    sem_minus = mean - stats.sem(cs1d_cs1_all_opto, axis=0, nan_policy='omit')
    plt.plot(x[0:60], mean[0:60], c='darkred', linewidth=3)
    plt.fill_between(x[0:60], sem_plus[0:60], sem_minus[0:60], alpha=0.2, color='darkred', lw=0)
    # mean = np.nanmean(cs1d_cs2_all_opto, axis=0)
    # sem_plus = mean + stats.sem(cs1d_cs2_all_opto, axis=0, nan_policy='omit')
    # sem_minus = mean - stats.sem(cs1d_cs2_all_opto, axis=0, nan_policy='omit')
    # plt.plot(x, mean, c='purple', linewidth=3)
    # plt.fill_between(x, sem_plus, sem_minus, alpha=0.2, color='purple', lw=0)
    # plt.ylabel('Stimulus activity\n(Normalized $\mathregular{Ca^{2+}}$transient $\mathregular{s^{-1}}$)')
    # plt.xlabel('Trial number')
    # plt.ylim(0, .13)
    # plt.xlim(1, 60)
    # plt.xticks([1, 60], ['First trial', 'Last trial'])
    # plt.yticks([0, .05, .1])
    # plt.subplot(2, 2, 2)
    # mean = np.nanmean(cs2d_cs2_all, axis=0)
    # sem_plus = mean + stats.sem(cs2d_cs2_all, axis=0, nan_policy='omit')
    # sem_minus = mean - stats.sem(cs2d_cs2_all, axis=0, nan_policy='omit')
    # plt.plot(x, mean, c='cadetblue', linewidth=3)
    # plt.fill_between(x, sem_plus, sem_minus, alpha=0.2, color='salmon', lw=0)
    mean = np.nanmean(cs2d_cs1_all, axis=0)
    sem_plus = mean + stats.sem(cs2d_cs1_all, axis=0, nan_policy='omit')
    sem_minus = mean - stats.sem(cs2d_cs1_all, axis=0, nan_policy='omit')
    plt.plot(x[0:60], mean[0:60], c='cadetblue', linewidth=3)
    plt.fill_between(x[0:60], sem_plus[0:60], sem_minus[0:60], alpha=0.2, color='cadetblue', lw=0)
    # mean = np.nanmean(cs2d_cs2_all_opto, axis=0)
    # sem_plus = mean + stats.sem(cs2d_cs2_all_opto, axis=0, nan_policy='omit')
    # sem_minus = mean - stats.sem(cs2d_cs2_all_opto, axis=0, nan_policy='omit')
    # plt.plot(x, mean, c='hotpink', linewidth=3)
    # plt.fill_between(x, sem_plus, sem_minus, alpha=0.2, color='hotpink', lw=0)
    mean = np.nanmean(cs2d_cs1_all_opto, axis=0)
    sem_plus = mean + stats.sem(cs2d_cs1_all_opto, axis=0, nan_policy='omit')
    sem_minus = mean - stats.sem(cs2d_cs1_all_opto, axis=0, nan_policy='omit')
    plt.plot(x[0:60], mean[0:60], c='hotpink', linewidth=3)
    plt.fill_between(x[0:60], sem_plus[0:60], sem_minus[0:60], alpha=0.2, color='hotpink', lw=0)
    plt.ylabel('Stimulus activity\n(Normalized $\mathregular{Ca^{2+}}$transient $\mathregular{s^{-1}}$)')
    # plt.xlabel('Trial number')
    plt.ylim(0, 1.2)
    plt.xlim(1, 61)
    plt.xticks([1, 60], ['First trial', 'Last trial'])
    # plt.yticks([0, .05, .1])
    sns.despine()

    plt.savefig(paths['base_path'] + '/NN/plots/activity_across_trials_grouped_' + group_type + '.png', bbox_inches='tight', dpi=200)
    plt.savefig(paths['base_path'] + '/NN/plots/activity_across_trials_grouped_' + group_type + '.pdf', bbox_inches='tight', dpi=200, transparent=True)
    plt.close()


def activity_reactivation_correlation(mice, sample_dates):
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(2.5, 6))
    m_colors = ['b', 'teal', 'slategray', 'green', 'darkolivegreen', 'darkorange', 'purple', 'darkred']

    corr_all_mice = []
    plt.subplot(2, 2, 1)
    for mouse in range(0, len(mice)):
        paths = preprocess.create_folders(mice[mouse], sample_dates[mouse])
        activity_all = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/activity.npy', allow_pickle=True)
        corr_all = []
        for i in range(0, len(activity_all[0])):
            if mouse < 5:
                correlation = activity_all[0][i]
                reactivation_prob = activity_all[1][i]
            if mouse > 4:
                correlation = activity_all[0][i]
                correlation = scipy.signal.resample_poly(correlation, 2, 1, padtype='smooth')[0:(len(correlation) * 2)]
                reactivation_prob = activity_all[1][i]
                reactivation_prob = scipy.signal.resample_poly(reactivation_prob, 2, 1, padtype='smooth')[0:(len(reactivation_prob) * 2)]
            smoothed_correlation = np.array(pd.DataFrame(correlation).rolling(8, min_periods=1,
                                                                                        center=True).mean())
            smoothed_reactivation_prob = np.array(pd.DataFrame(reactivation_prob).rolling(8, min_periods=1,
                                                                                              center=True).mean())

            # sos = signal.butter(2, .1 / 60 / 2, btype='highpass', output='sos', fs=1 / 60)
            # smoothed_correlation = signal.sosfilt(sos, smoothed_correlation)
            # smoothed_reactivation_prob = signal.sosfilt(sos, smoothed_reactivation_prob)
            # smoothed_correlation = smoothed_correlation[5:len(smoothed_correlation)]
            # smoothed_reactivation_prob = smoothed_reactivation_prob[5:len(smoothed_reactivation_prob)]

            corr_temp = np.corrcoef(np.concatenate(smoothed_correlation, axis=0),
                                    np.concatenate(smoothed_reactivation_prob, axis=0))[0][1]
            corr_all.append(corr_temp)

            if mouse == 2 and i == 2:
                print(corr_temp)
                sns.set(font_scale=1)
                sns.set_style("whitegrid", {'axes.grid': False})
                sns.set_style("ticks")
                plt.figure(figsize=(6, 6))
                x = range(1, 121)
                ax1 = plt.subplot(2, 2, 1)
                ax1.spines['top'].set_visible(False)
                ax1.plot(x, smoothed_correlation, c='k')
                plt.ylabel('Response similarity\n(correlation between\nS1 and S2)')
                plt.xlabel('Trial number')
                plt.yticks([.05, .1, .15, .2, .25])
                ax2 = ax1.twinx()
                ax2.spines['top'].set_visible(False)
                ax2.plot(x, smoothed_reactivation_prob, c='darkorange')
                ax2.spines['right'].set_color('darkorange')
                ax2.tick_params(axis='y', colors='darkorange')
                plt.ylabel('Reactivation rate (probablity $\mathregular{s^{-1}}$)', rotation=270, c='darkorange', labelpad=15)
                plt.xlim(1, 120)
                plt.xticks([1, 20, 40, 60, 80, 100, 120])
                plt.yticks([0, .05, .1, .15, .2, .25])
                plt.savefig(paths['base_path'] + '/NN/plots/activity_reactivation_correlation_example.png',
                            bbox_inches='tight', dpi=200)
                plt.savefig(paths['base_path'] + '/NN/plots/activity_reactivation_correlation_example.pdf',
                            bbox_inches='tight', dpi=200, transparent=True)
                plt.close()

        plt.axhline(y=0, color='black', linestyle='--', linewidth=1, snap=False)
        y1 = np.mean(corr_all)
        plt.errorbar(1, y1, yerr=0, c=m_colors[mouse], marker='o', mfc='none', mec=m_colors[mouse], ms=5, alpha=.3)
        corr_all_mice.append(y1)

    print(stats.ttest_1samp(corr_all_mice, 0))

    y1 = np.mean(corr_all_mice)
    y1_err = stats.sem(corr_all_mice)
    plt.errorbar(1.2, y1, yerr=y1_err, c='k', linewidth=2, marker='o', mfc='k', mec='k',
                 ms=7, mew=0, zorder=100)
    plt.ylabel('Correlation between\nresponse similarity\nand reactivation probability')
    plt.xlim(.5, 1.5)
    plt.ylim(-.62, .62)
    plt.yticks([-.6, -.4, -.2, 0, .2, .4, .6])
    plt.xticks([])
    sns.despine()

    plt.subplot(2, 2, 2)
    for mouse in range(0, len(mice)):
        paths = preprocess.create_folders(mice[mouse], sample_dates[mouse])
        activity_all = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/activity.npy',
                               allow_pickle=True)
        corr_all = []
        for i in range(0, len(activity_all[0])):
            if mouse < 5:
                correlation = activity_all[0][i]
                reactivation_prob = activity_all[1][i]
            if mouse > 4:
                correlation = activity_all[0][i]
                correlation = scipy.signal.resample_poly(correlation, 2, 1, padtype='smooth')[0:(len(correlation) * 2)]
                reactivation_prob = activity_all[1][i]
                reactivation_prob = scipy.signal.resample_poly(reactivation_prob, 2, 1, padtype='smooth')[0:(len(reactivation_prob) * 2)]
            smoothed_correlation = np.array(pd.DataFrame(correlation).rolling(8, min_periods=1,
                                                                              center=True).mean())
            smoothed_reactivation_prob = np.array(pd.DataFrame(reactivation_prob).rolling(8, min_periods=1,
                                                                                          center=True).mean())

            sos = signal.butter(2, .1 / 60 / 2, btype='highpass', output='sos', fs=1 / 60)
            smoothed_correlation = signal.sosfilt(sos, smoothed_correlation)
            smoothed_reactivation_prob = signal.sosfilt(sos, smoothed_reactivation_prob)
            smoothed_correlation = smoothed_correlation[5:len(smoothed_correlation)]
            smoothed_reactivation_prob = smoothed_reactivation_prob[5:len(smoothed_reactivation_prob)]

            corr_temp = np.corrcoef(np.concatenate(smoothed_correlation, axis=0),
                                    np.concatenate(smoothed_reactivation_prob, axis=0))[0][1]
            corr_all.append(corr_temp)

        plt.axhline(y=0, color='black', linestyle='--', linewidth=1, snap=False)
        y1 = np.mean(corr_all)
        plt.errorbar(1, y1, yerr=0, c=m_colors[mouse], marker='o', mfc='none', mec=m_colors[mouse], ms=5, alpha=.3)
        corr_all_mice.append(y1)

    print(stats.ttest_1samp(corr_all_mice, 0))

    y1 = np.mean(corr_all_mice)
    y1_err = stats.sem(corr_all_mice)
    plt.errorbar(1.2, y1, yerr=y1_err, c='k', linewidth=2, marker='o', mfc='k', mec='k',
                 ms=7, mew=0, zorder=100)
    plt.ylabel('Correlation between\nresponse similarity\nand reactivation probability')
    plt.xlim(.5, 1.5)
    plt.ylim(-.62, .62)
    plt.yticks([-.6, -.4, -.2, 0, .2, .4, .6])
    plt.xticks([])
    sns.despine()
    plt.savefig(paths['base_path'] + '/NN/plots/activity_reactivation_correlation.png', bbox_inches='tight',
                dpi=200)
    plt.savefig(paths['base_path'] + '/NN/plots/activity_reactivation_correlation.pdf', bbox_inches='tight',
                dpi=200, transparent=True)
    plt.close()
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(7, 6))
    plt.subplot(2, 2, 1)
    maxlags = 10
    xcorr_correlation_all = np.zeros((len(mice), (maxlags * 2) + 1))
    for mouse in range(0, len(mice)):
        paths = preprocess.create_folders(mice[mouse], sample_dates[mouse])
        activity_all = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/activity.npy',
                               allow_pickle=True)
        xcorr_correlation = np.zeros((len(activity_all[0]), (maxlags * 2) + 1))
        for i in range(0, len(activity_all[0])):
            if mouse < 5:
                correlation = activity_all[0][i]
                reactivation_prob = activity_all[1][i]
            if mouse > 4:
                correlation = activity_all[0][i]
                correlation = scipy.signal.resample_poly(correlation, 2, 1, padtype='smooth')[0:(len(correlation) * 2)]
                reactivation_prob = activity_all[1][i]
                reactivation_prob = scipy.signal.resample_poly(reactivation_prob, 2, 1, padtype='smooth')[0:(len(reactivation_prob) * 2)]
            smoothed_correlation = np.array(pd.DataFrame(correlation).rolling(8, min_periods=1,
                                                                                        center=True).mean())
            smoothed_reactivation_prob = np.array(pd.DataFrame(reactivation_prob).rolling(8, min_periods=1,
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

            smoothed_correlation = smoothed_correlation-np.mean(smoothed_correlation)
            smoothed_reactivation_prob = smoothed_reactivation_prob-np.mean(smoothed_reactivation_prob)

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
    plt.ylim(-.5, 1)
    sns.despine()

    plt.subplot(2, 2, 2)
    maxlags = 10
    xcorr_correlation_all = np.zeros((len(mice), (maxlags * 2) + 1))
    for mouse in range(0, 5):
        paths = preprocess.create_folders(mice[mouse], sample_dates[mouse])
        activity_all = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/activity.npy',
                               allow_pickle=True)
        xcorr_correlation = np.zeros((len(activity_all[0]), (maxlags * 2) + 1))
        for i in range(0, len(activity_all[0])):
            if mouse < 5:
                correlation = activity_all[0][i]
                reactivation_prob = activity_all[1][i]
            if mouse > 4:
                correlation = activity_all[0][i]
                correlation = scipy.signal.resample_poly(correlation, 2, 1, padtype='smooth')[0:(len(correlation) * 2)]
                reactivation_prob = activity_all[1][i]
                reactivation_prob = scipy.signal.resample_poly(reactivation_prob, 2, 1, padtype='smooth')[
                                    0:(len(reactivation_prob) * 2)]
            smoothed_correlation = np.array(pd.DataFrame(correlation).rolling(8, min_periods=1,
                                                                              center=True).mean())
            smoothed_reactivation_prob = np.array(pd.DataFrame(reactivation_prob).rolling(8, min_periods=1,
                                                                                          center=True).mean())
            smoothed_correlation = np.concatenate(smoothed_correlation, axis=0)
            smoothed_reactivation_prob = np.concatenate(smoothed_reactivation_prob, axis=0)

            sos = signal.butter(2, .1 / 60 / 2, btype='highpass', output='sos', fs=1 / 60)
            smoothed_correlation = signal.sosfilt(sos, smoothed_correlation)
            smoothed_reactivation_prob = signal.sosfilt(sos, smoothed_reactivation_prob)
            smoothed_correlation = smoothed_correlation[5:len(smoothed_correlation)]
            smoothed_reactivation_prob = smoothed_reactivation_prob[5:len(smoothed_reactivation_prob)]
            smoothed_correlation = smoothed_correlation - np.mean(smoothed_correlation)
            smoothed_reactivation_prob = smoothed_reactivation_prob - np.mean(smoothed_reactivation_prob)

            smoothed_correlation = smoothed_correlation - np.mean(smoothed_correlation)
            smoothed_reactivation_prob = smoothed_reactivation_prob - np.mean(smoothed_reactivation_prob)

            corr = signal.correlate(smoothed_correlation, smoothed_reactivation_prob)
            corr = corr / np.max(corr)
            num_trials = int((len(corr) - 1) / 2)
            xcorr_correlation[i, :] = corr[num_trials - maxlags:num_trials + maxlags + 1]

        mean = xcorr_correlation.mean(axis=0)
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
    plt.ylim(-.5, 1)
    sns.despine()

    plt.subplot(2, 2, 2)
    maxlags = 10
    xcorr_correlation_all = np.zeros((len(mice), (maxlags * 2) + 1))
    for mouse in range(5, len(mice)):
        paths = preprocess.create_folders(mice[mouse], sample_dates[mouse])
        activity_all = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/activity.npy',
                               allow_pickle=True)
        xcorr_correlation = np.zeros((len(activity_all[0]), (maxlags * 2) + 1))
        for i in range(0, len(activity_all[0])):
            if mouse < 5:
                correlation = activity_all[0][i]
                reactivation_prob = activity_all[1][i]
            if mouse > 4:
                correlation = activity_all[0][i]
                correlation = scipy.signal.resample_poly(correlation, 2, 1, padtype='smooth')[0:(len(correlation) * 2)]
                reactivation_prob = activity_all[1][i]
                reactivation_prob = scipy.signal.resample_poly(reactivation_prob, 2, 1, padtype='smooth')[
                                    0:(len(reactivation_prob) * 2)]
            smoothed_correlation = np.array(pd.DataFrame(correlation).rolling(8, min_periods=1,
                                                                              center=True).mean())
            smoothed_reactivation_prob = np.array(pd.DataFrame(reactivation_prob).rolling(8, min_periods=1,
                                                                                          center=True).mean())
            smoothed_correlation = np.concatenate(smoothed_correlation, axis=0)
            smoothed_reactivation_prob = np.concatenate(smoothed_reactivation_prob, axis=0)

            sos = signal.butter(2, .1 / 60 / 2, btype='highpass', output='sos', fs=1 / 60)
            smoothed_correlation = signal.sosfilt(sos, smoothed_correlation)
            smoothed_reactivation_prob = signal.sosfilt(sos, smoothed_reactivation_prob)
            smoothed_correlation = smoothed_correlation[5:len(smoothed_correlation)]
            smoothed_reactivation_prob = smoothed_reactivation_prob[5:len(smoothed_reactivation_prob)]
            smoothed_correlation = smoothed_correlation - np.mean(smoothed_correlation)
            smoothed_reactivation_prob = smoothed_reactivation_prob - np.mean(smoothed_reactivation_prob)

            smoothed_correlation = smoothed_correlation - np.mean(smoothed_correlation)
            smoothed_reactivation_prob = smoothed_reactivation_prob - np.mean(smoothed_reactivation_prob)

            corr = signal.correlate(smoothed_correlation, smoothed_reactivation_prob)
            corr = corr / np.max(corr)
            num_trials = int((len(corr) - 1) / 2)
            xcorr_correlation[i, :] = corr[num_trials - maxlags:num_trials + maxlags + 1]

        mean = xcorr_correlation.mean(axis=0)
        xcorr_correlation_all[mouse, :] = mean
    mean = xcorr_correlation_all.mean(axis=0)
    sem_plus = mean + stats.sem(xcorr_correlation_all, axis=0)
    sem_minus = mean - stats.sem(xcorr_correlation_all, axis=0)
    plt.plot(range(-maxlags, maxlags + 1), mean, '-', c='r', linewidth=3)
    plt.fill_between(range(-maxlags, maxlags + 1), sem_plus, sem_minus, alpha=0.2, color='r', lw=0)
    plt.axvline(0, color='black', linestyle='--', linewidth=1, snap=False)
    plt.ylabel('Cross correlation between\nstimulus orthogonalization\nand reactivation probability')
    plt.xlabel('Shift in trial')
    plt.xticks([-10, -5, 0, 5, 10])
    plt.ylim(-.25, .5)
    sns.despine()

    plt.savefig(paths['base_path'] + '/NN/plots/activity_reactivation_correlation_cross.png', bbox_inches='tight', dpi=200)
    plt.savefig(paths['base_path'] + '/NN/plots/activity_reactivation_correlation_cross.pdf', bbox_inches='tight', dpi=200, transparent=True)
    plt.close()


def reactivation_spatial(mice, sample_dates):
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(wspace=.45)
    m_colors = ['b', 'teal', 'slategray', 'green', 'darkolivegreen', 'darkorange', 'purple', 'darkred']
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

    [_, s_p_value] = stats.shapiro(area_por_cue_all)
    print(s_p_value)

    print(stats.f_oneway(area_li_cue_all, area_por_cue_all, area_p_cue_all, area_lm_cue_all))
    print(pairwise_tukeyhsd(np.concatenate([area_li_cue_all, area_por_cue_all, area_p_cue_all, area_lm_cue_all], axis=0),
                            ['li', 'li', 'li', 'li', 'li', 'li', 'li', 'li', 'por', 'por', 'por', 'por', 'por', 'por', 'por', 'por', 'p', 'p', 'p', 'p','p', 'p', 'p', 'p', 'lm', 'lm', 'lm', 'lm', 'lm', 'lm', 'lm', 'lm'], alpha=0.05))
    print(stats.f_oneway(area_li_reactivation_all, area_por_reactivation_all, area_p_reactivation_all, area_lm_reactivation_all))
    print(
        pairwise_tukeyhsd(np.concatenate([area_li_reactivation_all, area_por_reactivation_all, area_p_reactivation_all, area_lm_reactivation_all], axis=0),
                          ['li', 'li', 'li', 'li', 'li', 'li', 'li', 'li', 'por', 'por', 'por', 'por', 'por', 'por',
                           'por', 'por', 'p', 'p', 'p', 'p', 'p', 'p', 'p', 'p', 'lm', 'lm', 'lm', 'lm', 'lm', 'lm',
                           'lm', 'lm'], alpha=0.05))
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
    plt.ylim(0, .9)
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
    plt.ylim(0, .9)
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
    for mouse in range(0, len(mice)):
        paths = preprocess.create_folders(mice[mouse], sample_dates[mouse])
        reactivation_cue_pca_vec = np.load(paths['base_path'] + paths['mouse'] +
                                           '/data_across_days/reactivation_cue_vector.npy', allow_pickle=True)
        temp_s1 = np.zeros((len(reactivation_cue_pca_vec[0]), 60))
        temp_s1r = np.zeros((len(reactivation_cue_pca_vec[0]), 60))
        temp_s2 = np.zeros((len(reactivation_cue_pca_vec[0]), 60))
        temp_s2r = np.zeros((len(reactivation_cue_pca_vec[0]), 60))
        for i in range(0, len(reactivation_cue_pca_vec[0])):
            if mouse < 5:
                temp_s1[i, :] = reactivation_cue_pca_vec[0][i]
                temp_s1r[i, :] = reactivation_cue_pca_vec[1][i]
                temp_s2[i, :] = reactivation_cue_pca_vec[2][i]
                temp_s2r[i, :] = reactivation_cue_pca_vec[3][i]
            if mouse > 4:
                temp_s1[i, :] = scipy.signal.resample_poly(reactivation_cue_pca_vec[0][i], 2, 1, padtype='smooth')[0:(len(reactivation_cue_pca_vec[0][i])*2)]
                temp_s1r[i, :] = scipy.signal.resample_poly(reactivation_cue_pca_vec[1][i], 2, 1, padtype='smooth')[0:(len(reactivation_cue_pca_vec[1][i])*2)]
                temp_s2[i, :] = scipy.signal.resample_poly(reactivation_cue_pca_vec[2][i], 2, 1, padtype='smooth')[0:(len(reactivation_cue_pca_vec[2][i])*2)]
                temp_s2r[i, :] = scipy.signal.resample_poly(reactivation_cue_pca_vec[3][i], 2, 1, padtype='smooth')[0:(len(reactivation_cue_pca_vec[3][i])*2)]
        y_s1 = np.nanmean(temp_s1, axis=0)
        y_s1r = np.nanmean(temp_s1r, axis=0)
        y_s2 = np.nanmean(temp_s2, axis=0)
        y_s2r = np.nanmean(temp_s2r, axis=0)
        all_s1[mouse, :] = y_s1
        all_s1r[mouse, :] = y_s1r
        all_s2[mouse, :] = y_s2
        all_s2r[mouse, :] = y_s2r

    print(stats.ttest_rel(np.abs(np.mean(all_s1[:, 0:3], axis=1))-np.abs(np.mean(all_s1r[:, 0:3], axis=1)), np.abs(np.mean(all_s1[:, 57:60], axis=1)) - np.abs(np.mean(all_s1r[:, 57:60], axis=1)))[1])
    print(stats.ttest_rel(np.abs(np.mean(all_s2[:, 0:3], axis=1))-np.abs(np.mean(all_s2r[:, 0:3], axis=1)), np.abs(np.mean(all_s2[:, 57:60], axis=1)) - np.abs(np.mean(all_s2r[:, 57:60], axis=1)))[1])

    plt.subplot(2, 2, 1)
    plt.ylim(-1.35, .2)
    plt.yticks([0, -.5, -1])
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
    plt.xlim(0, 60)
    plt.xticks([0, 59], ['First trial', 'Last trial'])
    plt.subplot(2, 2, 2)
    plt.ylim(-1.35, .2)
    plt.yticks([0, -.5, -1])
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
    plt.xlim(0, 60)
    plt.xticks([0, 59], ['First trial', 'Last trial'])

    paths = preprocess.create_folders(mice[0], sample_dates[0])
    reactivation_cue_pca_vec = np.load(paths['base_path'] + paths['mouse'] +
                                       '/data_across_days/reactivation_cue_vector.npy', allow_pickle=True)
    all_s1 = np.empty((len(mice), len(reactivation_cue_pca_vec[0][0]))) * np.nan
    all_s1r = np.empty((len(mice), len(reactivation_cue_pca_vec[0][0]))) * np.nan
    all_s2 = np.empty((len(mice), len(reactivation_cue_pca_vec[0][0]))) * np.nan
    all_s2r = np.empty((len(mice), len(reactivation_cue_pca_vec[0][0]))) * np.nan
    all_s1_opto = np.empty((len(mice), len(reactivation_cue_pca_vec[0][0]))) * np.nan
    all_s1r_opto = np.empty((len(mice), len(reactivation_cue_pca_vec[0][0]))) * np.nan
    all_s2_opto = np.empty((len(mice), len(reactivation_cue_pca_vec[0][0]))) * np.nan
    all_s2r_opto = np.empty((len(mice), len(reactivation_cue_pca_vec[0][0]))) * np.nan
    for mouse in range(0, len(mice)):
        paths = preprocess.create_folders(mice[mouse], sample_dates[mouse])
        reactivation_cue_pca_vec = np.load(paths['base_path'] + paths['mouse'] +
                                           '/data_across_days/reactivation_cue_vector.npy', allow_pickle=True)
        temp_s1 = np.empty((len(reactivation_cue_pca_vec[0]), 60)) * np.nan
        temp_s1r = np.empty((len(reactivation_cue_pca_vec[0]), 60)) * np.nan
        temp_s2 = np.empty((len(reactivation_cue_pca_vec[0]), 60)) * np.nan
        temp_s2r = np.empty((len(reactivation_cue_pca_vec[0]), 60)) * np.nan
        temp_s1_opto = np.empty((len(reactivation_cue_pca_vec[0]), 60)) * np.nan
        temp_s1r_opto = np.empty((len(reactivation_cue_pca_vec[0]), 60)) * np.nan
        temp_s2_opto = np.empty((len(reactivation_cue_pca_vec[0]), 60)) * np.nan
        temp_s2r_opto = np.empty((len(reactivation_cue_pca_vec[0]), 60)) * np.nan
        for i in range(0, len(reactivation_cue_pca_vec[0])):
            if mouse < 5:
                temp_s1[i, :] = reactivation_cue_pca_vec[0][i]
                temp_s1r[i, :] = reactivation_cue_pca_vec[1][i]
                temp_s2[i, :] = reactivation_cue_pca_vec[2][i]
                temp_s2r[i, :] = reactivation_cue_pca_vec[3][i]
            if mouse > 4:
                temp_s1_opto[i, :] = scipy.signal.resample_poly(reactivation_cue_pca_vec[0][i], 2, 1, padtype='smooth')[
                                0:(len(reactivation_cue_pca_vec[0][i]) * 2)]
                temp_s1r_opto[i, :] = scipy.signal.resample_poly(reactivation_cue_pca_vec[1][i], 2, 1, padtype='smooth')[
                                 0:(len(reactivation_cue_pca_vec[1][i]) * 2)]
                temp_s2_opto[i, :] = scipy.signal.resample_poly(reactivation_cue_pca_vec[2][i], 2, 1, padtype='smooth')[
                                0:(len(reactivation_cue_pca_vec[2][i]) * 2)]
                temp_s2r_opto[i, :] = scipy.signal.resample_poly(reactivation_cue_pca_vec[3][i], 2, 1, padtype='smooth')[
                                 0:(len(reactivation_cue_pca_vec[3][i]) * 2)]
        y_s1 = np.nanmean(temp_s1, axis=0)
        y_s1r = np.nanmean(temp_s1r, axis=0)
        y_s2 = np.nanmean(temp_s2, axis=0)
        y_s2r = np.nanmean(temp_s2r, axis=0)
        all_s1[mouse, :] = y_s1
        all_s1r[mouse, :] = y_s1r
        all_s2[mouse, :] = y_s2
        all_s2r[mouse, :] = y_s2r
        y_s1_opto = np.nanmean(temp_s1_opto, axis=0)
        y_s1r_opto = np.nanmean(temp_s1r_opto, axis=0)
        y_s2_opto = np.nanmean(temp_s2_opto, axis=0)
        y_s2r_opto = np.nanmean(temp_s2r_opto, axis=0)
        all_s1_opto[mouse, :] = y_s1_opto
        all_s1r_opto[mouse, :] = y_s1r_opto
        all_s2_opto[mouse, :] = y_s2_opto
        all_s2r_opto[mouse, :] = y_s2r_opto

    # print(stats.ttest_rel(all_s2[:, 0:1], all_s2r[:, 59:60])[1])

    plt.subplot(2, 2, 3)
    plt.ylim(-1.5, .4)
    plt.yticks([0, -.5, -1, -1.5])
    plt.gca().invert_yaxis()
    mean = np.nanmean(all_s1, axis=0)
    sem_plus = mean + stats.sem(all_s1, axis=0, nan_policy='omit')
    sem_minus = mean - stats.sem(all_s1, axis=0, nan_policy='omit')
    plt.plot(x_label, mean, '-', c='darkgreen', linewidth=3)
    plt.fill_between(x_label, sem_plus, sem_minus, alpha=0.2, color='darkgreen', lw=0)
    mean = np.nanmean(all_s1r, axis=0)
    sem_plus = mean + stats.sem(all_s1r, axis=0, nan_policy='omit')
    sem_minus = mean - stats.sem(all_s1r, axis=0, nan_policy='omit')
    plt.plot(x_label, mean, '-', c='lime', linewidth=3)
    plt.fill_between(x_label, sem_plus, sem_minus, alpha=0.2, color='lime', lw=0)
    mean = np.nanmean(all_s1_opto, axis=0)
    sem_plus = mean + stats.sem(all_s1_opto, axis=0, nan_policy='omit')
    sem_minus = mean - stats.sem(all_s1_opto, axis=0, nan_policy='omit')
    plt.plot(x_label, mean, '-', c='purple', linewidth=3)
    plt.fill_between(x_label, sem_plus, sem_minus, alpha=0.2, color='purple', lw=0)
    mean = np.nanmean(all_s1r_opto, axis=0)
    sem_plus = mean + stats.sem(all_s1r_opto, axis=0, nan_policy='omit')
    sem_minus = mean - stats.sem(all_s1r_opto, axis=0, nan_policy='omit')
    plt.plot(x_label, mean, '-', c='orange', linewidth=3)
    plt.fill_between(x_label, sem_plus, sem_minus, alpha=0.2, color='orange', lw=0)
    plt.ylabel('Similarity to early vs. late\n S1 response pattern')
    plt.xlim(0, 60)
    plt.xticks([0, 59], ['First trial', 'Last trial'])
    plt.subplot(2, 2, 4)
    plt.ylim(-1.5, .4)
    plt.yticks([0, -.5, -1, -1.5])
    plt.gca().invert_yaxis()
    mean = np.nanmean(all_s2, axis=0)
    sem_plus = mean + stats.sem(all_s2, axis=0, nan_policy='omit')
    sem_minus = mean - stats.sem(all_s2, axis=0, nan_policy='omit')
    plt.plot(x_label, mean, '-', c='darkred', linewidth=3)
    plt.fill_between(x_label, sem_plus, sem_minus, alpha=0.2, color='darkred', lw=0)
    mean = np.nanmean(all_s2r, axis=0)
    sem_plus = mean + stats.sem(all_s2r, axis=0, nan_policy='omit')
    sem_minus = mean - stats.sem(all_s2r, axis=0, nan_policy='omit')
    plt.plot(x_label, mean, '-', c='hotpink', linewidth=3)
    plt.fill_between(x_label, sem_plus, sem_minus, alpha=0.2, color='hotpink', lw=0)
    mean = np.nanmean(all_s2_opto, axis=0)
    sem_plus = mean + stats.sem(all_s2_opto, axis=0, nan_policy='omit')
    sem_minus = mean - stats.sem(all_s2_opto, axis=0, nan_policy='omit')
    plt.plot(x_label, mean, '-', c='purple', linewidth=3)
    plt.fill_between(x_label, sem_plus, sem_minus, alpha=0.2, color='purple', lw=0)
    mean = np.nanmean(all_s2r_opto, axis=0)
    sem_plus = mean + stats.sem(all_s2r_opto, axis=0, nan_policy='omit')
    sem_minus = mean - stats.sem(all_s2r_opto, axis=0, nan_policy='omit')
    plt.plot(x_label, mean, '-', c='orange', linewidth=3)
    plt.fill_between(x_label, sem_plus, sem_minus, alpha=0.2, color='orange', lw=0)
    plt.ylabel('Similarity to early vs. late\n S2 response pattern)')
    plt.xlim(0, 60)
    plt.xticks([0, 59], ['First trial', 'Last trial'])
    sns.despine()
    plt.savefig(paths['base_path'] + '/NN/plots/reactivation_cue_vector.png', bbox_inches='tight', dpi=200)
    plt.savefig(paths['base_path'] + '/NN/plots/reactivation_cue_vector.pdf', bbox_inches='tight', dpi=200, transparent=True)
    #plt.close()


def reactivation_cue_vector_evolve(mice, sample_dates):
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(9, 9))
    plt.subplots_adjust(wspace=.3)
    x_label = list(range(0, 60))
    all_s1 = np.zeros((len(mice), 60))
    all_s1r = np.zeros((len(mice), 60))
    all_s2 = np.zeros((len(mice), 60))
    all_s2r = np.zeros((len(mice), 60))
    for mouse in range(0, len(mice)):
        paths = preprocess.create_folders(mice[mouse], sample_dates[mouse])
        reactivation_cue_pca_vec = np.load(paths['base_path'] + paths['mouse'] +
                                           '/data_across_days/reactivation_cue_vector_evolve.npy', allow_pickle=True)
        temp_s1 = np.zeros((len(reactivation_cue_pca_vec[0]), 60))
        temp_s1r = np.zeros((len(reactivation_cue_pca_vec[0]), 60))
        temp_s2 = np.zeros((len(reactivation_cue_pca_vec[0]), 60))
        temp_s2r = np.zeros((len(reactivation_cue_pca_vec[0]), 60))
        for i in range(0, len(reactivation_cue_pca_vec[0])):
            if mouse < 5:
                temp_s1r[i, :] = reactivation_cue_pca_vec[0][i]
                temp_s1[i, :] = reactivation_cue_pca_vec[2][i]
                temp_s2r[i, :] = reactivation_cue_pca_vec[1][i]
                temp_s2[i, :] = reactivation_cue_pca_vec[3][i]
            if mouse > 4:
                temp_s1r[i, :] = scipy.signal.resample_poly(reactivation_cue_pca_vec[0][i], 2, 1, padtype='smooth')[
                                0:(len(reactivation_cue_pca_vec[0][i]) * 2)]
                temp_s1[i, :] = scipy.signal.resample_poly(reactivation_cue_pca_vec[2][i], 2, 1, padtype='smooth')[
                                 0:(len(reactivation_cue_pca_vec[2][i]) * 2)]
                temp_s2r[i, :] = scipy.signal.resample_poly(reactivation_cue_pca_vec[1][i], 2, 1, padtype='smooth')[
                                0:(len(reactivation_cue_pca_vec[1][i]) * 2)]
                temp_s2[i, :] = scipy.signal.resample_poly(reactivation_cue_pca_vec[3][i], 2, 1, padtype='smooth')[
                                 0:(len(reactivation_cue_pca_vec[3][i]) * 2)]
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

    plt.subplot(3, 2, 1)
    plt.ylim(-1.35, .2)
    plt.yticks([0, -.5, -1])
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
    label_1 = mlines.Line2D([], [], color='darkgreen', linewidth=2, label='S1')
    label_2 = mlines.Line2D([], [], color='lime', linewidth=2, label='S1 modeled')
    plt.legend(handles=[label_1, label_2], frameon=False)
    plt.xlim(0, 60)
    plt.xticks([0, 59], ['First trial', 'Last trial'])
    plt.subplot(3, 2, 2)
    plt.ylim(-1.35, .2)
    plt.yticks([0, -.5, -1])
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
    label_1 = mlines.Line2D([], [], color='darkred', linewidth=2, label='S2')
    label_2 = mlines.Line2D([], [], color='hotpink', linewidth=2, label='S2 modeled')
    plt.legend(handles=[label_1, label_2], frameon=False)
    plt.xlim(0, 60)
    plt.xticks([0, 59], ['First trial', 'Last trial'])
    sns.despine()

    paths = preprocess.create_folders(mice[0], sample_dates[0])
    reactivation_cue_pca_vec = np.load(paths['base_path'] + paths['mouse'] +
                                       '/data_across_days/reactivation_cue_vector_evolve.npy', allow_pickle=True)
    all_s1 = np.empty((len(mice), len(reactivation_cue_pca_vec[0][0]))) * np.nan
    all_s1r = np.empty((len(mice), len(reactivation_cue_pca_vec[0][0]))) * np.nan
    all_s2 = np.empty((len(mice), len(reactivation_cue_pca_vec[0][0]))) * np.nan
    all_s2r = np.empty((len(mice), len(reactivation_cue_pca_vec[0][0]))) * np.nan
    all_s1_opto = np.empty((len(mice), len(reactivation_cue_pca_vec[0][0]))) * np.nan
    all_s1r_opto = np.empty((len(mice), len(reactivation_cue_pca_vec[0][0]))) * np.nan
    all_s2_opto = np.empty((len(mice), len(reactivation_cue_pca_vec[0][0]))) * np.nan
    all_s2r_opto = np.empty((len(mice), len(reactivation_cue_pca_vec[0][0]))) * np.nan
    for mouse in range(0, len(mice)):
        paths = preprocess.create_folders(mice[mouse], sample_dates[mouse])
        reactivation_cue_pca_vec = np.load(paths['base_path'] + paths['mouse'] +
                                           '/data_across_days/reactivation_cue_vector_evolve.npy', allow_pickle=True)
        temp_s1 = np.empty((len(reactivation_cue_pca_vec[0]), 60)) * np.nan
        temp_s1r = np.empty((len(reactivation_cue_pca_vec[0]), 60)) * np.nan
        temp_s2 = np.empty((len(reactivation_cue_pca_vec[0]), 60)) * np.nan
        temp_s2r = np.empty((len(reactivation_cue_pca_vec[0]), 60)) * np.nan
        temp_s1_opto = np.empty((len(reactivation_cue_pca_vec[0]), 60)) * np.nan
        temp_s1r_opto = np.empty((len(reactivation_cue_pca_vec[0]), 60)) * np.nan
        temp_s2_opto = np.empty((len(reactivation_cue_pca_vec[0]), 60)) * np.nan
        temp_s2r_opto = np.empty((len(reactivation_cue_pca_vec[0]), 60)) * np.nan
        for i in range(0, len(reactivation_cue_pca_vec[0])):
            if mouse < 5:
                temp_s1[i, :] = reactivation_cue_pca_vec[0][i]
                temp_s1r[i, :] = reactivation_cue_pca_vec[2][i]
                temp_s2[i, :] = reactivation_cue_pca_vec[1][i]
                temp_s2r[i, :] = reactivation_cue_pca_vec[3][i]
            if mouse > 4:
                temp_s1_opto[i, :] = scipy.signal.resample_poly(reactivation_cue_pca_vec[0][i], 2, 1, padtype='smooth')[
                                     0:(len(reactivation_cue_pca_vec[0][i]) * 2)]
                temp_s1r_opto[i, :] = scipy.signal.resample_poly(reactivation_cue_pca_vec[1][i], 2, 1,
                                                                 padtype='smooth')[
                                      0:(len(reactivation_cue_pca_vec[2][i]) * 2)]
                temp_s2_opto[i, :] = scipy.signal.resample_poly(reactivation_cue_pca_vec[2][i], 2, 1, padtype='smooth')[
                                     0:(len(reactivation_cue_pca_vec[1][i]) * 2)]
                temp_s2r_opto[i, :] = scipy.signal.resample_poly(reactivation_cue_pca_vec[3][i], 2, 1,
                                                                 padtype='smooth')[
                                      0:(len(reactivation_cue_pca_vec[3][i]) * 2)]
        y_s1 = np.nanmean(temp_s1, axis=0)
        y_s1r = np.nanmean(temp_s1r, axis=0)
        y_s2 = np.nanmean(temp_s2, axis=0)
        y_s2r = np.nanmean(temp_s2r, axis=0)
        all_s1[mouse, :] = y_s1
        all_s1r[mouse, :] = y_s1r
        all_s2[mouse, :] = y_s2
        all_s2r[mouse, :] = y_s2r
        y_s1_opto = np.nanmean(temp_s1_opto, axis=0)
        y_s1r_opto = np.nanmean(temp_s1r_opto, axis=0)
        y_s2_opto = np.nanmean(temp_s2_opto, axis=0)
        y_s2r_opto = np.nanmean(temp_s2r_opto, axis=0)
        all_s1_opto[mouse, :] = y_s1_opto
        all_s1r_opto[mouse, :] = y_s1r_opto
        all_s2_opto[mouse, :] = y_s2_opto
        all_s2r_opto[mouse, :] = y_s2r_opto

    # print(stats.ttest_rel(all_s2[:, 0:1], all_s2r[:, 59:60])[1])

    plt.subplot(3, 2, 3)
    plt.ylim(-1.5, .4)
    plt.yticks([0, -.5, -1, -1.5])
    plt.gca().invert_yaxis()
    mean = np.nanmean(all_s1, axis=0)
    sem_plus = mean + stats.sem(all_s1, axis=0, nan_policy='omit')
    sem_minus = mean - stats.sem(all_s1, axis=0, nan_policy='omit')
    plt.plot(x_label, mean, '-', c='darkgreen', linewidth=3)
    plt.fill_between(x_label, sem_plus, sem_minus, alpha=0.2, color='darkgreen', lw=0)
    mean = np.nanmean(all_s1r, axis=0)
    sem_plus = mean + stats.sem(all_s1r, axis=0, nan_policy='omit')
    sem_minus = mean - stats.sem(all_s1r, axis=0, nan_policy='omit')
    plt.plot(x_label, mean, '-', c='lime', linewidth=3)
    plt.fill_between(x_label, sem_plus, sem_minus, alpha=0.2, color='lime', lw=0)
    plt.ylabel('Similarity to early vs. late\n S1 response pattern')
    plt.xlim(0, 60)
    plt.xticks([0, 59], ['First trial', 'Last trial'])
    plt.subplot(3, 2, 4)
    plt.ylim(-1.5, .4)
    plt.yticks([0, -.5, -1, -1.5])
    plt.gca().invert_yaxis()
    mean = np.nanmean(all_s1_opto, axis=0)
    sem_plus = mean + stats.sem(all_s1_opto, axis=0, nan_policy='omit')
    sem_minus = mean - stats.sem(all_s1_opto, axis=0, nan_policy='omit')
    plt.plot(x_label, mean, '-', c='purple', linewidth=3)
    plt.fill_between(x_label, sem_plus, sem_minus, alpha=0.2, color='purple', lw=0)
    mean = np.nanmean(all_s1r_opto, axis=0)
    sem_plus = mean + stats.sem(all_s1r_opto, axis=0, nan_policy='omit')
    sem_minus = mean - stats.sem(all_s1r_opto, axis=0, nan_policy='omit')
    plt.plot(x_label, mean, '-', c='orange', linewidth=3)
    plt.fill_between(x_label, sem_plus, sem_minus, alpha=0.2, color='orange', lw=0)
    plt.ylabel('Similarity to early vs. late\n S1 response pattern')
    plt.xlim(0, 60)
    plt.xticks([0, 59], ['First trial', 'Last trial'])
    plt.subplot(3, 2, 5)
    plt.ylim(-1.5, .4)
    plt.yticks([0, -.5, -1, -1.5])
    plt.gca().invert_yaxis()
    mean = np.nanmean(all_s2, axis=0)
    sem_plus = mean + stats.sem(all_s2, axis=0, nan_policy='omit')
    sem_minus = mean - stats.sem(all_s2, axis=0, nan_policy='omit')
    plt.plot(x_label, mean, '-', c='darkred', linewidth=3)
    plt.fill_between(x_label, sem_plus, sem_minus, alpha=0.2, color='darkred', lw=0)
    mean = np.nanmean(all_s2r, axis=0)
    sem_plus = mean + stats.sem(all_s2r, axis=0, nan_policy='omit')
    sem_minus = mean - stats.sem(all_s2r, axis=0, nan_policy='omit')
    plt.plot(x_label, mean, '-', c='hotpink', linewidth=3)
    plt.fill_between(x_label, sem_plus, sem_minus, alpha=0.2, color='hotpink', lw=0)
    plt.ylabel('Similarity to early vs. late\n S2 response pattern)')
    plt.xlim(0, 60)
    plt.xticks([0, 59], ['First trial', 'Last trial'])
    sns.despine()
    plt.subplot(3, 2, 6)
    plt.ylim(-1.5, .4)
    plt.yticks([0, -.5, -1, -1.5])
    plt.gca().invert_yaxis()
    mean = np.nanmean(all_s2_opto, axis=0)
    sem_plus = mean + stats.sem(all_s2_opto, axis=0, nan_policy='omit')
    sem_minus = mean - stats.sem(all_s2_opto, axis=0, nan_policy='omit')
    plt.plot(x_label, mean, '-', c='purple', linewidth=3)
    plt.fill_between(x_label, sem_plus, sem_minus, alpha=0.2, color='purple', lw=0)
    mean = np.nanmean(all_s2r_opto, axis=0)
    sem_plus = mean + stats.sem(all_s2r_opto, axis=0, nan_policy='omit')
    sem_minus = mean - stats.sem(all_s2r_opto, axis=0, nan_policy='omit')
    plt.plot(x_label, mean, '-', c='orange', linewidth=3)
    plt.fill_between(x_label, sem_plus, sem_minus, alpha=0.2, color='orange', lw=0)
    plt.ylabel('Similarity to early vs. late\n S2 response pattern)')
    plt.xlim(0, 60)
    plt.xticks([0, 59], ['First trial', 'Last trial'])
    sns.despine()

    plt.savefig(paths['base_path'] + '/NN/plots/reactivation_cue_vector_evolve.png', bbox_inches='tight', dpi=200)
    plt.savefig(paths['base_path'] + '/NN/plots/reactivation_cue_vector_evolve.pdf', bbox_inches='tight', dpi=200,
                transparent=True)
    #plt.close()


def reactivation_cue_vector_cross_evolve(mice, sample_dates):
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(wspace=.35)
    m_colors = ['b', 'teal', 'slategray', 'green', 'darkolivegreen', 'darkorange', 'purple', 'darkred']
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

            if mouse < 5:
                s1_temp = np.array(reactivation_cue_pca_vec[2][i])
                s1r_temp = np.array(reactivation_cue_pca_vec[0][i])
                s2_temp = np.array(reactivation_cue_pca_vec[3][i])
                s2r_temp = np.array(reactivation_cue_pca_vec[1][i])
            if mouse > 4:
                s1_temp = np.array(scipy.signal.resample_poly(reactivation_cue_pca_vec[2][i], 2, 1, padtype='smooth')[
                                 0:(len(reactivation_cue_pca_vec[2][i]) * 2)])
                s1r_temp = np.array(scipy.signal.resample_poly(reactivation_cue_pca_vec[0][i], 2, 1, padtype='smooth')[
                                 0:(len(reactivation_cue_pca_vec[0][i]) * 2)])
                s2_temp = np.array(scipy.signal.resample_poly(reactivation_cue_pca_vec[3][i], 2, 1, padtype='smooth')[
                                 0:(len(reactivation_cue_pca_vec[3][i]) * 2)])
                s2r_temp = np.array(scipy.signal.resample_poly(reactivation_cue_pca_vec[1][i], 2, 1, padtype='smooth')[
                                 0:(len(reactivation_cue_pca_vec[1][i]) * 2)])
            s1_temp = signal.sosfilt(sos, s1_temp)
            s1_temp = s1_temp[5:len(s1_temp)]
            s1r_temp = signal.sosfilt(sos, s1r_temp)
            s1r_temp = s1r_temp[5:len(s1r_temp)]
            s2_temp = signal.sosfilt(sos, s2_temp)
            s2_temp = s2_temp[5:len(s2_temp)]
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
    plt.ylim(-.3, 1.05)
    plt.yticks([0, .5, 1])
    plt.subplot(2, 2, 2)
    mean = all_s2.mean(axis=0)
    sem_plus = mean + stats.sem(all_s2, axis=0)
    sem_minus = mean - stats.sem(all_s2, axis=0)
    plt.plot(range(-maxlags, maxlags + 1), mean, '-', c='salmon', linewidth=3)
    plt.fill_between(range(-maxlags, maxlags + 1), sem_plus, sem_minus, alpha=0.2, color='salmon', lw=0)
    plt.axvline(0, color='black', linestyle='--', linewidth=1, snap=False)
    plt.ylabel('Cross correlation between\nreal and modeled\nstimulus 2 responses')
    plt.xlabel('Shift in trial')
    plt.ylim(-.3, 1.05)
    plt.yticks([0, .5, 1])
    sns.despine()

    all_s1 = np.zeros((5, (maxlags * 2) + 1))
    all_s2 = np.zeros((5, (maxlags * 2) + 1))
    for mouse in range(0, 5):
        paths = preprocess.create_folders(mice[mouse], sample_dates[mouse])
        reactivation_cue_pca_vec = np.load(paths['base_path'] + paths['mouse'] +
                                           '/data_across_days/reactivation_cue_vector_evolve.npy', allow_pickle=True)
        temp_s1 = np.zeros((len(reactivation_cue_pca_vec[0]), (maxlags * 2) + 1))
        temp_s2 = np.zeros((len(reactivation_cue_pca_vec[0]), (maxlags * 2) + 1))
        for i in range(0, len(reactivation_cue_pca_vec[0])):

            sos = signal.butter(2, .1 / 60 / 2, btype='highpass', output='sos', fs=1 / 60)

            if mouse < 5:
                s1_temp = np.array(reactivation_cue_pca_vec[2][i])
                s1r_temp = np.array(reactivation_cue_pca_vec[0][i])
                s2_temp = np.array(reactivation_cue_pca_vec[3][i])
                s2r_temp = np.array(reactivation_cue_pca_vec[1][i])
            if mouse > 4:
                s1_temp = np.array(scipy.signal.resample_poly(reactivation_cue_pca_vec[2][i], 2, 1, padtype='smooth')[
                                   0:(len(reactivation_cue_pca_vec[2][i]) * 2)])
                s1r_temp = np.array(scipy.signal.resample_poly(reactivation_cue_pca_vec[0][i], 2, 1, padtype='smooth')[
                                    0:(len(reactivation_cue_pca_vec[0][i]) * 2)])
                s2_temp = np.array(scipy.signal.resample_poly(reactivation_cue_pca_vec[3][i], 2, 1, padtype='smooth')[
                                   0:(len(reactivation_cue_pca_vec[3][i]) * 2)])
                s2r_temp = np.array(scipy.signal.resample_poly(reactivation_cue_pca_vec[1][i], 2, 1, padtype='smooth')[
                                    0:(len(reactivation_cue_pca_vec[1][i]) * 2)])
            s1_temp = signal.sosfilt(sos, s1_temp)
            s1_temp = s1_temp[5:len(s1_temp)]
            s1r_temp = signal.sosfilt(sos, s1r_temp)
            s1r_temp = s1r_temp[5:len(s1r_temp)]
            s2_temp = signal.sosfilt(sos, s2_temp)
            s2_temp = s2_temp[5:len(s2_temp)]
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
        all_s1[mouse, :] = y_s1
        all_s2[mouse, :] = y_s2
    plt.subplot(2, 2, 3)
    mean = all_s1.mean(axis=0)
    sem_plus = mean + stats.sem(all_s1, axis=0)
    sem_minus = mean - stats.sem(all_s1, axis=0)
    plt.plot(range(-maxlags, maxlags + 1), mean, '-', c='mediumseagreen', linewidth=3)
    plt.fill_between(range(-maxlags, maxlags + 1), sem_plus, sem_minus, alpha=0.2, color='mediumseagreen', lw=0)
    plt.axvline(0, color='black', linestyle='--', linewidth=1, snap=False)
    plt.ylabel('Cross correlation\nbetween real and modeled\nstimulus 1 responses')
    plt.xlabel('Shift in trial')
    plt.ylim(-.3, 1.05)
    plt.yticks([0, .5, 1])
    plt.subplot(2, 2, 4)
    mean = all_s2.mean(axis=0)
    sem_plus = mean + stats.sem(all_s2, axis=0)
    sem_minus = mean - stats.sem(all_s2, axis=0)
    plt.plot(range(-maxlags, maxlags + 1), mean, '-', c='salmon', linewidth=3)
    plt.fill_between(range(-maxlags, maxlags + 1), sem_plus, sem_minus, alpha=0.2, color='salmon', lw=0)
    plt.axvline(0, color='black', linestyle='--', linewidth=1, snap=False)
    plt.ylabel('Cross correlation between\nreal and modeled\nstimulus 2 responses')
    plt.xlabel('Shift in trial')
    plt.ylim(-.3, 1.05)
    plt.yticks([0, .5, 1])
    sns.despine()

    all_s1 = np.zeros((3, (maxlags * 2) + 1))
    all_s2 = np.zeros((3, (maxlags * 2) + 1))
    for mouse in range(5, len(mice)):
        paths = preprocess.create_folders(mice[mouse], sample_dates[mouse])
        reactivation_cue_pca_vec = np.load(paths['base_path'] + paths['mouse'] +
                                           '/data_across_days/reactivation_cue_vector_evolve.npy', allow_pickle=True)
        temp_s1 = np.zeros((len(reactivation_cue_pca_vec[0]), (maxlags * 2) + 1))
        temp_s2 = np.zeros((len(reactivation_cue_pca_vec[0]), (maxlags * 2) + 1))
        for i in range(0, len(reactivation_cue_pca_vec[0])):

            sos = signal.butter(2, .1 / 60 / 2, btype='highpass', output='sos', fs=1 / 60)

            if mouse < 5:
                s1_temp = np.array(reactivation_cue_pca_vec[2][i])
                s1r_temp = np.array(reactivation_cue_pca_vec[0][i])
                s2_temp = np.array(reactivation_cue_pca_vec[3][i])
                s2r_temp = np.array(reactivation_cue_pca_vec[1][i])
            if mouse > 4:
                s1_temp = np.array(scipy.signal.resample_poly(reactivation_cue_pca_vec[2][i], 2, 1, padtype='smooth')[
                                   0:(len(reactivation_cue_pca_vec[2][i]) * 2)])
                s1r_temp = np.array(scipy.signal.resample_poly(reactivation_cue_pca_vec[0][i], 2, 1, padtype='smooth')[
                                    0:(len(reactivation_cue_pca_vec[0][i]) * 2)])
                s2_temp = np.array(scipy.signal.resample_poly(reactivation_cue_pca_vec[3][i], 2, 1, padtype='smooth')[
                                   0:(len(reactivation_cue_pca_vec[3][i]) * 2)])
                s2r_temp = np.array(scipy.signal.resample_poly(reactivation_cue_pca_vec[1][i], 2, 1, padtype='smooth')[
                                    0:(len(reactivation_cue_pca_vec[1][i]) * 2)])
            s1_temp = signal.sosfilt(sos, s1_temp)
            s1_temp = s1_temp[5:len(s1_temp)]
            s1r_temp = signal.sosfilt(sos, s1r_temp)
            s1r_temp = s1r_temp[5:len(s1r_temp)]
            s2_temp = signal.sosfilt(sos, s2_temp)
            s2_temp = s2_temp[5:len(s2_temp)]
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
        all_s1[mouse-5, :] = y_s1
        all_s2[mouse-5, :] = y_s2
    plt.subplot(2, 2, 3)
    mean = all_s1.mean(axis=0)
    sem_plus = mean + stats.sem(all_s1, axis=0)
    sem_minus = mean - stats.sem(all_s1, axis=0)
    plt.plot(range(-maxlags, maxlags + 1), mean, '-', c='r', linewidth=3)
    plt.fill_between(range(-maxlags, maxlags + 1), sem_plus, sem_minus, alpha=0.2, color='r', lw=0)
    plt.axvline(0, color='black', linestyle='--', linewidth=1, snap=False)
    plt.ylabel('Cross correlation\nbetween real and modeled\nstimulus 1 responses')
    plt.xlabel('Shift in trial')
    plt.ylim(-.3, 1.05)
    plt.yticks([0, .5, 1])
    plt.subplot(2, 2, 4)
    mean = all_s2.mean(axis=0)
    sem_plus = mean + stats.sem(all_s2, axis=0)
    sem_minus = mean - stats.sem(all_s2, axis=0)
    plt.plot(range(-maxlags, maxlags + 1), mean, '-', c='r', linewidth=3)
    plt.fill_between(range(-maxlags, maxlags + 1), sem_plus, sem_minus, alpha=0.2, color='r', lw=0)
    plt.axvline(0, color='black', linestyle='--', linewidth=1, snap=False)
    plt.ylabel('Cross correlation between\nreal and modeled\nstimulus 2 responses')
    plt.xlabel('Shift in trial')
    plt.ylim(-.3, 1.05)
    plt.yticks([0, .5, 1])
    sns.despine()
    plt.savefig(paths['base_path'] + '/NN/plots/reactivation_cue_vector_evolve_cross.png', bbox_inches='tight', dpi=200)
    plt.savefig(paths['base_path'] + '/NN/plots/reactivation_cue_vector_evolve_cross.pdf', bbox_inches='tight', dpi=200,
                transparent=True)
    plt.close()


def activity_across_trials_evolve(mice, sample_dates):
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(9, 6))
    plt.subplots_adjust(wspace=.33)

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
            smoothed_activity = np.array(
                pd.DataFrame(activity_data[0][i]).rolling(3, min_periods=1, center=True).mean())
            smoothed_activity = np.concatenate(smoothed_activity, axis=0)
            smoothed_activity_t = np.array(
                pd.DataFrame(activity_data_t[0][i]).rolling(3, min_periods=1, center=True).mean())
            smoothed_activity_t = np.concatenate(smoothed_activity_t, axis=0)
            if mouse < 5:
                activity[i, 0:len(smoothed_activity)] = smoothed_activity
                activity_t[i, 0:len(smoothed_activity_t)] = smoothed_activity_t
            if mouse > 4:
                smoothed_activity = scipy.signal.resample_poly(smoothed_activity, 2, 1, padtype='smooth')[
                                    0:(len(smoothed_activity) * 2)]
                smoothed_activity_t = scipy.signal.resample_poly(smoothed_activity_t, 2, 1, padtype='smooth')[
                                    0:(len(smoothed_activity) * 2)]
                activity[i, 0:len(smoothed_activity)] = smoothed_activity
                activity_t[i, 0:len(smoothed_activity_t)] = smoothed_activity_t
        activity = np.nanmean(activity, axis=0)
        activity_all[mouse, :] = activity
        activity_t = np.nanmean(activity_t, axis=0)
        activity_all_t[mouse, :] = activity_t
        x = range(1, len(activity)+1)

    # print(stats.ttest_rel(np.mean(activity_all_t[:, 0:120], axis=1), np.mean(activity_all[:, 0:120], axis=1))[1])

    mean = np.nanmean(activity_all_t, axis=0)
    sem_plus = mean + stats.sem(activity_all_t, axis=0, nan_policy='omit')
    sem_minus = mean - stats.sem(activity_all_t, axis=0, nan_policy='omit')
    plt.plot(x[0:120], mean[0:120], c='k', linewidth=3)
    plt.fill_between(x[0:120], sem_plus[0:120], sem_minus[0:120], alpha=0.2, color='k', lw=0)
    mean = np.nanmean(activity_all, axis=0)
    sem_plus = mean + stats.sem(activity_all, axis=0, nan_policy='omit')
    sem_minus = mean - stats.sem(activity_all, axis=0, nan_policy='omit')
    plt.plot(x[0:120], mean[0:120], c=[.6, 0, .6], linewidth=3)
    plt.fill_between(x[0:120], sem_plus[0:120], sem_minus[0:120], alpha=0.2, color=[.6, 0, .6], lw=0)
    plt.ylabel('Correlation between\nstimulus 1 and stimulus 2')
    plt.xlabel('Trial number')
    plt.ylim(0, .4)
    plt.xlim(1, 121)
    plt.xticks([1, 120], ['First trial', 'Last trial'])
    label_1 = mlines.Line2D([], [], color='k', linewidth=2, label='Real data')
    label_2 = mlines.Line2D([], [], color=[.6, 0, .6], linewidth=2, label='Modeled data\nusing reactivations')
    plt.legend(handles=[label_1, label_2], frameon=False)
    sns.despine()

    plt.subplot(2, 2, 2)
    activity_all = np.empty((len(mice), 128)) * np.nan
    activity_all_t = np.empty((len(mice), 128)) * np.nan
    for mouse in range(0, 5):
        paths = preprocess.create_folders(mice[mouse], sample_dates[mouse])
        activity_data = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/activity_evolve.npy',
                                allow_pickle=True)
        activity_data_t = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/activity.npy',
                                  allow_pickle=True)
        activity = np.empty((len(activity_data[0]), 128)) * np.nan
        activity_t = np.empty((len(activity_data[0]), 128)) * np.nan
        for i in range(0, len(activity_data[0])):
            smoothed_activity = np.array(
                pd.DataFrame(activity_data[0][i]).rolling(3, min_periods=1, center=True).mean())
            smoothed_activity = np.concatenate(smoothed_activity, axis=0)
            smoothed_activity_t = np.array(
                pd.DataFrame(activity_data_t[0][i]).rolling(3, min_periods=1, center=True).mean())
            smoothed_activity_t = np.concatenate(smoothed_activity_t, axis=0)
            if mouse < 5:
                activity[i, 0:len(smoothed_activity)] = smoothed_activity
                activity_t[i, 0:len(smoothed_activity_t)] = smoothed_activity_t
            if mouse > 4:
                smoothed_activity = scipy.signal.resample_poly(smoothed_activity, 2, 1, padtype='smooth')[
                                    0:(len(smoothed_activity) * 2)]
                smoothed_activity_t = scipy.signal.resample_poly(smoothed_activity_t, 2, 1, padtype='smooth')[
                                      0:(len(smoothed_activity) * 2)]
                activity[i, 0:len(smoothed_activity)] = smoothed_activity
                activity_t[i, 0:len(smoothed_activity_t)] = smoothed_activity_t
        activity = np.nanmean(activity, axis=0)
        activity_all[mouse, :] = activity
        activity_t = np.nanmean(activity_t, axis=0)
        activity_all_t[mouse, :] = activity_t
        x = range(1, len(activity) + 1)
        # plt.plot(x, activity, '--', c=m_colors[mouse], alpha=1)
        # plt.plot(x, activity_t, c=m_colors[mouse], alpha=1)

    # print(stats.ttest_rel(np.mean(activity_all_t[:, 0:120], axis=1), np.mean(activity_all[:, 0:120], axis=1))[1])

    mean = np.nanmean(activity_all_t, axis=0)
    sem_plus = mean + stats.sem(activity_all_t, axis=0, nan_policy='omit')
    sem_minus = mean - stats.sem(activity_all_t, axis=0, nan_policy='omit')
    plt.plot(x[0:120], mean[0:120], c='k', linewidth=3)
    plt.fill_between(x[0:120], sem_plus[0:120], sem_minus[0:120], alpha=0.2, color='k', lw=0)
    mean = np.nanmean(activity_all, axis=0)
    sem_plus = mean + stats.sem(activity_all, axis=0, nan_policy='omit')
    sem_minus = mean - stats.sem(activity_all, axis=0, nan_policy='omit')
    plt.plot(x[0:120], mean[0:120], c=[.6, 0, .6], linewidth=3)
    plt.fill_between(x[0:120], sem_plus[0:120], sem_minus[0:120], alpha=0.2, color=[.6, 0, .6], lw=0)
    plt.ylabel('Correlation between\nstimulus 1 and stimulus 2')
    plt.xlabel('Trial number')
    plt.ylim(-.5, .6)
    plt.xlim(1, 121)
    plt.xticks([1, 120], ['First trial', 'Last trial'])
    label_1 = mlines.Line2D([], [], color='k', linewidth=2, label='Real data')
    label_2 = mlines.Line2D([], [], color=[.6, 0, .6], linewidth=2, label='Modeled data\nusing reactivations')
    plt.legend(handles=[label_1, label_2], frameon=False)
    sns.despine()

    plt.subplot(2, 2, 2)
    activity_all = np.empty((len(mice), 128)) * np.nan
    activity_all_t = np.empty((len(mice), 128)) * np.nan
    for mouse in range(5, len(mice)):
        paths = preprocess.create_folders(mice[mouse], sample_dates[mouse])
        activity_data = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/activity_evolve.npy',
                                allow_pickle=True)
        activity_data_t = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/activity.npy',
                                  allow_pickle=True)
        activity = np.empty((len(activity_data[0]), 128)) * np.nan
        activity_t = np.empty((len(activity_data[0]), 128)) * np.nan
        for i in range(0, len(activity_data[0])):
            smoothed_activity = np.array(
                pd.DataFrame(activity_data[0][i]).rolling(3, min_periods=1, center=True).mean())
            smoothed_activity = np.concatenate(smoothed_activity, axis=0)
            smoothed_activity_t = np.array(
                pd.DataFrame(activity_data_t[0][i]).rolling(3, min_periods=1, center=True).mean())
            smoothed_activity_t = np.concatenate(smoothed_activity_t, axis=0)
            if mouse < 5:
                activity[i, 0:len(smoothed_activity)] = smoothed_activity
                activity_t[i, 0:len(smoothed_activity_t)] = smoothed_activity_t
            if mouse > 4:
                smoothed_activity = scipy.signal.resample_poly(smoothed_activity, 2, 1, padtype='smooth')[
                                    0:(len(smoothed_activity) * 2)]
                smoothed_activity_t = scipy.signal.resample_poly(smoothed_activity_t, 2, 1, padtype='smooth')[
                                      0:(len(smoothed_activity) * 2)]
                activity[i, 0:len(smoothed_activity)] = smoothed_activity
                activity_t[i, 0:len(smoothed_activity_t)] = smoothed_activity_t
        activity = np.nanmean(activity, axis=0)
        activity_all[mouse, :] = activity
        activity_t = np.nanmean(activity_t, axis=0)
        activity_all_t[mouse, :] = activity_t
        x = range(1, len(activity) + 1)
        # plt.plot(x, activity, '--', c=m_colors[mouse], alpha=1)
        # plt.plot(x, activity_t, c=m_colors[mouse], alpha=1)

    # print(stats.ttest_rel(np.mean(activity_all_t[:, 0:120], axis=1), np.mean(activity_all[:, 0:120], axis=1))[1])

    mean = np.nanmean(activity_all_t, axis=0)
    sem_plus = mean + stats.sem(activity_all_t, axis=0, nan_policy='omit')
    sem_minus = mean - stats.sem(activity_all_t, axis=0, nan_policy='omit')
    plt.plot(x[0:120], mean[0:120], c='dimgrey', linewidth=3)
    plt.fill_between(x[0:120], sem_plus[0:120], sem_minus[0:120], alpha=0.2, color='dimgrey', lw=0)
    mean = np.nanmean(activity_all, axis=0)
    sem_plus = mean + stats.sem(activity_all, axis=0, nan_policy='omit')
    sem_minus = mean - stats.sem(activity_all, axis=0, nan_policy='omit')
    plt.plot(x[0:120], mean[0:120], c='r', linewidth=3)
    plt.fill_between(x[0:120], sem_plus[0:120], sem_minus[0:120], alpha=0.2, color='r', lw=0)
    plt.ylabel('Correlation between\nstimulus 1 and stimulus 2')
    plt.xlabel('Trial number')
    plt.ylim(-.05, .6)
    plt.xlim(1, 121)
    plt.xticks([1, 120], ['First trial', 'Last trial'])
    label_1 = mlines.Line2D([], [], color='k', linewidth=2, label='Real data')
    label_2 = mlines.Line2D([], [], color='r', linewidth=2, label='Modeled data\nusing reactivations')
    plt.legend(handles=[label_1, label_2], frameon=False)
    sns.despine()

    plt.savefig(paths['base_path'] + '/NN/plots/correlation_across_trials_evolve.png', bbox_inches='tight', dpi=200)
    plt.savefig(paths['base_path'] + '/NN/plots/correlation_across_trials_evolve.pdf', bbox_inches='tight', dpi=200, transparent=True)
    #plt.close()


def activity_across_trials_cross_evolve(mice, sample_dates):
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(wspace=.35)
    m_colors = ['b', 'teal', 'slategray', 'green', 'darkolivegreen', 'darkorange', 'purple', 'darkred']
    maxlags = 10
    all_corr = np.zeros((len(mice), (maxlags * 2) + 1))
    for mouse in range(0, len(mice)):
        paths = preprocess.create_folders(mice[mouse], sample_dates[mouse])
        activity_data = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/activity_evolve.npy',
                                allow_pickle=True)
        activity_data_t = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/activity.npy',
                                  allow_pickle=True)
        temp_corr = np.zeros((len(activity_data[0]), (maxlags * 2) + 1))
        for i in range(0, len(activity_data[0])):

            sos = signal.butter(2, .1 / 112 / 2, btype='highpass', output='sos', fs=1 / 112)

            smoothed_activity = np.array(
                pd.DataFrame(activity_data[0][i]).rolling(3, min_periods=1, center=True).mean())
            smoothed_activity = np.concatenate(smoothed_activity, axis=0)[0:112]
            smoothed_activity_t = np.array(
                pd.DataFrame(activity_data_t[0][i]).rolling(3, min_periods=1, center=True).mean())
            smoothed_activity_t = np.concatenate(smoothed_activity_t, axis=0)[0:112]
            if mouse > 4:
                smoothed_activity = scipy.signal.resample_poly(smoothed_activity, 2, 1, padtype='smooth')[
                                    0:(len(smoothed_activity) * 2)][0:112]
                smoothed_activity_t = scipy.signal.resample_poly(smoothed_activity_t, 2, 1, padtype='smooth')[
                                      0:(len(smoothed_activity) * 2)][0:112]

            corr_temp = signal.sosfilt(sos, smoothed_activity)
            corr_temp = corr_temp[5:len(corr_temp)]
            corrt_temp = signal.sosfilt(sos, smoothed_activity_t)
            corrt_temp = corrt_temp[5:len(corrt_temp)]

            corr = corr_temp - np.mean(corr_temp)
            corrt = corrt_temp - np.mean(corrt_temp)

            corr_vec = signal.correlate(corr, corrt)
            corr_vec = corr_vec / np.max(corr_vec)
            num_trials = int((len(corr_vec) - 1) / 2)
            temp_corr[i, :] = corr_vec[num_trials - maxlags:num_trials + maxlags + 1]

        y_corr = np.nanmean(temp_corr, axis=0)
        plt.subplot(2, 2, 1)
        plt.plot(range(-maxlags, maxlags + 1), y_corr, '-', c=m_colors[mouse], alpha=.2, linewidth=2)
        all_corr[mouse, :] = y_corr
    plt.subplot(2, 2, 1)
    mean = all_corr.mean(axis=0)
    sem_plus = mean + stats.sem(all_corr, axis=0)
    sem_minus = mean - stats.sem(all_corr, axis=0)
    plt.plot(range(-maxlags, maxlags + 1), mean, '-', c='k', linewidth=3)
    plt.fill_between(range(-maxlags, maxlags + 1), sem_plus, sem_minus, alpha=0.2, color='k', lw=0)
    plt.axvline(0, color='black', linestyle='--', linewidth=1, snap=False)
    plt.ylabel('Cross correlation\nbetween real and modeled\nresponse similarity')
    plt.xlabel('Shift in trial')
    plt.ylim(-.5, 1)
    plt.yticks([0, 1])
    sns.despine()
    plt.savefig(paths['base_path'] + '/NN/plots/activity_across_trials_evolve_cross.png', bbox_inches='tight', dpi=200)
    plt.savefig(paths['base_path'] + '/NN/plots/activity_across_trials_evolve_cross.pdf', bbox_inches='tight', dpi=200,
                transparent=True)
    # plt.close()


def activity_across_trials_evolve_low_reactivation(mice, sample_dates):
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(9, 6))
    plt.subplots_adjust(wspace=.33)
    m_colors = ['b', 'purple', 'darkorange', 'green', 'darkred']

    plt.subplot(2, 2, 1)
    activity_all = np.empty((len(mice), 128)) * np.nan
    activity_all_t = np.empty((len(mice), 128)) * np.nan
    for mouse in range(0, len(mice)):
        paths = preprocess.create_folders(mice[mouse], sample_dates[mouse])
        activity_data = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/activity_evolve_low_reactivation.npy',
                               allow_pickle=True)
        activity_data_t = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/activity.npy',
                                allow_pickle=True)
        activity = np.empty((len(activity_data[0]), 128)) * np.nan
        activity_t = np.empty((len(activity_data[0]), 128)) * np.nan
        for i in range(0, len(activity_data[0])):
            smoothed_activity = np.array(
                pd.DataFrame(activity_data[0][i]).rolling(3, min_periods=1, center=True).mean())
            smoothed_activity = np.concatenate(smoothed_activity, axis=0)
            smoothed_activity_t = np.array(
                pd.DataFrame(activity_data_t[0][i]).rolling(3, min_periods=1, center=True).mean())
            smoothed_activity_t = np.concatenate(smoothed_activity_t, axis=0)
            if mouse < 5:
                activity[i, 0:len(smoothed_activity)] = smoothed_activity
                activity_t[i, 0:len(smoothed_activity_t)] = smoothed_activity_t
            if mouse > 4:
                smoothed_activity = scipy.signal.resample_poly(smoothed_activity, 2, 1, padtype='smooth')[
                                    0:(len(smoothed_activity) * 2)]
                smoothed_activity_t = scipy.signal.resample_poly(smoothed_activity_t, 2, 1, padtype='smooth')[
                                    0:(len(smoothed_activity) * 2)]
                activity[i, 0:len(smoothed_activity)] = smoothed_activity
                activity_t[i, 0:len(smoothed_activity_t)] = smoothed_activity_t
        activity = np.nanmean(activity, axis=0)
        activity_all[mouse, :] = activity
        activity_t = np.nanmean(activity_t, axis=0)
        activity_all_t[mouse, :] = activity_t
        x = range(1, len(activity)+1)
        # plt.plot(x, activity, '--', c=m_colors[mouse], alpha=1)
        # plt.plot(x, activity_t, c=m_colors[mouse], alpha=1)

    # print(stats.ttest_rel(np.mean(activity_all_t[:, 0:120], axis=1), np.mean(activity_all[:, 0:120], axis=1))[1])

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
    plt.xticks([1, 120], ['First trial', 'Last trial'])
    label_1 = mlines.Line2D([], [], color='k', linewidth=2, label='Real data')
    label_2 = mlines.Line2D([], [], color=[.6, 0, .6], linewidth=2, label='Modeled data\nusing reactivations')
    plt.legend(handles=[label_1, label_2], frameon=False)
    sns.despine()

    plt.savefig(paths['base_path'] + '/NN/plots/correlation_across_trials_evolve_low_reactivation.png', bbox_inches='tight', dpi=200)
    plt.savefig(paths['base_path'] + '/NN/plots/correlation_across_trials_evolve_low_reactivation.pdf', bbox_inches='tight', dpi=200, transparent=True)
    #plt.close()


def activity_across_trials_evolve_grouped(mice, sample_dates):
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(13.5, 9))
    plt.subplots_adjust(wspace=.35)

    plt.subplot(3, 3, 1)
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
                pd.DataFrame(activity_data[0][i][0]).rolling(3, min_periods=1, center=True).mean())
            smoothed_activity_same = np.concatenate(smoothed_activity_same, axis=0)
            smoothed_activity_increase = np.array(
                pd.DataFrame(activity_data[0][i][1]).rolling(3, min_periods=1, center=True).mean())
            smoothed_activity_increase = np.concatenate(smoothed_activity_increase, axis=0)
            smoothed_activity_decrease = np.array(
                pd.DataFrame(activity_data[0][i][2]).rolling(3, min_periods=1, center=True).mean())
            smoothed_activity_decrease = np.concatenate(smoothed_activity_decrease, axis=0)
            if mouse < 5:
                activity_same[i, 0:len(smoothed_activity_same)] = smoothed_activity_same
                activity_increase[i, 0:len(smoothed_activity_increase)] = smoothed_activity_increase
                activity_decrease[i, 0:len(smoothed_activity_decrease)] = smoothed_activity_decrease
            if mouse > 4:
                smoothed_activity_same = scipy.signal.resample_poly(smoothed_activity_same, 2, 1, padtype='smooth')[
                                         0:(len(smoothed_activity_same) * 2)]
                smoothed_activity_increase = scipy.signal.resample_poly(smoothed_activity_increase, 2, 1,
                                                                        padtype='smooth')[
                                             0:(len(smoothed_activity_increase) * 2)]
                smoothed_activity_decrease = scipy.signal.resample_poly(smoothed_activity_decrease, 2, 1,
                                                                        padtype='smooth')[
                                             0:(len(smoothed_activity_decrease) * 2)]
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

    slope, intercept, r_value, p_value_dec, std_err = stats.linregress(
        np.concatenate([x[0:120], x[0:120], x[0:120], x[0:120], x[0:120], x[0:120], x[0:120], x[0:120]]),
        np.concatenate(activity_decrease_all[:, 0:120]))
    print(r_value)
    slope, intercept, r_value, p_value_inc, std_err = stats.linregress(
        np.concatenate([x[0:120], x[0:120], x[0:120], x[0:120], x[0:120], x[0:120], x[0:120], x[0:120]]),
        np.concatenate(activity_increase_all[:, 0:120]))
    print(r_value)
    anova_results = []
    anova_results.append(p_value_dec)
    anova_results.append(p_value_inc)
    anova_results_corrected = multipletests(anova_results, alpha=.05, method='holm')
    print(anova_results_corrected[1])

    mean = np.nanmean(activity_increase_all, axis=0)
    sem_plus = mean + stats.sem(activity_increase_all, axis=0, nan_policy='omit')
    sem_minus = mean - stats.sem(activity_increase_all, axis=0, nan_policy='omit')
    plt.plot(x[0:120], mean[0:120], c='darkred', linewidth=3)
    plt.fill_between(x[0:120], sem_plus[0:120], sem_minus[0:120], alpha=0.2, color='darkred', lw=0)
    mean = np.nanmean(activity_decrease_all, axis=0)
    sem_plus = mean + stats.sem(activity_decrease_all, axis=0, nan_policy='omit')
    sem_minus = mean - stats.sem(activity_decrease_all, axis=0, nan_policy='omit')
    plt.plot(x[0:120], mean[0:120], c='darkblue', linewidth=3)
    plt.fill_between(x[0:120], sem_plus[0:120], sem_minus[0:120], alpha=0.2, color='darkblue', lw=0)
    plt.ylabel('Response similarity\n(correlation between\nS1 and S2)')
    plt.ylim(-.15, .35)
    plt.yticks([-.1, 0, .1, .2, .3])
    plt.xlim(1, 121)
    plt.xticks([1, 120], ['First trial', 'Last trial'])
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
    sns.despine()

    plt.subplot(3, 3, 1)
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
            if mouse < 5:
                activity_same[i, 0:len(smoothed_activity_same)] = smoothed_activity_same
                activity_increase[i, 0:len(smoothed_activity_increase)] = smoothed_activity_increase
                activity_decrease[i, 0:len(smoothed_activity_decrease)] = smoothed_activity_decrease
            if mouse > 4:
                smoothed_activity_same = scipy.signal.resample_poly(smoothed_activity_same, 2, 1, padtype='smooth')[0:(len(smoothed_activity_same)*2)]
                smoothed_activity_increase = scipy.signal.resample_poly(smoothed_activity_increase, 2, 1, padtype='smooth')[0:(len(smoothed_activity_increase)*2)]
                smoothed_activity_decrease = scipy.signal.resample_poly(smoothed_activity_decrease, 2, 1, padtype='smooth')[0:(len(smoothed_activity_decrease)*2)]
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

    slope, intercept, r_value, p_value_dec, std_err = stats.linregress(
        np.concatenate([x[0:120], x[0:120], x[0:120], x[0:120], x[0:120], x[0:120], x[0:120], x[0:120]]), np.concatenate(activity_decrease_all[:, 0:120]))
    print(r_value)
    slope, intercept, r_value, p_value_inc, std_err = stats.linregress(
        np.concatenate([x[0:120], x[0:120], x[0:120], x[0:120], x[0:120], x[0:120], x[0:120], x[0:120]]), np.concatenate(activity_increase_all[:, 0:120]))
    print(r_value)
    anova_results = []
    anova_results.append(p_value_dec)
    anova_results.append(p_value_inc)
    anova_results_corrected = multipletests(anova_results, alpha=.05, method='holm')
    print(anova_results_corrected[1])

    mean = np.nanmean(activity_increase_all, axis=0)
    sem_plus = mean + stats.sem(activity_increase_all, axis=0, nan_policy='omit')
    sem_minus = mean - stats.sem(activity_increase_all, axis=0, nan_policy='omit')
    plt.plot(x[0:120], mean[0:120], c='darkred', linewidth=3)
    plt.fill_between(x[0:120], sem_plus[0:120], sem_minus[0:120], alpha=0.2, color='darkred', lw=0)
    mean = np.nanmean(activity_decrease_all, axis=0)
    sem_plus = mean + stats.sem(activity_decrease_all, axis=0, nan_policy='omit')
    sem_minus = mean - stats.sem(activity_decrease_all, axis=0, nan_policy='omit')
    plt.plot(x[0:120], mean[0:120], c='darkblue', linewidth=3)
    plt.fill_between(x[0:120], sem_plus[0:120], sem_minus[0:120], alpha=0.2, color='darkblue', lw=0)
    plt.ylabel('Response similarity\n(correlation between\nS1 and S2)')
    plt.ylim(0, .37)
    plt.yticks([0, .1, .2, .3])
    plt.xlim(1, 121)
    plt.xticks([1, 120], ['First trial', 'Last trial'])
    sns.despine()

    activity_same_all = np.empty((len(mice), 128)) * np.nan
    activity_increase_all = np.empty((len(mice), 128)) * np.nan
    activity_decrease_all = np.empty((len(mice), 128)) * np.nan
    for mouse in range(0, 5):
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
            if mouse < 5:
                activity_same[i, 0:len(smoothed_activity_same)] = smoothed_activity_same
                activity_increase[i, 0:len(smoothed_activity_increase)] = smoothed_activity_increase
                activity_decrease[i, 0:len(smoothed_activity_decrease)] = smoothed_activity_decrease
            if mouse > 4:
                smoothed_activity_same = scipy.signal.resample_poly(smoothed_activity_same, 2, 1, padtype='smooth')[
                                         0:(len(smoothed_activity_same) * 2)]
                smoothed_activity_increase = scipy.signal.resample_poly(smoothed_activity_increase, 2, 1,
                                                                        padtype='smooth')[
                                             0:(len(smoothed_activity_increase) * 2)]
                smoothed_activity_decrease = scipy.signal.resample_poly(smoothed_activity_decrease, 2, 1,
                                                                        padtype='smooth')[
                                             0:(len(smoothed_activity_decrease) * 2)]
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

    plt.subplot(3, 3, 2)
    mean = np.nanmean(activity_increase_all, axis=0)
    sem_plus = mean + stats.sem(activity_increase_all, axis=0, nan_policy='omit')
    sem_minus = mean - stats.sem(activity_increase_all, axis=0, nan_policy='omit')
    plt.plot(x[0:120], mean[0:120], c='darkred', linewidth=3)
    plt.fill_between(x[0:120], sem_plus[0:120], sem_minus[0:120], alpha=0.2, color='darkred', lw=0)
    plt.ylabel('Response similarity\n(correlation between\nS1 and S2)')
    plt.ylim(0, .37)
    plt.yticks([0, .1, .2, .3])
    plt.xlim(1, 121)
    plt.xticks([1, 120], ['First trial', 'Last trial'])
    plt.subplot(3, 3, 3)
    mean = np.nanmean(activity_decrease_all, axis=0)
    sem_plus = mean + stats.sem(activity_decrease_all, axis=0, nan_policy='omit')
    sem_minus = mean - stats.sem(activity_decrease_all, axis=0, nan_policy='omit')
    plt.plot(x[0:120], mean[0:120], c='darkblue', linewidth=3)
    plt.fill_between(x[0:120], sem_plus[0:120], sem_minus[0:120], alpha=0.2, color='darkblue', lw=0)
    plt.ylabel('Response similarity\n(correlation between\nS1 and S2)')
    plt.ylim(0, .37)
    plt.yticks([0, .1, .2, .3])
    plt.xlim(1, 121)
    plt.xticks([1, 120], ['First trial', 'Last trial'])
    sns.despine()

    activity_same_all = np.empty((len(mice), 128)) * np.nan
    activity_increase_all = np.empty((len(mice), 128)) * np.nan
    activity_decrease_all = np.empty((len(mice), 128)) * np.nan
    for mouse in range(5, len(mice)):
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
            if mouse < 5:
                activity_same[i, 0:len(smoothed_activity_same)] = smoothed_activity_same
                activity_increase[i, 0:len(smoothed_activity_increase)] = smoothed_activity_increase
                activity_decrease[i, 0:len(smoothed_activity_decrease)] = smoothed_activity_decrease
            if mouse > 4:
                smoothed_activity_same = scipy.signal.resample_poly(smoothed_activity_same, 2, 1, padtype='smooth')[
                                         0:(len(smoothed_activity_same) * 2)]
                smoothed_activity_increase = scipy.signal.resample_poly(smoothed_activity_increase, 2, 1,
                                                                        padtype='smooth')[
                                             0:(len(smoothed_activity_increase) * 2)]
                smoothed_activity_decrease = scipy.signal.resample_poly(smoothed_activity_decrease, 2, 1,
                                                                        padtype='smooth')[
                                             0:(len(smoothed_activity_decrease) * 2)]
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

    plt.subplot(3, 3, 2)
    mean = np.nanmean(activity_increase_all, axis=0)
    sem_plus = mean + stats.sem(activity_increase_all, axis=0, nan_policy='omit')
    sem_minus = mean - stats.sem(activity_increase_all, axis=0, nan_policy='omit')
    plt.plot(x[0:120], mean[0:120], c='r', linewidth=3)
    plt.fill_between(x[0:120], sem_plus[0:120], sem_minus[0:120], alpha=0.2, color='r', lw=0)
    plt.ylabel('Response similarity\n(correlation between\nS1 and S2)')
    plt.ylim(0, .37)
    plt.yticks([0, .1, .2, .3])
    plt.xlim(1, 121)
    plt.xticks([1, 120], ['First trial', 'Last trial'])
    plt.subplot(3, 3, 3)
    mean = np.nanmean(activity_decrease_all, axis=0)
    sem_plus = mean + stats.sem(activity_decrease_all, axis=0, nan_policy='omit')
    sem_minus = mean - stats.sem(activity_decrease_all, axis=0, nan_policy='omit')
    plt.plot(x[0:120], mean[0:120], c='r', linewidth=3)
    plt.fill_between(x[0:120], sem_plus[0:120], sem_minus[0:120], alpha=0.2, color='r', lw=0)
    plt.ylabel('Response similarity\n(correlation between\nS1 and S2)')
    plt.ylim(0, .37)
    plt.yticks([0, .1, .2, .3])
    plt.xlim(1, 121)
    plt.xticks([1, 120], ['First trial', 'Last trial'])
    sns.despine()
    plt.savefig(paths['base_path'] + '/NN/plots/activity_across_trials_evolve_grouped.png', bbox_inches='tight', dpi=200)
    plt.savefig(paths['base_path'] + '/NN/plots/activity_across_trials_evolve_grouped.pdf', bbox_inches='tight', dpi=200, transparent=True)
    plt.close()


def activity_across_trials_evolve_grouped_separate(mice, sample_dates):
    activity_across_trials_evolve_grouped_separate_helper(mice, sample_dates, 'no_change')
    activity_across_trials_evolve_grouped_separate_helper(mice, sample_dates, 'increase')
    activity_across_trials_evolve_grouped_separate_helper(mice, sample_dates, 'decrease')


def activity_across_trials_evolve_grouped_separate_helper(mice, sample_dates, group_type):
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
        session_data = preprocess.load_data(paths)
        fr = session_data['framerate']
        activity_data = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/activity_grouped_' + group_type + '_evolve.npy',
                               allow_pickle=True)
        cs1d_cs1 = np.empty((len(activity_data[0]), 128)) * np.nan
        cs1d_cs2 = np.empty((len(activity_data[0]), 128)) * np.nan
        cs2d_cs2 = np.empty((len(activity_data[0]), 128)) * np.nan
        cs2d_cs1 = np.empty((len(activity_data[0]), 128)) * np.nan
        for i in range(0, len(activity_data[0])):
            if mouse < 5:
                cs1d_cs1[i, 0:len(activity_data[0][i])] = activity_data[0][i] * fr
                cs1d_cs2[i, 0:len(activity_data[1][i])] = activity_data[1][i] * fr
                cs2d_cs2[i, 0:len(activity_data[2][i])] = activity_data[2][i] * fr
                cs2d_cs1[i, 0:len(activity_data[3][i])] = activity_data[3][i] * fr
            if mouse > 4:
                vec_1 = scipy.signal.resample_poly(activity_data[0][i], 2, 1, padtype='smooth')[
                        0:(len(activity_data[0][i]) * 2)]
                cs1d_cs1[i, 0:len(vec_1)] = vec_1 * fr
                vec_2 = scipy.signal.resample_poly(activity_data[1][i], 2, 1, padtype='smooth')[
                        0:(len(activity_data[1][i]) * 2)]
                cs1d_cs2[i, 0:len(vec_2)] = vec_2 * fr
                vec_3 = scipy.signal.resample_poly(activity_data[2][i], 2, 1, padtype='smooth')[
                        0:(len(activity_data[2][i]) * 2)]
                cs2d_cs2[i, 0:len(vec_3)] = vec_3 * fr
                vec_4 = scipy.signal.resample_poly(activity_data[3][i], 2, 1, padtype='smooth')[
                        0:(len(activity_data[3][i]) * 2)]
                cs2d_cs1[i, 0:len(vec_4)] = vec_4 * fr
        cs1d_cs1 = np.nanmean(cs1d_cs1, axis=0)
        cs2d_cs2 = np.nanmean(cs2d_cs2, axis=0)
        cs1d_cs1_all[mouse, :] = (cs1d_cs1 + cs2d_cs2) / 2
        # cs1d_cs2 = np.nanmean(cs1d_cs2, axis=0)
        # cs1d_cs2_all[mouse, :] = cs1d_cs2
        # cs2d_cs2 = np.nanmean(cs2d_cs2, axis=0)
        # cs2d_cs2_all[mouse, :] = cs2d_cs2
        cs2d_cs1 = np.nanmean(cs2d_cs1, axis=0)
        cs1d_cs2 = np.nanmean(cs1d_cs2, axis=0)
        cs2d_cs1_all[mouse, :] = (cs2d_cs1 + cs1d_cs2) / 2
        x = range(1, len(cs1d_cs1)+1)

    print(stats.ttest_rel(np.mean(cs1d_cs1_all[:, 0:3], axis=1) - np.mean(cs2d_cs1_all[:, 0:3], axis=1),
                                         np.mean(cs2d_cs1_all[:, 57:60], axis=1) - np.mean(cs1d_cs1_all[:, 57:60], axis=1))[1])

    mean = np.nanmean(cs1d_cs1_all, axis=0)
    sem_plus = mean + stats.sem(cs1d_cs1_all, axis=0, nan_policy='omit')
    sem_minus = mean - stats.sem(cs1d_cs1_all, axis=0, nan_policy='omit')
    plt.plot(x[0:60], mean[0:60], c='darkslategray', linewidth=3)
    plt.fill_between(x[0:60], sem_plus[0:60], sem_minus[0:60], alpha=0.2, color='darkslategray', lw=0)
    # mean = np.nanmean(cs1d_cs2_all, axis=0)
    # sem_plus = mean + stats.sem(cs1d_cs2_all, axis=0, nan_policy='omit')
    # sem_minus = mean - stats.sem(cs1d_cs2_all, axis=0, nan_policy='omit')
    # plt.plot(x, mean, c='darkgreen', linewidth=3)
    # plt.fill_between(x, sem_plus, sem_minus, alpha=0.2, color='darkgreen', lw=0)
    # mean = np.nanmean(cs2d_cs2_all, axis=0)
    # sem_plus = mean + stats.sem(cs2d_cs2_all, axis=0, nan_policy='omit')
    # sem_minus = mean - stats.sem(cs2d_cs2_all, axis=0, nan_policy='omit')
    # plt.plot(x, mean, c='salmon', linewidth=3)
    # plt.fill_between(x, sem_plus, sem_minus, alpha=0.2, color='salmon', lw=0)
    mean = np.nanmean(cs2d_cs1_all, axis=0)
    sem_plus = mean + stats.sem(cs2d_cs1_all, axis=0, nan_policy='omit')
    sem_minus = mean - stats.sem(cs2d_cs1_all, axis=0, nan_policy='omit')
    plt.plot(x[0:60], mean[0:60], c='cadetblue', linewidth=3)
    plt.fill_between(x[0:60], sem_plus[0:60], sem_minus[0:60], alpha=0.2, color='cadetblue', lw=0)
    plt.ylabel('Stimulus activity\n(Normalized $\mathregular{Ca^{2+}}$transient $\mathregular{s^{-1}}$)')
    # plt.xlabel('Trial number')
    plt.ylim(0, 1)
    plt.xlim(1, 61)
    plt.xticks([1, 60], ['First trial', 'Last trial'])
    # plt.yticks([0, .05, .1])
    sns.despine()

    plt.savefig(paths['base_path'] + '/NN/plots/activity_across_trials_evolve_grouped_' + group_type + '.png', bbox_inches='tight', dpi=200)
    plt.savefig(paths['base_path'] + '/NN/plots/activity_across_trials_evolve_grouped_' + group_type + '.pdf', bbox_inches='tight', dpi=200, transparent=True)
    plt.close()


def reactivation_cue_vector_evolve_parametric(mice, sample_dates):
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(9, 6))
    plt.subplots_adjust(wspace=.3)
    m_colors = ['b', 'teal', 'slategray', 'green', 'darkolivegreen', 'darkorange', 'purple', 'darkred']

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
    plt.xlim(0, 1.05)
    plt.ylim(0, .65)
    plt.xticks([0, .2, .4, .6, .8, 1])
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
    plt.xlim(0, 1.05)
    plt.xticks([0, .2, .4, .6, .8, 1])
    plt.ylim(0, .65)
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
    m_colors = ['b', 'teal', 'slategray', 'green', 'darkolivegreen', 'darkorange', 'purple', 'darkred']
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
    plt.ylabel('Stimulus / reactivation activity')
    plt.axhline(1, color='black', linestyle='--', linewidth=1, snap=False)
    sns.despine()
    plt.savefig(paths['base_path'] + '/NN/plots/reactivation_cue_scale.png', bbox_inches='tight', dpi=200)
    plt.savefig(paths['base_path'] + '/NN/plots/reactivation_cue_scale.pdf', bbox_inches='tight', dpi=200,
                transparent=True)
    plt.close()


def reactivation_cue_difference(mice, sample_dates):
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(7, 6))
    m_colors = ['b', 'teal', 'slategray', 'green', 'darkolivegreen', 'darkorange', 'purple', 'darkred']
    x = [1, 2, 3]
    decrease_cs_1_all = []
    increase_cs_1_all = []
    no_change_cs_1_all = []
    decrease_cs_2_all = []
    increase_cs_2_all = []
    no_change_cs_2_all = []
    for mouse in range(0, len(mice)):
        paths = preprocess.create_folders(mice[mouse], sample_dates[mouse])
        session_data = preprocess.load_data(paths)
        fr = session_data['framerate']
        reactivation_influence = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/reactivation_influence.npy', allow_pickle=True)
        decrease_cs_1 = []
        increase_cs_1 = []
        no_change_cs_1 = []
        decrease_cs_2 = []
        increase_cs_2 = []
        no_change_cs_2 = []
        for i in range(0, len(reactivation_influence[0])):
            decrease_cs_1.append(reactivation_influence[2][i] * fr)
            increase_cs_1.append(reactivation_influence[1][i] * fr)
            no_change_cs_1.append(reactivation_influence[0][i] * fr)
            decrease_cs_2.append(reactivation_influence[5][i] * fr)
            increase_cs_2.append(reactivation_influence[4][i] * fr)
            no_change_cs_2.append(reactivation_influence[3][i] * fr)
        decrease_cs_1_all.append(np.nanmean(decrease_cs_1))
        increase_cs_1_all.append(np.nanmean(increase_cs_1))
        no_change_cs_1_all.append(np.nanmean(no_change_cs_1))
        decrease_cs_2_all.append(np.nanmean(decrease_cs_2))
        increase_cs_2_all.append(np.nanmean(increase_cs_2))
        no_change_cs_2_all.append(np.nanmean(no_change_cs_2))
        plt.subplot(2, 2, 1)
        plt.errorbar(x[2], np.nanmean(decrease_cs_1), yerr=0, c=m_colors[mouse], marker='o', mfc='none', mec=m_colors[mouse],
                     ms=5, alpha=.3)
        plt.errorbar(x[1], np.nanmean(no_change_cs_1), yerr=0, c=m_colors[mouse], marker='o', mfc='none', mec=m_colors[mouse],
                     ms=5, alpha=.3)
        plt.errorbar(x[0], np.nanmean(increase_cs_1), yerr=0, c=m_colors[mouse], marker='o', mfc='none', mec=m_colors[mouse],
                     ms=5, alpha=.3)
        plt.plot([3, 2, 1], [np.nanmean(decrease_cs_1), np.nanmean(no_change_cs_1), np.nanmean(increase_cs_1)], c=m_colors[mouse], linewidth=2, linestyle='-', alpha=.2)
        plt.subplot(2, 2, 2)
        plt.errorbar(x[2], np.nanmean(decrease_cs_2), yerr=0, c=m_colors[mouse], marker='o', mfc='none', mec=m_colors[mouse],
                     ms=5, alpha=.3)
        plt.errorbar(x[1], np.nanmean(no_change_cs_2), yerr=0, c=m_colors[mouse], marker='o', mfc='none',
                     mec=m_colors[mouse],
                     ms=5, alpha=.3)
        plt.errorbar(x[0], np.nanmean(increase_cs_2), yerr=0, c=m_colors[mouse], marker='o', mfc='none', mec=m_colors[mouse],
                     ms=5, alpha=.3)
        plt.plot([3, 2, 1], [np.nanmean(decrease_cs_2), np.nanmean(no_change_cs_2), np.nanmean(increase_cs_2)], c=m_colors[mouse], linewidth=2, linestyle='-', alpha=.2)


    print(stats.f_oneway(decrease_cs_1_all, no_change_cs_1_all, increase_cs_1_all))
    print(
        pairwise_tukeyhsd(np.concatenate([decrease_cs_1_all, no_change_cs_1_all, increase_cs_1_all], axis=0),
                          ['d', 'd', 'd', 'd', 'd', 'd', 'd', 'd', 'n', 'n', 'n', 'n', 'n', 'n', 'n', 'n','i', 'i', 'i', 'i', 'i', 'i', 'i', 'i'], alpha=0.05))
    print(stats.f_oneway(decrease_cs_2_all, no_change_cs_2_all, increase_cs_2_all))
    print(
        pairwise_tukeyhsd(np.concatenate([decrease_cs_2_all, no_change_cs_2_all, increase_cs_2_all], axis=0),
                          ['d', 'd', 'd', 'd', 'd', 'd', 'd', 'd', 'n', 'n', 'n', 'n', 'n', 'n', 'n', 'n','i', 'i', 'i', 'i', 'i', 'i', 'i', 'i'], alpha=0.05))

    plt.subplot(2, 2, 1)
    y1 = np.mean(decrease_cs_1_all)
    y1_err = stats.sem(decrease_cs_1_all)
    plt.errorbar(3.12, y1, yerr=y1_err, c='mediumseagreen', linewidth=2, marker='o', mfc='mediumseagreen', mec='mediumseagreen',
                 ms=7, mew=0, zorder=100)
    y2 = np.mean(no_change_cs_1_all)
    y2_err = stats.sem(no_change_cs_1_all)
    plt.errorbar(2.12, y2, yerr=y2_err, c='mediumseagreen', linewidth=2, marker='o', mfc='mediumseagreen', mec='mediumseagreen',
                 ms=7, mew=0, zorder=100)
    y3 = np.mean(increase_cs_1_all)
    y3_err = stats.sem(increase_cs_1_all)
    plt.errorbar(1.12, y3, yerr=y3_err, c='mediumseagreen', linewidth=2, marker='o', mfc='mediumseagreen', mec='mediumseagreen',
                 ms=7, mew=0, zorder=100)
    plt.plot([3.12, 2.12, 1.12], [y1, y2, y3], '-', c='mediumseagreen', linewidth=3)
    plt.fill_between([3.12, 2.12, 1.12], [y1 + y1_err, y2 + y2_err, y3 + y3_err], [y1 - y1_err, y2 - y2_err, y3 - y3_err], alpha=0.2, color='mediumseagreen',
                     lw=0)

    plt.xlim(.5, 3.5)
    plt.ylim(-0.61, .61)
    plt.yticks([-.6, -.4, -.2, 0, .2, .4, .6])
    plt.xticks([1, 2, 3])
    # plt.ylabel('Stimulus activity\n(normalized deconvolved\n$\mathregular{Ca^{2+}}$transient $\mathregular{s^{-1}}$)')
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1)

    plt.subplot(2, 2, 2)
    y1 = np.mean(decrease_cs_2_all)
    y1_err = stats.sem(decrease_cs_2_all)
    plt.errorbar(3.12, y1, yerr=y1_err, c='salmon', linewidth=2, marker='o', mfc='salmon', mec='salmon',
                 ms=7, mew=0, zorder=100)
    y2 = np.mean(no_change_cs_2_all)
    y2_err = stats.sem(no_change_cs_2_all)
    plt.errorbar(2.12, y2, yerr=y2_err, c='salmon', linewidth=2, marker='o', mfc='salmon', mec='salmon',
                 ms=7, mew=0, zorder=100)
    y3 = np.mean(increase_cs_2_all)
    y3_err = stats.sem(increase_cs_2_all)
    plt.errorbar(1.12, y3, yerr=y3_err, c='salmon', linewidth=2, marker='o', mfc='salmon', mec='salmon',
                 ms=7, mew=0, zorder=100)
    plt.plot([3.12, 2.12, 1.12], [y1, y2, y3], '-', c='salmon', linewidth=3)
    plt.fill_between([3.12, 2.12, 1.12], [y1 + y1_err, y2 + y2_err, y3 + y3_err], [y1 - y1_err, y2 - y2_err, y3 - y3_err], alpha=0.2, color='salmon',
                     lw=0)
    plt.xlim(.5, 3.5)
    plt.yticks([-.6, -.4, -.2, 0, .2, .4, .6])
    plt.ylim(-0.61, .61)
    plt.xticks([1, 2, 3])
    # plt.ylabel('Stimulus activity\n(normalized deconvolved\n$\mathregular{Ca^{2+}}$transient $\mathregular{s^{-1}}$)')
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1)

    decrease_cs_1_all = []
    increase_cs_1_all = []
    no_change_cs_1_all = []
    decrease_cs_2_all = []
    increase_cs_2_all = []
    no_change_cs_2_all = []
    decrease_cs_1_all_opto = []
    increase_cs_1_all_opto = []
    no_change_cs_1_all_opto = []
    decrease_cs_2_all_opto = []
    increase_cs_2_all_opto = []
    no_change_cs_2_all_opto = []
    for mouse in range(0, len(mice)):
        paths = preprocess.create_folders(mice[mouse], sample_dates[mouse])
        reactivation_influence = np.load(
            paths['base_path'] + paths['mouse'] + '/data_across_days/reactivation_influence.npy', allow_pickle=True)
        decrease_cs_1 = []
        increase_cs_1 = []
        no_change_cs_1 = []
        decrease_cs_2 = []
        increase_cs_2 = []
        no_change_cs_2 = []
        decrease_cs_1_opto = []
        increase_cs_1_opto = []
        no_change_cs_1_opto = []
        decrease_cs_2_opto = []
        increase_cs_2_opto = []
        no_change_cs_2_opto = []
        for i in range(0, len(reactivation_influence[0])):
            if mouse < 5:
                decrease_cs_1.append(reactivation_influence[2][i] * fr)
                increase_cs_1.append(reactivation_influence[1][i] * fr)
                no_change_cs_1.append(reactivation_influence[0][i] * fr)
                decrease_cs_2.append(reactivation_influence[5][i] * fr)
                increase_cs_2.append(reactivation_influence[4][i] * fr)
                no_change_cs_2.append(reactivation_influence[3][i] * fr)
            if mouse > 4:
                decrease_cs_1_opto.append(reactivation_influence[2][i] * fr)
                increase_cs_1_opto.append(reactivation_influence[1][i] * fr)
                no_change_cs_1_opto.append(reactivation_influence[0][i] * fr)
                decrease_cs_2_opto.append(reactivation_influence[5][i] * fr)
                increase_cs_2_opto.append(reactivation_influence[4][i] * fr)
                no_change_cs_2_opto.append(reactivation_influence[3][i] * fr)
        if mouse < 5:
            decrease_cs_1_all.append(np.nanmean(decrease_cs_1))
            increase_cs_1_all.append(np.nanmean(increase_cs_1))
            no_change_cs_1_all.append(np.nanmean(no_change_cs_1))
            decrease_cs_2_all.append(np.nanmean(decrease_cs_2))
            increase_cs_2_all.append(np.nanmean(increase_cs_2))
            no_change_cs_2_all.append(np.nanmean(no_change_cs_2))
        if mouse > 4:
            decrease_cs_1_all_opto.append(np.nanmean(decrease_cs_1_opto))
            increase_cs_1_all_opto.append(np.nanmean(increase_cs_1_opto))
            no_change_cs_1_all_opto.append(np.nanmean(no_change_cs_1_opto))
            decrease_cs_2_all_opto.append(np.nanmean(decrease_cs_2_opto))
            increase_cs_2_all_opto.append(np.nanmean(increase_cs_2_opto))
            no_change_cs_2_all_opto.append(np.nanmean(no_change_cs_2_opto))

    plt.subplot(2, 2, 3)
    y1 = np.mean(decrease_cs_1_all)
    y1_err = stats.sem(decrease_cs_1_all)
    plt.errorbar(3.12, y1, yerr=y1_err, c='mediumseagreen', linewidth=2, marker='o', mfc='mediumseagreen', mec='mediumseagreen',
                 ms=7, mew=0, zorder=100)
    y2 = np.mean(no_change_cs_1_all)
    y2_err = stats.sem(no_change_cs_1_all)
    plt.errorbar(2.12, y2, yerr=y2_err, c='mediumseagreen', linewidth=2, marker='o', mfc='mediumseagreen', mec='mediumseagreen',
                 ms=7, mew=0, zorder=100)
    y3 = np.mean(increase_cs_1_all)
    y3_err = stats.sem(increase_cs_1_all)
    plt.errorbar(1.12, y3, yerr=y3_err, c='mediumseagreen', linewidth=2, marker='o', mfc='mediumseagreen', mec='mediumseagreen',
                 ms=7, mew=0, zorder=100)
    plt.plot([3.12, 2.12, 1.12], [y1, y2, y3], '-', c='mediumseagreen', linewidth=3)
    plt.fill_between([3.12, 2.12, 1.12], [y1 + y1_err, y2 + y2_err, y3 + y3_err], [y1 - y1_err, y2 - y2_err, y3 - y3_err], alpha=0.2, color='mediumseagreen',
                     lw=0)
    y1 = np.mean(decrease_cs_1_all_opto)
    y1_err = stats.sem(decrease_cs_1_all_opto)
    plt.errorbar(3.12, y1, yerr=y1_err, c='r', linewidth=2, marker='o', mfc='r',
                 mec='r',
                 ms=7, mew=0, zorder=100)
    y2 = np.mean(no_change_cs_1_all_opto)
    y2_err = stats.sem(no_change_cs_1_all_opto)
    plt.errorbar(2.12, y2, yerr=y2_err, c='r', linewidth=2, marker='o', mfc='r',
                 mec='r',
                 ms=7, mew=0, zorder=100)
    y3 = np.mean(increase_cs_1_all_opto)
    y3_err = stats.sem(increase_cs_1_all_opto)
    plt.errorbar(1.12, y3, yerr=y3_err, c='r', linewidth=2, marker='o', mfc='r',
                 mec='r',
                 ms=7, mew=0, zorder=100)
    plt.plot([3.12, 2.12, 1.12], [y1, y2, y3], '-', c='r', linewidth=3)
    plt.fill_between([3.12, 2.12, 1.12], [y1 + y1_err, y2 + y2_err, y3 + y3_err],
                     [y1 - y1_err, y2 - y2_err, y3 - y3_err], alpha=0.2, color='r',
                     lw=0)
    plt.xlim(.5, 3.5)
    plt.ylim(-0.61, .61)
    plt.yticks([-.6, -.4, -.2, 0, .2, .4, .6])
    plt.xticks([1, 2, 3])
    # plt.ylabel('Stimulus activity\n(normalized deconvolved\n$\mathregular{Ca^{2+}}$transient $\mathregular{s^{-1}}$)')
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1)

    plt.subplot(2, 2, 4)
    y1 = np.mean(decrease_cs_2_all)
    y1_err = stats.sem(decrease_cs_2_all)
    plt.errorbar(3.12, y1, yerr=y1_err, c='salmon', linewidth=2, marker='o', mfc='salmon', mec='salmon',
                 ms=7, mew=0, zorder=100)
    y2 = np.mean(no_change_cs_2_all)
    y2_err = stats.sem(no_change_cs_2_all)
    plt.errorbar(2.12, y2, yerr=y2_err, c='salmon', linewidth=2, marker='o', mfc='salmon', mec='salmon',
                 ms=7, mew=0, zorder=100)
    y3 = np.mean(increase_cs_2_all)
    y3_err = stats.sem(increase_cs_2_all)
    plt.errorbar(1.12, y3, yerr=y3_err, c='salmon', linewidth=2, marker='o', mfc='salmon', mec='salmon',
                 ms=7, mew=0, zorder=100)
    plt.plot([3.12, 2.12, 1.12], [y1, y2, y3], '-', c='salmon', linewidth=3)
    plt.fill_between([3.12, 2.12, 1.12], [y1 + y1_err, y2 + y2_err, y3 + y3_err], [y1 - y1_err, y2 - y2_err, y3 - y3_err], alpha=0.2, color='salmon',
                     lw=0)
    y1 = np.mean(decrease_cs_2_all_opto)
    y1_err = stats.sem(decrease_cs_2_all_opto)
    plt.errorbar(3.12, y1, yerr=y1_err, c='r', linewidth=2, marker='o', mfc='r', mec='r',
                 ms=7, mew=0, zorder=100)
    y2 = np.mean(no_change_cs_2_all_opto)
    y2_err = stats.sem(no_change_cs_2_all_opto)
    plt.errorbar(2.12, y2, yerr=y2_err, c='r', linewidth=2, marker='o', mfc='r', mec='r',
                 ms=7, mew=0, zorder=100)
    y3 = np.mean(increase_cs_2_all_opto)
    y3_err = stats.sem(increase_cs_2_all_opto)
    plt.errorbar(1.12, y3, yerr=y3_err, c='r', linewidth=2, marker='o', mfc='r', mec='r',
                 ms=7, mew=0, zorder=100)
    plt.plot([3.12, 2.12, 1.12], [y1, y2, y3], '-', c='r', linewidth=3)
    plt.fill_between([3.12, 2.12, 1.12], [y1 + y1_err, y2 + y2_err, y3 + y3_err],
                     [y1 - y1_err, y2 - y2_err, y3 - y3_err], alpha=0.2, color='r',
                     lw=0)
    plt.xlim(.5, 3.5)
    plt.yticks([-.6, -.4, -.2, 0, .2, .4, .6])
    plt.ylim(-0.61, .61)
    plt.xticks([1, 2, 3])
    # plt.ylabel('Stimulus activity\n(normalized deconvolved\n$\mathregular{Ca^{2+}}$transient $\mathregular{s^{-1}}$)')
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1)

    sns.despine()
    plt.savefig(paths['base_path'] + '/NN/plots/reactivation_cue_difference.png', bbox_inches='tight', dpi=200)
    plt.savefig(paths['base_path'] + '/NN/plots/reactivation_cue_difference.pdf', bbox_inches='tight', dpi=200,
                transparent=True)
    # plt.close()


def reactivation_cue_pattern_difference(mice, sample_dates):
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(5, 6))
    m_colors = ['b', 'teal', 'slategray', 'green', 'darkolivegreen', 'darkorange', 'purple', 'darkred']

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
    anova_results.append(stats.ttest_rel(cs1c_i_all, cs1r_i_all)[1])
    anova_results.append(stats.ttest_rel(cs2c_i_all, cs2r_i_all)[1])
    anova_results_corrected = multipletests(anova_results, alpha=.05, method='holm')
    print(anova_results_corrected[1])

    anova_results = []
    anova_results.append(stats.ttest_rel(cs1c_d_all, cs1r_d_all)[1])
    anova_results.append(stats.ttest_rel(cs2c_d_all, cs2r_d_all)[1])
    anova_results_corrected = multipletests(anova_results, alpha=.05, method='holm')
    print(anova_results_corrected[1])

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

    cs1c_d_all = []
    cs2c_d_all = []
    cs1c_i_all = []
    cs2c_i_all = []
    cs1r_d_all = []
    cs2r_d_all = []
    cs1r_i_all = []
    cs2r_i_all = []
    cs1c_d_all_opto = []
    cs2c_d_all_opto = []
    cs1c_i_all_opto = []
    cs2c_i_all_opto = []
    cs1r_d_all_opto = []
    cs2r_d_all_opto = []
    cs1r_i_all_opto = []
    cs2r_i_all_opto = []
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
        cs1c_d_opto = []
        cs2c_d_opto = []
        cs1c_i_opto = []
        cs2c_i_opto = []
        cs1r_d_opto = []
        cs2r_d_opto = []
        cs1r_i_opto = []
        cs2r_i_opto = []
        for i in range(0, len(diff_all[0])):
            if mouse < 5:
                cs1c_d.append(diff_all[0][i])
                cs2c_d.append(diff_all[1][i])
                cs1c_i.append(diff_all[2][i])
                cs2c_i.append(diff_all[3][i])
                cs1r_d.append(diff_all[4][i])
                cs2r_d.append(diff_all[5][i])
                cs1r_i.append(diff_all[6][i])
                cs2r_i.append(diff_all[7][i])
            if mouse > 4:
                cs1c_d_opto.append(diff_all[0][i])
                cs2c_d_opto.append(diff_all[1][i])
                cs1c_i_opto.append(diff_all[2][i])
                cs2c_i_opto.append(diff_all[3][i])
                cs1r_d_opto.append(diff_all[4][i])
                cs2r_d_opto.append(diff_all[5][i])
                cs1r_i_opto.append(diff_all[6][i])
                cs2r_i_opto.append(diff_all[7][i])
        cs1c_d = np.nanmean(cs1c_d)
        cs2c_d = np.nanmean(cs2c_d)
        cs1c_i = np.nanmean(cs1c_i)
        cs2c_i = np.nanmean(cs2c_i)
        cs1r_d = np.nanmean(cs1r_d)
        cs2r_d = np.nanmean(cs2r_d)
        cs1r_i = np.nanmean(cs1r_i)
        cs2r_i = np.nanmean(cs2r_i)
        cs1c_d_opto = np.nanmean(cs1c_d_opto)
        cs2c_d_opto = np.nanmean(cs2c_d_opto)
        cs1c_i_opto = np.nanmean(cs1c_i_opto)
        cs2c_i_opto = np.nanmean(cs2c_i_opto)
        cs1r_d_opto = np.nanmean(cs1r_d_opto)
        cs2r_d_opto = np.nanmean(cs2r_d_opto)
        cs1r_i_opto = np.nanmean(cs1r_i_opto)
        cs2r_i_opto = np.nanmean(cs2r_i_opto)
        cs1c_d_all.append(cs1c_d)
        cs2c_d_all.append(cs2c_d)
        cs1c_i_all.append(cs1c_i)
        cs2c_i_all.append(cs2c_i)
        cs1r_d_all.append(cs1r_d)
        cs2r_d_all.append(cs2r_d)
        cs1r_i_all.append(cs1r_i)
        cs2r_i_all.append(cs2r_i)
        cs1c_d_all_opto.append(cs1c_d_opto)
        cs2c_d_all_opto.append(cs2c_d_opto)
        cs1c_i_all_opto.append(cs1c_i_opto)
        cs2c_i_all_opto.append(cs2c_i_opto)
        cs1r_d_all_opto.append(cs1r_d_opto)
        cs2r_d_all_opto.append(cs2r_d_opto)
        cs1r_i_all_opto.append(cs1r_i_opto)
        cs2r_i_all_opto.append(cs2r_i_opto)

    # anova_results = []
    # anova_results.append(stats.ttest_rel(cs1c_d_all, cs1r_d_all)[1])
    # anova_results.append(stats.ttest_rel(cs2c_d_all, cs2r_d_all)[1])
    # anova_results_corrected = multipletests(anova_results, alpha=.05, method='holm')
    # return [anova_results, anova_results_corrected[1]]

    y0 = np.nanmean(cs1c_d_all)
    y1 = np.nanmean(cs2c_d_all)
    y2 = np.nanmean(cs1c_i_all)
    y3 = np.nanmean(cs2c_i_all)
    y4 = np.nanmean(cs1r_d_all)
    y5 = np.nanmean(cs2r_d_all)
    y6 = np.nanmean(cs1r_i_all)
    y7 = np.nanmean(cs2r_i_all)
    y0_err = stats.sem(cs1c_d_all, nan_policy='omit')
    y1_err = stats.sem(cs2c_d_all, nan_policy='omit')
    y2_err = stats.sem(cs1c_i_all, nan_policy='omit')
    y3_err = stats.sem(cs2c_i_all, nan_policy='omit')
    y4_err = stats.sem(cs1r_d_all, nan_policy='omit')
    y5_err = stats.sem(cs2r_d_all, nan_policy='omit')
    y6_err = stats.sem(cs1r_i_all, nan_policy='omit')
    y7_err = stats.sem(cs2r_i_all, nan_policy='omit')
    plt.subplot(2, 2, 3)
    plt.plot(x[0:2], [y0, y4], '-', c='mediumseagreen', linewidth=3)
    plt.fill_between(x[0:2], [y0 + y0_err, y4 + y4_err], [y0 - y0_err, y4 - y4_err], alpha=0.2, color='mediumseagreen',
                     lw=0)
    plt.plot(x[2:4], [y1, y5], '-', c='salmon', linewidth=3)
    plt.fill_between(x[2:4], [y1 + y1_err, y5 + y5_err], [y1 - y1_err, y5 - y5_err], alpha=0.2, color='salmon', lw=0)
    plt.xticks([1, 2, 3, 4], ['S1', 'S1R', 'S2', 'S2R'])
    plt.ylabel('Ratio of decrease /\nno change neuron activity')
    plt.ylim((0, 2))
    plt.xlim((.5, 4.5))
    plt.subplot(2, 2, 4)
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

    y0 = np.nanmean(cs1c_d_all_opto)
    y1 = np.nanmean(cs2c_d_all_opto)
    y2 = np.nanmean(cs1c_i_all_opto)
    y3 = np.nanmean(cs2c_i_all_opto)
    y4 = np.nanmean(cs1r_d_all_opto)
    y5 = np.nanmean(cs2r_d_all_opto)
    y6 = np.nanmean(cs1r_i_all_opto)
    y7 = np.nanmean(cs2r_i_all_opto)
    y0_err = stats.sem(cs1c_d_all_opto, nan_policy='omit')
    y1_err = stats.sem(cs2c_d_all_opto, nan_policy='omit')
    y2_err = stats.sem(cs1c_i_all_opto, nan_policy='omit')
    y3_err = stats.sem(cs2c_i_all_opto, nan_policy='omit')
    y4_err = stats.sem(cs1r_d_all_opto, nan_policy='omit')
    y5_err = stats.sem(cs2r_d_all_opto, nan_policy='omit')
    y6_err = stats.sem(cs1r_i_all_opto, nan_policy='omit')
    y7_err = stats.sem(cs2r_i_all_opto, nan_policy='omit')
    plt.subplot(2, 2, 3)
    plt.plot(x[0:2], [y0, y4], '-', c='r', linewidth=3)
    plt.fill_between(x[0:2], [y0 + y0_err, y4 + y4_err], [y0 - y0_err, y4 - y4_err], alpha=0.2, color='r',
                     lw=0)
    plt.plot(x[2:4], [y1, y5], '-', c='r', linewidth=3)
    plt.fill_between(x[2:4], [y1 + y1_err, y5 + y5_err], [y1 - y1_err, y5 - y5_err], alpha=0.2, color='r', lw=0)
    plt.subplot(2, 2, 4)
    plt.plot(x[0:2], [y2, y6], '-', c='r', linewidth=3)
    plt.fill_between(x[0:2], [y2 + y2_err, y6 + y6_err], [y2 - y2_err, y6 - y6_err], alpha=0.2, color='r',
                     lw=0)
    plt.plot(x[2:4], [y3, y7], '-', c='r', linewidth=3)
    plt.fill_between(x[2:4], [y3 + y3_err, y7 + y7_err], [y3 - y3_err, y7 - y7_err], alpha=0.2, color='plum',
                     lw=0)
    sns.despine()
    plt.savefig(paths['base_path'] + '/NN/plots/reactivation_cue_pattern_difference.png', bbox_inches='tight', dpi=200)
    plt.savefig(paths['base_path'] + '/NN/plots/reactivation_cue_pattern_difference.pdf', bbox_inches='tight', dpi=200, transparent=True)
    plt.close()


def neuron_count_R2(mice, sample_dates):
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(9, 3))
    fig.subplots_adjust(hspace=.05)

    m_colors = ['b', 'teal', 'slategray', 'green', 'darkolivegreen', 'darkorange', 'purple', 'darkred']
    idx = [1, 2, 3, 4, 5, 8, 6, 9, 7, 10]
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    all_vec = np.empty((len(mice), len(x))) * np.nan
    mean_neuron = []
    mean_sig = []
    for mouse in range(0, len(mice)):
        paths = preprocess.create_folders(mice[mouse], sample_dates[mouse])
        grouped_count = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/grouped_count.npy', allow_pickle=True)
        vec = np.empty((100, len(x))) * np.nan
        for i in range(0, len(grouped_count[0])):
            for j in range(0, len(x)):
                vec[i, j] = grouped_count[idx[j]-1][i]
                if j == 0:
                    mean_neuron.append(grouped_count[idx[j]-1][i])
                    num_sig = grouped_count[idx[j+1]-1][i] + grouped_count[idx[j+2]-1][i] + grouped_count[idx[j+3]-1][i]
                    mean_sig.append(num_sig)
        vec = np.nanmean(vec, axis=0)

        for i in range(0, len(vec)):
            ax1.errorbar(x[i], vec[i], yerr=0, c=m_colors[mouse], marker='o', mfc='none', mec=m_colors[mouse],
                         ms=5, alpha=.3)
            ax2.errorbar(x[i], vec[i], yerr=0, c=m_colors[mouse], marker='o', mfc='none', mec=m_colors[mouse],
                             ms=5, alpha=.3)
            all_vec[mouse, i] = vec[i]
    print(np.mean(mean_neuron))
    print(stats.sem(mean_neuron))
    print(np.mean(mean_sig))
    print(stats.sem(mean_sig))
    for i in range(0, len(all_vec[0])):
        ax1.errorbar(x[i]+.2, np.mean(all_vec[:,i]), yerr=stats.sem(all_vec[:,i]), c='k', linewidth=2, marker='o', mfc='k', mec='k',
                     ms=7, mew=0, zorder=100)
        ax2.errorbar(x[i] + .2, np.mean(all_vec[:, i]), yerr=stats.sem(all_vec[:, i]), c='k', linewidth=2, marker='o',
                     mfc='k', mec='k',
                     ms=7, mew=0, zorder=100)
    ax1.set_ylim(4000, 8500)  # outliers only
    ax1.set_yticks([5000, 8000])
    ax2.set_ylim(0, 1000)  # most of the data
    ax2.set_yticks([0, 400, 800])
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.xaxis.set_ticks_position('none')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    d = .5
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                  linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax1.plot([0, 0], [0, 0], transform=ax1.transAxes, **kwargs)
    ax2.plot([0, 0], [1, 1], transform=ax2.transAxes, **kwargs)

    plt.xlim(.5, 10.5)
    plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], ['All neurons', 'S1+S2', 'S1', 'S2', 'S1+S2\nearly', 'S1+S2\nlate', 'S1\nearly', 'S1\nlate', 'S2\nearly', 'S2\nlate'])
    plt.ylabel('Number of neurons')

    label_0 = mlines.Line2D([], [], color='slategray', marker='.', linestyle='None', markersize=10, label='Mouse 1', linewidth=2)
    label_1 = mlines.Line2D([], [], color='b', marker='.', linestyle='None', markersize=10,  label='Mouse 2', linewidth=2)
    label_2 = mlines.Line2D([], [], color='green', marker='.', linestyle='None', markersize=10,  label='Mouse 3', linewidth=2)
    label_3 = mlines.Line2D([], [], color='teal', marker='.', linestyle='None', markersize=10,  label='Mouse 4', linewidth=2)
    label_4 = mlines.Line2D([], [], color='darkolivegreen', marker='.', linestyle='None', markersize=10,  label='Mouse 5', linewidth=2)
    label_5 = mlines.Line2D([], [], color='darkorange', marker='.', linestyle='None', markersize=10,  label='Mouse 6', linewidth=2)
    label_6 = mlines.Line2D([], [], color='purple', marker='.', linestyle='None', markersize=10,  label='Mouse 7', linewidth=2)
    label_7 = mlines.Line2D([], [], color='darkred', marker='.', linestyle='None', markersize=10,  label='Mouse 8', linewidth=2)
    plt.legend(handles=[label_0, label_1, label_2, label_3, label_4, label_5, label_6, label_7], frameon=False, labelspacing=.1)

    plt.savefig(paths['base_path'] + '/NN/plots/grouped_count.png', bbox_inches='tight', dpi=200)
    plt.savefig(paths['base_path'] + '/NN/plots/grouped_count.pdf', bbox_inches='tight', dpi=200,
                transparent=True)
    #plt.close()


def neuron_overlap_dist_R3(mice, sample_dates):
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(6, 6))
    m_colors = ['b', 'teal', 'slategray', 'green', 'darkolivegreen', 'darkorange', 'purple', 'darkred']
    paths = preprocess.create_folders(mice[0], sample_dates[0])
    dist_all = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/overlap_dist.npy',
                    allow_pickle=True)
    dist_total = np.zeros((len(mice), len(dist_all[0][0])))

    for mouse in range(0, len(mice)):
        paths = preprocess.create_folders(mice[mouse], sample_dates[mouse])
        dist_all = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/overlap_dist.npy', allow_pickle=True)
        dist = []
        for i in range(0, len(dist_all[0])):
            dist.append(dist_all[0][i])
        plt.subplot(2, 2, 1)
        plt.plot(list(range(0, len(np.mean(dist, axis=0)))), np.mean(dist, axis=0), c=m_colors[mouse], alpha=.2)
        dist_total[mouse, :] = np.mean(dist, axis=0)
    plt.subplot(2, 2, 1)
    mean_dist = dist_total.mean(axis=0)
    sem_plus = mean_dist + stats.sem(dist_total, axis=0)
    sem_minus = mean_dist - stats.sem(dist_total, axis=0)
    plt.plot(list(range(0, len(mean_dist))), mean_dist, c='k', linewidth=2)
    plt.fill_between(list(range(0, len(mean_dist))), sem_plus, sem_minus, alpha=0.2, color='k', lw=0)

    plt.ylim(0, 1.1)
    plt.ylabel('Density')
    plt.xlabel('Stimulus preference')
    plt.xticks([0, int(len(mean_dist)/4), int(len(mean_dist)/2), int(len(mean_dist)*3/4), len(mean_dist)],
               ['-1', '.5', '0', '.5', '1'])
    sns.despine()
    plt.savefig(paths['base_path'] + '/NN/plots/overlap_dist.png', bbox_inches='tight', dpi=200)
    plt.savefig(paths['base_path'] + '/NN/plots/overlap_dist.pdf', bbox_inches='tight', dpi=200, transparent=True)
    plt.close()


def reactivation_layer_R1(mice, sample_dates):
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(wspace=.45)
    m_colors = ['b', 'teal', 'slategray', 'green', 'darkolivegreen', 'darkorange', 'purple', 'darkred']
    x = [1, 2]
    upper_cue_all = []
    lower_cue_all = []
    upper_reactivation_all = []
    lower_reactivation_all = []
    for mouse in range(0, len(mice)):
        paths = preprocess.create_folders(mice[mouse], sample_dates[mouse])
        reactivation_layer = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/reactivation_layer.npy', allow_pickle=True)
        upper_cue = []
        lower_cue = []
        upper_reactivation = []
        lower_reactivation = []
        for i in range(0, len(reactivation_layer[0])):
            upper_cue.append(reactivation_layer[0][i])
            lower_cue.append(reactivation_layer[1][i])
            upper_reactivation.append(reactivation_layer[2][i])
            lower_reactivation.append(reactivation_layer[3][i])
        upper_cue_all.append(np.nanmean(upper_cue))
        lower_cue_all.append(np.nanmean(lower_cue))
        upper_reactivation_all.append(np.nanmean(upper_reactivation))
        lower_reactivation_all.append(np.nanmean(lower_reactivation))

        plt.subplot(2, 2, 1)
        plt.errorbar(x[0], np.nanmean(upper_cue), yerr=0, c=m_colors[mouse], marker='o', mfc='none', mec=m_colors[mouse],
                     ms=5, alpha=.3)
        plt.errorbar(x[1], np.nanmean(lower_cue), yerr=0, c=m_colors[mouse], marker='o', mfc='none', mec=m_colors[mouse],
                     ms=5, alpha=.3)
        plt.subplot(2, 2, 2)
        plt.errorbar(x[0], np.nanmean(upper_reactivation), yerr=0, c=m_colors[mouse], marker='o', mfc='none', mec=m_colors[mouse],
                     ms=5, alpha=.3)
        plt.errorbar(x[1], np.nanmean(lower_reactivation), yerr=0, c=m_colors[mouse], marker='o', mfc='none',
                     mec=m_colors[mouse],
                     ms=5, alpha=.3)

    [_, s_p_value] = stats.shapiro(upper_cue_all)
    print(s_p_value)

    print(stats.ranksums(upper_cue_all, lower_cue_all))
    print(stats.ranksums(upper_reactivation_all, lower_reactivation_all))

    plt.subplot(2, 2, 1)
    y1 = np.mean(upper_cue_all)
    y1_err = stats.sem(upper_cue_all)
    plt.errorbar(1.2, y1, yerr=y1_err, c='k', linewidth=2, marker='o', mfc='k', mec='k',
                 ms=7, mew=0, zorder=100)
    y2 = np.mean(lower_cue_all)
    y2_err = stats.sem(lower_cue_all)
    plt.errorbar(2.2, y2, yerr=y2_err, c='k', linewidth=2, marker='o', mfc='k', mec='k',
                 ms=7, mew=0, zorder=100)
    plt.xlim(.5, 2.5)
    plt.ylim(0, .9)
    plt.xticks([1, 2], ['Upper layer\nneurons (~140 µm)', 'Lower layer\nneurons (~270 µm)'])
    plt.ylabel('Stimulus activity\n(normalized deconvolved\n$\mathregular{Ca^{2+}}$transient $\mathregular{s^{-1}}$)')

    plt.subplot(2, 2, 2)
    y1 = np.mean(upper_reactivation_all)
    y1_err = stats.sem(upper_reactivation_all)
    plt.errorbar(1.2, y1, yerr=y1_err, c='k', linewidth=2, marker='o', mfc='k', mec='k',
                 ms=7, mew=0, zorder=100)
    y2 = np.mean(lower_reactivation_all)
    y2_err = stats.sem(lower_reactivation_all)
    plt.errorbar(2.2, y2, yerr=y2_err, c='k', linewidth=2, marker='o', mfc='k', mec='k',
                 ms=7, mew=0, zorder=100)
    plt.xlim(.5, 2.5)
    plt.ylim(0, .9)
    plt.xticks([1, 2], ['Upper layer\nneurons (~140 µm)', 'Lower layer\nneurons (~270 µm)'])
    plt.ylabel('Reactivation activity\n(normalized deconvolved\n$\mathregular{Ca^{2+}}$transient $\mathregular{s^{-1}}$)')

    sns.despine()
    plt.savefig(paths['base_path'] + '/NN/plots/cue_reactivation_layer.png', bbox_inches='tight', dpi=200)
    plt.savefig(paths['base_path'] + '/NN/plots/cue_reactivation_layer.pdf', bbox_inches='tight', dpi=200,
                transparent=True)
    plt.close()


def reactivation_syn_iti_R2(mice, sample_dates):
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(9, 6))
    m_colors = ['b', 'teal', 'slategray', 'green', 'darkolivegreen', 'darkorange', 'purple', 'darkred']

    plt.subplot(2, 2, 1)
    syn_all = []
    iti_all = []
    x = [1, 2]
    for mouse in range(0, len(mice)):
        paths = preprocess.create_folders(mice[mouse], sample_dates[mouse])
        vec_all = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/reactivation_syn_iti.npy',
                             allow_pickle=True)
        session_data = preprocess.load_data(paths)
        syn_vec = []
        iti_vec = []
        for i in range(0, len(vec_all[0])):
            syn = vec_all[0][i]
            iti = vec_all[1][i]
            syn_vec.append(syn)
            iti_vec.append(iti)
        syn_vec = np.mean(syn_vec)
        iti_vec = np.mean(iti_vec)
        plt.plot(x, [syn_vec, iti_vec], c=m_colors[mouse], linewidth=2, linestyle='-', alpha=.2)
        syn_all.append(syn_vec)
        iti_all.append(iti_vec)

    [_, s_p_value] = stats.shapiro(syn_all)
    print(s_p_value)
    print(stats.ttest_rel(syn_all, iti_all))

    y0 = np.mean(syn_all)
    y1 = np.mean(iti_all)
    y0_err = stats.sem(syn_all)
    y1_err = stats.sem(iti_all)
    plt.plot(x, [y0, y1], '-', c='k', linewidth=3)
    plt.fill_between(x, [y0 + y0_err, y1 + y1_err], [y0 - y0_err, y1 - y1_err], alpha=0.2, color='k', lw=0)
    plt.xlim((.5, 2.5))
    plt.ylim((0, 1))
    plt.xticks([1, 2], ['Synchronous', 'Other ITI'])
    # plt.yticks([.2, .25, .3, .35, .4])
    plt.gca().get_xticklabels()[1].set_color('k')
    plt.gca().get_xticklabels()[0].set_color('k')
    plt.ylabel('Reactivation rate (probablity $\mathregular{s^{-1}}$)')
    sns.despine()
    plt.savefig(paths['base_path'] + '/NN/plots/reactivation_rate_syn_iti.png', bbox_inches='tight', dpi=200)
    plt.savefig(paths['base_path'] + '/NN/plots/reactivation_rate_syn_iti.pdf', bbox_inches='tight', dpi=200, transparent=True)
    plt.close()


def top_activity_stable_R3(mice, sample_dates):
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(5, 6))
    plt.subplots_adjust(wspace=.45)
    m_colors = ['b', 'teal', 'slategray', 'green', 'darkolivegreen', 'darkorange', 'purple', 'darkred']
    x = [1, 2]
    cs_1_all = []
    cs_2_all = []
    for mouse in range(0, len(mice)):
        paths = preprocess.create_folders(mice[mouse], sample_dates[mouse])
        overlap_vec = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/top_activity_stable.npy', allow_pickle=True)
        cs_1 = []
        cs_2 = []
        for i in range(0, len(overlap_vec[0])):
            cs_1.append(overlap_vec[0][i])
            cs_2.append(overlap_vec[0][i])
        cs_1_all.append(np.nanmean(cs_1))
        cs_2_all.append(np.nanmean(cs_2))
        plt.subplot(2, 2, 1)
        plt.errorbar(x[0], np.nanmean(cs_1), yerr=0, c=m_colors[mouse], marker='o', mfc='none', mec=m_colors[mouse],
                     ms=5, alpha=.3)
        plt.errorbar(x[1], np.nanmean(cs_2), yerr=0, c=m_colors[mouse], marker='o', mfc='none', mec=m_colors[mouse],
                     ms=5, alpha=.3)


    plt.subplot(2, 2, 1)
    y1 = np.mean(cs_1_all)
    y1_err = stats.sem(cs_1_all)
    plt.errorbar(1.2, y1, yerr=y1_err, c='k', linewidth=2, marker='o', mfc='k', mec='k',
                 ms=7, mew=0, zorder=100)
    y2 = np.mean(cs_2_all)
    y2_err = stats.sem(cs_2_all)
    plt.errorbar(2.2, y2, yerr=y2_err, c='k', linewidth=2, marker='o', mfc='k', mec='k',
                 ms=7, mew=0, zorder=100)

    plt.xlim(.5, 2.5)
    plt.ylim(0, 1)
    plt.xticks([1, 2], ['S1', 'S2'])
    plt.ylabel('Fraction of top neurons\nshared early vs. late')

    sns.despine()
    plt.savefig(paths['base_path'] + '/NN/plots/top_activity_stable.png', bbox_inches='tight', dpi=200)
    plt.savefig(paths['base_path'] + '/NN/plots/top_activity_stable.pdf', bbox_inches='tight', dpi=200,
                transparent=True)
    plt.close()


def face_track_R1(mice, sample_dates):
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(4, 24))
    facemap_features = ['Nose (tip)', 'Nose (bottom)', 'Nose (top)', 'Mouth', 'Lower lip', 'Whisker 1', 'Whisker 2',
                        'Whisker 3']
    plt_idx = 0
    for feature in facemap_features:
        plt.subplot(len(facemap_features), 1, plt_idx+1)
        face_track_R1_helper(mice, sample_dates, plt_idx, feature)
        plt_idx += 1

    sns.despine()
    paths = preprocess.create_folders(mice[0], sample_dates[0])
    plt.savefig(paths['base_path'] + '/NN/plots/face_across_trials.png', bbox_inches='tight', dpi=200)
    plt.savefig(paths['base_path'] + '/NN/plots/face_across_trials.pdf', bbox_inches='tight', dpi=200, transparent=True)
    plt.close()


def face_track_R1_helper(mice, sample_dates, plt_idx, feature):
    m_colors = ['b', 'purple', 'darkorange', 'green', 'darkred']
    paths = preprocess.create_folders(mice[0], sample_dates[0])
    session_data = preprocess.load_data(paths)
    task_runs = int(session_data['task_runs'])
    dark_runs = int(session_data['dark_runs'])
    x_label = np.zeros(9)
    hours_per_run = 64000 / 31.25 / 60 / 60
    for i in range(0, dark_runs):
        x_label[dark_runs - i - 1] = - hours_per_run / 2
    for i in range(dark_runs, dark_runs + (task_runs * 2)):
        x_label[i] = hours_per_run / 4 + (hours_per_run / 2 * (i - dark_runs))
    mean_feature_mice = np.empty((len(mice), 9)) * np.nan
    for mouse in range(0, len(mice)):
        paths = preprocess.create_folders(mice[mouse], sample_dates[mouse])
        if os.path.isfile(paths['base_path'] + paths['mouse'] + '/data_across_days/facemap_features.npy'):
            binned_vec = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/facemap_features.npy',
                                 allow_pickle=True)
            mean_feature = np.empty((len(binned_vec), 9)) * np.nan
            for i in range(0, len(binned_vec)):
                if not isinstance(binned_vec[plt_idx][i], int):
                    mean_feature[i, :] = binned_vec[plt_idx][i]
            mean = np.nanmean(mean_feature, axis=0)
            plt.plot(x_label, mean, '-', c=m_colors[mouse], alpha=.2, linewidth=2)
            mean_feature_mice[mouse, :] = mean

    slope, intercept, r_value, p_value, std_err = stats.linregress(
        np.concatenate(
            [x_label[1:9], x_label[1:9], x_label[1:9]]),
        np.concatenate(mean_feature_mice[~np.isnan(mean_feature_mice).any(axis=1)][:, 1:9]))
    print(r_value)
    anova_results = []
    anova_results.append(stats.ttest_rel(mean_feature_mice[~np.isnan(mean_feature_mice).any(axis=1)][:, 1], mean_feature_mice[~np.isnan(mean_feature_mice).any(axis=1)][:, 0])[1])
    anova_results.append(p_value)
    anova_results_corrected = multipletests(anova_results, alpha=.05, method='holm')
    print(anova_results_corrected[1])

    mean = np.nanmean(mean_feature_mice, axis=0)
    sem_plus = mean + stats.sem(mean_feature_mice, axis=0, nan_policy='omit')
    sem_minus = mean - stats.sem(mean_feature_mice, axis=0, nan_policy='omit')
    plt.plot(x_label, mean, '-', c='k', linewidth=3)
    plt.fill_between(x_label, sem_plus, sem_minus, alpha=0.2, color='k', lw=0)
    plt.axvspan(-hours_per_run, 0, alpha=.1, color='gray', lw=0)
    plt.ylabel(feature + '\nmovement (abs mm)')
    plt.xlim(-hours_per_run, x_label[len(x_label) - 1] + hours_per_run / 4)
    plt.xlabel('Time relative to stimulus onset (h)')
    plt.ylim(0, .5)


def behavior_across_trials_R2(mice, sample_dates):
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(9, 6))
    plt.subplots_adjust(wspace=.33)
    m_colors = ['b', 'teal', 'slategray', 'green', 'darkolivegreen', 'darkorange', 'purple', 'darkred']

    plt.subplot(2, 2, 1)
    all_vec = np.empty((len(mice), 128)) * np.nan
    for mouse in range(0, len(mice)):
        paths = preprocess.create_folders(mice[mouse], sample_dates[mouse])
        all_data = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/behavior_across_days.npy',
                                allow_pickle=True)
        vec = np.empty((len(all_data[0]), 128)) * np.nan
        for i in range(0, len(all_data[0])):
            smoothed_vec = np.array(
                pd.DataFrame(all_data[1][i]).rolling(3, min_periods=1, center=True).mean())
            if mouse < 5:
                smoothed_vec = np.concatenate(smoothed_vec, axis=0)
                vec[i, 0:len(smoothed_vec)] = smoothed_vec
            if mouse > 4:
                smoothed_vec = np.concatenate(smoothed_vec, axis=0)
                smoothed_vec = scipy.signal.resample_poly(smoothed_vec, 2, 1, padtype='smooth')[0:(len(smoothed_vec) * 2)]
                vec[i, 0:len(smoothed_vec)] = smoothed_vec
        vec = np.nanmean(vec, axis=0)
        all_vec[mouse, :] = vec
        x = range(1, len(vec) + 1)
        plt.plot(x[0:121], vec[0:121], c=m_colors[mouse], alpha=.2)

    print(stats.ttest_rel(np.concatenate(all_vec[:, 0:3], axis=0), np.concatenate(all_vec[:, 118:121], axis=0), nan_policy='omit'))

    mean = np.nanmean(all_vec, axis=0)
    sem_plus = mean + stats.sem(all_vec, axis=0, nan_policy='omit')
    sem_minus = mean - stats.sem(all_vec, axis=0, nan_policy='omit')
    plt.plot(x[0:121], mean[0:121], c='k', linewidth=3)
    plt.fill_between(x[0:121], sem_plus[0:121], sem_minus[0:121], alpha=0.2, color='k', lw=0)

    plt.ylabel('Max normalized pupil area\nduring stimulus presentation')
    #plt.xlabel('Trial')
    plt.ylim(0, 1)
    plt.xlim(1, 122)
    plt.xticks([1, 121], ['First trial', 'Last trial'])

    plt.subplot(2, 2, 2)
    all_vec = np.empty((len(mice), 128)) * np.nan
    for mouse in range(0, len(mice)):
        paths = preprocess.create_folders(mice[mouse], sample_dates[mouse])
        all_data = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/behavior_across_days.npy',
                           allow_pickle=True)
        vec = np.empty((len(all_data[0]), 128)) * np.nan
        for i in range(0, len(all_data[0])):
            smoothed_vec = np.array(
                pd.DataFrame(all_data[0][i]).rolling(3, min_periods=1, center=True).mean())
            if mouse < 5:
                smoothed_vec = np.concatenate(smoothed_vec, axis=0)
                vec[i, 0:len(smoothed_vec)] = smoothed_vec
            if mouse > 4:
                smoothed_vec = np.concatenate(smoothed_vec, axis=0)
                smoothed_vec = scipy.signal.resample_poly(smoothed_vec, 2, 1, padtype='smooth')[0:(len(smoothed_vec) * 2)]
                vec[i, 0:len(smoothed_vec)] = smoothed_vec
        vec = np.nanmean(vec, axis=0)
        all_vec[mouse, :] = vec
        x = range(1, len(vec) + 1)
        plt.plot(x[0:121], vec[0:121], c=m_colors[mouse], alpha=.2)

    print(stats.ttest_rel(np.concatenate(all_vec[:, 0:3], axis=0), np.concatenate(all_vec[:, 118:121], axis=0),
                          nan_policy='omit'))

    mean = np.nanmean(all_vec, axis=0)
    sem_plus = mean + stats.sem(all_vec, axis=0, nan_policy='omit')
    sem_minus = mean - stats.sem(all_vec, axis=0, nan_policy='omit')
    plt.plot(x[0:121], mean[0:121], c='k', linewidth=3)
    plt.fill_between(x[0:121], sem_plus[0:121], sem_minus[0:121], alpha=0.2, color='k', lw=0)

    plt.ylabel('Max normalized pupil area\nduring stimulus presentation')
    # plt.xlabel('Trial')
    plt.ylim(0, 1)
    plt.xlim(1, 122)
    plt.xticks([1, 121], ['First trial', 'Last trial'])
    sns.despine()

    plt.savefig(paths['base_path'] + '/NN/plots/behavior_across_trials.png', bbox_inches='tight', dpi=200)
    plt.savefig(paths['base_path'] + '/NN/plots/behavior_across_trials.pdf', bbox_inches='tight', dpi=200, transparent=True)
    plt.close()


def reactivation_rate_day_comparison_R1(mice, sample_dates):
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(9, 6))
    plt.subplot(2, 2, 1)
    reactivation_all = np.zeros(5)
    reactivation_opto_all = np.zeros(3)
    reactivation_control_all = np.zeros(3)
    reactivation_mean_all = np.zeros(3)

    for mouse in range(0, 5):
        paths = preprocess.create_folders(mice[mouse], sample_dates[mouse])
        y_pred_binned_across_days = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/y_pred_binned.npy',
                                            allow_pickle=True)
        reactivation = np.zeros(len(y_pred_binned_across_days))
        for i in range(0, len(y_pred_binned_across_days)):
            reactivation[i] = np.mean(y_pred_binned_across_days[i][1:len(y_pred_binned_across_days[i])])
        reactivation_all[mouse] = reactivation.mean()
        plt.subplot(2, 2, 1)
        plt.errorbar(1, reactivation.mean(), yerr=0, c='k', marker='o', mfc='none',mec='k', ms=5, alpha=.3)
    for mouse in range(5, len(mice)):
        paths = preprocess.create_folders(mice[mouse], sample_dates[mouse])
        y_pred_binned_across_days = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/y_pred_binned.npy',
                                            allow_pickle=True)
        reactivation = np.zeros(len(y_pred_binned_across_days[0]))
        reactivation_opto = np.zeros(len(y_pred_binned_across_days[0]))
        reactivation_control = np.zeros(len(y_pred_binned_across_days[0]))
        for i in range(0, len(y_pred_binned_across_days[0])):
            reactivation[i] = np.mean([y_pred_binned_across_days[0][i][1:len(y_pred_binned_across_days[0][i])], y_pred_binned_across_days[1][i][1:len(y_pred_binned_across_days[1][i])]])
            reactivation_control[i] = np.mean(y_pred_binned_across_days[0][i][1:len(y_pred_binned_across_days[0][i])])
            reactivation_opto[i] = np.mean(y_pred_binned_across_days[1][i][1:len(y_pred_binned_across_days[1][i])])
        reactivation_mean_all[mouse-5] = reactivation.mean()
        reactivation_opto_all[mouse-5] = reactivation_opto.mean()
        reactivation_control_all[mouse-5] = reactivation_control.mean()
        plt.subplot(2, 2, 1)
        plt.errorbar(2, reactivation.mean(), yerr=0, c='r', marker='o', mfc='none',mec='darkred', ms=5, alpha=.3)
        plt.errorbar(3, reactivation_control.mean(), yerr=0, c='r', marker='o', mfc='none',mec='darkblue', ms=5, alpha=.3)
        plt.errorbar(4, reactivation_opto.mean(), yerr=0, c='r', marker='o', mfc='none',mec='r', ms=5, alpha=.3)

    anova_results = []
    anova_results.append(stats.ttest_ind(reactivation_all, reactivation_mean_all)[1])
    anova_results.append(stats.ttest_ind(reactivation_all, reactivation_control_all)[1])
    anova_results_corrected = multipletests(anova_results, alpha=.05, method='holm')
    print(anova_results_corrected[1])
    plt.subplot(2, 2, 1)
    y1 = np.mean(reactivation_all)
    y1_err = stats.sem(reactivation_all)
    plt.errorbar(1.2, y1, yerr=y1_err, c='k', linewidth=2, marker='o', mfc='k', mec='k',
                 ms=7, mew=0, zorder=100)
    y2 = np.mean(reactivation_mean_all)
    y2_err = stats.sem(reactivation_mean_all)
    plt.errorbar(2.2, y2, yerr=y2_err, c='darkred', linewidth=2, marker='o', mfc='darkred', mec='darkred',
                 ms=7, mew=0, zorder=100)
    y2 = np.mean(reactivation_control_all)
    y2_err = stats.sem(reactivation_control_all)
    plt.errorbar(3.2, y2, yerr=y2_err, c='darkblue', linewidth=2, marker='o', mfc='darkblue', mec='darkblue',
                 ms=7, mew=0, zorder=100)
    y2 = np.mean(reactivation_opto_all)
    y2_err = stats.sem(reactivation_opto_all)
    plt.errorbar(4.2, y2, yerr=y2_err, c='r', linewidth=2, marker='o', mfc='r', mec='r',
                 ms=7, mew=0, zorder=100)
    plt.xlim(.5, 4.5)
    plt.ylim(0, .1)
    plt.xticks([1, 2, 3, 4], ['Control\nmice', 'Inhibition\nmice\nall\ntrials', 'Inhibition\nmice\ncontrol\ntrials', 'Inhibition\nmice\ninhibition\ntrials'])
    plt.ylabel('Reactivation rate (probability $\mathregular{s^{-1}}$)')

    sns.despine()
    plt.savefig(paths['base_path'] + '/NN/plots/reactivation_rate_opto_comparison.png', bbox_inches='tight', dpi=200)
    plt.savefig(paths['base_path'] + '/NN/plots/reactivation_rate_opto_comparison.pdf', bbox_inches='tight', dpi=200, Transparent=True)
    plt.close()


def activity_across_trials_across_days_R1(mice, sample_dates):
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(20, 6))
    plt.subplots_adjust(wspace=.33, hspace=.2)

    plt.subplot(2, 2, 1)
    activity_all = []
    for mouse in range(0, len(mice)):
        paths = preprocess.create_folders(mice[mouse], sample_dates[mouse])
        activity_data = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/activity_across_days.npy',
                               allow_pickle=True)
        activity = np.empty((6, 118)) * np.nan
        for i in range(0, 6):
            smoothed_activity = np.array(
                pd.DataFrame(activity_data[0][i]).rolling(3, min_periods=1, center=True).mean())
            smoothed_activity = np.concatenate(smoothed_activity, axis=0)
            activity[i, :] = smoothed_activity[0:118]
        activity_all.append(activity)

    x = np.array(range(1, 119))
    start = np.empty((len(mice), 6)) * np.nan
    end = np.empty((len(mice), 6)) * np.nan
    for i in range(0, 6):
        activity = np.empty((len(mice), 118)) * np.nan
        for j in range(0, len(mice)):
            activity[j, :] = activity_all[j][i]
            start[j, i] = np.mean(activity_all[j][i][0:3])
            end[j, i] = np.mean(activity_all[j][i][115:118])
        mean = np.nanmean(activity, axis=0)
        sem_plus = mean + stats.sem(activity, axis=0, nan_policy='omit')
        sem_minus = mean - stats.sem(activity, axis=0, nan_policy='omit')
        plt.plot(x, mean, c='k', linewidth=3)
        plt.fill_between(x, sem_plus, sem_minus, alpha=0.2, color='k', lw=0)
        x += 118

    plt.ylabel('Correlation between\nstimulus 1 and stimulus 2')
    plt.xlabel('Trial number')
    plt.ylim(0, .6)
    plt.xlim(-10, 720)
    plt.yticks([0, .2, .4, .6])
    plt.xticks([1, 100, 200, 300, 400, 500, 600, 700])
    #plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
    sns.despine()

    plt.savefig(paths['base_path'] + '/NN/plots/activity_across_trial_across_days.png', bbox_inches='tight', dpi=200)
    plt.savefig(paths['base_path'] + '/NN/plots/activity_across_trial_across_days.pdf', bbox_inches='tight', dpi=200, transparent=True)
    plt.close()

    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(5, 6))
    plt.subplots_adjust(wspace=.33, hspace=.2)

    plt.subplot(2, 2, 1)
    y_pos = np.zeros(6)
    y_neg = np.zeros(6)
    y_value = np.zeros(6)
    y_all = np.zeros((5, 6))
    for i in range(0, 6):
        y_all[:, i] = start[:, i]
        y_value[i] = np.mean(start[:, i])
        y_pos[i] = np.mean(start[:, i]) + stats.sem(start[:, i])
        y_neg[i] = np.mean(start[:, i]) - stats.sem(start[:, i])

    anova_results = []
    for i in range(1, 6):
        anova_results.append(
            stats.ttest_rel(y_all[:, i], y_all[:, 0], alternative='less')[1])
    anova_results_corrected = multipletests(anova_results, alpha=.05, method='holm')
    print([anova_results, anova_results_corrected[1]])
    print(stats.ttest_rel(y_all[:, 5], y_all[:, 0])[1])

    plt.plot([1, 2, 3, 4, 5, 6], y_value, '-', c='k', linewidth=3)
    plt.fill_between([1, 2, 3, 4, 5, 6], y_pos, y_neg, alpha=0.2, color='k', lw=0)
    plt.xlim((.5, 6.5))
    plt.ylim((.0, .6))
    plt.xticks([1, 2, 3, 4, 5, 6])
    plt.yticks([0, .2, .4, .6])
    plt.xlabel('Day')
    plt.ylabel('Starting correlation between\nstimulus 1 and stimulus 2')
    sns.despine()

    plt.subplot(2, 2, 2)
    y_pos = np.zeros(6)
    y_neg = np.zeros(6)
    y_value = np.zeros(6)
    y_all = np.zeros((5, 6))
    for i in range(0, 6):
        y_all[:, i] = end[:, i]
        y_value[i] = np.mean(end[:, i])
        y_pos[i] = np.mean(end[:, i]) + stats.sem(end[:, i])
        y_neg[i] = np.mean(end[:, i]) - stats.sem(end[:, i])

    anova_results = []
    for i in range(1, 6):
        anova_results.append(
            stats.ttest_rel(y_all[:, i], y_all[:, 0], alternative='less')[1])
    anova_results_corrected = multipletests(anova_results, alpha=.05, method='holm')
    print([anova_results, anova_results_corrected[1]])
    print(stats.ttest_rel(y_all[:, 5], y_all[:, 0])[1])

    plt.plot([1, 2, 3, 4, 5, 6], y_value, '-', c='k', linewidth=3)
    plt.fill_between([1, 2, 3, 4, 5, 6], y_pos, y_neg, alpha=0.2, color='k', lw=0)
    plt.xlim((.5, 6.5))
    plt.ylim((.0, .6))
    plt.xticks([1, 2, 3, 4, 5, 6])
    plt.yticks([0, .2, .4, .6])
    plt.xlabel('Day')
    plt.ylabel('Ending correlation between\nstimulus 1 and stimulus 2')
    sns.despine()

    plt.subplot(2, 2, 3)
    y_pos = np.zeros(5)
    y_neg = np.zeros(5)
    y_value = np.zeros(5)
    y_all = np.zeros((5, 5))
    for i in range(0, 5):
        y_all[:, i] = start[:, i+1] -end[:, i]
        y_value[i] = np.mean(start[:, i+1]) - np.mean(end[:, i])
        y_pos[i] = np.mean(start[:, i+1] - end[:, i]) + stats.sem(start[:, i+1] - end[:, i])
        y_neg[i] = np.mean(start[:, i+1] - end[:, i]) - stats.sem(start[:, i+1] - end[:, i])
    anova_results = []
    for i in range(1, 5):
        anova_results.append(
            stats.ttest_rel(y_all[:, i], y_all[:, 0], alternative='less')[1])
    anova_results_corrected = multipletests(anova_results, alpha=.05, method='holm')
    print([anova_results, anova_results_corrected[1]])

    plt.plot([2, 3, 4, 5, 6], y_value, '-', c='k', linewidth=3)
    plt.fill_between([2, 3, 4, 5, 6], y_pos, y_neg, alpha=0.2, color='k', lw=0)
    plt.xlim((1.5, 6.5))
    plt.ylim((-.5, .5))
    plt.xticks([2, 3, 4, 5, 6])
    plt.yticks([-.4, -.2, 0, .2, .4])
    plt.xlabel('Day')
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1, snap=False)
    plt.ylabel('Starting next day - ending previous day\ncorrelation between\nstimulus 1 and stimulus 2')
    sns.despine()

    plt.savefig(paths['base_path'] + '/NN/plots/activity_across_trial_across_days_quant.png', bbox_inches='tight', dpi=200)
    plt.savefig(paths['base_path'] + '/NN/plots/activity_across_trial_across_days_quant.pdf', bbox_inches='tight', dpi=200,
                transparent=True)
    plt.close()


def reactivation_spatial_drift(mice, sample_dates):
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(9, 6))
    paths = preprocess.create_folders(mice[0], sample_dates[0])
    reactivation_spatial_drift_helper(mice, sample_dates, 0, 1)
    reactivation_spatial_drift_helper(mice, sample_dates, 2, 2)
    sns.despine()
    plt.savefig(paths['base_path'] + '/NN/plots/reactivation_spatial_drift.png', bbox_inches='tight', dpi=200)
    plt.savefig(paths['base_path'] + '/NN/plots/reactivation_spatial_drift.pdf', bbox_inches='tight', dpi=200, transparent=True)
    plt.close()


def reactivation_spatial_drift_helper(mice, sample_dates, idx, plt_idx):
    paths = preprocess.create_folders(mice[0], sample_dates[0])
    session_data = preprocess.load_data(paths)
    task_runs = int(session_data['task_runs'])
    x_label = np.zeros((task_runs * 2))
    hours_per_run = 64000 / 31.25 / 60 / 60
    for i in range(0, (task_runs * 2)):
        x_label[i] = hours_per_run / 4 + (hours_per_run / 2 * i)
    plt.subplot(2, 2, plt_idx)
    paths = preprocess.create_folders(mice[0], sample_dates[0])
    across_days = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/reactivation_spatial_drift.npy',
                               allow_pickle=True)
    drift_all_cs_1 = np.zeros((len(mice), len(across_days[0][0])))
    drift_all_cs_2 = np.zeros((len(mice), len(across_days[0][0])))
    for mouse in range(0, len(mice)):
        paths = preprocess.create_folders(mice[mouse], sample_dates[mouse])
        across_days = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/reactivation_spatial_drift.npy',
                                   allow_pickle=True)
        drift_cs_1 = np.zeros((len(across_days[0]), len(across_days[0][0])))
        drift_cs_2 = np.zeros((len(across_days[0]), len(across_days[0][0])))
        for i in range(0, len(across_days[0])):
            drift_cs_1[i, :] = (across_days[idx][i] - across_days[idx][i][0]) * 1500 / 512
            drift_cs_2[i, :] = (across_days[idx+1][i] - across_days[idx+1][i][0]) * 1500 / 512
        drift_all_cs_1[mouse, :] = np.nanmean(drift_cs_1, axis=0)
        drift_all_cs_2[mouse, :] = np.nanmean(drift_cs_2, axis=0)
        plt.plot(x_label, np.nanmean(drift_cs_1, axis=0), c='mediumseagreen', alpha=.2, linewidth=2)
        plt.plot(x_label, np.nanmean(drift_cs_2, axis=0), c='salmon', alpha=.2, linewidth=2)

    [_, s_p_value] = stats.shapiro(drift_all_cs_1[:, 5])
    print(s_p_value)

    [_, p1] = stats.ttest_rel(drift_all_cs_1[:, 0], drift_all_cs_1[:, len(drift_all_cs_1)-1])
    [_, p2] = stats.ttest_rel(drift_all_cs_2[:, 0], drift_all_cs_2[:, len(drift_all_cs_2)-1])

    print(statsmodels.stats.multitest.multipletests([p1, p2], method='holm'))

    mean = drift_all_cs_1.mean(axis=0)
    sem_plus = mean + stats.sem(drift_all_cs_1, axis=0)
    sem_minus = mean - stats.sem(drift_all_cs_1, axis=0)
    plt.plot(x_label, mean[0:len(mean)], 'mediumseagreen', linewidth=3)
    plt.fill_between(x_label, sem_plus[0:len(sem_plus)], sem_minus[0:len(sem_minus)], alpha=0.2, color='mediumseagreen', lw=0)
    mean = drift_all_cs_2.mean(axis=0)
    sem_plus = mean + stats.sem(drift_all_cs_2, axis=0)
    sem_minus = mean - stats.sem(drift_all_cs_2, axis=0)
    plt.plot(x_label, mean[0:len(mean)], 'salmon', linewidth=3)
    plt.fill_between(x_label, sem_plus[0:len(sem_plus)], sem_minus[0:len(sem_minus)], alpha=0.2, color='salmon',
                     lw=0)
    if plt_idx == 1:
        plt.ylabel('ΔReactivation location (μm, x)')
    if plt_idx == 2:
        plt.ylabel('ΔReactivation local (μm, y)')
    plt.xlabel('Time relative to stimulus onset (h)')
    plt.xlim(0, x_label[len(x_label)-1] + hours_per_run / 4)
    plt.ylim(-1000, 1000)
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1, snap=False)
    label_1 = mlines.Line2D([], [], color='mediumseagreen', linestyle='-', label='Reactivation stimulus 1', alpha=.2, linewidth=2)
    label_2 = mlines.Line2D([], [], color='salmon', linestyle='-', label='Reactivation stimulus 2', alpha=.2, linewidth=2)
    plt.legend(handles=[label_1, label_2], frameon=False, prop={'size': 8}, labelspacing=.1)


def activity_across_trials_layer(mice, sample_dates):
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(9, 6))
    plt.subplots_adjust(wspace=.33)

    plt.subplot(2, 2, 1)
    activity_all_upper = np.empty((len(mice), 128)) * np.nan
    activity_all_lower = np.empty((len(mice), 128)) * np.nan
    for mouse in range(0, len(mice)):
        paths = preprocess.create_folders(mice[mouse], sample_dates[mouse])
        activity_data = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/activity_layer.npy',
                               allow_pickle=True)
        activity_upper = np.empty((len(activity_data[0]), 128)) * np.nan
        activity_lower = np.empty((len(activity_data[0]), 128)) * np.nan
        for i in range(0, len(activity_data[0])):
            smoothed_activity_upper = np.array(pd.DataFrame(activity_data[0][i]).rolling(3, min_periods=1, center=True).mean())
            smoothed_activity_lower = np.array(pd.DataFrame(activity_data[1][i]).rolling(3, min_periods=1, center=True).mean())
            if mouse < 5:
                smoothed_activity_upper = np.concatenate(smoothed_activity_upper, axis=0)
                activity_upper[i, 0:len(smoothed_activity_upper)] = smoothed_activity_upper
                smoothed_activity_lower = np.concatenate(smoothed_activity_lower, axis=0)
                activity_lower[i, 0:len(smoothed_activity_lower)] = smoothed_activity_lower
            if mouse > 4:
                smoothed_activity_upper = np.concatenate(smoothed_activity_upper, axis=0)
                smoothed_activity_upper = scipy.signal.resample_poly(smoothed_activity_upper, 2, 1, padtype='smooth')[0:(len(smoothed_activity_upper)*2)]
                activity_upper[i, 0:len(smoothed_activity_upper)] = smoothed_activity_upper
                smoothed_activity_lower = np.concatenate(smoothed_activity_lower, axis=0)
                smoothed_activity_lower = scipy.signal.resample_poly(smoothed_activity_lower, 2, 1, padtype='smooth')[0:(len(smoothed_activity_lower)*2)]
                activity_lower[i, 0:len(smoothed_activity_lower)] = smoothed_activity_lower
        activity_upper = np.nanmean(activity_upper, axis=0)
        activity_lower = np.nanmean(activity_lower, axis=0)
        activity_all_upper[mouse, :] = activity_upper
        activity_all_lower[mouse, :] = activity_lower
        x = range(1, len(activity_upper)+1)

    print(stats.ttest_ind(np.nanmean(activity_all_upper, axis=1), np.nanmean(activity_all_lower, axis=1)))

    mean = np.nanmean(activity_all_upper, axis=0)
    sem_plus = mean + stats.sem(activity_all_upper, axis=0, nan_policy='omit')
    sem_minus = mean - stats.sem(activity_all_upper, axis=0, nan_policy='omit')
    plt.plot(x[0:120], mean[0:120], c=[0, 59/255, 116/255], linewidth=3)
    plt.fill_between(x[0:120], sem_plus[0:120], sem_minus[0:120], alpha=0.2, color=[0, 59/255, 116/255], lw=0)
    mean = np.nanmean(activity_all_lower, axis=0)
    sem_plus = mean + stats.sem(activity_all_lower, axis=0, nan_policy='omit')
    sem_minus = mean - stats.sem(activity_all_lower, axis=0, nan_policy='omit')
    plt.plot(x[0:120], mean[0:120], c=[0, 74/255, 31/255], linewidth=3)
    plt.fill_between(x[0:120], sem_plus[0:120], sem_minus[0:120], alpha=0.2, color=[0, 74/255, 31/255], lw=0)
    plt.ylabel('Correlation between\nstimulus 1 and stimulus 2')
    plt.ylim(-.05, .6)
    plt.xlim(1, 121)
    plt.xticks([1, 120], ['First trial', 'Last trial'])
    label_1 = mlines.Line2D([], [], color=[0, 59/255, 116/255], linestyle='-', label='Upper layer neurons', linewidth=3)
    label_2 = mlines.Line2D([], [], color=[0, 74/255, 31/255], linestyle='-', label='Lower layer neurons', linewidth=3)
    plt.legend(handles=[label_1, label_2], frameon=False, labelspacing=.1)
    sns.despine()

    plt.savefig(paths['base_path'] + '/NN/plots/correlation_across_trials_layer.png', bbox_inches='tight', dpi=200)
    plt.savefig(paths['base_path'] + '/NN/plots/correlation_across_trials_layer.pdf', bbox_inches='tight', dpi=200,
                transparent=True)
    plt.close()


def neuron_count_grouped_R2(mice, sample_dates):
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(9, 6))
    plt.subplots_adjust(wspace=.33)
    plt.subplot(2,2,1)
    m_colors = ['b', 'teal', 'slategray', 'green', 'darkolivegreen', 'darkorange', 'purple', 'darkred']
    x = [1, 2, 3]
    all_vec = np.empty((len(mice), len(x))) * np.nan
    for mouse in range(0, len(mice)):
        paths = preprocess.create_folders(mice[mouse], sample_dates[mouse])
        grouped_count = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/grouped_count.npy', allow_pickle=True)
        vec = np.empty((100, len(x))) * np.nan
        for i in range(0, len(grouped_count[0])):
            for j in range(10, 13):
                vec[i, j-10] = grouped_count[j][i]
        vec = np.nanmean(vec, axis=0)

        for i in range(0, len(vec)):
            plt.errorbar(x[i], vec[i], yerr=0, c=m_colors[mouse], marker='o', mfc='none', mec=m_colors[mouse],
                         ms=5, alpha=.3)
            all_vec[mouse, i] = vec[i]

    for i in range(0, len(all_vec[0])):
        plt.errorbar(x[i]+.2, np.mean(all_vec[:,i]), yerr=stats.sem(all_vec[:,i]), c='k', linewidth=2, marker='o', mfc='k', mec='k',
                     ms=7, mew=0, zorder=100)
    plt.ylim([0, 400])
    plt.xlim(.5, 3.5)
    plt.xticks([1, 2, 3], ['No change\nneurons', 'Increase\nneurons', 'Decrease\nneurons'])
    plt.ylabel('Number of neurons')
    sns.despine()
    plt.savefig(paths['base_path'] + '/NN/plots/grouped_count_incdec.png', bbox_inches='tight', dpi=200)
    plt.savefig(paths['base_path'] + '/NN/plots/grouped_count_incdec.pdf', bbox_inches='tight', dpi=200,
                transparent=True)
    #plt.close()


def activity_across_trials_grouped_baseline_R3(mice, sample_dates):
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(9, 6))
    plt.subplots_adjust(wspace=.3)
    m_colors = ['b', 'teal', 'slategray', 'green', 'darkolivegreen', 'darkorange', 'purple', 'darkred']
    dec_b_all = []
    inc_b_all = []
    nc_b_all = []
    x = [1, 2, 3]
    for mouse in range(0, len(mice)):
        paths = preprocess.create_folders(mice[mouse], sample_dates[mouse])
        activity_all = np.load(paths['base_path'] + paths['mouse'] +
                             '/data_across_days/baseline_activity_grouped.npy', allow_pickle=True)
        session_data = preprocess.load_data(paths)
        fr = session_data['framerate']
        dec_b = []
        inc_b = []
        nc_b = []
        for i in range(0, len(activity_all[0])):
            nc_b.append(activity_all[0][i][0]*fr)
            inc_b.append(activity_all[0][i][1]*fr)
            dec_b.append(activity_all[0][i][2]*fr)
        nc_b = np.mean(nc_b)
        inc_b = np.mean(inc_b)
        dec_b = np.nanmean(dec_b)
        plt.subplot(2, 2, 1)
        plt.errorbar(x[0], nc_b, yerr=0, c=m_colors[mouse], marker='o', mfc='none',
                     mec=m_colors[mouse],
                     ms=5, alpha=.3)
        plt.errorbar(x[1], inc_b, yerr=0, c=m_colors[mouse], marker='o', mfc='none',
                     mec=m_colors[mouse],
                     ms=5, alpha=.3)
        plt.errorbar(x[2], dec_b, yerr=0, c=m_colors[mouse], marker='o', mfc='none',
                     mec=m_colors[mouse],
                     ms=5, alpha=.3)
        nc_b_all.append(nc_b)
        inc_b_all.append(inc_b)
        dec_b_all.append(dec_b)
    y0 = np.mean(nc_b_all)
    y1 = np.mean(inc_b_all)
    y2 = np.mean(dec_b_all)
    y0_err = stats.sem(nc_b_all)
    y1_err = stats.sem(inc_b_all)
    y2_err = stats.sem(dec_b_all)

    print(stats.f_oneway(nc_b_all, inc_b_all, dec_b_all))
    print(
        pairwise_tukeyhsd(np.concatenate([nc_b_all, inc_b_all, dec_b_all], axis=0),
                          ['nc', 'nc', 'nc', 'nc', 'nc', 'nc', 'nc', 'nc', 'i', 'i', 'i', 'i', 'i', 'i',
                           'i', 'i', 'd', 'd', 'd', 'd', 'd', 'd', 'd', 'd'], alpha=0.05))

    plt.subplot(2, 2, 1)
    plt.errorbar(1.2, y0, yerr=y0_err, c='k', linewidth=2, marker='o', mfc='k', mec='k',
                 ms=7, mew=0, zorder=100)
    plt.errorbar(2.2, y1, yerr=y1_err, c='k', linewidth=2, marker='o', mfc='k', mec='k',
                 ms=7, mew=0, zorder=100)
    plt.errorbar(3.2, y2, yerr=y2_err, c='k', linewidth=2, marker='o', mfc='k', mec='k',
                 ms=7, mew=0, zorder=100)
    plt.xlim((.5, 3.5))
    plt.ylabel('Baseline activity\n(normalized\ndeconvolved$\mathregular{Ca^{2+}}$$\mathregular{s^{-1}}$)')
    plt.xticks([1, 2, 3], ['No change\nneurons', 'Increase\nneurons', 'Decrease\nneurons'])
    #plt.yticks([0, .05, .1, .15])
    plt.ylim((0, 1))

    sns.despine()
    plt.savefig(paths['base_path'] + '/NN/plots/baseline_activity_grouped.png', bbox_inches='tight', dpi=200)
    plt.savefig(paths['base_path'] + '/NN/plots/baseline_activity_grouped.pdf', bbox_inches='tight', dpi=200, transparent=True)
    plt.close()


def neuron_count_grouped_layer_R3(mice, sample_dates):
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(13.5, 6))
    m_colors = ['b', 'teal', 'slategray', 'green', 'darkolivegreen', 'darkorange', 'purple', 'darkred']

    plt.subplot(2, 2, 1)
    nc_upper_all = []
    nc_lower_all = []
    inc_upper_all = []
    inc_lower_all = []
    dec_upper_all = []
    dec_lower_all = []
    x = [1, 2, 3, 4, 5, 6]
    for mouse in range(0, len(mice)):
        paths = preprocess.create_folders(mice[mouse], sample_dates[mouse])
        vec_all = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/activity_grouped_count_layer.npy',
                             allow_pickle=True)
        nc_upper_vec = []
        nc_lower_vec = []
        inc_upper_vec = []
        inc_lower_vec = []
        dec_upper_vec = []
        dec_lower_vec = []
        for i in range(0, len(vec_all[0])):
            nc_upper_vec.append(vec_all[0][i] * 100)
            nc_lower_vec.append(vec_all[1][i] * 100)
            inc_upper_vec.append(vec_all[2][i] * 100)
            inc_lower_vec.append(vec_all[3][i] * 100)
            dec_upper_vec.append(vec_all[4][i] * 100)
            dec_lower_vec.append(vec_all[5][i] * 100)

        nc_upper_vec = np.mean(nc_upper_vec)
        nc_lower_vec = np.mean(nc_lower_vec)
        inc_upper_vec = np.mean(inc_upper_vec)
        inc_lower_vec = np.mean(inc_lower_vec)
        dec_upper_vec = np.mean(dec_upper_vec)
        dec_lower_vec = np.mean(dec_lower_vec)
        plt.plot(x[0:2], [nc_upper_vec, nc_lower_vec], c=m_colors[mouse], linewidth=2, linestyle='-', alpha=.2)
        plt.plot(x[2:4], [inc_upper_vec, inc_lower_vec], c=m_colors[mouse], linewidth=2, linestyle='-', alpha=.2)
        plt.plot(x[4:6], [dec_upper_vec, dec_lower_vec], c=m_colors[mouse], linewidth=2, linestyle='-', alpha=.2)

        nc_upper_all.append(nc_upper_vec)
        nc_lower_all.append(nc_lower_vec)
        inc_upper_all.append(inc_upper_vec)
        inc_lower_all.append(inc_lower_vec)
        dec_upper_all.append(dec_upper_vec)
        dec_lower_all.append(dec_lower_vec)

    anova_results = []
    anova_results.append(stats.ttest_rel(nc_upper_all, nc_lower_all)[1])
    anova_results.append(stats.ttest_rel(inc_upper_all, inc_lower_all)[1])
    anova_results.append(stats.ttest_rel(dec_upper_all, dec_lower_all)[1])
    anova_results_corrected = multipletests(anova_results, alpha=.05, method='holm')
    print(anova_results_corrected[1])

    y0 = np.mean(nc_upper_all)
    y1 = np.mean(nc_lower_all)
    y0_err = stats.sem(nc_upper_all)
    y1_err = stats.sem(nc_lower_all)
    plt.plot(x[0:2], [y0, y1], '-', c='k', linewidth=3)
    plt.fill_between(x[0:2], [y0 + y0_err, y1 + y1_err], [y0 - y0_err, y1 - y1_err], alpha=0.2, color='k', lw=0)

    y0 = np.mean(inc_upper_all)
    y1 = np.mean(inc_lower_all)
    y0_err = stats.sem(inc_upper_all)
    y1_err = stats.sem(inc_lower_all)
    plt.plot(x[2:4], [y0, y1], '-', c='k', linewidth=3)
    plt.fill_between(x[2:4], [y0 + y0_err, y1 + y1_err], [y0 - y0_err, y1 - y1_err], alpha=0.2, color='k', lw=0)

    y0 = np.mean(dec_upper_all)
    y1 = np.mean(dec_lower_all)
    y0_err = stats.sem(dec_upper_all)
    y1_err = stats.sem(dec_lower_all)
    plt.plot(x[4:6], [y0, y1], '-', c='k', linewidth=3)
    plt.fill_between(x[4:6], [y0 + y0_err, y1 + y1_err], [y0 - y0_err, y1 - y1_err], alpha=0.2, color='k', lw=0)

    plt.xlim((.5, 6.5))
    plt.ylim((0, 10))
    plt.xticks([1, 2, 3, 4, 5, 6], ['Upper\nlayer', 'Lower\nlayer', 'Upper\nlayer', 'Lower\nlayer', 'Upper\nlayer', 'Lower\nlayer'])
    # plt.yticks([.2, .25, .3, .35, .4])

    plt.ylabel('Percent of neurons in layer')
    sns.despine()
    plt.savefig(paths['base_path'] + '/NN/plots/grouped_count_layer.png', bbox_inches='tight', dpi=200)
    plt.savefig(paths['base_path'] + '/NN/plots/grouped_count_layer.pdf', bbox_inches='tight', dpi=200, transparent=True)
    plt.close()


def reactivation_spatial_grouped(mice, sample_dates):
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(12, 6))
    plt.subplots_adjust(wspace=.45)
    reactivation_spatial_grouped_helper(mice, sample_dates, 0, [1, 2, 3, 4])
    reactivation_spatial_grouped_helper(mice, sample_dates, 4, [5, 6, 7, 8])
    reactivation_spatial_grouped_helper(mice, sample_dates, 8, [9, 10, 11, 12])
    sns.despine()
    paths = preprocess.create_folders(mice[0], sample_dates[0])
    plt.savefig(paths['base_path'] + '/NN/plots/reactivation_spatial_grouped.png', bbox_inches='tight', dpi=200)
    plt.savefig(paths['base_path'] + '/NN/plots/reactivation_spatial_grouped.pdf', bbox_inches='tight', dpi=200,
                transparent=True)
    plt.close()


def reactivation_spatial_grouped_helper(mice, sample_dates, idx, x):
    m_colors = ['b', 'teal', 'slategray', 'green', 'darkolivegreen', 'darkorange', 'purple', 'darkred']
    area_li_all = []
    area_por_all = []
    area_p_all = []
    area_lm_all = []
    for mouse in range(0, len(mice)):
        paths = preprocess.create_folders(mice[mouse], sample_dates[mouse])
        reactivation_spatial = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/reactivation_spatial_grouped.npy', allow_pickle=True)
        area_li = []
        area_por = []
        area_p = []
        area_lm = []
        for i in range(0, len(reactivation_spatial[0])):
            area_li.append(100*reactivation_spatial[idx][i]/reactivation_spatial[12][i])
            area_por.append(100*reactivation_spatial[idx+1][i]/reactivation_spatial[13][i])
            area_p.append(100*reactivation_spatial[idx+2][i]/reactivation_spatial[14][i])
            area_lm.append(100*reactivation_spatial[idx+3][i]/reactivation_spatial[15][i])
        area_li_all.append(np.nanmean(area_li))
        area_por_all.append(np.nanmean(area_por))
        area_p_all.append(np.nanmean(area_p))
        area_lm_all.append(np.nanmean(area_lm))

        plt.subplot(2, 2, 1)
        plt.plot(x, [np.nanmean(area_li), np.nanmean(area_por), np.nanmean(area_p), np.nanmean(area_lm)],
                 c=m_colors[mouse], linewidth=2, linestyle='-', alpha=.2)

    print(stats.f_oneway(area_li_all, area_por_all, area_p_all, area_lm_all))
    print(
        pairwise_tukeyhsd(np.concatenate([area_li_all, area_por_all, area_p_all, area_lm_all], axis=0),
                          ['li', 'li', 'li', 'li', 'li', 'li', 'li', 'li', 'por', 'por', 'por', 'por', 'por', 'por',
                           'por', 'por', 'p', 'p', 'p', 'p', 'p', 'p', 'p', 'p', 'lm', 'lm', 'lm', 'lm', 'lm', 'lm',
                           'lm', 'lm'], alpha=0.05))

    plt.subplot(2, 2, 1)
    y0 = np.mean(area_li_all)
    y1 = np.mean(area_por_all)
    y2 = np.mean(area_p_all)
    y3 = np.mean(area_lm_all)
    y0_err = stats.sem(area_li_all)
    y1_err = stats.sem(area_por_all)
    y2_err = stats.sem(area_p_all)
    y3_err = stats.sem(area_lm_all)
    plt.plot(x, [y0, y1, y2, y3], '-', c='k', linewidth=3)
    plt.fill_between(x, [y0 + y0_err, y1 + y1_err, y2 + y2_err, y3 + y3_err],
                     [y0 - y0_err, y1 - y1_err, y2 - y2_err, y3 - y3_err], alpha=0.2, color='k', lw=0)

    plt.xlim(.5, 12.5)
    plt.ylim(0, 15)
    plt.yticks([0, 5, 10, 15])
    plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], ['LI', 'POR', 'P', 'LM', 'LI', 'POR', 'P', 'LM', 'LI', 'POR', 'P', 'LM'])
    plt.ylabel('Percent of all neurons in region')


def noise_grouped_R3(mice, sample_dates):
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(9, 6))
    plt.subplots_adjust(wspace=.45)
    m_colors = ['b', 'teal', 'slategray', 'green', 'darkolivegreen', 'darkorange', 'purple', 'darkred']
    x = [1, 2, 3]
    nc_all = []
    inc_all = []
    dec_all = []
    for mouse in range(0, len(mice)):
        paths = preprocess.create_folders(mice[mouse], sample_dates[mouse])
        noise_vec = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/noise_grouped.npy', allow_pickle=True)
        nc = []
        inc = []
        dec = []
        for i in range(0, len(noise_vec[0])):
            nc.append(noise_vec[0][i][0])
            inc.append(noise_vec[0][i][1])
            dec.append(noise_vec[0][i][2])
        nc_all.append(np.nanmean(nc))
        inc_all.append(np.nanmean(inc))
        dec_all.append(np.nanmean(dec))
        plt.subplot(2, 2, 1)
        plt.errorbar(x[0], np.nanmean(nc), yerr=0, c=m_colors[mouse], marker='o', mfc='none', mec=m_colors[mouse],
                     ms=5, alpha=.3)
        plt.errorbar(x[1], np.nanmean(inc), yerr=0, c=m_colors[mouse], marker='o', mfc='none', mec=m_colors[mouse],
                     ms=5, alpha=.3)
        plt.errorbar(x[2], np.nanmean(dec), yerr=0, c=m_colors[mouse], marker='o', mfc='none', mec=m_colors[mouse],
                     ms=5, alpha=.3)

    print(stats.f_oneway(nc_all, inc_all, dec_all))
    print(
        pairwise_tukeyhsd(np.concatenate([nc_all, inc_all, dec_all], axis=0),
                          ['nc', 'nc', 'nc', 'nc', 'nc', 'nc', 'nc', 'nc', 'i', 'i', 'i', 'i', 'i', 'i',
                           'i', 'i', 'd', 'd', 'd', 'd', 'd', 'd', 'd', 'd'], alpha=0.05))

    plt.subplot(2, 2, 1)
    y1 = np.mean(nc_all)
    y1_err = stats.sem(nc_all)
    plt.errorbar(1.2, y1, yerr=y1_err, c='k', linewidth=2, marker='o', mfc='k', mec='k',
                 ms=7, mew=0, zorder=100)
    y2 = np.mean(inc_all)
    y2_err = stats.sem(inc_all)
    plt.errorbar(2.2, y2, yerr=y2_err, c='k', linewidth=2, marker='o', mfc='k', mec='k',
                 ms=7, mew=0, zorder=100)
    y3 = np.mean(dec_all)
    y3_err = stats.sem(dec_all)
    plt.errorbar(3.2, y3, yerr=y3_err, c='k', linewidth=2, marker='o', mfc='k', mec='k',
                 ms=7, mew=0, zorder=100)
    plt.xlim(.5, 3.5)
    plt.ylim(0, .02)
    plt.xticks([1, 2, 3], ['No change\nneurons', 'Increase\nneurons', 'Decrease\nneurons'])
    plt.ylabel('Noise correlation (r)')

    sns.despine()
    plt.savefig(paths['base_path'] + '/NN/plots/noise_grouped.png', bbox_inches='tight', dpi=200)
    plt.savefig(paths['base_path'] + '/NN/plots/noise_grouped.pdf', bbox_inches='tight', dpi=200,
                transparent=True)
    plt.close()


def activity_across_trials_grouped_decrease_novelty_R1(mice, sample_dates):
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
        session_data = preprocess.load_data(paths)
        fr = session_data['framerate']
        activity_data = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/activity_grouped_no_change_decrease.npy',
                               allow_pickle=True)
        cs1d_cs1 = np.empty((len(activity_data[0]), 128)) * np.nan
        cs1d_cs2 = np.empty((len(activity_data[0]), 128)) * np.nan
        cs2d_cs2 = np.empty((len(activity_data[0]), 128)) * np.nan
        cs2d_cs1 = np.empty((len(activity_data[0]), 128)) * np.nan
        for i in range(0, len(activity_data[0])):
            if mouse < 5:
                cs1d_cs1[i, 0:len(activity_data[0][i])] = activity_data[0][i] * fr
                cs1d_cs2[i, 0:len(activity_data[1][i])] = activity_data[1][i] * fr
                cs2d_cs2[i, 0:len(activity_data[2][i])] = activity_data[2][i] * fr
                cs2d_cs1[i, 0:len(activity_data[3][i])] = activity_data[3][i] * fr
            if mouse > 4:
                vec_1 = scipy.signal.resample_poly(activity_data[0][i], 2, 1, padtype='smooth')[
                        0:(len(activity_data[0][i]) * 2)]
                cs1d_cs1[i, 0:len(vec_1)] = vec_1 * fr
                vec_2 = scipy.signal.resample_poly(activity_data[1][i], 2, 1, padtype='smooth')[
                        0:(len(activity_data[1][i]) * 2)]
                cs1d_cs2[i, 0:len(vec_2)] = vec_2 * fr
                vec_3 = scipy.signal.resample_poly(activity_data[2][i], 2, 1, padtype='smooth')[
                        0:(len(activity_data[2][i]) * 2)]
                cs2d_cs2[i, 0:len(vec_3)] = vec_3 * fr
                vec_4 = scipy.signal.resample_poly(activity_data[3][i], 2, 1, padtype='smooth')[
                        0:(len(activity_data[3][i]) * 2)]
                cs2d_cs1[i, 0:len(vec_4)] = vec_4 * fr
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
    plt.plot(x[0:60], mean[0:60], c='mediumseagreen', linewidth=3)
    plt.fill_between(x[0:60], sem_plus[0:60], sem_minus[0:60], alpha=0.2, color='mediumseagreen', lw=0)
    mean = np.nanmean(cs1d_cs2_all, axis=0)
    sem_plus = mean + stats.sem(cs1d_cs2_all, axis=0, nan_policy='omit')
    sem_minus = mean - stats.sem(cs1d_cs2_all, axis=0, nan_policy='omit')
    plt.plot(x[0:60], mean[0:60], c='salmon', linewidth=3)
    plt.fill_between(x[0:60], sem_plus[0:60], sem_minus[0:60], alpha=0.2, color='salmon', lw=0)
    plt.ylabel('Stimulus activity\n(Normalized $\mathregular{Ca^{2+}}$transient $\mathregular{s^{-1}}$)')
    plt.ylim(0, 1.4)
    plt.xlim(1, 61)
    plt.xticks([1, 60], ['First trial', 'Last trial'])
    plt.yticks([0, .2, .4, .6, .8, 1, 1.2, 1.4])
    label_1 = mlines.Line2D([], [], color='g', linewidth=2, label='Non-sel. Decrease Neurons, S1 trials')
    label_2 = mlines.Line2D([], [], color='r', linewidth=2, label='Non-sel. Decrease Neurons, S2 trials')
    plt.legend(handles=[label_1, label_2], frameon=False)
    sns.despine()
    plt.savefig(paths['base_path'] + '/NN/plots/activity_across_trials_grouped_no_change_decrease.png',
                bbox_inches='tight', dpi=200)
    plt.savefig(paths['base_path'] + '/NN/plots/activity_across_trials_grouped_no_change_decrease.pdf',
                bbox_inches='tight', dpi=200, transparent=True)
    plt.close()

    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(2.5, 6))
    plt.subplot(2, 2, 1)

    m_colors = ['b', 'teal', 'slategray', 'green', 'darkolivegreen', 'darkorange', 'purple', 'darkred']
    x = [1]
    count_all = []
    for mouse in range(0, len(mice)):
        paths = preprocess.create_folders(mice[mouse], sample_dates[mouse])
        grouped_count = np.load(
            paths['base_path'] + paths['mouse'] + '/data_across_days/activity_grouped_no_change_decrease.npy',
            allow_pickle=True)
        count = []
        for i in range(0, len(grouped_count[0])):
            count.append(grouped_count[4][i])

        count_all.append(np.nanmean(count))

        plt.errorbar(x[0], np.nanmean(count), yerr=0, c=m_colors[mouse], marker='o', mfc='none',
                     mec=m_colors[mouse],
                     ms=5, alpha=.3)

    y1 = np.mean(count_all)
    y1_err = stats.sem(count_all)
    plt.errorbar(1.2, y1, yerr=y1_err, c='k', linewidth=2, marker='o', mfc='k', mec='k',
                 ms=7, mew=0, zorder=100)
    plt.xlim(.5, 1.5)
    plt.ylim(0, 1)
    plt.xticks([1], ['Non-selective\nDecrease Neurons'])
    plt.ylabel('Percent of Decrease Neurons')
    sns.despine()

    plt.savefig(paths['base_path'] + '/NN/plots/activity_across_trials_grouped_no_change_decrease_count.png', bbox_inches='tight', dpi=200)
    plt.savefig(paths['base_path'] + '/NN/plots/activity_across_trials_grouped_no_change_decrease_count.pdf', bbox_inches='tight', dpi=200, transparent=True)
    plt.close()


def activity_across_trials_novelty_R1(mice, sample_dates):
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(9, 6))
    plt.subplots_adjust(wspace=.33)
    m_colors = ['b', 'teal', 'slategray', 'green', 'darkolivegreen', 'darkorange', 'purple', 'darkred']

    plt.subplot(2, 2, 1)
    activity_all = np.empty((len(mice), 128)) * np.nan
    activity_all_no_novelty = np.empty((len(mice), 128)) * np.nan
    for mouse in range(0, len(mice)):
        paths = preprocess.create_folders(mice[mouse], sample_dates[mouse])
        activity_data = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/activity.npy',
                               allow_pickle=True)
        activity_data_no_novelty = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/activity_no_novelty.npy',
                                allow_pickle=True)
        activity = np.empty((len(activity_data[0]), 128)) * np.nan
        activity_no_novelty = np.empty((len(activity_data[0]), 128)) * np.nan
        for i in range(0, len(activity_data[0])):
            smoothed_activity = np.array(
                pd.DataFrame(activity_data[0][i]).rolling(3, min_periods=1, center=True).mean())
            smoothed_activity = np.concatenate(smoothed_activity, axis=0)
            smoothed_activity_no_novelty = np.array(
                pd.DataFrame(activity_data_no_novelty[0][i]).rolling(3, min_periods=1, center=True).mean())
            smoothed_activity_no_novelty = np.concatenate(smoothed_activity_no_novelty, axis=0)
            if mouse < 5:
                activity[i, 0:len(smoothed_activity)] = smoothed_activity
                activity_no_novelty[i, 0:len(smoothed_activity)] = smoothed_activity_no_novelty
            if mouse > 4:
                smoothed_activity = scipy.signal.resample_poly(smoothed_activity, 2, 1, padtype='smooth')[0:(len(smoothed_activity) * 2)]
                activity[i, 0:len(smoothed_activity)] = smoothed_activity
                smoothed_activity_no_novelty = scipy.signal.resample_poly(smoothed_activity_no_novelty, 2, 1, padtype='smooth')[0:(len(smoothed_activity_no_novelty) * 2)]
                activity_no_novelty[i, 0:len(smoothed_activity_no_novelty)] = smoothed_activity_no_novelty

        activity = np.nanmean(activity, axis=0)
        activity_all[mouse, :] = activity
        activity_no_novelty = np.nanmean(activity_no_novelty, axis=0)
        activity_all_no_novelty[mouse, :] = activity_no_novelty
        x = range(1, len(activity)+1)

    print(stats.ttest_ind(np.nanmean(activity_all[0:120], axis=1), np.nanmean(activity_all_no_novelty[0:120], axis=1)))

    mean = np.nanmean(activity_all, axis=0)
    sem_plus = mean + stats.sem(activity_all, axis=0, nan_policy='omit')
    sem_minus = mean - stats.sem(activity_all, axis=0, nan_policy='omit')
    plt.plot(x[0:120], mean[0:120], c='k', linewidth=3)
    plt.fill_between(x[0:120], sem_plus[0:120], sem_minus[0:120], alpha=0.2, color='k', lw=0)
    mean_no_novelty = np.nanmean(activity_all_no_novelty, axis=0)
    sem_plus_no_novelty = mean + stats.sem(activity_all_no_novelty, axis=0, nan_policy='omit')
    sem_minus_no_novelty = mean - stats.sem(activity_all_no_novelty, axis=0, nan_policy='omit')
    plt.plot(x[0:120], mean_no_novelty[0:120], c='gray', linewidth=3)
    plt.fill_between(x[0:120], sem_plus_no_novelty[0:120], sem_minus_no_novelty[0:120], alpha=0.2, color='gray', lw=0)
    plt.ylabel('Correlation between\nstimulus 1 and stimulus 2')
    plt.ylim(-.05, .6)
    plt.xlim(1, 121)
    plt.xticks([1, 120], ['First trial', 'Last trial'])
    sns.despine()
    label_1 = mlines.Line2D([], [], color='k', linewidth=2,
                            label='All neurons')
    label_2 = mlines.Line2D([], [], color='gray', linewidth=2, label='Remove Non-sel. Decrease Neurons')
    plt.legend(handles=[label_1, label_2], frameon=False)

    plt.savefig(paths['base_path'] + '/NN/plots/activity_across_trials_no_novelty.png', bbox_inches='tight', dpi=200)
    plt.savefig(paths['base_path'] + '/NN/plots/activity_across_trials_no_novelty.pdf', bbox_inches='tight', dpi=200, transparent=True)
    plt.close()


def reactivation_cue_vector_novelty_R1(mice, sample_dates):
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(9, 6))
    plt.subplots_adjust(wspace=.3)
    reactivation_cue_vector_novelty_R1_helper(mice, sample_dates, '/data_across_days/reactivation_cue_vector.npy', 'darkgreen', 'lime', 'darkred', 'hotpink')
    reactivation_cue_vector_novelty_R1_helper(mice, sample_dates, '/data_across_days/reactivation_cue_vector_novelty.npy', 'k', 'gray', 'k', 'gray')
    paths = preprocess.create_folders(mice[0], sample_dates[0])
    plt.savefig(paths['base_path'] + '/NN/plots/reactivation_cue_vector_novelty.png', bbox_inches='tight', dpi=200)
    plt.savefig(paths['base_path'] + '/NN/plots/reactivation_cue_vector_novelty.pdf', bbox_inches='tight', dpi=200,
                transparent=True)
    plt.close()


def reactivation_cue_vector_novelty_R1_helper(mice, sample_dates, data_path, c1, c2, c3, c4):
    x_label = list(range(0, 60))

    paths = preprocess.create_folders(mice[0], sample_dates[0])
    reactivation_cue_pca_vec = np.load(paths['base_path'] + paths['mouse'] +
                                       '/data_across_days/reactivation_cue_vector.npy', allow_pickle=True)
    all_s1 = np.zeros((len(mice), len(reactivation_cue_pca_vec[0][0])))
    all_s1r = np.zeros((len(mice), len(reactivation_cue_pca_vec[0][0])))
    all_s2 = np.zeros((len(mice), len(reactivation_cue_pca_vec[0][0])))
    all_s2r = np.zeros((len(mice), len(reactivation_cue_pca_vec[0][0])))
    for mouse in range(0, len(mice)):
        paths = preprocess.create_folders(mice[mouse], sample_dates[mouse])
        reactivation_cue_pca_vec = np.load(paths['base_path'] + paths['mouse'] + data_path, allow_pickle=True)
        temp_s1 = np.zeros((len(reactivation_cue_pca_vec[0]), 60))
        temp_s1r = np.zeros((len(reactivation_cue_pca_vec[0]), 60))
        temp_s2 = np.zeros((len(reactivation_cue_pca_vec[0]), 60))
        temp_s2r = np.zeros((len(reactivation_cue_pca_vec[0]), 60))
        for i in range(0, len(reactivation_cue_pca_vec[0])):
            if mouse < 5:
                temp_s1[i, :] = reactivation_cue_pca_vec[0][i]
                temp_s1r[i, :] = reactivation_cue_pca_vec[1][i]
                temp_s2[i, :] = reactivation_cue_pca_vec[2][i]
                temp_s2r[i, :] = reactivation_cue_pca_vec[3][i]
            if mouse > 4:
                temp_s1[i, :] = scipy.signal.resample_poly(reactivation_cue_pca_vec[0][i], 2, 1, padtype='smooth')[
                                0:(len(reactivation_cue_pca_vec[0][i]) * 2)]
                temp_s1r[i, :] = scipy.signal.resample_poly(reactivation_cue_pca_vec[1][i], 2, 1, padtype='smooth')[
                                 0:(len(reactivation_cue_pca_vec[1][i]) * 2)]
                temp_s2[i, :] = scipy.signal.resample_poly(reactivation_cue_pca_vec[2][i], 2, 1, padtype='smooth')[
                                0:(len(reactivation_cue_pca_vec[2][i]) * 2)]
                temp_s2r[i, :] = scipy.signal.resample_poly(reactivation_cue_pca_vec[3][i], 2, 1, padtype='smooth')[
                                 0:(len(reactivation_cue_pca_vec[3][i]) * 2)]
        y_s1 = np.nanmean(temp_s1, axis=0)
        y_s1r = np.nanmean(temp_s1r, axis=0)
        y_s2 = np.nanmean(temp_s2, axis=0)
        y_s2r = np.nanmean(temp_s2r, axis=0)
        all_s1[mouse, :] = y_s1
        all_s1r[mouse, :] = y_s1r
        all_s2[mouse, :] = y_s2
        all_s2r[mouse, :] = y_s2r

    # print(stats.ttest_rel(all_s2[:, 0:1], all_s2r[:, 59:60])[1])

    plt.subplot(2, 2, 1)
    plt.ylim(-1.5, .2)
    plt.yticks([0, -.5, -1, -1.5])
    plt.gca().invert_yaxis()
    mean = all_s1.mean(axis=0)
    sem_plus = mean + stats.sem(all_s1, axis=0)
    sem_minus = mean - stats.sem(all_s1, axis=0)
    plt.plot(x_label, mean, '-', c=c1, linewidth=3)
    plt.fill_between(x_label, sem_plus, sem_minus, alpha=0.2, color=c1, lw=0)
    mean = all_s1r.mean(axis=0)
    sem_plus = mean + stats.sem(all_s1r, axis=0)
    sem_minus = mean - stats.sem(all_s1r, axis=0)
    plt.plot(x_label, mean, '-', c=c2, linewidth=3)
    plt.fill_between(x_label, sem_plus, sem_minus, alpha=0.2, color=c2, lw=0)
    plt.ylabel('Similarity to early vs. late\n S1 response pattern')
    plt.xlim(0, 60)
    plt.xticks([0, 59], ['First trial', 'Last trial'])
    plt.subplot(2, 2, 2)
    plt.ylim(-1.5, .2)
    plt.yticks([0, -.5, -1, -1.5])
    plt.gca().invert_yaxis()
    mean = all_s2.mean(axis=0)
    sem_plus = mean + stats.sem(all_s2, axis=0)
    sem_minus = mean - stats.sem(all_s2, axis=0)
    plt.plot(x_label, mean, '-', c=c3, linewidth=3)
    plt.fill_between(x_label, sem_plus, sem_minus, alpha=0.2, color=c3, lw=0)
    mean = all_s2r.mean(axis=0)
    sem_plus = mean + stats.sem(all_s2r, axis=0)
    sem_minus = mean - stats.sem(all_s2r, axis=0)
    plt.plot(x_label, mean, '-', c=c4, linewidth=3)
    plt.fill_between(x_label, sem_plus, sem_minus, alpha=0.2, color=c4, lw=0)
    plt.ylabel('Similarity to early vs. late\n S2 response pattern)')
    plt.xlim(0, 60)
    plt.xticks([0, 59], ['First trial', 'Last trial'])
    sns.despine()


def reactivation_cue_vector_layer_R1(mice, sample_dates):
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(9, 6))
    plt.subplots_adjust(wspace=.3)
    reactivation_cue_vector_layer_R1_helper(mice, sample_dates, '/data_across_days/reactivation_cue_vector_layer.npy',  [0, 59/255, 116/255], [0, 59/255, 216/255], [0, 59/255, 116/255], [0, 59/255, 216/255])
    reactivation_cue_vector_layer_R1_helper(mice, sample_dates, '/data_across_days/reactivation_cue_vector_layer.npy', [0, 74/255, 31/255], [0, 174/255, 31/255], [0, 74/255, 31/255], [0, 174/255, 31/255])
    paths = preprocess.create_folders(mice[0], sample_dates[0])
    plt.savefig(paths['base_path'] + '/NN/plots/reactivation_cue_vector_layer.png', bbox_inches='tight', dpi=200)
    plt.savefig(paths['base_path'] + '/NN/plots/reactivation_cue_vector_layer.pdf', bbox_inches='tight', dpi=200,
                transparent=True)
    plt.close()


def reactivation_cue_vector_layer_R1_helper(mice, sample_dates, data_path, c1, c2, c3, c4):
    x_label = list(range(0, 60))

    paths = preprocess.create_folders(mice[0], sample_dates[0])
    reactivation_cue_pca_vec = np.load(paths['base_path'] + paths['mouse'] +
                                       '/data_across_days/reactivation_cue_vector.npy', allow_pickle=True)
    all_s1 = np.zeros((len(mice), len(reactivation_cue_pca_vec[0][0])))
    all_s1r = np.zeros((len(mice), len(reactivation_cue_pca_vec[0][0])))
    all_s2 = np.zeros((len(mice), len(reactivation_cue_pca_vec[0][0])))
    all_s2r = np.zeros((len(mice), len(reactivation_cue_pca_vec[0][0])))
    for mouse in range(0, len(mice)):
        paths = preprocess.create_folders(mice[mouse], sample_dates[mouse])
        reactivation_cue_pca_vec = np.load(paths['base_path'] + paths['mouse'] + data_path, allow_pickle=True)
        temp_s1 = np.zeros((len(reactivation_cue_pca_vec[0]), 60))
        temp_s1r = np.zeros((len(reactivation_cue_pca_vec[0]), 60))
        temp_s2 = np.zeros((len(reactivation_cue_pca_vec[0]), 60))
        temp_s2r = np.zeros((len(reactivation_cue_pca_vec[0]), 60))
        for i in range(0, len(reactivation_cue_pca_vec[0])):
            if c1 == [0, 59/255, 116/255]:
                if mouse < 5:
                    temp_s1[i, :] = reactivation_cue_pca_vec[0][i]
                    temp_s1r[i, :] = reactivation_cue_pca_vec[1][i]
                    temp_s2[i, :] = reactivation_cue_pca_vec[2][i]
                    temp_s2r[i, :] = reactivation_cue_pca_vec[3][i]
                if mouse > 4:
                    temp_s1[i, :] = scipy.signal.resample_poly(reactivation_cue_pca_vec[0][i], 2, 1, padtype='smooth')[
                                    0:(len(reactivation_cue_pca_vec[0][i]) * 2)]
                    temp_s1r[i, :] = scipy.signal.resample_poly(reactivation_cue_pca_vec[1][i], 2, 1, padtype='smooth')[
                                     0:(len(reactivation_cue_pca_vec[1][i]) * 2)]
                    temp_s2[i, :] = scipy.signal.resample_poly(reactivation_cue_pca_vec[2][i], 2, 1, padtype='smooth')[
                                    0:(len(reactivation_cue_pca_vec[2][i]) * 2)]
                    temp_s2r[i, :] = scipy.signal.resample_poly(reactivation_cue_pca_vec[3][i], 2, 1, padtype='smooth')[
                                     0:(len(reactivation_cue_pca_vec[3][i]) * 2)]
            else:
                if mouse < 5:
                    temp_s1[i, :] = reactivation_cue_pca_vec[6][i]
                    temp_s1r[i, :] = reactivation_cue_pca_vec[7][i]
                    temp_s2[i, :] = reactivation_cue_pca_vec[8][i]
                    temp_s2r[i, :] = reactivation_cue_pca_vec[9][i]
                if mouse > 4:
                    temp_s1[i, :] = scipy.signal.resample_poly(reactivation_cue_pca_vec[6][i], 2, 1, padtype='smooth')[
                                    0:(len(reactivation_cue_pca_vec[6][i]) * 2)]
                    temp_s1r[i, :] = scipy.signal.resample_poly(reactivation_cue_pca_vec[7][i], 2, 1, padtype='smooth')[
                                     0:(len(reactivation_cue_pca_vec[7][i]) * 2)]
                    temp_s2[i, :] = scipy.signal.resample_poly(reactivation_cue_pca_vec[8][i], 2, 1, padtype='smooth')[
                                    0:(len(reactivation_cue_pca_vec[8][i]) * 2)]
                    temp_s2r[i, :] = scipy.signal.resample_poly(reactivation_cue_pca_vec[9][i], 2, 1, padtype='smooth')[
                                     0:(len(reactivation_cue_pca_vec[9][i]) * 2)]
        y_s1 = np.nanmean(temp_s1, axis=0)
        y_s1r = np.nanmean(temp_s1r, axis=0)
        y_s2 = np.nanmean(temp_s2, axis=0)
        y_s2r = np.nanmean(temp_s2r, axis=0)
        all_s1[mouse, :] = y_s1
        all_s1r[mouse, :] = y_s1r
        all_s2[mouse, :] = y_s2
        all_s2r[mouse, :] = y_s2r

    # print(stats.ttest_rel(all_s2[:, 0:1], all_s2r[:, 59:60])[1])

    plt.subplot(2, 2, 1)
    plt.ylim(-1, .2)
    plt.yticks([0, -.5, -1])
    plt.gca().invert_yaxis()
    mean = all_s1.mean(axis=0)
    sem_plus = mean + stats.sem(all_s1, axis=0)
    sem_minus = mean - stats.sem(all_s1, axis=0)
    plt.plot(x_label, mean, '-', c=c1, linewidth=3)
    plt.fill_between(x_label, sem_plus, sem_minus, alpha=0.2, color=c1, lw=0)
    mean = all_s1r.mean(axis=0)
    sem_plus = mean + stats.sem(all_s1r, axis=0)
    sem_minus = mean - stats.sem(all_s1r, axis=0)
    plt.plot(x_label, mean, '-', c=c2, linewidth=3)
    plt.fill_between(x_label, sem_plus, sem_minus, alpha=0.2, color=c2, lw=0)
    plt.ylabel('Similarity to early vs. late\n S1 response pattern')
    plt.xlim(0, 60)
    plt.xticks([0, 59], ['First trial', 'Last trial'])
    plt.subplot(2, 2, 2)
    plt.ylim(-1, .2)
    plt.yticks([0, -.5, -1])
    plt.gca().invert_yaxis()
    mean = all_s2.mean(axis=0)
    sem_plus = mean + stats.sem(all_s2, axis=0)
    sem_minus = mean - stats.sem(all_s2, axis=0)
    plt.plot(x_label, mean, '-', c=c3, linewidth=3)
    plt.fill_between(x_label, sem_plus, sem_minus, alpha=0.2, color=c3, lw=0)
    mean = all_s2r.mean(axis=0)
    sem_plus = mean + stats.sem(all_s2r, axis=0)
    sem_minus = mean - stats.sem(all_s2r, axis=0)
    plt.plot(x_label, mean, '-', c=c4, linewidth=3)
    plt.fill_between(x_label, sem_plus, sem_minus, alpha=0.2, color=c4, lw=0)
    plt.ylabel('Similarity to early vs. late\n S2 response pattern)')
    plt.xlim(0, 60)
    plt.xticks([0, 59], ['First trial', 'Last trial'])
    sns.despine()


def reactivation_cue_vector_across_days_R1(mice, sample_dates):
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(20, 12))
    reactivation_cue_vector_across_days_R1_helper(mice, sample_dates, '_within', 1, -1.5, .5)
    reactivation_cue_vector_across_days_R1_helper(mice, sample_dates, '_day1', 3, -1.75, 0)
    reactivation_cue_vector_across_days_R1_helper(mice, sample_dates, '', 5, -1.75, 0)

    paths = preprocess.create_folders(mice[0], sample_dates[0])
    plt.savefig(paths['base_path'] + '/NN/plots/reactivation_cue_vector_across_days.png', bbox_inches='tight', dpi=200)
    plt.savefig(paths['base_path'] + '/NN/plots/reactivation_cue_vector_across_days.pdf', bbox_inches='tight', dpi=200, transparent=True)
    plt.close()


def reactivation_cue_vector_across_days_R1_helper(mice, sample_dates, vector_type, plt_idx, y_min, y_max):
    all_s1 = []
    all_s1r = []
    all_s2 = []
    all_s2r = []
    for mouse in range(0, len(mice)):
        paths = preprocess.create_folders(mice[mouse], sample_dates[mouse])
        reactivation_cue_pca_vec = np.load(paths['base_path'] + paths['mouse'] +
                                           '/data_across_days/reactivation_cue_vector_across_days' + vector_type + '.npy', allow_pickle=True)
        temp_s1 = np.empty((6, 60)) * np.nan
        temp_s1r = np.empty((6, 60)) * np.nan
        temp_s2 = np.empty((6, 60)) * np.nan
        temp_s2r = np.empty((6, 60)) * np.nan
        for i in range(0, 6):
            temp_s1[i, :] = reactivation_cue_pca_vec[0][i][0:60]
            temp_s1r[i, :] = reactivation_cue_pca_vec[1][i][0:60]
            temp_s2[i, :] = reactivation_cue_pca_vec[2][i][0:60]
            temp_s2r[i, :] = reactivation_cue_pca_vec[3][i][0:60]
        all_s1.append(temp_s1)
        all_s1r.append(temp_s1r)
        all_s2.append(temp_s2)
        all_s2r.append(temp_s2r)

    x_label = np.array(range(1, 61))
    s1_s = []
    s1_e = []
    s2_s = []
    s2_e = []
    for i in range(0, 6):
        temp_s1 = np.empty((len(mice), 60)) * np.nan
        temp_s1r = np.empty((len(mice), 60)) * np.nan
        temp_s2 = np.empty((len(mice), 60)) * np.nan
        temp_s2r = np.empty((len(mice), 60)) * np.nan
        for j in range(0, len(mice)):
            temp_s1[j, :] = all_s1[j][i]
            temp_s1r[j, :] = all_s1r[j][i]
            temp_s2[j, :] = all_s2[j][i]
            temp_s2r[j, :] = all_s2r[j][i]

        if i == 0:
            s1_s = np.abs(np.mean(temp_s1[:, 0:3], axis=1)) - np.abs(np.mean(temp_s1r[:, 0:3], axis=1))
            s2_s = np.abs(np.mean(temp_s2[:, 0:3], axis=1)) - np.abs(np.mean(temp_s2r[:, 0:3], axis=1))
        if i == 5:
            s1_e = np.abs(np.mean(temp_s1[:, 57:60], axis=1)) - np.abs(np.mean(temp_s1r[:, 57:60], axis=1))
            s2_e =np.abs( np.mean(temp_s2[:, 57:60], axis=1)) - np.abs(np.mean(temp_s2r[:, 57:60], axis=1))

        plt.subplot(3, 2, plt_idx)
        plt.yticks([.5, 0, -.5, -1, -1.5, -2])
        plt.ylim(y_min, y_max)
        plt.gca().invert_yaxis()
        mean = np.nanmean(temp_s1, axis=0)
        sem_plus = mean + stats.sem(temp_s1, axis=0, nan_policy='omit')
        sem_minus = mean - stats.sem(temp_s1, axis=0, nan_policy='omit')
        plt.plot(x_label, mean, '-', c='darkgreen', linewidth=3)
        plt.fill_between(x_label, sem_plus, sem_minus, alpha=0.2, color='darkgreen', lw=0)
        mean = np.nanmean(temp_s1r, axis=0)
        sem_plus = mean + stats.sem(temp_s1r, axis=0, nan_policy='omit')
        sem_minus = mean - stats.sem(temp_s1r, axis=0, nan_policy='omit')
        plt.plot(x_label, mean, '-', c='lime', linewidth=3)
        plt.fill_between(x_label, sem_plus, sem_minus, alpha=0.2, color='lime', lw=0)
        plt.ylabel('Similarity to early vs. late\n S1 response pattern')
        plt.xlabel('Trial number')
        label_1 = mlines.Line2D([], [], color='darkgreen', linewidth=2, label='S1')
        label_2 = mlines.Line2D([], [], color='lime', linewidth=2, label='S1 reactivation')
        plt.xlim(-5, 365)
        plt.xticks([0, 49, 99, 149, 199, 249, 299, 349], ['1', '50', '100', '150', '200', '250', '300', '350'])

        plt.subplot(3, 2, plt_idx+1)
        plt.yticks([.5, 0, -.5, -1, -1.5, -2])
        plt.ylim(y_min, y_max)
        plt.gca().invert_yaxis()
        mean = np.nanmean(temp_s2, axis=0)
        sem_plus = mean + stats.sem(temp_s2, axis=0, nan_policy='omit')
        sem_minus = mean - stats.sem(temp_s2, axis=0, nan_policy='omit')
        plt.plot(x_label, mean, '-', c='darkred', linewidth=3)
        plt.fill_between(x_label, sem_plus, sem_minus, alpha=0.2, color='darkred', lw=0)
        mean = np.nanmean(temp_s2r, axis=0)
        sem_plus = mean + stats.sem(temp_s2r, axis=0, nan_policy='omit')
        sem_minus = mean - stats.sem(temp_s2r, axis=0, nan_policy='omit')
        plt.plot(x_label, mean, '-', c='hotpink', linewidth=3)
        plt.fill_between(x_label, sem_plus, sem_minus, alpha=0.2, color='hotpink', lw=0)
        plt.ylabel('Similarity to early vs. late\n S2 response pattern)')
        plt.xlabel('Trial number')
        label_1 = mlines.Line2D([], [], color='darkred', linewidth=2, label='S2')
        label_2 = mlines.Line2D([], [], color='hotpink', linewidth=2, label='S2 reactivation')

        x_label += 60

        plt.xlim(-5, 364)
        plt.xticks([0, 49, 99, 149, 199, 249, 299, 349], ['1', '50', '100', '150', '200', '250', '300', '350'])
    sns.despine()
    print(stats.ttest_rel(s1_s, s1_e)[1])
    print(stats.ttest_rel(s2_s, s2_e)[1])


def prior_R1(mice, sample_dates):
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(4, 6))
    m_colors = ['b', 'teal', 'slategray', 'green', 'darkolivegreen', 'darkorange', 'purple', 'darkred']
    x = [1, 2, 3]
    events_all = []
    corr_all = []
    for mouse in range(0, len(mice)):
        paths = preprocess.create_folders(mice[mouse], sample_dates[mouse])
        prior_vec = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/prior_R1.npy', allow_pickle=True)
        events = []
        corr = []
        for i in range(0, len(prior_vec[0])):
            events.append(prior_vec[0][i]/34)
            corr.append(prior_vec[1][i])

        events_all.append(np.nanmean(events))
        corr_all.append(np.nanmean(corr))

        plt.subplot(2, 2, 1)
        plt.errorbar(x[0], np.nanmean(events), yerr=0, c=m_colors[mouse], marker='o', mfc='none', mec=m_colors[mouse],
                     ms=5, alpha=.3)
        plt.subplot(2, 2, 2)
        plt.errorbar(x[0], np.nanmean(corr), yerr=0, c=m_colors[mouse], marker='o', mfc='none', mec=m_colors[mouse],
                     ms=5, alpha=.3)

    plt.subplot(2, 2, 1)
    y1 = np.mean(events_all)
    y1_err = stats.sem(events_all)
    plt.errorbar(1.2, y1, yerr=y1_err, c='k', linewidth=2, marker='o', mfc='k', mec='k',
                 ms=7, mew=0, zorder=100)
    plt.xlim(.5, 2)
    plt.ylim(0, 10)
    plt.xticks([])
    plt.ylabel('Number of synchronous\nevents per minute\nduring baseline period')
    plt.subplot(2, 2, 2)
    y1 = np.mean(corr_all)
    y1_err = stats.sem(corr_all)
    plt.errorbar(1.2, y1, yerr=y1_err, c='k', linewidth=2, marker='o', mfc='k', mec='k',
                 ms=7, mew=0, zorder=100)
    plt.xlim(.5, 2)
    plt.ylim(0, .2)
    plt.xticks([])
    plt.ylabel('Mean correlation between\nsynchronous events')

    sns.despine()
    plt.savefig(paths['base_path'] + '/NN/plots/prior_R1.png', bbox_inches='tight', dpi=200)
    plt.savefig(paths['base_path'] + '/NN/plots/prior_R1.pdf', bbox_inches='tight', dpi=200,
                transparent=True)
    plt.close()


def activity_across_trials_oddeven_R2R3(mice, sample_dates):
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(9, 6))
    plt.subplots_adjust(wspace=.33)

    plt.subplot(2, 2, 1)
    activity_all = np.empty((len(mice), 128)) * np.nan
    for mouse in range(0, len(mice)):
        paths = preprocess.create_folders(mice[mouse], sample_dates[mouse])
        activity_data = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/activity.npy',
                               allow_pickle=True)
        activity = np.empty((len(activity_data[0]), 128)) * np.nan
        for i in range(0, len(activity_data[0]), 2):
            smoothed_activity = np.array(pd.DataFrame(activity_data[0][i]).rolling(3, min_periods=1, center=True).mean())
            smoothed_activity = np.concatenate(smoothed_activity, axis=0)
            if mouse < 5:
                activity[i, 0:len(smoothed_activity)] = smoothed_activity
            if mouse > 4:
                smoothed_activity = scipy.signal.resample_poly(smoothed_activity, 2, 1, padtype='smooth')[0:(len(smoothed_activity) * 2)]
                activity[i, 0:len(smoothed_activity)] = smoothed_activity

        activity = np.nanmean(activity, axis=0) - np.mean(np.nanmean(activity, axis=0)[0:3])
        activity_all[mouse, :] = activity
        x = range(1, len(activity)+1)

    mean = np.nanmean(activity_all, axis=0)
    sem_plus = mean + stats.sem(activity_all, axis=0, nan_policy='omit')
    sem_minus = mean - stats.sem(activity_all, axis=0, nan_policy='omit')
    plt.plot(x[0:120], mean[0:120], c='b', linewidth=3)
    plt.fill_between(x[0:120], sem_plus[0:120], sem_minus[0:120], alpha=0.2, color='b', lw=0)

    activity_all = np.empty((len(mice), 128)) * np.nan
    for mouse in range(0, len(mice)):
        paths = preprocess.create_folders(mice[mouse], sample_dates[mouse])
        activity_data = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/activity.npy',
                                allow_pickle=True)
        activity = np.empty((len(activity_data[0]), 128)) * np.nan
        for i in range(1, len(activity_data[0]), 2):
            smoothed_activity = np.array(
                pd.DataFrame(activity_data[0][i]).rolling(3, min_periods=1, center=True).mean())
            smoothed_activity = np.concatenate(smoothed_activity, axis=0)
            if mouse < 5:
                activity[i, 0:len(smoothed_activity)] = smoothed_activity
            if mouse > 4:
                smoothed_activity = scipy.signal.resample_poly(smoothed_activity, 2, 1, padtype='smooth')[0:(len(smoothed_activity) * 2)]
                activity[i, 0:len(smoothed_activity)] = smoothed_activity
        activity = np.nanmean(activity, axis=0) - np.mean(np.nanmean(activity, axis=0)[0:3])
        activity_all[mouse, :] = activity
        x = range(1, len(activity) + 1)
        # plt.plot(x, activity, c=m_colors[mouse], alpha=.2)

    mean = np.nanmean(activity_all, axis=0)
    sem_plus = mean + stats.sem(activity_all, axis=0, nan_policy='omit')
    sem_minus = mean - stats.sem(activity_all, axis=0, nan_policy='omit')
    plt.plot(x[0:120], mean[0:120], c='r', linewidth=3)
    plt.fill_between(x[0:120], sem_plus[0:120], sem_minus[0:120], alpha=0.2, color='r', lw=0)
    plt.ylabel('ΔCorrelation between\nstimulus 1 and stimulus 2')
    # plt.ylim(-.1, .6)
    plt.xlim(1, 121)
    plt.xticks([1, 120], ['First trial', 'Last trial'])
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
    label_1 = mlines.Line2D([], [], color='r', linewidth=2, label='Odd days')
    label_2 = mlines.Line2D([], [], color='b', linewidth=2, label='Even days')
    plt.legend(handles=[label_1, label_2], frameon=False)
    sns.despine()

    plt.savefig(paths['base_path'] + '/NN/plots/activity_across_trials_oddeven.png', bbox_inches='tight', dpi=200)
    plt.savefig(paths['base_path'] + '/NN/plots/activity_across_trials_oddeven.pdf', bbox_inches='tight', dpi=200, transparent=True)
    plt.close()


def reactivation_cue_vector_oddeven_R2R3(mice, sample_dates):
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(9, 6))
    plt.subplots_adjust(wspace=.3)
    x_label = list(range(0, 60))

    paths = preprocess.create_folders(mice[0], sample_dates[0])
    reactivation_cue_pca_vec = np.load(paths['base_path'] + paths['mouse'] +
                                    '/data_across_days/reactivation_cue_vector.npy', allow_pickle=True)
    all_s1 = np.empty((len(mice), len(reactivation_cue_pca_vec[0][0]))) * np.nan
    for mouse in range(0, len(mice)):
        paths = preprocess.create_folders(mice[mouse], sample_dates[mouse])
        reactivation_cue_pca_vec = np.load(paths['base_path'] + paths['mouse'] +
                                           '/data_across_days/reactivation_cue_vector.npy', allow_pickle=True)
        temp_s1 = np.empty((len(reactivation_cue_pca_vec[0]), 60)) * np.nan
        for i in range(0, len(reactivation_cue_pca_vec[0]), 2):
            if mouse < 5:
                temp_s1[i, :] = (np.array(reactivation_cue_pca_vec[0][i]) + np.array(reactivation_cue_pca_vec[2][i])) / 2
            if mouse > 4:
                temp_s1[i, :] = (scipy.signal.resample_poly(reactivation_cue_pca_vec[0][i], 2, 1, padtype='smooth')[0:(len(reactivation_cue_pca_vec[0][i]) * 2)] +
                                 scipy.signal.resample_poly(reactivation_cue_pca_vec[2][i], 2, 1, padtype='smooth')[
                                 0:(len(reactivation_cue_pca_vec[2][i]) * 2)]) / 2
        y_s1 = np.nanmean(temp_s1, axis=0)
        all_s1[mouse, :] = y_s1

    plt.subplot(2, 2, 1)
    plt.ylim(-1.5, .1)
    plt.yticks([0, -.5, -1, -1.5])
    plt.gca().invert_yaxis()
    mean = all_s1.mean(axis=0)
    sem_plus = mean + stats.sem(all_s1, axis=0)
    sem_minus = mean - stats.sem(all_s1, axis=0)
    plt.plot(x_label, mean, '-', c='b', linewidth=3)
    plt.fill_between(x_label, sem_plus, sem_minus, alpha=0.2, color='b', lw=0)
    plt.ylabel('Similarity to early vs. late\n S response pattern')
    plt.xlim(0, 60)
    plt.xticks([0, 59], ['First trial', 'Last trial'])

    paths = preprocess.create_folders(mice[0], sample_dates[0])
    reactivation_cue_pca_vec = np.load(paths['base_path'] + paths['mouse'] +
                                       '/data_across_days/reactivation_cue_vector.npy', allow_pickle=True)
    all_s1 = np.empty((len(mice), len(reactivation_cue_pca_vec[0][0]))) * np.nan
    for mouse in range(0, len(mice)):
        paths = preprocess.create_folders(mice[mouse], sample_dates[mouse])
        reactivation_cue_pca_vec = np.load(paths['base_path'] + paths['mouse'] +
                                           '/data_across_days/reactivation_cue_vector.npy', allow_pickle=True)
        temp_s1 = np.empty((len(reactivation_cue_pca_vec[0]), 60)) * np.nan
        for i in range(1, len(reactivation_cue_pca_vec[0]), 2):
            if mouse < 5:
                temp_s1[i, :] = (np.array(reactivation_cue_pca_vec[0][i]) + np.array(
                    reactivation_cue_pca_vec[2][i])) / 2
            if mouse > 4:
                temp_s1[i, :] = (scipy.signal.resample_poly(reactivation_cue_pca_vec[0][i], 2, 1, padtype='smooth')[0:(len(reactivation_cue_pca_vec[0][i]) * 2)] +
                                 scipy.signal.resample_poly(reactivation_cue_pca_vec[2][i], 2, 1, padtype='smooth')[
                                 0:(len(reactivation_cue_pca_vec[2][i]) * 2)]) / 2
        y_s1 = np.nanmean(temp_s1, axis=0)
        all_s1[mouse, :] = y_s1

    plt.subplot(2, 2, 1)
    plt.ylim(-1.5, .1)
    plt.yticks([0, -.5, -1, -1.5])
    plt.gca().invert_yaxis()
    mean = all_s1.mean(axis=0)
    sem_plus = mean + stats.sem(all_s1, axis=0)
    sem_minus = mean - stats.sem(all_s1, axis=0)
    plt.plot(x_label, mean, '-', c='r', linewidth=3)
    plt.fill_between(x_label, sem_plus, sem_minus, alpha=0.2, color='r', lw=0)
    plt.ylabel('Similarity to early vs. late\n S response pattern')
    plt.xlim(0, 60)
    plt.xticks([0, 59], ['First trial', 'Last trial'])

    label_1 = mlines.Line2D([], [], color='r', linewidth=2, label='Odd days')
    label_2 = mlines.Line2D([], [], color='b', linewidth=2, label='Even days')
    plt.legend(handles=[label_1, label_2], frameon=False)

    sns.despine()
    plt.savefig(paths['base_path'] + '/NN/plots/reactivation_cue_vector_oddeven.png', bbox_inches='tight', dpi=200)
    plt.savefig(paths['base_path'] + '/NN/plots/reactivation_cue_vector_oddeven.pdf', bbox_inches='tight', dpi=200, transparent=True)
    plt.close()


def pupil_activity_reactivation_modulation(mice, sample_dates):
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(2.5, 6))
    m_colors = ['b', 'teal', 'slategray', 'green', 'darkolivegreen', 'darkorange', 'purple', 'darkred']
    x = [1, 2]
    rate_pupil_all = []
    rate_activity_all = []
    for mouse in range(0, len(mice)):
        paths = preprocess.create_folders(mice[mouse], sample_dates[mouse])
        data_vec = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/pupil_activity_reactivation.npy', allow_pickle=True)
        rate_pupil = np.zeros(len(data_vec[0]))
        rate_activity = np.zeros(len(data_vec[0]))
        for i in range(0, len(data_vec[0])):
            rate_pupil[i] = data_vec[0][i]
            rate_activity[i] = data_vec[1][i]
        rate_pupil_all.append(np.mean(rate_pupil))
        rate_activity_all.append(np.mean(rate_activity))
        plt.subplot(2, 2, 1)
        plt.errorbar(x[0], rate_pupil_all[mouse], yerr=0, c=m_colors[mouse], marker='o', mfc='none', mec=m_colors[mouse],
                     ms=5, alpha=.3)
        plt.subplot(2, 2, 2)
        plt.errorbar(x[0], rate_activity_all[mouse], yerr=0, c=m_colors[mouse], marker='o', mfc='none',
                     mec=m_colors[mouse],
                     ms=5, alpha=.3)

    print(stats.ttest_1samp(rate_pupil_all, 0))
    print(stats.ttest_1samp(rate_activity_all, 0))

    plt.subplot(2, 2, 1)
    plt.errorbar(x[0] + .2, np.mean(rate_pupil_all, axis=0), yerr=stats.sem(rate_pupil_all, axis=0), c='k', linewidth=2, marker='o',
                 mfc='k', mec='k',
                 ms=7, mew=0, zorder=100)
    plt.xlim(.5, 1.5)
    plt.ylim(-.31, .31)
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1, snap=False)
    plt.yticks([-.3, -.2, -.1, 0, .1, .2, .3])
    plt.xticks([])
    plt.ylabel('Correlation between reactivation rate\nand pupil area')
    plt.subplot(2, 2, 2)
    plt.errorbar(x[0] + .2, np.mean(rate_activity_all, axis=0), yerr=stats.sem(rate_activity_all, axis=0), c='k', linewidth=2,
                 marker='o',
                 mfc='k', mec='k',
                 ms=7, mew=0, zorder=100)
    plt.xlim(.5, 1.5)
    plt.ylim(-.31, .31)
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1, snap=False)
    plt.xticks([])
    plt.yticks([-.3, -.2, -.1, 0, .1, .2, .3])
    plt.ylabel('Correlation between reactivation rate\nand stimulus activity')
    sns.despine()
    plt.savefig(paths['base_path'] + '/NN/plots/pupil_activity_modulation.png', bbox_inches='tight', dpi=200)
    plt.savefig(paths['base_path'] + '/NN/plots/pupil_activity_modulation.pdf', bbox_inches='tight', dpi=200, transparent=True)
    #plt.close()


def ripples(mice, sample_dates):
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(14.5, 7))
    plt.subplot(2, 3, 1)
    m_colors = ['b', 'teal', 'slategray', 'green', 'darkolivegreen', 'darkorange', 'purple', 'darkred']

    paths = preprocess.create_folders(mice[0], sample_dates[0])

    with open(paths['base_path'] + '/NN/ripple/hardware_reference_traces.pkl', 'rb') as f:
        ripple_data = pickle.load(f)

    all_ripple = []
    mean_mice_heatmap = []
    days = []
    for i in range(0, len(ripple_data)):
        ripple_vec = []
        session_data = ripple_data[list(ripple_data)[i]]
        for j in range(0, len(session_data)):
            ripple_vec.append(scipy.signal.resample_poly(session_data[list(session_data)[j]], 1, 160, padtype='smooth'))
            mean_mice_heatmap.append(scipy.signal.resample_poly(session_data[list(session_data)[j]], 1, 160, padtype='smooth'))
        all_ripple.append(np.mean(ripple_vec, axis=0))
        days.append(len(session_data))

    test_vec_b = []
    test_vec_a = []
    for i in range(0, len(all_ripple)):
        test_vec_b.append(all_ripple[i][0])
        test_vec_a.append(all_ripple[i][int(len(all_ripple[0])/2)])

    [_, s_p_value] = stats.shapiro(test_vec_a)
    print(s_p_value)
    print(stats.ttest_rel(test_vec_a, test_vec_b))

    mean = np.mean(all_ripple, axis=0)
    sem_plus = mean + stats.sem(all_ripple, axis=0)
    sem_minus = mean - stats.sem(all_ripple, axis=0)
    plt.plot(mean, c='blueviolet', linewidth=2, ms=7)
    plt.fill_between(range(0, len(mean)), sem_plus, sem_minus, alpha=0.2, color='blueviolet', lw=0)
    plt.xticks([0, int(len(mean)/4), int(len(mean)/2), int(len(mean)*3/4), len(mean)], ['-20', '-10', '0', '10', '20'])
    plt.axvline(x=int(len(mean)/2), color='black', linestyle='--', linewidth=1, snap=False)
    plt.ylim(-.1, .5)
    plt.yticks([0, .25, .5])
    plt.xlabel('Time relative to reactivation onset (s)')
    plt.ylabel("Ripple-band power (z-scored)")
    sns.despine()

    plt.savefig(paths['base_path'] + '/NN/plots/ripple.png', bbox_inches='tight', dpi=200)
    plt.savefig(paths['base_path'] + '/NN/plots/ripple.pdf', bbox_inches='tight', dpi=200, transparent=True)
    plt.close()

    sns.set(font_scale=.7)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(4, 6))
    plt.subplot(2, 2, 1)
    sns.heatmap(mean_mice_heatmap, vmin=-0, vmax=.4, cmap="Reds", cbar=0)
    y_idx = 0
    for mouse in range(0, 5):
        plt.plot([-5, -5], [y_idx, y_idx + days[mouse]], linewidth=7, snap=False, solid_capstyle='butt')
        y_idx += days[mouse]
    plt.ylim(len(mean_mice_heatmap) + 3, -3)
    plt.xlim(0, len(mean))
    plt.axis('off')
    plt.savefig(paths['base_path'] + '/NN/plots/ripple_heatmap.png', bbox_inches='tight', dpi=500,
                transparent=True)


def reactivation_rate_day_vs_opto(mice, sample_dates):
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

    mean_reactivation_mice = np.zeros((5, len(binned_vec[0])))
    for mouse in range(0, 5):
        paths = preprocess.create_folders(mice[mouse], sample_dates[mouse])
        binned_vec = list(np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/y_pred_binned.npy'))
        mean_reactivation = np.zeros((len(binned_vec), len(binned_vec[0])))
        for i in range(0, len(binned_vec)):
            mean_reactivation[i, :] = binned_vec[i]
        mean = mean_reactivation.mean(axis=0)
        plt.plot(x_label, mean, '-', c='k', ms=0, alpha=.2, linewidth=2)
        mean_reactivation_mice[mouse, :] = mean

    mean = np.mean(mean_reactivation_mice, axis=0)
    sem_plus = mean + stats.sem(mean_reactivation_mice, axis=0)
    sem_minus = mean - stats.sem(mean_reactivation_mice, axis=0)
    plt.plot(x_label, mean, '-o', c='k', linewidth=3, ms=0)
    plt.fill_between(x_label, sem_plus, sem_minus, alpha=0.2, color='k', lw=0)

    mean_reactivation_mice = np.zeros((3, len(binned_vec[0])))
    for mouse in range(5, len(mice)):
        paths = preprocess.create_folders(mice[mouse], sample_dates[mouse])
        binned_vec = list(np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/y_pred_binned.npy', allow_pickle=True))
        mean_reactivation = np.zeros((len(binned_vec[0]), len(binned_vec[0][0])))
        for i in range(0, len(binned_vec[0])):
            mean_reactivation[i, :] = binned_vec[0][i]
        mean = mean_reactivation.mean(axis=0)
        plt.plot(x_label, mean, '-', c='r', ms=0, alpha=.2, linewidth=2)
        mean_reactivation_mice[mouse-5, :] = mean

    mean = np.mean(mean_reactivation_mice, axis=0)
    sem_plus = mean + stats.sem(mean_reactivation_mice, axis=0)
    sem_minus = mean - stats.sem(mean_reactivation_mice, axis=0)
    plt.plot(x_label, mean, '-o', c='r', linewidth=3, ms=0)
    plt.fill_between(x_label, sem_plus, sem_minus, alpha=0.2, color='r', lw=0)
    plt.axvspan(-hours_per_run, 0, alpha=.1, color='gray', lw=0)
    plt.ylabel('Reactivation rate (probablity $\mathregular{s^{-1}}$)')
    plt.xlabel('Time relative to first stimulus onset (h)')
    plt.ylim(0, .15)
    plt.xlim(-hours_per_run, x_label[len(x_label)-1] + hours_per_run / 4)
    plt.xticks([-.5, 0, .5, 1, 1.5, 2])
    label_1 = mlines.Line2D([], [], color='b', linestyle='-', label='Normal mice', alpha=.2, linewidth=2)
    label_2 = mlines.Line2D([], [], color='purple', linestyle='-', label='Inhibition mice', alpha=.2, linewidth=2)
    plt.legend(handles=[label_1, label_2], frameon=False, prop={'size': 8}, labelspacing=.1)
    sns.despine()
    # plt.savefig(paths['base_path'] + '/NN/plots/reactivation_rate_day.png', bbox_inches='tight', dpi=200, transparent=True)
    # plt.savefig(paths['base_path'] + '/NN/plots/reactivation_rate_day.pdf', bbox_inches='tight', dpi=200, transparent=True)
    # plt.close()


def quantify_num_across_days(mice, sample_dates):
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(3, 6))
    m_colors = ['b', 'teal', 'slategray', 'green', 'darkolivegreen', 'darkorange', 'purple', 'darkred']
    scale_all = []
    for mouse in range(0, len(mice)):
        paths = preprocess.create_folders(mice[mouse], sample_dates[mouse])
        reactivation_scale = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/num_across_days.npy', allow_pickle=True)
        scale = []
        for i in range(0, 6):
            scale.append(reactivation_scale[0][i])
        scale_all.append(np.mean(scale))
        plt.subplot(2, 2, 1)
        plt.errorbar(1, np.mean(scale), yerr=0, c=m_colors[mouse], marker='o', mfc='none', mec=m_colors[mouse],
                     ms=5, alpha=.3)
    print(np.mean(scale_all))
    print(stats.sem(scale_all))

    plt.subplot(2, 2, 1)
    y1 = np.mean(scale_all)
    y1_err = stats.sem(scale_all)
    plt.errorbar(1.2, y1, yerr=y1_err, c='k', linewidth=2, marker='o', mfc='k', mec='k',
                 ms=7, mew=0, zorder=100)
    plt.xlim(.5, 1.5)
    plt.ylim(0, 2000)
    plt.xticks([])
    plt.ylabel('Number of neurons\nfound across 6 days')
    sns.despine()
    plt.savefig(paths['base_path'] + '/NN/plots/num_across_days.png', bbox_inches='tight', dpi=200)
    plt.savefig(paths['base_path'] + '/NN/plots/num_across_days.pdf', bbox_inches='tight', dpi=200,
                transparent=True)
    plt.close()


def num_selective_grouped(mice, sample_dates):
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(9, 6))
    plt.subplots_adjust(wspace=.45)
    m_colors = ['b', 'teal', 'slategray', 'green', 'darkolivegreen', 'darkorange', 'purple', 'darkred']
    x = [1, 2, 3, 4, 5, 6]
    nc_i_all = []
    inc_i_all = []
    dec_i_all = []
    nc_d_all = []
    inc_d_all = []
    dec_d_all = []
    for mouse in range(0, len(mice)):
        paths = preprocess.create_folders(mice[mouse], sample_dates[mouse])
        vec = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/num_selective_grouped.npy', allow_pickle=True)
        nc_i = []
        inc_i = []
        dec_i = []
        nc_d = []
        inc_d = []
        dec_d = []
        for i in range(0, len(vec[0])):
            nc_i.append(vec[7][i])
            inc_i.append(vec[4][i])
            dec_i.append(vec[1][i])
            nc_d.append(vec[8][i])
            inc_d.append(vec[5][i])
            dec_d.append(vec[2][i])
        nc_i_all.append(np.nanmean(nc_i))
        inc_i_all.append(np.nanmean(inc_i))
        dec_i_all.append(np.nanmean(dec_i))
        nc_d_all.append(np.nanmean(nc_d))
        inc_d_all.append(np.nanmean(inc_d))
        dec_d_all.append(np.nanmean(dec_d))
        plt.subplot(2, 2, 1)

        # plt.errorbar(x[0], np.nanmean(nc_i), yerr=0, c=m_colors[mouse], marker='o', mfc='none', mec=m_colors[mouse],
        #              ms=5, alpha=.3)
        # plt.errorbar(x[1], np.nanmean(nc_d), yerr=0, c=m_colors[mouse], marker='o', mfc='none', mec=m_colors[mouse],
        #              ms=5, alpha=.3)
        #
        # plt.errorbar(x[2], np.nanmean(inc_i), yerr=0, c=m_colors[mouse], marker='o', mfc='none', mec=m_colors[mouse],
        #              ms=5, alpha=.3)
        # plt.errorbar(x[3], np.nanmean(inc_d), yerr=0, c=m_colors[mouse], marker='o', mfc='none', mec=m_colors[mouse],
        #              ms=5, alpha=.3)
        #
        # plt.errorbar(x[4], np.nanmean(dec_i), yerr=0, c=m_colors[mouse], marker='o', mfc='none', mec=m_colors[mouse],
        #              ms=5, alpha=.3)
        # plt.errorbar(x[5], np.nanmean(dec_d), yerr=0, c=m_colors[mouse], marker='o', mfc='none', mec=m_colors[mouse],
        #              ms=5, alpha=.3)

        plt.plot([1, 2], [np.nanmean(nc_i), np.nanmean(nc_d)], c=m_colors[mouse], linewidth=2, linestyle='-', alpha=.2)
        plt.plot([3, 4], [np.nanmean(inc_i), np.nanmean(inc_d)], c=m_colors[mouse], linewidth=2, linestyle='-', alpha=.2)
        plt.plot([5, 6], [np.nanmean(dec_i), np.nanmean(dec_d)], c=m_colors[mouse], linewidth=2, linestyle='-', alpha=.2)


    [_, p_nc] = stats.ttest_rel(nc_i_all, nc_d_all)
    [_, p_inc] = stats.ttest_rel(inc_i_all, inc_d_all)
    [_, p_dec] = stats.ttest_rel(dec_i_all, dec_d_all)
    anova_results = []
    anova_results.append(p_nc)
    anova_results.append(p_inc)
    anova_results.append(p_dec)
    anova_results_corrected = multipletests(anova_results, alpha=.05, method='holm')
    print(anova_results_corrected[1])
    plt.subplot(2, 2, 1)
    y1 = np.mean(nc_i_all)
    y1_err = stats.sem(nc_i_all)
    # plt.errorbar(1.2, y1, yerr=y1_err, c='k', linewidth=2, marker='o', mfc='k', mec='k',
    #              ms=7, mew=0, zorder=100)
    y2 = np.mean(nc_d_all)
    y2_err = stats.sem(nc_d_all)
    # plt.errorbar(2.2, y2, yerr=y2_err, c='k', linewidth=2, marker='o', mfc='k', mec='k',
    #              ms=7, mew=0, zorder=100)
    plt.plot([1, 2], [y1, y2], '-', c='k', linewidth=3)
    plt.fill_between([1, 2], [y1 + y1_err, y2 + y2_err],
                     [y1 - y1_err, y2 - y2_err], alpha=0.2, color='k',
                     lw=0)

    y1 = np.mean(inc_i_all)
    y1_err = stats.sem(inc_i_all)
    # plt.errorbar(4.2, y1, yerr=y1_err, c='k', linewidth=2, marker='o', mfc='k', mec='k',
    #              ms=7, mew=0, zorder=100)
    y2 = np.mean(inc_d_all)
    y2_err = stats.sem(inc_d_all)
    # plt.errorbar(5.2, y2, yerr=y2_err, c='k', linewidth=2, marker='o', mfc='k', mec='k',
    #              ms=7, mew=0, zorder=100)
    plt.plot([3, 4], [y1, y2], '-', c='k', linewidth=3)
    plt.fill_between([3, 4], [y1 + y1_err, y2 + y2_err],
                     [y1 - y1_err, y2 - y2_err], alpha=0.2, color='k',
                     lw=0)

    y1 = np.mean(dec_i_all)
    y1_err = stats.sem(dec_i_all)
    # plt.errorbar(7.2, y1, yerr=y1_err, c='k', linewidth=2, marker='o', mfc='k', mec='k',
    #              ms=7, mew=0, zorder=100)
    y2 = np.mean(dec_d_all)
    y2_err = stats.sem(dec_d_all)
    # plt.errorbar(8.2, y2, yerr=y2_err, c='k', linewidth=2, marker='o', mfc='k', mec='k',
    #              ms=7, mew=0, zorder=100)
    plt.plot([5, 6], [y1, y2], '-', c='k', linewidth=3)
    plt.fill_between([5, 6], [y1 + y1_err, y2 + y2_err],
                     [y1 - y1_err, y2 - y2_err], alpha=0.2, color='k',
                     lw=0)

    plt.xlim(0.5, 6.5)
    plt.ylim(-.01, .35)
    plt.xticks([1, 2, 3, 4, 5, 6], ['+', '-', '+', '-', '+', '-'])
    plt.ylabel('Percent of neurons with\nincrease (+) or decrease (-)\nin stimulus selectivity\nfrom early to late trials')

    nc_i_all = []
    inc_i_all = []
    dec_i_all = []
    nc_d_all = []
    inc_d_all = []
    dec_d_all = []
    for mouse in range(0, 5):
        paths = preprocess.create_folders(mice[mouse], sample_dates[mouse])
        vec = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/num_selective_grouped.npy',
                      allow_pickle=True)
        nc_i = []
        inc_i = []
        dec_i = []
        nc_d = []
        inc_d = []
        dec_d = []
        for i in range(0, len(vec[0])):
            nc_i.append(vec[7][i])
            inc_i.append(vec[4][i])
            dec_i.append(vec[1][i])
            nc_d.append(vec[8][i])
            inc_d.append(vec[5][i])
            dec_d.append(vec[2][i])
        nc_i_all.append(np.nanmean(nc_i))
        inc_i_all.append(np.nanmean(inc_i))
        dec_i_all.append(np.nanmean(dec_i))
        nc_d_all.append(np.nanmean(nc_d))
        inc_d_all.append(np.nanmean(inc_d))
        dec_d_all.append(np.nanmean(dec_d))
        plt.subplot(2, 2, 2)

        # plt.errorbar(x[0], np.nanmean(nc_i), yerr=0, c=m_colors[mouse], marker='o', mfc='none', mec=m_colors[mouse],
        #              ms=5, alpha=.3)
        # plt.errorbar(x[1], np.nanmean(nc_d), yerr=0, c=m_colors[mouse], marker='o', mfc='none', mec=m_colors[mouse],
        #              ms=5, alpha=.3)
        #
        # plt.errorbar(x[2], np.nanmean(inc_i), yerr=0, c=m_colors[mouse], marker='o', mfc='none', mec=m_colors[mouse],
        #              ms=5, alpha=.3)
        # plt.errorbar(x[3], np.nanmean(inc_d), yerr=0, c=m_colors[mouse], marker='o', mfc='none', mec=m_colors[mouse],
        #              ms=5, alpha=.3)
        #
        # plt.errorbar(x[4], np.nanmean(dec_i), yerr=0, c=m_colors[mouse], marker='o', mfc='none', mec=m_colors[mouse],
        #              ms=5, alpha=.3)
        # plt.errorbar(x[5], np.nanmean(dec_d), yerr=0, c=m_colors[mouse], marker='o', mfc='none', mec=m_colors[mouse],
        #              ms=5, alpha=.3)

    [_, p_nc] = stats.ttest_rel(nc_i_all, nc_d_all)
    [_, p_inc] = stats.ttest_rel(inc_i_all, inc_d_all)
    [_, p_dec] = stats.ttest_rel(dec_i_all, dec_d_all)
    anova_results = []
    anova_results.append(p_nc)
    anova_results.append(p_inc)
    anova_results.append(p_dec)
    anova_results_corrected = multipletests(anova_results, alpha=.05, method='holm')
    print(anova_results_corrected[1])
    y1 = np.mean(nc_i_all)
    y1_err = stats.sem(nc_i_all)
    # plt.errorbar(1.2, y1, yerr=y1_err, c='k', linewidth=2, marker='o', mfc='k', mec='k',
    #              ms=7, mew=0, zorder=100)
    y2 = np.mean(nc_d_all)
    y2_err = stats.sem(nc_d_all)
    # plt.errorbar(2.2, y2, yerr=y2_err, c='k', linewidth=2, marker='o', mfc='k', mec='k',
    #              ms=7, mew=0, zorder=100)
    plt.plot([1, 2], [y1, y2], '-', c='k', linewidth=3)
    plt.fill_between([1, 2], [y1 + y1_err, y2 + y2_err],
                     [y1 - y1_err, y2 - y2_err], alpha=0.2, color='k',
                     lw=0)

    y1 = np.mean(inc_i_all)
    y1_err = stats.sem(inc_i_all)
    # plt.errorbar(4.2, y1, yerr=y1_err, c='k', linewidth=2, marker='o', mfc='k', mec='k',
    #              ms=7, mew=0, zorder=100)
    y2 = np.mean(inc_d_all)
    y2_err = stats.sem(inc_d_all)
    # plt.errorbar(5.2, y2, yerr=y2_err, c='k', linewidth=2, marker='o', mfc='k', mec='k',
    #              ms=7, mew=0, zorder=100)
    plt.plot([3, 4], [y1, y2], '-', c='k', linewidth=3)
    plt.fill_between([3, 4], [y1 + y1_err, y2 + y2_err],
                     [y1 - y1_err, y2 - y2_err], alpha=0.2, color='k',
                     lw=0)

    y1 = np.mean(dec_i_all)
    y1_err = stats.sem(dec_i_all)
    # plt.errorbar(7.2, y1, yerr=y1_err, c='k', linewidth=2, marker='o', mfc='k', mec='k',
    #              ms=7, mew=0, zorder=100)
    y2 = np.mean(dec_d_all)
    y2_err = stats.sem(dec_d_all)
    # plt.errorbar(8.2, y2, yerr=y2_err, c='k', linewidth=2, marker='o', mfc='k', mec='k',
    #              ms=7, mew=0, zorder=100)
    plt.plot([5, 6], [y1, y2], '-', c='k', linewidth=3)
    plt.fill_between([5, 6], [y1 + y1_err, y2 + y2_err],
                     [y1 - y1_err, y2 - y2_err], alpha=0.2, color='k',
                     lw=0)

    nc_i_all = []
    inc_i_all = []
    dec_i_all = []
    nc_d_all = []
    inc_d_all = []
    dec_d_all = []
    for mouse in range(5, len(mice)):
        paths = preprocess.create_folders(mice[mouse], sample_dates[mouse])
        vec = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/num_selective_grouped.npy',
                      allow_pickle=True)
        nc_i = []
        inc_i = []
        dec_i = []
        nc_d = []
        inc_d = []
        dec_d = []
        for i in range(0, len(vec[0])):
            nc_i.append(vec[7][i])
            inc_i.append(vec[4][i])
            dec_i.append(vec[1][i])
            nc_d.append(vec[8][i])
            inc_d.append(vec[5][i])
            dec_d.append(vec[2][i])
        nc_i_all.append(np.nanmean(nc_i))
        inc_i_all.append(np.nanmean(inc_i))
        dec_i_all.append(np.nanmean(dec_i))
        nc_d_all.append(np.nanmean(nc_d))
        inc_d_all.append(np.nanmean(inc_d))
        dec_d_all.append(np.nanmean(dec_d))
        plt.subplot(2, 2, 2)

        # plt.errorbar(x[0], np.nanmean(nc_i), yerr=0, c=m_colors[mouse], marker='o', mfc='none', mec=m_colors[mouse],
        #              ms=5, alpha=.3)
        # plt.errorbar(x[1], np.nanmean(nc_d), yerr=0, c=m_colors[mouse], marker='o', mfc='none', mec=m_colors[mouse],
        #              ms=5, alpha=.3)
        #
        # plt.errorbar(x[2], np.nanmean(inc_i), yerr=0, c=m_colors[mouse], marker='o', mfc='none', mec=m_colors[mouse],
        #              ms=5, alpha=.3)
        # plt.errorbar(x[3], np.nanmean(inc_d), yerr=0, c=m_colors[mouse], marker='o', mfc='none', mec=m_colors[mouse],
        #              ms=5, alpha=.3)
        #
        # plt.errorbar(x[4], np.nanmean(dec_i), yerr=0, c=m_colors[mouse], marker='o', mfc='none', mec=m_colors[mouse],
        #              ms=5, alpha=.3)
        # plt.errorbar(x[5], np.nanmean(dec_d), yerr=0, c=m_colors[mouse], marker='o', mfc='none', mec=m_colors[mouse],
        #              ms=5, alpha=.3)

    [_, p_nc] = stats.ttest_rel(nc_i_all, nc_d_all)
    [_, p_inc] = stats.ttest_rel(inc_i_all, inc_d_all)
    [_, p_dec] = stats.ttest_rel(dec_i_all, dec_d_all)
    anova_results = []
    anova_results.append(p_nc)
    anova_results.append(p_inc)
    anova_results.append(p_dec)
    anova_results_corrected = multipletests(anova_results, alpha=.05, method='holm')
    print(anova_results_corrected[1])
    y1 = np.mean(nc_i_all)
    y1_err = stats.sem(nc_i_all)
    # plt.errorbar(1.2, y1, yerr=y1_err, c='k', linewidth=2, marker='o', mfc='k', mec='k',
    #              ms=7, mew=0, zorder=100)
    y2 = np.mean(nc_d_all)
    y2_err = stats.sem(nc_d_all)
    # plt.errorbar(2.2, y2, yerr=y2_err, c='k', linewidth=2, marker='o', mfc='k', mec='k',
    #              ms=7, mew=0, zorder=100)
    plt.plot([1, 2], [y1, y2], '-', c='r', linewidth=3)
    plt.fill_between([1, 2], [y1 + y1_err, y2 + y2_err],
                     [y1 - y1_err, y2 - y2_err], alpha=0.2, color='r',
                     lw=0)

    y1 = np.mean(inc_i_all)
    y1_err = stats.sem(inc_i_all)
    # plt.errorbar(4.2, y1, yerr=y1_err, c='k', linewidth=2, marker='o', mfc='k', mec='k',
    #              ms=7, mew=0, zorder=100)
    y2 = np.mean(inc_d_all)
    y2_err = stats.sem(inc_d_all)
    # plt.errorbar(5.2, y2, yerr=y2_err, c='k', linewidth=2, marker='o', mfc='k', mec='k',
    #              ms=7, mew=0, zorder=100)
    plt.plot([3, 4], [y1, y2], '-', c='r', linewidth=3)
    plt.fill_between([3, 4], [y1 + y1_err, y2 + y2_err],
                     [y1 - y1_err, y2 - y2_err], alpha=0.2, color='r',
                     lw=0)

    y1 = np.mean(dec_i_all)
    y1_err = stats.sem(dec_i_all)
    # plt.errorbar(7.2, y1, yerr=y1_err, c='k', linewidth=2, marker='o', mfc='k', mec='k',
    #              ms=7, mew=0, zorder=100)
    y2 = np.mean(dec_d_all)
    y2_err = stats.sem(dec_d_all)
    # plt.errorbar(8.2, y2, yerr=y2_err, c='k', linewidth=2, marker='o', mfc='k', mec='k',
    #              ms=7, mew=0, zorder=100)
    plt.plot([5, 6], [y1, y2], '-', c='r', linewidth=3)
    plt.fill_between([5, 6], [y1 + y1_err, y2 + y2_err],
                     [y1 - y1_err, y2 - y2_err], alpha=0.2, color='r',
                     lw=0)

    plt.xlim(0.5, 6.5)
    plt.ylim(-.01, .35)
    plt.xticks([1, 2, 3, 4, 5, 6], ['+', '-', '+', '-', '+', '-'])
    plt.ylabel(
        'Percent of neurons with\nincrease (+) or decrease (-)\nin stimulus selectivity\nfrom early to late trials')
    sns.despine()
    plt.savefig(paths['base_path'] + '/NN/plots/selective_grouped.png', bbox_inches='tight', dpi=200)
    plt.savefig(paths['base_path'] + '/NN/plots/selective_grouped.pdf', bbox_inches='tight', dpi=200,
                transparent=True)
    # plt.close()


def reactivation_difference_tunedflip_R2(mice, sample_dates):
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(5, 6))
    m_colors = ['b', 'teal', 'slategray', 'green', 'darkolivegreen', 'darkorange', 'purple', 'darkred']
    x = [1, 2, 3]
    S1_S2_1_all = []
    S1_S2_2_all = []
    S2_S1_1_all = []
    S2_S1_2_all = []
    for mouse in range(0, len(mice)):
        paths = preprocess.create_folders(mice[mouse], sample_dates[mouse])
        session_data = preprocess.load_data(paths)
        fr = session_data['framerate']
        reactivation_influence = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/reactivation_influence_flip.npy', allow_pickle=True)
        S1_S2_1 = []
        S1_S2_2 = []
        S2_S1_1 = []
        S2_S1_2 = []
        for i in range(0, len(reactivation_influence[0])):
            S1_S2_1.append(reactivation_influence[4][i] * fr)
            S1_S2_2.append(reactivation_influence[5][i] * fr)
            S2_S1_1.append(reactivation_influence[6][i] * fr)
            S2_S1_2.append(reactivation_influence[7][i] * fr)

        S1_S2_1_all.append(np.nanmean(S1_S2_1))
        S1_S2_2_all.append(np.nanmean(S1_S2_2))
        S2_S1_1_all.append(np.nanmean(S2_S1_1))
        S2_S1_2_all.append(np.nanmean(S2_S1_2))

        plt.subplot(2, 2, 1)
        plt.errorbar(x[0], np.nanmean(S1_S2_1), yerr=0, c=m_colors[mouse], marker='o', mfc='none', mec=m_colors[mouse],
                     ms=5, alpha=.3)
        plt.errorbar(x[1], np.nanmean(S1_S2_2), yerr=0, c=m_colors[mouse], marker='o', mfc='none', mec=m_colors[mouse],
                     ms=5, alpha=.3)
        plt.plot([1, 2], [np.nanmean(S1_S2_1), np.nanmean(S1_S2_2)], c=m_colors[mouse], linewidth=2, linestyle='-', alpha=.2)
        plt.subplot(2, 2, 2)
        plt.errorbar(x[0], np.nanmean(S2_S1_1), yerr=0, c=m_colors[mouse], marker='o', mfc='none', mec=m_colors[mouse],
                     ms=5, alpha=.3)
        plt.errorbar(x[1], np.nanmean(S2_S1_2), yerr=0, c=m_colors[mouse], marker='o', mfc='none', mec=m_colors[mouse],
                     ms=5, alpha=.3)
        plt.plot([1, 2], [np.nanmean(S2_S1_1), np.nanmean(S2_S1_2)], c=m_colors[mouse], linewidth=2, linestyle='-',
                 alpha=.2)

    print(stats.ttest_rel(S1_S2_1_all, S1_S2_2_all))
    print(stats.ttest_rel(S2_S1_1_all, S2_S1_2_all))

    plt.subplot(2, 2, 1)
    y1 = np.mean(S1_S2_1_all)
    y1_err = stats.sem(S1_S2_1_all)
    plt.errorbar(1.12, y1, yerr=y1_err, c='mediumseagreen', linewidth=2, marker='o', mfc='mediumseagreen', mec='mediumseagreen',
                 ms=7, mew=0, zorder=100)
    y2 = np.mean(S1_S2_2_all)
    y2_err = stats.sem(S1_S2_2_all)
    plt.errorbar(2.12, y2, yerr=y2_err, c='mediumseagreen', linewidth=2, marker='o', mfc='mediumseagreen', mec='mediumseagreen',
                 ms=7, mew=0, zorder=100)
    plt.plot([1.12, 2.12], [y1, y2], '-', c='mediumseagreen', linewidth=3)
    plt.fill_between([1.12, 2.12], [y1 + y1_err, y2 + y2_err], [y1 - y1_err, y2 - y2_err], alpha=0.2, color='mediumseagreen',
                     lw=0)
    plt.xlim(.5, 2.5)
    plt.yticks([-.4, -.2, 0, .2, .4])
    plt.ylim(-0.4, .4)
    plt.xticks([1, 2])
    # plt.ylabel('Stimulus activity\n(normalized deconvolved\n$\mathregular{Ca^{2+}}$transient $\mathregular{s^{-1}}$)')
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1)

    plt.subplot(2, 2, 2)
    y1 = np.mean(S2_S1_1_all)
    y1_err = stats.sem(S2_S1_1_all)
    plt.errorbar(1.12, y1, yerr=y1_err, c='salmon', linewidth=2, marker='o', mfc='salmon', mec='salmon',
                 ms=7, mew=0, zorder=100)
    y2 = np.mean(S2_S1_2_all)
    y2_err = stats.sem(S2_S1_2_all)
    plt.errorbar(2.12, y2, yerr=y2_err, c='salmon', linewidth=2, marker='o', mfc='salmon', mec='salmon',
                 ms=7, mew=0, zorder=100)
    plt.plot([1.12, 2.12], [y1, y2], '-', c='salmon', linewidth=3)
    plt.fill_between([1.12, 2.12], [y1 + y1_err, y2 + y2_err], [y1 - y1_err, y2 - y2_err], alpha=0.2, color='salmon',
                     lw=0)
    plt.xlim(.5, 2.5)
    plt.yticks([-.4, -.2, 0, .2, .4])
    plt.ylim(-0.4, .4)
    plt.xticks([1, 2])
    # plt.ylabel('Stimulus activity\n(normalized deconvolved\n$\mathregular{Ca^{2+}}$transient $\mathregular{s^{-1}}$)')
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
    sns.despine()

    sns.despine()
    plt.savefig(paths['base_path'] + '/NN/plots/reactivation_cue_difference_flip.png', bbox_inches='tight', dpi=200)
    plt.savefig(paths['base_path'] + '/NN/plots/reactivation_cue_difference_flip.pdf', bbox_inches='tight', dpi=200,
                transparent=True)
    plt.close()


def grouped_neurons_dist(mice, sample_dates):

    nc_l_all = []
    nc_h_all = []
    inc_all = []
    dec_all = []
    for mouse in range(0, len(mice)):
        paths = preprocess.create_folders(mice[mouse], sample_dates[mouse])
        vec = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/group_neurons_dist.npy', allow_pickle=True)
        for i in range(0, len(vec[0])):
            inc_all.append(vec[0][i])
            dec_all.append(vec[2][i])
            nc_l_all.append(vec[4][i])
            nc_h_all.append(vec[6][i])
            inc_all.append(vec[1][i])
            dec_all.append(vec[3][i])
            nc_l_all.append(vec[5][i])
            nc_h_all.append(vec[7][i])
    print(np.mean(dec_all))
    print(stats.sem(dec_all))
    print(np.min(dec_all))
    print(np.max(dec_all))

































