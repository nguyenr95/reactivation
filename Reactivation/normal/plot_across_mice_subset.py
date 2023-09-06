from scipy import stats
import random
import preprocess
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings('ignore')


def reactivation_bias_day(mice, sample_dates):
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(9, 6))
    sa = 1
    paths = preprocess.create_folders(mice[0], sample_dates[0])
    session_data = preprocess.load_data(paths)
    task_runs = int(session_data['task_runs'])
    x_label = np.zeros((task_runs * 2))
    hours_per_run = 64000 / 31.25 / 60 / 60
    for i in range(0, (task_runs * 2)):
        x_label[i] = hours_per_run / 4 + (hours_per_run / 2 * i)
    plt.subplot(2, 2, 1)
    paths = preprocess.create_folders(mice[0], sample_dates[0])
    bias_across_days = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/y_pred_bias_binned_subset_' + str(sa) + '.npy',
                               allow_pickle=True)
    bias_all = np.zeros((len(mice), len(bias_across_days[0][0])))
    bias_all_original = np.zeros((len(mice), len(bias_across_days[0][0])))
    for mouse in range(0, len(mice)):
        paths = preprocess.create_folders(mice[mouse], sample_dates[mouse])
        bias_across_days = np.load(
            paths['base_path'] + paths['mouse'] + '/data_across_days/y_pred_bias_binned_subset_' + str(sa) + '.npy',
            allow_pickle=True)
        bias_across_days_original = np.load(
            paths['base_path'] + paths['mouse'] + '/data_across_days/y_pred_bias_binned.npy', allow_pickle=True)
        bias = np.zeros((len(bias_across_days[0]), len(bias_across_days[0][0])))
        bias_original = np.zeros((len(bias_across_days[0]), len(bias_across_days[0][0])))
        for i in range(0, len(bias_across_days[0])):
            bias[i, :] = bias_across_days[0][i]
            bias_original[i, :] = bias_across_days_original[0][i]
        bias_all[mouse, :] = np.nanmean(bias, axis=0)
        bias_all_original[mouse, :] = np.nanmean(bias_original, axis=0)

    print(stats.ttest_rel(np.mean(bias_all_original[:, 0:8], axis=1), np.mean(bias_all[:, 0:8], axis=1)))

    mean = bias_all_original.mean(axis=0)
    print(mean[0])
    sem_plus = mean + stats.sem(bias_all_original, axis=0)
    sem_minus = mean - stats.sem(bias_all_original, axis=0)
    plt.plot(x_label, mean[0:len(mean)], '-k', linewidth=3)
    plt.fill_between(x_label, sem_plus[0:len(sem_plus)], sem_minus[0:len(sem_minus)], alpha=0.2, color='k', lw=0)
    mean = bias_all.mean(axis=0)
    print(mean[0])

    sem_plus = mean + stats.sem(bias_all, axis=0)
    sem_minus = mean - stats.sem(bias_all, axis=0)
    plt.plot(x_label, mean[0:len(mean)], '-', c=[.6, 0, .6], linewidth=3)
    plt.fill_between(x_label, sem_plus[0:len(sem_plus)], sem_minus[0:len(sem_minus)], alpha=0.2, color=[.6, 0, .6],
                     lw=0)
    plt.ylabel('Bias in reactivation rate\n toward the previous stimulus')
    plt.xlabel('Time relative to stimulus onset (h)')
    plt.xlim(0, x_label[len(x_label)-1] + hours_per_run / 4)
    plt.ylim(-1, 1)
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1, snap=False)
    label_1 = mlines.Line2D([], [], color='k', linestyle='-', label='All neurons', linewidth=3)
    label_2 = mlines.Line2D([], [], color=[.6, 0, .6], linestyle='-', label='Random 10% neurons', linewidth=3)
    plt.legend(handles=[label_1, label_2], frameon=False, prop={'size': 8}, labelspacing=.1)
    sns.despine()
    plt.savefig(paths['base_path'] + '/NN/plots/reactivation_bias_day_subset.png', bbox_inches='tight', dpi=200)
    plt.savefig(paths['base_path'] + '/NN/plots/reactivation_bias_day_subset.pdf', bbox_inches='tight', dpi=200, transparent=True)
    plt.close()


def reactivation_difference(mice, sample_dates):
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks")
    plt.figure(figsize=(6, 4.5))
    m_colors = ['b', 'teal', 'slategray', 'green', 'darkolivegreen', 'darkorange', 'purple', 'darkred']
    x = [1, 3, 5, 7, 9]
    subset_amounts = [9, 7, 5, 3, 1]
    false_positive_all = np.zeros((len(mice), len(subset_amounts)))
    false_negative_all = np.zeros((len(mice), len(subset_amounts)))
    idx = 0
    for sa in subset_amounts:
        for mouse in range(0, len(mice)):
            paths = preprocess.create_folders(mice[mouse], sample_dates[mouse])
            reactivation_difference_all = np.load(paths['base_path'] + paths['mouse'] + '/data_across_days/reactivation_difference_subset_continuous_' + str(sa) + '.npy', allow_pickle=True)
            false_positive = []
            false_negative = []
            for i in range(0, len(reactivation_difference_all)):
                false_positive.append(reactivation_difference_all[1][i]/reactivation_difference_all[0][i])
                false_negative.append(reactivation_difference_all[2][i]/reactivation_difference_all[0][i])
            false_positive_all[mouse, idx] = np.mean(false_positive) * 100
            false_negative_all[mouse, idx] = np.mean(false_negative) * 100
        idx += 1
    for mouse in range(0, len(mice)):
        plt.subplot(2, 2, 1)
        plt.errorbar(x, false_negative_all[mouse, :], yerr=0, c=m_colors[mouse], linewidth=2, linestyle='-', zorder=0,
                     alpha=.2)
        plt.subplot(2, 2, 2)
        plt.errorbar(x, false_positive_all[mouse, :], yerr=0, c=m_colors[mouse], linewidth=2, linestyle='-', zorder=0,
                     alpha=.2)

    [_, s_p_value] = stats.shapiro(false_positive_all[:, 0])
    print(s_p_value)

    anova_results = []
    for i in range(1, len(false_positive_all[0])):
        anova_results.append(stats.ttest_rel(false_positive_all[:, 0], false_positive_all[:, i])[1])
    anova_results_corrected = multipletests(anova_results, alpha=.05, method='holm')
    print([anova_results, anova_results_corrected[1]])

    anova_results = []
    for i in range(1, len(false_negative_all[0])):
        anova_results.append(stats.ttest_rel(false_negative_all[:, 0], false_negative_all[:, i])[1])
    anova_results_corrected = multipletests(anova_results, alpha=.05, method='holm')
    print([anova_results, anova_results_corrected[1]])
    ggg
    y0 = np.mean(false_negative_all, axis=0)
    y1 = np.mean(false_positive_all, axis=0)
    y0_err = stats.sem(false_negative_all, axis=0)
    y1_err = stats.sem(false_positive_all, axis=0)
    plt.subplot(2, 2, 1)
    plt.plot(x, y0, c='k', linewidth=3, linestyle='-', zorder=0)
    plt.fill_between(x, y0 + y0_err, y0-y0_err, alpha=0.2, color='k', lw=0)
    plt.ylim(0, 100)
    plt.ylabel('Percent of false\nnegative reactivations')
    plt.xlabel('Percent of neurons\nused in classifier')
    plt.xticks([1, 3, 5, 7, 9], ['90', '70', '50', '30', '10'])
    plt.xlim(.5, 9.5)
    plt.subplot(2, 2, 2)
    plt.plot(x, y1, c='k', linewidth=3, linestyle='-', zorder=0)
    plt.fill_between(x, y1 + y1_err, y1 - y1_err, alpha=0.2, color='k', lw=0)
    plt.ylim(0, 140)
    plt.yticks([0, 20, 40, 60, 80, 100, 120, 140])
    plt.ylabel('Percent of false\npositive reactivations')
    plt.xticks([1, 3, 5, 7, 9], ['90', '70', '50', '30', '10'])
    plt.xlabel('Percent of neurons\nused in classifier')
    plt.xlim(.5, 9.5)
    sns.despine()
    plt.savefig(paths['base_path'] + '/NN/plots/reactivation_difference_subset.png', bbox_inches='tight', dpi=200)
    plt.savefig(paths['base_path'] + '/NN/plots/reactivation_difference_subset.pdf', bbox_inches='tight', dpi=200, transparent=True)
    plt.close()










































