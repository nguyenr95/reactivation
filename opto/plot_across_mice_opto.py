import warnings
import matplotlib
import preprocess_opto
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

    print(stats.ttest_rel(np.mean(binned_reactivation_all[:, 1:9], axis=1),
                          np.mean(binned_reactivation_opto_all[:, 1:9], axis=1)))

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

    print(stats.ttest_rel(np.mean(binned_reactivation_all, axis=1), np.mean(binned_reactivation_opto_all, axis=1)))

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

    print(stats.ttest_rel(np.mean(mean_norm_mice, axis=1), np.mean(mean_opto_mice, axis=1)))

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

    print(stats.ttest_rel(np.mean(mean_norm_mice, axis=1), np.mean(mean_opto_mice, axis=1)))

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
































