import plot
import random
import warnings
import classify
import preprocess
import plot_subset
import numpy as np
import pandas as pd
from os import path
warnings.filterwarnings('ignore')


def process(mouse, date, day, days):
    """
    runs reactivation
    :param mouse: mouse
    :param date: date
    :return: all pre-processed data
    """

    # # create folders to save files
    # paths = preprocess.create_folders(mouse, date)
    # # import data for mouse and date as dict
    # session_data = preprocess.load_data(paths)
    # # process and plot behavior
    # behavior = preprocess.process_behavior(session_data, paths)
    # # normalize activity
    # norm_deconvolved = preprocess.normalize_deconvolved([], behavior, paths, 0)
    # # Gaussian filter activity
    # norm_moving_deconvolved_filtered = preprocess.difference_gaussian_filter(norm_deconvolved, 4, behavior, paths, 0)
    #
    # subset_amounts = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    #
    # for sa in subset_amounts:
    #
    #     y_pred_subset_all = []
    #
    #     for i in range(0, 10):
    #         # random subset of cells
    #         rand_cells = random.sample(range(0, len(norm_deconvolved)), int(len(norm_deconvolved)*(sa/10)))
    #         norm_deconvolved_subset = norm_deconvolved.loc[rand_cells, :]
    #         norm_deconvolved_subset = pd.DataFrame(np.array(norm_deconvolved_subset))
    #         norm_moving_deconvolved_filtered_subset = norm_moving_deconvolved_filtered[rand_cells, :]
    #         # make trial averaged traces and baseline subtract
    #         mean_cs_1_responses_df = preprocess.normalized_trial_averaged(norm_deconvolved_subset, behavior, 'cs_1')
    #         mean_cs_2_responses_df = preprocess.normalized_trial_averaged(norm_deconvolved_subset, behavior, 'cs_2')
    #         # get sig cells
    #         [cs_1_poscells, cs_1_negcells] = preprocess.sig_test(norm_deconvolved_subset, behavior, 'cs_1')
    #         [cs_2_poscells, cs_2_negcells] = preprocess.sig_test(norm_deconvolved_subset, behavior, 'cs_2')
    #         [both_poscells, both_sigcells] = preprocess.combine_sig(cs_1_poscells, cs_1_negcells, cs_2_poscells,
    #                                                                 cs_2_negcells)
    #         # get idx of top cell differences
    #         idx = preprocess.get_index(behavior, mean_cs_1_responses_df, mean_cs_2_responses_df, cs_1_poscells,
    #                                    cs_2_poscells, both_poscells, both_sigcells, paths, 2)
    #         # get prior for synchronous cue activity
    #         prior = classify.prior(norm_moving_deconvolved_filtered_subset, idx['cs_1'], idx['cs_2'], behavior, [])
    #         # logistic regression
    #         y_pred = classify.log_regression(behavior, norm_deconvolved_subset, norm_moving_deconvolved_filtered_subset,
    #                                          both_poscells, prior)
    #         # process classified output
    #         y_pred = classify.process_classified(y_pred, prior, paths, 2)
    #
    #         y_pred_subset_all.append(y_pred)
    #
    #     days_path = paths['base_path'] + paths['mouse'] + '/data_across_days/'
    #     if days:
    #         if path.isfile(days_path + 'y_pred_subset_' + str(sa) + '.npy') == 0 or day == 0:
    #             y_pred_across_days = [list(range(0, days))]
    #             y_pred_across_days[0][day] = y_pred_subset_all
    #             np.save(days_path + 'y_pred_subset_' + str(sa), y_pred_across_days)
    #         else:
    #             y_pred_across_days = np.load(days_path + 'y_pred_subset_' + str(sa) + '.npy', allow_pickle=True)
    #             y_pred_across_days[0][day] = y_pred_subset_all
    #             np.save(days_path + 'y_pred_subset_' + str(sa), y_pred_across_days)

    # paths = preprocess.create_folders(mouse, date)
    # session_data = preprocess.load_data(paths)
    # behavior = preprocess.process_behavior(session_data, paths)
    # y_pred_original = classify.process_classified([], [], paths, 0)
    # times_considered = preprocess.get_times_considered(y_pred_original, behavior)
    # reactivation_cs_1 = y_pred_original[:, 0].copy()
    # reactivation_cs_2 = y_pred_original[:, 1].copy()
    # p_threshold = .75
    # cs_1_peak = 0
    # cs_2_peak = 0
    # i = 0
    # reactivation_original_frames = np.zeros(len(reactivation_cs_1))
    # next_r = 0
    # while i < len(reactivation_cs_1) - 1:
    #     i += 1
    #     if reactivation_cs_1[i] > 0 or reactivation_cs_2[i] > 0:
    #         if next_r == 0:
    #             r_start = i
    #             next_r = 1
    #         if reactivation_cs_1[i] > cs_1_peak:
    #             cs_1_peak = reactivation_cs_1[i]
    #         if reactivation_cs_2[i] > cs_2_peak:
    #             cs_2_peak = reactivation_cs_2[i]
    #         if reactivation_cs_1[i + 1] == 0 and reactivation_cs_2[i + 1] == 0:
    #             r_end = i + 1
    #             next_r = 0
    #             if cs_1_peak > p_threshold:
    #                 reactivation_original_frames[r_start:r_end] = 1
    #             if cs_2_peak > p_threshold:
    #                 reactivation_original_frames[r_start:r_end] = 1
    #             i = r_end
    #             cs_1_peak = 0
    #             cs_2_peak = 0
    # y_pred_original[reactivation_original_frames == 0, 0] = 0
    # y_pred_original[reactivation_original_frames == 0, 1] = 0
    # days_path = paths['base_path'] + paths['mouse'] + '/data_across_days/'
    # subset_amounts = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    # for sa in subset_amounts:
    #     y_pred_across_days = np.load(days_path + 'y_pred_subset_' + str(sa) + '.npy', allow_pickle=True)
    #     y_pred_all = y_pred_across_days[0][day]
    #     sum_reactivation_original_all = []
    #     false_positive_all = []
    #     false_negative_all = []
    #     for i in range(0, 10):
    #         y_pred = y_pred_all[i]
    #         # reactivation comparison to real
    #         [sum_original, false_positive, false_negative] = plot_subset.reactivation_difference(y_pred_original,
    #                                                                                              y_pred, behavior,
    #                                                                                              times_considered)
    #         sum_reactivation_original_all.append(sum_original)
    #         false_positive_all.append(false_positive)
    #         false_negative_all.append(false_negative)
    #     # reactivation comparison to real
    #     sum_reactivation_original_all = np.mean(sum_reactivation_original_all)
    #     false_positive_all = np.mean(false_positive_all)
    #     false_negative_all = np.mean(false_negative_all)
    #     plot_subset.save_reactivation_difference(sum_reactivation_original_all, false_positive_all, false_negative_all,
    #                                              sa, paths, day, days)

    paths = preprocess.create_folders(mouse, date)
    session_data = preprocess.load_data(paths)
    behavior = preprocess.process_behavior(session_data, paths)
    days_path = paths['base_path'] + paths['mouse'] + '/data_across_days/'
    subset_amounts = [1]
    for sa in subset_amounts:
        y_pred_across_days = np.load(days_path + 'y_pred_subset_' + str(sa) + '.npy', allow_pickle=True)
        y_pred_all = y_pred_across_days[0][day]
        y_pred_binned_all = []
        y_pred_binned_bias_all = []
        for i in range(0, 10):
            y_pred = y_pred_all[i]
            y_pred = plot.true_reactivations(y_pred)
            # reactivation rate day
            y_pred_binned = plot_subset.reactivation_rate(y_pred, behavior)
            y_pred_binned_all.append(y_pred_binned)
            # reactivation bias day
            y_pred_binned_bias = plot_subset.reactivation_bias(y_pred, behavior)
            y_pred_binned_bias_all.append(y_pred_binned_bias)
        # reactivation rate day
        y_pred_binned_all = np.mean(y_pred_binned_all, axis=0)
        plot_subset.save_reactivation_rate(y_pred_binned_all, sa, paths, day)
        # reactivation bias day
        y_pred_binned_bias_all = np.nanmean(y_pred_binned_bias_all, axis=0)
        plot_subset.save_reactivation_bias(y_pred_binned_bias_all, sa, paths, day, days)

    # idx_original = preprocess.get_index(behavior, [], [], [], [], [], [], paths, 0)
    # plot_subset.reactivation_raster(behavior, norm_deconvolved, y_pred, y_pred_original, idx_original['cs_1_df'], idx_original['cs_2_df'], idx_original['both'], paths, '2')









