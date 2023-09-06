import warnings
import classify_opto
import preprocess_opto
warnings.filterwarnings('ignore')


def process(mouse, date, day, days):
    """
    runs reactivation
    :param mouse: mouse
    :param date: date
    :return: all pre-processed data
    """

    # create folders to save files
    paths = preprocess_opto.create_folders(mouse, date)
    # process and plot behavior
    behavior = preprocess_opto.process_behavior(paths)
    # save masks so can run in matlab to process other planes
    preprocess_opto.cell_masks(paths, 0)

    # grab activity
    deconvolved = preprocess_opto.process_activity(paths, 'spks', 3, 0)
    # normalize activity
    norm_deconvolved = preprocess_opto.normalize_deconvolved(deconvolved, behavior, paths, 0)
    # Gaussian filter activity
    norm_moving_deconvolved_filtered = preprocess_opto.difference_gaussian_filter(norm_deconvolved, 4, behavior, paths,
                                                                                  0)
    # make trial averaged traces and baseline subtract
    mean_cs_1_responses_df = preprocess_opto.normalized_trial_averaged(norm_deconvolved, behavior, 'cs_1')
    mean_cs_2_responses_df = preprocess_opto.normalized_trial_averaged(norm_deconvolved, behavior, 'cs_2')
    # get sig cells
    [cs_1_poscells, cs_1_negcells] = preprocess_opto.sig_test(norm_deconvolved, behavior, 'cs_1')
    [cs_2_poscells, cs_2_negcells] = preprocess_opto.sig_test(norm_deconvolved, behavior, 'cs_2')
    [both_poscells, both_sigcells] = preprocess_opto.combine_sig(cs_1_poscells, cs_1_negcells, cs_2_poscells,
                                                                 cs_2_negcells)
    # get idx of top cell differences
    idx = preprocess_opto.get_index(behavior, mean_cs_1_responses_df, mean_cs_2_responses_df, cs_1_poscells,
                                    cs_2_poscells, both_poscells, both_sigcells, paths, 0)

    # get prior for synchronous cue activity
    prior = classify_opto.prior(norm_moving_deconvolved_filtered, idx['cs_1'], idx['cs_2'], behavior, [])
    # logistic regression
    y_pred = classify_opto.log_regression(behavior, norm_deconvolved, norm_moving_deconvolved_filtered, both_poscells,
                                          prior)
    # process classified output
    y_pred = classify_opto.process_classified(y_pred, prior, paths, 1)

    preprocess_opto.sig_reactivated_cells(norm_deconvolved, norm_moving_deconvolved_filtered, idx, y_pred,
                                          behavior, paths, 1)
    # # p distribution shuffle
    # classify_opto.p_distribution_shuffle(norm_moving_deconvolved_filtered, norm_deconvolved, behavior, idx,
    #                                      both_poscells, paths, day, days)
