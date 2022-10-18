import warnings
import classify
import preprocess
warnings.filterwarnings('ignore')


def process(mouse, date, day, days):
    """
    runs reactivation
    :param mouse: mouse
    :param date: date
    :return: all pre-processed data
    """

    # create folders to save files
    paths = preprocess.create_folders(mouse, date)
    # import data for mouse and date as dict
    session_data = preprocess.load_data(paths)
    # process and plot behavior
    behavior = preprocess.process_behavior(session_data, paths)
    # save masks so can run in matlab to process other planes
    preprocess.cell_masks(paths, 0)

    # grab activity
    deconvolved = preprocess.process_activity(paths, 'spks', 3, 0)
    # normalize activity
    norm_deconvolved = preprocess.normalize_deconvolved(deconvolved, behavior, paths, 0)
    # Gaussian filter activity
    norm_moving_deconvolved_filtered = preprocess.difference_gaussian_filter(norm_deconvolved, 4, behavior, paths, 0)
    # make trial averaged traces and baseline subtract
    mean_cs_1_responses_df = preprocess.normalized_trial_averaged(norm_deconvolved, behavior, 'cs_1')
    mean_cs_2_responses_df = preprocess.normalized_trial_averaged(norm_deconvolved, behavior, 'cs_2')
    # get sig cells
    [cs_1_poscells, cs_1_negcells] = preprocess.sig_test(norm_deconvolved, behavior, 'cs_1')
    [cs_2_poscells, cs_2_negcells] = preprocess.sig_test(norm_deconvolved, behavior, 'cs_2')
    [both_poscells, both_sigcells] = preprocess.combine_sig(cs_1_poscells, cs_1_negcells, cs_2_poscells, cs_2_negcells)
    # get idx of top cell differences
    idx = preprocess.get_index(behavior, mean_cs_1_responses_df, mean_cs_2_responses_df, cs_1_poscells, cs_2_poscells,
                               both_poscells, both_sigcells, paths, 0)

    # get prior for synchronous cue activity
    prior = classify.prior(norm_moving_deconvolved_filtered, idx['cs_1'], idx['cs_2'], behavior, [])
    # logistic regression
    y_pred = classify.log_regression(behavior, norm_deconvolved, norm_moving_deconvolved_filtered, both_poscells, prior)
    # process classified output
    y_pred = classify.process_classified(y_pred, prior, paths, 1)

    # # plot p distribution shuffled
    # classify.p_distribution_shuffle(norm_moving_deconvolved_filtered, norm_deconvolved, behavior, idx, both_poscells,
    #                                 paths, day, days)

