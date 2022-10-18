import plot
import warnings
import classify
import preprocess
warnings.filterwarnings('ignore')


def process(mouse, date, day, days):
    """
    runs reactivation plotting
    :param mouse: mouse
    :param date: date
    :param day: day order
    :param days: total number of days
    :return: all plots
    """
    # create folders to save files
    paths = preprocess.create_folders(mouse, date)
    # import data for mouse and date as dict
    session_data = preprocess.load_data(paths)
    # process and plot behavior
    behavior = preprocess.process_behavior(session_data, paths)
    # normalize activity
    norm_deconvolved = preprocess.normalize_deconvolved([], behavior, paths, 0)
    # # Gaussian filter activity
    # norm_moving_deconvolved_filtered = preprocess.difference_gaussian_filter(norm_deconvolved, 4, behavior, paths, 0)
    # get idx of top cell differences
    idx = preprocess.get_index(behavior, [], [], [], [], [], [], paths, 0)
    # process classified output
    y_pred = classify.process_classified([], [], paths, 0)
    # # get sig cells
    # [cs_1_poscells, cs_1_negcells] = preprocess.sig_test(norm_deconvolved, behavior, 'cs_1')
    # [cs_2_poscells, cs_2_negcells] = preprocess.sig_test(norm_deconvolved, behavior, 'cs_2')
    # [both_poscells, _] = preprocess.combine_sig(cs_1_poscells, cs_1_negcells, cs_2_poscells, cs_2_negcells)

    # # reactivation rates
    # plot.reactivation_rate(y_pred, behavior, paths, day)
    # # reactivation bias over time
    # plot.reactivation_bias(y_pred, behavior, paths, day, days)
    # # reactivation rate trial
    # plot.rate_within_trial(y_pred, behavior, paths, day, days)
    # # reactivation bias trial
    # plot.bias_within_trial(y_pred, behavior, paths, day, days)
    # # iti activity across session
    # plot.iti_activity_across_trials(norm_deconvolved, y_pred, idx, behavior, paths, day)
    # # iti activity within trial
    # plot.iti_activity_within_trial(norm_deconvolved, y_pred, idx, behavior, paths, day, days)
    # # pupil across session
    # plot.pupil_across_trials(y_pred, behavior, paths, day)
    # # pupil wihtin trial
    # plot.pupil_within_trial(y_pred, behavior, paths, day, days)
    # # trial history modulation
    # plot.trial_history(norm_deconvolved, idx, y_pred, behavior, paths, 1, day, days)
    # # pupil reactivation rate modulation
    # plot.pupil_reactivation_modulation(behavior, y_pred, paths, day, days)
    # # physical evoked reactivations
    # plot.reactivation_physical(y_pred, behavior, paths, day, days)
    # # activity change with reactivation rates over time
    # plot.activity_across_trials(norm_deconvolved, behavior, y_pred, idx, paths, day, days)
    # # activity change with reactivation rates over time grouped
    # plot.activity_across_trials_grouped(norm_deconvolved, behavior, idx, paths, day, days)
    # # plot activity across trials grouped for increase or decrease cells
    # plot.activity_across_trials_grouped_decrease(norm_deconvolved, behavior, idx, paths, day, days)
    # # plot reactivation cue activity
    # plot.cue_reactivation_activity(norm_deconvolved, behavior, y_pred, idx, paths, day, days)
    # # iti activity across session
    # plot.iti_activity_across_trials(norm_deconvolved, y_pred, idx, behavior, paths, day)
    # # iti activity bias
    # plot.iti_activity_bias(norm_deconvolved, norm_moving_deconvolved_filtered, behavior, y_pred, idx, paths, day, days)
    # # reactivation duration pre vs post
    # plot.reactivation_duration(y_pred, behavior, paths, day, days)
    # # activity of top vs other cells during reactivation
    # plot.reactivation_top_bottom_activity(norm_deconvolved, idx, y_pred, behavior, both_poscells, paths, day, days)
    # # correlation between cue and reactivation
    # plot.reactivation_cue_correlation(norm_deconvolved, idx, y_pred, behavior, paths, day, days)
    # # correlation between cue and reactivation shuffled
    # plot.reactivation_cue_correlation_shuffle(norm_deconvolved, idx, y_pred, behavior, paths, day, days)
    # # correlation between cue and cue
    # plot.cue_cue_correlation(norm_deconvolved, idx, y_pred, behavior, paths, day, days)
    # # correlation between cue and cue shuffled
    # plot.cue_cue_correlation_shuffle(norm_deconvolved, idx, y_pred, behavior, paths, day, days)
    # # plot spatial region of cue and reactivation
    # plot.reactivation_spatial(norm_deconvolved, y_pred, behavior, 3, idx, paths, day, days)
    # # plot spatial region of cue and reactivation percent
    # plot.reactivation_spatial_percent(norm_deconvolved, y_pred, behavior, 3, idx, paths, day, days)
    # # plot reactivation cue vector
    # plot.reactivation_cue_vector(norm_deconvolved, idx, y_pred, behavior, paths, day, days)
    # # plot reactivation cue vector evolve
    # plot.reactivation_cue_vector_evolve(norm_deconvolved, idx, y_pred, behavior, 0, [], paths, day, days)
    # # activity across trials evolve
    # plot.activity_across_trials_evolve(norm_deconvolved, behavior, y_pred, idx, paths, day, days)
    # # plot activity across trials evolve grouped
    # plot.activity_across_trials_evolve_grouped(norm_deconvolved, behavior, y_pred, idx, paths, day, days)
    # # plot activity across trials evolve grouped in/decrease
    # plot.activity_across_trials_evolve_grouped_decrease(norm_deconvolved, behavior, y_pred, idx, paths, day, days)
    # # plot reactivation cue vector evolve parametric
    # plot.reactivation_cue_vector_evolve_parametric(norm_deconvolved, idx, y_pred, behavior, paths, day, days)
    # # plot scale between cue and reactivation
    # plot.reactivation_cue_scale(norm_deconvolved, idx, y_pred, behavior, paths, day, days)
    # # plot reactivation influence
    # plot.reactivation_influence(norm_deconvolved, behavior, y_pred, idx, paths, day, days)
    # # plot number of grouped neurons
    # plot.grouped_count(norm_deconvolved, behavior, idx, paths, day, days)
    # # plot activity difference grouped
    # plot.activity_difference_grouped(norm_deconvolved, behavior, y_pred, idx, paths, day, days)
