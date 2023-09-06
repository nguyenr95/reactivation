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
    # # get sig cells
    # [cs_1_poscells, cs_1_negcells] = preprocess.sig_test(norm_deconvolved, behavior, 'cs_1')
    # [cs_2_poscells, cs_2_negcells] = preprocess.sig_test(norm_deconvolved, behavior, 'cs_2')
    # [both_poscells, _] = preprocess.combine_sig(cs_1_poscells, cs_1_negcells, cs_2_poscells, cs_2_negcells)
    # process classified output
    y_pred = classify.process_classified([], [], paths, 0)
    # # upper and lower layer cells
    # [upper_layer_cells, lower_layer_cells] = preprocess.process_plane_activity_R1(paths, 'spks', 3)
    # # process cells across days
    # preprocess.process_activity_across_days_R1(paths, 3, 0, mouse, day)
    # # process sig cells aligned across days
    # preprocess.grab_align_cells(paths, idx, day, days)
    # # align cells across days
    # preprocess.align_cells(paths)


    # # plot neuron count
    # plot.neuron_count_R2(norm_deconvolved, behavior, idx, paths, day, days)
    # # plot neuron dist
    # plot.neuron_dist_R3(norm_deconvolved, behavior, idx, paths, day, days)
    # # plot reactivation participation upper vs lower
    # plot.reactivation_layer_R1(norm_deconvolved, upper_layer_cells, lower_layer_cells, y_pred, behavior, idx, paths, day,
    #                            days)
    # # classify iti as well
    # prior = classify.prior(norm_moving_deconvolved_filtered, idx['cs_1'], idx['cs_2'], behavior, [])
    # y_pred = classify.log_regression_R2(behavior, norm_deconvolved, both_poscells, prior)
    # classify.process_classified_R2(norm_moving_deconvolved_filtered, idx['cs_1'], idx['cs_2'], behavior, y_pred, paths, day, days)
    # # plot top activity stable
    # plot.top_activity_stable_R3(norm_deconvolved, idx, behavior, paths, day, days)
    # # plot pupil and brain motion during stimulus period
    # plot.behavior_across_trials_R2(behavior, paths, day, days)
    # # correlation across days
    # plot.activity_across_trials_across_days_R1(norm_deconvolved, behavior, y_pred, paths, day, days)
    # # plot drift reactivation
    # plot.reactivation_spatial_drift_R3(norm_deconvolved, y_pred, behavior, 3, idx, paths, day, days)
    # # plot activity across trials layer
    # plot.activity_across_trials_layer_R1(norm_deconvolved, upper_layer_cells, lower_layer_cells, idx, y_pred, behavior,
    #                                 paths, day, days)
    # # plot activity across trials grouped omit
    # plot.activity_across_trials_grouped_omit(norm_deconvolved, behavior, idx, paths, day, days)
    # # plot baseline activity of groups
    # plot.activity_across_trials_grouped_baseline_R3(norm_deconvolved, behavior, idx, paths, day, days)
    # # plot grouped count layer
    # plot.neuron_count_grouped_layer_R3(norm_deconvolved, behavior, upper_layer_cells, lower_layer_cells, idx, paths, day,
    #                               days)
    # # reactivation grouped location
    # plot.reactivation_spatial_grouped_R3(norm_deconvolved, behavior, 3, idx, paths, day, days)
    # # noise correlation grouped
    # plot.noise_correlation_grouped(norm_deconvolved, behavior, idx, paths, day, days)
    # novelty decrease cells separate and count
    plot.activity_across_trials_grouped_decrease_novelty_R1(norm_deconvolved, behavior, idx, paths, day, days)
    # activity across trials no novelty
    plot.activity_across_trials_novelty_R1(norm_deconvolved, behavior, y_pred, idx, paths, day, days)
    # # plot face tracking
    # plot.face_track_R1(mouse, date, paths, day, days)
    # vector projection without novelty
    plot.reactivation_cue_vector_novelty_R1(norm_deconvolved, idx, y_pred, behavior, paths, day, days)
    # # plot reactivation cue vector upper vs lower
    # plot.reactivation_cue_vector_layer_R1(norm_deconvolved, upper_layer_cells, lower_layer_cells, idx, y_pred, behavior,
    #                                  paths, day, days)
    # # vector projection across days
    # plot.reactivation_cue_vector_across_days_R1(norm_deconvolved, mouse, y_pred, behavior, paths, day, days)
    # # plot prior for other synchronous events
    # plot.prior_R1(norm_deconvolved, norm_moving_deconvolved_filtered, behavior, y_pred, idx, paths)
    # # plot pupil/activity modulation of reactivation
    # plot.pupil_activity_reactivation_modulation(norm_deconvolved, behavior, y_pred, idx, paths, day, days)
    # # how many cells across days
    # plot.quantify_num_across_days(paths, day, days)
    # # plot spatial masks aligned across days
    # plot.reactivation_spatial_aligned(y_pred, behavior, 3, idx, paths, day, days)


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
    # plot.iti_activity_within_trial(norm_deconvolved, idx, behavior, paths, day, days)
    # # pupil across session
    # plot.pupil_across_trials(y_pred, behavior, paths, day)
    # # pupil within trial
    # plot.pupil_within_trial(behavior, paths, day, days)
    # # trial history modulation
    # plot.trial_history(norm_deconvolved, idx, y_pred, behavior, paths, 1, day, days)
    # # physical evoked reactivations
    # plot.reactivation_physical(y_pred, behavior, paths, day, days)
    # # activity change with reactivation rates over time
    # plot.activity_across_trials(norm_deconvolved, behavior, y_pred, idx, paths, day, days)
    # # activity change with reactivation rates over time grouped
    # plot.activity_across_trials_grouped(norm_deconvolved, behavior, idx, paths, day, days)
    # # plot activity across trials grouped for increase or decrease cells
    # plot.activity_across_trials_grouped_separate(norm_deconvolved, behavior, idx, paths, day, days)
    # # reactivation duration pre vs post
    # plot.reactivation_duration(y_pred, behavior, paths, day, days)
    # # activity of top vs other cells during reactivation
    # plot.reactivation_top_bottom_activity(norm_deconvolved, idx, y_pred, behavior, both_poscells, paths, day, days)
    # # plot spatial region of cue and reactivation
    # plot.reactivation_spatial(norm_deconvolved, y_pred, behavior, 3, idx, paths, day, days)
    # # plot reactivation cue vector
    # plot.reactivation_cue_vector(norm_deconvolved, idx, y_pred, behavior, paths, day, days)
    # # plot reactivation cue vector evolve
    # plot.reactivation_cue_vector_evolve(norm_deconvolved, idx, y_pred, behavior, 0, [], paths, day, days)
    # # activity across trials evolve
    # plot.activity_across_trials_evolve(norm_deconvolved, behavior, y_pred, idx, paths, day, days)
    # # activity across trials evolve
    # plot.activity_across_trials_evolve_low_reactivation(norm_deconvolved, behavior, y_pred, idx, paths, day, days)
    # # plot activity across trials evolve grouped
    # plot.activity_across_trials_evolve_grouped(norm_deconvolved, behavior, y_pred, idx, paths, day, days)
    # # plot activity across trials evolve grouped in/decrease
    # plot.activity_across_trials_evolve_grouped_separate(norm_deconvolved, behavior, y_pred, idx, paths, day, days)
    # # plot reactivation cue vector evolve parametric
    # plot.reactivation_cue_vector_evolve_parametric(norm_deconvolved, idx, y_pred, behavior, paths, day, days)
    # # plot scale between cue and reactivation
    # plot.reactivation_cue_scale(norm_deconvolved, idx, y_pred, behavior, paths, day, days)
    # # plot reactivation cue difference
    # plot.reactivation_cue_difference(norm_deconvolved, behavior, y_pred, idx, paths, day, days)
    # # plot number of grouped neurons
    # plot.grouped_count(norm_deconvolved, behavior, idx, paths, day, days)
    # # plot activity pattern cue vs reactivation
    # plot.reactivation_cue_pattern_difference(norm_deconvolved, behavior, y_pred, idx, paths, day, days)
    # # plot num selective grouped
    # plot.num_selective_grouped(norm_deconvolved, behavior, idx, paths, day, days)
    # # plot reactivation difference tuned flip
    # # plot.reactivation_difference_tunedflip_R2(norm_deconvolved, behavior, y_pred, idx, paths, day, days)
    # # plot range of sd cutoff groups
    # plot.group_neurons_range(norm_deconvolved, idx, behavior, paths, day, days)
