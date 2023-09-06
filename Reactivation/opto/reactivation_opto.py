import warnings
import plot_opto
import classify_opto
import preprocess_opto
warnings.filterwarnings('ignore')


def process(mouse, date, day, days):
    """
    runs reactivation opto plots
    :param mouse: mouse
    :param date: date
    :param day: day order
    :param days: total number of days
    :return: all plots
    """
    # create folders to save files
    paths = preprocess_opto.create_folders(mouse, date)
    # process and plot behavior
    behavior = preprocess_opto.process_behavior(paths)
    # normalize activity
    norm_deconvolved = preprocess_opto.normalize_deconvolved([], behavior, paths, 0)
    # # Gaussian filter activity
    # norm_moving_deconvolved_filtered = preprocess_opto.difference_gaussian_filter(norm_deconvolved, 4, behavior, paths,
    #                                                                               0)
    # # get sig cells
    # [cs_1_poscells, cs_1_negcells] = preprocess_opto.sig_test(norm_deconvolved, behavior, 'cs_1')
    # [cs_2_poscells, cs_2_negcells] = preprocess_opto.sig_test(norm_deconvolved, behavior, 'cs_2')
    # [both_poscells, both_sigcells] = preprocess_opto.combine_sig(cs_1_poscells, cs_1_negcells, cs_2_poscells,
    #                                                              cs_2_negcells)
    # get idx of top cell differences
    idx = preprocess_opto.get_index(behavior, [], [], [], [], [], [], paths, 0)
    # process classified output
    y_pred = classify_opto.process_classified([], [], paths, 0)
    # # upper and lower layer cells
    # [upper_layer_cells, lower_layer_cells] = preprocess_opto.process_plane_activity_R1(paths, 'spks', 3)
    # # process cells across days
    # preprocess_opto.process_activity_across_days_R1(paths, 3, 0, mouse, day)
    # # process sig cells aligned across days
    # preprocess_opto.grab_align_cells(paths, idx, day, days)
    # # align cells across days
    # preprocess_opto.align_cells(paths)


    # # plot neuron count
    # plot_opto.neuron_count_R2(norm_deconvolved, behavior, idx, paths, day, days)
    # # plot neuron dist
    # plot_opto.neuron_dist_R3(norm_deconvolved, behavior, idx, paths, day, days)
    # # plot reactivation cue layer
    # plot_opto.reactivation_layer_R1(norm_deconvolved, upper_layer_cells, lower_layer_cells, y_pred, behavior, idx, paths,
    #                                 day, days)
    # # activity of top vs other cells during reactivation
    # plot_opto.reactivation_top_bottom_activity(norm_deconvolved, idx, y_pred, behavior, both_poscells, paths, day, days)
    # # classify iti as well
    # prior = classify_opto.prior(norm_moving_deconvolved_filtered, idx['cs_1'], idx['cs_2'], behavior, [])
    # y_pred = classify_opto.log_regression_R2(behavior, norm_deconvolved, both_poscells, prior)
    # classify_opto.process_classified_R2(norm_moving_deconvolved_filtered, idx['cs_1'], idx['cs_2'], behavior, y_pred, paths,
    #                                day, days)
    # # plot top activity stable
    # plot_opto.top_activity_stable_R3(norm_deconvolved, idx, behavior, paths, day, days)
    # # reactivation duration pre vs post
    # plot_opto.reactivation_duration(y_pred, behavior, paths, day, days)
    # # trial history modulation
    # plot_opto.trial_history(norm_deconvolved, idx, y_pred, behavior, paths, 1, day, days)
    # # iti activity across session
    # plot_opto.iti_activity_across_trials(norm_deconvolved, y_pred, idx, behavior, paths, day)
    # # iti activity within trial
    # plot_opto.iti_activity_within_trial(norm_deconvolved, idx, behavior, paths, day, days)
    # # pupil across session
    # plot_opto.pupil_across_trials(y_pred, behavior, paths, day)
    # # pupil within trial
    # plot_opto.pupil_within_trial(behavior, paths, day, days)
    # # plot pupil and brain motion during stimulus period
    # plot_opto.behavior_across_trials_R2(behavior, paths, day, days)
    # # correlation across days
    # plot_opto.activity_across_trials_across_days_R1(norm_deconvolved, behavior, y_pred, paths, day, days)
    # # plot drift reactivation
    # plot_opto.reactivation_spatial_drift_R3(norm_deconvolved, y_pred, behavior, 3, idx, paths, day, days)
    # # plot activity across trials layer
    # plot_opto.activity_across_trials_layer_R1(norm_deconvolved, upper_layer_cells, lower_layer_cells, idx, y_pred, behavior,
    #                                           paths, day, days)
    # # plot activity across trials grouped omit
    # plot_opto.activity_across_trials_grouped_omit(norm_deconvolved, behavior, idx, paths, day, days)
    # # plot baseline activity of groups
    # plot_opto.activity_across_trials_grouped_baseline_R3(norm_deconvolved, behavior, idx, paths, day, days)
    # # plot grouped count layer
    # plot_opto.neuron_count_grouped_layer_R3(norm_deconvolved, behavior, upper_layer_cells, lower_layer_cells, idx, paths,
    #                                    day, days)
    # # reactivation grouped location
    # plot_opto.reactivation_spatial_grouped_R3(norm_deconvolved, behavior, 3, idx, paths, day, days)
    # # noise correlation grouped
    # plot_opto.noise_correlation_grouped(norm_deconvolved, behavior, idx, paths, day, days)
    # novelty decrease cells separate and count
    plot_opto.activity_across_trials_grouped_decrease_novelty_R1(norm_deconvolved, behavior, idx, paths, day, days)
    # activity across trials no novelty
    plot_opto.activity_across_trials_novelty_R1(norm_deconvolved, behavior, y_pred, idx, paths, day, days)
    # # plot reactivation cue vector upper vs lower
    # plot_opto.reactivation_cue_vector_layer_R1(norm_deconvolved, upper_layer_cells, lower_layer_cells, idx, y_pred, behavior,
    #                                       paths, day, days)
    # # plot pupil/activity modulation of reactivation
    # plot_opto.pupil_activity_reactivation_modulation(norm_deconvolved, behavior, y_pred, idx, paths, day, days)

    # # reactivation rate
    # plot_opto.reactivation_rate(y_pred, behavior, paths, day, days)
    # # bias over time in session
    # plot_opto.reactivation_bias(y_pred, behavior, paths, day, days)
    # # rate within trial
    # plot_opto.rate_within_trial(y_pred, behavior, paths, day, days)
    # # bias within trial
    # plot_opto.bias_within_trial(y_pred, behavior, paths, day, days)
    # # physical evoked reactivations
    # plot_opto.reactivation_physical(y_pred, behavior, paths, day, days)
    # # pupil control
    # plot_opto.pupil_control(behavior, paths, day, days)
    # # activity control during iti
    # plot_opto.activity_control(norm_deconvolved, behavior, idx, paths, day, days)
    # # plot rate within trial long
    # plot_opto.rate_within_trial_controlopto_R1(y_pred, behavior, paths, day, days)
    # # plot bias within trial long
    # plot_opto.bias_within_trial_controlopto_R1(y_pred, behavior, paths, day, days)
    # # plot pupil matched baseline
    # plot_opto.reactivation_rate_pupil_control_baseline_R1(y_pred, behavior, paths, day, days)
    # # activity across trials
    # plot_opto.activity_across_trials(norm_deconvolved, behavior, y_pred, idx, paths, day, days)
    # # activity across trials grouped
    # plot_opto.activity_across_trials_grouped(norm_deconvolved, behavior, idx, paths, day, days)
    # # activity across trials grouped decrease
    # plot_opto.activity_across_trials_grouped_separate(norm_deconvolved, behavior, idx, paths, day, days)
    # # plot cue vector
    # plot_opto.reactivation_cue_vector(norm_deconvolved, idx, y_pred, behavior, paths, day, days)
    # # plot diff cue reactivation
    # plot_opto.reactivation_cue_difference(norm_deconvolved, behavior, y_pred, idx, paths, day, days)
    # # plot activity pattern cue vs reactivation
    # plot_opto.reactivation_cue_pattern_difference(norm_deconvolved, behavior, y_pred, idx, paths, day, days)
    # # plot cue vector evolve
    # plot_opto.reactivation_cue_vector_evolve(norm_deconvolved, idx, y_pred, behavior, 0, [], paths, day, days)
    # # activity across trials evolve
    # plot_opto.activity_across_trials_evolve(norm_deconvolved, behavior, y_pred, idx, paths, day, days)
    # # activity across trials evolve
    # plot_opto.activity_across_trials_evolve_low_reactivation(norm_deconvolved, behavior, y_pred, idx, paths, day, days)
    # # plot activity across trials evolve grouped
    # plot_opto.activity_across_trials_evolve_grouped(norm_deconvolved, behavior, y_pred, idx, paths, day, days)
    # # plot activity across trials evolve grouped in/decrease
    # plot_opto.activity_across_trials_evolve_grouped_separate(norm_deconvolved, behavior, y_pred, idx, paths, day, days)
    # # plot spatial region of cue and reactivation
    # plot_opto.reactivation_spatial(norm_deconvolved, y_pred, behavior, 3, idx, paths, day, days)
    # # plot scale between cue and reactivation
    # plot_opto.reactivation_cue_scale(norm_deconvolved, idx, y_pred, behavior, paths, day, days)
    # vector projection without novelty
    plot_opto.reactivation_cue_vector_novelty_R1(norm_deconvolved, idx, y_pred, behavior, paths, day, days)
    # # plot reactivation cue vector evolve parametric
    # plot_opto.reactivation_cue_vector_evolve_parametric(norm_deconvolved, idx, y_pred, behavior, paths, day, days)
    # # plot prior for other synchronous events
    # plot_opto.prior_R1(norm_deconvolved, norm_moving_deconvolved_filtered, behavior, y_pred, idx, paths)
    # # plot num selective grouped
    # plot_opto.num_selective_grouped(norm_deconvolved, behavior, idx, paths, day, days)
    # # plot reactivation difference tuned flip
    # plot_opto.reactivation_difference_tunedflip_R2(norm_deconvolved, behavior, y_pred, idx, paths, day, days)
    # plot sd cutoff grouped
    # plot_opto.group_neurons_range(norm_deconvolved, idx, behavior, paths, day, days)
