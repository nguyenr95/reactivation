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
    # get idx of top cell differences
    idx = preprocess_opto.get_index(behavior, [], [], [], [], [], [], paths, 0)
    # process classified output
    y_pred = classify_opto.process_classified([], [], paths, 0)

    # reactivation rate
    plot_opto.reactivation_rate(y_pred, behavior, paths, day, days)
    # bias over time in session
    plot_opto.reactivation_bias(y_pred, behavior, paths, day, days)
    # rate within trial
    plot_opto.rate_within_trial(y_pred, behavior, paths, day, days)
    # bias within trial
    plot_opto.bias_within_trial(y_pred, behavior, paths, day, days)
    # # physical evoked reactivations
    # plot_opto.reactivation_physical(y_pred, behavior, paths, day, days)
    # # pupil control
    # plot_opto.pupil_control(behavior, paths, day, days)
    # # activity control during iti
    # plot_opto.activity_control(norm_deconvolved, behavior, idx, paths, day, days)









