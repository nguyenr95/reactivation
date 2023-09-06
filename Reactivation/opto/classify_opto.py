import random
import warnings
import preprocess_opto
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from os import path
from scipy import stats
from sklearn.linear_model import LogisticRegression
warnings.filterwarnings('ignore')


def prior(norm_moving_deconvolved_filtered, cs_1_idx, cs_2_idx, behavior, threshold):
    """
    get cue prior
    :param norm_moving_deconvolved_filtered: processed activity
    :param cs_1_idx: cs 1 index
    :param cs_2_idx: cs 2 index
    :param behavior: behavior
    :param threshold: threshold
    :return: prior
    """
    cs_1_prior = mean_activity_prior(norm_moving_deconvolved_filtered, cs_1_idx, behavior, threshold)
    cs_2_prior = mean_activity_prior(norm_moving_deconvolved_filtered, cs_2_idx, behavior, threshold)
    combined_prior = cs_1_prior + cs_2_prior
    combined_prior[combined_prior > 0] = 1
    return combined_prior


def mean_activity_prior(activity, cs_idx, behavior, threshold):
    """
    prior
    :param activity: activity
    :param cs_idx: index of cells
    :param behavior: behavior
    :param threshold: threshold
    :return: prior
    """
    if not threshold:
        threshold = 5
    activity[activity < 0] = 0
    activity = pd.DataFrame(activity)
    mean_vec_filtered_cs = activity.reindex(cs_idx.index[0:int(len(cs_idx) / 20)]).mean()
    mean_vec_filtered_cs = preprocess_opto.filter_cues(behavior, mean_vec_filtered_cs)
    mean_vec_filtered_cs = preprocess_opto.filter_opto(behavior, mean_vec_filtered_cs)
    mean_vec_filtered_cs[behavior['relevant_times'] == 0] = float("nan")
    mean_vec_filtered_cs = filter_classified(behavior, mean_vec_filtered_cs, float("nan"))
    mean_vec_filtered_cs = (mean_vec_filtered_cs - mean_vec_filtered_cs.mean()) / mean_vec_filtered_cs.std()
    mean_vec_filtered_cs[mean_vec_filtered_cs < threshold] = 0
    mean_vec_filtered_cs[mean_vec_filtered_cs != mean_vec_filtered_cs] = 0
    mean_vec_filtered_cs[mean_vec_filtered_cs > 0] = 1
    return mean_vec_filtered_cs


def log_regression(behavior, train_fluorescence, test_fluorescence, idx, cue_prior):
    """
    classify
    :param behavior: dict of behavior
    :param train_fluorescence: normalized task
    :param test_fluorescence: normalized dark
    :param idx: index of cells
    :param cue_prior: prior
    :return: y pred
    """
    cue_type = np.zeros(int(behavior['frames_per_run']) * (behavior['task_runs']+behavior['dark_runs']))
    cue_offset_to_remove = np.zeros(int(behavior['frames_per_run']) * (behavior['task_runs']+behavior['dark_runs']))
    opto_to_remove = np.zeros(int(behavior['frames_per_run']) * (behavior['task_runs']+behavior['dark_runs']))
    trial_times = behavior['onsets']
    for i in range(0, len(trial_times)):
        cue_onset = int(behavior['onsets'][i])
        cue_offset = int(behavior['offsets'][i])
        cue_time = cue_offset - cue_onset + 1
        if behavior['cue_codes'][i] == behavior['cs_1_code']:
            for j in range(0, cue_time):
                cue_type[j + cue_onset] = 1
            for k in range(0, int(behavior['framerate'])*6):
                cue_offset_to_remove[k + cue_offset + 1] = 1
        if behavior['cue_codes'][i] == behavior['cs_2_code']:
            for j in range(0, cue_time):
                cue_type[j + cue_onset] = 2
            for k in range(0, int(behavior['framerate'])*6):
                cue_offset_to_remove[k + cue_offset + 1] = 1
        if (behavior['cue_codes'][i] == behavior['cs_1_opto_code'] or
                behavior['cue_codes'][i] == behavior['cs_2_opto_code']):
            opto_onset = int(behavior['opto_onsets'][i]) - int(.3 * behavior['framerate'])
            opto_time = int(behavior['opto_length'] * behavior['framerate']) + int(5.3 * behavior['framerate'])
            for k in range(0, opto_time):
                opto_to_remove[k + opto_onset] = 1

    test_fluorescence = pd.DataFrame(test_fluorescence)

    cue_prior = cue_prior + cue_offset_to_remove + opto_to_remove
    cue_prior[behavior['relevant_times'] == 0] = 1
    cue_prior = filter_classified(behavior, cue_prior, 1)

    y_pred_all = []
    num_split = 2
    for i in range(0, num_split):
        total_frames = int(behavior['frames_per_run']) * (behavior['task_runs'])
        start = (int(total_frames / num_split) * i) + int(behavior['frames_per_run'])
        end = (int(total_frames / num_split) * (i + 1)) + int(behavior['frames_per_run'])
        idx_frames = np.zeros(int(behavior['frames_per_run']) * (behavior['task_runs'] + behavior['dark_runs']))
        idx_frames[start:end] = 1
        idx_frames[0:behavior['frames_per_run']] = 1
        train_fluorescence_part = train_fluorescence.iloc[:, idx_frames == 1]
        cue_prior_part = cue_prior[idx_frames == 1]
        cue_type_part = cue_type[idx_frames == 1]
        x_train = train_fluorescence_part.loc[idx > 0, cue_prior_part < 1].T
        y_train = cue_type_part[cue_prior_part < 1]
        logistic_model = LogisticRegression(solver='lbfgs', penalty='l2', C=.1, class_weight='balanced',
                                            multi_class='multinomial')
        logistic_model.fit(x_train, y_train)
        x_test = test_fluorescence.copy().iloc[idx > 0, :].T.multiply(1.5)
        y_pred = logistic_model.predict_proba(x_test)
        y_pred_all.append(y_pred)

    return y_pred_all


def process_classified(y_pred, cue_prior, paths, save):
    """
    process classified output
    :param y_pred: output
    :param cue_prior: prior
    :param paths: path to data
    :param save: save or not
    :return: processed output
    """
    if path.isfile(paths['save_path'] + 'saved_data/y_pred.npy') and save == 0:
        y_pred = np.load(paths['save_path'] + 'saved_data/y_pred.npy')
        return y_pred
    else:
        num_split = 2
        y_pred_1 = np.array(y_pred[0][:, 1:3] * np.transpose([cue_prior, cue_prior]))
        y_pred_2 = np.array(y_pred[1][:, 1:3] * np.transpose([cue_prior, cue_prior]))
        y_pred_final = np.array(y_pred[0][:, 1:3] * np.transpose([cue_prior, cue_prior]))
        for i in range(0, len(cue_prior)):
            temp_y_pred_1 = y_pred_1[i, 0] + y_pred_1[i, 1]
            temp_y_pred_2 = y_pred_2[i, 0] + y_pred_2[i, 1]
            if temp_y_pred_1 == temp_y_pred_2:
                if 0 <= i < int(len(y_pred_1[:, 0]) / num_split):
                    y_pred_final[i, 0] = y_pred_1[i, 0]
                    y_pred_final[i, 1] = y_pred_1[i, 1]
                if int(len(y_pred_1[:, 0]) / num_split) <= i < int(len(y_pred_1[:, 0]) / num_split) * 2:
                    y_pred_final[i, 0] = y_pred_2[i, 0]
                    y_pred_final[i, 1] = y_pred_2[i, 1]
            else:
                temp_y_pred = [temp_y_pred_1, temp_y_pred_2]
                max_value = max(temp_y_pred)
                max_idx = temp_y_pred.index(max_value)
                if max_idx == 0:
                    y_pred_final[i, 0] = y_pred_1[i, 0]
                    y_pred_final[i, 1] = y_pred_1[i, 1]
                if max_idx == 1:
                    y_pred_final[i, 0] = y_pred_2[i, 0]
                    y_pred_final[i, 1] = y_pred_2[i, 1]
        if save == 1:
            np.save(paths['save_path'] + 'saved_data/y_pred', y_pred_final)
        return y_pred_final


def filter_classified(behavior, vector, output):
    """
    filter reactivations
    :param behavior: behavior
    :param vector: regression probabilities
    :param output: output
    :return: reactivation filtered
    """
    moving_frames = int(behavior['framerate'])
    filtered_vec = vector.copy()
    runs = int(behavior['task_runs'] + behavior['dark_runs'])
    frames_per_run = int(behavior['frames_per_run'])
    for i in range(0, runs):
        start_frame = i * frames_per_run
        end_frame = (i + 1) * frames_per_run
        filtered_vec[0 + start_frame:moving_frames + start_frame] = output
        filtered_vec[end_frame - moving_frames:end_frame] = output
    filtered_vec[preprocess_opto.moving_average(abs(behavior['running']), moving_frames) > 0] = output
    filtered_vec[preprocess_opto.moving_average(abs(behavior['licking']), moving_frames) > 0] = output
    pupil_movement_thresh = behavior['pupil_movement'].copy()
    pupil_movement_thresh = stats.zscore(pupil_movement_thresh)
    pupil_movement_thresh[pupil_movement_thresh < 6] = 0
    filtered_vec[preprocess_opto.moving_average(pupil_movement_thresh, moving_frames) > 0] = output
    return filtered_vec


def prior_shuffle(norm_moving_deconvolved_filtered, cs_1_idx, cs_2_idx, behavior, threshold):
    """
    get cue prior
    :param norm_moving_deconvolved_filtered: processed activity
    :param cs_1_idx: cs 1 index
    :param cs_2_idx: cs 2 index
    :param behavior: behavior
    :param threshold: threshold
    :return: prior
    """
    cs_1_prior = mean_activity_prior_shuffle(norm_moving_deconvolved_filtered, cs_1_idx, behavior, threshold)
    cs_2_prior = mean_activity_prior_shuffle(norm_moving_deconvolved_filtered, cs_2_idx, behavior, threshold)
    combined_prior = cs_1_prior + cs_2_prior
    combined_prior[combined_prior > 0] = 1
    return combined_prior


def mean_activity_prior_shuffle(activity, cs_idx, behavior, threshold):
    """
    prior
    :param activity: activity
    :param cs_idx: index of cells
    :param behavior: behavior
    :param threshold: threshold
    :return: prior
    """
    if not threshold:
        threshold = 5
    activity[activity < 0] = 0
    activity = pd.DataFrame(activity)
    rand_idx = random.sample(list(range(0, len(activity))), len(cs_idx))
    mean_vec_filtered_cs = activity.reindex(rand_idx[0:int(len(cs_idx) / 20)]).mean()
    mean_vec_filtered_cs = preprocess_opto.filter_cues(behavior, mean_vec_filtered_cs)
    mean_vec_filtered_cs = preprocess_opto.filter_opto(behavior, mean_vec_filtered_cs)
    mean_vec_filtered_cs[behavior['relevant_times'] == 0] = float("nan")
    mean_vec_filtered_cs = filter_classified(behavior, mean_vec_filtered_cs, float("nan"))
    mean_vec_filtered_cs = (mean_vec_filtered_cs - mean_vec_filtered_cs.mean()) / mean_vec_filtered_cs.std()
    mean_vec_filtered_cs[mean_vec_filtered_cs < threshold] = 0
    mean_vec_filtered_cs[mean_vec_filtered_cs != mean_vec_filtered_cs] = 0
    mean_vec_filtered_cs[mean_vec_filtered_cs > 0] = 1
    return mean_vec_filtered_cs


def log_regression_shuffle(behavior, train_fluorescence, test_fluorescence, idx, cue_prior, it):
    """
    classify
    :param behavior: dict of behavior
    :param train_fluorescence: normalized task
    :param test_fluorescence: normalized dark
    :param idx: index of cells
    :param cue_prior: prior
    :param it: iterations
    :return: y pred
    """
    cue_type = np.zeros(int(behavior['frames_per_run']) * (behavior['task_runs']+behavior['dark_runs']))
    cue_offset_to_remove = np.zeros(int(behavior['frames_per_run']) * (behavior['task_runs']+behavior['dark_runs']))
    opto_to_remove = np.zeros(int(behavior['frames_per_run']) * (behavior['task_runs']+behavior['dark_runs']))
    trial_times = behavior['onsets']
    for i in range(0, len(trial_times)):
        cue_onset = int(behavior['onsets'][i])
        cue_offset = int(behavior['offsets'][i])
        cue_time = cue_offset - cue_onset + 1
        if behavior['cue_codes'][i] == behavior['cs_1_code']:
            for j in range(0, cue_time):
                cue_type[j + cue_onset] = 1
            for k in range(0, int(behavior['framerate'])*6):
                cue_offset_to_remove[k + cue_offset + 1] = 1
        if behavior['cue_codes'][i] == behavior['cs_2_code']:
            for j in range(0, cue_time):
                cue_type[j + cue_onset] = 2
            for k in range(0, int(behavior['framerate'])*6):
                cue_offset_to_remove[k + cue_offset + 1] = 1
        if (behavior['cue_codes'][i] == behavior['cs_1_opto_code'] or
                behavior['cue_codes'][i] == behavior['cs_2_opto_code']):
            opto_onset = int(behavior['opto_onsets'][i]) - int(.3 * behavior['framerate'])
            opto_time = int(behavior['opto_length'] * behavior['framerate']) + int(5.3 * behavior['framerate'])
            for k in range(0, opto_time):
                opto_to_remove[k + opto_onset] = 1

    test_fluorescence = pd.DataFrame(test_fluorescence)

    cue_prior = cue_prior + cue_offset_to_remove + opto_to_remove
    cue_prior[behavior['relevant_times'] == 0] = 1
    cue_prior = filter_classified(behavior, cue_prior, 1)

    y_pred_total = []
    num_split = 2
    for i in range(0, num_split):
        start = int(len(cue_type) / num_split) * i
        end = int(len(cue_type) / num_split) * (i + 1)
        train_fluorescence_part = train_fluorescence.copy().iloc[:, start:end]
        cue_prior_part = cue_prior.copy()[start:end]
        cue_type_part = cue_type.copy()[start:end]
        x_train = train_fluorescence_part.loc[idx > 0, cue_prior_part < 1].T
        y_train = cue_type_part[cue_prior_part < 1]
        logistic_model = LogisticRegression(solver='lbfgs', penalty='l2', C=.1, class_weight='balanced',
                                            multi_class='multinomial')
        logistic_model.fit(x_train, y_train)
        x_test = test_fluorescence.copy().iloc[idx > 0, :].T.multiply(1.5)
        for j in range(0, it):
            y_pred = logistic_model.predict_proba(x_test.sample(frac=1))
            y_pred_total.append(y_pred)

    y_pred_all = []
    for j in range(0, it):
        y_pred_temp = [y_pred_total[j], y_pred_total[j + it]]
        y_pred_all.append(y_pred_temp)
    return y_pred_all


def p_distribution_shuffle(norm_moving_deconvolved_filtered, norm_deconvolved, behavior, idx, both_poscells, paths, day,
                           days):
    """
    shuffle classifier and get p dist
    :param norm_moving_deconvolved_filtered: filtered activity
    :param norm_deconvolved: activity
    :param behavior: behavior
    :param idx: index
    :param both_poscells: pos modulation cells
    :param paths: path
    :param day: day
    :param days: days
    :return: shuffled p vdist
    """
    y_pred = process_classified([], [], paths, 0)
    cs_1 = y_pred[:, 0].copy()
    cs_2 = y_pred[:, 1].copy()
    cs_1 = cs_1[cs_1 > 0]
    cs_2 = cs_2[cs_2 > 0]
    p_norm = np.concatenate((cs_1, cs_2))

    all_p_dist = []
    it = 10
    for i in range(0, it):
        prior_temp = prior_shuffle(norm_moving_deconvolved_filtered, idx['cs_1'], idx['cs_2'], behavior, [])
        y_pred = log_regression(behavior, norm_deconvolved, norm_moving_deconvolved_filtered, both_poscells, prior_temp)
        y_pred = process_classified(y_pred, prior_temp, paths, 2)
        cs_1 = y_pred[:, 0].copy()
        cs_2 = y_pred[:, 1].copy()
        cs_1 = cs_1[cs_1 > 0]
        cs_2 = cs_2[cs_2 > 0]
        p_shuffle = np.concatenate((cs_1, cs_2))
        all_p_dist.append(p_shuffle)
    prior_norm = prior(norm_moving_deconvolved_filtered, idx['cs_1'], idx['cs_2'], behavior, [])
    y_pred_all = log_regression_shuffle(behavior, norm_deconvolved, norm_moving_deconvolved_filtered, both_poscells,
                                        prior_norm, it)
    for i in range(0, it):
        y_pred = y_pred_all[i]
        y_pred = process_classified(y_pred, prior_norm, paths, 2)
        cs_1 = y_pred[:, 0].copy()
        cs_2 = y_pred[:, 1].copy()
        cs_1 = cs_1[cs_1 > 0]
        cs_2 = cs_2[cs_2 > 0]
        p_shuffle = np.concatenate((cs_1, cs_2))
        all_p_dist.append(p_shuffle)
    all_p_dist.append(p_norm)

    data_all = sns.kdeplot(data=all_p_dist, clip=(0, 1), common_norm=True).get_lines()
    plt.close()
    p_rand_beta = np.zeros((it, len(data_all[0].get_data()[1])))
    p_rand_prior = np.zeros((it, len(data_all[0].get_data()[1])))
    p_norm = np.zeros((len(data_all[0].get_data()[1])))
    beta_idx = 0
    prior_idx = 0
    for i in range(0, len(data_all)):
        if i == 0:
            p_norm = data_all[i].get_data()[1]
        if 0 < i <= it:
            p_rand_beta[beta_idx, :] = data_all[i].get_data()[1]
            beta_idx += 1
        if it < i <= it*2:
            p_rand_prior[prior_idx, :] = data_all[i].get_data()[1]
            prior_idx += 1
    p_rand_beta = np.mean(p_rand_beta, axis=0)
    p_rand_prior = np.mean(p_rand_prior, axis=0)

    days_path = paths['base_path'] + paths['mouse'] + '/data_across_days/'
    if days:
        if path.isfile(days_path + 'p_shuffle.npy') == 0 or day == 0:
            p_shuffle_days = [list(range(0, days)), list(range(0, days)), list(range(0, days)), list(range(0, days))]
            p_shuffle_days[0][day] = p_norm
            p_shuffle_days[1][day] = p_rand_prior
            p_shuffle_days[2][day] = p_rand_beta
            p_shuffle_days[3][day] = data_all[0].get_data()[0]
            np.save(days_path + 'p_shuffle', p_shuffle_days)
        else:
            p_shuffle_days = np.load(days_path + 'p_shuffle.npy', allow_pickle=True)
            p_shuffle_days[0][day] = p_norm
            p_shuffle_days[1][day] = p_rand_prior
            p_shuffle_days[2][day] = p_rand_beta
            p_shuffle_days[3][day] = data_all[0].get_data()[0]
            np.save(days_path + 'p_shuffle', p_shuffle_days)


def log_regression_R2(behavior, activity, idx, cue_prior):
    """
    classify reactivations
    :param behavior: dict of behavior
    :param train_fluorescence: normalized task
    :param test_fluorescence: normalized dark
    :param idx: index of cells
    :param cue_prior: prior
    :return: y pred
    """
    cue_type = np.zeros(int(behavior['frames_per_run']) * (behavior['task_runs']+behavior['dark_runs']))
    cue_offset_to_remove = np.zeros(int(behavior['frames_per_run']) * (behavior['task_runs']+behavior['dark_runs']))
    opto_to_remove = np.zeros(int(behavior['frames_per_run']) * (behavior['task_runs'] + behavior['dark_runs']))
    trial_times = behavior['onsets']
    for i in range(0, len(trial_times)):
        cue_onset = int(behavior['onsets'][i])
        cue_offset = int(behavior['offsets'][i])
        cue_time = cue_offset - cue_onset + 1

        if i < len(trial_times)-1:
            next_cue = int(behavior['onsets'][i+1]) - cue_time - cue_onset
        if i == len(trial_times)-1:
            next_cue = len(cue_type) - cue_time - cue_onset

        if behavior['cue_codes'][i] == behavior['cs_1_code']:
            for j in range(0, cue_time):
                cue_type[j + cue_onset] = 1
            for k in range(cue_time, next_cue):
                cue_offset_to_remove[k + cue_onset] = 1
        if behavior['cue_codes'][i] == behavior['cs_2_code']:
            for j in range(0, cue_time):
                cue_type[j + cue_onset] = 2
            for k in range(cue_time, next_cue):
                cue_offset_to_remove[k + cue_onset] = 1
        if (behavior['cue_codes'][i] == behavior['cs_1_opto_code'] or
                behavior['cue_codes'][i] == behavior['cs_2_opto_code']):
            opto_onset = int(behavior['opto_onsets'][i]) - int(.3 * behavior['framerate'])
            opto_time = int(behavior['opto_length'] * behavior['framerate']) + int(5.3 * behavior['framerate'])
            for k in range(0, opto_time):
                opto_to_remove[k + opto_onset] = 1

    test_fluorescence = pd.DataFrame(activity)

    cue_prior = cue_prior + cue_offset_to_remove + opto_to_remove
    cue_prior[behavior['relevant_times'] == 0] = 1
    cue_prior = filter_classified(behavior, cue_prior, 1)

    y_pred_all = []
    num_split = 2
    for i in range(0, num_split):
        total_frames = int(behavior['frames_per_run']) * (behavior['task_runs'])
        start = (int(total_frames / num_split) * i) + int(behavior['frames_per_run'])
        end = (int(total_frames / num_split) * (i + 1)) + int(behavior['frames_per_run'])
        idx_frames = np.zeros(int(behavior['frames_per_run']) * (behavior['task_runs'] + behavior['dark_runs']))
        idx_frames[start:end] = 1
        idx_frames[0:behavior['frames_per_run']] = 1
        train_fluorescence_part = activity.iloc[:, idx_frames == 1]
        cue_prior_part = cue_prior[idx_frames == 1]
        cue_type_part = cue_type[idx_frames == 1]
        x_train = train_fluorescence_part.loc[idx > 0, cue_prior_part < 1].T
        y_train = cue_type_part[cue_prior_part < 1]
        logistic_model = LogisticRegression(solver='lbfgs', penalty='l2', C=.1, class_weight='balanced',
                                            multi_class='multinomial')
        logistic_model.fit(x_train, y_train)
        x_test = test_fluorescence.copy().iloc[idx > 0, :].T.multiply(1.5)
        y_pred = logistic_model.predict_proba(x_test)
        y_pred_all.append(y_pred)

    return y_pred_all


def prior_R2(norm_moving_deconvolved_filtered, cs_1_idx, cs_2_idx, behavior):
    """
    get cue prior
    :param norm_moving_deconvolved_filtered: processed activity
    :param cs_1_idx: cs 1 index
    :param cs_2_idx: cs 2 index
    :param behavior: behavior
    :param threshold: threshold
    :return: prior
    """
    [cs_1_prior, cs_1_prior_iti] = mean_activity_prior_R2(norm_moving_deconvolved_filtered, cs_1_idx, behavior, [])
    [cs_2_prior, cs_2_prior_iti] = mean_activity_prior_R2(norm_moving_deconvolved_filtered, cs_2_idx, behavior, [])
    combined_prior = cs_1_prior + cs_2_prior
    combined_prior[combined_prior > 0] = 1
    combined_prior_iti = cs_1_prior_iti + cs_2_prior_iti
    combined_prior_iti[combined_prior_iti > 1] = 1
    return [combined_prior, combined_prior_iti]


def mean_activity_prior_R2(activity, cs_idx, behavior, threshold):
    """
    prior helper
    :param activity: activity
    :param cs_idx: index of cells
    :param behavior: behavior
    :param threshold: threshold std for synchronous activity
    :return: prior
    """
    if not threshold:
        threshold = 5
    activity[activity < 0] = 0
    activity = pd.DataFrame(activity)
    mean_vec_filtered_cs = activity.reindex(cs_idx.index[0:int(len(cs_idx) / 20)]).mean()
    mean_vec_filtered_cs = preprocess_opto.filter_cues(behavior, mean_vec_filtered_cs)
    mean_vec_filtered_cs = preprocess_opto.filter_opto(behavior, mean_vec_filtered_cs)
    mean_vec_filtered_cs[behavior['relevant_times'] == 0] = float("nan")
    mean_vec_filtered_cs = filter_classified(behavior, mean_vec_filtered_cs, float("nan"))
    mean_vec_filtered_cs = (mean_vec_filtered_cs - mean_vec_filtered_cs.mean()) / mean_vec_filtered_cs.std()
    mean_vec_filtered_cs_iti = mean_vec_filtered_cs.copy()

    mean_vec_filtered_cs[mean_vec_filtered_cs < threshold] = 0
    mean_vec_filtered_cs[mean_vec_filtered_cs != mean_vec_filtered_cs] = 0
    mean_vec_filtered_cs[mean_vec_filtered_cs > 0] = 1

    mean_vec_filtered_cs_iti[mean_vec_filtered_cs_iti < threshold] = 1
    mean_vec_filtered_cs_iti[mean_vec_filtered_cs_iti >= threshold] = 0
    mean_vec_filtered_cs_iti[mean_vec_filtered_cs_iti != mean_vec_filtered_cs_iti] = 0
    mean_vec_filtered_cs_iti[mean_vec_filtered_cs_iti > 0] = 1

    return [mean_vec_filtered_cs, mean_vec_filtered_cs_iti]


def process_classified_R2(norm_moving_deconvolved_filtered, cs_1_idx, cs_2_idx, behavior, y_pred, paths, day, days):

    [combined_prior, combined_prior_iti] = prior_R2(norm_moving_deconvolved_filtered, cs_1_idx, cs_2_idx, behavior)
    rate_syn = process_classified_R2_helper(y_pred, combined_prior, behavior)
    rate_iti = process_classified_R2_helper(y_pred, combined_prior_iti, behavior)

    days_path = paths['base_path'] + paths['mouse'] + '/data_across_days/'
    if days:
        if path.isfile(days_path + 'reactivation_syn_iti.npy') == 0 or day == 0:
            reactivation_rate_days = [list(range(0, days)), list(range(0, days))]
            reactivation_rate_days[0][day] = rate_syn
            reactivation_rate_days[1][day] = rate_iti
            np.save(days_path + 'reactivation_syn_iti', reactivation_rate_days)
        else:
            reactivation_rate_days = np.load(days_path + 'reactivation_syn_iti.npy', allow_pickle=True)
            reactivation_rate_days[0][day] = rate_syn
            reactivation_rate_days[1][day] = rate_iti
            np.save(days_path + 'reactivation_syn_iti', reactivation_rate_days)


def process_classified_R2_helper(y_pred, cue_prior, behavior):
    num_split = 2
    y_pred_1 = np.array(y_pred[0][:, 1:3] * np.transpose([cue_prior, cue_prior]))
    y_pred_2 = np.array(y_pred[1][:, 1:3] * np.transpose([cue_prior, cue_prior]))
    y_pred_final = np.array(y_pred[0][:, 1:3] * np.transpose([cue_prior, cue_prior]))
    for i in range(0, len(cue_prior)):
        temp_y_pred_1 = y_pred_1[i, 0] + y_pred_1[i, 1]
        temp_y_pred_2 = y_pred_2[i, 0] + y_pred_2[i, 1]
        if temp_y_pred_1 == temp_y_pred_2:
            if 0 <= i < int(len(y_pred_1[:, 0]) / num_split):
                y_pred_final[i, 0] = y_pred_1[i, 0]
                y_pred_final[i, 1] = y_pred_1[i, 1]
            if int(len(y_pred_1[:, 0]) / num_split) <= i < int(len(y_pred_1[:, 0]) / num_split) * 2:
                y_pred_final[i, 0] = y_pred_2[i, 0]
                y_pred_final[i, 1] = y_pred_2[i, 1]
        else:
            temp_y_pred = [temp_y_pred_1, temp_y_pred_2]
            max_value = max(temp_y_pred)
            max_idx = temp_y_pred.index(max_value)
            if max_idx == 0:
                y_pred_final[i, 0] = y_pred_1[i, 0]
                y_pred_final[i, 1] = y_pred_1[i, 1]
            if max_idx == 1:
                y_pred_final[i, 0] = y_pred_2[i, 0]
                y_pred_final[i, 1] = y_pred_2[i, 1]
    reactivation_cs_1 = y_pred_final[:, 0]
    reactivation_cs_2 = y_pred_final[:, 1]
    y_pred_all = (reactivation_cs_1 + reactivation_cs_2)
    y_pred_all = sum(y_pred_all[behavior['frames_per_run']:len(reactivation_cs_1)])/sum(cue_prior[behavior['frames_per_run']:len(reactivation_cs_1)])
    return y_pred_all
