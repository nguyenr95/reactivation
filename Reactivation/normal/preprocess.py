import os
import math
import pickle
import classify
import numpy as np
import pandas as pd
from os import path
from os import makedirs
from scipy import stats
from scipy.io import loadmat
from scipy.io import savemat
from scipy.ndimage import gaussian_filter1d
import warnings
warnings.filterwarnings('ignore')


def create_folders(mouse, date):
    """
    creates folders
    :param mouse: mouse name
    :param date: date
    :return: folders
    """
    base_path = 'D:/2p_data/scan/'
    save_path = base_path + mouse + '/' + date + '_' + mouse + '/processed_data/'
    if not path.exists(save_path + 'plots'):
        makedirs(save_path + 'plots')
    if not path.exists(save_path + 'movies'):
        makedirs(save_path + 'movies')
    if not path.exists(save_path + 'saved_data'):
        makedirs(save_path + 'saved_data')
    if not path.exists(base_path + mouse + '/data_across_days'):
        makedirs(base_path + mouse + '/data_across_days')
    if not path.exists(base_path + mouse + '/data_across_days/plots'):
        makedirs(base_path + mouse + '/data_across_days/plots')
    if not path.exists(save_path + 'movies/reactivation'):
        makedirs(save_path + 'movies/reactivation')
    return {'base_path': base_path, 'mouse': mouse, 'date': date, 'save_path': save_path}


def load_data(paths):
    """
    Load in data for mouse and date
    :param paths: path to data
    :return: neural and behavioral data and path to it
    """
    session_data = loadmat(paths['save_path'] + 'saved_data/behavior_file.mat')
    return session_data


def process_behavior(session_data, paths):
    """
    Processes behavioral data for mouse and date
    :param session_data: neural data from suite2p
    :param paths: path to data
    :return: behavioral time stamps
    """
    onsets = session_data['onsets'].astype(int) - 1
    offsets = session_data['offsets'].astype(int) - 1
    cue_codes = session_data['cue_code']
    cs_1_code = int(session_data['CS_1_code'])
    cs_2_code = int(session_data['CS_2_code'])
    task_runs = int(session_data['task_runs'])
    dark_runs = int(session_data['dark_runs'])
    unique, counts = np.unique(cue_codes, return_counts=True)
    num_cs_1_trials = counts[0]
    num_cs_2_trials = counts[1]
    frames_per_run = int(session_data['frames_per_run'])
    framerate = session_data['framerate']
    frames_before = int(framerate * 2)
    frames_after = int(framerate * 8) + 1
    iti = int(session_data['ITI'])
    ops_path = paths['base_path'] + paths['mouse'] + '/' + paths['date'] + '_' + paths['mouse'] + \
                    '/suite2p_plane_1/suite2p/plane0/'
    ops = np.load(ops_path + 'ops.npy', allow_pickle=True).item()
    y_off = np.diff(ops['yoff']) * np.diff(ops['yoff'])
    x_off = np.diff(ops['xoff']) * np.diff(ops['xoff'])
    brain_motion = np.sqrt([y_off + x_off])
    brain_motion = np.concatenate((np.zeros(1), brain_motion[0]))
    corrXY = ops['corrXY']
    pupil = session_data['pupil'][0]
    pupil_movement = session_data['pupil_movement'][0]
    running = session_data['running'][0]
    task_times = np.ones(frames_per_run*(task_runs+dark_runs))
    relevant_times = np.ones(frames_per_run*(task_runs+dark_runs))
    start_trials = []
    end_trials = []
    if "end_trials" in session_data:
        end_trials = session_data['end_trials'][0].astype(int) - 1
        start_trials = session_data['start_trials'][0].astype(int) - 1
    else:
        for i in range(0, task_runs):
            end_trials.append(int(len(onsets)/task_runs) * (i+1) - 1)
            start_trials.append(int(len(onsets)/task_runs) * i)
    for i in range(0, task_runs):
        start_1 = (frames_per_run*dark_runs) + (i*frames_per_run)
        end_1 = int(onsets[start_trials[i]])
        start_2 = int(offsets[end_trials[i]]) + 1
        end_2 = (frames_per_run*dark_runs) + ((i+1)*frames_per_run)
        task_times[start_1:end_1] = 0
        task_times[start_2:end_2] = 0
        task_times[0:frames_per_run*dark_runs] = 0
        relevant_times[start_2:end_2] = 0
    temp_pupil = np.sort(pupil[relevant_times == 1])
    pupil_max = np.mean(temp_pupil[len(temp_pupil)-int(len(pupil)*.01):len(temp_pupil)])
    pupil_min = np.mean(temp_pupil[0:int(len(pupil)*.01)])
    temp_pupil_movement = np.sort(pupil_movement[relevant_times == 1])
    pupil_movement_max = np.mean(temp_pupil_movement[len(temp_pupil_movement) - int(len(pupil_movement) * .01):len(temp_pupil_movement)])
    licking = session_data['licking'][0]
    licking[licking > .1] = 1
    licking[licking < .1] = 0
    if len(licking) == 1:
        licking = np.zeros((1, len(pupil)))
    cue_offset = 6
    behavior = {'onsets': onsets, 'offsets': offsets, 'cue_codes': cue_codes, 'cs_1_code': cs_1_code,
                'cs_2_code': cs_2_code, 'licking': licking, 'running': running, 'pupil': pupil,
                'cs_1_trials': num_cs_1_trials, 'cs_2_trials': num_cs_2_trials, 'frames_before': frames_before,
                'frames_after': frames_after, 'framerate': framerate, 'frames_per_run': frames_per_run,
                'task_runs': task_runs, 'iti': iti, 'dark_runs': dark_runs, 'brain_motion': brain_motion,
                'pupil_max': pupil_max, 'pupil_min': pupil_min, 'task_times': task_times,
                'relevant_times': relevant_times, 'cue_offset': cue_offset, 'pupil_movement': pupil_movement,
                'pupil_movement_max': pupil_movement_max, 'end_trials': end_trials, 'start_trials': start_trials,
                'phase_correlation': corrXY}
    return behavior


def cell_masks(paths, save):
    """
    make cell masks
    :param paths: path to data
    :param save: to overwrite or not
    :return:
    """
    if path.isfile(paths['save_path'] + 'saved_data/overlap_plane_2.mat') == 0 or save == 1:
        data_path_1 = paths['base_path'] + paths['mouse'] + '/' + paths['date'] + '_' + paths['mouse'] + \
                      '/suite2p_plane_1/suite2p/plane0/'
        make_cell_masks(data_path_1, paths, 1)
        data_path_2 = paths['base_path'] + paths['mouse'] + '/' + paths['date'] + '_' + paths['mouse'] + \
                      '/suite2p_plane_2/suite2p/plane0/'
        if os.path.isdir(data_path_2):
            make_cell_masks(data_path_2, paths, 2)
        data_path_3 = paths['base_path'] + paths['mouse'] + '/' + paths['date'] + '_' + paths['mouse'] + \
                      '/suite2p_plane_3/suite2p/plane0/'
        if os.path.isdir(data_path_3):
            make_cell_masks(data_path_3, paths, 3)


def make_cell_masks(data_path, paths, plane):
    """
    make cell masks and save as matlab for registration across planes
    :param data_path: path to data
    :param paths: path to data
    :param plane: plane
    :return: cell_masks
    """
    stat = np.load(data_path + 'stat.npy', allow_pickle=True)
    accepted_cells = np.load(data_path + 'iscell.npy')[:, 0]
    fluorescence = np.load(data_path + 'F.npy').sum(axis=1)
    accepted_cells[fluorescence == 0] = 0
    accepted_cells = accepted_cells == 1
    stat = stat[accepted_cells]
    ops = np.load(data_path + 'ops.npy', allow_pickle=True).item()
    im = np.zeros((ops['Ly'], ops['Lx'], len(stat)))
    for n in range(0, len(stat)):
        ypix = stat[n]['ypix'][~stat[n]['overlap']]
        xpix = stat[n]['xpix'][~stat[n]['overlap']]
        im[ypix, xpix, n] = 1
    np.save(paths['save_path'] + 'saved_data/plane_' + str(plane) + '_cell_masks.npy', im)


def process_activity(paths, activity_type, planes, to_delete_save):
    """
    Processes matrix of neural activity of all real cells during task and quiet waking
    :param paths: path to data
    :param activity_type: deconvolved or fluorescence
    :param planes: number of planes
    :param to_delete_save: load to delete or not
    :return: processed fluorescence and events
    """
    activity = []
    for plane in range(1, planes+1):
        plane_path = paths['base_path'] + paths['mouse'] + '/' + paths['date'] + '_' + paths['mouse'] + \
                  '/suite2p_plane_' + str(plane) + '/suite2p/plane0/'
        accepted_cells = np.load(plane_path + 'iscell.npy')[:, 0]
        fluorescence = np.load(plane_path + 'F.npy').sum(axis=1)
        accepted_cells[fluorescence == 0] = 0
        accepted_cells = accepted_cells == 1
        all_activity = np.load(plane_path + activity_type + '.npy')
        if activity_type == 'F':
            all_activity = all_activity - np.load(plane_path + 'Fneu.npy')
        if plane == 1:
            activity = all_activity[accepted_cells, :]
        else:
            activity_plane = all_activity[accepted_cells, :]
            to_delete = cells_to_delete(paths, plane, plane_path, to_delete_save)
            activity_plane = activity_plane[to_delete == 1, :]
            activity = np.concatenate((activity, activity_plane))
    return activity


def cells_to_delete(paths, plane, plane_path, save):
    """
    find cells from other planes to delete
    :param paths: path to data
    :param plane: which plane to compare
    :param plane_path: path to plane
    :param save: to overwrite
    :return: vector of cells to delete
    """
    if path.isfile(paths['save_path'] + 'saved_data/to_delete_plane_' + str(plane) + '.npy') and save == 0:
        to_delete = np.load(paths['save_path'] + 'saved_data/to_delete_plane_' + str(plane) + '.npy')
    else:
        overlap_cells = loadmat(paths['save_path'] + 'saved_data/overlap_plane_' + str(plane))
        overlap_cells = overlap_cells['overlap_vec']

        accepted_cells = np.load(plane_path + 'iscell.npy')[:, 0]
        fluorescence = np.load(plane_path + 'F.npy').sum(axis=1)
        accepted_cells[fluorescence == 0] = 0
        accepted_cells = accepted_cells == 1
        all_activity = np.load(plane_path + 'F.npy')
        all_activity = all_activity - np.load(plane_path + 'Fneu.npy')
        fluorescence_plane_other = all_activity[accepted_cells, :]

        plane_1_path = paths['base_path'] + paths['mouse'] + '/' + paths['date'] + '_' + paths['mouse'] + \
                       '/suite2p_plane_1/suite2p/plane0/'
        accepted_cells = np.load(plane_1_path + 'iscell.npy')[:, 0]
        fluorescence = np.load(plane_1_path + 'F.npy').sum(axis=1)
        accepted_cells[fluorescence == 0] = 0
        accepted_cells = accepted_cells == 1
        all_activity = np.load(plane_1_path + 'F.npy')
        all_activity = all_activity - np.load(plane_1_path + 'Fneu.npy')
        fluorescence_plane_1 = all_activity[accepted_cells, :]

        r = np.corrcoef(fluorescence_plane_1, fluorescence_plane_other)
        r_idx = r[len(fluorescence_plane_1):len(r), 0:len(fluorescence_plane_1)]
        to_delete = np.ones(len(overlap_cells))
        for i in range(0, len(overlap_cells)):
            if overlap_cells[i, 1] > 0:
                idx_exclude = np.ones(len(fluorescence_plane_1))
                idx_exclude[int(overlap_cells[i, 0]) - 1] = 0
                corr_non = r_idx[i, idx_exclude == 1]
                max_corr_non = np.max(corr_non)
                overlap_corr = r_idx[i, int(overlap_cells[i, 0]) - 1]
                if overlap_corr > max_corr_non:
                    to_delete[i] = 0
        np.save(paths['save_path'] + 'saved_data/to_delete_plane_' + str(plane), to_delete)
    return to_delete


def normalize_deconvolved(deconvolved_vec, behavior, paths, save):
    """
    normalize deconvolved to peak
    :param deconvolved_vec:  dark or task events vector
    :param behavior: behavior
    :param paths: path to data
    :param save: overwrite saved filed or not
    :return: normalized deconvolved dataframe
    """
    if path.isfile(paths['save_path'] + 'saved_data/norm_deconvolved.npy') and save == 0:
        norm_deconvolved = np.load(paths['save_path'] + 'saved_data/norm_deconvolved.npy')
        norm_deconvolved = pd.DataFrame(norm_deconvolved)
    else:
        norm_vec = np.empty(len(deconvolved_vec))
        norm_vec[:] = np.nan
        for i in range(0, len(deconvolved_vec)):
            temp_deconvolved = deconvolved_vec[i, behavior['relevant_times'] == 1][deconvolved_vec[i,
                                                                                   behavior['relevant_times'] == 1] > 0]
            temp_deconvolved = np.flip(np.sort(temp_deconvolved))
            norm_value = np.mean(temp_deconvolved[0:int(len(temp_deconvolved) / 100)])
            norm_vec[i] = norm_value
        deconvolved_vec = pd.DataFrame(deconvolved_vec)
        norm_deconvolved = deconvolved_vec.divide(norm_vec, axis=0)
        np.save(paths['save_path'] + 'saved_data/norm_deconvolved', norm_deconvolved)
    return norm_deconvolved


def difference_gaussian_filter(deconvolved_vec, fwhm, behavior, paths, save):
    """
    min difference of gaussian filter
    :param deconvolved_vec: vector of activity
    :param fwhm: full width at half max
    :param behavior: behavior
    :param paths: path to data
    :param save: to save or not
    :return: filtered vector
    """
    if path.isfile(paths['save_path'] + 'saved_data/deconvolved_filtered.npy') and save == 0:
        norm_moving_deconvolved_filtered = np.load(paths['save_path'] + 'saved_data/deconvolved_filtered.npy')
        norm_moving_deconvolved_filtered[norm_moving_deconvolved_filtered < 0] = 0
        return norm_moving_deconvolved_filtered
    else:
        cue_times_vec = cue_times(behavior, behavior['cue_offset'], 0)
        times_to_use = behavior['relevant_times'].copy()
        times_to_use[cue_times_vec[0, :] > 0] = 0
        deconvolved_vector = deconvolved_vec.iloc[:, times_to_use == 1]
        deconvolved_vector = deconvolved_vector.rolling(window=4, axis=1, center=True, min_periods=1).max()
        deconvolved_vector = np.array(deconvolved_vector)

        sigma_0 = fwhm / (2 * math.sqrt(2 * np.log(2)))
        sigma_1 = math.pow(fwhm, 2) / (2 * math.sqrt(2 * np.log(2)))
        sigma_2 = math.pow(fwhm, 3) / (2 * math.sqrt(2 * np.log(2)))
        sigma_3 = math.pow(fwhm, 4) / (2 * math.sqrt(2 * np.log(2)))

        filtered_s0 = pd.DataFrame(gaussian_filter1d(deconvolved_vector, sigma_0))
        filtered_s1 = pd.DataFrame(gaussian_filter1d(deconvolved_vector, sigma_1))
        filtered_s2 = pd.DataFrame(gaussian_filter1d(deconvolved_vector, sigma_2))
        filtered_s3 = pd.DataFrame(gaussian_filter1d(deconvolved_vector, sigma_3))

        deconvolved_vector_filter_1 = filtered_s0 - filtered_s1
        deconvolved_vector_filter_2 = filtered_s0 - filtered_s2
        deconvolved_vector_filter_3 = filtered_s0 - filtered_s3

        deconvolved_vector_filter_min = pd.concat(
            [deconvolved_vector_filter_1, deconvolved_vector_filter_2,
             deconvolved_vector_filter_3]).min(level=0)

        deconvolved_vector_filter_min_final = np.array(deconvolved_vec)
        deconvolved_vector_filter_min = np.array(deconvolved_vector_filter_min)

        frame = 0
        for i in range(0, len(times_to_use)):
            if times_to_use[i] == 1:
                deconvolved_vector_filter_min_final[:, i] = deconvolved_vector_filter_min[:, frame]
                frame = frame + 1

        np.save(paths['save_path'] + 'saved_data/deconvolved_filtered', deconvolved_vector_filter_min_final)
        return deconvolved_vector_filter_min_final


def normalized_trial_averaged(activity, behavior, trial_type):
    """
    makes normalized trial averaged trace for each cue type
    :param activity: fluorescence
    :param behavior: dict of behavior
    :param trial_type: type of trial
    :return: trial averaged trace for given cell
    """
    activity = activity.to_numpy()
    index_frames = []
    num_trials = 0
    trial_times = behavior['onsets'][behavior['cue_codes'] == behavior[trial_type + '_code']]
    for i in trial_times:
        if i > 0:
            num_trials = num_trials + 1
            for j in range(-behavior['frames_before'], behavior['frames_after']):
                index_frames.append(int(i) + j)
    activity_task_idx = activity[:, index_frames]
    activity_task_idx = np.reshape(activity_task_idx, (activity_task_idx.shape[0], num_trials,
                                                       behavior['frames_before'] + behavior['frames_after']))
    activity_task_mean = pd.DataFrame(activity_task_idx.mean(axis=1))
    frames_before = behavior['frames_before']
    activity_task_mean_df = activity_task_mean.subtract(
        activity_task_mean.iloc[:, int(frames_before / 2):frames_before].mean(axis=1), axis=0)
    return activity_task_mean_df


def sig_test(activity, behavior, trial_type):
    """
    sig test for cue cells
    :param activity: fluorescence
    :param behavior: dict of behavior
    :param trial_type: trial type
    :return: which cells are sig for which cue(s)
    """
    activity = activity.to_numpy()
    frames_before = behavior['frames_before']
    index_frames = []
    num_trials = 0
    trial_times = behavior['onsets'][behavior['cue_codes'] == behavior[trial_type + '_code']]
    for i in trial_times:
        if i > 0:
            num_trials = num_trials + 1
            for j in range(-frames_before, frames_before):
                index_frames.append(int(i) + j)
    activity_idx = activity[:, index_frames]
    activity_idx = np.reshape(activity_idx, (activity_idx.shape[0], num_trials, frames_before * 2))
    pos_sig_cells = np.zeros(len(activity_idx))
    neg_sig_cells = np.zeros(len(activity_idx))
    for j in range(len(activity_idx)):
        before = np.reshape(activity_idx[j, :, 0:frames_before], (num_trials * frames_before))
        after = np.reshape(activity_idx[j, :, frames_before:frames_before * 2], (num_trials * frames_before))
        res = stats.ranksums(before, after)
        if res.statistic < 0 and res.pvalue < .01:
            pos_sig_cells[j] = 1
        if res.statistic > 0 and res.pvalue < .01:
            neg_sig_cells[j] = 1
    return [pos_sig_cells, neg_sig_cells]


def combine_sig(cs_1_poscells, cs_1_negcells, cs_2_poscells, cs_2_negcells):
    """
    combines sig cells
    :param cs_1_poscells: cs 1 pos cells
    :param cs_1_negcells: cs 1 neg cells
    :param cs_2_poscells: cs 2 pos cells
    :param cs_2_negcells: cs 2 neg cells
    :return: both
    """
    both_poscells = cs_1_poscells + cs_2_poscells
    both_poscells[both_poscells > 1] = 1
    both_sigcells = cs_1_poscells + cs_2_poscells + cs_1_negcells + cs_2_negcells
    both_sigcells[both_sigcells > 1] = 1
    return [both_poscells, both_sigcells]


def get_index(behavior, mean_cs_1_responses_df, mean_cs_2_responses_df, cs_1_poscells, cs_2_poscells, both_poscells,
              both_sigcells, paths, save):
    """
    gets index of top cells
    :param behavior: behavior
    :param mean_cs_1_responses_df: cs 1 response
    :param mean_cs_2_responses_df: cs 2 reponse
    :param cs_1_poscells: cs 1 pos cells
    :param cs_2_poscells: cs 2 pos cells
    :param both_poscells: all pos cells
    :param both_sigcells: all sig cells
    :param paths: path to data
    :param save: to save or not
    :return: indices
    """
    if path.isfile(paths['save_path'] + 'saved_data/idx.npy') and save == 0:
        idx = np.load(paths['save_path'] + 'saved_data/idx.npy', allow_pickle=True)
        return idx.item()
    else:
        cs_1_idx = sort_cells(behavior, mean_cs_1_responses_df, cs_1_poscells, 'Mean', 2, 'descending')
        cs_2_idx = sort_cells(behavior, mean_cs_2_responses_df, cs_2_poscells, 'Mean', 2, 'descending')
        cs_1_idx_df = sort_cells(behavior, mean_cs_1_responses_df-mean_cs_2_responses_df, cs_1_poscells, 'Mean', 2,
                                 'descending')
        cs_2_idx_df = sort_cells(behavior, mean_cs_2_responses_df-mean_cs_1_responses_df, cs_2_poscells, 'Mean', 2,
                                 'descending')
        both_idx = sort_cells(behavior, mean_cs_2_responses_df+mean_cs_1_responses_df, both_poscells, 'Mean', 2,
                              'descending')
        both_sig = pd.DataFrame(both_sigcells)
        both_sig = both_sig[(both_sig.T != 0).any()]
        idx = {'cs_1': cs_1_idx, 'cs_2': cs_2_idx, 'cs_1_df': cs_1_idx_df, 'cs_2_df': cs_2_idx_df, 'both': both_idx,
               'all': both_sig}
        if save == 1:
            np.save(paths['save_path'] + 'saved_data/idx', idx)
        return idx


def sort_cells(behavior, mean_responses, sig_cells, sort, seconds, direction):
    """
    gets sorting (peak, max, etc)
    :param behavior: dict of behavior
    :param mean_responses: dataframe of mean responses
    :param sig_cells: sig cells
    :param sort: sort type
    :param seconds: how many seconds after onset to sort by, scale of frames before
    :param direction: descending or ascending
    :return: sorted heatmap plots
    """
    if len(sig_cells) > 0:
        mean_responses = mean_responses.iloc[sig_cells > 0, :]
    frames_before = behavior['frames_before']
    idx = []
    if sort == 'Peak':
        idx = pd.DataFrame({'peak_fluorescence': mean_responses.idxmax(1)})
        idx = idx.sort_values(by=['peak_fluorescence'])
    if sort == 'Mean':
        mean_responses_idx = pd.DataFrame({'mean_fluorescence': mean_responses.
                                          iloc[:, frames_before:int(frames_before * seconds)].mean(axis=1)})
        if direction == 'descending':
            idx = mean_responses_idx.sort_values(by=['mean_fluorescence'], ascending=0)
        if direction == 'ascending':
            idx = mean_responses_idx.sort_values(by=['mean_fluorescence'], ascending=1)
    if sort == 'Max':
        mean_responses_idx = pd.DataFrame({'max_fluorescence': mean_responses.
                                          iloc[:, frames_before:int(frames_before * seconds)].max(axis=1)})
        idx = mean_responses_idx.sort_values(by=['max_fluorescence'], ascending=0)
    return idx


def sig_reactivated_cells(activity, norm_moving_deconvolved_filtered, idx, y_pred, behavior, paths, save):

    if path.isfile(paths['save_path'] + 'saved_data/reactivated.npy') and save == 0:
        sig_cells = np.load(paths['save_path'] + 'saved_data/reactivated.npy')
        return sig_cells
    else:
        activity = activity.to_numpy()
        activity = activity[idx['both'].index]

        prior = classify.prior(norm_moving_deconvolved_filtered, idx['cs_1'], idx['cs_2'], behavior, 1)

        reactivation_cs_1 = y_pred[:, 0].copy()
        reactivation_cs_2 = y_pred[:, 1].copy()
        p_threshold = .75
        cs_1_peak = 0
        cs_2_peak = 0
        i = 0
        activity_r_cs_1_idx = np.zeros(len(reactivation_cs_1))
        activity_r_cs_2_idx = np.zeros(len(reactivation_cs_1))
        activity_rand_cs_1_idx = []
        activity_rand_cs_2_idx = []
        next_r = 0
        while i < len(reactivation_cs_1) - 1:
            i += 1
            if reactivation_cs_1[i] > 0 or reactivation_cs_2[i] > 0:
                if next_r == 0:
                    r_start = i
                    next_r = 1
                if reactivation_cs_1[i] > cs_1_peak:
                    cs_1_peak = reactivation_cs_1[i]
                if reactivation_cs_2[i] > cs_2_peak:
                    cs_2_peak = reactivation_cs_2[i]
                if reactivation_cs_1[i + 1] == 0 and reactivation_cs_2[i + 1] == 0:
                    r_end = i + 1
                    next_r = 0
                    if cs_1_peak > p_threshold and r_start > int(behavior['onsets'][0]):
                        activity_r_cs_1_idx[r_start:r_end] = 1
                        num_frames = r_end - r_start
                        idx = 1
                        while num_frames > 0:
                            if prior[r_start-idx] == 0:
                                activity_rand_cs_1_idx.append(r_start-idx)
                                num_frames -= 1
                            idx += 1
                    if cs_2_peak > p_threshold and r_start > int(behavior['onsets'][0]):
                        activity_r_cs_2_idx[r_start:r_end] = 1
                        num_frames = r_end - r_start
                        idx = 1
                        while num_frames > 0:
                            if prior[r_start - idx] == 0:
                                activity_rand_cs_2_idx.append(r_start-idx)
                                num_frames -= 1
                            idx += 1
                    i = r_end
                    cs_1_peak = 0
                    cs_2_peak = 0

        activity_r_cs_1 = activity[:, activity_r_cs_1_idx == 1]
        activity_r_cs_2 = activity[:, activity_r_cs_2_idx == 1]
        activity_rand_cs_1 = activity[:, activity_rand_cs_1_idx]
        activity_rand_cs_2 = activity[:, activity_rand_cs_2_idx]

        sig_cells = np.zeros(len(activity_rand_cs_1))
        for j in range(len(activity_rand_cs_1)):
            before = activity_rand_cs_1[j, :]
            after = activity_r_cs_1[j, :]
            res = stats.mannwhitneyu(before, after, alternative='less')
            if res.pvalue < .05:
                sig_cells[j] = 1
            before = activity_rand_cs_2[j, :]
            after = activity_r_cs_2[j, :]
            res = stats.mannwhitneyu(before, after, alternative='less')
            if res.pvalue < .05:
                sig_cells[j] = 2
        if save == 1:
            np.save(paths['save_path'] + 'saved_data/reactivated', sig_cells)
        return sig_cells


def group_neurons(activity, behavior, trial_type):

    index_frames_start = []
    index_frames_end = []
    num_trials_total = 10
    num_trials = 0
    trial_times = behavior['onsets'][behavior['cue_codes'] == behavior[trial_type + '_code']]
    for i in trial_times:
        if i > 0 and num_trials < num_trials_total:
            for j in range(0, behavior['frames_before']):
                index_frames_start.append(int(i) + j)
            num_trials = num_trials + 1
    num_trials = 0
    for i in reversed(trial_times):
        if i > 0 and num_trials < num_trials_total:
            for j in range(0, behavior['frames_before']):
                index_frames_end.append(int(i) + j)
            num_trials = num_trials + 1

    activity_start = activity[:, index_frames_start]
    activity_start = np.reshape(activity_start, (activity_start.shape[0], num_trials_total, behavior['frames_before']))
    activity_end = activity[:, index_frames_end]
    activity_end = np.reshape(activity_end, (activity_end.shape[0], num_trials_total, behavior['frames_before']))

    increase_sig_cells = np.zeros(len(activity_start))
    decrease_sig_cells = np.zeros(len(activity_start))
    no_change_cells = np.zeros(len(activity_start))

    dist = np.zeros(len(activity_start))
    for j in range(len(activity_start)):
        before = np.reshape(activity_start[j, :, 0:behavior['frames_before']], (num_trials * behavior['frames_before']))
        after = np.reshape(activity_end[j, :, 0:behavior['frames_before']], (num_trials * behavior['frames_before']))
        dist[j] = (np.mean(after)-np.mean(before))/np.mean(before)

    for j in range(len(activity_start)):
        if dist[j] > np.mean(dist) + np.std(dist):
            increase_sig_cells[j] = 1
        if dist[j] < np.mean(dist) - np.std(dist):
            decrease_sig_cells[j] = 1
        if np.mean(dist) - (np.std(dist)/2) < dist[j] < np.mean(dist) + (np.std(dist)/2):
            no_change_cells[j] = 1

    return [no_change_cells, increase_sig_cells, decrease_sig_cells]


def filter_cues(behavior, vector):
    """
    filter cues
    :param behavior: behavior
    :param vector: activity vector
    :return: cue filtered
    """
    cue_idx = cue_times(behavior, behavior['cue_offset'], 0)
    vector[cue_idx[0, :] > 0] = float("nan")
    return vector


def cue_times(behavior, offset, preonset):
    """
    get cue times
    :param behavior: behavior
    :param offset: how much after cue to include
    :param preonset: how much before cue to include
    :return: vector of cue times
    """
    runs = int(behavior['task_runs']) + int(behavior['dark_runs'])
    frames_per_run = int(behavior['frames_per_run'])
    cue_idx = np.zeros((1, runs * frames_per_run))
    for i in range(len(behavior['onsets'])):
        cue_number = []
        if behavior['cue_codes'][i] == behavior['cs_1_code']:
            cue_number = 1
        if behavior['cue_codes'][i] == behavior['cs_2_code']:
            cue_number = 2
        cue_onset = int(behavior['onsets'][i]) - int(preonset * behavior['framerate'])
        cue_offset = int(behavior['offsets'][i])
        cue_time = cue_offset - cue_onset + 1 + int(offset * behavior['framerate'])
        for j in range(0, cue_time):
            idx = cue_onset + j
            cue_idx[0, idx] = cue_number
    return cue_idx


def moving_average(vec, num):
    """
    calculates moving average
    :param vec: vector
    :param num: frame to average over
    :return: moving average vec
    """
    moving_vec = vec.copy()
    for i in range(0, len(vec)):
        if i >= int(num / 2) and i + int(num / 2) <= len(vec):
            if (num % 2) == 0:
                moving_vec[i] = np.mean(vec[i - int(num / 2):i + int(num / 2)])
            if (num % 2) == 1:
                moving_vec[i] = np.mean(vec[i - int(num / 2):i + int(num / 2) + 1])
    for i in range(0, int(num / 2)):
        moving_vec[i] = np.mean(vec[0:num])
    for i in range(len(vec) - int(num / 2), len(vec)):
        moving_vec[i] = np.mean(vec[len(vec) - num:len(vec) + 1])
    return moving_vec


def get_times_considered(y_pred, behavior):
    """
    get times considered for reactivations
    :param y_pred: y pred
    :param behavior: behavior
    :return: times considered
    """
    times_considered = pd.DataFrame([1] * int(len(y_pred[:, 0])))
    times_considered = filter_cues(behavior, times_considered)
    times_considered[times_considered != times_considered] = 0
    times_considered = np.array(times_considered.iloc[:, 0])
    times_considered = classify.filter_classified(behavior, times_considered, 0)
    times_considered[behavior['relevant_times'] < 1] = 0
    return times_considered


def get_trial_reactivations(y_pred, behavior, trial_type, pupil_norm):
    """
    gets reactivation for each cue type after presentation
    :param y_pred: y_pred
    :param behavior: behavior
    :param trial_type: trial type
    :param pupil_norm: normalize pupil or not
    :return: trial averaged matrix of reactivations
    """
    times_considered = get_times_considered(y_pred, pupil_norm, behavior)

    end_trials = behavior['end_trials']

    fr = behavior['framerate']
    duration = int(fr * (behavior['iti'] + 5)) + 1
    reactivation_data_1 = np.zeros((sum(behavior['cue_codes'] == behavior[trial_type + '_code'])[0] -
                                    sum(behavior['cue_codes'][end_trials] == behavior[trial_type + '_code'])[0],
                                    duration))
    reactivation_data_2 = np.zeros((sum(behavior['cue_codes'] == behavior[trial_type + '_code'])[0] -
                                    sum(behavior['cue_codes'][end_trials] == behavior[trial_type + '_code'])[0],
                                    duration))
    reactivation_times_considered = np.zeros((sum(behavior['cue_codes'] == behavior[trial_type + '_code'])[0] -
                                    sum(behavior['cue_codes'][end_trials] == behavior[trial_type + '_code'])[0],
                                    duration))

    trial_number = 0
    for i in range(0, len(behavior['cue_codes'])):
        if behavior['cue_codes'][i] == behavior[trial_type + '_code'] and i not in end_trials:
            for j in range(0, duration):
                idx = int(behavior['onsets'][i]) + j
                reactivation_data_1[trial_number, j] = y_pred[:, 0][idx]
                reactivation_data_2[trial_number, j] = y_pred[:, 1][idx]
                reactivation_times_considered[trial_number, j] = times_considered[idx]
            trial_number += 1
    return [reactivation_data_1, reactivation_data_2, reactivation_times_considered]


def get_reactivation_times(y_pred, behavior, threshold, paths, returns):
    """
    get time of reactivations
    :param y_pred: regression probabilities
    :param behavior: behavior
    :param threshold: threshold
    :param paths: path to data
    :param returns: return data or not
    :return: reactivation times
    """
    reactivation_times_cs_1 = y_pred[:, 0].copy() - y_pred[:, 1].copy()
    reactivation_times_cs_2 = y_pred[:, 1].copy() - y_pred[:, 0].copy()
    reactivation_times_cs_1[reactivation_times_cs_1 < threshold] = 0
    reactivation_times_cs_2[reactivation_times_cs_2 < threshold] = 0
    for i in range(0, len(reactivation_times_cs_1)):
        if reactivation_times_cs_1[i] > 0:
            for j in range(i + 1, len(reactivation_times_cs_1)):
                if reactivation_times_cs_1[j] == 0:
                    break
                if reactivation_times_cs_1[j] > 0:
                    reactivation_times_cs_1[j] = 0
            for j in range(i + 1, i + 6):
                if reactivation_times_cs_1[j] > 0:
                    reactivation_times_cs_1[j] == 0
        if reactivation_times_cs_2[i] > 0:
            for j in range(i + 1, len(reactivation_times_cs_2)):
                if reactivation_times_cs_2[j] == 0:
                    break
                if reactivation_times_cs_2[j] > 0:
                    reactivation_times_cs_2[j] = 0
            for j in range(i + 1, i + 6):
                if reactivation_times_cs_2[j] > 0:
                    reactivation_times_cs_2[j] == 0
    reactivation_times_cs_1[0:behavior['onsets'][0][0]] = 0
    reactivation_times_cs_2[0:behavior['onsets'][0][0]] = 0
    reactivation_times_cs_1 = np.nonzero(reactivation_times_cs_1)
    reactivation_times_cs_2 = np.nonzero(reactivation_times_cs_2)
    savemat(paths['save_path'] + 'saved_data/reactivation_times.mat',
            {'reactivation_times_cs_1': reactivation_times_cs_1, 'reactivation_times_cs_2': reactivation_times_cs_2})
    if returns == 1:
        return [reactivation_times_cs_1, reactivation_times_cs_2]


def reactivated_cells(y_pred, behavior, deconvolved, idx):
    """
    cell participation in cue and reactivations
    :param y_pred: y pred
    :param behavior: behavior
    :param deconvolved: activity matrix
    :param idx: sorted indices
    :return: vec of participation
    """
    threshold = 0
    reactivation_times = y_pred[:, 0] + y_pred[:, 1]
    reactivation_times[reactivation_times < threshold] = 0
    reactivation_times[reactivation_times >= threshold] = 1
    reactivation_times[0:behavior['onsets'][0][0]] = 0

    cue_time = cue_times(behavior, 0, 0)

    deconvolved = pd.DataFrame(deconvolved)
    deconvolved = deconvolved.reindex(idx.index)

    sorted_deconvolved_reactivation_times = deconvolved.loc[:, reactivation_times == 1]
    sorted_deconvolved_cue_times = deconvolved.loc[:, cue_time[0] > 0]

    reactivation_participation = np.array(sorted_deconvolved_reactivation_times.mean(axis=1)) * behavior['framerate']
    cue_participation = np.array(sorted_deconvolved_cue_times.mean(axis=1)) * behavior['framerate']

    return [reactivation_participation[0, :], cue_participation[0, :]]


def process_plane_activity_R1(paths, activity_type, planes):
    """
    Processes matrix of neural activity of all real cells during task and quiet waking
    :param paths: path to data
    :param activity_type: deconvolved or fluorescence
    :param planes: number of planes
    :param to_delete_save: load to delete or not
    :return: processed fluorescence and events
    """
    upper_plane_cells = []
    lower_plane_cells = []
    for plane in range(1, planes+1):
        plane_path = paths['base_path'] + paths['mouse'] + '/' + paths['date'] + '_' + paths['mouse'] + \
                  '/suite2p_plane_' + str(plane) + '/suite2p/plane0/'
        accepted_cells = np.load(plane_path + 'iscell.npy')[:, 0]
        fluorescence = np.load(plane_path + 'F.npy').sum(axis=1)
        accepted_cells[fluorescence == 0] = 0
        accepted_cells = accepted_cells == 1
        all_activity = np.load(plane_path + activity_type + '.npy')
        if activity_type == 'F':
            all_activity = all_activity - np.load(plane_path + 'Fneu.npy')
        if plane == 1:
            activity = all_activity[accepted_cells, :]
        else:
            activity_plane = all_activity[accepted_cells, :]
            [to_delete, to_keep] = cells_to_delete_R1(paths, plane, plane_path)
            activity_plane = activity_plane[to_delete == 1, :]

            if plane == 2:
                upper_plane_cells = to_keep[~np.isnan(to_keep)]
                upper_plane_cells = np.concatenate((upper_plane_cells, np.array(list(range(len(activity), len(activity)+len(activity_plane))))))
            if plane == 3:
                lower_plane_cells = to_keep[~np.isnan(to_keep)]
                lower_plane_cells = np.concatenate((lower_plane_cells, np.array(list(range(len(activity), len(activity)+len(activity_plane))))))

            activity = np.concatenate((activity, activity_plane))

    return [upper_plane_cells, lower_plane_cells]


def cells_to_delete_R1(paths, plane, plane_path):
    if path.isfile(paths['save_path'] + 'saved_data/to_delete_layer_' + str(plane) + '.npz'):
        temp_file = np.load(paths['save_path'] + 'saved_data/to_delete_layer_' + str(plane) + '.npz')
        to_delete = temp_file['to_delete']
        to_keep = temp_file['to_keep']
    else:
        overlap_cells = loadmat(paths['save_path'] + 'saved_data/overlap_plane_' + str(plane))
        overlap_cells = overlap_cells['overlap_vec']

        accepted_cells = np.load(plane_path + 'iscell.npy')[:, 0]
        fluorescence = np.load(plane_path + 'F.npy').sum(axis=1)
        accepted_cells[fluorescence == 0] = 0
        accepted_cells = accepted_cells == 1
        all_activity = np.load(plane_path + 'F.npy')
        all_activity = all_activity - np.load(plane_path + 'Fneu.npy')
        fluorescence_plane_other = all_activity[accepted_cells, :]

        plane_1_path = paths['base_path'] + paths['mouse'] + '/' + paths['date'] + '_' + paths['mouse'] + \
                       '/suite2p_plane_1/suite2p/plane0/'
        accepted_cells = np.load(plane_1_path + 'iscell.npy')[:, 0]
        fluorescence = np.load(plane_1_path + 'F.npy').sum(axis=1)
        accepted_cells[fluorescence == 0] = 0
        accepted_cells = accepted_cells == 1
        all_activity = np.load(plane_1_path + 'F.npy')
        all_activity = all_activity - np.load(plane_1_path + 'Fneu.npy')
        fluorescence_plane_1 = all_activity[accepted_cells, :]

        r = np.corrcoef(fluorescence_plane_1, fluorescence_plane_other)
        r_idx = r[len(fluorescence_plane_1):len(r), 0:len(fluorescence_plane_1)]
        to_delete = np.ones(len(overlap_cells))
        to_keep = np.empty(len(overlap_cells)) * np.nan
        for i in range(0, len(overlap_cells)):
            if overlap_cells[i, 1] > 0:
                idx_exclude = np.ones(len(fluorescence_plane_1))
                idx_exclude[int(overlap_cells[i, 0]) - 1] = 0
                corr_non = r_idx[i, idx_exclude == 1]
                max_corr_non = np.max(corr_non)
                overlap_corr = r_idx[i, int(overlap_cells[i, 0]) - 1]
                if overlap_corr > max_corr_non:
                    to_delete[i] = 0
                    to_keep[i] = int(overlap_cells[i, 0]) - 1
        np.savez(paths['save_path'] + 'saved_data/to_delete_layer_' + str(plane), to_delete=to_delete, to_keep=to_keep)

    return [to_delete, to_keep]


def normalized_trial_averaged_R3(activity, behavior, trial_type, period):
    """
    makes normalized trial averaged trace for each cue type
    :param activity: fluorescence
    :param behavior: dict of behavior
    :param trial_type: type of trial
    :return: trial averaged trace for given cell
    """
    activity = activity.to_numpy()
    index_frames = []
    if period == 'start':
        num_trials = 0
        total_trials = 10
        trial_times = behavior['onsets'][behavior['cue_codes'] == behavior[trial_type + '_code']]
        for i in trial_times:
            if i > 0 and num_trials < total_trials:
                num_trials = num_trials + 1
                for j in range(-behavior['frames_before'], behavior['frames_after']):
                    index_frames.append(int(i) + j)
    if period == 'end':
        num_trials = 0
        total_trials = 10
        trial_times = behavior['onsets'][behavior['cue_codes'] == behavior[trial_type + '_code']]
        for i in reversed(trial_times):
            if i > 0 and num_trials < total_trials:
                num_trials = num_trials + 1
                for j in range(-behavior['frames_before'], behavior['frames_after']):
                    index_frames.append(int(i) + j)
    activity_task_idx = activity[:, index_frames]
    activity_task_idx = np.reshape(activity_task_idx, (activity_task_idx.shape[0], num_trials,
                                                       behavior['frames_before'] + behavior['frames_after']))
    activity_task_mean = pd.DataFrame(activity_task_idx.mean(axis=1))
    frames_before = behavior['frames_before']
    activity_task_mean_df = activity_task_mean.subtract(
        activity_task_mean.iloc[:, int(frames_before / 2):frames_before].mean(axis=1), axis=0)
    return activity_task_mean_df


def process_activity_across_days_R1(paths, planes, to_delete_save, mouse, date):
    ucids = []
    for plane in range(1, planes+1):
        plane_path = paths['base_path'] + paths['mouse'] + '/' + paths['date'] + '_' + paths['mouse'] + \
                  '/suite2p_plane_' + str(plane) + '/suite2p/plane0/'
        accepted_cells = np.load(plane_path + 'iscell.npy')[:, 0]
        fluorescence = np.load(plane_path + 'F.npy').sum(axis=1)
        accepted_cells[fluorescence == 0] = 0
        accepted_cells = accepted_cells == 1

        with open(paths['base_path'] + '/NN/aligned/plane_' + str(plane) + '/' + mouse + '/ROICaT.tracking.results.stringency_1.3.pkl', 'rb') as f:
            results = pickle.load(f)
        cluster_score = results['cluster_quality_metrics']['cs_sil']
        with open('D:/2p_data/scan/NN/aligned/labelsAndSampleSilhouette.pkl', 'rb') as f:
            labels = pickle.load(f)
            file_path = labels['/media/rich/bigSSD/analysis_data/ROICaT/ROI_tracking/Andermann_lab/Nghia/bigRun_20230507/plane_' + str(plane) + '/' + mouse + '/ROICaT.tracking.rundata.pkl']['ROICaT.tracking.results.stringency_1.3.pkl']
            labels = file_path['labels']['labels_bySession']
            day_len = []
            for i in range(0, len(labels)):
                day_len.append(len(labels[i]))
            sample_sil = file_path['sample_silhouette']
            sil_all = []
            start = 0
            end = day_len[0]
            for i in range(0, len(day_len)):
                sil_all.append(sample_sil[start:end])
                if i < len(day_len) - 1:
                    start += day_len[i]
                    end += day_len[i + 1]
        ucids_all = np.array(labels, dtype=object)[date]

        # ucids_all[sil_all[date] < .4] = -1

        if plane == 1:
            ucids = ucids_all[accepted_cells]
            for i in range(0, len(ucids)):
                if cluster_score[ucids[i]+1] <= .2:
                    ucids[i] = -100000
        else:
            ucids_plane = ucids_all[accepted_cells]
            to_delete = cells_to_delete(paths, plane, plane_path, to_delete_save)
            ucids_plane = ucids_plane[to_delete == 1]
            for i in range(0, len(ucids_plane)):
                if cluster_score[ucids_plane[i]+1] <= .2:
                    ucids_plane[i] = -100000
            ucids_plane += plane*10000
            ucids = np.concatenate((ucids, ucids_plane))

    np.save(paths['save_path'] + 'saved_data/cross_day_alignment', ucids)


def grab_align_cells(paths, idx, day, days):
    days_path = paths['base_path'] + paths['mouse'] + '/data_across_days/'
    if day == 0:
        align_vec = np.load(paths['save_path'] + 'saved_data/cross_day_alignment.npy')
        sig_cells_aligned = pd.DataFrame(align_vec)
        sig_cells_aligned = sig_cells_aligned.reindex(idx['both'].index)
        sig_cells_aligned = sig_cells_aligned.to_numpy()
        sig_cells_aligned = np.concatenate(sig_cells_aligned, axis=0)
        # sig_cells = sig_reactivated_cells([], [], [], [], [], paths, 0)
        # sig_cells_aligned = sig_cells_aligned[sig_cells > 0]
        if days:
            across_days = [list(range(0, days)), list(range(0, days))]
            across_days[0][day] = align_vec
            across_days[1][day] = sig_cells_aligned
            np.save(days_path + 'alignment_across_days', across_days)
    else:
        align_vec = np.load(paths['save_path'] + 'saved_data/cross_day_alignment.npy')
        sig_cells_aligned = pd.DataFrame(align_vec)
        sig_cells_aligned = sig_cells_aligned.reindex(idx['both'].index)
        sig_cells_aligned = sig_cells_aligned.to_numpy()
        sig_cells_aligned = np.concatenate(sig_cells_aligned, axis=0)
        # sig_cells = sig_reactivated_cells([], [], [], [], [], paths, 0)
        # sig_cells_aligned = sig_cells_aligned[sig_cells > 0]
        across_days = np.load(days_path + 'alignment_across_days.npy', allow_pickle=True)
        across_days[0][day] = align_vec
        across_days[1][day] = sig_cells_aligned
        np.save(days_path + 'alignment_across_days', across_days)


def align_cells(paths):
    days = 6
    days_path = paths['base_path'] + paths['mouse'] + '/data_across_days/'
    align_vec = np.load(days_path + 'alignment_across_days.npy', allow_pickle=True)
    intersec_vec = []
    for i in range(0, days):
        for j in range(0, len(align_vec[1][i])):
            intersec_vec.append(align_vec[1][i][j])
    intersec_vec = np.unique(intersec_vec)

    for i in range(0, days):
        temp_vec = align_vec[0][i]
        for j in range(0, len(intersec_vec)):
            if intersec_vec[j] not in temp_vec:
                intersec_vec[j] = -10000
    intersec_vec = intersec_vec[intersec_vec >= 0]

    for i in range(0, days):
        temp_vec = align_vec[0][i]
        for j in range(0, len(temp_vec)):
            if temp_vec[j] not in intersec_vec:
                temp_vec[j] = -10000
        align_vec[0][i] = temp_vec
    np.save(days_path + 'alignment_across_days_intersect', align_vec)

    days = 6
    days_path = paths['base_path'] + paths['mouse'] + '/data_across_days/'
    align_vec = np.load(days_path + 'alignment_across_days.npy', allow_pickle=True)
    intersec_vec = []
    for i in range(0, days):
        for j in range(0, len(align_vec[0][i])):
            intersec_vec.append(align_vec[0][i][j])
    intersec_vec = np.unique(intersec_vec)

    for i in range(0, days):
        temp_vec = align_vec[0][i]
        for j in range(0, len(intersec_vec)):
            if intersec_vec[j] not in temp_vec:
                intersec_vec[j] = -10000
    intersec_vec = intersec_vec[intersec_vec >= 0]

    for i in range(0, days):
        temp_vec = align_vec[0][i]
        for j in range(0, len(temp_vec)):
            if temp_vec[j] not in intersec_vec:
                temp_vec[j] = -10000
        align_vec[0][i] = temp_vec
    np.save(days_path + 'alignment_across_days_intersect_all', align_vec)


def no_change_decrease_neurons_novelty_R1(activity, behavior, decrease_sig_cells):

    index_frames_start_cs_1 = []
    index_frames_end_cs_1 = []
    index_frames_start_cs_2 = []
    index_frames_end_cs_2 = []
    num_trials_total = 10
    num_trials = 0
    trial_times_cs_1 = behavior['onsets'][behavior['cue_codes'] == behavior['cs_1_code']]
    trial_times_cs_2 = behavior['onsets'][behavior['cue_codes'] == behavior['cs_2_code']]
    for i in trial_times_cs_1:
        if i > 0 and num_trials < num_trials_total:
            for j in range(0, behavior['frames_before']):
                index_frames_start_cs_1.append(int(i) + j)
            num_trials = num_trials + 1
    num_trials = 0
    for i in reversed(trial_times_cs_1):
        if i > 0 and num_trials < num_trials_total:
            for j in range(0, behavior['frames_before']):
                index_frames_end_cs_1.append(int(i) + j)
            num_trials = num_trials + 1
    num_trials = 0
    for i in trial_times_cs_2:
        if i > 0 and num_trials < num_trials_total:
            for j in range(0, behavior['frames_before']):
                index_frames_start_cs_2.append(int(i) + j)
            num_trials = num_trials + 1
    num_trials = 0
    for i in reversed(trial_times_cs_2):
        if i > 0 and num_trials < num_trials_total:
            for j in range(0, behavior['frames_before']):
                index_frames_end_cs_2.append(int(i) + j)
            num_trials = num_trials + 1

    activity_start_cs_1 = activity[:, index_frames_start_cs_1]
    activity_start_cs_1 = np.reshape(activity_start_cs_1, (activity_start_cs_1.shape[0], num_trials_total, behavior['frames_before']))
    activity_end_cs_1 = activity[:, index_frames_end_cs_1]
    activity_end_cs_1 = np.reshape(activity_end_cs_1, (activity_end_cs_1.shape[0], num_trials_total, behavior['frames_before']))
    activity_start_cs_2 = activity[:, index_frames_start_cs_2]
    activity_start_cs_2 = np.reshape(activity_start_cs_2,
                                     (activity_start_cs_2.shape[0], num_trials_total, behavior['frames_before']))
    activity_end_cs_2 = activity[:, index_frames_end_cs_2]
    activity_end_cs_2 = np.reshape(activity_end_cs_2,
                                   (activity_end_cs_2.shape[0], num_trials_total, behavior['frames_before']))

    no_change_decrease_cells = np.zeros(len(decrease_sig_cells))
    for j in range(len(activity_start_cs_1)):
        before_cs_1 = np.reshape(activity_start_cs_1[j, :, 0:behavior['frames_before']],
                                 (num_trials * behavior['frames_before']))
        after_cs_1 = np.reshape(activity_end_cs_1[j, :, 0:behavior['frames_before']],
                                (num_trials * behavior['frames_before']))
        before_cs_2 = np.reshape(activity_start_cs_2[j, :, 0:behavior['frames_before']],
                                 (num_trials * behavior['frames_before']))
        after_cs_2 = np.reshape(activity_end_cs_2[j, :, 0:behavior['frames_before']],
                                (num_trials * behavior['frames_before']))

        res = stats.ttest_rel(before_cs_1 - after_cs_1, before_cs_2 - after_cs_2)
        if res.pvalue > .05:
            no_change_decrease_cells[j] = 1

        vec_1 = (before_cs_1 - after_cs_1)/(before_cs_1)
        vec_1[np.isnan(vec_1)] = 0
        vec_2 = (before_cs_2 - after_cs_2) / (before_cs_2)
        vec_2[np.isnan(vec_2)] = 0
        res = stats.ttest_rel(vec_1, vec_2)
        if res.pvalue > .05:
            no_change_decrease_cells[j] = 1

    return no_change_decrease_cells


def sig_test_R2(activity, behavior, trial_type, period):
    """
    sig test for cue cells
    :param activity: fluorescence
    :param behavior: dict of behavior
    :param trial_type: trial type
    :return: which cells are sig for which cue(s)
    """
    activity = activity.to_numpy()
    frames_before = behavior['frames_before']
    index_frames = []
    if period == 'start':
        num_trials = 0
        total_trials = 10
        trial_times = behavior['onsets'][behavior['cue_codes'] == behavior[trial_type + '_code']]
        for i in trial_times:
            if i > 0 and num_trials < total_trials:
                num_trials = num_trials + 1
                for j in range(-frames_before, frames_before):
                    index_frames.append(int(i) + j)
    if period == 'end':
        num_trials = 0
        total_trials = 10
        trial_times = behavior['onsets'][behavior['cue_codes'] == behavior[trial_type + '_code']]
        for i in reversed(trial_times):
            if i > 0 and num_trials < total_trials:
                num_trials = num_trials + 1
                for j in range(-frames_before, frames_before):
                    index_frames.append(int(i) + j)
    activity_idx = activity[:, index_frames]
    activity_idx = np.reshape(activity_idx, (activity_idx.shape[0], num_trials, frames_before * 2))
    pos_sig_cells = np.zeros(len(activity_idx))
    neg_sig_cells = np.zeros(len(activity_idx))
    for j in range(len(activity_idx)):
        before = np.reshape(activity_idx[j, :, 0:frames_before], (num_trials * frames_before))
        after = np.reshape(activity_idx[j, :, frames_before:frames_before * 2], (num_trials * frames_before))
        res = stats.ranksums(before, after)
        if res.statistic < 0 and res.pvalue < .05:
            pos_sig_cells[j] = 1
        if res.statistic > 0 and res.pvalue < .05:
            neg_sig_cells[j] = 1
    return [pos_sig_cells, neg_sig_cells]


def selectivity_grouped(activity, behavior, sig_cells):

    index_frames_start_cs_1 = []
    index_frames_end_cs_1 = []
    index_frames_start_cs_2 = []
    index_frames_end_cs_2 = []
    num_trials_total = 10
    num_trials = 0
    trial_times_cs_1 = behavior['onsets'][behavior['cue_codes'] == behavior['cs_1_code']]
    trial_times_cs_2 = behavior['onsets'][behavior['cue_codes'] == behavior['cs_2_code']]
    for i in trial_times_cs_1:
        if i > 0 and num_trials < num_trials_total:
            for j in range(0, behavior['frames_before']):
                index_frames_start_cs_1.append(int(i) + j)
            num_trials = num_trials + 1
    num_trials = 0
    for i in reversed(trial_times_cs_1):
        if i > 0 and num_trials < num_trials_total:
            for j in range(0, behavior['frames_before']):
                index_frames_end_cs_1.append(int(i) + j)
            num_trials = num_trials + 1
    num_trials = 0
    for i in trial_times_cs_2:
        if i > 0 and num_trials < num_trials_total:
            for j in range(0, behavior['frames_before']):
                index_frames_start_cs_2.append(int(i) + j)
            num_trials = num_trials + 1
    num_trials = 0
    for i in reversed(trial_times_cs_2):
        if i > 0 and num_trials < num_trials_total:
            for j in range(0, behavior['frames_before']):
                index_frames_end_cs_2.append(int(i) + j)
            num_trials = num_trials + 1

    activity_start_cs_1 = activity[:, index_frames_start_cs_1]
    activity_start_cs_1 = np.reshape(activity_start_cs_1, (activity_start_cs_1.shape[0], num_trials_total, behavior['frames_before']))
    activity_end_cs_1 = activity[:, index_frames_end_cs_1]
    activity_end_cs_1 = np.reshape(activity_end_cs_1, (activity_end_cs_1.shape[0], num_trials_total, behavior['frames_before']))
    activity_start_cs_2 = activity[:, index_frames_start_cs_2]
    activity_start_cs_2 = np.reshape(activity_start_cs_2,
                                     (activity_start_cs_2.shape[0], num_trials_total, behavior['frames_before']))
    activity_end_cs_2 = activity[:, index_frames_end_cs_2]
    activity_end_cs_2 = np.reshape(activity_end_cs_2,
                                   (activity_end_cs_2.shape[0], num_trials_total, behavior['frames_before']))

    import scipy
    def statistic(x, y, a, b):
        return np.abs((np.mean(x)-np.mean(y))/(np.mean(x)+np.mean(y))) - np.abs((np.mean(a)-np.mean(b))/(np.mean(a)+np.mean(b)))

    sig_cells_vec = np.zeros(len(sig_cells))
    for j in range(len(activity_start_cs_1)):
        if sig_cells[j] == 1:
            before_cs_1 = np.reshape(activity_start_cs_1[j, :, 0:behavior['frames_before']],
                                     (num_trials * behavior['frames_before']))
            after_cs_1 = np.reshape(activity_end_cs_1[j, :, 0:behavior['frames_before']],
                                    (num_trials * behavior['frames_before']))
            before_cs_2 = np.reshape(activity_start_cs_2[j, :, 0:behavior['frames_before']],
                                     (num_trials * behavior['frames_before']))
            after_cs_2 = np.reshape(activity_end_cs_2[j, :, 0:behavior['frames_before']],
                                    (num_trials * behavior['frames_before']))

            res = scipy.stats.permutation_test((before_cs_1, before_cs_2, after_cs_1, after_cs_2), statistic,
                                               n_resamples=1000, permutation_type='samples')

            if res.pvalue > .05:
                sig_cells_vec[j] = 1
            if res.statistic < 0 and res.pvalue < .05:
                sig_cells_vec[j] = 2
            if res.statistic > 0 and res.pvalue < .05:
                sig_cells_vec[j] = 3

    return sig_cells_vec








