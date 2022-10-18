import os
import math
import warnings
import classify_opto
import numpy as np
import pandas as pd
from os import path
from os import makedirs
from scipy import stats
from scipy.io import loadmat
from scipy.io import savemat
from scipy.ndimage import gaussian_filter1d
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


def process_behavior(paths):
    """
    Processes behavioral data for mouse and date
    :param paths: path to data
    :return: processed fluorescence and events
    """
    session_data = loadmat(paths['save_path'] + 'saved_data/behavior_file.mat')
    onsets = session_data['onsets'].astype(int) - 1
    offsets = session_data['offsets'].astype(int) - 1
    opto_onsets = session_data['opto_onsets'].astype(int) - 1
    opto_length = int(session_data['opto_length'])
    cue_codes = session_data['cue_code']
    cs_1_code = int(session_data['CS_1_code'])
    cs_2_code = int(session_data['CS_2_code'])
    cs_1_opto_code = int(session_data['CS_1_code_opto'])
    cs_2_opto_code = int(session_data['CS_2_code_opto'])
    task_runs = int(session_data['task_runs'])
    dark_runs = int(session_data['dark_runs'])
    unique, counts = np.unique(cue_codes, return_counts=True)
    num_cs_1_opto_trials = counts[0]
    num_cs_1_trials = counts[1]
    num_cs_2_opto_trials = counts[2]
    num_cs_2_trials = counts[3]
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
        start_2 = int(offsets[end_trials[i]]) + int(np.round(framerate/4))
        end_2 = (frames_per_run*dark_runs) + ((i+1)*frames_per_run)
        task_times[start_1:end_1] = 0
        task_times[start_2:end_2] = 0
        task_times[0:frames_per_run*dark_runs] = 0
        relevant_times[start_2:end_2] = 0
    temp_pupil = np.sort(pupil[relevant_times == 1])
    pupil_max = np.mean(temp_pupil[len(temp_pupil)-int(len(pupil)*.01):len(temp_pupil)])
    pupil_min = np.mean(temp_pupil[0:int(len(pupil)*.01)])
    licking = session_data['licking'][0]
    licking[licking > .1] = 1
    licking[licking < .1] = 0
    if len(licking) == 1:
        licking = np.zeros((1, len(pupil)))
    cue_offset = 6
    opto_offset = .3
    behavior = {'onsets': onsets, 'offsets': offsets, 'cue_codes': cue_codes, 'cs_1_code': cs_1_code,
                'cs_2_code': cs_2_code, 'licking': licking, 'running': running, 'pupil': pupil,
                'cs_1_trials': num_cs_1_trials, 'cs_2_trials': num_cs_2_trials, 'frames_before': frames_before,
                'frames_after': frames_after, 'framerate': framerate, 'frames_per_run': frames_per_run,
                'task_runs': task_runs, 'iti': iti, 'dark_runs': dark_runs, 'brain_motion': brain_motion,
                'pupil_max': pupil_max, 'pupil_min': pupil_min, 'task_times': task_times,
                'relevant_times': relevant_times, 'cue_offset': cue_offset, 'pupil_movement': pupil_movement,
                'end_trials': end_trials, 'start_trials': start_trials, 'cs_1_opto_code': cs_1_opto_code,
                'cs_2_opto_code': cs_2_opto_code, 'opto_onsets': opto_onsets, 'opto_length': opto_length,
                'opto_offset': opto_offset, 'cs_1_opto_trials': num_cs_1_opto_trials,
                'cs_2_opto_trials': num_cs_2_opto_trials}
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
    make cell masks and save as matlab for cellreg
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
    :param to_delete_save: reload or not to delete vector
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
    moving average of deconvolved
    :param deconvolved_vec:  dark or task events vector
    :param behavior: behavior
    :param paths: path to data
    :param save: overwrite saved filed or not
    :return: 4 frame moving max of dark events
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
        opto_times_vec = opto_times(behavior, behavior['opto_offset'], 0)
        times_to_use = behavior['relevant_times'].copy()
        times_to_use[cue_times_vec[0, :] > 0] = 0
        times_to_use[opto_times_vec[0, :] > 0] = 0
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
    sig test for cells
    :param activity: fluorescence
    :param behavior: dict of behavior
    :param trial_type: trial type
    :return: sig cells
    """
    activity = activity.to_numpy()
    frames_before = behavior['frames_before']
    index_frames = []
    num_trials = 0
    trial_times = behavior['onsets'][behavior['cue_codes'] == behavior[trial_type + '_code']]
    for i in trial_times:
        if i > 0:
            num_trials = num_trials + 1
            for j in range(-behavior['frames_before'], behavior['frames_before']):
                index_frames.append(int(i) + j)
    activity_idx = activity[:, index_frames]
    activity_idx = np.reshape(activity_idx, (activity_idx.shape[0], num_trials,
                                             behavior['frames_before'] + behavior['frames_before']))
    pos_sig_cells = np.zeros(len(activity_idx))
    neg_sig_cells = np.zeros(len(activity_idx))
    for j in range(len(activity_idx)):
        before = np.reshape(activity_idx[j, :, 0:frames_before], (num_trials * behavior['frames_before']))
        after = np.reshape(activity_idx[j, :, frames_before:frames_before * 2],
                           (num_trials * behavior['frames_before']))
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


def filter_opto(behavior, vector):
    """
    filter opto times
    :param behavior: behavior
    :param vector: activity vector
    :return: cue filtered
    """
    cue_idx = opto_times(behavior, behavior['opto_offset'], 0)
    vector[cue_idx[0, :] > 0] = float("nan")
    return vector


def cue_times(behavior, offset, preonset):
    """
    get cue times
    :param behavior: behavior
    :param offset: how much after cue to include
    :param preonset: how much before cue to include
    :return:
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
        if behavior['cue_codes'][i] == behavior['cs_1_opto_code']:
            cue_number = 3
        if behavior['cue_codes'][i] == behavior['cs_2_opto_code']:
            cue_number = 4
        cue_onset = int(behavior['onsets'][i]) - int(preonset * behavior['framerate'])
        cue_offset = int(behavior['offsets'][i])
        cue_time = cue_offset - cue_onset + 1 + int(offset * behavior['framerate'])
        for j in range(0, cue_time):
            idx = cue_onset + j
            cue_idx[0, idx] = cue_number
    return cue_idx


def opto_times(behavior, offset, preonset):
    """
    get opto times
    :param behavior: behavior
    :param offset: how much after opto to include
    :param preonset: how much before opto to include
    :return:
    """
    runs = int(behavior['task_runs']) + int(behavior['dark_runs'])
    frames_per_run = int(behavior['frames_per_run'])
    opto_idx = np.zeros((1, runs * frames_per_run))
    for i in range(len(behavior['opto_onsets'])):
        if behavior['opto_onsets'][i] > 0:
            cue_number = []
            if behavior['cue_codes'][i] == behavior['cs_1_opto_code']:
                cue_number = 3
            if behavior['cue_codes'][i] == behavior['cs_2_opto_code']:
                cue_number = 4
            opto_onset = int(behavior['opto_onsets'][i]) - int(preonset * behavior['framerate'])
            opto_time = int(behavior['opto_length'] * behavior['framerate']) + 1 + int(offset * behavior['framerate'])
            for j in range(0, opto_time):
                idx = opto_onset + j
                opto_idx[0, idx] = cue_number
    return opto_idx


def moving_average(vec, num):
    """
    moving average
    :param vec: vec
    :param num: frames
    :return: moving average
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
    get times considered
    :param y_pred: y pred
    :param behavior: behavior
    :return: times considered
    """
    times_considered = pd.DataFrame([1] * int(len(y_pred[:, 0])))
    times_considered = filter_cues(behavior, times_considered)
    times_considered = filter_opto(behavior, times_considered)
    times_considered[times_considered != times_considered] = 0
    times_considered = np.array(times_considered.iloc[:, 0])
    times_considered = classify_opto.filter_classified(behavior, times_considered, 0)
    times_considered[behavior['relevant_times'] < 1] = 0
    return times_considered


def get_trial_reactivations(y_pred, behavior, trial_type, pupil_norm):
    """
    gets reactivation for each cue type
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


def opto_cells(planes, paths):
    """
    get cells to remove from opto blanking
    :param planes: planes
    :param paths: path
    :return: which opto cells to remove
    """
    stat = []
    for plane in range(1, planes + 1):
        plane_path = paths['base_path'] + paths['mouse'] + '/' + paths['date'] + '_' + paths['mouse'] + \
                     '/suite2p_plane_' + str(plane) + '/suite2p/plane0/'
        accepted_cells = np.load(plane_path + 'iscell.npy')[:, 0]
        fluorescence = np.load(plane_path + 'F.npy').sum(axis=1)
        accepted_cells[fluorescence == 0] = 0
        accepted_cells = accepted_cells == 1
        stat_plane = np.load(plane_path + 'stat.npy', allow_pickle=True)
        if plane == 1:
            stat = stat_plane[accepted_cells]
        else:
            stat_plane = stat_plane[accepted_cells]
            to_delete = cells_to_delete(paths, plane, plane_path, 0)
            stat_plane = stat_plane[to_delete == 1]
            stat = np.concatenate((stat, stat_plane))
    opto_cells_to_remove = np.zeros(len(stat))
    for n in range(0, len(stat)):
        ypix = np.min(stat[n]['ypix'][~stat[n]['overlap']])
        if ypix < 94:
            opto_cells_to_remove[n] = 1
    return opto_cells_to_remove
