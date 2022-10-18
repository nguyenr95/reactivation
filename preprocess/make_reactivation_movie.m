%% get movie around reactivation
format long g
load(['D:\2p_data\scan\',mouse,'\',date,'_',mouse,'\processed_data\saved_data\reactivation_times.mat'])
framerate = 31.25/planes;
frames_before = round(framerate*2);
frames_after = round(framerate*3);
reactivation_times_vec = {reactivation_times_cs_1, reactivation_times_cs_2};
cue_names = {'CS_1','CS_2'};
for cue = 1:2
    reactivation_dff_all = zeros(size(regmovie,1),size(regmovie,2),frames_before+frames_after+1);
    reactivation_times = reactivation_times_vec{1,cue};
    for r = 1:size(reactivation_times,2)
        % get movie reactivtion
        time = reactivation_times(r);
        reactivation = regmovie(:,:,time-frames_before:time+frames_after);
        reactivation = movmean(reactivation,round(framerate/2),3);
        % make dff movie
        reactivation_dff = zeros(size(regmovie,1),size(regmovie,2),frames_before+frames_after+1);
        meanbefore = mean(reactivation(:,:,1:frames_before),3);
        for i = 1:size(reactivation,3)
            reactivation_dff(:,:,i) = (reactivation(:,:,i) - meanbefore);
        end  
        % save dff movie
        % cd (['D:\2p_data\scan\',mouse,'\',date,'_',mouse,'\suite2p_plane_1\suite2p\plane0\movies\reactivation'])
        % reactivation_dff(746:781,40:75,frames_before+1:frames_before+8) = 1000000;
        % pipe.io.write_tiff(imrotate(flip(reactivation_dff),-90),[cue_names{1,cue},'_sample_reactivation_dff_',num2str(r)]);
        % make mean
        reactivation_dff_all = reactivation_dff_all + reactivation_dff;
    end
    reactivation_dff_all = reactivation_dff_all/size(reactivation_times,2);
    % save averaged reactivation
    cd (['D:\2p_data\scan\',mouse,'\',date,'_',mouse,'\processed_data\movies\reactivation'])
    pipe.io.write_tiff(imrotate(flip(reactivation_dff_all),-90),[cue_names{1,cue},'_mean_reactivation_df']);
end
