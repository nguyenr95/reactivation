%%for both cue types
cue_types_1 = [4 3];
cue_types_2 = [8 7];
cue_names = {'Normal','Opto'};
mean_img_vec = {};
movie_vecs = {};
for ct = 1:size(cue_types_1,2)
    % make trial averaged cue evoked movie
    planes = 3;
    framerate = 31.25/planes;
    frames_before = round(framerate*2);
    frames_after = round(framerate*8);
    % get movie
    movie_vec = zeros(size(regmovie,1),size(regmovie,2),frames_before+frames_after+1);
    cue_onset = onsets(cue_code == cue_types_1(ct) | cue_code == cue_types_2(ct)); %%%%%%%%%%%%%%%%%%%%
    for i = 1:size(cue_onset,1)
        t = cue_onset(i);
        movie_vec = movie_vec + double(regmovie(:,:,t-frames_before:t+frames_after));
    end
    movie_vec = movie_vec/size(cue_onset,1);
    % make dff movie
    mean_before = mean(movie_vec(:,:,1:frames_before),3);
    movie_vec_dff = zeros(size(regmovie,1),size(regmovie,2),frames_before+frames_after+1);
    for i = 1:size(movie_vec,3)
        movie_vec_dff(:,:,i) = (movie_vec(:,:,i) - mean_before); %./mean_before;
    end
    % put into vector
    mean_img_vec{ct} = mean(movie_vec_dff(:,:,frames_before+1:frames_before*2),3)';
    movie_vecs{ct} = movie_vec_dff;
    %save as tiff
    cd (['D:\2p_data\scan\',mouse,'\',date,'_',mouse,'\processed_data\movies'])
%     if ct == 2
%        movie_vec_dff(100:135,100:135,frames_before+3-round(frames_before/2):frames_before*3+3-round(frames_before/2)) = 100000;
%     end
    movie_vec_dff(100:135,150:185,frames_before+1:frames_before*2+1) = 100000;
    pipe.io.write_tiff(imrotate(flip(movie_vec_dff),-90),[cue_names{ct},'_cue_trial_averaged_df']);
end