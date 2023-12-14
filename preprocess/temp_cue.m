%% for both cue types
planes = 3;
framerate = 31.25/planes;
frames_before = round(framerate*2);
frames_after = round(framerate*8);
% get movie
movie_vec = zeros(size(regmovie,1),size(regmovie,2),frames_before+frames_after+1);
cue_onset = bdata.onsets;
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
%save as tiff
movie_vec_dff(100:135,150:185,frames_before+1:frames_before*2+1) = 100000;
pipe.io.write_tiff(movie_vec_dff,'dff');



%%
