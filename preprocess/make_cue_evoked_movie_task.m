%%for both cue types
cue_types = [3 8];
cue_names = {'CS_1','CS_2'};
mean_img_vec = {};
movie_vecs = {};
for ct = 1:size(cue_types,2)
    % make trial averaged cue evoked movie
    planes = 3;
    framerate = 31.25/planes;
    frames_before = round(framerate*2);
    frames_after = round(framerate*8);
    % get movie
    movie_vec = zeros(size(regmovie,1),size(regmovie,2),frames_before+frames_after+1);
    cue_onset = onsets(cue_code == cue_types(ct)); %%%%%%%%%%%%%%%%%%%%
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
%     cd (['D:\2p_data\scan\',mouse,'\',date,'_',mouse,'\processed_data\movies'])
%     movie_vec_dff(746:781,40:75,frames_before+1:frames_before*2+1) = 100000;
    pipe.io.write_tiff(imrotate(flip(movie_vec_dff),-90),[cue_names{ct},'_cue_trial_averaged_df']);
end
%% make cue evoked mean image difference
mean_img_diff = mean_img_vec{1} - mean_img_vec{2};
figure
set(gcf,'position',[50 50 796 512]); set(gcf,'color',[1 1 1])
imagesc(mean_img_diff,[-1000 1000]);
%a = colorbar;
%a.Label.String = 'DF/F';
%cMap = interp1([0;1],[1 .5 .4; .235 .7 .44],linspace(0,1,256));
%colormap(cMap)
colormap(gray);
box off; set(gca,'xtick',[]); set(gca,'ytick',[]); set(gca,'FontSize',17)
%set(gcf,'papersize',[7.96 512]); set(gcf,'PaperPosition',[0 0 11 6.75]);
saveas(gcf,'stimulus_difference','pdf');
% %% make overlaid mean images
% figure
% C = imfuse(mean_img_vec{1},mean_img_vec{2}, 'ColorChannels', [1 2 0], 'scaling','none');
% imshow(imgaussfilt(imadjust(C,[0 0 0; .4 .4 .5],[]),1))
% box off; set(gca,'xtick',[]); set(gca,'ytick',[]); set(gca,'FontSize',17)
% set(gcf,'papersize',[12 8]); set(gcf,'PaperPosition',[0 0 11 6.75]);
% saveas(gcf,'mean_cue_img_overlay','pdf'); 
%  %% make temporal images
% for ct = 1:size(cue_types,2)
%     [M,I] = max(movie_vecs{ct},[],3); I = rot90(I); I = flip(I);
%     figure
%     set(gcf,'position',[50 50 796 512]); set(gcf,'color',[1 1 1])
%     imagesc(imgaussfilt(I,1),[frames_before frames_before*2.2]);
%     colormapNghia
%     colorbar
%     box off; set(gca,'xtick',[]); set(gca,'ytick',[]); set(gca,'FontSize',17)
%     saveas(gcf,[cue_names{ct},'_temporal_map'],'pdf');
% end
close all