%%for both cue types
cue_types = [5 12];
cue_names = {'No_opto','Opto'};
for ct = 1:size(cue_types,2)
    % make trial averaged cue evoked movie
    framerate = 10.42;
    frames_before = round(framerate*4);
    frames_after = round(framerate*8);
    % get movie
    movie_vec = zeros(size(regmovie,1),size(regmovie,2),frames_before+frames_after+1);
    if ct == 1
        cue_onset = bdata.onsets(bdata.condition == cue_types(ct)); 
    end
    if ct == 2
        cue_onset = bdata.opto_onsets(bdata.condition == cue_types(ct)); 
    end
    for i = 1:size(cue_onset,1)
        t = double(cue_onset(i));
        movie_vec = movie_vec + double(regmovie(:,:,t-frames_before:t+frames_after));
    end
    movie_vec = movie_vec/size(cue_onset,1);
    % make dff movie
    mean_before = mean(movie_vec(:,:,1:frames_before),3);
    movie_vec_dff = zeros(size(regmovie,1),size(regmovie,2),frames_before+frames_after+1);
    for i = 1:size(movie_vec,3)
        movie_vec_dff(:,:,i) = (movie_vec(:,:,i) - mean_before); % ./mean_before;
    end
    %save as tiff
    movie_vec_dff(40:75,40:75,round(framerate*3):round(framerate*7)) = 100000;
    movie_vec_dff(100:135,40:75,round(framerate*4):round(framerate*6)) = 100000;
    pipe.io.write_tiff(movie_vec_dff,[cue_names{ct},'_cue_trial_averaged_df']);
end

%%
%%for both cue types
cue_types = [5 13];
c = ['k', 'r'];
for ct = 1:size(cue_types,2)
    % make trial averaged cue evoked movie
    framerate = 15.63;
    frames_before = round(framerate*4);
    frames_after = round(framerate*8);
    % plot
    cue_onset = onsets(cue_code == cue_types(ct)); %%%%%%%%%%%%%%%%%%%%
    for i = 1:size(cue_onset,1)
        if ct == 1
            t = cue_onset(i);
        end
        if ct == 2
            t = cue_onset(i) + round(framerate);
        end
        vec = (F(t-frames_before:t+frames_after) - mean(F(t-frames_before:t))) ./ mean(F(t-frames_before:t));
        x = -frames_before:frames_after;
        x = x/15.63;
        plot(x, vec, 'color', c(ct), 'LineWidth', 2);
        hold on
    end
end
box off
set(gcf, 'color', 'w')
yLim = get(gca,'YLim');
p = patch([round(framerate*-1)/15.63 round(framerate*3)/15.63 round(framerate*3)/15.63 round(framerate*-1)/15.63]...
,[yLim(1) yLim(1) yLim(2) yLim(2)],'r','LineStyle','none');
alpha(p,.3)
hold on
p = patch([round(framerate*0)/15.63 round(framerate*2)/15.63 round(framerate*2)/15.63 round(framerate*0)/15.63]...
,[yLim(1) yLim(1) yLim(2) yLim(2)],'b','LineStyle','none');
alpha(p,.3)
ylabel('GCaMP7s DF/F')
xlabel('Time from cue onset (s)')
set(gca, 'Fontsize', 20)
xlim([-4, 8])


%%
framerate = 10.42;
frames_before = round(framerate*15);
frames_after = round(framerate*65);
% plot
cue_onset = opto_onset; 
c = 1;
for i = 1:size(cue_onset,2)
    t = cue_onset(i);
    vec = (F(t-frames_before:t+frames_after) - mean(F(t-frames_before:t))) ./ mean(F(t-frames_before:t));
    x = -frames_before:frames_after;
    x = x/framerate;
    plot(x, vec, 'color', [c 0 0], 'LineWidth', 2);
    hold on
    c = c-.2;
end

box off
set(gcf, 'color', 'w')
yLim = get(gca,'YLim');
p = patch([round(framerate*0)/framerate round(framerate*49)/framerate round(framerate*49)/framerate round(framerate*0)/framerate]...
,[yLim(1) yLim(1) yLim(2) yLim(2)],'r','LineStyle','none');
alpha(p,.3)
ylabel('jCaMP7s DF/F')
xlabel('Time from opto onset (s)')
set(gca, 'Fontsize', 15)
xlim([-15, 65])


