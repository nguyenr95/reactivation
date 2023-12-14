%% process photometry data
% fluorescence
fluorescence = movmean(zscore(data(3,:)),frequency*5);
% cue times
onsets = data(5,:);
onsets = diff(onsets);
onsets = find(onsets > 1);
% running
running = data(4,:);
running = diff(running);
running = find(running > 4);
speed = speed(1:size(running,2));
running_final = zeros(1,size(data,2));
running_final(running) = speed;
%running_final = movmean(running_final,frequency);
% plot fluorescence vs running
x = 1:1:size(data,2);
x = x/frequency/60;
fig = figure;
right_color = [0 .6 0];
left_color = [0 0 0];
set(fig,'defaultAxesColorOrder',[left_color; right_color]);
yyaxis right
plot(x, fluorescence, 'Color', right_color, 'LineWidth', 1.5);
ylabel('GRAB-NE fluorescence (SD)', 'Color', right_color);
yyaxis left
plot(x, running_final, 'Color', left_color, 'LineWidth', 1.5);
xlabel('Time (minutes)');
ylabel('Running (cm/s)', 'Color', left_color);
xlim([0 x(size(x,2))]);
box off
set(gcf,'Color','w');
set(gca,'FontSize',17);
set(gcf,'papersize',[12 8])
set(gcf, 'PaperPosition', [0 2 10 4.5]);
saveas(gcf,'Running_vs_NE.pdf')
%% make trial averaged cue response
cue_response = {};
for i = 1:size(onsets,2)
    cue_response{i} = fluorescence(onsets(i)-(frequency*2):onsets(i)+(frequency*8)) - mean(fluorescence(onsets(i)-frequency*2:onsets(i)-1));
end
cue_response = cell2mat(cue_response');
figure; imagesc(cue_response);
    
