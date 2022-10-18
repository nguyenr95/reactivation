%% Plot Licking Across Trials For Each Trial Type

%% Licking vector
lickingvec = zeros(1,size(bdata.running,2));
lickingvec(bdata.licking) = -1;

%% Ensure vector
ensurevec = zeros(1,size(bdata.running,2));
ensurevec(bdata.ensure(bdata.ensure>0)) = 2;

pl = 1;
cl = {[0 1 0],[1 0 0],[0 .5 1]};
ttl = {'Plus Cue','Neutral Cue 1','Neutral Cue 2'};
idx = 0;
for cue = [3 8]
idx = idx+1;
%% Make cue alignment vec for cue
totaltrials = size(bdata.onsets,1);
onset = bdata.onsets(1:totaltrials);
condition = bdata.condition(1:totaltrials);

cueonset = onset(condition == cue);

t = round(bdata.framerate)+1; %~16fps
t2 = round(bdata.framerate*2); %31 fp2s

licking = {};
ensure = {};
for i = 1:size(cueonset,1)
    time = cueonset(i,1);
    if size(lickingvec,2) >= time+(t*10)
    templick = lickingvec(:,time-t:time+(t*10));
    licking{i,1} = templick;
    tempensure = ensurevec(:,time-t:time+(t*10));
    ensure{i,1} = tempensure;
    end
end
licking = cell2mat(licking);
ensure = cell2mat(ensure);

%% plot licking and ensure times
if pl == 1
        figure('Position', [0 200 2000 700])
end

subplot(2,3,pl)
imagesc(licking+ensure,[-1 1]);
set(gca,'FontSize',17);
colormapNghia
line([t+1 t+1],[0 1000],'Color','k','LineStyle','--');
line([t+1+t2 t+1+t2],[0 1000],'Color','k','LineStyle','--');
xticks([t+1 t+1+t2]);
xticklabels({'Cue ON','Cue OFF'});
xtickangle(45)
ylabel('Trial #')
title(ttl{idx})
set(gcf,'color','w');
subplot(2,3,pl+3)
x = 1:1:t*11+1;
licking = licking*-1;
errorbar_shade(x,movmean(mean(licking),8),movmean(std(licking)/sqrt(size(licking,1)),8),cl{idx});
line([t+1 t+1],[0 .6],'Color','k','LineStyle','--');
line([t+1+t2 t+1+t2],[0 .6],'Color','k','LineStyle','--');
xticks([t+1 t+1+t2]);
xticklabels({'Cue ON','Cue OFF'});
xtickangle(45)
ylabel('Lick Frequency')
ylim([0 .6])
set(gcf,'color','w');
set(gca,'FontSize',17);
pl = pl+1;
end



 