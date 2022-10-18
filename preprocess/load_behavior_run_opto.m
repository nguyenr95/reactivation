%% mouse and date
mouse = 'NN';
date = '22';
planes = 3;

%% movie imaging files from nasquatch
make_folders('D:/2p_data/scan/',mouse,date,0,planes);
copy_folders('//nasquatch/data/2p/nghia/','D:/2p_data/scan/',mouse,date,5,0);

%% get behavior
runs = {'1','2','3','4','5'};
for r = 1:size(runs,2)

    run = runs{r};

    %% load behavioral data 
    bdata = pipe.io.trial_times_opto_ML2(mouse,date,run);
    bdata.framerate = 31.25/planes;

    %% running
    running = [];
    quad_path = ['D:\2p_data\scan\',mouse,'\',date,'_',mouse,'\',date,'_',mouse,'_00',run,'\',mouse,'_',date,'_00',run,'_quadrature.mat'];
    quadfile = builtin('load', quad_path, '-mat');
    if ~isempty(quadfile)
        running = double(quadfile.quad_data);
        last = round(size(running,2)/planes)*planes;
        nframes = size(running,2);
        running = pipe.misc.position_to_speed(running,bdata.framerate);
        running = resample(running,last/planes,size(running,2));
    end
    bdata.running = running;

    %% pupil
    pipe.pupil.masks(mouse,date,run);
    [dx, dy, psum, area, quality] = pipe.pupil.extract(mouse,date,run);
    area = resample(area,last/planes,size(area,2));
    bdata.area = area;

    %% licking
    licking = zeros(1,nframes);
    licking(bdata.licking) = 1;
    licking = resample(licking,last/planes,size(licking,2));
    bdata.licking = licking;
    
    %% round down behavior if multiplane to aligned to 1st plane
    if run ~= '1'
        bdata.onsets = floor(bdata.onsets/planes);
        bdata.offsets = floor(bdata.offsets/planes);
        bdata.ensure = floor(bdata.ensure/planes);
        bdata.quinine = floor(bdata.quinine/planes);
        bdata.opto_onsets = floor(bdata.opto_onsets/planes);
    end
end

%% delete sbx and other large files
%delete_sbx_files(mouse,date,5);
%close all; clear all; clc