%%%%% reads in registered movie from suite2p
format long g

mice = {'NN8', 'NN9', 'NN11', 'NN23', 'NN28'};
dates_NN8 = {'210312', '210314', '210316', '210318', '210320', '210321'};
dates_NN9 = {'210428', '210429', '210501', '210502', '210503', '210505'};
dates_NN11 = {'210626', '210627', '210628', '210629', '210630', '210701'};
dates_NN23 = {'220416', '220417', '220418', '220419', '220420', '220421'};
dates_NN28 = {'230210', '230211', '230212', '230214', '230216', '230217'};
mice_dates = {dates_NN8, dates_NN9, dates_NN11, dates_NN23, dates_NN28};

for i = 1:size(mice,2)
    %% Mouse info, change
    mouse = mice{i};
    dates = mice_dates{i};
    
    for j = 1:size(dates,2)
        date = dates{j};

        %% Get path to movie
        %base = 'D:/2p_data/scan/';
        base = '//nasquatch/data/2p/nghia/';
        movie_path = [base,mouse,'\',date,'_',mouse,'\suite2p_plane_1\suite2p\plane0\data.bin'];

        %% open registered binary movie
        fid = fopen(movie_path,'r');

        %% dimensions
        Ly = 512;
        Lx = 796;
        nframes = 6000; %21333*5;

        %% open movie 
        regmovie = fread(fid,Ly*Lx*nframes,'*int16');

        %% reshape into array
        regmovie = reshape(regmovie,Lx,Ly,nframes);

        %% write tiff
        % regmovie = double(regmovie(:,:,5000:7000));
        % pipe.io.write_tiff(regmovie,[mouse,'_',date]);

        cd (['D:/2p_data/scan/',mouse,'\',date,'_',mouse,'\processed_data\movies'])
        pipe.io.write_tiff(double(mean(regmovie(:,:,1001:6000), 3)'),'example_fov');
    end
end
