%%%%% reads in registered movie from suite2p
format long g

%% Mouse info, change
mouse = 'NN11';
date = '210627';

%% Get path to movie
%base = 'D:/2p_data/scan/';
base = '//nasquatch/data/2p/nghia/';
movie_path = [base,mouse,'\',date,'_',mouse,'\suite2p_plane_1\suite2p\plane0\data.bin'];

%% open registered binary movie
fid = fopen(movie_path,'r');

%% dimensions
Ly = 512;
Lx = 796;
nframes = 1000; %21333*5;

%% open movie 
regmovie = fread(fid,Ly*Lx*nframes,'*int16');

%% reshape into array
regmovie = reshape(regmovie,Lx,Ly,nframes);

%% write tiff
% regmovie = double(regmovie(:,:,5000:7000));
% pipe.io.write_tiff(regmovie,[mouse,'_',date]);


