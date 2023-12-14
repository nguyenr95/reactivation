%% This function makes tiff files for suite2p from all runs (task+quietwaking)
%% runs
% pause(1800)
runs = {'5'};

%% info
mouse = 'NN28';
date = '230217';
base = 'D:/2p_data/scan/';
%base = '//nasquatch/data/2p/nghia/';

%% multiplane? bidrectional?
planes = 3;
bidirectional = 1;

%% make tiffs
for r = 1:size(runs,2)
    
    %%run
    run = runs{r};

    %% read in movie
    tempmovie = pipe.io.read_sbx([base,mouse,'/',date,'_',mouse,'/',date,'_',mouse,'_00',run,'/',mouse,'_',date,'_00',run,'.sbx'],1,-1,2);
    
    %% if multiplane make multiple temp movies
    movies = {};
    last = floor(size(tempmovie,3)/planes)*planes;
    for j = 1:planes
        movies{1,j} = tempmovie(:,:,j:planes:last);
    end
    
    %% shift movie for bidirectional scanning if needed
    if bidirectional == 1
        loop = 0;
        bin_size = 16;
        check_movie_raw = movies{1}(:,:,5001:5000+(bin_size*180));
        check_movie_bin = {};
        for i = 1:size(check_movie_raw,3)/bin_size
            check_movie_bin{i} = uint16(mean(check_movie_raw(:,:,((i-1)*bin_size)+1:i*bin_size),3));
        end
        check_movie_bin = cell2mat(check_movie_bin); 
        check_movie_bin = reshape(check_movie_bin, size(check_movie_raw,1), size(check_movie_raw,2), size(check_movie_raw,3)/bin_size);
        handle = implay(check_movie_bin);
        handle.Parent.Position = [100 200 796 512];
        while true
            if loop ~= 1
                shift = input('Enter shift : ');
                if shift == 0
                    break
                end
                check_movie_temp = check_movie_bin;
                for i = 1:2:size(check_movie_temp,1)
                    check_movie_temp(i,:,:) = circshift(check_movie_temp(i,:,:),shift);
                end
                handle = implay(check_movie_temp);
                handle.Parent.Position = [150 200 796 512];
            else
                break
            end
            loop = input('Proceed? (1/0) : '); 
        end
    end
%     shift = -2;
    %% apply shift
    if shift ~= 0
        for j = 1:size(movies,2)
            for i = 1:2:size(movies{j},1)
                movies{j}(i,:,:) = circshift(movies{j}(i,:,:),shift);
            end
        end
    end
    
    %% save movies as tifs, no larger than 4GB per tiff
    maxframes = 5000;
    for i = 1:planes
        cd ([base,mouse,'\',date,'_',mouse])
        if ~exist(['suite2p_plane_',num2str(i)], 'dir')
            mkdir (['suite2p_plane_',num2str(i)])
        end
        cd ([base,mouse,'\',date,'_',mouse,'\suite2p_plane_',num2str(i)])
        tempmovie = movies{1,i};
        tiffnum = 1;
        for t = 1:ceil(size(tempmovie,3)/maxframes)
            start = 1 + ((t-1)*maxframes);
            last = t*maxframes;
            if last > size(tempmovie,3)
                last = size(tempmovie,3);
            end
            pipe.io.write_tiff(tempmovie(:,:,start:last),[mouse,'_',date,'_plane_',num2str(i),'_run_00',run,'_',num2str(tiffnum)]);
            tiffnum = tiffnum + 1;
        end
    end
    cd ([base,mouse,'\',date,'_',mouse])
end
close all; clear all; clc