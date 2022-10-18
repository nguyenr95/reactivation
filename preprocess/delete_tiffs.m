function [] = delete_tiffs(mouse, date)
    % This function delete tiff files 
    % runs
    runs = {'1','2', '3', '4', '5'};
    % info
    planes = 3;
    % delete tiffs
    for r = 1:size(runs,2)  
        %run
        run = runs{r};
        % delete
        for i = 1:planes
            cd (['D:\2p_data\scan\',mouse,'\',date,'_',mouse,'\suite2p_plane_',num2str(i)])
            tiffnum = 1;
            for t = 1:6
                if run ~= '1' && plane ~= 1 && tiffnum ~= 1
                    delete ([mouse,'_',date,'_plane_',num2str(i),'_run_00',run,'_',num2str(tiffnum),'.tif']);
                end
                tiffnum = tiffnum + 1;
            end
        end
    end
    cd (['D:\2p_data\scan\',mouse,'\',date,'_',mouse])
end
