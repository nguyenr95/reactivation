function [] = make_folders(base, mouse, date, runs, planes)
    % This function makes folders
    % info
    if ~exist([base,mouse,'\',date,'_',mouse], 'dir')
        mkdir ([base,mouse,'\',date,'_',mouse])
        if strcmp(base,'D:/2p_data/scan/') == 1
            mkdir ([base,mouse,'\',date,'_',mouse,'\processed_data\saved_data',])
        end
    end
 
    if runs > 0
       for r = 1:runs
           if ~exist([base,mouse,'\',date,'_',mouse,'\',date,'_',mouse,'_00',num2str(r)], 'dir')
                mkdir ([base,mouse,'\',date,'_',mouse,'\',date,'_',mouse,'_00',num2str(r)])
           end
       end
    end
    
    if planes > 0
       for p = 1:planes
           if ~exist([base,mouse,'\',date,'_',mouse,'\suite2p_plane_',num2str(p),'\suite2p\plane0'], 'dir')
                mkdir ([base,mouse,'\',date,'_',mouse,'\suite2p_plane_',num2str(p),'\suite2p\plane0'])
           end
       end
    end
end
