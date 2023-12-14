function [] = copy_folders(base1, base2, mouse, date, runs, planes)
    % This function moves folders
    if runs > 0
        for r = 1:runs 
            run = num2str(r);
            
            file_to_copy = [mouse,'\',date,'_',mouse,'\',date,'_',mouse,'_00',run];
            if ~exist([base2,file_to_copy])
                copyfile([base1,file_to_copy], [base2,file_to_copy])
            end
            
        end
    end
    
    if planes > 0
        for p = 1:planes 
            files = dir([base1,mouse,'\',date,'_',mouse,'\suite2p_plane_',num2str(p),'\suite2p\plane0']);
            for file = 1:size(files,1)
                file_to_copy = [mouse,'\',date,'_',mouse,'\suite2p_plane_',num2str(p),'\suite2p\plane0\',files(file).name];
                if ~exist([base2,file_to_copy]) && strcmp(files(file).name,'data.bin') == 0 && size(files(file).name,2) > 2
                    copyfile([base1,file_to_copy], [base2,file_to_copy])
                end
            end
        end
    end
end
