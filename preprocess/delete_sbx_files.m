function [] = delete_sbx_files(mouse, date, runs)
    % This function deletes sbx and other large files once processed
    base = 'D:/2p_data/scan/';
    if runs > 0
        for r = 1:runs 
            run = num2str(r);
            
            file_to_delete = [mouse,'\',date,'_',mouse,'\',date,'_',mouse,'_00',run,'\',mouse,'_',date,'_00',run,'.sbx'];
            if exist([base,file_to_delete])
                delete ([base,file_to_delete])
            end
            
            file_to_delete = [mouse,'\',date,'_',mouse,'\',date,'_',mouse,'_00',run,'\',mouse,'_',date,'_00',run,'_ball.mj2'];
            if exist([base,file_to_delete])
                delete ([base,file_to_delete])
            end
            
            file_to_delete = [mouse,'\',date,'_',mouse,'\',date,'_',mouse,'_00',run,'\',mouse,'_',date,'_00',run,'_eye.mj2'];
            if exist([base,file_to_delete])
                delete ([base,file_to_delete])
            end
            
            file_to_delete = [mouse,'\',date,'_',mouse,'\',date,'_',mouse,'_00',run,'\',mouse,'_',date,'_00',run,'_eye.mat'];
            if exist([base,file_to_delete])
                delete ([base,file_to_delete])
            end
            
            file_to_delete = [mouse,'\',date,'_',mouse,'\',date,'_',mouse,'_00',run,'\',mouse,'_',date,'_00',run,'.ephys'];
            if exist([base,file_to_delete])
                delete ([base,file_to_delete])
            end
            
        end
    end
end
