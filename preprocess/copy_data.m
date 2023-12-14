mice = {'NN8', 'NN9', 'NN11', 'NN23', 'NN28', 'NN13', 'NN16', 'NN17'};

dates_NN8 = {'210312', '210314', '210316', '210318', '210320', '210321', '210327', '210329', '210330'};
dates_NN9 = {'210428', '210429', '210501', '210502', '210503', '210505', '210506', '210507', '210509', '210510', '210511', '210512', '210513', '210514'};
dates_NN11 = {'210626', '210627', '210628', '210629', '210630', '210701', '210703', '210704', '210705', '210706'};
dates_NN23 = {'220416', '220417', '220418', '220419', '220420', '220421', '220422', '220423', '220424'};
dates_NN28 = {'230210', '230211', '230212', '230214', '230216', '230217'};
dates_NN13 = {'210811', '210812', '210813', '210814', '210815', '210816', '210817', '210818'};
dates_NN16 = {'211014', '211015', '211016', '211017', '211018', '211019', '211020', '211021', '211022'};
dates_NN17 = {'211025', '211026', '211028', '211029', '211030', '211031', '211101'};

dates_all = {dates_NN8, dates_NN9, dates_NN11, dates_NN23, dates_NN28, dates_NN13, dates_NN16, dates_NN17};
base = '//nasquatch/data/2p/nghia/';
for i = 1:size(mice,2)
    mouse = mice{i};
    dates = dates_all{i};
    for d = 1:size(dates,2)
        date = dates{d};
        mkdir (['C:/Users/nnguyen/Desktop/',mouse,'/',date]);
        source = [base,mouse,'/',date,'_',mouse,'/suite2p_plane_3/suite2p/plane0/stat.npy'];
        destination = ['C:/Users/nnguyen/Desktop/',mouse,'/',date];
        copyfile(source, destination); 
        source = [base,mouse,'/',date,'_',mouse,'/suite2p_plane_3/suite2p/plane0/ops.npy'];
        destination = ['C:/Users/nnguyen/Desktop/',mouse,'/',date];
        copyfile(source, destination); 
    end
end



%%
mice = {'NN8', 'NN9'};
dates_NN8 = {'210312', '210314', '210316', '210318', '210320', '210321', '210327', '210329', '210330'};
dates_NN9 = {'210428', '210429', '210501', '210502', '210503', '210505', '210506', '210507', '210509', '210510', '210511', '210512', '210513', '210514'};
dates_NN11 = {'210626', '210627', '210628', '210629', '210630', '210701', '210703', '210704', '210705', '210706'};
dates_NN13 = {'210811', '210812', '210813', '210814', '210815', '210816', '210817', '210818'};

dates_all = {dates_NN8, dates_NN9};
base = 'D:/2p_data/scan/';
for i = 1:size(mice,2)
    mouse = mice{i};
    dates = dates_all{i};
    for d = 1:size(dates,2)
        date = dates{d};
        mkdir (['C:/Users/nnguyen/Desktop/',mouse,'/',date]);
        source = [base,mouse,'/',date,'_',mouse,'/processed_data/plots'];
        destination = ['C:/Users/nnguyen/Desktop/',mouse,'/',date];
        copyfile(source, destination);  
        %cd ([base,mouse,'/',date,'_',mouse,'/processed_data/plots'])
        %delete * 
    end
    mkdir (['C:/Users/nnguyen/Desktop/',mouse,'/data_across_days']);
    source = [base,mouse,'/data_across_days/plots'];
    destination = ['C:/Users/nnguyen/Desktop/',mouse,'/data_across_days'];
    copyfile(source, destination);  
end
        
%% delete tifs 
dates_NN8 = {'210312', '210314', '210316', '210318', '210320', '210321', '210327', '210329', '210330'};
dates_NN9 = {'210428', '210429', '210501', '210502', '210503', '210505', '210506', '210507', '210509', '210510', '210511', '210512', '210513', '210514'};
dates_NN11 = {'210626', '210627', '210628', '210629', '210630', '210701', '210703', '210704', '210705', '210706'};
dates_NN13 = {'210811', '210812', '210813', '210814', '210815', '210816', '210817', '210818'};
dates_NN16 = {'211014', '211015', '211016', '211017', '211018', '211019', '211020', '211021', '211022'};
dates_NN17 = {'211025', '211026', '211028', '211029', '211030', '211031', '211101'};

mice = {'NN17'};
dates_all = {dates_NN17};
base = '//nasquatch/data/2p/nghia/';
for i = 1:size(mice,2)
    mouse = mice{i};
    dates = dates_all{i};
    for d = 1:size(dates,2) 
        date = dates{d};
        cd ([base,mouse,'/',date,'_',mouse,'/suite2p_plane_1'])
        delete *.tif
        cd ([base,mouse,'/',date,'_',mouse,'/suite2p_plane_2'])
        delete *.tif
        cd ([base,mouse,'/',date,'_',mouse,'/suite2p_plane_3'])
        delete *.tif
    end
end
        

%%
mice = {'NN8', 'NN9', 'NN11', 'NN23', 'NN28', 'NN13', 'NN16', 'NN17'};

dates_NN8 = {'210312', '210314', '210316', '210318', '210320', '210321', '210327', '210329', '210330'};
dates_NN9 = {'210428', '210429', '210501', '210502', '210503', '210505', '210506', '210507', '210509', '210510', '210511', '210512', '210513', '210514'};
dates_NN11 = {'210626', '210627', '210628', '210629', '210630', '210701', '210703', '210704', '210705', '210706'};
dates_NN23 = {'220416', '220417', '220418', '220419', '220420', '220421', '220422', '220423', '220424'};
dates_NN28 = {'230210', '230211', '230212', '230214', '230216', '230217'};
dates_NN13 = {'210811', '210812', '210813', '210814', '210815', '210816', '210817', '210818'};
dates_NN16 = {'211014', '211015', '211016', '211017', '211018', '211019', '211020', '211021', '211022'};
dates_NN17 = {'211025', '211026', '211028', '211029', '211030', '211031', '211101'};

dates_all = {dates_NN8, dates_NN9, dates_NN11, dates_NN23, dates_NN28, dates_NN13, dates_NN16, dates_NN17};
base = 'D:/2p_data/scan/';
for i = 1:size(mice,2)
    mouse = mice{i};
    dates = dates_all{i};
    for d = 1:size(dates,2)
        date = dates{d};
        for p = 1:3
            mkdir (['//nasquatch/data/2p/nghia/nature_data','/',mouse,'/',date,'_plane_',num2str(p)]);
            mkdir (['//nasquatch/data/2p/nghia/nature_data','/',mouse,'/','/',date,'_behavior_data']);

            source = [base,mouse,'/',date,'_',mouse,'/suite2p_plane_',num2str(p),'/suite2p/plane0/ops.npy'];
            destination = ['//nasquatch/data/2p/nghia/nature_data','/',mouse,'/',date,'_plane_',num2str(p)];
            copyfile(source, destination); 
            
            source = [base,mouse,'/',date,'_',mouse,'/suite2p_plane_',num2str(p),'/suite2p/plane0/iscell.npy'];
            destination = ['//nasquatch/data/2p/nghia/nature_data','/',mouse,'/',date,'_plane_',num2str(p)];
            copyfile(source, destination); 
            
            source = [base,mouse,'/',date,'_',mouse,'/suite2p_plane_',num2str(p),'/suite2p/plane0/stat.npy'];
            destination = ['//nasquatch/data/2p/nghia/nature_data','/',mouse,'/',date,'_plane_',num2str(p)];
            copyfile(source, destination); 
            
            source = [base,mouse,'/',date,'_',mouse,'/suite2p_plane_',num2str(p),'/suite2p/plane0/spks.npy'];
            destination = ['//nasquatch/data/2p/nghia/nature_data','/',mouse,'/',date,'_plane_',num2str(p)];
            copyfile(source, destination); 
            
        end
        
        source = [base,mouse,'/',date,'_',mouse,'/processed_data/saved_data/behavior_file.mat'];
        destination = ['//nasquatch/data/2p/nghia/nature_data','/',mouse,'/','/',date,'_behavior_data'];
        copyfile(source, destination); 
    end
end
     