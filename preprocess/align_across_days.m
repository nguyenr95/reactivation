% mice = {'NN8', 'NN9', 'NN11'};
dates_NN8 = {'210312', '210314', '210316', '210318', '210320', '210321', '210327', '210329', '210330'};
dates_NN9 = {'210428', '210429', '210501', '210502', '210503', '210505', '210506', '210507', '210509', '210510', '210511', '210512', '210513', '210514'};
dates_NN11 = {'210626', '210627', '210628', '210629', '210630', '210701', '210703', '210704', '210705', '210706'};
dates_NN23 = {'220416', '220417', '220418', '220419', '220420', '220421', '220422', '220423', '220424'};

% mice_dates = {dates_NN8, dates_NN9, dates_NN11};

mice = {'NN8'};
mice_dates = {dates_NN8};
it = 1;
for m = 1:size(mice,2)
    mouse = mice{m};
    dates = mice_dates{m};
    for d = 1:size(dates,2)
        date = dates{d};
        cell_masks = readNPY(['D:\2p_data\scan\',mouse,'\',date,'_',mouse,'\processed_data\saved_data\plane_1_cell_masks.npy']);
        cell_masks = permute(cell_masks, [3, 1, 2]);
        save(sprintf('spatial_footprints_0%1i.mat',it), 'cell_masks', '-v7.3')
        it = it + 1;
    end
end


