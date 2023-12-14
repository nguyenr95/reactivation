%%
mice = {'NN8', 'NN9', 'NN11'};
dates_NN8 = {'210312', '210314', '210316', '210318', '210320', '210321', '210327', '210329', '210330'};
dates_NN9 = {'210428', '210429', '210501', '210502', '210503', '210505', '210506', '210507', '210509', '210510', '210511', '210512', '210513', '210514'};
dates_NN11 = {'210626', '210627', '210628', '210629', '210630', '210701', '210703', '210704', '210705', '210706'};
mice_dates = {dates_NN8, dates_NN9, dates_NN11};

mice = {'NN8'};
dates_NN8 = {'210312'};
mice_dates = {dates_NN8};

for m = 1:size(mice,2)
    mouse = mice{m};
    dates = mice_dates{m};
    for d = 1:size(dates,2)
        date = dates{d};
        cd (['D:/2p_data/scan/',mouse,'\',date,'_',mouse,'\processed_data\saved_data'])
        runs = {'1', '2', '3', '4', '5'};
        vv = VideoWriter([mouse,'_',date,'_face.mj2'], 'Motion JPEG 2000');
        open(vv);
        for r = 1:size(runs,2)
            run = runs{r};
            face_path = ['//nasquatch/data/2p/nghia/',mouse,'\',date,'_',mouse,'\',date,'_',mouse,'_','00',run,'\',mouse,'_',date,'_','00',run,'_ball.mj2'];
            v = VideoReader(face_path);
            curr_frame = 0;
            for i = 1:v.Framerate * v.Duration
                curr_frame = curr_frame + 1;
                if mod(curr_frame, 16) == 0 || curr_frame == 1
                    writeVideo(vv, read(v, curr_frame));
                end
            end
        end
        close(vv);
    end
end
