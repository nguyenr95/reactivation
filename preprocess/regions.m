%% do first for each mouse
% load in movie for first day to determine scaling factor in imajeg
regmovie = im2double(regmovie);
align = mean(regmovie,3)';
imwrite(align,'align.jpeg')

base = NN17_base;

%NN8: 362/128 NN9:457.34/162 NN11: 2.82 NN23: 2.82 NN28: 2.72 NN13:2.735
%NN17: 2.72 NN16: 2.72
base_r = imresize(base, 2.72);

figure; imshow(base_r)
LI = roipoly;
ALL = LI;

figure; imshowpair(base_r, ALL)
POR = roipoly;
ALL = LI+POR;

figure; imshowpair(base_r, ALL)
P = roipoly;
ALL = LI+POR+P;

figure; imshowpair(base_r, ALL)
LM = roipoly;
ALL = LI+POR+P+LM;

figure;imshowpair(base_r, ALL)

save('retinotopy')

%% For each day find same point x, y ex: [406, 471] align, [656 903] base
regmovie = im2double(regmovie);
align = mean(regmovie,3)';
align_r = zeros(size(base_r,1), size(base_r,2));
align_r(1:size(align,1), 1:size(align,2)) = align;
figure; imshow(align_r, [.5,.8])
% figure; imshow(base_r)

%% save for each mouse
imaging_c = [390 343];
base_c = [627 752];

diff_x = base_c(1)-imaging_c(1);
diff_y = base_c(2)-imaging_c(2);

align_r = zeros(size(base_r,1), size(base_r,2));
align_r(diff_y:diff_y+size(align,1)-1, diff_x:diff_x+size(align,2)-1) = align;

figure;imshowpair(align_r, base_r)

base = 'D:/2p_data/scan/';
cd ([base,mouse,'\',date,'_',mouse,'\processed_data\saved_data'])
save('retinotopy_day', 'base_c', 'imaging_c', 'align_r', 'base_r')

