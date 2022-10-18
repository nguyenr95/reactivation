%% make avi
v = VideoWriter('NoiseBarHorz22.avi');
open(v)
for i = 1:120
load(['NoiseBarHorz2_preprocessed',num2str(i)]);
M = reshape(M,1280,1024)';
x = 426;
y = 341;
%M(x+1:size(M,1), :) = 9276301;
%M(1:342, :) = 9276301;
M(:,1:size(M,2)-427) = 9276301;
M2 = double((M/(13552846/205)));
M2 = M2/255;
imagesc(M2);
writeVideo(v,M2)
end
close(v)

%% new pizza
for i = 1:120
load(['NoiseBarHorz1_preprocessed',num2str(i)]);
M = reshape(M,1280,1024)';
M = imscaled(:,:,i);
M(M > .5) = 13552846;
M(M < .5) = 3486517;

im = M;
r = 8000;
cy = floor(size(im,1)/2);
cx = floor(size(im,2)/2);
for j = 0
    t = linspace(-j,-j-38.66,128);
    x = [cx, cx+r*cosd(t), cx];
    y = [cy, cy+r*sind(t), cy];
    bw = poly2mask( x, y, size(im,1),size(im,2));
    M(bw~=1) = 9276301;
end

M(1:512,1:640) = flip(M(1:512,641:1280));
M(513:1024,:) = flip(flip(M(1:512,:),2));
%imagesc(M)

% M2 = M;
% M2(1:512,:) = flip(M2(1:512,:),2);
% M2(513:1024, :) = flip(M2(513:1024, :),2);
%imagesc(M2)

M = uint32(reshape(M',1280*1024,1));
save(['NoiseBarHorz1_preprocessed',num2str(i)],'M','xis','yis','xisbuf','yisbuf')
end

for i = 1:120
load(['NoiseBarHorz2_preprocessed',num2str(i)]);
M = reshape(M,1280,1024)';
M = imscaled(:,:,i);
M(M > .5) = 13552846;
M(M < .5) = 3486517;

im = M;
r = 8000;
cy = floor(size(im,1)/2);
cx = floor(size(im,2)/2);
for j = 0
    t = linspace(-j,-j-38.66,128);
    x = [cx, cx+r*cosd(t), cx];
    y = [cy, cy+r*sind(t), cy];
    bw = poly2mask( x, y, size(im,1),size(im,2));
    M(bw~=1) = 9276301;
end

M(1:512,1:640) = flip(M(1:512,641:1280));
M(513:1024,:) = flip(flip(M(1:512,:),2));
%imagesc(M)

M2 = M;
M2(1:512,:) = flip(M2(1:512,:),2);
M2(513:1024, :) = flip(M2(513:1024, :),2);
%imagesc(M2)
M = M2;

M = uint32(reshape(M',1280*1024,1));
save(['NoiseBarHorz2_preprocessed',num2str(i)],'M','xis','yis','xisbuf','yisbuf')
end

%% new box
for i = 1:120
load(['NoiseBarHorz1_preprocessed',num2str(i)]);
M = reshape(M,1280,1024)';
M = imscaled(:,:,i);
M(M > .5) = 13552846;
M(M < .5) = 3486517;

x = 426;
y = 341;

M(x+2:x*2+1, 1:y) = 9276301;
M(x+2:x*2+1, y*2+1:y*3) = 9276301;
M(2:x+1, y+1:y*2) = 9276301;
M(x*2+2:x*3+1, y+1:y*2) =9276301;

M(1,:) = 9276301;
M(1280,:) = 9276301;
M(:,1024) = 9276301;

M(x*2+2:x*3+1, 1:y) = M(2:x+1, 1:y);
M(x+2:x*2+1, y+1:y*2) = M(2:x+1, 1:y);
M(2:x+1, y*2+1:y*3) = M(2:x+1, 1:y);
M(x*2+2:x*3+1, y*2+1:y*3) = M(2:x+1, 1:y);

M = uint32(reshape(M,1280*1024,1));
save(['NoiseBarHorz1_preprocessed',num2str(i)],'M','xis','yis','xisbuf','yisbuf')
end
for i = 1:120
load(['NoiseBarHorz2_preprocessed',num2str(i)]);
M = reshape(M,1280,1024)';
M = imscaled(:,:,i);
M(M > .5) = 13552846;
M(M < .5) = 3486517;

x = 426;
y = 341;

M(x*2+2:x*3+1, 1:y) = 9276301;
M(x+2:x*2+1, y+1:y*2) = 9276301;
M(2:x+1, y*2+1:y*3) = 9276301;
M(x*2+2:x*3+1, y*2+1:y*3) = 9276301;

M(x+2:x*2+1, 1:y) = M(2:x+1, 1:y);
M(x+2:x*2+1, y*2+1:y*3) = M(2:x+1, 1:y);
M(2:x+1, y+1:y*2) = M(2:x+1, 1:y);
M(x*2+2:x*3+1, y+1:y*2) =M(2:x+1, 1:y);

M(2:x+1, 1:y) = 9276301;

M(1,:) = 9276301;
M(1280,:) = 9276301;
M(:,1024) = 9276301;

M = uint32(reshape(M,1280*1024,1));
save(['NoiseBarHorz2_preprocessed',num2str(i)],'M','xis','yis','xisbuf','yisbuf')
end













%% old
for i = 1:120
load(['NoiseBarHorz1_preprocessed',num2str(i)]);
M1 = reshape(M,1280,1024)';
M2 = M1;
M2(:) = 9276301;
M2(1:256,428:427+426) = M1(1:256,428:853);
M2(513:512+256,428:427+426) = M1(1:256,428:853);
M2(257:512,1:426) = M1(1:256,428:853);
M2(257+512:512+512,1:426) = M1(1:256,428:853);
M2 = reshape(M2',1280*1024,1);
M = M2;
save(['NoiseBarHorz2_preprocessed',num2str(i)],'M','xis','yis','xisbuf','yisbuf')
end
for i = 1:120
load(['NoiseBarHorz1_preprocessed',num2str(i)]);
M1 = reshape(M,1280,1024)';
M2 = M1;
M2(:) = 9276301;
M2(1:256,428+426:427+426+426) = M1(1:256,428:853);
M2(513:512+256,428+426:427+426+426) = M1(1:256,428:853);
M2(257:512,1+427:426+427) = M1(1:256,428:853);
M2(257+512:512+512,1+427:426+427) = M1(1:256,428:853);
M2 = reshape(M2',1280*1024,1);
M = M2;
save(['NoiseBarHorz3_preprocessed',num2str(i)],'M','xis','yis','xisbuf','yisbuf')
end
for i = 1:120
load(['NoiseBarHorz1_preprocessed',num2str(i)]);
M1 = reshape(M,1280,1024)';
M2 = M1;
M2(:) = 9276301;
M2(1:256,1:426) = M1(1:256,428:853);
M2(1+256:256+256,428+426:427+426+426) = M1(1:256,428:853);
M2(1+512:256+512,1:426) = M1(1:256,428:853);
M2(513+256:512+512,428+426:427+426+426) = M1(1:256,428:853);
M2 = reshape(M2',1280*1024,1);
M = M2;
save(['NoiseBarHorz4_preprocessed',num2str(i)],'M','xis','yis','xisbuf','yisbuf')
end

%%
for i = 1:120
load(['NoiseBarHorz1_preprocessed',num2str(i)]);
M1 = reshape(M,1280,1024)';
M2 = [];
M2 = horzcat(ones(1024,320)*9276301,M1,ones(1024,320)*9276301);
M2 = vertcat(ones(28,1920)*9276301,M2,ones(28,1920)*9276301);
M2(:) = 9276301;
M2(1+28:256+28,1:480) = M1(1:256,1:480);
M2(256*2+1+28:256*3+28,1:480) = M1(1:256,1:480);
M2(1+28:256+28,480*2+1:480*3) = M1(1:256,1:480);
M2(256*2+1+28:256*3+28,480*2+1:480*3) = M1(1:256,1:480);
M2(256+1+28:256*2+28,480+1:480*2) = M1(1:256,1:480);
M2(256*3+1+28:256*4+28,480+1:480*2) = M1(1:256,1:480);
M2(256+1+28:256*2+28,480*3+1:480*4) = M1(1:256,1:480);
M2(256+1+28:256*2+28,480*3+1:480*4) = M1(1:256,1:480);
M2(256*3+1+28:256*4+28,480*3+1:480*4) = M1(1:256,1:480);
M2 = reshape(M2',1920*1080,1);
M = M2;
xis = 1920;
xisbuf = 1920;
yis = 1080;
yisbuf = 1080;
save(['NoiseBarHorz1_preprocessed',num2str(i)],'M','xis','yis','xisbuf','yisbuf')
end
%%
for i = 1:120
load(['NoiseBarHorz1_preprocessed',num2str(i)]);
M1 = reshape(M,1280,1024)';
M2 = horzcat(ones(1024,320)*9276301,M1,ones(1024,320)*9276301);
M2 = vertcat(ones(28,1920)*9276301,M2,ones(28,1920)*9276301);
M2(:) = 9276301;
M2(1+28:256+28,480+1:480*2) = M1(1:256,600:1079);
M2(1+28:256+28,480*3+1:480*4) = M1(1:256,600:1079);
M2(256+1+28:256*2+28,480*2+1:480*3) = M1(1:256,600:1079);
M2(256+1+28:256*2+28,1:480) = M1(1:256,600:1079);
M2(256*2+1+28:256*3+28,480+1:480*2) = M1(1:256,600:1079);
M2(256*2+1+28:256*3+28,480*3+1:480*4) = M1(1:256,600:1079);
M2(256*3+1+28:256*4+28,1:480) = M1(1:256,600:1079);
M2(256*3+1+28:256*4+28,480*2+1:480*3) = M1(1:256,600:1079);
M2 = reshape(M2',1920*1080,1);
M = M2;
xis = 1920;
xisbuf = 1920;
yis = 1080;
yisbuf = 1080;
save(['NoiseBarHorz2_preprocessed',num2str(i)],'M','xis','yis','xisbuf','yisbuf')
end
