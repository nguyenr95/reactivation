rand('state',sum(100*clock))

tic

%%% stimulus/display parameters

% user set-up
movtype=3;
binarize=1;
contrast_luminance=0.8;
background = 140;

cmod=1; step=2; bar = 3;  Xpatches=4; Ypatches=4; %%%% movietypes

imsize = 1280;                %% size in pixels Liang: 136
imheight = 1024;
framerate = 60;             %% Hz
imageMag=1;                 %% magnification that movie will be played at Liang 4
screenWidthPix = 1280;        %% Screen width in Pixels
screenWidthCm =37.5;         %% Width in cm
screenDistanceCm = 22;      %% Distance in cm
% 
duration = 1/30;               %% duration in minutes
maxSpatFreq = 0.32;         %% spatial frequency cutoff (cpd)
maxTempFreq = 4;          %% temporal frequency cutoff
contrastSigma =0.5;         %% one-sigma value for contrast

%% derived parameters

screenWidthDeg = 2*atan(0.5*screenWidthCm/screenDistanceCm)*180/pi
degperpix = (screenWidthDeg/screenWidthPix)*imageMag
nframes = framerate*60*duration;

%% frequency intervals for FFT

nyq_pix = 0.5;
nyq_deg=nyq_pix/degperpix;
freqInt_deg = nyq_deg / (0.5*imsize);
freqInt_pix = nyq_pix / (0.5*imsize);
nyq = framerate/2;
tempFreq_int = nyq/(0.5*nframes)

%% cutoffs in terms of frequency intervals

tempCutoff = round(maxTempFreq/tempFreq_int);
maxFreq_pix = maxSpatFreq*degperpix;
spatCutoff = round(maxFreq_pix / freqInt_pix);
offsetFreq=0.05;
offset=round(offsetFreq*degperpix/freqInt_pix)

%%% generate frequency spectrum (invFFT)
alpha=-1;

% offset=3;
range_mult =1;

%for noise that extends past cutoff parameter (i.e. if cutoff = 1sigma)
%range_mult=2;
spaceRange = (imsize/2 - range_mult*spatCutoff : imsize/2 + range_mult*spatCutoff)+1;
tempRange =   (nframes /2 - range_mult*tempCutoff : nframes/2 + range_mult*tempCutoff)+1;
[x y z] = meshgrid(-range_mult*spatCutoff:range_mult*spatCutoff,-range_mult*spatCutoff:range_mult*spatCutoff,-range_mult*tempCutoff:range_mult*tempCutoff);

%% e.g. gaussian spectrum

% use = exp(-1*((0.5*x.^2/spatCutoff^2) + (0.5*y.^2/spatCutoff^2) + (0.5*z.^2/tempCutoff^2)));
%  use =single(((x.^2 + y.^2)<=(spatCutoff^2))& ((z.^2)<(tempCutoff^2)) );
% use =single(((x.^2 + y.^2)<=(spatCutoff^2))& ((z.^2)<(tempCutoff^2)) ).*(sqrt(x.^2 + y.^2 +offset).^alpha);

use =single(((x.^2 + y.^2)<=(spatCutoff^2))& ((z.^2)<(tempCutoff^2)) ).*((sqrt(x.^2 + y.^2) +offset).^alpha);
clear x y z;

%%%

invFFT = zeros(imsize,imsize,nframes,'single');
mu = zeros(size(spaceRange,2), size(spaceRange,2), size(tempRange,2));
sig = ones(size(spaceRange,2), size(spaceRange,2), size(tempRange,2));
invFFT(spaceRange, spaceRange, tempRange) = single(use .* normrnd(mu,sig).*exp(2*pi*i*rand(size(spaceRange,2), size(spaceRange,2), size(tempRange,2))));
clear use;

%% symmetric

fullspace = -range_mult*spatCutoff:range_mult*spatCutoff; halftemp = 1:range_mult*tempCutoff;
halfspace = 1:range_mult*spatCutoff;
invFFT(imsize/2 + fullspace+1, imsize/2+fullspace+1, nframes/2 + halftemp+1) = ...
    conj(invFFT(imsize/2 - fullspace+1, imsize/2-fullspace+1, nframes/2 - halftemp+1));
invFFT(imsize/2+fullspace+1, imsize/2 + halfspace+1,nframes/2+1) = ...
    conj( invFFT(imsize/2-fullspace+1, imsize/2 - halfspace+1,nframes/2+1));
invFFT(imsize/2+halfspace+1, imsize/2 +1,nframes/2+1) = ...
    conj( invFFT(imsize/2-halfspace+1, imsize/2+1,nframes/2+1));

figure
imagesc(abs(invFFT(:,:,nframes/2+1)));
figure
imagesc(angle(invFFT(:,:,nframes/2)));

pack
shiftinvFFT = ifftshift(invFFT);
clear invFFT;

%%% invert FFT and scale it to 0 -255

imraw = real(ifftn(shiftinvFFT));
clear shiftinvFFT;
immean = mean(imraw(:))
immax = std(imraw(:))/contrastSigma
immin = -1*immax
imscaled = (imraw - immin-immean) / (immax - immin);
clear imfiltered;
contrast_period =10;

%%% modify movie for different patterns (patches, bars, etc)
if binarize
    imscaled(imscaled<0.5)=1-contrast_luminance;
    imscaled(imscaled>0.5)=1*contrast_luminance;
end
%%
%imscaled = imsave;
imscaled(imheight+1:end,:,:) = [];
imscaled = imscaled(1:imheight,1:imsize,:);
t = 0;
for i = 1:Xpatches
    frame = ones(imheight,imsize,nframes).*(background/256);
    frame(:,t+1:t+(imsize/Xpatches),:) = imscaled(:,1:(imsize/Xpatches),:);
    t = t + (imsize/Xpatches);
    for j = 1:size(frame,3)
        F(j) = im2frame(repmat(frame(:,:,j),1,1,3));
    end
    v = VideoWriter(strcat('D:\Analysis_scripts\Dropbox\AndermannLab\users\jasmine\stimuli\retinotopy\NoiseBarVert',num2str(i),'of',num2str(Xpatches),'.avi'));
    open(v)
    writeVideo(v,F)
    close(v)
end

t = 0;
for i = 1:Ypatches
    frame = ones(imheight,imsize,nframes).*(background/256);
    frame(t+1:t+(imheight/Xpatches),:,:) = imscaled(1:(imheight/Xpatches),:,:);
    t = t + (imheight/Xpatches);
    for j = 1:size(frame,3)
        F(j) = im2frame(repmat(frame(:,:,j),1,1,3));
    end
    v = VideoWriter(strcat('D:\Analysis_scripts\Dropbox\AndermannLab\users\jasmine\stimuli\retinotopy\NoiseBarHorz',num2str(i),'of',num2str(Ypatches),'.avi'));
    open(v)
    writeVideo(v,F)
    close(v)
end

