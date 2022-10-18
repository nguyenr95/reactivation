%load movie
v = VideoReader('NN11_210705_001_eye.mj2');
frames = read(v,[1 Inf]);
%% save movie
m = VideoWriter('pupil.avi');
open(m)
writeVideo(m,frames)
close(m)