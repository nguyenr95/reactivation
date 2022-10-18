function h = errorbar_shade(x,y,dy,color,varargin)

if nargin<5, linewidth = 1; else, linewidth = varargin{1}; end
if nargin<6, linespec = '-'; else, linespec = varargin{2}; end
if nargin<7, markersize = 10; else, markersize = varargin{3}; end
% function h = errorbar_patch(x,y,dy,color,linewidth)
% plots the line x vs y and a shaded background defined by dx
%       and returns the handles to the line & patch objects
% y + dy(:,1) is used for the upper boundary of the shaded region
% y - dy(:,2) is used for the upper boundary of the shaded region
% if dy is a vector y - dy(:,1) is used for the lower boundary 
% if color has two rows the 1st is for the line & the 2nd for the shading
% if color has one row a lighter version of the line color is used for the
% shading
% example: 
% figure; errorbar_patch(1:100,rand(100,1)+10,+.1,[0 .6 1],2); shg
% - MAS 9/6/2007



if nargin < 4, color = [1 0 0]; end
if nargin < 5, linewidth = 2; end
if size(color,1)<2, color(2,1:3)=1-[1-color(1:3)]/4; end
if size(x,1)==1; x=x'; end
if size(y,1)==1; y=y'; dy=dy'; end
if size(dy,2)==1, dy = [dy,dy]; end

ylow = y-dy(:,1); yhigh=y+dy(:,2);
h2 = patch([x;flipud(x)],[ylow;flipud(yhigh)],color(2,1:3)); hold on; 
h1 = plot(x,y,linespec,'linewidth',linewidth); 
set(h2,'edgecolor','none');
set(h1,'color',color(1,1:3),'markerfacecolor',color(1,1:3),'markersize',markersize);
h=[h1;h2];
alpha(.75)
