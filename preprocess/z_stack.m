%%
z_stack = zeros(512, 796, 7);
for i = 1:7
    z_stack(:,:,i) = uint16(mean(movies{1,i}, 3));
end



%%
z_stack = {};
p = [4, 3, 2, 1, 5, 6, 7];
for i = 1:7
    z_stack{i} = uint16(mean(movies{1,p(i)},3));
end
z_stack = cell2mat(z_stack); 
z_stack = reshape(z_stack, 512, 796, 7);

% for i = 1:7
%     z_stack(i,:,:) = uint16(mean(movies{1,p(i)},3));
% end
pipe.io.write_tiff(z_stack,'z_stack');
implay(z_stack);