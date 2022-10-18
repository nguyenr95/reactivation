function colormapNghia()
    for i = 1:255
        if i == 1
        map1 = [0 0 255];
        else
            map1 = vertcat(map1,[i i 255]);
        end
    end
    
    for i = 1:255
        if i == 1
        map2 = [255 0 0];
        else
            map2 = vertcat(map2,[255 i i]);
        end
    end
    map2 = flip(map2);
    
    map = vertcat(map1,map2);

    colormap(map/255);
end