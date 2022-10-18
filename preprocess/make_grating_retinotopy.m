%% 
j = 1;
for i = 1:120
    load(['C:\Users\nnguyen\Desktop\CSp_CSn_CSm_delay\135deg_FF_primetime_preprocessed',num2str(j)]);
    M2 = reshape(M,1280,1024)';
    load(['NoiseBarVert1_preprocessed',num2str(i)]);
    M = reshape(M,1280,1024)';
    M2(:,1:960) = 9276301;
    M = M2;
    M = uint32(reshape(M',1280*1024,1));
    save(['NoiseBarVert1_preprocessed',num2str(i)],'M','xis','yis','xisbuf','yisbuf')
    j = j+1;
    if mod(i,30) == 0
        j = 1;
    end
end
j = 1;
for i = 1:120
    load(['C:\Users\nnguyen\Desktop\CSp_CSn_CSm_delay\135deg_FF_primetime_preprocessed',num2str(j)]);
    M2 = reshape(M,1280,1024)';
    load(['NoiseBarVert2_preprocessed',num2str(i)]);
    M = reshape(M,1280,1024)';
    M2(:,1:640) = 9276301;
    M2(:,961:1280) = 9276301;
    M = M2;
    M = uint32(reshape(M',1280*1024,1));
    save(['NoiseBarVert2_preprocessed',num2str(i)],'M','xis','yis','xisbuf','yisbuf')
    j = j+1;
    if mod(i,30) == 0
        j = 1;
    end
end
j = 1;
for i = 1:120
    load(['C:\Users\nnguyen\Desktop\CSp_CSn_CSm_delay\135deg_FF_primetime_preprocessed',num2str(j)]);
    M2 = reshape(M,1280,1024)';
    load(['NoiseBarVert3_preprocessed',num2str(i)]);
    M = reshape(M,1280,1024)';
    M2(:,1:320) = 9276301;
    M2(:,641:1280) = 9276301;
    M = M2;
    M = uint32(reshape(M',1280*1024,1));
    save(['NoiseBarVert3_preprocessed',num2str(i)],'M','xis','yis','xisbuf','yisbuf')
    j = j+1;
    if mod(i,30) == 0
        j = 1;
    end
end
j = 1;
for i = 1:120
    load(['C:\Users\nnguyen\Desktop\CSp_CSn_CSm_delay\135deg_FF_primetime_preprocessed',num2str(j)]);
    M2 = reshape(M,1280,1024)';
    load(['NoiseBarVert4_preprocessed',num2str(i)]);
    M = reshape(M,1280,1024)';
    M2(:,321:1280) = 9276301;
    M = M2;
    M = uint32(reshape(M',1280*1024,1));
    save(['NoiseBarVert4_preprocessed',num2str(i)],'M','xis','yis','xisbuf','yisbuf')
    j = j+1;
    if mod(i,30) == 0
        j = 1;
    end
end
j = 1;
for i = 1:120
    load(['C:\Users\nnguyen\Desktop\CSp_CSn_CSm_delay\135deg_FF_primetime_preprocessed',num2str(j)]);
    M2 = reshape(M,1280,1024)';
    load(['NoiseBarHorz1_preprocessed',num2str(i)]);
    M = reshape(M,1280,1024)';
    M2(:,1:640) = 9276301;
    M = M2;
    M = uint32(reshape(M',1280*1024,1));
    save(['NoiseBarHorz1_preprocessed',num2str(i)],'M','xis','yis','xisbuf','yisbuf')
    j = j+1;
    if mod(i,30) == 0
        j = 1;
    end
end
j = 1;
for i = 1:120
    load(['C:\Users\nnguyen\Desktop\CSp_CSn_CSm_delay\135deg_FF_primetime_preprocessed',num2str(j)]);
    M2 = reshape(M,1280,1024)';
    load(['NoiseBarHorz2_preprocessed',num2str(i)]);
    M = reshape(M,1280,1024)';
    M2(:,641:1280) = 9276301;
    M = M2;
    M = uint32(reshape(M',1280*1024,1));
    save(['NoiseBarHorz2_preprocessed',num2str(i)],'M','xis','yis','xisbuf','yisbuf')
    j = j+1;
    if mod(i,30) == 0
        j = 1;
    end
end
%%
Ma = zeros(1024,1280,120);
for i = 1:120
load(['NoiseBarVert4_preprocessed',num2str(i)]);
M = reshape(M,1280,1024)';
Ma(:,:,i) = M;
end
pipe.io.write_tiff(Ma,'test');