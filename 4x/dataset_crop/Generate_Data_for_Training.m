%% Initialization
clear all;
clc;

%% Parameters setting
angRes = 5;                 % Angular Resolution, options, e.g., 3, 5, 7, 9. Default: 5
factor =4;                 % SR factor
patchsize = 12 * factor;  	% Spatial resolution of each SAI patch
stride = patchsize/2;       % stride between two patches. Default: 32
downRatio = 1/factor;
src_data_path = 'E:\数据集(异构)\TrainSet/';
src_datasets = dir(src_data_path);
% src_datasets(1:2) = [];
num_scene = length(src_datasets); 


%% Training data generation
idx_save = 0;
    for index_scene = 3 : num_scene
        
        % Load LF image
        idx_scene_save = 0;
        name_scene = src_datasets(index_scene).name;
        fprintf('Generating training data of Scene_%s in Dataset......\n', name_scene);
        data_path = [src_data_path, name_scene];
        data = im2single(imread(data_path));
%         imshow(data)
        [h,w,c] = size(data);
        h = h/9;
        w=w/9;

        LF= reshape(data,[9,h,9,w,3]);
        LF = permute(LF,[1,3,2,4,5]);
        [U, V, ~, ~, ~] = size(LF);
         
        % Extract central angRes*angRes views   (9x9----------->5x5)
        LF = LF(0.5*(U-angRes+2):0.5*(U+angRes), 0.5*(V-angRes+2):0.5*(V+angRes), :, :, 1:3); 
        [U, V, H, W, ~] = size(LF);
                
        % Generate patches of size 32*32
        for h = 1 : stride : H - patchsize + 1
            for w = 1 : stride : W - patchsize + 1
                 idx_save = idx_save + 1;
                idx_scene_save = idx_scene_save + 1;
                Hr_SAI_y = single(zeros(U * patchsize, V * patchsize));
                Lr_SAI_y = single(zeros(U * patchsize * downRatio, V * patchsize * downRatio));             

                for u = 1 : U
                    for v = 1 : V     
                        x = (u-1) * patchsize + 1;
                        y = (v-1) * patchsize + 1;
                        
                        % Convert to YCbCr
                        patch_Hr_rgb = double(squeeze(LF(u, v, h : h+patchsize-1, w : w+patchsize-1, :)));
%                         imshow(patch_Hr_rgb)
                          patch_Hr_ycbcr = rgb2ycbcr(patch_Hr_rgb);
                          patch_Hr_y = squeeze(patch_Hr_ycbcr(:,:,1)); 
%                          imshow(patch_Hr_y)
                                                
                        patchsize_Lr = patchsize / factor;
                        Hr_SAI_y(x:x+patchsize-1, y:y+patchsize-1) = single(patch_Hr_y);
                        patch_Sr_y = imresize(patch_Hr_y, downRatio);
                        Lr_SAI_y((u-1)*patchsize_Lr+1 : u*patchsize_Lr, (v-1)*patchsize_Lr+1:v*patchsize_Lr) = single(patch_Sr_y);
%                         imshow(Lr_SAI_y)
                    end
                end

                SavePath = ['E:\dataset(fixed point)\data_for_training/SR_', num2str(angRes), 'x' , num2str(angRes), '_' ,num2str(factor), 'x/' ];
                if exist(SavePath, 'dir')==0
                    mkdir(SavePath);
                end

                SavePath_H5 = [SavePath, num2str(idx_save,'%06d'),'.h5'];
%                 Lr_SAI_y
                h5create(SavePath_H5, '/Lr_SAI_y', size(Lr_SAI_y), 'Datatype', 'single');
                h5write(SavePath_H5, '/Lr_SAI_y', single(Lr_SAI_y), [1,1], size(Lr_SAI_y));
                
                h5create(SavePath_H5, '/Hr_SAI_y', size(Hr_SAI_y), 'Datatype', 'single');
                h5write(SavePath_H5, '/Hr_SAI_y', single(Hr_SAI_y), [1,1], size(Hr_SAI_y));
                
            end
        end
        fprintf([num2str(idx_scene_save), ' training samples have been generated\n']);
    end



