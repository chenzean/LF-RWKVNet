%% Initialization
clear all;
clc;

%% Parameters setting
angRes = 5;                 % Angular Resolution, options, e.g., 3, 5, 7, 9. Default: 5
factor = 8;                 % SR factor
patchsize = 12*factor;  	% Spatial resolution of each SAI patch
stride = patchsize/2;       % stride between two patches. Default: 16
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
        name_temp = strsplit(name_scene,'.');
        name_out = name_temp{1,1}
        fprintf('Generating training data of Scene_%s in Dataset......\n', name_scene);
        data_path = [src_data_path, name_scene];
        data = im2single(imread(data_path));
%         imshow(data)
        [h,w,c] = size(data);
        h = h/9;
        w=w/9;

        LF= reshape(data,[9,h,9,w,3]);
        LF = permute(LF,[1,3,2,4,5]);
        [U, V, H, W, C] = size(LF);
        
        ref =zeros(H, W, C);
%         num=1;
%         cd('E:\异构超分2\data\ref_mat')
        for u=1:9
            for l=1:9
                if u==5 && l==9
                    ref(:,:,:)=LF(u,l,:,:,:);
%                     num=num+1;
                end
            end
        end
%              save([num2str(name_out),'.mat'],'ref')
            [height,width,c]=size(ref);        % 56   434   625     3
%             Hr_ref = single(zeros(n,patchsize,patchsize, c));
%             ref_sai = single(zeros(n,patchsize,patchsize, 1));
             for y = 1:stride:(height - patchsize + 1)
                  for x = 1:stride:(width - patchsize + 1)
                                                   idx_save = idx_save + 1;
%                          if x+patchsize-1<height && y+patchsize-1<width

                            Hr_ref = single(double(ref(y:y+patchsize-1,x:x+patchsize-1,:) ));
                            Hr_ref_ycbcr = rgb2ycbcr(Hr_ref(:,:,:));
                            ref_sai(:,:,:) = squeeze(Hr_ref_ycbcr(:,:,1));
%                             sad = sad+1
%                                 for m=1:n
%                                     Hr_ref_ycbcr = rgb2ycbcr(squeeze(Hr_ref(m,:,:,:)));
%                                     ref_sai(:,:,:) = squeeze(Hr_ref_ycbcr(:,:,1));
%                                 end
                 % 生成相应长度的 START、COUNT 和 STRIDE 向量
                dataset_dims = size(ref_sai);   % , 64, 64

%                 START = ones(1, numel(dataset_dims));
%                 COUNT = dataset_dims;
%                 STRIDE = ones(1, numel(dataset_dims));
                SavePath = ['E:\dataset(fixed point)\data_for_training\ref_8x\' ];
                
                SavePath_H5 = [SavePath, num2str(idx_save,'%06d'),'.h5'];
                h5create(SavePath_H5, '/ref_sai', size(ref_sai), 'Datatype', 'single');
                h5write(SavePath_H5, '/ref_sai', single(ref_sai), [1,1], size(ref_sai));
%                          end
            
                  end
             end

            
            
            
            
            
            
            
%            for o=1:56
%                ref_temp = squeeze(ref(o,:,:,:));
%                [height, width, ~] = size(ref_temp);
%                ref_temp_ycbcr = rgb2ycbcr(ref_temp);
%                ref_Hr_y = squeeze(ref_temp_ycbcr(:,:,1));
%                eeee=1;
%                Hr_ref=zeros(128,128);
%                for y = 1:stride:(height - patchsize + 1)
%                    for x = 1:stride:(width - patchsize + 1)
%                        if x+patchsize-1<height &&y+patchsize-1<width
%                             Hr_ref(x:x+patchsize-1, y:y+patchsize-1) = single(ref_Hr_y(x:x+patchsize-1, y:y+patchsize-1));
%                             eeee=eeee+1
%                        end
%                    end
%                end
%                
%                
%            end
            
    
    end