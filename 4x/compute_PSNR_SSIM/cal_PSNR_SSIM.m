clear all
clc 
%%
 path1 = 'E:\工作点2\code 上传版本\x8\LFSSR-code\log\SR_5x5_8x\ALL\LF-DET\results\TEST\Test\';
% path1 = 'E:\工作点2\多尺度 loss等价\LFSSR-code\log\SR_5x5_8x\ALL\LF-DET\results\TEST\Test\';
path2 = 'E:\混合域学习用于基于从异构式成像的光场空间超分辨率\其他算法的运行结果Results\gt_sai\';
total_psnr = zeros(1, 70);
total_ssim = zeros(1, 70);
file_path_dis1 = dir(path1);
file_path_dis1=file_path_dis1(3:end);
% file_path_dis1=file_path_dis1(3:end);

file_path_dis2= dir(path2);
file_path_dis2=file_path_dis2(3:end);

scene_num = length(file_path_dis1);

for scene_i = 1:scene_num
    scene_i
    sub_path1= [path1,file_path_dis1(scene_i).name,'/views'];
%     sub_path1= [path1,file_path_dis1(scene_i).name]; 
    sub_path2= [path2,file_path_dis2(scene_i).name]; 
    temp1=0;
    temp2=0;
    for p = 1:5
        for q= 1:5
        im1 = imread([sub_path1,'/',num2str(scene_i),'_',sprintf('%01d',p-1),'_',sprintf('%01d',q-1),'.bmp']);
        im2 = imread([sub_path2,'/',sprintf('%02d',p),'_',sprintf('%02d',q),'.png']); 
            
        ima1 = rgb2ycbcr(im1);
        ima2 = rgb2ycbcr(im2);
        ima1 = im2double(ima1);
        ima2 = im2double(ima2);
%         [h1,w1,c]=size(ima2);
%         h1_ = h1-1;
%         w1_ = w1-1;
        ima2 =  ima2(2:h1_,2:w1_,:);
%         [h1,w1,c]=size(ima1);
%         h1_ = h1-1;
%         w1_ = w1-1;
        ima1 =  ima1(2:h1_,2:w1_,:);
        
        val_ima_psnr = cal_psnr(ima2(:,:,1), ima1(:,:,1));
        val_ima_ssim = ssim(ima2(:,:,1), ima1(:,:,1));
%             im1 = im2single(imread([sub_path1,'/',num2str(scene_i),'_',sprintf('%01d',p-1),'_',sprintf('%01d',q-1),'.bmp']));
%             im1 = im2single(imread([sub_path1,'/',sprintf('%02d',p),'_',sprintf('%02d',q),'.png']));
%             im2 = im2single(imread([sub_path2,'/',sprintf('%02d',p),'_',sprintf('%02d',q),'.png'])); 
%             im1_ycbcr = rgb2ycbcr(im1);  
%             im_y1_new = im1_ycbcr(:,:,1);
%             [h1,w1]=size(im_y1_new);
%             h1_ = h1-1;
%             w1_ = w1-1;
%             im_y1 =  im_y1_new(2:h1_,2:w1_);
            
%             im2_ycbcr = rgb2ycbcr(im2);  
%             im_y2_new = im2_ycbcr(:,:,1);
%             [h2,w2]=size(im_y2_new);
%             h2_ = h2-1;
%             w2_ = w2-1;
%             im_y2 =  im_y2_new(2:h2_,2:w2_);

            
%             val_ima_psnr = cal_psnr(im_y1_new, im_y2_new);
%             val_ima_ssim = ssim(im_y1_new, im_y2_new);
            
            temp1 = val_ima_psnr+temp1;
            temp2 = val_ima_ssim+temp2;
        end
    end
    temp2 = temp2/25;
    temp1 = temp1/25;
    total_psnr(scene_i)= temp1;
    total_ssim(scene_i)= temp2;
    save('epoch=80_final_PSNR_and_SSIM.mat', 'total_psnr', 'total_ssim');
    
end
mean_ssim = mean(total_psnr);
mean_psnr = mean(total_ssim);




function psnr_value = cal_psnr(dis,ref)

% Input image: [0,1]
dis = double(dis);
ref = double(ref);
element_num = length(dis(:));
diff = (dis-ref).^2;
mse = sum(diff(:))/element_num;
psnr_value = 10*log10(1^2/mse);

end

function q = ssim(I,J)
K = [0.01,0.03];
L = 1;
C1 = (K(1)*L)^2;
C2 = (K(2)*L)^2;
window =  fspecial('gaussian', 11, 1.5);	
window = window/sum(window(:));

I1 = double(I);
I2 = double(J);

mu1   = filter2(window, I1(:,:), 'same');
mu2   = filter2(window, I2(:,:), 'same');

mu1_sq = mu1.*mu1;    % E(x)^2
mu2_sq = mu2.*mu2;    % E(y)^2
mu1_mu2 = mu1.*mu2;   % E(x)*E(y)
sigma1_sq = filter2(window, I1(:,:).*I1(:,:), 'same') - mu1_sq;       % D(x)=E(x.^2)- E(x)^2
sigma2_sq = filter2(window, I2(:,:).*I2(:,:), 'same') - mu2_sq;       % D(y)=E(y.^2)- E(y)^2
sigma12 = filter2(window, I1(:,:).*I2(:,:), 'same') - mu1_mu2;        % cov(x,y)=E(x*y)- E(x)*E(y)
cal_SSIM_map = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))./((mu1_sq + mu2_sq + C1).*(sigma1_sq + sigma2_sq + C2));

q= mean(cal_SSIM_map(:));
end