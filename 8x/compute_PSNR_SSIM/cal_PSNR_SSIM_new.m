clear all
clc 
%%  需要修改内容：model_name， up_sample，path1
model_name = 'SPL';
up_sample = '8x';

% 路径要对否则可能会显示segments有关的错误
path1 = 'C:\Users\Administrator\Desktop\RWKV蛇形卷积_错误修正\log\SR_5x5_8x\ALL\LF-RWKVNet\results\TEST\Test\Results\';
path2 = 'E:\work 3\Results from different methods\gt-sai\';
total_psnr = zeros(1, 70);
total_ssim = zeros(1, 70);
file_path_dis1 = dir(path1);
file_path_dis1 = file_path_dis1(3:end);

file_path_dis2 = dir(path2);
file_path_dis2 = file_path_dis2(3:end);

scene_num = length(file_path_dis1);

formats = {'.png', '.bmp'};
naming_conventions = {
    @(scene_i, p, q) sprintf('%d_%01d_%01d', scene_i, p-1, q-1),
    @(scene_i, p, q) sprintf('%d_%02d_%02d', scene_i, p, q),
    @(scene_i, p, q) sprintf('%02d_%02d', p, q),
    @(scene_i, p, q) sprintf('%01d_%01d', p, q),
    @(scene_i, p, q) sprintf('0%01d_0%01d', p, q)
};

for scene_i = 1:scene_num
    scene_i
    sub_path1 = [path1, file_path_dis1(scene_i).name]; 
    sub_path2 = [path2, file_path_dis2(scene_i).name]; 
    temp1 = 0;
    temp2 = 0;
    for p = 1:5
        for q = 1:5
            im1 = [];
            for format_idx = 1:length(formats)
                format = formats{format_idx};
                for name_idx = 1:length(naming_conventions)
                    naming = naming_conventions{name_idx};
                    try
                        im1 = imread([sub_path1, '/', naming(scene_i, p, q), format]);
                        break;
                    catch
                        % Continue to next format if im1 is not found
                    end
                end
                if ~isempty(im1)
                    break;
                end
            end
            
            if isempty(im1)
                error('Image im1 not found in any of the specified formats.');
            end
            
            im2 = imread([sub_path2, '/', sprintf('%02d_%02d', p, q), '.png']);
            
            ima1 = rgb2ycbcr(im1);
            ima2 = rgb2ycbcr(im2);
            ima1 = im2double(ima1);
            ima2 = im2double(ima2);
            val_ima_psnr = cal_psnr(ima2(:,:,1), ima1(:,:,1));
            val_ima_ssim = ssim(ima2(:,:,1), ima1(:,:,1));
            temp1 = val_ima_psnr + temp1;
            temp2 = val_ima_ssim + temp2;
        end
    end
    temp2 = temp2 / 25;
    temp1 = temp1 / 25;
    total_psnr(scene_i) = temp1;
    total_ssim(scene_i) = temp2;
    
    % 定义分段
    segments = {
        1:10,
        11:14,
        15:16,
        17:26,
        27:36,
        37:38,
        39:70
    };
end

mean_ssim = mean(total_psnr);
mean_psnr = mean(total_ssim);

% 初始化存储平均值的数组
means1 = zeros(1, length(segments));
means2 = zeros(1, length(segments));

% 保存结果
save([up_sample, '_', model_name, '_final_PSNR_and_SSIM_indicators.mat'], 'total_psnr', 'total_ssim');

% 计算每个分段的平均值
for i = 1:length(segments)
    segment = segments{i};
    means1(i) = mean(total_psnr(segment));
    means2(i) = mean(total_ssim(segment));
end

% 显示结果
fprintf('EPFL: %.2f  %.3f \n', means1(1), means2(1));
fprintf('HCI new: %.2f  %.3f \n', means1(2), means2(2));
fprintf('HCI old: %.2f  %.3f \n', means1(3), means2(3));
fprintf('INRIA: %.2f  %.3f  \n', means1(4), means2(4));
fprintf('Kalantari: %.2f  %.3f  \n', means1(5), means2(5));
fprintf('STFGantry: %.2f  %.3f  \n', means1(6), means2(6));
fprintf('STFLytro: %.2f  %.3f  \n', means1(7), means2(7));

fprintf('ave: %.2f  %.3f  \n', mean(means1), mean(means2));

function psnr_value = cal_psnr(dis, ref)
    dis = double(dis);
    ref = double(ref);
    element_num = length(dis(:));
    diff = (dis - ref).^2;
    mse = sum(diff(:)) / element_num;
    psnr_value = 10 * log10(1^2 / mse);
end

function q = ssim(I, J)
    K = [0.01, 0.03];
    L = 1;
    C1 = (K(1) * L)^2;
    C2 = (K(2) * L)^2;
    window = fspecial('gaussian', 11, 1.5);    
    window = window / sum(window(:));

    I1 = double(I);
    I2 = double(J);

    mu1 = filter2(window, I1(:,:), 'same');
    mu2 = filter2(window, I2(:,:), 'same');

    mu1_sq = mu1 .* mu1;
    mu2_sq = mu2 .* mu2;
    mu1_mu2 = mu1 .* mu2;
    sigma1_sq = filter2(window, I1(:,:) .* I1(:,:), 'same') - mu1_sq;
    sigma2_sq = filter2(window, I2(:,:) .* I2(:,:), 'same') - mu2_sq;
    sigma12 = filter2(window, I1(:,:) .* I2(:,:), 'same') - mu1_mu2;
    cal_SSIM_map = ((2 * mu1_mu2 + C1) .* (2 * sigma12 + C2)) ./ ((mu1_sq + mu2_sq + C1) .* (sigma1_sq + sigma2_sq + C2));

    q = mean(cal_SSIM_map(:));
end
