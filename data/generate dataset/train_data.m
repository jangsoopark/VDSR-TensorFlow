clear;
clc;

addpath('./utils/');

%% data path
data_path = '../train_data/291';
save_path = '../train_data/vdsr_train.h5';

%% data configuration
patch_size = 41;
stride = 41;

flip_parameter = [0 2]; % [0 1 2];
rotation_parameter = [0 90]; % [0 90 180 270];

scale = [2 3 4];

%% file path
file_list = [];
file_list = [file_list; dir(fullfile(data_path, '*.jpg'))];
file_list = [file_list; dir(fullfile(data_path, '*.bmp'))];


%% data pairs
gt_list = zeros(patch_size, patch_size, 1, 1);
lr_list = zeros(patch_size, patch_size, 1, 1);

count = 0;

%% generate dataset
for i = 1:numel(file_list)
    %% read image
    disp(file_list(i).name);
    image = imread(fullfile(data_path, file_list(i).name));
    image = rgb2ycbcr(image);
    image = im2double(image(:, :, 1));
    
    %% rotation
    for r = 1:numel(rotation_parameter)
        %% flip
        for f = 1:numel(flip_parameter)
            image_aug = image_augmentation(image, rotation_parameter(r), flip_parameter(f));
            
            for s = 1:numel(scale)
                
                gt = modcrop(image_aug, scale(s));
                [h, w] = size(gt);
                lr = imresize(imresize(gt, 1/scale(s), 'bicubic'), [h, w], 'bicubic');
                
                for y = 1:stride:h - patch_size + 1
                    for x = 1:stride:w - patch_size + 1
                        
                        gt_patch = gt(y: y + patch_size - 1, x: x + patch_size - 1);
                        lr_patch = lr(y: y + patch_size - 1, x: x + patch_size - 1);
                        count = count + 1;
                        
                        gt_list(:, :, 1, count) = gt_patch;
                        lr_list(:, :, 1, count) = lr_patch;
                    end
                end
                
            end
            
        end
    end
end

order = randperm(count);
lr_list = lr_list(:, :, 1, order);
gt_list = gt_list(:, :, 1, order); 

%% writing to HDF5
chunksz = 64;
created_flag = false;
totalct = 0;

for batchno = 1:floor(count/chunksz)
    last_read=(batchno-1)*chunksz;
    batch_lr = lr_list(:,:,1,last_read+1:last_read+chunksz); 
    batch_gt = gt_list(:,:,1,last_read+1:last_read+chunksz);

    startloc = struct('lr',[1,1,1,totalct+1], 'gt', [1,1,1,totalct+1]);
    curr_dat_sz = store2hdf5(save_path, batch_lr, batch_gt, ~created_flag, startloc, chunksz); 
    created_flag = true;
    totalct = curr_dat_sz(end);
end

h5disp(save_path);