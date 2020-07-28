clear all
clc

format long

%% Read the Raw EEG Data and Create Dataset
Stack_Dataset = [];
num_subject   = 20;
num_trial     = 84;
num_channel   = 64;
num_data      = 640;
Time_consider = 4 / 10;
Data_points   = Time_consider * 160;

for i = 1:num_channel
    Dataset = ['Dataset_', num2str(i), '.mat'];
    Dataset = load(Dataset);
    Dataset = Dataset.Dataset;
    Dataset = reshape(Dataset, num_subject*num_trial, num_data);
    
    [row, column] = size(Dataset);
    Dataset = reshape(Dataset', 1, row*column);
    Stack_Dataset = [Stack_Dataset; Dataset];
end

% Eliminate the influence of Reference Electrode
Stack_Dataset = Stack_Dataset - mean(Stack_Dataset, 1);

% Normalized the signals
Stack_Dataset = reshape(Stack_Dataset, [num_channel, num_subject*num_trial, num_data]);
[m, n, k] = size(Stack_Dataset);
for i = 1:m
    for j = 1:k
        mean_x = mean(Stack_Dataset(i, :, j));
        std_x  = std(Stack_Dataset(i, :, j));
        Stack_Dataset(i, :, j) = (Stack_Dataset(i, :, j) - mean_x) / std_x;
    end
end

Stack_Dataset = reshape(Stack_Dataset, [num_channel, num_subject*num_trial*num_data]);

[~, columns] = size(Dataset);
Dataset = reshape(Stack_Dataset, [num_channel*Data_points, columns/Data_points]);
Dataset = Dataset';

%% Read and Create Labels
Labels = load('Labels_1.mat');
Labels = Labels.Labels;
Labels = reshape(Labels, num_subject*num_trial, 4);
[row, column] = size(Labels);

New_Labels = [];
parfor i = 1:row
    location = find(Labels(i, :) == 1);
    location = location - 1;
    New_Labels = [New_Labels; location];
end
Labels = New_Labels;

Extend_Labels = [];
parfor i =1:num_data
    Extend_Labels = [Extend_Labels, Labels];
end
Labels = Extend_Labels;

[row, column] = size(Labels);
Labels = reshape(Labels', 1, row*column);

[~, column] = size(Labels);
Labels = reshape(Labels, [Data_points, column / Data_points]);
Labels = Labels(1, :);
Labels = Labels';

%% Connecting the Dataset and Corresponding Labels
% and shuffle the total Dataset
ALL = [Dataset, Labels];
rowrank = randperm(size(ALL, 1));
ALL_Dataset = ALL(rowrank, :);
[row, column] = size(ALL_Dataset);

%% Make Dataset for One-shot Learning
% if the samples are the same class, then the label will be 1
% Otherwise, the label is 0
half_row = row / 2;
new_labels = [];

for i = 1: half_row
    if ALL_Dataset(i, end) == ALL_Dataset(half_row + i, end)
        new_label = 1;
    else
        new_label = 0;
    end
    new_labels = [new_labels; new_label];
end

%% Save Dataset
data_set_1 = ALL_Dataset(1:half_row, 1:4096);
data_set_2 = ALL_Dataset(half_row:end, 1:4096);
final_labels = new_labels;

training_set_1 = data_set_1(1:fix(half_row/10*9),     1:4096);
test_set_1     = data_set_1(fix(half_row/10*9)+1:end, 1:4096);

training_set_2 = data_set_2(1:fix(half_row/10*9),     1:4096);
test_set_2     = data_set_2(fix(half_row/10*9)+1:end, 1:4096);

training_label = final_labels(1:fix(half_row/10*9),     end);
test_label     = final_labels(fix(half_row/10*9)+1:end, end);

%% Save Dataset as CSV Files (on Macbook) 
xlswrite('training_set_1.xlsx', training_set_1);
xlswrite('test_set_1.xlsx', test_set_1);

xlswrite('training_set_2.xlsx', training_set_2);
xlswrite('test_set_2.xlsx', test_set_2);

xlswrite('training_label.xlsx', training_label);
xlswrite('test_label.xlsx', test_label);
