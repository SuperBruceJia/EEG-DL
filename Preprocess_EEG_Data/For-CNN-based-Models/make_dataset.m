clear all
clc

format long

%% Read the Data and Create Dataset
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
ALL = [Dataset, Labels];
rowrank = randperm(size(ALL, 1));
ALL_Dataset = ALL(rowrank, :);
[row, ~] = size(ALL_Dataset);

%%
training_set   = ALL_Dataset(1:fix(row/10*9),     1:4096);
test_set       = ALL_Dataset(fix(row/10*9)+1:end, 1:4096);

training_label = ALL_Dataset(1:fix(row/10*9),     end);
test_label     = ALL_Dataset(fix(row/10*9)+1:end, end);

all_data       = ALL_Dataset(:, 1:4096);
all_labels     = ALL_Dataset(:, end);

%%
xlswrite('training_set.xlsx', training_set);
xlswrite('test_set.xlsx', test_set);

xlswrite('training_label.xlsx', training_label);
xlswrite('test_label.xlsx', test_label);

xlswrite('all_data.xlsx', all_data);
xlswrite('all_labels.xlsx', all_labels);
