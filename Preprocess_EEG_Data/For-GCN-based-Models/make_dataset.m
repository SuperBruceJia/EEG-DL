clear all
clc

format long

%%
% Read the Data and Create Dataset
Stack_Dataset = [];
num_trial = 20;

for i = 1:64
    Dataset = ['Dataset_', num2str(i), '.mat'];
    Dataset = load(Dataset);
    Dataset = Dataset.Dataset;
    Dataset = reshape(Dataset, num_trial*84, 640);
    
    [row, column] = size(Dataset);
    Dataset = reshape(Dataset', 1, row*column);
    Stack_Dataset = [Stack_Dataset; Dataset];
end

Stack_Dataset = Stack_Dataset - mean(Stack_Dataset, 1);
Stack_Dataset = Stack_Dataset';

%% Compute Covariance Matrix
covariance_matrix = cov(Stack_Dataset);
xlswrite('covariance_matrix.xlsx', covariance_matrix);

figure(1)
imagesc(covariance_matrix)
axis square
title('Covariance Matrix for 20 Subjects', 'FontName', 'Times New Roman', 'FontSize', 16, 'FontWeight', 'bold')
xlabel('Channels'), ylabel('Channels')
set(gca, 'FontName', 'Times New Roman', 'FontSize', 16, 'FontWeight', 'bold');
colorbar

print('Covariance_Matrix_for_20_Subjects', '-dpng',  '-r600')

%% Compute Pearson Matrix
Pearson_matrix = corrcoef(Stack_Dataset);
xlswrite('Pearson_matrix.xlsx', Pearson_matrix);

figure(2)
imagesc(Pearson_matrix)
axis square
title('Pearson Matrix for 20 Subjects', 'FontName', 'Times New Roman', 'FontSize', 16, 'FontWeight', 'bold')
xlabel('Channels'), ylabel('Channels')
set(gca, 'FontName', 'Times New Roman', 'FontSize', 16, 'FontWeight', 'bold');
colorbar

print('Pearson_matrix_for_20_Subjects', '-dpng',  '-r600')

%% Compute Absolute Pearson Matrix
Absolute_Pearson_matrix = abs(Pearson_matrix);
xlswrite('Absolute_Pearson_matrix.xlsx', Absolute_Pearson_matrix);

figure(3)
imagesc(Absolute_Pearson_matrix)
axis square
title('Absolute Pearson Matrix for 20 Subjects', 'FontName', 'Times New Roman', 'FontSize', 16, 'FontWeight', 'bold')
xlabel('Channels'), ylabel('Channels')
set(gca, 'FontName', 'Times New Roman', 'FontSize', 16, 'FontWeight', 'bold');
colorbar

print('Absolute_Pearson_matrix_for_20_Subjects', '-dpng',  '-r600')

%% Compute Adjacency Matrix
Eye_Matrix = eye(64, 64);
Adjacency_Matrix = Absolute_Pearson_matrix - Eye_Matrix;
xlswrite('Adjacency_Matrix.xlsx', Adjacency_Matrix);

figure(4)
imagesc(Adjacency_Matrix)
axis square
title('Adjacency Matrix for 20 Subjects', 'FontName', 'Times New Roman', 'FontSize', 16, 'FontWeight', 'bold')
xlabel('Channels'), ylabel('Channels')
set(gca, 'FontName', 'Times New Roman', 'FontSize', 16, 'FontWeight', 'bold');
colorbar

print('Adjacency_Matrix_for_20_Subjects', '-dpng',  '-r600')

%% Compute Degree Matrix
diagonal_vector = sum(Adjacency_Matrix, 2);
Degree_Matrix = diag(diagonal_vector);

figure(5)
imagesc(Degree_Matrix)
axis square
title('Degree Matrix for 20 Subjects', 'FontName', 'Times New Roman', 'FontSize', 16, 'FontWeight', 'bold')
xlabel('Channels'), ylabel('Channels')
set(gca, 'FontName', 'Times New Roman', 'FontSize', 16, 'FontWeight', 'bold');
colorbar

print('Degree_Matrix_for_20_Subjects', '-dpng',  '-r600')

%% Compute Laplacian Matrix
Laplacian_Matrix = Degree_Matrix - Adjacency_Matrix;

figure(6)
imagesc(Laplacian_Matrix)
axis square
title('Laplacian Matrix for 20 Subjects', 'FontName', 'Times New Roman', 'FontSize', 16, 'FontWeight', 'bold')
xlabel('Channels'), ylabel('Channels')
set(gca, 'FontName', 'Times New Roman', 'FontSize', 16, 'FontWeight', 'bold');
colorbar

print('Laplacian_Matrix_for_20_Subjects', '-dpng',  '-r600')

%% Read and Create Labels
Labels = load('Labels_1.mat');
Labels = Labels.Labels;
Labels = reshape(Labels, num_trial*84, 4);
[row, column] = size(Labels);

New_Labels = [];
parfor i = 1:row
    location = find(Labels(i, :) == 1);
    location = location - 1;
    New_Labels = [New_Labels; location];
end
Labels = New_Labels;

Extend_Labels = [];
parfor i =1:640
    Extend_Labels = [Extend_Labels, Labels];
end
Labels = Extend_Labels;

[row, column] = size(Labels);
Labels = reshape(Labels', 1, row*column);
Labels = Labels';

%%
All_Data = [Stack_Dataset, Labels];
rowrank = randperm(size(All_Data, 1));
All_Dataset = All_Data(rowrank, :);
[row, ~] = size(All_Dataset);

training_set   = All_Dataset(1:fix(row/10*9),     1:64);
test_set       = All_Dataset(fix(row/10*9)+1:end, 1:64);
training_label = All_Dataset(1:fix(row/10*9),     end);
test_label     = All_Dataset(fix(row/10*9)+1:end, end);

xlswrite('training_set.xlsx', training_set);
xlswrite('test_set.xlsx', test_set);
xlswrite('training_label.xlsx', training_label);
xlswrite('test_label.xlsx', test_label);
