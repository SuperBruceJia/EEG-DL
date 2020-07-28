clear all
clc

format long

model1_acc  = readmatrix("Model-20-1/run-.-tag-Global_Average_Accuracy_numpy.csv");
model2_acc  = readmatrix("Model-20-2/run-.-tag-Global_Average_Accuracy_numpy.csv");
model3_acc  = readmatrix("Model-20-3/run-.-tag-Global_Average_Accuracy_numpy.csv");
model4_acc  = readmatrix("Model-20-4/run-.-tag-Global_Average_Accuracy_numpy.csv");
model5_acc  = readmatrix("Model-20-5/run-.-tag-Global_Average_Accuracy_numpy.csv");
model6_acc  = readmatrix("Model-20-6/run-.-tag-Global_Average_Accuracy_numpy.csv");
model7_acc  = readmatrix("Model-20-7/run-.-tag-Global_Average_Accuracy_numpy.csv");
model8_acc  = readmatrix("Model-20-8/run-.-tag-Global_Average_Accuracy_numpy.csv");
model9_acc  = readmatrix("Model-20-9/run-.-tag-Global_Average_Accuracy_numpy.csv");
model10_acc = readmatrix("Model-20-10/run-.-tag-Global_Average_Accuracy_numpy.csv");

model1_Kappa  = readmatrix("Model-20-1/run-.-tag-Kappa_Metric_numpy.csv");
model2_Kappa  = readmatrix("Model-20-2/run-.-tag-Kappa_Metric_numpy.csv");
model3_Kappa  = readmatrix("Model-20-3/run-.-tag-Kappa_Metric_numpy.csv");
model4_Kappa  = readmatrix("Model-20-4/run-.-tag-Kappa_Metric_numpy.csv");
model5_Kappa  = readmatrix("Model-20-5/run-.-tag-Kappa_Metric_numpy.csv");
model6_Kappa  = readmatrix("Model-20-6/run-.-tag-Kappa_Metric_numpy.csv");
model7_Kappa  = readmatrix("Model-20-7/run-.-tag-Kappa_Metric_numpy.csv");
model8_Kappa  = readmatrix("Model-20-8/run-.-tag-Kappa_Metric_numpy.csv");
model9_Kappa  = readmatrix("Model-20-9/run-.-tag-Kappa_Metric_numpy.csv");
model10_Kappa = readmatrix("Model-20-10/run-.-tag-Kappa_Metric_numpy.csv");

model1_precision  = readmatrix("Model-20-1/run-.-tag-Macro_Global_Precision_numpy.csv");
model2_precision  = readmatrix("Model-20-2/run-.-tag-Macro_Global_Precision_numpy.csv");
model3_precision  = readmatrix("Model-20-3/run-.-tag-Macro_Global_Precision_numpy.csv");
model4_precision  = readmatrix("Model-20-4/run-.-tag-Macro_Global_Precision_numpy.csv");
model5_precision  = readmatrix("Model-20-5/run-.-tag-Macro_Global_Precision_numpy.csv");
model6_precision  = readmatrix("Model-20-6/run-.-tag-Macro_Global_Precision_numpy.csv");
model7_precision  = readmatrix("Model-20-7/run-.-tag-Macro_Global_Precision_numpy.csv");
model8_precision  = readmatrix("Model-20-8/run-.-tag-Macro_Global_Precision_numpy.csv");
model9_precision  = readmatrix("Model-20-9/run-.-tag-Macro_Global_Precision_numpy.csv");
model10_precision = readmatrix("Model-20-10/run-.-tag-Macro_Global_Precision_numpy.csv");

model1_recall  = readmatrix("Model-20-1/run-.-tag-Macro_Global_Recall_numpy.csv");
model2_recall  = readmatrix("Model-20-2/run-.-tag-Macro_Global_Recall_numpy.csv");
model3_recall  = readmatrix("Model-20-3/run-.-tag-Macro_Global_Recall_numpy.csv");
model4_recall  = readmatrix("Model-20-4/run-.-tag-Macro_Global_Recall_numpy.csv");
model5_recall  = readmatrix("Model-20-5/run-.-tag-Macro_Global_Recall_numpy.csv");
model6_recall  = readmatrix("Model-20-6/run-.-tag-Macro_Global_Recall_numpy.csv");
model7_recall  = readmatrix("Model-20-7/run-.-tag-Macro_Global_Recall_numpy.csv");
model8_recall  = readmatrix("Model-20-8/run-.-tag-Macro_Global_Recall_numpy.csv");
model9_recall  = readmatrix("Model-20-9/run-.-tag-Macro_Global_Recall_numpy.csv");
model10_recall = readmatrix("Model-20-10/run-.-tag-Macro_Global_Recall_numpy.csv");

model1_f1  = readmatrix("Model-20-1/run-.-tag-Macro_Global_F1_Score_numpy.csv");
model2_f1  = readmatrix("Model-20-2/run-.-tag-Macro_Global_F1_Score_numpy.csv");
model3_f1  = readmatrix("Model-20-3/run-.-tag-Macro_Global_F1_Score_numpy.csv");
model4_f1  = readmatrix("Model-20-4/run-.-tag-Macro_Global_F1_Score_numpy.csv");
model5_f1  = readmatrix("Model-20-5/run-.-tag-Macro_Global_F1_Score_numpy.csv");
model6_f1  = readmatrix("Model-20-6/run-.-tag-Macro_Global_F1_Score_numpy.csv");
model7_f1  = readmatrix("Model-20-7/run-.-tag-Macro_Global_F1_Score_numpy.csv");
model8_f1  = readmatrix("Model-20-8/run-.-tag-Macro_Global_F1_Score_numpy.csv");
model9_f1  = readmatrix("Model-20-9/run-.-tag-Macro_Global_F1_Score_numpy.csv");
model10_f1 = readmatrix("Model-20-10/run-.-tag-Macro_Global_F1_Score_numpy.csv");

model1_acc = model1_acc(end, 3);
model2_acc = model2_acc(end, 3);
model3_acc = model3_acc(end, 3);
model4_acc = model4_acc(end, 3);
model5_acc = model5_acc(end, 3);
model6_acc = model6_acc(end, 3);
model7_acc = model7_acc(end, 3);
model8_acc = model8_acc(end, 3);
model9_acc = model9_acc(end, 3);
model10_acc = model10_acc(end, 3);

model1_Kappa = model1_Kappa(end, 3);
model2_Kappa = model2_Kappa(end, 3);
model3_Kappa = model3_Kappa(end, 3);
model4_Kappa = model4_Kappa(end, 3);
model5_Kappa = model5_Kappa(end, 3);
model6_Kappa = model6_Kappa(end, 3);
model7_Kappa = model7_Kappa(end, 3);
model8_Kappa = model8_Kappa(end, 3);
model9_Kappa = model9_Kappa(end, 3);
model10_Kappa = model10_Kappa(end, 3);

model1_precision = model1_precision(end, 3);
model2_precision = model2_precision(end, 3);
model3_precision = model3_precision(end, 3);
model4_precision = model4_precision(end, 3);
model5_precision = model5_precision(end, 3);
model6_precision = model6_precision(end, 3);
model7_precision = model7_precision(end, 3);
model8_precision = model8_precision(end, 3);
model9_precision = model9_precision(end, 3);
model10_precision = model10_precision(end, 3);

model1_recall = model1_recall(end, 3);
model2_recall = model2_recall(end, 3);
model3_recall = model3_recall(end, 3);
model4_recall = model4_recall(end, 3);
model5_recall = model5_recall(end, 3);
model6_recall = model6_recall(end, 3);
model7_recall = model7_recall(end, 3);
model8_recall = model8_recall(end, 3);
model9_recall = model9_recall(end, 3);
model10_recall = model10_recall(end, 3);

model1_f1 = model1_f1(end, 3);
model2_f1 = model2_f1(end, 3);
model3_f1 = model3_f1(end, 3);
model4_f1 = model4_f1(end, 3);
model5_f1 = model5_f1(end, 3);
model6_f1 = model6_f1(end, 3);
model7_f1 = model7_f1(end, 3);
model8_f1 = model8_f1(end, 3);
model9_f1 = model9_f1(end, 3);
model10_f1 = model10_f1(end, 3);

model_acc = [model1_acc, model2_acc, model3_acc, model4_acc, model5_acc, ...
             model6_acc, model7_acc, model8_acc, model9_acc, model10_acc];
model_Kappa = [model1_Kappa, model2_Kappa, model3_Kappa, model4_Kappa, model5_Kappa, ...
               model6_Kappa, model7_Kappa, model8_Kappa, model9_Kappa, model10_Kappa];
model_precision = [model1_precision, model2_precision, model3_precision, model4_precision, model5_precision, ...
             model6_precision, model7_precision, model8_precision, model9_precision, model10_precision];
model_recall = [model1_recall, model2_recall, model3_recall, model4_recall, model5_recall, ...
             model6_recall, model7_recall, model8_recall, model9_recall, model10_recall];
model_f1 = [model1_f1, model2_f1, model3_f1, model4_f1, model5_f1, ...
             model6_f1, model7_f1, model8_f1, model9_f1, model10_f1];
         
% Number of intended boxes in the figure
num_boxes = 5;          

% Generating random data
data = cell(1, num_boxes);   
data{1} = model_acc;
data{2} = model_Kappa;
data{3} = model_precision;
data{4} = model_recall;
data{5} = model_f1;

% Using the "figure_boxplot.m" function to plot the boxplot figure using the data, 
label_axes = {'Evaluation Metrics', 'Percentage'}; 
label_boxes = {'GAA', 'Kappa', 'Precision', 'Recall', 'F1 Score'};
figure_boxplot(data, label_axes, label_boxes, '.'); 
grid on

title({'Box Plot for 10-fold Cross-validation'}, 'FontName', 'Times New Roman', 'FontSize', 16, 'FontWeight', 'bold')
set(gca, 'FontName', 'Times New Roman', 'FontSize', 16, 'FontWeight', 'bold');
box on

print('box_cross_validation', '-dpng',  '-r600')











