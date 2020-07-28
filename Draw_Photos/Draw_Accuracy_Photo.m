clear all
clc

format long

model1     = readmatrix("Model_1/run-.-tag-accuracy.csv");
model2     = readmatrix("Model_2/run-.-tag-accuracy.csv");
model3     = readmatrix("Model_3/run-.-tag-accuracy.csv");
model4     = readmatrix("Model_4/run-.-tag-accuracy.csv");
model5     = readmatrix("Model_5/run-.-tag-accuracy.csv");
model6     = readmatrix("Model_6/run-.-tag-accuracy.csv");
model7     = readmatrix("Model_7/run-.-tag-accuracy.csv");
model8     = readmatrix("Model_8/run-.-tag-accuracy.csv");
model9     = readmatrix("Model_9/run-.-tag-accuracy.csv");
model10    = readmatrix("Model_10/run-.-tag-accuracy.csv");
model11    = readmatrix("Model_11/run-.-tag-accuracy.csv");
model12    = readmatrix("Model_12/run-.-tag-accuracy.csv");

model1_x_axis  = model1(:, 2);
model1_y_axis  = model1(:, 3);
model2_x_axis  = model2(:, 2);
model2_y_axis  = model2(:, 3);
model3_x_axis  = model3(:, 2);
model3_y_axis  = model3(:, 3);
model4_x_axis  = model4(:, 2);
model4_y_axis  = model4(:, 3);
model5_x_axis  = model5(:, 2);
model5_y_axis  = model5(:, 3);
model6_x_axis  = model6(:, 2);
model6_y_axis  = model6(:, 3);
model7_x_axis  = model7(:, 2);
model7_y_axis  = model7(:, 3);
model8_x_axis  = model8(:, 2);
model8_y_axis  = model8(:, 3);
model9_x_axis  = model9(:, 2);
model9_y_axis  = model9(:, 3);
model10_x_axis = model10(:, 2);
model10_y_axis = model10(:, 3);
model11_x_axis = model11(:, 2);
model11_y_axis = model11(:, 3);
model12_x_axis = model12(:, 2);
model12_y_axis = model12(:, 3);

color=[1 0 0; 0 1 0; 0 0 1; 0.5 1 1; 
       1 1 0.5; 1 0.5 1; 0 0 0.5; 0.5 0 0;
       0 0.5 0; 1 0.5 0.5; 0.5 1 0.5; 0.5 0.5 1;
       1 1 0;0 1 1;1 0 1];

% Draw the Images
figure(1)
plot(model1_x_axis,  model1_y_axis,  'linewidth', 1.2, 'color', color(8, :));
hold on
plot(model2_x_axis,  model2_y_axis,  'linewidth', 1.2, 'color', color(2, :));
hold on
plot(model3_x_axis,  model3_y_axis,  'linewidth', 1.2, 'color', color(3, :));
hold on
plot(model4_x_axis,  model4_y_axis,  'linewidth', 1.2, 'color', color(4, :));
hold on 
plot(model5_x_axis,  model5_y_axis,  'linewidth', 1.2, 'color', color(5, :));
hold on
plot(model6_x_axis,  model6_y_axis,  'linewidth', 1.2, 'color', color(6, :));
hold on
plot(model7_x_axis,  model7_y_axis,  'linewidth', 1.2, 'color', color(7, :));
hold on
plot(model8_x_axis,  model8_y_axis,  'linewidth', 1.2, 'color', color(1, :));
hold on
plot(model9_x_axis,  model9_y_axis,  'linewidth', 1.2, 'color', color(9, :));
hold on
plot(model10_x_axis, model10_y_axis, 'linewidth', 1.2, 'color', color(10, :));
hold on
plot(model11_x_axis, model11_y_axis, 'linewidth', 1.2, 'color', 'k');
hold on
plot(model12_x_axis, model12_y_axis, 'linewidth', 1.2, 'color', color(12, :));
hold on

grid on

xlim([0, 310])

title({'GAA w.r.t. RNN-based Models'}, 'FontName', 'Times New Roman', 'FontSize', 16, 'FontWeight', 'bold')
xlabel('Iterations')
ylabel('Global Average Accuracy')
set(gca, 'FontName', 'Times New Roman', 'FontSize', 16, 'FontWeight', 'bold');

legend('RNN', 'BiRNN', 'RNN with Attention', 'BiRNN with Attention', ...
       'LSTM', 'BiLSTM', 'LSTM with Attention', 'BiLSTM with Attention', ...
       'GRU', 'BiGRU', 'GRU with Attention', 'BiGRU with Attention', ...
       'location', 'EastOutside', 'FontName', 'Times New Roman', 'FontSize', 16)
legend('boxoff')

print('GAA_RNN_basedModels', '-dpng',  '-r600')

