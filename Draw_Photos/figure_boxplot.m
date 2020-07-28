function bh = figure_boxplot(data_input,...
    text_label_in,label_xaxis_data_in,text_title_in,label_orientation_choice_in,...
    box_color_in,box_lineWidth_in,box_widths_value_in,box_color_transparency_in,...
    median_lineWidth_in,median_color_in,...
    whisker_value_in,...
    outlier_marker_in,outlier_markerSize_in,outlier_marker_edgeWidth_in,outlier_marker_edgeColor_in,outlier_jitter_value_in,...
    savefig_in,savefig_name_in,fig_width_cm_in,fig_height_cm_in,...
    ylim_min_in,ylim_max_in)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Adaption of MATLAB boxplot function to plot beautiful publication-quality boxplots easily
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Written by: Ahmed Abdul Quadeer
% E-mail: ahmedaq@gmail.com
% Last change: May 29, 2018
% Copyright (c) Ahmed Abdul Quadeer, 2019
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Inputs:   
%
% - data_input: Input data       
%   * N-length cell including data, with each cell having data of one boxplot 
%     to be compared. G matrix to be used in the default MATLAB "boxplot" 
%     function is generated accordingly.
%   * a matrix of size MxN, where M = no. of observation and N = no. of
%     variables. Boxplot is made for each variable (column) of input matrix
%
% - text_label_in: Labels for x-axis and y-axis
%   Cell of length 2, with first cell having text of x-axis label and 
%   second cell having text of y-axis label
%
% - label_xaxis_data_in: Label for each box plot on the x-axis
%   N-length cell (containing text) or N-length vector numerical data
%
% - text_title_in: Title of the box plot
%   N-length cell (containing text) or N-length vector numerical data
%
% - label_orientation_choice_in: Choice for orientation of boxplot labels
%   * 'horizontal' (default) or
%   * 'inline' (vertical) suitable for long labels
%
% - box_color_in: Color of boxplots
%   Nx3 sized matrix with first row specifying RGB color for the first box
%   plot and so on.
%
% - box_lineWidth_in: Width of boxplots outline
%   Scalar value
%
% - box_widths_value_in: Width of boxplots
%   Scalar value
%
% - box_color_transparency_in: Transparency of box plots
%   Scalar value between 0 (transparent) and 1 (opaque)
%
% - median_lineWidth_in: Width of the line representing the median
%   Scalar value
%
% - median_color_in: Color of the line representing the median
%   3-length row vector specifying the RGB color
%
% - whisker_value_in: Limit of whiskers 
%   Scalar value "a" in the definition of whiskers:a * inter-quartile-range
%
% - outlier_marker_in: Marker to represent outliers
%   e.g. 'o', 'x', '.', or '+'
%
% - outlier_markerSize_in: Size of the marker representing outliers
%   Scalar value
%
% - outlier_marker_edgeWidth_in: Width of the edge (outline) of the marker
%   represeting outliers
%   Scalar value
%
% - outlier_marker_edgeColor_in: Color of the edge (outline) of the marker
%   represeting outliers
%   3-length row vector specifying the RGB color. Black ('k') and white
%   ('w') (default) generally are good options.
%
% - outlier_jitter_value_in: Variance of the random noise added to outliers
%   for visualization purpose
%   Scalar value
%
% - savefig_in: Option to save figure automatically (in png format)
%   * 1 - Yes, save the figure
%   * 0 - No
%
% - savefig_name_in: Name for the figure file
%   Text data
%
% - fig_width_cm_in: Width of the output figure (in cm)
%   Scalar value
%
% - fig_height_cm_in: Height of the output figure (in cm)
%   Scalar value
%
% - ylim_min_in: Lower limit of y-axis 
%   Scalar value
%
% - ylim_max_in: Upper limit of y-axis 
%   Scalar value
%
%
%           
% Output: Box plot figure
% 
%
%% Basic settings 

%%% Font type and size setting %%%

% Using Arial as default because all journals normally require the font to
% be either Arial or Helvetica
set(0,'DefaultAxesFontName','Arial')
set(0,'DefaultTextFontName','Arial')
set(0,'DefaultAxesFontSize',10)
set(0,'DefaultTextFontSize',10)

%%% Color definition %%%

% Using Nature color scheme
% Reason: Because it is really cool!
% Drawback: Only 9 colors
% You can use any other scheme as well, for example check:
% https://ggsci.net/index.html for other journal templates

% color_scheme_npg_hex = ['#E64B35';'#4DBBD5';'#00A087';'#3C5488';....
%     '#F39B7F';'#8491B4';'#91D1C2';'#DC0000';'#7E6148';'#B09C85'];
% color_scheme_npg = hex2rgb(color_scheme_npg_hex);
color_scheme_npg = [...
    0.9020    0.2941    0.2078; ...
    0.3020    0.7333    0.8353; ...
         0    0.6275    0.5294; ...
    0.2353    0.3294    0.5333; ...
    0.9529    0.6078    0.4980; ...
    0.5176    0.5686    0.7059; ...
    0.5686    0.8196    0.7608; ...
    0.8627         0         0; ...
    0.4941    0.3804    0.2824; ...
    0.6902    0.6118    0.5216 ];

%% Processing with respect to number of input arguments

%Number of boxplots
if iscell(data_input) %if input data is in cell form
    no_boxplots = length(data_input);
else                  %if input data is in matrix form
    no_boxplots = size(data_input,2);
end

if no_boxplots>9 || isempty(data_input)
    fprintf('\nWarning: The number of box plots to be plotted is greater than colors in the selected scheme. Using one color for all boxes.')
end

%Defaults
box_lineWidth = 0.5;
box_widths_value = 0.3;

if no_boxplots>9
   box_color = [0.9020    0.2941    0.2078]; 
else
    box_color = color_scheme_npg(1:no_boxplots,:);
end

%Defaults
box_lineWidth = 0.5;
box_widths_value = 0.3;
box_color = color_scheme_npg(1:no_boxplots,:);
box_color_transparency = 0.7; %faceAlpha
median_lineWidth = 2;
median_color = 'k';
whisker_value = 1.5;
outlier_marker = 'o';
outlier_markerSize = 4;
outlier_marker_edgeWidth = 0.1;
outlier_marker_edgeColor = 'w';
outlier_jitter_value = 0.3;
label_xaxis_data = 1:no_boxplots;
text_label{1} = '';
text_label{2} = '';
text_title = '';
label_orientation_choice = 'horizontal'; %'inline'
savefig = 0;
savefig_name = 'fig_boxplot_noname';
fig_width_cm = 10;
fig_height_cm = 5;


if nargin == 2
    text_label = text_label_in;
elseif nargin == 3
    text_label = text_label_in;
    label_xaxis_data = label_xaxis_data_in;
elseif nargin == 4
    text_label = text_label_in;
    label_xaxis_data = label_xaxis_data_in;
    text_title = text_title_in;
elseif nargin == 5
    text_label = text_label_in;
    label_xaxis_data = label_xaxis_data_in;
    text_title = text_title_in;
    label_orientation_choice = label_orientation_choice_in;
elseif nargin == 6
    text_label = text_label_in;
    label_xaxis_data = label_xaxis_data_in;
    text_title = text_title_in;
    label_orientation_choice = label_orientation_choice_in;
    box_color = box_color_in;
elseif nargin == 7
    text_label = text_label_in;
    label_xaxis_data = label_xaxis_data_in;
    text_title = text_title_in;
    label_orientation_choice = label_orientation_choice_in;
    box_color = box_color_in;
    box_lineWidth = box_lineWidth_in;
elseif nargin == 8
    text_label = text_label_in;
    label_xaxis_data = label_xaxis_data_in;
    text_title = text_title_in;
    label_orientation_choice = label_orientation_choice_in;
    box_color = box_color_in;
    box_lineWidth = box_lineWidth_in;
    box_widths_value = box_widths_value_in;
elseif nargin == 9
    text_label = text_label_in;
    label_xaxis_data = label_xaxis_data_in;
    text_title = text_title_in;
    label_orientation_choice = label_orientation_choice_in;
    box_color = box_color_in;
    box_lineWidth = box_lineWidth_in;
    box_widths_value = box_widths_value_in;
    box_color_transparency = box_color_transparency_in;
elseif nargin == 10
    text_label = text_label_in;
    label_xaxis_data = label_xaxis_data_in;
    text_title = text_title_in;
    label_orientation_choice = label_orientation_choice_in;
    box_color = box_color_in;
    box_lineWidth = box_lineWidth_in;
    box_widths_value = box_widths_value_in;
    box_color_transparency = box_color_transparency_in;
    median_lineWidth = median_lineWidth_in;
elseif nargin == 11
    text_label = text_label_in;
    label_xaxis_data = label_xaxis_data_in;
    text_title = text_title_in;
    label_orientation_choice = label_orientation_choice_in;
    box_color = box_color_in;
    box_lineWidth = box_lineWidth_in;
    box_widths_value = box_widths_value_in;
    box_color_transparency = box_color_transparency_in;
    median_lineWidth = median_lineWidth_in;
    median_color = median_color_in;
elseif nargin == 12
    text_label = text_label_in;
    label_xaxis_data = label_xaxis_data_in;
    text_title = text_title_in;
    label_orientation_choice = label_orientation_choice_in;
    box_color = box_color_in;
    box_lineWidth = box_lineWidth_in;
    box_widths_value = box_widths_value_in;
    box_color_transparency = box_color_transparency_in;
    median_lineWidth = median_lineWidth_in;
    median_color = median_color_in;
    whisker_value = whisker_value_in;
elseif nargin == 13
    text_label = text_label_in;
    label_xaxis_data = label_xaxis_data_in;
    text_title = text_title_in;
    label_orientation_choice = label_orientation_choice_in;
    box_color = box_color_in;
    box_lineWidth = box_lineWidth_in;
    box_widths_value = box_widths_value_in;
    box_color_transparency = box_color_transparency_in;
    median_lineWidth = median_lineWidth_in;
    median_color = median_color_in;
    whisker_value = whisker_value_in;
    outlier_marker = outlier_marker_in;
elseif nargin == 14
    text_label = text_label_in;
    label_xaxis_data = label_xaxis_data_in;
    text_title = text_title_in;
    label_orientation_choice = label_orientation_choice_in;
    box_color = box_color_in;
    box_lineWidth = box_lineWidth_in;
    box_widths_value = box_widths_value_in;
    box_color_transparency = box_color_transparency_in;
    median_lineWidth = median_lineWidth_in;
    median_color = median_color_in;
    whisker_value = whisker_value_in;
    outlier_marker = outlier_marker_in;
    outlier_markerSize = outlier_markerSize_in;
elseif nargin == 15
    text_label = text_label_in;
    label_xaxis_data = label_xaxis_data_in;
    text_title = text_title_in;
    label_orientation_choice = label_orientation_choice_in;
    box_color = box_color_in;
    box_lineWidth = box_lineWidth_in;
    box_widths_value = box_widths_value_in;
    box_color_transparency = box_color_transparency_in;
    median_lineWidth = median_lineWidth_in;
    median_color = median_color_in;
    whisker_value = whisker_value_in;
    outlier_marker = outlier_marker_in;
    outlier_markerSize = outlier_markerSize_in;
    outlier_marker_edgeWidth = outlier_marker_edgeWidth_in;
elseif nargin == 16
    text_label = text_label_in;
    label_xaxis_data = label_xaxis_data_in;
    text_title = text_title_in;
    label_orientation_choice = label_orientation_choice_in;
    box_color = box_color_in;
    box_lineWidth = box_lineWidth_in;
    box_widths_value = box_widths_value_in;
    box_color_transparency = box_color_transparency_in;
    median_lineWidth = median_lineWidth_in;
    median_color = median_color_in;
    whisker_value = whisker_value_in;
    outlier_marker = outlier_marker_in;
    outlier_markerSize = outlier_markerSize_in;
    outlier_marker_edgeWidth = outlier_marker_edgeWidth_in;
    outlier_marker_edgeColor = outlier_marker_edgeColor_in;
elseif nargin == 17 
    text_label = text_label_in;
    label_xaxis_data = label_xaxis_data_in;
    text_title = text_title_in;
    label_orientation_choice = label_orientation_choice_in;
    box_color = box_color_in;
    box_lineWidth = box_lineWidth_in;
    box_widths_value = box_widths_value_in;
    box_color_transparency = box_color_transparency_in;
    median_lineWidth = median_lineWidth_in;
    median_color = median_color_in;
    whisker_value = whisker_value_in;
    outlier_marker = outlier_marker_in;
    outlier_markerSize = outlier_markerSize_in;
    outlier_marker_edgeWidth = outlier_marker_edgeWidth_in;
    outlier_marker_edgeColor = outlier_marker_edgeColor_in;
    outlier_jitter_value = outlier_jitter_value_in;
elseif nargin == 18
    text_label = text_label_in;
    label_xaxis_data = label_xaxis_data_in;
    text_title = text_title_in;
    label_orientation_choice = label_orientation_choice_in;
    box_color = box_color_in;
    box_lineWidth = box_lineWidth_in;
    box_widths_value = box_widths_value_in;
    box_color_transparency = box_color_transparency_in;
    median_lineWidth = median_lineWidth_in;
    median_color = median_color_in;
    whisker_value = whisker_value_in;
    outlier_marker = outlier_marker_in;
    outlier_markerSize = outlier_markerSize_in;
    outlier_marker_edgeWidth = outlier_marker_edgeWidth_in;
    outlier_marker_edgeColor = outlier_marker_edgeColor_in;
    outlier_jitter_value = outlier_jitter_value_in;
    savefig = savefig_in;
elseif nargin == 19 
    text_label = text_label_in;
    label_xaxis_data = label_xaxis_data_in;
    text_title = text_title_in;
    label_orientation_choice = label_orientation_choice_in;
    box_color = box_color_in;
    box_lineWidth = box_lineWidth_in;
    box_widths_value = box_widths_value_in;
    box_color_transparency = box_color_transparency_in;
    median_lineWidth = median_lineWidth_in;
    median_color = median_color_in;
    whisker_value = whisker_value_in;
    outlier_marker = outlier_marker_in;
    outlier_markerSize = outlier_markerSize_in;
    outlier_marker_edgeWidth = outlier_marker_edgeWidth_in;
    outlier_marker_edgeColor = outlier_marker_edgeColor_in;
    outlier_jitter_value = outlier_jitter_value_in;
    savefig = savefig_in;
    savefig_name = savefig_name_in;    
elseif nargin == 20 
    text_label = text_label_in;
    label_xaxis_data = label_xaxis_data_in;
    text_title = text_title_in;
    label_orientation_choice = label_orientation_choice_in;
    box_color = box_color_in;
    box_lineWidth = box_lineWidth_in;
    box_widths_value = box_widths_value_in;
    box_color_transparency = box_color_transparency_in;
    median_lineWidth = median_lineWidth_in;
    median_color = median_color_in;
    whisker_value = whisker_value_in;
    outlier_marker = outlier_marker_in;
    outlier_markerSize = outlier_markerSize_in;
    outlier_marker_edgeWidth = outlier_marker_edgeWidth_in;
    outlier_marker_edgeColor = outlier_marker_edgeColor_in;
    outlier_jitter_value = outlier_jitter_value_in;
    savefig = savefig_in;
    savefig_name = savefig_name_in;    
    fig_width_cm = fig_width_cm_in;
elseif nargin == 21    
    text_label = text_label_in;
    label_xaxis_data = label_xaxis_data_in;
    text_title = text_title_in;
    label_orientation_choice = label_orientation_choice_in;
    box_color = box_color_in;
    box_lineWidth = box_lineWidth_in;
    box_widths_value = box_widths_value_in;
    box_color_transparency = box_color_transparency_in;
    median_lineWidth = median_lineWidth_in;
    median_color = median_color_in;
    whisker_value = whisker_value_in;
    outlier_marker = outlier_marker_in;
    outlier_markerSize = outlier_markerSize_in;
    outlier_marker_edgeWidth = outlier_marker_edgeWidth_in;
    outlier_marker_edgeColor = outlier_marker_edgeColor_in;
    outlier_jitter_value = outlier_jitter_value_in;
    savefig = savefig_in;
    savefig_name = savefig_name_in;    
    fig_width_cm = fig_width_cm_in;
    fig_height_cm = fig_height_cm_in;
elseif nargin == 22        
    text_label = text_label_in;
    label_xaxis_data = label_xaxis_data_in;
    text_title = text_title_in;
    label_orientation_choice = label_orientation_choice_in;
    box_color = box_color_in;
    box_lineWidth = box_lineWidth_in;
    box_widths_value = box_widths_value_in;
    box_color_transparency = box_color_transparency_in;
    median_lineWidth = median_lineWidth_in;
    median_color = median_color_in;
    whisker_value = whisker_value_in;
    outlier_marker = outlier_marker_in;
    outlier_markerSize = outlier_markerSize_in;
    outlier_marker_edgeWidth = outlier_marker_edgeWidth_in;
    outlier_marker_edgeColor = outlier_marker_edgeColor_in;
    outlier_jitter_value = outlier_jitter_value_in;
    savefig = savefig_in;
    savefig_name = savefig_name_in;    
    fig_width_cm = fig_width_cm_in;
    fig_height_cm = fig_height_cm_in;
    ylim_min = ylim_min_in;
elseif nargin == 23        
    text_label = text_label_in;
    label_xaxis_data = label_xaxis_data_in;
    text_title = text_title_in;
    label_orientation_choice = label_orientation_choice_in;
    box_color = box_color_in;
    box_lineWidth = box_lineWidth_in;
    box_widths_value = box_widths_value_in;
    box_color_transparency = box_color_transparency_in;
    median_lineWidth = median_lineWidth_in;
    median_color = median_color_in;
    whisker_value = whisker_value_in;
    outlier_marker = outlier_marker_in;
    outlier_markerSize = outlier_markerSize_in;
    outlier_marker_edgeWidth = outlier_marker_edgeWidth_in;
    outlier_marker_edgeColor = outlier_marker_edgeColor_in;
    outlier_jitter_value = outlier_jitter_value_in;
    savefig = savefig_in;
    savefig_name = savefig_name_in;    
    fig_width_cm = fig_width_cm_in;
    fig_height_cm = fig_height_cm_in;
    ylim_min = ylim_min_in;
    ylim_max = ylim_max_in;
end

%% Preprocessing data for plotting

data = [];
G = [];
if ~iscell(data_input)   %if input data is in matrix form
    for kk = 1:size(data_input,2)
        data_temp = data_input(:,kk);
        data = [data data_temp.'];
        G = [G kk*ones(1,size(data_input,1))];
    end
else                      %if input data is in cell form
    for kk = 1:no_boxplots
        data_temp = data_input{kk};
        data = [data data_temp(:).'];
        G = [G kk*ones(1,length(data_input{kk}))];
    end
end

%% Main box plot

% figure;
bh = boxplot(data,G,...
    'whisker',whisker_value,'symbol',outlier_marker,...
    'color','k','jitter',outlier_jitter_value,...
    'labels',label_xaxis_data,...
    'widths',box_widths_value,'LabelOrientation',label_orientation_choice);

set(bh,'linewidth',box_lineWidth);
xlabel(text_label{1});
ylabel(text_label{2})
title(text_title)

%% Rotate xaxis labels by 45 degrees and make their font smaller only if
%% labels>10

if length(label_xaxis_data)>10
    set(gca,'XTickLabelRotation',45)
    xL = xlabel(text_label{1});
    ax = ancestor(gca, 'axes');
    xrule = ax.XAxis;
    xrule.FontSize = 8;
    xL.FontSize = 10;
end

%% Coloring each box

h = findobj(gca,'Tag','Box');
if size(box_color,1) ~= 1   %if colors provided for each box
    for kk = 1:length(h)
        patch(get(h(kk),'XData'),get(h(kk),'YData'),box_color(length(h)-kk+1,:),'FaceAlpha',box_color_transparency);
    end
else
    for kk = 1:length(h)
        patch(get(h(kk),'XData'),get(h(kk),'YData'),box_color,'FaceAlpha',box_color_transparency);
    end
end

% Sending patch to back of the figure so that median can be drawn on top of it
set(gca,'children',flipud(get(gca,'children'))) 

%% Adjusting median

h=findobj(gca,'tag','Median');
for kk = 1:length(h)
    h(kk).LineWidth = median_lineWidth;
    h(kk).Color = median_color;
end

%% Adjusting outliers

h=findobj(gca,'tag','Outliers');
for kk = 1:length(h)
    if size(box_color,1) ~= 1   %if colors provided for each box
        h(kk).MarkerFaceColor = box_color(length(h)-kk+1,:); alpha(box_color_transparency)
    else
        h(kk).MarkerFaceColor = box_color; alpha(box_color_transparency)
    end
    h(kk).MarkerEdgeColor = outlier_marker_edgeColor;
    h(kk).MarkerSize = outlier_markerSize;
    h(kk).LineWidth = outlier_marker_edgeWidth;
end

% ylim([ylim_min ylim_max])

%% Further post-processing the figure

set(gca, ...
    'Box'         , 'off'     , ...
    'TickDir'     , 'out'     , ...
    'TickLength'  , [.01 .01] , ...
    'XColor'      , [.1 .1 .1], ...
    'YColor'      , [.1 .1 .1], ...
    'XTick'       , 1:1:100,... 
    'LineWidth'   , .5        );

%% Saving figure

if savefig == 1
    set(gcf,'PaperUnits','centimeters','PaperPosition',[0 0 fig_width_cm fig_height_cm])
    print(savefig_name,'-dpng','-r300')
end
