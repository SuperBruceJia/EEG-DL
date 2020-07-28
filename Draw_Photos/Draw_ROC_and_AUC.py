#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import metrics

# Read labels and scores
labels_Subject0 = pd.read_csv('20-person/summaries/Model-20-1/labels.csv')
labels_Subject0 = np.array(labels_Subject0).astype('float32')
new_labels_Subject0 = []
for i in range(np.shape(labels_Subject0)[0]):
    if labels_Subject0[i] == 0.:
        new_labels_Subject0.append([1, 0, 0, 0])
    if labels_Subject0[i] == 1.:
        new_labels_Subject0.append([0, 1, 0, 0])
    if labels_Subject0[i] == 2.:
        new_labels_Subject0.append([0, 0, 1, 0])
    if labels_Subject0[i] == 3.:
        new_labels_Subject0.append([0, 0, 0, 1])
labels_Subject0 = np.array(new_labels_Subject0).astype('float32')

prediction_Subject0 = pd.read_csv('20-person/summaries/Model-20-1/prediction.csv')
prediction_Subject0 = np.array(prediction_Subject0).astype('float32')
new_prediction_Subject0 = []
for i in range(np.shape(labels_Subject0)[0]):
    if prediction_Subject0[i] == 0.:
        new_prediction_Subject0.append([1, 0, 0, 0])
    if prediction_Subject0[i] == 1.:
        new_prediction_Subject0.append([0, 1, 0, 0])
    if prediction_Subject0[i] == 2.:
        new_prediction_Subject0.append([0, 0, 1, 0])
    if prediction_Subject0[i] == 3.:
        new_prediction_Subject0.append([0, 0, 0, 1])
prediction_Subject0 = np.array(new_prediction_Subject0).astype('float32')

# Read labels and scores
labels_Subject1 = pd.read_csv('50-person/Model-50/labels.csv')
labels_Subject1 = np.array(labels_Subject1).astype('float32')
new_labels_Subject1 = []
for i in range(np.shape(labels_Subject1)[0]):
    if labels_Subject1[i] == 0.:
        new_labels_Subject1.append([1, 0, 0, 0])
    if labels_Subject1[i] == 1.:
        new_labels_Subject1.append([0, 1, 0, 0])
    if labels_Subject1[i] == 2.:
        new_labels_Subject1.append([0, 0, 1, 0])
    if labels_Subject1[i] == 3.:
        new_labels_Subject1.append([0, 0, 0, 1])
labels_Subject1 = np.array(new_labels_Subject1).astype('float32')

prediction_Subject1 = pd.read_csv('50-person/Model-50/prediction.csv')
prediction_Subject1 = np.array(prediction_Subject1).astype('float32')
new_prediction_Subject1 = []
for i in range(np.shape(labels_Subject1)[0]):
    if prediction_Subject1[i] == 0.:
        new_prediction_Subject1.append([1, 0, 0, 0])
    if prediction_Subject1[i] == 1.:
        new_prediction_Subject1.append([0, 1, 0, 0])
    if prediction_Subject1[i] == 2.:
        new_prediction_Subject1.append([0, 0, 1, 0])
    if prediction_Subject1[i] == 3.:
        new_prediction_Subject1.append([0, 0, 0, 1])
prediction_Subject1 = np.array(new_prediction_Subject1).astype('float32')

fpr_Subject0, tpr_Subject0, thresholds_Subject0 = metrics.roc_curve(labels_Subject0.ravel(), prediction_Subject0.ravel())
auc_Subject0 = metrics.auc(fpr_Subject0, tpr_Subject0)

fpr_Subject1, tpr_Subject1, thresholds_Subject1 = metrics.roc_curve(labels_Subject1.ravel(), prediction_Subject1.ravel())
auc_Subject1 = metrics.auc(fpr_Subject1, tpr_Subject1)

mpl.rcParams['font.sans-serif'] = 'Times New Roman'
mpl.rcParams['axes.unicode_minus'] = False

plt.plot(fpr_Subject0, tpr_Subject0, lw = 1, alpha = 1.0, label = u'AUC of 20 subjects = %.3f' % auc_Subject0)
plt.plot(fpr_Subject1, tpr_Subject1, lw = 1, alpha = 1.0, label = u'AUC of 50 subjects = %.3f' % auc_Subject1)

plt.plot((0, 1), (0, 1), c = 'b', lw = 1, ls = '--', alpha = 1.0)
plt.xlim((-0.01, 1.02))
plt.ylim((-0.01, 1.02))
plt.xticks(np.arange(0, 1.1, 0.1))
plt.yticks(np.arange(0, 1.1, 0.1))

plt.title('ROC Curve for Group-wise Prediction', fontsize=12, fontweight='bold')
plt.xlabel('False Positive Rate',  fontsize=12, fontweight='bold')
plt.ylabel('True Positive Rate',   fontsize=12, fontweight='bold')

plt.grid(b=True, ls=':')
legend = plt.legend(loc='lower right', fancybox=True, framealpha=1.0, frameon=False, markerscale=0.1, prop={'weight':'bold'}, fontsize=13)
frame = legend.get_frame()
frame.set_alpha(1)
frame.set_facecolor('none')

plt.savefig('ROC_Group_Wise.png', format='png', bbox_inches='tight', transparent=True, dpi=600)
plt.show()
