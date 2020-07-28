#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib as mpl

# Read labels and scores
labels = pd.read_csv('labels.csv', header=None)
labels = np.array(labels).astype('float32')
prediction = pd.read_csv('prediction.csv', header=None)
prediction = np.array(prediction).astype('float32')

# Set photo parameters
sns.set()
mpl.rcParams['font.sans-serif'] = 'Times New Roman'
mpl.rcParams['axes.unicode_minus'] = False

# Draw Confusion Matrix
f, ax = plt.subplots()
C2 = confusion_matrix(labels, prediction, labels=[0, 1, 2, 3])
C2 = np.around(C2/sum(C2), 4)
# print(C2)

sns.heatmap(C2, annot=True, ax=ax, fmt='.4f')
ax.set(xticklabels=['L', 'R', 'B', 'F'], yticklabels=['L', 'R', 'B', 'F'])
ax.set_title('Confusion Matrix of Subject Nine', fontsize=16, fontweight='bold')
ax.set_xlabel('Predicted Class', fontsize=14, fontweight='bold')
ax.set_ylabel('True Class', fontsize=14, fontweight='bold')
plt.savefig('Confusion_Matrix.png', format='png', bbox_inches='tight', transparent=True, dpi=600)
plt.show()
