from math import atan, pi, cos
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.dummy import DummyClassifier

# Run each module in turn:
# Note: quadrant edge size is set in feature_builder.py
# Will need to edit this in order to build features based on different quadrant dimensions.
import feature_builder
import KNN
import Log_reg

# Build a combined ROC curve showing dummy, KNN and Log Reg:
# Plot ROC curve for each model
figure3 = plt.figure()
plt.rc('font', size=18)
plt.rcParams['figure.constrained_layout.use'] = True
plt.plot(KNN.fpr_knn, KNN.tpr_knn)
plt.plot(Log_reg.fpr_lr, Log_reg.tpr_lr)
plt.plot([0, 1], [0, 1], color='green', linestyle='dashdot')
plt.legend(["kNN", "Logistic Regression", "Baseline"])
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.show()




