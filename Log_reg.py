# -- IMPORT LIBRARIES --
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, \
    f1_score, roc_curve
from sklearn.model_selection import cross_val_score, train_test_split

# -- READ IN TRAINING DATA --
df = pd.read_csv("feature_file.csv")
X1 = df.iloc[:, 0] # quad 0 fire size x wind vector
X2 = df.iloc[:, 1] # quad 1 fire size x wind vector
X3 = df.iloc[:, 2] # quad 2 fire size x wind vector
X4 = df.iloc[:, 3] # quad 3 fire size x wind vector
X5 = df.iloc[:, 4] # quad 4 fire size x wind vector
X6 = df.iloc[:, 5] # quad 5 fire size x wind vector
X7 = df.iloc[:, 6] # quad 6 fire size x wind vector
X8 = df.iloc[:, 7] # quad 7 fire size x wind vector
X9 = df.iloc[:, 8] # quad 8 fire size x wind vector
X10 = df.iloc[:, 9] # quad 9 fire size x wind vector
X11 = df.iloc[:, 10] # quad 10 fire size x wind vector
X12 = df.iloc[:, 11] # quad 11 fire size x wind vector
X13 = df.iloc[:, 12] # quad 12 fire size x wind vector
X14 = df.iloc[:, 13] # quad 13 fire size x wind vector
X15 = df.iloc[:, 14] # quad 14 fire size x wind vector
X16 = df.iloc[:, 15] # quad 15 fire size x wind vector
X17 = df.iloc[:, 16] # percipitation
X18 = df.iloc[:, 17] # temperature
X = np.column_stack((X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11, X12, X13, X14, X15, X16, X17))
y = df.iloc[:, 18] # air quality index


# -- CROSS VALIDATION FOR C --
mean_error=[]
std_error=[]
scores=[]
C_range = np.arange(0.01, 50, 5)

for C in C_range:
    # -- TRAIN THE MODEL --
    model = LogisticRegression(penalty='l2', C=C, max_iter=1000).fit(X, y)
    # -- CROSS VALIDATION --
    scores = cross_val_score(model, X, y, cv=5, scoring='f1')  # F1 score, 5-fold cross validation
    # -- GET THE MEAN F1 SCORE AND STANDARD ERROR --
    mean_error.append(np.array(scores).mean())
    std_error.append(np.array(scores).std())

# -- ERROR BAR PLOT --
plt.errorbar(C_range, mean_error, yerr=std_error)
plt.xlabel('C')
plt.ylabel('F1 Score')
plt.title("Cross Validation for C")
plt.xlim((0, 50))
plt.show()

# -- SPLIT DATA INTO TRAINING AND TEST DATA --
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2)

# -- MODEL EVALUATION --
C = 5 # Select optimum C from cross validation and evaluate the model
model = LogisticRegression(penalty='l2', C=C, max_iter=1000).fit(Xtrain, ytrain)
ypred = model.predict(Xtest)
# -- CONFUSION MATRIX --
cm = metrics.confusion_matrix(ytest, ypred)
tn, fp, fn, tp = cm.ravel()
print("TN, FP, FN, TP: ", tn, fp, fn, tp)
print("Confusion Matrix for Logistic Regression Model:")
print(cm)
# -- CLASSIFICATION REPORT --
print("Classification report for C = ", C)
print(classification_report(ytest, ypred))

# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(ytest, ypred)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(ytest, ypred)
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(ytest, ypred)
print('Recall: %f' % recall)
# specificity(fp rate): fp / (tn + fp)
specificity = (fp / (tn + fp))
print('Specificity: %f' % specificity)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(ytest, ypred)
print('F1 score: %f' % f1)

# -- ROC CURVE --
fpr_lr, tpr_lr, _ = roc_curve(ytest, model.predict_proba(Xtest)[:, 1])
# Plot ROC curve for each model
figure2 = plt.figure()
plt.rc('font', size=18)
plt.rcParams['figure.constrained_layout.use'] = True
plt.plot(fpr_lr, tpr_lr)
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.plot([0, 1], [0, 1], color='green', linestyle='dashdot')
plt.legend(["Logistic Regression", "Baseline"])
plt.show()

# # -- PLOT PREDICTIONS  --
# plt.rc('font', size=20)
# pos_ypred = plt.scatter(X1[ypred > 0], X2[ypred > 0],
#                         color='green', marker=".", linewidths=4, label='Healthy AQI Prediction')
#
# pos_ypred = plt.scatter(X1[ypred < 0], X2[ypred < 0],
#                         color='red', marker=".", linewidths=4, label='Unealthy AQI Prediction')
# #
# pos_ypred = plt.scatter(X1[ypred > 0], X2[ypred > 0], X3[ypred > 0], X4[ypred > 0], X5[ypred > 0], X6[ypred > 0],
#                         X7[ypred > 0], X8[ypred > 0], X9[ypred > 0], X10[ypred > 0], X11[ypred > 0], X12[ypred > 0],
#                         X13[ypred > 0], X14[ypred > 0], X15[ypred > 0], X16[ypred > 0], X17[ypred > 0],
#                         color='green', marker=".", linewidths=4, label='Healthy AQI Prediction')
#
# pos_ypred = plt.scatter(X1[ypred < 0], X2[ypred < 0], X3[ypred < 0], X4[ypred < 0], X5[ypred < 0], X6[ypred < 0],
#                         X7[ypred < 0], X8[ypred < 0], X9[ypred < 0], X10[ypred < 0], X11[ypred < 0], X12[ypred < 0],
#                         X13[ypred < 0], X14[ypred < 0], X15[ypred < 0], X16[ypred < 0], X17[ypred < 0],
#                         color='red', marker=".", linewidths=4, label='Unealthy AQI Prediction')

# # !! Plotting needs to be fixed !!
# plt.rcParams['figure.constrained_layout.use'] = True
# plt.title("Plot of Predictions for Salt Lake City's Air Quality")
# plt.xlabel("X1")
# plt.ylabel("X2")
# plt.legend(loc='upper right')
# plt.show()


# -- TRAIN BASELINE PREDICTOR --
dummy = DummyClassifier(strategy="most_frequent").fit(Xtest, ytest) # Always classifies with most common label
ydummy = dummy.predict(Xtest)
print(confusion_matrix(ytest, ydummy))
tn, fp, fn, tp = confusion_matrix(ytest, ydummy).ravel()
print("TN, FP, FN, TP: ", tn, fp, fn, tp)
print("Classification report for dummy classifier")
print(classification_report(ytest, ydummy))

# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(ytest, ydummy)
# precision tp / (tp + fp)
precision = precision_score(ytest, ydummy)
# recall: tp / (tp + fn)
recall = recall_score(ytest, ydummy)
# specificity(fp rate): fp / (tn + fp)
specificity = (fp / (tn + fp))
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(ytest, ydummy)

print('Accuracy: %f' % accuracy)
print('Precision: %f' % precision)
print('Recall: %f' % recall)
print('Specificity: %f' % specificity)
print('F1 score: %f' % f1)
