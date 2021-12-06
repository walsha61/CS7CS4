import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.dummy import DummyClassifier

# Import data file
feature_dataFrame = pd.read_csv("feature_file.csv", header=None)
print(feature_dataFrame.head())
feature_cols = range(0, 18)
X = feature_dataFrame.iloc[:, feature_cols]  # Features
y = feature_dataFrame.iloc[:, 18]

mean_error = []
std_error = []
Neighbours_range = [1, 2, 3, 4, 5, 6, 7, 8]
optimal_num_neighbours = 3 # This value was only chosen after the cross-validation had already been run
# Perform kfold cross-validation for number of neighbours
for Neighbour in Neighbours_range:
    model = KNeighborsClassifier(n_neighbors=Neighbour, weights='uniform')
    kf = KFold(n_splits=5)
    scores = cross_val_score(model, X.values, y, cv=kf, scoring='f1')
    mean_error.append(np.array(scores).mean())
    std_error.append(np.array(scores).std())
    # Now perform some evaluations for our chosen number of neighbours
    # This subsection added after cv was initially run and optimal k identified
    if Neighbour == optimal_num_neighbours:
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2)
        model.fit(Xtrain, ytrain)
        y_pred = model.predict(Xtest)
        cm = metrics.confusion_matrix(ytest, y_pred)
        print("Confusion Matrix for kNN Model:")
        print(cm)
        tn, fp, fn, tp = confusion_matrix(ytest, y_pred).ravel()
        print("TN, FP, FN, TP: ", tn, fp, fn, tp)
        # accuracy: (tp + tn) / (p + n)
        accuracy = accuracy_score(ytest, y_pred)
        print('Accuracy: %f' % accuracy)
        # precision tp / (tp + fp)
        precision = precision_score(ytest, y_pred)
        print('Precision: %f' % precision)
        # recall: tp / (tp + fn)
        recall = recall_score(ytest, y_pred)
        print('Recall: %f' % recall)
        # specificity(fp rate): fp / (tn + fp)
        specificity = (fp / (tn + fp))
        print('Specificity: %f' % specificity)
        # f1: 2 tp / (2 tp + fp + fn)
        f1 = f1_score(ytest, y_pred)
        print('F1 score: %f' % f1)
        fpr_knn, tpr_knn, _ = roc_curve(ytest, model.predict_proba(Xtest)[:, 1])
        print(metrics.classification_report(ytest, y_pred))

# Plot results of cross-validation
plt.rc('font', size=18)
plt.rcParams['figure.constrained_layout.use'] = True
plt.errorbar(Neighbours_range, mean_error, yerr=std_error, linewidth=3)
plt.xlabel('Number of Neighbours')
plt.ylabel('F1Score')
plt.show()

# Train a dummy classifier to act as a baseline comparison:
dummy_classifier = DummyClassifier(strategy="most_frequent")
dummy_classifier.fit(Xtrain, ytrain)
y_pred = dummy_classifier.predict(Xtest)
cm = metrics.confusion_matrix(ytest, y_pred)
print("Confusion Matrix for Dummy model (Most frequent:")
print(cm)
tn, fp, fn, tp = confusion_matrix(ytest, y_pred).ravel()
print("TN, FP, FN, TP: ", tn, fp, fn, tp)
# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(ytest, y_pred)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(ytest, y_pred)
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(ytest, y_pred)
print('Recall: %f' % recall)
# specificity(fp rate): fp / (tn + fp)
specificity = (fp / (tn + fp))
print('Specificity: %f' % specificity)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(ytest, y_pred)
print('F1 score: %f' % f1)

# Plot ROC curve for each model
figure2 = plt.figure()
plt.rc('font', size=18)
plt.rcParams['figure.constrained_layout.use'] = True
plt.plot(fpr_knn, tpr_knn)
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.plot([0, 1], [0, 1], color='green', linestyle='dashdot')
plt.legend(["kNN", "Baseline"])
plt.show()

