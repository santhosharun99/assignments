#SANTHOSH ARUNAGIRI
#201586816

# Importing necessary packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Load dataset
data = "dataset_assignment1.csv"
df = pd.read_csv(data)

# Print data information
print('DATASET INFORMATION')
print('\n')
print(df.info())

np.random.seed(51)
# visualize data
# Separate data by class
class_0 = df[df['class'] == 0]
class_1 = df[df['class'] == 1]

# Create a figure with 9 subplots, one for each feature
fig, axs = plt.subplots(3, 3, figsize=(12, 12))

# Loop over features and plot histograms on each subplot
for i, feature in enumerate(df.columns[:-1]):  # exclude the last column (class)
    row = i // 3
    col = i % 3
    axs[row, col].hist(class_0[feature], alpha=0.5, label='Class 0')
    axs[row, col].hist(class_1[feature], alpha=0.5, label='Class 1')
    axs[row, col].set_title(feature)
    axs[row, col].legend()

plt.tight_layout()
plt.show()

print(df['class'].value_counts())

# Separate data by class
class_0 = df[df['class'] == 0]
class_1 = df[df['class'] == 1]

print('\n')

# Print statistical description of features for each class
for label, data in [('Class 0', class_0), ('Class 1', class_1)]:
    print(f"Statistical description of features for {label}:")
    print(data.describe())
    print()
    print('\n')
    
# Separate the features and target
X = df.drop('class', axis=1)
y = df['class']

# Split data into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Scale the features
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Decision Tree
dt_clf = DecisionTreeClassifier()
dt_param_grid = {'max_depth': [3, 5, 7, 9]}
dt_grid_search = GridSearchCV(dt_clf, dt_param_grid, cv=5)
dt_grid_search.fit(X_train, y_train)
dt_best_clf = dt_grid_search.best_estimator_
dt_pred = dt_best_clf.predict(X_test)

# SVM
svm_clf = SVC()
svm_param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}
svm_grid_search = GridSearchCV(svm_clf, svm_param_grid, cv=5)
svm_grid_search.fit(X_train, y_train)
svm_best_clf = svm_grid_search.best_estimator_
svm_pred = svm_best_clf.predict(X_test)


# Random Forest
rf_clf = RandomForestClassifier()
rf_param_grid = {'n_estimators': [100, 200, 300], 'max_depth': [3, 5, 7, 9]}
rf_grid_search = GridSearchCV(rf_clf, rf_param_grid, cv=5)
rf_grid_search.fit(X_train, y_train)
rf_best_clf = rf_grid_search.best_estimator_
rf_pred = rf_best_clf.predict(X_test)

# Print performance metrics and confusion matrix for Random Forest
print("\nRandom Forest Performance Metrics:\n")
print("Precision: {:.2f}%".format(round(precision_score(y_test, rf_pred, average='weighted')*100, 2)))
print("Recall: {:.2f}%".format(round(recall_score(y_test, rf_pred, average='weighted')*100, 2)))
print("Accuracy: {:.2f}%".format(round(accuracy_score(y_test, rf_pred)*100, 2)))
print("F1 Score: {:.2f}%".format(round(f1_score(y_test, rf_pred, average='weighted')*100, 2)))
print(confusion_matrix(y_test, rf_pred))

# Compute confusion matrix for Random Forest
cm_rf = confusion_matrix(y_test, rf_pred)

# Plot confusion matrix using seaborn
sns.heatmap(cm_rf, annot=True, cmap="Blues", fmt="d")
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# Print performance metrics and confusion matrix for SVM
print("\nSVM Performance Metrics:\n")
print("Precision: {:.2f}%".format(round(precision_score(y_test, svm_pred, average='weighted')*100,2)))
print("Recall: {:.2f}%".format(round(recall_score(y_test, svm_pred, average='weighted')*100,2)))
print("Accuracy: {:.2f}%".format(round(accuracy_score(y_test, svm_pred)*100,2)))
print("F1 Score: {:.2f}%".format(round(f1_score(y_test, svm_pred, average='weighted')*100,2)))
print(confusion_matrix(y_test, svm_pred))

# Compute confusion matrix for SVM
cm_svm = confusion_matrix(y_test, svm_pred)

# Plot confusion matrix using seaborn
sns.heatmap(cm_svm, annot=True, cmap="Blues", fmt="d")
plt.title("Confusion Matrix - SVM")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()


# Print performance metrics and confusion matrix for Decision Tree
print("\nDecision Tree Performance Metrics:\n")
print("Precision: {:.2f}%".format(round(precision_score(y_test, dt_pred, average='weighted')*100,2)))
print("Recall: {:.2f}%".format(round(recall_score(y_test, dt_pred, average='weighted')*100,2)))
print("Accuracy: {:.2f}%".format(round(accuracy_score(y_test, dt_pred)*100,2)))
print("F1 Score: {:.2f}%".format(round(f1_score(y_test, dt_pred, average='weighted')*100,2)))
print(confusion_matrix(y_test, dt_pred))

# Compute confusion matrix for decision tree
cm_dt = confusion_matrix(y_test, dt_pred)

# Plot confusion matrix using seaborn
sns.heatmap(cm_dt, annot=True, cmap="Blues", fmt="d")
plt.title("Confusion Matrix - DECISION TREE")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

#Compare the classifiers
classifiers = ['Random Forest', 'SVM', 'Decision Tree']
accuracies = [accuracy_score(y_test, rf_pred), accuracy_score(y_test, svm_pred), accuracy_score(y_test, dt_pred)]
f1_scores = [f1_score(y_test, rf_pred, average='weighted'), f1_score(y_test, svm_pred, average='weighted'), f1_score(y_test, dt_pred, average='weighted')]
precision_scores = [precision_score(y_test, rf_pred, average='weighted'), precision_score(y_test, svm_pred, average='weighted'), precision_score(y_test, dt_pred, average='weighted')]
recall_scores = [recall_score(y_test, rf_pred, average='weighted'), recall_score(y_test, svm_pred, average='weighted'), recall_score(y_test, dt_pred, average='weighted')]

metrics_df = pd.DataFrame({'Classifiers': classifiers, 'Accuracy': accuracies, 'F1 Score': f1_scores, 'Precision': precision_scores, 'Recall': recall_scores})
print("\nMetrics of all classifiers:\n")
print(metrics_df)

#Find the best classifier based on accuracy
best_acc = max(accuracies)
best_acc_idx = accuracies.index(best_acc)
best_acc_clf = classifiers[best_acc_idx]

print("\nThe best classifier based on accuracy is: {}".format(best_acc_clf))

#Find the best classifier based on F1 score
best_f1 = max(f1_scores)
best_f1_idx = f1_scores.index(best_f1)
best_f1_clf = classifiers[best_f1_idx]

print("\nThe best classifier based on F1 score is: {}".format(best_f1_clf))

#Find the best classifier based on precision score
best_precision = max(precision_scores)
best_precision_idx = precision_scores.index(best_precision)
best_precision_clf = classifiers[best_precision_idx]

print("\nThe best classifier based on precision score is: {}".format(best_precision_clf))

#Find the best classifier based on recall score
best_recall = max(recall_scores)
best_recall_idx = recall_scores.index(best_recall)
best_recall_clf = classifiers[best_recall_idx]

print("\nThe best classifier based on recall score is: {}".format(best_recall_clf))

# plot bar chart for accuracy
plt.figure(figsize=(10, 6))
plt.bar(classifiers, accuracies, color='green')
plt.xlabel("Classifiers")
plt.ylabel("Accuracy")
plt.title("Accuracy of all classifiers")
# add value of each bar on the chart
for i, v in enumerate(accuracies):
    plt.text(i-0.1, v+0.01, str(round(v,2)), color='black', fontweight='bold')
plt.show()


# plot bar chart for F1 score
plt.figure(figsize=(10, 6))
plt.bar(classifiers, f1_scores, color='blue')
plt.xlabel("Classifiers")
plt.ylabel("F1 Score")
plt.title("F1 Score of all classifiers")
# add value of each bar on the chart
for i, v in enumerate(f1_scores):
    plt.text(i-0.1, v+0.01, str(round(v,2)), color='black', fontweight='bold')
plt.show()

# plot bar chart for precision score
plt.figure(figsize=(10, 6))
plt.bar(classifiers, precision_scores, color='red')
plt.xlabel("Classifiers")
plt.ylabel("Precision")
plt.title("Precision of all classifiers")
# add value of each bar on the chart
for i, v in enumerate(precision_scores):
    plt.text(i-0.1, v+0.01, str(round(v,2)), color='black', fontweight='bold')
plt.show()

# plot bar chart for recall score
plt.figure(figsize=(10, 6))
plt.bar(classifiers, recall_scores, color='orange')
plt.xlabel("Classifiers")
plt.ylabel("Recall")
plt.title("Recall of all classifiers")
# add value of each bar on the chart
for i, v in enumerate(recall_scores):
    plt.text(i-0.1, v+0.01, str(round(v,2)), color='black', fontweight='bold')
plt.show()
