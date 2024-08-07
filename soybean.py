import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import warnings
from sklearn.metrics import make_scorer, f1_score

warnings.filterwarnings("ignore", category=FutureWarning)


df = pd.read_csv('C:/Users/genis/Desktop/çalışmalar/2024 bahar/neuro/2.donemproje/dataset/data.csv')
df = df.drop(columns=["Cultivar"])


y = df["Season"]
X = df.drop("Season", axis=1)


scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)


k = 10
random_seed = 42
f1_scorer = make_scorer(f1_score, average='macro')



print("Logistic Regression")
model = LogisticRegression(C=8, solver='liblinear', random_state=random_seed)
scores = cross_val_score(model, X_normalized, y, cv=k)
mean_accuracy = scores.mean()
print(f"Mean Accuracy (k={k}): {mean_accuracy:.4f}")

precisions = cross_val_score(model, X_normalized, y, cv=k, scoring='precision')
recalls = cross_val_score(model, X_normalized, y, cv=k, scoring='recall')
f1_scores = cross_val_score(model, X_normalized, y, cv=k, scoring=f1_scorer)

mean_precision = precisions.mean()
mean_recall = recalls.mean()
mean_f1 = f1_scores.mean()

print(f"Mean Precision (k={k}): {mean_precision:.4f}")
print(f"Mean Recall (k={k}): {mean_recall:.4f}")
print(f"Mean F1 Score (k={k}): {mean_f1:.4f}")


print("Support Vector Classifier")
model = SVC(C=0.25)
scores = cross_val_score(model, X_normalized, y, cv=k)
mean_accuracy = scores.mean()
print(f"Mean Accuracy (k={k}): {mean_accuracy:.4f}")

precisions = cross_val_score(model, X_normalized, y, cv=k, scoring='precision')
recalls = cross_val_score(model, X_normalized, y, cv=k, scoring='recall')
f1_scores = cross_val_score(model, X_normalized, y, cv=k, scoring=f1_scorer)

mean_precision = precisions.mean()
mean_recall = recalls.mean()
mean_f1 = f1_scores.mean()

print(f"Mean Precision (k={k}): {mean_precision:.4f}")
print(f"Mean Recall (k={k}): {mean_recall:.4f}")
print(f"Mean F1 Score (k={k}): {mean_f1:.4f}")


print("K-Nearest Neighbors Classifier")
model = KNeighborsClassifier(n_neighbors=13)
scores = cross_val_score(model, X_normalized, y, cv=k)
mean_accuracy = scores.mean()
print(f"Mean Accuracy (k={k}): {mean_accuracy:.4f}")

precisions = cross_val_score(model, X_normalized, y, cv=k, scoring='precision')
recalls = cross_val_score(model, X_normalized, y, cv=k, scoring='recall')
f1_scores = cross_val_score(model, X_normalized, y, cv=k, scoring=f1_scorer)

mean_precision = precisions.mean()
mean_recall = recalls.mean()
mean_f1 = f1_scores.mean()

print(f"Mean Precision (k={k}): {mean_precision:.4f}")
print(f"Mean Recall (k={k}): {mean_recall:.4f}")
print(f"Mean F1 Score (k={k}): {mean_f1:.4f}")


print(" Random Forest Classifier")
model = RandomForestClassifier(max_depth=128, n_estimators=7, random_state=random_seed)
scores = cross_val_score(model, X_normalized, y, cv=k)
mean_accuracy = scores.mean()
print(f"Mean Accuracy (k={k}): {mean_accuracy:.4f}")

precisions = cross_val_score(model, X_normalized, y, cv=k, scoring='precision')
recalls = cross_val_score(model, X_normalized, y, cv=k, scoring='recall')
f1_scores = cross_val_score(model, X_normalized, y, cv=k, scoring=f1_scorer)

mean_precision = precisions.mean()
mean_recall = recalls.mean()
mean_f1 = f1_scores.mean()

print(f"Mean Precision (k={k}): {mean_precision:.4f}")
print(f"Mean Recall (k={k}): {mean_recall:.4f}")
print(f"Mean F1 Score (k={k}): {mean_f1:.4f}")


print("Gaussian Naive Bayes")
model = GaussianNB()
scores = cross_val_score(model, X_normalized, y, cv=k)
mean_accuracy = scores.mean()
print(f"Mean Accuracy (k={k}): {mean_accuracy:.4f}")

precisions = cross_val_score(model, X_normalized, y, cv=k, scoring='precision')
recalls = cross_val_score(model, X_normalized, y, cv=k, scoring='recall')
f1_scores = cross_val_score(model, X_normalized, y, cv=k, scoring=f1_scorer)

mean_precision = precisions.mean()
mean_recall = recalls.mean()
mean_f1 = f1_scores.mean()

print(f"Mean Precision (k={k}): {mean_precision:.4f}")
print(f"Mean Recall (k={k}): {mean_recall:.4f}")
print(f"Mean F1 Score (k={k}): {mean_f1:.4f}")


print("Gradient Boosting Classifier")
model = GradientBoostingClassifier(random_state=random_seed)
scores = cross_val_score(model, X_normalized, y, cv=k)
mean_accuracy = scores.mean()
print(f"Mean Accuracy (k={k}): {mean_accuracy:.4f}")

precisions = cross_val_score(model, X_normalized, y, cv=k, scoring='precision')
recalls = cross_val_score(model, X_normalized, y, cv=k, scoring='recall')
f1_scores = cross_val_score(model, X_normalized, y, cv=k, scoring=f1_scorer)

mean_precision = precisions.mean()
mean_recall = recalls.mean()
mean_f1 = f1_scores.mean()

print(f"Mean Precision (k={k}): {mean_precision:.4f}")
print(f"Mean Recall (k={k}): {mean_recall:.4f}")
print(f"Mean F1 Score (k={k}): {mean_f1:.4f}")


print("Multi-Layer Perceptron Classifier")
model = MLPClassifier(hidden_layer_sizes=(50, 50), activation='tanh', max_iter=1000, learning_rate_init=0.0001, learning_rate='constant', random_state=random_seed)
scores = cross_val_score(model, X_normalized, y, cv=k)
mean_accuracy = scores.mean()
print(f"Mean Accuracy (k={k}): {mean_accuracy:.4f}")

precisions = cross_val_score(model, X_normalized, y, cv=k, scoring='precision')
recalls = cross_val_score(model, X_normalized, y, cv=k, scoring='recall')
f1_scores = cross_val_score(model, X_normalized, y, cv=k, scoring=f1_scorer)

mean_precision = precisions.mean()
mean_recall = recalls.mean()
mean_f1 = f1_scores.mean()

print(f"Mean Precision (k={k}): {mean_precision:.4f}")
print(f"Mean Recall (k={k}): {mean_recall:.4f}")
print(f"Mean F1 Score (k={k}): {mean_f1:.4f}")


print("Decision Tree Classifier")
kf = KFold(n_splits=k, shuffle=True, random_state=random_seed)
model = DecisionTreeClassifier(max_depth=7, min_samples_split=2, random_state=random_seed)
scores = cross_val_score(model, X_normalized, y, cv=kf)
mean_accuracy = scores.mean()
print(f"Mean Accuracy (k={k}): {mean_accuracy:.4f}")

precisions = cross_val_score(model, X_normalized, y, cv=k, scoring='precision')
recalls = cross_val_score(model, X_normalized, y, cv=k, scoring='recall')
f1_scores = cross_val_score(model, X_normalized, y, cv=k, scoring=f1_scorer)

mean_precision = precisions.mean()
mean_recall = recalls.mean()
mean_f1 = f1_scores.mean()

print(f"Mean Precision (k={k}): {mean_precision:.4f}")
print(f"Mean Recall (k={k}): {mean_recall:.4f}")
print(f"Mean F1 Score (k={k}): {mean_f1:.4f}")


print("Polynomial SVM")
poly_svm = SVC(kernel='poly', C=1, degree=2)
accuracy_scores = cross_val_score(poly_svm, X_normalized, y, cv = k, scoring='accuracy')
precision_scores = cross_val_score(poly_svm, X_normalized, y, cv= k, scoring='precision_macro')
recall_scores = cross_val_score(poly_svm, X_normalized, y, cv=k, scoring='recall_macro')
f1_scores = cross_val_score(model, X_normalized, y, cv=k, scoring=f1_scorer)
mean_accuracy = accuracy_scores.mean()
mean_precision = precision_scores.mean()
mean_recall = recall_scores.mean()
mean_f1 = f1_scores.mean()
print(f"Mean Accuracy: {mean_accuracy:.4f}")
print(f"Mean Precision : {mean_precision:.4f}")
print(f"Mean Recall : {mean_recall:.4f}")
print(f"Mean F1 Score (k={k}): {mean_f1:.4f}")


print("Support Vector Classifier with RBF Kernel")
rbf_svm = SVC(kernel='rbf', C=16384, gamma=0.000977)
accuracy_scores = cross_val_score(rbf_svm, X_normalized, y, cv=k, scoring='accuracy')
precision_scores = cross_val_score(rbf_svm, X_normalized, y, cv=k, scoring='precision_macro')
recall_scores = cross_val_score(rbf_svm, X_normalized, y, cv=k, scoring='recall_macro')
f1_scores = cross_val_score(model, X_normalized, y, cv=k, scoring=f1_scorer)

mean_accuracy = accuracy_scores.mean()
mean_precision = precision_scores.mean()
mean_recall = recall_scores.mean()
mean_f1 = f1_scores.mean()

print(f"Mean Accuracy : {mean_accuracy:.4f}")
print(f"Mean Precision : {mean_precision:.4f}")
print(f"Mean Recall  : {mean_recall:.4f}")
print(f"Mean F1 Score (k={k}): {mean_f1:.4f}")


print("Support Vector Classifier with Sigmoid Kernel")
sigmoid_svm = SVC(kernel='sigmoid', C=16384.0, gamma=0.031250)
accuracy_scores = cross_val_score(sigmoid_svm, X_normalized, y, cv=k, scoring='accuracy')
precision_scores = cross_val_score(sigmoid_svm, X_normalized, y, cv=k, scoring='precision_macro')
recall_scores = cross_val_score(sigmoid_svm, X_normalized, y, cv=k, scoring='recall_macro')
f1_scores = cross_val_score(model, X_normalized, y, cv=k, scoring=f1_scorer)

mean_accuracy = accuracy_scores.mean()
mean_precision = precision_scores.mean()
mean_recall = recall_scores.mean()
mean_f1 = f1_scores.mean()

print(f"Mean Accuracy : {mean_accuracy:.4f}")
print(f"Mean Precision : {mean_precision:.4f}")
print(f"Mean Recall  : {mean_recall:.4f}")
print(f"Mean F1 Score (k={k}): {mean_f1:.4f}")
