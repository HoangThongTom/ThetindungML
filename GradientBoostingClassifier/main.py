from models.GradientBoosting import GradientBoostingClassifier
from metrics.Accuracy import accuracy_score
from metrics.ClassificationReport import classification_report
from metrics.ConfusionMatrix import confusion_matrix
from visualization.ConfusionMatrixDisplay import ConfusionMatrixDisplay

X = [[1], [2], [3], [4]]
y = [0, 1, 0, 1]

model = GradientBoostingClassifier()

model.fit(X, y)

y_pred = model.predict(X)

acc = accuracy_score(y, y_pred)
report = classification_report(y, y_pred)
cm = confusion_matrix(y, y_pred)

disp = ConfusionMatrixDisplay(cm)

print("Accuracy:", acc)
print("Report:", report)
disp.plot()

print("Xong")