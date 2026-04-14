from models.gradient_boosting import gradient_boosting_classifier
from metrics.accuracy import accuracy_score
from metrics.classification_report import classification_report
from metrics.confusion_matrix import confusion_matrix
from visualization.confusion_matrix_display import confusion_matrix_display
X = [[1], [2], [3], [4]]
y = [0, 1, 0, 1]

model = gradient_boosting_classifier()

model.fit(X, y)

y_pred = model.predict(X)

acc = accuracy_score(y, y_pred)
report = classification_report(y, y_pred)
cm = confusion_matrix(y, y_pred)

disp = confusion_matrix_display(cm)

print("Accuracy:", acc)
print("Report:", report)
disp.plot()

print("Xong")