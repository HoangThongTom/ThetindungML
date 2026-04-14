import models.gradient_boosting as gb
import metrics.accuracy as acc
import metrics.classification_report as cr
import metrics.confusion_matrix as cm   
import visualization.confusion_matrix_display as cmd
print("Đường dẫn file Python đang đọc:", gb.__file__, acc.__file__, cr.__file__, cm.__file__, cmd.__file__)
print("Danh sách những thứ tìm thấy trong file:", dir(gb), dir(acc))

# from models.gradient_boosting import GradientBoostingClassifier
# from metrics.accuracy import accuracy_score
# from metrics.classification_report import classification_report
# from metrics.confusion_matrix import confusion_matrix
# from visualization.confusion_matrix_display import confusion_matrix_display
# X = [[1], [2], [3], [4]]
# y = [0, 1, 0, 1]

# model = GradientBoostingClassifier()
# model.fit(X, y)
# y_pred = model.predict(X)


# acc = accuracy_score(y, y_pred)
# report = classification_report(y, y_pred)
# cm = confusion_matrix(y, y_pred)

# disp = confusion_matrix_display(cm)

# print("Accuracy:", acc)
# print("Report:", report)
# disp.plot()

# print("Xong")