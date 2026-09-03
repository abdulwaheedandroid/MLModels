from sklearn.metrics import (
accuracy_score,
confusion_matrix,
classification_report
)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

import pandas as pd

data = {
    "StudyHours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Attendance": [45, 50, 55, 60, 65, 72, 78, 85, 90, 95],
    "Result": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)
print(df)
X_test = df[["StudyHours", "Attendance"]]
y = df["Result"]

X_train, X_test, y_train, y_test = train_test_split(
    X_test,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = LogisticRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(predictions)

#Accuracy
accuracy = accuracy_score(y_test,predictions)
print(accuracy)

#Confusion Matrix
confusion_matrix = confusion_matrix(y_test, predictions)
print(confusion_matrix)

#Classification Report
classification_report = classification_report(y_test, predictions)
print(classification_report)



