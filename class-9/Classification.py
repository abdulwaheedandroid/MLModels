import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

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

result = pd.DataFrame({
    "Actual" : y_test,
    "Predicted" : predictions
})

print(result)

new_student = pd.DataFrame(
    [[6, 80]],
    columns=["StudyHours", "Attendance"]
)
prediction = model.predict(new_student)
print(prediction)

probability = model.predict_proba(new_student)
print(probability)

accuracy = accuracy_score(y_test, predictions)
print(accuracy)

cm = confusion_matrix(y_test, predictions)
print(cm)





