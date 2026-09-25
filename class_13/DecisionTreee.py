import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.tree import plot_tree

data = {
    "StudyHours": [2, 3, 4, 5, 6, 7, 8, 9],
    "Attendance": [50, 60, 70, 75, 80, 85, 90, 95],
    "Assignment": [40, 55, 65, 70, 80, 85, 90, 95],
    "Result": [0, 0, 0, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)
print(df)

X = df[["StudyHours", "Attendance", "Assignment"]]
y = df["Result"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

model = DecisionTreeClassifier(
    max_depth=3,
    random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print(predictions)

results = pd.DataFrame({
    "Actual": y_test,
    "Predicted": predictions
})

print(results)

accuracy_score = accuracy_score(y_test, predictions)
print("Accuracy", accuracy_score)

report = classification_report(y_test, predictions)
print(report)

# Study Hours = 6
# Attendance = 82
# Assignment = 78

new_student = pd.DataFrame(
    [[6, 82, 78]],
    columns=["StudyHours", "Attendance", "Assignment"]
)

prediction = model.predict(new_student)

print(prediction)

plt.figure(figsize=(12, 8))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=["Fail", "Pass"],
    filled=True
)

plt.show()






