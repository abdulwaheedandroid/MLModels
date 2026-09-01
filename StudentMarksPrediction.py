import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = {
    "StudyHours": [1, 2, 3, 4, 5],
    "Marks": [40, 48, 55, 65, 75]
}

df = pd.DataFrame(data)
print(df)

X = df[["StudyHours"]]
y = df["Marks"]

model = LinearRegression()
model.fit(X,y)
prediction = model.predict([[6]])
print(prediction)

plt.scatter(X,y)
plt.plot(X, model.predict(X))
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Study Hours VS Marks")
plt.show()
