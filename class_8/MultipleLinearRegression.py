import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

data = {
    "Size": [1000, 1500, 1800, 2200, 2800],
    "Bedrooms": [2, 3, 3, 4, 5],
    "Bathrooms": [1, 2, 2, 3, 4],
    "Age": [10, 8, 5, 3, 2],
    "Price": [5000000, 7500000, 9000000, 12000000, 16000000]
}

df = pd.DataFrame(data)
print(df)

X = df[["Size", "Bedrooms", "Bathrooms", "Age"]]
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(predictions)

results = pd.DataFrame({
    "Actual": y_test,
    "Predicted": predictions
})
print(results)

print(model.coef_)
print(model.intercept_)
