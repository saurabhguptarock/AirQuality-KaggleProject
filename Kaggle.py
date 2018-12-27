import pandas as pd
import csv
from sklearn.linear_model import LinearRegression

df = pd.read_csv('Train.csv')
dft = pd.read_csv('Test.csv')

x = df.values
y = df.target
x = x[:, :-1]
x_test = dft.values

lr = LinearRegression(normalize=True)
lr.fit(x, y)

y_pred = lr.predict(x_test)

with open('submission.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['Id', 'target'])
    for i in range(400):
        w.writerow([i, y_pred[i]])
