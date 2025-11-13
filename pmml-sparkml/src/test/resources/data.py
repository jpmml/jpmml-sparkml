from sklearn.datasets import dump_svmlight_file
from sklearn.preprocessing import LabelEncoder

import pandas

df = pandas.read_csv("csv/Housing.csv")

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

dump_svmlight_file(X, y, "libsvm/Housing.libsvm", zero_based = False)

df = pandas.read_csv("csv/Iris.csv")

X = df.iloc[:, :-1].values

le = LabelEncoder()
y = le.fit_transform(df.iloc[:, -1].values)

dump_svmlight_file(X, y, "libsvm/Iris.libsvm", zero_based = False)
