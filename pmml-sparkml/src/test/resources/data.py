from sklearn.preprocessing import LabelEncoder

import math
import pandas

def read_csv(name):
	df = pandas.read_csv("csv/" + name + ".csv", na_values = ["", "NA", "N/A"])

	X = df.iloc[:, :-1].values
	y = df.iloc[:, -1].values

	return (X, y)

def write_libsvm(X, y, name):
	n_rows, n_cols = X.shape

	with open("libsvm/" + name + ".libsvm", "w") as file:
		for row in range(n_rows):
			cells = []
			label = y[row]
			cells.append("{:g}".format(label))
			for col in range(n_cols):
				value = X[row, col]
				if not math.isnan(value):
					cells.append("{}:{:g}".format(col + 1, value))
			file.write(" ".join(cells) + "\n")

#
# Auto
#

auto_X, auto_y = read_csv("Auto")

write_libsvm(auto_X, auto_y, "Auto")

auto_X, auto_y = read_csv("AutoNA")

write_libsvm(auto_X, auto_y, "AutoNA")

#
# Housing
#

housing_X, housing_y = read_csv("Housing")

write_libsvm(housing_X, housing_y, "Housing")

#
# Iris
#

iris_X, iris_y = read_csv("Iris")

iris_le = LabelEncoder()
iris_y = iris_le.fit_transform(iris_y)

write_libsvm(iris_X, iris_y, "Iris")
