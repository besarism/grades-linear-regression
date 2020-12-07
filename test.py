import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

data = pd.read_csv("student-mat.csv", sep=";")
data = data[["G1","G2","G3","studytime","failures","absences"]]

predict = "G3"
x = np.array(data.drop([predict], 1)) # Features
y = np.array(data[predict]) # Labels

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

# best = 0
# for i in range(30):
#     x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
#     linear = linear_model.LinearRegression()
#     linear.fit(x_train, y_train)
#     acc = linear.score(x_test, y_test)
#     if acc > best:
#         print(acc)
#         best = acc
#         with open("studentmodel.pickle","wb") as f:
#             pickle.dump(linear,f)

pickle_in = open("studentmodel.pickle","rb")
linear = pickle.load(pickle_in)


print('Coefficient: \n', linear.coef_) # These are each slope value
print('Intercept: \n', linear.intercept_) # This is the intercept

predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(round(predictions[x]), x_test[x], y_test[x])

style.use("ggplot")
pyplot.scatter(data["absences"], data[predict])
pyplot.xlabel("G1")
pyplot.ylabel("Final grade")
pyplot.show()
