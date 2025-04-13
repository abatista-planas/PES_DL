import pickle

from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

# load data
iris = datasets.load_iris()
x = iris.data
y = iris.target

# train model
clf = RandomForestClassifier(max_depth=2, n_estimators=5, random_state=47)
clf.fit(x, y)

# save model
with open("clf.pickle", "wb") as f:
    pickle.dump(clf, f)
