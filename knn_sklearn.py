import matplot.lib as plt
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import random
from scipy.stats as ss

'''
    Sample from PH526x Using Python for Research
'''

def distance(p1, p2):
  return np.sqrt(np.sum(np.power(p2 - p1, 2)))

def majority_vote(votes):
  vote_counts = {}
  for vote in votes:
    if vote in vote_counts:
      vote_counts[vote] += 1
    else:
      vote_counts[vote] = 1

  winners = []
  max_count = max(vote_counts.values())
  for vote, count in vote_counts.items():
    if count == max_count:
      winners.append(vote)

  return random.choice(winners)

'''
Simple Version
import scipy.stats as ss

def majority_vote(votes):
  mode, count = ss.mstats.mode(votes)
  return mode
'''

def find_nearest_neighborgs(p, points, k=5):
  distances = np.zeros(points.shape[0])
  for i in range(len(distances)):
    distances[i] = distance(p, points[i])
    ind = np.argsort(distances)
  return ind[:k]

def knn_predict(p, points, outcomes, k=5):
  ind = find_nearest_neighborgs(p, points, k)
  return majority_vote(outcomes[ind])

def generate_synth_data(n=50):
    points = np.concatenate((ss.norm(0, 1).rvs((n, 2)), ss.norm(1,1).rvs((n,2))), axis=0)
    outcomes = np.concatenate((np.repeat(0,n), np.repear(1,n)))
    return (points, outcomes)

def make_prediction_grid(predictors, outcomes, limits, h, k):
    (x_min, x_max, y_min, y_max) = limits

    xs = np.arange(x_min, x_max, h)
    ys = np.arange(y_min, y_max, h)
    xx, yy = np.meshgrid(xs, ys)

    prediction_grid = np.zeros(xx.shape, dtype=int)
    for i,x in enumerate(xs):
        for j,y in enumerate(ys):
            p = np.array([x,y])
            prediction_grid[j,i] =  knn_predict(p, predictors, outcomes, k)
    return (xx, yy, prediction_grid)

iris = datasets.load_iris()

predictors = iris.data[:, 0:2]
outcomes = iris.target

k = 5
h = 0.1
limits = (4, 8, 1.5, 4.5)
(xx, yy, prediction_grid) = make_prediction_grid(predictors, outcomes, limits, k, h)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(predictors, outcomes)
sk_predictions = knn.predict(predictors)
print(sk_predictions)

my_predictions = np.array([knn_predict(p, predictors, outcomes, k, n) for p in predictors])
print(sk_predictions == my_predictions)
print(100 * np.mean(sk_predictions == my_predictions))
print(100 * np.mean(sk_predictions == outcomes))
print(100 * np.mean(my_predictions == outcomes))

plt.plot(predictors[outcomes==0][:, 0], predictors[outcomes==0][:, 1], 'ro')
plt.plot(predictors[outcomes==1][:, 0], predictors[outcomes==1][:, 1], 'go')
plt.plot(predictors[outcomes==2][:, 0], predictors[outcomes==0][:, 2], 'bo')
plt.show()
