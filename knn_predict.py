'''
    Sample from PH526x Using Python for Research
'''
import numpy as np
import random
from scipy.stats as ss

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

points, outcomes = generate_synth_data(20)
p = np.array([2.5, 2.7])

print(knn_predict(p, points, outcomes, k=5))
