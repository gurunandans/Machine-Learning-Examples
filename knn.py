import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import warnings
from matplotlib import style
from collections import Counter
style.use('fivethirtyeight')

#Example points
plot1 = [1,3]
plot2 = [2,5]

#Using Euclidean distance formula
euclidean_distance = sqrt((plot1[0] - plot2[0])**2 + (plot1[1] - plot2[1])**2)
print(euclidean_distance)

#Plots(Features) and Classes(Label) -Training
dataset = {'k':[[1,2],[2,3],[3,1]], 'r':[[6,5],[7,7],[8,6]]}
#Testing
new_features = [3,4]

#Plotting Training set
for cl in dataset:
    for pt in dataset[cl]:
        plt.scatter(pt[0], pt[1], s=100, color=cl)

def k_nearest_neighbours(data, predict, k=3):
    if(len(data)>=3):
        warnings.warn('K is set to a value less than total voting groups')

    distances = []
    for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))
            distances.append([euclidean_distance, group])

    votes = [i[1] for i in sorted(distances)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]
    return vote_result

result = k_nearest_neighbours(dataset, new_features)
print(result)

plt.scatter(new_features[0], new_features[1], s=150, color=result)
plt.title("K-Nearest-Neighbour Algorithm").set_size(15)
plt.show()