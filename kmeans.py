import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
style.use("ggplot")

#Training set
X = np.array([[1,2],
              [3.5,10],
              [5,8],
              [8,8],
              [1,0.6],
              [9,11],
              [1, 3],
              [0, 9],
              [0, 3],
              [5, 4],
              [6, 4]])

#Plotting the given points
plt.scatter(X[:,0], X[:,1], s=150)
plt.show()

colors = 10*["g","r","c","b","k"]

class K_Means:
    def __init__(self, k=3, tol=0.001, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, data):
        self.centroids = {}

        # Cluster : Centroids
        for i in range(self.k):
            self.centroids[i] = data[i]

        for i in range(self.max_iter):
            self.classifications = {}

            #Each cluster contains list of points
            for i in range(self.k):
                self.classifications[i] = []

            #Assigning points to clusters
            for featureset in data:
                distances = [np.linalg.norm(featureset - self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset)

            prev_centroids = dict(self.centroids)

            #New centroids
            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification], axis=0)

            #Optimization: Carry on iteration until centroids do not change
            optimized =True

            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid-original_centroid)/original_centroid*100.0) > self.tol:
                    optimized = False

            if optimized:
                break


    def predict(self, data):
        distances = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification

clf = K_Means()
clf.fit(X)

#Plotting centroids
for centroid in clf.centroids:
    plt.scatter(clf.centroids[centroid][0], clf.centroids[centroid][1], marker="o", color="k", s=175, linewidths=5)

#Plotting given points(Training set)
for classification in clf.classifications:
    color = colors[classification]
    for featureset in clf.classifications[classification]:
        plt.scatter(featureset[0], featureset[1], marker="x", color=color, s=150, linewidths=5)

#Testing set
unknowns = np.array([[5,5],
                     [3,1.2],
                     [4,8.3],
                     [3,7]])

#Plotting unknowns
for unknown in unknowns:
    classification = clf.predict(unknown)
    plt.scatter(unknown[0], unknown[1], marker="*", color=colors[classification], s=150)

plt.show()