import os

import numpy as np
from matplotlib import pyplot as plt
from math import sqrt
import json

from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

class ClusteringModule():
    
    def __init__ (self, filePath, size):
        dirname = os.path.dirname(__file__)
        path = os.path.join(dirname,filePath)  
        self.embeddings = self.loadDataFromFile(path, size)
        self.keywords = self.loadKeywordsFromFile()

    def loadDataFromFile (self, file, size):
        data = np.loadtxt(file)
        data = data.reshape((size, 384))
        return data

    def loadKeywordsFromFile(self):  
        dirname = os.path.dirname(__file__)
        path = os.path.join(dirname,"../Files/keywords.json")  
        file = open(path, "r")
        data = json.load(file)
        return data
    
    def scaleNormalize (self, data):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(data)
        
        # Normalizing the Data
        X_normalized = normalize(X_scaled)

        return X_normalized

    def generatePC (self, data, number_of_components):        
        # Reducing the dimensions of the data
        pca = PCA(n_components = number_of_components)
        X_principal = pca.fit_transform(data)
        print(pca.explained_variance_ratio_.cumsum())
        
        return X_principal

    def generateSpectralClusters (self, data, number_of_clusters = 15):
        spectral_model_rbf = SpectralClustering(n_clusters = number_of_clusters, affinity ='rbf')
        
        # Training the model and Storing the predicted cluster labels
        labels_rbf = spectral_model_rbf.fit_predict(data)

        #print(labels_rbf)

        return labels_rbf

    def conductRawSprectral (self):
        self.generateSpectralClusters(self.embeddings)

    def conductSpectralPC (self, number_of_clusters):
        normalizedData = self.scaleNormalize(self.embeddings)
        principalComponents = self.generatePC(normalizedData, 50)
        return self.generateSpectralClusters(principalComponents, number_of_clusters)
    
    def findOptimalClusters (self, data, show_graph = False):
        SilhouetteList = []
        number_of_clusters = []
        
        for i in range(10, 150, 5):
            labels = self.conductSpectralPC(i) #self.generateSpectralClusters(data, i)
            SilhouetteList.append(silhouette_score(data, labels))
            number_of_clusters.append(i)

        if (show_graph):
            plt.plot(number_of_clusters, SilhouetteList)
            plt.show()
        
        return number_of_clusters[np.argmax(SilhouetteList)]

    def findCentralPoints(self, labels, number_of_clusters):
        clusters = []
        
        for i in range(0, number_of_clusters):
            clusterSubList = []
            for j in range(0, len(labels)):
                if (i == labels[j]):
                    clusterSubList.append(j)
            clusters.append(clusterSubList)
        
        centroid = []

        for i in range(0, len(clusters)):
            Sum = self.embeddings[clusters[i][0]]
            for j in range(0, len(clusters[i])):
                if (i==j):
                    continue
                Sum = [Sum[k] + self.embeddings[j][k] for k in range(0, len(self.embeddings[0]))]
            Sum = [k / len(clusters[i]) for k in Sum]
            centroid.append(Sum)
        
        centralPoint = []

        for i in range(0, len(centroid)):
            distanceMatrix = []
            X = [(centroid[i][m] - self.embeddings[clusters[i][0]][m])**2 for m in range(0, len(self.embeddings[0]))]
            for k in range(1, len(clusters[i])):
                X = [X[m] + (centroid[i][m] - self.embeddings[k][m])**2 for m in range(0, len(self.embeddings[0]))]
            Sum = sum(X)
            Sum = sqrt(Sum)
            distanceMatrix.append(Sum)
            centralPoint.append(clusters[i][np.argmin(distanceMatrix)])
        #print(self.keywords[centralPoint[-1]])
        
        print(len(centralPoint))
        print(number_of_clusters)
        #self.printResults(centralPoint)

        self.clusters = clusters

        self.printResults(centralPoint)

        return centralPoint
    
    def printResults(self, centralPoint):
        clusters = self.clusters

        for i in range(0, len(clusters)):
            print("Cluster number - ", i)
            for j in range(0, len(clusters[i])):
                print (self.keywords[clusters[i][j]])
            print("Central Word - ", self.keywords[centralPoint[i]]) 
            print("\n\n")

    def reduceKeywords (self, ShowGraph = False):
        numClusters = self.findOptimalClusters(self.embeddings, show_graph=ShowGraph)
        labels = self.generateSpectralClusters(self.embeddings, number_of_clusters=numClusters)
        centers = self.findCentralPoints(labels, numClusters)
        Keywords = [self.keywords[i][0] for i in centers]
        
        self.saveClusteredKeywords(Keywords)
        return Keywords

    # Made so that twitter search can run independently on already made clustered keywords
    def saveClusteredKeywords(self,Keywords):
        dirname = os.path.dirname(__file__)
        path = os.path.join(dirname,"../Files/clusteredKeywords.json")
        file = open(path, "w")
        json.dump(Keywords, file)

# MyModel = ClusteringModule("../Files/embeddings.txt", 219)
# numClusters = MyModel.findOptimalClusters(MyModel.embeddings, show_graph=True)
# labels = MyModel.generateSpectralClusters(MyModel.embeddings, number_of_clusters=numClusters)
# centers = MyModel.findCentralPoints(labels, numClusters)
# MyModel.printResults(centers)







#LEGACY FUNCTIONS
# def generateDBScanClusters (self, data, Eps = 1.0, Min_samples = 4, Metric = 'cosine'):
#     clustering = DBSCAN(eps=Eps, min_samples = Min_samples, metric=Metric).fit(data)

#     print(clustering.labels_)
#     print("\n\n", len(set(clustering.labels_)))

#     return clustering.labels_

# def drawDistanceGraph (self, data):
#     neighbors = NearestNeighbors(n_neighbors=20)
#     neighbors_fit = neighbors.fit(data)
#     distances, indices = neighbors_fit.kneighbors(data)

#     distances = np.sort(distances, axis=0)
#     distances = distances[:,1]
#     plt.plot(distances)

# def conductTSNE (self, data, perp):
#     tsne = TSNE(n_components = 2, perplexity = perp)
#     Y = tsne.fit_transform(data)
#     return Y

# def conductRawDBScan (self, data):
#     self.generateDBScanClusters(data) #self.embeddings)

# def conductDBScanPC (self):
#     normalizedData = self.scaleNormalize(self.embeddings)
#     principalComponents = self.generatePC(normalizedData, 50)
#     self.generateDBScanClusters(principalComponents)