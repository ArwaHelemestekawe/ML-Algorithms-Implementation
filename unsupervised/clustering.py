import numpy as np 
import os
os.system("cls")
import matplotlib.pyplot as plt
def euclidean_distance(x1,x2):
    # each point represented as a vector
    return np.sqrt(np.sum((x1-x2)**2))


class Clustering :
    def __init__(self,n_iteration=100,initial_number_of_clusters=3,tol=1e4):
        self.n_iteration=n_iteration
        self.initial_number_of_clusters=initial_number_of_clusters
        self.centroid=None
        self.labels=None
        self.tol=tol

    def initialize_centeroid(self,x):
        self.x=x
        self.number_of_sampels,self.number_of_features=x.shape
        # inititialisze random centeroid from data
        random_centeroid_index=np.randome.choice(self.number_of_sampels,self.initial_number_of_clusters,replace=False)
        self.random_centroid=[self.x[i] for i in random_centeroid_index]
        for i in self.n_iteration:
            self.labels=self._create_clusters(self.random_centroid)
            self.old_centroid=self.random_centroid
            self.new_centeroid=self.update_centroid(self.labels)
            if self.convergence(self.old_centroid,self.new_centeroid):
                break


    def _create_clusters(self, random_centroid):
        self.clusters=[[] for clusters in range(self.initial_number_of_clusters)]
        for point_index,point_value in enumerate(self.x):
           cluster_index=self.nearest_cluster(random_centroid,point_value)
           self.clusters[cluster_index].append(point_index)
           return self.clusters


    def nearest_cluster(self,randome_centroid,point_value):
        distances=[euclidean_distance(centroid,point_value) for centroid in self.random_centroid]
        smallest_distance_index_centroid=np.argmin(distances)
        return smallest_distance_index_centroid

    def _update_centroid(self,clusters):
        
        # number of features == number of axises , as we calculate the mean on each dimension >> so number of features= coloumn
        number_of_coloumn=len(self.x[0])
        #x=[frist_cluster=[1,2,3,4,5],       each slot is index in x dataset
        #  second_cluster=[4,6,8,9,3]]
        # number of rows == number of dimension 
        mean_centroids=np.zeros(len(clusters),number_of_coloumn)
        for each_cluster_index,list_of_indices_inside_each_cluster in enumerate(clusters):
         points_of_each_cluster= [self.x[i] for i in list_of_indices_inside_each_cluster]
         # points=[frist_point_in_cluster-1=[3.14,60,50],
         #        second_point_in_cluster_1=[4.56,78,30],
         #         third_point_in_cluster_1=[9.87,45,60]]
         mean_centroids[each_cluster_index]=np.mean(points_of_each_cluster,axis=0)
         return mean_centroids
        
    
    
    def _convergence(self,oldcentroid,new_centeroid):
        distance_between_each_old_centeriod_and_new=[euclidean_distance(oldcentroid[i],new_centeroid[i]) for i in range(self.clusters)]
        sum_of_all_distancs=np.sum(distance_between_each_old_centeriod_and_new)
        return sum_of_all_distancs==self.tol

        # لحد هنا انا رجعت رقم الكلاستر او الانديكس 
        #When checking for convergence, the algorithm evaluates the changes in the positions of all centroids.
        #If any of the centroids still move beyond the predefined threshold, the algorithm will continue iterating.
        #Convergence is reached only when all centroids stop moving significantly between iterations.
        #In essence, all clusters need to converge together. If one centroid has reached its optimal position,
        #  but others have not, the algorithm will continue to adjust all centroids until they all meet the convergence criteria.
        # so we have to sum all the values of the array
        #  This ensures that the final clustering solution is stable and that no centroids are still changing
