import numpy as np
from sklearn.cluster import KMeans

class Cluster:

    def run(self, dict_data):
        volume = []
        for key, value in dict_data.items():
            n_vol_sku = np.array(value)
            sum_vol_sku = np.sum(n_vol_sku)
            volume.append(sum_vol_sku)
        
        np_volume = np.array(volume)
        kmeans_model = KMeans(n_clusters=3).fit(np_volume)
        print("Centers are: ")
        print(kmeans_model.cluster_centers_)


