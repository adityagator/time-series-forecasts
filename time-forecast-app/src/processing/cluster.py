import numpy as np
from sklearn.cluster import KMeans
from collections import Counter

class Cluster:

    def run(dict_data):
        print("in cluster")
        volume = np.array([])
        vol_dict = {}
        norm_dict = {}

        # initializing maps for volume and intermittency
        for key, value in dict_data.items():
            n_vol_sku = np.array(value)
            sum_vol_sku = np.sum(n_vol_sku)
            vol_dict[key] = int(sum_vol_sku)
            volume = np.append(volume, int(sum_vol_sku))
            mean = np.mean(n_vol_sku)
            sd = np.std(n_vol_sku)
            norm_arr = []
            for num in value:
                norm = (num - mean)/sd
                norm_arr.append(norm)
            norm_dict[key] = norm_arr

        # calculating volume
        volume = volume.reshape(-1, 1)
        kmeans_model = KMeans(n_clusters=3).fit(volume)
        centers = kmeans_model.cluster_centers_
        centers = [int(centers[0]), int(centers[1]), int(centers[2])]
        centers.sort()
        final_vol_dict = {}
        for key, value in vol_dict.items():
            if value < centers[0]:
                final_vol_dict[key] = "low"
            elif value >= centers[0] and value <= centers[1]:
                final_vol_dict[key] = "medium"
            else:
                final_vol_dict[key] = "high"

        # calculating intermittency
        sum_arr = np.array([])
        int_dict = {}
        for key, value in norm_dict.items():
            diff_arr = np.array([])
            for i in range(0, len(value) - 1):
                diff = value[i + 1] - value[i]
                diff_arr = np.append(diff_arr, diff)
            sum = np.sum(diff_arr)
            sum_arr = np.append(sum_arr, sum)
            int_dict[key] = float(sum)

        sum_arr = sum_arr.reshape(-1, 1)
        kmeans_model = KMeans(n_clusters=3).fit(sum_arr)
        centers = kmeans_model.cluster_centers_
        centers = [centers[0], centers[1], centers[2]]
        centers.sort()
        final_int_dict = {}
        for key, value in int_dict.items():
            if value < centers[0]:
                final_int_dict[key] = "low"
            elif value >= centers[0] and value <= centers[1]:
                final_int_dict[key] = "medium"
            else:
                final_int_dict[key] = "high"
        
        return final_vol_dict, final_int_dict





