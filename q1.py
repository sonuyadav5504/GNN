from matplotlib import markers
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

dimensions = [1, 2, 4, 8, 16, 32, 64]
ratios = []

for d in dimensions:
    dataset = np.random.rand(1000000, d).round(5)
    query_points = dataset[np.random.choice(len(dataset), 100, replace=False)]

    x1 = []
    x2 = []
    x3 = []

    for query_point in query_points:
        distances_l1 = cdist([query_point], dataset, metric='cityblock').flatten()
        distances_l2 = cdist([query_point], dataset, metric='euclidean').flatten()
        distances_linf = cdist([query_point], dataset, metric='chebyshev').flatten()

        farthest_distance_l1 = np.max(distances_l1)
        farthest_distance_l2 = np.max(distances_l2)
        farthest_distance_linf = np.max(distances_linf)
        nearest_distance_l1 = np.min(distances_l1[distances_l1 > 0])  # Avoid division by zero
        nearest_distance_l2 = np.min(distances_l2[distances_l2 > 0])  # Avoid division by zero
        nearest_distance_linf = np.min(distances_linf[distances_linf > 0])  # Avoid division by zero

        x1.append((farthest_distance_l1 / nearest_distance_l1).round(5))
        x2.append((farthest_distance_l2 / nearest_distance_l2).round(5))
        x3.append((farthest_distance_linf / nearest_distance_linf).round(5))

    ratios.append(np.mean(x1))
    ratios.append(np.mean(x2))
    ratios.append(np.mean(x3))

    
#all the three plot's combine observaion 
plt.figure(figsize=(12, 4))
plt.plot(dimensions, [ratios[i] for i in range(0, 21,3)], label='l1_ratio', marker="o", color='red')
plt.plot(dimensions, [ratios[i] for i in range(1,21,3)], label='l2_ratio', marker="o", color='blue')
plt.plot(dimensions, [ratios[i] for i in range(2,21,3)], label='linf_ratio', marker="o", color='green')

plt.xlabel("Dimension (d)")
plt.ylabel("Average Ratio (ratios)")
plt.legend()
plt.grid()
plt.show()    
