# def extract_features_from_query(range_query, table_stats, considered_cols):
#     # feat:     [c1_begin, c1_end, c2_begin, c2_end, ... cn_begin, cn_end, AVI_sel, EBO_sel, Min_sel]
#     #           <-                   range features                    ->, <-     est features     ->
#     feature = []
#     # YOUR CODE HERE: extract features from query
#     for col in considered_cols:
#         if col in range_query.column_names():
#             if col in range_query.col_left:
#                 feature.append(min_max_normalize(range_query.col_left[col],table_stats.columns[col].min_val(),table_stats.columns[col].max_val()))
#                 if col in range_query.col_right:
#                     feature.append(min_max_normalize(range_query.col_right[col],table_stats.columns[col].min_val(),table_stats.columns[col].max_val()))
#                 else:
#                     feature.append(min_max_normalize(table_stats.columns[col].max_val(),table_stats.columns[col].min_val(),table_stats.columns[col].max_val()))
#             else:
#                 feature.append(0)
#                 feature.append(min_max_normalize(range_query.col_right[col],table_stats.columns[col].min_val(),table_stats.columns[col].max_val()))
#         else:
#             feature.append(0)
#             feature.append(1)
#     feature.append(stats.AVIEstimator.estimate(range_query,table_stats))
#     feature.append(stats.ExpBackoffEstimator.estimate(range_query,table_stats))
#     feature.append(stats.MinSelEstimator.estimate(range_query,table_stats))
#     return feature


import numpy as np
import torch
from sklearn.cluster import KMeans
from collections import Counter


Array1 = [
    [1, 2, 3],
    [4, 5, 6],
    [3, 2, 5],
    [4, 2, 3]
]

Array2 = [
    [7, 8, 9],
    [8, 4, 6]

]

Array3 = [
    [1],
    [2],
    [11],
    [22],
    [55],
    [34],
    [128],
    [345]
]

km = KMeans(n_clusters=2,random_state=123)
km.fit(Array3)
distinct_values = Counter(km.labels_)
print(len(distinct_values))
print(km.labels_)







