import torch
import torch.nn.functional as F
from sklearn import preprocessing
import numpy as np
import json

operators = ["Projection", "Selection", "Sort", "HashAgg", "HashJoin", "TableScan", "IndexScan", "TableReader",
             "IndexReader", "IndexLookUp"]

def get_operator_enc_dict(operators):
    operators_x = np.array(operators).reshape(len(operators), 1)
    enc = preprocessing.OneHotEncoder()
    operator_enc = enc.fit_transform(operators_x).toarray()
    print(operator_enc)
    result_dict = {}
    for index,item in enumerate(operators):
        result_dict[item] = operator_enc[index]
    return result_dict


s = '{"id":"TableReader_7","est_rows":"3351.01","task":"root","acc_obj":"","op_info":"data:Selection_6, row_size: 196","children":[{"id":"Selection_6","est_rows":"3351.01","task":"cop","acc_obj":"","op_info":"gt(imdb.title.episode_of_id, 0), gt(imdb.title.kind_id, 0), lt(imdb.title.episode_of_id, 2528185), lt(imdb.title.kind_id, 5), row_size: 196","children":[{"id":"TableFullScan_5","est_rows":"10000.00","task":"cop","acc_obj":"table:title","op_info":"keep order:false, row_size: 196","children":null}]}]}'

d = json.loads(s)
print(d)



