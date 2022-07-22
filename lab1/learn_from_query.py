import evaluation_utils as eval_utils
import matplotlib.pyplot as plt
import numpy as np
import range_query as rq
import json
import torch
import torch.nn as nn
import statistics as stats
import xgboost as xgb
from models.MLP import MLP
import torch.optim as optim
from tqdm import tqdm
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import ExtraTreeRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor
import warnings
warnings.filterwarnings('ignore')

EPOCHS = 100


def min_max_normalize(v, min_v, max_v):
    # The function may be useful when dealing with lower/upper bounds of columns.
    assert max_v > min_v
    return (v-min_v)/(max_v-min_v)


def extract_features_from_query(range_query, table_stats, considered_cols):
    # feat:     [c1_begin, c1_end, c2_begin, c2_end, ... cn_begin, cn_end, AVI_sel, EBO_sel, Min_sel]
    #           <-                   range features                    ->, <-     est features     ->
    feature = []
    # YOUR CODE HERE: extract features from query
    for col in considered_cols:
        if col in range_query.column_names():
            if col in range_query.col_left:
                feature.append(range_query.col_left[col])
                if col in range_query.col_right:
                    feature.append(range_query.col_right[col])
                else:
                    feature.append(table_stats.columns[col].max_val())
            else:
                feature.append(table_stats.columns[col].min_val())
                feature.append(range_query.col_right[col])
        else:
            feature.append(table_stats.columns[col].min_val())
            feature.append(table_stats.columns[col].max_val())
    feature.append(stats.AVIEstimator.estimate(range_query,table_stats))
    feature.append(stats.ExpBackoffEstimator.estimate(range_query,table_stats))
    feature.append(stats.MinSelEstimator.estimate(range_query,table_stats))
    return feature


def preprocess_queries(queris, table_stats, columns):
    """
    preprocess_queries turn queries into features and labels, which are used for regression model.
    """
    features, labels = [], []
    for item in queris:
        query, act_rows = item['query'], item['act_rows']
        feature, label = None, None
        # YOUR CODE HERE: transform (query, act_rows) to (feature, label)
        range_query = rq.ParsedRangeQuery.parse_range_query(query)
        feature = extract_features_from_query(range_query,table_stats,columns)
        label = act_rows
        label = np.log(label)
        # Some functions like rq.ParsedRangeQuery.parse_range_query and extract_features_from_query may be helpful.
        features.append(torch.Tensor(feature))
        labels.append(label)
    return features, labels


class QueryDataset(torch.utils.data.Dataset):
    def __init__(self, queries, table_stats, columns):
        super().__init__()
        self.query_data = list(zip(*preprocess_queries(queries, table_stats, columns)))

    def __getitem__(self, index):
        return self.query_data[index]

    def __len__(self):
        return len(self.query_data)


def est_mlp(train_data, test_data, table_stats, columns):
    """
    est_mlp uses MLP to produce estimated rows for train_data and test_data
    """
    train_dataset = QueryDataset(train_data, table_stats, columns)
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=1)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=False)
    train_est_rows, train_act_rows = [], []
    # YOUR CODE HERE: train procedure

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ============================ step 2/5 模型 ============================
    net = MLP(input_layers=len(columns) * 2 + 3).to(device)
    net.init_weights()
    # ============================ step 3/5 损失函数 ============================
    def loss_fn(predict_rows, actual_rows):
        est_rows = torch.abs(predict_rows)
        return torch.mean(torch.square((est_rows - actual_rows) / (est_rows + actual_rows)))

    loss_fn1 = torch.nn.MSELoss(reduce=True, size_average=True)

    # ============================ step 4/5 优化器 ============================
    # optimizer = optim.SGD(net.parameters(), lr=0.00001, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    # ============================ step 5/5 训练 ============================

    train_curve = []
    log_interval = 1000
    for epoch in range(EPOCHS):

        loss_mean = 0.
        net.train()

        for i,data in enumerate(train_loader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            if epoch == EPOCHS - 1:
                train_act_rows += list(labels.cpu().numpy())
            outputs = net(inputs)
            outputs = outputs.to(torch.float)
            if epoch == EPOCHS - 1:
                train_est_rows += list(outputs.squeeze().cpu().detach().numpy())
            labels = labels.to(torch.float)

            # backward
            optimizer.zero_grad()
            loss = loss_fn(outputs, labels)
            loss.backward()

            # update weights
            optimizer.step()

            # print train information
            loss_mean += loss.item()
            # print(loss.item())
            train_curve.append(loss.item())
            if (i + 1) % log_interval == 0:
                loss_mean = loss_mean / log_interval
                print("Training:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f}".format(
                    epoch, 100, i + 1, len(train_loader), loss_mean))
                loss_mean = 0.
        # scheduler.step()


    test_dataset = QueryDataset(test_data, table_stats, columns)
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=True, num_workers=1)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=True)
    test_est_rows, test_act_rows = [], []
    # YOUR CODE HERE: test procedure
    for i,data in enumerate(test_loader):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        test_act_rows += list(labels.cpu().numpy())
        outputs = net(inputs)
        test_est_rows += list(outputs.squeeze().cpu().detach().numpy())
    train_act_rows = [np.e**i for i in train_act_rows]
    train_est_rows = [np.e**i for i in train_est_rows]
    test_est_rows = [np.e**i for i in test_est_rows]
    test_act_rows = [np.e**i for i in test_act_rows]

    return train_est_rows, train_act_rows, test_est_rows, test_act_rows


def est_xgb(train_data, test_data, table_stats, columns):
    """
    est_xgb uses xgboost to produce estimated rows for train_data and test_data
    """
    print("estimate row counts by xgboost")
    train_x, train_y = preprocess_queries(train_data, table_stats, columns)
    train_x = [i.numpy().tolist() for i in train_x]
    train_est_rows, train_act_rows = [], []
    # YOUR CODE HERE: train procedure
    train_act_rows = [np.e**i for i in train_y]
    models = [LinearRegression(), KNeighborsRegressor(), SVR(), Ridge(), Lasso(), MLPRegressor(alpha=20),
              DecisionTreeRegressor(), ExtraTreeRegressor(), XGBRegressor(), RandomForestRegressor(),
              AdaBoostRegressor(), GradientBoostingRegressor(), BaggingRegressor()]
    models_str = ['LinearRegression', 'KNNRegressor', 'SVR', 'Ridge', 'Lasso', 'MLPRegressor', 'DecisionTree',
                  'ExtraTree', 'XGBoost', 'RandomForest', 'AdaBoost', 'GradientBoost', 'Bagging']
    # models = [KNeighborsRegressor()]
    # models_str = ['KNNRegressor']
    score_ = []

    train_x = pd.DataFrame(train_x)
    train_y = pd.DataFrame(train_y)



    test_x, test_y = preprocess_queries(test_data, table_stats, columns)
    test_est_rows, test_act_rows = [], []
    test_act_rows = [np.e**i for i in test_y]
    # YOUR CODE HERE: test procedure
    test_x = [i.numpy().tolist() for i in test_x]
    test_x = pd.DataFrame(test_x)
    test_y = pd.DataFrame(test_y)


    for name, model in zip(models_str, models):
        print('开始训练模型：' + name)
        model = model  # 建立模型
        model.fit(train_x, train_y)
        y_pred = model.predict(test_x)
        score = model.score(test_x, test_y)
        score_.append(str(score)[:5])
        print(name + ' 得分:' + str(score))
        if name == 'XGBoost':
            test_est_rows = [np.e**i for i in np.array(y_pred).flatten().tolist()]
            y_trai_pred = model.predict(train_x)
            train_est_rows = [np.e**i for i in np.array(y_trai_pred).flatten().tolist()]

    return train_est_rows, train_act_rows, test_est_rows, test_act_rows


def eval_model(model, train_data, test_data, table_stats, columns):
    if model == 'mlp':
        est_fn = est_mlp
    else:
        est_fn = est_xgb

    train_est_rows, train_act_rows, test_est_rows, test_act_rows = est_fn(train_data, test_data, table_stats, columns)

    name = f'{model}_train_{len(train_data)}'
    eval_utils.draw_act_est_figure(name, train_act_rows, train_est_rows)
    p50, p80, p90, p99 = eval_utils.cal_p_error_distribution(train_act_rows, train_est_rows)
    print(f'{name}, p50:{p50}, p80:{p80}, p90:{p90}, p99:{p99}')

    name = f'{model}_test_{len(test_data)}'
    eval_utils.draw_act_est_figure(name, test_act_rows, test_est_rows)
    p50, p80, p90, p99 = eval_utils.cal_p_error_distribution(test_act_rows, test_est_rows)
    print(f'{name}, p50:{p50}, p80:{p80}, p90:{p90}, p99:{p99}')


if __name__ == '__main__':
    stats_json_file = './data/title_stats.json'
    train_json_file = './data/query_train_20000.json'
    test_json_file = './data/query_test_5000.json'
    columns = ['kind_id', 'production_year', 'imdb_id', 'episode_of_id', 'season_nr', 'episode_nr']
    table_stats = stats.TableStats.load_from_json_file(stats_json_file, columns)
    with open(train_json_file, 'r') as f:
        train_data = json.load(f)
    with open(test_json_file, 'r') as f:
        test_data = json.load(f)

    eval_model('mlp', train_data, test_data, table_stats, columns)
    eval_model('xgb', train_data, test_data, table_stats, columns)
