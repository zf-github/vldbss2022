from enum import Enum
from .statistics import Histogram, MIN_VAL, MAX_VAL
import csv
from sklearn.cluster import KMeans
import numpy as np
from networkx import from_numpy_matrix, connected_components
import lab1.range_query as rq
import statistics as stats
import json


class Operation(Enum):
    CREATE_LEAF = 1
    SPLIT_COLS = 2
    SPLIT_ROWS = 3


class NodeScope:
    """
    NodeScope indicates the scope of the input dataset a node can see in our SPN.
    """
    def __init__(self, row_idxs, col_idxs):
        self.row_idxs = row_idxs
        self.col_idxs = col_idxs

    def n_rows(self):
        return len(self.row_idxs)

    def n_cols(self):
        return len(self.col_idxs)

    def __repr__(self):
        return f'NodeScope{{rows:[{self.row_idxs}), cols:{self.col_idxs}}}'

    @staticmethod
    def full_scope(dataset):
        assert(len(dataset) > 0) # cannot be empty
        row_idxs = [i for i in range(0, len(dataset))]
        col_idxs = [i for i in range(0, len(dataset[0]))]
        return NodeScope(row_idxs, col_idxs)


class LeafNode():
    """
    LeafNode represents a leaf node in our SPN.
    It uses a histogram to capture data distribution.
    """
    def __init__(self, dataset, col_names, scope):
        assert(scope.n_cols() == 1)
        self.dataset = dataset
        self.col_names = col_names
        self.scope = scope
        # create a histogram over this scope
        col_idx = scope.col_idxs[0]
        vals = []
        for row_idx in scope.row_idxs:
            vals.append(dataset[row_idx][col_idx])
        self.hist = Histogram.construct_from(vals, 5)

    def estimate(self, range_query):
        # YOUR CODE HERE
        assert(self.scope.n_cols() == 1)
        columns = range_query.column_names()
        if self.col_names[self.scope.col_idxs[0]] not in columns:
            return 1
        left = range_query.col_left[self.col_names[self.scope.col_idxs[0]]]
        right = range_query.col_right[self.col_names[self.scope.col_idxs[0]]]
        range_count = self.hist.between_row_count(left, right, None) * 1.0
        return range_count/self.scope.n_rows()


    def debug_print(self, prefix, indent):
        print('%sLeafNode: %s, %s' % (prefix, self.scope, self.hist))


class SumNode():
    """
    SumNode represents a sum node in our SPN.
    """
    def __init__(self, scope, lchild, rchild):
        self.scope = scope
        self.lchild = lchild
        self.rchild = rchild

    def estimate(self, range_query):
        # YOUR CODE HERE
        sel_left = self.lchild.estimate(range_query)
        sel_right = self.rchild.estimate(range_query)
        left_nums = self.lchild.scope.n_rows()
        right_nums = self.rchild.scope.n_rows()
        total = left_nums + right_nums
        sel = sel_left * float(left_nums)/total + sel_right * float(right_nums)/total
        return  sel

    def debug_print(self, prefix, indent):
        print('%sSumNode: %s' % (prefix, self.scope))
        self.lchild.debug_print(prefix+indent, indent)
        self.rchild.debug_print(prefix+indent, indent)


class ProductNode():
    """
    ProductNode represents a product node in our SPN.
    """
    def __init__(self, scope, lchild, rchild):
        self.scope = scope
        self.lchild = lchild
        self.rchild = rchild

    def estimate(self, range_query):
        # YOUR CODE HERE
        return self.lchild.estimate(range_query) * self.rchild.estimate(range_query)

    def debug_print(self, prefix, indent):
        print('%sProductNode: %s' % (prefix, self.scope))
        self.lchild.debug_print(prefix+indent, indent)
        self.rchild.debug_print(prefix+indent, indent)


class SPN:
    """
    SPN represents a sum-product network.
    """
    def __init__(self, root):
        self.root = root

    def estimate(self, range_query):
        """
        estimate returns the estimated cardinality of the range query on this SPN.
        The input argument range_query is supposed to be a ParsedRangeQuery structure. 
        """
        return self.root.estimate(range_query)

    def debug_print(self, prefix, indent):
        self.root.debug_print(prefix, indent)
        pass


    @staticmethod
    def construct_np_array(dataset, scope):
        """
        construct_np_array constructs a numpy array on the dataset according to the scope.
        """
        row_vals = []
        for row_idx in scope.row_idxs:
            row = []
            for col_idx in scope.col_idxs:
                row.append(dataset[row_idx][col_idx])
            row_vals.append(row)
        return np.array(row_vals)


    @staticmethod
    def split_rows(dataset, scope):
        """
        split_rows splits these rows that specified by dataset and scope into two parts.
        It uses kmeans algorithm to split these rows.
        """
        # YOUR CODE HERE
        np_array = SPN.construct_np_array(dataset, scope)
        assert (len(np_array) > 0)
        same = True
        for r in np_array:
            if not np.array_equal(r, np_array[0]):
                same = False
                break
        if same:
            n_rows = int(scope.n_rows() / 2)
            return NodeScope(scope.row_idxs[:n_rows], scope.col_idxs), NodeScope(scope.row_idxs[n_rows:],scope.col_idxs)

        kmeans = KMeans(n_clusters=2, random_state=0).fit(np_array)
        l_rows = []
        r_rows = []
        for i in range(0, scope.n_rows()):
            row_idx = scope.row_idxs[i]
            if kmeans.labels_[i] == 0:
                l_rows.append(row_idx)
            else:
                r_rows.append(row_idx)
        return NodeScope(l_rows, scope.col_idxs), NodeScope(r_rows, scope.col_idxs)

    @staticmethod
    def split_cols(dataset, scope, force):
        """
        split_cols splits these columns that specified by dataset and scope into two parts according to their correlation.
        For simplicity, we use Pearson correlation coefficients to measure their correlation.
        First, we calculate Pearson correlation of each two columns.
        And then we put columns whose correlation are large than a fixed threshold into a gorup.
        """
        assert (scope.n_cols() > 1)
        # use Pearson correlation coefficients to split cols
        np_array = SPN.construct_np_array(dataset, scope).T
        corr_matrix = np.corrcoef(np_array)
        corr_matrix = np.nan_to_num(corr_matrix)

        threshold = 0.3
        while True:
            edge = corr_matrix.copy()
            edge[edge>=threshold] = True
            edge[edge<threshold] = False
            graph = from_numpy_matrix(edge)
            groups = sorted(connected_components(graph), key=len, reverse=True)
            if len(groups) == 1: 
                # all nodes are connected so we cannot split a group of cols in this case
                if force is True:
                    threshold *= 1.5
                    continue
                else:
                    return NodeScope([], []), NodeScope([], [])
            max_group = groups[0]
            l_cols = []
            r_cols = []
            for i in range(scope.n_cols()):
                col = scope.col_idxs[i]
                if i in max_group:
                    l_cols.append(col)
                else:
                    r_cols.append(col)
            return NodeScope(scope.row_idxs, l_cols), NodeScope(scope.row_idxs, r_cols)

    @staticmethod
    def get_next_op(scope, row_batch_threshold, split_col_failed):
        """
        get_next_op returns the next operation to do when constructing a SPN.
        """
        # YOUR CODE HERE: return the next operation
        if scope.n_cols() == 1:
            # YOUR CODE HERE
            if scope.n_rows() <= row_batch_threshold:
                return Operation.CREATE_LEAF, False
            else:
                return Operation.SPLIT_ROWS, False
        if split_col_failed:
            return Operation.SPLIT_ROWS, False

        if scope.n_rows() <= row_batch_threshold:
            return Operation.SPLIT_COLS, True
        return Operation.SPLIT_COLS, False


    @staticmethod
    def construct_top_down(dataset, col_names, scope, row_batch_threshold):
        """
        construct_top_down constructs a SPN top-down.
        """
        split_col_failed = False
        split_col_force = False
        while True:
            next_op, split_col_force = SPN.get_next_op(scope, row_batch_threshold, split_col_failed)

            if next_op == Operation.SPLIT_ROWS:
                left_scope, right_scope = SPN.split_rows(dataset, scope)
                lchild = SPN.construct_top_down(dataset, col_names, left_scope, row_batch_threshold)
                rchild = SPN.construct_top_down(dataset, col_names, right_scope, row_batch_threshold)
                return SumNode(scope, lchild, rchild)

            elif next_op == Operation.SPLIT_COLS:
                left_scope, right_scope = SPN.split_cols(dataset, scope, split_col_force)
                if left_scope.n_cols() == 0 or right_scope.n_cols() == 0:
                    split_col_failed = True
                    continue
                lchild = SPN.construct_top_down(dataset, col_names, left_scope, row_batch_threshold)
                rchild = SPN.construct_top_down(dataset, col_names, right_scope, row_batch_threshold)
                return ProductNode(scope, lchild, rchild)

            elif next_op == Operation.CREATE_LEAF:
                return LeafNode(dataset, col_names, scope)


    @staticmethod
    def construct_from_dataset(dataset, col_names, row_batch_threshold):
        """
        construct_from_dataset constructs a SPN from the specified dataset.
        """
        root = SPN.construct_top_down(dataset, col_names, NodeScope.full_scope(dataset), row_batch_threshold)
        return SPN(root)


    @staticmethod
    def construct_for_imdb_title(csv_file_path, row_batch_threshold):
        """
        construct_for_imdb_title constructs a SPN from the specified csv file.
        The input csv file is supposed to be sampling data of imdb.title table.
        Not all columns of imdb.title are used to construct the SPN, while only some INT columns 
            are used: kind_id, production_year, imdb_id, episode_of_id, season_nr, episode_nr.
        """
        col_names = ['kind_id', 'production_year', 'imdb_id', 'episode_of_id', 'season_nr', 'episode_nr']
        col_offsets = [3, 4, 5, 7, 8, 9]
        dataset = []
        with open(csv_file_path) as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                if len(line) != 12: # imdb.title has 12 columns
                    continue
                row = []
                for offset in col_offsets:
                    if line[offset] == '':
                        row.append(0)
                    else:
                        row.append(int(line[offset]))
                dataset.append(row)
        return SPN.construct_from_dataset(dataset, col_names, row_batch_threshold)


def get_selectivity_by_query(query):
    csvfile = '/Users/zhangfei/Documents/projects/pycharm-projects/pytorch-projects/vldbss2022/lab1/data/title_sample_10000.csv'
    print("SPN estimation on %s" % csvfile)
    spn = SPN.construct_for_imdb_title(csvfile, 100)
    range_query = rq.ParsedRangeQuery.parse_range_query(query)
    sel = spn.estimate(range_query)
    return sel

def get_spn():
    csvfile = '/Users/zhangfei/Documents/projects/pycharm-projects/pytorch-projects/vldbss2022/lab1/data/title_sample_10000.csv'
    spn = SPN.construct_for_imdb_title(csvfile, 100)
    return spn


if __name__ == '__main__':
    stats_json_file = 'data/title_stats.json'
    train_json_file = 'data/query_train_2000.json'
    test_json_file = 'data/query_test_500.json'
    columns = ['kind_id', 'production_year', 'imdb_id', 'episode_of_id', 'season_nr', 'episode_nr']

    table_stats = stats.TableStats.load_from_json_file(stats_json_file, columns)
    with open(train_json_file, 'r') as f:
        train_data = json.load(f)
    with open(test_json_file, 'r') as f:
        test_data = json.load(f)

    est_avi, est_ebo, est_min_sel, act = [], [], [], []
    for item in test_data:
        range_query = rq.ParsedRangeQuery.parse_range_query(item['query'])
        est_avi.append(stats.AVIEstimator.estimate(range_query, table_stats) * table_stats.row_count)
        est_ebo.append(stats.ExpBackoffEstimator.estimate(range_query, table_stats) * table_stats.row_count)
        est_min_sel.append(stats.MinSelEstimator.estimate(range_query, table_stats) * table_stats.row_count)
        act.append(item['act_rows'])

    csvfile = './data/title_sample_10000.csv'
    print("SPN estimation on %s" % csvfile)
    spn = SPN.construct_for_imdb_title(csvfile, 100)

    est = []
    for item in train_data:
        range_query = rq.ParsedRangeQuery.parse_range_query(item['query'])
        sel = spn.estimate(range_query)
        est.append(sel * table_stats.row_count)
    print(est)
