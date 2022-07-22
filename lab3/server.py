from http.server import HTTPServer, BaseHTTPRequestHandler
import json
from lab1.learn_from_data import get_selectivity_by_query, get_spn
import lab1.range_query as rq
import torch
from lab2.cost_learning import YourModel
from lab2.plan import Plan, Operator
from lab2.cost_learning import PlanFeatureCollector

host = ('localhost', 8888)

spn = get_spn()
lab2_model = torch.load('/Users/zhangfei/Documents/projects/pycharm-projects/pytorch-projects/vldbss2022/lab2/doc/lab2.pkl')

class Resquest(BaseHTTPRequestHandler):

    def handle_cardinality_estimate(self, req_data):
        # YOUR CODE HERE: use your model in lab1
        print("cardinality_estimate post_data: " + str(req_data))
        query = "select * from imdb.title where " + str(req_data).replace("'", '')[1:]
        print('query:', query)
        range_query = rq.ParsedRangeQuery.parse_range_query(query)
        sel = spn.estimate(range_query)
        print('sel:', sel)
        return {"selectivity": sel, "err_msg": ""} # return the selectivity

    def parse_plan_from_str(self, plan_str):
        plan_obj = json.loads(plan_str)
        op = Operator(plan_obj['id'], plan_obj['est_rows'], 0, 0, 0, 0, 0, 0, 0, 0)
        if plan_obj['children']:
            for child in plan_obj['children']:
                op.children.append((self.parse_plan_from_str(json.dumps(child))))
        return op


    def handle_cost_estimate(self, req_data):
        # YOUR CODE HERE: use your model in lab2
        print("cost_estimate post_data: " + str(req_data))
        plan_str = str(req_data).replace("'", '')[1:]
        plan = self.parse_plan_from_str(plan_str)
        # p = Plan.parse_plan('',plan)
        plan_feature_collect = PlanFeatureCollector()
        feature = plan_feature_collect.walk_operator_tree(plan)
        if len(feature) < 264:
            feature += [0] * (264 - len(feature))
        cost = lab2_model(torch.tensor(feature, dtype=torch.float)).tolist()[0]
        print('cost:', cost)
        return {"cost": cost, "err_msg": ""} # return the cost


    def handle_cost_estimate2(self, req_data):
        # YOUR CODE HERE: use your model in lab2
        print("cost_estimate post_data: " + str(req_data))
        plan = req_data
        p = Plan.parse_plan('',plan)
        plan_feature_collect = PlanFeatureCollector()
        feature = plan_feature_collect.walk_operator_tree(p.root)
        if len(feature) < 264:
            feature += [0] * (264 - len(feature))
        cost = lab2_model(torch.tensor(feature, dtype=torch.float)).tolist()[0]
        print('cost:', cost)
        return {"cost": cost, "err_msg": ""} # return the cost

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        req_data = self.rfile.read(content_length)
        resp_data = ""
        if self.path == "/cardinality":
            resp_data = self.handle_cardinality_estimate(req_data)
        elif self.path == "/cost":
            resp_data = self.handle_cost_estimate(req_data)

        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(resp_data).encode())


if __name__ == '__main__':
    server = HTTPServer(host, Resquest)
    print("Starting server, listen at: %s:%s" % host)
    server.serve_forever()

    # plan = [
    #   "id\testRows\testCost\tactRows\ttask\taccess object\texecution info\toperator info\tmemory\tdisk",
    #   "Projection_6\t97039.00\t1540822569.21\t97039\troot\t\ttime:1.82s, loops:96, Concurrency:1\timdb.title.production_year, imdb.title.kind_id, row_size: 16\t47.5 KB\tN/A",
    #   "└─Sort_7\t97039.00\t1537911393.21\t97039\troot\t\ttime:1.82s, loops:96\timdb.title.season_nr, row_size: 48\t3.19 MB\t0 Bytes",
    #   "  └─IndexReader_11\t97039.00\t1489684047.38\t97039\troot\t\ttime:1.8s, loops:114, cop_task: {num: 3, max: 709.3ms, min: 493.9ms, avg: 597.5ms, p95: 709.3ms, max_proc_keys: 925477, p95_proc_keys: 925477, tot_proc: 1.77s, tot_wait: 4ms, rpc_num: 3, rpc_time: 1.79s, copr_cache: disabled}\tindex:Selection_10, row_size: 48\t1.89 MB\tN/A",
    #   "    └─Selection_10\t97039.00\t1480222724.88\t97039\tcop[tikv]\t\ttikv_task:{proc max:676ms, min:489ms, p80:676ms, p95:676ms, iters:2482, tasks:3}, scan_detail: {total_process_keys: 2528312, total_process_keys_size: 161811968, total_keys: 2528318, rocksdb: {delete_skipped_count: 0, key_skipped_count: 2528315, block: {cache_hit_count: 1042, read_count: 504, read_byte: 31.4 MB}}}\teq(imdb.title.kind_id, 3), row_size: 48\tN/A\tN/A",
    #   "      └─IndexFullScan_9\t2528312.00\t1404373364.88\t2528312\tcop[tikv]\ttable:title, index:idx1(production_year, kind_id, season_nr)\ttikv_task:{proc max:651ms, min:459ms, p80:651ms, p95:651ms, iters:2482, tasks:3}\tkeep order:false, row_size: 48\tN/A\tN/A"
    # ]
    # request = Resquest()
    # request.handle_cost_estimate2(plan)

