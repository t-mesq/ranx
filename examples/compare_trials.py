import numpy as np

from ranx import Qrels, Run, compare
from ranx_extensions import compare_trials

qs = 500

q_ids = [f'q_{i}' for i in range(qs)]
d_ids = [f'd_{i}' for i in range(qs)]
ds_ids = [d_ids for _ in range(qs)]
d_scores = []

qrels = Qrels()
qrels.add_multi(q_ids, list(map(lambda x: [x], d_ids)), [[1] for _ in d_ids])

run_1_1 = Run()
run_1_1.add_multi(q_ids, ds_ids, np.random.rand(qs, qs))
run_1_1.name = 'run_1_1'
run_1_2 = Run()
run_1_2.add_multi(q_ids, ds_ids, np.random.rand(qs, qs))
run_1_2.name = 'run_1_2'
run_2_1 = Run()
run_2_1.add_multi(q_ids, ds_ids, np.random.rand(qs, qs))
run_2_1.name = 'run_2_1'
run_2_2 = Run()
run_2_2.add_multi(q_ids, ds_ids, np.random.rand(qs, qs))
run_2_2.name = 'run_2_2'

qrels = Qrels.from_file("data/qrels.txt")
run_1_1 = Run.from_file("data/run_1.txt")
run_1_2 = Run.from_file("data/run_2.txt")
run_2_1 = Run.from_file("data/run_3.txt")
run_2_2 = Run.from_file("data/run_4.txt")

# for compute in ('full', 'full_avg', 'random', 'random_avg'):
#     report = compare_trials(qrels, [[run_1_1], [run_1_2], [run_2_1], [run_2_2]], ['mrr@10', 'recall@3'], 1000, 0.01, compute=compute)
#     print(f'Single trial sanity check without trials with {compute} compute:', report, dict(report.comparisons), sep='\n')

for compute in ('avg', 'full', 'full_avg', 'random', 'random_avg'):
    report = compare_trials(qrels, [[run_1_1, run_1_2], [run_2_1, run_2_2]], ['mrr@10', 'recall@3'], 10, 0.01, compute=compute)
    print(f'Without trials with {compute} compute:', report.to_table(1, True), dict(report.comparisons), sep='\n')

report = compare(qrels, [run_1_1, run_1_2, run_2_1, run_2_2], ['mrr@10', 'recall@3'], 1000, 0.01)
print('With trials:', report, dict(report.comparisons), sep='\n')

