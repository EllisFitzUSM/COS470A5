from ranx import Run, Qrels, evaluate, compare
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os

evaluation_folder = r'.\evaluate_results'
os.makedirs(evaluation_folder, exist_ok=True)
def main():
	argparser = argparse.ArgumentParser()
	argparser.add_argument('qrel')
	argparser.add_argument('runs', nargs='+')
	argparser.add_argument('-m', '--metrics', nargs='+', default=['ndcg@5', 'ndcg@10', 'precision@5',  'precision@10', 'map', 'bpref', 'mrr'])
	args = argparser.parse_args()

	qrel = Qrels.from_file(args.qrel, kind='trec')
	runs = list(Run.from_file(run_path, kind='trec', name=run_path.split('.')[0]) for run_path in args.runs)
	report = compare(qrels=qrel, runs=runs, metrics=args.metrics, max_p=0.01)
	with open(os.path.join(evaluation_folder, 'report_test.txt'), 'w', encoding='utf-8') as f:
		f.write(str(report))
		f.write(report.to_latex())

	for run in runs:
		run_dict = run.to_dict()
		p10_run = Run.from_dict(run_dict)
		evaluate(qrel, run, metrics=args.metrics, return_mean=True)
		print(run.mean_scores)
		evaluate(qrel, p10_run, metrics='precision@10', return_mean=False)
		df = pd.DataFrame(list(p10_run.scores['precision@10'].items()), columns=['qID', 'precision@10'])
		df.sort_values(by=['precision@10'], ascending=False, inplace=True)
		print(df)
		df.plot(x='qID',
				y='precision@10',
				kind='bar',
				title=f'Precision@5 scores for {run.name}'
		)

		plt.title(f'Precision@10 scores for {run.name}')
		plt.xlabel('qID')
		plt.ylabel('Precision@10')
		plt.show()

if __name__ == '__main__':
	main()