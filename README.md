# BM25-Okapi + cross-encoder/ms-marco-MiniLM-L-6-v2

### Author: Ellis Fitzgerald


~| Model                               |     NDCG@5 | NDCG@10   |   P@5 | P@10   | MAP     |  BPref  |  MRR
| :-------- | :-------- | :-------- |:-------- |:-------- |:-------- |:-------- |:-------- |:-------- |
a    | results\res_BM25_Cross_ft_ReRank_test  |   0.466 | 0.478ᵇ   |  0.452 | 0.310ᵇ | 0.401ᵇ   |   nan | 0.702
b    | results\res_BM25_test                  |   0.413 | 0.420    | 0.388  | 0.254  | 0.346    |   nan | 0.702

This Information-Retrieval repository uses PyTerrier's (wrapper for Terrier) BM25-Okapi implementation, re-ranked by an MS-MARCO trained Cross-Encoder from Sentence-Transformers (SBERT). The Cross-Encoder has been fine-tuned with Topic-Answer pairings, from `topics_1.json` and QRELs for validation seen in `qrel_split_e4`




## Installation

After cloning the repository, to get the necessary dependency run this line:

```bash
  pip install -r requirements.txt
```
    
## How to Run

There are three files to run:

- `retrieval_script.py`
- `train_script.py`
- `evaluate_script.py`

Only the first two will be covered here, as evaluation is more self explanatory.
Retrieval and Training **require** two (2) positional parameters, the rest are *"optional"*

```
Answers.json topics_1.json ... topics_N.json
```
Only one (1) Answers.json can be supplied, but any (N) many query files can be supplied. As long as they are in order/sequence as the results files will be ordered in the same way.

### retrieval_script.py

Optional parameters/flags (all have defaults):

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `-test`    | `string` | Path to a train qrel from a training split. |
| `-name`    | `string` | Custom run name to be appended to all result files. |
| `-pti`    | `string` | Path to PyTerrier index. |
| `-ft`    | `string` | Path to checkpoint/local model (fine-tuned). |
| `-sym` | `string` | If the query and answer dataset is symmetrical (UNUSED). |
| `-res`    | `string` | Path to a directory for results to be saved. |

### train_script.py

Optional parameters/flags (all have defaults)

| Parameter | Type     | Description                | 
| :-------- | :------- | :------------------------- |
| `-save`    | `string` | Path to save fine-tuned Cross-Encoder. |
| `-model`    | `string` | Name of Cross-Encoder model to download from HuggingFace. |
| `-q` or `-qrel`    | `string` |  Path to qrel_1.tsv |
| `-qs` or `-qrel_splits`    | 3 `string` | Train, Evaluation, and Test QREL |
| `-qsp`    | `string` | Directory to save QREL splits (from qrel_1.tsv) |
| `-e`    | `int` | Path to a directory for results to be saved. |


## License

[MIT](https://choosealicense.com/licenses/mit/)
