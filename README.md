# incnoise
Incremental noisy channel sentence processing model

This repository contains Python 3 code for replicating [Futrell & Levy (2017, EACL)](http://aclweb.org/anthology/E/E17/E17-1065.pdf).

```
@inproceedings{futrell2017noisy,
author={Richard Futrell and Roger Levy},
title={Noisy-context surprisal as a human sentence processing cost model},
year={2017},
booktitle={Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics: Volume 1, Long Papers},
pages={688-698},
address={Valencia, Spain}}
```

To get started: `pip3 install -r requirements.txt`.

## Figure 1
```python
import experiments
_, english = experiments.verb_forgetting_conditions(m=.5, r=.5, e=.2, s=.8)
_, german = experiments.verb_forgetting_conditions(m=.5, r=.5, e=.2, s=0)
```
The resulting numbers, divided by log2, are plotted against reading time data in `shravanplot.R`.

## Figure 2
To generate Figure 2, do:
```python
import experiments
df = experiments.verb_forgetting_grid()
experiments.verb_forgetting_plot(df)
```
This will bring up a matplotlib plot of Figure 2.

## Table 2
Table 2 uses data from the Google Syntactic N-grams. Supposing you have the ngrams at `$PATH`, use `syntngrams_depmi.py` to extract the appropriate counts:
```
$ zcat $PATH/arcs* | python3 syntngrams_depmi.py 01 01 | sort | sh uniqsum.sh > arcs_01-01
```
The script takes two arguments, `match_code` and `get_code`. `match_code` tells the script what dependency structures to filter for. For example, `01` means a head and its direct dependent. `012` means a chain of a word w_0, w_0's dependent w_1, and w_1's dependent w_2. `011` means to look at structures with one head and two dependents. `get_code` tells the script which two words to extract wordforms for. The example above looks for direct dependencies and takes the wordforms of head and dependent.

The resulting file `arcs_01_01` contains joint counts of two words in the specified dependency relationship.
Now generate the vocabulary file for the frequency cutoff:
```
$ cat arcs_01-01 | sed "s/^.* //g" | sort | uniqsum.sh > vocab
```
Then use the vocab file to calculate MI:
```
$ cat arcs_01-01 | python3 compute_mi.py vocab 10000
```



