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
The relevant code is in the directory `code`. To generate the numbers for Figure 1, do:
```python3
import experiments
_, english = experiments.verb_forgetting_conditions(m=.5, r=.5, e=.2, s=.8)
_, german = experiments.verb_forgetting_conditions(m=.5, r=.5, e=.2, s=0)
```
The resulting numbers, divided by log2, are plotted against reading time data in `shravanplot.R`.

## Figure 2
To generate Figure 2, do:
```python3
import experiments
df = experiments.verb_forgetting_grid()
experiments.verb_forgetting_plot(df)
```
This will bring up a matplotlib plot of Figure 2.
