# IE498_Project


## Prel. Results

After **10** epochs of training:

Epoch | Loss
------|------
1 | 1.841061
2 | 1.493707
3 | 1.408885
4 | 1.360092
5 | 1.326156
6 | 1.300284
7 | 1.279234
8 | 1.261251
9 | 1.245867
10 | 1.232213

metric | score
------|------
`Bleu_1` | 0.659
`Bleu_2` | 0.476
`Bleu_3` | 0.333
`Bleu_4` | 0.234


Project Report: [Link to overleaf](https://www.overleaf.com/2839924692cpzssjdybsby)

Repo from paper: [Link to repo](https://github.com/nikhilmaram/Show_and_Tell.git)

## Structure

```
├── data										
│   ├── train
│   │   └── images
│   └── val
│       └── images
├── README.md
└── src
    ├── config.py                           <== Collect hyperparams etc.
    ├── dataset.py							<== Dataloader
    ├── main.py								<== Run training/testing
    ├── model.py							<== Define Model
    └── utils
        ├── coco                            <== Dataset API
        │   ├── coco.py
        │   ├── __init__.py
        │   ├── license.txt
        │   ├── pycocoevalcap               <== Scoring/Eval
        │   │   ├── bleu
        │   │   │   ├── bleu.py
        │   │   │   ├── bleu_scorer.py
        │   │   │   ├── bleu_scorer.py.bak
        │   │   │   ├── __init__.py
        │   │   │   └── LICENSE
        │   │   ├── eval.py
        │   │   ├── __init__.py
        │   │   ├── readme.md
        │   │   └── tokenizer
        │   │       ├── __init__.py
        │   │       ├── ptbtokenizer.py
        │   │       └── stanford-corenlp-3.4.1.jar
        │   └── readme.md
        ├── ilsvrc_2012_mean.npy
        ├── misc.py
        ├── __pycache__
        │   └── vocabulary.cpython-37.pyc
        └── vocabulary.py                   <== Generate Vocab

```
