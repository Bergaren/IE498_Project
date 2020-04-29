# IE498_Project

Project Report: [Link to overleaf](https://www.overleaf.com/2839924692cpzssjdybsby)

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
