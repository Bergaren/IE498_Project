# IE498_Project


Project Report: [Link to overleaf](https://www.overleaf.com/2839924692cpzssjdybsby)

Repo from paper: [Link to repo](https://github.com/nikhilmaram/Show_and_Tell.git)

## Structure

```
data/
├── test
│   └── images
├── train
│   ├── captions_train2014.json
│   └── images
├── val
│   ├── captions_val2014.json
│   ├── images
└── vocabulary.csv

```

```
src/
├── config.py                                    <== Define hyperparams etc. 
├── dataset.py                                   <== Dataset class **
├── main.py                                      <== Training and eval loops
├── model.py                                     <== Define model architecture
├── models                                       <== Save models
└── utils                                        
    ├── coco                             
    │   ├── coco.py                              <== Read and preprocess dataset
    │   ├── __init__.py
    │   ├── license.txt
    │   ├── pycocoevalcap                        (https://github.com/tylin/coco-caption.git)
    │   │   ├── bleu                             <== Bleu evaluation code *
    │   │   │   ├── bleu.py
    │   │   │   ├── bleu_scorer.py
    │   │   │   ├── bleu_scorer.py.bak
    │   │   │   ├── __init__.py
    │   │   │   ├── LICENSE
    │   │   ├── cider                            <== Cider evaluation code *
    │   │   │   ├── cider.py
    │   │   │   ├── cider_scorer.py
    │   │   │   ├── __init__.py
    │   │   ├── eval.py                          <== Run eval algorithms *
    │   │   ├── __init__.py
    │   │   ├── meteor                           <== Meteor Eval code *
    │   │   │   ├── data
    │   │   │   │   └── paraphrase-en.gz
    │   │   │   ├── __init__.py
    │   │   │   ├── meteor-1.5.jar
    │   │   │   ├── meteor.py
    │   │   ├── readme.md
    │   │   └── tokenizer                       <== Tokenizer used by eval code *
    │   │       ├── __init__.py
    │   │       ├── ptbtokenizer.py
    │   │       └── stanford-corenlp-3.4.1.jar
    │   └── readme.md
    ├── ilsvrc_2012_mean.npy                    <== Data used for normalizing image data
    ├── misc.py                                 <== Image loader and preprocess *
    └── vocabulary.py                           <== Build the vocabulary from COCO data *

```

`*` = Code (slightly modified/adopted) from other source
`**` = Code heavily modified but based on material from other source (underlying paper repo - see link at top of README)

## Instructions

1. Download MSCOCO training and eval data and place in the folders indicated above
2. Add modules `python/2.0.1` and `java` (for special tokenizer and eval scripts).
3. Run training: `cd src && python3 main.py <new|load> train` depending on new or loaded model. If the dataset has not not been previously processed, this will be done and the resulting files/annotations will be placed in the correct locations
4. Run eval: `cd src && python3 main.py load eval`. The resulting captions are available in `data/val/results.json`

Please note that pretrained models and dataset have not been provided due to their large file sizes (several gigabytes).

