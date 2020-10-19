# X-SRL Dataset and mBERT Word Aligner

Code associated with the paper **X-SRL: A Parallel Cross-lingual Semantic Role Labeling Dataset**, to be presented @EMNLP 2020.

## Citation

If you use our data, our word alignment tool or the annotation projection tool for your experiments, please cite us as follows:

```bibtex
@inproceedings{xsrl-aligner2020,
 author = {Angel Daza and Anette Frank},
 title = {X-SRL: A Parallel Cross-lingual Semantic Role Labeling Dataset},
 year = {2020},
 booktitle = {Proceedings of EMNLP},
}
```

## Datasets

Unfortunately, the English SRL data is licenced with LDC, therefore we cannot distribute it. You can obtain the English original annotations [here](https://catalog.ldc.upenn.edu/LDC2012T04). We are currently figuring out the licensing for our created dataset in the other languages.

We created SRL annotated data (in [CoNLL-U Format](https://universaldependencies.org/format.html)) for German, French and Spanish. All data was automatically translated using [DeepL](https://www.deepl.com/translator). The *train/dev* portions were automatically annotated using our projection tool and the *test* is human-validated both for translation quality and labeling. For the test sets, we also include references that map the translated sentences to their original index in the English test sets, include the assigned translation quality, and the translation as plain text. 

## Code
You can use the code in this repository in two ways:

1. As an **out-of-the-box Word Alignment Tool** for obtaining word level alignments given any parallel corpus. In this case, the system takes a pair of CoNLL Files (source-left, target-right) and outputs a file with word alignments in the widely used “Pharaoh format”: Each line is a sequence of pairs i-j, where a pair i-j indicates that the ith word (zero-indexed) of the source sentence (left) is aligned to the jth word of the target sentence (right).

2. As an **SRL Annotation Projection Tool**. In this case the system takes a pair of CoNLL Files (source with annotations and target to be annotated) and outputs a third, populated CoNLL file with the target sentences containing projected SRL labels.

Note that the code was tested for English, German, French and Spanish only; however, you can easily modify it to work with any language included in mBERT (for word alignments and basic annotation projection). Additionally, for the projection tool, if you want to include POS and syntactic information inside your generated conll target files, you need to install the pertinent SpaCy language or you can plug-in any other NLP tool that uses your preferred language.


## Installation

1. Create a new virtual environment. For example using conda:

```sh
conda create --name mbert-aligner python=3.6.3
```

2. Activate environment and move to the main folder:
```sh
source activate mbert-aligner
cd xsrl_mbert_aligner
```

3. Install Requirements:
```sh
pip install -r requirements.txt
```


Install SpaCy Languages (as required):
```sh
python -m spacy download en
python -m spacy download de
python -m spacy download fr
python -m spacy download es
```

## mBERT Word Aligner
To use the code as a multilingual alignment tool: 

### Pre-process Text

This step is necessary to tokenize plain text files (using SpaCy) and convert them into CoNLL Format:

```python
python pre_process/text_to_CoNLL.py \
    --src_file trial_data/SentsOnly_ES.txt \
    --out_file trial_data/ES_template_trial.conll \
    --lang ES
```

If you don't provide a valid language, then the script uses `split()` to tokenize the text.

### Run Word Aligner Tool 

There are three modes:
* S2T - Only *source* to *target* mBERT alignments.
* T2S - Only *target* to *source* mBERT alignments.
* **INTER** - Intersection of S2T and T2S mBERT alignments (recommended!)

```python
python word_aligner.py \
    --src_file trial_data/mini_X-SRL_Gold_EN.conll \
    --tgt_file trial_data/ES_template_trial.conll \
    --src_lang EN \
    --tgt_lang ES \
    --align_mode INTER

```

## SRL annotation projection
We developed this tool specifically for SRL annotation projection. It is based on cosine similarity of mBERT embeddings and enhanced with filters to project source SRL labels to the *closest valid word* on the target side. 

### Pre-process Text
This is optional to tokenize a text file and convert it to CoNLL Format:
```python
python pre_process/text_to_CoNLL.py \
    --src_file trial_data/SentsOnly_ES.txt \
    --out_file trial_data/ES_template_trial.syn.conll \
    --lang ES \
    --add_syntax True
```

### Converting English CoNLL-09 to CoNLL-U Format

In case you already have access to the English corpus, you can run the following script to make it compatible with our code:

```python
python pre_process/CoNLL_converter.py 
    --src_file CoNLL2009-ST-English/CoNLL2009-ST-evaluation-English.txt \
    --only_verbs True \
    --mode 09toUP
```

### Run Annotation Projection Tool

There are five main modes:
* BERT-S2T - Only *source* to *target* mBERT alignments.
* BERT-T2S - Only *target* to *source* mBERT alignments.
* BERT-INTER - Intersection of S2T and T2S mBERT alignments.
* INTER - Intersection of S2T and T2S **with SRL filters** (the highest precision but low recall).
* **S2T** - Only *source* to *target* alignments **with SRL filters** (recommended for high precision AND high recall).

This will create a new CoNLL file with the form `<ORIGINAL_TGT_FILE>.<ALIGN_MODE>.populated` containing the annotations that the algorithm projected into the target.

```python
python project_srl_annotations.py \
    --src_file trial_data/mini_X-SRL_Gold_EN.conll \
    --tgt_file trial_data/ES_template_trial.syn.conll --tgt_lang ES \
    --align_mode S2T
```

### Test vs Gold Annotations

If there are gold annotations available, turning on the *tgt_has_gold* flag allows to evaluate the tool's projections vs the gold annotations:

```python
python project_srl_annotations.py \
    --src_file trial_data/mini_X-SRL_Gold_EN.conll \
    --tgt_file trial_data/mini_X-SRL_Gold_ES.conll --tgt_lang ES \
    --align_mode S2T \
    --tgt_has_gold True
```
