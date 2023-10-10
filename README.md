# DocGNRE
This repo contains the code used for the EMNLP 2023 paper "Semi-automatic Data Enhancement for Document-Level Relation Extraction with Distant Supervision from Large Language Models".

## Requirements
+ Python 3.8
+ Python packages
  + PyTorch 2.0+
  + transformers 4.24.0
  + openai
  + tqdm
  + wandb
  + pandas

## Dataset
The [DocRED](https://github.com/thunlp/DocRED/tree/master/data) and [Re-DocRED](https://github.com/tonytan48/Re-DocRED/tree/main/data) 
dataset can be downloaded following the instructions at corresponding links. 
``enhancement_data`` is the enhanced dataset.
The expected structure of files is:

resource \
├── DocRED \
│   ├── dev.json \
│   └── train_annotated.json \
├── enhancement_data \
│   ├── docred_train_data_enhancement.json \
│   ├── docred_train_data_enhancement_more.json \
│   ├── re_docred_test_data_enhancement.json \
│   ├── re_docred_train_data_enhancement.json \
│   └── re_docred_train_data_enhancement_more.json \
├── meta \
│   ├── rel2hypothesis.txt \
│   ├── rel2id.json \
│   └── rel_info.json \
└── Re-DocRED \
│   ├── dev_revised.json \
│   ├── test_revised.json \
│   └── train_revised.json 

## Getting Started

### Automatic Data Generation
#### GPT Results as Proposals
**STEP1** \
The ``automatic_data_generation/I_gpt_proposal.py`` script generates additional triples for each document in the original dataset. \
Example: 
```
python I_gpt_proposal.py -i ./resource/DocRED/train_annotated.json -o docred_train_gpt
```
Arguments: 
  + -i, --input: Path to the input file.
  + -o, --output: Path to the output file.

The ``automatic_data_generation/I_gpt_proposal_more.py`` script generates more additional triples through an iterative approach by feeding the previous GPT answers as input.

**STEP2** \
The ``automatic_data_generation/II_gpt_triples_postprocess.py`` script filters undesired or illegal triples. \
Example: 
```
python II_gpt_triples_postprocess.py -i docred_train_gpt -o docred_train_gpt_postprocess
```
Arguments: 
  + -i, --input: Path to the input file.
  + -o, --output: Path to the output file.

#### NLI as an Annotator
**STEP1** \
The ``automatic_data_generation/III_nli_annotator.py`` script calculates the entailment scores used for predefined relation types. \
Example: 
```
python III_nli_annotator.py -i docred_train_gpt_postprocess -o docred_train_nli
```
Arguments: 
  + -i, --input: Path to the input file.
  + -o, --output: Path to the output file.

**STEP2** \
The ``automatic_data_generation/IV_nli_score_postprocess.py`` script post-processes the entailment scores to ensure the high quality of newly produced relation triples. \
Example: 
```
python IV_nli_score_postprocess.py -origin ./resource/DocRED/train_annotated.json -i docred_train_nli -o ./resource/enhancement_data/docred_train_data_enhancement.json
```
Arguments: 
  + -origin, --origin: Path to the original file.
  + -i, --input: Path to the input file.
  + -o, --output: Path to the output file.

### Training
The codebase of this repo is extended from [DREEAM](https://github.com/YoumiMa/dreeam). 
This work mainly designs an automated annotation method, so there is basically no difference between model training and evaluation. 
Just change the training set file in the ``./scripts/run_bert_gpt.sh`` and ``./scripts/run_roberta_gpt.sh`` to complete the training. \
Run below:
```
bash scripts/run_bert_gpt.sh ${name} ${lambda} ${seed} # for BERT
bash scripts/run_roberta_gpt.sh ${name} ${lambda} ${seed} # for RoBERTa
```
where ``${name}`` is the identifier of this run displayed in wandb, 
``${lambda}`` is the scaler that controls the weight of evidence loss, 
and ``${seed}`` is the value of random seed.
### Evaluation
Make predictions on enhanced test set with the commands below:
```
bash scripts/isf_bert.sh ${name} ${model_dir} resource/enhancement_data/re_docred_test_data_enhancement # for BERT
bash scripts/isf_roberta.sh ${name} ${model_dir} resource/enhancement_data/re_docred_test_data_enhancement # for RoBERTa
```
where ``${model_dir}`` is the directory that contains the checkpoint we are going to evaluate. 
The program will generate a test file ``result.json`` in the official evaluation format. 

### Data Format
Generated input example:
```json
{
  "title": "Culrav", 
  "sents": [
      ["Culrav", "is", "a", "cultural", "festival", "of", "Motilal", "Nehru", "National", ...], 
      ["Culrav", "gives", "a", "platform", "to", "the", "students", "of", "MNNIT", ...], 
      ...],
  "vertexSet": [
      [
        {
          "sent_id": 0, 
          "type": "MISC", 
          "pos": [0, 1], 
          "name": "Culrav", 
          "global_pos": [0, 0], 
          "index": "0_0"
        }, 
        {
          "sent_id": 1, 
          "type": "MISC", 
          "pos": [0, 1], 
          "name": "Culrav", 
          "global_pos": [15, 15], 
          "index": "0_1"},
        ...],
      ...],
  "labels": [
      {
        "r": "P17",
        "h": 0, 
        "t": 3, 
        "evidence": [0, 1, 3]
      }, 
      {
        "r": "P131",
        "h": 1, 
        "t": 2, 
        "evidence": [0]
      },
      ...],
  "gpt_labels": [
      {
        "h": 0,
        "r": "P127",
        "t": 1, 
        "score": 0.758
      }, 
      {
        "h": 0,
        "r": "P276", 
        "t": 4, 
        "score": 0.662
      },
      ...]
}
```

## Citation

## Acknowledgements
The codebase of this repo is extended from [DREEAM](https://github.com/YoumiMa/dreeam)
