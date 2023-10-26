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

## Datasets

### Original DocRED and Re-DocRED
The [DocRED](https://github.com/thunlp/DocRED/tree/master/data) and [Re-DocRED](https://github.com/tonytan48/Re-DocRED/tree/main/data) 
dataset can be downloaded following the instructions at the corresponding links. 

### DocGNRE
Our enhanced dataset can be obtained with the following command:
```commandline
wget https://bigai-nlco.s3.ap-southeast-1.amazonaws.com/DocGNRE/enhancement_data.zip
```
We provide an enhanced test set (in "enhancement_data/re_docred_test_data_enhancement.json") after manually refining. We also provide four training datasets enhanced by our distant annotations.



## Automatic Relation Generation

You can run **one command to automatically generate distantly enhanced datasets**.
```commandline
bash Automatical_Relation_Generation/run.sh
```

### GPT Results as Proposals

**STEP1** \
The ``Automatical_Relation_Generation/I_gpt_proposal.py`` script generates additional triples for each document in the original dataset. \
This code requires OpenAI's model APIs.  Accessing the API requires an API key, which you can obtain by creating an account and going to the official website. \
Example to run (``cd Automatical_Relation_Generation``): 
```
python I_gpt_proposal.py -i ${input_file_path} -o ${output_file_path}
```
Arguments: 
  + -i, --input: Path to the input file, such as the train file path of Re-DocRED.
  + -o, --output: Path to the output file.

The ``Automatical_Relation_Generation/I_gpt_proposal_more.py`` script generates more additional triples through an iterative approach by feeding the previous GPT answers as input.

**STEP2** \
The ``Automatical_Relation_Generation/II_gpt_triples_postprocess.py`` script filters undesired or illegal triples. \
Example to run: 
```
python II_gpt_triples_postprocess.py -i ${step1_output_file_path} -o ${output_file_path}
```
Arguments: 
  + -i, --input: Path to the input file.
  + -o, --output: Path to the output file.


### NLI as an Annotator

**STEP3** \
The ``Automatical_Relation_Generation/III_nli_annotator.py`` script calculates the entailment scores used for predefined relation types. \
Example to run: 
```
python III_nli_annotator.py -i ${step2_output_file_path} -o ${output_file_path}
```
Arguments: 
  + -i, --input: Path to the input file.
  + -o, --output: Path to the output file.

**STEP4** \
The ``Automatical_Relation_Generation/IV_nli_score_postprocess.py`` script supplements relations according to entailment scores to ensure the high quality of newly added triples. \
Example to run: 
```
python IV_nli_score_postprocess.py -origin ${step1_input_file_path} -i ${step3_output_file_path} -o ${output_file_path}
```
Arguments: 
  + -origin, --origin: Path to the original file.
  + -i, --input: Path to the input file.
  + -o, --output: Path to the output file.



## DocRE Models

### Training
The codebase of this repo is extended from [DREEAM](https://github.com/YoumiMa/dreeam). 
This work mainly designs an automated annotation method, so there is basically no difference between model training and evaluation. 
Just change the training set file in the ``DocRE/scripts/run_bert_gpt.sh`` and ``DocRE/scripts/run_roberta_gpt.sh`` to complete the training. \
Run below:
```
bash scripts/run_bert_gpt.sh ${name} ${lambda} ${seed} # for BERT
bash scripts/run_roberta_gpt.sh ${name} ${lambda} ${seed} # for RoBERTa
```
where ``${name}`` is the identifier of this run displayed in wandb, 
``${lambda}`` is the scaler that controls the weight of evidence loss, 
and ``${seed}`` is the value of random seed.

### Evaluation
Make predictions on the enhanced test set with the commands below:
```
bash DocRE/scripts/isf_bert.sh ${name} ${model_dir} ${test_file_path} # for BERT
bash DocRE/scripts/isf_roberta.sh ${name} ${model_dir} ${test_file_path} # for RoBERTa
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
The codebase of this repo is extended from [DREEAM](https://github.com/YoumiMa/dreeam).
