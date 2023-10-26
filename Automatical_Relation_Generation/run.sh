DATA_FILE=./resource/DocRED/train_annotated

python I_gpt_proposal.py -i ${DATA_FILE}.json -o ${DATA_FILE}_gpt


python II_gpt_triples_postprocess.py -i ${DATA_FILE}_gpt -o ${DATA_FILE}_gpt_postprocess


python III_nli_annotator.py -i ${DATA_FILE}_gpt_postprocess -o ${DATA_FILE}_nli

python IV_nli_score_postprocess.py -origin ${DATA_FILE}.json -i ${DATA_FILE}_nli -o ./resource/enhancement_data/docred_train_data_enhancement.json
