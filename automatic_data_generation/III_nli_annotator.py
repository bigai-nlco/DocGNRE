# !usr/bin/env python
# -*- coding:utf-8 _*-
"""
@Author:lijunpeng@bigai.ai
 
@File: III_nli_annotator.py
@Time: 2023/4/20 上午11:05
"""


import sys
import torch
import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from argparse import ArgumentParser


device = torch.device("cuda")
tokenizer = AutoTokenizer.from_pretrained("t5_xxl_true_nli_mixture")
model = AutoModelForSeq2SeqLM.from_pretrained("t5_xxl_true_nli_mixture")
model.to(device)


def get_t5_score(input_list):
    input_ids = tokenizer(input_list, return_tensors='pt', padding=True).input_ids
    input_ids = input_ids.to(device)

    decoder_input_ids1 = torch.tensor([tokenizer.pad_token_id] * len(input_list), dtype=torch.long).unsqueeze(1).to(device)
    decoder_input_ids2 = torch.tensor([tokenizer.pad_token_id, 3], dtype=torch.long).unsqueeze(0).repeat(
        len(input_list), 1).to(device)
    decoder_input_ids3 = torch.tensor([tokenizer.pad_token_id, 209], dtype=torch.long).unsqueeze(0).repeat(
        len(input_list), 1).to(device)

    with torch.no_grad():
        second_outputs = model(input_ids, decoder_input_ids=decoder_input_ids1)
        third_outpus_not_entail = model(input_ids, decoder_input_ids=decoder_input_ids2)
        third_outpus_entail = model(input_ids, decoder_input_ids=decoder_input_ids3)

        not_entail_second_logit = second_outputs.logits[:, -1, 3].cpu().tolist()
        not_entail_third_logit_632 = third_outpus_not_entail.logits[:, -1, 632].cpu().tolist()
        not_entail_third_logit_1 = third_outpus_not_entail.logits[:, -1, 1].cpu().tolist()
        entail_second_logit = second_outputs.logits[:, -1, 209].cpu().tolist()
        entail_third_logit_632 = third_outpus_entail.logits[:, -1, 632].cpu().tolist()
        entail_third_logit_1 = third_outpus_entail.logits[:, -1, 1].cpu().tolist()
    return not_entail_third_logit_632, not_entail_third_logit_1, entail_third_logit_632, entail_third_logit_1, not_entail_second_logit, entail_second_logit


def main():
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', help='Path to the input file')
    parser.add_argument('-o', '--output', help='Path to the output file')
    args = parser.parse_args()


    rel_list = []
    rel_natural_list = []
    with open('../resource/meta/rel2hypothesis.txt', 'r') as f:
        for line in f:
            line = line.strip()
            items = line.split("\t")
            if len(items) != 2:
                print("error: ", line)
                exit(0)
            rel_list.append(items[0].strip())
            rel_natural_list.append(items[1].strip())

    f_out = open(args.output, 'w')
    with open(args.input, 'r') as f:
        for line in f:
            line = line.strip()

            data_info = json.loads(line)
            new_triple_list = data_info['new_triples']
            triple2nli = {}
            for triple in new_triple_list:
                str1 = ' '.join(triple)
                input_list1 = []
                for rel_natural in rel_natural_list:
                    str2 = rel_natural.replace('sub.', triple[0])
                    str3 = str2.replace('obj.', triple[2])
                    input1 = "premise: " + str1 + ". hypothesis: " + str3 + '.'

                    input_list1.append(input1)

                # # split into multi-batches
                not_entail_third_logit_632_1, not_entail_third_logit_1_1, \
                    entail_third_logit_632_1, entail_third_logit_1_1, \
                    not_entail_second_logit_1, entail_second_logit_1 = \
                    get_t5_score(input_list1[:24])
                not_entail_third_logit_632_2, not_entail_third_logit_1_2, \
                    entail_third_logit_632_2, entail_third_logit_1_2,\
                    not_entail_second_logit_2, entail_second_logit_2 = \
                    get_t5_score(input_list1[24:48])
                not_entail_third_logit_632_3, not_entail_third_logit_1_3, \
                    entail_third_logit_632_3, entail_third_logit_1_3,\
                    not_entail_second_logit_3, entail_second_logit_3 = \
                    get_t5_score(input_list1[48:72])
                not_entail_third_logit_632_4, not_entail_third_logit_1_4, \
                    entail_third_logit_632_4, entail_third_logit_1_4,\
                    not_entail_second_logit_4, entail_second_logit_4 = \
                    get_t5_score(input_list1[72:])

                input_list2 = []
                for rel_natural in rel_natural_list:
                    str2 = rel_natural.replace('sub.', triple[2])
                    str3 = str2.replace('obj.', triple[0])
                    input2 = "premise: " + str1 + ". hypothesis: " + str3 + '.'
                    input_list2.append(input2)

                not_entail_third_logit_632_5, not_entail_third_logit_1_5, \
                    entail_third_logit_632_5, entail_third_logit_1_5,\
                    not_entail_second_logit_5, entail_second_logit_5 = \
                    get_t5_score(input_list2[:24])
                not_entail_third_logit_632_6, not_entail_third_logit_1_6, \
                    entail_third_logit_632_6, entail_third_logit_1_6,\
                    not_entail_second_logit_6, entail_second_logit_6 = \
                    get_t5_score(input_list2[24:48])
                not_entail_third_logit_632_7, not_entail_third_logit_1_7, \
                    entail_third_logit_632_7, entail_third_logit_1_7,\
                    not_entail_second_logit_7, entail_second_logit_7 = \
                    get_t5_score(input_list2[48:72])
                not_entail_third_logit_632_8, not_entail_third_logit_1_8, \
                    entail_third_logit_632_8, entail_third_logit_1_8,\
                    not_entail_second_logit_8, entail_second_logit_8 = \
                    get_t5_score(input_list2[72:])

                triple2nli['\t'.join(triple)] = {
                    'not_entail_second_logit': not_entail_second_logit_1 + not_entail_second_logit_2 +
                                                not_entail_second_logit_3 + not_entail_second_logit_4 +
                                                not_entail_second_logit_5 + not_entail_second_logit_6 +
                                                not_entail_second_logit_7 + not_entail_second_logit_8,
                    'entail_second_logit': entail_second_logit_1 + entail_second_logit_2 +
                                                entail_second_logit_3 + entail_second_logit_4 +
                                                entail_second_logit_5 + entail_second_logit_6 +
                                                entail_second_logit_7 + entail_second_logit_8,
                    'not_entail_third_logit_632': not_entail_third_logit_632_1 + not_entail_third_logit_632_2 +
                                                  not_entail_third_logit_632_3 + not_entail_third_logit_632_4 +
                                                  not_entail_third_logit_632_5 + not_entail_third_logit_632_6 +
                                                  not_entail_third_logit_632_7 + not_entail_third_logit_632_8,
                    'not_entail_third_logit_1': not_entail_third_logit_1_1 + not_entail_third_logit_1_2 +
                                                not_entail_third_logit_1_3 + not_entail_third_logit_1_4 +
                                                not_entail_third_logit_1_5 + not_entail_third_logit_1_6 +
                                                not_entail_third_logit_1_7 + not_entail_third_logit_1_8,
                    'entail_third_logit_632': entail_third_logit_632_1 + entail_third_logit_632_2 +
                                              entail_third_logit_632_3 + entail_third_logit_632_4 +
                                              entail_third_logit_632_5 + entail_third_logit_632_6 +
                                              entail_third_logit_632_7 + entail_third_logit_632_8,
                    'entail_third_logit_1': entail_third_logit_1_1 + entail_third_logit_1_2 +
                                            entail_third_logit_1_3 + entail_third_logit_1_4 +
                                            entail_third_logit_1_5 + entail_third_logit_1_6 +
                                            entail_third_logit_1_7 + entail_third_logit_1_8,
                }
            data_info['nli_result'] = triple2nli
            f_out.write(json.dumps(data_info) + '\n')



if __name__ == "__main__":
    main()
