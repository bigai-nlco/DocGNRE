# !usr/bin/env python
# -*- coding:utf-8 _*-
"""
@Author:lijunpeng@bigai.ai
 
@File: IV_nli_score_postprocess.py
@Time: 2023/5/29 上午10:44
"""

import os
import json
import numpy as np
from argparse import ArgumentParser


def softmax(x):
    x -= np.max(x, axis=1, keepdims=True)
    x = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
    return x


def dedump(labels):
    labels_dict = {}
    for label in labels:
        relation = label['r']
        if str(label['h']) + '\t' + relation + '\t' + str(label['t']) in labels_dict:
            pre_score = labels_dict[str(label['h']) + '\t' + relation + '\t' + str(label['t'])]
            if pre_score > 0:
                if pre_score < label['score']:
                    labels_dict[str(label['h']) + '\t' + relation + '\t' + str(label['t'])] = label['score']
            else:
                if pre_score > label['score']:
                    labels_dict[str(label['h']) + '\t' + relation + '\t' + str(label['t'])] = label['score']
        else:
            labels_dict[str(label['h']) + '\t' + relation + '\t' + str(label['t'])] = label['score']

    count = 0
    new_labels = []
    for key, score in labels_dict.items():
        h_str, rel, t_str = key.split('\t')
        label = {
            'h': int(h_str),
            'r': rel,
            't': int(t_str),
            'score': score,
        }
        if score > 0:
            count += 1
        new_labels.append(label)
    return new_labels, count


def main():
    parser = ArgumentParser()
    parser.add_argument('-origin', '--origin', help='Path to the input file')
    parser.add_argument('-i', '--input', help='Path to the input file')
    parser.add_argument('-o', '--output', help='Path to the output file')
    args = parser.parse_args()

    
    title2docred_data = {}
    docred_train_data = json.load(open(args.origin, 'r'))
    for data in docred_train_data:
        title2docred_data[data['title']] = data

    train_data = json.load(open('../resource/Re-DocRED/train_revised.json', 'r'))
    relation2head_type = {}
    relation2tail_type = {}
    for data in train_data:
        vertex_list = data['vertexSet']
        labels = data['labels']
        for label in labels:
            head = vertex_list[label['h']]
            tail = vertex_list[label['t']]
            relation = label['r']
            head_type = head[0]['type']
            tail_type = tail[0]['type']
            if relation not in relation2head_type:
                relation2head_type[relation] = set()
                relation2tail_type[relation] = set()
            relation2head_type[relation].add(head_type)
            relation2tail_type[relation].add(tail_type)
    
    rel_list = []
    with open('../resource/meta/rel2hypothesis.txt', 'r') as f:
        for line in f:
            line = line.strip()
            items = line.split("\t")
            if len(items) != 2:
                print("error: ", line)
                exit(0)
            rel_list.append(items[0].strip())


    train_data_top1_diff_reward = []
    with open(args.input, 'r') as f:
        for line in f:
            line = line.strip()
            try:
                data_info = json.loads(line)
            except:
                continue
            origin_data_info = title2docred_data[data_info['title']]
            labels = origin_data_info['labels']
            ground_truth_set = set()
            for label in labels:
                ground_truth_set.add(str(label['h']) + '\t' + str(label['r']) + '\t' + str(label['t']))

            mention2index = {}
            for index, vertex in enumerate(origin_data_info['vertexSet']):
                for entity in vertex:
                    mention2index[entity['name']] = index

            gpt_labels_diff_reward = []
            triple2nli = data_info['nli_result']
            nli_result2gpt_triple = {}

            for triple, nli_result in triple2nli.items():
                head, _, tail = triple.split('\t')
                h = mention2index[head]
                t = mention2index[tail]
                if h == t:
                    continue
                head_type = origin_data_info['vertexSet'][h][0]['type']
                tail_type = origin_data_info['vertexSet'][t][0]['type']

                entail_second_logit = np.array(nli_result['entail_second_logit']).reshape(-1, 1)
                not_entail_second_logit = np.array(nli_result['not_entail_second_logit']).reshape(-1, 1)

                not_entail_third_logit_632 = np.array(nli_result['not_entail_third_logit_632']).reshape(-1, 1)
                not_entail_third_logit_1 = np.array(nli_result['not_entail_third_logit_1']).reshape(-1, 1)

                entail_third_logit_632 = np.array(nli_result['entail_third_logit_632']).reshape(-1, 1)
                entail_third_logit_1 = np.array(nli_result['entail_third_logit_1']).reshape(-1, 1)

                score_base_second = softmax(np.concatenate((not_entail_second_logit, entail_second_logit), axis=1))
                score_base_not_entail_third = softmax(
                    np.concatenate((not_entail_third_logit_632, not_entail_third_logit_1), axis=1))
                score_base_entail_third = softmax(
                    np.concatenate((entail_third_logit_632, entail_third_logit_1), axis=1))

                not_entail_prob = score_base_second[:, 0].reshape(-1, 1) * score_base_not_entail_third
                entail_prob = score_base_second[:, 1].reshape(-1, 1) * score_base_entail_third
                score_all = np.concatenate((not_entail_prob, entail_prob), axis=1)
                diff_score_dict = {}
                score_index = 0
                for rel_index, rel in enumerate(rel_list):
                    if head_type in relation2head_type[rel] and tail_type in relation2tail_type[rel]:
                        not_entail_score = score_all[score_index][0]
                        entail_score = score_all[score_index][3]
                        diff_score_dict[str(h) + '\t' + str(rel) + '\t' + str(t)] = [entail_score - not_entail_score,
                                                                                     score_base_second[
                                                                                         score_index],
                                                                                     score_base_not_entail_third[
                                                                                         score_index],
                                                                                     score_base_entail_third[
                                                                                         score_index],
                                                                                     score_all[score_index]]
                    score_index += 1

                for rel_index, rel in enumerate(rel_list):
                    if tail_type in relation2head_type[rel] and head_type in relation2tail_type[rel]:
                        not_entail_score = score_all[score_index][0]
                        entail_score = score_all[score_index][3]
                        diff_score_dict[str(t) + '\t' + str(rel) + '\t' + str(h)] = [entail_score - not_entail_score,
                                                                                     score_base_second[
                                                                                         score_index],
                                                                                     score_base_not_entail_third[
                                                                                         score_index],
                                                                                     score_base_entail_third[
                                                                                         score_index],
                                                                                     score_all[score_index]]
                    score_index += 1
                diff_score_list = sorted(diff_score_dict.items(), key=lambda x: x[1][0], reverse=True)
                
                for positive in diff_score_list[:1]:
                    new_triple, reward = positive
                    h_str, rel, t_str = new_triple.split('\t')
                    if (reward[0] > 0.6) and (new_triple not in ground_truth_set):
                        gpt_labels_diff_reward.append({
                        'h': int(h_str),
                        't': int(t_str),
                        'r': rel,
                        'score': reward[0]})
                        nli_result2gpt_triple[new_triple] = triple
            gpt_labels, pos_count = dedump(gpt_labels_diff_reward)
            train_data_top1_diff_reward.append({
                'title': origin_data_info['title'],
                'vertexSet': origin_data_info['vertexSet'],
                'labels': origin_data_info['labels'],
                'sents': origin_data_info['sents'],
                'gpt_labels': gpt_labels,
                'nli_result2gpt_triple': nli_result2gpt_triple,
            })

    json_file = open(args.output, 'w')
    json.dump(train_data_top1_diff_reward, json_file)


if __name__ == "__main__":
    main()
