# !usr/bin/env python
# -*- coding:utf-8 _*-
"""
@Author:lijunpeng@bigai.ai
 
@File: II_gpt_triples_postprocess.py
@Time: 2023/6/1 下午2:05
"""


import json
import re
from argparse import ArgumentParser
strinfo1 = re.compile(r'<.*>')


def main():
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', help='Path to the input file')
    parser.add_argument('-o', '--output', help='Path to the output file')
    args = parser.parse_args()


    f_out = open(args.output, 'w')
    count = 0
    with open(args.input, 'r') as f:
        for line in f:
            line = line.strip()
            try:
                data_info = json.loads(line)
            except:
                continue
            labels = data_info['labels']
            ground_truth_set = set()
            for label in labels:
                ground_truth_set.add(str(label['h']) + '\t' + str(label['r']) + '\t' + str(label['t']))

            mention2index = {}
            for index, vertex in enumerate(data_info['vertexSet']):
                for entity in vertex:
                    mention2index[entity['name']] = index

            answer_list = [data_info['answer']]
            triple_dict = {}
            for answer in answer_list:
                result = strinfo1.findall(answer)
                for triple in result:
                    triple = triple.replace('\\', '')
                    triple = triple[1:-1].strip()
                    if '"' in triple:
                        triple = triple.replace('\"', '\'')
                    items = triple.split('\', \'')
                    if len(items) != 3:
                        continue
                    head, relation, tail = items
                    head = head.strip()[1:].strip()
                    relation = relation.strip().strip()
                    tail = tail.strip()[:-1].strip()
                    if head in mention2index and tail in mention2index:
                        triple_dict[':::'.join([head, relation, tail])] = 0
            new_triple_list = []
            for triple in triple_dict.keys():
                head, relation, tail = triple.split(':::')
                h = mention2index[head]
                t = mention2index[tail]
                if str(h) + '\t' + relation + '\t' + str(t) not in ground_truth_set:
                    new_triple_list.append([head, relation, tail])
            count += len(new_triple_list)
            data_info['new_triples'] = new_triple_list
            f_out.write(json.dumps(data_info) + '\n')
    f_out.close()
    print(count)


if __name__ == "__main__":
    main()
