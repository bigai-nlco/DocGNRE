# !usr/bin/env python
# -*- coding:utf-8 _*-
"""
@Author:lijunpeng@bigai.ai
 
@File: I_gpt_proposal.py
@Time: 2023/4/23 上午11:10
"""

import json
import openai
from argparse import ArgumentParser
openai.organization = "org-XXX"
openai.api_key = "sk-XXX"

def documentize(sents):
    doc = ''
    for sent in sents:
        doc += ' '.join(sent[:-1])
        doc += (sent[-1] + ' ')
    return doc[:-1]


def get_GPT4_rsp(prompt1, prompt2):
    rsp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": prompt2},
            {"role": "user", "content": prompt1},
        ]
    )
    answer = (rsp.get("choices")[0]["message"]["content"])
    response = {
        'created': rsp.get("created"),
        'model': rsp.get("model"),
        'object': rsp.get("object"),
        'usage': rsp.get("usage"),
        'id': rsp.get("id")
    }
    return answer, response


def main():
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', help='Path to the input file')
    parser.add_argument('-o', '--output', help='Path to the output file')
    args = parser.parse_args()

    rel_info = json.load(open('../resource/meta/rel_info.json'))
    train_data = json.load(open(args.input, 'r'))
    f_out = open(args.output, 'w')
    for data_index, data in enumerate(train_data):
        doc_content = documentize(data['sents'])
        vertex_list = data['vertexSet']
        entity_list = [repr(vertex[0]['name']) for vertex in vertex_list]
        entity_str = ', '.join(entity_list)
        relation_set_str = ', '.join([repr(value) for _, value in rel_info.items()])
        prompt1 = 'Context: ' + repr(doc_content) + \
                  '\nAnd entity list: [ ' + entity_str + \
                  ' ] \n relation list: [' + relation_set_str + ' ]. \nFor example, <\'Caribbean\', \'part of\', \'Atlantic\'>, \n<\'Atlantic\', \'has part\', \'Caribbean\'>'
        prompt2 = 'Please generate at least 20 triples that are considered to be true with respect to below context using only given entities and relations from the entity and relation list. Please answer using triples in form of <\'entity1\', \'relation\', \'entity2\'>. \'entity1\' and \'entity2\' are from the below entity list and \'relation\' is from the below relation list. It is true if \'entity2\' is \'relation\' of \'entity1\''

        answer, rsp = get_GPT4_rsp(prompt1, prompt2)

        data['answer'] = answer
        data['rsp'] = rsp
        f_out.write(json.dumps(data) + '\n')
    f_out.close()


if __name__ == "__main__":
    main()
