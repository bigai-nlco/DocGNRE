# !usr/bin/env python
# -*- coding:utf-8 _*-
"""
@Author:lijunpeng@bigai.ai
 
@File: I_gpt_proposal_more.py
@Time: 2023/5/11 上午11:10
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


class Chat:
    def __init__(self, prompt_for_sys):
        self.conversation_list = [{"role": "system", "content": prompt_for_sys}]

    def ask(self, prompt):
        self.conversation_list.append({"role": "user", "content": prompt})
        rsp = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=self.conversation_list)   
        answer = rsp.get("choices")[0]["message"]["content"]
        self.conversation_list.append({"role": "assistant", "content": answer})
       
        return answer


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
        triple_list = []
        
        for label in data['labels']:
            head_name = repr(vertex_list[label['h']][0]['name'])
            tail_name = repr(vertex_list[label['t']][0]['name'])
            relation = repr(rel_info[label['r']])
            triple_list.append('<\'' + head_name + '\', \'' + relation + '\', \'' + tail_name + '\'>')
        
        if not triple_list:
            triple_list = ['<\'Caribbean\', \'is part of\', \'Atlantic\'>', '<\'Atlantic\', \'has part\', \'Caribbean\'>']
        elif len(triple_list) > 2:
            triple_list = triple_list[:2]
        prompt_for_sys = 'Context: ' + repr(doc_content) + \
                  '\nAnd entity list: [' + entity_str + \
                  ']. \nFor example, ' + '\n'.join(triple_list)
        gpt35_chat = Chat(prompt_for_sys)
        
        first_prompt = 'Please generate at least 20 triples that are considered to be true with respect to below context using only given entities from the entity list. Please answer using triples in form of <\'entity1\', \'relation\', \'entity2\'>. \'entity1\' and \'entity2\' are from the below entity list.' 
        first_answer = gpt35_chat.ask(first_prompt)

        second_prompt = 'Please generate 20 more triples using only given entities from the entity list.'
        second_answer = gpt35_chat.ask(second_prompt)
        
        third_prompt = 'Please keep generating 20 more triples using only given entities from the entity list.'
        third_answer = gpt35_chat.ask(third_prompt)
        data['answers'] = [first_answer, second_answer, third_answer]
        f_out.write(json.dumps(data) + '\n')


if __name__ == "__main__":
    main()
