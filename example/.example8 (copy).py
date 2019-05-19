#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Han Xiao <artex.xh@gmail.com> <https://hanxiao.github.io>

# NOTE: First install bert-as-service via
# $
# $ pip install bert-serving-server
# $ pip install bert-serving-client
# $

# simple similarity search on FAQ

import numpy as np
from bert_serving.client import BertClient
from termcolor import colored
from flask import Flask, request
from flask_json import FlaskJSON, as_json, JsonError
import json

app = app = Flask(__name__)

prefix_q = 'Q: '
prefix_a = 'A: '
topk = 5

with open('./QA_TravelAgancy.txt') as fp:
    questions = [v.replace(prefix_q, '').strip() for v in fp if v.strip() and v.startswith(prefix_q)]
    print('%d questions loaded, avg. len of %d' % (len(questions), np.mean([len(d.split()) for d in questions])))

with open('./QA_TravelAgancy.txt') as fp:
    answers = [v.replace(prefix_a, '').strip() for v in fp if v.strip() and v.startswith(prefix_a)]
    print('%d answers loaded, avg. len of %d' % (len(answers), np.mean([len(d.split()) for d in answers])))


bc = BertClient(port=5555, port_out=5556, check_length=False)

doc_vecs = bc.encode(questions)


@app.route('/')
@as_json
def hello_world():
    return {'message':'Hello World'}

@app.route('/gsug', methods=['POST'])
@as_json
def sendAnswers():
    data = request.form
    query = data['query']
    query_vec = bc.encode([query])[0]
    score = np.sum(query_vec * doc_vecs, axis=1)
    topk_idx = np.argsort(score)[::-1][:topk]
    response = {}
    suggestions = []
    for idx in topk_idx:
        suggestions.append(answers[idx])
    response['suggestions'] = [answer for answer in suggestions]
    for a in response['suggestions']:
        print(a)
    return json.dumps(response)


FlaskJSON(app)

if __name__ == '__main__':
    app.run()


# with BertClient(port=5555, port_out=5556, check_length=False) as bc:
#     doc_vecs = bc.encode(questions)
#     while True:
#         query = input(colored('your question: ', 'green'))
#         query_vec = bc.encode([query])[0]
#         # compute simple dot product as score
#         score = np.sum(query_vec * doc_vecs, axis=1)
#         topk_idx = np.argsort(score)[::-1][:topk]
#         print('top %d questions similar to "%s"' % (topk, colored(query, 'green')))
#         print(topk_idx)
#         for idx in topk_idx:
#             print('> %s\t%s' % (colored('%.1f' % score[idx], 'cyan'), colored(questions[idx], 'yellow')))
#             print('> %s\t%s' % (colored('%.1f' % score[idx], 'cyan'), colored(answers[idx], 'green')))
        # break
            
# while True:
#     query = input('your question: ')
#     query_vec = bc.encode([query])[0]
#     # compute simple dot product as score
#     score = np.sum(query_vec * doc_vecs, axis=1)
#     topk_idx = np.argsort(score)[::-1][:topk]
#     for idx in topk_idx:
#         print('> %s\t%s' % (score[idx], questions[idx])