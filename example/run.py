#!/usr/bin/env python3

import numpy as np
from bert_serving.client import BertClient
from termcolor import colored
from flask import Flask, request
from flask_json import FlaskJSON, as_json, JsonError
from flask_cors import CORS
import json
from dotenv import load_dotenv
import os 

dir_path = os.path.dirname(os.path.realpath(__file__))
dotenv_path = dir_path + "/app.env"

load_dotenv(dotenv_path)

app = Flask(__name__)
CORS(app)

prefix_q = 'Q: '
prefix_a = 'A: '
topk = 3

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
    data = request.args.get('query')
    query = data
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
    return (response)


FlaskJSON(app)

if __name__ == '__main__':
    app.run(port=5557, host='0.0.0.0')

