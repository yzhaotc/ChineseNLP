# -*- coding: utf-8 -*-

"""
命名实体识别接口
"""
from entity_extractor import predict

__author__ = 'Yuxuan'

from flask import Blueprint, make_response, request, current_app
from flask import jsonify
person = Blueprint('person', __name__)

model_dir, batch_size, id2label, label_list, graph, input_ids_p, input_mask_p, pred_ids, tokenizer, sess, max_seq_length = None, None, None, None, None, None, None, None, None, None, None
@person.route('/extract', methods=['POST'])

def extract():
    params = request.get_json()
    if 'text' not in params or params['text'] is None:
        return make_response(jsonify({'error': '文本长度不符合要求'}), 400)
    sentence = params['text']
    # 成句
    sentence = sentence + "。" if not sentence.endswith(("，", "。", "！", "？")) else sentence
    # 利用模型提取
    pred_rs, pred_label_result = predict(sentence, current_app.config['PERSON_LABELS'], model_dir, batch_size, id2label,
                                         label_list, graph, input_ids_p,
                                         input_mask_p,
                                         pred_ids, tokenizer, sess, max_seq_length)
    print(sentence)
    return jsonify(pred_rs)

if __name__ == '__main__':
    pass