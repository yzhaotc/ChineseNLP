# -*- coding: utf-8 -*-
"""
flask初始化
"""
from logging.config import dictConfig
from flask import Flask
from flask_cors import CORS
import person_ner_resource
from entity_extractor import person_model_init
from person_ner_resource import person

__author__ = 'Yuxuan'

def create_app(config_type):
    dictConfig({
        'version': 1,
        'formatters': {'default': {
            'format': '[%(asctime)s] %(name)s %(levelname)s in %(module)s %(lineno)d: %(message)s',
        }},
        'handlers': {'wsgi': {
            'class': 'logging.StreamHandler',
            'stream': 'ext://flask.logging.wsgi_errors_stream',
            'formatter': 'default'
        }},
        'root': {
            'level': 'DEBUG',
            # 'level': 'WARN',
            # 'level': 'INFO',
            'handlers': ['wsgi']
        }
    })
    # 加载flask配置信息
    app = Flask(__name__, static_folder='static', static_url_path='')
    # CORS(app, resources=r'/*',origins=['192.168.1.104'])  # r'/*' 是通配符，允许跨域请求本服务器所有的URL，"origins": '*'表示允许所有ip跨域访问本服务器的url
    CORS(app, resources={r"/*": {"origins": '*'}})  # r'/*' 是通配符，允许跨域请求本服务器所有的URL，"origins": '*'表示允许所有ip跨域访问本服务器的url
    app.config.from_object(config_type)
    app.register_blueprint(person, url_prefix='/person')
    # 初始化上下文
    ctx = app.app_context()
    ctx.push()
    return app

def loadProjContext():
    # 加载人名提取模型
    model_dir, batch_size, id2label, label_list, graph, input_ids_p, input_mask_p, pred_ids, tokenizer, sess, max_seq_length = person_model_init()
    person_ner_resource.model_dir = model_dir
    person_ner_resource.batch_size = batch_size
    person_ner_resource.id2label = id2label
    person_ner_resource.label_list = label_list
    person_ner_resource.graph = graph
    person_ner_resource.input_ids_p = input_ids_p
    person_ner_resource.input_mask_p = input_mask_p
    person_ner_resource.pred_ids = pred_ids
    person_ner_resource.tokenizer = tokenizer
    person_ner_resource.sess = sess
    person_ner_resource.max_seq_length = max_seq_length