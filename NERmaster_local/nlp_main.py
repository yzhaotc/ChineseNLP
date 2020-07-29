# -*- coding: utf-8 -*-

"""
flask 入口
"""
import os
import nlp_config as nc
from flaskr import create_app, loadProjContext

__author__ = 'Yuxuan'

from flask import jsonify, make_response, redirect

# 加载flask配置信息
# app = create_app('config.DevelopmentConfig')
app = create_app(nc.config['default'])
# 加载项目上下文信息
loadProjContext()

@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)

@app.errorhandler(400)
def not_found(error):
    return make_response(jsonify({'error': '400 Bad Request,参数或参数内容异常'}), 400)

@app.route('/')
def index_sf():
    # return render_template('index.html')
    return redirect('index.html')

if __name__ == '__main__':
    app.run('localhost', 5006, app, use_reloader=False)