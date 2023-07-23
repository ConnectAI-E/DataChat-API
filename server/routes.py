import os
import asyncio
import logging
import json
import base64
import queue
import requests
import threading
from functools import partial
from uuid import uuid4
from time import time
from urllib.parse import quote
from flask import request, session, jsonify, Response, copy_current_request_context, redirect, make_response, send_file
from app import app
from models import (
    ObjID, User, Collection, Documents, Embedding,
    get_user,
    save_user,
    get_collections,
    get_collection_by_id,
    get_collection_id_by_hash,
    get_hash_by_collection_id,
    get_data_by_hash,
    save_collection,
    update_collection_by_id,
    delete_collection_by_id,
    get_documents_by_collection_id,
    remove_document_by_id,
    query_by_collection_id,
    chat_on_collection,
    get_bot_list,
    get_bot_by_hash,
    create_bot,
    update_bot_by_collection_id_and_action,
)
from celery_app import embed_documents, get_status_by_id
from sse import ServerSentEvents


class InternalError(Exception): pass
class PermissionDenied(Exception): pass
class NeedAuth(Exception): pass


def create_access_token(user):
    extra = user.extra
    expires = extra.get('permission').get('expires', 0)
    privilege = extra.get('permission').get('has_privilege', False)
    app.logger.debug("create_access_token %r expires %r time %r", user.extra, expires, time())
    if privilege and expires > time():
        return session.sid, int(expires)
    raise PermissionDenied()


@app.after_request
def after_request_callback(response):
    app.logger.info(
        "%s - %s %s %s %s %sms",
        request.remote_addr,
        request.method,
        request.path,
        response.status_code,
        response.content_length,
        int((time() - request.environ['REQUEST_TIME']) * 1000),
    )
    return response


@app.before_request 
def before_request_callback(): 
    request.environ['REQUEST_TIME'] = time()
    if request.path in [
        '/api/access_token',
        '/api/login', '/login', '/api/code2session',
        '/', '/favicon.ico',
    ]:
        return
    if '/embed' in request.path and '/chat/completions' in request.path:
        # 这个接口不使用session校验，而是通过hash判断是否可用
        return
    access_token = session.get('access_token', '')
    expired = session.get('expired', 0)
    user_id = session.get('user_id', '')
    if access_token and user_id:
        if expired > time():
            pass
        else:
            raise PermissionDenied()
    else:
        raise NeedAuth()
        # return jsonify({'code': -1, 'msg': 'auth required'})

# 这里的几个页面是模拟接入站点的接口: /login, /api/code2session
@app.route('/login', methods=['GET', 'POST'])
def login_form():
    # 模拟客户的登录页面，
    if request.method == 'GET':
        return '''
<h1>登录</h1>
<form action="/login" method="post">
  <input name="name" /><br />
  <input name="passwd" type="password" /><br />
  <button type="submit">登录</button>
</form>
    '''
    elif request.method == 'POST':
        name = request.form.get('name')
        passwd = request.form.get('passwd')
        app.logger.info("debug %r", (name, passwd))
        # TODO 这里模拟登录，不校验用户名密码，只要能
        # TODO 后面需要完善注册登录逻辑
        user = {
            'name': name,
            'openid': base64.urlsafe_b64encode(name.encode()).decode(),
            'permission': {
                'has_privilege': True,
                'expires': time() + 100,
                # TODO
                # 'collection_size': 10,
                # 'bot_size': 1,
            }
        }
        code = base64.b64encode(json.dumps(user).encode()).decode()
        return redirect('{}/api/login?code={}'.format(app.config['DOMAIN'], code))

@app.route('/favicon.ico', methods=['GET'])
def faviconico():
    return ''

@app.route('/', methods=['GET'])
def home():
    return '<h1>首页</h1><a href="/api/login">登录</a>'

@app.route('/api/code2session', methods=['GET'])
def code2session():
    # 模拟客户的code2session接口
    code = request.args.get('code', default='', type=str)
    user = json.loads(base64.urlsafe_b64decode(code).decode())
    app.logger.debug('user %r', user)
    return jsonify({'data': user})


# 以下是自己的url
@app.route('/api/login', methods=['GET'])
def login_check():
    # 如果没有权限，
    # user_id = session.get('user_id', '')
    # if user_id:
    #     return redirect('/api/login?code={}'.format(code))
    code = request.args.get('code', default='', type=str)
    if not code:
        # 这里使用配置的站点的登录url
        return redirect(app.config['SYSTEM_LOGIN_URL'])

    user_info = requests.get('{}?code={}'.format(
        app.config['SYSTEM_URL'], code,
    )).json()

    assert 'data' in user_info and 'openid' in user_info['data'], '获取用户信息失败'
    user = save_user(**user_info['data'])

    access_token, expired = create_access_token(user)
    # set session
    session['access_token'] = access_token
    session['expired'] = expired
    session['openid'] = user.openid
    session['user_id'] = str(user.id)

    # return redirect('/')
    # 使用html进行跳转
    resp = make_response('<meta http-equiv="refresh" content="0;url={}/">'.format(app.config['DOMAIN']))
    resp.set_cookie("__sid__", session.sid, max_age=86400)
    app.logger.info("session %r", session)
    # 登录成功，返回前端首页
    return resp


@app.route('/api/access_token', methods=['GET'])
def get_access_token():
    code = request.args.get('code', default='', type=str)
    # TODO mock
    if code == 'JhHogaYEJId1lWLN':
        user_info = {'data': {'openid': 'JhHogaYEJId1lWLN', 'name': 'mock user'}}
    else:
        # user_info = requests.get('{}/api/code2session?code={}'.format(
        user_info = requests.get('{}?code={}'.format(
            app.config['SYSTEM_URL'], code,
        )).json()

    assert 'data' in user_info and 'openid' in user_info['data'], '获取用户信息失败'
    user = save_user(**user_info['data'])

    access_token, expired = create_access_token(user)
    # set session
    session['access_token'] = access_token
    session['expired'] = expired
    session['openid'] = user.openid
    session['user_id'] = str(user.id)
    app.logger.info("session %r", session)

    return jsonify({'code': 0, 'msg': 'success', 'access_token': access_token, 'expired': expired})


@app.route('/api/account', methods=['GET'])
def get_account():
    user = get_user(session.get('user_id', ''))
    return jsonify({
        'code': 0,
        'msg': 'success',
        'data': {
            'id': user.id,
            'name': user.name,
            'openid': user.openid,
        },
    })


@app.route('/api/collection', methods=['GET'])
def api_collections():
    page = request.args.get('page', default=1, type=int)
    size = request.args.get('size', default=20, type=int)
    user_id = session.get('user_id', '')
    collections, total = get_collections(user_id, page, size)

    return jsonify({
        'code': 0,
        'msg': 'success',
        'data': [{
            'id': collection.id,
            'name': collection.name,
            'description': collection.description,
            'created': int(collection.created.timestamp() * 1000),
        } for collection in collections],
        'total': total,
    })


@app.route('/api/collection', methods=['POST'])
def api_save_collection():
    user_id = session.get('user_id', '')
    name = request.json.get('name')
    description = request.json.get('description')
    app.logger.info("debug %r", [name, description])
    collection_id = save_collection(user_id, name, description)

    return jsonify({
        'code': 0,
        'msg': 'success',
        'data': {
            'id': collection_id,
            'collection_id': collection_id,
        },
    })


@app.route('/api/collection/<collection_id>', methods=['GET'])
def api_collection_by_id(collection_id):
    user_id = session.get('user_id', '')
    collection = get_collection_by_id(user_id, collection_id)
    assert collection, '找不到知识库或者没有权限'

    return jsonify({
        'code': 0,
        'msg': 'success',
        'data': {
            'id': collection.id,
            'name': collection.name,
            'description': collection.description,
            'created': int(collection.created.timestamp() * 1000),
        },
    })


@app.route('/api/collection/<collection_id>', methods=['PUT'])
def api_update_collection_by_id(collection_id):
    user_id = session.get('user_id', '')
    name = request.json.get('name')
    description = request.json.get('description')
    update_collection_by_id(user_id, collection_id, name, description)

    return jsonify({
        'code': 0,
        'msg': 'success',
    })


@app.route('/api/collection/<collection_id>', methods=['DELETE'])
def api_delete_collection_by_id(collection_id):
    user_id = session.get('user_id', '')
    delete_collection_by_id(user_id, collection_id)

    return jsonify({
        'code': 0,
        'msg': 'success',
    })


@app.route('/api/collection/<collection_id>/documents', methods=['GET'])
def api_get_documents_by_collection_id(collection_id):
    page = request.args.get('page', default=1, type=int)
    size = request.args.get('size', default=20, type=int)
    user_id = session.get('user_id', '')
    documents, total = get_documents_by_collection_id(user_id, collection_id, page, size)

    return jsonify({
        'code': 0,
        'msg': 'success',
        'data': [{
            'id': document.id,
            'name': document.name,
            'path': document.path,
            'type': document.type,
            'created': int(document.created.timestamp() * 1000),
        } for document in documents],
        'total': total,
    })


@app.route('/api/collection/<collection_id>/document/<document_id>', methods=['DELETE'])
def api_remove_document_by_id(collection_id, document_id):
    user_id = session.get('user_id', '')
    remove_document_by_id(user_id, collection_id, document_id)

    return jsonify({
        'code': 0,
        'msg': 'success',
    })


@app.route('/api/collection/<collection_id>/documents', methods=['POST'])
def api_embed_documents(collection_id):
    fileName = request.json.get('fileName')
    fileUrl = request.json.get('fileUrl')
    fileType = request.json.get('fileType')
    user_id = session.get('user_id', '')
    collection = get_collection_by_id(user_id, collection_id)
    assert collection, '找不到知识库或者没有权限'
    # isopenai=False
    task = embed_documents.delay(fileUrl, fileType, fileName, collection_id, False)

    return jsonify({
        'code': 0,
        'msg': 'success',
        'data': {
            'task_id': task.id,
        },
    })


@app.route('/api/collection/<collection_id>/task/<task_id>', methods=['GET'])
def api_get_task_by_id(collection_id, task_id):
    user_id = session.get('user_id', '')
    collection = get_collection_by_id(user_id, collection_id)
    assert collection, '找不到知识库或者没有权限'

    task = get_status_by_id(task_id)

    return jsonify({
        'code': 0,
        'msg': 'success',
        'data': {
            'task_id': task.id,
            'status': task.status,
            'result': task.result if isinstance(task.result, list) else str(task.result),
        },
    })


@app.route('/api/collection/<collection_id>/query', methods=['GET'])
def api_query_by_collection_id(collection_id):
    q = request.args.get('q', default='', type=str)
    page = request.args.get('page', default=1, type=int)
    size = request.args.get('size', default=20, type=int)
    user_id = session.get('user_id', '')
    collection = get_collection_by_id(user_id, collection_id)
    assert collection, '找不到知识库或者没有权限'

    documents, total = query_by_collection_id(collection_id, q, page, size)

    app.logger.info("%r %r", documents, total)
    app.logger.info("debug Documents %r", [(d.document, distance) for d, distance in documents])
    return jsonify({
        'code': 0,
        'msg': 'success',
        'data': [{
            'document_id': document.document_id,
            'document_name': document.document_name,
            'document': document.document,
            'distance': distance,
        } for document, distance in documents],
        'total': total,
    })


class ThreadTask(threading.Thread):
     def __init__(self, main, *args, **kwargs):
         super(ThreadTask, self).__init__()
         self.main = partial(main, *args, **kwargs)
 
     def run(self):
         return self.main()


@app.route('/api/collection/<collection_id>/openai/deployments/<deployment_name>/chat/completions', methods=['POST'])
def azure_chat_on_collection(collection_id, deployment_name):
    data = request.json
    # user_id = session.get('user_id', '')
    # collection = get_collection_by_id(user_id, collection_id)
    # assert collection, '找不到知识库或者没有权限'

    sse = ServerSentEvents()

    request.environ['STARTED'] = False
    def on_llm_new_token(token):
        if not request.environ['STARTED']:
            sse.send('assistant', event='role')
            app.logger.info("send role")
            request.environ['STARTED'] = True

        # app.logger.info("streaming on_llm_new_token %r", token)
        sse.send(token)

    @copy_current_request_context
    def main():
        result = chat_on_collection(
            collection_id, deployment_name,
            on_llm_new_token=on_llm_new_token,
            **data,
        )
        app.logger.info("%r", result)
        sse.send(None, event='end')
        return result

    # stream模式
    if data.get('stream'):
        t = ThreadTask(main)
        t.start()
        return sse.response()

    result = main()
    return jsonify({
        'id': uuid4(),
        'object': 'chat.completion',
        'created': int(time()),
        'choices': [{
            'index': 0,
            'message': {
                'role': 'assistant',
                'content': result.get('answer', result.get('result', '')),  # 兼容处理
            },
            'finish_reason': 'stop',
        }],
        'usage': result.get('usage', {}),
        # 这里返回相关文档信息
        'source_documents': [{
            'content': i.page_content,
            'metadata': i.metadata,
        } for i in result.get('source_documents', [])],
    })


# https://api.openai.com/v1/chat/completions
@app.route('/embed/<hash>/v1/chat/completions', methods=['POST'])
@app.route('/api/collection/<collection_id>/v1/chat/completions', methods=['POST'])
def openai_chat_on_collection(collection_id):
    if '/embed' in request.path and '/chat/completions' in request.path:
        # 这里使用hash判断是否有权限，并且
        hash = collection_id
        collection_id = get_collection_id_by_hash(hash)
        data = get_data_by_hash(hash, request.json)
    else:
        data = request.json
    # user_id = session.get('user_id', '')
    # collection = get_collection_by_id(user_id, collection_id)
    # assert collection, '找不到知识库或者没有权限'

    sse = ServerSentEvents()

    request.environ['STARTED'] = False
    def on_llm_new_token(token):
        if not request.environ['STARTED']:
            sse.send('assistant', event='role')
            app.logger.info("send role")
            request.environ['STARTED'] = True

        # app.logger.info("streaming on_llm_new_token %r", token)
        sse.send(token)

    @copy_current_request_context
    def main():
        result = chat_on_collection(
            collection_id, None,
            on_llm_new_token=on_llm_new_token,
            **data,
        )
        app.logger.info("%r", result)
        sse.send('', event='end')
        return result

    # stream模式
    if data.get('stream'):
        t = ThreadTask(main)
        t.start()
        return sse.response()

    result = main()
    return jsonify({
        'id': uuid4(),
        'object': 'chat.completion',
        'created': int(time()),
        'choices': [{
            'index': 0,
            'message': {
                'role': 'assistant',
                'content': result.get('answer', result.get('result', '')),  # 兼容处理
            },
            'finish_reason': 'stop',
        }],
        'usage': result.get('usage', {}),
        # 这里返回相关文档信息
        'source_documents': [{
            'content': i.page_content,
            'metadata': i.metadata,
        } for i in result.get('source_documents', [])],
    })


@app.route('/api/file/<user_id>/<filename>', methods=['GET'])
def get_file(user_id, filename):
    app.logger.info('filename %r', filename)
    return send_file(app.config['UPLOAD_PATH'] + '/' + user_id + '/' + filename)


@app.route('/api/upload', methods=['POST'])
def upload():
    # app.logger.info("file %r", request.files['file'])
    if 'file' not in request.files:
        raise InternalError()
    file = request.files['file']
    user_id = session.get('user_id', '')
    directory = app.config['UPLOAD_PATH'] + '/' + user_id
    filename = file.filename.replace('/', '_')
    if not os.path.exists(directory):
        os.makedirs(directory)
    file.save(directory + '/' + filename)
    return {
        'url': app.config['DOMAIN'] + '/api/file/' + user_id + '/' + quote(filename) + '?__sid__=' + session.sid,
    }


@app.route('/api/bot', methods=['GET'])
def get_bot_list_handler():
    page = request.args.get('page', default=1, type=int)
    size = request.args.get('size', default=20, type=int)
    user_id = session.get('user_id', '')
    bots, total = get_bot_list(user_id, '', page, size)
    return jsonify({
        'code': 0,
        'msg': 'success',
        'data': [{
            'bot_id': bot.id,
            'user_id': bot.user_id,
            'collection_id': bot.collection_id,
            'hash': bot.hash,
            'extra': bot.extra,
        } for bot in bots],
        'total': total,
    })


@app.route('/api/bot/<hash>', methods=['GET'])
def get_bot_by_hash_handler(hash):
    bot = get_bot_by_hash(hash)
    return jsonify({
        'code': 0,
        'msg': 'success',
        'data': {
            'bot_id': bot.id,
            'user_id': bot.user_id,
            'collection_id': bot.collection_id,
            'hash': bot.hash,
            'extra': bot.extra,
        },
    })


@app.route('/api/collection/<collection_id>/bot', methods=['GET'])
def get_hash_by_collection_id_handler(collection_id):
    hash = get_hash_by_collection_id(collection_id)
    return jsonify({
        'code': 0,
        'msg': 'success',
        'data': hash,
    })


@app.route('/api/collection/<collection_id>/bot', methods=['POST'])
def create_bot_handler(collection_id):
    user_id = session.get('user_id', '')
    hash = create_bot(user_id, collection_id, **request.json)
    return jsonify({
        'code': 0,
        'msg': 'success',
        'data': hash,
    })


@app.route('/api/collection/<collection_id>/bot', methods=['PUT'])
def update_bot_handler(collection_id):
    action = request.json.get('action', 'start')  # action=start/stop/remove/refresh
    hash = update_bot_by_collection_id_and_action(collection_id, action)
    return jsonify({
        'code': 0,
        'msg': 'success',
        'data': hash,
    })



