import os
import httpx
from functools import cached_property
from tempfile import NamedTemporaryFile
from time import time
from datetime import timedelta
from celery import Celery
from app import app
from models import save_document, save_embedding, purge_document_by_id, ObjID
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.document_loaders import (
    PyMuPDFLoader,
    TextLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader,
    UnstructuredFileLoader,
)
from langchain.document_loaders.sitemap import SitemapLoader
from langchain.schema import Document


LARK_HOST = 'https://open.feishu.cn'


def create_celery_app(app=None):
    """
    Create a new Celery object and tie together the Celery config to the app's
    config. Wrap all tasks in the context of the application.

    :param app: Flask app
    :return: Celery app
    """

    app.config['CELERY_BROKER_URL'] = 'redis://redis:6379/0'
    app.config['CELERY_RESULT_BACKEND'] = 'redis://redis:6379/0'
    celery = Celery(app.import_name, broker=app.config['CELERY_BROKER_URL'], backend=app.config['CELERY_RESULT_BACKEND'])

    celery.conf.update(app.config.get("CELERY_CONFIG", {}))
    TaskBase = celery.Task

    class ContextTask(TaskBase):
        abstract = True

        def __call__(self, *args, **kwargs):
            with app.app_context():
                return TaskBase.__call__(self, *args, **kwargs)

    celery.Task = ContextTask
    return celery



celery = create_celery_app(app)

celery.conf.beat_schedule = {
    "sync_feishudoc": {
        "task": "celery_app.sync_feishudoc",
        "schedule": timedelta(hours=1), # 定时1hours执行一次
        # "schedule": timedelta(seconds=5), # 定时2hours执行一次
        # "schedule": 10.0, # 每10秒执行一次
        # "schedule": crontab(minute='*/1'), # 定时每分钟执行一次
        # 使用的是crontab表达式
        "args": (False) # 函数传参的值
    }
}


LOADER_MAPPING = {
    "pdf": (PyMuPDFLoader, {}),
    "word": (UnstructuredWordDocumentLoader, {}),
    "excel": (UnstructuredExcelLoader, {}),
    "markdown": (UnstructuredMarkdownLoader, {}),
    "ppt": (UnstructuredPowerPointLoader, {}),
    "txt": (TextLoader, {"encoding": "utf8"}),
    "html": (UnstructuredHTMLLoader, {}),
    "sitemap": (SitemapLoader, {}),
    "default": (UnstructuredFileLoader, {}),
}


def embedding_single_document(doc, fileUrl, fileType, fileName, collection_id, openai=False, uniqid='', version=0):
    # 初始化embeddings
    if openai:
        embeddings = OpenAIEmbeddings()
    else:
        embeddings = HuggingFaceEmbeddings(model_name="/m3e-base")
    # 初始化加载器
    # text_splitter = CharacterTextSplitter(chunk_size=1024, chunk_overlap=0)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    split_docs = text_splitter.split_documents([doc])
    # 先生成document_id，等向量保存完成之后，再保存文档
    document_id = ObjID.new_id()
    # document_ids.append(document_id)
    try:
        doc_result = embeddings.embed_documents([d.page_content for d in split_docs])
        for chunk_index, doc in enumerate(split_docs):
            save_embedding(
                collection_id, document_id,
                chunk_index, len(doc.page_content),
                doc.page_content,
                doc_result[chunk_index],  # embed
            )
        save_document(
            collection_id, fileName or fileUrl, fileUrl, len(split_docs), fileType,
            uniqid=uniqid, version=version,
            document_id=document_id,
        )
        return document_id
    except Exception as e:
        # 出错的时候移除
        purge_document_by_id(document_id)


def get_status_by_id(task_id):
    return celery.AsyncResult(task_id)


def embed_query(text, openai=False):
    # 初始化embeddings
    if openai:
        embeddings = OpenAIEmbeddings()
    else:
        embeddings = HuggingFaceEmbeddings(model_name="/m3e-base")

    return embeddings.embed_query(text)


class Lark(object):
    def __init__(self, app_id=None, secret_key=None, app_secret=None, verification_token=None, validation_token=None, encript_key=None, encrypt_key=None, host=LARK_HOST, **kwargs):
        self.app_id = app_id
        self.app_secret = app_secret or secret_key
        self.encrypt_key = encrypt_key or encript_key
        self.verification_token = verification_token or validation_token
        self.host = host

    @cached_property
    def _tenant_access_token(self):
        # https://open.feishu.cn/document/ukTMukTMukTM/ukDNz4SO0MjL5QzM/auth-v3/auth/tenant_access_token_internal
        url = f'{self.host}/open-apis/auth/v3/tenant_access_token/internal'
        result = self.post(url, json={
            'app_id': self.app_id,
            'app_secret': self.app_secret,
        }).json()
        if "tenant_access_token" not in result:
            raise Exception('get tenant_access_token error')
        return result['tenant_access_token'], result['expire'] + time()

    @property
    def tenant_access_token(self):
        token, expired = self._tenant_access_token
        if not token or expired < time():
            # retry get_tenant_access_token
            del self._tenant_access_token
            token, expired = self._tenant_access_token
        return token

    def request(self, method, url, headers=dict(), **kwargs):
        if 'tenant_access_token' not in url:
            headers['Authorization'] = 'Bearer {}'.format(self.tenant_access_token)
        return httpx.request(method, url, headers=headers, **kwargs)

    def get(self, url, **kwargs):
        return self.request('GET', url, **kwargs)

    def post(self, url, **kwargs):
        return self.request('POST', url, **kwargs)

class LarkDocLoader(object):
    def __init__(self, fileUrl, document_id, **kwargs):
        app.logger.info("debug %r", kwargs)
        self.client = Lark(**kwargs)
        self.fileUrl = fileUrl
        if not document_id:
            t = fileUrl.split('?')[0].split('/')
            document_id = t.pop()
            type_ = t.pop()
            # https://open.feishu.cn/open-apis/wiki/v2/spaces/get_node
            # https://xxx.feishu.cn/docx/ExGmdqrg4oz2evx7SRuciY78nRe
            # https://xxx.feishu.cn/wiki/V0LuwIeWCiL3yWkq0zBcn1g0nua
            if type_ == 'wiki':
                url = f"{self.client.host}/open-apis/wiki/v2/spaces/get_node?token={document_id}"
                res = self.client.get(url).json()
                if 'data' not in res or 'node' not in res['data']:
                    app.logger.error("error get node %r", res)
                    if res['code'] == 131006:
                        raise Exception('「企联 AI 飞书助手」应用权限配置不正确，请检查以后重新配置')
                    raise Exception('「企联 AI 飞书助手」无该文档访问权限')
                document_id = res['data']['node']['obj_token']
                type_ = res['data']['node']['obj_type']

            if type_ not in ['docx', 'doc']:
                app.logger.error("unsupport type %r", type_)
                raise Exception('「企联 AI 飞书助手」无该文档访问权限')
                # raise Exception(f'unsupport type {type_}')
        self.document_id = document_id
        # TODO (文档只有所有者可以订阅) 查询订阅状态
        # url = f"{self.client.host}/open-apis/drive/v1/files/{document_id}/get_subscribe?file_type={type_}"
        # res = self.client.get(url).json()
        # if not res.get('data', {}).get('is_subscribe'):
        #     url = f"{self.client.host}/open-apis/drive/v1/files/{document_id}/subscribe?file_type={type_}"
        #     res = self.client.post(url).json()
        #     app.logger.info("debug subscribe %r", res)

    def load(self):
        # https://open.feishu.cn/document/server-docs/docs/docs/docx-v1/document/raw_content
        # https://open.feishu.cn/open-apis/docx/v1/documents/:document_id/raw_content
        url = f"{self.client.host}/open-apis/docx/v1/documents/{self.document_id}/raw_content"
        res = self.client.get(url).json()
        if 'data' not in res or 'content' not in res['data']:
            app.logger.error("error get content %r", res)
            raise Exception('「企联 AI 飞书助手」无该文档访问权限')
            # raise Exception(f'error get content for document')
        return Document(page_content=res['data']['content'], metadata=dict(
            fileUrl=self.fileUrl,
            document_id=self.document_id,
            revision_id=self.version,
            title=self.title,
        ))

    @property
    def version(self):
        return self.file_info.get('revision_id', 0)

    @property
    def title(self):
        return self.file_info.get('title', '')

    @cached_property
    def file_info(self):
        url = f"{self.client.host}/open-apis/docx/v1/documents/{self.document_id}"
        res = self.client.get(url).json()
        app.logger.info("debug file_info %r %r", url, res)
        return res.get('data', {}).get('document', {})



