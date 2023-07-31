import logging
import bson
import json
from uuid import uuid4
from copy import deepcopy
from datetime import datetime
from sqlalchemy import String, text, select, and_
from sqlalchemy.orm import column_property
from pgvector.sqlalchemy import Vector
from langchain.schema import Document, BaseRetriever
from langchain.chat_models import ChatOpenAI, AzureChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
)
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks import get_openai_callback
from app import db, app


class NotFound(Exception): pass


class ObjID(db.LargeBinary):
    """基于bson.ObjectId用于mysql主键的自定义类型"""
    def bind_processor(self, dialect):
        def processor(value):
            return bson.ObjectId(value).binary if bson.ObjectId.is_valid(value) else value

        return processor

    def result_processor(self, dialect, coltype):
        def processor(value):
            if not isinstance(value, bytes):
                value = bytes(value)
            return str(bson.ObjectId(value)) if bson.ObjectId.is_valid(value) else value

        return processor

    @staticmethod
    def new_id():
        return str(bson.ObjectId())

    @staticmethod
    def is_valid(value):
        return bson.ObjectId.is_valid(value)


class JSONStr(String):
    """自动转换 str 和 dict 的自定义类型"""
    def bind_processor(self, dialect):
        def processor(value):
            try:
                if isinstance(value, str) and (value[0] == '%' or value[-1] == '%'):
                    # 使用like筛选的情况
                    return value
                return json.dumps(value, ensure_ascii=False)
            except Exception as e:
                logging.exception(e)
                return value
        return processor

    def result_processor(self, dialect, coltype):
        def processor(value):
            try:
                return json.loads(value)
            except Exception as e:
                logging.exception(e)
                return value
        return processor

    @staticmethod
    def is_valid(value):
        try:
            json.loads(value)
            return True
        except Exception as e:
            logging.exception(e)
            return False


class User(db.Model):
    __tablename__ = 'user'
    id = db.Column(ObjID(12), primary_key=True)
    openid = db.Column(db.String(128), nullable=True, comment="外部用户ID")
    name = db.Column(db.String(128), nullable=True, comment="用户名")
    extra = db.Column(JSONStr(1024), nullable=True, server_default=text("'{}'"), comment="用户其他字段")
    status = db.Column(db.Integer, nullable=True, default=0, server_default=text("0"))
    created = db.Column(db.TIMESTAMP, nullable=False, default=datetime.utcnow)
    modified = db.Column(db.TIMESTAMP, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)


class Collection(db.Model):
    __tablename__ = 'collection'
    id = db.Column(ObjID(12), primary_key=True)
    user_id = db.Column(ObjID(12), nullable=True, comment="用户ID")
    name = db.Column(db.String(128), nullable=True, comment="知识库名称")
    description = db.Column(db.String(512), nullable=True, comment="知识库描述")
    status = db.Column(db.Integer, nullable=True, default=0, server_default=text("0"))
    created = db.Column(db.TIMESTAMP, nullable=False, default=datetime.utcnow)
    modified = db.Column(db.TIMESTAMP, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)


class Documents(db.Model):
    __tablename__ = 'documents'
    id = db.Column(ObjID(12), primary_key=True)
    collection_id = db.Column(ObjID(12), nullable=True, comment="知识库ID")
    type = db.Column(db.String(128), nullable=True, comment="文件类型")
    name = db.Column(db.String(512), nullable=True, comment="文件名称")
    path = db.Column(db.String(512), nullable=True, comment="文件地址")
    chunks = db.Column(db.Integer, nullable=True, default=0, server_default=text("0"), comment="文件分片数量")
    status = db.Column(db.Integer, nullable=True, default=0, server_default=text("0"))
    created = db.Column(db.TIMESTAMP, nullable=False, default=datetime.utcnow)
    modified = db.Column(db.TIMESTAMP, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)


class Embedding(db.Model):
    __tablename__ = 'embedding'
    id = db.Column(ObjID(12), primary_key=True)
    document_id = db.Column(ObjID(12), nullable=True, comment="文件ID")
    collection_id = db.Column(ObjID(12), nullable=True, comment="知识库ID")
    chunk_index = db.Column(db.Integer, nullable=True, default=0, server_default=text("0"), comment="文件分片索引")
    chunk_size = db.Column(db.Integer, nullable=True, default=0, server_default=text("0"), comment="文件分片大小")
    document = db.Column(db.Text, nullable=True, comment="分片内容")
    embedding = db.Column(Vector, nullable=True, comment="分片向量")
    status = db.Column(db.Integer, nullable=True, default=0, server_default=text("0"))
    created = db.Column(db.TIMESTAMP, nullable=False, default=datetime.utcnow)
    modified = db.Column(db.TIMESTAMP, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)


class Bot(db.Model):
    __tablename__ = 'bot'
    id = db.Column(ObjID(12), primary_key=True)
    user_id = db.Column(ObjID(12), nullable=True, comment="用户ID")
    collection_id = db.Column(ObjID(12), nullable=True, comment="知识库ID")
    hash = db.Column(String(128), nullable=True, comment="hash")
    extra = db.Column(JSONStr(1024), nullable=True, server_default=text("'{}'"), comment="机器人配置信息")
    status = db.Column(db.Integer, nullable=True, default=0, server_default=text("0"))
    created = db.Column(db.TIMESTAMP, nullable=False, default=datetime.utcnow)
    modified = db.Column(db.TIMESTAMP, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)


def get_user(user_id):
    user = db.session.query(User).filter(
        User.id == user_id,
        User.status == 0,
    ).first()
    if not user:
        raise NotFound()
    return user


def save_user(openid='', name='', **kwargs):
    user = db.session.query(User).filter(
        User.openid == openid,
        User.status == 0,
    ).first()
    if not user:
        user = User(
            id=ObjID.new_id(),
            openid=openid,
            name=name,
            extra=kwargs,
        )
        db.session.add(user)
        db.session.commit()
    else:
        db.session.query(User).filter(User.id == user.id).update(dict(
            openid=openid,
            name=name,
            extra=kwargs,
        ), synchronize_session=False)
        db.session.commit()
    return user


def query_one_page(query, page, size):
    offset = (page - 1) * int(size)
    return query.offset(
        offset if offset > 0 else 0
    ).limit(
        size if size > 0 else 0
    ).all()


class CollectionWithDocumentCount(Collection):

    document_count = column_property(
        select(func.count(Documents.id)).where(and_(
            Documents.collection_id == Collection.id,
            Documents.status == 0,
        ))
    )


def get_collections(user_id, page, size):
    query = db.session.query(CollectionWithDocumentCount).filter(
        Collection.user_id == user_id,
        Collection.status == 0,
    )
    total = query.count()
    if total == 0:
        return [], 0
    return query_one_page(query, page, size), total


def get_collection_by_id(user_id, collection_id):
    return db.session.query(Collection).filter(
        Collection.user_id == user_id,
        Collection.id == collection_id,
        Collection.status == 0,
    ).first()


def save_collection(user_id, name, description):
    collection_id = ObjID.new_id()
    db.session.add(Collection(
        id=collection_id,
        user_id=user_id,
        name=name,
        description=description,
    ))
    db.session.commit()
    return collection_id


def update_collection_by_id(user_id, collection_id, name, description):

    db.session.query(Collection).filter(
        Collection.user_id == user_id,
        Collection.id == collection_id,
    ).update(dict(
        name=name,
        description=description,
    ), synchronize_session=False)
    db.session.commit()


def delete_collection_by_id(user_id, collection_id):

    db.session.query(Collection).filter(
        Collection.user_id == user_id,
        Collection.id == collection_id,
    ).update(dict(
        status=-1,
    ), synchronize_session=False)
    db.session.commit()


def get_documents_by_collection_id(user_id, collection_id, page, size):
    collection = get_collection_by_id(user_id, collection_id)
    assert collection, '找不到对应知识库'
    query = db.session.query(Documents).filter(
        Documents.collection_id == collection_id,
        Documents.status == 0,
    ).order_by(
        Documents.created.desc(),
    )
    total = query.count()
    if total == 0:
        return [], 0
    return query_one_page(query, page, size), total


def remove_document_by_id(user_id, collection_id, document_id):
    collection = get_collection_by_id(user_id, collection_id)
    assert collection, '找不到对应知识库'
    db.session.query(Documents).filter(
        Documents.id == document_id,
    ).update(dict(
        status=-1,
    ), synchronize_session=False)
    db.session.query(Embedding).filter(
        Embedding.document_id == document_id,
    ).update(dict(
        status=-1,
    ), synchronize_session=False)
    db.session.commit()


def save_document(collection_id, name, url, chunks, type):
    did = ObjID.new_id()
    db.session.add(Documents(
        id=did,
        collection_id=collection_id,
        type=type,
        name=name,
        path=url,
        chunks=chunks,
    ))
    db.session.commit()
    return did


def save_embedding(collection_id, document_id, chunk_index, chunk_size, document, embedding):
    eid = ObjID.new_id()
    app.logger.info("debug save_embedding %r", len(embedding))
    db.session.add(Embedding(
        id=eid,
        collection_id=collection_id,
        document_id=document_id,
        chunk_index=chunk_index,
        chunk_size=chunk_size,
        document=document,
        embedding=embedding,
    ))
    db.session.commit()
    return eid


def get_bot_list(user_id, collection_id, page, size):
    query = db.session.query(Bot).filter(
        Bot.user_id == user_id,
        Bot.collection_id == collection_id if collection_id else True,
        Bot.status > -1,
    ).order_by(
        Bot.created.desc(),
    )
    total = query.count()
    if total == 0:
        return [], 0
    return query_one_page(query, page, size), total


def get_bot_by_hash(hash):
    bot = db.session.query(Bot).filter(
        Bot.hash == hash,
        # Bot.status == 1,
    ).first()
    if not bot:
        raise NotFound()
    return bot


def create_bot(user_id, collection_id, **extra):
    # 移除这个逻辑
    # if db.session.query(Bot.id).filter(
    #     Bot.collection_id == collection_id,
    #     Bot.status >= 0,
    # ).limit(1).scalar():
    #     raise Exception('already create bot')
    hash = str(uuid4())
    db.session.add(Bot(
        id=ObjID.new_id(),
        user_id=user_id,
        collection_id=collection_id,
        hash=hash,
        extra=extra,
        status=1,  # 启用
    ))
    db.session.commit()
    return hash


def update_bot_by_collection_id_and_action(collection_id, action, hash=''):
    bot_id = db.session.query(Bot.id).filter(
        Bot.collection_id == collection_id if collection_id else True,
        Bot.hash == hash if hash else '',
        Bot.status >= 0,
    )
    # action=start/stop/remove/refresh
    if action == 'refresh':
        hash = str(uuid4())
        db.session.query(Bot).filter(
            Bot.id == bot_id,
        ).update(dict(hash=hash), synchronize_session=False)
        db.session.commit()
        return hash
    else:
        if 'start' == action:
            status = 1
        elif 'remove' == action:
            status = -1
        else:
            status = 0
        db.session.query(Bot).filter(
            Bot.id == bot_id,
        ).update(dict(status=status), synchronize_session=False)
        db.session.commit()
        return status


def get_collection_id_by_hash(hash):
    collection_id = db.session.query(Bot.collection_id).filter(
        Bot.hash == hash,
        Bot.status == 1,  # 这里是前端使用的，需要启用链接才能使用
    ).limit(1).scalar()
    return collection_id


def get_hash_by_collection_id(collection_id):
    hash = db.session.query(Bot.hash).filter(
        Bot.collection_id == collection_id,
        Bot.status >= 0,
    ).limit(1).scalar()
    return hash


def get_data_by_hash(hash, json):
    extra = db.session.query(Bot.extra).filter(
        Bot.hash == hash,
        Bot.status >= 0,
    ).limit(1).scalar()
    if extra:
        messages = json.get('messages', [])
        if extra.get('prompt', ''):
            messages = [{
                'role': 'system',
                'content': extra.get('prompt', '')
            }] + messages
        json.update(
            model=extra.get('model', 'gpt-3.5-turbo'),
            temperature=extra.get('temperature', 0.7),
            messages=messages,
        )
    return json


class EmbeddingWithDocument(Embedding):

    document_name = column_property(
        select(Documents.name).where(and_(
            Documents.id == Embedding.document_id,
        )).limit(1)
    )


def query_by_collection_id(collection_id, q, page, size):
    from tasks import embed_query
    embed = embed_query(q)
    # Embedding.embedding.l2_distance(embed),
    # Embedding.embedding.max_inner_product(embed),
    column = Embedding.embedding.cosine_distance(embed)
    # column = Embedding.embedding.l2_distance(embed)
    # column = Embedding.embedding.max_inner_product(embed)
    query = db.session.query(
        EmbeddingWithDocument,
        column.label('distinct'),
    ).filter(
        Embedding.collection_id == collection_id,
        Embedding.status == 0,
    ).order_by(
        column,
    )
    total = query.count()
    if total == 0:
        return [], 0
    return query_one_page(query, page, size), total



class Retriever(BaseRetriever):
    collection_id: str = ''
    similarity: float = 0
    limit: int = 4

    def get_relevant_documents(self, query: str):
        """Get texts relevant for a query.

        Args:
            query: string to find relevant texts for

        Returns:
            List of relevant documents
        """
        documents, total = query_by_collection_id(self.collection_id, query, 1, self.limit)
        # documents = list(filter(lambda i: i[1] > similarity, documents))
        app.logger.info("debug Documents %r", [documents, total, self.collection_id, query])
        app.logger.info("debug Documents %r", [(d.document, distance) for d, distance in documents])
        # 这里
        return [Document(
            page_content=document.document,
            metadata={
                'collection_id': self.collection_id,
                'document_id': document.document_id,
                'document_name': document.document_name,
                'distance': distance,
            }
        ) for document, distance in documents]

    async def aget_relevant_documents(self, query):
        # 这里不用真的异步
        return self.get_relevant_documents(query)


def chat_on_collection(
    collection_id,
    deployment_name=None,
    on_llm_new_token=None, stream=False,
    temperature=0.7,
    similarity=0.8,
    limit=4,
    messages=list(), **kwargs,
):
    retriever = Retriever(collection_id=collection_id, limit=limit)
    # TODO 这里需要一个简介的输出，可能需要调整模板
    system_template = """Use the following context to answer the user's question.
-----------
{{context}}
-----------
Question: {{question}}
Helpful Answer:"""

    assert len(messages) > 0, '问题为空'
    # 从原始的messages里面读取system message
    # TODO 如果用户传了system role，可能会被丢弃？
    # 取出除了system role之外的消息
    system_message = list(filter(lambda m: m.get('role') == 'system', messages))
    chat_history = list(filter(lambda m: m.get('role') != 'system', messages[:-1]))
    # 最后一条消息是提问消息
    assert messages[-1].get('role') == 'user', '问题为空'
    question = messages[-1].get('content')

    # 2023-06-22 客户想要使用system message，我们放弃使用langchain内置的prompt模式。而是直接更改chat_history，将context放进去就可以了?
    # 构建初始 messages 列表，这里可以理解为是 openai 传入的 messages 参数
    if len(system_message) > 0:
        chat_messages = [
          SystemMessagePromptTemplate.from_template(system_message[0]['content'], template_format="jinja2"),
          HumanMessagePromptTemplate.from_template(system_template, template_format="jinja2"),
        ]
    else:
        chat_messages = [
          SystemMessagePromptTemplate.from_template(system_template, template_format="jinja2"),
        ]

    # qa()函数传chat_history不起作用，将history放到prompt里面
    for m in chat_history:
        if m['role'] == 'assistant':
            message = AIMessagePromptTemplate.from_template(m['content'], template_format="jinja2")
        else:
            message = HumanMessagePromptTemplate.from_template(m['content'], template_format="jinja2")
        chat_messages.append(message)

    chat_messages.append(HumanMessagePromptTemplate.from_template('{{question}}', template_format="jinja2"))

    # 初始化 prompt 对象
    prompt = ChatPromptTemplate.from_messages(chat_messages)

    # 问答API一些配置信息
    app.config.setdefault('OPENAI_API_KEY', None)
    app.config.setdefault('OPENAI_API_BASE', None)
    app.config.setdefault('OPENAI_API_PROXY', '')
    openai_api_key = app.config['OPENAI_API_KEY']
    openai_api_base = app.config['OPENAI_API_BASE']
    openai_proxy = app.config['OPENAI_API_PROXY']
    # 这个是azure的版本号
    app.config.setdefault('OPENAI_API_VERSION', '2023-03-15-preview')
    openai_api_version = app.config['OPENAI_API_VERSION']

    params = deepcopy(kwargs)
    params.update(
        temperature=temperature,
        verbose=True,  # 调试信息
    )
    # 初始化问答API
    if stream:
        # 如果开启了stream，就增加一个callback_manager
        class StreamingCallback(BaseCallbackHandler):
            def on_llm_new_token(self, token: str, **kwargs):
                if on_llm_new_token:
                    on_llm_new_token(token)

        params.update(
            streaming=True,  # openai的接口使用的是stream，但是ChatOpenAI的参数名称是streaming
            callback_manager=CallbackManager([StreamingCallback()]),
            openai_api_key=openai_api_key,
            openai_api_base=openai_api_base,
            openai_proxy=openai_proxy,
        )
    else:
        params.update(
            openai_api_key=openai_api_key,
            openai_api_base=openai_api_base,
            openai_proxy=openai_proxy,
        )
    # Azure
    if deployment_name:
        params.update(
            deployment_name=deployment_name,
            openai_api_version=openai_api_version,
        )
        chat = AzureChatOpenAI(**params)
    else:
        chat = ChatOpenAI(**params)

    # 初始化问答链
    # qa = ConversationalRetrievalChain.from_llm(
    #     chat, retriever,
    #     # prompt=prompt,
    # )
    # result = qa({'question': question, 'chat_history': chat_history})
    # qa = RetrievalQA.from_chain_type(
    #     llm=chat,
    #     # prompt=prompt,  # chain_type_kwargs
    #     chain_type="stuff",
    #     retriever=retriever,
    #     return_source_documents=True,
    # )
    # 自定义模板
    qa = RetrievalQA.from_llm(
        llm=chat,
        prompt=prompt,  # chain_type_kwargs
        retriever=retriever,
        return_source_documents=True,
    )
    with get_openai_callback() as cb:
        result = qa({'query': question})
        app.logger.info("cb %r", cb)
        result['usage'] = {
            'prompt_tokens': cb.prompt_tokens,
            'completion_tokens': cb.completion_tokens,
            'total_tokens': cb.total_tokens,
        }
    # 这里的result['answer']就是回答
    return result


