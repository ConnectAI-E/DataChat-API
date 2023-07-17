import logging
from datetime import datetime
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_session import Session, SqlAlchemySessionInterface
from itsdangerous import BadSignature, want_bytes
from flask_cors import CORS


app = Flask(__name__)
# 从环境变量读取配置信息，统一使用前缀
# FLASK_SQLALCHEMY_DATABASE_URI --> app.config["SQLALCHEMY_DATABASE_URI"]
app.config.from_prefixed_env()
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///app.db'

# 初始化Session
# db = SQLAlchemy(app, engine_options={'isolation_level': 'REPEATABLE READ'})
db = SQLAlchemy(app, engine_options={'isolation_level': 'AUTOCOMMIT'})
# db = SQLAlchemy(app)
# app.config['SESSION_SQLALCHEMY'] = db
# Session(app)

CORS(app, allow_headers=["Authorization", "X-Requested-With"], supports_credentials=True)


class SessionInterface(SqlAlchemySessionInterface):
    def open_session(self, app, request):
        sid = request.headers.get('Authorization', '')[7:]
        if not sid:
            sid = self._generate_sid()
            return self.session_class(sid=sid, permanent=self.permanent)
        if self.use_signer:
            signer = self._get_signer(app)
            if signer is None:
                return None
            try:
                sid_as_bytes = signer.unsign(sid)
                sid = sid_as_bytes.decode()
            except BadSignature:
                sid = self._generate_sid()
                return self.session_class(sid=sid, permanent=self.permanent)

        store_id = self.key_prefix + sid
        saved_session = self.sql_session_model.query.filter_by(
            session_id=store_id).first()
        if saved_session and saved_session.expiry <= datetime.utcnow():
            # Delete expired session
            self.db.session.delete(saved_session)
            self.db.session.commit()
            saved_session = None
        if saved_session:
            try:
                val = saved_session.data
                data = self.serializer.loads(want_bytes(val))
                return self.session_class(data, sid=sid)
            except Exception as e:
                # app.logger.exception(e)
                return self.session_class(sid=sid, permanent=self.permanent)
        return self.session_class(sid=sid, permanent=self.permanent)


app.config.setdefault('SESSION_PERMANENT', True)
app.config.setdefault('SESSION_USE_SIGNER', False)
app.config.setdefault('SESSION_KEY_PREFIX', 'session:')
app.config.setdefault('SESSION_SQLALCHEMY_TABLE', 'sessions')
app.session_interface = SessionInterface(
    app,
    db,
    app.config['SESSION_SQLALCHEMY_TABLE'],
    app.config['SESSION_KEY_PREFIX'],
    app.config['SESSION_USE_SIGNER'],
    app.config['SESSION_PERMANENT']
)

gunicorn_logger = logging.getLogger('gunicorn.error')
app.logger.handlers = gunicorn_logger.handlers
app.logger.setLevel(gunicorn_logger.level)

openai_logger = logging.getLogger("openai")
openai_logger.handlers = gunicorn_logger.handlers
openai_logger.setLevel(gunicorn_logger.level)


