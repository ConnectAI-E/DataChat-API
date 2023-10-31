import redis
import logging
from datetime import timedelta
from flask import Flask
from flask_session import Session, RedisSessionInterface
from itsdangerous import BadSignature, want_bytes
from flask_cors import CORS
from flasgger import Swagger


app = Flask(__name__)

app.config["SESSION_TYPE"] = 'redis'  # 指定session存储的类型
app.config["SESSION_REDIS"] = redis.Redis(host="redis", port=6379)  # 创建redis连接
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(minutes=86400)  # 执行session的有效时间

app.config.setdefault('DELTA', 0.5)

# 从环境变量读取配置信息，统一使用前缀
# FLASK_SQLALCHEMY_DATABASE_URI --> app.config["SQLALCHEMY_DATABASE_URI"]
app.config.from_prefixed_env()

CORS(app, allow_headers=["Authorization", "X-Requested-With"], supports_credentials=True)
# Session(app)  # 使用Session对session的存储机制重新定义

class SessionInterface(RedisSessionInterface):

    def open_session(self, app, request):
        # 　　从cookie中获取session
        arg_sid = request.args.get('__sid__', default='', type=str)
        cookie_sid = request.cookies.get('__sid__', '')
        cookie_session = request.cookies.get('session', '')
        sid = cookie_sid or cookie_session or arg_sid or request.headers.get('Authorization', '')[7:]
        # 　　首次访问如没有获取到session  ID
        if not sid:
            # 　设置一个随机字符串，使用uuid
            sid = self._generate_sid()

            #返回特殊字典   <RedisSession {'_permanent': True}>

            return self.session_class(sid=sid, permanent=self.permanent) #session_class = RedisSession()
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

        val = self.redis.get(self.key_prefix + sid)
        if val is not None:
            try:
                data = self.serializer.loads(val)
                return self.session_class(data, sid=sid)
            except:
                return self.session_class(sid=sid, permanent=self.permanent)
        return self.session_class(sid=sid, permanent=self.permanent)


app.session_interface = SessionInterface(
    redis=app.config["SESSION_REDIS"],
    key_prefix="know"
)

swagger_config = Swagger.DEFAULT_CONFIG
Swagger(app, config=swagger_config)


gunicorn_logger = logging.getLogger('gunicorn.error')
app.logger.handlers = gunicorn_logger.handlers
app.logger.setLevel(gunicorn_logger.level)

openai_logger = logging.getLogger("openai")
openai_logger.handlers = gunicorn_logger.handlers
openai_logger.setLevel(gunicorn_logger.level)


app.logger.info("Swagger %r", swagger_config)

