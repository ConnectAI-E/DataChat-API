import redis
import logging
from datetime import timedelta
from flask import Flask
from flask_session import Session
from itsdangerous import BadSignature, want_bytes
from flask_cors import CORS


app = Flask(__name__)

app.config["SESSION_TYPE"] = 'redis'  # 指定session存储的类型
app.config["SESSION_REDIS"] = redis.Redis(host="redis", port=6379)  # 创建redis连接
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(minutes=1)  # 执行session的有效时间

# 从环境变量读取配置信息，统一使用前缀
# FLASK_SQLALCHEMY_DATABASE_URI --> app.config["SQLALCHEMY_DATABASE_URI"]
app.config.from_prefixed_env()

CORS(app, allow_headers=["Authorization", "X-Requested-With"], supports_credentials=True)
Session(app)  # 使用Session对session的存储机制重新定义


gunicorn_logger = logging.getLogger('gunicorn.error')
app.logger.handlers = gunicorn_logger.handlers
app.logger.setLevel(gunicorn_logger.level)

openai_logger = logging.getLogger("openai")
openai_logger.handlers = gunicorn_logger.handlers
openai_logger.setLevel(gunicorn_logger.level)


