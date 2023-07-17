FROM python:3.8

RUN echo "start"
RUN sed -i "s@http://deb.debian.org@http://mirrors.aliyun.com@g" /etc/apt/sources.list

RUN apt-get update && apt-get install -y libcurl4-openssl-dev libffi-dev libxml2-dev g++\
  && pip3 install requests Flask gunicorn gevent bson psycopg2 Flask-Session Flask-SQLAlchemy Flask-Migrate pgvector ujson pycurl bcrypt langchain sentence_transformers celery[sqlalchemy] pdf2image pytesseract -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn --no-cache-dir

RUN pip3 install redis unstructured PyMuPDF bs4 openai flask[async] -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn --no-cache-dir

ADD ./docker/entrypoint.sh /entrypoint.sh
ADD ./docker/wait-for-it.sh /wait-for-it.sh
RUN chmod a+x /entrypoint.sh

ADD ./all-MiniLM-L6-v2.tar.gz /
ADD ./punkt.tar.gz /usr/lib/nltk_data/tokenizers
ADD ./averaged_perceptron_tagger.tar.gz /usr/lib/nltk_data/taggers
ADD ./web_base.py /usr/local/lib/python3.8/site-packages/langchain/document_loaders/web_base.py

WORKDIR /server
ENTRYPOINT ["/entrypoint.sh"]

RUN pip3 install Cython flask-cors -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn --no-cache-dir

ADD ./setup.py /setup.py
ADD ./server /server

RUN cd / && python3 setup.py build_ext
RUN cd /server && rm *.c tasks.py sse.py models.py app.py && cp ../build/lib.linux-x86_64-3.8/* ./


CMD ["gunicorn", "--worker-class=gevent", "--workers", "8", "--bind", "0.0.0.0:80", "-t", "600", "--keep-alive", "60", "--log-level=debug", "server:app"]

