FROM python:3.8-bullseye

RUN echo "start"
RUN sed -i "s@http://deb.debian.org@http://mirrors.aliyun.com@g" /etc/apt/sources.list
RUN sed -i "s@http://security.debian.org@http://mirrors.aliyun.com@g" /etc/apt/sources.list

RUN apt-get update && apt-get install -y libcurl4-openssl-dev libffi-dev libxml2-dev g++ libreoffice \
  && pip3 install torch==2.0.1+cpu torchvision==0.15.2+cpu -f https://download.pytorch.org/whl/torch_stable.html  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  && pip3 install requests Flask gunicorn gevent bson Flask-Session==0.5.0 Flask-SQLAlchemy ujson pycurl bcrypt langchain=0.0.246 sentence_transformers pdf2image pytesseract elasticsearch_dsl redis unstructured PyMuPDF bs4 openai==0.28.1 flask[async] Cython flask-cors python-docx python-pptx markdown pandas openpyxl celery -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn --no-cache-dir

ADD ./docker/entrypoint.sh /entrypoint.sh
ADD ./docker/wait-for-it.sh /wait-for-it.sh
RUN chmod a+x /entrypoint.sh

RUN pip3 install flasgger httpx -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn --no-cache-dir

# ADD ./m3e-base.tar.gz /
ADD ./m3e-base /m3e-base

ADD ./punkt.tar.gz /usr/lib/nltk_data/tokenizers
ADD ./averaged_perceptron_tagger.tar.gz /usr/lib/nltk_data/taggers
ADD ./web_base.py /usr/local/lib/python3.8/site-packages/langchain/document_loaders/web_base.py

WORKDIR /server
ENTRYPOINT ["/entrypoint.sh"]

ADD ./setup.py /setup.py
ADD ./server /server

RUN cd / && python3 setup.py build_ext
RUN cd /server && rm *.c tasks.py sse.py models.py app.py && cp ../build/lib.linux-x86_64-3.8/* ./


CMD ["gunicorn", "--worker-class=gevent", "--workers", "1", "--bind", "0.0.0.0:80", "-t", "600", "--keep-alive", "60", "--log-level=debug", "server:app"]

