#!/bin/bash

/wait-for-it.sh pgvector:5432 -- python3 server.py && celery -A celery_app.celery worker &

exec "$@"

