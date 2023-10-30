#!/bin/bash

/wait-for-it.sh pgvector:5432 -- python3 server.py && celery -A celery_app.celery worker -l DEBUG -c 1 &
/wait-for-it.sh pgvector:5432 -- celery -A celery_app.celery beat -l DEBUG &

exec "$@"

