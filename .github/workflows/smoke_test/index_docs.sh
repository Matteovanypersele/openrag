#!/usr/bin/env bash

source venv/bin/activate

docker container ls
OPENRAG_ADDR=`docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' openrag-openrag-cpu-1`
docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' openrag-openrag-cpu-1

docker logs openrag-openrag-cpu-1

sleep 180s 

python3 utility/data_indexer.py \
    -u http://${OPENRAG_ADDR}:8080 \
    -d .github/workflows/data/simplewiki-500/ \
    -p simplewiki-500

docker logs openrag-openrag-cpu-1

.github/workflows/smoke_test/wait_for_tasks_completed.sh openrag-openrag-cpu-1 8080 500

