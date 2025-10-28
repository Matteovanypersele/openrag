#!/usr/bin/env bash

NAME="${1:-openrag-vllm-cpu-1}"
PORT="${2:-8000}"
ADDR=`docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' ${NAME}`

while ! curl -fs "${ADDR}:${PORT}/health" >/dev/null 2>&1;
do
  if docker ps --format '{{.Names}}' | grep -qw "$NAME"; then
    echo "Container '$NAME' is running but not healthy yet ..."
  else
    echo "Container '$NAME' has stopped or was never started."
    exit 1
  fi

  echo "Waiting for ${NAME} to start at ${ADDR}:${PORT}"
  sleep 10s

done

echo "${NAME} at ${ADDR}:${PORT} is healthy"

