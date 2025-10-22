#!/usr/bin/env bash

NAME="${1:-openrag-openrag-cpu-1}"
PORT="${2:-8080}"
NUM=$3
ADDR=`docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' ${NAME}`

while true
do
  tf=`curl -fs "${ADDR}:${PORT}/queue/info" 2>/dev/null | jq '.tasks.total_failed'`

  if [ "${tf}" -ne 0 ]
  then
    df -h
    docker logs openrag-openrag-cpu-1
    echo "ERROR: ${tf} tasks failed. Aborting."
    exit 1
  fi

  tc=`curl -fs "${ADDR}:${PORT}/queue/info" 2>/dev/null | jq '.tasks.total_completed'`

  if [ "${tc}" -eq ${NUM} ]
  then
    echo "${tc} tasks completed."
    break
  fi

  curl -fs "${ADDR}:${PORT}/queue/info"
  curl -fs "${ADDR}:${PORT}/queue/tasks"

  echo "Waiting: ${tc} tasks completed, ${tf} tasks failed on ${ADDR}:${PORT}"
  sleep 10s
done

