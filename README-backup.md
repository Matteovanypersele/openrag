
# How to backup OpenRag partition ?


```
docker compose \
    run \
    --build \
    --rm \
    -v /my-backup-dir/:/backup:rw \
    --entrypoint "bash /app/openrag/scripts/entrypoint-backup.sh ${PARTITION_NAME}" \
    openrag-cpu
```
It's better to stop `openrag-cpu` (or `openrag`) service before starting backup.

By default backup script creates plan text uncomressed file. To make things faster you can use multithread compressor the following way:

```
docker compose \
    run \
    --build \
    --rm \
    -v /my-backup-dir/:/backup:rw \
    --entrypoint "bash /app/openrag/scripts/entrypoint-backup-mt.sh ${PARTITION_NAME}" \
    openrag-cpu
```


# How to restore OpenRag partition ?

Start with dry run to ensure the backup file is correct:

```
docker compose \
    run \
    --build \
    --rm \
    -v /my-backup-dir/:/backup:ro \
    --entrypoint "bash /app/openrag/scripts/entrypoint-restore-dry-run.sh backup-file-without-path parition-name" \
    openrag-cpu
```
Backup files are expected to be in `/my-backup-dir/`. If the dry run is successful, run the following script to insert the data :

```
docker compose \
    run \
    --build \
    --rm \
    -v /my-backup-dir/:/backup:ro \
    --entrypoint "bash /app/openrag/scripts/entrypoint-restore.sh backup-file-without-path parition-name" \
    openrag-cpu
```

