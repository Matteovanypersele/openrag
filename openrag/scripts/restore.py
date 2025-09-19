#!/usr/bin/env python3

import sys
import os
import json
import time

from pymilvus import MilvusClient

from config import load_config
from utils.logger import get_logger
from components.indexer.vectordb.utils import PartitionFileManager


def read_rdb_section(fh, pfm, include_only, added_documents, existing_partitions, logger, verbose=False, dry_run=False):
    part = json.loads(next(fh))
    if part['name'] in existing_partitions:
        raise Exception(f'Partition \"{part["name"]}\" already exists')

    if verbose:
        logger.info(f'Read rdb section | partition=\"{part["name"]}\"')

    for line in fh:
        line = line.strip()

        if 0 == len(line):
            break

        if include_only is not None and len(include_only) > 0 and part['name'] not in include_only:
            continue

        try:
            doc = json.loads(line)
        except Exception as e:
            logger.exception(f'Failed while parsing the following json:\n{line}\n')
            raise

        if not dry_run:
            try:
                res = pfm.add_file_to_partition(doc['file_id'], part['name'], doc)
            except Exception as e:
                logger.exception(f'{type(e)} in add_file_to_partition({doc["file_id"]}, {part["name"]}, ...)\n' + str(e))
                raise
        else:
            res = True

        if res:
            if part['name'] not in added_documents:
                added_documents[part['name']] = set()
            added_documents[part['name']].add(doc['file_id'])
        else:
            logger.error(f'Can\'t add file {doc["file_id"]} to partition {part["name"]}')


def insert_into_vdb(client, collection_name, batch, logger, verbose=False, dry_run=False):
    before = time.time()
    try:
        if not dry_run:
            res = client.insert(collection_name=collection_name, data=batch)
    except Exception as e:
        logger.exception(f'{type(e)} in client.insert({collection_name}, {len(batch)} items)')
        raise
    elapsed = time.time() - before
    if verbose:
        logger.info(f'Inserting {len(batch)} items took {elapsed:.2f}s')


def read_vdb_section(fh, collection_name, added_documents, client, batch_size, logger, verbose=False, dry_run=False):
    if verbose:
        logger.info(f'Read vdb section')

    batch = []
    for line in fh:
        # End of section
        if 0 == len(line):
            break

        if len(batch) >= batch_size:
            insert_into_vdb(client, collection_name, batch, logger, verbose, dry_run)
            batch = []

        chunk = json.loads(line)

        if chunk['partition'] in added_documents and chunk['file_id'] in added_documents[chunk['partition']]:
            chunk.pop('_id', None)
            batch.append(chunk)

    if len(batch) > 0:
        insert_into_vdb(client, collection_name, batch, logger, verbose, dry_run)


def main():
    """
    Main entry point:
      - Parses CLI arguments.
      - Loads OpenRAG configuration.
      - Connects to RDB (PostgreSQL) and VDB (Milvus).
      - Retrieves and filters partitions.
      - Dumps RDB and VDB data.

    Parameters:
        None (arguments are parsed from sys.argv)

    Returns:
        int: Exit code (0 on success, non-zero on failure).
    """
    def load_openrag_config(logger):
        """
        Loads OpenRAG configuration.

        Parameters:
            logger: Logger instance.

        Returns:
            tuple:
                rdb (dict): Relational database configuration.
                vdb (dict): Vector database configuration.
        """
        from config import load_config

        try:
            config = load_config()
        except Exception as e:
            logger.error(f'Failed while trying to obtain OpenRAG config: {e}')
            raise

        return config['rdb'], config['vectordb']


    # Arguments and configs
    import argparse
    parser = argparse.ArgumentParser(description='OpenRAG restore from backup tool')
    parser.add_argument('-i', '--include-only', nargs='*', help='Include only listed partitions')
    parser.add_argument('-b', '--batch-size', default=1024, type=int, help='Batch size used to iterate Milvus')
    parser.add_argument('-v', '--verbose', default=False, action='store_true', help='Be verbose')
    parser.add_argument('-d', '--dry-run', default=False, action='store_true', help='Don\'t change the target database')
    parser.add_argument('input', help='input file name')


    args = parser.parse_args()

    logger = get_logger()

    rdb, vdb = load_openrag_config(logger)

    if args.verbose:
        logger.info(f'rdb @ {rdb["host"]}:{rdb["port"]} | vdb @ {vdb["host"]}:{vdb["port"]} | collection: {vdb["collection_name"]}')


    # List existing partitions
    try:
        pfm = PartitionFileManager(
            database_url=f"postgresql://{rdb['user']}:{rdb['password']}@{rdb['host']}:{rdb['port']}/partitions_for_collection_{vdb['collection_name']}",
            logger=logger,
        )

        existing_partitions = { item['partition']: item for item in pfm.list_partitions() }
    except Exception as e:
        logger.error(f'Failed while accessing PartitionFileManager at {rdb["host"]}:{rdb["port"]}\n{e}')
        raise


    if args.include_only:
        for part_name in args.include_only:
            if part_name in existing_partitions:
                logger.error(f'Partition "{part_name}" already exists')
                return -1


    client = MilvusClient(uri=f"http://{vdb['host']}:{vdb['port']}")

    if args.input.endswith('.xz'):
        import lzma
        fh = lzma.open(args.input, 'rt', encoding='utf-8')
    else:
        fh = open(args.input, 'rt', encoding='utf-8')


    added_documents = {}

    for line in fh:
        line = line.strip()

        if line in [ 'rdb' ]:
            read_rdb_section(fh, pfm, args.include_only, added_documents, existing_partitions, logger, args.verbose, args.dry_run)

        if line in [ 'vdb' ]:
            read_vdb_section(fh, vdb['collection_name'], added_documents, client, args.batch_size, logger, args.verbose, args.dry_run)


if __name__ == '__main__':
    sys.exit(main())

