#!/usr/bin/env python3

import sys
import os
import json

from typing import Dict, Any, Tuple, Optional, List, IO

from pymilvus import connections, Collection

from utils.logger import get_logger
from components.indexer.vectordb.utils import PartitionFileManager


def dump_rdb_part(
        out_fh: IO[str],
        pfm: PartitionFileManager,
        partitions: Dict[str, Dict[str, Any]],
        logger: Any,
        verbose: bool = False):
    """
    Dumps relational DB data into the backup file:
      - writes only requested partitions
      - lines are groupped by partition

    Parameters:
        out_fh:      File handle opened for writing.
        pfm:         OpenRAG PartitionFileManager
        partitions:  Mapping of partition name -> partition metadata.
        logger:      Logger instance.
        verbose:     Be verbose.

    Returns:
        None
    """
    for part_name in sorted(partitions):
        # Header
        out_fh.write('rdb\n')
        if verbose:
            logger.info('Writting rdb data')

        # Partition details
        out_fh.write(json.dumps({ 'name': part_name, 'created': partitions[part_name]['created_at'] }, ensure_ascii=False, sort_keys=True) + '\n')

        try:
            files = pfm.list_partition_files(part_name)
        except Exception as e:
            logger.error(f'Failed while requesting the list of files in partition \'{part_name}\'\n{e}')
            raise

        files['files'].sort(key=lambda v: v['file_id'])

        for f in files['files']:
            f.pop('partition', None)
            out_fh.write(json.dumps(f, ensure_ascii=False, sort_keys=True) + '\n')

        # Separator
        out_fh.write('\n')

        if verbose:
            logger.info(f'Partition \'{part_name}\' - {len(files["files"])} files')


def dump_vdb_part(
        out_fh: IO[str],
        collection: Collection,
        partitions: Dict[str, Dict[str, Any]],
        logger: Any,
        batch_size: int = 1024,
        verbose: bool = False):
    """
    Dumps vector DB data into the backup file:
      - writes one chunk per line
      - writes only chunks belonging to partitions requested
      - no particular order guaranteed

    Parameters:
        out_fh:      File handle opened for writing backup data.
        collection:  Milvus collection to query from.
        partitions:  Mapping of partition name -> partition metadata.
        logger:      Logger instance.
        batch_size:  Number of chunks per batch.
        verbose:     Be verbose.

    Returns:
        None
    """
    try:
        collection.load()
    except Exception as e:
        logger.error(f'Failed while loading Milvus collection: {e}')
        raise

    try:
        iterator = collection.query_iterator(
            batch_size=batch_size,        # size of each batch
            output_fields=["*"]           # all fields
        )
    except Exception as e:
        logger.error(f'Failed while trying to obtain Milvus collection iterator: {e}')
        raise

    out_fh.write('vdb\n')
    if verbose:
        logger.info('Writting vdb data')
        cnt = 0

    while True:
        try:
            batch = iterator.next()
        except Exception as e:
            logger.error(f'iterator.next() failed with: {e}')
            raise

        if not batch:  # no more data
            break

        for entity in batch:
            if entity['partition'] in partitions:
                entity.pop('_id', None)
                out_fh.write(json.dumps(entity, ensure_ascii=False, sort_keys=True) + '\n')
                if verbose:
                    cnt += 1

    if verbose:
        logger.info(f'{cnt} chunks written')


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
    parser = argparse.ArgumentParser(description='OpenRAG backup tool')
    parser.add_argument('-i', '--include-only', nargs='*', help='Include only listed partitions')
    parser.add_argument('-o', '--output', required=True, help='Output file name')
    parser.add_argument('-b', '--batch-size', default=1024, type=int, help='Batch size used to iterate Milvus')
    parser.add_argument('-v', '--verbose', default=False, action='store_true', help='Be verbose')

    args = parser.parse_args()

    logger = get_logger()

    rdb, vdb = load_openrag_config(logger)

    if args.verbose:
        logger.info(f'rdb @ {rdb["host"]}:{rdb["port"]} | vdb @ {vdb["host"]}:{vdb["port"]} | collection: {vdb["collection_name"]}')


    # Open output file
    if os.path.isfile(args.output):
        logger.error(f'File \'{args.output}\' already exists.')
        return -1

    # List existing partitions
    try:
        pfm = PartitionFileManager(
            database_url=f"postgresql://{rdb['user']}:{rdb['password']}@{rdb['host']}:{rdb['port']}/partitions_for_collection_{vdb['collection_name']}",
            logger=logger,
        )

        partitions = pfm.list_partitions()
    except Exception as e:
        logger.error(f'Failed while accessing PartitionFileManager at {rdb["host"]}:{rdb["port"]}\n{e}')
        raise

    if args.include_only:
        partitions = [ item for item in partitions if item['partition'] in args.include_only ]
    else:
        partitions = [ item for item in partitions ]

    partitions = { item['partition']: item for item in partitions }

    if 0 == len(partitions):
        logger.error(f'No paritions meet given conditions.')
        return -1

    if args.verbose:
        partitions_str = ', '.join(partitions)
        logger.info(f'partitions: {partitions_str}')


    # Connect to Milvus
    try:
        connections.connect("default", host=vdb['host'], port=vdb['port'])
    except Exception as e:
        logger.error(f'Can\'t connect to Milvus at {vdb["host"]}:{vdb["port"]}\n{e}')
        raise

    try:
        vdb_collection = Collection(vdb['collection_name'])
    except Exception as e:
        logger.error(f'Can\'t access Milvus collection {vdb["collection_name"]} at {vdb["host"]}:{vdb["port"]}\n{e}')
        raise

    with open(args.output, 'wt', encoding='utf-8') as out_fh:
        # Dump data from RDB (one line per document)
        dump_rdb_part(out_fh, pfm, partitions, logger, args.verbose)

        # Dump data from VDB (one line per chunk)
        dump_vdb_part(out_fh, vdb_collection, partitions, logger, args.batch_size, args.verbose)


    return 0


if __name__ == '__main__':
    sys.exit(main())

