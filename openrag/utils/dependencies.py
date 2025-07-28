import ray
import ray.actor
from components import ABCVectorDB
from components.indexer.indexer import Indexer, TaskStateManager
from components.indexer.loaders.pdf_loaders.marker import MarkerPool
from components.indexer.loaders.serializer import SerializerQueue
from components.indexer.vectordb.vectordb import MilvusDB
from config import load_config


def get_or_create_actor(name, cls, namespace="openrag", **options):
    try:
        return ray.get_actor(name, namespace=namespace)
    except ValueError:
        return cls.options(name=name, namespace=namespace, **options).remote()
    except Exception:
        raise


# load config
config = load_config()


def get_task_state_manager():
    return get_or_create_actor(
        "TaskStateManager", TaskStateManager, lifetime="detached"
    )


def get_serializer_queue():
    return get_or_create_actor("SerializerQueue", SerializerQueue)


def get_marker_pool():
    if config.loader.file_loaders.get("pdf") == "MarkerLoader":
        return get_or_create_actor("MarkerPool", MarkerPool)


def get_indexer():
    return get_or_create_actor("Indexer", Indexer)


def get_vectordb() -> ABCVectorDB:
    return get_or_create_actor("Vectordb", MilvusDB)


task_state_manager = get_task_state_manager()
serializer_queue = get_serializer_queue()
vectordb = get_vectordb()
indexer = get_indexer()
marker_pool = get_marker_pool()
