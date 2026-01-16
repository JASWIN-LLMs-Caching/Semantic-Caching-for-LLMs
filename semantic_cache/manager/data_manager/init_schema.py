from redis import Redis
from redis.exceptions import RedisError

from pymilvus import (
    connections,
    utility,
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType,
    MilvusException,
)

from semantic_cache.core.settings import settings
from semantic_cache.utils.logger import logger
from redis.commands.search.field import TagField, VectorField, NumericField
from redis.commands.search.index_definition import IndexDefinition, IndexType


# -------------------------------------------------------------------
# Redis Schema Initialization
# -------------------------------------------------------------------
def init_redis_schema() -> None:
    """
    Initialize Redis Search Index for semantic cache.
    MUST be idempotent and schema-safe.
    """
    try:
        redis = Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            decode_responses=False,
        )
        redis.ping()
    except RedisError as e:
        raise RedisError(f"Redis connection failed: {e}") from e

    try:
        info = redis.ft(settings.REDIS_INDEX_NAME).info()
        logger.info("[Redis] Index '%s' already exists", settings.REDIS_INDEX_NAME)

        attributes = {attr["attribute"] for attr in info["attributes"]}
        required_fields = {"course_id", "vector", "access_count", "last_accessed"}

        if not required_fields.issubset(attributes):
            raise RedisError(
                f"Redis index schema mismatch. "
                f"Expected {required_fields}, found {attributes}"
            )

        logger.info("[Redis] Schema validation OK")

    except RedisError:
        logger.info("[Redis] Creating index '%s'", settings.REDIS_INDEX_NAME)

        schema = (
            TagField("course_id"),
            VectorField(
                "vector",
                "HNSW",
                {
                    "TYPE": "FLOAT32",
                    "DIM": settings.VECTOR_DIM,
                    "DISTANCE_METRIC": "COSINE",
                    "M": 16,
                    "EF_CONSTRUCTION": 200,
                },
            ),
            NumericField("access_count"),
            NumericField("last_accessed"),
        )

        definition = IndexDefinition(
            prefix=["cache:"],
            index_type=IndexType.HASH,
        )

        try:
            redis.ft(settings.REDIS_INDEX_NAME).create_index(
                schema,
                definition=definition,
                stopwords=[],
            )
            logger.info("[Redis] Index created successfully")
        except RedisError as e:
            raise RedisError(f"Failed to create Redis index: {e}") from e


# -------------------------------------------------------------------
# Milvus Schema Initialization
# -------------------------------------------------------------------
def init_milvus_schema() -> None:
    """
    Initialize Milvus collection for LMS semantic knowledge base.
    MUST be idempotent, partition-safe, and schema-locked.
    """
    try:
        connections.connect(
            alias="default",
            host=settings.MILVUS_HOST,
            port=settings.MILVUS_PORT,
        )
    except MilvusException as e:
        raise MilvusException(f"Milvus connection failed: {e}") from e

    if utility.has_collection(settings.MILVUS_COLLECTION_NAME):
        logger.info(
            "[Milvus] Collection '%s' already exists",
            settings.MILVUS_COLLECTION_NAME,
        )

        col = Collection(settings.MILVUS_COLLECTION_NAME)
        fields = {f.name: f for f in col.schema.fields}

        # ---- Schema validation (HARD FAILS) ----
        if "course_id" not in fields or not fields["course_id"].is_partition_key:
            raise MilvusException(
                "Milvus schema invalid: 'course_id' must be a partition key"
            )

        if "vector" not in fields:
            raise MilvusException("Milvus schema invalid: missing 'vector' field")

        if fields["vector"].dim != settings.VECTOR_DIM:
            raise MilvusException(
                f"Vector dimension mismatch: expected {settings.VECTOR_DIM}, "
                f"found {fields['vector'].dim}"
            )

        logger.info("[Milvus] Schema validation OK")
        return

    logger.info(
        "[Milvus] Creating collection '%s'",
        settings.MILVUS_COLLECTION_NAME,
    )

    fields = [
        FieldSchema(
            name="id",
            dtype=DataType.INT64,
            is_primary=True,
            auto_id=True,
        ),
        FieldSchema(
            name="course_id",
            dtype=DataType.INT64,
            is_partition_key=True,
        ),
        FieldSchema(
            name="vector",
            dtype=DataType.FLOAT_VECTOR,
            dim=settings.VECTOR_DIM,
        ),
        FieldSchema(
            name="doc_ref_id",
            dtype=DataType.VARCHAR,
            max_length=128,
        ),
        FieldSchema(
            name="chunk_index",
            dtype=DataType.INT32,
        ),
        FieldSchema(
            name="created_at",
            dtype=DataType.INT64,
            description="Unix timestamp",
        ),
    ]

    schema = CollectionSchema(
        fields,
        description="LMS secured semantic course content",
    )

    try:
        col = Collection(
            name=settings.MILVUS_COLLECTION_NAME,
            schema=schema,
            consistency_level="Strong",
        )

        index_params = {
            "metric_type": "COSINE",
            "index_type": "HNSW",
            "params": {
                "M": 16,
                "efConstruction": 200,
            },
        }

        col.create_index("vector", index_params)
        logger.info("[Milvus] Collection created and indexed")

    except MilvusException as e:
        raise MilvusException(f"Failed to create Milvus collection: {e}") from e

