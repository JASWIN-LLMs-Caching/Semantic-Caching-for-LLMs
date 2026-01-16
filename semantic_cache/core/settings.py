from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    # Redis
    REDIS_HOST: str
    REDIS_PORT: int
    REDIS_INDEX_NAME: str

    # Milvus
    MILVUS_HOST: str
    MILVUS_PORT: int
    MILVUS_COLLECTION_NAME: str

    VECTOR_DIM: int

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()

