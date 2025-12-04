from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

class DatabaseManager:
    def __init__(self, database_url: str):
        if database_url.startswith("postgresql://"):
            database_url = database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
        
        self.engine = create_async_engine(
            database_url,
            echo=False,
            pool_size=20,
            max_overflow=10,
            pool_pre_ping=True,
        )
        self.async_session = async_sessionmaker(self.engine, expire_on_commit=False, class_=AsyncSession)