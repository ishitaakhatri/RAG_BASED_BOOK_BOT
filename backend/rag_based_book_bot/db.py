from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase
from app_config import get_settings  # <--- CHANGED: Use get_settings instead of get_config

# Load the Pydantic Settings object (AppSettings)
settings = get_settings()

# Fix Render/Heroku style URLs for AsyncPG (they often provide postgres://)
# CHANGED: Access DATABASE_URL directly from the flat settings object
db_url = settings.DATABASE_URL

if db_url:
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql+asyncpg://", 1)
    elif not db_url.startswith("postgresql+asyncpg://") and "sqlite" not in db_url:
        # Fallback for standard postgresql://
        db_url = db_url.replace("postgresql://", "postgresql+asyncpg://", 1)

# Default to sqlite for local dev if no URL provided
if not db_url:
    db_url = "sqlite+aiosqlite:///./local_dev.db"

# CHANGED: Derive echo_sql from the LOG_LEVEL or APP_ENV
# If log level is DEBUG or env is development, we echo SQL.
should_echo = (settings.LOG_LEVEL.upper() == "DEBUG") or (settings.APP_ENV == "development")

engine = create_async_engine(
    db_url,
    echo=should_echo,
    pool_size=20,
    max_overflow=10
)

# SQLAlchemy 2.0 Async Session Factory
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False
)

class Base(DeclarativeBase):
    pass

async def init_db():
    """Create tables if they don't exist"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

async def get_db():
    """Dependency for dependency injection"""
    async with AsyncSessionLocal() as session:
        yield session