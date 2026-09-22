from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="TECHNOVA_AI_",
        extra="ignore",
    )

    data_dir: Path = Path("data")
    artifact_dir: Path = Path("artifacts")

    @property
    def raw_retail_path(self) -> Path:
        return self.data_dir / "raw" / "online_retail" / "Online Retail.xlsx"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
