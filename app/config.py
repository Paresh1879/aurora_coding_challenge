from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    openai_api_key: str
    member_api_url: str = "https://november7-730026606190.europe-west1.run.app"
    openai_model: str = "gpt-4"
    
    class Config:
        env_file = ".env"

settings = Settings()

