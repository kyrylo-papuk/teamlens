from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str = ""
    jira_base_url: str = ""
    jira_email: str = ""
    jira_api_token: str = ""
    jira_project_key: str = "PAY"
    jira_team_key: str = "PAY"  # HRM team that works in this Jira project
    langfuse_public_key: str = ""  # tracing is off unless both keys are set
    langfuse_secret_key: str = ""
    langfuse_host: str = ""
    people_agent_url: str = "http://localhost:9101/"
    people_agent_token: str = "dev-only-token"  # set a real secret outside local runs
