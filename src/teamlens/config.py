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
    # A public demo has no login, so anyone can make it call the model. This caps the requests per day that do;
    # 0 means no cap (local use).
    teamlens_daily_model_requests: int = 0
    teamlens_notice: str = ""  # one line shown at the top of the page, e.g. "Public demo on fictional data"
    # Upstash Redis over REST, under either pair of names its integrations use. Set on the public demo only,
    # where several instances have to share one copy of the fictional company: see web/shared.py.
    kv_rest_api_url: str = ""
    kv_rest_api_token: str = ""
    upstash_redis_rest_url: str = ""
    upstash_redis_rest_token: str = ""
    people_agent_url: str = "http://localhost:9101/"
    people_agent_token: str = "dev-only-token"  # set a real secret outside local runs
