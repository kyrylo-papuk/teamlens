"""Run the People agent:  python -m teamlens.people_agent   (http://localhost:9101)"""

from __future__ import annotations

import asyncio
import os

import uvicorn
from mcp import Client

from teamlens import live
from teamlens.config import Settings
from teamlens.hrm import Hrm
from teamlens.mcp_servers import hrm_server
from teamlens.people_agent.advisor import build_advisor
from teamlens.people_agent.service import READ, Caller, build_app
from teamlens.seed import generate


async def main() -> None:
    s = Settings()
    c = generate() if os.environ.get("TEAMLENS_FAKE") == "1" else live.company()
    hrm = Hrm(c.employees, c.absences)
    tokens = {s.people_agent_token: Caller(name="delivery-agent", scopes={READ})}
    async with Client(hrm_server(hrm, c.today, c.sprint_start, c.sprint_end)) as hrm_client:
        advisor = build_advisor(hrm_client, s.anthropic_api_key, c.today) if s.anthropic_api_key else None
        app = build_app(hrm, c.today, tokens, s.people_agent_url, advisor)
        host = os.environ.get("TEAMLENS_HOST", "127.0.0.1")  # a container sets 0.0.0.0
        await uvicorn.Server(uvicorn.Config(app, host=host, port=9101, log_level="warning")).serve()


if __name__ == "__main__":
    asyncio.run(main())
