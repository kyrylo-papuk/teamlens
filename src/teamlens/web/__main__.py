import os

import uvicorn

from teamlens.web.app import create_app

if __name__ == "__main__":
    # Local runs listen on this machine only; a container sets TEAMLENS_HOST=0.0.0.0.
    # Hosting platforms pass the port to listen on in PORT.
    uvicorn.run(create_app(), host=os.environ.get("TEAMLENS_HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "8000")),
                log_level="warning")
