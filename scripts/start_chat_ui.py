"""Run local Ari-LLM ChatGPT-style Web UI server."""

import os
import uvicorn
from arillm.api.app import app

def main():
    print("=" * 65)
    print("          Starting Ari-LLM Chat Studio Server")
    print("=" * 65)
    print("--> Web UI available at: http://127.0.0.1:8000")
    print("--> API Docs available at: http://127.0.0.1:8000/docs")
    print("Press Ctrl+C to stop the server.\n")

    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")

if __name__ == "__main__":
    main()
