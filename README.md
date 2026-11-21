# Orbit Backend Service

Production-ready backend API service for the Orbit Agent Workspace.

## Tech Stack
- **Framework**: FastAPI (Python 3.11)
- **ASGI Server**: Uvicorn
- **Database**: Neon Serverless PostgreSQL (`psycopg3`, `psycopg_pool`)
- **Memory & Graph**: LangGraph Checkpointer, Qdrant Cloud Vector Store
- **Cache & Scheduling**: Upstash Redis, Upstash QStash
- **Object Storage**: S3-compatible Tigris Storage

## Running Locally

1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables in `.env` (refer to `.env.example`).

4. Run the server:
   ```bash
   uvicorn server:app --host 0.0.0.0 --port 8000
   ```

## API Documentation
- Interactive Swagger UI: `/docs`
- OpenAPI Schema: `/openapi.json`
- Health check: `/health`
