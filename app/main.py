from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from .routes import indices

load_dotenv()

app = FastAPI(title="Market Map API", version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # For local development
        "https://calm-field-0e1a5ae1e.2.azurestaticapps.net",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(indices.router)

@app.get("/")
async def root():
    return {"message": "Market Map API", "version": "1.0.0"}

@app.get("/debug/env")
def debug_environment():
    """Debug endpoint to check environment variables"""
    import os
    return {
        "mongodb_uri_exists": bool(os.getenv("MONGODB_URI")),
        "db_name": os.getenv("DB_NAME", "not_set"),
        "collection_name": os.getenv("COLLECTION_NAME", "not_set"),
        "environment_vars": list(os.environ.keys())  # List all env vars (names only)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)