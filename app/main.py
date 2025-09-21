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
        "http://localhost:3000",  # React dev server
        "https://calm-field-0e1a5ae1e.azurestaticapps.net",  # Your Azure Static Web App
        "https://your-custom-domain.com"  # If you have a custom domain
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=80)