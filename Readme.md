# Market Map Backend

FastAPI backend for the Market Map application.

## Prerequisites

- Python 3.9+
- MongoDB database (local or cloud)
- Environment variables configured

## Setup

### 1. Activate Virtual Environment

```bash
cd backend
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows
```

### 2. Install Dependencies

If you haven't already installed dependencies:

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file in the `backend` directory with the following variables:

```env
# MongoDB Configuration
MONGODB_URI=your_mongodb_connection_string
DB_NAME=marketmap
COLLECTION_NAME=indices
NEWS_COLLECTION_NAME=financial_news

# OpenAI API (for embeddings - optional but recommended)
OPENAI_API_KEY=your_openai_api_key

# Google GenAI (for news search)
# Make sure you have GOOGLE_API_KEY set in your environment or .env
```

### 4. Run the Server

#### Option 1: Using uvicorn directly (Recommended)

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

#### Option 2: Using Python directly

```bash
python -m app.main
```

#### Option 3: Using the module syntax

```bash
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

Once running, the API will be available at `http://localhost:8000`

- **API Docs**: http://localhost:8000/docs (Swagger UI)
- **Alternative Docs**: http://localhost:8000/redoc (ReDoc)
- **Root**: http://localhost:8000/

### Main Endpoints

- `GET /api/indices/status` - Get all indices with market data
- `GET /api/indices/{symbol}` - Get specific index details
- `GET /api/indices/{symbol}/news` - Get news for a specific index
- `GET /api/indices/with-news` - Get all indices that have news
- `POST /api/news/fetch` - Trigger news fetching
- `GET /api/news/latest` - Get latest news articles

## Development

### Hot Reload

The `--reload` flag enables automatic reloading when code changes are detected.

### Testing

Test the API endpoints:

```bash
# Test root endpoint
curl http://localhost:8000/

# Test indices status
curl http://localhost:8000/api/indices/status

# Test news fetching
curl -X POST http://localhost:8000/api/news/fetch?max_results=10
```

## Troubleshooting

### Port Already in Use

If port 8000 is already in use, change it:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8001
```

### MongoDB Connection Issues

1. Check your `MONGODB_URI` is correct
2. Ensure MongoDB is running (if local)
3. Check network connectivity (if cloud)
4. Verify credentials are correct

### Missing Dependencies

If you get import errors:

```bash
pip install -r requirements.txt
```

### Environment Variables Not Loading

Make sure:
1. `.env` file exists in the `backend` directory
2. `python-dotenv` is installed
3. You're running from the correct directory

## Production Deployment

For production, use gunicorn with uvicorn workers:

```bash
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

