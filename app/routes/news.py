"""
Routes for financial news operations
"""

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from datetime import datetime
from typing import List, Optional
from ..agent.financial_news_agent import FinancialNewsAgent

router = APIRouter(prefix="/api/news", tags=["news"])

@router.post("/fetch")
async def fetch_news(
    background_tasks: BackgroundTasks,
    max_results: int = Query(default=10, ge=1, le=50, description="Maximum number of articles to fetch")
):
    """
    Trigger news fetching and processing (runs in background)
    """
    try:
        agent = FinancialNewsAgent()
        
        # Run in background
        background_tasks.add_task(agent.fetch_and_process_news, max_results=max_results)
        
        return {
            "status": "started",
            "message": f"News fetching started (max {max_results} articles)",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting news fetch: {str(e)}")

@router.get("/latest")
async def get_latest_news(
    limit: int = Query(default=20, ge=1, le=100, description="Maximum number of articles to return"),
    category: Optional[str] = Query(None, description="Filter by category"),
    sentiment: Optional[str] = Query(None, description="Filter by sentiment (bullish, bearish, neutral)")
):
    """Get latest financial news articles"""
    try:
        agent = FinancialNewsAgent()
        
        # Build query
        query = {}
        if category:
            query["category"] = category
        if sentiment:
            query["sentiment"] = sentiment
        
        # Fetch articles
        articles = list(agent.news_collection.find(query)
                       .sort("published_at", -1)
                       .limit(limit))
        
        # Convert to JSON-serializable format
        for article in articles:
            if "_id" in article:
                article["_id"] = str(article["_id"])
            if "published_at" in article and isinstance(article["published_at"], datetime):
                article["published_at"] = article["published_at"].isoformat()
            if "fetched_at" in article and isinstance(article["fetched_at"], datetime):
                article["fetched_at"] = article["fetched_at"].isoformat()
            # Remove embedding from response
            if "embedding" in article:
                del article["embedding"]
        
        return {
            "articles": articles,
            "count": len(articles),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching latest news: {str(e)}")

@router.get("/{article_id}")
async def get_article(article_id: str):
    """Get a specific news article by ID"""
    try:
        agent = FinancialNewsAgent()
        article = agent.news_collection.find_one({"id": article_id})
        
        if not article:
            raise HTTPException(status_code=404, detail=f"Article {article_id} not found")
        
        # Convert to JSON-serializable format
        if "_id" in article:
            article["_id"] = str(article["_id"])
        if "published_at" in article and isinstance(article["published_at"], datetime):
            article["published_at"] = article["published_at"].isoformat()
        if "fetched_at" in article and isinstance(article["fetched_at"], datetime):
            article["fetched_at"] = article["fetched_at"].isoformat()
        # Remove embedding from response
        if "embedding" in article:
            del article["embedding"]
        
        return article
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching article: {str(e)}")

