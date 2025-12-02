from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class NewsCategory(str, Enum):
    """Categories for financial news"""
    MARKET_MOVING = "market_moving"
    EARNINGS = "earnings"
    ECONOMIC_DATA = "economic_data"
    CORPORATE_ACTION = "corporate_action"
    REGULATORY = "regulatory"
    GENERAL = "general"


class SentimentType(str, Enum):
    """Sentiment classification"""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class CompanyTicker(BaseModel):
    """Model for company ticker information"""
    symbol: str = Field(..., description="Stock ticker symbol (e.g., AAPL, MSFT)")
    name: str = Field(..., description="Company name")
    exchange: Optional[str] = Field(None, description="Stock exchange (e.g., NASDAQ, NYSE)")
    sector: Optional[str] = Field(None, description="Industry sector")
    market_cap: Optional[float] = Field(None, description="Market capitalization")
    country: Optional[str] = Field(None, description="Country of incorporation")
    
    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "AAPL",
                "name": "Apple Inc.",
                "exchange": "NASDAQ",
                "sector": "Technology",
                "market_cap": 3000000000000,
                "country": "United States"
            }
        }


class NewsSource(BaseModel):
    """Model for news source information"""
    name: str = Field(..., description="Source name (e.g., Reuters, Bloomberg)")
    url: Optional[str] = Field(None, description="Article URL")
    published_at: Optional[datetime] = Field(None, description="Publication timestamp")
    credibility_score: Optional[float] = Field(None, ge=0, le=1, description="Source credibility score")


class NewsArticle(BaseModel):
    """Model for financial news article with embedding"""
    id: str = Field(..., description="Unique article identifier")
    title: str = Field(..., description="Article title")
    summary: str = Field(..., description="Article summary")
    content: Optional[str] = Field(None, description="Full article content")
    sources: List[NewsSource] = Field(default_factory=list, description="News sources")
    
    # Categorization
    category: NewsCategory = Field(default=NewsCategory.GENERAL, description="News category")
    sentiment: SentimentType = Field(default=SentimentType.NEUTRAL, description="Overall sentiment")
    sentiment_score: float = Field(default=0.0, ge=-1, le=1, description="Sentiment score (-1 to 1)")
    
    # Market impact
    market_impact_level: str = Field(default="LOW", description="Market impact: HIGH, MEDIUM, LOW")
    affected_indices: List[str] = Field(default_factory=list, description="List of affected index symbols")
    affected_tickers: List[str] = Field(default_factory=list, description="List of affected company tickers")
    
    # Company information
    companies: List[CompanyTicker] = Field(default_factory=list, description="Related companies")
    
    # Embedding for similarity search
    embedding: Optional[List[float]] = Field(None, description="Article embedding vector for similarity search")
    
    # Timestamps
    published_at: datetime = Field(..., description="Publication timestamp")
    fetched_at: datetime = Field(default_factory=datetime.utcnow, description="When article was fetched")
    
    # Metadata
    tags: List[str] = Field(default_factory=list, description="Additional tags")
    language: str = Field(default="en", description="Article language")
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "news_20240101_001",
                "title": "Fed Raises Interest Rates by 0.25%",
                "summary": "The Federal Reserve announced a 0.25% interest rate increase...",
                "category": "market_moving",
                "sentiment": "bearish",
                "sentiment_score": -0.6,
                "market_impact_level": "HIGH",
                "affected_indices": ["^GSPC", "^DJI"],
                "affected_tickers": [],
                "companies": [],
                "published_at": "2024-01-01T14:00:00Z",
                "tags": ["fed", "interest_rates", "monetary_policy"]
            }
        }


class IndexNewsMapping(BaseModel):
    """Model for mapping news articles to market indices"""
    index_symbol: str = Field(..., description="Index symbol (e.g., ^GSPC)")
    article_id: str = Field(..., description="News article ID")
    relevance_score: float = Field(..., ge=0, le=1, description="Relevance score (0-1)")
    mapped_at: datetime = Field(default_factory=datetime.utcnow, description="When mapping was created")
    
    class Config:
        json_schema_extra = {
            "example": {
                "index_symbol": "^GSPC",
                "article_id": "news_20240101_001",
                "relevance_score": 0.95,
                "mapped_at": "2024-01-01T14:00:00Z"
            }
        }

