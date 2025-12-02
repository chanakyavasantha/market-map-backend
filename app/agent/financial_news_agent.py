"""
Financial News Agent for fetching, processing, and tagging financial news articles.

This agent:
1. Searches for latest financial news using Google Search grounding
2. Generates structured JSON responses
3. Creates embeddings for deduplication
4. Tags news with market indices from indices_metadata
5. Deduplicates using cosine similarity
6. Saves to MongoDB with proper indexing
"""

import os
import json
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError

from ..models.news_models import NewsArticle, CompanyTicker, NewsSource, NewsCategory, SentimentType

load_dotenv()

# Initialize Google GenAI client
client = genai.Client()

# MongoDB configuration
MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("DB_NAME", "marketmap")
NEWS_COLLECTION = "financial_news"
MAPPINGS_COLLECTION = "index_news_mappings"


class FinancialNewsAgent:
    """Agent for fetching and processing financial news"""
    
    def __init__(self):
        """Initialize the Financial News Agent"""
        self.client = client
        self.grounding_tool = types.Tool(google_search=types.GoogleSearch())
        self.config = types.GenerateContentConfig(tools=[self.grounding_tool])
        
        # Initialize MongoDB
        if not MONGODB_URI:
            raise ValueError("MONGODB_URI environment variable is not set")
        
        self.mongo_client = MongoClient(
            MONGODB_URI,
            tls=True,
            tlsAllowInvalidCertificates=True,
            serverSelectionTimeoutMS=60000,
            connectTimeoutMS=60000,
            socketTimeoutMS=60000
        )
        self.db = self.mongo_client[DB_NAME]
        self.news_collection = self.db[NEWS_COLLECTION]
        self.mappings_collection = self.db[MAPPINGS_COLLECTION]
        
        # Load indices metadata
        self.indices_metadata = self._load_indices_metadata()
        
        # Create indexes for better performance
        self._create_indexes()
        
        print("Financial News Agent initialized successfully")
    
    def _load_indices_metadata(self) -> Dict[str, Dict]:
        """Load indices metadata from JSON file"""
        # Get the path relative to this file's location
        # __file__ is at app/agent/financial_news_agent.py
        # We need to go to app/data/indices_metadata.json
        current_dir = os.path.dirname(os.path.abspath(__file__))
        metadata_path = os.path.join(current_dir, "..", "data", "indices_metadata.json")
        metadata_path = os.path.abspath(metadata_path)
        try:
            with open(metadata_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading indices metadata: {e}")
            print(f"Looking for file at: {metadata_path}")
            return {}
    
    def _create_indexes(self):
        """Create MongoDB indexes for better query performance"""
        try:
            # Index on article ID
            self.news_collection.create_index("id", unique=True)
            # Index on published_at for time-based queries
            self.news_collection.create_index("published_at")
            # Index on affected_indices for filtering
            self.news_collection.create_index("affected_indices")
            # Index on embedding for similarity search (if using vector search)
            # Note: For production, consider using MongoDB Atlas Vector Search
            
            # Index on mappings
            self.mappings_collection.create_index([("index_symbol", 1), ("article_id", 1)], unique=True)
            self.mappings_collection.create_index("index_symbol")
            self.mappings_collection.create_index("article_id")
            
            print("MongoDB indexes created successfully")
        except Exception as e:
            print(f"Error creating indexes: {e}")
    
    def search_latest_financial_news(self, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Search for latest financial news using Google Search grounding
        
        Args:
            max_results: Maximum number of news articles to fetch
            
        Returns:
            List of structured news articles
        """
        print(f"Searching for latest financial news (max {max_results} articles)...")
        
        # Build search prompt for structured JSON response
        search_prompt = f"""
        Search for the latest financial news from the past 24 hours that could impact global stock markets and indices.
        
        Focus on:
        - Major market-moving events (Fed decisions, economic data releases, geopolitical events)
        - Corporate earnings announcements from major companies
        - Economic indicators and data releases
        - Regulatory changes affecting markets
        - Major corporate actions (M&A, IPOs, bankruptcies)
        
        For each news article found, provide a structured JSON response with the following format:
        {{
            "articles": [
                {{
                    "title": "Article headline",
                    "summary": "Brief summary of the news (2-3 sentences)",
                    "content": "Full article content if available, or extended summary",
                    "source": {{
                        "name": "Source name (e.g., Reuters, Bloomberg, WSJ)",
                        "url": "Article URL if available",
                        "published_at": "ISO format timestamp (YYYY-MM-DDTHH:MM:SSZ)"
                    }},
                    "category": "One of: market_moving, earnings, economic_data, corporate_action, regulatory, general",
                    "sentiment": "One of: bullish, bearish, neutral",
                    "sentiment_score": -1.0 to 1.0 (negative for bearish, positive for bullish),
                    "market_impact_level": "HIGH, MEDIUM, or LOW",
                    "affected_indices": ["List of index symbols like ^GSPC, ^NSEI, etc. if mentioned"],
                    "affected_tickers": ["List of company tickers like AAPL, MSFT if mentioned"],
                    "companies": [
                        {{
                            "symbol": "Ticker symbol",
                            "name": "Company name",
                            "exchange": "Exchange name if known",
                            "sector": "Sector if known"
                        }}
                    ],
                    "tags": ["Relevant tags like 'fed', 'earnings', 'inflation', etc."]
                }}
            ]
        }}
        
        Return ONLY valid JSON, no additional text. Include up to {max_results} articles.
        Focus on news from the last 24 hours that has actual or potential market impact.
        """
        
        try:
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=search_prompt,
                config=self.config,
            )
            
            # Parse JSON response
            response_text = response.text.strip()
            
            # Clean response text (remove markdown code blocks if present)
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            # Parse JSON
            news_data = json.loads(response_text)
            articles = news_data.get("articles", [])
            
            print(f"Found {len(articles)} news articles")
            return articles
            
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response: {e}")
            print(f"Response text: {response_text[:500]}")
            return []
        except Exception as e:
            print(f"Error searching for financial news: {e}")
            return []
    
    def _generate_article_id(self, title: str, published_at: str) -> str:
        """Generate unique article ID"""
        # Create hash from title and published_at
        hash_input = f"{title}_{published_at}"
        hash_value = hashlib.md5(hash_input.encode()).hexdigest()[:8]
        timestamp = datetime.now().strftime("%Y%m%d")
        return f"news_{timestamp}_{hash_value}"
    
    def _parse_datetime(self, datetime_str: str) -> datetime:
        """Parse datetime string in various formats"""
        formats = [
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(datetime_str, fmt)
            except ValueError:
                continue
        
        # Default to current time if parsing fails
        return datetime.utcnow()
    
    def _generate_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generate embedding for text using FinBERT model
        FinBERT is a financial domain-specific BERT model for better financial text understanding
        """
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
            
            # Use FinBERT model for financial text embeddings
            # ProsusAI/finbert is trained on financial news and reports, better for embeddings
            model_name = "ProsusAI/finbert"  # Financial BERT model for embeddings
            
            # Lazy load model and tokenizer (cache after first load)
            if not hasattr(self, '_finbert_tokenizer'):
                print("Loading FinBERT model (first time, this may take a moment)...")
                self._finbert_tokenizer = AutoTokenizer.from_pretrained(model_name)
                self._finbert_model = AutoModel.from_pretrained(model_name)
                self._finbert_model.eval()  # Set to evaluation mode
                print("FinBERT model loaded successfully")
            
            # Tokenize and encode text
            inputs = self._finbert_tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self._finbert_model(**inputs)
                # Use mean pooling of last hidden state
                embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()
                # Convert to numpy and then to list
                embedding_list = embeddings.cpu().numpy().tolist()
            
            return embedding_list
        except ImportError as e:
            print(f"Warning: transformers or torch not installed, skipping embedding generation: {e}")
            print("Install with: pip install transformers torch sentencepiece")
            return None
        except Exception as e:
            print(f"Error generating FinBERT embedding: {e}")
            return None
    
    def _load_existing_articles(self, days: int = 7) -> List[Dict]:
        """Load existing articles from MongoDB for deduplication"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        try:
            articles = list(self.news_collection.find({
                "published_at": {"$gte": cutoff_date},
                "embedding": {"$exists": True, "$ne": None}
            }))
            print(f"Loaded {len(articles)} existing articles for deduplication")
            return articles
        except Exception as e:
            print(f"Error loading existing articles: {e}")
            return []
    
    def _check_duplicate(self, new_article: Dict, existing_articles: List[Dict], threshold: float = 0.85) -> bool:
        """
        Check if article is duplicate using cosine similarity
        
        Args:
            new_article: New article with embedding
            existing_articles: List of existing articles with embeddings
            threshold: Similarity threshold (0-1)
            
        Returns:
            True if duplicate found, False otherwise
        """
        new_embedding = new_article.get("embedding")
        if not new_embedding or not existing_articles:
            return False
        
        # Extract embeddings from existing articles
        existing_embeddings = []
        for article in existing_articles:
            emb = article.get("embedding")
            if emb:
                existing_embeddings.append(emb)
        
        if not existing_embeddings:
            return False
        
        # Calculate cosine similarity
        try:
            similarities = cosine_similarity([new_embedding], existing_embeddings)[0]
            max_similarity = float(np.max(similarities))
            
            if max_similarity >= threshold:
                print(f"Duplicate detected with similarity: {max_similarity:.3f}")
                return True
        except Exception as e:
            print(f"Error checking duplicate: {e}")
        
        return False
    
    def _tag_with_indices(self, article: Dict) -> List[str]:
        """
        Tag article with relevant market indices based on content
        
        Args:
            article: Article dictionary
            
        Returns:
            List of index symbols that are relevant
        """
        # Get indices from article if already mentioned
        mentioned_indices = article.get("affected_indices", [])
        
        # Extract text for analysis
        title = article.get("title", "")
        summary = article.get("summary", "")
        content = article.get("content", summary)
        text = f"{title} {content}".lower()
        
        # Map index names/symbols to metadata
        relevant_indices = set(mentioned_indices)
        
        # Check for index mentions in text
        for symbol, metadata in self.indices_metadata.items():
            index_name = metadata.get("name", "").lower()
            country = metadata.get("country", "").lower()
            region = metadata.get("region", "").lower()
            city = metadata.get("city", "").lower()
            
            # Check if index name, country, region, or city is mentioned
            if (index_name in text or 
                country in text or 
                region in text or 
                city in text or
                symbol.replace("^", "").lower() in text):
                relevant_indices.add(symbol)
        
        # Use AI to determine relevance if we have many potential matches
        if len(relevant_indices) > 5:
            # Use AI to filter to most relevant
            relevant_indices = self._ai_filter_indices(article, list(relevant_indices))
        
        return list(relevant_indices)
    
    def _ai_filter_indices(self, article: Dict, candidate_indices: List[str]) -> List[str]:
        """Use AI to filter to most relevant indices"""
        # For now, return top 5 most relevant based on simple heuristics
        # In production, you could use an LLM call here
        return candidate_indices[:5]
    
    def process_articles(self, raw_articles: List[Dict]) -> List[Dict]:
        """
        Process raw articles: generate IDs, embeddings, tag with indices, check duplicates
        
        Args:
            raw_articles: List of raw article dictionaries from search
            
        Returns:
            List of processed articles ready for storage
        """
        print(f"Processing {len(raw_articles)} articles...")
        
        # Load existing articles for deduplication
        existing_articles = self._load_existing_articles()
        
        processed_articles = []
        
        for raw_article in raw_articles:
            try:
                # Generate article ID
                source = raw_article.get("source", {})
                published_at_str = source.get("published_at", datetime.utcnow().isoformat())
                published_at = self._parse_datetime(published_at_str)
                
                article_id = self._generate_article_id(
                    raw_article.get("title", ""),
                    published_at_str
                )
                
                # Create article document
                article_doc = {
                    "id": article_id,
                    "title": raw_article.get("title", ""),
                    "summary": raw_article.get("summary", ""),
                    "content": raw_article.get("content", raw_article.get("summary", "")),
                    "sources": [{
                        "name": source.get("name", "Unknown"),
                        "url": source.get("url"),
                        "published_at": published_at,
                        "credibility_score": None
                    }],
                    "category": raw_article.get("category", "general"),
                    "sentiment": raw_article.get("sentiment", "neutral"),
                    "sentiment_score": float(raw_article.get("sentiment_score", 0.0)),
                    "market_impact_level": raw_article.get("market_impact_level", "LOW"),
                    "affected_indices": raw_article.get("affected_indices", []),
                    "affected_tickers": raw_article.get("affected_tickers", []),
                    "companies": raw_article.get("companies", []),
                    "tags": raw_article.get("tags", []),
                    "published_at": published_at,
                    "fetched_at": datetime.utcnow(),
                    "language": "en"
                }
                
                # Generate embedding for deduplication
                embedding_text = f"{article_doc['title']} {article_doc['summary']}"
                embedding = self._generate_embedding(embedding_text)
                if embedding:
                    article_doc["embedding"] = embedding
                
                # Check for duplicates
                if self._check_duplicate(article_doc, existing_articles):
                    print(f"Skipping duplicate article: {article_doc['title'][:50]}...")
                    continue
                
                # Tag with indices
                tagged_indices = self._tag_with_indices(article_doc)
                article_doc["affected_indices"] = list(set(article_doc["affected_indices"] + tagged_indices))
                
                processed_articles.append(article_doc)
                existing_articles.append(article_doc)  # Add to existing for subsequent checks
                
            except Exception as e:
                print(f"Error processing article: {e}")
                continue
        
        print(f"Processed {len(processed_articles)} unique articles")
        return processed_articles
    
    def save_articles(self, articles: List[Dict]):
        """Save articles to MongoDB and create index mappings"""
        if not articles:
            print("No articles to save")
            return
        
        print(f"Saving {len(articles)} articles to MongoDB...")
        
        saved_count = 0
        mapping_count = 0
        
        for article in articles:
            try:
                # Save article
                self.news_collection.update_one(
                    {"id": article["id"]},
                    {"$set": article},
                    upsert=True
                )
                saved_count += 1
                
                # Create index mappings
                for index_symbol in article.get("affected_indices", []):
                    if index_symbol in self.indices_metadata:
                        mapping = {
                            "index_symbol": index_symbol,
                            "article_id": article["id"],
                            "relevance_score": 0.8,  # Default relevance, could be calculated
                            "mapped_at": datetime.utcnow()
                        }
                        self.mappings_collection.update_one(
                            {"index_symbol": index_symbol, "article_id": article["id"]},
                            {"$set": mapping},
                            upsert=True
                        )
                        mapping_count += 1
                
            except Exception as e:
                print(f"Error saving article {article.get('id')}: {e}")
        
        print(f"Saved {saved_count} articles and {mapping_count} index mappings")
    
    def fetch_and_process_news(self, max_results: int = 10) -> Dict[str, Any]:
        """
        Main method to fetch and process financial news
        
        Args:
            max_results: Maximum number of articles to fetch
            
        Returns:
            Dictionary with processing results
        """
        print("\n" + "="*60)
        print("FETCHING AND PROCESSING FINANCIAL NEWS")
        print("="*60)
        
        # Step 1: Search for news
        raw_articles = self.search_latest_financial_news(max_results=max_results)
        
        if not raw_articles:
            return {
                "status": "no_articles_found",
                "articles_fetched": 0,
                "articles_processed": 0,
                "articles_saved": 0
            }
        
        # Step 2: Process articles
        processed_articles = self.process_articles(raw_articles)
        
        # Step 3: Save to database
        if processed_articles:
            self.save_articles(processed_articles)
        
        return {
            "status": "success",
            "articles_fetched": len(raw_articles),
            "articles_processed": len(processed_articles),
            "articles_saved": len(processed_articles),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def get_news_by_index(self, index_symbol: str, limit: int = 20) -> List[Dict]:
        """
        Get news articles for a specific index
        
        Args:
            index_symbol: Index symbol (e.g., ^GSPC)
            limit: Maximum number of articles to return
            
        Returns:
            List of news articles
        """
        try:
            # Find article IDs mapped to this index
            mappings = list(self.mappings_collection.find(
                {"index_symbol": index_symbol}
            ).sort("mapped_at", -1).limit(limit))
            
            article_ids = [m["article_id"] for m in mappings]
            
            if not article_ids:
                return []
            
            # Fetch articles
            articles = list(self.news_collection.find(
                {"id": {"$in": article_ids}}
            ).sort("published_at", -1).limit(limit))
            
            return articles
            
        except Exception as e:
            print(f"Error fetching news for index {index_symbol}: {e}")
            return []


# Example usage
if __name__ == "__main__":
    agent = FinancialNewsAgent()
    result = agent.fetch_and_process_news(max_results=10)
    print(f"\nProcessing complete: {result}")
