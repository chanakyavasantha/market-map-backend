# Standard library imports
import os
import json
import time
from datetime import datetime, timedelta, timezone as dt_timezone
import jsonschema

# Third-party imports
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from openai import OpenAI
from pymongo import MongoClient, UpdateOne
from pytz import timezone
from pydantic import BaseModel

import resend

# AI Modules
from google import genai

# Local application imports
from .utils.market_prompts import MarketPromptManager
from .utils.market_schema import MarketDataSchema

# Load environment variables from .env file
load_dotenv()


# Configure Resend
RESEND_API_KEY = os.getenv("RESEND_API_KEY")
if not RESEND_API_KEY:
    raise ValueError("Please set RESEND_API_KEY in your .env file")
resend.api_key = RESEND_API_KEY

# Configure email recipients
EMAIL_RECIPIENTS = [
    "chanakyavasantha@gmail.com",
]

# MongoDB configuration
MONGO_DB_URI = os.getenv("MONGODB_URI")
MONGO_DB = os.getenv("DB_NAME")
MONGO_COLLECTION_MARKET = os.getenv("NEWS_COLLECTION_NAME")

# Market category mapping
MARKET_CATEGORY_MAP = {
    'MMH': 'market_moving_headlines',
    'EDR': 'economic_data_releases', 
    'CEV': 'corporate_events',
}

class MarketNewsAgent:
    """
    Market News Agent for collecting and processing market-moving financial news.
    
    This agent searches for market-relevant news articles, validates their market impact,
    grounds their datetime information with market session context, and stores them in MongoDB.
    
    Key Features:
    - Searches for market-moving headlines using OpenAI with web search
    - Filters articles by market impact relevance
    - Grounds article datetime with trading session context (pre-market, market hours, etc.)
    - Generates embeddings for similarity search and deduplication
    - Provides sentiment analysis and volatility scoring
    - Sends email notifications with market intelligence summaries
    
    Market Focus Areas:
    - Market Moving Headlines: Fed decisions, major economic data, geopolitical events
    - Economic Data Releases: Employment, inflation, GDP, consumer data
    - Corporate Events: Earnings, M&A, management changes, regulatory approvals
    
    Grounding Policy:
    - Articles are rejected if datetime grounding fails (no fallback to current time)
    - Only articles with successfully grounded datetime and market session info are processed
    - Market session context is crucial for understanding trading impact
    """
    
    def __init__(self):
        """Initialize the Market News Agent"""
        self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
        
        # Initialize MongoDB
        self.mongo_client = MongoClient(
            MONGO_DB_URI,
            connectTimeoutMS=60000,
            serverSelectionTimeoutMS=60000,
            socketTimeoutMS=60000
        )
        
        self.mongo_db = self.mongo_client[MONGO_DB]
        self.articles_col = self.mongo_db[MONGO_COLLECTION_MARKET]
        
        # Market news schema
        self.market_schema = self._get_market_data_schema()
        print("Market data schema loaded for validation")
        print("Market News Agent initialized successfully")

    def _get_market_data_schema(self):
        """Return the JSON schema for market news data"""
        return MarketDataSchema.get_schema()

    def load_market_prompts(self):
        """Load market news prompts for valid categories only."""
        valid_categories = [k for k in MARKET_CATEGORY_MAP.keys()]
        prompts = {
            cat: getattr(MarketPromptManager, cat)
            for cat in valid_categories
            if hasattr(MarketPromptManager, cat)
        }
        print(f"Loaded {len(prompts)} market prompts successfully")
        return prompts

    def _load_market_definition(self):
        """Load market news definition from prompts"""
        return MarketPromptManager.market_news_definition

    def search_for_market_titles(self, category, prompt_content, max_titles=5):
        """
        Phase 1: Search for relevant market news titles/headlines only
        
        Args:
            category (str): The market category (MMH, EDR, CEV)
            prompt_content (str): The category definition
            max_titles (int): Maximum number of titles to return
            
        Returns:
            list: List of market news titles with basic metadata
        """
        try:
            # Get current UTC date and time for context
            current_utc = self._get_utc_now()
            current_utc_str = current_utc.strftime('%Y-%m-%d %H:%M:%S UTC')
            
            # Determine market session
            market_session = self._get_market_session(current_utc)
            
            # Build focused search prompt for market news titles
            title_search_prompt = f"""
            Search for the most important market-moving news from the past 24 hours that match this category.

            CURRENT TIME CONTEXT: {current_utc_str}
            CURRENT MARKET SESSION: {market_session}

            MARKET CATEGORY: {category} - {MARKET_CATEGORY_MAP.get(category, 'unknown')}
            
            CATEGORY DEFINITION:
            {prompt_content}

            SEARCH INSTRUCTIONS:
            1. Find ONLY confirmed news events that have moved or will move equity markets
            2. Focus on news from the last 24 hours that affects US equity markets
            3. Return ONLY {max_titles} most market-impactful events
            4. Each result should be a distinct, separate market event
            5. Prioritize news that affects major indices (S&P 500, NASDAQ, Dow)
            6. Include market session timing when the event occurred or was announced

            For each news item, provide:
            "title": "Exact headline/title from financial news source",
            "brief_description": "One sentence summary of market impact",
            "market_impact_level": "HIGH|MEDIUM|LOW",
            "primary_source": "Name of financial news source (Reuters, Bloomberg, WSJ, etc.)",
            "news_type": "Specific type within {category}",
            "affected_indices": "Which major indices are affected (S&P 500, NASDAQ, Dow)",
            "expected_volatility": "High Vol Expected|Moderate Impact|Low Impact",
            "preliminary_timing": "When the event occurred or was announced"

            CRITICAL MARKET RELEVANCE:
            - Return ONLY confirmed market events with actual or expected price impact
            - Each title must represent a DIFFERENT market-moving event
            - Prioritize recency (last 24 hours) and immediate market impact potential
            - Focus on actionable market intelligence for traders and investors

            Return valid JSON array only, no additional text.
            """
            
            # Use OpenAI with web search
            response = self.openai_client.responses.parse(
                model="gpt-4o",
                input=[
                    {
                        "role": "system", 
                        "content": title_search_prompt
                    }
                ],
                reasoning={},
                tools=[
                    {
                        "type": "web_search_preview",
                        "user_location": {
                            "type": "approximate"
                        },
                        "search_context_size": "medium"
                    }
                ],
                temperature=0.7,
                max_output_tokens=8000,
                top_p=1,
                store=True,
                text_format=MultiMarketTitleSearch
            )
            
            # Extract response content
            response_json = response.output_parsed.model_dump()
            titles = response_json.get('titles', [])
            
            print(f"Found {len(titles)} market news titles for category {category}")
            
            # Add metadata
            for title_obj in titles:
                title_obj['search_category'] = category
                title_obj['search_timestamp'] = current_utc_str
                title_obj['market_session'] = market_session
                title_obj['phase'] = 'title_search'
            
            return titles
            
        except Exception as e:
            print(f"Error searching for market titles: {e}")
            return []

    def search_detailed_market_article(self, title_obj, prompt_content):
        """
        Phase 2: Deep dive search for specific market news to get full article details
        
        Args:
            title_obj (dict): Title object from Phase 1
            prompt_content (str): Category definition for context
            
        Returns:
            dict: Detailed market article following MarketData schema or None if not found
        """
        try:
            title = title_obj.get('title', '')
            category = title_obj.get('search_category', '')
            impact_level = title_obj.get('market_impact_level', '')
            
            # Build detailed search prompt
            detailed_search_prompt = f"""
            Search for comprehensive details about this specific market-moving event and structure according to MarketData schema.

            SPECIFIC EVENT TO RESEARCH:
            Title: "{title}"
            Impact Level: "{impact_level}"
            Category: {category} - {MARKET_CATEGORY_MAP.get(category, 'unknown')}

            CATEGORY CONTEXT:
            {prompt_content}

            SEARCH TASK:
            1. Find the exact article or related coverage about this specific market event
            2. Gather comprehensive market impact analysis
            3. Structure response according to MarketData schema with market intelligence

            REQUIRED MARKET ANALYSIS:
            - Sentiment analysis (-1 to +1 scale) for market impact
            - Expected volatility impact and time horizon (immediate, short-term, etc.)
            - Affected market indices, sectors, and individual stocks
            - Technical levels if mentioned (support/resistance for indices)
            - Market session timing relevance and trading implications
            - Forward-looking market implications and trader considerations
            - Options activity and institutional positioning if relevant

            MarketData Schema Requirements:
            {json.dumps(self._get_market_data_schema(), indent=2)}

            CRITICAL MARKET INTELLIGENCE REQUIREMENTS:
            - Generate unique ID using format: "MKT-{category}-{timestamp}-{hash}"
            - Include precise market impact assessment with quantified expectations
            - Provide actionable intelligence for trading decisions
            - Include sentiment breakdown (bullish/bearish indicators)
            - Assess market session timing impact on price discovery
            - Identify key price levels and volatility expectations
            - Include institutional vs retail impact considerations

            INSTRUCTIONS:
            - Focus specifically on this market event: "{title}"
            - Provide detailed market impact analysis with trading implications
            - Include technical analysis and key price levels if available
            - Generate comprehensive sentiment and volatility assessment
            - If this specific event cannot be verified with market impact, return null

            Return valid JSON object following the MarketData schema, or null if event cannot be verified.
            """
            
            response = self.openai_client.responses.create(
                model="gpt-4o",
                input=[
                    {
                        "role": "system", 
                        "content": detailed_search_prompt
                    }
                ],
                tools=[
                    {
                        "type": "web_search_preview",
                        "user_location": {
                            "type": "approximate"
                        },
                        "search_context_size": "large"
                    }
                ],
                temperature=0.3,
                max_output_tokens=12000,
                top_p=1,
                store=True
            )
            
            # Parse the detailed article
            response_text = response.output[0].content[0].text
            article = self._parse_json_response(response_text, expect_object=True)
            
            if article:
                # Add metadata from title search
                article['_metadata'] = {
                    'phase': 'detailed_search',
                    'original_title_search': title_obj,
                    'search_timestamp': self._get_utc_now().isoformat(),
                    'market_session': title_obj.get('market_session', 'unknown')
                }
                print(f"Successfully retrieved detailed market article for: {title[:50]}...")
                return article
            else:
                print(f"Could not verify or find detailed market information for: {title[:50]}...")
                return None
                
        except Exception as e:
            print(f"Error getting detailed market article for '{title[:50]}...': {e}")
            return None

    def classify_market_relevance(self, article, category=None):
        """
        Determine if an article is truly relevant for market trading decisions
        
        Args:
            article (dict): The article to classify
            category (str): The market category to classify against
        """
        try:
            # Extract title and content
            title_obj = article.get('title', {})
            title = title_obj.get('title', '') if isinstance(title_obj, dict) else str(title_obj)
            
            # Extract summary from sources
            summary = ""
            sources = article.get('sources', [])
            if sources and isinstance(sources, list) and len(sources) > 0:
                first_source = sources[0]
                if isinstance(first_source, dict):
                    summary = first_source.get('summary', '')
            
            # Extract market impact information
            market_impact_text = ""
            market_impact = article.get('marketImpact', {})
            if market_impact:
                impact_level = market_impact.get('level', '')
                affected_indices = market_impact.get('affectedIndices', [])
                expected_vol = market_impact.get('expectedVolatility', '')
                market_impact_text = f"Impact Level: {impact_level}, Indices: {', '.join(affected_indices)}, Volatility: {expected_vol}"
            
            # Use provided category or get from article
            if category is None:
                category = article.get('marketCategory', 'unknown')
            
            # Build classification prompt
            classification_prompt = f"""
            Evaluate if this financial news article has genuine market-moving potential for equity markets and active trading decisions.

            Article Information:
            - Title: {title}
            - Summary: {summary}
            - Market Impact Info: {market_impact_text}
            - Category: {category}
            
            MARKET RELEVANCE CRITERIA:
            Answer "true" ONLY if the article:
            - Reports on specific events that typically cause immediate stock/index price movements
            - Contains NEW market-moving information not already priced in
            - Affects major companies (S&P 500), sectors, or overall market indices
            - Involves Fed policy, major economic data releases, or geopolitical market events
            - Discusses earnings surprises, M&A announcements, or major corporate events
            - Has clear implications for trading decisions within hours/days
            - Shows quantifiable market impact (price movements, volume spikes, volatility)
            
            Answer "false" if the article:
            - Is general market commentary without actionable new information
            - Discusses theoretical scenarios or historical market analysis
            - Is educational content about markets/trading without current events
            - Reports on minor companies with minimal market impact
            - Contains opinion pieces without concrete market-moving catalysts
            - Shows outdated information already reflected in current prices
            - Lacks specific, actionable market intelligence for traders

            MARKET IMPACT FOCUS:
            Consider whether this news would cause:
            - Major index movements (>0.5% S&P 500)
            - Sector rotation or significant sector moves (>2%)
            - Individual large-cap stock movements (>3%)
            - Options volatility spikes or unusual trading activity
            - Changes in Fed policy expectations or interest rate forecasts

            Answer ONLY with "true" if this article provides actionable market intelligence for active traders/investors making immediate decisions, or "false" if not.
            No explanations, just "true" or "false".
            """
            
            response = self.openai_client.responses.create(
                model="gpt-4o-mini",
                input=[
                    {
                        "role": "system", 
                        "content": classification_prompt
                    }
                ],
                temperature=0.1,
                max_output_tokens=10,
                store=True
            )
            
            relevance_result = response.output[0].content[0].text.strip().lower()
            is_relevant = relevance_result == "true"
            
            # Add relevance info to article metadata
            article['_metadata'] = article.get('_metadata', {})
            article['_metadata']['is_market_relevant'] = is_relevant
            article['_metadata']['relevance_score'] = 1.0 if is_relevant else 0.0
            article['_metadata']['category'] = category
            
            # Update marketCategory if relevant
            if is_relevant and category in MARKET_CATEGORY_MAP.keys():
                article['marketCategory'] = category
            
            # Log result
            if is_relevant:
                print(f"Article classified as MARKET RELEVANT to {category}: '{title[:60]}...'")
            else:
                print(f"Article classified as NOT MARKET RELEVANT to {category}: '{title[:60]}...'")
                    
        except Exception as e:
            print(f"Error classifying market relevance: {e}")
            article['_metadata'] = article.get('_metadata', {})
            article['_metadata']['is_market_relevant'] = False
            article['_metadata']['relevance_score'] = 0.0
            article['_metadata']['classification_error'] = str(e)

    def _ground_market_datetime_utc(self, article):
        """
        Ground market article datetime with trading session context
        
        Args:
            article (dict): Article to ground with accurate datetime and market session
            
        Returns:
            dict: Article with updated UTC datetime and market session info, or None if grounding failed
        """
        try:
            # Extract article information
            title_obj = article.get('title', {})
            title = title_obj.get('title', '') if isinstance(title_obj, dict) else str(title_obj)
            
            summary = ""
            sources = article.get('sources', [])
            if sources and isinstance(sources, list) and len(sources) > 0:
                first_source = sources[0]
                if isinstance(first_source, dict):
                    summary = first_source.get('summary', '')

            # Create grounding prompt with market context
            grounding_prompt = f"""
            Find the exact date and time when this market event occurred or was announced, with precise market session context.

            Article Title: "{title}"
            Article Summary: "{summary}"
            
            Task Instructions:
            1. Search for this exact market event by title and content
            2. From the coverage, extract:
               - Primary: The exact time the market event occurred/was announced (not article publication time)
               - Secondary: If event time unavailable, use first reliable news source publication time
               - Tertiary: Economic data release time, earnings announcement time, Fed decision time
            3. Convert the final datetime to UTC with high precision
            4. Determine the US Eastern Time market session when this occurred:
               - pre_market: 4:00-9:30 AM ET (pre-market trading)
               - market_hours: 9:30 AM-4:00 PM ET (regular trading session)
               - after_hours: 4:00-8:00 PM ET (after-hours trading)
               - overnight: 8:00 PM-4:00 AM ET (overnight/international hours)
            
            MARKET TIMING IMPORTANCE:
            - Pre-market: Events often cause gaps at market open
            - Market hours: Immediate price discovery and high volume impact
            - After-hours: Limited liquidity but can show direction for next day
            - Overnight: International reaction, futures trading impact
            
            If accurate market timing cannot be determined with confidence, return found: false.
            
            Market session timing is CRITICAL for understanding:
            - Price impact magnitude and timing
            - Trading volume and liquidity effects  
            - Institutional vs retail participation
            - Gap-up/gap-down potential at market open
            """
            
            try:
                print(f"Grounding market datetime for: {title[:50]}...")
                response = self.openai_client.responses.parse(
                    model="gpt-4o",
                    input=[{
                        "role": "system",
                        "content": grounding_prompt
                    }],
                    reasoning={},
                    tools=[{
                        "type": "web_search_preview",
                        "user_location": {
                            "type": "approximate"
                        },
                        "search_context_size": "medium"
                    }],
                    temperature=0.3,
                    max_output_tokens=2048,
                    text_format=MarketDateTime
                )
                
                grounding_result = response.output_parsed.model_dump()
                print(f"Market grounding result: {grounding_result}")
                
            except Exception as e:
                print(f"Market grounding failed: {e}")
                grounding_result = None

            # Reject if grounding failed
            if not grounding_result or not grounding_result.get('found', False):
                print(f"REJECTING MARKET ARTICLE: Grounding failed for: {title[:50]}...")
                return None

            # Update article with grounded datetime and market session
            article['groundedDateTime'] = {
                "found": True,
                "original_datetime": grounding_result.get('original_datetime'),
                "original_timezone": grounding_result.get('original_timezone'),
                "utc_datetime": grounding_result.get('utc_datetime'),
                "utc_iso": grounding_result.get('utc_iso'),
                "source_verified": grounding_result.get('source_verified'),
                "confidence": grounding_result.get('confidence', 0.0),
                "market_session": grounding_result.get('market_session'),
                "grounding_timestamp": self._get_utc_now().isoformat()
            }
            
            # Update market impact with session timing if high confidence
            if grounding_result.get('confidence', 0) >= 0.7:
                market_impact = article.get('marketImpact', {})
                market_impact['sessionTiming'] = grounding_result.get('market_session')
                article['marketImpact'] = market_impact
                
                # Update source datetime
                sources = article.get('sources', [])
                if sources and len(sources) > 0:
                    utc_datetime_str = grounding_result.get('utc_datetime', '')
                    if utc_datetime_str:
                        try:
                            utc_dt = self._parse_datetime(utc_datetime_str)
                            if utc_dt:
                                sources[0]['dateTimePublished'] = {
                                    "date": utc_dt.strftime("%Y-%m-%d"),
                                    "time": utc_dt.strftime("%H:%M:%S"),
                                    "region": "UTC",
                                    "verified": True,
                                    "market_session": grounding_result.get('market_session')
                                }
                        except Exception as e:
                            print(f"Error processing market datetime: {e}")
            
            print(f"Market grounding successful for: {title[:50]}... (session: {grounding_result.get('market_session', 'unknown')})")
            return article
                
        except Exception as e:
            print(f"Error in market datetime grounding: {e}")
            return None

    def _get_market_session(self, utc_datetime):
        """
        Determine current market session based on UTC time
        
        Args:
            utc_datetime (datetime): UTC datetime object
            
        Returns:
            str: Market session identifier
        """
        # Convert UTC to Eastern Time (market timezone)
        eastern = timezone('US/Eastern')
        market_time = utc_datetime.astimezone(eastern)
        
        hour = market_time.hour
        minute = market_time.minute
        
        # Market sessions in Eastern Time
        if 4 <= hour < 9 or (hour == 9 and minute < 30):
            return "pre_market"
        elif 9 <= hour < 16 or (hour == 9 and minute >= 30):
            return "market_hours"
        elif 16 <= hour < 20:
            return "after_hours"
        else:
            return "overnight"

    def _parse_datetime(self, datetime_str):
        """Parse datetime string with multiple format attempts"""
        datetime_formats = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%d",
        ]
        
        for fmt in datetime_formats:
            try:
                return datetime.strptime(datetime_str, fmt)
            except ValueError:
                continue
        return None

    def _get_utc_now(self):
        """Get current UTC datetime"""
        return datetime.now(dt_timezone.utc)

    def _load_existing_market_articles(self):
        """Load existing market articles from MongoDB (last 7 days)"""
        print("Loading existing market articles from MongoDB...")
        
        cutoff_utc = self._get_utc_now() - timedelta(days=7)
        cutoff_datetime_str = cutoff_utc.strftime('%Y-%m-%d %H:%M:%S')
        
        query = {
            "$and": [
                {"groundedDateTime.found": True},
                {"groundedDateTime.utc_datetime": {"$gte": cutoff_datetime_str}}
            ]
        }
        
        try:
            existing_articles = list(self.articles_col.find(query))
            print(f"Loaded {len(existing_articles)} existing market articles")
            
            # Log statistics
            total_grounded = self.articles_col.count_documents({"groundedDateTime": {"$exists": True}})
            total_successful = self.articles_col.count_documents({"groundedDateTime.found": True})
            total_failed = self.articles_col.count_documents({"groundedDateTime.found": False})
            
            print(f"Market database grounding statistics:")
            print(f"  - Total articles with grounding attempts: {total_grounded}")
            print(f"  - Successfully grounded: {total_successful}")
            print(f"  - Failed grounding: {total_failed}")
            print(f"  - Loaded for deduplication: {len(existing_articles)}")
            
            return existing_articles
        except Exception as e:
            print(f"Error loading existing market articles: {e}")
            return []

    def _filter_market_articles_by_date(self, articles):
        """Filter market articles to last 7 days using grounded datetime"""
        filtered_articles = []
        current_time = self._get_utc_now()
        cutoff_time = current_time - timedelta(days=7)
        
        for article in articles:
            article_date = current_time  # Default to current time
            grounded_datetime = article.get('groundedDateTime', {})
            
            utc_datetime_str = grounded_datetime.get('utc_datetime')
            if utc_datetime_str:
                article_date = self._parse_datetime(utc_datetime_str)
                if article_date is None:
                    article_date = current_time
            
            if article_date >= cutoff_time:
                filtered_articles.append(article)
        
        return filtered_articles

    def _generate_market_url(self, title):
        """Generate a Google search URL for the market article title"""
        import urllib.parse
        search_query = f'"{title}" market news'
        encoded_query = urllib.parse.quote(search_query)
        return f"https://www.google.com/search?q={encoded_query}&tbm=nws"

    def _add_market_embeddings(self, articles):
        """Generate embeddings for market articles"""
        print(f"Generating embeddings for {len(articles)} market articles")
        try:
            texts = []
            for article in articles:
                title = article.get('title', {})
                if isinstance(title, dict):
                    title = title.get('title', '')
                
                summary = ""
                sources = article.get('sources', [])
                if sources and isinstance(sources, list) and len(sources) > 0:
                    first_source = sources[0]
                    if isinstance(first_source, dict):
                        summary = first_source.get('summary', '')
                
                # Include market impact information in embedding
                market_impact = article.get('marketImpact', {})
                impact_text = f"Impact: {market_impact.get('level', '')}"
                
                text = f"{title} {summary} {impact_text}".strip()
                texts.append(text)
            
            # Process in batches
            batch_size = 20
            batches = [texts[i:i+batch_size] for i in range(0, len(texts), batch_size)]
            
            all_embeddings = []
            for i, batch in enumerate(batches):
                print(f"Processing embedding batch {i+1}/{len(batches)}")
                
                try:
                    response = self.openai_client.embeddings.create(
                        model="text-embedding-3-small",
                        input=batch
                    )
                    
                    batch_embeddings = [item.embedding for item in response.data]
                    all_embeddings.extend(batch_embeddings)
                    
                    if i < len(batches) - 1:
                        time.sleep(0.5)
                        
                except Exception as e:
                    print(f"Error in embedding batch {i+1}: {e}")
                    all_embeddings.extend([None] * len(batch))
            
            # Add embeddings to articles
            for i, article in enumerate(articles):
                if i < len(all_embeddings):
                    article['embedding'] = all_embeddings[i]
                    
            print(f"Successfully added embeddings to {len(articles)} market articles")
            
        except Exception as e:
            print(f"Error generating market embeddings: {e}")

    def _deduplicate_market_articles(self, new_articles, existing_articles):
        """
        Deduplicate market articles using similarity search
        
        Args:
            new_articles (list): New market articles
            existing_articles (list): Existing market articles
            
        Returns:
            list: Unique new articles with sequence relationships marked
        """
        if not new_articles:
            return []
        
        if not existing_articles:
            return new_articles
        
        print(f"Deduplicating market articles: {len(new_articles)} new vs {len(existing_articles)} existing")
        
        unique_articles = []
        duplicate_count = 0
        sequence_count = 0
        
        for i, new_article in enumerate(new_articles):
            print(f"Processing market article {i+1}/{len(new_articles)} for similarity")
            
            new_title = self._extract_title(new_article)
            new_category = new_article.get('marketCategory', 'unknown')
            
            # Find similar articles
            similar_articles = self._find_similar_market_articles(new_article, existing_articles, top_k=10)
            
            if not similar_articles:
                # Add market URL
                new_article['market_url'] = self._generate_market_url(new_title)
                unique_articles.append(new_article)
                continue
            
            # Compare with similar articles using OpenAI
            comparison_result = self._compare_market_articles(new_article, similar_articles)
            
            if comparison_result.startswith("DUPLICATE:"):
                duplicate_count += 1
                print(f"Found DUPLICATE market article: '{new_title[:50]}...'")
                continue
                
            elif comparison_result.startswith("SEQUENCE:"):
                sequence_count += 1
                # Extract sequence index and add prev field
                try:
                    sequence_index = int(comparison_result.split(":")[1].strip()) - 1
                    if 0 <= sequence_index < len(similar_articles):
                        matched_article = similar_articles[sequence_index]
                        new_article['prev'] = {
                            'id': matched_article.get('id'),
                            'title': self._extract_title(matched_article),
                            'category': matched_article.get('marketCategory', 'unknown'),
                            'relationship': 'market_sequence'
                        }
                        print(f"Found MARKET SEQUENCE: '{new_title[:50]}...' linked to previous event")
                except (ValueError, IndexError):
                    pass
                
                # Add market URL and include in unique articles
                new_article['market_url'] = self._generate_market_url(new_title)
                unique_articles.append(new_article)
            else:
                # Add market URL for unique articles
                new_article['market_url'] = self._generate_market_url(new_title)
                unique_articles.append(new_article)
        
        print(f"Market deduplication complete: {duplicate_count} duplicates, {sequence_count} sequences, {len(unique_articles)} unique")
        return unique_articles

    def _extract_title(self, article):
        """Extract title string from article"""
        title = article.get('title', {})
        if isinstance(title, dict):
            return title.get('title', '')
        return str(title)

    def _find_similar_market_articles(self, new_article, existing_articles, top_k=10):
        """Find similar market articles using embedding similarity"""
        new_embedding = new_article.get('embedding')
        if not new_embedding:
            return []
        
        existing_embeddings = []
        valid_existing = []
        
        for article in existing_articles:
            embedding = article.get('embedding')
            if embedding:
                existing_embeddings.append(embedding)
                valid_existing.append(article)
        
        if not existing_embeddings:
            return []
        
        try:
            similarities = cosine_similarity([new_embedding], existing_embeddings)[0]
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            similar_articles = [valid_existing[i] for i in top_indices if similarities[i] > 0.4]
            return similar_articles
            
        except Exception as e:
            print(f"Error finding similar market articles: {e}")
            return []

    def _compare_market_articles(self, new_article, similar_articles):
        """Compare new market article with similar existing articles"""
        new_title = self._extract_title(new_article)
        new_summary = self._extract_summary(new_article)
        new_category = new_article.get('marketCategory', 'unknown')
        
        similar_data = []
        for idx, article in enumerate(similar_articles):
            similar_data.append({
                'index': idx + 1,
                'title': self._extract_title(article),
                'summary': self._extract_summary(article)[:150],
                'category': article.get('marketCategory', 'unknown')
            })
        
        similar_text = "\n".join([
            f"{art['index']}. Title: \"{art['title']}\" | Summary: \"{art['summary']}\" | Category: {art['category']}"
            for art in similar_data
        ])
        
        comparison_prompt = f"""
        Compare this NEW market article against SIMILAR EXISTING articles.

        NEW MARKET ARTICLE:
        Title: "{new_title}"
        Summary: "{new_summary[:150]}"
        Category: "{new_category}"

        SIMILAR EXISTING ARTICLES:
        {similar_text}

        MARKET ARTICLE RELATIONSHIPS:
        - DUPLICATE: Same specific market event (same Fed decision, same company earnings, same economic data release)
        - SEQUENCE: Related market events in sequence (multiple Fed speeches, ongoing earnings season, related economic releases)
        - UNIQUE: Different market event entirely

        Look for specific matching details: company names, data release types, Fed policy decisions, dates, economic indicators, earnings periods.

        For market events, consider:
        - Fed policy: Same FOMC meeting vs different Fed speeches
        - Earnings: Same company same quarter vs different companies or quarters  
        - Economic data: Same release vs different indicators
        - Corporate events: Same M&A deal vs different transactions

        RESPONSE FORMAT:
        "DUPLICATE: <number>" - if reporting same specific market event
        "SEQUENCE: <number>" - if related ongoing market story  
        "UNIQUE" - if different market event

        Answer only with the specified format.
        """
        
        try:
            response = self.openai_client.responses.create(
                model="gpt-4o-mini",
                input=[{
                    "role": "system",
                    "content": comparison_prompt
                }],
                temperature=0.1,
                max_output_tokens=50,
                store=True
            )
            
            return response.output[0].content[0].text.strip()
            
        except Exception as e:
            print(f"Error comparing market articles: {e}")
            return "UNIQUE"

    def _extract_summary(self, article):
        """Extract summary from article sources"""
        sources = article.get('sources', [])
        if sources and isinstance(sources, list) and len(sources) > 0:
            first_source = sources[0]
            if isinstance(first_source, dict):
                return first_source.get('summary', '')
        return ""

    def _save_market_articles_to_mongo(self, articles):
        """Save market articles to MongoDB"""
        if not articles:
            print("No market articles to save")
            return
        
        print(f"Saving {len(articles)} market articles to MongoDB...")
        try:
            ops = []
            for article in articles:
                article = self._add_market_metadata(article)
                ops.append(UpdateOne(
                    {"id": article["id"]}, 
                    {"$set": article}, 
                    upsert=True
                ))
            
            if ops:
                self.articles_col.bulk_write(ops)
                print(f"Successfully saved {len(articles)} market articles")
                
        except Exception as e:
            print(f"Error saving market articles: {e}")

    def _add_market_metadata(self, article):
        """Add market-specific metadata to article"""
        current_time = self._get_utc_now()
        date_str = current_time.strftime("%Y-%m-%d")
        time_str = current_time.strftime("%H:%M:%S")
        
        # Add timezone info with market context
        timezone_info = {
            "UTC": current_time.strftime("%Y-%m-%d %H:%M:%S"),
            "ET": current_time.astimezone(timezone('US/Eastern')).strftime("%Y-%m-%d %H:%M:%S"),
            "market_session": self._get_market_session(current_time)
        }
        
        article['lastUpdatedAt'] = {
            "date": date_str,
            "time": timezone_info
        }
        
        if 'approval' not in article:
            article['approval'] = {
                "status": "pending",
                "date": date_str,
                "timezone_info": timezone_info
            }
        
        return article

    def _send_market_email_notification(self, articles):
        """Send market intelligence email notification"""
        try:
            html_content = self._generate_market_email_template(articles)
            
            import uuid
            unique_id = str(uuid.uuid4())[:8]
            current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            if articles:
                subject = f"Market Intelligence Report - {datetime.now().strftime('%Y-%m-%d')} - {len(articles)} Articles - ID:{unique_id}"
            else:
                subject = f"Market Intelligence Report - {datetime.now().strftime('%Y-%m-%d')} - No New Articles - ID:{unique_id}"
            
            for recipient in EMAIL_RECIPIENTS:
                response = resend.Emails.send({
                    "from": "onboarding@resend.dev",
                    "to": recipient,
                    "subject": subject,
                    "html": html_content,
                    "headers": {
                        "X-Report-ID": unique_id,
                        "X-Report-Timestamp": current_time,
                        "X-Report-Type": "market_intelligence_daily"
                    }
                })
                print(f"Market email sent to {recipient} with ID: {unique_id}")

        except Exception as e:
            print(f"Error sending market email: {e}")

    def _generate_market_email_template(self, articles):
        """Generate HTML email template for market intelligence"""
        # Brand colors
        primary_blue = "#1E40AF"  # Professional blue
        secondary_blue = "#3B82F6"  # Lighter blue
        success_green = "#10B981"  # Green for positive
        warning_red = "#EF4444"  # Red for negative
        neutral_gray = "#6B7280"  # Gray for neutral
        light_gray = "#F9FAFB"
        
        # Market category descriptions
        market_categories = {
            'MMH': 'Market Moving Headlines - Major events causing immediate price movements',
            'EDR': 'Economic Data Releases - Scheduled macroeconomic statistics',
            'CEV': 'Corporate Events - Company-specific announcements and earnings'
        }
        
        report_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        current_session = self._get_market_session(self._get_utc_now())
        
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Market Intelligence Report - {datetime.now().strftime('%Y-%m-%d')}</title>
        </head>
        <body style="margin: 0; padding: 0; font-family: Arial, sans-serif; background-color: {light_gray}; line-height: 1.6;">
            <div style="max-width: 800px; margin: 0 auto; background-color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                
                <!-- Header -->
                <div style="background: linear-gradient(135deg, {primary_blue} 0%, {secondary_blue} 100%); padding: 30px; text-align: center;">
                    <h1 style="color: white; margin: 0; font-size: 28px; font-weight: bold;">
                        📈 Market Intelligence Report
                    </h1>
                    <p style="color: white; margin: 10px 0 0 0; font-size: 16px;">
                        Daily Market Moving News Summary
                    </p>
                    <p style="color: #E5E7EB; margin: 5px 0 0 0; font-size: 14px;">
                        {datetime.now().strftime('%B %d, %Y')} | Session: {current_session.replace('_', ' ').title()} | Report ID: {report_timestamp}
                    </p>
                </div>
        """
        
        if articles:
            # Group articles by category
            categories = {}
            for article in articles:
                category = article.get('marketCategory', 'Unknown')
                if category not in categories:
                    categories[category] = []
                categories[category].append(article)
            
            # Summary section with market session context
            html_content += f"""
                <!-- Summary Statistics -->
                <div style="padding: 30px; background-color: {light_gray};">
                    <h2 style="color: {primary_blue}; margin: 0 0 20px 0; font-size: 24px; text-align: center;">
                        📊 Market Intelligence Summary
                    </h2>
                    
                    <div style="background-color: white; padding: 25px; border-radius: 10px; text-align: center; margin-bottom: 20px;">
                        <h3 style="color: {primary_blue}; margin: 0 0 15px 0; font-size: 20px;">📈 Market Articles Today</h3>
                        <div style="font-size: 36px; font-weight: bold; color: {primary_blue}; margin-bottom: 10px;">{len(articles)}</div>
                        <p style="color: {neutral_gray}; margin: 0; font-size: 14px;">
                            New market-moving events across {len(categories)} categories | Current: {current_session.replace('_', ' ').title()}
                        </p>
                    </div>
                    
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;">
            """
            
            # Category breakdown with market impact colors
            for category, category_articles in categories.items():
                category_name = market_categories.get(category, f"{category} - Unknown")
                # Count high impact articles
                high_impact_count = sum(1 for art in category_articles 
                                      if art.get('marketImpact', {}).get('level') == 'HIGH')
                impact_color = warning_red if high_impact_count > 0 else secondary_blue
                
                html_content += f"""
                        <div style="background-color: white; padding: 15px; border-radius: 8px; text-align: center; border-left: 4px solid {impact_color};">
                            <div style="font-size: 24px; font-weight: bold; color: {primary_blue}; margin-bottom: 5px;">{len(category_articles)}</div>
                            <div style="font-size: 12px; color: {neutral_gray}; font-weight: bold;">{category}</div>
                            {f'<div style="font-size: 10px; color: {warning_red}; margin-top: 5px;">⚡ {high_impact_count} HIGH</div>' if high_impact_count > 0 else ''}
                        </div>
                """
            
            html_content += """
                    </div>
                </div>
                
                <!-- Articles Section -->
                <div style="padding: 30px;">
                    <h2 style="color: """ + primary_blue + """; margin: 0 0 30px 0; font-size: 24px; text-align: center;">
                        🚨 Market Moving News
                    </h2>
            """
            
            # Articles by category with market-specific formatting
            for category, category_articles in categories.items():
                category_name = market_categories.get(category, f"{category} - Unknown")
                html_content += f"""
                    <div style="margin-bottom: 40px;">
                        <h3 style="color: {primary_blue}; margin: 0 0 20px 0; font-size: 20px; padding: 15px; background-color: {light_gray}; border-radius: 8px; border-left: 4px solid {secondary_blue};">
                            {category} - {category_name.split(' - ')[1] if ' - ' in category_name else category_name}
                        </h3>
                """
                
                for i, article in enumerate(category_articles):
                    title = self._extract_title(article)
                    summary = self._extract_summary(article)
                    
                    # Get market impact
                    market_impact = article.get('marketImpact', {})
                    impact_level = market_impact.get('level', 'UNKNOWN')
                    affected_indices = market_impact.get('affectedIndices', [])
                    session_timing = article.get('groundedDateTime', {}).get('market_session', 'unknown')
                    
                    # Color code by impact level
                    impact_color = success_green if impact_level == 'LOW' else warning_red if impact_level == 'HIGH' else '#F59E0B'
                    
                    # Get sentiment
                    sentiment = article.get('sentiment', {})
                    sentiment_score = sentiment.get('overall', 0)
                    sentiment_emoji = "📈" if sentiment_score > 0.2 else "📉" if sentiment_score < -0.2 else "➡️"
                    
                    # Session emoji
                    session_emoji = {"pre_market": "🌅", "market_hours": "🔔", "after_hours": "🌆", "overnight": "🌙"}.get(session_timing, "⏰")
                    
                    # Create search URL
                    import urllib.parse
                    search_query = f'"{title}" market news'
                    encoded_query = urllib.parse.quote(search_query)
                    market_search_url = f"https://www.google.com/search?q={encoded_query}&tbm=nws"
                    
                    title_html = f'<a href="{market_search_url}" target="_blank" style="color: {primary_blue}; text-decoration: none; font-weight: bold;">{title}</a>'
                    
                    html_content += f"""
                        <div style="margin-bottom: 25px; padding: 20px; border: 1px solid {light_gray}; border-radius: 10px; background-color: white;">
                            <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 10px;">
                                <h4 style="margin: 0; font-size: 16px; line-height: 1.4; color: {primary_blue}; flex: 1;">{title_html}</h4>
                                <div style="display: flex; gap: 8px; margin-left: 15px; align-items: center;">
                                    <span style="background-color: {impact_color}; color: white; padding: 4px 8px; border-radius: 4px; font-size: 11px; font-weight: bold;">{impact_level}</span>
                                    <span style="font-size: 16px;" title="Market Sentiment">{sentiment_emoji}</span>
                                    <span style="font-size: 16px;" title="Session: {session_timing.replace('_', ' ').title()}">{session_emoji}</span>
                                </div>
                            </div>
                            <p style="margin: 0 0 10px 0; color: {neutral_gray}; line-height: 1.6; font-size: 14px;">{summary}</p>
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 12px;">
                                <p style="margin: 0; color: {neutral_gray}; font-size: 11px;">
                                    🔍 <strong>Indices:</strong> {', '.join(affected_indices[:3]) if affected_indices else 'Broad Market'}
                                </p>
                                <p style="margin: 0; color: {neutral_gray}; font-size: 11px; font-style: italic;">
                                    Article {i+1} of {len(category_articles)} in {category}
                                </p>
                            </div>
                        </div>
                    """
                
                html_content += "</div>"
        else:
            # No articles found
            html_content += f"""
                <div style="padding: 30px; background-color: {light_gray};">
                    <div style="background-color: white; padding: 30px; border-radius: 10px; text-align: center;">
                        <div style="font-size: 48px; margin-bottom: 20px;">📊</div>
                        <h3 style="color: {primary_blue}; margin: 0 0 15px 0; font-size: 20px;">No New Market Articles Found</h3>
                        <p style="color: {neutral_gray}; margin: 0; font-size: 16px; line-height: 1.6;">
                            No new market-moving articles were found in the past 24 hours. Markets may be in a quiet period or awaiting major announcements.
                        </p>
                        <p style="color: {neutral_gray}; margin: 10px 0 0 0; font-size: 14px;">
                            Current Session: {current_session.replace('_', ' ').title()}
                        </p>
                    </div>
                </div>
            """
        
        # Category reference and footer
        html_content += f"""
                <!-- Category Reference -->
                <div style="padding: 30px;">
                    <h2 style="color: {primary_blue}; margin: 0 0 30px 0; font-size: 24px; text-align: center;">
                        📋 Market Category Reference
                    </h2>
                    
                    <div style="background-color: {light_gray}; padding: 25px; border-radius: 10px;">
                        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); gap: 15px;">
        """
        
        for code, description in market_categories.items():
            html_content += f"""
                            <div style="background-color: white; padding: 15px; border-radius: 8px; border-left: 4px solid {secondary_blue};">
                                <div style="font-weight: bold; color: {primary_blue}; margin-bottom: 5px; font-size: 14px;">{code}</div>
                                <div style="color: {neutral_gray}; font-size: 12px; line-height: 1.4;">{description}</div>
                            </div>
            """
        
        # Market session info
        html_content += f"""
                        </div>
                        
                        <div style="margin-top: 20px; padding: 15px; background-color: white; border-radius: 8px;">
                            <h4 style="color: {primary_blue}; margin: 0 0 10px 0; font-size: 14px;">📅 Market Sessions (US Eastern Time)</h4>
                            <div style="font-size: 12px; color: {neutral_gray}; line-height: 1.8;">
                                🌅 <strong>Pre-Market:</strong> 4:00-9:30 AM ET | 🔔 <strong>Market Hours:</strong> 9:30 AM-4:00 PM ET<br>
                                🌆 <strong>After-Hours:</strong> 4:00-8:00 PM ET | 🌙 <strong>Overnight:</strong> 8:00 PM-4:00 AM ET
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Footer -->
                <div style="background: linear-gradient(135deg, {primary_blue} 0%, {secondary_blue} 100%); padding: 30px; text-align: center;">
                    <p style="color: white; margin: 0 0 10px 0; font-size: 14px;">
                        Market Intelligence System - Automated Market News Analysis
                    </p>
                    <p style="color: #E5E7EB; margin: 0; font-size: 12px;">
                        Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p UTC')} | Session: {current_session.replace('_', ' ').title()} | Report ID: {report_timestamp}
                    </p>
                    <div style="margin-top: 20px; padding-top: 20px; border-top: 1px solid rgba(255,255,255,0.2);">
                        <p style="color: white; margin: 0; font-size: 12px; font-weight: bold;">
                            📈 Stay informed. Trade smart. Profit consistently.
                        </p>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html_content

    def _parse_json_response(self, response_text, expect_object=False):
        """Parse JSON response from API calls"""
        try:
            json_content = None
            
            if "```json" in response_text:
                json_content = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_content = response_text.split("```")[1].split("```")[0].strip()
            else:
                import re
                if expect_object:
                    match = re.search(r'\{.*\}', response_text, re.DOTALL)
                else:
                    match = re.search(r'\[.*\]', response_text, re.DOTALL)
                if match:
                    json_content = match.group(0)
            
            if json_content:
                try:
                    return json.loads(json_content)
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON: {e}")
                    return [] if not expect_object else None
            else:
                print("No valid JSON found in response")
                return [] if not expect_object else None
                
        except Exception as e:
            print(f"Error parsing JSON response: {e}")
            return [] if not expect_object else None

    def collect_market_news_two_phase(self):
        """
        Two-phase market news collection process
        
        Returns:
            list: Collected and processed market articles
        """
        print("\n" + "="*60)
        print("STARTING TWO-PHASE MARKET NEWS COLLECTION")
        print("="*60)
        
        # Load existing articles for deduplication
        existing_articles = self._load_existing_market_articles()
        prompts = self.load_market_prompts()
        
        if not prompts:
            print("No market prompt files found. Please check the prompts directory.")
            return existing_articles
        
        all_articles = []
        
        # PHASE 1: Collect market news titles
        print("\n" + "="*50)
        print("PHASE 1: SEARCHING FOR MARKET NEWS TITLES")
        print("="*50)
        
        all_titles = []
        for category, prompt_content in prompts.items():
            if category.lower() == "market_news_definition":
                continue
                
            print(f"\nSearching for {category} market news titles...")
            titles = self.search_for_market_titles(category, prompt_content, max_titles=3)
            all_titles.extend(titles)
            time.sleep(2)  # Rate limiting
        
        print(f"\nPhase 1 Complete: Found {len(all_titles)} market news titles")
        
        # PHASE 2: Deep dive into each title
        print("\n" + "="*50)
        print("PHASE 2: DETAILED MARKET ARTICLE RESEARCH")
        print("="*50)
        
        for i, title_obj in enumerate(all_titles):
            title = title_obj.get('title', 'Unknown')
            category = title_obj.get('search_category', 'Unknown')
            
            print(f"\nProcessing {i+1}/{len(all_titles)}: {category} - {title[:60]}...")
            
            prompt_content = prompts.get(category, '')
            detailed_article = self.search_detailed_market_article(title_obj, prompt_content)
            
            if detailed_article:
                all_articles.append(detailed_article)
                print(f"✓ Successfully processed market article")
            else:
                print(f"✗ Could not retrieve detailed market article")
            
            time.sleep(3)  # Rate limiting
        
        print(f"\nPhase 2 Complete: Retrieved {len(all_articles)} detailed market articles")
        
        # Continue with processing pipeline
        return self._process_market_articles_pipeline(all_articles, existing_articles)
    
    def _process_market_articles_pipeline(self, articles, existing_articles):
        """Process market articles through the pipeline"""
        # STEP 3: RELEVANCE FILTER
        print("\n" + "="*50)
        print("STEP 3: MARKET RELEVANCE FILTERING")
        print("="*50)
        
        relevant_articles = []
        for article in articles:
            if '_metadata' not in article:
                article['_metadata'] = {}
            
            category = article.get('marketCategory', '')
            self.classify_market_relevance(article, category)
            
            if article.get('_metadata', {}).get('is_market_relevant', False):
                relevant_articles.append(article)
        
        print(f"Market relevance filtering: {len(articles)} -> {len(relevant_articles)} articles")
        
        # STEP 4: MARKET DATETIME GROUNDING
        print("\n" + "="*50)
        print("STEP 4: GROUNDING MARKET DATETIME WITH SESSION CONTEXT")
        print("="*50)
        
        grounded_articles = []
        for article in relevant_articles:
            grounded = self._ground_market_datetime_utc(article)
            if grounded:
                grounded_articles.append(grounded)
            time.sleep(3)  # Rate limiting for grounding calls
        
        print(f"Market grounding: {len(relevant_articles)} -> {len(grounded_articles)} articles")
        
        # STEP 5: DATE FILTER
        print("\n" + "="*50)
        # STEP 5: DATE FILTER
        print("\n" + "="*50)
        print("STEP 5: FILTERING BY DATE (7 days)")
        print("="*50)
        
        date_filtered_articles = self._filter_market_articles_by_date(grounded_articles)
        print(f"Date filtering: {len(grounded_articles)} -> {len(date_filtered_articles)} articles")
        
        # STEP 6: EMBEDDINGS
        print("\n" + "="*50)
        print("STEP 6: ADDING MARKET EMBEDDINGS")
        print("="*50)
        
        if date_filtered_articles:
            self._add_market_embeddings(date_filtered_articles)
        
        # STEP 7: DEDUPLICATION
        print("\n" + "="*50)
        print("STEP 7: MARKET DEDUPLICATION")
        print("="*50)
        
        unique_articles = self._deduplicate_market_articles(date_filtered_articles, existing_articles)
        
        # STEP 8: SAVE AND NOTIFY
        print("\n" + "="*50)
        print("STEP 8: SAVING MARKET ARTICLES")
        print("="*50)
        
        if unique_articles:
            self._save_market_articles_to_mongo(unique_articles)
            self._send_market_email_notification(unique_articles)
        else:
            self._send_market_email_notification([])
        
        return existing_articles + unique_articles

    def run(self):
        """Run the enhanced two-phase market collection process"""
        articles = self.collect_market_news_two_phase()
        
        # Generate summary with market-specific metrics
        categories = {}
        high_impact_count = 0
        session_breakdown = {}
        
        for article in articles:
            category = article.get('marketCategory', 'unknown')
            if category not in categories:
                categories[category] = []
            categories[category].append(article)
            
            # Count high impact articles
            if article.get('marketImpact', {}).get('level') == 'HIGH':
                high_impact_count += 1
                
            # Count by market session
            session = article.get('groundedDateTime', {}).get('market_session', 'unknown')
            session_breakdown[session] = session_breakdown.get(session, 0) + 1
        
        result = {
            "total_articles": len(articles),
            "high_impact_articles": high_impact_count,
            "articles_by_category": {category: len(articles) for category, articles in categories.items()},
            "articles_by_session": session_breakdown,
            "current_market_session": self._get_market_session(self._get_utc_now()),
            "generated_at": datetime.now().isoformat(),
            "date": datetime.now().strftime("%Y-%m-%d"),
            "collection_method": "two_phase_market_intelligence"
        }
        
        return result


# Utility classes for market data schema
class MarketDataSchema:
    """Schema definition for market news data"""
    
    @staticmethod
    def get_schema():
        return {
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "description": "Unique identifier in format MKT-{category}-{timestamp}-{hash}"
                },
                "title": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "sentiment_score": {"type": "number", "minimum": -1, "maximum": 1}
                    },
                    "required": ["title"]
                },
                "sources": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "url": {"type": "string"},
                            "summary": {"type": "string"},
                            "credibility_score": {"type": "number", "minimum": 0, "maximum": 1},
                            "dateTimePublished": {
                                "type": "object",
                                "properties": {
                                    "date": {"type": "string"},
                                    "time": {"type": "string"},
                                    "region": {"type": "string"},
                                    "verified": {"type": "boolean"},
                                    "market_session": {"type": "string"}
                                }
                            }
                        },
                        "required": ["name", "summary"]
                    },
                    "minItems": 1
                },
                "marketImpact": {
                    "type": "object",
                    "properties": {
                        "level": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]},
                        "affectedIndices": {"type": "array", "items": {"type": "string"}},
                        "affectedSectors": {"type": "array", "items": {"type": "string"}},
                        "expectedVolatility": {"type": "string"},
                        "timeHorizon": {"type": "string", "enum": ["immediate", "short_term", "medium_term", "long_term"]},
                        "sessionTiming": {"type": "string", "enum": ["pre_market", "market_hours", "after_hours", "overnight"]}
                    },
                    "required": ["level", "expectedVolatility", "timeHorizon"]
                },
                "marketCategory": {"type": "string", "enum": ["MMH", "EDR", "CEV"]},
                "sentiment": {
                    "type": "object",
                    "properties": {
                        "overall": {"type": "number", "minimum": -1, "maximum": 1},
                        "bullish_indicators": {"type": "array", "items": {"type": "string"}},
                        "bearish_indicators": {"type": "array", "items": {"type": "string"}},
                        "uncertainty_level": {"type": "number", "minimum": 0, "maximum": 1}
                    },
                    "required": ["overall"]
                },
                "technicalLevels": {
                    "type": "object",
                    "properties": {
                        "supportLevels": {"type": "array", "items": {"type": "number"}},
                        "resistanceLevels": {"type": "array", "items": {"type": "number"}},
                        "keyMovingAverages": {"type": "object"},
                        "volatilityMeasures": {"type": "object"}
                    }
                },
                "tradingImplications": {
                    "type": "object",
                    "properties": {
                        "shortTerm": {"type": "string"},
                        "mediumTerm": {"type": "string"},
                        "keyLevelsToWatch": {"type": "array", "items": {"type": "number"}},
                        "volumeExpectations": {"type": "string"},
                        "optionsActivity": {"type": "string"}
                    }
                },
                "groundedDateTime": {
                    "type": "object",
                    "properties": {
                        "found": {"type": "boolean"},
                        "original_datetime": {"type": ["string", "null"]},
                        "original_timezone": {"type": ["string", "null"]},
                        "utc_datetime": {"type": ["string", "null"]},
                        "utc_iso": {"type": ["string", "null"]},
                        "source_verified": {"type": ["string", "null"]},
                        "confidence": {"type": ["number", "null"]},
                        "market_session": {"type": ["string", "null"]},
                        "grounding_timestamp": {"type": "string"}
                    },
                    "required": ["found"]
                }
            },
            "required": ["id", "title", "sources", "marketImpact", "marketCategory", "sentiment"]
        }


# Example usage
if __name__ == "__main__":
    # Create a market news agent
    agent = MarketNewsAgent()
    
    # Run the daily market collection process
    result = agent.run()
    
    print(f"\nMarket Two-Phase Collection Complete!")
    print(f"Total articles: {result['total_articles']}")
    print(f"High impact articles: {result['high_impact_articles']}")
    print(f"Current market session: {result['current_market_session']}")
    print(f"Collection method: {result['collection_method']}")
    
    print(f"\nArticles by category:")
    for category, count in result['articles_by_category'].items():
        print(f"- {category}: {count} articles")
    
    print(f"\nArticles by market session:")
    for session, count in result['articles_by_session'].items():
        print(f"- {session.replace('_', ' ').title()}: {count} articles")