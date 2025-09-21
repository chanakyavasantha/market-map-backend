class MarketPromptManager:
    """
    Market News prompts focused on the three core categories with highest market impact.
    Each prompt includes primary keywords, alternative terms, and search variations for market-moving financial news.
    """
    
    def __init__(self):
        """Initialize the MarketPromptManager class with predefined prompts."""
        pass
    
    # Market News Definition Prompt
    market_news_definition = """
    ## Market News Definition
    A market-moving event is any confirmed financial development that causes immediate or expected significant movement in equity markets, indices, or individual stock prices within minutes to hours of publication. This includes events that trigger algorithmic trading responses, create immediate volatility spikes, and typically move broad market indices by 1%+ intraday. Only actualized market developments with measurable price impact are considered—speculative analysis or theoretical market scenarios are excluded.
    """
    
    # Market Moving Headlines - MMH
    MMH = """
    ## Market Moving Headlines
    **Description:** News stories that have immediate, significant impact on asset prices or market direction within minutes/hours of publication.
    
    **Characteristics:**
    - Triggers algorithmic trading responses
    - Creates immediate volatility spikes
    - Often involves monetary policy, geopolitical events, or major corporate announcements
    - Typically moves broad market indices by 1%+ intraday
    
    **Primary Search Terms:** Fed rate decision, central bank announcement, geopolitical crisis, major bank failure, market crash, emergency policy, circuit breaker, flash crash, market halt, systemic risk
    
    **Secondary Keywords:** FOMC surprise, hawkish dovish, military conflict, trade war escalation, banking crisis, liquidity crisis, margin call, risk-off sentiment, safe haven flows, currency crisis
    
    **Examples include but are not limited to:**
    - Federal Reserve emergency rate cuts or surprise policy changes
    - Major bank failures or financial institution collapses
    - Geopolitical conflicts affecting global trade and markets
    - Unexpected earnings surprises from mega-cap stocks (Apple, Microsoft, NVIDIA)
    - Currency crises or sovereign debt defaults
    - Natural disasters affecting major supply chains
    - Terrorist attacks or major security incidents
    - Technology system failures affecting trading infrastructure
    - Oil supply disruptions or OPEC emergency meetings
    - Presidential election results or major political upheavals
    - Pandemic declarations or major health emergencies
    - Major cryptocurrency exchanges collapses or regulatory crackdowns
    
    **Market Impact Thresholds:** S&P 500 >1% move, NASDAQ >1.5% move, Dow >300 points, VIX >20% spike, individual mega-cap >5% move
    
    **Trading Response Indicators:** Algorithm triggered selling, volatility halt activations, after-hours/pre-market extreme moves, options volume spikes
    
    **Related Incident Types:** trading system failures, market manipulation discoveries, flash crashes, liquidity freezes, cross-market contagion
    """
    
    # Economic Data Releases - EDR
    EDR = """
    ## Economic Data Releases
    **Description:** Scheduled publication of macroeconomic statistics that measure the health and performance of national or regional economies.
    
    **Key Features:**
    - Released on predetermined schedules (monthly, quarterly, annually)
    - Compared against consensus forecasts
    - Market reaction depends on deviation from expectations
    - Influence central bank policy decisions
    
    **Primary Search Terms:** Non-farm payrolls, unemployment rate, CPI inflation, GDP growth, retail sales, consumer confidence, manufacturing PMI, housing starts, initial jobless claims, trade balance
    
    **Secondary Keywords:** core PCE, wage growth, labor force participation, consumer spending, business investment, productivity, unit labor costs, import prices, export growth, current account
    
    **Major Categories:**
    - **Employment:** Non-farm payrolls, unemployment rate, wage growth, job openings, quits rate
    - **Inflation:** CPI, PPI, PCE deflator, core inflation, breakeven rates
    - **Growth:** GDP, personal income, consumer spending, business investment
    - **Manufacturing:** ISM PMI, industrial production, capacity utilization, factory orders
    - **Consumer Behavior:** Retail sales, consumer confidence, consumer sentiment, personal consumption
    - **Housing:** Housing starts, building permits, existing home sales, new home sales
    - **Trade:** Trade balance, import/export data, current account balance
    
    **Examples include but are not limited to:**
    - Non-farm payrolls significantly above/below consensus estimates
    - CPI inflation readings exceeding Fed targets or expectations
    - GDP growth rates showing recession or expansion surprises
    - Unemployment rate changes indicating labor market shifts
    - Consumer confidence collapses or surges
    - Manufacturing PMI crossing expansion/contraction thresholds
    - Housing market data showing bubble or crash indicators
    - Trade deficit changes affecting currency and policy
    - Regional Fed surveys showing economic turning points
    - International economic data affecting global growth outlook
    
    **Market Impact Thresholds:** >0.2% deviation from consensus, >2 standard deviation surprise, Fed policy relevance, recession/expansion implications
    
    **Timing Sensitivity:** Pre-market hours (8:30 AM ET releases), intraday policy implications, next FOMC meeting relevance
    
    **Related Incident Types:** data revisions, methodology changes, seasonal adjustment errors, early data leaks, government shutdowns affecting releases
    """

    # Corporate Events - CEV
    CEV = """
    ## Corporate Events
    **Description:** Company-specific announcements or occurrences that affect individual stock prices and potentially broader sector performance.
    
    **Types:**
    - **Earnings releases:** Quarterly financial results vs. analyst estimates
    - **Corporate actions:** Mergers, acquisitions, spinoffs, dividend changes
    - **Management changes:** CEO transitions, board appointments
    - **Product/service announcements:** New launches, partnerships, regulatory approvals
    - **Legal/regulatory:** Lawsuits, investigations, compliance issues
    
    **Primary Search Terms:** earnings beat, earnings miss, merger announcement, CEO resignation, dividend cut, stock buyback, FDA approval, product recall, guidance cut, bankruptcy filing
    
    **Secondary Keywords:** quarterly results, EPS surprise, revenue miss, forward guidance, special dividend, share repurchase, strategic acquisition, management departure, regulatory approval, class action lawsuit
    
    **Earnings Releases:**
    - S&P 500 companies with significant earnings beats or misses (>10% vs. estimates)
    - Revenue growth acceleration or deceleration vs. expectations
    - Forward guidance raises or cuts affecting future earnings expectations
    - Margin expansion or compression indicating operational changes
    - Management commentary on business outlook and market conditions
    - Analyst estimate revisions following earnings announcements
    
    **Corporate Actions:**
    - Major merger and acquisition announcements (>$1B deal value)
    - Hostile takeover attempts and acquisition defenses
    - Dividend increases, cuts, or suspensions by major companies
    - Large share buyback program announcements
    - Corporate spin-offs and asset divestiture plans
    - Bankruptcy filings and Chapter 11 restructuring announcements
    
    **Management Changes:**
    - CEO departures at major corporations (especially unplanned)
    - CFO changes during critical business periods
    - Board of directors changes and activist investor appointments
    - Founder departures at major technology companies
    - Regulatory-forced management changes at financial institutions
    
    **Product/Service Announcements:**
    - Major product launches by technology companies (iPhone, new chips)
    - FDA drug approvals and rejections for pharmaceutical companies
    - Major partnership agreements and strategic alliances
    - New market entry announcements by large corporations
    - Patent approvals and intellectual property developments
    
    **Legal/Regulatory:**
    - Major lawsuit settlements affecting company finances
    - Regulatory investigations and enforcement actions
    - Antitrust investigations and approval processes
    - SEC investigations and accounting irregularities
    - Environmental and safety violations with financial impact
    
    **Market Impact Criteria:** Individual stock >5% movement, market cap change >$5B, sector spillover effects, index inclusion/exclusion implications
    
    **Sector Implications:** Technology disruptions, healthcare regulatory changes, financial sector regulatory impacts, energy transition effects
    
    """
    
    @classmethod
    def get_prompt(cls, prompt_name):
        """Get a specific prompt by name."""
        return getattr(cls, prompt_name, None)

    @classmethod
    def get_all_prompts(cls):
        """Get all market news prompts as a dictionary."""
        prompts = {}
        for attr in dir(cls):
            if not attr.startswith('__') and not callable(getattr(cls, attr)) and attr != 'market_news_definition':
                prompts[attr] = getattr(cls, attr)
        return prompts

    @classmethod
    def list_available_prompts(cls):
        """List all available prompt names."""
        return [attr for attr in dir(cls) if not attr.startswith('__') and not callable(getattr(cls, attr)) and attr != 'market_news_definition']

    @classmethod
    def get_market_categories(cls):
        """Get market category mapping."""
        return {
            'MMH': 'Market Moving Headlines',
            'EDR': 'Economic Data Releases', 
            'CEV': 'Corporate Events'
        }

    @classmethod
    def get_category_description(cls, category_code):
        """Get description for a specific market category."""
        descriptions = {
            'MMH': 'News stories that have immediate, significant impact on asset prices or market direction within minutes/hours of publication.',
            'EDR': 'Scheduled publication of macroeconomic statistics that measure the health and performance of national or regional economies.',
            'CEV': 'Company-specific announcements or occurrences that affect individual stock prices and potentially broader sector performance.'
        }
        return descriptions.get(category_code, 'Unknown category')

    @classmethod
    def get_market_impact_thresholds(cls):
        """Get market impact thresholds for different categories."""
        return {
            'MMH': {
                'indices': 'S&P 500 >1% move, NASDAQ >1.5% move, Dow >300 points',
                'volatility': 'VIX >20% spike',
                'individual': 'Mega-cap >5% move'
            },
            'EDR': {
                'deviation': '>0.2% from consensus',
                'surprise': '>2 standard deviations',
                'policy_relevance': 'Fed policy implications'
            },
            'CEV': {
                'individual': 'Stock >5% movement',
                'market_cap': 'Change >$5B',
                'sector': 'Spillover effects'
            }
        }


# Example usage
if __name__ == "__main__":
    # Test market news search capabilities
    manager = MarketPromptManager()
    
    # List all available prompts
    prompt_names = manager.list_available_prompts()
    print(f"Available market news prompts: {prompt_names}")
    
    # Get all prompts as a dictionary
    all_prompts = manager.get_all_prompts()
    print(f"Total market news prompts: {len(all_prompts)}")
    
    # Get market categories
    categories = manager.get_market_categories()
    print(f"Market categories: {categories}")
    
    # Get specific prompt
    mmh_prompt = manager.get_prompt('MMH')
    if mmh_prompt:
        print(f"MMH prompt length: {len(mmh_prompt)} characters")
    
    # Get category descriptions
    for code in ['MMH', 'EDR', 'CEV']:
        description = manager.get_category_description(code)
        print(f"{code}: {description}")
    
    # Get market impact thresholds
    thresholds = manager.get_market_impact_thresholds()
    print(f"Market impact thresholds: {thresholds}")