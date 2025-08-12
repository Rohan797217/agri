import requests
from typing import Dict, Any, List
from datetime import datetime, timedelta
from agents.base_agent import BaseAgent

class MarketAgent(BaseAgent):
    """Specialized agent for market prices, trends, and trading advice"""
    
    def __init__(self):
        super().__init__("MarketAgent")
        self.market_data_sources = self.initialize_market_sources()
    
    def initialize_market_sources(self):
        """Initialize market data sources"""
        return {
            "agmarknet": "https://agmarknet.gov.in/",
            "enam": "https://www.enam.gov.in/",
            "commodity_boards": [
                "https://indianspices.com/",
                "https://coffeeboard.gov.in/",
                "https://www.teaboard.gov.in/"
            ]
        }
    
    def get_market_advice(self, query: str, user_context: Dict) -> str:
        """Generate market-related advice and price information"""
        try:
            location = user_context.get("location", "")
            crops = user_context.get("crops", [])
            
            # Extract commodity information from query
            detected_commodity = self.detect_commodity_from_query(query, crops)
            
            # Try to get real market data
            market_data = self.fetch_market_data(detected_commodity, location)
            
            # Build system message for market advice
            system_message = """You are an agricultural market analyst and commodity trading expert. 
            Provide practical market advice including price trends, best selling times, market analysis, 
            and trading strategies for farmers. Always include risk factors and market volatility warnings."""
            
            # Build enhanced query
            enhanced_query = self.build_market_query(query, detected_commodity, location, market_data)
            
            # Generate response
            response = self.generate_response(enhanced_query, system_message)
            
            # Add market data if available
            if market_data.get("prices"):
                market_summary = self.format_market_data(market_data)
                response = f"{market_summary}\n\n**Market Analysis:**\n{response}"
            
            # Add market risk disclaimer
            response += self.add_market_disclaimer()
            
            return response
            
        except Exception as e:
            fallback_response = self.web_search_fallback(f"market price {query}", "market")
            if fallback_response != "No relevant information found through web search.":
                return f"Using fallback data:\n{fallback_response}"
            return f"Error generating market advice: {str(e)}"
    
    def detect_commodity_from_query(self, query: str, user_crops: List = None) -> str:
        """Detect commodity mentioned in query"""
        query_lower = query.lower()
        
        # Common commodities
        commodities = [
            "rice", "wheat", "cotton", "sugarcane", "maize", "corn",
            "soybean", "groundnut", "onion", "potato", "tomato", 
            "chilli", "turmeric", "cardamom", "pepper", "coffee", "tea"
        ]
        
        for commodity in commodities:
            if commodity in query_lower:
                return commodity
        
        # Check user's crops if no specific commodity mentioned
        if user_crops:
            return user_crops[0].lower()
        
        return "general"
    
    def fetch_market_data(self, commodity: str, location: str) -> Dict:
        """Fetch market data from available sources"""
        try:
            # This is a placeholder for real market data fetching
            # In a real implementation, you would integrate with actual market APIs
            
            # Simulated market data structure (replace with real API calls)
            return {
                "commodity": commodity,
                "location": location,
                "prices": {
                    "current_price": "Data not available - API integration needed",
                    "price_trend": "Please check local mandis or official sources"
                },
                "markets": [
                    f"Local mandi in {location}",
                    "Nearby wholesale markets"
                ]
            }
            
        except Exception as e:
            return {"error": f"Failed to fetch market data: {str(e)}"}
    
    def build_market_query(self, query: str, commodity: str, location: str, market_data: Dict) -> str:
        """Build enhanced market query with available data"""
        enhanced_parts = [query]
        
        if commodity != "general":
            enhanced_parts.append(f"Commodity focus: {commodity}")
        
        if location:
            enhanced_parts.append(f"Location: {location}")
        
        if market_data and not market_data.get("error"):
            enhanced_parts.append(f"Market context: {market_data}")
        
        return " ".join(enhanced_parts)
    
    def format_market_data(self, market_data: Dict) -> str:
        """Format market data for display"""
        if market_data.get("error"):
            return f"**Market Data Error:** {market_data['error']}"
        
        info_parts = ["**📈 Market Information:**"]
        
        commodity = market_data.get("commodity", "Unknown")
        location = market_data.get("location", "Unknown")
        
        info_parts.append(f"• **Commodity:** {commodity.title()}")
        info_parts.append(f"• **Location:** {location}")
        
        prices = market_data.get("prices", {})
        if prices:
            if "current_price" in prices:
                info_parts.append(f"• **Current Price:** {prices['current_price']}")
            if "price_trend" in prices:
                info_parts.append(f"• **Trend:** {prices['price_trend']}")
        
        markets = market_data.get("markets", [])
        if markets:
            market_list = ", ".join(markets)
            info_parts.append(f"• **Available Markets:** {market_list}")
        
        return "\n".join(info_parts)
    
    def add_market_disclaimer(self) -> str:
        """Add market risk disclaimer"""
        return """
        
**⚠️ Market Risk Disclaimer:**
• Market prices are subject to volatility and rapid changes
• Always verify prices from multiple sources before making decisions
• Consider transportation, storage, and handling costs
• Market timing depends on multiple factors beyond price
• Consult local traders and commission agents for current rates
• Check official sources: AgMarkNet, eNAM, local mandi boards
        """
    
    def get_price_trend_analysis(self, commodity: str, days: int = 30) -> str:
        """Get price trend analysis for a commodity"""
        try:
            system_message = """You are a commodity price analyst. Provide trend analysis and 
            price forecasting based on seasonal patterns, demand-supply factors, and market conditions."""
            
            prompt = f"""
            Provide price trend analysis for {commodity} over the last {days} days:
            
            Include:
            1. Historical price patterns for this season
            2. Factors affecting current prices
            3. Supply and demand dynamics
            4. Seasonal price variations
            5. Short-term price forecast (next 2-4 weeks)
            6. Best selling strategies
            """
            
            return self.generate_response(prompt, system_message)
            
        except Exception as e:
            return f"Error generating price trend analysis: {str(e)}"
    
    def get_market_timing_advice(self, commodity: str, quantity: float, location: str) -> str:
        """Get advice on optimal market timing"""
        try:
            system_message = """You are a market timing specialist for agricultural commodities. 
            Provide strategic advice on when to sell crops for maximum profit."""
            
            prompt = f"""
            Provide market timing advice for:
            Commodity: {commodity}
            Quantity: {quantity} tonnes
            Location: {location}
            
            Consider:
            1. Current market conditions
            2. Seasonal price patterns
            3. Storage costs vs waiting for better prices
            4. Transportation and logistics
            5. Risk factors (weather, policy changes)
            6. Alternative marketing channels (direct, FPO, contract farming)
            """
            
            return self.generate_response(prompt, system_message)
            
        except Exception as e:
            return f"Error generating market timing advice: {str(e)}"
    
    def get_value_addition_opportunities(self, commodity: str, farm_size: float) -> str:
        """Suggest value addition opportunities"""
        try:
            system_message = """You are an agribusiness consultant specializing in value addition 
            and post-harvest processing opportunities for farmers."""
            
            prompt = f"""
            Suggest value addition opportunities for {commodity} for a farm of {farm_size} acres:
            
            Include:
            1. Processing opportunities at farm level
            2. Equipment and investment requirements
            3. Market demand for processed products
            4. Skill development needed
            5. Financial support and schemes available
            6. Partnership opportunities with processors
            7. Risk assessment and returns
            """
            
            return self.generate_response(prompt, system_message)
            
        except Exception as e:
            return f"Error generating value addition advice: {str(e)}"
