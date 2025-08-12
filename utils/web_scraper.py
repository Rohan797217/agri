import trafilatura
import requests
from typing import List, Dict, Optional
import time
import random
from urllib.parse import urljoin, urlparse
import re

def get_website_text_content(url: str) -> str:
    """
    This function takes a url and returns the main text content of the website.
    The text content is extracted using trafilatura and easier to understand.
    The results is not directly readable, better to be summarized by LLM before consume
    by the user.

    Some common website to crawl information from:
    MLB scores: https://www.mlb.com/scores/YYYY-MM-DD
    """
    try:
        # Send a request to the website
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return ""
        
        text = trafilatura.extract(downloaded)
        return text if text else ""
        
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return ""

class AgriWebScraper:
    """Specialized web scraper for agricultural information"""
    
    def __init__(self):
        self.agricultural_sites = self.get_agricultural_sites()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    def get_agricultural_sites(self) -> Dict[str, List[str]]:
        """Get list of agricultural websites for different categories"""
        return {
            "weather": [
                "https://www.weather.com/",
                "https://www.accuweather.com/",
                "https://mausam.imd.gov.in/",
                "https://www.timeanddate.com/weather/"
            ],
            "market_prices": [
                "https://agmarknet.gov.in/",
                "https://www.enam.gov.in/",
                "https://www.commodityonline.com/",
                "https://www.mcxindia.com/"
            ],
            "agricultural_news": [
                "https://www.financialexpress.com/industry/agriculture/",
                "https://www.business-standard.com/topic/agriculture",
                "https://economictimes.indiatimes.com/news/economy/agriculture",
                "https://www.thehindu.com/news/national/agriculture/"
            ],
            "government_schemes": [
                "https://www.india.gov.in/",
                "https://pmkisan.gov.in/",
                "https://www.pmfby.gov.in/",
                "https://agricoop.nic.in/"
            ],
            "research_institutions": [
                "https://www.icar.org.in/",
                "https://www.icrisat.org/",
                "https://www.irri.org/",
                "https://www.cimmyt.org/"
            ],
            "crop_information": [
                "https://farmer.gov.in/",
                "https://mkisan.gov.in/",
                "https://www.krishi.icar.gov.in/"
            ]
        }
    
    def scrape_weather_info(self, location: str, query: str) -> str:
        """Scrape weather information from multiple sources"""
        try:
            weather_content = ""
            
            # Construct weather-specific search URLs
            weather_urls = [
                f"https://www.weather.com/weather/today/l/{location}",
                f"https://www.timeanddate.com/weather/{location.replace(' ', '-')}"
            ]
            
            for url in weather_urls[:2]:  # Limit to avoid rate limiting
                try:
                    content = get_website_text_content(url)
                    if content and len(content) > 100:
                        weather_content += f"\n--- Weather from {urlparse(url).netloc} ---\n"
                        weather_content += content[:800] + "...\n"
                        break  # Use first successful result
                except Exception as e:
                    continue
            
            return weather_content if weather_content else "Could not retrieve weather information."
            
        except Exception as e:
            return f"Weather scraping error: {str(e)}"
    
    def scrape_market_prices(self, commodity: str, location: str = "") -> str:
        """Scrape market price information"""
        try:
            market_content = ""
            
            # Try to get market price information
            market_urls = [
                f"https://agmarknet.gov.in/SearchCmmMkt.aspx?Tx_Commodity={commodity}",
                f"https://www.commodityonline.com/mandiprices/{commodity}-price"
            ]
            
            for url in market_urls[:1]:  # Limit to one source to avoid overloading
                try:
                    content = get_website_text_content(url)
                    if content and len(content) > 100:
                        market_content += f"\n--- Market data from {urlparse(url).netloc} ---\n"
                        market_content += content[:1000] + "...\n"
                        break
                except Exception as e:
                    continue
            
            return market_content if market_content else "Could not retrieve market price information."
            
        except Exception as e:
            return f"Market price scraping error: {str(e)}"
    
    def scrape_government_schemes(self, scheme_type: str = "farmer") -> str:
        """Scrape government schemes information"""
        try:
            schemes_content = ""
            
            # Government scheme URLs
            scheme_urls = [
                "https://pmkisan.gov.in/",
                "https://www.pmfby.gov.in/",
                "https://www.india.gov.in/topics/agriculture"
            ]
            
            for url in scheme_urls[:2]:  # Limit to avoid overloading
                try:
                    content = get_website_text_content(url)
                    if content and len(content) > 100:
                        schemes_content += f"\n--- Scheme info from {urlparse(url).netloc} ---\n"
                        schemes_content += content[:1000] + "...\n"
                except Exception as e:
                    continue
            
            return schemes_content if schemes_content else "Could not retrieve government scheme information."
            
        except Exception as e:
            return f"Government scheme scraping error: {str(e)}"
    
    def scrape_crop_information(self, crop: str, information_type: str = "general") -> str:
        """Scrape crop-specific information"""
        try:
            crop_content = ""
            
            # Crop information URLs
            crop_urls = [
                f"https://farmer.gov.in/search?q={crop}",
                f"https://www.icar.org.in/search?q={crop}",
                f"https://mkisan.gov.in/search?q={crop}"
            ]
            
            for url in crop_urls[:2]:  # Limit to avoid overloading
                try:
                    content = get_website_text_content(url)
                    if content and len(content) > 100:
                        crop_content += f"\n--- Crop info from {urlparse(url).netloc} ---\n"
                        crop_content += content[:1000] + "...\n"
                except Exception as e:
                    continue
            
            return crop_content if crop_content else f"Could not retrieve information about {crop}."
            
        except Exception as e:
            return f"Crop information scraping error: {str(e)}"
    
    def scrape_agricultural_news(self, topic: str = "agriculture") -> str:
        """Scrape latest agricultural news"""
        try:
            news_content = ""
            
            # Agricultural news URLs
            news_urls = [
                f"https://www.financialexpress.com/industry/agriculture/",
                f"https://economictimes.indiatimes.com/news/economy/agriculture"
            ]
            
            for url in news_urls[:1]:  # Limit to one source
                try:
                    content = get_website_text_content(url)
                    if content and len(content) > 100:
                        news_content += f"\n--- Agricultural news from {urlparse(url).netloc} ---\n"
                        news_content += content[:1200] + "...\n"
                        break
                except Exception as e:
                    continue
            
            return news_content if news_content else "Could not retrieve agricultural news."
            
        except Exception as e:
            return f"Agricultural news scraping error: {str(e)}"
    
    def scrape_research_information(self, topic: str) -> str:
        """Scrape research information from agricultural institutions"""
        try:
            research_content = ""
            
            # Research institution URLs
            research_urls = [
                f"https://www.icar.org.in/search?q={topic}",
                f"https://www.icrisat.org/search/?q={topic}"
            ]
            
            for url in research_urls[:1]:  # Limit to one source
                try:
                    content = get_website_text_content(url)
                    if content and len(content) > 100:
                        research_content += f"\n--- Research from {urlparse(url).netloc} ---\n"
                        research_content += content[:1000] + "...\n"
                        break
                except Exception as e:
                    continue
            
            return research_content if research_content else f"Could not retrieve research information about {topic}."
            
        except Exception as e:
            return f"Research information scraping error: {str(e)}"
    
    def intelligent_scrape(self, query: str, context: Dict = None) -> str:
        """Intelligently scrape based on query type and context"""
        try:
            query_lower = query.lower()
            scraped_content = ""
            
            # Determine scraping strategy based on query
            if any(keyword in query_lower for keyword in ["weather", "rain", "temperature", "forecast"]):
                location = context.get("location", "India") if context else "India"
                scraped_content += self.scrape_weather_info(location, query)
            
            elif any(keyword in query_lower for keyword in ["price", "market", "rate", "mandi"]):
                # Extract commodity from query
                commodity = self.extract_commodity_from_query(query)
                location = context.get("location", "") if context else ""
                scraped_content += self.scrape_market_prices(commodity, location)
            
            elif any(keyword in query_lower for keyword in ["scheme", "subsidy", "loan", "government", "policy"]):
                scraped_content += self.scrape_government_schemes()
            
            elif any(keyword in query_lower for keyword in ["crop", "seed", "planting", "harvest", "cultivation"]):
                crop = self.extract_crop_from_query(query)
                scraped_content += self.scrape_crop_information(crop)
            
            elif any(keyword in query_lower for keyword in ["news", "latest", "recent", "update"]):
                scraped_content += self.scrape_agricultural_news(query)
            
            elif any(keyword in query_lower for keyword in ["research", "study", "technique", "method"]):
                scraped_content += self.scrape_research_information(query)
            
            else:
                # General agricultural search
                scraped_content += self.scrape_crop_information(query)
            
            return scraped_content if scraped_content else "Could not find relevant information through web search."
            
        except Exception as e:
            return f"Intelligent scraping error: {str(e)}"
    
    def extract_commodity_from_query(self, query: str) -> str:
        """Extract commodity name from query"""
        commodities = [
            "rice", "wheat", "cotton", "sugarcane", "maize", "soybean",
            "groundnut", "onion", "potato", "tomato", "chilli", "turmeric"
        ]
        
        query_lower = query.lower()
        for commodity in commodities:
            if commodity in query_lower:
                return commodity
        
        return "agricultural products"
    
    def extract_crop_from_query(self, query: str) -> str:
        """Extract crop name from query"""
        crops = [
            "rice", "wheat", "cotton", "sugarcane", "maize", "corn", "soybean",
            "groundnut", "onion", "potato", "tomato", "chilli", "turmeric",
            "cardamom", "pepper", "coffee", "tea", "banana", "mango"
        ]
        
        query_lower = query.lower()
        for crop in crops:
            if crop in query_lower:
                return crop
        
        return "crops"
    
    def rate_limit_delay(self, min_delay: float = 1.0, max_delay: float = 3.0):
        """Add random delay to avoid rate limiting"""
        delay = random.uniform(min_delay, max_delay)
        time.sleep(delay)
    
    def validate_url(self, url: str) -> bool:
        """Validate URL before scraping"""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False
    
    def clean_scraped_content(self, content: str) -> str:
        """Clean and format scraped content"""
        try:
            # Remove excessive whitespace
            content = re.sub(r'\n+', '\n', content)
            content = re.sub(r'\s+', ' ', content)
            
            # Remove special characters
            content = re.sub(r'[^\w\s\u0900-\u097F.,;:!?()-]', ' ', content)
            
            # Remove very short lines
            lines = content.split('\n')
            cleaned_lines = [line.strip() for line in lines if len(line.strip()) > 20]
            
            return '\n'.join(cleaned_lines[:20])  # Limit to 20 lines
            
        except Exception:
            return content
