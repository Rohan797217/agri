import os
from typing import Dict, Any, List, Optional
import requests
import json
from groq import Groq

class BaseAgent:
    """Base class for all specialized agents"""
    
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.groq_client = None
        self.initialize_groq()
    
    def initialize_groq(self):
        """Initialize Groq client"""
        try:
            api_key = os.getenv("GROQ_API_KEY")
            if api_key:
                self.groq_client = Groq(api_key=api_key)
            else:
                print(f"Warning: GROQ_API_KEY not found for {self.agent_name}")
        except Exception as e:
            print(f"Error initializing Groq for {self.agent_name}: {str(e)}")
    
    def generate_response(self, prompt: str, system_message: str = None) -> str:
        """Generate response using Groq LLM"""
        try:
            if not self.groq_client:
                return "Error: Groq client not initialized. Please check your API key."
            
            messages = []
            
            if system_message:
                messages.append({
                    "role": "system",
                    "content": system_message
                })
            
            messages.append({
                "role": "user", 
                "content": prompt
            })
            
            response = self.groq_client.chat.completions.create(
                model="mixtral-8x7b-32768",
                messages=messages,
                temperature=0.3,
                max_tokens=2000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def web_search_fallback(self, query: str, domain: str = None) -> str:
        """Web search fallback when other data sources fail"""
        try:
            from utils.web_scraper import get_website_text_content
            
            # Construct search URLs based on domain
            urls = self.get_search_urls(query, domain)
            
            content = ""
            for url in urls[:2]:  # Limit to 2 URLs to avoid rate limiting
                try:
                    page_content = get_website_text_content(url)
                    if page_content:
                        content += f"\n--- Content from {url} ---\n{page_content[:1000]}...\n"
                except Exception as e:
                    continue
            
            return content if content else "No relevant information found through web search."
            
        except Exception as e:
            return f"Web search error: {str(e)}"
    
    def get_search_urls(self, query: str, domain: str = None) -> List[str]:
        """Get relevant URLs for web searching based on domain"""
        base_urls = {
            "weather": [
                f"https://weather.com/search?query={query.replace(' ', '+')}"
            ],
            "crop": [
                f"https://www.icar.org.in/search?q={query.replace(' ', '+')}"
            ],
            "market": [
                f"https://agmarknet.gov.in/SearchCmmMkt.aspx?Tx_Commodity={query.replace(' ', '+')}"
            ],
            "policy": [
                f"https://www.india.gov.in/search/site/{query.replace(' ', '+')}"
            ]
        }
        
        return base_urls.get(domain, [])
    
    def format_response_with_sources(self, content: str, sources: List[str] = None) -> str:
        """Format response with source citations"""
        if sources:
            source_text = "\n\n**Sources:**\n" + "\n".join([f"- {source}" for source in sources])
            return content + source_text
        return content
    
    def validate_response_quality(self, response: str) -> Dict[str, Any]:
        """Validate response quality and detect potential hallucinations"""
        quality_metrics = {
            "length": len(response),
            "has_specific_data": any(char.isdigit() for char in response),
            "has_sources": "source" in response.lower() or "reference" in response.lower(),
            "confidence_score": 0.8  # Default confidence
        }
        
        # Simple hallucination detection
        warning_phrases = [
            "i'm not sure", "i think", "probably", "might be", 
            "could be", "seems like", "appears to"
        ]
        
        uncertainty_count = sum(1 for phrase in warning_phrases if phrase in response.lower())
        quality_metrics["uncertainty_indicators"] = uncertainty_count
        
        if uncertainty_count > 2:
            quality_metrics["confidence_score"] = 0.5
            quality_metrics["warning"] = "⚠️ This response contains uncertainty indicators. Please verify information from official sources."
        
        return quality_metrics
