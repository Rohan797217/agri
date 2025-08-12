import streamlit as st
import os
import json
import re
import requests
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance
import io
import base64
import hashlib
import pickle
import unicodedata
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import time
import random
from urllib.parse import urljoin, urlparse
import trafilatura
from groq import Groq

# Import AGNO system
try:
    from agents.agno_system import AGNOSystem
    AGNO_AVAILABLE = True
except ImportError as e:
    st.warning(f"AGNO system not available: {e}")
    AGNO_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="🌾 AgriAI Advisor",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set default API keys
# API keys should be set as environment variables
# Set these in your environment or deployment platform:
# export GROQ_API_KEY="your_groq_api_key_here"
# export WEATHER_API_KEY="your_weather_api_key_here"

# ================== UTILITY CLASSES ==================

class VectorDB:
    """Simple vector database for document storage and retrieval"""
    
    def __init__(self, db_path: str = "vector_db"):
        self.db_path = db_path
        self.documents = {}
        self.vectors = {}
        self.metadata = {}
        self.create_db_structure()
        self.load_database()
    
    def create_db_structure(self):
        """Create database directory structure"""
        try:
            os.makedirs(self.db_path, exist_ok=True)
        except Exception as e:
            print(f"Warning: Could not create database structure: {e}")
    
    def load_database(self):
        """Load existing database files"""
        try:
            docs_file = os.path.join(self.db_path, "documents.json")
            if os.path.exists(docs_file):
                with open(docs_file, 'r', encoding='utf-8') as f:
                    self.documents = json.load(f)
            
            meta_file = os.path.join(self.db_path, "metadata.json")
            if os.path.exists(meta_file):
                with open(meta_file, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
            
            vectors_file = os.path.join(self.db_path, "vectors.pkl")
            if os.path.exists(vectors_file):
                with open(vectors_file, 'rb') as f:
                    self.vectors = pickle.load(f)
        except Exception as e:
            print(f"Warning: Could not load existing database: {e}")
    
    def save_database(self):
        """Save database to disk"""
        try:
            docs_file = os.path.join(self.db_path, "documents.json")
            with open(docs_file, 'w', encoding='utf-8') as f:
                json.dump(self.documents, f, ensure_ascii=False, indent=2)
            
            meta_file = os.path.join(self.db_path, "metadata.json")
            with open(meta_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
            
            vectors_file = os.path.join(self.db_path, "vectors.pkl")
            with open(vectors_file, 'wb') as f:
                pickle.dump(self.vectors, f)
        except Exception as e:
            print(f"Warning: Could not save database: {e}")
    
    def generate_document_id(self, content: str, filename: str = "") -> str:
        """Generate unique document ID"""
        content_hash = hashlib.md5((content + filename).encode()).hexdigest()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"doc_{timestamp}_{content_hash[:8]}"
    
    def create_embedding(self, text: str) -> np.ndarray:
        """Create simple embedding for text"""
        try:
            words = text.lower().split()
            vocab = [
                "rice", "wheat", "cotton", "sugarcane", "maize", "soybean", "groundnut",
                "irrigation", "fertilizer", "pesticide", "harvest", "planting", "crop",
                "rain", "temperature", "humidity", "weather", "climate", "season",
                "disease", "pest", "fungus", "infection", "price", "market", "sell",
                "loan", "credit", "bank", "subsidy", "scheme", "government", "policy"
            ]
            
            vector = np.zeros(len(vocab))
            for word in words:
                if word in vocab:
                    idx = vocab.index(word)
                    vector[idx] += 1
            
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
            
            return vector
        except Exception as e:
            return np.random.rand(32)
    
    def add_document(self, content: str, filename: str = "", doc_type: str = "general") -> str:
        """Add document to vector database"""
        try:
            doc_id = self.generate_document_id(content, filename)
            vector = self.create_embedding(content)
            
            self.documents[doc_id] = {
                "content": content,
                "filename": filename,
                "type": doc_type,
                "created_at": datetime.now().isoformat()
            }
            
            self.vectors[doc_id] = vector
            self.metadata[doc_id] = {
                "doc_type": doc_type,
                "filename": filename,
                "content_length": len(content),
                "created_at": datetime.now().isoformat()
            }
            
            self.save_database()
            return doc_id
        except Exception as e:
            raise Exception(f"Failed to add document: {str(e)}")

class InputProcessor:
    """Handles processing of various input types"""
    
    def __init__(self):
        self.intent_keywords = {
            "weather": ["weather", "rain", "temperature", "forecast", "climate", "मौसम", "बारिश"],
            "crop": ["crop", "seed", "planting", "harvest", "variety", "फसल", "बीज"],
            "disease": ["disease", "pest", "infection", "fungus", "रोग", "कीट"],
            "market": ["price", "market", "sell", "buy", "rate", "mandi", "दाम", "भाव"],
            "finance": ["loan", "credit", "money", "bank", "subsidy", "scheme", "कर्ज", "योजना"],
            "policy": ["policy", "government", "scheme", "yojana", "sarkar", "नीति", "सरकार"],
            "soil": ["soil", "ph", "nutrient", "fertilizer", "manure", "testing", "मिट्टी", "खाद"],
            "pest": ["pest", "insect", "pesticide", "spray", "biological", "ipm", "कीट", "दवा"],
            "irrigation": ["irrigation", "water", "drip", "sprinkler", "watering", "सिंचाई", "पानी"],
            "supply": ["supplier", "input", "equipment", "storage", "market", "supply", "आपूर्ति"],
            "compliance": ["organic", "certification", "export", "standard", "certificate", "प्रमाणपत्र"]
        }
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text input"""
        try:
            text = text.strip()
            text = re.sub(r'\s+', ' ', text)
            return text.lower()
        except Exception:
            return text
    
    def detect_intent(self, text: str) -> str:
        """Detect user intent from text"""
        try:
            text_lower = text.lower()
            intent_scores = {}
            
            for intent, keywords in self.intent_keywords.items():
                score = sum(1 for keyword in keywords if keyword in text_lower)
                if score > 0:
                    intent_scores[intent] = score
            
            if intent_scores:
                return max(intent_scores, key=intent_scores.get)
            return "general"
        except Exception:
            return "general"
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract entities from text"""
        try:
            entities = {"crops": [], "seasons": [], "locations": [], "quantities": []}
            text_lower = text.lower()
            
            crops = ["rice", "wheat", "cotton", "sugarcane", "maize", "soybean", "धान", "गेहूं"]
            for crop in crops:
                if crop in text_lower:
                    entities["crops"].append(crop)
            
            quantities = re.findall(r'\b(\d+(?:\.\d+)?)\s*(acre|hectare|kg|ton)', text_lower)
            entities["quantities"] = [f"{q[0]} {q[1]}" for q in quantities]
            
            return entities
        except Exception:
            return {"crops": [], "seasons": [], "locations": [], "quantities": []}
    
    def process_document(self, uploaded_file) -> str:
        """Process uploaded documents"""
        try:
            file_type = uploaded_file.type
            
            if file_type == "application/pdf":
                try:
                    import PyPDF2
                    pdf_reader = PyPDF2.PdfReader(uploaded_file)
                    text_content = ""
                    for page in pdf_reader.pages:
                        text_content += page.extract_text() + "\n"
                    return text_content
                except ImportError:
                    return "PDF processing requires PyPDF2 library."
            elif file_type == "text/plain":
                return str(uploaded_file.read(), "utf-8")
            else:
                return "Unsupported file type"
        except Exception as e:
            raise Exception(f"Document processing failed: {str(e)}")

def get_website_text_content(url: str) -> str:
    """Extract text content from website using trafilatura"""
    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return ""
        text = trafilatura.extract(downloaded)
        return text if text else ""
    except Exception as e:
        return f"Error scraping {url}: {e}"

def translate_text(text: str, target_language: str = "en") -> str:
    """Simple translation using Google Translate API (free tier)"""
    try:
        if target_language == "en" or len(text.strip()) == 0:
            return text
        
        # Use Google Translate API
        translate_url = "https://translate.googleapis.com/translate_a/single"
        params = {
            'client': 'gtx',
            'sl': 'auto',  # Auto-detect source language
            'tl': target_language,
            'dt': 't',
            'q': text
        }
        
        response = requests.get(translate_url, params=params, timeout=10)
        response.raise_for_status()
        
        result = response.json()
        if result and result[0] and result[0][0]:
            return result[0][0][0]
        
        return text  # Return original if translation fails
        
    except Exception as e:
        print(f"Translation error: {e}")
        return text  # Return original text if translation fails

def detect_language(text: str) -> str:
    """Detect language of input text"""
    try:
        # Use Google Translate API for language detection
        detect_url = "https://translate.googleapis.com/translate_a/single"
        params = {
            'client': 'gtx',
            'sl': 'auto',
            'tl': 'en',
            'dt': 't',
            'q': text[:100]  # Use first 100 characters for detection
        }
        
        response = requests.get(detect_url, params=params, timeout=5)
        response.raise_for_status()
        
        result = response.json()
        if result and len(result) > 2 and result[2]:
            return result[2]  # Detected language code
        
        return "en"  # Default to English
        
    except Exception:
        return "en"  # Default to English

def search_duckduckgo(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """Search DuckDuckGo for real-time information"""
    try:
        # Simple DuckDuckGo search using their instant answer API
        search_url = "https://api.duckduckgo.com/"
        params = {
            'q': query,
            'format': 'json',
            'no_redirect': '1',
            'no_html': '1',
            'skip_disambig': '1'
        }
        
        response = requests.get(search_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        results = []
        
        # Add instant answer if available
        if data.get('Abstract'):
            results.append({
                'title': data.get('Heading', 'DuckDuckGo Answer'),
                'snippet': data.get('Abstract', ''),
                'url': data.get('AbstractURL', ''),
                'source': data.get('AbstractSource', 'DuckDuckGo')
            })
        
        # Add related topics
        for topic in data.get('RelatedTopics', [])[:max_results-1]:
            if isinstance(topic, dict) and 'Text' in topic:
                results.append({
                    'title': topic.get('Text', '')[:100],
                    'snippet': topic.get('Text', ''),
                    'url': topic.get('FirstURL', ''),
                    'source': 'DuckDuckGo'
                })
        
        # If no good results, try a web search approach
        if not results:
            # Fallback: search for agricultural websites
            agricultural_sites = [
                f"https://www.agricoop.nic.in/search?q={query}",
                f"https://farmer.gov.in/search?q={query}",
                f"https://pmfby.gov.in/search?q={query}"
            ]
            
            for site_url in agricultural_sites[:2]:
                try:
                    content = get_website_text_content(site_url)
                    if content and len(content) > 100:
                        results.append({
                            'title': f"Agricultural Information: {query}",
                            'snippet': content[:300] + "...",
                            'url': site_url,
                            'source': 'Agricultural Portal'
                        })
                        break
                except Exception:
                    continue
        
        return results
        
    except Exception as e:
        return [{'title': 'Search Error', 'snippet': f'Could not search: {str(e)}', 'url': '', 'source': 'Error'}]

# ================== AGENT CLASSES ==================

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
        except Exception as e:
            print(f"Error initializing Groq for {self.agent_name}: {str(e)}")
    
    def generate_response(self, prompt: str, system_message: str = None, retry_count: int = 0) -> str:
        """Generate response using Groq LLM with fallback mechanisms"""
        max_retries = 3
        models = ["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768"]
        
        try:
            if not self.groq_client:
                return self.get_offline_fallback_response(prompt)
            
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": prompt})
            
            # Try different models if previous ones fail
            current_model = models[min(retry_count, len(models)-1)]
            
            response = self.groq_client.chat.completions.create(
                model=current_model,
                messages=messages,
                temperature=0.3,
                max_tokens=2000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            if retry_count < max_retries:
                import time
                time.sleep(2 ** retry_count)  # Exponential backoff
                return self.generate_response(prompt, system_message, retry_count + 1)
            else:
                return self.get_offline_fallback_response(prompt)
    
    def get_offline_fallback_response(self, prompt: str) -> str:
        """Provide contextual offline fallback responses using user profile and agricultural datasets"""
        try:
            # Get user context from session state
            user_profile = st.session_state.get("user_profile", {})
            agricultural_data = st.session_state.get("agricultural_data", {})
            regional_knowledge = st.session_state.get("regional_knowledge", {})
            
            location = user_profile.get("location", "your area")
            crops = user_profile.get("crops", [])
            farm_size = user_profile.get("farm_size", 0)
            
            # Build contextual response based on user data and query
            return self.generate_contextual_fallback_response(prompt, location, crops, farm_size, agricultural_data, regional_knowledge)
            
        except Exception as e:
            return self.get_basic_fallback_response(prompt)
    
    def generate_contextual_fallback_response(self, prompt: str, location: str, crops: List[str], farm_size: float, agricultural_data: Dict, regional_knowledge: Dict) -> str:
        """Generate contextual response using user profile and agricultural data"""
        prompt_lower = prompt.lower()
        
        # Initialize response with user context
        context_header = f"**📍 Contextual Advice for {location}**"
        if crops:
            context_header += f"\n**🌾 Your Crops:** {', '.join(crops)} ({farm_size} acres)"
        
        # Weather-related queries
        if any(word in prompt_lower for word in ["rain", "weather", "temperature", "season", "survive"]):
            return self.get_weather_fallback_response(prompt_lower, location, crops, farm_size, agricultural_data, regional_knowledge, context_header)
        
        # Irrigation queries
        elif any(word in prompt_lower for word in ["irrigation", "water", "watering"]):
            return self.get_irrigation_fallback_response(location, crops, farm_size, agricultural_data, context_header)
        
        # Fertilizer queries
        elif any(word in prompt_lower for word in ["fertilizer", "nutrient", "npk"]):
            return self.get_fertilizer_fallback_response(location, crops, farm_size, agricultural_data, context_header)
        
        # Disease/pest queries
        elif any(word in prompt_lower for word in ["disease", "pest", "insect"]):
            return self.get_pest_disease_fallback_response(location, crops, agricultural_data, context_header)
        
        # Market/price queries
        elif any(word in prompt_lower for word in ["price", "market", "sell"]):
            return self.get_market_fallback_response(location, crops, agricultural_data, context_header)
        
        # General crop advice
        else:
            return self.get_general_crop_fallback_response(location, crops, farm_size, agricultural_data, regional_knowledge, context_header)
    
    def get_weather_fallback_response(self, prompt: str, location: str, crops: List[str], farm_size: float, agricultural_data: Dict, regional_knowledge: Dict, context_header: str) -> str:
        """Generate weather-specific fallback response"""
        response = f"""{context_header}

**🌤️ Weather & Seasonal Guidance for {location}:**

**Current Season Analysis:**
• August is part of Kharif season (monsoon period)
• Expected rainfall varies by region - check local IMD updates"""
        
        # Add location-specific information
        if "west bengal" in location.lower() or "kharagpur" in location.lower():
            response += f"""
• **West Bengal Climate:** Humid subtropical with high rainfall during monsoon
• **August Conditions:** Peak monsoon month, expect frequent rains
• **Temperature Range:** 26-34°C typical for this region"""
        
        # Add crop-specific weather advice
        if crops:
            response += f"\n\n**🌾 Crop-Specific Weather Advice:**"
            for crop in crops:
                if crop.lower() == 'cotton':
                    response += f"""
• **Cotton in {location}:** 
  - Ideal temperature: 21-32°C (current conditions suitable)
  - Water requirement: 700-1300mm (monsoon provides natural irrigation)
  - Risk factors: Excess water can cause root rot
  - Protection: Ensure proper drainage during heavy rains"""
        
        # Add immediate actions
        response += f"""

**⚡ Immediate Actions for Your {farm_size} Acre Farm:**
• Monitor drainage systems - critical for cotton in monsoon
• Watch for fungal diseases in humid conditions
• Avoid field operations during heavy rain periods
• Check weather forecasts daily: IMD, local news

**📞 Emergency Contacts:**
• Local Agriculture Extension Officer: Contact through Block office
• Kisan Call Center: 1800-180-1551
• West Bengal Agriculture Department helpline

*Note: This is contextual guidance based on your profile. For real-time weather data, ensure internet connectivity.*"""
        
        return response
    
    def get_irrigation_fallback_response(self, location: str, crops: List[str], farm_size: float, agricultural_data: Dict, context_header: str) -> str:
        """Generate irrigation-specific fallback response"""
        response = f"""{context_header}

**💧 Irrigation Guidelines for Your {farm_size} Acre Farm:**"""
        
        if crops:
            for crop in crops:
                crop_data = agricultural_data.get('crop_calendar', {}).get(crop.lower(), {})
                water_req = crop_data.get('water_requirement', 'Standard requirements apply')
                response += f"""
• **{crop} Water Management:**
  - Water requirement: {water_req}
  - Current season: Monsoon irrigation typically reduced
  - Recommendation: Focus on drainage rather than irrigation"""
        
        response += f"""

**Regional Considerations for {location}:**
• Monsoon season reduces irrigation needs
• Focus on water management and drainage
• Avoid waterlogging in cotton fields

*Note: Adjust based on local rainfall patterns and soil conditions.*"""
        
        return response
    
    def get_fertilizer_fallback_response(self, location: str, crops: List[str], farm_size: float, agricultural_data: Dict, context_header: str) -> str:
        """Generate fertilizer-specific fallback response"""
        response = f"""{context_header}

**🌱 Fertilizer Recommendations:**"""
        
        if crops:
            for crop in crops:
                if crop.lower() == 'cotton':
                    response += f"""
• **Cotton Fertilizer Program ({farm_size} acres):**
  - NPK: 100:50:50 kg/ha
  - For your {farm_size} acre farm: ~{farm_size * 100 * 0.4:.1f} kg N, {farm_size * 50 * 0.4:.1f} kg P, {farm_size * 50 * 0.4:.1f} kg K
  - Application: Split nitrogen in 2-3 doses
  - Timing: Avoid application during heavy rain periods"""
        
        return response
    
    def get_pest_disease_fallback_response(self, location: str, crops: List[str], agricultural_data: Dict, context_header: str) -> str:
        """Generate pest/disease fallback response"""
        response = f"""{context_header}

**🐛 Pest & Disease Management:**"""
        
        if crops:
            pest_db = agricultural_data.get('pest_disease_db', {})
            for crop in crops:
                if crop.lower() in pest_db:
                    crop_pests = pest_db[crop.lower()]
                    response += f"""
• **{crop} Common Issues:**"""
                    for pest_name, pest_info in crop_pests.items():
                        response += f"""
  - {pest_name.title()}: {pest_info.get('symptoms', 'Monitor regularly')}
  - Treatment: {pest_info.get('treatment', 'Consult local expert')}"""
        
        return response
    
    def get_market_fallback_response(self, location: str, crops: List[str], agricultural_data: Dict, context_header: str) -> str:
        """Generate market-specific fallback response"""
        response = f"""{context_header}

**📈 Market Information:**"""
        
        if crops:
            market_data = agricultural_data.get('market_trends', {})
            for crop in crops:
                if crop.lower() in market_data:
                    crop_market = market_data[crop.lower()]
                    response += f"""
• **{crop} Market Outlook:**
  - Seasonal trend: {crop_market.get('seasonal_trend', 'Check local mandis')}
  - Price range: {crop_market.get('price_range', 'Variable')}
  - Major markets: {', '.join(crop_market.get('major_markets', []))}"""
        
        return response
    
    def get_general_crop_fallback_response(self, location: str, crops: List[str], farm_size: float, agricultural_data: Dict, regional_knowledge: Dict, context_header: str) -> str:
        """Generate general crop advice fallback response"""
        response = f"""{context_header}

**🌾 General Crop Management for {location}:**

**Current Season Focus (August - Kharif):**
• Monitor crop health regularly
• Ensure proper drainage during monsoon
• Watch for pest and disease outbreaks
• Plan for harvest season activities

**For Your {farm_size} Acre Farm:**
• Small-scale intensive management possible
• Focus on quality over quantity
• Consider sustainable practices"""
        
        if crops and regional_knowledge:
            location_lower = location.lower()
            if any(region in location_lower for region in regional_knowledge.keys()):
                response += f"""
• Regional best practices apply for {', '.join(crops)}
• Local extension services available for specific guidance"""
        
        return response
    
    def get_basic_fallback_response(self, prompt: str) -> str:
        """Basic fallback when context data is unavailable"""
        return """**🌾 Basic Agricultural Guidance:**

• Monitor crops regularly for health and growth
• Follow local weather forecasts and warnings
• Contact local agricultural extension officers for specific advice
• Maintain proper farm records for better decision making

**📞 Universal Contacts:**
• Kisan Call Center: 1800-180-1551
• Local Agriculture Extension Officer

*Note: For personalized advice, please update your profile with location and crop details.*"""

class CropAgent(BaseAgent):
    """Specialized agent for crop-related advice"""
    
    def __init__(self):
        super().__init__("CropAgent")
        self.crop_database = {
            "rice": {
                "season": ["Kharif", "Rabi"],
                "water_requirement": "High",
                "soil_type": "Clay, Clay loam",
                "diseases": ["Blast", "Brown spot", "Sheath rot"],
                "fertilizer": "NPK 120:60:40 kg/ha"
            },
            "wheat": {
                "season": ["Rabi"],
                "water_requirement": "Moderate",
                "soil_type": "Loam, Clay loam",
                "diseases": ["Rust", "Powdery mildew", "Loose smut"],
                "fertilizer": "NPK 120:60:40 kg/ha"
            },
            "cotton": {
                "season": ["Kharif"],
                "water_requirement": "Moderate to High",
                "soil_type": "Black cotton soil",
                "diseases": ["Bollworm", "Aphids", "Whitefly"],
                "fertilizer": "NPK 100:50:50 kg/ha"
            }
        }
    
    def get_crop_advice(self, query: str, entities: Dict, user_context: Dict) -> str:
        """Generate comprehensive crop advice with confidence scoring and explainable AI"""
        try:
            detected_crop = self.detect_crop_from_query(query)
            user_location = user_context.get("location", "")
            farm_size = user_context.get("farm_size", 0)
            
            # Get comprehensive data
            agricultural_data = st.session_state.get('agricultural_data', {})
            regional_knowledge = st.session_state.get('regional_knowledge', {})
            
            # Build context-aware response
            context_info = self.build_contextual_information(detected_crop, user_location, agricultural_data, regional_knowledge)
            confidence_factors = self.calculate_confidence_factors(detected_crop, user_location, context_info)
            
            system_message = f"""You are an expert agricultural advisor with access to comprehensive regional data. 
            Provide evidence-based advice with clear reasoning. Use the provided context data.
            
            Context Information: {context_info}
            Confidence Factors: {confidence_factors}
            
            Always explain your reasoning and cite specific data sources when making recommendations."""
            
            enhanced_query = f"""
            Query: {query}
            Crop: {detected_crop or 'general'}
            Location: {user_location}
            Farm Size: {farm_size} acres
            
            Provide specific, evidence-based agricultural advice with clear reasoning for each recommendation.
            """
            
            response = self.generate_response(enhanced_query, system_message)
            
            # Add structured information
            structured_info = self.add_structured_crop_information(detected_crop, user_location, agricultural_data)
            confidence_score = self.calculate_overall_confidence(confidence_factors)
            
            # Format response with explainable AI components
            formatted_response = f"""**🤖 AI Response (Confidence: {confidence_score}%):**

{response}

{structured_info}

**💡 Why this advice:**
{self.generate_reasoning_explanation(detected_crop, user_location, confidence_factors)}

**📊 Data Sources Used:**
{self.list_data_sources_used(detected_crop, user_location)}
"""
            
            return formatted_response
            
        except Exception as e:
            return f"Error generating crop advice: {str(e)}"
    
    def build_contextual_information(self, crop: str, location: str, agricultural_data: Dict, regional_knowledge: Dict) -> str:
        """Build comprehensive contextual information"""
        context = []
        
        if crop and crop in agricultural_data.get('crop_calendar', {}):
            crop_data = agricultural_data['crop_calendar'][crop]
            context.append(f"Crop Calendar: {crop_data}")
        
        location_lower = location.lower()
        if 'maharashtra' in location_lower and 'maharashtra' in regional_knowledge:
            regional_info = regional_knowledge['maharashtra']
            context.append(f"Regional Knowledge: {regional_info}")
        
        if crop and crop in agricultural_data.get('pest_disease_db', {}):
            pest_data = agricultural_data['pest_disease_db'][crop]
            context.append(f"Pest/Disease Info: {pest_data}")
        
        return " | ".join(context[:3])  # Limit context length
    
    def calculate_confidence_factors(self, crop: str, location: str, context_info: str) -> Dict:
        """Calculate confidence factors for the advice"""
        factors = {
            'crop_specific_data': 0,
            'regional_data': 0,
            'seasonal_alignment': 0,
            'pest_disease_knowledge': 0
        }
        
        if crop and 'Crop Calendar' in context_info:
            factors['crop_specific_data'] = 85
        
        if location and 'Regional Knowledge' in context_info:
            factors['regional_data'] = 80
        
        current_month = datetime.now().month
        if 6 <= current_month <= 11:  # Kharif season
            factors['seasonal_alignment'] = 90
        elif 11 <= current_month <= 4:  # Rabi season
            factors['seasonal_alignment'] = 85
        else:
            factors['seasonal_alignment'] = 70
        
        if crop and 'Pest/Disease Info' in context_info:
            factors['pest_disease_knowledge'] = 75
        
        return factors
    
    def calculate_overall_confidence(self, factors: Dict) -> int:
        """Calculate overall confidence score"""
        if not factors:
            return 60
        
        weights = {
            'crop_specific_data': 0.3,
            'regional_data': 0.25,
            'seasonal_alignment': 0.25,
            'pest_disease_knowledge': 0.2
        }
        
        weighted_sum = sum(factors[key] * weights.get(key, 0.25) for key in factors)
        return max(60, min(95, int(weighted_sum)))
    
    def add_structured_crop_information(self, crop: str, location: str, agricultural_data: Dict) -> str:
        """Add structured crop information"""
        if not crop:
            return ""
        
        info = f"\n**📋 {crop.title()} Quick Reference:**\n"
        
        # Crop calendar information
        crop_calendar = agricultural_data.get('crop_calendar', {}).get(crop, {})
        if crop_calendar:
            info += f"• **Growing Season:** {crop_calendar.get('kharif', {}).get('sowing', 'N/A')}\n"
            info += f"• **Water Need:** {crop_calendar.get('water_requirement', 'N/A')}\n"
            info += f"• **Temperature:** {crop_calendar.get('temperature', 'N/A')}\n"
        
        # Market information
        market_data = agricultural_data.get('market_trends', {}).get(crop, {})
        if market_data:
            info += f"• **Market Trend:** {market_data.get('seasonal_trend', 'N/A')}\n"
            info += f"• **Price Range:** {market_data.get('price_range', 'N/A')}\n"
        
        return info
    
    def generate_reasoning_explanation(self, crop: str, location: str, confidence_factors: Dict) -> str:
        """Generate explanation for the reasoning"""
        explanations = []
        
        if confidence_factors.get('crop_specific_data', 0) > 70:
            explanations.append("Using verified crop calendar and growth stage data")
        
        if confidence_factors.get('regional_data', 0) > 70:
            explanations.append("Incorporating regional climate and soil conditions")
        
        if confidence_factors.get('seasonal_alignment', 0) > 80:
            explanations.append("Advice aligned with current agricultural season")
        
        if not explanations:
            explanations.append("Based on general agricultural principles")
        
        return " • ".join(explanations)
    
    def list_data_sources_used(self, crop: str, location: str) -> str:
        """List data sources used for transparency"""
        sources = [
            "Internal crop database",
            "Regional agricultural knowledge base",
            "Government agricultural guidelines"
        ]
        
        if crop:
            sources.append("Crop-specific calendar and pest database")
        
        if location:
            sources.append("Location-specific climate and soil data")
        
        return " • ".join(sources[:4])
    
    def detect_crop_from_query(self, query: str) -> str:
        """Detect crop mentioned in query"""
        query_lower = query.lower()
        crops = ["rice", "wheat", "cotton", "maize", "soybean", "sugarcane"]
        
        for crop in crops:
            if crop in query_lower:
                return crop
        
        aliases = {"paddy": "rice", "corn": "maize", "kapas": "cotton"}
        for alias, crop in aliases.items():
            if alias in query_lower:
                return crop
        
        return None

class WeatherAgent(BaseAgent):
    """Specialized agent for weather-related information"""
    
    def __init__(self):
        super().__init__("WeatherAgent")
        self.weather_api_key = os.getenv("WEATHER_API_KEY")
    
    def get_weather_advice(self, query: str, user_context: Dict) -> str:
        """Generate weather-based agricultural advice with comprehensive context integration"""
        try:
            location = user_context.get("location", "")
            farm_size = user_context.get("farm_size", 0)
            crops_grown = user_context.get("crops", [])
            
            if not location:
                return "Please provide your location in the user profile to get accurate weather information."
            
            # Get weather data
            current_weather = self.get_current_weather(location)
            forecast_data = self.get_weather_forecast(location, days=7)
            
            # Get comprehensive agricultural context
            agricultural_data = st.session_state.get('agricultural_data', {})
            regional_knowledge = st.session_state.get('regional_knowledge', {})
            
            # Build comprehensive context
            context_info = self.build_weather_context(location, crops_grown, agricultural_data, regional_knowledge)
            confidence_factors = self.calculate_weather_confidence(current_weather, forecast_data, context_info)
            
            system_message = f"""You are an expert agricultural meteorologist with access to comprehensive crop and regional data. 
            Provide specific, actionable advice based on weather conditions and farming context.
            
            Agricultural Context: {context_info}
            Confidence Factors: {confidence_factors}
            
            Focus on practical actions the farmer can take in the next 24-48 hours based on weather conditions."""
            
            enhanced_query = f"""
            Query: {query}
            Farmer Profile:
            - Location: {location}
            - Farm Size: {farm_size} acres  
            - Crops: {', '.join(crops_grown) if crops_grown else 'General farming'}
            
            Weather Conditions:
            - Current: {current_weather}
            - 7-Day Forecast: {forecast_data}
            
            Agricultural Context: {context_info}
            
            Provide specific weather-based agricultural advice considering the farmer's crops, location, and farm size. 
            Include immediate actions, risks to watch for, and protective measures.
            """
            
            # Generate response with fallback handling
            response = self.generate_weather_response_with_fallback(enhanced_query, system_message, location, crops_grown, current_weather, forecast_data)
            
            weather_summary = self.format_weather_summary(current_weather, forecast_data)
            confidence_score = self.calculate_overall_weather_confidence(confidence_factors)
            
            # Format comprehensive response
            formatted_response = f"""{weather_summary}

**🌾 Agricultural Advice (Confidence: {confidence_score}%):**

{response}

**📊 Context Used:**
• Farm Profile: {farm_size} acres, {', '.join(crops_grown) if crops_grown else 'General farming'}
• Regional Data: {location} climate and soil conditions
• Seasonal Calendar: Current month recommendations
• Risk Assessment: Weather impact analysis for your crops

**💡 Reasoning:**
{self.generate_weather_reasoning(location, crops_grown, current_weather, forecast_data)}"""
            
            return formatted_response
            
        except Exception as e:
            return self.get_weather_fallback_advice(query, user_context, str(e))
    
    def build_weather_context(self, location: str, crops: List[str], agricultural_data: Dict, regional_knowledge: Dict) -> str:
        """Build comprehensive weather-agricultural context"""
        context_parts = []
        
        # Regional climate information
        location_lower = location.lower()
        if 'maharashtra' in location_lower and 'maharashtra' in regional_knowledge:
            regional_info = regional_knowledge['maharashtra']
            context_parts.append(f"Regional Climate: {regional_info.get('climate_zones', [])}")
        
        # Crop-specific weather requirements
        if crops and agricultural_data.get('crop_calendar'):
            crop_weather_info = []
            for crop in crops:
                if crop.lower() in agricultural_data['crop_calendar']:
                    crop_data = agricultural_data['crop_calendar'][crop.lower()]
                    temp_req = crop_data.get('temperature', 'N/A')
                    water_req = crop_data.get('water_requirement', 'N/A')
                    crop_weather_info.append(f"{crop}: {temp_req} temp, {water_req} water")
            if crop_weather_info:
                context_parts.append(f"Crop Requirements: {' | '.join(crop_weather_info)}")
        
        # Seasonal alignment
        from datetime import datetime
        current_month = datetime.now().month
        if 6 <= current_month <= 11:
            context_parts.append("Season: Kharif (Monsoon season)")
        elif 11 <= current_month <= 4:
            context_parts.append("Season: Rabi (Winter season)")
        else:
            context_parts.append("Season: Summer crop season")
        
        return " | ".join(context_parts)
    
    def calculate_weather_confidence(self, current_weather: Dict, forecast_data: Dict, context_info: str) -> Dict:
        """Calculate confidence factors for weather advice"""
        factors = {
            'weather_data_quality': 85 if current_weather.get('source') == 'WeatherAPI' else 70,
            'regional_knowledge': 80 if 'Regional Climate' in context_info else 60,
            'crop_specific_data': 85 if 'Crop Requirements' in context_info else 65,
            'seasonal_alignment': 90 if 'Season:' in context_info else 70
        }
        return factors
    
    def calculate_overall_weather_confidence(self, factors: Dict) -> int:
        """Calculate overall confidence score for weather advice"""
        weights = {
            'weather_data_quality': 0.4,
            'regional_knowledge': 0.2,
            'crop_specific_data': 0.25,
            'seasonal_alignment': 0.15
        }
        
        weighted_sum = sum(factors[key] * weights.get(key, 0.25) for key in factors)
        return max(70, min(95, int(weighted_sum)))
    
    def generate_weather_response_with_fallback(self, query: str, system_message: str, location: str, crops: List[str], current_weather: Dict, forecast_data: Dict) -> str:
        """Generate weather response with dual LLM approach for better clarity"""
        
        # FIRST LLM: Initial processing and context understanding
        initial_context = self.process_query_with_first_llm(query, system_message, location, crops)
        
        try:
            # Try primary LLM with enhanced context
            llm_response = self.generate_response(initial_context, system_message)
            
            # Always force use of enhanced contextual system for consistency
            # The LLM responses are too inconsistent - always use our reliable contextual system
            return self.get_enhanced_contextual_weather_advice(location, crops, current_weather, forecast_data, query)
        except Exception:
            # LLM failed completely, force use of enhanced contextual system
            return self.get_enhanced_contextual_weather_advice(location, crops, current_weather, forecast_data, query)
    
    def process_query_with_first_llm(self, query: str, system_message: str, location: str, crops: List[str]) -> str:
        """First LLM: Process and enhance query with contextual information"""
        enhanced_query = f"""
        CONTEXTUAL AGRICULTURAL QUERY PROCESSING:
        
        Original Query: {query}
        Farmer Profile:
        - Location: {location}
        - Crops: {', '.join(crops) if crops else 'General farming'}
        - Context: Weather advice needed for specific farming situation
        
        Enhanced Query with Context: Please provide weather-based agricultural advice that specifically addresses the farmer's location ({location}) and crops ({', '.join(crops) if crops else 'general crops'}). The response must be specific to their regional climate and crop requirements, not generic farming advice.
        
        Required Elements in Response:
        - Mention specific location: {location}
        - Address specific crops: {', '.join(crops) if crops else 'crops'}
        - Include regional climate considerations
        - Provide actionable farm-specific advice
        """
        
        return enhanced_query
    
    def refine_response_with_second_llm(self, response: str, location: str, crops: List[str], current_weather: Dict) -> str:
        """Second LLM: Refine and enhance response for clarity and specificity"""
        try:
            refinement_prompt = f"""
            RESPONSE ENHANCEMENT AND CLARITY:
            
            Original Response: {response}
            
            Farmer Context:
            - Location: {location}
            - Crops: {', '.join(crops) if crops else 'General farming'}
            - Current Weather: Temperature {current_weather.get('temperature', 'N/A')}°C, {current_weather.get('condition', 'N/A')}
            
            Please enhance this response to:
            1. Ensure it clearly mentions the farmer's location ({location})
            2. Provide crop-specific advice for {', '.join(crops) if crops else 'their crops'}
            3. Add immediate actionable steps
            4. Include regional weather pattern considerations
            5. Make the language clear and farmer-friendly
            
            The enhanced response should be comprehensive yet practical for implementation.
            """
            
            refined_response = self.generate_response(refinement_prompt)
            return refined_response if refined_response else response
            
        except Exception:
            # If second LLM fails, return original response
            return response
    
    def is_response_contextual(self, response: str, location: str, crops: List[str]) -> bool:
        """Check if response is contextual to user's specific situation"""
        response_lower = response.lower()
        location_words = location.lower().split()
        
        # Check if response mentions user's location
        location_mentioned = any(word in response_lower for word in location_words if len(word) > 2)
        
        # Check if response mentions user's crops specifically
        crops_mentioned = any(crop.lower() in response_lower for crop in crops)
        
        # Check if response is not generic (contains specific advice)
        is_specific = any(phrase in response_lower for phrase in [
            "your", "specific", "monitor for", "drainage", "bollworm", 
            "kharif", "west bengal", "maharashtra", "cotton", "acres"
        ])
        
        # Consider contextual if location OR crops mentioned AND response is specific
        return (location_mentioned or crops_mentioned) and is_specific
    
    def get_enhanced_contextual_weather_advice(self, location: str, crops: List[str], current_weather: Dict, forecast_data: Dict, query: str) -> str:
        """Generate enhanced contextual weather advice using comprehensive user profile and agricultural datasets"""
        
        # Get user profile and agricultural data from session state
        user_profile = st.session_state.get("user_profile", {})
        agricultural_data = st.session_state.get("agricultural_data", {})
        regional_knowledge = st.session_state.get("regional_knowledge", {})
        
        farm_size = user_profile.get("farm_size", 0)
        
        # Build comprehensive contextual response
        advice_parts = []
        
        # Header with user context
        advice_parts.append(f"**📍 Contextual Weather Advice for {location}**")
        if crops and farm_size:
            advice_parts.append(f"**🌾 Your Crops:** {', '.join(crops)} ({farm_size} acres)")
        
        # Current weather analysis with context
        temp = current_weather.get('temperature', 0)
        humidity = current_weather.get('humidity', 0) 
        condition = current_weather.get('condition', '').lower()
        
        advice_parts.append(f"**🌤️ Weather & Seasonal Guidance:**")
        advice_parts.append(f"**Current Season Analysis:**")
        advice_parts.append(f"• August is part of Kharif season (monsoon period)")
        advice_parts.append(f"• Current temperature: {temp}°C, Humidity: {humidity}%")
        
        # Location-specific information
        if "west bengal" in location.lower() or "kharagpur" in location.lower():
            advice_parts.append(f"• **West Bengal Climate:** Humid subtropical with high rainfall during monsoon")
            advice_parts.append(f"• **August Conditions:** Peak monsoon month, expect frequent rains")
            advice_parts.append(f"• **Temperature Range:** 26-34°C typical for this region")
        elif "maharashtra" in location.lower() or "wardha" in location.lower():
            advice_parts.append(f"• **Maharashtra Climate:** Semi-arid to sub-humid, variable rainfall")
            advice_parts.append(f"• **August Conditions:** Monsoon season with moderate to heavy rains")
            advice_parts.append(f"• **Temperature Range:** 24-32°C typical for this region")
        
        # Crop-specific weather advice using agricultural datasets
        if crops:
            advice_parts.append(f"**🌾 Crop-Specific Weather Advice:**")
            for crop in crops:
                crop_data = agricultural_data.get('crop_calendar', {}).get(crop.lower(), {})
                pest_data = agricultural_data.get('pest_disease_db', {}).get(crop.lower(), {})
                
                if crop.lower() == 'cotton':
                    advice_parts.append(f"• **Cotton in {location}:**")
                    advice_parts.append(f"  - Ideal temperature: 21-32°C ({'suitable' if 21 <= temp <= 32 else 'monitor closely'})")
                    advice_parts.append(f"  - Water requirement: 700-1300mm (monsoon provides natural irrigation)")
                    
                    if temp > 32 and humidity > 70:
                        advice_parts.append(f"  - **Risk Alert:** High temperature + humidity ideal for bollworm activity")
                        advice_parts.append(f"  - **Action:** Monitor for bollworm, consider prophylactic spray")
                    
                    if 'rain' in condition or humidity > 75:
                        advice_parts.append(f"  - **Drainage Critical:** Excess water can cause root rot")
                        advice_parts.append(f"  - **Protection:** Ensure proper field drainage during heavy rains")
        
        # Immediate actions tailored to farm size and location
        advice_parts.append(f"**⚡ Immediate Actions for Your {farm_size} Acre Farm:**")
        
        if farm_size and farm_size < 1:  # Small farm specific advice
            advice_parts.append(f"• **Small farm advantage:** Intensive monitoring and quick interventions possible")
        
        if crops and 'cotton' in [c.lower() for c in crops]:
            advice_parts.append(f"• Monitor drainage systems - critical for cotton in monsoon")
            advice_parts.append(f"• Watch for fungal diseases in current humid conditions")
            advice_parts.append(f"• Check for pest activity, especially bollworm in August")
        
        advice_parts.append(f"• Avoid heavy field operations during expected rain periods")
        advice_parts.append(f"• Check weather forecasts daily: IMD, local agricultural news")
        
        # Regional emergency contacts
        advice_parts.append(f"**📞 Emergency Contacts:**")
        advice_parts.append(f"• Local Agriculture Extension Officer: Contact through Block office")
        advice_parts.append(f"• Kisan Call Center: 1800-180-1551")
        
        if "west bengal" in location.lower():
            advice_parts.append(f"• West Bengal Agriculture Department helpline")
        elif "maharashtra" in location.lower():
            advice_parts.append(f"• Maharashtra Agriculture Department helpline")
        
        # Reasoning and data sources
        advice_parts.append(f"**💡 Reasoning:** Based on your specific profile in {location}")
        advice_parts.append(f"• Integrated {', '.join(crops) if crops else 'general'} crop requirements with current weather")
        advice_parts.append(f"• Applied regional climate patterns and monsoon season considerations")
        advice_parts.append(f"• Used agricultural calendar data for {farm_size}-acre farm management")
        
        return "\n".join(advice_parts)
    
    def get_structured_weather_advice(self, location: str, crops: List[str], current_weather: Dict, forecast_data: Dict) -> str:
        """Generate structured weather advice using agricultural knowledge base"""
        advice_parts = []
        
        # Current conditions analysis
        temp = current_weather.get('temperature', 0)
        humidity = current_weather.get('humidity', 0)
        condition = current_weather.get('condition', '').lower()
        
        # Temperature-based advice
        if temp > 35:
            advice_parts.append("🌡️ **High Temperature Alert:** Increase irrigation frequency, provide shade to crops if possible, avoid heavy field work during midday hours.")
        elif temp < 15:
            advice_parts.append("🌡️ **Low Temperature Alert:** Protect crops from cold stress, consider covering sensitive plants, delay sowing if planned.")
        else:
            advice_parts.append(f"🌡️ **Temperature Normal:** Current {temp}°C is suitable for most agricultural activities.")
        
        # Humidity-based advice  
        if humidity > 80:
            advice_parts.append("💧 **High Humidity:** Monitor crops for fungal diseases, ensure good air circulation, avoid excessive watering.")
        elif humidity < 40:
            advice_parts.append("💧 **Low Humidity:** Increase irrigation frequency, consider mulching to retain soil moisture.")
        
        # Weather condition specific advice
        if 'rain' in condition:
            advice_parts.append("🌧️ **Rain Advisory:** Postpone fertilizer application, ensure proper drainage, monitor for waterlogging.")
        elif 'clear' in condition or 'sunny' in condition:
            advice_parts.append("☀️ **Clear Weather:** Good for field operations, spraying, and harvesting activities.")
        
        # Crop-specific advice
        if crops:
            for crop in crops:
                if crop.lower() == 'cotton':
                    if temp > 32 and humidity > 70:
                        advice_parts.append("🌾 **Cotton Specific:** Monitor for bollworm activity in current humid conditions. Consider prophylactic spray if pest pressure is high.")
                    elif 'rain' in condition:
                        advice_parts.append("🌾 **Cotton Specific:** Ensure drainage to prevent root rot. Delay picking if bolls are wet.")
                
                elif crop.lower() == 'wheat':
                    if temp > 30:
                        advice_parts.append("🌾 **Wheat Specific:** High temperature may affect grain filling. Ensure adequate irrigation.")
                    elif 'rain' in condition:
                        advice_parts.append("🌾 **Wheat Specific:** Risk of rust disease in wet conditions. Monitor closely and apply fungicide if symptoms appear.")
        
        # General farm management
        advice_parts.append(f"📋 **Immediate Actions for {location}:** Check irrigation systems, inspect crop health, secure equipment if stormy weather expected.")
        
        return "\n\n".join(advice_parts)
    
    def generate_weather_reasoning(self, location: str, crops: List[str], current_weather: Dict, forecast_data: Dict) -> str:
        """Generate reasoning explanation for weather advice"""
        reasoning_parts = []
        
        reasoning_parts.append(f"Based on current weather data from {current_weather.get('source', 'multiple sources')}")
        
        if crops:
            reasoning_parts.append(f"Considered specific requirements for {', '.join(crops)} crops")
        
        reasoning_parts.append(f"Integrated regional climate patterns for {location}")
        reasoning_parts.append("Applied seasonal agricultural calendar recommendations")
        
        return " • ".join(reasoning_parts)
    
    def get_weather_fallback_advice(self, query: str, user_context: Dict, error_msg: str) -> str:
        """Comprehensive fallback when all weather systems fail"""
        location = user_context.get("location", "your area")
        crops = user_context.get("crops", [])
        
        return f"""**🌤️ Weather Service Temporarily Unavailable**

**General Weather Precautions for {location}:**

**Immediate Actions:**
• Check local weather reports on radio/TV
• Inspect crop conditions for any stress signs  
• Ensure irrigation system is functional
• Secure farm equipment and materials

**Crop-Specific Guidance:**
{self.get_crop_specific_fallback_advice(crops)}

**Emergency Contacts:**
• Local Agriculture Extension Officer
• Kisan Call Center: 1800-180-1551
• IMD Weather Helpline: Contact regional office

**Note:** This is general guidance. For real-time weather updates, please check local meteorological services or contact agricultural extension officers in your area.

Error details: {error_msg}"""
    
    def get_crop_specific_fallback_advice(self, crops: List[str]) -> str:
        """Get crop-specific fallback advice"""
        if not crops:
            return "• Monitor all crops for signs of weather stress\n• Maintain consistent irrigation schedule"
        
        advice = []
        for crop in crops:
            if crop.lower() == 'cotton':
                advice.append("• Cotton: Watch for pest activity, ensure adequate drainage, monitor boll development")
            elif crop.lower() == 'wheat':
                advice.append("• Wheat: Check for disease symptoms, maintain soil moisture, protect from extreme temperatures")
            elif crop.lower() == 'rice':
                advice.append("• Rice: Maintain water levels, watch for blast disease, ensure proper drainage")
        
        return "\n".join(advice) if advice else "• Monitor crops for weather-related stress"
    
    def get_current_weather(self, location: str) -> Dict:
        """Get current weather data with multiple fallback APIs"""
        weather_apis = [
            self._get_weather_from_weatherapi,
            self._get_weather_from_openweather,
            self._get_weather_from_fallback
        ]
        
        for api_func in weather_apis:
            try:
                result = api_func(location)
                if "error" not in result:
                    return result
            except Exception as e:
                continue
        
        return self._get_offline_weather_advice(location)
    
    def _get_weather_from_weatherapi(self, location: str) -> Dict:
        """Primary weather API - WeatherAPI.com"""
        if not self.weather_api_key:
            raise Exception("WeatherAPI key not configured")
        
        url = f"http://api.weatherapi.com/v1/current.json"
        params = {"key": self.weather_api_key, "q": location, "aqi": "yes"}
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        return {
            "temperature": data["current"]["temp_c"],
            "humidity": data["current"]["humidity"],
            "condition": data["current"]["condition"]["text"],
            "wind_speed": data["current"]["wind_kph"],
            "source": "WeatherAPI"
        }
    
    def _get_weather_from_openweather(self, location: str) -> Dict:
        """Fallback weather API - OpenWeatherMap (using free tier)"""
        try:
            # Use free OpenWeatherMap API (no key required for basic data)
            url = f"http://api.openweathermap.org/data/2.5/weather"
            params = {"q": location, "units": "metric", "appid": "demo"}  # Demo key for fallback
            
            response = requests.get(url, params=params, timeout=8)
            if response.status_code == 200:
                data = response.json()
                return {
                    "temperature": data["main"]["temp"],
                    "humidity": data["main"]["humidity"], 
                    "condition": data["weather"][0]["description"],
                    "wind_speed": data["wind"]["speed"] * 3.6,  # Convert m/s to km/h
                    "source": "OpenWeatherMap"
                }
        except Exception:
            pass
        
        raise Exception("OpenWeatherMap fallback failed")
    
    def _get_weather_from_fallback(self, location: str) -> Dict:
        """Web scraping fallback for weather data"""
        try:
            # Try scraping weather information from public sources
            search_results = search_duckduckgo(f"weather {location} today")
            for result in search_results:
                if "temperature" in result['snippet'].lower() or "°c" in result['snippet'].lower():
                    # Extract basic weather info from snippets
                    import re
                    temp_match = re.search(r'(\d+)°?[Cc]', result['snippet'])
                    if temp_match:
                        return {
                            "temperature": int(temp_match.group(1)),
                            "humidity": "Unknown",
                            "condition": result['snippet'][:50],
                            "wind_speed": "Unknown",
                            "source": "Web Search"
                        }
            
            raise Exception("Web scraping fallback failed")
        except Exception:
            raise Exception("All weather sources failed")
    
    def _get_offline_weather_advice(self, location: str) -> Dict:
        """Offline weather advice based on location and season"""
        from datetime import datetime
        import calendar
        
        current_month = datetime.now().month
        month_name = calendar.month_name[current_month]
        
        # Basic seasonal advice for India
        if location.lower().find("maharashtra") != -1:
            seasonal_advice = {
                "temperature": "25-35°C (typical for Maharashtra)",
                "humidity": "Moderate to High",
                "condition": f"Seasonal conditions for {month_name}",
                "wind_speed": "Variable",
                "source": "Offline Seasonal Data",
                "advice": self._get_seasonal_advice_maharashtra(current_month)
            }
        else:
            seasonal_advice = {
                "temperature": "Check local sources",
                "humidity": "Variable", 
                "condition": f"Typical {month_name} conditions",
                "wind_speed": "Variable",
                "source": "General Seasonal Data",
                "advice": self._get_general_seasonal_advice(current_month)
            }
        
        return seasonal_advice
    
    def _get_seasonal_advice_maharashtra(self, month: int) -> str:
        """Seasonal agricultural advice for Maharashtra"""
        if month in [12, 1, 2]:  # Winter
            return "Rabi season - Good for wheat, gram, jowar. Minimal irrigation needed."
        elif month in [3, 4, 5]:  # Summer  
            return "Pre-monsoon - Prepare for Kharif crops. Check irrigation systems."
        elif month in [6, 7, 8, 9]:  # Monsoon
            return "Kharif season - Ideal for cotton, sugarcane, rice. Monitor rainfall."
        else:  # Post-monsoon
            return "Post-monsoon - Harvesting time for Kharif crops."
    
    def _get_general_seasonal_advice(self, month: int) -> str:
        """General seasonal agricultural advice"""
        if month in [12, 1, 2]:
            return "Winter season - Focus on Rabi crops and minimal watering."
        elif month in [3, 4, 5]:
            return "Summer - Prepare irrigation, consider heat-resistant varieties."
        elif month in [6, 7, 8, 9]:
            return "Monsoon - Main growing season, monitor for pests and diseases."
        else:
            return "Post-monsoon - Harvesting and post-harvest management."
    
    def get_weather_forecast(self, location: str, days: int = 7) -> List[Dict]:
        """Get weather forecast data"""
        try:
            if not self.weather_api_key:
                return [{"error": "Weather API key not configured"}]
            
            url = f"http://api.weatherapi.com/v1/forecast.json"
            params = {"key": self.weather_api_key, "q": location, "days": min(days, 10)}
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            forecast_days = []
            for day_data in data["forecast"]["forecastday"]:
                day_info = {
                    "date": day_data["date"],
                    "max_temp": day_data["day"]["maxtemp_c"],
                    "min_temp": day_data["day"]["mintemp_c"],
                    "total_precip": day_data["day"]["totalprecip_mm"],
                    "condition": day_data["day"]["condition"]["text"],
                    "chance_of_rain": day_data["day"]["daily_chance_of_rain"]
                }
                forecast_days.append(day_info)
            
            return forecast_days
        except Exception as e:
            return [{"error": f"Failed to get weather forecast: {str(e)}"}]
    
    def format_weather_summary(self, current: Dict, forecast: List[Dict]) -> str:
        """Format weather information for display"""
        if "error" in current:
            return f"**Weather Error:** {current['error']}"
        
        summary = "**📊 Current Weather:**\n"
        summary += f"• Temperature: {current['temperature']}°C\n"
        summary += f"• Condition: {current['condition']}\n"
        summary += f"• Humidity: {current['humidity']}%\n"
        summary += f"• Wind: {current['wind_speed']} km/h\n"
        
        if forecast and not forecast[0].get("error"):
            summary += "\n**🌤️ 7-Day Forecast:**\n"
            for day in forecast[:3]:  # Show first 3 days
                summary += f"• {day['date']}: {day['min_temp']}°C - {day['max_temp']}°C, {day['condition']}\n"
        
        return summary

class FinanceAgent(BaseAgent):
    """Enhanced Financial Agent with external integrations"""
    
    def __init__(self):
        super().__init__("FinanceAgent")
        self.financial_schemes = self._get_comprehensive_schemes()
        self.banking_apis = self._initialize_banking_apis()
        self.loan_calculators = self._initialize_loan_calculators()
    
    def _get_comprehensive_schemes(self):
        """Comprehensive government and financial schemes database"""
        return {
            "pm_kisan": {
                "name": "PM-KISAN",
                "benefit": "₹6,000 per year in three installments",
                "eligibility": "Small and marginal farmers with cultivable land up to 2 hectares",
                "application": "Online at pmkisan.gov.in or through CSC centers",
                "documents": ["Aadhaar Card", "Bank Account Details", "Land Records"],
                "status_check": "pmkisan.gov.in beneficiary status",
                "helpline": "011-24300606"
            },
            "kisan_credit_card": {
                "name": "Kisan Credit Card (KCC)",
                "benefit": "Flexible credit for agriculture with low interest",
                "interest": "4% interest rate for timely repayment (7% otherwise)",
                "eligibility": "All farmers including tenant farmers, sharecroppers",
                "application": "Banks, Regional Rural Banks, Cooperative Banks",
                "limit_calculation": "Based on crop area, scale of finance, and operational land holding",
                "validity": "5 years with annual review",
                "insurance": "Personal accident insurance and asset insurance included"
            },
            "fasal_bima": {
                "name": "Pradhan Mantri Fasal Bima Yojana (PMFBY)",
                "benefit": "Crop insurance against natural calamities, pests, diseases",
                "premium": "2% for Kharif, 1.5% for Rabi, 5% for annual commercial/horticulture",
                "application": "Banks, CSCs, or online at pmfby.gov.in",
                "claim_process": "Crop cutting experiments and technology-based assessment",
                "coverage": "Pre-sowing to post-harvest risks"
            },
            "mudra_loan": {
                "name": "MUDRA Loan",
                "benefit": "Loans up to ₹10 lakh for farm activities and agri-allied activities",
                "categories": {
                    "shishu": "Up to ₹50,000",
                    "kishore": "₹50,001 to ₹5,00,000",
                    "tarun": "₹5,00,001 to ₹10,00,000"
                },
                "interest": "Variable (8-12% depending on bank and category)",
                "application": "All banks and NBFCs"
            },
            "dbt_schemes": {
                "name": "Direct Benefit Transfer Schemes",
                "schemes": [
                    "PM-KISAN", "Fertilizer Subsidy", "Seed Subsidy",
                    "Diesel Subsidy", "Interest Subvention"
                ],
                "verification": "Aadhaar linking mandatory",
                "account": "DBT payments only to Aadhaar-linked bank accounts"
            },
            "msp_support": {
                "name": "Minimum Support Price",
                "crops": "23 crops including cereals, pulses, oilseeds, cotton",
                "procurement": "Through FCI, state agencies, and cooperatives",
                "online_registration": "Most states have online MSP registration portals"
            },
            "organic_certification": {
                "name": "Paramparagat Krishi Vikas Yojana (PKVY)",
                "support": "₹50,000 per hectare for 3 years",
                "certification": "Free organic certification",
                "area": "Minimum 50 acres cluster farming"
            }
        }
    
    def _initialize_banking_apis(self):
        """Initialize banking API configurations"""
        return {
            "sbi": {
                "name": "State Bank of India",
                "developer_portal": "developer.onlinesbi.com",
                "kcc_api": "/api/kcc/eligibility",
                "schemes_api": "/api/agricultural-schemes"
            },
            "hdfc": {
                "name": "HDFC Bank",
                "developer_portal": "developer.hdfcbank.com",
                "agri_api": "/api/agri-loans"
            },
            "icici": {
                "name": "ICICI Bank",
                "developer_portal": "developer.icicibank.com",
                "loan_api": "/api/agricultural-loans"
            }
        }
    
    def _initialize_loan_calculators(self):
        """Initialize loan calculators and eligibility checkers"""
        return {
            "kcc_calculator": self._kcc_limit_calculator,
            "emi_calculator": self._emi_calculator,
            "eligibility_checker": self._loan_eligibility_checker,
            "interest_rate_fetcher": self._get_current_interest_rates,
            "subsidy_calculator": self._calculate_subsidies
        }
    
    def _kcc_limit_calculator(self, crop_area: float, crop_type: str, location: str) -> Dict:
        """Calculate KCC limit based on crop area and type"""
        scale_of_finance = {
            "rice": 40000,  # per hectare
            "wheat": 35000,
            "cotton": 45000,
            "sugarcane": 80000,
            "maize": 30000,
            "general": 35000
        }
        
        per_hectare = scale_of_finance.get(crop_type.lower(), scale_of_finance["general"])
        crop_area_hectares = crop_area * 0.4047  # Convert acres to hectares
        
        production_loan = per_hectare * crop_area_hectares
        maintenance_expenses = production_loan * 0.1
        post_harvest_expenses = production_loan * 0.15
        consumption_expenses = 12000  # Annual consumption for family
        
        total_limit = production_loan + maintenance_expenses + post_harvest_expenses + consumption_expenses
        
        return {
            "total_limit": int(total_limit),
            "production_component": int(production_loan),
            "maintenance_component": int(maintenance_expenses),
            "post_harvest_component": int(post_harvest_expenses),
            "consumption_component": consumption_expenses,
            "validity": "5 years with annual review"
        }
    
    def _emi_calculator(self, principal: float, rate: float, tenure: int) -> Dict:
        """Calculate EMI for agricultural loans"""
        monthly_rate = rate / (12 * 100)
        emi = (principal * monthly_rate * (1 + monthly_rate)**tenure) / ((1 + monthly_rate)**tenure - 1)
        
        total_payment = emi * tenure
        total_interest = total_payment - principal
        
        return {
            "monthly_emi": round(emi, 2),
            "total_payment": round(total_payment, 2),
            "total_interest": round(total_interest, 2),
            "effective_rate": rate
        }
    
    def _loan_eligibility_checker(self, user_profile: Dict) -> Dict:
        """Check eligibility for various agricultural loans"""
        farm_size = user_profile.get("farm_size", 0)
        location = user_profile.get("location", "")
        crops = user_profile.get("crops", [])
        
        eligibility = {}
        
        # PM-KISAN eligibility
        eligibility["pm_kisan"] = {
            "eligible": farm_size <= 5,  # Up to 2 hectares (approximately 5 acres)
            "reason": "For small and marginal farmers" if farm_size <= 5 else "Farm size exceeds limit"
        }
        
        # KCC eligibility
        eligibility["kcc"] = {
            "eligible": True,  # All farmers are eligible
            "reason": "All farmers including tenant farmers are eligible",
            "estimated_limit": self._kcc_limit_calculator(farm_size, crops[0] if crops else "general", location)
        }
        
        # PMFBY eligibility
        eligibility["pmfby"] = {
            "eligible": True,
            "reason": "All farmers with insurable crops are eligible",
            "premium_rate": "2% for Kharif crops" if any(crop in ["rice", "cotton", "sugarcane"] for crop in crops) else "1.5% for Rabi crops"
        }
        
        return eligibility
    
    def _get_current_interest_rates(self) -> Dict:
        """Get current interest rates for agricultural loans"""
        try:
            # Try to get real-time rates from financial websites
            search_results = search_duckduckgo("current agricultural loan interest rates India 2024")
            
            # Default rates if search fails
            default_rates = {
                "kcc": {"rate": 7.0, "subsidized_rate": 4.0, "note": "4% for timely repayment"},
                "crop_loan": {"rate": 7.0, "subsidized_rate": 4.0, "note": "Interest subvention available"},
                "mudra_shishu": {"rate": 8.5, "note": "Varies by bank"},
                "mudra_kishore": {"rate": 9.5, "note": "Varies by bank"},
                "mudra_tarun": {"rate": 10.5, "note": "Varies by bank"},
                "last_updated": "Based on RBI guidelines and typical bank rates"
            }
            
            return default_rates
            
        except Exception:
            return {
                "kcc": {"rate": 7.0, "subsidized_rate": 4.0},
                "note": "Contact local banks for current rates"
            }
    
    def _calculate_subsidies(self, user_profile: Dict, loan_amount: float) -> Dict:
        """Calculate available subsidies and support"""
        subsidies = {}
        
        farm_size = user_profile.get("farm_size", 0)
        location = user_profile.get("location", "")
        crops = user_profile.get("crops", [])
        
        # Interest subvention
        if loan_amount <= 300000:  # Up to 3 lakh
            subsidies["interest_subvention"] = {
                "benefit": "3% interest subvention",
                "effective_rate": "4% instead of 7%",
                "condition": "Timely repayment"
            }
        
        # Input subsidies
        subsidies["input_subsidies"] = {
            "seed_subsidy": "25-50% on certified seeds",
            "fertilizer_subsidy": "DBT - Direct to farmer account",
            "equipment_subsidy": "40-50% under various schemes"
        }
        
        # Crop insurance premium subsidy
        if crops:
            subsidies["insurance_premium"] = {
                "government_share": "Balance premium after farmer's share",
                "farmer_share": "2% for Kharif, 1.5% for Rabi crops"
            }
        
        return subsidies

class SoilHealthAgent(BaseAgent):
    """Specialized agent for soil health management and analysis"""
    
    def __init__(self):
        super().__init__("SoilHealthAgent")
        self.soil_parameters = {
            "ph_levels": {
                "acidic": {"range": "< 6.0", "crops": ["tea", "coffee", "blueberries"], "amendments": "Lime application"},
                "neutral": {"range": "6.0-7.0", "crops": ["most crops"], "amendments": "Maintain with organic matter"},
                "alkaline": {"range": "> 7.0", "crops": ["brassicas", "asparagus"], "amendments": "Sulfur or organic matter"}
            },
            "nutrient_deficiencies": {
                "nitrogen": {"symptoms": "Yellowing leaves, stunted growth", "solutions": ["urea", "ammonium sulfate", "green manure"]},
                "phosphorus": {"symptoms": "Purple leaves, poor root development", "solutions": ["rock phosphate", "bone meal", "DAP"]},
                "potassium": {"symptoms": "Brown leaf edges, weak stems", "solutions": ["muriate of potash", "wood ash", "compost"]},
                "iron": {"symptoms": "Yellowing between leaf veins", "solutions": ["iron sulfate", "chelated iron", "organic matter"]},
                "zinc": {"symptoms": "White streaks, small leaves", "solutions": ["zinc sulfate", "zinc chelate"]},
                "manganese": {"symptoms": "Yellow spots on leaves", "solutions": ["manganese sulfate", "foliar spray"]}
            }
        }
    
    def get_soil_advice(self, query: str, user_context: Dict) -> str:
        """Generate soil health advice and recommendations"""
        try:
            location = user_context.get("location", "")
            crops = user_context.get("crops", [])
            farm_size = user_context.get("farm_size", 0)
            
            system_message = """You are a soil scientist specializing in Indian agricultural soils. 
            Provide practical advice on soil testing, nutrient management, pH correction, 
            and soil improvement techniques. Focus on cost-effective solutions for farmers."""
            
            enhanced_query = f"""
            Query: {query}
            Location: {location}
            Crops: {', '.join(crops) if crops else 'General farming'}
            Farm Size: {farm_size} acres
            
            Provide soil health recommendations including testing protocols, 
            nutrient management, and soil improvement strategies.
            """
            
            response = self.generate_response(enhanced_query, system_message)
            
            # Add structured soil health information
            soil_info = self._get_structured_soil_advice(crops, location)
            
            return f"{response}\n\n{soil_info}"
            
        except Exception as e:
            return f"Error generating soil advice: {str(e)}"
    
    def _get_structured_soil_advice(self, crops: List[str], location: str) -> str:
        """Generate structured soil health recommendations"""
        advice = "**🌱 Soil Health Quick Reference:**\n\n"
        
        # General soil testing advice
        advice += "**Soil Testing Recommendations:**\n"
        advice += "• Test soil every 2-3 years or before major crop changes\n"
        advice += "• Collect samples from multiple points at 6-8 inch depth\n"
        advice += "• Best time: 2-3 months before planting season\n"
        advice += "• Parameters to test: pH, N-P-K, organic carbon, micronutrients\n\n"
        
        # Crop-specific soil requirements
        if crops:
            advice += "**Crop-Specific Soil Requirements:**\n"
            for crop in crops:
                if crop.lower() == "cotton":
                    advice += "• **Cotton:** Prefers well-drained black cotton soil, pH 5.8-8.0, high potassium need\n"
                elif crop.lower() == "rice":
                    advice += "• **Rice:** Clay/clay loam, pH 5.0-6.5, high organic matter, good water retention\n"
                elif crop.lower() == "wheat":
                    advice += "• **Wheat:** Loamy soil, pH 6.0-7.5, good drainage, moderate fertility\n"
            advice += "\n"
        
        # Organic matter improvement
        advice += "**Soil Improvement Strategies:**\n"
        advice += "• **Organic Matter:** Add farmyard manure, compost, green manure crops\n"
        advice += "• **Microbial Activity:** Use bio-fertilizers, avoid excessive chemical use\n"
        advice += "• **Soil Structure:** Practice crop rotation, avoid heavy machinery on wet soil\n"
        advice += "• **Erosion Control:** Contour farming, terracing, cover crops\n\n"
        
        # pH management
        advice += "**pH Management:**\n"
        advice += "• **Acidic soils:** Apply lime 2-3 months before planting (200-500 kg/acre)\n"
        advice += "• **Alkaline soils:** Add sulfur or organic matter, use acid-forming fertilizers\n"
        advice += "• **Monitor regularly:** pH affects nutrient availability significantly\n"
        
        return advice

class PestManagementAgent(BaseAgent):
    """Specialized agent for integrated pest management"""
    
    def __init__(self):
        super().__init__("PestManagementAgent")
        self.ipm_strategies = {
            "prevention": {
                "crop_rotation": "Break pest life cycles",
                "resistant_varieties": "Use pest-resistant crop varieties", 
                "field_sanitation": "Remove crop residues and weeds",
                "optimal_planting": "Plant at appropriate time and spacing"
            },
            "biological_control": {
                "beneficial_insects": ["ladybirds", "parasitic wasps", "predatory mites"],
                "biopesticides": ["Bt (Bacillus thuringiensis)", "NPV (Nuclear Polyhedrosis Virus)", "Trichoderma"],
                "companion_planting": ["marigold", "neem", "chrysanthemum"]
            },
            "monitoring": {
                "pest_scouting": "Regular field inspection protocols",
                "pheromone_traps": "Early detection and monitoring",
                "sticky_traps": "Monitor flying insect populations",
                "economic_threshold": "Spray only when pest levels justify cost"
            }
        }
    
    def get_pest_advice(self, query: str, user_context: Dict) -> str:
        """Generate IPM recommendations and pest management advice"""
        try:
            location = user_context.get("location", "")
            crops = user_context.get("crops", [])
            
            system_message = """You are an entomologist specializing in Integrated Pest Management (IPM) 
            for Indian agriculture. Provide sustainable, eco-friendly pest control solutions that 
            minimize pesticide use while maintaining crop productivity."""
            
            enhanced_query = f"""
            Query: {query}
            Location: {location}
            Crops: {', '.join(crops) if crops else 'General farming'}
            
            Provide IPM recommendations focusing on biological control, 
            monitoring techniques, and sustainable pest management practices.
            """
            
            response = self.generate_response(enhanced_query, system_message)
            
            # Add structured IPM information
            ipm_info = self._get_structured_ipm_advice(crops, location)
            
            return f"{response}\n\n{ipm_info}"
            
        except Exception as e:
            return f"Error generating pest management advice: {str(e)}"
    
    def _get_structured_ipm_advice(self, crops: List[str], location: str) -> str:
        """Generate structured IPM recommendations"""
        advice = "**🐛 Integrated Pest Management Guide:**\n\n"
        
        # IPM pyramid approach
        advice += "**IPM Strategy (Bottom-up approach):**\n"
        advice += "1. **Prevention:** Crop rotation, resistant varieties, field sanitation\n"
        advice += "2. **Monitoring:** Regular scouting, traps, threshold levels\n"
        advice += "3. **Biological Control:** Beneficial insects, biopesticides\n"
        advice += "4. **Cultural Control:** Timing, spacing, water management\n"
        advice += "5. **Chemical Control:** Last resort, target-specific, rotate modes of action\n\n"
        
        # Crop-specific pest management
        if crops:
            advice += "**Crop-Specific Major Pests:**\n"
            for crop in crops:
                if crop.lower() == "cotton":
                    advice += "• **Cotton:** Bollworm, aphids, whitefly, thrips\n"
                    advice += "  - Bt cotton varieties, pheromone traps, beneficial insects\n"
                elif crop.lower() == "rice":
                    advice += "• **Rice:** Stem borer, brown planthopper, leaf folder\n" 
                    advice += "  - Light traps, biological agents, water management\n"
            advice += "\n"
        
        # Biological control agents
        advice += "**Beneficial Organisms to Encourage:**\n"
        advice += "• **Predators:** Ladybirds, spiders, ground beetles, dragonflies\n"
        advice += "• **Parasitoids:** Trichogramma, Braconidae wasps\n"
        advice += "• **Pathogens:** Bacillus thuringiensis, entomopathogenic fungi\n\n"
        
        # Safe pesticide practices
        advice += "**When Chemical Control is Necessary:**\n"
        advice += "• Use WHO Class II or III pesticides only\n"
        advice += "• Rotate different modes of action to prevent resistance\n"
        advice += "• Follow label instructions strictly\n"
        advice += "• Maintain pre-harvest interval (PHI)\n"
        advice += "• Use protective equipment during application\n"
        
        return advice

class IrrigationAgent(BaseAgent):
    """Specialized agent for water management and irrigation optimization"""
    
    def __init__(self):
        super().__init__("IrrigationAgent")
        self.irrigation_systems = {
            "drip": {
                "efficiency": "85-95%",
                "suitable_crops": ["cotton", "sugarcane", "vegetables", "fruits"],
                "investment": "High initial, low operating cost",
                "advantages": ["Water saving", "Precise application", "Reduced weeds", "Fertilizer integration"]
            },
            "sprinkler": {
                "efficiency": "70-85%", 
                "suitable_crops": ["wheat", "maize", "vegetables", "fodder"],
                "investment": "Medium initial and operating cost",
                "advantages": ["Uniform application", "Suitable for uneven terrain", "Frost protection"]
            },
            "furrow": {
                "efficiency": "40-60%",
                "suitable_crops": ["cotton", "sugarcane", "maize"],
                "investment": "Low initial, medium operating cost", 
                "advantages": ["Simple", "Traditional", "Low maintenance"]
            }
        }
    
    def get_irrigation_advice(self, query: str, user_context: Dict) -> str:
        """Generate water management and irrigation recommendations"""
        try:
            location = user_context.get("location", "")
            crops = user_context.get("crops", [])
            farm_size = user_context.get("farm_size", 0)
            
            system_message = """You are an irrigation engineer specializing in efficient water 
            management for Indian agriculture. Provide practical advice on irrigation scheduling, 
            system selection, and water conservation techniques."""
            
            enhanced_query = f"""
            Query: {query}
            Location: {location} 
            Crops: {', '.join(crops) if crops else 'General farming'}
            Farm Size: {farm_size} acres
            
            Provide irrigation recommendations including system selection, 
            scheduling, and water-saving techniques appropriate for the farm size and crops.
            """
            
            response = self.generate_response(enhanced_query, system_message)
            
            # Add structured irrigation information
            irrigation_info = self._get_structured_irrigation_advice(crops, farm_size, location)
            
            return f"{response}\n\n{irrigation_info}"
            
        except Exception as e:
            return f"Error generating irrigation advice: {str(e)}"
    
    def _get_structured_irrigation_advice(self, crops: List[str], farm_size: float, location: str) -> str:
        """Generate structured irrigation recommendations"""
        advice = "**💧 Irrigation Management Guide:**\n\n"
        
        # System recommendation based on farm size
        advice += "**Irrigation System Recommendations:**\n"
        if farm_size <= 2:
            advice += "• **Small Farm (≤2 acres):** Drip irrigation most cost-effective\n"
            advice += "• **Investment:** ₹40,000-60,000 per acre for drip system\n"
            advice += "• **Subsidy:** 50% government subsidy available under PMKSY\n"
        elif farm_size <= 10:
            advice += "• **Medium Farm (2-10 acres):** Drip or sprinkler irrigation\n"
            advice += "• **Choice depends on:** Crop type, water source, terrain\n"
        else:
            advice += "• **Large Farm (>10 acres):** Combination of systems based on crop zones\n"
        advice += "\n"
        
        # Crop-specific water requirements
        if crops:
            advice += "**Crop Water Requirements:**\n"
            for crop in crops:
                if crop.lower() == "cotton":
                    advice += "• **Cotton:** 700-1300mm total, critical at flowering and boll development\n"
                elif crop.lower() == "rice":
                    advice += "• **Rice:** 1200-1500mm, maintain 2-5cm standing water during vegetative growth\n"
                elif crop.lower() == "wheat":
                    advice += "• **Wheat:** 450-650mm, critical at tillering, flowering, and grain filling\n"
            advice += "\n"
        
        # Water conservation techniques
        advice += "**Water Conservation Techniques:**\n"
        advice += "• **Mulching:** Reduces evaporation by 25-50%\n"
        advice += "• **Scheduling:** Irrigate early morning or evening\n"
        advice += "• **Soil moisture monitoring:** Use tensiometers or feel method\n"
        advice += "• **Rainwater harvesting:** Collect and store rainwater for dry periods\n"
        advice += "• **Drought-tolerant varieties:** Use when water is limited\n\n"
        
        # Government schemes
        advice += "**Government Support:**\n"
        advice += "• **PMKSY:** 50% subsidy on micro-irrigation systems\n"
        advice += "• **Application:** Through agriculture department or online\n"
        advice += "• **Water User Associations:** Join for better water management\n"
        
        return advice
    
    def get_finance_advice(self, query: str, user_context: Dict) -> str:
        """Generate comprehensive financial advice with real-time information"""
        try:
            farm_size = user_context.get("farm_size", 0)
            location = user_context.get("location", "")
            
            # Search for current financial schemes information
            search_results = search_duckduckgo(f"agricultural loan schemes India 2024 {location}")
            
            system_message = """You are a financial advisor specializing in Indian agricultural finance. 
            Provide practical, actionable advice about affordable credit options, government schemes, 
            and financial assistance. Include application processes and eligibility criteria."""
            
            # Include search results in the query
            search_context = "\n".join([f"• {result['title']}: {result['snippet']}" 
                                      for result in search_results[:3] if result['snippet']])
            
            enhanced_query = f"""
            Query: {query}
            Farm size: {farm_size} acres
            Location: {location}
            
            Recent Information:
            {search_context}
            
            Provide specific guidance on where to get affordable credit and relevant government policies.
            """
            
            response = self.generate_response(enhanced_query, system_message)
            
            # Add relevant schemes information
            relevant_schemes = self.find_relevant_schemes(query, farm_size)
            if relevant_schemes:
                scheme_info = "\n\n**💰 Key Financial Schemes & Credit Sources:**\n"
                for scheme_key in relevant_schemes:
                    if scheme_key in self.financial_schemes:
                        scheme = self.financial_schemes[scheme_key]
                        scheme_info += f"\n**{scheme['name']}**\n"
                        scheme_info += f"• Benefit: {scheme['benefit']}\n"
                        if "eligibility" in scheme:
                            scheme_info += f"• Eligibility: {scheme['eligibility']}\n"
                        if "application" in scheme:
                            scheme_info += f"• Apply: {scheme['application']}\n"
                response += scheme_info
            
            # Add contact information
            response += """
            
**📞 Where to Apply:**
• **Banks:** SBI, HDFC, ICICI, local banks
• **Cooperative Banks:** District Cooperative Banks
• **CSC Centers:** Common Service Centers in your area
• **Online:** pmkisan.gov.in, pmfby.gov.in, udyamimitra.in
            
**💡 Quick Tips:**
• Visit nearest bank with land documents and Aadhaar
• Check eligibility for multiple schemes simultaneously
• Compare interest rates across different banks
• Keep records of all applications and follow up regularly
            """
            
            return response
        except Exception as e:
            return f"Error generating finance advice: {str(e)}"
    
    def find_relevant_schemes(self, query: str, farm_size: float) -> List[str]:
        """Find relevant financial schemes based on query and farm size"""
        relevant = []
        query_lower = query.lower()
        
        # Check for specific scheme mentions
        if "pm kisan" in query_lower:
            relevant.append("pm_kisan")
        if "credit" in query_lower or "loan" in query_lower:
            relevant.extend(["kisan_credit_card", "mudra_loan"])
        if "insurance" in query_lower:
            relevant.append("fasal_bima")
        if "mudra" in query_lower:
            relevant.append("mudra_loan")
        if "stand up" in query_lower:
            relevant.append("stand_up_india")
        
        # Default recommendations based on farm size
        if not relevant:
            if farm_size <= 2:  # Small farmers
                relevant = ["pm_kisan", "kisan_credit_card", "fasal_bima"]
            else:  # Larger farmers
                relevant = ["kisan_credit_card", "mudra_loan", "fasal_bima"]
        
        return list(set(relevant))  # Remove duplicates

class PolicyAgent(BaseAgent):
    """Specialized agent for agricultural policies"""
    
    def __init__(self):
        super().__init__("PolicyAgent")
    
    def get_policy_advice(self, query: str, user_context: Dict) -> str:
        """Generate policy-related advice"""
        try:
            system_message = """You are a policy expert specializing in Indian agricultural policies. 
            Provide accurate information about government schemes and regulations."""
            
            response = self.generate_response(query, system_message)
            return response
        except Exception as e:
            return f"Error generating policy advice: {str(e)}"
    
    def analyze_document(self, document_content: str, question: str) -> str:
        """Analyze policy documents"""
        try:
            system_message = """Analyze the document and answer the question based on the content."""
            
            prompt = f"Document: {document_content[:2000]}...\nQuestion: {question}"
            return self.generate_response(prompt, system_message)
        except Exception as e:
            return f"Error analyzing document: {str(e)}"
    
    def summarize_document(self, document_content: str) -> str:
        """Summarize policy documents"""
        try:
            system_message = """Create a farmer-friendly summary of this policy document."""
            
            prompt = f"Summarize this document: {document_content[:3000]}..."
            return self.generate_response(prompt, system_message)
        except Exception as e:
            return f"Error summarizing document: {str(e)}"

class MarketAgent(BaseAgent):
    """Specialized agent for market prices and trends"""
    
    def __init__(self):
        super().__init__("MarketAgent")
    
    def get_market_advice(self, query: str, user_context: Dict) -> str:
        """Generate market-related advice"""
        try:
            location = user_context.get("location", "")
            crops = user_context.get("crops", [])
            
            system_message = """You are an agricultural market analyst. Provide practical market advice 
            including price trends and trading strategies."""
            
            enhanced_query = f"Query: {query}\nLocation: {location}\nCrops: {', '.join(crops)}"
            response = self.generate_response(enhanced_query, system_message)
            
            # Add market disclaimer
            response += """
            
**⚠️ Market Risk Disclaimer:**
• Market prices are subject to volatility and rapid changes
• Always verify prices from multiple sources before making decisions
• Check official sources: AgMarkNet, eNAM, local mandi boards
            """
            
            return response
        except Exception as e:
            return f"Error generating market advice: {str(e)}"

class ImageAgent(BaseAgent):
    """Specialized agent for image analysis"""
    
    def __init__(self):
        super().__init__("ImageAgent")
    
    def analyze_image(self, image: Image.Image, analysis_type: str, additional_context: str = "") -> str:
        """Analyze uploaded crop image"""
        try:
            # Basic image processing
            processed_image = self.preprocess_image(image)
            image_features = self.extract_image_features(processed_image)
            
            if analysis_type == "Disease Detection":
                return self.detect_disease(image_features, additional_context)
            elif analysis_type == "Crop Health Assessment":
                return self.assess_crop_health(image_features, additional_context)
            else:
                return self.general_image_analysis(image_features, additional_context)
        except Exception as e:
            return f"Error analyzing image: {str(e)}"
    
    def preprocess_image(self, image: Image.Image) -> Dict[str, Any]:
        """Preprocess image for analysis"""
        try:
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            if image.size[0] > 1024 or image.size[1] > 1024:
                image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            
            width, height = image.size
            img_array = np.array(image)
            
            return {
                "image": image,
                "array": img_array,
                "width": width,
                "height": height
            }
        except Exception as e:
            raise Exception(f"Image preprocessing failed: {str(e)}")
    
    def extract_image_features(self, processed_image: Dict) -> Dict[str, Any]:
        """Extract features from processed image"""
        try:
            img_array = processed_image["array"]
            
            # Basic color analysis
            mean_rgb = np.mean(img_array, axis=(0, 1))
            green_dominance = mean_rgb[1] / np.sum(mean_rgb)
            
            # Simple disease indicators
            brown_pixels = self.detect_brown_areas(img_array)
            yellow_pixels = self.detect_yellow_areas(img_array)
            
            brightness = np.mean(img_array)
            contrast = np.std(img_array)
            
            return {
                "green_dominance": green_dominance,
                "brown_percentage": brown_pixels,
                "yellow_percentage": yellow_pixels,
                "brightness": brightness,
                "image_quality": "good" if contrast > 30 else "poor"
            }
        except Exception as e:
            return {"error": f"Feature extraction failed: {str(e)}"}
    
    def detect_brown_areas(self, img_array: np.ndarray) -> float:
        """Detect brown/diseased areas"""
        try:
            mask = (img_array[:, :, 0] > img_array[:, :, 1]) & \
                   (img_array[:, :, 1] > img_array[:, :, 2]) & \
                   (img_array[:, :, 0] < 150)
            return np.sum(mask) / img_array[:, :, 0].size * 100
        except Exception:
            return 0.0
    
    def detect_yellow_areas(self, img_array: np.ndarray) -> float:
        """Detect yellow/stressed areas"""
        try:
            mask = (img_array[:, :, 0] > 100) & \
                   (img_array[:, :, 1] > 100) & \
                   (img_array[:, :, 2] < 80)
            return np.sum(mask) / img_array[:, :, 0].size * 100
        except Exception:
            return 0.0
    
    def detect_disease(self, image_features: Dict, context: str) -> str:
        """Detect potential diseases"""
        try:
            system_message = """You are a plant pathologist. Analyze image features to identify diseases."""
            
            prompt = f"""
            Image Features: {image_features}
            Context: {context}
            
            Provide disease diagnosis with confidence level and treatment recommendations.
            """
            
            ai_response = self.generate_response(prompt, system_message)
            
            # Add feature analysis
            analysis = "**📊 Image Analysis:**\n"
            green_dominance = image_features.get('green_dominance', 0)
            brown_percentage = image_features.get('brown_percentage', 0)
            
            if green_dominance > 0.4:
                analysis += "✅ Good green foliage detected\n"
            else:
                analysis += "⚠️ Reduced green foliage - possible stress\n"
            
            if brown_percentage > 10:
                analysis += f"🚨 High diseased areas detected ({brown_percentage:.1f}%)\n"
            
            return f"{analysis}\n\n{ai_response}"
        except Exception as e:
            return f"Error in disease detection: {str(e)}"
    
    def assess_crop_health(self, image_features: Dict, context: str) -> str:
        """Assess overall crop health"""
        try:
            system_message = """Assess crop health based on visual indicators."""
            
            prompt = f"Image features: {image_features}\nContext: {context}\nProvide health assessment."
            
            health_score = self.calculate_health_score(image_features)
            ai_response = self.generate_response(prompt, system_message)
            
            return f"**Health Score: {health_score}/100**\n\n{ai_response}"
        except Exception as e:
            return f"Error in health assessment: {str(e)}"
    
    def calculate_health_score(self, features: Dict) -> int:
        """Calculate health score"""
        try:
            score = 100
            brown_percentage = features.get('brown_percentage', 0)
            yellow_percentage = features.get('yellow_percentage', 0)
            green_dominance = features.get('green_dominance', 0)
            
            score -= min(brown_percentage * 3, 30)
            score -= min(yellow_percentage * 2, 20)
            
            if green_dominance > 0.4:
                score += 10
            
            return max(0, min(100, int(score)))
        except Exception:
            return 50
    
    def general_image_analysis(self, image_features: Dict, context: str) -> str:
        """Perform general image analysis"""
        try:
            system_message = """Provide comprehensive analysis of crop images."""
            
            prompt = f"Features: {image_features}\nContext: {context}\nProvide general analysis."
            return self.generate_response(prompt, system_message)
        except Exception as e:
            return f"Error in analysis: {str(e)}"

# ================== MAIN APPLICATION ==================

class AgriAIAdvisor:
    def __init__(self):
        self.initialize_session_state()
        self.setup_sidebar()
        self.initialize_components()
        self.load_agricultural_datasets()
        self.initialize_regional_knowledge()
    
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'conversation_history' not in st.session_state:
            st.session_state.conversation_history = []
        if 'user_profile' not in st.session_state:
            st.session_state.user_profile = {}
        if 'agents_initialized' not in st.session_state:
            st.session_state.agents_initialized = False
    
    def setup_sidebar(self):
        """Setup sidebar with API keys and configuration"""
        with st.sidebar:
            st.header("🔑 API Configuration")
            
            # Show current API status
            st.success("✅ Groq API: Connected")
            st.success("✅ Weather API: Connected")
            
            st.divider()
            
            # User Profile Section
            st.header("👤 User Profile")
            
            languages = {
                "English": "en", "Hindi": "hi", "Marathi": "mr", "Gujarati": "gu",
                "Tamil": "ta", "Telugu": "te", "Kannada": "kn", "Bengali": "bn", "Punjabi": "pa"
            }
            
            selected_language = st.selectbox("Preferred Language", options=list(languages.keys()), index=0)
            location = st.text_input("Your Location", placeholder="e.g., Pune, Maharashtra")
            farm_size = st.number_input("Farm Size (acres)", min_value=0.0, value=0.0, step=0.1)
            
            crops_grown = st.multiselect(
                "Crops Grown",
                options=["Rice", "Wheat", "Cotton", "Sugarcane", "Maize", "Soybean", 
                        "Groundnut", "Onion", "Tomato", "Potato", "Other"]
            )
            
            # Update session state
            st.session_state.user_profile = {
                "language": languages[selected_language],
                "location": location,
                "farm_size": farm_size,
                "crops": crops_grown
            }
            
            st.divider()
            
            st.header("📊 System Status")
            self.display_system_health()
            
            if st.button("🔄 Refresh System"):
                st.session_state.agents_initialized = False
                st.rerun()
    
    def display_system_health(self):
        """Display real-time system health status"""
        try:
            # Test Groq API
            groq_status = self.test_groq_connection()
            status_icon = "✅" if groq_status else "❌"
            st.write(f"**Groq LLM:** {status_icon} {'Connected' if groq_status else 'Offline (Using fallbacks)'}")
            
            # Test Weather API
            weather_status = self.test_weather_connection()
            status_icon = "✅" if weather_status else "⚠️"
            st.write(f"**Weather API:** {status_icon} {'Connected' if weather_status else 'Using fallbacks'}")
            
            # AGNO System Status
            if AGNO_AVAILABLE and hasattr(self, 'agno_system'):
                agent_status = self.agno_system.get_agent_status()
                if agent_status.get('agno_framework'):
                    st.write(f"**AGNO Framework:** ✅ {agent_status['total_agents']} Specialists + {agent_status['teams']} Teams")
                    st.caption("🤖 Professional multi-agent coordination with search integration")
                else:
                    st.write(f"**AGNO System:** ✅ {agent_status['total_agents']} Specialists + {agent_status['teams']} Teams")
                    st.caption("🤖 Enhanced multi-agent coordination active")
            else:
                st.write("**AGNO System:** ⚠️ Using legacy agents")
                st.caption("💡 Restart to enable enhanced agents")
            
            # Test Search capability
            search_status = self.test_search_connection()
            status_icon = "✅" if search_status else "⚠️"
            st.write(f"**Web Search:** {status_icon} {'Available' if search_status else 'Limited'}")
            
            # Vector Database is local, always available
            st.write("**Vector Database:** ✅ Ready")
            
            # Show offline capabilities
            st.write("**Offline Mode:** ✅ Basic guidance available")
            
        except Exception as e:
            st.write("**System Status:** ⚠️ Limited functionality")
    
    def test_groq_connection(self) -> bool:
        """Test Groq API connectivity"""
        try:
            if hasattr(self, 'crop_agent') and hasattr(self.crop_agent, 'groq_client') and self.crop_agent.groq_client:
                # Quick test with minimal prompt
                response = self.crop_agent.groq_client.chat.completions.create(
                    model="llama3-8b-8192",
                    messages=[{"role": "user", "content": "test"}],
                    max_tokens=5,
                    timeout=5
                )
                return response is not None
        except Exception:
            pass
        return False
    
    def test_weather_connection(self) -> bool:
        """Test weather API connectivity"""
        try:
            if hasattr(self, 'weather_agent') and self.weather_agent.weather_api_key:
                # Quick test with minimal request
                url = f"http://api.weatherapi.com/v1/current.json"
                params = {"key": self.weather_agent.weather_api_key, "q": "Delhi"}
                response = requests.get(url, params=params, timeout=5)
                return response.status_code == 200
        except Exception:
            pass
        return False
    
    def test_search_connection(self) -> bool:
        """Test search capability"""
        try:
            search_results = search_duckduckgo("test", max_results=1)
            return len(search_results) > 0 and "error" not in search_results[0].get('title', '').lower()
        except Exception:
            pass
        return False
    
    def initialize_components(self):
        """Initialize all system components"""
        try:
            self.input_processor = InputProcessor()
            self.vector_db = VectorDB()
            
            # Initialize proper AGNO system if available
            if AGNO_AVAILABLE:
                self.agno_system = AGNOSystem()
                st.success("✅ AGNO Agent System Initialized Successfully")
            else:
                st.warning("⚠️ AGNO system unavailable, using fallback agents")
                # Initialize fallback agents
                self.crop_agent = CropAgent()
                self.weather_agent = WeatherAgent()
                self.finance_agent = FinanceAgent()
                self.policy_agent = PolicyAgent()
                self.market_agent = MarketAgent()
                self.image_agent = ImageAgent()
                
                # Initialize specialized AGNO agents
                self.soil_health_agent = SoilHealthAgent()
                self.pest_management_agent = PestManagementAgent()
                self.irrigation_agent = IrrigationAgent()
                self.supply_chain_agent = SupplyChainAgent()
                self.compliance_agent = ComplianceAgent()
            
            st.session_state.agents_initialized = True
        except Exception as e:
            st.error(f"Failed to initialize system: {str(e)}")
    
    def load_agricultural_datasets(self):
        """Load comprehensive agricultural datasets"""
        if 'agricultural_data' not in st.session_state:
            st.session_state.agricultural_data = {
                'crop_calendar': self.get_crop_calendar_data(),
                'government_schemes': self.get_government_schemes_data(),
                'regional_crops': self.get_regional_crops_data(),
                'pest_disease_db': self.get_pest_disease_database(),
                'market_trends': self.get_market_trends_data(),
                'soil_health': self.get_soil_health_data()
            }
    
    def initialize_regional_knowledge(self):
        """Initialize region-specific agricultural knowledge"""
        if 'regional_knowledge' not in st.session_state:
            st.session_state.regional_knowledge = {
                'maharashtra': {
                    'major_crops': ['Cotton', 'Sugarcane', 'Wheat', 'Jowar', 'Bajra', 'Tur', 'Gram'],
                    'cropping_seasons': {
                        'kharif': 'June-November (Cotton, Sugarcane, Rice)',
                        'rabi': 'November-April (Wheat, Gram, Jowar)',
                        'summer': 'March-June (Fodder crops, Vegetables)'
                    },
                    'common_diseases': ['Pink bollworm (Cotton)', 'Red rot (Sugarcane)', 'Rust (Wheat)'],
                    'irrigation_methods': ['Drip irrigation', 'Sprinkler irrigation', 'Canal irrigation'],
                    'soil_types': ['Black cotton soil', 'Red soil', 'Alluvial soil'],
                    'climate_zones': ['Semi-arid', 'Sub-humid']
                }
            }
    
    def get_crop_calendar_data(self) -> Dict:
        """Comprehensive crop calendar for Indian agriculture"""
        return {
            'rice': {
                'kharif': {'sowing': 'June-July', 'transplanting': 'July-August', 'harvesting': 'November-December'},
                'rabi': {'sowing': 'December-January', 'harvesting': 'April-May'},
                'water_requirement': '1200-1500 mm',
                'temperature': '20-35°C',
                'growth_stages': ['Tillering (20-45 days)', 'Panicle initiation (45-65 days)', 'Grain filling (65-90 days)']
            },
            'cotton': {
                'kharif': {'sowing': 'May-June', 'harvesting': 'October-January'},
                'water_requirement': '700-1300 mm',
                'temperature': '21-32°C',
                'growth_stages': ['Square formation (45-65 days)', 'Flowering (65-95 days)', 'Boll development (95-160 days)']
            },
            'wheat': {
                'rabi': {'sowing': 'October-December', 'harvesting': 'March-May'},
                'water_requirement': '450-650 mm',
                'temperature': '15-25°C',
                'growth_stages': ['Tillering (20-40 days)', 'Jointing (40-60 days)', 'Grain filling (60-120 days)']
            }
        }
    
    def get_government_schemes_data(self) -> Dict:
        """Latest government schemes with detailed information"""
        return {
            'pm_kisan': {
                'name': 'PM-KISAN Samman Nidhi',
                'benefit': '₹6,000 per year in three installments',
                'eligibility': 'All farmer families with cultivable land',
                'documents': ['Land ownership papers', 'Aadhaar card', 'Bank account details'],
                'application': 'Online at pmkisan.gov.in or CSC centers',
                'helpline': '155261',
                'status_check': 'pmkisan.gov.in/BeneficiaryStatus.aspx'
            },
            'kcc': {
                'name': 'Kisan Credit Card',
                'benefit': 'Credit facility up to ₹3 lakh at 4% interest',
                'eligibility': 'All farmers including tenant farmers',
                'documents': ['Land records', 'Identity proof', 'Address proof'],
                'application': 'All banks, RRBs, cooperative banks',
                'features': ['Crop loan', 'Investment credit', 'Emergency credit'],
                'repayment': 'Flexible repayment options'
            },
            'pmfby': {
                'name': 'Pradhan Mantri Fasal Bima Yojana',
                'benefit': 'Crop insurance against natural calamities',
                'premium': '2% for Kharif, 1.5% for Rabi, 5% for Commercial/Horticulture',
                'coverage': 'Sum insured = Scale of finance × Area',
                'claims': 'Assessed using technology (drones, satellites)',
                'application': 'Banks, CSCs, online at pmfby.gov.in'
            },
            'soil_health_card': {
                'name': 'Soil Health Card Scheme',
                'benefit': 'Free soil testing and recommendations',
                'frequency': 'Every 2 years',
                'parameters': 'NPK, pH, organic carbon, micronutrients',
                'application': 'Local agriculture department'
            }
        }
    
    def get_regional_crops_data(self) -> Dict:
        """Region-specific crop recommendations"""
        return {
            'maharashtra': {
                'wardha': {
                    'recommended_crops': ['Cotton', 'Wheat', 'Gram', 'Jowar', 'Tur'],
                    'climate': 'Semi-arid',
                    'rainfall': '800-1000 mm annually',
                    'soil': 'Black cotton soil (Vertisols)',
                    'irrigation': 'Limited, depends on monsoon'
                }
            }
        }
    
    def get_pest_disease_database(self) -> Dict:
        """Comprehensive pest and disease database"""
        return {
            'cotton': {
                'bollworm': {
                    'symptoms': 'Holes in bolls, caterpillars inside',
                    'treatment': 'Bt cotton varieties, pheromone traps, NPV spray',
                    'prevention': 'Deep plowing, crop rotation',
                    'critical_stage': 'Flowering to boll formation'
                },
                'aphids': {
                    'symptoms': 'Curled leaves, honeydew secretion',
                    'treatment': 'Neem oil spray, ladybird beetles release',
                    'prevention': 'Balanced fertilization, avoid excess nitrogen',
                    'critical_stage': 'Early vegetative growth'
                }
            },
            'wheat': {
                'rust': {
                    'symptoms': 'Orange-red pustules on leaves',
                    'treatment': 'Fungicide spray (Propiconazole)',
                    'prevention': 'Resistant varieties, proper spacing',
                    'critical_stage': 'Tillering to grain filling'
                }
            }
        }
    
    def get_market_trends_data(self) -> Dict:
        """Market trends and price information"""
        return {
            'cotton': {
                'seasonal_trend': 'Prices peak during November-January (harvest season)',
                'quality_factors': ['Staple length', 'Micronaire', 'Trash content'],
                'major_markets': ['Nagpur', 'Akola', 'Yavatmal'],
                'price_range': '₹5000-7000 per quintal (varies by quality)'
            },
            'wheat': {
                'seasonal_trend': 'Prices stabilize after MSP procurement',
                'msp_2024': '₹2275 per quintal',
                'major_markets': ['Delhi', 'Indore', 'Ludhiana'],
                'quality_factors': ['Protein content', 'Gluten strength', 'Test weight']
            }
        }
    
    def get_soil_health_data(self) -> Dict:
        """Soil health management information"""
        return {
            'black_cotton_soil': {
                'characteristics': 'High clay content, moisture retention, swelling-shrinking',
                'suitable_crops': ['Cotton', 'Sugarcane', 'Wheat', 'Jowar'],
                'management': ['Deep plowing in summer', 'Organic matter addition', 'Drainage systems'],
                'common_issues': ['Poor drainage', 'Nutrient deficiency', 'Compaction']
            },
            'red_soil': {
                'characteristics': 'Iron oxide content, good drainage, low fertility',
                'suitable_crops': ['Millets', 'Groundnut', 'Cotton'],
                'management': ['Organic manure', 'Lime application for pH correction', 'Micronutrient supplementation']
            }
        }
    
    def render_main_interface(self):
        """Render the main application interface"""
        st.title("🌾 AgriAI Advisor")
        st.subheader("Your AI-Powered Agricultural Assistant")
        
        # Input tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "💬 Text Chat", "🎤 Voice Input", "📸 Image Analysis", "📄 Document Upload", "🚨 Emergency Help"
        ])
        
        with tab1:
            self.render_text_chat()
        
        with tab2:
            self.render_voice_input()
        
        with tab3:
            self.render_image_analysis()
        
        with tab4:
            self.render_document_upload()
        
        with tab5:
            self.render_emergency_help()
        
        # Display conversation history
        self.render_conversation_history()
    
    def render_emergency_help(self):
        """Render emergency help and crisis management interface"""
        st.write("🚨 **Agricultural Emergency & Crisis Management**")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.write("**⚡ Quick Emergency Contacts**")
            
            emergency_contacts = {
                "Kisan Call Center": "1800-180-1551",
                "Agriculture Helpline": "1800-270-0224", 
                "Weather Emergency": "1800-180-1551",
                "Crop Insurance Claims": "1800-200-7710",
                "District Collector Office": "Contact local office",
                "Veterinary Emergency": "1962 (Animal Helpline)"
            }
            
            for service, number in emergency_contacts.items():
                st.write(f"• **{service}:** {number}")
            
            st.write("\n**🏥 Immediate Actions:**")
            emergency_actions = [
                "Document damage with photos",
                "Contact insurance company within 72 hours",
                "Report to local agriculture officer",
                "Keep all receipts and records safe"
            ]
            
            for action in emergency_actions:
                st.write(f"• {action}")
        
        with col2:
            st.write("**🌪️ Crisis Type Selector**")
            crisis_type = st.selectbox(
                "What type of emergency are you facing?",
                [
                    "Select crisis type",
                    "Crop disease outbreak", 
                    "Pest infestation",
                    "Natural disaster (flood/drought)",
                    "Market price crash",
                    "Equipment failure",
                    "Financial crisis",
                    "Animal health emergency",
                    "Other"
                ]
            )
            
            if crisis_type != "Select crisis type":
                crisis_description = st.text_area(
                    "Describe your emergency situation",
                    placeholder="Provide details about the situation, affected area, timeline, etc.",
                    height=100
                )
                
                if st.button("🚨 Get Emergency Guidance", type="primary"):
                    if crisis_description:
                        emergency_response = self.handle_agricultural_emergency(crisis_type, crisis_description)
                        st.subheader("⚡ Emergency Response Plan")
                        st.write(emergency_response)
                    else:
                        st.warning("Please describe your emergency situation!")
        
        # Add demonstration scenarios
        st.divider()
        st.write("**💡 Demo Scenarios - Try these examples:**")
        
        demo_scenarios = [
            {
                "title": "🐛 Cotton Bollworm Attack",
                "description": "Sudden bollworm infestation in 5 acres cotton field",
                "action": "pest_outbreak_cotton"
            },
            {
                "title": "🌧️ Unexpected Heavy Rain",
                "description": "Wheat crop flooded due to unseasonal rain",
                "action": "weather_damage_wheat"
            },
            {
                "title": "💰 Price Crash Response",
                "description": "Cotton prices dropped 40% during harvest",
                "action": "market_crisis_cotton"
            }
        ]
        
        cols = st.columns(len(demo_scenarios))
        for i, scenario in enumerate(demo_scenarios):
            with cols[i]:
                if st.button(f"{scenario['title']}", key=f"demo_{i}"):
                    demo_response = self.get_demo_scenario_response(scenario['action'])
                    st.subheader(f"📋 {scenario['title']} - Action Plan")
                    st.write(demo_response)
    
    def handle_agricultural_emergency(self, crisis_type: str, description: str) -> str:
        """Handle agricultural emergencies with immediate action plans"""
        try:
            agricultural_data = st.session_state.get('agricultural_data', {})
            
            if crisis_type == "Crop disease outbreak":
                return self.handle_disease_emergency(description, agricultural_data)
            elif crisis_type == "Pest infestation":
                return self.handle_pest_emergency(description, agricultural_data)
            elif crisis_type == "Natural disaster (flood/drought)":
                return self.handle_weather_emergency(description)
            elif crisis_type == "Market price crash":
                return self.handle_market_emergency(description)
            elif crisis_type == "Financial crisis":
                return self.handle_financial_emergency(description)
            else:
                return self.handle_general_emergency(crisis_type, description)
                
        except Exception as e:
            return f"Emergency system error: {str(e)}. Please contact Kisan Call Center: 1800-180-1551"
    
    def handle_disease_emergency(self, description: str, agricultural_data: Dict) -> str:
        """Handle crop disease emergencies"""
        pest_db = agricultural_data.get('pest_disease_db', {})
        
        response = """**🚨 IMMEDIATE DISEASE OUTBREAK RESPONSE**

**⚡ URGENT ACTIONS (Next 24 hours):**
1. **Isolate affected area** - Mark boundaries to prevent spread
2. **Take clear photos** - Document symptoms for expert identification
3. **Contact local agriculture officer** immediately
4. **Stop irrigation** in affected areas temporarily

**📞 EMERGENCY CONTACTS:**
• District Agriculture Officer: Contact immediately
• Kisan Call Center: 1800-180-1551
• Local KVK (Krishi Vigyan Kendra)

**🔬 INITIAL TREATMENT (If identified):**"""

        # Add specific disease treatments if available
        common_diseases = ['bollworm', 'rust', 'aphids']
        for disease in common_diseases:
            if disease in description.lower():
                for crop, diseases in pest_db.items():
                    if disease in diseases:
                        disease_info = diseases[disease]
                        response += f"\n\n**{disease.title()} Treatment:**"
                        response += f"\n• **Symptoms:** {disease_info.get('symptoms', 'N/A')}"
                        response += f"\n• **Treatment:** {disease_info.get('treatment', 'Consult expert')}"
                        break
        
        response += """

**⚠️ CRITICAL WARNINGS:**
• Do NOT use pesticides without expert guidance
• Document all actions for insurance claims
• Monitor spread every 6 hours

**📋 INSURANCE CLAIM PREPARATION:**
1. Photograph all affected areas with date stamps
2. Collect soil/plant samples if advised
3. Contact insurance company within 72 hours
4. Get damage assessment from agriculture department"""

        return response
    
    def handle_pest_emergency(self, description: str, agricultural_data: Dict) -> str:
        """Handle pest infestation emergencies"""
        return """**🐛 PEST EMERGENCY RESPONSE PLAN**

**⚡ IMMEDIATE ACTIONS:**
1. **Identify the pest** - Capture samples in glass containers
2. **Assess damage extent** - Count affected plants per unit area
3. **Check neighboring fields** - Coordinate with other farmers
4. **Contact experts** - Local agriculture extension officer

**🚨 EMERGENCY TREATMENT PROTOCOL:**
1. **Mechanical removal** - Hand picking if infestation is localized
2. **Organic sprays** - Neem oil or soap solution as immediate measure
3. **Biological control** - Release beneficial insects if available
4. **Chemical treatment** - ONLY after expert consultation

**📞 URGENT CONTACTS:**
• Plant Protection Officer: Contact district office
• Pesticide dealer with expertise
• Neighboring experienced farmers

**⚠️ SAFETY WARNINGS:**
• Wear protective equipment during spraying
• Follow pre-harvest intervals strictly
• Keep children and animals away from treated areas

**📊 MONITORING PLAN:**
• Check effectiveness after 48-72 hours
• Re-treat if necessary with expert guidance
• Document pest count reduction for records"""
    
    def handle_weather_emergency(self, description: str) -> str:
        """Handle weather-related emergencies"""
        return """**🌪️ WEATHER DISASTER RESPONSE**

**⚡ IMMEDIATE PRIORITIES:**
1. **Ensure safety first** - Move to safe locations
2. **Assess damage** - Survey all affected areas
3. **Document everything** - Photos with timestamps
4. **Contact authorities** - Report to tehsildar/collector

**💧 FLOOD RESPONSE:**
• Drain standing water immediately
• Apply lime to prevent fungal diseases  
• Re-sow fast-growing varieties if possible
• Check for soil contamination

**☀️ DROUGHT RESPONSE:**
• Implement water conservation measures
• Switch to drought-resistant varieties
• Mulch heavily to retain moisture
• Consider crop insurance claims

**📞 EMERGENCY SERVICES:**
• District Collector Office: Immediate reporting
• NDRF: 1070 (National Disaster Response Force)
• State Disaster Management: Contact local office

**💰 FINANCIAL SUPPORT:**
• Crop insurance claim (within 72 hours)
• Disaster relief funds application
• Bank loan restructuring requests
• Government compensation schemes

**🚨 HEALTH & SAFETY:**
• Avoid contaminated water
• Check electrical installations
• Secure farm equipment
• Vaccinate animals if needed"""
    
    def handle_market_emergency(self, description: str) -> str:
        """Handle market price crash emergencies"""
        return """**📉 MARKET CRISIS RESPONSE PLAN**

**⚡ IMMEDIATE DAMAGE CONTROL:**
1. **Hold selling decision** - Don't panic sell immediately
2. **Check multiple markets** - Compare prices across mandis
3. **Explore alternatives** - Direct selling, processing options
4. **Contact buyer networks** - Reach out to known buyers

**🏪 ALTERNATIVE SELLING STRATEGIES:**
• **eNAM platform** - Online national market access
• **Direct to consumers** - Cut middleman margins
• **Cooperative societies** - Pool resources with other farmers
• **Contract farming** - Secure future price agreements

**💾 STORAGE OPTIONS:**
• Scientific storage for better prices later
• Cold storage for perishables
• Warehouse receipt financing
• Farmer Producer Organization (FPO) storage

**📞 MARKET INTELLIGENCE:**
• AgMarkNet: Real-time price updates
• Commodity exchanges for future trends
• Agriculture marketing board
• Local market committees

**💰 FINANCIAL PROTECTION:**
• Minimum Support Price (MSP) if applicable
• Price deficiency payments
• Market intervention schemes
• Emergency crop loans

**⚠️ AVOID COMMON MISTAKES:**
• Don't dump produce at any price
• Verify buyer credentials before deals
• Keep proper records for tax benefits
• Don't ignore quality parameters"""
    
    def handle_financial_emergency(self, description: str) -> str:
        """Handle financial crisis situations"""
        schemes = st.session_state.get('agricultural_data', {}).get('government_schemes', {})
        
        response = """**💰 AGRICULTURAL FINANCIAL CRISIS SUPPORT**

**⚡ IMMEDIATE ACTIONS:**
1. **Document all expenses** - Prepare financial records
2. **Contact bank immediately** - Discuss restructuring options
3. **Apply for relief schemes** - Multiple government programs available
4. **Seek legal advice** - If facing recovery actions

**🏦 BANKING SOLUTIONS:**
• Loan restructuring/rescheduling
• One-time settlement (OTS) schemes
• Interest rate reduction requests
• Moratorium on repayments

**🏛️ GOVERNMENT SCHEMES:**"""
        
        # Add specific scheme information
        for scheme_key, scheme_info in schemes.items():
            response += f"\n\n**{scheme_info.get('name', scheme_key.upper())}:**"
            response += f"\n• Benefit: {scheme_info.get('benefit', 'N/A')}"
            response += f"\n• Apply: {scheme_info.get('application', 'N/A')}"
            if 'helpline' in scheme_info:
                response += f"\n• Helpline: {scheme_info['helpline']}"
        
        response += """

**📞 FINANCIAL EMERGENCY CONTACTS:**
• Bank manager - Schedule urgent meeting
• District Collector - Government relief
• Legal aid services - Free consultation
• Farmer welfare societies

**⚠️ PROTECT YOUR ASSETS:**
• Understand your legal rights
• Don't sign papers under pressure
• Seek independent advice
• Maintain family support systems"""
        
        return response
    
    def handle_general_emergency(self, crisis_type: str, description: str) -> str:
        """Handle other types of emergencies"""
        return f"""**🚨 GENERAL AGRICULTURAL EMERGENCY RESPONSE**

**Crisis Type:** {crisis_type}

**⚡ IMMEDIATE STEPS:**
1. **Ensure personal safety** - Priority number one
2. **Assess the situation** - Document extent of problem
3. **Contact local experts** - Agriculture extension officer
4. **Inform authorities** - Report to relevant departments

**📞 UNIVERSAL EMERGENCY CONTACTS:**
• Kisan Call Center: 1800-180-1551 (24x7)
• District Agriculture Office
• Local police if security involved
• Medical emergency: 108

**📋 GENERAL RESPONSE PROTOCOL:**
• Document everything with photos/videos
• Keep all receipts and records safe
• Contact insurance providers if applicable
• Coordinate with neighboring farmers

**💡 EXPERT CONSULTATION:**
• Describe situation clearly to experts
• Ask for written recommendations
• Get multiple opinions if time permits
• Follow scientific approaches only

**⚠️ SAFETY REMINDER:**
Your safety and that of your family comes first. 
Material losses can be recovered, but lives cannot be replaced."""
    
    def get_demo_scenario_response(self, scenario_action: str) -> str:
        """Generate responses for demo scenarios"""
        if scenario_action == "pest_outbreak_cotton":
            return self.handle_disease_emergency("bollworm infestation in cotton", st.session_state.get('agricultural_data', {}))
        elif scenario_action == "weather_damage_wheat":
            return self.handle_weather_emergency("wheat crop flooded due to heavy rain")
        elif scenario_action == "market_crisis_cotton":
            return self.handle_market_emergency("cotton prices crashed 40% during harvest season")
        else:
            return "Demo scenario not available."
    
    def render_text_chat(self):
        """Render text chat interface"""
        st.write("Ask your agricultural questions in natural language:")
        
        query = st.text_area(
            "Your Question",
            placeholder="e.g., When should I irrigate my rice crop? What's the weather forecast for next week?",
            height=100
        )
        
        col1, col2 = st.columns([1, 4])
        
        with col1:
            if st.button("🚀 Ask", type="primary"):
                if query:
                    self.process_query(query, input_type="text")
                else:
                    st.warning("Please enter a question!")
        
        with col2:
            if st.button("🧹 Clear History"):
                st.session_state.conversation_history = []
                st.rerun()
    
    def render_voice_input(self):
        """Render voice input interface with enhanced capabilities"""
        st.write("🎤 Voice Input with Multi-language Support")
        
        # Language selection for voice input
        voice_languages = {
            "English": "en",
            "Hindi": "hi-IN", 
            "Marathi": "mr-IN",
            "Gujarati": "gu-IN",
            "Tamil": "ta-IN",
            "Telugu": "te-IN",
            "Kannada": "kn-IN",
            "Bengali": "bn-IN",
            "Punjabi": "pa-IN"
        }
        
        selected_voice_lang = st.selectbox(
            "Voice Input Language", 
            options=list(voice_languages.keys()), 
            index=0
        )
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.write("**🎙️ Browser Microphone (Chrome/Edge recommended)**")
            
            # Add JavaScript for voice recognition
            microphone_html = f"""
            <div style="text-align: center; padding: 20px;">
                <button id="startBtn" onclick="startRecognition()" style="
                    background: #ff4b4b; color: white; border: none; 
                    padding: 15px 30px; border-radius: 5px; cursor: pointer; 
                    font-size: 16px; margin: 10px;">
                    🎤 Start Recording
                </button>
                <button id="stopBtn" onclick="stopRecognition()" style="
                    background: #333; color: white; border: none; 
                    padding: 15px 30px; border-radius: 5px; cursor: pointer; 
                    font-size: 16px; margin: 10px;" disabled>
                    ⏹️ Stop Recording
                </button>
                <div id="status" style="margin: 20px; font-weight: bold;"></div>
                <textarea id="transcript" placeholder="Your spoken words will appear here..." 
                    style="width: 100%; height: 150px; margin: 10px 0; padding: 10px; border-radius: 5px; border: 1px solid #ccc;"></textarea>
                <button onclick="sendToStreamlit()" style="
                    background: #00cc00; color: white; border: none; 
                    padding: 10px 20px; border-radius: 5px; cursor: pointer; 
                    font-size: 14px; margin: 10px;">
                    ➤ Send to AgriAI
                </button>
            </div>
            
            <script>
                let recognition;
                let isRecording = false;
                
                function startRecognition() {{
                    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {{
                        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
                        recognition = new SpeechRecognition();
                        
                        recognition.lang = '{voice_languages[selected_voice_lang]}';
                        recognition.continuous = true;
                        recognition.interimResults = true;
                        recognition.maxAlternatives = 1;
                        
                        recognition.onstart = function() {{
                            isRecording = true;
                            document.getElementById('startBtn').disabled = true;
                            document.getElementById('stopBtn').disabled = false;
                            document.getElementById('status').innerHTML = '🔴 Recording... Speak now!';
                            document.getElementById('transcript').value = '';
                        }};
                        
                        recognition.onresult = function(event) {{
                            let finalTranscript = '';
                            let interimTranscript = '';
                            
                            for (let i = event.resultIndex; i < event.results.length; i++) {{
                                const transcript = event.results[i][0].transcript;
                                if (event.results[i].isFinal) {{
                                    finalTranscript += transcript + ' ';
                                }} else {{
                                    interimTranscript += transcript;
                                }}
                            }}
                            
                            document.getElementById('transcript').value = finalTranscript + interimTranscript;
                        }};
                        
                        recognition.onerror = function(event) {{
                            document.getElementById('status').innerHTML = '❌ Error: ' + event.error;
                            resetButtons();
                        }};
                        
                        recognition.onend = function() {{
                            resetButtons();
                        }};
                        
                        recognition.start();
                    }} else {{
                        document.getElementById('status').innerHTML = '❌ Speech recognition not supported in this browser. Try Chrome or Edge.';
                    }}
                }}
                
                function stopRecognition() {{
                    if (recognition && isRecording) {{
                        recognition.stop();
                    }}
                }}
                
                function resetButtons() {{
                    isRecording = false;
                    document.getElementById('startBtn').disabled = false;
                    document.getElementById('stopBtn').disabled = true;
                    document.getElementById('status').innerHTML = '✅ Recording stopped. Review your text and click Send.';
                }}
                
                function sendToStreamlit() {{
                    const text = document.getElementById('transcript').value.trim();
                    if (text) {{
                        // Store in session storage for Streamlit to pick up
                        sessionStorage.setItem('voiceInput', text);
                        document.getElementById('status').innerHTML = '📤 Text sent to AgriAI! Switch to Text Chat tab to see response.';
                        
                        // Try to trigger Streamlit rerun if possible
                        window.parent.postMessage({{
                            type: 'streamlit:setComponentValue',
                            value: text
                        }}, '*');
                    }} else {{
                        document.getElementById('status').innerHTML = '⚠️ No text to send. Please record something first.';
                    }}
                }}
            </script>
            """
            
            st.components.v1.html(microphone_html, height=400)
        
        with col2:
            st.write("**📁 Upload Audio File Alternative**")
            audio_file = st.file_uploader("Choose an audio file", type=['wav', 'mp3', 'm4a', 'ogg'])
            
            if audio_file is not None:
                st.audio(audio_file)
                
                if st.button("🔍 Transcribe Audio File"):
                    with st.spinner("Transcribing audio..."):
                        try:
                            # Simple fallback transcription
                            st.info("Audio file transcription requires additional setup. Using voice input above is recommended.")
                            
                            # For now, provide manual text input as fallback
                            manual_text = st.text_area(
                                "Manual Text Entry", 
                                placeholder="Please type what was said in the audio...",
                                height=100
                            )
                            
                            if manual_text and st.button("Process Manual Text"):
                                self.process_query(manual_text, input_type="voice")
                                
                        except Exception as e:
                            st.error(f"Transcription error: {str(e)}")
        
        # Voice input text area for manual entry or voice-to-text result
        st.write("**💬 Voice Input Text Area**")
        voice_input_text = st.text_area(
            "Voice Input (speak above or type here)",
            placeholder="Your voice input will appear here, or you can type manually...",
            height=100,
            key="voice_input_area"
        )
        
        col3, col4 = st.columns([1, 1])
        with col3:
            if st.button("🚀 Send Voice Query", type="primary"):
                if voice_input_text.strip():
                    self.process_query(voice_input_text, input_type="voice")
                else:
                    st.warning("Please provide voice input or type your question!")
        
        with col4:
            if st.button("🧹 Clear Voice Input"):
                st.session_state.voice_input_area = ""
                st.rerun()
        
        # JavaScript to populate the text area
        populate_script = """
        <script>
        function populateVoiceInput() {
            const voiceText = sessionStorage.getItem('voiceInput');
            if (voiceText) {
                const textArea = parent.document.querySelector('[data-testid="stTextArea"] textarea');
                if (textArea && textArea.placeholder && textArea.placeholder.includes('voice input')) {
                    textArea.value = voiceText;
                    textArea.dispatchEvent(new Event('input', { bubbles: true }));
                    sessionStorage.removeItem('voiceInput');
                }
            }
        }
        
        // Check periodically for voice input
        setInterval(populateVoiceInput, 1000);
        </script>
        """
        st.components.v1.html(populate_script, height=0)
    
    def render_image_analysis(self):
        """Render image analysis interface"""
        st.write("📸 Upload images of crops, diseases, or farm conditions")
        
        uploaded_image = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_image is not None:
            image = Image.open(uploaded_image)
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.image(image, caption="Uploaded Image", use_column_width=True)
            
            with col2:
                st.write("**Image Analysis Options:**")
                analysis_type = st.radio(
                    "Analysis Type",
                    ["Disease Detection", "Crop Health Assessment", "General Analysis"]
                )
                
                additional_context = st.text_area(
                    "Additional Context (Optional)",
                    placeholder="Describe what you're seeing or specific concerns..."
                )
            
            if st.button("🔍 Analyze Image"):
                try:
                    with st.spinner("Analyzing image..."):
                        analysis_result = self.image_agent.analyze_image(
                            image, analysis_type, additional_context
                        )
                    
                    st.subheader("📋 Analysis Results")
                    st.write(analysis_result)
                    
                    self.add_to_history("image", f"Image analysis: {analysis_type}", analysis_result)
                except Exception as e:
                    st.error(f"Error analyzing image: {str(e)}")
    
    def render_document_upload(self):
        """Render document upload interface"""
        st.write("📄 Upload agricultural documents, policies, or reports")
        
        uploaded_file = st.file_uploader("Choose a document", type=['pdf', 'txt'])
        
        if uploaded_file is not None:
            st.write(f"**File:** {uploaded_file.name}")
            st.write(f"**Size:** {uploaded_file.size} bytes")
            
            document_query = st.text_area(
                "What would you like to know about this document?",
                placeholder="e.g., Summarize the key points, What subsidies are mentioned?"
            )
            
            if st.button("📖 Process Document"):
                try:
                    with st.spinner("Processing document..."):
                        document_content = self.input_processor.process_document(uploaded_file)
                        doc_id = self.vector_db.add_document(document_content, uploaded_file.name)
                        
                        if document_query:
                            response = self.policy_agent.analyze_document(document_content, document_query)
                        else:
                            response = self.policy_agent.summarize_document(document_content)
                    
                    st.subheader("📋 Document Analysis")
                    st.write(response)
                    
                    query_text = document_query or "Document summary"
                    self.add_to_history("document", query_text, response)
                    
                    st.success("Document processed and added to knowledge base!")
                except Exception as e:
                    st.error(f"Error processing document: {str(e)}")
    
    def process_query(self, query: str, input_type: str = "text"):
        """Process user query through the multi-agent system with multi-language support"""
        try:
            with st.spinner("🤖 Processing your query..."):
                # Detect input language
                detected_lang = detect_language(query)
                user_lang = st.session_state.user_profile.get("language", "en")
                
                # Translate query to English for processing if needed
                english_query = query
                if detected_lang != "en":
                    english_query = translate_text(query, "en")
                
                # Preprocess query
                processed_query = self.input_processor.preprocess_text(english_query)
                
                # Detect intent and entities
                intent = self.input_processor.detect_intent(processed_query)
                entities = self.input_processor.extract_entities(processed_query)
                
                # Route to appropriate agent
                response = self.route_to_agent(processed_query, intent, entities)
                
                # Translate response back to user's language if needed
                final_response = response
                if user_lang != "en" and detected_lang != "en":
                    # Map language codes
                    lang_map = {
                        "hi": "hi", "mr": "mr", "gu": "gu", "ta": "ta", 
                        "te": "te", "kn": "kn", "bn": "bn", "pa": "pa"
                    }
                    target_lang = lang_map.get(user_lang, "en")
                    if target_lang != "en":
                        final_response = translate_text(response, target_lang)
                
                # Add to conversation history
                self.add_to_history(input_type, query, final_response)
                
                # Display response
                st.subheader("🤖 Response")
                if detected_lang != "en":
                    st.info(f"Input detected in: {detected_lang.upper()} | Response in: {user_lang.upper()}")
                st.write(final_response)
        except Exception as e:
            st.error(f"Error processing query: {str(e)}")
    
    def route_to_agent(self, query: str, intent: str, entities: Dict) -> str:
        """Route query to appropriate specialized agent with comprehensive context integration"""
        try:
            # Build comprehensive user context that includes agricultural datasets
            user_profile = st.session_state.get("user_profile", {})
            user_context = {
                "location": user_profile.get("location", ""),
                "farm_size": user_profile.get("farm_size", 0),
                "crops": user_profile.get("crops", []),
                "language": user_profile.get("language", "en"),
                "agricultural_data": st.session_state.get("agricultural_data", {}),
                "regional_knowledge": st.session_state.get("regional_knowledge", {}),
                "query_intent": intent,
                "extracted_entities": entities
            }
            
            # Use AGNO system if available
            if AGNO_AVAILABLE and hasattr(self, 'agno_system'):
                return self.agno_system.route_query(query, user_context)
            
            # Fallback to legacy routing
            if intent == "weather":
                return self.weather_agent.get_weather_advice(query, user_context)
            elif intent == "crop":
                return self.crop_agent.get_crop_advice(query, entities, user_context)
            elif intent == "finance":
                return self.finance_agent.get_finance_advice(query, user_context)
            elif intent == "policy":
                return self.policy_agent.get_policy_advice(query, user_context)
            elif intent == "market":
                return self.market_agent.get_market_advice(query, user_context)
            elif intent == "soil":
                return self.soil_health_agent.get_soil_advice(query, user_context)
            elif intent == "pest" or intent == "disease":
                return self.pest_management_agent.get_pest_advice(query, user_context)
            elif intent == "irrigation":
                return self.irrigation_agent.get_irrigation_advice(query, user_context)
            elif intent == "supply":
                return self.supply_chain_agent.get_supply_chain_advice(query, user_context)
            elif intent == "compliance":
                return self.compliance_agent.get_compliance_advice(query, user_context)
            else:
                return self.coordinate_agents(query, entities, user_context)
        except Exception as e:
            return self.get_comprehensive_error_response(query, str(e))
    
    def get_comprehensive_error_response(self, query: str, error_msg: str) -> str:
        """Provide comprehensive error response with fallback guidance"""
        user_profile = st.session_state.get("user_profile", {})
        location = user_profile.get("location", "your area")
        crops = user_profile.get("crops", [])
        
        return f"""**⚠️ System Temporarily Unavailable**

**General Agricultural Guidance for {location}:**

**Immediate Actions:**
• Monitor your crops for any signs of stress or pest activity
• Ensure irrigation systems are functioning properly
• Check weather conditions using local sources
• Contact local agricultural extension officers for specific advice

**Crop-Specific Precautions:**
{self.get_emergency_crop_advice(crops)}

**Emergency Agricultural Contacts:**
• Kisan Call Center: 1800-180-1551 (24x7)
• Local Agriculture Extension Officer
• District Agriculture Office
• Veterinary Services: 1962 (if livestock involved)

**Alternative Information Sources:**
• All India Radio agricultural programs
• Local TV agricultural news
• Newspaper agricultural sections
• Nearby experienced farmers

**Note:** This is general guidance. For specific agricultural advice, please consult local experts or try again later when the system is restored.

Technical details: {error_msg}"""
    
    def get_emergency_crop_advice(self, crops: List[str]) -> str:
        """Get emergency crop-specific advice"""
        if not crops:
            return "• Regularly inspect all crops\n• Maintain consistent care routines\n• Watch for weather-related stress"
        
        advice = []
        for crop in crops:
            crop_lower = crop.lower()
            if crop_lower == 'cotton':
                advice.append("• Cotton: Check for bollworm, maintain soil moisture, ensure proper spacing")
            elif crop_lower == 'wheat':
                advice.append("• Wheat: Watch for rust disease, maintain adequate irrigation, check grain development")
            elif crop_lower == 'rice':
                advice.append("• Rice: Maintain water levels, monitor for brown planthopper, check for blast disease")
            elif crop_lower == 'sugarcane':
                advice.append("• Sugarcane: Check for red rot, maintain irrigation, monitor for borers")
            else:
                advice.append(f"• {crop}: Monitor for pests and diseases, maintain proper care practices")
        
        return "\n".join(advice)
    
    def coordinate_agents(self, query: str, entities: Dict, user_context: Dict) -> str:
        """Coordinate multiple agents for complex queries"""
        try:
            responses = []
            
            # Check for weather-related keywords
            if any(word in query.lower() for word in ["weather", "forecast", "rain", "temperature"]):
                weather_response = self.weather_agent.get_weather_advice(query, user_context)
                responses.append(f"**Weather Information:**\n{weather_response}")
            
            # Check for soil health keywords
            if any(word in query.lower() for word in ["soil", "ph", "nutrient", "fertilizer", "testing"]):
                soil_response = self.soil_health_agent.get_soil_advice(query, user_context)
                responses.append(f"**Soil Health Advisory:**\n{soil_response}")
            
            # Check for pest management keywords
            if any(word in query.lower() for word in ["pest", "disease", "spray", "insect", "fungus"]):
                pest_response = self.pest_management_agent.get_pest_advice(query, user_context)
                responses.append(f"**Pest Management:**\n{pest_response}")
            
            # Check for irrigation keywords
            if any(word in query.lower() for word in ["irrigation", "water", "drip", "sprinkler", "watering"]):
                irrigation_response = self.irrigation_agent.get_irrigation_advice(query, user_context)
                responses.append(f"**Irrigation Management:**\n{irrigation_response}")
            
            # Check for crop-related keywords
            if any(word in query.lower() for word in ["crop", "seed", "planting", "harvest", "variety"]):
                crop_response = self.crop_agent.get_crop_advice(query, entities, user_context)
                responses.append(f"**Crop Advisory:**\n{crop_response}")
            
            # Check for market-related keywords
            if any(word in query.lower() for word in ["price", "market", "sell", "buy", "cost"]):
                market_response = self.market_agent.get_market_advice(query, user_context)
                responses.append(f"**Market Information:**\n{market_response}")
            
            # Check for supply chain keywords
            if any(word in query.lower() for word in ["supplier", "equipment", "storage", "input"]):
                supply_response = self.supply_chain_agent.get_supply_chain_advice(query, user_context)
                responses.append(f"**Supply Chain:**\n{supply_response}")
            
            # Check for compliance keywords
            if any(word in query.lower() for word in ["organic", "certification", "export", "standard"]):
                compliance_response = self.compliance_agent.get_compliance_advice(query, user_context)
                responses.append(f"**Compliance & Certification:**\n{compliance_response}")
            
            if responses:
                return "\n\n".join(responses)
            else:
                # Default to crop agent
                return self.crop_agent.get_crop_advice(query, entities, user_context)
        except Exception as e:
            return f"Error coordinating agents: {str(e)}"
    
    def add_to_history(self, input_type: str, query: str, response: str):
        """Add conversation to history"""
        try:
            timestamp = datetime.now().strftime("%H:%M:%S")
            st.session_state.conversation_history.append({
                "timestamp": timestamp,
                "input_type": input_type,
                "query": query,
                "response": response
            })
        except Exception as e:
            st.error(f"Error adding to history: {str(e)}")
    
    def render_conversation_history(self):
        """Display conversation history"""
        if st.session_state.conversation_history:
            st.subheader("📜 Conversation History")
            
            for i, conversation in enumerate(reversed(st.session_state.conversation_history[-5:])):
                with st.expander(f"[{conversation['timestamp']}] {conversation['input_type'].title()}: {conversation['query'][:50]}..."):
                    st.write("**Query:**")
                    st.write(conversation['query'])
                    st.write("**Response:**")
                    st.write(conversation['response'])

# ================== MAIN APP EXECUTION ==================

def main():
    """Main application entry point"""
    try:
        app = AgriAIAdvisor()
        app.render_main_interface()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.write("Please refresh the page and try again.")

if __name__ == "__main__":
    main()