import streamlit as st
import os
import time
from datetime import datetime
import logging
from dotenv import load_dotenv
import base64
import streamlit.components.v1 as components

# Import health check utility
try:
    from utils.streamlit_health import add_health_endpoint, show_cache_stats
except ImportError:
    def add_health_endpoint(): pass
    def show_cache_stats(): pass

# Ensure DiseaseDetector symbol exists for any legacy references without importing heavy modules
DiseaseDetector = None  # Avoid NameError in any cached/legacy paths; real loading happens via utils.performance_cache

 

# Force load environment variables from .env file
import os
load_dotenv(override=True)

# Debug: Check if .env file exists and try to load it explicitly
env_file_path = os.path.join(os.path.dirname(__file__), '.env')
print(f"Looking for .env file at: {env_file_path}")
print(f".env file exists: {os.path.exists(env_file_path)}")

# Try loading from explicit path
if os.path.exists(env_file_path):
    load_dotenv(env_file_path, override=True)
    print("Loaded .env file explicitly")

# Initialize messaging service for production use
try:
    from utils.messaging_service import messaging_service
    MESSAGING_AVAILABLE = True
    print("✅ Production messaging service initialized")
except ImportError:
    messaging_service = None
    MESSAGING_AVAILABLE = False
    print("⚠️ Messaging service not available")

import numpy as np
from PIL import Image
from typing import Optional, Dict, Any

# Safe Translation import with fallback (using deep-translator instead of googletrans)
try:
    from deep_translator import GoogleTranslator
    TRANSLATOR_AVAILABLE = True
    print("✅ Deep Translator imported successfully")
except ImportError as e:
    print(f"⚠️ Deep Translator import failed: {e}")
    print("🔄 Translation features will be disabled")
    GoogleTranslator = None
    TRANSLATOR_AVAILABLE = False
except Exception as e:
    print(f"❌ Deep Translator error: {e}")
    GoogleTranslator = None
    TRANSLATOR_AVAILABLE = False

# Import performance optimizations
from utils.performance_cache import (
    optimize_streamlit_config, 
    preload_critical_components,
    get_disease_detector,
    get_voice_processor, 
    get_agricultural_agents,
    cached_weather_data,
    cached_translation
)

# Apply performance optimizations (sets page config first)
optimize_streamlit_config()
preload_critical_components()
# Import utilities - will be loaded lazily
from utils.ui_helpers import UIHelpers
from utils.auth_manager import AuthManager
from utils.database_manager import DatabaseManager
from utils.ocr_processor import OCRProcessor
# Agricultural knowledge will be loaded from JSON file

# Configure logging
logging.basicConfig(level=logging.INFO)

 

# Inject a one-time client-side cache buster to avoid stale JS chunk errors
def _inject_cache_buster():
    components.html(
        """
        <script>
        (async () => {
          try {
            const KEY = 'st_cache_busted_v3';
            if (window.sessionStorage.getItem(KEY)) return;

            // Unregister any service workers
            if ('serviceWorker' in navigator) {
              try {
                const regs = await navigator.serviceWorker.getRegistrations();
                for (const r of regs) { try { await r.unregister(); } catch (e) {} }
              } catch (e) {}
            }

            // Clear caches (static chunks)
            if (window.caches && caches.keys) {
              try {
                const names = await caches.keys();
                await Promise.all(names.map(n => caches.delete(n)));
              } catch (e) {}
            }

            // Mark as done and reload with cache-busting query param
            window.sessionStorage.setItem(KEY, '1');
            const url = new URL(window.location.href);
            url.searchParams.set('_cb', Date.now().toString());
            window.location.replace(url.toString());
          } catch (e) {
            // no-op
          }
        })();
        </script>
        """,
        height=0,
    )

_inject_cache_buster()

# Dark mode CSS fix - remove box outlines (after page config)
st.markdown("""
<style>
    .stSelectbox > div > div {
        border: none !important;
        box-shadow: none !important;
    }
    .stTextInput > div > div {
        border: none !important;
        box-shadow: none !important;
    }
    .stFileUploader > div {
        border: none !important;
        box-shadow: none !important;
    }
    .stButton > button {
        border: none !important;
        box-shadow: none !important;
    }
    [data-testid="stSidebar"] {
        border-right: none !important;
    }
    .stTabs [data-baseweb="tab"] {
        border: none !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize authentication manager
auth_manager = AuthManager()

# Initialize session state (preserved for backward compatibility)
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'user_profile' not in st.session_state:
    st.session_state.user_profile = {}
if 'language' not in st.session_state:
    st.session_state.language = 'English'
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'  # Default to light mode
if 'background_image' not in st.session_state:
    st.session_state.background_image = None  # Start with no background

# Docker service endpoints configuration
DOCKER_SERVICES = {
    'disease_service': os.getenv('DISEASE_SERVICE_URL', 'http://disease-service:8899'),
    'ocr_service': os.getenv('OCR_SERVICE_URL', 'http://ocr-service:8898'),
    'redis_cache': os.getenv('REDIS_URL', 'redis://redis-cache:6379')
}

def validate_docker_services():
    """Validate Docker service connectivity with comprehensive error handling"""
    service_status = {}
    
    for service_name, service_url in DOCKER_SERVICES.items():
        try:
            if service_name == 'redis_cache':
                # Skip Redis validation for now, handle separately
                service_status[service_name] = {'status': 'available', 'url': service_url}
                continue
                
            # Test health endpoint
            import requests
            response = requests.get(f"{service_url}/health", timeout=5)
            
            if response.status_code == 200:
                service_status[service_name] = {
                    'status': 'healthy', 
                    'url': service_url,
                    'response': response.json()
                }
            else:
                service_status[service_name] = {
                    'status': 'unhealthy', 
                    'url': service_url,
                    'error': f"HTTP {response.status_code}"
                }
                
        except Exception as e:
            service_status[service_name] = {
                'status': 'unavailable', 
                'url': service_url,
                'error': str(e)
            }
    
    return service_status

# Initialize services (removed caching to ensure updated classes are loaded)
def initialize_services():
    """Initialize all services without caching to ensure updates are loaded"""
    try:
        # Validate Docker services first
        service_status = validate_docker_services()
        
        # Initialize vector store with agricultural knowledge (lazy load)
        from utils.vector_store import AgricultureKnowledgeBase
        vector_store = AgricultureKnowledgeBase()
        
        # Initialize other services with lazy loading
        disease_detector = get_disease_detector()
        from utils.weather_service import WeatherService
        weather_service = WeatherService()
        voice_processor = get_voice_processor()
        agent_team = get_agricultural_agents()
        ui_helpers = UIHelpers()
        
        # Initialize translator with fallback (using deep-translator)
        if TRANSLATOR_AVAILABLE and GoogleTranslator:
            try:
                # deep-translator doesn't need initialization, it's used directly
                translator = GoogleTranslator
                print("✅ Deep Translator initialized successfully")
            except Exception as e:
                print(f"⚠️ Deep Translator initialization failed: {e}")
                translator = None
        else:
            translator = None
            print("🔄 Deep Translator not available - translation features disabled")
            
        ocr_processor = OCRProcessor()
        
        # Add missing methods directly if not present (workaround for caching issues)
        if not hasattr(agent_team, 'get_government_schemes'):
            def get_government_schemes(user_profile, query=""):
                """Get government schemes based on user profile and query"""
                try:
                    if query:
                        schemes = agent_team.policy_agent.search_schemes_by_query(query, user_profile)
                        analysis = agent_team.policy_agent.get_scheme_analysis(schemes, user_profile)
                        return {
                            "schemes": schemes,
                            "analysis": analysis,
                            "search_type": "query",
                            "query": query,
                            "total_found": len(schemes)
                        }
                    else:
                        schemes = agent_team.policy_agent.get_profile_based_schemes(user_profile)
                        analysis = agent_team.policy_agent.get_scheme_analysis(schemes, user_profile)
                        return {
                            "schemes": schemes,
                            "analysis": analysis,
                            "search_type": "profile",
                            "total_found": len(schemes)
                        }
                except Exception as e:
                    return {
                        "schemes": [],
                        "analysis": f"Error retrieving government schemes: {str(e)}",
                        "search_type": "error",
                        "total_found": 0
                    }
            
            def get_scheme_updates_info():
                """Get information about scheme cache status and updates"""
                try:
                    return agent_team.policy_agent.get_scheme_updates()
                except Exception as e:
                    return {
                        "error": f"Error getting scheme updates: {str(e)}",
                        "last_update": None,
                        "total_schemes_cached": 0,
                        "cache_valid": False
                    }
            
            def force_update_schemes():
                """Force update of government schemes cache"""
                try:
                    return agent_team.policy_agent.force_update_schemes()
                except Exception as e:
                    return False
            
            # Add methods to the agent_team instance
            agent_team.get_government_schemes = get_government_schemes
            agent_team.get_scheme_updates_info = get_scheme_updates_info
            agent_team.force_update_schemes = force_update_schemes
        
        return {
            'vector_store': vector_store,
            'disease_detector': disease_detector,
            'weather_service': weather_service,
            'voice_processor': voice_processor,
            'agent_team': agent_team,
            'ui_helpers': ui_helpers,
            'translator': translator,
            'ocr_processor': ocr_processor
        }
    except Exception as e:
        st.error(f"Error initializing services: {str(e)}")
        return None

# Load services
services = initialize_services()

if not services:
    st.error("Failed to initialize services. Please check your configuration.")
    st.stop()

# Background image functions
def load_background_image(image_path):
    """Load and encode background image to base64"""
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        return encoded_string
    except Exception as e:
        st.error(f"Error loading background image: {e}")
        return None

def get_available_backgrounds():
    """Get list of available background images"""
    bg_folder = "bg_images"
    backgrounds = {"None": None}
    
    if os.path.exists(bg_folder):
        files = sorted([f for f in os.listdir(bg_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        # Add default background (first image)
        if files:
            backgrounds["Default"] = os.path.join(bg_folder, files[0])
        
        # Add numbered backgrounds
        for i, file in enumerate(files, 1):
            backgrounds[f"Background {i}"] = os.path.join(bg_folder, file)
    
    return backgrounds

# Apply custom CSS for background and styling
def apply_custom_css(theme="light", background_image=None):
    """Apply custom CSS with theme support and background images"""
    
    # Background image CSS if provided
    bg_css = ""
    if background_image:
        bg_css = f"""
        .stApp {{
            background-image: linear-gradient(rgba(0,0,0,0.3), rgba(0,0,0,0.3)), url("data:image/png;base64,{background_image}") !important;
            background-size: cover !important;
            background-position: center !important;
            background-repeat: no-repeat !important;
            background-attachment: fixed !important;
        }}
        
        .main > div {{
            background: rgba(255, 255, 255, 0.95) !important;
            backdrop-filter: blur(5px) !important;
            border-radius: 15px !important;
            margin: 15px !important;
            padding: 25px !important;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1) !important;
        }}
        
        .block-container {{
            background: rgba(255, 255, 255, 0.9) !important;
            border-radius: 15px !important;
            padding: 20px !important;
            margin: 10px !important;
        }}
        """
    
    if theme == "dark":
        # Dark theme styles
        css = f"""
        <style>
        {bg_css}
        
        .main {{
            {"background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);" if not background_image else ""}
            color: #ffffff !important;
        }}
        
        .stApp {{
            {"background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);" if not background_image else ""}
        }}
        
        /* Enhanced visibility for dark theme with backgrounds */
        .stMarkdown, .stMarkdown p, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {{
            color: #ffffff !important;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.8) !important;
        }}
        
        .block-container {{
            background: rgba(26, 26, 46, 0.95) !important;
            backdrop-filter: blur(10px) !important;
            border-radius: 15px !important;
            padding: 25px !important;
            margin: 15px !important;
            border: none !important;
            box-shadow: none !important;
        }}
        
        .stButton > button {{
            background: rgba(76, 175, 80, 0.9) !important;
            color: #ffffff !important;
            border: none !important;
            box-shadow: none !important;
            backdrop-filter: blur(5px) !important;
        }}
        
        .stSelectbox > div > div {{
            background: rgba(45, 45, 45, 0.95) !important;
            color: #ffffff !important;
            border: none !important;
            box-shadow: none !important;
            backdrop-filter: blur(5px) !important;
        }}
        
        .typing-animation {{
            border-right: 2px solid #4CAF50;
            white-space: nowrap;
            overflow: hidden;
            animation: typing 3s steps(40, end), blink-caret 0.75s step-end infinite;
            color: #ffffff;
        }}
        
        @keyframes typing {{
            from {{ width: 0; }}
            to {{ width: 100%; }}
        }}
        
        @keyframes blink-caret {{
            from, to {{ border-color: transparent; }}
            50% {{ border-color: #4CAF50; }}
        }}
        
        .stSelectbox > div > div {{
            background-color: rgba(30, 30, 30, 0.9);
            color: #ffffff;
        }}
        
        .agent-response {{
            background: linear-gradient(135deg, rgba(45, 80, 22, 0.8) 0%, rgba(62, 107, 31, 0.9) 100%);
            border: 1px solid rgba(76, 175, 80, 0.3);
            padding: 15px;
            border-radius: 15px;
            margin: 10px 0;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            color: #ffffff;
            backdrop-filter: blur(10px);
        }}
        
        .disease-result {{
            background: linear-gradient(135deg, rgba(74, 44, 42, 0.8) 0%, rgba(109, 76, 65, 0.9) 100%);
            border: 1px solid rgba(255, 152, 0, 0.3);
            padding: 15px;
            border-radius: 15px;
            margin: 10px 0;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            color: #ffffff;
            backdrop-filter: blur(10px);
        }}
        
        .weather-info {{
            background: linear-gradient(135deg, rgba(26, 35, 126, 0.8) 0%, rgba(40, 53, 147, 0.9) 100%);
            border: 1px solid rgba(33, 150, 243, 0.3);
            padding: 15px;
            border-radius: 15px;
            margin: 10px 0;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            color: #ffffff;
            backdrop-filter: blur(10px);
        }}
        
        .theme-toggle {{
            background: #4CAF50;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 20px;
            cursor: pointer;
            margin: 5px;
        }}
        
        /* Dark theme for Streamlit components */
        .stTextInput > div > div > input {{
            background-color: #2d2d2d;
            color: #ffffff;
        }}
        
        .stTextArea > div > div > textarea {{
            background-color: #2d2d2d;
            color: #ffffff;
        }}
        
        .stMarkdown {{
            color: #ffffff;
        }}
        
        .stSidebar {{
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%) !important;
            backdrop-filter: blur(10px) !important;
        }}
        
        .stSidebar .stMarkdown {{
            color: #ffffff !important;
            text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.8) !important;
        }}
        
        .stSidebar .stSelectbox > div > div {{
            background: rgba(45, 45, 45, 0.95) !important;
            color: #ffffff !important;
            border: none !important;
            box-shadow: none !important;
        }}
        
        .stSidebar .stButton > button {{
            background: rgba(76, 175, 80, 0.9) !important;
            color: #ffffff !important;
            border: none !important;
            box-shadow: none !important;
        }}
        
        /* Improve card visibility */
        div[data-testid="stVerticalBlock"] > div {{
            background: rgba(26, 26, 46, 0.9) !important;
            border-radius: 10px !important;
            padding: 15px !important;
            margin: 10px 0 !important;
            backdrop-filter: blur(8px) !important;
            border: none !important;
            box-shadow: none !important;
        }}
        
        /* Improve metric boxes */
        div[data-testid="metric-container"] {{
            background: rgba(26, 26, 46, 0.95) !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 10px !important;
            backdrop-filter: blur(5px) !important;
        }}
        
        /* Remove green outlines/focus rings and borders across widgets */
        *:focus {{ outline: none !important; box-shadow: none !important; }}
        .stTextInput > div > div,
        .stTextArea > div > div,
        .stNumberInput > div > div,
        .stFileUploader > div,
        .stMultiSelect > div > div,
        .stSelectbox > div > div,
        .stDateInput > div > div,
        .stTimeInput > div > div {{
            border: none !important;
            box-shadow: none !important;
        }}
        
        /* BaseWeb (Streamlit) select/radio/checkbox focus + borders */
        [data-baseweb="select"],
        [data-baseweb="radio"],
        [data-baseweb="checkbox"] {{
            border: none !important;
            box-shadow: none !important;
        }}
        
        /* Tabs highlight line */
        .stTabs [data-baseweb="tab-highlight"] {{
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
        }}
        
        /* Code blocks and markdown containers */
        [data-testid="stMarkdownContainer"] {{
            border: none !important;
            box-shadow: none !important;
        }}
        </style>
        """
    else:
        # Light theme styles (default)
        css = f"""
        <style>
        {bg_css}
        
        .main {{
            {"background-image: linear-gradient(rgba(46, 125, 50, 0.1), rgba(139, 195, 74, 0.1)), url('https://pixabay.com/get/g9836796093aea7b1d746ecb5d2e8c808d0b7f7862e8d88430f5ed4fabe325ae828373c8f1b978dcf521dd72e659099bc84cb0a9a5507a3c9200e7e2e3a44a94c_1280.jpg'); background-size: cover; background-attachment: fixed; background-repeat: no-repeat;" if not background_image else ""}
        }}
        
        .typing-animation {{
            border-right: 2px solid #2E7D32;
            white-space: nowrap;
            overflow: hidden;
            animation: typing 3s steps(40, end), blink-caret 0.75s step-end infinite;
        }}
        
        @keyframes typing {{
            from {{ width: 0; }}
            to {{ width: 100%; }}
        }}
        
        @keyframes blink-caret {{
            from, to {{ border-color: transparent; }}
            50% {{ border-color: #2E7D32; }}
        }}
        
        .stSelectbox > div > div {{
            background-color: rgba(255, 255, 255, 0.9);
        }}
        
        .agent-response {{
            background: linear-gradient(135deg, #E8F5E8 0%, #F1F8E9 100%);
            border-left: 4px solid #4CAF50;
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            color: #2E7D32 !important;
        }}
        
        .disease-result {{
            background: linear-gradient(135deg, #FFF3E0 0%, #FFCC80 30%);
            border-left: 4px solid #FF9800;
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
            color: #E65100 !important;
        }}
        
        .weather-info {{
            background: linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 30%);
            border-left: 4px solid #2196F3;
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
            color: #1565C0 !important;
        }}
        
        .theme-toggle {{
            background: #4CAF50;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 20px;
            cursor: pointer;
            margin: 5px;
        }}
        </style>
        """
    
    st.markdown(css, unsafe_allow_html=True)

# Apply CSS with current theme and background
background_encoded = None
if st.session_state.background_image:
    background_encoded = load_background_image(st.session_state.background_image)

apply_custom_css(st.session_state.theme, background_encoded)

# Sidebar navigation
st.sidebar.title("🌾 Agricultural AI Advisor")
st.sidebar.markdown("---")

# Authentication check - show login if not authenticated
if not auth_manager.is_authenticated() and not st.session_state.get('guest_mode', False):
    # Show authentication form
    if not auth_manager.show_auth_form():
        st.stop()  # Stop execution until user logs in

# Show user info in sidebar
auth_manager.show_user_info()

page = st.sidebar.selectbox(
    "Navigate to:",
    [
        "🏠 Home",
        "🤖 AI Chat Assistant", 
        "🔬 Disease Detection",
        "📷 OCR Text Extraction",
        "🌤️ Weather Information",
        "💰 Financial Advisor",
        "📚 Knowledge Base",
        "🏛️ Government Schemes",
        "📢 Notifications",
        "👤 User Profile",
        "🗄️ Data Management",
        "🔐 Login"
    ]
)

# Language selection
language = st.sidebar.selectbox(
    "Select Language / भाषा चুनें:",
    ["English", "हिन्दी (Hindi)", "தமிழ் (Tamil)", "తెలుగు (Telugu)", "বাংলা (Bengali)"]
)
st.session_state.language = language

# Theme toggle
st.sidebar.markdown("---")
theme_col1, theme_col2 = st.sidebar.columns(2)

with theme_col1:
    if st.button("☀️ Light", key="light_theme", help="Switch to light theme"):
        st.session_state.theme = 'light'
        st.rerun()

with theme_col2:
    if st.button("🌙 Dark", key="dark_theme", help="Switch to dark theme"):
        st.session_state.theme = 'dark'
        st.rerun()

# Show current theme
current_theme_emoji = "☀️" if st.session_state.theme == "light" else "🌙"
st.sidebar.markdown(f"**Current Theme:** {current_theme_emoji} {st.session_state.theme.title()}")

# Background image selector
st.sidebar.markdown("---")
st.sidebar.markdown("### 🖼️ Background")
available_backgrounds = get_available_backgrounds()

# Debug info
st.sidebar.write(f"Available backgrounds: {len(available_backgrounds)}")
if st.session_state.background_image:
    st.sidebar.write(f"Current: {os.path.basename(st.session_state.background_image)}")
else:
    st.sidebar.write("Current: None")

# Get background options and find current selection
bg_options = list(available_backgrounds.keys())
current_selection = "None"  # Default

# Find which option matches current background
for option_name, option_path in available_backgrounds.items():
    if option_path == st.session_state.background_image:
        current_selection = option_name
        break

# Get index safely
try:
    current_index = bg_options.index(current_selection)
except ValueError:
    current_index = 0  # Default to first option if not found

selected_bg = st.sidebar.selectbox(
    "Choose Background:",
    bg_options,
    index=current_index,
    key="bg_selector"
)

# Update background if changed
new_bg_path = available_backgrounds[selected_bg]
if new_bg_path != st.session_state.background_image:
    st.session_state.background_image = new_bg_path
    st.sidebar.success(f"Background changed to: {selected_bg}")
    st.rerun()

# User location for weather
st.sidebar.markdown("---")
location = st.sidebar.text_input("📍 Enter your location:", placeholder="e.g., Mumbai, Maharashtra")

def display_typing_effect(text: str, container):
    """Display text with typing animation effect"""
    placeholder = container.empty()
    displayed_text = ""
    
    for char in text:
        displayed_text += char
        placeholder.markdown(f'<div class="typing-animation">{displayed_text}</div>', unsafe_allow_html=True)
        time.sleep(0.02)  # Adjust speed here
    
    # Final display without cursor
    placeholder.markdown(displayed_text)

# Main content based on page selection
if page == "🏠 Home":
    st.title("🌾 Welcome to Agricultural AI Advisor")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #E8F5E8 0%, #C8E6C8 100%); border-radius: 15px; margin: 20px 0; border: 1px solid #4CAF50;">
        <h2 style="color: #2E7D32; margin-bottom: 10px;">🚀 Empowering Indian Farmers with AI</h2>
        <p style="font-size: 18px; color: #1B5E20;">
        Your comprehensive AI-powered agricultural assistant providing expert guidance on:
        </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Feature cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="agent-response">
        <h4>🔬 Disease Detection</h4>
        <p>Upload plant images for instant disease identification using advanced AI models</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="weather-info">
        <h4>🌤️ Weather Insights</h4>
        <p>Get real-time weather data and agricultural forecasts for your location</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="disease-result">
        <h4>💰 Financial Guidance</h4>
        <p>Access loans, subsidies, and market price information</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="agent-response">
        <h4>🤖 AI Chat Assistant</h4>
        <p>Voice and text support in multiple Indian languages</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Profile completion prompt for new users
    if not st.session_state.user_profile.get('name') or st.session_state.user_profile.get('name') == '':
        st.markdown("---")
        st.markdown("""
        <div style="background: linear-gradient(135deg, #ff6b6b, #ffa500); padding: 25px; border-radius: 15px; margin: 20px 0; color: white; text-align: center;">
            <h3>🎯 New User? Let's Get Started!</h3>
            <p style="font-size: 16px; margin: 10px 0;">
                For the best personalized experience, please complete your profile first!
            </p>
            <p style="font-size: 14px; opacity: 0.9;">
                This helps us provide tailored crop recommendations, weather alerts, and government schemes specific to your location and farming needs.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Complete My Profile Now", type="primary", use_container_width=True):
                st.session_state.page = "👤 User Profile"
                st.rerun()
    
    # Quick stats
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🌱 Crop Diseases Detected", "38+", "+5")
    with col2:
        st.metric("🗣️ Languages Supported", "5+", "+2")
    with col3:
        st.metric("🌍 Locations Covered", "All India", "✓")
    with col4:
        st.metric("⚡ Response Time", "< 3s", "-1s")

elif page == "🤖 AI Chat Assistant":
    st.title("🤖 AI Chat Assistant")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Ask me anything about agriculture!")
        
        # Input method selection with navigation routing
        input_method = st.radio("Choose input method:", ["💬 Text", "🎤 Voice", "📷 Image + Text", "📄 PDF Documents"])
        
        # Navigation routing for Image + Text to OCR service
        if input_method == "📷 Image + Text":
            st.info("🔄 For OCR text extraction, redirecting to OCR service...")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🚀 Go to OCR Processing", type="primary"):
                    st.session_state.page = "📷 OCR Text Extraction"
                    st.rerun()
            with col2:
                if st.button("🔬 Go to Disease Detection", type="secondary"):
                    st.session_state.page = "🔬 Disease Detection"
                    st.rerun()
        
        user_input = ""
        uploaded_image = None
        
        if input_method == "💬 Text":
            user_input = st.text_area("Your question:", placeholder="e.g., When should I plant tomatoes in Mumbai?", height=100)
        
        elif input_method == "🎤 Voice":
            st.markdown("🎤 **Voice Input** (Click record and speak)")
            
            # Audio recorder component
            audio_bytes = st.audio_input("Record your question")
            
            if audio_bytes:
                try:
                    # Process voice input
                    with st.spinner("🔄 Processing voice..."):
                        user_input = services['voice_processor'].speech_to_text(audio_bytes)
                        if user_input:
                            st.success(f"🎤 Detected: {user_input}")
                except Exception as e:
                    st.error(f"Voice processing error: {str(e)}")
        
        elif input_method == "📷 Image + Text":
            uploaded_image = st.file_uploader("Upload plant image:", type=['jpg', 'jpeg', 'png'])
            
        elif input_method == "📄 PDF Documents":
            st.markdown("### 📄 PDF Document Q&A")
            
            # PDF upload section
            uploaded_pdf = st.file_uploader("Upload agricultural PDF document:", type=['pdf'])
            
            if uploaded_pdf is not None:
                # Initialize vector DB handler
                from utils.vector_db_handler import VectorDBHandler
                vector_db = VectorDBHandler()
                
                # Check if this PDF is already processed
                doc_stats = vector_db.get_document_stats()
                pdf_processed = uploaded_pdf.name in doc_stats['filenames']
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    if st.button("📤 Process PDF", type="primary"):
                        with st.spinner("📄 Processing PDF document..."):
                            try:
                                # Debug info
                                st.info(f"📁 Processing: {uploaded_pdf.name} ({uploaded_pdf.size} bytes)")
                                
                                # Extract text from PDF
                                pdf_content = vector_db.extract_text_from_pdf(uploaded_pdf)
                                
                                if pdf_content and len(pdf_content.strip()) > 0:
                                    st.info(f"📄 Extracted {len(pdf_content)} characters of text")
                                    
                                    # Show preview of extracted text
                                    with st.expander("📄 Preview extracted text"):
                                        st.text(pdf_content[:500] + "..." if len(pdf_content) > 500 else pdf_content)
                                    
                                    # Add to vector database
                                    chunks = vector_db.chunk_text(pdf_content)
                                    st.info(f"🧩 Creating {len(chunks)} vector chunks...")
                                    
                                    success = vector_db.add_document(
                                        filename=uploaded_pdf.name,
                                        content=pdf_content,
                                        metadata={
                                            "file_size": uploaded_pdf.size,
                                            "processed_at": str(datetime.now()),
                                            "total_chars": len(pdf_content),
                                            "total_chunks": len(chunks)
                                        }
                                    )
                                    
                                    if success:
                                        st.success(f"✅ Successfully processed '{uploaded_pdf.name}'")
                                        st.info(f"📊 Document split into {len(chunks)} searchable chunks")
                                        
                                        # Show final stats
                                        stats = vector_db.get_document_stats()
                                        st.success(f"📊 Database now contains {stats['total_chunks']} chunks from {stats['unique_documents']} documents")
                                    else:
                                        st.error("❌ Failed to add document to vector database")
                                else:
                                    st.error("⚠️ No text content extracted from PDF. The PDF might be:")
                                    st.markdown("- Scanned images (not text-selectable)\n- Password protected\n- Corrupted or empty")
                                    
                            except Exception as e:
                                st.error(f"❌ Error processing PDF: {str(e)}")
                                st.info("💡 Try a different PDF file or check if the file is readable")
                
                with col2:
                    if pdf_processed and st.button("🗑️ Remove PDF"):
                        if vector_db.delete_document(uploaded_pdf.name):
                            st.success("✅ PDF removed from database")
                            st.rerun()
                        else:
                            st.error("❌ Failed to remove PDF")
                
                # Show document statistics
                if doc_stats['unique_documents'] > 0:
                    with st.expander("📊 Document Database Stats"):
                        st.write(f"📄 Total documents: {doc_stats['unique_documents']}")
                        st.write(f"🧩 Total chunks: {doc_stats['total_chunks']}")
                        st.write("📁 Files:", ", ".join(doc_stats['filenames'][:5]))
                        if len(doc_stats['filenames']) > 5:
                            st.write(f"... and {len(doc_stats['filenames']) - 5} more")
                        
                        if st.button("🗑️ Clear All Documents"):
                            if vector_db.clear_all_documents():
                                st.success("✅ All documents cleared")
                                st.rerun()
            
            # OCR processing if image is uploaded
            extracted_text = ""
            if uploaded_image is not None:
                st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)
                
                # OCR text extraction
                with st.spinner("🔍 Extracting text from image..."):
                    try:
                        ocr_result = services['ocr_processor'].extract_text_from_image(uploaded_image)
                        
                        if ocr_result['success'] and ocr_result['cleaned_text']:
                            extracted_text = ocr_result['cleaned_text']
                            
                            # Display OCR results in an expander
                            with st.expander("📄 Extracted Text from Image", expanded=True):
                                st.markdown(services['ocr_processor'].format_ocr_results(ocr_result))
                                
                                # Option to use extracted text
                                if st.button("📝 Use Extracted Text in Query", key="use_ocr_text"):
                                    st.session_state.ocr_text = extracted_text
                                    st.rerun()
                        else:
                            st.info("📄 No text detected in the image. You can still ask questions about the image content.")
                            
                    except Exception as e:
                        st.warning(f"OCR processing failed: {str(e)}. You can still analyze the image content.")
            
            # Text input area with OCR integration
            default_text = st.session_state.get('ocr_text', '')
            user_input = st.text_area(
                "Additional question about the image:", 
                value=default_text,
                placeholder="e.g., What's wrong with my plant? Or ask about the extracted text above.",
                height=120,
                help="💡 Tip: If text was extracted from your image, you can ask questions about it or modify the extracted text above."
            )
            
            # Clear OCR text button
            if st.session_state.get('ocr_text'):
                if st.button("🗑️ Clear Extracted Text", key="clear_ocr_text"):
                    st.session_state.ocr_text = ""
                    st.rerun()
        
        # Submit button
        if st.button("🚀 Get AI Response", type="primary"):
            if user_input or uploaded_image:
                with st.spinner("🤖 AI is thinking..."):
                    try:
                        # CRITICAL: Force PDF context integration
                        pdf_context = ""
                        pdf_sources = []
                        
                        try:
                            from utils.vector_db_handler import VectorDBHandler
                            vector_db = VectorDBHandler()
                            
                            # Get actual database content
                            stats = vector_db.get_document_stats()
                            
                            if stats['total_chunks'] > 0:
                                # Force search for any PDF content
                                search_results = vector_db.search_documents(user_input, top_k=5)
                                
                                if search_results:
                                    # Build comprehensive context
                                    pdf_context = f"""

🚨 CRITICAL: PDF CONTENT AVAILABLE 🚨
The following content is extracted from uploaded PDF documents in the vector database:
Database contains {stats['total_chunks']} chunks from {stats['unique_documents']} documents.
Files: {', '.join(stats['filenames'])}

--- EXTRACTED PDF CONTENT ---
"""
                                    
                                    for i, result in enumerate(search_results, 1):
                                        filename = result['metadata'].get('filename', 'PDF Document')
                                        content = result['content'].strip()
                                        score = result['score']
                                        
                                        pdf_context += f"""
📄 SOURCE {i}: FROM '{filename}' (RELEVANCE: {score:.2f}/1.0)
CONTENT: {content}
{'─' * 60}
"""
                                        
                                        pdf_sources.append({
                                            'filename': filename,
                                            'content': content[:150] + "..." if len(content) > 150 else content,
                                            'score': score
                                        })
                                    
                                    # Force display of PDF usage
                                    st.success(f"✅ USING PDF CONTENT: {len(search_results)} sources from {stats['unique_documents']} documents")
                                    
                                    # Debug: Show what's being sent
                                    with st.expander("🔍 Debug: PDF Content Being Used", expanded=True):
                                        st.json({
                                            "total_chunks": stats['total_chunks'],
                                            "documents": stats['filenames'],
                                            "sources_found": len(search_results),
                                            "first_source": search_results[0]['content'][:200] if search_results else "None"
                                        })
                                else:
                                    st.warning("⚠️ PDF database exists but no relevant content found")
                            else:
                                st.error("❌ No PDF documents in vector database - upload and process PDF first")
                                
                        except Exception as e:
                            st.error(f"❌ PDF Database Error: {str(e)}")
                            st.info("💡 Ensure PDF is uploaded and processed using the 'Process PDF' button above")
                        
                        # Combine user input with PDF context
                        enhanced_query = user_input
                        if pdf_context:
                            enhanced_query = f"{user_input}\n\n{pdf_context}"
                            
                        # Debug: Show what we're sending to AI
                        if pdf_context:
                            st.info(f"📄 Including {len(pdf_sources)} PDF sources in query")
                            with st.expander("📊 Debug: Enhanced Query", expanded=False):
                                st.text(f"Original: {user_input}")
                                st.text(f"Enhanced: {enhanced_query[:500]}...")
                        
                        # Determine query type and route to appropriate agent
                        response = services['agent_team'].process_query(
                            text=enhanced_query,
                            image=uploaded_image,
                            location=location,
                            language=st.session_state.language,
                            user_profile=st.session_state.user_profile
                        )
                        
                        # Determine output language based on detected language of input text
                        detected_code = None
                        try:
                            # Prefer language auto-detected by STT for voice inputs
                            detected_code = services['voice_processor'].last_detected_lang_code
                            if not detected_code and user_input:
                                detected_code = services['voice_processor'].detect_language(user_input)
                        except Exception:
                            detected_code = None

                        # Optional translation to detected input language
                        response_text = response.get('text', 'No response available')
                        try:
                            if detected_code and services.get('translator'):
                                # Use deep-translator API (source='auto', target=detected_code)
                                translated = services['translator'](source='auto', target=detected_code).translate(response_text)
                                if translated:
                                    response_text = translated
                        except Exception:
                            pass

                        # Display response with typing effect
                        response_container = st.container()
                        with response_container:
                            st.markdown('<div class="agent-response">', unsafe_allow_html=True)
                            
                            # Create placeholder for typing effect
                            typing_placeholder = st.empty()
                            display_typing_effect(response_text, typing_placeholder)
                            
                            # Display PDF sources if available
                            if pdf_sources:
                                st.markdown("### 📄 Sources from PDF Documents")
                                for source in pdf_sources:
                                    with st.expander(f"📄 {source['filename']}"):
                                        st.text(source['content'])
                            
                            # Audio response: synthesize TTS in selected language
                            try:
                                language_map = {
                                    "English": "en",
                                    "हिन्दी (Hindi)": "hi",
                                    "தமிழ் (Tamil)": "ta",
                                    "తెలుగు (Telugu)": "te",
                                    "বাংলা (Bengali)": "bn"
                                }
                                # Prefer detected language; fallback to UI selection
                                # Normalize detected code through voice processor for gTTS compatibility
                                tts_lang = services['voice_processor'].normalize_lang_code(
                                    detected_code or language_map.get(st.session_state.language, "en")
                                )
                                audio_bytes = services['voice_processor'].create_voice_response(
                                    response_text, language=tts_lang
                                )
                                if audio_bytes:
                                    st.audio(audio_bytes, format="audio/mp3")
                                    st.markdown("🔊 Click play to hear the response")
                                elif response.get('audio_url'):
                                    st.audio(response['audio_url'])
                                    st.markdown("🔊 Click play to hear the response")
                            except Exception as _:
                                pass
                            
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Update chat history with persistent storage
                        auth_manager.save_chat_message(
                            user_input,
                            response.get('text', ''),
                            'chat',
                            {
                                'location': location, 
                                'language': st.session_state.language,
                                'timestamp': time.time(),
                                'image': uploaded_image is not None
                            }
                        )
                        
                    except Exception as e:
                        st.error(f"Error processing request: {str(e)}")
            else:
                st.warning("Please provide input (text, voice, image, or PDF query)")
    
    with col2:
        st.markdown("### 💡 Suggested Questions")
        
        suggestions = [
            "What crops grow best in monsoon season?",
            "How to identify and treat crop diseases?",
            "Current market prices for wheat",
            "Government subsidies available for farmers",
            "Best fertilizers for rice cultivation",
            "Weather forecast for next week",
            "Soil health management tips",
            "Organic farming techniques"
        ]
        
        for suggestion in suggestions:
            if st.button(suggestion, key=f"suggest_{suggestion[:20]}"):
                # Set the suggestion as input and trigger processing
                st.session_state.suggested_query = suggestion
                st.rerun()
        
        # Check if there's a suggested query to process
        if hasattr(st.session_state, 'suggested_query'):
            user_input = st.session_state.suggested_query
            del st.session_state.suggested_query
            
            with st.spinner("🤖 Processing suggested query..."):
                try:
                    response = services['agent_team'].process_query(
                        text=user_input,
                        location=location,
                        language=st.session_state.language,
                        user_profile=st.session_state.user_profile
                    )
                    
                    # Detect language from suggested text and translate response to match
                    detected_code = None
                    try:
                        detected_code = services['voice_processor'].last_detected_lang_code or (
                            services['voice_processor'].detect_language(user_input) if user_input else None
                        )
                    except Exception:
                        pass
                    
                    try:
                        if detected_code and services.get('translator'):
                            translated = services['translator'](source='auto', target=detected_code).translate(response.get('text', 'No response available'))
                            if translated:
                                response_text = translated
                            else:
                                response_text = response.get('text', 'No response available')
                        else:
                            response_text = response.get('text', 'No response available')
                    except Exception:
                        response_text = response.get('text', 'No response available')
                    
                    st.markdown('<div class="agent-response">', unsafe_allow_html=True)
                    st.markdown(response_text)
                    
                    # Optional TTS for suggested queries as well
                    try:
                        language_map = {
                            "English": "en",
                            "हिन्दी (Hindi)": "hi",
                            "தமிழ் (Tamil)": "ta",
                            "తెలుగు (Telugu)": "te",
                            "বাংলা (Bengali)": "bn",
                        }
                        tts_lang = services['voice_processor'].normalize_lang_code(
                            detected_code or language_map.get(st.session_state.language, "en")
                        )
                        audio_bytes = services['voice_processor'].create_voice_response(
                            response_text, language=tts_lang
                        )
                        if audio_bytes:
                            st.audio(audio_bytes, format="audio/mp3")
                    except Exception:
                        pass
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Error processing suggested query: {str(e)}")
        
        # Chat history
        if st.session_state.chat_history:
            st.markdown("### 📜 Recent Conversations")
            for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):
                with st.expander(f"💬 {chat['user'][:30]}..."):
                    st.write(f"**You:** {chat['user']}")
                    st.write(f"**AI:** {chat['assistant']}")

elif page == "🔬 Disease Detection":
    st.title("🔬 Plant Disease Detection")
    
    st.markdown("""
    <div class="disease-result">
    <h4>📱 Upload a photo of your plant to get instant disease diagnosis</h4>
    <p>Our AI model can identify 38+ different plant diseases and conditions</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose a plant image...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear image of the affected plant part"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Predict disease
            if st.button("🔍 Analyze Disease", type="primary"):
                with st.spinner("🔬 Analyzing image..."):
                    try:
                        # Use enhanced Docker service communication
                        result = services['disease_detector'].predict_disease(uploaded_file)
                        
                        # Check if result contains error
                        if 'error' in result:
                            st.error(f"Disease detection failed: {result['error']}")
                        elif not all(key in result for key in ['disease_name', 'confidence', 'category']):
                            # Validate required keys in result
                            required_keys = ['disease_name', 'confidence', 'category']
                            missing_keys = [key for key in required_keys if key not in result]
                            st.error(f"Invalid response from disease service. Missing keys: {missing_keys}")
                            st.error(f"Received response: {result}")
                        else:
                            with col2:
                                st.markdown(f"""
                                <div class="disease-result">
                                <h3>🎯 Analysis Results</h3>
                                <h4>Detected Condition:</h4>
                                <h2 style="color: #FF5722;">{result['disease_name']}</h2>
                                <p><strong>Confidence:</strong> {result['confidence']:.1f}%</p>
                                <p><strong>Category:</strong> {result['category']}</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Get treatment recommendations
                            if result['disease_name'] != 'healthy':
                                treatment_response = services['agent_team'].get_treatment_advice(
                                    result['disease_name'],
                                    st.session_state.language
                                )
                                
                                st.markdown("### 💊 Treatment Recommendations")
                                st.markdown(f'<div class="agent-response">{treatment_response}</div>', 
                                          unsafe_allow_html=True)
                            else:
                                st.success("🎉 Your plant appears to be healthy!")
                                st.balloons()
                    
                    except Exception as e:
                        st.error(f"Error analyzing image: {str(e)}")
                        st.error("Please try again or check if the disease service is running properly.")
    
    with col2:
        if not uploaded_file:
            st.markdown("""
            ### 📋 Supported Plant Types:
            - 🍎 Apple
            - 🫐 Blueberry  
            - 🍒 Cherry
            - 🌽 Corn (Maize)
            - 🍇 Grape
            - 🍊 Orange
            - 🍑 Peach
            - 🌶️ Pepper (Bell)
            - 🥔 Potato
            - 🍓 Strawberry
            - 🍅 Tomato
            - And more...
            
            ### 💡 Tips for Best Results:
            - Take clear, well-lit photos
            - Focus on affected plant parts
            - Avoid blurry or dark images
            - Include diseased leaves/fruits in frame
            """)

elif page == "📷 OCR Text Extraction":
    st.title("📷 OCR Text Extraction Service")
    
    st.markdown("""
    <div class="agent-response">
    <h4>🔍 Extract text from images using advanced OCR technology</h4>
    <p>Upload images containing text (fertilizer packages, documents, labels) to extract readable text</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_image = st.file_uploader(
            "Choose an image with text...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear image containing text to extract"
        )
        
        if uploaded_image is not None:
            # Display uploaded image
            st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)
            
            # OCR processing
            with st.spinner("🔍 Extracting text from image..."):
                try:
                    # Use OCR service to extract text
                    ocr_result = services['ocr_processor'].extract_text_from_image(uploaded_image)
                    
                    if ocr_result['success'] and ocr_result['cleaned_text']:
                        extracted_text = ocr_result['cleaned_text']
                        
                        # Display OCR results
                        st.success("✅ Text extraction successful!")
                        
                        # Show extracted text in expandable section
                        with st.expander("📄 Extracted Text", expanded=True):
                            st.text_area(
                                "Extracted Text:",
                                value=extracted_text,
                                height=200,
                                help="You can copy this text or edit it as needed"
                            )
                        
                        # Show OCR details
                        with st.expander("🔧 OCR Processing Details"):
                            st.markdown(services['ocr_processor'].format_ocr_results(ocr_result))
                        
                        # Option to use in chat
                        if st.button("💬 Use Text in AI Chat", type="primary"):
                            st.session_state.ocr_extracted_text = extracted_text
                            st.session_state.page = "🤖 AI Chat Assistant"
                            st.rerun()
                            
                    else:
                        st.warning("⚠️ No text detected in the image. Please try with a clearer image containing visible text.")
                        
                except Exception as e:
                    st.error(f"❌ OCR processing failed: {str(e)}")
                    st.info("💡 Try uploading a clearer image or check if the OCR service is running.")
    
    with col2:
        if not uploaded_image:
            st.markdown("""
            ### 📋 Supported Text Types:
            - 📦 Fertilizer package labels
            - 📄 Agricultural documents
            - 🏷️ Product labels and tags
            - 📰 Newspaper articles
            - 📋 Forms and certificates
            - 🔤 Any printed or handwritten text
            
            ### 💡 Tips for Best OCR Results:
            - Use high-resolution images
            - Ensure good lighting
            - Keep text horizontal and straight
            - Avoid shadows and glare
            - Clean, uncrumpled documents work best
            
            ### 🔧 OCR Engines Used:
            - **TrOCR**: Microsoft's Transformer OCR
            - **EasyOCR**: Multi-language support
            - **Tesseract**: Traditional OCR fallback
            """)

elif page == "🌤️ Weather Information":
    st.title("🌤️ Weather Information & Agricultural Insights")
    
    if location:
        try:
            with st.spinner("🌡️ Fetching weather data..."):
                weather_data = services['weather_service'].get_weather_data(location)
                
                if weather_data:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown(f"""
                        <div class="weather-info">
                        <h3>🌡️ Current Weather</h3>
                        <h2>{weather_data['current']['temp_c']}°C</h2>
                        <p><strong>Condition:</strong> {weather_data['current']['condition']['text']}</p>
                        <p><strong>Humidity:</strong> {weather_data['current']['humidity']}%</p>
                        <p><strong>Wind:</strong> {weather_data['current']['wind_kph']} km/h</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class="weather-info">
                        <h3>📊 Agricultural Conditions</h3>
                        <p><strong>UV Index:</strong> {weather_data['current']['uv']}</p>
                        <p><strong>Visibility:</strong> {weather_data['current']['vis_km']} km</p>
                        <p><strong>Pressure:</strong> {weather_data['current']['pressure_mb']} mb</p>
                        <p><strong>Feels like:</strong> {weather_data['current']['feelslike_c']}°C</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        # Agricultural recommendations based on weather
                        agricultural_advice = services['agent_team'].get_weather_advice(
                            weather_data, st.session_state.language
                        )
                        
                        st.markdown(f"""
                        <div class="agent-response">
                        <h3>🌾 Agricultural Recommendations</h3>
                        {agricultural_advice}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # 7-day forecast
                    if 'forecast' in weather_data:
                        st.markdown("### 📅 7-Day Forecast")
                        
                        forecast_cols = st.columns(7)
                        for i, day in enumerate(weather_data['forecast']['forecastday'][:7]):
                            with forecast_cols[i]:
                                st.markdown(f"""
                                <div style="text-align: center; padding: 10px; background: rgba(255,255,255,0.8); border-radius: 10px; margin: 5px;">
                                <h5>{day['date']}</h5>
                                <p>🌡️ {day['day']['maxtemp_c']}°/{day['day']['mintemp_c']}°C</p>
                                <p>💧 {day['day']['daily_chance_of_rain']}% rain</p>
                                <p>{day['day']['condition']['text']}</p>
                                </div>
                                """, unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"Error fetching weather data: {str(e)}")
    else:
        st.info("👆 Please enter your location in the sidebar to get weather information")
        
        # Show sample weather info for major agricultural regions
        st.markdown("### 🌍 Major Agricultural Regions in India")
        
        regions = {
            "Punjab": "Major wheat and rice producing state",
            "Uttar Pradesh": "Largest producer of food grains",
            "Maharashtra": "Leading in sugarcane and cotton",
            "West Bengal": "Major rice and jute producer",
            "Andhra Pradesh": "Leading in rice and cotton",
            "Gujarat": "Major cotton and groundnut producer"
        }
        
        for region, description in regions.items():
            st.markdown(f"**{region}:** {description}")

elif page == "💰 Financial Advisor":
    st.title("💰 Agricultural Financial Advisor")
    
    st.markdown("""
    <div class="agent-response">
    <h3>💳 Get information about loans, subsidies, and market prices</h3>
    <p>Our AI agent searches for the latest financial schemes and market data</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Financial query categories
    financial_category = st.selectbox(
        "Select financial topic:",
        ["🏦 Loans & Credit", "💰 Government Subsidies", "📈 Market Prices", "📊 Financial Planning", "🌾 Crop Insurance"]
    )
    
    query = st.text_area(
        "Ask your financial question:",
        placeholder="e.g., What are the current interest rates for agricultural loans?",
        height=100
    )
    
    if st.button("💼 Get Financial Advice", type="primary"):
        if query:
            with st.spinner("💰 Searching for financial information..."):
                try:
                    financial_response = services['agent_team'].get_financial_advice(
                        query=query, 
                        category=financial_category, 
                        location=location, 
                        language=st.session_state.language
                    )
                    
                    # Ensure we have a valid response
                    if financial_response and isinstance(financial_response, dict):
                        advice_text = financial_response.get('advice', 'No advice available')
                    else:
                        advice_text = 'Unable to generate financial advice at this time. Please try again.'
                    
                    st.markdown(f"""
                    <div class="agent-response">
                    <h3>💼 Financial Advice</h3>
                    {advice_text}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display sources if available
                    if financial_response.get('sources'):
                        st.markdown("### 📚 Sources:")
                        for source in financial_response['sources']:
                            st.markdown(f"- [{source['title']}]({source['url']})")
                
                except Exception as e:
                    st.error(f"Error getting financial advice: {str(e)}")
        else:
            st.warning("Please enter your financial question")
    
    # Quick access to common financial topics
    st.markdown("### 🔗 Quick Access")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🏦 PM-KISAN Scheme"):
            st.info("The PM-KISAN scheme provides ₹6,000 per year to all landholding farmers in three equal installments.")
    
    with col2:
        if st.button("💳 Kisan Credit Card"):
            st.info("KCC provides flexible credit facility with interest rates as low as 4% per annum with government subsidy.")
    
    with col3:
        if st.button("📊 Market Rates"):
            with st.spinner("Fetching market rates..."):
                try:
                    market_response = services['agent_team'].get_financial_advice(
                        query="Current market rates for major crops in India", 
                        category="📈 Market Prices", 
                        location=location, 
                        language=st.session_state.language
                    )
                    
                    # Ensure we have a valid response
                    if market_response and isinstance(market_response, dict):
                        market_advice = market_response.get('advice', 'No market data available')
                    else:
                        market_advice = 'Unable to fetch market rates at this time. Please try again.'
                    
                    st.markdown(f"""
                    <div class="agent-response">
                    {market_advice}
                    </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error fetching market rates: {str(e)}")

elif page == "📚 Knowledge Base":
    st.title("📚 Agricultural Knowledge Base")
    
    st.markdown("""
    <div class="agent-response">
    <h3>🔍 Search our comprehensive agricultural database</h3>
    <p>Find information about crops, farming techniques, best practices, and more</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Search functionality
    search_query = st.text_input("🔍 Search knowledge base:", placeholder="e.g., organic farming techniques")
    
    if search_query:
        try:
            search_results = services['vector_store'].search_knowledge(search_query, max_results=5)
            
            if search_results:
                st.markdown("### 📖 Search Results")
                
                for i, result in enumerate(search_results):
                    with st.expander(f"📄 {result['title']}"):
                        st.markdown(result['content'])
                        if result.get('category'):
                            st.markdown(f"**Category:** {result['category']}")
                        if result.get('keywords'):
                            st.markdown(f"**Keywords:** {', '.join(result['keywords'])}")
            else:
                st.info("No results found. Try different keywords.")
        
        except Exception as e:
            st.error(f"Search error: {str(e)}")
    
    # Browse by categories
    st.markdown("### 📂 Browse by Category")
    
    categories = [
        "🌱 Crop Cultivation",
        "🐛 Pest Management", 
        "💧 Irrigation",
        "🌱 Soil Health",
        "🌾 Harvest Techniques",
        "💰 Financial Management",
        "🌿 Organic Farming",
        "🌡️ Climate Adaptation"
    ]
    
    selected_category = st.selectbox("Select category:", categories)
    
    if st.button("📚 Browse Category"):
        try:
            category_content = services['vector_store'].get_category_information(selected_category)
            
            if category_content:
                for content in category_content:
                    with st.expander(content['title']):
                        st.markdown(content['content'])
                        if content.get('keywords'):
                            st.markdown(f"**Keywords:** {', '.join(content['keywords'])}")
            else:
                st.info(f"No content found for category: {selected_category}")
        
        except Exception as e:
            st.error(f"Error browsing category: {str(e)}")

elif page == "🏛️ Government Schemes":
    st.title("🏛️ Government Schemes & Policies")
    st.markdown("Discover government schemes and policies relevant to your farming profile")
    
    # Get user profile for scheme recommendations
    user_profile = st.session_state.user_profile
    
    # Tabs for different scheme features
    tab1, tab2, tab3 = st.tabs(["🎯 Personalized Schemes", "🔍 Search Schemes", "⚙️ Settings"])
    
    with tab1:
        st.markdown("### 🎯 Schemes Recommended for Your Profile")
        
        if not user_profile.get('name'):
            st.info("💡 Complete your user profile to get personalized scheme recommendations!")
            if st.button("Go to User Profile"):
                st.session_state.page = "👤 User Profile"
                st.rerun()
        else:
            # Display user profile summary
            with st.expander("📋 Your Profile Summary"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Name:** {user_profile.get('name', 'Not set')}")
                    st.write(f"**Location:** {user_profile.get('location', 'Not set')}")
                    st.write(f"**Farm Size:** {user_profile.get('farm_size', 0)} acres")
                with col2:
                    st.write(f"**Primary Crops:** {', '.join(user_profile.get('primary_crops', []))}")
                    st.write(f"**Farming Type:** {user_profile.get('farming_type', 'Not set')}")
                    st.write(f"**Experience:** {user_profile.get('experience', 0)} years")
            
            if st.button("🔄 Get Personalized Schemes", type="primary"):
                with st.spinner("Analyzing your profile and fetching relevant schemes..."):
                    try:
                        scheme_data = services['agent_team'].get_government_schemes(user_profile)
                        
                        if scheme_data['total_found'] > 0:
                            st.success(f"Found {scheme_data['total_found']} relevant schemes for your profile!")
                            
                            # Display AI analysis
                            st.markdown("### 🤖 AI Analysis & Recommendations")
                            st.markdown(scheme_data['analysis'])
                            
                            # Display individual schemes
                            st.markdown("### 📋 Scheme Details")
                            for i, scheme in enumerate(scheme_data['schemes'][:5], 1):
                                with st.expander(f"{i}. {scheme.get('title', 'Unknown Scheme')} (Score: {scheme.get('relevance_score', 0)})"):
                                    st.markdown(f"**Description:** {scheme.get('description', 'No description available')}")
                                    if scheme.get('source_url'):
                                        st.markdown(f"**Source:** [View Details]({scheme['source_url']})")
                                    if scheme.get('keywords'):
                                        st.markdown(f"**Keywords:** {', '.join(scheme['keywords'][:10])}")
                                    st.markdown(f"**Scraped:** {scheme.get('scraped_at', 'Unknown')}")
                        else:
                            st.warning("No schemes found matching your profile. Try updating your profile or searching manually.")
                            
                    except Exception as e:
                        st.error(f"Error fetching schemes: {str(e)}")
    
    with tab2:
        st.markdown("### 🔍 Search Government Schemes")
        
        search_query = st.text_input(
            "Enter keywords to search for schemes:",
            placeholder="e.g., cotton farming subsidy, irrigation loan, organic certification"
        )
        
        if st.button("🔍 Search Schemes") and search_query:
            with st.spinner("Searching government schemes..."):
                try:
                    scheme_data = services['agent_team'].get_government_schemes(user_profile, search_query)
                    
                    if scheme_data['total_found'] > 0:
                        st.success(f"Found {scheme_data['total_found']} schemes matching '{search_query}'")
                        
                        # Display AI analysis
                        st.markdown("### 🤖 Search Analysis")
                        st.markdown(scheme_data['analysis'])
                        
                        # Display search results
                        st.markdown("### 📋 Search Results")
                        for i, scheme in enumerate(scheme_data['schemes'], 1):
                            with st.expander(f"{i}. {scheme.get('title', 'Unknown Scheme')} (Match Score: {scheme.get('match_score', 0)})"):
                                st.markdown(f"**Description:** {scheme.get('description', 'No description available')}")
                                if scheme.get('source_url'):
                                    st.markdown(f"**Source:** [View Details]({scheme['source_url']})")
                                if scheme.get('keywords'):
                                    st.markdown(f"**Keywords:** {', '.join(scheme['keywords'][:10])}")
                    else:
                        st.warning(f"No schemes found for '{search_query}'. Try different keywords.")
                        
                except Exception as e:
                    st.error(f"Error searching schemes: {str(e)}")
    
    with tab3:
        st.markdown("### ⚙️ Scheme Database & Scheduling Settings")
        
        # Storage Statistics
        try:
            storage_stats = services['agent_team'].policy_agent.get_storage_statistics()
            scheduler_status = services['agent_team'].policy_agent.get_scheduler_status()
            
            st.markdown("#### 📊 Storage Statistics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Schemes", storage_stats.get('total_schemes', 0))
                st.metric("Active Schemes", storage_stats.get('active_schemes', 0))
            
            with col2:
                if storage_stats.get('last_update'):
                    st.write(f"**Last Updated:** {storage_stats['last_update'][:19]}")
                else:
                    st.write("**Last Updated:** Never")
                
                storage_type = storage_stats.get('storage_type', 'persistent')
                st.write(f"**Storage Type:** {storage_type.title()}")
            
            with col3:
                db_size = storage_stats.get('database_size', 0)
                if db_size > 0:
                    st.metric("Database Size", f"{db_size / 1024:.1f} KB")
                else:
                    st.metric("Database Size", "N/A")
            
            st.markdown("---")
            
            # Scheduler Status
            st.markdown("#### ⏰ Automatic Scraping Schedule")
            
            if scheduler_status.get('scheduler_available', True):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Scheduler Status:** {'🟢 Running' if scheduler_status.get('is_running', False) else '🔴 Stopped'}")
                    st.write(f"**Current Status:** {scheduler_status.get('current_status', 'Unknown').title()}")
                    st.write(f"**Schedule Time:** {scheduler_status.get('schedule_time', 'Not set')} daily")
                
                with col2:
                    if scheduler_status.get('next_scheduled_run'):
                        st.write(f"**Next Run:** {scheduler_status['next_scheduled_run'][:19]}")
                    else:
                        st.write("**Next Run:** Not scheduled")
                    
                    if scheduler_status.get('last_scheduled_run'):
                        st.write(f"**Last Scheduled:** {scheduler_status['last_scheduled_run'][:19]}")
                    else:
                        st.write("**Last Scheduled:** Never")
                
                # Schedule Update
                st.markdown("##### Update Schedule")
                schedule_col1, schedule_col2, schedule_col3 = st.columns([2, 2, 1])
                
                with schedule_col1:
                    new_hour = st.selectbox("Hour (24h format):", range(24), index=2)
                
                with schedule_col2:
                    new_minute = st.selectbox("Minute:", [0, 15, 30, 45], index=0)
                
                with schedule_col3:
                    if st.button("Update Schedule"):
                        try:
                            success = services['agent_team'].policy_agent.update_schedule(new_hour, new_minute)
                            if success:
                                st.success(f"✅ Schedule updated to {new_hour:02d}:{new_minute:02d}")
                                st.rerun()
                            else:
                                st.error("❌ Failed to update schedule")
                        except Exception as e:
                            st.error(f"Error updating schedule: {str(e)}")
            else:
                st.info("📝 Automatic scheduling not available (using memory cache)")
            
            st.markdown("---")
            
            # Manual Update Section
            st.markdown("#### 🔄 Manual Database Update")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write("Force an immediate update of the scheme database from government sources.")
                st.write("This will scrape the latest schemes and update the persistent storage.")
            
            with col2:
                if st.button("🔄 Force Update Now", type="primary"):
                    with st.spinner("Updating scheme database from mygov.in..."):
                        try:
                            success = services['agent_team'].force_update_schemes()
                            if success:
                                st.success("✅ Scheme database updated successfully!")
                                st.rerun()
                            else:
                                st.error("❌ Failed to update scheme database. Please try again later.")
                        except Exception as e:
                            st.error(f"Error updating schemes: {str(e)}")
            
            st.markdown("---")
            
            # Scraping History
            st.markdown("#### 📜 Recent Scraping History")
            
            try:
                history = services['agent_team'].policy_agent.get_scraping_history(5)
                
                if history:
                    for i, entry in enumerate(history):
                        with st.expander(f"Run {i+1}: {entry.get('scraping_started_at', 'Unknown')[:19]} ({entry.get('scraping_source', 'Unknown')})"):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write(f"**Status:** {entry.get('status', 'Unknown').title()}")
                                st.write(f"**Schemes Found:** {entry.get('schemes_found', 0)}")
                                st.write(f"**Schemes Added:** {entry.get('schemes_added', 0)}")
                            
                            with col2:
                                st.write(f"**Schemes Updated:** {entry.get('schemes_updated', 0)}")
                                st.write(f"**Source:** {entry.get('scraping_source', 'Unknown').title()}")
                                
                                if entry.get('error_message'):
                                    st.error(f"Error: {entry['error_message']}")
                else:
                    st.info("No scraping history available")
                    
            except Exception as e:
                st.warning(f"Could not load scraping history: {str(e)}")
            
            st.markdown("---")
            
            # Maintenance Section
            st.markdown("#### 🧹 Database Maintenance")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Clean Old Schemes**")
                days_old = st.number_input("Remove schemes older than (days):", min_value=30, max_value=365, value=90)
                
                if st.button("🧹 Clean Old Schemes"):
                    try:
                        cleaned = services['agent_team'].policy_agent.cleanup_old_schemes(days_old)
                        if cleaned > 0:
                            st.success(f"✅ Cleaned {cleaned} old schemes")
                        else:
                            st.info("No old schemes found to clean")
                    except Exception as e:
                        st.error(f"Error cleaning schemes: {str(e)}")
            
            with col2:
                st.write("**Storage Information**")
                st.info("💡 Schemes are automatically updated daily at the scheduled time. Manual updates can be triggered anytime using the 'Force Update Now' button.")
            
        except Exception as e:
            st.error(f"Error loading settings: {str(e)}")

elif page == "🗄️ Data Management":
    st.title("🗄️ Data Management")
    st.markdown("Manage stored data safely. Use confirmations to avoid accidental deletion.")

    # Vector Database Management
    st.markdown("---")
    st.subheader("📄 Vector Database (PDF Embeddings)")
    try:
        from utils.vector_db_handler import VectorDBHandler
        vector_db = VectorDBHandler()
        stats = vector_db.get_document_stats()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Documents", stats.get('unique_documents', 0))
        with col2:
            st.metric("Total Chunks", stats.get('total_chunks', 0))
        with col3:
            filenames = stats.get('filenames', [])
            st.write(f"Examples: {', '.join(filenames[:3])}{'…' if len(filenames) > 3 else ''}")

        confirm_vec = st.checkbox("I understand this will remove ALL PDF embeddings", key="confirm_clear_vector")
        if st.button("🧹 Clear All PDF Embeddings", key="btn_clear_vector"):
            if confirm_vec:
                try:
                    if vector_db.clear_all_documents():
                        st.success("✅ Cleared all PDF embeddings from vector DB")
                        st.rerun()
                    else:
                        st.error("❌ Failed to clear vector database")
                except Exception as e:
                    st.error(f"Error clearing vector DB: {str(e)}")
            else:
                st.warning("Please check the confirmation box before clearing.")
    except Exception as e:
        st.warning(f"Vector DB unavailable: {str(e)}")

    # Chat History Management
    st.markdown("---")
    st.subheader("💬 Chat History (SQLite)")

    dbm = auth_manager.db_manager

    colA, colB = st.columns(2)

    with colA:
        st.markdown("#### Current User")
        uid = auth_manager.get_user_id()
        if uid:
            confirm_user = st.checkbox("Confirm clear my entire chat history", key="confirm_clear_user_chat")
            if st.button("🗑️ Clear My Chat History", key="btn_clear_user_chat"):
                if confirm_user:
                    try:
                        deleted = dbm.clear_chat_history_for_user(uid)
                        st.success(f"✅ Deleted {deleted} messages from your history")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error clearing your chat history: {str(e)}")
                else:
                    st.warning("Please confirm before clearing your history.")
        else:
            st.info("Login to manage your stored chat history. Guest mode is session-only.")

        st.markdown("#### Prune History (All Users)")
        keep_n = st.number_input("Keep last N messages per user", min_value=0, max_value=1000, value=50, step=10)
        if st.button("✂️ Prune Chat History", key="btn_prune_chat"):
            try:
                deleted = dbm.prune_chat_history_keep_last_n_per_user(int(keep_n))
                st.success(f"✅ Pruned old messages across users. Deleted {deleted} rows.")
            except Exception as e:
                st.error(f"Error pruning chat history: {str(e)}")

    with colB:
        st.markdown("#### All Users")
        confirm_all = st.checkbox("Confirm clear ALL chat history for ALL users", key="confirm_clear_all_chat")
        if st.button("🧹 Clear ALL Chat History", key="btn_clear_all_chat"):
            if confirm_all:
                try:
                    deleted = dbm.clear_all_chat_history()
                    st.success(f"✅ Deleted {deleted} chat records from the database")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error clearing chat history: {str(e)}")
            else:
                st.warning("Please confirm before clearing ALL chat history.")

    st.info("Government scheme scraping and storage are separate and unaffected by these actions.")

elif page == "📢 Notifications":
    st.title("📢 Notification Settings & Alerts")
    
    if st.session_state.get('user_id'):
        user_id = st.session_state['user_id']
        
        # Create tabs for different notification sections
        tab1, tab2, tab3, tab4 = st.tabs(["⚙️ Settings", "📊 Statistics", "📜 History", "🧪 Test"])
        
        with tab1:
            st.markdown("### 🔔 Notification Preferences")
            
            # Get current preferences from user profile
            try:
                user_profile = auth_manager.get_user_profile(user_id)
                preferences = {
                    'whatsapp_number': user_profile.get('phone', '') if user_profile else '',
                    'email': user_profile.get('email', '') if user_profile else '',
                    'sms_enabled': True,
                    'whatsapp_enabled': True,
                    'email_enabled': True
                }
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 📧 Email Notifications")
                    email_enabled = st.checkbox("Enable Email Notifications", 
                                               value=preferences.get('email_notifications', True))
                    
                    email_address = st.text_input("Email Address", 
                                                 value=preferences.get('email_address', ''),
                                                 placeholder="your.email@example.com")
                    
                    st.markdown("#### 📱 WhatsApp Notifications")
                    whatsapp_enabled = st.checkbox("Enable WhatsApp Notifications", 
                                                  value=preferences.get('whatsapp_notifications', False))
                    
                    whatsapp_number = st.text_input("WhatsApp Number", 
                                                   value=preferences.get('whatsapp_number', ''),
                                                   placeholder="+91XXXXXXXXXX")
                
                with col2:
                    st.markdown("#### 🚨 Alert Types")
                    disease_alerts = st.checkbox("Disease Detection Alerts", 
                                                value=preferences.get('disease_alerts', True))
                    
                    disaster_alerts = st.checkbox("Disaster & Weather Alerts", 
                                                 value=preferences.get('disaster_alerts', True))
                    
                    scheme_alerts = st.checkbox("Government Scheme Alerts", 
                                              value=preferences.get('scheme_alerts', True))
                    
                    st.markdown("#### ⏰ Notification Frequency")
                    frequency = st.selectbox("Notification Frequency", 
                                            ["immediate", "daily_digest", "weekly_digest"],
                                            index=["immediate", "daily_digest", "weekly_digest"].index(
                                                preferences.get('notification_frequency', 'immediate')))
                
                st.markdown("---")
                
                # Save preferences
                if st.button("💾 Save Notification Preferences", type="primary"):
                    new_preferences = {
                        'email_notifications': email_enabled,
                        'whatsapp_notifications': whatsapp_enabled,
                        'email_address': email_address,
                        'whatsapp_number': whatsapp_number,
                        'disease_alerts': disease_alerts,
                        'disaster_alerts': disaster_alerts,
                        'scheme_alerts': scheme_alerts,
                        'notification_frequency': frequency
                    }
                    
                    try:
                        success = services['agent_team'].save_notification_preferences(user_id, new_preferences)
                        if success:
                            st.success("✅ Notification preferences saved successfully!")
                            st.rerun()
                        else:
                            st.error("❌ Failed to save preferences. Please try again.")
                    except Exception as e:
                        st.error(f"Error saving preferences: {str(e)}")
                
                # Configuration status
                st.markdown("#### 🔧 Configuration Status")
                
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.write("Check if your notification credentials are properly configured:")
                
                with col2:
                    if st.button("🔄 Reload Config", help="Reload environment variables from .env file"):
                        try:
                            success = services['agent_team'].reload_notification_config()
                            if success:
                                st.success("✅ Configuration reloaded!")
                                st.rerun()
                            else:
                                st.error("❌ Failed to reload configuration")
                        except Exception as e:
                            st.error(f"Error reloading config: {str(e)}")
                
                try:
                    test_results = services['agent_team'].test_notification_setup(user_id)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Email Configuration:**")
                        if test_results.get('email_configured'):
                            st.success("✅ Server configured")
                        else:
                            st.error("❌ Server not configured")
                        
                        if test_results.get('user_email_set'):
                            st.success("✅ Email address set")
                        else:
                            st.warning("⚠️ Email address not set")
                    
                    with col2:
                        st.write("**WhatsApp Configuration:**")
                        if test_results.get('whatsapp_configured'):
                            st.success("✅ API configured")
                        else:
                            st.error("❌ API not configured")
                        
                        if test_results.get('user_whatsapp_set'):
                            st.success("✅ Phone number set")
                        else:
                            st.warning("⚠️ Phone number not set")
                    
                    # Show current environment variable status
                    st.markdown("---")
                    st.markdown("**🔍 Current Environment Variables:**")
                    
                    import os
                    env_status = {
                        "TWILIO_ACCOUNT_SID": "✅ Set" if os.getenv('TWILIO_ACCOUNT_SID') else "❌ Not set",
                        "TWILIO_AUTH_TOKEN": "✅ Set" if os.getenv('TWILIO_AUTH_TOKEN') else "❌ Not set",
                        "EMAIL_USER": "✅ Set" if os.getenv('EMAIL_USER') else "❌ Not set",
                        "EMAIL_APP_PASSWORD": "✅ Set" if os.getenv('EMAIL_APP_PASSWORD') else "❌ Not set"
                    }
                    
                    for var, status in env_status.items():
                        st.write(f"• **{var}**: {status}")
                
                except Exception as e:
                    st.error(f"Error checking configuration: {str(e)}")
                    
            except Exception as e:
                st.error(f"Error loading notification preferences: {str(e)}")
        
        with tab2:
            st.markdown("### 📊 Notification Statistics")
            
            try:
                stats = services['agent_team'].get_notification_statistics(user_id)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Notifications", stats.get('total_notifications', 0))
                    st.metric("Success Rate", f"{stats.get('success_rate', 0):.1f}%")
                
                with col2:
                    st.metric("Successful Notifications", stats.get('successful_notifications', 0))
                    st.metric("Recent Notifications (30 days)", stats.get('recent_notifications', 0))
                
                with col3:
                    notifications_by_type = stats.get('notifications_by_type', {})
                    st.write("**Notifications by Type:**")
                    for alert_type, count in notifications_by_type.items():
                        st.write(f"• {alert_type.replace('_', ' ').title()}: {count}")
                
                # Visualization
                if notifications_by_type:
                    st.markdown("#### 📈 Notification Distribution")
                    import pandas as pd
                    
                    df = pd.DataFrame(list(notifications_by_type.items()), 
                                    columns=['Alert Type', 'Count'])
                    df['Alert Type'] = df['Alert Type'].str.replace('_', ' ').str.title()
                    
                    st.bar_chart(df.set_index('Alert Type'))
                    
            except Exception as e:
                st.error(f"Error loading statistics: {str(e)}")
        
        with tab3:
            st.markdown("### 📜 Notification History")
            
            try:
                history = services['agent_team'].get_notification_history(user_id, 20)
                
                if history:
                    for i, notification in enumerate(history):
                        with st.expander(f"📧 {notification.get('title', 'Notification')} - {notification.get('sent_at', '')[:19]}"):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write(f"**Type:** {notification.get('alert_category', 'Unknown').replace('_', ' ').title()}")
                                st.write(f"**Method:** {notification.get('delivery_method', 'Unknown').title()}")
                                st.write(f"**Status:** {notification.get('delivery_status', 'Unknown').title()}")
                            
                            with col2:
                                st.write(f"**Sent:** {notification.get('sent_at', 'Unknown')[:19]}")
                                if notification.get('error_message'):
                                    st.error(f"Error: {notification['error_message']}")
                            
                            st.write("**Message:**")
                            st.text_area("", value=notification.get('message', ''), height=100, key=f"msg_{i}", disabled=True)
                else:
                    st.info("No notification history available")
                    
            except Exception as e:
                st.error(f"Error loading notification history: {str(e)}")
        
        with tab4:
            st.markdown("### 🧪 Test Notifications")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🦠 Test Disease Alert")
                if st.button("Send Test Disease Alert"):
                    test_disease_info = {
                        'disease': 'Late Blight',
                        'confidence': 85.7,
                        'recommendations': [
                            'Apply copper-based fungicide immediately',
                            'Remove affected plant parts',
                            'Improve air circulation around plants',
                            'Avoid overhead watering'
                        ]
                    }
                    
                    try:
                        # Send disease alert via production messaging service
                        user_profile = auth_manager.get_user_profile(user_id)
                        contact_info = {
                            'phone': user_profile.get('phone', '') if user_profile else '',
                            'whatsapp': user_profile.get('phone', '') if user_profile else '',
                            'email': user_profile.get('email', '') if user_profile else ''
                        }
                        
                        disease_message = f"🦠 DISEASE ALERT: {test_disease_info['disease']} detected in {test_disease_info['crop']}. Confidence: {test_disease_info['confidence']}%. Take immediate action to prevent spread."
                        
                        if MESSAGING_AVAILABLE:
                            result = messaging_service.send_notification(
                                contact_info,
                                disease_message,
                                f"🦠 {test_disease_info['disease']} Alert"
                            )
                            st.success(f"Disease alert sent! Results: {result}")
                        else:
                            st.error("Messaging service not available")
                    except Exception as e:
                        st.error(f"Error sending test alert: {str(e)}")
                
                st.markdown("#### 🌪️ Test Disaster Alert")
                disaster_type = st.selectbox("Disaster Type", ["cyclone", "hailstorm", "drought"])
                
                if st.button("Send Test Disaster Alert"):
                    try:
                        # Get user's contact info from profile
                        user_profile = auth_manager.get_user_profile(user_id)
                        whatsapp_number = user_profile.get('phone', '') if user_profile else ''
                        
                        st.info(f"🔍 **Debug Info:**\n"
                               f"• Sending to: {whatsapp_number}\n"
                               f"• Twilio Account SID: {os.getenv('TWILIO_ACCOUNT_SID', 'Not set')[:10]}...\n"
                               f"• Twilio Auth Token: {'Set' if os.getenv('TWILIO_AUTH_TOKEN') else 'Not set'}")
                        
                        # Create real disaster alert
                        disaster_alert = {
                            'type': disaster_type,
                            'severity': 'high',
                            'message': f"🚨 URGENT: {disaster_type} alert in your area. Take immediate precautions and follow safety guidelines.",
                            'timestamp': datetime.now().isoformat(),
                            'location': 'Your registered location'
                        }
                        
                        # Get user contact info
                        user_profile = auth_manager.get_user_profile(user_id)
                        contact_info = {
                            'phone': whatsapp_number,
                            'whatsapp': whatsapp_number,
                            'email': user_profile.get('email') if user_profile else None
                        }
                        
                        # Send via production messaging service
                        if MESSAGING_AVAILABLE:
                            result = messaging_service.send_notification(
                                contact_info,
                                disaster_alert['message'],
                                f"🚨 {disaster_type} Alert"
                            )
                        else:
                            result = {'success': False, 'error': 'Messaging service not available'}
                        
                        # Display results for all messaging channels
                        if result.get('success'):
                            st.success(f"✅ Alert sent successfully!")
                            
                            # Show detailed results
                            for service, service_result in result.get('results', {}).items():
                                if service_result.get('success'):
                                    st.success(f"📱 {service.upper()}: Delivered")
                                else:
                                    st.error(f"❌ {service.upper()}: {service_result.get('error', 'Failed')}")
                        else:
                            st.error("❌ Failed to send alert via any channel")
                            if result.get('error'):
                                st.error(f"Error: {result['error']}")
                        
                        st.write(f"**Full Results:** {result}")
                        
                        # Show recent Twilio logs suggestion
                        st.info("💡 **Check Twilio Logs:**\n"
                               "Go to Twilio Console → Messaging → Logs to see the exact error code and troubleshoot further.")
                        
                    except Exception as e:
                        st.error(f"Error sending test alert: {str(e)}")
            
            with col2:
                st.markdown("#### 🏛️ Test Scheme Alert")
                if st.button("Send Test Scheme Alert"):
                    test_schemes = [
                        {
                            'title': 'PM-KISAN Samman Nidhi Yojana',
                            'description': 'Direct income support to farmers with ₹6000 per year in three installments.'
                        },
                        {
                            'title': 'Pradhan Mantri Fasal Bima Yojana',
                            'description': 'Crop insurance scheme providing financial support against crop loss.'
                        }
                    ]
                    
                    try:
                        result = services['agent_team'].send_scheme_notification(user_id, test_schemes)
                        st.success(f"Test scheme alert sent! Results: {result}")
                    except Exception as e:
                        st.error(f"Error sending test alert: {str(e)}")
                
                st.markdown("#### ⚙️ Configuration Test")
                if st.button("Test Notification Setup"):
                    try:
                        test_results = services['agent_team'].test_notification_setup(user_id)
                        
                        st.write("**Configuration Status:**")
                        for key, value in test_results.items():
                            if key.endswith('_test'):
                                continue
                            status = "✅" if value else "❌"
                            st.write(f"{status} {key.replace('_', ' ').title()}: {value}")
                        
                        if test_results.get('email_test') is not None:
                            if test_results['email_test']:
                                st.success("✅ Test email sent successfully!")
                            else:
                                st.error("❌ Failed to send test email")
                                
                    except Exception as e:
                        st.error(f"Error testing setup: {str(e)}")
            
            st.markdown("---")
            st.info("💡 **Setup Instructions:**\n\n"
                   "**For Email Notifications:**\n"
                   "1. Set EMAIL_USER and EMAIL_APP_PASSWORD environment variables\n"
                   "2. Use Gmail app-specific password for EMAIL_APP_PASSWORD\n\n"
                   "**For WhatsApp Notifications:**\n"
                   "1. Set TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN environment variables\n"
                   "2. Get credentials from Twilio Console\n"
                   "3. Enable WhatsApp sandbox for testing")
    
    else:
        st.warning("Please log in to access notification settings.")
        if st.button("Go to Login"):
            st.session_state.page = "🔐 Login"
            st.rerun()

elif page == "👤 User Profile":
    st.title("👤 User Profile & Preferences")
    
    st.markdown("### 👤 Personal Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        name = st.text_input("Name:", value=st.session_state.user_profile.get('name', ''))
        location_profile = st.text_input("Location:", value=st.session_state.user_profile.get('location', ''))
        farm_size = st.number_input("Farm Size (acres):", min_value=0.0, 
                                   value=st.session_state.user_profile.get('farm_size', 0.0))
    
    with col2:
        # Handle primary crops dropdown safely
        available_crops = ["Rice", "Wheat", "Cotton", "Sugarcane", "Pulses", "Vegetables", "Fruits", "Spices", "Oilseeds", "Tea", "Coffee"]
        current_crops = st.session_state.user_profile.get('primary_crops', [])
        
        # Ensure current crops exist in available list
        valid_crops = [crop for crop in current_crops if crop in available_crops]
        if not valid_crops:
            valid_crops = []
            
        primary_crops = st.multiselect(
            "Primary Crops:",
            available_crops,
            default=valid_crops
        )
        
        # Handle farming type dropdown safely
        farming_types = ["Traditional", "Organic", "Mixed", "Commercial"]
        current_farming_type = st.session_state.user_profile.get('farming_type', 'Traditional')
        
        # Ensure current value is in the list
        if current_farming_type not in farming_types:
            current_farming_type = "Traditional"
            
        farming_type = st.selectbox(
            "Farming Type:",
            farming_types,
            index=farming_types.index(current_farming_type)
        )
        
        experience = st.slider("Years of Experience:", 0, 50, 
                              st.session_state.user_profile.get('experience', 5))
    
    st.markdown("### ⚙️ Preferences")
    
    col1, col2 = st.columns(2)
    
    with col1:
        preferred_language = st.selectbox(
            "Preferred Language:",
            ["English", "हिन्दी", "தமிழ்", "తెలుగు", "বাংলা"],
            index=["English", "हिन्दी", "தமিழ்", "తెలుగు", "বাংলা"].index(
                st.session_state.user_profile.get('preferred_language', 'English')
            )
        )
        
        notification_preferences = st.multiselect(
            "Notification Preferences:",
            ["Weather Alerts", "Disease Outbreaks", "Market Prices", "Government Schemes"],
            default=st.session_state.user_profile.get('notification_preferences', [])
        )
    
    with col2:
        interests = st.multiselect(
            "Areas of Interest:",
            ["Crop Diseases", "Weather Forecasting", "Financial Planning", "New Technologies"],
            default=st.session_state.user_profile.get('interests', [])
        )
    
    if st.button("💾 Save Profile", type="primary"):
        # Update session state
        st.session_state.user_profile.update({
            'name': name,
            'location': location_profile,
            'farm_size': farm_size,
            'primary_crops': primary_crops,
            'farming_type': farming_type,
            'experience': experience,
            'preferred_language': preferred_language,
            'notification_preferences': notification_preferences,
            'interests': interests
        })
        
        # Save to persistent storage
        if auth_manager.save_user_profile():
            st.success("✅ Profile saved successfully to your account!")
        else:
            st.success("✅ Profile saved to session!")
        st.balloons()
    
    # Display profile summary
    if st.session_state.user_profile:
        st.markdown("### 📊 Profile Summary")
        
        profile_summary = f"""
        <div class="agent-response">
        <h4>👤 {st.session_state.user_profile.get('name', 'User')}</h4>
        <p><strong>📍 Location:</strong> {st.session_state.user_profile.get('location', 'Not specified')}</p>
        <p><strong>🌾 Farm Size:</strong> {st.session_state.user_profile.get('farm_size', 0)} acres</p>
        <p><strong>🌱 Primary Crops:</strong> {', '.join(st.session_state.user_profile.get('primary_crops', []))}</p>
        <p><strong>🚜 Farming Type:</strong> {st.session_state.user_profile.get('farming_type', 'Not specified')}</p>
        <p><strong>📅 Experience:</strong> {st.session_state.user_profile.get('experience', 0)} years</p>
        </div>
        """
        
        st.markdown(profile_summary, unsafe_allow_html=True)

elif page == "🔐 Login":
    st.title("🔐 Login & Security Management")
    
    # Import PasswordManager
    from utils.password_manager import PasswordManager
    password_manager = PasswordManager()
    
    if st.session_state.get('user_id'):
        user_id = st.session_state['user_id']
        
        st.success(f"👋 Welcome back! You are logged in.")
        
        # Create tabs for different security sections
        tab1, tab2, tab3 = st.tabs(["🔑 Update Password", "🔐 API Keys & Credentials", "📊 Security Audit"])
        
        with tab1:
            st.markdown("### 🔑 Update Your Password")
            
            with st.form("update_password_form"):
                current_password = st.text_input("Current Password", type="password", help="Enter your current password")
                new_password = st.text_input("New Password", type="password", help="Minimum 8 characters, include uppercase, lowercase, numbers, and special characters")
                confirm_password = st.text_input("Confirm New Password", type="password", help="Re-enter your new password")
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    submitted = st.form_submit_button("🔄 Update Password", type="primary")
                with col2:
                    if st.form_submit_button("🎲 Generate Strong Password"):
                        generated_password = password_manager.generate_secure_password()
                        st.info(f"Generated password: `{generated_password}`")
                        st.code(f"Generated password: {generated_password}")
                
                if submitted:
                    if not all([current_password, new_password, confirm_password]):
                        st.error("❌ Please fill in all fields")
                    elif new_password != confirm_password:
                        st.error("❌ New passwords don't match")
                    elif len(new_password) < 8:
                        st.error("❌ Password must be at least 8 characters")
                    else:
                        # Check password strength
                        strength_score = password_manager.check_password_strength(new_password)
                        if strength_score < 3:
                            st.warning(f"⚠️ Password strength: {['Very Weak', 'Weak', 'Medium', 'Strong', 'Very Strong'][strength_score]}")
                            st.info("Consider using a stronger password with uppercase, lowercase, numbers, and special characters")
                        
                        # Update password
                        success, message = password_manager.update_user_password(user_id, current_password, new_password)
                        if success:
                            st.success("✅ Password updated successfully!")
                            st.balloons()
                            # Clear the form
                            st.rerun()
                        else:
                            st.error(f"❌ {message}")
        
        with tab2:
            st.markdown("### 🔐 API Keys & Credentials Management")
            st.info("🔒 All credentials are securely encrypted and stored. Enter your API keys here to use throughout the application without .env files.")
            
            # Essential API Keys Section
            st.markdown("#### 🌟 Essential Service Credentials")
            st.markdown("*Configure these primary services for full application functionality:*")
            
            # Get current API keys
            api_keys = password_manager.get_user_api_keys(user_id) if hasattr(password_manager, 'get_user_api_keys') else {}
            
            # Essential services configuration
            essential_services = {
                "groq_api": {
                    "name": "🤖 Groq AI API",
                    "description": "For AI-powered agricultural advice and analysis",
                    "placeholder": "gsk_...",
                    "help": "Get your API key from https://console.groq.com/keys"
                },
                "weather_api": {
                    "name": "🌤️ Weather API",
                    "description": "For real-time weather data and forecasts",
                    "placeholder": "your-weather-api-key",
                    "help": "Get your API key from OpenWeatherMap or similar service"
                },
                "twilio_account_sid": {
                    "name": "📱 Twilio Account SID",
                    "description": "For SMS notifications and WhatsApp integration",
                    "placeholder": "AC...",
                    "help": "Find in your Twilio Console Dashboard"
                },
                "twilio_auth_token": {
                    "name": "🔐 Twilio Auth Token",
                    "description": "Authentication token for Twilio services",
                    "placeholder": "your-auth-token",
                    "help": "Find in your Twilio Console Dashboard"
                },
                "twilio_phone_number": {
                    "name": "📞 Twilio Phone Number",
                    "description": "Your Twilio phone number for sending messages",
                    "placeholder": "+1234567890",
                    "help": "Format: +1234567890 (include country code)"
                }
            }
            
            # Display and manage essential services
            with st.form("essential_api_keys_form"):
                st.markdown("##### 🔑 Enter Your API Credentials")
                
                form_data = {}
                for service_key, service_info in essential_services.items():
                    current_value = ""
                    if api_keys and service_key in api_keys:
                        current_value = "●●●●●●●●●●●●" if api_keys[service_key].get('key') else ""
                    
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        form_data[service_key] = st.text_input(
                            service_info["name"],
                            value="" if not current_value else current_value,
                            placeholder=service_info["placeholder"],
                            help=f"{service_info['description']}. {service_info['help']}",
                            type="password",
                            key=f"form_{service_key}"
                        )
                    with col2:
                        if current_value:
                            if st.form_submit_button(f"👁️", help=f"Show {service_info['name']} key", key=f"show_{service_key}"):
                                if api_keys and service_key in api_keys:
                                    st.code(api_keys[service_key]['key'])
                
                # Form submission
                col1, col2, col3 = st.columns([2, 2, 1])
                with col1:
                    if st.form_submit_button("💾 Save All Credentials", type="primary"):
                        saved_count = 0
                        errors = []
                        
                        for service_key, value in form_data.items():
                            if value and value != "●●●●●●●●●●●●" and len(value.strip()) > 0:
                                try:
                                    # Validate API key format
                                    if service_key == "groq_api" and not value.startswith("gsk_"):
                                        errors.append(f"Groq API key should start with 'gsk_'")
                                        continue
                                    if service_key == "twilio_account_sid" and not value.startswith("AC"):
                                        errors.append(f"Twilio Account SID should start with 'AC'")
                                        continue
                                    if service_key == "twilio_phone_number" and not value.startswith("+"):
                                        errors.append(f"Twilio phone number should include country code (+)")
                                        continue
                                    
                                    # Save to session state for immediate use
                                    st.session_state[f"api_{service_key}"] = value
                                    
                                    # Save to password manager if available
                                    if hasattr(password_manager, 'update_api_key'):
                                        if password_manager.update_api_key(user_id, service_key, value):
                                            saved_count += 1
                                    else:
                                        saved_count += 1
                                        
                                except Exception as e:
                                    errors.append(f"Error saving {service_key}: {str(e)}")
                        
                        if saved_count > 0:
                            st.success(f"✅ Successfully saved {saved_count} API credentials!")
                            st.balloons()
                            # Update environment variables for immediate use
                            import os
                            for service_key, value in form_data.items():
                                if value and value != "●●●●●●●●●●●●":
                                    os.environ[service_key.upper()] = value
                            st.rerun()
                        
                        if errors:
                            for error in errors:
                                st.error(f"❌ {error}")
                
                with col2:
                    if st.form_submit_button("🧪 Test Connections"):
                        st.info("Testing API connections...")
                        test_results = []
                        
                        # Test Groq API
                        if form_data.get('groq_api'):
                            try:
                                # Safe import and test
                                import requests
                                headers = {"Authorization": f"Bearer {form_data['groq_api']}"}
                                response = requests.get("https://api.groq.com/openai/v1/models", headers=headers, timeout=5)
                                if response.status_code == 200:
                                    test_results.append("✅ Groq API: Connected")
                                else:
                                    test_results.append("❌ Groq API: Invalid key")
                            except Exception as e:
                                test_results.append(f"⚠️ Groq API: {str(e)[:50]}...")
                        
                        # Test Weather API
                        if form_data.get('weather_api'):
                            try:
                                import requests
                                response = requests.get(f"http://api.openweathermap.org/data/2.5/weather?q=London&appid={form_data['weather_api']}", timeout=5)
                                if response.status_code == 200:
                                    test_results.append("✅ Weather API: Connected")
                                else:
                                    test_results.append("❌ Weather API: Invalid key")
                            except Exception as e:
                                test_results.append(f"⚠️ Weather API: {str(e)[:50]}...")
                        
                        # Test Twilio
                        if form_data.get('twilio_account_sid') and form_data.get('twilio_auth_token'):
                            try:
                                # Safe import and test
                                from twilio.rest import Client
                                client = Client(form_data['twilio_account_sid'], form_data['twilio_auth_token'])
                                account = client.api.accounts(form_data['twilio_account_sid']).fetch()
                                test_results.append("✅ Twilio: Connected")
                            except Exception as e:
                                test_results.append(f"⚠️ Twilio: {str(e)[:50]}...")
                        
                        for result in test_results:
                            if "✅" in result:
                                st.success(result)
                            elif "❌" in result:
                                st.error(result)
                            else:
                                st.warning(result)
            
            # Current API Keys Status
            st.markdown("#### 📊 Current API Keys Status")
            
            if api_keys:
                status_data = []
                for service_key, service_info in essential_services.items():
                    if service_key in api_keys:
                        key_data = api_keys[service_key]
                        status = "🟢 Active" if key_data.get('key') else "🔴 Empty"
                        last_updated = key_data.get('updated_at', 'Unknown')
                        status_data.append({
                            "Service": service_info["name"],
                            "Status": status,
                            "Last Updated": last_updated
                        })
                    else:
                        status_data.append({
                            "Service": service_info["name"],
                            "Status": "⚪ Not Set",
                            "Last Updated": "Never"
                        })
                
                if status_data:
                    import pandas as pd
                    df = pd.DataFrame(status_data)
                    st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.info("No API keys configured yet. Use the form above to add your credentials.")
            
            # Additional Services Section
            st.markdown("#### 🔧 Additional Services (Optional)")
            
            additional_services = [
                "google_translate", "email_smtp", "database_url", "redis_url", 
                "aws_access_key", "azure_key", "custom_api_1", "custom_api_2"
            ]
            
            with st.expander("➕ Add More API Keys"):
                with st.form("additional_api_keys_form"):
                    service_name = st.selectbox(
                        "Service Type", 
                        additional_services,
                        help="Select the type of service you want to add"
                    )
                    custom_service_name = st.text_input(
                        "Custom Service Name (Optional)",
                        help="Enter a custom name if not using predefined services"
                    )
                    api_key = st.text_input(
                        "API Key/Credential", 
                        type="password", 
                        help="Enter your API key or credential"
                    )
                    expiry_date = st.date_input(
                        "Expiry Date (Optional)",
                        help="Set an expiry date for this API key"
                    )
                    
                    if st.form_submit_button("➕ Add Additional Credential", type="secondary"):
                        final_service_name = custom_service_name if custom_service_name else service_name
                        if api_key and final_service_name:
                            try:
                                # Save to session state
                                st.session_state[f"api_{final_service_name}"] = api_key
                                if expiry_date:
                                    st.session_state[f"api_{final_service_name}_expiry"] = expiry_date
                                
                                # Save to password manager if available
                                if hasattr(password_manager, 'update_api_key'):
                                    password_manager.update_api_key(user_id, final_service_name, api_key)
                                
                                st.success(f"✅ {final_service_name} credential added successfully!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ Failed to add credential: {str(e)}")
                        else:
                            st.error("❌ Please provide both service name and API key")
            
            # Import Safety Status
            st.markdown("#### 🛡️ Import Safety Status")
            
            # Check import status for critical services
            import_status = []
            
            # Check Groq import
            try:
                import groq
                import_status.append({"Service": "🤖 Groq AI", "Status": "✅ Available", "Version": getattr(groq, '__version__', 'Unknown')})
            except ImportError as e:
                import_status.append({"Service": "🤖 Groq AI", "Status": "❌ Import Error", "Error": str(e)[:50]})
            
            # Check Google Translate import
            try:
                from googletrans import Translator
                import_status.append({"Service": "🌐 Google Translate", "Status": "✅ Available", "Version": "4.0.0rc1"})
            except ImportError as e:
                import_status.append({"Service": "🌐 Google Translate", "Status": "❌ Import Error", "Error": str(e)[:50]})
            
            # Check Twilio import
            try:
                from twilio.rest import Client
                import twilio
                import_status.append({"Service": "📱 Twilio", "Status": "✅ Available", "Version": getattr(twilio, '__version__', 'Unknown')})
            except ImportError as e:
                import_status.append({"Service": "📱 Twilio", "Status": "❌ Import Error", "Error": str(e)[:50]})
            
            # Check Weather API libraries
            try:
                import requests
                import_status.append({"Service": "🌤️ Weather Requests", "Status": "✅ Available", "Version": getattr(requests, '__version__', 'Unknown')})
            except ImportError as e:
                import_status.append({"Service": "🌤️ Weather Requests", "Status": "❌ Import Error", "Error": str(e)[:50]})
            
            # Display import status
            if import_status:
                import pandas as pd
                df_imports = pd.DataFrame(import_status)
                st.dataframe(df_imports, use_container_width=True, hide_index=True)
            
            # Quick Actions
            st.markdown("#### ⚡ Quick Actions")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if st.button("🔄 Refresh Status", help="Refresh all API key and import status"):
                    st.rerun()
            
            with col2:
                if st.button("📋 Export Config", help="Export API configuration (keys masked)"):
                    config_data = {
                        "services_configured": len(api_keys) if api_keys else 0,
                        "essential_services": list(essential_services.keys()),
                        "import_status": {item["Service"]: item["Status"] for item in import_status}
                    }
                    st.download_button(
                        "💾 Download Config",
                        data=str(config_data),
                        file_name="api_config.json",
                        mime="application/json"
                    )
            
            with col3:
                if st.button("🧹 Clear All Keys", help="Clear all API keys (requires confirmation)"):
                    if st.session_state.get('confirm_clear_keys'):
                        # Clear session state API keys
                        keys_to_clear = [key for key in st.session_state.keys() if key.startswith('api_')]
                        for key in keys_to_clear:
                            del st.session_state[key]
                        st.success("✅ All API keys cleared from session!")
                        st.session_state['confirm_clear_keys'] = False
                        st.rerun()
                    else:
                        st.session_state['confirm_clear_keys'] = True
                        st.warning("⚠️ Click again to confirm clearing all API keys")
            
            with col4:
                if st.button("📖 API Help", help="Show API key setup instructions"):
                    with st.expander("📚 API Setup Instructions", expanded=True):
                        st.markdown("""
                        ### 🔑 How to Get API Keys:
                        
                        **🤖 Groq AI API:**
                        1. Visit [console.groq.com](https://console.groq.com/keys)
                        2. Sign up/login to your account
                        3. Generate a new API key
                        4. Copy the key (starts with 'gsk_')
                        
                        **🌤️ Weather API:**
                        1. Visit [OpenWeatherMap](https://openweathermap.org/api)
                        2. Sign up for a free account
                        3. Generate API key from dashboard
                        4. Copy your API key
                        
                        **📱 Twilio (SMS/WhatsApp):**
                        1. Visit [twilio.com](https://www.twilio.com/)
                        2. Create account and verify phone
                        3. Get Account SID and Auth Token from Console
                        4. Purchase/setup a phone number
                        
                        **🔒 Security Tips:**
                        - Never share your API keys publicly
                        - Use environment variables in production
                        - Set expiry dates when possible
                        - Monitor usage and costs
                        - Rotate keys regularly
                        """)
        
        with tab3:
            st.markdown("### 📊 Security Audit")
            
            # Get security information
            security_info = password_manager.get_security_audit_info(user_id)
            
            if security_info and 'error' not in security_info:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Password Age", f"{security_info.get('password_age_days', 0)} days")
                with col2:
                    st.metric("Last Login", security_info.get('last_login', 'Never'))
                with col3:
                    st.metric("Credentials Configured", security_info.get('api_keys_count', 0))
                
                # Security recommendations
                st.markdown("#### 🔍 Security Recommendations")
                
                recommendations = []
                if security_info.get('password_age_days', 0) > 90:
                    recommendations.append("🔴 Consider changing your password (older than 90 days)")
                if security_info.get('api_keys_count', 0) == 0:
                    recommendations.append("🟡 Add API keys and credentials for enhanced functionality")
                if security_info.get('api_keys_count', 0) > 0 and not security_info.get('backup_keys'):
                    recommendations.append("🟡 Generate backup credentials for redundancy")
                
                if recommendations:
                    for rec in recommendations:
                        st.warning(rec)
                else:
                    st.success("✅ Your security settings look good!")
                
                # Export security settings
                if st.button("📥 Export Security Report"):
                    report = password_manager.export_security_report(user_id)
                    st.download_button(
                        label="📄 Download Security Report",
                        data=report,
                        file_name=f"security_report_{user_id}_{datetime.now().strftime('%Y%m%d')}.json",
                        mime="application/json"
                    )
            else:
                st.error("Unable to load security audit information. Please try again later.")
                if security_info and 'error' in security_info:
                    st.error(f"Error: {security_info['error']}")
    else:
        st.markdown("### 🔐 Please Log In")
        st.info("You need to log in to access password management and backup credentials.")
        
        # Show authentication form if not authenticated
        if not auth_manager.show_auth_form():
            st.warning("Authentication required to access security features.")

# Add health check endpoint
add_health_endpoint()

# Show cache stats in sidebar
show_cache_stats()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 20px; color: #666;">
<p>🌾 <strong>Agricultural AI Advisor</strong> - Empowering Indian farmers with AI technology</p>
<p>Made with ❤️ for Indian Agriculture | Version 1.0</p>
</div>
""", unsafe_allow_html=True)
