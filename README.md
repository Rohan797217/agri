# AgriAI Advisor

## 🌾 AI-Powered Agricultural Advisory Platform for Indian Farmers

AgriAI Advisor is a comprehensive multimodal agricultural advisory system designed specifically for Indian farmers. The application provides intelligent farming guidance through specialized AI agents that handle crop management, weather analysis, financial planning, policy information, market insights, and image-based disease detection.

### 🚀 Features

- **Multimodal Input Processing**: Text, voice, images, and PDF document support
- **8 Specialized AGNO Agents**: Expert advice across agricultural domains
- **Team Coordination**: Multi-agent collaboration for complex queries
- **Real-time Search Integration**: Live agricultural information from the web
- **Multilingual Support**: Indian languages with transliteration
- **Location-aware**: Regional recommendations based on user location
- **Financial Advisory**: Government schemes and loan guidance

### 📋 Prerequisites

- Python 3.11+
- Groq API Key (get from [console.groq.com](https://console.groq.com))
- Optional: Weather API key for enhanced weather services

### 🔧 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/agriAI-advisor.git
cd agriAI-advisor
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
export GROQ_API_KEY="your_groq_api_key_here"
# Optional:
export WEATHER_API_KEY="your_weather_api_key_here"
```

4. Run the application:
```bash
streamlit run main.py
```

### 🌐 Deployment

#### Deploy on Streamlit Cloud

1. Fork this repository to your GitHub account
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click "New app" and select your forked repository
4. Set `main.py` as the main file path
5. Add your API keys in the "Secrets" section:
```toml
GROQ_API_KEY = "your_groq_api_key"
WEATHER_API_KEY = "your_weather_api_key"  # optional
```
6. Click "Deploy"

### 📁 Project Structure

```
agriAI-advisor/
├── main.py                 # Main Streamlit application
├── agents/
│   └── agno_system.py     # AGNO multi-agent system
├── utils/
│   ├── language.py        # Language processing utilities
│   ├── weather.py         # Weather API integration
│   └── image_analysis.py  # Image processing for crop analysis
├── data/
│   └── knowledge_base.json # Agricultural knowledge database
├── vector_db/
│   └── documents.json     # Vector database for documents
├── .streamlit/
│   └── config.toml        # Streamlit configuration
└── requirements.txt       # Python dependencies
```

### 🤖 AGNO Agent System

The application uses a sophisticated multi-agent system with 8 specialized agents:

1. **Crop Management Specialist** - Seed varieties, cultivation practices
2. **Weather & Climate Advisor** - Seasonal guidance, weather-based decisions  
3. **Soil Health Specialist** - Soil testing, nutrient management
4. **Agricultural Finance Advisor** - Government schemes, loans, insurance
5. **IPM Specialist** - Integrated Pest Management strategies
6. **Water Management Engineer** - Irrigation systems, water conservation
7. **Market Intelligence Analyst** - Commodity prices, market timing
8. **Supply Chain Expert** - Input sourcing, storage, certification

### 🔄 Agent Teams

Three coordinated teams handle complex queries:
- **Farm Planning Team**: Integrated crop, soil, weather, and irrigation advice
- **Financial Strategy Team**: Combined financial, market, and supply chain guidance
- **Crop Protection Team**: Integrated pest, weather, and soil management

### 📖 Usage Examples

**Cotton Cultivation Query:**
```
"I'm growing cotton in Maharashtra. When should I irrigate given the current weather?"
```

**Financial Planning:**
```
"What government schemes are available for drip irrigation installation on my 2-acre farm?"
```

**Pest Management:**
```
"My tomato plants have yellowing leaves. What IPM strategy should I follow?"
```

### 🔒 Security

- API keys are stored as environment variables
- No user data is permanently stored
- All communications are encrypted

### 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

### 📞 Support

For technical support or agricultural queries, contact:
- GitHub Issues: Create an issue for bugs or feature requests
- Email: [Your email]

### 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Made with ❤️ for Indian farmers**