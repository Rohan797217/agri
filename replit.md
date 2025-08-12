# AgriAI Advisor

## Overview

AgriAI Advisor is a comprehensive multimodal agricultural advisory system designed for Indian farmers. The application provides intelligent farming guidance through specialized AI agents that handle crop management, weather analysis, financial planning, policy information, market insights, and image-based disease detection. Built with Streamlit for the frontend and powered by Groq's LLM services, the system processes multiple input types including text, voice, images, and PDFs to deliver contextual agricultural advice in multiple Indian languages.

## User Preferences

Preferred communication style: Simple, everyday language.

## Recent Changes (August 2025)

- **AGNO System Successfully Implemented**: Enhanced multi-agent system with 8 specialized agricultural advisors and 3 coordinated teams now fully operational
- **Model Updated**: Migrated from deprecated mixtral-8x7b-32768 to llama3-70b-8192 for improved performance
- **Search Integration**: Real-time DuckDuckGo search integration using ddgs library for current agricultural information
- **Deployment Ready**: Complete GitHub deployment configuration with Streamlit Cloud integration guide
- **User Testing Confirmed**: Agricultural Finance Advisor successfully providing comprehensive credit and policy guidance

## System Architecture

### Enhanced AGNO Agent Architecture
The system follows a comprehensive multi-agent design pattern with specialized AGNO (Agricultural Guidance and Networking Operations) agents:

#### Core AGNO Agents:
- **Base Agent**: Provides common functionality including Groq LLM integration and response generation
- **Crop Agent**: Handles crop-specific advice, variety selection, and cultivation recommendations
- **Weather Agent**: Processes weather data with dual LLM approach and provides climate-based farming guidance
- **Finance Agent**: Enhanced with external banking integrations, loan calculators, and eligibility checkers
- **Policy Agent**: Specializes in government schemes and agricultural policies
- **Market Agent**: Provides market prices, trends, and trading advice
- **Image Agent**: Analyzes crop images for disease and pest detection

#### Specialized AGNO Agents (Enhanced August 2025):
- **Soil Health Agent**: Soil testing interpretation, pH management, nutrient analysis, and organic matter recommendations
- **Pest Management Agent**: Integrated Pest Management (IPM) strategies, biological control, and sustainable pest solutions
- **Irrigation Agent**: Water management optimization, irrigation system selection, and conservation techniques
- **Supply Chain Agent**: Input sourcing, storage solutions, market connections, and value chain optimization
- **Compliance Agent**: Organic certification guidance, export compliance, and quality standards management

#### AGNO Framework Implementation (Enhanced August 2025):
- **Professional AGNO Library Integration**: Using official `agno` framework with proper Team coordination
- **GroqChatModel Wrapper**: Custom model wrapper for seamless Groq integration with agno agents
- **Enhanced Agent Specialization**: 8 expert agricultural agents with comprehensive domain knowledge
- **Team Coordination Mode**: Multi-agent teams using coordinate mode for complex integrated responses
- **Advanced Search Integration**: DuckDuckGo and YFinance tools integrated into all relevant agents
- **Context-Aware Routing**: Intelligent query routing with enhanced user profile integration
- **Success Criteria Definition**: Each team has specific success criteria for quality assurance

### Multimodal Input Processing
The input processing layer handles diverse data types through specialized processors:

- **Text Processing**: Natural language queries with multilingual support
- **Voice Input**: Speech-to-text conversion for local language support
- **Image Analysis**: Crop disease detection and visual assessment
- **PDF Processing**: Policy documents and agricultural reports
- **Location Services**: GPS-based regional recommendations

### Knowledge Management System
A comprehensive knowledge base provides structured agricultural information:

- **Crop Database**: Detailed information on Indian crop varieties, growing conditions, and cultivation practices
- **Disease Database**: Crop disease identification and treatment recommendations  
- **Weather Patterns**: Regional climate data and seasonal calendars
- **Government Schemes**: Current agricultural policies and financial assistance programs
- **Market Information**: Price trends and trading insights

### Vector Database Integration
Custom vector database implementation for document storage and semantic search:

- **Document Storage**: Efficient storage of agricultural documents and knowledge
- **Semantic Search**: Vector-based similarity matching for relevant information retrieval
- **Metadata Management**: Contextual information for enhanced search results

### Language Support Infrastructure
Comprehensive multilingual capabilities for Indian languages:

- **Language Detection**: Automatic identification of input language
- **Transliteration**: Support for romanized inputs of Indian languages
- **Regional Terms**: Agricultural terminology in local languages
- **Code-switching**: Mixed language query processing

### Web Scraping Capabilities
Automated data collection from agricultural websites:

- **Weather Data**: Real-time meteorological information
- **Market Prices**: Current commodity prices from various sources
- **Policy Updates**: Latest government scheme information

## External Dependencies

### AI/ML Services
- **Groq API**: Primary LLM service for response generation using Mixtral-8x7b-32768 model
- **Weather API**: External weather service integration for climate data

### Core Framework
- **Streamlit**: Frontend framework for web application interface
- **PIL (Python Imaging Library)**: Image processing and manipulation

### Data Processing
- **Pandas**: Data analysis and manipulation
- **NumPy**: Numerical computing for vector operations
- **Trafilatura**: Web content extraction for data scraping

### Development Tools
- **Asyncio**: Asynchronous programming support
- **Requests**: HTTP client for API communications
- **JSON**: Data serialization and storage
- **Pickle**: Object serialization for vector storage

### Optional Services
- **IMD (India Meteorological Department)**: Government weather data source
- **AgMarkNet**: Agricultural market price information
- **eNAM**: National Agriculture Market platform integration