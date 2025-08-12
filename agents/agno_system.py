"""
Enhanced AGNO (Agricultural Guidance and Networking Operations) System
Multi-agent coordination with serialization-safe implementation
"""

import os
import json
from typing import Dict, List, Any
import streamlit as st
from groq import Groq

def search_duckduckgo(query: str) -> List[Dict]:
    """Search using DuckDuckGo with error handling"""
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            results = []
            for result in ddgs.text(query, max_results=5):
                results.append({
                    'title': result.get('title', ''),
                    'snippet': result.get('body', ''),
                    'url': result.get('href', '')
                })
            return results[:3]  # Limit to top 3 results
    except Exception as e:
        return [{'title': 'Search Error', 'snippet': f'Search unavailable: {str(e)}', 'url': ''}]

class AGNOAgent:
    """Specialized AGNO Agent with Groq integration and search"""
    
    def __init__(self, name: str, role: str, instructions: str, specialization: str):
        self.name = name
        self.role = role
        self.instructions = instructions
        self.specialization = specialization
        
        # Initialize Groq client
        api_key = os.getenv("GROQ_API_KEY")
        if api_key:
            self.groq_client = Groq(api_key=api_key)
        else:
            self.groq_client = None
    
    def run(self, query: str, context: Dict = None) -> str:
        """Execute agent with query and context"""
        try:
            if not self.groq_client:
                return f"Error: {self.name} - Groq API not available"
            
            # Get relevant search results
            search_query = f"{query} {self.specialization} agriculture India"
            search_results = search_duckduckgo(search_query)
            
            # Format search context
            search_context = ""
            if search_results:
                search_context = "Recent Information:\n"
                for result in search_results:
                    if result['title'] != 'Search Error':
                        search_context += f"• {result['title']}: {result['snippet'][:150]}...\n"
            
            # Build enhanced prompt
            system_prompt = f"""
            You are a {self.name} - {self.role}
            
            {self.instructions}
            
            {search_context}
            
            User Context: {json.dumps(context or {}, indent=2)}
            
            Provide specific, actionable advice with:
            - Exact quantities and measurements
            - Cost estimates in Indian Rupees
            - Timeline and scheduling
            - Sources and references
            - Step-by-step procedures
            """
            
            # Generate response
            response = self.groq_client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                temperature=0.7,
                max_tokens=1500
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error from {self.name}: {str(e)}"

class AGNOTeam:
    """Coordinated team of AGNO agents"""
    
    def __init__(self, name: str, agents: List[AGNOAgent], coordination_strategy: str):
        self.name = name
        self.agents = agents
        self.coordination_strategy = coordination_strategy
        
        # Initialize Groq for coordination
        api_key = os.getenv("GROQ_API_KEY")
        if api_key:
            self.groq_client = Groq(api_key=api_key)
        else:
            self.groq_client = None
    
    def run(self, query: str, context: Dict = None) -> str:
        """Execute team coordination"""
        try:
            if not self.groq_client:
                return "Error: Team coordination - Groq API not available"
            
            # Get individual agent responses
            agent_responses = []
            for agent in self.agents:
                response = agent.run(query, context)
                agent_responses.append(f"**{agent.name}:**\n{response}")
            
            # Coordinate responses
            coordination_prompt = f"""
            You are coordinating responses from agricultural specialists for: {query}
            
            Strategy: {self.coordination_strategy}
            
            Specialist Responses:
            {chr(10).join(agent_responses)}
            
            Provide a unified, comprehensive response that:
            1. Integrates all specialist insights
            2. Eliminates contradictions and redundancy
            3. Provides clear priorities and action steps
            4. Shows connections between different aspects
            5. Includes specific costs, quantities, and timelines
            6. Cites sources from the specialists
            
            Format as a well-structured agricultural advisory.
            """
            
            coordinated_response = self.groq_client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[
                    {"role": "system", "content": "You are an expert agricultural coordinator integrating specialist advice."},
                    {"role": "user", "content": coordination_prompt}
                ],
                temperature=0.6,
                max_tokens=2500
            )
            
            return coordinated_response.choices[0].message.content
            
        except Exception as e:
            return f"Team coordination error: {str(e)}"

class AGNOSystem:
    """Enhanced AGNO system with multi-agent coordination"""
    
    def __init__(self):
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        if not self.groq_api_key:
            st.warning("GROQ_API_KEY not found. Please configure API key.")
            return
        
        self.initialize_agents()
        self.setup_teams()
    
    def initialize_agents(self):
        """Initialize specialized agricultural agents"""
        
        # Crop Management Specialist
        self.crop_agent = AGNOAgent(
            name="Crop Management Specialist",
            role="Expert in crop selection, cultivation, and management for Indian agriculture",
            specialization="crop variety seed cultivation farming",
            instructions="""Provide expert advice on:
            - Crop selection based on soil, climate, and market conditions
            - Seed variety recommendations for different regions and seasons
            - Sowing timing and techniques for Indian agricultural seasons
            - Fertilizer and nutrient management with specific quantities
            - Irrigation scheduling and growth monitoring
            
            Focus on practical, implementable solutions for Indian farmers.
            Include specific varieties, quantities, costs, and seasonal timing.
            Consider local climate conditions and market demand."""
        )
        
        # Weather & Climate Advisor
        self.weather_agent = AGNOAgent(
            name="Weather & Climate Advisor",
            role="Expert meteorologist for agricultural weather services",
            specialization="weather climate forecast monsoon season",
            instructions="""Provide weather-based agricultural guidance:
            - Current and forecast weather impact on farming
            - Seasonal climate patterns (Kharif, Rabi, Zaid)
            - Weather-based crop recommendations
            - Risk assessment for weather-related damage
            - Protective measures and contingency planning
            
            Focus on actionable weather-based decisions.
            Include timing recommendations and risk mitigation.
            Consider regional weather variations across India."""
        )
        
        # Soil Health Specialist
        self.soil_agent = AGNOAgent(
            name="Soil Health Specialist",
            role="Expert soil scientist for Indian agricultural soils",
            specialization="soil health testing pH nutrients organic",
            instructions="""Provide soil management expertise:
            - Soil testing protocols and interpretation
            - pH management and nutrient optimization
            - Organic matter improvement strategies
            - Micronutrient deficiency corrections
            - Sustainable soil management practices
            
            Include specific testing procedures, amendment quantities,
            and implementation timelines. Focus on cost-effective solutions."""
        )
        
        # Financial Advisory Expert
        self.finance_agent = AGNOAgent(
            name="Agricultural Finance Advisor",
            role="Expert in agricultural finance and government schemes",
            specialization="loan credit scheme subsidy insurance financial",
            instructions="""Provide financial guidance:
            - Government schemes (PM-KISAN, KCC, PMFBY, MUDRA)
            - Loan eligibility and application processes
            - Insurance coverage and procedures
            - Investment planning and optimization
            - Subsidy calculations and benefits
            
            Include specific eligibility criteria, application steps,
            contact information, and financial projections."""
        )
        
        # Pest Management Specialist
        self.pest_agent = AGNOAgent(
            name="IPM Specialist",
            role="Expert in Integrated Pest Management",
            specialization="pest disease IPM biological pesticide",
            instructions="""Provide pest management expertise:
            - Integrated Pest Management strategies
            - Biological control methods
            - Sustainable pesticide usage
            - Disease identification and treatment
            - Preventive measures and monitoring
            
            Prioritize eco-friendly solutions.
            Include application rates, timing, and safety measures."""
        )
        
        # Water Management Engineer
        self.irrigation_agent = AGNOAgent(
            name="Water Management Engineer",
            role="Expert in irrigation and water management",
            specialization="irrigation water drip sprinkler conservation",
            instructions="""Provide water management guidance:
            - Irrigation system selection and design
            - Water scheduling and requirements
            - Conservation techniques
            - Government subsidies for irrigation
            - Groundwater management
            
            Include system costs, ROI calculations,
            and installation procedures."""
        )
        
        # Market Intelligence Analyst
        self.market_agent = AGNOAgent(
            name="Market Intelligence Analyst",
            role="Expert in agricultural markets and trading",
            specialization="market price trading commodity MSP",
            instructions="""Provide market intelligence:
            - Current commodity prices and trends
            - Market timing strategies
            - Value-addition opportunities
            - Export potential and requirements
            - Government procurement schemes
            
            Include specific price data, forecasts,
            and trading recommendations."""
        )
        
        # Supply Chain Expert
        self.supply_agent = AGNOAgent(
            name="Supply Chain Expert",
            role="Specialist in agricultural supply chains",
            specialization="supply storage certification organic export",
            instructions="""Provide supply chain guidance:
            - Input sourcing and evaluation
            - Storage and post-harvest management
            - Certification processes
            - Quality assurance systems
            - Direct marketing channels
            
            Include supplier contacts, procedures,
            and compliance requirements."""
        )
    
    def setup_teams(self):
        """Setup coordinated agent teams"""
        
        # Farm Planning Team
        self.farm_planning_team = AGNOTeam(
            name="Farm Planning Team",
            agents=[self.crop_agent, self.soil_agent, self.weather_agent, self.irrigation_agent],
            coordination_strategy="Integrate crop, soil, weather, and water management for comprehensive farm planning"
        )
        
        # Financial Strategy Team
        self.financial_team = AGNOTeam(
            name="Financial Strategy Team",
            agents=[self.finance_agent, self.market_agent, self.supply_agent],
            coordination_strategy="Combine financial planning, market intelligence, and supply chain optimization"
        )
        
        # Crop Protection Team
        self.protection_team = AGNOTeam(
            name="Crop Protection Team",
            agents=[self.pest_agent, self.weather_agent, self.soil_agent],
            coordination_strategy="Integrate pest management with weather and soil considerations"
        )
    
    def route_query(self, query: str, user_context: Dict) -> str:
        """Route query to appropriate agent or team"""
        try:
            query_lower = query.lower()
            
            # Enhanced context with query analysis
            enhanced_context = {
                **user_context,
                "query_analysis": {
                    "original_query": query,
                    "detected_keywords": [word for word in query_lower.split() if len(word) > 3]
                }
            }
            
            # Route to specialized agents
            if any(keyword in query_lower for keyword in ["soil", "ph", "nutrient", "testing", "fertilizer"]):
                return self.soil_agent.run(query, enhanced_context)
            
            elif any(keyword in query_lower for keyword in ["pest", "disease", "spray", "insect", "ipm"]):
                return self.pest_agent.run(query, enhanced_context)
            
            elif any(keyword in query_lower for keyword in ["irrigation", "water", "drip", "sprinkler"]):
                return self.irrigation_agent.run(query, enhanced_context)
            
            elif any(keyword in query_lower for keyword in ["weather", "rain", "forecast", "climate", "season"]):
                return self.weather_agent.run(query, enhanced_context)
            
            elif any(keyword in query_lower for keyword in ["loan", "credit", "scheme", "subsidy", "finance"]):
                return self.finance_agent.run(query, enhanced_context)
            
            elif any(keyword in query_lower for keyword in ["price", "market", "sell", "buy", "trading"]):
                return self.market_agent.run(query, enhanced_context)
            
            elif any(keyword in query_lower for keyword in ["seed", "variety", "crop", "planting", "cultivation"]):
                return self.crop_agent.run(query, enhanced_context)
            
            elif any(keyword in query_lower for keyword in ["organic", "certification", "export", "supply"]):
                return self.supply_agent.run(query, enhanced_context)
            
            # Route to teams for complex queries
            elif any(keyword in query_lower for keyword in ["planning", "setup", "farm", "comprehensive"]):
                return self.farm_planning_team.run(query, enhanced_context)
            
            elif any(keyword in query_lower for keyword in ["investment", "profitability", "business", "roi"]):
                return self.financial_team.run(query, enhanced_context)
            
            elif any(keyword in query_lower for keyword in ["protection", "treatment", "problem", "management"]):
                return self.protection_team.run(query, enhanced_context)
            
            else:
                # Default to crop agent for general queries
                return self.crop_agent.run(query, enhanced_context)
                
        except Exception as e:
            return f"Error processing query: {str(e)}"
    
    def get_agent_status(self) -> Dict:
        """Get system status"""
        return {
            "total_agents": 8,
            "teams": 3,
            "groq_configured": bool(self.groq_api_key),
            "agno_framework": True,
            "search_enabled": True,
            "agents": {
                "crop": "Crop Management Specialist",
                "weather": "Weather & Climate Advisor",
                "finance": "Agricultural Finance Advisor",
                "soil": "Soil Health Specialist", 
                "pest": "IPM Specialist",
                "irrigation": "Water Management Engineer",
                "market": "Market Intelligence Analyst",
                "supply": "Supply Chain Expert"
            },
            "teams": {
                "farm_planning": "Farm Planning Team",
                "financial": "Financial Strategy Team",
                "protection": "Crop Protection Team"
            }
        }