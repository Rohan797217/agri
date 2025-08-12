from typing import Dict, Any, List
from agents.base_agent import BaseAgent

class PolicyAgent(BaseAgent):
    """Specialized agent for agricultural policies and government schemes"""
    
    def __init__(self):
        super().__init__("PolicyAgent")
        self.policy_database = self.load_policy_information()
    
    def load_policy_information(self) -> Dict:
        """Load basic policy and scheme information"""
        return {
            "central_schemes": {
                "pm_kisan": "Direct income support of ₹6000/year to farmer families",
                "pm_fasal_bima": "Comprehensive crop insurance scheme",
                "soil_health_card": "Soil testing and nutrient management",
                "pm_kusum": "Solar pump and grid-connected solar power scheme",
                "paramparagat_krishi": "Organic farming promotion scheme",
                "rashtriya_gokul_mission": "Cattle development and conservation"
            },
            "recent_policies": {
                "msp_2024": "Minimum Support Price announced for major crops",
                "digital_agriculture": "Technology adoption in farming practices",
                "fpo_promotion": "Farmer Producer Organizations support",
                "export_promotion": "Agricultural export incentive schemes"
            }
        }
    
    def get_policy_advice(self, query: str, user_context: Dict) -> str:
        """Generate policy-related advice and information"""
        try:
            location = user_context.get("location", "")
            crops = user_context.get("crops", [])
            farm_size = user_context.get("farm_size", 0)
            
            # Build enhanced context
            system_message = """You are a policy expert specializing in Indian agricultural policies. 
            Provide accurate, up-to-date information about government schemes, policies, and regulations. 
            Include implementation details, eligibility criteria, and application processes."""
            
            enhanced_query = self.build_policy_query(query, location, crops, farm_size)
            
            # Generate response
            response = self.generate_response(enhanced_query, system_message)
            
            # Add relevant policy information
            relevant_policies = self.find_relevant_policies(query)
            if relevant_policies:
                policy_info = self.format_policy_information(relevant_policies)
                response += f"\n\n{policy_info}"
            
            return response
            
        except Exception as e:
            fallback_response = self.web_search_fallback(f"agriculture policy {query}", "policy")
            if fallback_response != "No relevant information found through web search.":
                return f"Using fallback data:\n{fallback_response}"
            return f"Error generating policy advice: {str(e)}"
    
    def build_policy_query(self, query: str, location: str, crops: List, farm_size: float) -> str:
        """Build enhanced policy query with context"""
        enhanced_parts = [query]
        
        if location:
            enhanced_parts.append(f"Location context: {location}")
        
        if crops:
            enhanced_parts.append(f"Relevant crops: {', '.join(crops)}")
        
        if farm_size > 0:
            size_category = "small/marginal" if farm_size <= 2 else "medium/large"
            enhanced_parts.append(f"Farm size category: {size_category} farmer")
        
        return " ".join(enhanced_parts)
    
    def find_relevant_policies(self, query: str) -> List[str]:
        """Find policies relevant to the query"""
        relevant = []
        query_lower = query.lower()
        
        # Check for scheme mentions
        policy_keywords = {
            "pm_kisan": ["pm kisan", "pmkisan", "income support", "6000"],
            "pm_fasal_bima": ["fasal bima", "crop insurance", "insurance"],
            "soil_health_card": ["soil health", "soil card", "soil testing"],
            "pm_kusum": ["kusum", "solar pump", "solar power"],
            "paramparagat_krishi": ["organic", "organic farming", "paramparagat"],
            "msp_2024": ["msp", "minimum support price", "procurement"],
            "fpo_promotion": ["fpo", "farmer producer", "collective farming"]
        }
        
        for policy, keywords in policy_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                relevant.append(policy)
        
        return relevant
    
    def format_policy_information(self, policy_keys: List[str]) -> str:
        """Format policy information for display"""
        info_parts = ["**📋 Relevant Government Policies & Schemes:**"]
        
        for policy_key in policy_keys:
            # Check central schemes
            if policy_key in self.policy_database.get("central_schemes", {}):
                scheme_info = self.policy_database["central_schemes"][policy_key]
                info_parts.append(f"• **{policy_key.upper()}:** {scheme_info}")
            
            # Check recent policies
            elif policy_key in self.policy_database.get("recent_policies", {}):
                policy_info = self.policy_database["recent_policies"][policy_key]
                info_parts.append(f"• **{policy_key.upper()}:** {policy_info}")
        
        return "\n".join(info_parts)
    
    def analyze_document(self, document_content: str, question: str) -> str:
        """Analyze policy documents and answer specific questions"""
        try:
            system_message = """You are a policy analyst. Analyze the provided document and answer 
            the specific question accurately based on the document content. If the information is not 
            in the document, clearly state that."""
            
            prompt = f"""
            Document content: {document_content[:3000]}...
            
            Question: {question}
            
            Provide a detailed answer based on the document content. Include:
            1. Direct answer to the question
            2. Relevant sections or clauses from the document
            3. Any eligibility criteria or conditions mentioned
            4. Implementation details if available
            """
            
            return self.generate_response(prompt, system_message)
            
        except Exception as e:
            return f"Error analyzing document: {str(e)}"
    
    def summarize_document(self, document_content: str) -> str:
        """Summarize policy documents"""
        try:
            system_message = """You are a policy summarization expert. Create clear, concise summaries 
            of policy documents that are easy to understand for farmers and agricultural stakeholders."""
            
            prompt = f"""
            Summarize this policy document in a farmer-friendly format:
            
            {document_content[:4000]}...
            
            Include:
            1. Main objectives and benefits
            2. Eligibility criteria
            3. How to apply or participate
            4. Key deadlines or timelines
            5. Contact information for more details
            """
            
            return self.generate_response(prompt, system_message)
            
        except Exception as e:
            return f"Error summarizing document: {str(e)}"
    
    def get_state_specific_schemes(self, state: str) -> str:
        """Get state-specific agricultural schemes"""
        try:
            system_message = """You are an expert on state-specific agricultural schemes in India. 
            Provide accurate information about schemes specific to different Indian states."""
            
            prompt = f"""
            Provide information about agricultural schemes specific to {state}:
            
            Include:
            1. Major state government schemes for farmers
            2. Subsidies and incentives available
            3. State-specific crop promotion programs
            4. Water management and irrigation schemes
            5. Market linkage initiatives
            6. Application procedures and contact details
            """
            
            return self.generate_response(prompt, system_message)
            
        except Exception as e:
            return f"Error getting state-specific schemes: {str(e)}"
    
    def check_policy_updates(self, policy_area: str) -> str:
        """Check for recent policy updates in specific areas"""
        try:
            system_message = """You are a policy tracking expert. Provide information about recent 
            updates and changes in Indian agricultural policies."""
            
            prompt = f"""
            Provide recent updates (2024) in {policy_area} policy area:
            
            Include:
            1. Recent policy announcements
            2. Changes in existing schemes
            3. New initiatives launched
            4. Budget allocations and changes
            5. Implementation timeline
            """
            
            response = self.generate_response(prompt, system_message)
            
            # Add disclaimer about information currency
            disclaimer = """
            
**📅 Information Currency Note:**
Policy information may change frequently. For the most current updates:
• Visit official government portals
• Contact local agricultural departments
• Check with agricultural extension officers
            """
            
            return response + disclaimer
            
        except Exception as e:
            return f"Error checking policy updates: {str(e)}"
