import json
from typing import Dict, Any, List
from agents.base_agent import BaseAgent

class FinanceAgent(BaseAgent):
    """Specialized agent for agricultural finance and policy information"""
    
    def __init__(self):
        super().__init__("FinanceAgent")
        self.financial_schemes = self.load_financial_schemes()
    
    def load_financial_schemes(self) -> Dict:
        """Load information about financial schemes and programs"""
        return {
            "pm_kisan": {
                "name": "PM-KISAN (Pradhan Mantri Kisan Samman Nidhi)",
                "benefit": "₹6,000 per year in three installments",
                "eligibility": "Small and marginal farmers with cultivable land",
                "application": "Online through PM-KISAN portal or CSC centers"
            },
            "kisan_credit_card": {
                "name": "Kisan Credit Card (KCC)",
                "benefit": "Flexible credit for agriculture and allied activities",
                "eligibility": "Farmers with land ownership or tenant farmers",
                "features": "4% interest rate, insurance coverage"
            },
            "fasal_bima": {
                "name": "Pradhan Mantri Fasal Bima Yojana (PMFBY)",
                "benefit": "Crop insurance against natural calamities",
                "premium": "2% for Kharif, 1.5% for Rabi crops",
                "coverage": "Sum insured based on scale of finance"
            },
            "mudra_loan": {
                "name": "MUDRA Loan for Agriculture",
                "benefit": "Loans up to ₹10 lakh for micro enterprises",
                "categories": "Shishu (up to ₹50,000), Kishore (₹50,000-₹5 lakh), Tarun (₹5-10 lakh)",
                "interest": "Variable, typically 8-12% per annum"
            }
        }
    
    def get_finance_advice(self, query: str, user_context: Dict) -> str:
        """Generate financial advice and scheme information"""
        try:
            location = user_context.get("location", "")
            farm_size = user_context.get("farm_size", 0)
            crops = user_context.get("crops", [])
            
            # Build context for financial advice
            system_message = """You are a financial advisor specializing in agricultural finance. 
            Provide accurate information about government schemes, loans, subsidies, and financial planning for farmers. 
            Always include eligibility criteria, application processes, and contact information when relevant."""
            
            # Enhance query with user context and scheme database
            enhanced_query = self.build_financial_query(query, location, farm_size, crops)
            
            # Generate response
            response = self.generate_response(enhanced_query, system_message)
            
            # Add relevant scheme information
            relevant_schemes = self.find_relevant_schemes(query, farm_size)
            if relevant_schemes:
                scheme_info = self.format_scheme_information(relevant_schemes)
                response += f"\n\n{scheme_info}"
            
            # Add disclaimer
            response += self.add_financial_disclaimer()
            
            return response
            
        except Exception as e:
            fallback_response = self.web_search_fallback(f"agriculture finance schemes {query}", "policy")
            if fallback_response != "No relevant information found through web search.":
                return f"Using fallback data:\n{fallback_response}"
            return f"Error generating finance advice: {str(e)}"
    
    def build_financial_query(self, query: str, location: str, farm_size: float, crops: List) -> str:
        """Build enhanced financial query with user context"""
        enhanced_parts = [query]
        
        if farm_size > 0:
            if farm_size <= 2:
                enhanced_parts.append("User is a small/marginal farmer (≤2 acres).")
            elif farm_size <= 5:
                enhanced_parts.append("User is a small farmer (2-5 acres).")
            else:
                enhanced_parts.append("User is a medium/large farmer (>5 acres).")
        
        if location:
            enhanced_parts.append(f"User location: {location}")
        
        if crops:
            enhanced_parts.append(f"Crops grown: {', '.join(crops)}")
        
        return " ".join(enhanced_parts)
    
    def find_relevant_schemes(self, query: str, farm_size: float) -> List[str]:
        """Find relevant financial schemes based on query and user profile"""
        relevant = []
        query_lower = query.lower()
        
        # Check for specific scheme mentions
        if "pm kisan" in query_lower or "pmkisan" in query_lower or "6000" in query_lower:
            relevant.append("pm_kisan")
        
        if "credit card" in query_lower or "kcc" in query_lower or "loan" in query_lower:
            relevant.append("kisan_credit_card")
        
        if "insurance" in query_lower or "crop insurance" in query_lower or "fasal bima" in query_lower:
            relevant.append("fasal_bima")
        
        if "mudra" in query_lower or "micro finance" in query_lower:
            relevant.append("mudra_loan")
        
        # If no specific schemes mentioned, suggest based on farm size
        if not relevant:
            if farm_size <= 2:  # Small/marginal farmers
                relevant = ["pm_kisan", "fasal_bima", "kisan_credit_card"]
            else:
                relevant = ["kisan_credit_card", "fasal_bima", "mudra_loan"]
        
        return relevant
    
    def format_scheme_information(self, scheme_keys: List[str]) -> str:
        """Format detailed scheme information"""
        info_parts = ["**💰 Relevant Financial Schemes:**"]
        
        for scheme_key in scheme_keys:
            if scheme_key in self.financial_schemes:
                scheme = self.financial_schemes[scheme_key]
                info_parts.append(f"\n**{scheme['name']}**")
                info_parts.append(f"• Benefit: {scheme['benefit']}")
                
                if "eligibility" in scheme:
                    info_parts.append(f"• Eligibility: {scheme['eligibility']}")
                
                if "application" in scheme:
                    info_parts.append(f"• Application: {scheme['application']}")
                
                if "premium" in scheme:
                    info_parts.append(f"• Premium: {scheme['premium']}")
                
                if "interest" in scheme:
                    info_parts.append(f"• Interest Rate: {scheme['interest']}")
        
        return "\n".join(info_parts)
    
    def add_financial_disclaimer(self) -> str:
        """Add standard financial disclaimer"""
        return """
        
**⚠️ Important Disclaimer:**
Financial schemes and eligibility criteria may change. Please verify current information from:
• Official government websites
• Local bank branches
• Agricultural extension officers
• Common Service Centers (CSC)
        """
    
    def calculate_loan_eligibility(self, farm_size: float, annual_income: float, 
                                 existing_loans: float = 0) -> Dict[str, Any]:
        """Calculate approximate loan eligibility"""
        try:
            # Basic eligibility calculation (simplified)
            land_value_per_acre = 200000  # Approximate value
            total_land_value = farm_size * land_value_per_acre
            
            # KCC calculation (simplified)
            kcc_limit = min(total_land_value * 0.1, 300000)  # 10% of land value or ₹3 lakh
            
            # MUDRA loan eligibility
            mudra_eligibility = min(annual_income * 2, 1000000)  # 2x annual income or ₹10 lakh
            
            # Factor in existing loans
            available_kcc = max(0, kcc_limit - existing_loans)
            available_mudra = max(0, mudra_eligibility - existing_loans)
            
            return {
                "kcc_eligible_amount": available_kcc,
                "mudra_eligible_amount": available_mudra,
                "total_land_value": total_land_value,
                "debt_to_income_ratio": existing_loans / annual_income if annual_income > 0 else 0
            }
            
        except Exception as e:
            return {"error": f"Error calculating eligibility: {str(e)}"}
    
    def get_subsidy_information(self, crop_type: str, location: str) -> str:
        """Get subsidy information for specific crops and locations"""
        try:
            system_message = """You are an expert on agricultural subsidies in India. 
            Provide accurate, up-to-date information about subsidies available for specific crops and regions."""
            
            prompt = f"""
            Provide subsidy information for:
            Crop: {crop_type}
            Location: {location}
            
            Include:
            1. Central government subsidies
            2. State government schemes
            3. Input subsidies (seeds, fertilizers, equipment)
            4. Application procedures
            5. Required documents
            """
            
            return self.generate_response(prompt, system_message)
            
        except Exception as e:
            return f"Error getting subsidy information: {str(e)}"
