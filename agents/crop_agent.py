import json
from typing import Dict, Any, List
from agents.base_agent import BaseAgent

class CropAgent(BaseAgent):
    """Specialized agent for crop-related advice and recommendations"""
    
    def __init__(self):
        super().__init__("CropAgent")
        self.crop_database = self.load_crop_knowledge()
    
    def load_crop_knowledge(self) -> Dict:
        """Load basic crop knowledge database"""
        return {
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
            },
            "maize": {
                "season": ["Kharif", "Rabi"],
                "water_requirement": "Moderate",
                "soil_type": "Well-drained loam",
                "diseases": ["Downy mildew", "Leaf blight", "Stalk rot"],
                "fertilizer": "NPK 120:60:40 kg/ha"
            }
        }
    
    def get_crop_advice(self, query: str, entities: Dict, user_context: Dict) -> str:
        """Generate comprehensive crop advice"""
        try:
            # Extract crop information from entities
            detected_crop = self.detect_crop_from_query(query)
            user_location = user_context.get("location", "")
            user_crops = user_context.get("crops", [])
            
            # Build context for LLM
            system_message = """You are an expert agricultural advisor specializing in crop management. 
            Provide practical, actionable advice based on scientific principles and local conditions.
            Always mention specific timings, quantities, and methods when relevant.
            If uncertain about specific local conditions, recommend consulting local agricultural extension officers."""
            
            # Enhance prompt with available data
            enhanced_query = self.build_enhanced_query(query, detected_crop, user_location, user_crops)
            
            # Generate response
            response = self.generate_response(enhanced_query, system_message)
            
            # Add crop-specific recommendations
            if detected_crop and detected_crop in self.crop_database:
                crop_info = self.crop_database[detected_crop]
                additional_info = self.format_crop_specific_info(detected_crop, crop_info)
                response += f"\n\n{additional_info}"
            
            # Validate and add warnings if needed
            quality_check = self.validate_response_quality(response)
            if quality_check.get("warning"):
                response += f"\n\n{quality_check['warning']}"
            
            return response
            
        except Exception as e:
            return f"Error generating crop advice: {str(e)}. Please try rephrasing your question."
    
    def detect_crop_from_query(self, query: str) -> str:
        """Detect crop mentioned in the query"""
        query_lower = query.lower()
        
        for crop in self.crop_database.keys():
            if crop in query_lower:
                return crop
        
        # Check for common aliases
        crop_aliases = {
            "paddy": "rice",
            "corn": "maize",
            "kapas": "cotton"
        }
        
        for alias, crop in crop_aliases.items():
            if alias in query_lower:
                return crop
        
        return None
    
    def build_enhanced_query(self, query: str, crop: str, location: str, user_crops: List) -> str:
        """Build enhanced query with context"""
        enhanced_parts = [query]
        
        if crop:
            enhanced_parts.append(f"The query is specifically about {crop} crop.")
        
        if location:
            enhanced_parts.append(f"The user is located in {location}.")
        
        if user_crops:
            enhanced_parts.append(f"The user grows: {', '.join(user_crops)}.")
        
        return " ".join(enhanced_parts)
    
    def format_crop_specific_info(self, crop: str, crop_info: Dict) -> str:
        """Format crop-specific technical information"""
        info_parts = [f"**{crop.title()} Crop Information:**"]
        
        if "season" in crop_info:
            seasons = ", ".join(crop_info["season"])
            info_parts.append(f"• **Growing Season:** {seasons}")
        
        if "water_requirement" in crop_info:
            info_parts.append(f"• **Water Requirement:** {crop_info['water_requirement']}")
        
        if "soil_type" in crop_info:
            info_parts.append(f"• **Suitable Soil:** {crop_info['soil_type']}")
        
        if "fertilizer" in crop_info:
            info_parts.append(f"• **Recommended Fertilizer:** {crop_info['fertilizer']}")
        
        if "diseases" in crop_info:
            diseases = ", ".join(crop_info["diseases"])
            info_parts.append(f"• **Common Diseases:** {diseases}")
        
        return "\n".join(info_parts)
    
    def get_irrigation_advice(self, crop: str, growth_stage: str, weather_context: Dict) -> str:
        """Provide irrigation-specific advice"""
        try:
            system_message = """You are an irrigation specialist. Provide specific irrigation advice 
            considering crop type, growth stage, weather conditions, and water conservation principles."""
            
            prompt = f"""
            Provide irrigation advice for {crop} crop at {growth_stage} stage.
            Weather context: {weather_context}
            Include:
            1. Optimal irrigation timing
            2. Water quantity recommendations
            3. Irrigation method suggestions
            4. Signs to watch for water stress
            """
            
            return self.generate_response(prompt, system_message)
            
        except Exception as e:
            return f"Error generating irrigation advice: {str(e)}"
    
    def get_pest_disease_advice(self, crop: str, symptoms: str, image_analysis: Dict = None) -> str:
        """Provide pest and disease management advice"""
        try:
            system_message = """You are a plant pathologist and entomologist. Provide accurate 
            pest and disease identification and management advice. Always recommend integrated 
            pest management approaches."""
            
            prompt = f"""
            Analyze pest/disease issue for {crop} crop.
            Symptoms described: {symptoms}
            """
            
            if image_analysis:
                prompt += f"Image analysis results: {image_analysis}"
            
            prompt += """
            Provide:
            1. Possible pest/disease identification
            2. Immediate action steps
            3. Preventive measures
            4. Organic and chemical control options
            5. When to seek expert help
            """
            
            return self.generate_response(prompt, system_message)
            
        except Exception as e:
            return f"Error generating pest/disease advice: {str(e)}"
