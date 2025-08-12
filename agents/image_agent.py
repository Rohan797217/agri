import numpy as np
from PIL import Image, ImageEnhance
import io
import base64
from typing import Dict, Any, List, Tuple
from agents.base_agent import BaseAgent

class ImageAgent(BaseAgent):
    """Specialized agent for image analysis and crop disease detection"""
    
    def __init__(self):
        super().__init__("ImageAgent")
        self.disease_database = self.load_disease_database()
    
    def load_disease_database(self) -> Dict:
        """Load crop disease information database"""
        return {
            "rice": {
                "blast": {
                    "symptoms": "Gray-green lesions with white centers on leaves",
                    "treatment": "Fungicide application, resistant varieties",
                    "prevention": "Proper spacing, avoid excess nitrogen"
                },
                "brown_spot": {
                    "symptoms": "Brown spots with yellow halo on leaves",
                    "treatment": "Seed treatment, balanced fertilization",
                    "prevention": "Use certified seeds, proper drainage"
                }
            },
            "wheat": {
                "rust": {
                    "symptoms": "Orange/brown pustules on leaves and stems",
                    "treatment": "Fungicide spray, resistant varieties",
                    "prevention": "Early sowing, proper ventilation"
                },
                "powdery_mildew": {
                    "symptoms": "White powdery coating on leaves",
                    "treatment": "Fungicide application",
                    "prevention": "Avoid dense planting, good air circulation"
                }
            },
            "tomato": {
                "blight": {
                    "symptoms": "Dark spots with concentric rings on leaves",
                    "treatment": "Remove affected parts, apply fungicide",
                    "prevention": "Crop rotation, avoid overhead watering"
                },
                "wilt": {
                    "symptoms": "Yellowing and wilting of leaves",
                    "treatment": "Soil fumigation, resistant varieties",
                    "prevention": "Proper drainage, crop rotation"
                }
            }
        }
    
    def analyze_image(self, image: Image.Image, analysis_type: str, 
                     additional_context: str = "") -> str:
        """Analyze uploaded crop image"""
        try:
            # Basic image processing
            processed_image = self.preprocess_image(image)
            image_features = self.extract_image_features(processed_image)
            
            # Generate analysis based on type
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
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize for consistent processing
            if image.size[0] > 1024 or image.size[1] > 1024:
                image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            
            # Extract basic properties
            width, height = image.size
            
            # Convert to numpy array for analysis
            img_array = np.array(image)
            
            return {
                "image": image,
                "array": img_array,
                "width": width,
                "height": height,
                "channels": img_array.shape[2] if len(img_array.shape) == 3 else 1
            }
            
        except Exception as e:
            raise Exception(f"Image preprocessing failed: {str(e)}")
    
    def extract_image_features(self, processed_image: Dict) -> Dict[str, Any]:
        """Extract features from processed image"""
        try:
            img_array = processed_image["array"]
            
            # Color analysis
            mean_rgb = np.mean(img_array, axis=(0, 1))
            
            # Simple color-based features
            green_dominance = mean_rgb[1] / np.sum(mean_rgb)  # Green channel dominance
            brown_pixels = self.detect_brown_areas(img_array)
            yellow_pixels = self.detect_yellow_areas(img_array)
            
            # Brightness and contrast
            brightness = np.mean(img_array)
            contrast = np.std(img_array)
            
            return {
                "mean_rgb": mean_rgb.tolist(),
                "green_dominance": green_dominance,
                "brown_percentage": brown_pixels,
                "yellow_percentage": yellow_pixels,
                "brightness": brightness,
                "contrast": contrast,
                "image_quality": "good" if contrast > 30 else "poor"
            }
            
        except Exception as e:
            return {"error": f"Feature extraction failed: {str(e)}"}
    
    def detect_brown_areas(self, img_array: np.ndarray) -> float:
        """Detect brown/diseased areas in image"""
        try:
            # Simple brown color detection (HSV would be better)
            # Brown is typically low brightness with red > green > blue
            mask = (img_array[:, :, 0] > img_array[:, :, 1]) & \
                   (img_array[:, :, 1] > img_array[:, :, 2]) & \
                   (img_array[:, :, 0] < 150)  # Not too bright
            
            return np.sum(mask) / img_array[:, :, 0].size * 100
            
        except Exception:
            return 0.0
    
    def detect_yellow_areas(self, img_array: np.ndarray) -> float:
        """Detect yellow/stressed areas in image"""
        try:
            # Yellow detection: high red and green, low blue
            mask = (img_array[:, :, 0] > 100) & \
                   (img_array[:, :, 1] > 100) & \
                   (img_array[:, :, 2] < 80)
            
            return np.sum(mask) / img_array[:, :, 0].size * 100
            
        except Exception:
            return 0.0
    
    def detect_disease(self, image_features: Dict, context: str) -> str:
        """Detect potential diseases from image features"""
        try:
            system_message = """You are a plant pathologist specializing in crop disease diagnosis. 
            Analyze the provided image features and context to identify potential diseases. 
            Provide specific recommendations for treatment and prevention."""
            
            # Build analysis prompt
            prompt = f"""
            Analyze this crop image for disease detection:
            
            Image Features:
            - Green dominance: {image_features.get('green_dominance', 'unknown'):.2f}
            - Brown areas: {image_features.get('brown_percentage', 0):.1f}%
            - Yellow areas: {image_features.get('yellow_percentage', 0):.1f}%
            - Brightness: {image_features.get('brightness', 0):.1f}
            - Image quality: {image_features.get('image_quality', 'unknown')}
            
            Additional context: {context}
            
            Provide:
            1. Possible disease identification
            2. Confidence level of diagnosis
            3. Immediate treatment recommendations
            4. Preventive measures
            5. When to seek expert help
            6. Expected recovery timeline
            """
            
            # Generate AI response
            ai_response = self.generate_response(prompt, system_message)
            
            # Add feature-based analysis
            feature_analysis = self.analyze_features_for_disease(image_features)
            
            return f"{feature_analysis}\n\n**AI Analysis:**\n{ai_response}"
            
        except Exception as e:
            return f"Error in disease detection: {str(e)}"
    
    def analyze_features_for_disease(self, features: Dict) -> str:
        """Analyze image features for disease indicators"""
        analysis_parts = ["**📊 Image Analysis Results:**"]
        
        green_dominance = features.get('green_dominance', 0)
        brown_percentage = features.get('brown_percentage', 0)
        yellow_percentage = features.get('yellow_percentage', 0)
        
        # Health indicators
        if green_dominance > 0.4:
            analysis_parts.append("✅ Good green foliage detected")
        else:
            analysis_parts.append("⚠️ Reduced green foliage - possible stress")
        
        # Disease indicators
        if brown_percentage > 10:
            analysis_parts.append(f"🚨 High brown/diseased areas detected ({brown_percentage:.1f}%)")
        elif brown_percentage > 5:
            analysis_parts.append(f"⚠️ Moderate brown areas detected ({brown_percentage:.1f}%)")
        
        if yellow_percentage > 15:
            analysis_parts.append(f"⚠️ Yellowing detected ({yellow_percentage:.1f}%) - possible nutrient deficiency")
        
        # Image quality
        image_quality = features.get('image_quality', 'unknown')
        if image_quality == 'poor':
            analysis_parts.append("📷 Image quality is poor - consider taking clearer photos")
        
        return "\n".join(analysis_parts)
    
    def assess_crop_health(self, image_features: Dict, context: str) -> str:
        """Assess overall crop health from image"""
        try:
            system_message = """You are a crop health specialist. Assess the overall health and 
            vigor of crops based on visual indicators. Provide recommendations for improving crop health."""
            
            prompt = f"""
            Assess crop health based on these image features:
            
            {image_features}
            
            Context: {context}
            
            Provide:
            1. Overall health assessment (Excellent/Good/Fair/Poor)
            2. Specific health indicators observed
            3. Areas of concern
            4. Recommendations for improvement
            5. Nutritional status assessment
            6. Growth stage evaluation if possible
            """
            
            ai_response = self.generate_response(prompt, system_message)
            
            # Add quantitative health score
            health_score = self.calculate_health_score(image_features)
            
            return f"**Health Score: {health_score}/100**\n\n{ai_response}"
            
        except Exception as e:
            return f"Error in crop health assessment: {str(e)}"
    
    def calculate_health_score(self, features: Dict) -> int:
        """Calculate a simple health score based on image features"""
        try:
            score = 100
            
            # Reduce score for disease indicators
            brown_percentage = features.get('brown_percentage', 0)
            yellow_percentage = features.get('yellow_percentage', 0)
            green_dominance = features.get('green_dominance', 0)
            
            # Penalties
            score -= min(brown_percentage * 3, 30)  # Max 30 points for brown areas
            score -= min(yellow_percentage * 2, 20)  # Max 20 points for yellowing
            
            # Bonus for good green coverage
            if green_dominance > 0.4:
                score += 10
            
            return max(0, min(100, int(score)))
            
        except Exception:
            return 50  # Default moderate score
    
    def general_image_analysis(self, image_features: Dict, context: str) -> str:
        """Perform general agricultural image analysis"""
        try:
            system_message = """You are an agricultural image analysis expert. Provide comprehensive 
            analysis of crop images including growth stage, health status, and management recommendations."""
            
            prompt = f"""
            Perform general analysis of this agricultural image:
            
            Image Features: {image_features}
            Context: {context}
            
            Provide:
            1. Crop identification if possible
            2. Growth stage assessment
            3. General health observations
            4. Recommended actions
            5. Best practices suggestions
            6. Optimal timing for next steps
            """
            
            return self.generate_response(prompt, system_message)
            
        except Exception as e:
            return f"Error in general image analysis: {str(e)}"
    
    def get_treatment_recommendations(self, disease: str, crop: str) -> str:
        """Get specific treatment recommendations"""
        try:
            # Check disease database
            if crop.lower() in self.disease_database:
                crop_diseases = self.disease_database[crop.lower()]
                
                for disease_key, disease_info in crop_diseases.items():
                    if disease.lower() in disease_key or disease_key in disease.lower():
                        return self.format_treatment_info(disease_key, disease_info)
            
            # Fallback to AI generation
            system_message = """You are a plant protection specialist. Provide specific treatment 
            protocols for crop diseases including organic and chemical options."""
            
            prompt = f"""
            Provide detailed treatment recommendations for {disease} in {crop}:
            
            Include:
            1. Immediate treatment steps
            2. Organic treatment options
            3. Chemical treatment protocols
            4. Application timings and rates
            5. Safety precautions
            6. Expected results timeline
            """
            
            return self.generate_response(prompt, system_message)
            
        except Exception as e:
            return f"Error getting treatment recommendations: {str(e)}"
    
    def format_treatment_info(self, disease: str, disease_info: Dict) -> str:
        """Format disease treatment information"""
        info_parts = [f"**Treatment for {disease.replace('_', ' ').title()}:**"]
        
        if "symptoms" in disease_info:
            info_parts.append(f"**Symptoms:** {disease_info['symptoms']}")
        
        if "treatment" in disease_info:
            info_parts.append(f"**Treatment:** {disease_info['treatment']}")
        
        if "prevention" in disease_info:
            info_parts.append(f"**Prevention:** {disease_info['prevention']}")
        
        return "\n".join(info_parts)
