import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import pandas as pd

class KnowledgeBase:
    """Comprehensive agricultural knowledge base for Indian farming"""
    
    def __init__(self):
        self.crop_data = self.load_crop_database()
        self.weather_patterns = self.load_weather_patterns()
        self.government_schemes = self.load_government_schemes()
        self.market_data = self.load_market_information()
        self.disease_database = self.load_disease_database()
        self.fertilizer_guide = self.load_fertilizer_guide()
        self.irrigation_guide = self.load_irrigation_guide()
        self.seasonal_calendar = self.load_seasonal_calendar()
        self.regional_data = self.load_regional_data()
    
    def load_crop_database(self) -> Dict[str, Dict]:
        """Load comprehensive crop information database"""
        return {
            "rice": {
                "scientific_name": "Oryza sativa",
                "local_names": {
                    "hindi": "धान",
                    "marathi": "तांदूळ",
                    "tamil": "அரிசி",
                    "telugu": "వరి",
                    "gujarati": "ચોખા"
                },
                "varieties": [
                    "Basmati", "IR64", "Swarna", "BPT 5204", "Sona Masuri", 
                    "MTU 1010", "Pusa 1121", "CSR 30"
                ],
                "growing_conditions": {
                    "temperature_range": "20-35°C",
                    "rainfall": "1000-2000mm",
                    "soil_ph": "5.5-7.0",
                    "soil_type": "Clay, Clay loam",
                    "altitude": "0-2000m"
                },
                "cultivation": {
                    "seasons": ["Kharif", "Rabi"],
                    "duration": "120-150 days",
                    "seed_rate": "25-30 kg/ha",
                    "plant_spacing": "20cm x 15cm",
                    "depth": "2-3 cm"
                },
                "fertilizer": {
                    "recommended_dose": "NPK 120:60:40 kg/ha",
                    "organic": "FYM 5-10 tonnes/ha",
                    "micronutrients": "Zinc sulphate 25 kg/ha"
                },
                "irrigation": {
                    "water_requirement": "1200-1500mm",
                    "critical_stages": ["Tillering", "Panicle initiation", "Grain filling"],
                    "method": "Flood irrigation, SRI method"
                },
                "common_diseases": [
                    "Blast", "Brown spot", "Sheath rot", "False smut", "Bacterial blight"
                ],
                "common_pests": [
                    "Stem borer", "Brown plant hopper", "Leaf folder", "Gall midge"
                ],
                "harvest": {
                    "indicators": "Golden yellow grains, 80% maturity",
                    "method": "Manual harvesting, Combine harvester",
                    "yield": "5-6 tonnes/ha"
                },
                "market_info": {
                    "peak_season": "October-December (Kharif), March-May (Rabi)",
                    "storage": "14% moisture content",
                    "grades": "Grade A, Grade B, Grade C"
                }
            },
            "wheat": {
                "scientific_name": "Triticum aestivum",
                "local_names": {
                    "hindi": "गेहूं",
                    "marathi": "गहू",
                    "gujarati": "ઘઉં",
                    "punjabi": "ਕਣਕ"
                },
                "varieties": [
                    "HD 2967", "PBW 550", "DBW 88", "WH 542", "HD 3086",
                    "PBW 725", "HD 2733", "WH 1105"
                ],
                "growing_conditions": {
                    "temperature_range": "15-25°C",
                    "rainfall": "300-400mm",
                    "soil_ph": "6.0-7.5",
                    "soil_type": "Loam, Clay loam",
                    "altitude": "0-1500m"
                },
                "cultivation": {
                    "seasons": ["Rabi"],
                    "duration": "120-150 days",
                    "seed_rate": "100 kg/ha",
                    "plant_spacing": "22.5cm row spacing",
                    "depth": "3-5 cm"
                },
                "fertilizer": {
                    "recommended_dose": "NPK 120:60:40 kg/ha",
                    "organic": "FYM 8-10 tonnes/ha",
                    "micronutrients": "Zinc sulphate 25 kg/ha"
                },
                "irrigation": {
                    "water_requirement": "450-650mm",
                    "critical_stages": ["Crown root initiation", "Tillering", "Jointing", "Flowering"],
                    "method": "Furrow irrigation, Sprinkler irrigation"
                },
                "common_diseases": [
                    "Rust (Yellow, Brown, Black)", "Powdery mildew", "Loose smut", "Karnal bunt"
                ],
                "common_pests": [
                    "Aphid", "Termite", "Army worm", "Wheat midge"
                ],
                "harvest": {
                    "indicators": "Golden yellow color, grain hardness",
                    "method": "Combine harvester, Manual cutting",
                    "yield": "4-5 tonnes/ha"
                }
            },
            "cotton": {
                "scientific_name": "Gossypium spp.",
                "local_names": {
                    "hindi": "कपास",
                    "marathi": "कापूस",
                    "gujarati": "કપાસ",
                    "telugu": "పత్తి"
                },
                "varieties": [
                    "Bt cotton varieties", "Bollgard II", "RCH 659", "MRC 7361",
                    "Ajeet 155", "RCH 773", "VBN 08-M3"
                ],
                "growing_conditions": {
                    "temperature_range": "21-32°C",
                    "rainfall": "500-1000mm",
                    "soil_ph": "6.0-8.0",
                    "soil_type": "Black cotton soil, Red soil",
                    "altitude": "0-1000m"
                },
                "cultivation": {
                    "seasons": ["Kharif"],
                    "duration": "180-210 days",
                    "seed_rate": "1.5-2 kg/ha",
                    "plant_spacing": "90cm x 45-60cm",
                    "depth": "2-3 cm"
                },
                "fertilizer": {
                    "recommended_dose": "NPK 100:50:50 kg/ha",
                    "organic": "FYM 5-8 tonnes/ha",
                    "micronutrients": "Boron, Zinc application"
                },
                "irrigation": {
                    "water_requirement": "700-1300mm",
                    "critical_stages": ["Squaring", "Flowering", "Boll development"],
                    "method": "Furrow irrigation, Drip irrigation"
                },
                "common_diseases": [
                    "Fusarium wilt", "Verticillium wilt", "Alternaria leaf spot", "Grey mildew"
                ],
                "common_pests": [
                    "Bollworm", "Aphids", "Whitefly", "Thrips", "Jassids"
                ]
            }
        }
    
    def load_weather_patterns(self) -> Dict[str, Dict]:
        """Load regional weather patterns and climate data"""
        return {
            "monsoon_patterns": {
                "southwest_monsoon": {
                    "period": "June-September",
                    "regions": ["Western Ghats", "Central India", "Eastern India"],
                    "rainfall_percentage": 75,
                    "crop_activities": ["Sowing of Kharif crops", "Land preparation"]
                },
                "northeast_monsoon": {
                    "period": "October-December",
                    "regions": ["Tamil Nadu", "Karnataka", "Andhra Pradesh"],
                    "rainfall_percentage": 25,
                    "crop_activities": ["Rabi sowing", "Kharif harvesting"]
                }
            },
            "seasonal_temperatures": {
                "summer": {"months": [3, 4, 5], "avg_temp_range": "25-45°C"},
                "monsoon": {"months": [6, 7, 8, 9], "avg_temp_range": "20-35°C"},
                "winter": {"months": [12, 1, 2], "avg_temp_range": "10-25°C"},
                "post_monsoon": {"months": [10, 11], "avg_temp_range": "15-30°C"}
            },
            "regional_variations": {
                "north_india": {
                    "temperature_range": "0-48°C",
                    "rainfall": "400-1200mm",
                    "climate": "Continental"
                },
                "south_india": {
                    "temperature_range": "15-40°C",
                    "rainfall": "600-3000mm",
                    "climate": "Tropical"
                },
                "west_india": {
                    "temperature_range": "10-45°C",
                    "rainfall": "300-2000mm",
                    "climate": "Semi-arid to humid"
                }
            }
        }
    
    def load_government_schemes(self) -> Dict[str, Dict]:
        """Load comprehensive government schemes database"""
        return {
            "central_schemes": {
                "pm_kisan": {
                    "full_name": "Pradhan Mantri Kisan Samman Nidhi",
                    "launched": 2019,
                    "benefit": "₹6,000 per year in three installments of ₹2,000",
                    "eligibility": "Small and marginal farmer families with cultivable land",
                    "beneficiaries": "Over 11 crore farmers",
                    "application": "Online through PM-KISAN portal, CSC centers",
                    "documents_required": ["Aadhar card", "Land records", "Bank details"],
                    "website": "https://pmkisan.gov.in/"
                },
                "pm_fasal_bima": {
                    "full_name": "Pradhan Mantri Fasal Bima Yojana",
                    "launched": 2016,
                    "benefit": "Comprehensive crop insurance coverage",
                    "premium_rates": {
                        "kharif": "2% of sum insured",
                        "rabi": "1.5% of sum insured",
                        "annual": "5% of sum insured"
                    },
                    "coverage": "Natural calamities, pests, diseases",
                    "application": "Through banks, insurance companies",
                    "website": "https://pmfby.gov.in/"
                },
                "kisan_credit_card": {
                    "full_name": "Kisan Credit Card Scheme",
                    "launched": 1998,
                    "benefit": "Flexible credit for agriculture and allied activities",
                    "interest_rate": "4% per annum",
                    "credit_limit": "Based on cropping pattern and scale of finance",
                    "validity": "5 years",
                    "features": ["Personal Accident Insurance", "Crop Insurance"],
                    "application": "Through banks"
                },
                "soil_health_card": {
                    "full_name": "Soil Health Card Scheme",
                    "launched": 2015,
                    "benefit": "Free soil testing and nutrient recommendations",
                    "frequency": "Every 3 years",
                    "parameters_tested": ["pH", "EC", "N", "P", "K", "S", "Zn", "Fe", "Cu", "Mn", "B"],
                    "application": "Through agriculture department"
                }
            },
            "state_schemes": {
                "maharashtra": [
                    "Mahatma Phule Jan Arogya Yojana",
                    "Pradhan Mantri Krishi Sinchayee Yojana",
                    "Maharashtra State Crop Insurance Scheme"
                ],
                "punjab": [
                    "Crop Diversification Scheme",
                    "Punjab State Farmers' Debt Relief Scheme",
                    "Direct Benefit Transfer for Fertilizer"
                ],
                "karnataka": [
                    "Raitha Siri Scheme",
                    "Bhoochetana Scheme",
                    "Karnataka State Crop Insurance Scheme"
                ]
            }
        }
    
    def load_market_information(self) -> Dict[str, Dict]:
        """Load market-related information"""
        return {
            "msp_crops": [
                "Paddy", "Wheat", "Jowar", "Bajra", "Maize", "Ragi", "Arhar", "Moong", 
                "Urad", "Cotton", "Groundnut", "Sunflower", "Soybean", "Safflower",
                "Niger", "Jute", "Sugarcane"
            ],
            "marketing_channels": {
                "traditional": {
                    "mandis": "APMC mandis",
                    "commission_agents": "2-8% commission",
                    "traders": "Local and regional traders"
                },
                "modern": {
                    "fpo": "Farmer Producer Organizations",
                    "contract_farming": "Direct contract with companies",
                    "online_platforms": "eNAM, commodity exchanges"
                }
            },
            "post_harvest": {
                "storage_methods": ["Warehouses", "Cold storage", "Silos"],
                "processing": ["Primary processing", "Value addition"],
                "transportation": ["Road", "Rail", "Air (perishables)"]
            },
            "export_opportunities": {
                "major_commodities": ["Basmati rice", "Cotton", "Spices", "Tea", "Coffee"],
                "key_markets": ["Middle East", "Europe", "USA", "Southeast Asia"]
            }
        }
    
    def load_disease_database(self) -> Dict[str, Dict]:
        """Load comprehensive disease and pest database"""
        return {
            "fungal_diseases": {
                "blast": {
                    "crops_affected": ["Rice"],
                    "symptoms": "Elliptical lesions with gray centers and brown margins",
                    "conditions": "High humidity, moderate temperature",
                    "management": {
                        "cultural": ["Resistant varieties", "Balanced fertilization"],
                        "chemical": ["Tricyclazole", "Isoprothiolane"],
                        "biological": ["Pseudomonas fluorescens"]
                    }
                },
                "powdery_mildew": {
                    "crops_affected": ["Wheat", "Barley"],
                    "symptoms": "White powdery coating on leaves",
                    "conditions": "Cool, humid weather",
                    "management": {
                        "cultural": ["Proper spacing", "Resistant varieties"],
                        "chemical": ["Propiconazole", "Tebuconazole"]
                    }
                }
            },
            "bacterial_diseases": {
                "bacterial_blight": {
                    "crops_affected": ["Rice", "Cotton"],
                    "symptoms": "Water-soaked lesions turning yellow",
                    "management": {
                        "cultural": ["Seed treatment", "Crop rotation"],
                        "chemical": ["Streptocycline", "Copper oxychloride"]
                    }
                }
            },
            "viral_diseases": {
                "yellow_mosaic": {
                    "crops_affected": ["Soybean", "Black gram"],
                    "symptoms": "Yellow mosaic patterns on leaves",
                    "vector": "Whitefly",
                    "management": ["Vector control", "Resistant varieties"]
                }
            },
            "major_pests": {
                "bollworm": {
                    "crops_affected": ["Cotton", "Tomato", "Chickpea"],
                    "damage": "Bores into bolls and fruits",
                    "management": {
                        "cultural": ["Crop rotation", "Deep ploughing"],
                        "biological": ["NPV", "Trichogramma"],
                        "chemical": ["Emamectin benzoate", "Flubendiamide"]
                    }
                }
            }
        }
    
    def load_fertilizer_guide(self) -> Dict[str, Dict]:
        """Load fertilizer recommendations"""
        return {
            "macronutrients": {
                "nitrogen": {
                    "functions": ["Protein synthesis", "Chlorophyll formation"],
                    "deficiency_symptoms": ["Yellowing of older leaves", "Stunted growth"],
                    "sources": ["Urea", "CAN", "Ammonium sulphate"],
                    "application_timing": ["Basal", "Top dressing"]
                },
                "phosphorus": {
                    "functions": ["Root development", "Energy transfer"],
                    "deficiency_symptoms": ["Purple discoloration", "Poor root growth"],
                    "sources": ["DAP", "SSP", "TSP"],
                    "application_timing": ["Basal application"]
                },
                "potassium": {
                    "functions": ["Water regulation", "Disease resistance"],
                    "deficiency_symptoms": ["Leaf margin burning", "Weak stems"],
                    "sources": ["MOP", "SOP", "Potassium nitrate"],
                    "application_timing": ["Basal", "Split application"]
                }
            },
            "micronutrients": {
                "zinc": {
                    "deficiency_symptoms": ["White patches", "Short internodes"],
                    "sources": ["Zinc sulphate", "Zinc oxide"],
                    "application": ["Soil", "Foliar spray"]
                },
                "iron": {
                    "deficiency_symptoms": ["Interveinal chlorosis"],
                    "sources": ["Ferrous sulphate", "Iron chelates"],
                    "application": ["Foliar spray", "Soil application"]
                }
            },
            "organic_sources": {
                "farmyard_manure": {
                    "npk_content": "0.5-1.5% N, 0.4-0.8% P, 0.5-1.5% K",
                    "application_rate": "10-15 tonnes/ha",
                    "benefits": ["Soil structure improvement", "Microbial activity"]
                },
                "vermicompost": {
                    "npk_content": "1.5-2% N, 1-1.5% P, 1-1.5% K",
                    "application_rate": "2-3 tonnes/ha",
                    "benefits": ["Slow release", "Enzyme activity"]
                }
            }
        }
    
    def load_irrigation_guide(self) -> Dict[str, Dict]:
        """Load irrigation guidelines"""
        return {
            "methods": {
                "surface_irrigation": {
                    "types": ["Basin", "Border", "Furrow", "Check basin"],
                    "efficiency": "40-60%",
                    "suitable_crops": ["Rice", "Wheat", "Sugarcane"],
                    "advantages": ["Low cost", "Simple technology"],
                    "disadvantages": ["Water wastage", "Labor intensive"]
                },
                "sprinkler_irrigation": {
                    "types": ["Portable", "Semi-permanent", "Permanent"],
                    "efficiency": "70-80%",
                    "suitable_crops": ["Vegetables", "Cereals", "Oilseeds"],
                    "advantages": ["Water saving", "Uniform distribution"],
                    "disadvantages": ["High initial cost", "Wind effect"]
                },
                "drip_irrigation": {
                    "types": ["Surface drip", "Subsurface drip"],
                    "efficiency": "85-95%",
                    "suitable_crops": ["Fruits", "Vegetables", "Cash crops"],
                    "advantages": ["Maximum water saving", "Fertigation possible"],
                    "disadvantages": ["High cost", "Clogging issues"]
                }
            },
            "water_requirements": {
                "factors": ["Crop type", "Growth stage", "Climate", "Soil type"],
                "calculation_methods": ["Penman-Monteith", "Blaney-Criddle", "Pan evaporation"],
                "critical_stages": {
                    "rice": ["Transplanting", "Tillering", "Panicle initiation"],
                    "wheat": ["Crown root", "Jointing", "Flowering", "Grain filling"],
                    "cotton": ["Squaring", "Flowering", "Boll formation"]
                }
            }
        }
    
    def load_seasonal_calendar(self) -> Dict[str, Dict]:
        """Load seasonal agricultural calendar"""
        return {
            "kharif": {
                "season": "June - November",
                "crops": ["Rice", "Cotton", "Sugarcane", "Maize", "Soybean", "Groundnut"],
                "activities": {
                    "may": ["Land preparation", "Seed procurement"],
                    "june": ["Sowing", "Transplanting"],
                    "july": ["First weeding", "Fertilizer application"],
                    "august": ["Irrigation", "Pest management"],
                    "september": ["Second fertilizer dose", "Disease monitoring"],
                    "october": ["Pre-harvest activities", "Harvesting begins"],
                    "november": ["Harvesting", "Post-harvest operations"]
                }
            },
            "rabi": {
                "season": "November - April",
                "crops": ["Wheat", "Barley", "Chickpea", "Mustard", "Potato"],
                "activities": {
                    "october": ["Land preparation", "Seed treatment"],
                    "november": ["Sowing", "Irrigation"],
                    "december": ["First irrigation", "Fertilizer application"],
                    "january": ["Weeding", "Second irrigation"],
                    "february": ["Third irrigation", "Pest management"],
                    "march": ["Final irrigation", "Pre-harvest spray"],
                    "april": ["Harvesting", "Threshing"]
                }
            },
            "summer": {
                "season": "April - June",
                "crops": ["Fodder crops", "Green vegetables", "Summer rice"],
                "activities": {
                    "march": ["Land preparation", "Irrigation setup"],
                    "april": ["Sowing", "Regular irrigation"],
                    "may": ["Maintenance", "Harvesting fodder"],
                    "june": ["Final harvesting", "Land preparation for Kharif"]
                }
            }
        }
    
    def load_regional_data(self) -> Dict[str, Dict]:
        """Load region-specific agricultural data"""
        return {
            "agro_climatic_zones": {
                "zone_1": {
                    "name": "Western Himalayan Region",
                    "states": ["Himachal Pradesh", "Uttarakhand", "J&K"],
                    "characteristics": "Hill farming, temperate climate",
                    "major_crops": ["Apple", "Rice", "Wheat", "Maize"]
                },
                "zone_2": {
                    "name": "Eastern Himalayan Region", 
                    "states": ["Assam", "West Bengal hills", "Sikkim"],
                    "characteristics": "High rainfall, tea cultivation",
                    "major_crops": ["Tea", "Rice", "Maize", "Orange"]
                },
                "zone_3": {
                    "name": "Lower Gangetic Plains",
                    "states": ["West Bengal", "Bihar", "Eastern UP"],
                    "characteristics": "High fertility, rice-wheat system",
                    "major_crops": ["Rice", "Wheat", "Jute", "Potato"]
                }
            },
            "soil_types": {
                "alluvial": {
                    "distribution": "Northern plains, river valleys",
                    "characteristics": "High fertility, good water retention",
                    "suitable_crops": ["Rice", "Wheat", "Sugarcane"],
                    "management": "Regular organic matter addition"
                },
                "black": {
                    "distribution": "Deccan plateau",
                    "characteristics": "High clay content, water retention",
                    "suitable_crops": ["Cotton", "Sugarcane", "Wheat"],
                    "management": "Drainage improvement, deep ploughing"
                },
                "red": {
                    "distribution": "Southern and eastern India",
                    "characteristics": "Well drained, low fertility",
                    "suitable_crops": ["Rice", "Ragi", "Cotton"],
                    "management": "Organic matter, lime application"
                }
            }
        }
    
    def get_crop_information(self, crop_name: str) -> Dict:
        """Get comprehensive crop information"""
        crop_name_lower = crop_name.lower()
        
        # Direct match
        if crop_name_lower in self.crop_data:
            return self.crop_data[crop_name_lower]
        
        # Search in local names
        for crop, data in self.crop_data.items():
            local_names = data.get("local_names", {})
            for lang, name in local_names.items():
                if name == crop_name or crop_name_lower in name.lower():
                    return data
        
        return {}
    
    def get_seasonal_recommendations(self, month: int, region: str = None) -> Dict:
        """Get seasonal recommendations for a specific month"""
        recommendations = {
            "current_activities": [],
            "crops_to_sow": [],
            "crops_to_harvest": [],
            "general_advice": []
        }
        
        # Determine season
        if month in [6, 7, 8, 9, 10, 11]:  # Kharif season
            season_data = self.seasonal_calendar.get("kharif", {})
        elif month in [11, 12, 1, 2, 3, 4]:  # Rabi season
            season_data = self.seasonal_calendar.get("rabi", {})
        else:  # Summer season
            season_data = self.seasonal_calendar.get("summer", {})
        
        # Get month name
        month_names = [
            "january", "february", "march", "april", "may", "june",
            "july", "august", "september", "october", "november", "december"
        ]
        
        if month >= 1 and month <= 12:
            month_name = month_names[month - 1]
            activities = season_data.get("activities", {}).get(month_name, [])
            recommendations["current_activities"] = activities
        
        recommendations["crops_to_sow"] = season_data.get("crops", [])
        
        return recommendations
    
    def get_disease_management(self, crop: str, disease: str) -> Dict:
        """Get disease management recommendations"""
        management_info = {
            "identification": "",
            "symptoms": "",
            "management": {
                "cultural": [],
                "chemical": [],
                "biological": []
            },
            "prevention": []
        }
        
        # Search in disease database
        for category, diseases in self.disease_database.items():
            if disease.lower() in diseases:
                disease_info = diseases[disease.lower()]
                
                if crop.lower() in [c.lower() for c in disease_info.get("crops_affected", [])]:
                    management_info["symptoms"] = disease_info.get("symptoms", "")
                    management_info["management"] = disease_info.get("management", {})
                    break
        
        return management_info
    
    def get_fertilizer_recommendations(self, crop: str, growth_stage: str, 
                                    soil_test_results: Dict = None) -> Dict:
        """Get fertilizer recommendations"""
        recommendations = {
            "npk_dose": "",
            "application_schedule": [],
            "micronutrients": [],
            "organic_options": []
        }
        
        crop_info = self.get_crop_information(crop)
        if crop_info and "fertilizer" in crop_info:
            fertilizer_info = crop_info["fertilizer"]
            recommendations["npk_dose"] = fertilizer_info.get("recommended_dose", "")
            recommendations["organic_options"] = [fertilizer_info.get("organic", "")]
        
        # Add micronutrient recommendations based on soil test
        if soil_test_results:
            for nutrient, value in soil_test_results.items():
                if nutrient.lower() in ["zn", "zinc"] and float(value) < 0.6:
                    recommendations["micronutrients"].append("Zinc sulphate 25 kg/ha")
                elif nutrient.lower() in ["fe", "iron"] and float(value) < 4.5:
                    recommendations["micronutrients"].append("Ferrous sulphate 50 kg/ha")
        
        return recommendations
    
    def get_irrigation_schedule(self, crop: str, growth_stage: str, 
                              weather_forecast: Dict = None) -> Dict:
        """Get irrigation recommendations"""
        schedule = {
            "next_irrigation": "",
            "water_quantity": "",
            "critical_stages": [],
            "method_recommended": ""
        }
        
        crop_info = self.get_crop_information(crop)
        if crop_info and "irrigation" in crop_info:
            irrigation_info = crop_info["irrigation"]
            schedule["critical_stages"] = irrigation_info.get("critical_stages", [])
            schedule["method_recommended"] = irrigation_info.get("method", "")
            
            # Adjust based on weather forecast
            if weather_forecast:
                rainfall_expected = weather_forecast.get("rainfall", 0)
                if rainfall_expected > 20:  # mm
                    schedule["next_irrigation"] = "Postpone irrigation due to expected rainfall"
                else:
                    schedule["next_irrigation"] = "Irrigate as per crop requirement"
        
        return schedule
    
    def get_market_intelligence(self, commodity: str, region: str = None) -> Dict:
        """Get market intelligence for commodity"""
        intelligence = {
            "msp_status": False,
            "peak_season": "",
            "marketing_channels": [],
            "value_addition_opportunities": [],
            "export_potential": False
        }
        
        # Check if commodity has MSP
        if commodity.title() in self.market_data.get("msp_crops", []):
            intelligence["msp_status"] = True
        
        # Check export opportunities
        major_commodities = self.market_data.get("export_opportunities", {}).get("major_commodities", [])
        if any(commodity.lower() in comm.lower() for comm in major_commodities):
            intelligence["export_potential"] = True
        
        # Marketing channels
        intelligence["marketing_channels"] = [
            "APMC Mandis", "FPO", "Direct Marketing", "Contract Farming"
        ]
        
        return intelligence
    
    def search_knowledge_base(self, query: str, category: str = None) -> List[Dict]:
        """Search across the knowledge base"""
        results = []
        query_lower = query.lower()
        
        # Search in crop database
        if not category or category == "crops":
            for crop_name, crop_info in self.crop_data.items():
                if (query_lower in crop_name or 
                    any(query_lower in str(value).lower() for value in crop_info.values())):
                    results.append({
                        "category": "crops",
                        "title": crop_name.title(),
                        "content": crop_info,
                        "relevance": "high"
                    })
        
        # Search in schemes
        if not category or category == "schemes":
            central_schemes = self.government_schemes.get("central_schemes", {})
            for scheme_name, scheme_info in central_schemes.items():
                if (query_lower in scheme_name or
                    any(query_lower in str(value).lower() for value in scheme_info.values())):
                    results.append({
                        "category": "schemes",
                        "title": scheme_info.get("full_name", scheme_name),
                        "content": scheme_info,
                        "relevance": "high"
                    })
        
        return results[:10]  # Limit results
    
    def get_regional_recommendations(self, location: str) -> Dict:
        """Get location-specific recommendations"""
        recommendations = {
            "agro_climatic_zone": "",
            "suitable_crops": [],
            "soil_type": "",
            "climate_characteristics": "",
            "special_considerations": []
        }
        
        # Simple location matching (can be enhanced with geolocation)
        location_lower = location.lower()
        
        for zone_id, zone_data in self.regional_data.get("agro_climatic_zones", {}).items():
            states = [state.lower() for state in zone_data.get("states", [])]
            if any(state in location_lower for state in states):
                recommendations["agro_climatic_zone"] = zone_data.get("name", "")
                recommendations["suitable_crops"] = zone_data.get("major_crops", [])
                recommendations["climate_characteristics"] = zone_data.get("characteristics", "")
                break
        
        return recommendations
    
    def validate_agricultural_practice(self, practice: Dict) -> Dict:
        """Validate agricultural practices against knowledge base"""
        validation = {
            "is_valid": True,
            "warnings": [],
            "suggestions": [],
            "confidence": 0.8
        }
        
        crop = practice.get("crop", "").lower()
        season = practice.get("season", "").lower()
        location = practice.get("location", "")
        
        # Validate crop-season combination
        crop_info = self.get_crop_information(crop)
        if crop_info:
            suitable_seasons = [s.lower() for s in crop_info.get("cultivation", {}).get("seasons", [])]
            if season and season not in suitable_seasons:
                validation["warnings"].append(f"{crop.title()} is not typically grown in {season} season")
                validation["confidence"] *= 0.7
        
        return validation
