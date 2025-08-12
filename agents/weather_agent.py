import requests
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List
from agents.base_agent import BaseAgent

class WeatherAgent(BaseAgent):
    """Specialized agent for weather-related information and agricultural advice"""
    
    def __init__(self):
        super().__init__("WeatherAgent")
        self.weather_api_key = None
        self.initialize_weather_api()
    
    def initialize_weather_api(self):
        """Initialize WeatherAPI client"""
        import os
        self.weather_api_key = os.getenv("WEATHER_API_KEY")
        if not self.weather_api_key:
            print("Warning: WEATHER_API_KEY not found")
    
    def get_weather_advice(self, query: str, user_context: Dict) -> str:
        """Generate weather-based agricultural advice"""
        try:
            location = user_context.get("location", "")
            
            if not location:
                return """Please provide your location in the user profile to get accurate weather information. 
                Weather data is crucial for agricultural decision-making."""
            
            # Get current and forecast weather
            current_weather = self.get_current_weather(location)
            forecast_data = self.get_weather_forecast(location, days=7)
            
            # Generate agricultural advice based on weather
            system_message = """You are an agricultural meteorologist. Provide practical advice 
            based on weather conditions for farming activities. Focus on actionable recommendations 
            for irrigation, planting, harvesting, pest management, and crop protection."""
            
            weather_context = {
                "current": current_weather,
                "forecast": forecast_data,
                "location": location
            }
            
            enhanced_query = f"""
            Query: {query}
            Location: {location}
            Current Weather: {current_weather}
            7-Day Forecast: {forecast_data}
            
            Provide specific agricultural advice considering these weather conditions.
            Include timing recommendations and precautionary measures.
            """
            
            response = self.generate_response(enhanced_query, system_message)
            
            # Add weather summary
            weather_summary = self.format_weather_summary(current_weather, forecast_data)
            return f"{weather_summary}\n\n**Agricultural Advice:**\n{response}"
            
        except Exception as e:
            fallback_response = self.web_search_fallback(f"weather agriculture {query}", "weather")
            if fallback_response != "No relevant information found through web search.":
                return f"Using fallback data:\n{fallback_response}"
            return f"Error getting weather advice: {str(e)}"
    
    def get_current_weather(self, location: str) -> Dict:
        """Get current weather data from WeatherAPI"""
        try:
            if not self.weather_api_key:
                return {"error": "Weather API key not configured"}
            
            url = f"http://api.weatherapi.com/v1/current.json"
            params = {
                "key": self.weather_api_key,
                "q": location,
                "aqi": "yes"
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            return {
                "temperature": data["current"]["temp_c"],
                "feels_like": data["current"]["feelslike_c"],
                "humidity": data["current"]["humidity"],
                "wind_speed": data["current"]["wind_kph"],
                "wind_direction": data["current"]["wind_dir"],
                "pressure": data["current"]["pressure_mb"],
                "visibility": data["current"]["vis_km"],
                "uv_index": data["current"]["uv"],
                "condition": data["current"]["condition"]["text"],
                "is_day": data["current"]["is_day"],
                "last_updated": data["current"]["last_updated"]
            }
            
        except Exception as e:
            return {"error": f"Failed to get current weather: {str(e)}"}
    
    def get_weather_forecast(self, location: str, days: int = 7) -> List[Dict]:
        """Get weather forecast data"""
        try:
            if not self.weather_api_key:
                return [{"error": "Weather API key not configured"}]
            
            url = f"http://api.weatherapi.com/v1/forecast.json"
            params = {
                "key": self.weather_api_key,
                "q": location,
                "days": min(days, 10),  # API limits to 10 days
                "aqi": "no",
                "alerts": "yes"
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            forecast_days = []
            
            for day_data in data["forecast"]["forecastday"]:
                day_info = {
                    "date": day_data["date"],
                    "max_temp": day_data["day"]["maxtemp_c"],
                    "min_temp": day_data["day"]["mintemp_c"],
                    "avg_temp": day_data["day"]["avgtemp_c"],
                    "max_wind": day_data["day"]["maxwind_kph"],
                    "total_precip": day_data["day"]["totalprecip_mm"],
                    "avg_humidity": day_data["day"]["avghumidity"],
                    "condition": day_data["day"]["condition"]["text"],
                    "uv_index": day_data["day"]["uv"],
                    "sunrise": day_data["astro"]["sunrise"],
                    "sunset": day_data["astro"]["sunset"],
                    "chance_of_rain": day_data["day"]["daily_chance_of_rain"],
                    "will_it_rain": day_data["day"]["daily_will_it_rain"]
                }
                forecast_days.append(day_info)
            
            return forecast_days
            
        except Exception as e:
            return [{"error": f"Failed to get weather forecast: {str(e)}"}]
    
    def format_weather_summary(self, current: Dict, forecast: List[Dict]) -> str:
        """Format weather information for display"""
        if "error" in current:
            return f"**Weather Error:** {current['error']}"
        
        summary_parts = []
        
        # Current conditions
        summary_parts.append("**📊 Current Weather:**")
        summary_parts.append(f"• Temperature: {current['temperature']}°C (feels like {current['feels_like']}°C)")
        summary_parts.append(f"• Condition: {current['condition']}")
        summary_parts.append(f"• Humidity: {current['humidity']}%")
        summary_parts.append(f"• Wind: {current['wind_speed']} km/h {current['wind_direction']}")
        summary_parts.append(f"• UV Index: {current['uv_index']}")
        
        # Forecast highlights
        if forecast and not forecast[0].get("error"):
            summary_parts.append("\n**🌤️ 7-Day Forecast Highlights:**")
            
            # Rain prediction
            rainy_days = [day for day in forecast if day["chance_of_rain"] > 50]
            if rainy_days:
                rain_dates = [day["date"] for day in rainy_days]
                summary_parts.append(f"• Rain expected: {', '.join(rain_dates)}")
            
            # Temperature range
            max_temps = [day["max_temp"] for day in forecast]
            min_temps = [day["min_temp"] for day in forecast]
            summary_parts.append(f"• Temperature range: {min(min_temps):.1f}°C to {max(max_temps):.1f}°C")
            
            # Extreme weather warnings
            extreme_conditions = []
            for day in forecast:
                if day["max_temp"] > 35:
                    extreme_conditions.append(f"High temperature ({day['max_temp']}°C) on {day['date']}")
                if day["total_precip"] > 25:
                    extreme_conditions.append(f"Heavy rain ({day['total_precip']}mm) on {day['date']}")
                if day["max_wind"] > 40:
                    extreme_conditions.append(f"Strong winds ({day['max_wind']}km/h) on {day['date']}")
            
            if extreme_conditions:
                summary_parts.append("• ⚠️ Weather alerts: " + "; ".join(extreme_conditions))
        
        return "\n".join(summary_parts)
    
    def get_irrigation_recommendations(self, current_weather: Dict, forecast: List[Dict]) -> str:
        """Generate irrigation recommendations based on weather"""
        try:
            system_message = """You are an irrigation specialist. Based on weather data, 
            provide specific irrigation timing and quantity recommendations."""
            
            prompt = f"""
            Based on this weather data:
            Current: {current_weather}
            7-day forecast: {forecast}
            
            Provide irrigation recommendations including:
            1. Should irrigation be done today?
            2. Optimal irrigation timing for the week
            3. Water quantity adjustments based on weather
            4. Any weather-related irrigation precautions
            """
            
            return self.generate_response(prompt, system_message)
            
        except Exception as e:
            return f"Error generating irrigation recommendations: {str(e)}"
