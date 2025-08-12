import re
import json
from typing import Dict, Any, List, Optional
import io
from PIL import Image

class InputProcessor:
    """Handles processing of various input types including text, audio, and documents"""
    
    def __init__(self):
        self.language_patterns = self.load_language_patterns()
        self.intent_keywords = self.load_intent_keywords()
        self.entity_patterns = self.load_entity_patterns()
    
    def load_language_patterns(self) -> Dict:
        """Load patterns for language detection and processing"""
        return {
            "hindi": {
                "patterns": ["कैसे", "क्या", "कब", "कहाँ", "क्यों", "कितना"],
                "transliterations": {
                    "kaise": "कैसे",
                    "kya": "क्या",
                    "kab": "कब",
                    "kahan": "कहाँ"
                }
            },
            "marathi": {
                "patterns": ["कसे", "काय", "केव्हा", "कुठे", "का", "किती"],
                "transliterations": {
                    "kase": "कसे",
                    "kay": "काय",
                    "kevha": "केव्हा"
                }
            }
        }
    
    def load_intent_keywords(self) -> Dict:
        """Load keywords for intent classification"""
        return {
            "weather": [
                "weather", "rain", "temperature", "forecast", "climate", 
                "मौसम", "बारिश", "तापमान", "हवामान"
            ],
            "crop": [
                "crop", "seed", "planting", "harvest", "irrigation", "fertilizer",
                "फसल", "बीज", "सिंचाई", "खाद", "पीक", "लागवड"
            ],
            "disease": [
                "disease", "pest", "infection", "fungus", "bug", "problem",
                "रोग", "कीट", "बीमारी", "फंगस"
            ],
            "market": [
                "price", "market", "sell", "buy", "rate", "mandi",
                "दाम", "भाव", "बाजार", "मंडी", "विक्री"
            ],
            "finance": [
                "loan", "credit", "money", "bank", "subsidy", "scheme",
                "कर्ज", "पैसा", "बँक", "योजना", "सब्सिडी"
            ],
            "policy": [
                "policy", "government", "scheme", "yojana", "sarkar",
                "नीति", "सरकार", "योजना", "नियम"
            ]
        }
    
    def load_entity_patterns(self) -> Dict:
        """Load patterns for entity extraction"""
        return {
            "crops": [
                "rice", "wheat", "cotton", "sugarcane", "maize", "corn", "soybean",
                "groundnut", "onion", "potato", "tomato", "chilli", "turmeric",
                "धान", "गहूं", "कपास", "गन्ना", "मक्का", "सोयाबीन", "मूंगफली"
            ],
            "seasons": [
                "kharif", "rabi", "summer", "zaid", "monsoon", "winter",
                "खरीफ", "रबी", "गर्मी", "सर्दी", "बरसात"
            ],
            "locations": [
                "maharashtra", "punjab", "haryana", "uttar pradesh", "bihar",
                "gujarat", "rajasthan", "madhya pradesh", "karnataka", "tamil nadu"
            ]
        }
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text input for better understanding"""
        try:
            # Basic cleaning
            text = text.strip()
            text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single
            text = re.sub(r'[^\w\s\u0900-\u097F]', ' ', text)  # Keep alphanumeric and Devanagari
            
            # Handle common transliterations
            text = self.handle_transliterations(text)
            
            return text.lower()
            
        except Exception as e:
            return text  # Return original if processing fails
    
    def handle_transliterations(self, text: str) -> str:
        """Handle common transliterations"""
        try:
            # Simple transliteration handling
            transliterations = {
                "kaise": "how",
                "kya": "what",
                "kab": "when",
                "kahan": "where",
                "kyun": "why",
                "kitna": "how much",
                "paani": "water",
                "kheti": "farming",
                "fasal": "crop",
                "baarish": "rain"
            }
            
            for hindi, english in transliterations.items():
                text = re.sub(r'\b' + hindi + r'\b', english, text, flags=re.IGNORECASE)
            
            return text
            
        except Exception:
            return text
    
    def detect_intent(self, text: str) -> str:
        """Detect user intent from text"""
        try:
            text_lower = text.lower()
            intent_scores = {}
            
            for intent, keywords in self.intent_keywords.items():
                score = sum(1 for keyword in keywords if keyword in text_lower)
                if score > 0:
                    intent_scores[intent] = score
            
            if intent_scores:
                return max(intent_scores, key=intent_scores.get)
            
            # Default intent based on common patterns
            if any(word in text_lower for word in ["when", "time", "schedule"]):
                return "crop"
            elif any(word in text_lower for word in ["cost", "expensive", "cheap"]):
                return "market"
            else:
                return "general"
                
        except Exception:
            return "general"
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract entities from text"""
        try:
            entities = {
                "crops": [],
                "seasons": [],
                "locations": [],
                "quantities": [],
                "dates": []
            }
            
            text_lower = text.lower()
            
            # Extract crops
            for crop in self.entity_patterns["crops"]:
                if crop in text_lower:
                    entities["crops"].append(crop)
            
            # Extract seasons
            for season in self.entity_patterns["seasons"]:
                if season in text_lower:
                    entities["seasons"].append(season)
            
            # Extract locations
            for location in self.entity_patterns["locations"]:
                if location in text_lower:
                    entities["locations"].append(location)
            
            # Extract quantities (simple regex)
            quantities = re.findall(r'\b(\d+(?:\.\d+)?)\s*(acre|hectare|kg|ton|litre|bag)', text_lower)
            entities["quantities"] = [f"{q[0]} {q[1]}" for q in quantities]
            
            # Extract dates (simple patterns)
            date_patterns = re.findall(r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})', text)
            entities["dates"] = date_patterns
            
            return entities
            
        except Exception:
            return {"crops": [], "seasons": [], "locations": [], "quantities": [], "dates": []}
    
    def process_audio(self, audio_file) -> str:
        """Process audio file and convert to text"""
        try:
            # Placeholder for Whisper ASR integration
            # In a real implementation, you would use OpenAI Whisper or similar
            
            # For now, return a placeholder message
            return "Audio transcription feature requires Whisper ASR integration. Please use text input."
            
        except Exception as e:
            raise Exception(f"Audio processing failed: {str(e)}")
    
    def process_document(self, uploaded_file) -> str:
        """Process uploaded documents and extract text"""
        try:
            file_type = uploaded_file.type
            file_content = ""
            
            if file_type == "application/pdf":
                file_content = self.extract_pdf_text(uploaded_file)
            elif file_type == "text/plain":
                file_content = str(uploaded_file.read(), "utf-8")
            elif file_type in ["application/msword", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
                file_content = self.extract_word_text(uploaded_file)
            else:
                raise Exception(f"Unsupported file type: {file_type}")
            
            # Clean and preprocess extracted text
            cleaned_content = self.clean_extracted_text(file_content)
            
            return cleaned_content
            
        except Exception as e:
            raise Exception(f"Document processing failed: {str(e)}")
    
    def extract_pdf_text(self, pdf_file) -> str:
        """Extract text from PDF file"""
        try:
            import PyPDF2
            
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text_content = ""
            
            for page in pdf_reader.pages:
                text_content += page.extract_text() + "\n"
            
            return text_content
            
        except ImportError:
            # Fallback if PyPDF2 not available
            return "PDF processing requires PyPDF2 library. Please install it for PDF support."
        except Exception as e:
            raise Exception(f"PDF extraction failed: {str(e)}")
    
    def extract_word_text(self, word_file) -> str:
        """Extract text from Word documents"""
        try:
            import docx
            
            doc = docx.Document(word_file)
            text_content = ""
            
            for paragraph in doc.paragraphs:
                text_content += paragraph.text + "\n"
            
            return text_content
            
        except ImportError:
            return "Word document processing requires python-docx library."
        except Exception as e:
            raise Exception(f"Word document extraction failed: {str(e)}")
    
    def clean_extracted_text(self, text: str) -> str:
        """Clean extracted text from documents"""
        try:
            # Remove excessive whitespace
            text = re.sub(r'\n+', '\n', text)
            text = re.sub(r'\s+', ' ', text)
            
            # Remove special characters but keep useful punctuation
            text = re.sub(r'[^\w\s\u0900-\u097F.,;:!?()-]', ' ', text)
            
            # Remove very short lines (likely noise)
            lines = text.split('\n')
            cleaned_lines = [line.strip() for line in lines if len(line.strip()) > 10]
            
            return '\n'.join(cleaned_lines)
            
        except Exception:
            return text  # Return original if cleaning fails
    
    def detect_language(self, text: str) -> str:
        """Detect language of input text"""
        try:
            # Simple detection based on script
            if re.search(r'[\u0900-\u097F]', text):  # Devanagari
                return "hi"  # Hindi/Marathi (need more sophisticated detection)
            elif re.search(r'[\u0B80-\u0BFF]', text):  # Tamil
                return "ta"
            elif re.search(r'[\u0C80-\u0CFF]', text):  # Kannada
                return "kn"
            elif re.search(r'[\u0C00-\u0C7F]', text):  # Telugu
                return "te"
            elif re.search(r'[\u0A80-\u0AFF]', text):  # Gujarati
                return "gu"
            elif re.search(r'[\u0A00-\u0A7F]', text):  # Punjabi
                return "pa"
            else:
                return "en"  # Default to English
                
        except Exception:
            return "en"
    
    def normalize_query(self, text: str, detected_language: str = "en") -> str:
        """Normalize query based on detected language"""
        try:
            # Language-specific normalization
            if detected_language in ["hi", "mr"]:  # Hindi/Marathi
                # Add specific normalization rules
                text = self.normalize_indic_text(text)
            
            # General normalization
            text = self.preprocess_text(text)
            
            return text
            
        except Exception:
            return text
    
    def normalize_indic_text(self, text: str) -> str:
        """Normalize Indic language text"""
        try:
            # Unicode normalization for Indic scripts
            import unicodedata
            text = unicodedata.normalize('NFC', text)
            
            # Remove zero-width characters
            text = re.sub(r'[\u200b-\u200f\ufeff]', '', text)
            
            return text
            
        except Exception:
            return text
