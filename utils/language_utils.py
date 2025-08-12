import re
from typing import Dict, List, Optional, Tuple
import unicodedata
from datetime import datetime

class LanguageUtils:
    """Utility class for handling multilingual support and language processing"""
    
    def __init__(self):
        self.supported_languages = self.load_supported_languages()
        self.transliteration_maps = self.load_transliteration_maps()
        self.language_patterns = self.load_language_patterns()
        self.regional_terms = self.load_regional_terms()
    
    def load_supported_languages(self) -> Dict[str, Dict]:
        """Load supported languages with their metadata"""
        return {
            "en": {
                "name": "English",
                "native_name": "English",
                "script": "Latin",
                "direction": "ltr",
                "enabled": True
            },
            "hi": {
                "name": "Hindi",
                "native_name": "हिंदी",
                "script": "Devanagari",
                "direction": "ltr",
                "enabled": True
            },
            "mr": {
                "name": "Marathi",
                "native_name": "मराठी",
                "script": "Devanagari",
                "direction": "ltr",
                "enabled": True
            },
            "gu": {
                "name": "Gujarati",
                "native_name": "ગુજરાતી",
                "script": "Gujarati",
                "direction": "ltr",
                "enabled": True
            },
            "ta": {
                "name": "Tamil",
                "native_name": "தமிழ்",
                "script": "Tamil",
                "direction": "ltr",
                "enabled": True
            },
            "te": {
                "name": "Telugu",
                "native_name": "తెలుగు",
                "script": "Telugu",
                "direction": "ltr",
                "enabled": True
            },
            "kn": {
                "name": "Kannada",
                "native_name": "ಕನ್ನಡ",
                "script": "Kannada",
                "direction": "ltr",
                "enabled": True
            },
            "bn": {
                "name": "Bengali",
                "native_name": "বাংলা",
                "script": "Bengali",
                "direction": "ltr",
                "enabled": True
            },
            "pa": {
                "name": "Punjabi",
                "native_name": "ਪੰਜਾਬੀ",
                "script": "Gurmukhi",
                "direction": "ltr",
                "enabled": True
            }
        }
    
    def load_transliteration_maps(self) -> Dict[str, Dict]:
        """Load transliteration mappings for different languages"""
        return {
            "hi": {
                # Common Hindi transliterations
                "kaise": "कैसे",
                "kya": "क्या",
                "kab": "कब",
                "kahan": "कहाँ",
                "kyun": "क्यों",
                "kitna": "कितना",
                "paani": "पानी",
                "kheti": "खेती",
                "fasal": "फसल",
                "baarish": "बारिश",
                "mausam": "मौसम",
                "kisan": "किसान",
                "zameen": "जमीन",
                "beej": "बीज",
                "ped": "पेड़",
                "phal": "फल",
                "sabzi": "सब्जी",
                "dhaan": "धान",
                "gehun": "गेहूं",
                "makka": "मक्का",
                "kapas": "कपास"
            },
            "mr": {
                # Common Marathi transliterations
                "kase": "कसे",
                "kay": "काय",
                "kevha": "केव्हा",
                "kuthe": "कुठे",
                "ka": "का",
                "kiti": "किती",
                "paani": "पाणी",
                "sheti": "शेती",
                "pik": "पीक",
                "paus": "पाऊस",
                "havaaman": "हवामान",
                "shetkari": "शेतकरी",
                "jameen": "जमीन",
                "biyaane": "बियाणे"
            },
            "gu": {
                # Common Gujarati transliterations
                "kem": "કેમ",
                "shu": "શું",
                "kyare": "ક્યારે",
                "kya": "ક્યાં",
                "ketlu": "કેટલું",
                "paani": "પાણી",
                "kheti": "ખેતી",
                "fasal": "ફસલ",
                "varsa": "વરસા"
            }
        }
    
    def load_language_patterns(self) -> Dict[str, List[str]]:
        """Load language detection patterns"""
        return {
            "hi": [
                "कैसे", "क्या", "कब", "कहाँ", "क्यों", "कितना", "है", "हूँ", "हैं",
                "खेती", "फसल", "किसान", "पानी", "बारिश", "मौसम"
            ],
            "mr": [
                "कसे", "काय", "केव्हा", "कुठे", "का", "किती", "आहे", "आहेत",
                "शेती", "पीक", "शेतकरी", "पाणी", "पाऊस", "हवामान"
            ],
            "gu": [
                "કેમ", "શું", "ક્યારે", "ક્યાં", "કેટલું", "છે", "છો", "છીએ",
                "ખેતી", "ફસલ", "પાણી", "વરસા"
            ],
            "ta": [
                "எப்படி", "என்ன", "எப்போது", "எங்கே", "ஏன்", "எவ்வளவு", "இருக்கிறது",
                "விவசாயம்", "பயிர்", "நீர்", "மழை"
            ],
            "te": [
                "ఎలా", "ఏమి", "ఎప్పుడు", "ఎక్కడ", "ఎందుకు", "ఎంత", "ఉంది",
                "వ్యవసాయం", "పంట", "నీరు", "వర్షం"
            ],
            "kn": [
                "ಹೇಗೆ", "ಏನು", "ಯಾವಾಗ", "ಎಲ್ಲಿ", "ಏಕೆ", "ಎಷ್ಟು", "ಇದೆ",
                "ಕೃಷಿ", "ಬೆಳೆ", "ನೀರು", "ಮಳೆ"
            ],
            "bn": [
                "কেমন", "কী", "কখন", "কোথায়", "কেন", "কত", "আছে",
                "কৃষি", "ফসল", "পানি", "বৃষ্টি"
            ],
            "pa": [
                "ਕਿਵੇਂ", "ਕੀ", "ਕਦੋਂ", "ਕਿੱਥੇ", "ਕਿਉਂ", "ਕਿੰਨਾ", "ਹੈ",
                "ਖੇਤੀ", "ਫਸਲ", "ਪਾਣੀ", "ਮੀਂਹ"
            ]
        }
    
    def load_regional_terms(self) -> Dict[str, Dict]:
        """Load regional agricultural terms"""
        return {
            "crops": {
                "en": ["rice", "wheat", "cotton", "sugarcane", "maize", "soybean"],
                "hi": ["धान", "गेहूं", "कपास", "गन्ना", "मक्का", "सोयाबीन"],
                "mr": ["तांदूळ", "गहू", "कापूस", "ऊस", "मका", "सोयाबीन"],
                "gu": ["ચોખા", "ઘઉં", "કપાસ", "શેરડી", "મકાઈ", "સોયાબીન"],
                "ta": ["அரிசி", "கோதுமை", "பருத்தி", "கருவாளம்", "சோளம்", "சோயாபீன்"],
                "te": ["వరి", "గోధుమ", "పత్తి", "చెరకు", "మొక్కజొన్న", "సోయాబీన్"],
                "kn": ["ಅಕ್ಕಿ", "ಗೋಧಿ", "ಹತ್ತಿ", "ಕಬ್ಬು", "ಜೋಳ", "ಸೋಯಾಬೀನ್"],
                "bn": ["চাল", "গম", "তুলা", "আখ", "ভুট্টা", "সয়াবিন"],
                "pa": ["ਚਾਵਲ", "ਕਣਕ", "ਕਪਾਹ", "ਗੰਨਾ", "ਮੱਕੀ", "ਸੋਇਆਬੀਨ"]
            },
            "seasons": {
                "en": ["kharif", "rabi", "summer", "monsoon", "winter"],
                "hi": ["खरीफ", "रबी", "गर्मी", "बरसात", "सर्दी"],
                "mr": ["खरीप", "रब्बी", "उन्हाळा", "पावसाळा", "हिवाळा"],
                "gu": ["ખરીફ", "રબી", "ઉનાળો", "વરસાદ", "શિયાળો"],
                "ta": ["கரீப்", "ரபி", "கோடை", "மழைக்காலம்", "குளிர்காலம்"],
                "te": ["ఖరీఫ్", "రబీ", "వేసవి", "వర్షాకాలం", "చలికాలం"],
                "kn": ["ಖರೀಫ್", "ರಬಿ", "ಬೇಸಿಗೆ", "ಮಳೆಗಾಲ", "ಚಳಿಗಾಲ"],
                "bn": ["খরিপ", "রবি", "গ্রীষ্ম", "বর্ষা", "শীত"],
                "pa": ["ਖਰੀਫ", "ਰਬੀ", "ਗਰਮੀ", "ਮਾਨਸੂਨ", "ਸਰਦੀ"]
            },
            "farming_terms": {
                "en": ["irrigation", "fertilizer", "pesticide", "harvest", "planting"],
                "hi": ["सिंचाई", "खाद", "कीटनाशक", "फसल कटाई", "रोपण"],
                "mr": ["सिंचन", "खत", "कीटकनाशक", "कापणी", "लागवड"],
                "gu": ["સિંચાઈ", "ખાતર", "જંતુનાશક", "કાપણી", "વાવેતર"],
                "ta": ["நீர்ப்பாசனம்", "உரம்", "பூச்சிக்கொல்லி", "அறுவடை", "நடவு"],
                "te": ["నీటిపారుదల", "ఎరువులు", "కీటకనాశకాలు", "కోత", "నాటడం"],
                "kn": ["ನೀರಾವರಿ", "ಗೊಬ್ಬರ", "ಕೀಟನಾಶಕ", "ಸುಗ್ಗಿ", "ನೆಡುವಿಕೆ"],
                "bn": ["সেচ", "সার", "কীটনাশক", "ফসল কাটা", "রোপণ"],
                "pa": ["ਸਿੰਚਾਈ", "ਖਾਦ", "ਕੀੜੇਮਾਰ", "ਵਾਢੀ", "ਬਿਜਾਈ"]
            }
        }
    
    def detect_language(self, text: str) -> Tuple[str, float]:
        """
        Detect language of input text with confidence score
        Returns: (language_code, confidence_score)
        """
        try:
            text = text.strip().lower()
            
            if not text:
                return "en", 0.0
            
            # Script-based detection first
            script_lang = self.detect_by_script(text)
            if script_lang:
                return script_lang, 0.9
            
            # Pattern-based detection
            pattern_scores = {}
            
            for lang_code, patterns in self.language_patterns.items():
                score = 0
                words = text.split()
                
                for pattern in patterns:
                    if pattern.lower() in text:
                        score += 1
                
                if words:
                    pattern_scores[lang_code] = score / len(words)
            
            if pattern_scores:
                best_lang = max(pattern_scores, key=pattern_scores.get)
                confidence = min(pattern_scores[best_lang] * 2, 1.0)  # Scale confidence
                
                if confidence > 0.1:
                    return best_lang, confidence
            
            # Default to English
            return "en", 0.3
            
        except Exception as e:
            print(f"Language detection error: {e}")
            return "en", 0.0
    
    def detect_by_script(self, text: str) -> Optional[str]:
        """Detect language by Unicode script"""
        try:
            # Count characters by script
            script_counts = {}
            
            for char in text:
                if char.isspace() or char.isdigit() or char in ".,;:!?()-":
                    continue
                
                # Get Unicode category
                category = unicodedata.category(char)
                if category.startswith('L'):  # Letter category
                    code_point = ord(char)
                    
                    # Devanagari (Hindi/Marathi/Sanskrit)
                    if 0x0900 <= code_point <= 0x097F:
                        script_counts['devanagari'] = script_counts.get('devanagari', 0) + 1
                    # Tamil
                    elif 0x0B80 <= code_point <= 0x0BFF:
                        script_counts['tamil'] = script_counts.get('tamil', 0) + 1
                    # Telugu
                    elif 0x0C00 <= code_point <= 0x0C7F:
                        script_counts['telugu'] = script_counts.get('telugu', 0) + 1
                    # Kannada
                    elif 0x0C80 <= code_point <= 0x0CFF:
                        script_counts['kannada'] = script_counts.get('kannada', 0) + 1
                    # Gujarati
                    elif 0x0A80 <= code_point <= 0x0AFF:
                        script_counts['gujarati'] = script_counts.get('gujarati', 0) + 1
                    # Bengali
                    elif 0x0980 <= code_point <= 0x09FF:
                        script_counts['bengali'] = script_counts.get('bengali', 0) + 1
                    # Gurmukhi (Punjabi)
                    elif 0x0A00 <= code_point <= 0x0A7F:
                        script_counts['gurmukhi'] = script_counts.get('gurmukhi', 0) + 1
            
            if not script_counts:
                return None
            
            # Map scripts to language codes
            script_to_lang = {
                'devanagari': 'hi',  # Default to Hindi for Devanagari
                'tamil': 'ta',
                'telugu': 'te',
                'kannada': 'kn',
                'gujarati': 'gu',
                'bengali': 'bn',
                'gurmukhi': 'pa'
            }
            
            dominant_script = max(script_counts, key=script_counts.get)
            return script_to_lang.get(dominant_script)
            
        except Exception:
            return None
    
    def transliterate_text(self, text: str, from_lang: str = "en", to_lang: str = "hi") -> str:
        """Transliterate text between languages"""
        try:
            if from_lang == "en" and to_lang in self.transliteration_maps:
                trans_map = self.transliteration_maps[to_lang]
                
                words = text.split()
                transliterated_words = []
                
                for word in words:
                    word_lower = word.lower()
                    if word_lower in trans_map:
                        transliterated_words.append(trans_map[word_lower])
                    else:
                        transliterated_words.append(word)
                
                return " ".join(transliterated_words)
            
            return text  # Return original if no transliteration available
            
        except Exception:
            return text
    
    def normalize_text(self, text: str, language: str = "en") -> str:
        """Normalize text for processing"""
        try:
            # Unicode normalization
            text = unicodedata.normalize('NFC', text)
            
            # Remove zero-width characters
            text = re.sub(r'[\u200b-\u200f\ufeff]', '', text)
            
            # Language-specific normalization
            if language in ["hi", "mr"]:
                # Handle Devanagari specifics
                text = self.normalize_devanagari(text)
            elif language == "ta":
                # Handle Tamil specifics
                text = self.normalize_tamil(text)
            
            # General cleanup
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()
            
            return text
            
        except Exception:
            return text
    
    def normalize_devanagari(self, text: str) -> str:
        """Normalize Devanagari script text"""
        try:
            # Remove nukta variations
            text = re.sub(r'[\u093C]', '', text)
            
            # Normalize zero-width joiner
            text = re.sub(r'[\u200C\u200D]', '', text)
            
            return text
            
        except Exception:
            return text
    
    def normalize_tamil(self, text: str) -> str:
        """Normalize Tamil script text"""
        try:
            # Tamil-specific normalizations
            # Handle Tamil-specific Unicode issues
            return text
            
        except Exception:
            return text
    
    def get_regional_term(self, english_term: str, target_language: str, 
                         category: str = "crops") -> str:
        """Get regional term for English agricultural term"""
        try:
            if category in self.regional_terms:
                terms = self.regional_terms[category]
                
                if "en" in terms and target_language in terms:
                    en_terms = terms["en"]
                    target_terms = terms[target_language]
                    
                    if english_term.lower() in [t.lower() for t in en_terms]:
                        # Find index of English term
                        for i, term in enumerate(en_terms):
                            if term.lower() == english_term.lower():
                                if i < len(target_terms):
                                    return target_terms[i]
            
            return english_term  # Return original if no translation found
            
        except Exception:
            return english_term
    
    def translate_agricultural_terms(self, text: str, target_language: str) -> str:
        """Translate agricultural terms in text to target language"""
        try:
            words = text.split()
            translated_words = []
            
            for word in words:
                # Check each category for translations
                translated = False
                
                for category in ["crops", "seasons", "farming_terms"]:
                    regional_term = self.get_regional_term(word, target_language, category)
                    if regional_term != word:
                        translated_words.append(regional_term)
                        translated = True
                        break
                
                if not translated:
                    translated_words.append(word)
            
            return " ".join(translated_words)
            
        except Exception:
            return text
    
    def is_supported_language(self, lang_code: str) -> bool:
        """Check if language is supported"""
        return (lang_code in self.supported_languages and 
                self.supported_languages[lang_code].get("enabled", False))
    
    def get_language_info(self, lang_code: str) -> Dict:
        """Get language information"""
        return self.supported_languages.get(lang_code, {})
    
    def get_supported_languages_list(self) -> List[Dict]:
        """Get list of supported languages"""
        return [
            {
                "code": code,
                **info
            }
            for code, info in self.supported_languages.items()
            if info.get("enabled", False)
        ]
    
    def format_multilingual_response(self, response: str, user_language: str) -> str:
        """Format response for multilingual output"""
        try:
            if user_language == "en":
                return response
            
            # Add language-specific formatting
            if user_language in ["hi", "mr"]:
                # Add Devanagari number formatting if needed
                response = self.format_devanagari_numbers(response)
            
            # Translate key agricultural terms
            response = self.translate_agricultural_terms(response, user_language)
            
            return response
            
        except Exception:
            return response
    
    def format_devanagari_numbers(self, text: str) -> str:
        """Format numbers in Devanagari script if appropriate"""
        try:
            # Simple number formatting (can be expanded)
            devanagari_digits = ['०', '१', '२', '३', '४', '५', '६', '७', '८', '९']
            
            # Only format standalone numbers, not numbers within words
            def replace_number(match):
                number = match.group()
                devanagari_number = ""
                for digit in number:
                    if digit.isdigit():
                        devanagari_number += devanagari_digits[int(digit)]
                    else:
                        devanagari_number += digit
                return devanagari_number
            
            # Replace standalone numbers
            text = re.sub(r'\b\d+\b', replace_number, text)
            
            return text
            
        except Exception:
            return text
    
    def clean_mixed_script_text(self, text: str) -> str:
        """Clean text with mixed scripts"""
        try:
            # Handle common mixed script issues
            text = re.sub(r'[\u200b-\u200f\ufeff]', '', text)  # Remove zero-width chars
            text = re.sub(r'\s+', ' ', text)  # Normalize spaces
            text = text.strip()
            
            return text
            
        except Exception:
            return text
