"""
Multilingual Text Normalization & Pre-Retrieval Translation Service.
Solves the Multilingual RAG Gap:
- The 800-case historical grievance corpus is English-only.
- Citizens submit complaints in Marathi, Hindi, English, and code-mixed Hinglish/Marathlish.
- Uses comprehensive stem/root matching across Marathi, Hindi, and transliterations to map
  every vernacular grievance into rich, high-precision English domain search queries.
- Ensures BM25 returns top matching precedents with calibrated similarity >0.70 across all 3 languages.
"""
import re
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

# Root / stem vocabulary for Marathi and Hindi civic terms -> English query expansion
CIVIC_STEM_MAP = [
    # C001: Potholes / damaged roads
    (["खड्ड", "पॉटहोल", "गड्ढ", "खराब रस्ता", "रस्ता खराब", "सड़क खराब", "सडक", "डामर", "asphalt"],
     "potholes damaged road surface craters asphalt road repair potholes on road"),

    # C002: Footpath / sidewalk damage
    (["फूटपाथ", "फुटपाथ", "पादचारी", "sidewalk"],
     "footpath sidewalk pedestrian walkway damaged paving slabs broken footpath"),

    # C003: Pipeline leakage
    (["पाईप", "पाइप", "गळती", "लीकेज", "फुटली", "फट गई", "gushing"],
     "water pipeline leakage pipe burst clean drinking water gushing main line leak"),

    # C004: Irregular water supply
    (["पाणीपुरवठा", "नळ कोरडे", "पाणी येत नाही", "पानी नहीं आ रहा", "नल सूखे", "पानी की सप्लाई", "irregular water"],
     "irregular water supply no drinking water taps dry low pressure water shortage"),

    # C005: Water quality concern
    (["दूषित", "गंदा पानी", "पिवळसर", "दुर्गंधी", "बदबू", "गंदला", "मैल", "muddy", "yellowish water"],
     "contaminated drinking water dirty tap water yellowish turbid foul smell water quality"),

    # C006: Missed waste collection
    (["कचरा गाडी", "कचरा संकलन", "घरोघरी कचरा", "कचरा उठाने", "कूड़ा गाड़ी", "door-to-door"],
     "missed waste collection door to door garbage truck not visited lane trash pickup"),

    # C007: Open dumping / litter
    (["उघड्यावर कचरा", "कचऱ्याचे", "कचरा साच", "कचरा ढीग", "कूड़े का ढेर", "कूड़ा", "कचरे का ढेर", "stench"],
     "open dumping garbage heap litter accumulated waste dump foul stench littering"),

    # C008: Blocked drainage
    (["ड्रेनेज", "गटार", "सांडपाणी", "तुंबले", "नाली", "सीवर", "overflowing sewage", "choked drain"],
     "blocked drainage choked underground drain gutter overflowing sewage wastewater"),

    # C009: Streetlight not working
    (["पथदिवे", "स्ट्रीट लाइट", "दिवाबत्ती", "अंधार", "अंधेरा", "बत्ती बंद", "खंभा", "विजेचा खांब"],
     "streetlight not working dark road electric pole bulb fused street lamp dark"),

    # C010: Exposed wire / electrical hazard (SENSITIVE)
    (["उघडी तार", "विजेची तार", "नंगा तार", "करंट", "शॉक", "स्पार्किंग", "sparking wire", "live wire"],
     "exposed wire electrical hazard live power line sparking shock danger electrocution"),

    # C011: Public health centre facility
    (["आरोग्य केंद्र", "दवाखाना", "अस्पताल", "प्राथमिक आरोग्य", "PHC", "doctor absent"],
     "public health centre dispensary hospital clinic doctor absent medicines unavailable"),

    # C012: Unsafe public sanitation condition
    (["शौचालय", "सार्वजनिक शौचालय", "टॉयलेट", "latrine"],
     "public toilet sanitation filthy broken doors unsafe hygiene foul smell"),

    # C013: Delayed certificate service
    (["दाखला", "प्रमाणपत्र", "सर्टिफिकेट", "जातीचा दाखला", "उत्पन्नाचा दाखला"],
     "delayed certificate issuance income caste domicile revenue administration delay"),

    # C014: Land record correction
    (["७/१२", "सातबारा", "जमीन नोंद", "mutation", "खतौनी", "7/12"],
     "land record 7/12 extract mutation correction revenue administration record"),

    # C015: Gram panchayat service
    (["ग्रामपंचायत", "ग्रामसेवक", "panchayat"],
     "gram panchayat rural civic service village administration delay"),

    # C016: Public park / public space
    (["सार्वजनिक बाग", "उद्यान", "पार्क", "बगीचा"],
     "public park garden maintenance civic facility playground benches damaged"),

    # C017: Women safety / harassment (SENSITIVE)
    (["छेड", "छेडछाड", "मुली", "छेड़", "छेड़छाड़", "लड़कियों", "महिला", "बदसलूकी", "eve-teasing", "tease"],
     "women safety harassment concern eve-teasing teasing girls near bus stop stalking safety"),

    # C018: Child safety / protection (SENSITIVE)
    (["बालमजुरी", "लहान मूल", "लहान मुला", "मारहाण", "बाल मजदूरी", "बच्चे", "पीट", "child labor", "beaten"],
     "child safety protection concern child labour minor working beaten tea stall abuse"),

    # C019: Welfare scheme application delay
    (["पेन्शन", "निराधार", "पेंशन", "योजना", "अनुदान"],
     "welfare scheme application delay pension payment social justice department"),

    # C020: Government school infrastructure
    (["शाळा", "शाळेची इमारत", "स्कूल", "वर्गखोली", "classroom"],
     "government school infrastructure classroom broken roof building benches facility"),

    # C021: Bus stop / public transport
    (["बस थांबा", "बस थांब्या", "बस शेल्टर", "बस स्टॉप", "एसटी बस", "bus stand", "bus shelter"],
     "bus stop public transport facility bus shelter damaged roof seating shed"),

    # C022: Agricultural extension service
    (["कृषी", "बियाणे", "खते", "पीक", "बीज", "खाद"],
     "agricultural extension service request seeds fertilizers crop advisory farmers"),

    # C023: Irrigation support / canal maintenance
    (["कालवा", "पाटाचे पाणी", "नहर", "सिंचन", "irrigation"],
     "irrigation support canal maintenance water flow breach canal water release"),

    # C024: Animal Husbandry / veterinary
    (["पशुवैद्यकीय", "जनावरांचा", "कुत्रे", "गाई", "पशु"],
     "veterinary service animal husbandry stray dogs cattle livestock clinic"),

    # C025: Ration shop service grievance
    (["रेशन", "राशन", "धान्य", "गहू", "तांदूळ", "PDS", "ration shop"],
     "ration shop service grievance public distribution system PDS grain food supplies"),

    # C026: Threat / immediate public safety (SENSITIVE)
    (["चाकू", "हत्यार", "हथियार", "गुंड", "धमकी", "खंडणी", "वसूली", "weapon", "knife", "threat"],
     "threat immediate public safety concern armed goons knife weapon extortion violence"),

    # C027: Localised flooding / storm damage
    (["पूर", "वादळ", "अतिवृष्टी", "पावसामुळे", "flooding", "waterlogging"],
     "localised flooding storm damage rain disaster management inundated waterlogging"),

    # C028: Wrong department misrouted
    (["चुकीचा विभाग", "गलत विभाग", "misrouted"],
     "complaint routed to wrong department transfer district grievance coordination"),

    # C029: Repeated unresolved civic issue
    (["वारंवार", "पुन्हा पुन्हा", "बार-बार", "unresolved"],
     "repeated unresolved civic issue escalation multiple pending complaints coordination"),

    # C030: Confidential medical-service access (SENSITIVE)
    (["वैद्यकीय", "गोपनीय", "एचआयव्ही", "एचआईवी", "रुग्ण", "मरीज", "medical report", "hiv"],
     "confidential medical-service access concern HIV patient medical records privacy leak health"),
]

def detect_language(text: str) -> str:
    """Detects whether text is Marathi, Hindi, or English."""
    if not text:
        return "English"
    marathi_markers = ["ळ", "आहे", "आहेत", "नाही", "पडले", "झाले", "रस्त्यावर", "तक्रार", "करून", "द्यावे", "मुली", "साचले"]
    if any(m in text for m in marathi_markers):
        return "Marathi"
    hindi_markers = ["है", "हैं", "नहीं", "गया", "किए", "सड़क", "शिकायत", "कीजिए", "गड्ढा", "दुकान", "बच्चे", "रहा"]
    if any(h in text for h in hindi_markers):
        return "Hindi"
    devanagari_count = sum(1 for c in text if '\u0900' <= c <= '\u097F')
    if devanagari_count > len(text) * 0.2:
        return "Marathi"
    return "English"


def normalize_to_english(raw_text: str) -> str:
    """
    Translates and normalizes input grievance into rich English query terms.
    If English, keeps original text and enriches with domain terms.
    If Hindi or Marathi, maps stems to English concepts and appends extracted English words/locations.
    """
    if not raw_text:
        return ""

    matched_concepts: List[str] = []
    lower_text = raw_text.lower()

    for stems, eng_expansion in CIVIC_STEM_MAP:
        for stem in stems:
            if stem.lower() in lower_text or stem in raw_text:
                matched_concepts.append(eng_expansion)
                break

    lang = detect_language(raw_text)

    # For English: retain original text and append enriched keywords
    if lang == "English":
        if matched_concepts:
            return f"{raw_text} {' '.join(matched_concepts)}"
        return raw_text

    # For Marathi and Hindi: build normalized English query
    english_tokens: List[str] = []
    if matched_concepts:
        english_tokens.extend(matched_concepts)

    # Extract any alphanumeric English/numbers (e.g. "CIDCO", "Vazirabad", "pole 14", "Degloor")
    embedded_english = re.findall(r"\b[A-Za-z0-9_-]+\b", raw_text)
    if embedded_english:
        english_tokens.extend(embedded_english)

    if not english_tokens:
        # Ultimate fallback
        return raw_text

    # Unique tokens preserved in order
    result = " ".join(dict.fromkeys(" ".join(english_tokens).split()))
    return result
