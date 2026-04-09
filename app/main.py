from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import json
import re
from datetime import datetime

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")


class ChatRequest(BaseModel):
    message: str


def load_faq():
    with open("data/faq.json", "r", encoding="utf-8") as file:
        return json.load(file)


FAQ_DATA = load_faq()


SYNONYMS = {
    "sarakstu": "grafiks",
    "saraksts": "grafiks",
    "plānu": "grafiks",
    "plāns": "grafiks",
    "kalendārs": "grafiks",

    "stundu": "lekciju",
    "stundas": "lekciju",
    "nodarbību": "lekciju",
    "nodarbības": "lekciju",

    "epastu": "epasts",
    "e-pastu": "epasts",
    "epasta": "epasts",
    "emails": "epasts",
    "mail": "epasts",

    "naudas": "stipendija",
    "nauda": "stipendija",
    "atbalstu": "stipendija",
    "atbalsts": "stipendija",
    "finansējums": "stipendija",
    "finansiāls": "stipendija",

    "iestāties": "uzņemšana",
    "iestāšanās": "uzņemšana",
    "uzņemt": "uzņemšana",
    "uzņemšanas": "uzņemšana",

    "grāmata": "bibliotēka",
    "grāmatas": "bibliotēka",
    "lasītava": "bibliotēka",
    "bibliotekā": "bibliotēka",

    "priekšmetus": "kursi",
    "priekšmets": "kursi",
    "kurss": "kursi",
    "kursus": "kursi",

    "sesijas": "eksāmeni",
    "sesija": "eksāmeni",
    "eksāmenu": "eksāmeni",
    "ieskaite": "eksāmeni",
    "ieskaites": "eksāmeni",
    "pārbaudījums": "eksāmeni",

    "rēķinu": "rēķins",
    "rēķina": "rēķins",
    "apmaksa": "maksa",
    "maksājums": "maksa",
    "maksāt": "maksa",
    "samaksa": "maksa",

    "praktika": "prakse",
    "prakses": "prakse",
    "internship": "prakse",

    "ārzemēs": "erasmus",
    "ārzemes": "erasmus",
    "mobilitāte": "erasmus",
    "apmaiņa": "erasmus",

    "vērtējumi": "sekmes",
    "vērtējums": "sekmes",
    "atzīme": "sekmes",
    "atzīmes": "sekmes",
    "rezultāti": "sekmes",

    "deadline": "termiņš",
    "deadlines": "termiņš",
    "kavējums": "termiņš",
    "nokavēju": "termiņš",
    "nokavēts": "termiņš",

    "diplomdarbs": "bakalaura",
    "noslēguma": "bakalaura",
    "vadītājs": "bakalaura",

    "konsultācijas": "konsultācija",
    "pieņemšana": "konsultācija",

    "kopmītnes": "dzīvošana",
    "kopmītnēs": "dzīvošana",
    "dienesta": "dzīvošana",
    "viesnīca": "dzīvošana",
    "istaba": "dzīvošana",

    "apliecība": "studenta",
    "karte": "studenta",
    "karti": "studenta",

    "konts": "epasts",
    "profils": "epasts",

    "apelācija": "atzīme",
    "iebildums": "atzīme",
    "pārskatīt": "atzīme"
}


STOPWORDS = {
    "kur", "kā", "kas", "vai", "es", "man", "mani", "manu", "ir",
    "būtu", "var", "varu", "ar", "par", "no", "uz", "pa", "pie",
    "ko", "kam", "kad", "lai", "atrast", "redzēt", "dabūt", "notiek",
    "gribu", "vajag", "manam", "mana", "mans", "lūdzu", "parādīt"
}


def normalize_text(text: str):
    text = text.lower()
    text = re.sub(r"[^\w\sāčēģīķļņšūž-]", "", text)
    return text


def simple_stem(word: str):
    endings = [
        "ām", "iem", "ais", "ajā", "ajai", "oju", "oju", "iem",
        "am", "as", "ai", "us", "os", "ēm", "ēs",
        "u", "a", "i", "e", "s", "o"
    ]

    for ending in endings:
        if word.endswith(ending) and len(word) > len(ending) + 2:
            return word[:-len(ending)]

    return word


def normalize_word(word: str):
    word = normalize_text(word)
    word = SYNONYMS.get(word, word)
    word = simple_stem(word)
    return word


def tokenize(text: str):
    words = normalize_text(text).split()
    result = []

    for word in words:
        normalized = normalize_word(word)
        if normalized and normalized not in STOPWORDS:
            result.append(normalized)

    return result


def partial_match(token1: str, token2: str):
    if token1 == token2:
        return 1.0

    min_len = min(len(token1), len(token2))
    if min_len < 4:
        return 0.0

    # prefiksa līdzība
    common_prefix = 0
    for a, b in zip(token1, token2):
        if a == b:
            common_prefix += 1
        else:
            break

    similarity = common_prefix / max(len(token1), len(token2))

    if similarity >= 0.6:
        return similarity

    return 0.0


def score_keyword(user_tokens: list, keyword: str):
    keyword_tokens = tokenize(keyword)

    if not keyword_tokens:
        return 0

    score = 0.0

    # Pilna frāzes sakritība
    if len(keyword_tokens) > 1:
        matched = 0
        for kt in keyword_tokens:
            token_score = 0
            for ut in user_tokens:
                sim = partial_match(kt, ut)
                if sim > token_score:
                    token_score = sim
            if token_score > 0:
                matched += 1
                score += token_score

        if matched == len(keyword_tokens):
            score += 3
        elif matched > 0:
            score += matched * 0.5

    else:
        kt = keyword_tokens[0]
        best = 0
        for ut in user_tokens:
            sim = partial_match(kt, ut)
            if sim > best:
                best = sim

        score += best * 2

    return score


def find_best_matches(user_message: str, top_n: int = 3):
    user_tokens = tokenize(user_message)
    scored_items = []

    for item in FAQ_DATA:
        total_score = 0.0

        for keyword in item["keywords"]:
            total_score += score_keyword(user_tokens, keyword)

        scored_items.append({
            "item": item,
            "score": round(total_score, 2)
        })

    scored_items.sort(key=lambda x: x["score"], reverse=True)
    return scored_items[:top_n]


def find_answer(user_message: str):
    matches = find_best_matches(user_message, top_n=3)
    best = matches[0] if matches else None

    if best and best["score"] >= 2.0:
        item = best["item"]
        return {
            "reply": item["answer"],
            "source": item.get("source", ""),
            "url": item.get("url", ""),
            "next_step": item.get("next_step", ""),
            "matched_question": item.get("question", ""),
            "score": best["score"],
            "suggestions": []
        }

    suggestions = []
    for match in matches:
        if match["score"] > 0:
            suggestions.append(match["item"].get("question", ""))

    return {
        "reply": "Es neatradu pietiekami precīzu atbildi.",
        "source": "",
        "url": "",
        "next_step": "Pamēģini uzdot jautājumu citādi vai izvēlies kādu no līdzīgajām tēmām.",
        "matched_question": "",
        "score": best["score"] if best else 0,
        "suggestions": suggestions[:3]
    }


def log_chat(user_message: str, result: dict):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "user_message": user_message,
        "reply": result.get("reply", ""),
        "source": result.get("source", ""),
        "url": result.get("url", ""),
        "next_step": result.get("next_step", ""),
        "matched_question": result.get("matched_question", ""),
        "score": result.get("score", 0),
        "suggestions": result.get("suggestions", [])
    }

    with open("logs/chat_logs.jsonl", "a", encoding="utf-8") as file:
        file.write(json.dumps(log_entry, ensure_ascii=False) + "\n")


@app.get("/")
def read_root():
    return FileResponse("static/index.html")


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/chat")
def chat(request: ChatRequest):
    result = find_answer(request.message)
    log_chat(request.message, result)
    return result