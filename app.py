from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import spacy, re, dateparser, tempfile, os
from typing import List, Optional

# Load spaCy English pipeline for PERSON / MONEY / DATE
# Run once: python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")

try:
    from faster_whisper import WhisperModel
    # Use a small model with low precision for CPU
    WHISPER_MODEL = WhisperModel("small", device="cpu", compute_type="int8")
except Exception:
    WHISPER_MODEL = None  # Graceful fallback if model not installed

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

class ParseIn(BaseModel):
    text: str

def extract_amount(text: str, doc):
    money_ents = [ent for ent in doc.ents if ent.label_ == "MONEY"]
    if money_ents:
        m = re.search(r"[\d,]+(?:\.\d{1,2})?", money_ents[0].text.replace(",", ""))
        if m:
            try:
                return float(m.group(0))
            except:
                pass
    m = re.search(r"(?:₹\s*)?(\d+(?:\.\d{1,2})?)", text)
    return float(m.group(1)) if m else None

def extract_payer(text: str, doc):
    persons = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    if not persons:
        return None
    paid_idx = text.lower().find("paid")
    if paid_idx != -1:
        best = None
        for ent in doc.ents:
            if ent.label_ == "PERSON" and ent.end_char <= paid_idx:
                best = ent.text
        if best: return best
    return persons[0]

def extract_participants(text: str, doc, payer: Optional[str]):
    persons = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    if payer and payer in persons:
        persons = [p for p in persons if p != payer]
    lower = text.toLowerCase() if hasattr(text, 'toLowerCase') else text.lower()
    for kw in [" for ", " with ", " among "]:
        if kw in lower:
            tail = text[lower.find(kw)+len(kw):]
            names = re.split(r",| and ", tail)
            cands = []
            for n in names:
                n = n.strip()
                for ent in doc.ents:
                    if ent.label_ == "PERSON" and ent.text in n:
                        cands.append(ent.text)
            if cands:
                # de-duplicate preserving order
                seen = set()
                res = []
                for c in cands:
                    if c not in seen:
                        res.append(c); seen.add(c)
                return res
    # fallback
    seen = set()
    res = []
    for p in persons:
        if p not in seen:
            res.append(p); seen.add(p)
    return res

def extract_date(text: str, doc):
    dates = [ent.text for ent in doc.ents if ent.label_ == "DATE"]
    if not dates:
        return None
    dt = dateparser.parse(dates[0])
    return dt.date().isoformat() if dt else None

def extract_desc(text: str, amount, payer, participants):
    m = re.search(r"\bfor\b(.+)", text, flags=re.IGNORECASE)
    if m:
        desc = m.group(1).strip().strip(".")
        return desc[:120]
    m2 = re.search(r"\bpaid\b(.*?)\bfor\b", text, flags=re.IGNORECASE)
    if m2:
        desc = m2.group(1).strip(" .,-:")
        if desc:
            return desc[:120]
    t = text
    if payer:
        t = re.sub(re.escape(payer), "", t)
    for p in participants or []:
        t = re.sub(re.escape(p), "", t)
    if amount:
        t = re.sub(str(amount), "", t)
    t = re.sub(r"\s+", " ", t).strip(" .,-:")
    return (t or "Expense").split(",")[0][:120]

@app.post("/parse_text")
def parse_text(inp: ParseIn):
    text = inp.text.strip()
    if not text:
        return {"desc": None, "amount": None, "payer": None, "participants": [], "date_iso": None}
    doc = nlp(text)
    amount = extract_amount(text, doc)
    payer  = extract_payer(text, doc)
    participants = extract_participants(text, doc, payer)
    date_iso = extract_date(text, doc)
    desc = extract_desc(text, amount, payer, participants)
    return {
        "desc": desc, "amount": amount, "payer": payer,
        "participants": participants, "date_iso": date_iso
    }

@app.post("/transcribe")
def transcribe(file: UploadFile = File(...)):
    if WHISPER_MODEL is None:
        return {"error": "Whisper model not available"}
    suffix = os.path.splitext(file.filename or "audio.webm")[-1] or ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file.file.read())
        tmp_path = tmp.name
    try:
        segments, info = WHISPER_MODEL.transcribe(tmp_path, language="en", beam_size=5)
        text = "".join([seg.text for seg in segments]).strip()
        return {"text": text}
    finally:
        os.unlink(tmp_path)
