#!/usr/bin/env python3
# better_split_scispacy.py
# Robust sentence separation for S2ORC-like JSON using SciSpaCy + PDF artifacts cleanup.
# Now with: merging very short sentences into the previous one (configurable threshold).

import json, re, argparse, unicodedata
from collections import defaultdict, OrderedDict

import spacy

# ---------- Regexes for cleanup ----------
URL_RE = re.compile(r'(https?://\S+|www\.\S+)', re.IGNORECASE)
NATURE_LICENSE_RE = re.compile(
    r'Users may view, print, copy, and download.*?editorial_policies/license\.html#terms',
    re.IGNORECASE | re.DOTALL
)
MULTISPACE_RE = re.compile(r'\s+')

# [6] [7] [8] trains  -> <CIT>
BRACKETED_CITS_RE = re.compile(r'(?:\[\s*\d+\s*\]\s*){1,}', re.UNICODE)
# numeric trains that are clearly citations: ", 12" ", 3, 4" near words
INLINE_NUM_CITS_RE = re.compile(r'(?<=\w)\s*(?:\d{1,3}\s*,\s*)+\d{1,3}\b')

# Orphan figure/panel fragments like "Fig." / "1a )." / "2c )" / ")"
FIG_TOKEN_RE   = re.compile(r'^(fig\.?|figure\.?)$', re.IGNORECASE)
PANEL_FRAG_RE  = re.compile(r'^[A-Za-z]?\d+[a-z]?\s*\)\.?$')  # "1a ).", "2c )"
JUST_PAREN_RE  = re.compile(r'^[\)\(]+\.?$')

# Caption/callout prefixes to route/ignore if wanted
CAPTION_START_RE = re.compile(r'^(supplementary|table|fig\.?|figure)\b', re.IGNORECASE)

# ---------- Helpers ----------
def nfkc(s: str) -> str:
    return unicodedata.normalize("NFKC", s or "")

def clean_block_text(t: str) -> str:
    """Lightweight, targeted cleanup before sentence splitting."""
    if not t: return ""
    t = nfkc(t)

    # Drop publisher license blobs
    t = NATURE_LICENSE_RE.sub(' ', t)
    # Strip URLs (keep prose clean)
    t = URL_RE.sub(' ', t)
    # Collapse bracketed citation trains
    t = BRACKETED_CITS_RE.sub(' <CIT> ', t)
    # Normalize inline numeric citation trains like "11, 12"
    t = INLINE_NUM_CITS_RE.sub(' <CIT> ', t)

    # Normalize whitespace
    t = MULTISPACE_RE.sub(' ', t).strip()
    return t

def build_nlp(model: str = "en_core_sci_md") -> spacy.language.Language:
    """
    Load a SciSpaCy model and ensure we have a sentence boundary component.
    SciSpaCy models usually include sentence segmentation; if not, add sentencizer.
    """
    try:
        nlp = spacy.load(model, disable=[])
    except OSError as e:
        raise RuntimeError(
            f"Could not load SciSpaCy model '{model}'. "
            f"Install it, e.g.:\n"
            f"  pip install spacy\n"
            f"  pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/{model}-0.5.4.tar.gz"
        ) from e
    # Ensure sentence boundaries exist
    if "senter" not in nlp.pipe_names and "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer", first=True)
    return nlp

def sent_split(nlp: spacy.language.Language, text: str) -> list[str]:
    if not text: return []
    doc = nlp(text)
    return [s.text.strip() for s in doc.sents if s.text.strip()]

def should_route_caption(sent: str) -> bool:
    return bool(CAPTION_START_RE.match(sent.strip()))

def is_orphan_fragment(sent: str) -> bool:
    s = sent.strip()
    if not s: return True
    if len(s) <= 2: return True
    if FIG_TOKEN_RE.match(s): return True
    if PANEL_FRAG_RE.match(s): return True
    if JUST_PAREN_RE.match(s): return True
    return False

def merge_fragments(sentences: list[str]) -> list[str]:
    """Merge orphan fragments into the previous real sentence when possible."""
    out = []
    for s in sentences:
        if out and is_orphan_fragment(s):
            out[-1] = (out[-1] + ' ' + s).strip()
        else:
            out.append(s)
    if len(out) >= 2 and is_orphan_fragment(out[-1]):
        out[-2] = (out[-2] + ' ' + out[-1]).strip()
        out.pop()
    return out

# ---------- NEW: merge very short sentences safely ----------
SHORT_SENT_EXCEPTIONS = tuple([
    "see methods.", "data not shown.", "see supplementary.", "see supplement.",
    "results follow.", "conclusion follows.", "in summary.", "in conclusion.",
])

ACRONYM_RE = re.compile(r'^[A-Z]{2,6}\.?$')  # e.g., EMDB., PDB.

def is_exception_short(s: str) -> bool:
    st = s.strip()
    if not st:
        return True
    # Keep short imperative/info lines that are common in scientific prose
    if st.lower() in SHORT_SENT_EXCEPTIONS:
        return True
    # Keep pure acronyms like "EMDB." or "PDB."
    if ACRONYM_RE.match(st.replace(" ", "")):
        return True
    # Keep questions/exclamations as standalone
    if st.endswith("?") or st.endswith("!"):
        return True
    return False

def merge_short_sentences(sentences: list[str], min_words: int = 6) -> list[str]:
    """
    Merge very short sentences into the previous one, with exceptions.
    - Only merges within the same chunk call (we already process per paragraph chunk).
    - Does NOT merge across section/paragraph boundaries (by construction).
    """
    if not sentences:
        return sentences

    out = [sentences[0]]
    for s in sentences[1:]:
        wc = len(s.split())
        if wc < min_words and not is_exception_short(s):
            # Merge to previous; add a space if previous doesn't end with punctuation
            prev = out[-1].rstrip()
            sep = "" if (prev.endswith(('.', '!', '?', ':'))) else " "
            out[-1] = (prev + sep + s.strip()).strip()
        else:
            out.append(s)
    return out

# ---------- Main flattening ----------
def flatten(input_path: str, output_path: str, keep_captions: bool = False, model: str = "en_core_sci_md", short_merge_threshold: int = 6):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    nlp = build_nlp(model=model)
    out = []
    captions = []

    # Title
    title = nfkc(data.get('title') or data.get('pdf_parse', {}).get('title') or '')
    if title:
        out.append({'sid': 'title-0-0', 'section': 'Title', 'text': title})

    # Abstract
    abs_chunks = (data.get('pdf_parse', {}) or {}).get('abstract') or []
    if not abs_chunks and data.get('abstract'):
        abs_chunks = [{'text': data['abstract']}]
    for ai, ch in enumerate(abs_chunks):
        text = clean_block_text(ch.get('text', ''))
        sents = sent_split(nlp, text)
        sents = merge_fragments(sents)
        sents = merge_short_sentences(sents, min_words=short_merge_threshold)
        for si, s in enumerate(sents):
            out.append({'sid': f'abstract-{ai}-{si}', 'section': 'Abstract', 'text': s})

    # Body
    body = (data.get('pdf_parse', {}) or {}).get('body_text') or []
    section_order = OrderedDict()
    para_counters = defaultdict(int)

    for chunk in body:
        raw_sec = nfkc(chunk.get('section') or 'Unsectioned')
        if raw_sec not in section_order:
            section_order[raw_sec] = len(section_order)
        sec_idx = section_order[raw_sec]

        text = clean_block_text(chunk.get('text', ''))
        if not text:
            continue

        sents = sent_split(nlp, text)
        sents = merge_fragments(sents)
        sents = merge_short_sentences(sents, min_words=short_merge_threshold)

        routed = []
        for s in sents:
            if should_route_caption(s):
                captions.append({'section': raw_sec, 'text': s})
            else:
                routed.append(s)
        sents = routed if not keep_captions else sents

        if not sents:
            continue

        para_idx = para_counters[raw_sec]
        for si, s in enumerate(sents):
            out.append({
                'sid': f'{sec_idx}-{para_idx}-{si}',
                'section': raw_sec,
                'text': s
            })
        para_counters[raw_sec] += 1

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(out)} sentences to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Better sentence separation for S2ORC-like JSON (SciSpaCy).")
    parser.add_argument("-i", "--input", required=True, help="Path to S2ORC-like JSON (e.g., test_bio.json)")
    parser.add_argument("-o", "--output", default="sentences_better_scispacy.json", help="Output sentences JSON")
    parser.add_argument("--keep-captions", action="store_true", help="Keep caption-like lines in main output")
    parser.add_argument("--model", default="en_core_sci_lg", help="SciSpaCy model name (e.g., en_core_sci_md, en_core_sci_lg)")
    parser.add_argument("--short-merge-threshold", type=int, default=6, help="Merge sentences shorter than this many words into the previous sentence")
    args = parser.parse_args()

    flatten(
        args.input,
        args.output,
        keep_captions=args.keep_captions,
        model=args.model,
        short_merge_threshold=args.short_merge_threshold
    )
