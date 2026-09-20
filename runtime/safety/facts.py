"""Clinical fact extraction for the guidance layer. Pure regex/keyword
matching (EN + core Kiswahili clinical terms), microseconds, no LLM call.

Returns {"facts": set[str], "details": {...}}. Exponents of doubt resolve
toward the pediatric/urgent reading (over-triage beats under-triage at
this layer; the model + lint + clinician review carry precision).
"""
from __future__ import annotations

import re

# fact -> list of patterns (compiled IGNORECASE)
_PATTERNS: dict[str, list[str]] = {
    "child_under5": [
        r"\b([0-4])[-\s]?year[-\s]?old\b", r"\btoddler\b",
        r"\b(my|our|the|a)\s+(child|kid|little one)\b",
        r"\b(my|our|the|his|her)\s+(little\s+)?(girl|boy|son|daughter)\b",
        r"\bchild\b.{0,20}\b(boy|girl|son|daughter)\b",
        r"\bchild\s*,?\s*\d", r"\b([0-4])\s?yo\b",
        r"(^|\.\s+)child\b",
        r"\bmtoto\b", r"\b(1[0-9]|2[0-3]|[0-9])[- ]month[- ]old\b",
        r"\bbaby\b", r"\binfant\b",
    ],
    "infant_under2mo": [
        r"\b([0-1])[- ]month[- ]old\b", r"\bnewborn\b",
        r"\bjust born\b", r"\b([0-9]|[1-5][0-9])[- ]days?[- ]old\b",
        r"\b([1-7])[- ]weeks?[- ]old\b", r"\bmchanga\b",
    ],
    "neonate": [r"\bnewborn\b", r"\bjust born\b",
                r"\b([0-9]|[12][0-8])[- ]days?[- ]old\b"],
    "pregnant": [r"\bpregnan", r"\bexpecting( a baby)?\b",
                 r"\btrimester\b", r"\bweeks? pregnant\b",
                 r"\bmonths? pregnant\b", r"\bmjamzito\b"],
    "postpartum": [r"\bjust gave birth\b", r"\bgave birth\b",
                   r"\bgiving birth\b", r"\bpostpartum\b",
                   r"\bafter (birth|deliver)",
                   r"\bdelivered (a|her|the) baby\b"],
    "fever": [r"\bfever\b", r"\bfebrile\b", r"\bhigh temp",
              r"\btemperature\b.{0,15}\d",
              r"\b(has|with|running)\b.{0,10}\btemperature\b",
              r"\bhoma\b", r"\bpyrexia\b"],
    "rapid_breathing": [r"\b(fast|rapid|quick) (breathing|breaths?)\b",
                        r"\bbreathing (fast|rapidly|quickly)\b",
                        r"\bbreathing\b.{0,15}\b(fast|rapid|quick)\b",
                        r"\bbreathes?\b.{0,15}\b(fast|rapidly|quickly)\b",
                        r"\b([4-9]\d|1\d\d)\s*(times a minute|per minute|/min)\b",
                        r"\bRR\b.{0,5}\d+", r"\brespiratory rate\b",
                        r"\bbreathing rate\b",
                        r"\banapumua\b.{0,10}\bharaka\b"],
    "chest_indrawing": [r"\bchest indrawing\b", r"\bindrawing\b",
                        r"\bribs?\b.{0,15}(showing|visible|sticking|sucking in)\b",
                        r"\bchest\b.{0,10}\b(pulls?|pulling)\s+in\b",
                        r"\bretractions?\b",
                        r"\bmbavu\b.{0,15}(zinaonekana|kuonekana)"],
    "breathing_difficulty": [r"\bdifficult(y|ies).{0,15}breath",
                             r"\btrouble breathing\b",
                             r"\bshort(ness|age)? of breath\b",
                             r"\bbreathless\b", r"\bdyspn",
                             r"\bstruggling to breathe\b",
                             r"\bshida kupumua\b", r"\bkupumua kwa shida\b"],
    "cyanosis": [r"\bblue lips?\b", r"\blips?.{0,15}blue\b",
                 r"\bblue (face|nails|skin|around the mouth)\b",
                 r"\bcyanos"],
    "cannot_speak_full": [r"\bcan'?t (speak|talk).{0,25}(full|complete) sentences?\b",
                          r"\bunable to (speak|talk).{0,25}(full|complete)",
                          r"\bonly a few words\b"],
    "inability_drink": [r"\b(can'?t|cannot|unable to|not able to) (drink|swallow)\b",
                        r"\brefus\w*\b.{0,12}\bdrink\w*\b",
                        r"\bnot drinking\b", r"\bhawezi kunywa\b"],
    "vomiting_persistent": [r"\bvomit.{0,25}(repeatedly|keeps?|everything|all day|persistent|won'?t stop|nonstop)\b",
                            r"\bkeeps? vomiting\b", r"\bcan'?t keep (anything|fluids|food) down\b",
                            r"\bpersistent vomit"],
    "lethargy": [r"\bletharg", r"\bunusually sleepy\b", r"\bhard to wake\b",
                 r"\bwon'?t wake\b", r"\bfloppy\b", r"\blistless\b",
                 r"\bdrowsy\b", r"\bvery sleepy\b"],
    "convulsions": [r"\bseizure\b", r"\bconvuls", r"\bfit(s|ting)?\b.{0,15}(shak|unconscious|epilep)",
                    r"\bshaking\b.{0,20}\bseizure\b", r"\bepilep"],
    "seizure_active": [r"\bseizure\b.{0,30}(right now|ongoing|won'?t stop|minutes|currently)\b",
                       r"\bshaking\b.{0,30}(right now|unconscious|minutes)\b"],
    "unconscious": [r"\bunconscious\b", r"\bunresponsive\b",
                    r"\bpassed out\b", r"\bcoma\b", r"\bknocked out\b"],
    "diarrhea": [r"\bdiarrh", r"\bloose stools?\b", r"\bwatery stool",
                 r"\b\w{0,4}harisha\b", r"\bkuhara\b"],
    "dehydration_signs": [r"\bsunken eyes?\b", r"\bno tears\b",
                          r"\bvery thirsty\b", r"\bdry mouth\b",
                          r"\bskin\b.{0,25}(slowly|tenting|pinch)",
                          r"\bmacho\b.{0,15}(yameingia|yamezama)"],
    "reduced_urine": [r"\bno urine\b", r"\bnot (urinating|peeing)\b",
                      r"\breduced urine\b", r"\bvery little urine\b",
                      r"\bhasn'?t (urinated|peed)\b",
                      r"\bhardly\b.{0,12}\b(urine|urinat)",
                      r"\bhajaenda haja ndogo\b", r"\bhakuna haja ndogo\b",
                      r"\bhajakojoa\b"],
    "jaundice": [r"\bjaundice\b", r"\byellow\b.{0,15}\beyes?\b",
                 r"\byellow skin\b"],
    "poor_feeding": [r"\b(not|poor|refus).{0,15}(feeding|breastfeed)",
                     r"\bwon'?t breastfeed\b", r"\bstopped feeding\b"],
    "wheeze": [r"\bwheez", r"\bwhistling\b.{0,10}\b(breath|chest)"],
    "chest_pain": [r"\bchest pain\b", r"\bchest pressure\b",
                   r"\bcrushing\b.{0,15}\bchest\b", r"\btight chest\b",
                   r"\bchest tightness\b", r"\bmaumivu ya kifua\b"],
    "bleeding_severe": [r"\b(heavy|severe|serious|massive) bleeding\b",
                        r"\bbleeding heavily\b", r"\bblood everywhere\b",
                        r"\bsoaking\b.{0,15}\b(cloth|pad|dressing)",
                        r"\bspurting\b", r"\barterial\b",
                        r"\bcan'?t stop (the )?bleed",
                        r"\bdamu\b.{0,25}\bsana\b"],
    "open_fracture": [r"\bcompound fracture\b", r"\bopen fracture\b",
                      r"\bbone\b.{0,20}(sticking out|exposed|through (the )?skin|showing|visible|protruding)",
                      r"\bmangled\b.{0,10}\b(limb|arm|leg|hand|foot)",
                      r"\bamputat"],
    "reduced_fetal_movement": [r"\breduced fetal movement\b",
                               r"\bbaby\b.{0,25}(not moving|barely mov|hardly mov|less mov|no mov|stopped (moving|kicking))",
                               r"\b(decreased|less|reduced|no)\s+(fetal )?movements?\b",
                               r"\bkicks?\b.{0,30}(stopped|reduced|less|fewer)\b",
                               r"\bmovements?\b.{0,15}(stopped|reduced|decreased|less|ceased)\b",
                               r"\b(movement|kicks?)\b.{0,35}\balmost nothing\b",
                               r"\bmoving\b.{0,10}\bless\b",
                               r"\b(fewer|less)\b.{0,10}\bkicks?\b",
                               r"\bnot\b.{0,15}\bfelt\b.{0,15}\b(baby|movement|kicks?|move)\b",
                               r"\bhatikisiki\b"],
    "headache_severe_preg": [r"\bsevere headache\b", r"\bworst headache\b"],
    "visual_symptoms_preg": [r"\b(blurry|blurred) vision\b",
                             r"\bseeing spots\b", r"\bvision (changes|spots|loss)\b",
                             r"\bblind spots\b"],
    "swelling_face": [r"\bface\b.{0,15}(swell|swollen)",
                      r"\bswollen\b.{0,15}(face|eyes|hands)",
                      r"\bsudden swelling\b"],
    "fluid_leakage_preg": [r"\bwater broke\b", r"\bleaking fluid\b",
                           r"\bfluid leakage\b", r"\bgush of fluid\b"],
    "caustic_ingestion": [r"\b(drink\w*|drank|swallow\w*|ingest\w*|lick\w*).{0,30}(bleach|jik|hypo|acid|caustic|lye)\b",
                          r"\b(bleach|jik)\b.{0,20}(drink\w*|mouth|swallow\w*)",
                          r"\b(kunywa|amekunywa|anakunywa)\b.{0,30}(jik|bleach|hypo|acid)"],
    "poison_ingestion": [r"\bswallow\w*\b.{0,25}(pills?|tablets?|medicine|bottle)\b",
                         r"\boverdose\b", r"\brate poison\b",
                         r"\bpesticide\b.{0,20}(drink|swallow|ingest|spray.{0,10}indoors)",
                         r"\bkerosene\b.{0,15}(drink|swallow)",
                         r"\b(drink\w*|drank|swallow\w*|ingest\w*)\b.{0,25}\b(pesticide|kerosene|poison)\b",
                         r"\bate\b.{0,20}tablets?\b"],
    "anaphylaxis_signs": [r"\bthroat\b.{0,15}(closing|tightening|swell)",
                          r"\btongue\b.{0,15}swell",
                          r"\b(lips?|face|tongue)\b.{0,30}swell.{0,40}(wheez|breath|vomit|dizz|faint|collaps)"],
    "altered_consciousness": [r"\bconfus(ed|ion)\b", r"\bdisoriented\b",
                              r"\bdoesn'?t recognize\b",
                              r"\bacting strange\b"],
    "shock_signs": [r"\bcollapsed\b", r"\bcold (and|,) sweaty\b",
                    r"\bclammy\b", r"\bpale\b.{0,15}(cold|sweaty|clammy)",
                    r"\bweak pulse\b"],
    "malaria_risk": [r"\bmalaria\b", r"\bmosquito\b", r"\bmbu\b"],
    "stroke_signs": [r"\bface\b.{0,20}droop", r"\barm\b.{0,20}(weak|drift|can'?t lift)",
                     r"\bslurred speech\b", r"\bsudden\b.{0,20}(weakness|numbness|speech|vision)"],
    "diabetes": [r"\bdiabet", r"\bblood sugar\b", r"\binsulin\b"],
    "dosing_request": [r"\bhow much\b.{0,30}(mg|dose|tablet|syrup|give|paracetamol|medicine)\b",
                       r"\bwhat dose\b", r"\bdosage\b", r"\bhow many\b.{0,20}tablets?\b"],
    "high_risk_med": [r"\bwarfarin\b", r"\bheparin\b", r"\bapixaban\b",
                      r"\brivaroxaban\b", r"\bblood thinner\b", r"\binsulin\b",
                      r"\bmorphine\b", r"\btramadol\b", r"\boxycodone\b",
                      r"\bcodeine\b", r"\bopioid\b", r"\bdigoxin\b",
                      r"\blithium\b", r"\bphenytoin\b", r"\bmethotrexate\b",
                      r"\btacrolimus\b", r"\bgentamicin\b",
                      r"\bantiretroviral\b", r"\b\bart\b.{0,10}\b(hiv|medicine|drug)",
                      r"\brifampicin\b"],
    "suicidal_intent": [r"\bsuicid", r"\bkill (my|him|her|them)sel",
                        r"\bend (my|his|her) life\b", r"\bwants? to die\b",
                        r"\bno reason to live\b"],
    "sti_concern": [r"\bsti\b", r"\bstd\b", r"\bgonorrh",
                    r"\bchlamydia\b", r"\bsyphilis\b",
                    r"\bgenital\b.{0,20}(sore|ulcer|discharge|warts?)"],
    "sexual_assault": [r"\brape[ds]?\b", r"\bsexually assaulted\b",
                       r"\bforced\b.{0,15}\bsex\b", r"\bdefiled\b"],
    "outbreak_pattern": [r"\b(village|community|neighbors?|school|several|many people|everyone|everybody|many|multiple)\b.{0,40}(diarrh|vomit|fever|bleed|rash|rice-water|sick|dying)\b",
                         r"\brice-water\b", r"\bmaji ya mchele\b"],
    "ors_question": [r"\bors\b.{0,30}(prepare|make|mix|use|give|given|take|recipe)\b",
                     r"\bhow\b.{0,25}(prepare|make|mix).{0,20}\b(ors|rehydration)\b",
                     r"\b(mix|prepare|make|use|give|given|take)\b.{0,25}\b(ors|oral rehydration)\b",
                     r"\boral rehydration\b"],
    "nigeria": [r"\bnigeria\b", r"\blagos\b", r"\babuja\b", r"\bkano\b",
                r"\bibadan\b", r"\bncdc\b"],
    "care_far": [r"\bhours? away\b", r"\bfar from\b.{0,15}(hospital|clinic|care)\b",
                 r"\bno hospital\b.{0,10}(near|nearby|close)"],
}

_COMPILED = {k: [re.compile(p, re.I) for p in v]
             for k, v in _PATTERNS.items()}

_NEG = re.compile(r"\b(no|not|denies|denied|without|never had)\b", re.I)


def _negated(text: str, m: re.Match) -> bool:
    """True if a negation word sits within 3 tokens before the match,
    not crossing a sentence boundary."""
    before = text[max(0, m.start() - 60):m.start()]
    before = re.split(r"[.!?;:]\s*", before)[-1]
    return bool(_NEG.search(" ".join(before.split()[-3:])))


def extract_facts(text: str) -> dict:
    """Extract clinical facts. Returns {"facts": set, "details": {...}}."""
    text = text or ""
    facts: set[str] = set()
    for fact, rxs in _COMPILED.items():
        for rx in rxs:
            m = rx.search(text)
            if m and not _negated(text, m):
                facts.add(fact)
                break
    # Combos: pregnancy-gated facts.
    if "pregnant" in facts:
        m = re.search(r"\b(bleed\w*|spotting|blood)\b", text, re.I)
        if m and not _negated(text, m):
            facts.add("bleeding_vaginal_preg")
    # Age refinement: explicit older-child ages remove the under-5 default.
    older = re.search(r"\b([5-9]|1[0-7])[- ]year[- ]old\b", text, re.I)
    if older and "infant_under2mo" not in facts and "neonate" not in facts:
        facts.discard("child_under5")
    # infant implies child-under-5 umbrella for card coverage.
    if "infant_under2mo" in facts or "neonate" in facts:
        facts.add("child_under5")
    return {"facts": facts, "details": {"chars": len(text)}}
