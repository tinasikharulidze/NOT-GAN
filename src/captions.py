"""
Turns a CartoonSet attribute row into a randomized natural-language caption
for CLIP conditioning (Section 6.2 of the pdf report).

Deliberately randomized at every level -- synonym choice, feature ordering,
opener phrase -- so the fine-tuned CLIP encoder sees the full distribution
of ways a real user might phrase the same request (see Appendix B of the
thesis for the full caption-generation logic writeup).
"""

import random


def generate_clip_caption(row):
    """
    Safely sanitizes row datatypes before dictionary lookups.
    Integrates the 111 hair geometry system with dynamic hair color synonyms, 
    facial hair maps, and explicit negative-prompt phrases for both 
    missing glasses and clean-shaven faces.
    """
    # Helper to force safe integer evaluation from pandas float/string formats
    def safe_int(val):
        try:
            return int(float(str(val).strip()))
        except (ValueError, TypeError):
            return 0

    # Extract clean integer targets safely
    face_color_idx = safe_int(row.get('face_color', 0))
    face_shape_idx = safe_int(row.get('face_shape', 0))
    chin_len_idx   = safe_int(row.get('chin_length', 0))
    eye_color_idx  = safe_int(row.get('eye_color', 0))
    eyebrow_idx    = safe_int(row.get('eyebrow_thickness', 0))
    hair_idx       = safe_int(row.get('hair', 0))
    hair_color_idx = safe_int(row.get('hair_color', 0)) 
    facial_idx     = safe_int(row.get('facial_hair', 14))
    glasses_idx    = safe_int(row.get('glasses', 11))
    g_color_idx    = safe_int(row.get('glasses_color', 0))

    # --- DICTIONARY DIALECT MAPS ---
    openers = [
        "A vector illustration of", 
        "An artistic cartoon avatar of", 
        "A clean 2D graphic of", 
        "A cartoon profile depicting"
    ]
    
    face_shapes = {
        0: ["a long, rectangular face", "an oblong head shape with a flat chin", "a rigid rectangular jawline"],
        1: ["a diamond facial structure", "prominent cheekbones with a narrow chin", "a geometric diamond face silhouette"],
        2: ["a pear-shaped jaw", "a wide, heavy lower jawline", "a triangular face widening at the base"],
        3: ["a perfectly round face", "a circular head outline", "soft, rounded facial contours"],
        4: ["a square jawline", "a boxy, sharp face shape", "a distinct angular face"],
        5: ["a heart-shaped face with a pointed chin", "a tapered triangular face", "a delicate inverted-triangle head shape"],
        6: ["an oval countenance", "an elongated, smooth face", "a classic oval head structure"]
    }

    chin_lengths = {
        0: ["a petite, short chin", "a subtle jaw edge"],
        1: ["a balanced, medium-sized chin", "a normal jaw length"],
        2: ["an elongated, long chin", "a highly prominent jaw extension"]
    }
    
    face_colors = {
        0: ["an ultra-deep, dark skin tone", "a rich, velvety dark complexion", "deep midnight skin tones"],
        1: ["deep dark skin", "a highly saturated dark complexion", "a very deep skin tone"],
        2: ["dark brown skin", "a deep espresso complexion", "a striking dark skin tone"],
        3: ["medium deep brown skin", "a rich milk-chocolate complexion", "a prominent brown skin tone"],
        4: ["light brown skin", "a smooth caramel complexion", "a medium-brown skin tone"],
        5: ["tan, olive skin", "a deeply bronzed complexion", "rich warm tan skin"],
        6: ["medium warm skin", "an olive complexion", "a healthy neutral-toned skin tone"],
        7: ["a warm, light skin tone", "subtle golden undertones", "a light warm yellowish complexion"],
        8: ["a fair skin tone", "a light peach complexion", "natural fair skin"],
        9: ["very fair, pale skin", "a light pinkish complexion", "pale fair skin"],
        10: ["extremely pale, alabaster skin", "a stark off-white complexion", "a very light bleached skin tone"]
    }
    
    eye_colors = {
        0: ["brown eyes", "dark brown eyes", "classic brown eyes", "deep brown eyes"],
        1: ["blue eyes", "bright blue eyes", "vivid blue eyes", "dark blue eyes"],
        2: ["green eyes", "vivid green eyes", "bright green eyes", "dark green eyes"],
        3: ["grey eyes", "pale grey eyes", "cool grey eyes", "light grey eyes"],
        4: ["black eyes", "solid black eyes", "dark black eyes", "pitch black eyes"]
    }

    eyebrow_thicknesses = {
        0: ["delicately ultra-thin eyebrows", "barely visible brow lines"],
        1: ["slender, thin eyebrows", "neatly trimmed, fine brows"],
        2: ["moderately arched eyebrows", "medium-weight brow structures"],
        3: ["bold eyebrows", "heavy, thick prominent brows"]
    }

    # --- HAIR COLOR SYNONYM MAP ---
    hair_colors = {
        0: ["very light blonde", "platinum blonde", "pale bleach-blonde", "ultra-light blonde"],
        1: ["blonde", "golden blonde", "honey blonde", "classic blonde"],
        2: ["orange", "vibrant orange", "bright orange", "ginger"],
        3: ["red", "fiery red", "crimson red", "vivid red"],
        4: ["caramel", "warm caramel", "rich caramel", "light brown tone"],
        5: ["brunette", "classic brunette", "chestnut brown", "medium brown"],
        6: ["dark brown", "deep dark brown", "espresso brown", "rich dark brown"],
        7: ["black", "jet black", "pitch black", "midnight black"],
        8: ["grey", "ash grey", "charcoal grey", "silver-grey"],
        9: ["silver", "completely white", "pale silver", "bright silver", "white"]
    }

    # ── HAIR STYLE GEOMETRY (111 styles, IDs 0–110) ────────────────────────
    _special_hair = {
        0:   ["a completely shaved head", "a bald profile", "no hair at all"],
        1:   ["an almost bald style",  "a closely shaved head", "a very short crop"],
        2:   ["a buzz cut",            "a very short crop",     "a shaved style"],
        3:   ["twin pigtails",         "dual side pigtails",    "symmetrical pigtails"],
        4:   ["a voluminous afro",     "a large round afro",    "a full natural afro"],
        5:   ["long voluminous curls", "a big curly mane",      "flowing full ringlets"],
        6:   ["long dreadlocks",       "flowing locs",          "long twisted dreadlocks"],
        13:  ["long curls",            "long curly hair",       "lengthy ringlets"],
        15:  ["long curls",            "long curly hair",       "flowing ringlets"],
        20:  ["a very short crop",     "a near-bald style",     "closely shaved hair"],
        21:  ["a very short crop",     "a near-bald cut",       "an ultra-short style"],
        22:  ["a very short cut",      "a closely cropped style", "a minimal crop"],
        29:  ["twin pigtails",         "dual ponytails",        "symmetrical pigtails"],
        42:  ["long curly hair",       "long curly locks",      "lengthy ringlets"],
        43:  ["long wavy hair",        "long flowing waves",    "lengthy waves"],
        47:  ["long straight hair",    "long flowing hair",     "lengthy straight hair"],
        55:  ["a sleek bun updo",      "hair gathered in a bun","a neat top bun"],
        72:  ["long straight hair",    "long flowing locks",    "lengthy straight hair"],
        76:  ["long straight hair",    "sleek flowing hair",    "long straight hair"],
        77:  ["long wavy hair",        "long flowing waves",    "flowing lengthy hair"],
        90:  ["long curly hair",       "long flowing curls",    "lengthy ringlets"],
        91:  ["long wavy hair",        "long flowing waves",    "lengthy waves"],
        92:  ["long curly hair",       "long curly locks",      "lengthy curls"],
        93:  ["long wavy hair",        "flowing long waves",    "lengthy wavy locks"],
        96:  ["twin pigtails",         "symmetrical ponytails", "dual side pigtails"],
        102: ["long curly hair",       "lengthy ringlets",      "long flowing curls"],
        103: ["long curly hair",       "lengthy ringlets",      "long flowing curls"],
        108: ["twin ponytails",        "symmetrical pigtails",  "dual twin tails"],
        109: ["a short crop",          "a short cut",           "a short cropped style"],
        110: ["a short cut",           "a closely cropped style", "a minimal crop"],
    }

    _short_curly   = {9, 10, 25, 26, 27, 28, 32, 34, 85, 97}
    _short_wavy    = {12, 14, 30, 31, 37, 48, 51, 56, 82, 94}
    _short_straight= {7, 8, 11, 16, 17, 19, 23, 38, 39, 49, 50, 52, 53, 57, 73, 74, 81, 89}
    _med_curly     = {18, 33, 41, 54, 58, 86, 87, 99, 100, 104, 105, 106, 107}
    _med_wavy      = {24, 36, 40, 44, 45, 46, 59, 75, 83, 84, 95, 98}
    _med_straight  = {60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71}
    _long_wavy     = {35, 78, 79, 80, 88}
    _long_straight = {101}

    def _get_hair_geometry(val):
        val = int(val)
        if val in _special_hair:
            return random.choice(_special_hair[val])
        if val in _short_curly:
            return random.choice(["short curly hair", "a short curly cut", "cropped curly hair"])
        if val in _short_wavy:
            return random.choice(["short wavy hair", "a short wavy cut", "wavy cropped hair"])
        if val in _short_straight:
            return random.choice(["short straight hair", "a neat short cut", "a short cropped style"])
        if val in _med_curly:
            return random.choice(["medium curly hair", "shoulder-length curls", "mid-length curly hair"])
        if val in _med_wavy:
            return random.choice(["medium wavy hair", "shoulder-length waves", "mid-length wavy hair"])
        if val in _med_straight:
            return random.choice(["medium straight hair", "shoulder-length straight hair", "a medium straight cut"])
        if val in _long_wavy:
            return random.choice(["long wavy hair", "long flowing waves", "lengthy wavy locks"])
        if val in _long_straight:
            return random.choice(["long straight hair", "long flowing straight hair", "sleek long hair"])
        return random.choice(["medium length hair", "shoulder-length hair", "a medium hairstyle"])

    chosen_geometry = _get_hair_geometry(hair_idx)
    chosen_color    = random.choice(hair_colors.get(hair_color_idx, ["colored"]))

    # ── WEAVE COLOR INTO GEOMETRY ATTRIBUTE ──
    # If completely bald (ID 0), color injections are dropped
    if hair_idx == 0:
        chosen_hair = chosen_geometry
    else:
        # Ordered structural anchor words to place the color phrase logically
        structural_nouns = [
            "hair", "locks", "locs", "ringlets", "curls", "waves", "bun", 
            "updo", "afro", "dreadlocks", "pigtails", "ponytails", "tails", 
            "mane", "crop", "cut", "style"
        ]
        injected = False
        for noun in structural_nouns:
            if noun in chosen_geometry:
                chosen_hair = chosen_geometry.replace(noun, f"{chosen_color} {noun}", 1)
                injected = True
                break
        if not injected:
            chosen_hair = f"{chosen_color} {chosen_geometry}"

    # --- FACIAL HAIR GEOMETRY MAP ---
    facial_hair_styles = {
        0: ["a thin beard and mustache setup", "a light beard paired with a mustache", "a subtle mustache and thin beard combo"],
        1: ["facial hair growing only around the ears", "sideburn hair extending down near the ears", "facial hair localized near the ears"],
        2: ["a long beard", "a lengthy, full beard", "a rugged long beard"],
        3: ["a medium-length beard", "a fully grown medium beard", "a classic medium beard"],
        4: ["hair exclusively on the chin", "a chin-only beard patch", "facial hair limited to the chin area"],
        5: ["a standalone mustache", "a distinct mustache with a clean-shaven jaw", "a simple mustache setup"],
        6: ["a mustache and a chin beard patch", "a mustache paired with central chin hair", "a classic mustache and chin stubble combo"],
        7: ["a mustache paired with a goatee on the chin", "a classic mustache and pointed goatee combo", "a goatee style featuring a mustache and chin hair"],
        8: ["a medium beard and mustache combo", "a full medium beard with a mustache", "a classic medium beard and mustache build"],
        9: ["a short, neatly trimmed beard and mustache", "a light short beard with a mustache", "a clean short beard and mustache setup"],
        10: ["a trimmed short beard and mustache", "a short beard accompanied by a mustache", "a neat short beard and mustache variation"],
        11: ["a neatly groomed short beard and mustache", "a short beard with a mustache", "a tailored short beard and mustache layout"],
        12: ["a tidy short beard and mustache", "a compact short beard paired with a mustache", "a short beard and mustache structure"],
        13: ["hair growing only on the chin", "a clean chin beard with no mustache", "facial hair restricted solely to the chin"]
    }
    
    # --- ACCESSORIES MAPS ---
    glasses_types = {
        0: "thick square-framed specs", 1: "classic round-framed glasses", 2: "oval wire-rimmed spectacles",
        3: "semi-rimless rectangular glasses", 4: "retro wayfarer glasses", 5: "sharp cat-eye glasses",
        6: "oversized circular wire spectacles", 7: "classic teardrop aviators", 8: "geometric hexagonal frames",
        9: "sporty round sunglasses", 10: "heart-shaped glasses"
    }
    glasses_colors = {0: "black", 1: "white", 2: "blue", 3: "red", 4: "grey", 5: "yellow", 6: "pink"}

    # --- COMPILE CAPTION COMPONENTS ---
    features_pool = []
    
    features_pool.append(random.choice(face_colors.get(face_color_idx, ["skin"])))
    features_pool.append(random.choice(face_shapes.get(face_shape_idx, ["a face outline"])))
    features_pool.append(random.choice(chin_lengths.get(chin_len_idx, ["a chin"])))
    features_pool.append(random.choice(eye_colors.get(eye_color_idx, ["eyes"])))
    features_pool.append(random.choice(eyebrow_thicknesses.get(eyebrow_idx, ["eyebrows"])))
    features_pool.append(chosen_hair)
    
    # Process corrected facial hair logic (Includes dynamic negative phrasing for ID 14)
    if facial_idx != 14:
        chosen_facial_hair = random.choice(facial_hair_styles.get(facial_idx, ["facial hair"]))
        features_pool.append(f"complemented by {chosen_facial_hair}")
    else:
        no_beard_phrases = [
            "with a clean-shaven face", 
            "showing no facial hair", 
            "where facial hair does not outline the face shape", 
            "with a completely bare jawline", 
            "having no beard or mustache present",
            "with no beard"
        ]
        features_pool.append(random.choice(no_beard_phrases))
        
    if glasses_idx != 11:
        g_color = glasses_colors.get(g_color_idx, "neutral-toned")
        g_style = glasses_types.get(glasses_idx, "eyewear")
        features_pool.append(f"wearing a pair of {g_color} {g_style}")
    else:
        no_glasses_phrases = [
            "with no eyewear present", 
            "not wearing glasses", 
            "with no glasses", 
            "without any glasses", 
            "lacking any eyewear"
        ]
        features_pool.append(random.choice(no_glasses_phrases))

    # Scramble visual feature sequence order to strip positional bias for CLIP
    random.shuffle(features_pool)
    
    opener_phrase = random.choice(openers)
    joined_features = ", ".join(features_pool[:-1]) + f", and {features_pool[-1]}."
    
    return f"{opener_phrase} a person featuring {joined_features}"
