from pathlib import Path
import json
import re
from paddleocr import PaddleOCR

SAMPLES_DIR = Path("samples")
OUTPUT_DIR = Path("result")

# -----------------------------
# 1) Nettoyage des lignes OCR
# -----------------------------
def clean_line(text: str) -> str:
    if not text:
        return ""

    # espaces propres
    text = re.sub(r"\s+", " ", text).strip()

    # corriger quelques collages fréquents
    text = text.replace("attestationest", "attestation est")
    text = text.replace("touteauthentification", "toute authentification")

    return text


def is_useful_line(text: str) -> bool:
    if not text:
        return False

    # ignorer lignes trop courtes
    if len(text.strip()) < 3:
        return False

    # ignorer bruit très probable
    # ex: hv / cY / ご / Ry / oiy / === etc.
    if re.fullmatch(r"[^A-Za-zÀ-ÿ0-9]+", text):
        return False

    # trop peu de lettres/chiffres utiles
    useful_chars = re.findall(r"[A-Za-zÀ-ÿ0-9]", text)
    if len(useful_chars) < 3:
        return False

    return True


# -----------------------------
# 2) Extraire texte + score
# -----------------------------
def extract_lines(result) -> list[dict]:
    lines = []

    for page in result:
        rec_texts = page.get("rec_texts", [])
        rec_scores = page.get("rec_scores", [])

        for text, score in zip(rec_texts, rec_scores):
            text = clean_line(str(text))
            lines.append({
                "text": text,
                "score": float(score)
            })

    return lines


# -----------------------------
# 3) Filtrer les lignes utiles
# -----------------------------
def filter_lines(lines: list[dict], min_score: float = 0.60) -> list[dict]:
    filtered = []

    for item in lines:
        text = item["text"]
        score = item["score"]

        if score < min_score:
            continue

        if not is_useful_line(text):
            continue

        filtered.append(item)

    return filtered


# -----------------------------
# 4) Extraire quelques champs
# -----------------------------
def extract_fields(clean_text: str) -> dict:
    fields = {
        "nom": None,
        "prenom": None,
        "date_naissance": None,
        "adresse": None,
        "numero_affiliation": None,
        "qualite": None,
        "organisme_declarant": None,
        "date_document": None,
    }

    patterns = {
        "nom": r"Nom\s*[:\-]?\s*(.+)",
        "prenom": r"Pr[eé]nom\s*[:\-]?\s*(.+)",
        "date_naissance": r"Date et lieu de Naissance\s*[:\-]?\s*(.+)",
        "adresse": r"Adresse\s*[:\-]?\s*(.+)",
        "numero_affiliation": r"Sous le num[eé]ro\s*[:\-]?\s*(.+)",
        "qualite": r"Qualit[eé]\s*[:\-]?\s*(.+)",
        "organisme_declarant": r"Organisme D[eé]clarant\s*[:\-]?\s*(.+)",
        "date_document": r"Fait le\s*[:\-]?\s*(.+)",
    }

    lines = clean_text.splitlines()

    for line in lines:
        for field, pattern in patterns.items():
            match = re.search(pattern, line, re.IGNORECASE)
            if match and not fields[field]:
                fields[field] = match.group(1).strip()

    return fields


# -----------------------------
# 5) Programme principal
# -----------------------------
def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    image_path = SAMPLES_DIR / "test2.jpg"

    if not image_path.exists():
        print("Image introuvable.")
        print("Place une image dans : samples/test2.jpg")
        return

    print("Chargement du modèle OCR...")

    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False
    )

    print("Lancement OCR...")
    result = ocr.predict(str(image_path))

    # Extraction brute
    all_lines = extract_lines(result)

    # Sauvegarde brute
    raw_file = OUTPUT_DIR / "resultat_brut2.txt"
    with raw_file.open("w", encoding="utf-8") as f:
        for item in all_lines:
            f.write(f"{item['score']:.4f} | {item['text']}\n")

    # Filtrage
    filtered_lines = filter_lines(all_lines, min_score=0.60)

    # Construire texte propre
    clean_text = "\n".join(item["text"] for item in filtered_lines)

    clean_file = OUTPUT_DIR / "resultat_propre2.txt"
    clean_file.write_text(clean_text, encoding="utf-8")

    # Extraction de champs
    fields = extract_fields(clean_text)

    json_file = OUTPUT_DIR / "champs_extraits2.json"
    json_file.write_text(json.dumps(fields, indent=4, ensure_ascii=False), encoding="utf-8")

    print("\nOCR terminé avec succès.")
    print(f"Fichier brut       : {raw_file}")
    print(f"Fichier nettoyé    : {clean_file}")
    print(f"Champs extraits    : {json_file}")

    print("\n--- Aperçu des champs extraits ---")
    for key, value in fields.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()