"""Télécharge du texte français pour entraîner Unbound AI.

Exemples :
    pip install datasets
    python prepare_data.py --out data/wikipedia_fr.txt --chars 4000000
    python prepare_data.py --source wikisource --out data/wikisource_fr.txt

Le texte est lu en "streaming" : on ne télécharge que ce qu'il faut pour
atteindre le nombre de caractères demandé, pas toute la base de données.
"""

import argparse

# Jeux de données hébergés sur Hugging Face : (nom, configuration)
SOURCES = {
    # Articles de Wikipédia en français : texte propre et varié
    "wikipedia": ("wikimedia/wikipedia", "20231101.fr"),
    # Textes de Wikisource en français : plus littéraire, mais contenu très mélangé
    "wikisource": ("OpenLLM-France/wikisource", None),
}

# Caractères conservés. Notre modèle travaille caractère par caractère : plus il
# y a de caractères rares (cyrillique, chinois...), plus il gaspille de capacité
# à les apprendre. On garde donc le français et la ponctuation usuelle.
ALLOWED = set(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    "àâäçéèêëîïôöùûüÿœæÀÂÄÇÉÈÊËÎÏÔÖÙÛÜŸŒÆ"
    " \n.,;:!?'\"()[]-–—«»’‘“”…%&/+=*°€$#@_<>"
)
# Les espaces "spéciales" deviennent des espaces normales
SPACES = {ord(" "): " ", ord(" "): " ", ord(" "): " ", ord("\t"): " "}


def clean(text):
    """Ne garde que les caractères autorisés."""
    text = text.translate(SPACES)
    return "".join(c for c in text if c in ALLOWED)


def build_corpus(texts, out_path, max_chars, min_doc_chars=500):
    """Écrit dans `out_path` les textes nettoyés jusqu'à `max_chars` caractères.
    Les documents trop courts (listes, ébauches) sont ignorés.
    Retourne (nombre de documents, nombre de caractères) écrits."""
    n_docs, n_chars = 0, 0
    with open(out_path, "w", encoding="utf-8") as f:
        for text in texts:
            text = clean(text).strip()
            if len(text) < min_doc_chars:
                continue
            f.write(text + "\n\n")
            n_docs += 1
            n_chars += len(text) + 2
            if n_chars >= max_chars:
                break
    return n_docs, n_chars


def main():
    p = argparse.ArgumentParser(description="Télécharge du texte français")
    p.add_argument("--source", choices=SOURCES, default="wikipedia")
    p.add_argument("--out", default="data/wikipedia_fr.txt", help="fichier de sortie")
    p.add_argument("--chars", type=int, default=4_000_000, help="nombre de caractères visé")
    args = p.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("Il manque la bibliothèque 'datasets' : pip install datasets")

    name, config = SOURCES[args.source]
    print(f"Lecture de {name} en streaming...")
    ds = load_dataset(name, config, split="train", streaming=True)
    # Mélange léger pour ne pas lire que les premiers articles dans l'ordre
    ds = ds.shuffle(seed=1, buffer_size=10_000)

    n_docs, n_chars = build_corpus((row["text"] for row in ds), args.out, args.chars)
    print(f"{n_docs} documents, {n_chars:,} caractères écrits dans {args.out}")


if __name__ == "__main__":
    main()
