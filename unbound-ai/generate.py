"""Génération de texte avec un modèle entraîné.

Exemple :
    python generate.py --prompt "Bienvenue" --n 300 --temperature 0.8
"""

import argparse
import os

import torch

from model import UnboundConfig, UnboundAI
from tokenizer import CharTokenizer


def main():
    p = argparse.ArgumentParser(description="Génère du texte avec Unbound AI")
    p.add_argument("--model", default="unbound-ai.pt", help="fichier créé par train.py")
    p.add_argument("--prompt", default="\n", help="début du texte à compléter")
    p.add_argument("--n", type=int, default=300, help="nombre de caractères à générer")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Message clair si le fichier n'existe pas (au lieu d'une longue erreur)
    if not os.path.exists(args.model):
        raise SystemExit(
            f"Modèle introuvable : {args.model}\n"
            "Entraîne-le d'abord avec train.py, ou indique le bon fichier avec --model."
        )

    # Le fichier contient tout le nécessaire : poids, configuration, vocabulaire
    ckpt = torch.load(args.model, map_location=device, weights_only=True)
    tok = CharTokenizer(ckpt["chars"])
    model = UnboundAI(UnboundConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model"])

    ids = tok.encode(args.prompt)
    if not ids:  # prompt vide ou caractères inconnus : on part du premier caractère
        ids = [0]
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    out = model.generate(idx, args.n, temperature=args.temperature, top_k=args.top_k)
    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
