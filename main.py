"""Unbound AI : point d'entrée. C'est ici qu'on pose de vrais prompts.

Utilisation :
    python main.py                          # mode interactif
    python main.py --prompt "La France est" # une seule génération
    python main.py --model unbound-ai/unbound-wiki.pt --temperature 0.5

Important : Unbound AI est un modèle qui COMPLÈTE du texte, il ne répond pas
à des questions comme un assistant. Écris le début d'une phrase ou d'un
article, et il écrit la suite.
"""

import argparse
import os
import sys

# Le moteur (modèle, tokenizer, entraînement) vit dans le dossier unbound-ai/
ENGINE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "unbound-ai")
sys.path.insert(0, ENGINE)

import torch

from model import UnboundAI, UnboundConfig
from tokenizer import CharTokenizer


def load_model(path, device):
    """Charge un modèle entraîné par train.py : poids, configuration et vocabulaire."""
    if not os.path.exists(path):
        raise SystemExit(
            f"Modèle introuvable : {path}\n"
            "Entraîne-le avec train.py (voir unbound-ai/README.md) "
            "ou indique le bon fichier avec --model."
        )
    ckpt = torch.load(path, map_location=device, weights_only=True)
    tok = CharTokenizer(ckpt["chars"])
    model = UnboundAI(UnboundConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, tok


def complete(model, tok, prompt, device, n=300, temperature=0.6, top_k=40):
    """Écrit la suite du prompt. Les caractères inconnus du modèle sont ignorés."""
    ids = tok.encode(prompt) or [0]  # prompt vide ou inconnu : on part du 1er caractère
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(idx, n, temperature=temperature, top_k=top_k)
    return tok.decode(out[0].tolist())


def main():
    p = argparse.ArgumentParser(description="Pose un prompt à Unbound AI")
    p.add_argument("--model", default=os.path.join(ENGINE, "unbound-ai.pt"),
                   help="fichier modèle créé par train.py")
    p.add_argument("--prompt", default=None, help="un seul prompt, sans mode interactif")
    p.add_argument("--n", type=int, default=300, help="nombre de caractères à générer")
    p.add_argument("--temperature", type=float, default=0.6,
                   help="bas = prudent et répétitif, haut = créatif et risqué")
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, tok = load_model(args.model, device)

    def run(prompt):
        print(complete(model, tok, prompt, device, args.n, args.temperature, args.top_k))

    if args.prompt is not None:
        run(args.prompt)
        return

    print("Hello there ! I am Unbound AI, ask something to begin.")
    print("(I complete your text: start a sentence and I will write the rest. Empty line to quit.)")
    while True:
        try:
            prompt = input("Enter a question : ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not prompt.strip():
            break
        run(prompt)
        print()


if __name__ == "__main__":
    main()
