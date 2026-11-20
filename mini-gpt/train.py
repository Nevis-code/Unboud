"""Entraînement du mini-GPT.

Exemple :
    python train.py --data data/exemple.txt --steps 3000

Principe de la boucle, répété des milliers de fois :
    1. prendre des extraits de texte (x) et les mêmes décalés d'un caractère (y)
    2. le modèle prédit y à partir de x
    3. on mesure l'erreur (loss)
    4. on corrige légèrement les poids pour réduire l'erreur
"""

import argparse
import math
import time

import torch

from model import GPTConfig, MiniGPT
from tokenizer import CharTokenizer


def parse_args():
    p = argparse.ArgumentParser(description="Entraîne un mini-GPT sur un fichier texte")
    p.add_argument("--data", default="data/exemple.txt", help="fichier texte d'entraînement")
    p.add_argument("--out", default="mini-gpt.pt", help="fichier du modèle sauvegardé")
    p.add_argument("--ctx", type=int, default=128, help="longueur du contexte")
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--batch", type=int, default=32, help="extraits traités en parallèle")
    p.add_argument("--steps", type=int, default=3000, help="nombre d'étapes d'entraînement")
    p.add_argument("--lr", type=float, default=3e-4, help="learning rate maximal")
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", default=None, help="cuda, cpu... (auto par défaut)")
    return p.parse_args()


def get_batch(data, ctx, batch, device):
    """Tire `batch` extraits au hasard. y = x décalé d'un cran :
    pour chaque position, la réponse attendue est le caractère suivant."""
    starts = torch.randint(len(data) - ctx - 1, (batch,))
    x = torch.stack([data[s : s + ctx] for s in starts])
    y = torch.stack([data[s + 1 : s + ctx + 1] for s in starts])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, splits, args, device, iters=20):
    """Moyenne de la loss sur plusieurs lots, pour train et validation.
    La validation = texte que le modèle n'a JAMAIS vu à l'entraînement :
    c'est elle qui dit s'il apprend vraiment ou s'il récite par cœur."""
    model.eval()
    out = {}
    for name, data in splits.items():
        losses = torch.zeros(iters)
        for i in range(iters):
            x, y = get_batch(data, args.ctx, args.batch, device)
            _, loss = model(x, y)
            losses[i] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def lr_at(step, args, warmup=100):
    """Learning rate : montée progressive, puis descente en cosinus.
    Démarrer doucement évite de casser le modèle au tout début."""
    if step < warmup:
        return args.lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, args.steps - warmup)
    return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Appareil : {device}")

    # --- Données ---------------------------------------------------------
    with open(args.data, encoding="utf-8") as f:
        text = f.read()
    tok = CharTokenizer.from_text(text)
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    if len(data) < 4 * args.ctx:
        raise SystemExit(
            f"Texte trop court ({len(data)} caractères) pour un contexte de {args.ctx}. "
            "Ajoute du texte ou baisse --ctx."
        )
    split = int(0.9 * len(data))  # 90 % entraînement, 10 % validation
    splits = {"train": data[:split], "val": data[split:]}
    print(f"{len(data):,} caractères, {tok.vocab_size} caractères distincts")

    # --- Modèle ----------------------------------------------------------
    cfg = GPTConfig(
        vocab_size=tok.vocab_size, ctx=args.ctx, dim=args.dim,
        heads=args.heads, layers=args.layers, dropout=args.dropout,
    )
    model = MiniGPT(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Modèle : {n_params / 1e6:.2f} M de paramètres")

    # AdamW : l'optimiseur qui ajuste les poids à chaque étape
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # --- Boucle d'entraînement ------------------------------------------
    best_val = float("inf")
    t0 = time.time()
    for step in range(args.steps + 1):
        for group in opt.param_groups:
            group["lr"] = lr_at(step, args)

        # Évaluation régulière + sauvegarde du meilleur modèle
        if step % args.eval_every == 0 or step == args.steps:
            losses = estimate_loss(model, splits, args, device)
            print(f"étape {step:5d} | train {losses['train']:.3f} | val {losses['val']:.3f}"
                  f" | {time.time() - t0:.0f}s")
            if losses["val"] < best_val:
                best_val = losses["val"]
                torch.save(
                    {"model": model.state_dict(), "config": cfg.to_dict(), "chars": tok.chars},
                    args.out,
                )
        if step == args.steps:
            break

        x, y = get_batch(splits["train"], args.ctx, args.batch, device)
        _, loss = model(x, y)           # 1. prédiction + erreur
        opt.zero_grad(set_to_none=True)
        loss.backward()                 # 2. dans quel sens corriger chaque poids ?
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # évite les corrections trop brutales
        opt.step()                      # 3. on corrige

    print(f"Terminé. Meilleur modèle (val {best_val:.3f}) sauvegardé dans {args.out}")
    print("Teste-le avec : python generate.py --prompt \"Bonjour\"")


if __name__ == "__main__":
    main()
