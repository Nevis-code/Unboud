# Unbound AI

Une petite IA de génération de texte, écrite from scratch en PyTorch : un modèle de type transformer qui apprend à **prédire le caractère suivant**, puis écrit du texte en répétant cette prédiction.

Unbound AI complète le texte qu'on lui donne ; ce n'est pas un assistant qui répond à des questions.

## Démarrage rapide

```bash
python -m venv venv && source venv/bin/activate   # shell fish : activate.fish
pip install torch numpy

# 1. Récupérer du texte français (Wikipédia) et entraîner
pip install datasets
python unbound-ai/prepare_data.py --out unbound-ai/data/wikipedia_fr.txt
python unbound-ai/train.py --data unbound-ai/data/wikipedia_fr.txt --out unbound-ai/unbound-ai.pt \
  --dim 384 --layers 6 --heads 6 --ctx 256 --batch 64 --steps 10000

# 2. Poser un prompt
python main.py --prompt "La France est"
python main.py                                    # mode interactif
```

## Structure

| Fichier | Rôle |
|---|---|
| `main.py` | Point d'entrée : charger le modèle et poser des prompts |
| `unbound-ai/model.py` | Le réseau (embedding, blocs attention + MLP, sortie) |
| `unbound-ai/tokenizer.py` | Texte en nombres et inversement |
| `unbound-ai/train.py` | Entraînement |
| `unbound-ai/generate.py` | Génération en ligne de commande |
| `unbound-ai/prepare_data.py` | Téléchargement de texte français |

Le détail de l'entraînement, la lecture des courbes `train` / `val` et les réglages conseillés sont dans [`unbound-ai/README.md`](unbound-ai/README.md).

Les poids du modèle (`*.pt`) et les données d'entraînement ne sont pas versionnés dans Git.
