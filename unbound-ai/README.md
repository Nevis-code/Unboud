# Unbound AI

Un petit modèle de langage écrit en PyTorch, pour comprendre comment fonctionne une IA qui génère du texte. Le code est volontairement court et commenté en français.

Le modèle apprend une seule chose : **prédire le caractère suivant**. En répétant cette prédiction, il génère du texte.

## Structure

| Fichier | Rôle |
|---|---|
| `tokenizer.py` | Transforme le texte en nombres (un nombre par caractère) et inversement |
| `model.py` | Le réseau : embedding, blocs transformer (attention + MLP), couche de sortie |
| `train.py` | Boucle d'entraînement, évaluation, sauvegarde du meilleur modèle |
| `generate.py` | Génère du texte à partir d'un modèle entraîné |
| `data/exemple.txt` | Court texte de démonstration |

## Installation (Linux / CachyOS)

```bash
sudo pacman -S python python-pip git
python -m venv venv
source venv/bin/activate        # shell fish : source venv/bin/activate.fish
pip install torch numpy
```

Les paquets PyTorch pour Linux embarquent CUDA : il suffit d'avoir le pilote NVIDIA (`nvidia-smi` doit fonctionner). Pour AMD (ROCm) ou sans GPU, prends la commande adaptée sur [pytorch.org](https://pytorch.org).

## Utilisation

```bash
# 1. Entraîner. Le meilleur modèle est sauvegardé dans le fichier --out.
python train.py --data data/mon_texte.txt --out unbound-ai.pt

# 2. Générer du texte
python generate.py --model unbound-ai.pt --prompt "Bonjour" --n 300 --temperature 0.6
```

`generate.py` utilise par défaut `unbound-ai.pt`. Si tu as sauvegardé ton modèle sous un autre nom avec `--out`, indique le même nom avec `--model`, sinon tu ne génères pas avec le modèle que tu crois.

Options utiles de `train.py` : `--dim`, `--layers`, `--heads`, `--ctx`, `--batch`, `--lr`, `--steps`, `--dropout`. Tape `python train.py --help` pour tout voir.

## Lire les résultats

Pendant l'entraînement, deux valeurs s'affichent :

- `train` : l'erreur sur les données vues à l'entraînement.
- `val` : l'erreur sur des données gardées de côté, jamais vues.

| Ce que tu observes | Signification |
|---|---|
| `train` et `val` baissent | Le modèle apprend |
| `train` baisse mais `val` remonte | Surapprentissage : il apprend par cœur. Il faut plus de texte ou un modèle plus petit |
| Les deux stagnent | Modèle trop petit, ou learning rate mal réglé |

Quand `val` remonte, `train.py` affiche un avertissement. Le fichier sauvegardé est toujours celui où `val` était la plus basse.

## Quelle taille de modèle pour quelle quantité de texte

Plus le modèle est gros, plus il lui faut de texte, sinon il récite par cœur.

| Quantité de texte | Réglages conseillés |
|---|---|
| Moins de 300 000 caractères (une pièce de théâtre) | `--dim 256 --layers 4 --heads 4 --dropout 0.2` |
| 1 à 5 millions de caractères (plusieurs romans) | `--dim 384 --layers 6 --heads 6 --ctx 256` |

Exemple mesuré sur une pièce de théâtre d'environ 185 000 caractères : un modèle de 10,8 millions de paramètres atteint son meilleur score de validation (2,53) dès l'étape 500, puis surapprend ; un modèle de 3,2 millions de paramètres avec `--dropout 0.2` arrive à 1,37.

## Pour aller plus loin

- Ajouter beaucoup plus de texte (livres libres de droits sur [Project Gutenberg](https://www.gutenberg.org), documentation, FAQ...).
- Remplacer le tokenizer par caractères par un tokenizer BPE (`tiktoken`, `tokenizers`).
- Fine-tuner un modèle open source existant (Qwen, Mistral, Llama...) avec LoRA pour obtenir un vrai assistant.

## Publication

Les poids (`*.pt`) et les données (`data/`, sauf l'exemple) sont ignorés par Git, voir `.gitignore`. Pour partager un modèle entraîné, utilise le [Hugging Face Hub](https://huggingface.co). Ajoute un fichier `LICENSE` à la racine du dépôt (MIT ou Apache-2.0 par exemple), vérifie les droits des textes utilisés pour l'entraînement et anonymise toute donnée venant de vrais utilisateurs.
