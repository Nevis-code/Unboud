# Mini-GPT

Un petit modèle de langage de type GPT, écrit en PyTorch, pour comprendre comment fonctionne une IA de génération de texte. Le code est volontairement court et commenté en français.

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
python -m venv venv && source venv/bin/activate

# NVIDIA (CUDA)
pip install torch --index-url https://download.pytorch.org/whl/cu124
# AMD (ROCm)  : pip install torch --index-url https://download.pytorch.org/whl/rocm6.2
# Sans GPU    : pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Les versions CUDA et ROCm évoluent : vérifie la commande exacte sur [pytorch.org](https://pytorch.org).

## Utilisation

```bash
# 1. Entraîner (le meilleur modèle est sauvegardé dans mini-gpt.pt)
python train.py --data data/exemple.txt --steps 3000

# 2. Générer du texte
python generate.py --prompt "Bienvenue" --n 300 --temperature 0.8
```

Options utiles de `train.py` : `--dim`, `--layers`, `--heads`, `--ctx`, `--batch`, `--lr`, `--steps`. Tape `python train.py --help` pour tout voir.

## Lire les résultats

Pendant l'entraînement, deux valeurs s'affichent :

- `train` : l'erreur sur les données vues à l'entraînement.
- `val` : l'erreur sur des données gardées de côté, jamais vues.

| Ce que tu observes | Signification |
|---|---|
| `train` et `val` baissent | Le modèle apprend |
| `train` baisse mais `val` remonte | Surapprentissage : il apprend par cœur. Il faut plus de données ou un modèle plus petit |
| Les deux stagnent | Modèle trop petit, ou learning rate mal réglé |

Avec `data/exemple.txt` (moins de 3 000 caractères), le surapprentissage arrive très vite : c'est normal, ce texte sert seulement à vérifier que tout fonctionne. Pour un résultat intéressant, il faut **plusieurs centaines de milliers de caractères** de texte (des livres libres de droits, un wiki, une FAQ...). Mets ton fichier dans `data/` et lance `python train.py --data data/mon_texte.txt`.

## Pour aller plus loin

- Remplacer le tokenizer par caractères par un tokenizer BPE (`tiktoken`, `tokenizers`).
- Agrandir le modèle (`--dim 384 --layers 6 --heads 6`) quand tu as un GPU et beaucoup de données.
- Fine-tuner un modèle open source existant (Qwen, Mistral, Llama...) avec LoRA pour obtenir un vrai assistant.

## Publication

Les poids (`*.pt`) et les données personnelles sont ignorés par Git (voir `.gitignore`). Pour partager un modèle entraîné, utilise le [Hugging Face Hub](https://huggingface.co). Pense à ajouter un fichier `LICENSE` à la racine du dépôt (MIT ou Apache-2.0 par exemple) et à anonymiser toute donnée provenant de vrais utilisateurs.
