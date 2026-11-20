"""Le modèle : un petit GPT (architecture "transformer décodeur").

Chemin d'un texte dans le modèle :
    numéros de tokens
      -> embedding (token + position)
      -> N blocs [attention -> MLP]
      -> couche de sortie : un score pour chaque token possible

Le modèle n'apprend qu'une chose : prédire le token suivant.
"""

from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int          # nombre de tokens différents
    ctx: int = 128           # longueur max du contexte (combien de tokens le modèle "voit")
    dim: int = 128           # taille des vecteurs (largeur du modèle)
    heads: int = 4           # nombre de têtes d'attention (dim doit être divisible par heads)
    layers: int = 4          # nombre de blocs transformer empilés
    dropout: float = 0.1     # "oubli" aléatoire pendant l'entraînement, limite le par-cœur

    def to_dict(self):
        return asdict(self)


class CausalSelfAttention(nn.Module):
    """Attention : chaque token regarde les tokens PRÉCÉDENTS pour décider
    lesquels sont importants. "Causale" = interdiction de regarder le futur,
    sinon le modèle tricherait en lisant la réponse."""

    def __init__(self, cfg):
        super().__init__()
        assert cfg.dim % cfg.heads == 0, "dim doit être divisible par heads"
        self.heads = cfg.heads
        self.qkv = nn.Linear(cfg.dim, 3 * cfg.dim)   # calcule Q, K, V en une fois
        self.proj = nn.Linear(cfg.dim, cfg.dim)      # recombine les têtes
        self.dropout = cfg.dropout

    def forward(self, x):
        B, T, C = x.shape  # batch, longueur, dimension
        # Q (ce que je cherche), K (ce que j'offre), V (ce que je transmets)
        q, k, v = self.qkv(x).split(C, dim=2)
        # On sépare en plusieurs têtes : chacune peut s'intéresser à autre chose
        q = q.view(B, T, self.heads, C // self.heads).transpose(1, 2)
        k = k.view(B, T, self.heads, C // self.heads).transpose(1, 2)
        v = v.view(B, T, self.heads, C // self.heads).transpose(1, 2)
        # is_causal=True applique le masque "pas de regard vers le futur"
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # on recolle les têtes
        return self.proj(y)


class MLP(nn.Module):
    """Petit réseau appliqué à chaque token séparément pour transformer
    l'information récupérée par l'attention."""

    def __init__(self, cfg):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.dim, 4 * cfg.dim),
            nn.GELU(),
            nn.Linear(4 * cfg.dim, cfg.dim),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Un bloc transformer = attention puis MLP.
    Les `x = x + ...` sont des connexions résiduelles : on AJOUTE le résultat
    à l'entrée au lieu de la remplacer, ce qui facilite l'apprentissage.
    Les LayerNorm stabilisent les valeurs numériques."""

    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.dim)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.dim)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.dim)  # numéro de token -> vecteur
        self.pos_emb = nn.Embedding(cfg.ctx, cfg.dim)         # position -> vecteur
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.Sequential(*[Block(cfg) for _ in range(cfg.layers)])
        self.ln_f = nn.LayerNorm(cfg.dim)
        self.head = nn.Linear(cfg.dim, cfg.vocab_size)        # vecteur -> score par token
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        # Petits poids de départ aléatoires : point de départ standard des GPT
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        """idx : (batch, longueur) de numéros de tokens.
        Si `targets` est donné, on calcule aussi la loss (l'erreur du modèle)."""
        B, T = idx.shape
        assert T <= self.cfg.ctx, "séquence plus longue que le contexte"
        positions = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(positions))
        x = self.blocks(x)
        logits = self.head(self.ln_f(x))  # (batch, longueur, vocab)

        loss = None
        if targets is not None:
            # cross-entropy : pénalise le modèle quand il donne une faible
            # probabilité au vrai token suivant
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Génère du texte token par token.
        temperature : <1 = prudent et répétitif, >1 = créatif et risqué.
        top_k : ne tire qu'au sein des k tokens les plus probables."""
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.ctx:]            # on garde les derniers tokens
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)  # dernier token seulement
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)            # scores -> probabilités
            next_id = torch.multinomial(probs, num_samples=1)  # tirage au sort pondéré
            idx = torch.cat([idx, next_id], dim=1)       # on l'ajoute et on recommence
        return idx
