"""Tokenizer au niveau du caractère.

Un tokenizer transforme du texte en nombres (que le réseau comprend) et
inversement. Ici, chaque caractère distinct du texte d'entraînement reçoit un
numéro. C'est le plus simple possible : idéal pour apprendre. Les vrais LLM
utilisent des tokenizers plus malins (BPE) qui découpent en morceaux de mots.
"""


class CharTokenizer:
    def __init__(self, chars):
        # `chars` est la liste triée de tous les caractères connus.
        self.chars = list(chars)
        self.stoi = {c: i for i, c in enumerate(self.chars)}  # caractère -> numéro
        self.itos = {i: c for i, c in enumerate(self.chars)}  # numéro -> caractère

    @classmethod
    def from_text(cls, text):
        """Construit le vocabulaire à partir d'un texte."""
        return cls(sorted(set(text)))

    @property
    def vocab_size(self):
        return len(self.chars)

    def encode(self, text):
        """Texte -> liste de numéros. Les caractères inconnus sont ignorés."""
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, ids):
        """Liste de numéros -> texte."""
        return "".join(self.itos[i] for i in ids)
