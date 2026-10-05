import numpy as np


class FakeModel:
    """Embeds text as letter counts, so texts sharing letters get similar vectors. No download needed."""

    def encode(self, texts):
        def vec(t):
            v = np.zeros(26)
            for ch in t.lower():
                if "a" <= ch <= "z":
                    v[ord(ch) - 97] += 1
            return v + 1e-9
        return np.array([vec(t) for t in texts]) if isinstance(texts, list) else vec(texts)
