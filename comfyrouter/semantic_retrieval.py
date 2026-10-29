"""Flat multilingual dense retrieval over a frozen physical workflow library.

No task-family or leaf classifier is called. Only the media I/O contract may
filter candidates; a common business decomposition can retrieve several Bases.
"""
from __future__ import annotations
from pathlib import Path

class E5Encoder:
    """Local, pinned multilingual-e5 with masked mean pooling on CPU."""

    def __init__(self, model_dir: Path):
        import torch
        from transformers import AutoModel, AutoTokenizer
        torch.set_num_threads(4)
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
        self.model = AutoModel.from_pretrained(str(model_dir), local_files_only=True).eval()

    def encode(self, texts: list[str], *, query: bool=False):
        import numpy as np
        outputs = []
        for start in range(0, len(texts), 16):
            batch = self.tokenizer([('query: ' if query else 'passage: ') + s for s in texts[start:start + 16]], padding=True, truncation=True, max_length=512, return_tensors='pt')
            with self.torch.inference_mode():
                hidden = self.model(**batch).last_hidden_state
                mask = batch['attention_mask'].unsqueeze(-1).bool()
                vector = hidden.masked_fill(~mask, 0).sum(1) / mask.sum(1)
                vector = self.torch.nn.functional.normalize(vector, p=2, dim=1)
            outputs.append(vector.numpy())
        return np.concatenate(outputs)

class SemanticWorkflowIndex:

    def __init__(self, documents: list[dict], vectors, encoder):
        import numpy as np
        self.documents = documents
        self.vectors = np.asarray(vectors, dtype=np.float32)
        self.encoder = encoder
        if len(documents) != len(vectors) or not np.isfinite(self.vectors).all():
            raise ValueError('invalid workflow vectors')
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError('zero workflow vector')
        self.vectors = self.vectors / norms

    def search(self, description: str, modality: str, top_k: int=5) -> list[dict]:
        import numpy as np
        indices = [i for i, row in enumerate(self.documents) if row['modality'] == modality]
        if not indices:
            raise ValueError(f'no workflow for media contract {modality}')
        query = self.encoder.encode([description], query=True)[0]
        query = query / np.linalg.norm(query)
        scores = self.vectors[indices] @ query
        ranked = sorted(zip(indices, scores), key=lambda item: (-float(item[1]), self.documents[item[0]]['workflow_id']))[:top_k]
        return [{**self.documents[i], 'similarity': float(score)} for i, score in ranked]
