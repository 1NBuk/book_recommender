
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from src.models.base import BaseRecommender


class InteractionDataset(Dataset):
    def __init__(self, user_idx, item_idx, labels):
        self.users = user_idx
        self.items = item_idx
        self.labels = labels

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.users[idx], dtype=torch.long),
            torch.tensor(self.items[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.float32),
        )


class TwoTowerNetwork(nn.Module):
    def __init__(self, num_users, num_items, text_embeddings, embedding_dim=128, hidden_dim=256):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        text_dim = text_embeddings.shape[1]
        self.text_projection = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

        self.user_tower = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

        self.item_tower = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

        self.register_buffer(
            "text_embeddings",
            torch.tensor(text_embeddings, dtype=torch.float32),
        )

    def forward(self, user_idx, item_idx):
        user_emb = self.user_embedding(user_idx)
        user_vec = self.user_tower(user_emb)

        item_id_emb = self.item_embedding(item_idx)
        item_text_emb = self.text_projection(self.text_embeddings[item_idx])
        item_input = torch.cat([item_id_emb, item_text_emb], dim=1)
        item_vec = self.item_tower(item_input)

        user_vec = nn.functional.normalize(user_vec, dim=1)
        item_vec = nn.functional.normalize(item_vec, dim=1)
        return (user_vec * item_vec).sum(dim=1)

    def encode_user(self, user_idx):
        user_emb = self.user_embedding(user_idx)
        user_vec = self.user_tower(user_emb)
        return nn.functional.normalize(user_vec, dim=1)

    def encode_items(self, item_idx):
        item_id_emb = self.item_embedding(item_idx)
        item_text_emb = self.text_projection(self.text_embeddings[item_idx])
        item_input = torch.cat([item_id_emb, item_text_emb], dim=1)
        item_vec = self.item_tower(item_input)
        return nn.functional.normalize(item_vec, dim=1)


class TwoTowerRecommender(BaseRecommender):
    def __init__(self, config):
        super().__init__(config, name="TwoTower")
        self.model = None
        self.user_to_idx = {}
        self.item_to_idx = {}
        self.idx_to_item = {}
        self.device = None

    def _select_device(self):
        requested = self.config.get("runtime.device", "auto")
        if requested == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(requested)

    def fit(self, interactions, items=None, embeddings=None):
        if embeddings is None:
            raise ValueError("TwoTower requires embeddings")

        self.device = self._select_device()

        user_ids = interactions["user_id"].unique()
        item_ids = interactions["item_id"].unique()
        self.user_to_idx = {u: i for i, u in enumerate(user_ids)}
        self.item_to_idx = {it: i for i, it in enumerate(item_ids)}
        self.idx_to_item = {i: it for it, i in self.item_to_idx.items()}

        item_id_to_row = {item_id: i for i, item_id in enumerate(items["item_id"].values)}
        item_text_embs = np.zeros((len(item_ids), embeddings.shape[1]), dtype=np.float32)
        for item_id, idx in self.item_to_idx.items():
            if item_id in item_id_to_row:
                item_text_embs[idx] = embeddings[item_id_to_row[item_id]]

        user_idx = interactions["user_id"].map(self.user_to_idx).values
        item_idx = interactions["item_id"].map(self.item_to_idx).values
        labels = interactions["is_explicit"].values.astype(np.float32)

        dataset = InteractionDataset(user_idx, item_idx, labels)
        loader = DataLoader(
            dataset,
            batch_size=self.config.get("models.two_tower.batch_size", 512),
            shuffle=True,
        )

        self.model = TwoTowerNetwork(
            num_users=len(self.user_to_idx),
            num_items=len(self.item_to_idx),
            text_embeddings=item_text_embs,
            embedding_dim=self.config.get("models.two_tower.embedding_dim", 128),
            hidden_dim=self.config.get("models.two_tower.hidden_dim", 256),
        ).to(self.device)

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.get("models.two_tower.learning_rate", 0.001),
            weight_decay=1e-5,
        )
        criterion = nn.BCEWithLogitsLoss()

        epochs = self.config.get("models.two_tower.epochs", 5)
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for users, items_b, labels_b in loader:
                users = users.to(self.device)
                items_b = items_b.to(self.device)
                labels_b = labels_b.to(self.device)

                optimizer.zero_grad()
                scores = self.model(users, items_b)
                loss = criterion(scores, labels_b)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / max(len(loader), 1)
            print(f"TwoTower epoch {epoch + 1}/{epochs} - loss: {avg_loss:.4f}")

        self.seen_by_user = interactions.groupby("user_id")["item_id"].apply(set).to_dict()
        self.is_fitted = True
        return self

    def recommend(self, user_id, n=10, exclude_seen=True):
        if user_id not in self.user_to_idx:
            return pd.DataFrame(columns=["user_id", "item_id", "score"])

        self.model.eval()
        user_idx = torch.tensor([self.user_to_idx[user_id]], dtype=torch.long, device=self.device)
        all_items = torch.arange(len(self.item_to_idx), dtype=torch.long, device=self.device)

        with torch.no_grad():
            user_vec = self.model.encode_user(user_idx)
            item_vecs = self.model.encode_items(all_items)
            scores = (user_vec @ item_vecs.T).squeeze(0)
            scores_np = scores.cpu().numpy()

        if exclude_seen and user_id in self.seen_by_user:
            seen = self.seen_by_user[user_id]
            for item_id in seen:
                if item_id in self.item_to_idx:
                    scores_np[self.item_to_idx[item_id]] = -np.inf

        top_idx = np.argsort(scores_np)[::-1][:n]
        result = pd.DataFrame({
            "user_id": user_id,
            "item_id": [self.idx_to_item[i] for i in top_idx],
            "score": scores_np[top_idx],
        })
        return result
