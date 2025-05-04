import torch
from torch.utils.data import DataLoader, Dataset
from model import TextClassifier
import torch.nn.functional as F
import torch.optim as optim

class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab):
        self.texts = [
            torch.tensor(
                [vocab[word] for word in t.split() if word in vocab],
                dtype=torch.long
            ) if any(word in vocab for word in t.split())
            else torch.tensor([vocab['<PAD>']])
            for t in texts
        ]
        self.labels = torch.tensor(labels)
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

def pad_collate(batch):
    texts, labels = zip(*batch)
    padded = torch.nn.utils.rnn.pad_sequence(texts, batch_first=True)
    return padded, torch.tensor(labels)

def build_vocab(texts):
    vocab = {'<PAD>': 0, '<UNK>': 1}
    idx = 2
    for text in texts:
        for word in text.split():
            if word not in vocab:
                vocab[word] = idx
                idx += 1
    return vocab

def train_model(user_id, texts, labels, categories):
    vocab = build_vocab(texts)
    dataset = TextDataset(texts, labels, vocab)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=pad_collate)

    model = TextClassifier(len(vocab), embedding_dim=50, num_classes=len(categories))
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        for batch_x, batch_y in dataloader:
            output = model(batch_x)
            loss = F.cross_entropy(output, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    torch.save({
        'model': model.state_dict(),
        'vocab': vocab,
        'categories': categories
    }, f'models/model_user_{user_id}.pt')

    print(f'Modelo entrenado para el usuario {user_id}')