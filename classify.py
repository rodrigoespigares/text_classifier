import torch
from model import TextClassifier
import torch.nn.functional as F

def classify_text(user_id, text):
    checkpoint = torch.load(f'models/model_user_{user_id}.pt')
    model = TextClassifier(len(checkpoint['vocab']), embedding_dim=50, num_classes=len(checkpoint['categories']))
    model.load_state_dict(checkpoint['model'])
    model.eval()

    unk_idx = checkpoint['vocab'].get('<UNK>', 1)
    tokens = [checkpoint['vocab'].get(word, unk_idx) for word in text.split()]
    if len(tokens) == 0:
        return "desconocido"
    input_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)  # batch=1

    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)
        predicted_index = torch.argmax(probs).item()
        return checkpoint['categories'][predicted_index]