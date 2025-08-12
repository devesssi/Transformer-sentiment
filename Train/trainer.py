import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.sentiment_model import TransformerSentimentClassifier
from datasets import load_dataset
from utils import tokenize_batch
 
#` Initialize model parameters`
EPOCHS = 10
BATCH_SIZE = 32
LR = 1e-4
D_MODEL = 128
D_FF = 512
HEADS = 8
LAYERS = 4
DROPOUT = 0.1
NUM_CLASSES = 2 

dataset = load_dataset('imdb')

from collections import Counter
import re

def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower()) #\w+ for all the words numbers and characters, \b for word boundaries

all_tokens =[]
for example in dataset["train"]:
    all_tokens.extend(tokenize(example["text"]))
    
# vocab = {word: i+2 for i, (word , _) in enumerate(Counter(all_tokens).most_common(20000))}


vocab = {  word:i+2 for i,(word, _) in enumerate (Counter(all_tokens).most_common(20000))}
vocab['<pad>'] = 0
vocab['<unk>'] = 1

def encode(text, max_length=200):
    tokens = tokenize(text)
    ids = [vocab.get(t, 1) for t in tokens]
    ids += [0]*(max_length - len(ids))
    return ids

class IMDBDataset(torch.utils.data.Dataset):
    def __init__(self, split):
        self.data = dataset[split]
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        text = self.data[idx]["text"]
        label = int(self.data[idx]["label"])
        return torch.tensor(encode(text)), torch.tensor(label)

train_loader = DataLoader(IMDBDataset("train"), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(IMDBDataset("test"), batch_size=BATCH_SIZE)

# 4. Model
model = TransformerSentimentClassifier(D_MODEL, D_FF, len(vocab), LAYERS, HEADS, DROPOUT, NUM_CLASSES)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 5. Loss & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# 6. Training loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f}")

    # Validation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    print(f"Validation Accuracy: {correct/total:.4f}")

# 7. Save model
torch.save(model.state_dict(), "sentiment_transformer.pth")