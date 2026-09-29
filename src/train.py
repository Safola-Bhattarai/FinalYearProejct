import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          get_linear_schedule_with_warmup)
from sklearn.metrics import accuracy_score, f1_score

# ---------- Settings ----------
MODEL_NAME = "distilbert-base-multilingual-cased"   # supports English + Nepali
SAVE_DIR   = "models/distilbert_category"
MAX_LEN    = 96
BATCH      = 16
EPOCHS     = 4
LR         = 3e-5
torch.manual_seed(42)

# ---------- Load data ----------
train = pd.read_csv("dataset/train.csv")
val   = pd.read_csv("dataset/validation.csv")

label_map = (train[["category_label", "category"]].drop_duplicates()
             .sort_values("category_label"))
id2label = {int(i): c for i, c in zip(label_map.category_label, label_map.category)}
label2id = {c: i for i, c in id2label.items()}
print("Classes:", id2label)

# ---------- Tokenize ----------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def make_loader(df, shuffle):
    enc = tokenizer(df["description"].tolist(), padding="max_length",
                    truncation=True, max_length=MAX_LEN, return_tensors="pt")
    labels = torch.tensor(df["category_label"].values)
    ds = TensorDataset(enc["input_ids"], enc["attention_mask"], labels)
    return DataLoader(ds, batch_size=BATCH, shuffle=shuffle)

train_loader = make_loader(train, True)
val_loader   = make_loader(val, False)

# ---------- Model ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=len(id2label), id2label=id2label, label2id=label2id
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)

def evaluate(loader):
    model.eval()
    preds, golds = [], []
    with torch.no_grad():
        for ids, mask, y in loader:
            out = model(input_ids=ids.to(device), attention_mask=mask.to(device))
            preds += out.logits.argmax(dim=-1).cpu().tolist()
            golds += y.tolist()
    return accuracy_score(golds, preds), f1_score(golds, preds, average="macro")

# ---------- Train ----------
best_f1 = 0
for epoch in range(1, EPOCHS + 1):
    model.train()
    running = 0
    for step, (ids, mask, y) in enumerate(train_loader, 1):
        optimizer.zero_grad()
        out = model(input_ids=ids.to(device), attention_mask=mask.to(device),
                    labels=y.to(device))
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        running += out.loss.item()
        if step % 25 == 0:
            print(f"  epoch {epoch} step {step}/{len(train_loader)} loss {running/step:.4f}")

    acc, f1 = evaluate(val_loader)
    print(f"Epoch {epoch} DONE | val accuracy {acc:.4f} | val macro-F1 {f1:.4f}")

    if f1 > best_f1:
        best_f1 = f1
        model.save_pretrained(SAVE_DIR)
        tokenizer.save_pretrained(SAVE_DIR)
        print(f"  Saved best model to {SAVE_DIR}")

print("\nTraining finished. Best val macro-F1:", round(best_f1, 4))