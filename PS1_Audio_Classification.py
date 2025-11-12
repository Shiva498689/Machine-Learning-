# ====================================================
# Audio Classification with ResNet152 (Pretrained)
# ====================================================

import os
import random
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from tqdm import tqdm

# ====================================================
# CONFIG
# ====================================================
SEED = 42
SR = 22050
MAX_LEN_SEC = 5.0
N_MELS = 128
BATCH_SIZE = 16      # ResNet152 is heavier - reducing batch size helps
EPOCHS = 40
LEARNING_RATE = 1e-4
PATIENCE = 6

# ====================================================
# SEED FIX
# ====================================================
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed()

# ====================================================
# AUDIO UTILS
# ====================================================
def load_and_fix_length(file_path, sr=SR, max_len_sec=MAX_LEN_SEC):
    """Loads audio, trims silence, and pads/truncates to a fixed length."""
    audio, sr = librosa.load(file_path, sr=sr)
    audio, _ = librosa.effects.trim(audio)
    max_samples = int(max_len_sec * sr)
    if len(audio) < max_samples:
        # Pad with zeros
        audio = np.pad(audio, (0, max_samples - len(audio)))
    else:
        # Truncate
        audio = audio[:max_samples]
    return audio.astype(np.float32), sr

def extract_log_mel(audio, sr=SR, n_mels=N_MELS):
    """Extracts a normalized log-Mel spectrogram."""
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    # Normalize to [0, 1] range
    log_mel = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-6)
    return log_mel.astype(np.float32)

# ====================================================
# DATASET
# ====================================================
class AudioDataset(Dataset):
    def __init__(self, files, labels=None):
        self.files = files
        self.labels = labels

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fpath = self.files[idx]
        audio, sr = load_and_fix_length(fpath)
        mel = extract_log_mel(audio, sr)
        
        # Stack the 1-channel mel spectrogram to 3 channels (R, G, B) to fit ImageNet pretrained models
        mel = np.stack([mel, mel, mel], axis=0)
        
        mel_tensor = torch.tensor(mel, dtype=torch.float32)
        
        if self.labels is None:
            # For test set
            return mel_tensor, os.path.basename(fpath)
        else:
            # For train/validation sets
            return mel_tensor, int(self.labels[idx])

# ====================================================
# PATHS
# ====================================================
# NOTE: Update these paths to your specific environment if not running in a Kaggle notebook
train_path = "/kaggle/input/the-frequency-quest/the-frequency-quest - Copy/train/train"
test_path = "/kaggle/input/the-frequency-quest/the-frequency-quest - Copy/test/test"

# ====================================================
# FILES & LABELS
# ====================================================
def list_wavs_in_train(train_root):
    """Lists all .wav files and their corresponding class labels in the train directory."""
    files, labels = [], []
    for label_name in sorted(os.listdir(train_root)):
        folder = os.path.join(train_root, label_name)
        if not os.path.isdir(folder): continue
        for file in os.listdir(folder):
            if file.lower().endswith(".wav"):
                files.append(os.path.join(folder, file))
                labels.append(label_name)
    return files, labels

def list_wavs_in_test(test_root):
    """Lists all .wav files in the test directory."""
    files = []
    for file in sorted(os.listdir(test_root)):
        if file.lower().endswith(".wav"):
            files.append(os.path.join(test_root, file))
    return files

train_files, train_labels_str = list_wavs_in_train(train_path)
test_files = list_wavs_in_test(test_path)
print(f"Found {len(train_files)} train files, {len(test_files)} test files.")

encoder = LabelEncoder()
train_labels = encoder.fit_transform(train_labels_str)
num_classes = len(encoder.classes_)
print(f"Detected {num_classes} classes.")

# ====================================================
# SPLIT
# ====================================================
X_train, X_val, y_train, y_val = train_test_split(
    train_files, train_labels, test_size=0.2, random_state=SEED, stratify=train_labels
)

train_ds = AudioDataset(X_train, y_train)
val_ds = AudioDataset(X_val, y_val)
test_ds = AudioDataset(test_files, labels=None)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)


# ====================================================
# MODEL: ResNet152 (Pretrained)
# ====================================================

# Load pretrained ResNet152
model = models.resnet152(weights='IMAGENET1K_V1')

# Replace final classification layer (model.fc) for your number of classes
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, num_classes)

model = model.to(device)

# Criterion, Optimizer, Scheduler
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3)

print("✅ Using ResNet152 as backbone.")


# ====================================================
# TRAIN LOOP
# ====================================================
best_val_acc = 0.0
patience_counter = 0

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0

    # Training phase
    for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} (Train)"):
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * xb.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == yb).sum().item()
        total_samples += xb.size(0)

    train_loss = total_loss / total_samples
    train_acc = total_correct / total_samples

    # Validation phase
    model.eval()
    val_correct, val_samples, val_loss = 0, 0, 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            val_loss += loss.item() * xb.size(0)
            preds = logits.argmax(dim=1)
            val_correct += (preds == yb).sum().item()
            val_samples += xb.size(0)

    val_loss /= val_samples
    val_acc = val_correct / val_samples
    scheduler.step(val_acc)

    print(f"Epoch {epoch:02d} | Train Loss/Acc: {train_loss:.4f}/{train_acc:.4f} | "
          f"Val Loss/Acc: {val_loss:.4f}/{val_acc:.4f}")

    # Checkpoint and Early Stopping
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        # Save the best model weights
        torch.save(model.state_dict(), "resnet_152_model.pth")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("⏹ Early stopping triggered.")
            break

print(f"✅ Training complete. Best Val Acc: {best_val_acc:.4f}")

# ====================================================
# TEST PREDICTIONS
# ====================================================
model.load_state_dict(torch.load("resnet_152_model.pth"))
model.eval()

test_preds, test_files_out = [], []
with torch.no_grad():
    for xb, fnames in tqdm(test_loader, desc="Predicting Test Set"):
        xb = xb.to(device)
        logits = model(xb)
        preds = logits.argmax(dim=1).cpu().numpy()
        test_preds.extend(preds.tolist())
        test_files_out.extend(fnames)

# Inverse transform the numeric predictions back to class labels
test_labels = encoder.inverse_transform(np.array(test_preds))
test_results = pd.DataFrame({
    "ID": test_files_out,
    "Class": test_labels
})
test_results.to_csv("test_predictions_resnet.csv", index=False)
print("✅ Test predictions saved to test_predictions_resnet.csv")