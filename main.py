"""
main.py

Script principale del progetto. Genera il file di label, crea gli
split train/test del dataset, istanzia i DataLoader (batch size 32) e
allena entrambe le reti (TriConvNet e TriConvNet2), stampando i
risultati ottenuti.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from mri_datautils import generate_label
from dataset_utils import train_test_split_dataset
from models import TriConvNet, TriConvNet2
from trainings import train_and_save, check_accuracy


# ----------------------------------------------------------------------
# Configurazione
# ----------------------------------------------------------------------
DATA_DIR = "./data/mri_images"          # cartella con i file .nii
LABELS_CSV = "./data/labels.csv"        # file CSV generato con generate_label
BATCH_SIZE = 32
SPLIT_FACTOR = 0.8
LEARNING_RATE = 0.001
EPOCHS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    # 1. Generazione del file di label (se non già presente)
    generate_label(DATA_DIR, LABELS_CSV)

    # 2. Creazione degli split train/test del dataset
    train_dataset, test_dataset = train_test_split_dataset(
        LABELS_CSV, DATA_DIR, split_factor=SPLIT_FACTOR
    )

    # 3. DataLoader: i dati vengono caricati in batch di 32 per evitare
    #    di dover tenere in memoria l'intero dataset 3D tutto insieme.
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                               shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                              shuffle=False)

    criterion = nn.CrossEntropyLoss()

    # ------------------------------------------------------------------
    # Training rete 1: TriConvNet
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Training TriConvNet")
    print("=" * 60)

    net1 = TriConvNet(num_classes=2)
    optimizer1 = optim.Adam(net1.parameters(), lr=LEARNING_RATE)

    history1 = train_and_save(
        net1, EPOCHS, criterion, optimizer1,
        train_loader, test_loader,
        save_path="./weights/triconvnet_best.pth",
        device=DEVICE,
    )

    print(f"TriConvNet - Best accuracy: {history1['best_accuracy']:.2f}%")

    # ------------------------------------------------------------------
    # Training rete 2: TriConvNet2 (versione semplificata)
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Training TriConvNet2")
    print("=" * 60)

    net2 = TriConvNet2(num_classes=2)
    optimizer2 = optim.Adam(net2.parameters(), lr=LEARNING_RATE)

    history2 = train_and_save(
        net2, EPOCHS, criterion, optimizer2,
        train_loader, test_loader,
        save_path="./weights/triconvnet2_best.pth",
        device=DEVICE,
    )

    print(f"TriConvNet2 - Best accuracy: {history2['best_accuracy']:.2f}%")


if __name__ == "__main__":
    main()
