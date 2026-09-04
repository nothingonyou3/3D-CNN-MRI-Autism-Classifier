"""
mri_datautils.py

Contiene:
    - NiiDataset: classe custom che eredita da torch.utils.data.Dataset,
      usata per caricare le immagini MRI in formato .nii dal disco.
    - read_nii: funzione di supporto che legge un file .nii tramite nibabel,
      lo converte in tensore PyTorch, aggiunge la dimensione dei canali
      ed esegue il crop delle porzioni vuote dell'immagine.
    - plot_nii_images: funzione di debug per visualizzare tre sezioni
      centrali (assiale, coronale, sagittale) di un'immagine, usata per
      verificare che il crop non tagliasse via informazione utile.
    - generate_label: funzione per generare il file CSV di label richiesto
      dalla classe Dataset di PyTorch.
"""

import os
import csv
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


# Valori di crop trovati per tentativi, tali da rimuovere solo le porzioni
# di spazio vuoto (non informative) attorno al cervello.
# Applicati sui tre assi dell'immagine originale (61 x 73 x 61).
CROP_X = (7, 54)   # asse 0
CROP_Y = (5, 68)   # asse 1
CROP_Z = (6, 55)   # asse 2


def read_nii(filepath, crop=True):
    """
    Legge un file .nii dal disco e lo converte in un tensore PyTorch
    correttamente formattato per una rete 3D (C, D, H, W).

    Parameters
    ----------
    filepath : str
        Percorso del file .nii da leggere.
    crop : bool
        Se True, applica il crop per rimuovere le porzioni vuote
        dell'immagine (definite da CROP_X, CROP_Y, CROP_Z).

    Returns
    -------
    torch.Tensor
        Tensore di forma (1, D, H, W), dove il primo canale rappresenta
        il numero di canali (1, trattandosi di immagini in scala di grigi).
    """
    img = nib.load(filepath)
    data = img.get_fdata()  # array numpy grezzo, shape (61, 73, 61)

    if crop:
        data = data[CROP_X[0]:CROP_X[1],
                     CROP_Y[0]:CROP_Y[1],
                     CROP_Z[0]:CROP_Z[1]]

    # Conversione in tensore float32
    tensor = torch.from_numpy(np.asarray(data, dtype=np.float32))

    # Aggiunta della dimensione dei canali: (D, H, W) -> (1, D, H, W)
    tensor = tensor.unsqueeze(0)

    return tensor


def plot_nii_images(filepath, crop=True):
    """
    Plotta tre sezioni centrali (una per asse) di un'immagine .nii,
    utile per verificare visivamente che il crop non rimuova informazione.

    Parameters
    ----------
    filepath : str
        Percorso del file .nii da visualizzare.
    crop : bool
        Se True, mostra la versione croppata dell'immagine.
    """
    tensor = read_nii(filepath, crop=crop)
    volume = tensor.squeeze(0).numpy()  # (D, H, W)

    d, h, w = volume.shape
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(volume[d // 2, :, :], cmap="viridis")
    axes[0].set_title("Sezione asse 0 (centrale)")

    axes[1].imshow(volume[:, h // 2, :], cmap="viridis")
    axes[1].set_title("Sezione asse 1 (centrale)")

    axes[2].imshow(volume[:, :, w // 2], cmap="viridis")
    axes[2].set_title("Sezione asse 2 (centrale)")

    plt.suptitle("Cropped" if crop else "Not cropped")
    plt.tight_layout()
    plt.show()


def generate_label(data_dir, output_csv, positive_prefixes=None):
    """
    Genera il file CSV di label richiesto dalla classe Dataset di PyTorch.

    Scansiona data_dir alla ricerca di file .nii e assegna una label
    (1 = Autismo, 0 = controllo) in base al nome del file. Il criterio
    di assegnazione può essere personalizzato passando i prefissi/pattern
    che identificano i soggetti con Autismo.

    Parameters
    ----------
    data_dir : str
        Cartella contenente le immagini .nii.
    output_csv : str
        Percorso del file CSV da generare (colonne: filename, label).
    positive_prefixes : list[str] o None
        Lista di sotto-stringhe che, se presenti nel nome del file,
        identificano un soggetto con label 1 (Autismo). Se None, viene
        usato il prefisso di default "ASD".
    """
    if positive_prefixes is None:
        positive_prefixes = ["ASD"]

    rows = []
    for fname in sorted(os.listdir(data_dir)):
        if not fname.lower().endswith((".nii", ".nii.gz")):
            continue

        label = 1 if any(p in fname for p in positive_prefixes) else 0
        rows.append((fname, label))

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label"])
        writer.writerows(rows)

    print(f"Generato {output_csv} con {len(rows)} campioni "
          f"({sum(l for _, l in rows)} positivi, "
          f"{len(rows) - sum(l for _, l in rows)} negativi).")


class NiiDataset(Dataset):
    """
    Dataset custom per immagini MRI in formato .nii.

    Eredita da torch.utils.data.Dataset e richiede un file CSV
    (generato tramite generate_label) contenente il nome del file
    e la relativa label.
    """

    def __init__(self, labels_csv, data_dir, crop=True, transform=None):
        """
        Parameters
        ----------
        labels_csv : str
            Percorso del CSV con colonne 'filename' e 'label'.
        data_dir : str
            Cartella contenente le immagini .nii.
        crop : bool
            Se applicare il crop delle porzioni vuote dell'immagine.
        transform : callable o None
            Trasformazione opzionale da applicare al tensore immagine.
        """
        self.data_dir = data_dir
        self.crop = crop
        self.transform = transform

        self.samples = []
        with open(labels_csv, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append((row["filename"], int(row["label"])))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, label = self.samples[idx]
        filepath = os.path.join(self.data_dir, filename)

        image = read_nii(filepath, crop=self.crop)

        if self.transform:
            image = self.transform(image)

        return image, label
