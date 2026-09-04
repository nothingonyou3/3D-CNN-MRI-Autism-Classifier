"""
dataset_utils.py

Funzioni per generare gli split del NiiDataset:
    - train_test_split_dataset: split semplice train/test in base a un
      fattore di split.
    - kfold_split_dataset: split k-fold per cross validation (definita
      ma non utilizzata nel progetto finale per motivi di tempo di
      calcolo).
"""

import random
from torch.utils.data import Subset

from mri_datautils import NiiDataset


def train_test_split_dataset(labels_csv, data_dir, split_factor=0.8,
                               crop=True, shuffle=True, seed=42):
    """
    Genera due istanze di NiiDataset (train e test) a partire da un
    unico dataset, usando il fattore di split indicato.

    Parameters
    ----------
    labels_csv : str
        Percorso del CSV di label (generato con generate_label).
    data_dir : str
        Cartella contenente le immagini .nii.
    split_factor : float
        Frazione di campioni da assegnare al train set (es. 0.8 = 80%
        train, 20% test).
    crop : bool
        Se applicare il crop alle immagini.
    shuffle : bool
        Se mescolare gli indici prima dello split.
    seed : int
        Seed per la riproducibilità dello shuffle.

    Returns
    -------
    (Subset, Subset)
        Tupla (train_dataset, test_dataset).
    """
    full_dataset = NiiDataset(labels_csv, data_dir, crop=crop)

    indices = list(range(len(full_dataset)))
    if shuffle:
        random.Random(seed).shuffle(indices)

    split_point = int(len(indices) * split_factor)
    train_indices = indices[:split_point]
    test_indices = indices[split_point:]

    train_dataset = Subset(full_dataset, train_indices)
    test_dataset = Subset(full_dataset, test_indices)

    return train_dataset, test_dataset


def kfold_split_dataset(labels_csv, data_dir, k_fold=5, crop=True,
                          shuffle=True, seed=42):
    """
    Genera k coppie di NiiDataset (train, test) per la k-fold cross
    validation. Non utilizzata nella versione finale del progetto per
    via del tempo di calcolo richiesto, ma implementata per
    completezza.

    Parameters
    ----------
    labels_csv : str
        Percorso del CSV di label.
    data_dir : str
        Cartella contenente le immagini .nii.
    k_fold : int
        Numero di fold.
    crop : bool
        Se applicare il crop alle immagini.
    shuffle : bool
        Se mescolare gli indici prima di creare i fold.
    seed : int
        Seed per la riproducibilità dello shuffle.

    Returns
    -------
    list[tuple[Subset, Subset]]
        Lista di k coppie (train_dataset, test_dataset), ciascuna
        corrispondente a una porzione di n_campioni / k_fold usata
        come test set.
    """
    full_dataset = NiiDataset(labels_csv, data_dir, crop=crop)

    indices = list(range(len(full_dataset)))
    if shuffle:
        random.Random(seed).shuffle(indices)

    fold_size = len(indices) // k_fold
    folds = []

    for i in range(k_fold):
        start = i * fold_size
        # L'ultimo fold assorbe eventuali campioni rimanenti
        end = (i + 1) * fold_size if i < k_fold - 1 else len(indices)

        test_indices = indices[start:end]
        train_indices = indices[:start] + indices[end:]

        train_dataset = Subset(full_dataset, train_indices)
        test_dataset = Subset(full_dataset, test_indices)

        folds.append((train_dataset, test_dataset))

    return folds
