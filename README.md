# MRI Autism Classification — 3D CNN

Progetto per il corso di **Artificial Intelligence 2022/2023**.

Autori: Samuele Capacci (70/90/00341), Giulia Avanzato (70/90/00356)

## Descrizione

L'obiettivo del progetto è classificare immagini di risonanza magnetica
(MRI) tridimensionali, distinguendo tra soggetti con diagnosi di
**Autismo** e soggetti di controllo, tramite una rete neurale
convoluzionale 3D implementata in **PyTorch**.

Il dataset è composto da circa **900 immagini tridimensionali**
provenienti da diversi centri di ricerca, tutte con dimensioni
`61 x 73 x 61` e un solo canale (scala di grigi).

## Struttura del progetto

```
.
├── mri_datautils.py   # Dataset custom, lettura/crop delle immagini .nii, generazione label
├── dataset_utils.py   # Split train/test e k-fold cross validation
├── models.py           # Definizione delle reti TriConvNet e TriConvNet2
├── trainings.py         # Funzioni di training e calcolo dell'accuracy
├── main.py              # Script principale: dataset, DataLoader, training
└── README.md
```

### `mri_datautils.py`

- **`NiiDataset`**: classe che eredita da `torch.utils.data.Dataset`,
  usata per caricare le immagini `.nii` dal disco insieme alle
  rispettive label, lette da un file CSV.
- **`read_nii`**: legge un file `.nii` tramite [nibabel](https://nipy.org/nibabel/),
  lo converte in tensore PyTorch, aggiunge la dimensione dei canali
  (formato atteso: `(C, D, H, W)`) ed effettua il crop delle porzioni
  vuote dell'immagine.
- **`plot_nii_images`**: plotta tre sezioni centrali (una per asse) di
  un'immagine, usata per verificare visivamente che il crop non
  rimuova informazione utile.
- **`generate_label`**: genera il file CSV di label (`filename, label`)
  richiesto dalla classe `Dataset`.

### `dataset_utils.py`

- **`train_test_split_dataset`**: genera due `NiiDataset` (train/test)
  a partire da un fattore di split.
- **`kfold_split_dataset`**: genera `k` coppie di dataset per la
  k-fold cross validation. *Non utilizzata* nella versione finale del
  progetto per via del tempo di calcolo richiesto, ma implementata per
  completezza.

### `models.py`

Due reti convoluzionali 3D, costruite componendo blocchi
`nn.Sequential`:

- **`TriConvNet`**: rete completa, ispirata a
  [3D-CNN-PyTorch (C3DNet)](https://github.com/xmuyzz/3D-CNN-PyTorch/blob/master/models/C3DNet.py).
  Struttura:
  - 6 layer convoluzionali (`Conv3d` + `BatchNorm3d` + `ReLU`, alcuni
    con `MaxPool3d`)
  - 2 layer di upsample (`Upsample`, scale factor 2)
  - 3 layer fully connected (`Linear` + `ReLU` + `Dropout`)

- **`TriConvNet2`**: versione semplificata di `TriConvNet`.
  Struttura:
  - 4 layer convoluzionali
  - 2 layer fully connected

Entrambe espongono anche un metodo `scale()`, usato in fase di debug
per calcolare la dimensione (numero di parametri) del modello.

### `trainings.py`

- **`train_and_save`**: allena una rete per un numero di epoche
  specificato, con il `criterion` e l'`optimizer` indicati, valutando
  l'accuracy ad ogni epoca e salvando (opzionalmente) i pesi migliori.
- **`check_accuracy`**: calcola l'accuracy confrontando le classi
  predette con le label reali.

Implementazione basata sul tutorial ufficiale PyTorch
["Training a classifier"](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html).

### `main.py`

Script principale: genera il file di label, crea gli split
train/test, istanzia i `DataLoader` (batch size **32**, necessari per
non dover caricare l'intero dataset 3D in memoria) e allena entrambe
le reti.

## Requisiti

```
torch
nibabel
numpy
matplotlib
```

Installazione:

```bash
pip install torch nibabel numpy matplotlib
```

## Utilizzo

1. Posizionare le immagini `.nii` in una cartella (es. `data/mri_images/`).
2. Modificare, se necessario, il criterio di assegnazione delle label
   in `generate_label` (`mri_datautils.py`), in base alla convenzione
   di naming del proprio dataset.
3. Configurare i percorsi in `main.py` (`DATA_DIR`, `LABELS_CSV`).
4. Avviare il training:

```bash
python main.py
```

I pesi del modello con la miglior accuracy verranno salvati nella
cartella `weights/`.

## Risultati

Sono stati effettuati diversi esperimenti variando **learning rate**
(`0.001` e `0.01`) e **numero di epoche** (fino a 200), su entrambe le
reti.

| Rete        | Learning rate | Epoche | Accuracy (range osservato) |
|-------------|:-------------:|:------:|:---------------------------:|
| TriConvNet  | 0.001         | 2–200  | ~39% – 66%                  |
| TriConvNet  | 0.01          | 5–10   | ~49% – 60%                  |
| TriConvNet2 | 0.001         | 2–200  | ~38% – 57%                  |
| TriConvNet2 | 0.01          | 5–10   | ~52% – 55%                  |

Nonostante una progressiva diminuzione della loss durante il
training, l'accuracy non ha mai superato in modo stabile il **55%**,
un risultato deludente considerando che il task è una classificazione
binaria bilanciata (dove il livello di riferimento casuale è ~50%).

## Possibili sviluppi futuri

- Rivedere in modo non banale l'architettura delle reti (profondità,
  numero di filtri, tipo di pooling).
- Sperimentare con optimizer e criterion diversi.
- Data augmentation specifica per immagini 3D.
- Utilizzare la cross validation k-fold (già implementata in
  `dataset_utils.py`) per una stima più robusta delle performance.
- Normalizzazione/standardizzazione più accurata dell'intensità dei
  voxel prima dell'addestramento.

## Riferimenti

- [PyTorch — Datasets & DataLoaders](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)
- [PyTorch — Training a Classifier](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
- [torch.nn.BatchNorm3d](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm3d.html)
- [torch.nn.Dropout](https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html)
- [torch.nn.Upsample](https://pytorch.org/docs/stable/generated/torch.nn.Upsample.html)
- [3D-CNN-PyTorch (C3DNet)](https://github.com/xmuyzz/3D-CNN-PyTorch/blob/master/models/C3DNet.py)
- [nibabel](https://nipy.org/nibabel/)
