"""
trainings.py

Contiene:
    - train_and_save: allena una rete per un numero specificato di
      epoche, con il criterion e l'optimizer indicati, valutandola ad
      ogni epoca sul test loader e salvando (opzionalmente) i pesi
      che ottengono la miglior accuracy.
    - check_accuracy: calcola l'accuracy di una rete confrontando le
      predizioni con le label reali.

Implementazione basata sul tutorial ufficiale PyTorch
"Training a classifier":
https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
"""

import torch


def check_accuracy(net, loader, device="cpu"):
    """
    Calcola l'accuracy della rete sul DataLoader fornito, confrontando
    la classe predetta con la label reale.

    Parameters
    ----------
    net : torch.nn.Module
        Rete da valutare.
    loader : torch.utils.data.DataLoader
        DataLoader contenente i dati su cui valutare l'accuracy.
    device : str
        Device su cui eseguire i calcoli ("cpu" o "cuda").

    Returns
    -------
    float
        Accuracy percentuale (0-100).
    """
    net.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total if total > 0 else 0.0
    return accuracy


def train_and_save(net, epochs, criterion, optimizer, train_loader,
                    test_loader, save_path=None, device="cpu",
                    verbose=True):
    """
    Allena la rete `net` per `epochs` epoche usando `criterion` e
    `optimizer`, valutando l'accuracy ad ogni epoca su `test_loader`.
    Se `save_path` è specificato, salva i pesi della rete ogni volta
    che viene raggiunta una nuova miglior accuracy.

    Parameters
    ----------
    net : torch.nn.Module
        Rete da allenare.
    epochs : int
        Numero di epoche di training.
    criterion : callable
        Funzione di loss (es. nn.CrossEntropyLoss()).
    optimizer : torch.optim.Optimizer
        Optimizer da usare per l'aggiornamento dei pesi.
    train_loader : torch.utils.data.DataLoader
        DataLoader per il training set.
    test_loader : torch.utils.data.DataLoader
        DataLoader per il test set, usato per valutare l'accuracy.
    save_path : str o None
        Percorso dove salvare i pesi migliori. Se None, non salva.
    device : str
        Device su cui eseguire il training ("cpu" o "cuda").
    verbose : bool
        Se stampare a video loss e accuracy ad ogni epoca.

    Returns
    -------
    dict
        Dizionario con le liste 'loss' e 'accuracy' registrate ad ogni
        epoca, e 'best_accuracy' con il valore massimo raggiunto.
    """
    net.to(device)
    best_accuracy = 0.0
    history = {"loss": [], "accuracy": []}

    for epoch in range(epochs):
        net.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = check_accuracy(net, test_loader, device=device)

        history["loss"].append(epoch_loss)
        history["accuracy"].append(epoch_accuracy)

        if verbose:
            print(f"Epoch [{epoch + 1}/{epochs}] "
                  f"Loss: {epoch_loss:.4f} "
                  f"Accuracy: {epoch_accuracy:.2f}%")

        if save_path is not None and epoch_accuracy > best_accuracy:
            best_accuracy = epoch_accuracy
            torch.save(net.state_dict(), save_path)
            if verbose:
                print(f"  -> Nuovo best model salvato in "
                      f"{save_path} (accuracy {best_accuracy:.2f}%)")

    history["best_accuracy"] = best_accuracy
    return history
