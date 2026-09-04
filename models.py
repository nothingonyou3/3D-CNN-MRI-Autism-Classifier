"""
models.py

Definisce le due reti neurali convoluzionali 3D usate nel progetto:

    - TriConvNet: rete completa con 6 layer convoluzionali,
      2 layer di upsample (scale factor 2) e 3 layer fully connected.
      Struttura ispirata a:
      https://github.com/xmuyzz/3D-CNN-PyTorch/blob/master/models/C3DNet.py

    - TriConvNet2: versione semplificata di TriConvNet, con
      4 layer convoluzionali e 2 layer fully connected.
"""

import torch
import torch.nn as nn


class TriConvNet(nn.Module):
    """
    Rete 3D-CNN completa.

    Struttura:
        - 6 layer convoluzionali (alcuni seguiti da max pooling)
        - 2 layer di upsample con scale_factor=2
        - 3 layer fully connected
    """

    def __init__(self, num_classes=2):
        super(TriConvNet, self).__init__()

        # Blocco convoluzionale 1: conv + maxpool
        self.conv1 = self._layer_conv_maxpool(1, 16, stride=1)
        # Blocco convoluzionale 2: conv + maxpool
        self.conv2 = self._layer_conv_maxpool(16, 32, stride=1)
        # Blocco convoluzionale 3: conv semplice
        self.conv3 = self._layer_conv(32, 64, stride=1)
        # Blocco convoluzionale 4: conv + maxpool
        self.conv4 = self._layer_conv_maxpool(64, 128, stride=1)

        # Upsampling: aumenta la risoluzione spaziale delle feature map
        self.upsample1 = nn.Upsample(scale_factor=2, mode="nearest")
        # Blocco convoluzionale 5: conv semplice dopo upsample
        self.conv5 = self._layer_conv(128, 64, stride=1)

        self.upsample2 = nn.Upsample(scale_factor=2, mode="nearest")
        # Blocco convoluzionale 6: conv semplice dopo upsample
        self.conv6 = self._layer_conv(64, 32, stride=1)

        # Global average pooling per rendere la rete indipendente
        # dalla dimensione esatta del volume in ingresso
        self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))

        # Layer fully connected
        self.fc1 = self._layer_fully_connected(32, 128)
        self.fc2 = self._layer_fully_connected(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    # ------------------------------------------------------------------
    # Blocchi ausiliari
    # ------------------------------------------------------------------
    def _layer_conv(self, c_in, c_out, stride):
        """
        Blocco Conv3d + BatchNorm3d + ReLU, incapsulato in un
        nn.Sequential.
        """
        return nn.Sequential(
            nn.Conv3d(c_in, c_out, kernel_size=(3, 3, 3),
                      stride=stride, padding=1),
            nn.BatchNorm3d(num_features=c_out),
            nn.ReLU(),
        )

    def _layer_conv_maxpool(self, c_in, c_out, stride):
        """
        Blocco Conv3d + BatchNorm3d + ReLU + MaxPool3d, incapsulato
        in un nn.Sequential.
        """
        return nn.Sequential(
            nn.Conv3d(c_in, c_out, kernel_size=(3, 3, 3),
                      stride=stride, padding=1),
            nn.BatchNorm3d(num_features=c_out),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 2)),
        )

    def _layer_fully_connected(self, c_in, c_out, p=0.5):
        """
        Blocco Linear + ReLU + Dropout, incapsulato in un
        nn.Sequential.
        """
        return nn.Sequential(
            nn.Linear(c_in, c_out),
            nn.ReLU(),
            nn.Dropout(p=p),
        )

    # ------------------------------------------------------------------
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = self.upsample1(x)
        x = self.conv5(x)

        x = self.upsample2(x)
        x = self.conv6(x)

        x = self.gap(x)
        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x

    def scale(self, n1):
        """
        Calcola la scala (numero di parametri complessivi) della rete,
        utile in fase di debug per verificare la dimensione del modello.
        """
        total_params = sum(p.numel() for p in self.parameters())
        return total_params * n1


class TriConvNet2(nn.Module):
    """
    Versione semplificata di TriConvNet.

    Struttura:
        - 4 layer convoluzionali
        - 2 layer fully connected
    """

    def __init__(self, num_classes=2):
        super(TriConvNet2, self).__init__()

        self.conv1 = self._layer_conv_maxpool(1, 16, stride=1)
        self.conv2 = self._layer_conv_maxpool(16, 32, stride=1)
        self.conv3 = self._layer_conv_maxpool(32, 64, stride=1)
        self.conv4 = self._layer_conv_maxpool(64, 128, stride=1)

        self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.fc1 = self._layer_fully_connected(128, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def _layer_conv(self, c_in, c_out, stride):
        return nn.Sequential(
            nn.Conv3d(c_in, c_out, kernel_size=(3, 3, 3),
                      stride=stride, padding=1),
            nn.BatchNorm3d(num_features=c_out),
            nn.ReLU(),
        )

    def _layer_conv_maxpool(self, c_in, c_out, stride):
        return nn.Sequential(
            nn.Conv3d(c_in, c_out, kernel_size=(3, 3, 3),
                      stride=stride, padding=1),
            nn.BatchNorm3d(num_features=c_out),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 2)),
        )

    def _layer_fully_connected(self, c_in, c_out, p=0.5):
        return nn.Sequential(
            nn.Linear(c_in, c_out),
            nn.ReLU(),
            nn.Dropout(p=p),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = self.gap(x)
        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = self.fc2(x)

        return x

    def scale(self, n1):
        total_params = sum(p.numel() for p in self.parameters())
        return total_params * n1
