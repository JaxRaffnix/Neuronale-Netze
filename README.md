# Neuronale Netze in der Bildverarbeitung

Vorlesung von Prof. Bauer an der Hochschule Karlsruhe. Anlehnung an die Stanford CS class [cs231n](https://cs231n.github.io/) von [Justin Johnson](https://web.eecs.umich.edu/~justincj/).


Eigene Präsentation eines Moells unter [talk/Präsentation Pytorch.pdf](talk/Präsentation%20Pytorch.pdf)

## Inhalt

- Notebooks mit Theorie, Aufgaben und Loesungen: [notebooks/](notebooks/)
- Python-Implementierungen (Layer, Solver, Utils): [src/EITB712M/](src/EITB712M/)
- Datensaetze (CIFAR-10, Hymenoptera, Kaggle Cats vs Dogs): [data/](data/)
- Ergebnisse und Modelle: [artifacts/](artifacts/)
- Vortrag und Quellen: [talk/](talk/)

## Notebooks

- [notebooks/01_python_tutorial.ipynb](notebooks/01_python_tutorial.ipynb)
- [notebooks/02_SimpleClassification.ipynb](notebooks/02_SimpleClassification.ipynb)
- [notebooks/03_knn.ipynb](notebooks/03_knn.ipynb)
- [notebooks/04_msvm.ipynb](notebooks/04_msvm.ipynb)
- [notebooks/05_two_layer_net.ipynb](notebooks/05_two_layer_net.ipynb)
- [notebooks/06_image_features.ipynb](notebooks/06_image_features.ipynb)
- [notebooks/07_Modular_Backpropagation.ipynb](notebooks/07_Modular_Backpropagation.ipynb)
- [notebooks/08_ConvolutionalNetworks.ipynb](notebooks/08_ConvolutionalNetworks.ipynb)
- [notebooks/09_PyTorch.ipynb](notebooks/09_PyTorch.ipynb)
- [notebooks/10_TransferLearning.ipynb](notebooks/10_TransferLearning.ipynb)
- [notebooks/11_SemantischeSegmentierung.ipynb](notebooks/11_SemantischeSegmentierung.ipynb)
- [notebooks/12_StyleTransfer.ipynb](notebooks/12_StyleTransfer.ipynb)

## Schnellstart

Voraussetzung: Python >= 3.14 (laut [pyproject.toml](pyproject.toml)).

1) Virtuelle Umgebung erstellen und aktivieren

```bash
python -m venv .venv
```

```powershell
.\.venv\Scripts\Activate.ps1
```

2) Abhaengigkeiten installieren

```bash
pip install -e .
```

Optional mit uv:

```bash
uv sync
```

3) Jupyter Notebook starten

```bash
jupyter notebook
```

## Daten

Die Datensaetze liegen unter [data/](data/). CIFAR-10 ist bereits enthalten, weitere Datensaetze (Hymenoptera, Kaggle Cats vs Dogs) sind dort strukturiert abgelegt. Falls du eigene Daten nutzt, lege sie in einem neuen Unterordner unter [data/](data/) ab.

## Ergebnisse

Beispielgrafik der Accuracy aus Kapitel 9: ![talk/images/all_trials_accuracy.png](talk/images/all_trials_accuracy.png)

Trainierte Modelle und Artefakte befinden sich unter [artifacts/](artifacts/).