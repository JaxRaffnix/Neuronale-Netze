
"""
visualize_results.py — Visualization suite for CIFAR-10 CNN model
===============================================================
Generates and saves figures for:
- Predictions grid
- Confusion matrix
- Per-class accuracy
- Feature maps
- Grad-CAM heatmap
- t-SNE embeddings
"""

import os
import torch
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
from torchvision import transforms
from tqdm import tqdm
from datetime import datetime
import torch.nn.functional as F
from torchviz import make_dot
import pandas as pd
import ast
import hiddenlayer as hl
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.io as pio

sns.set_theme(context="talk", style="whitegrid")

CMAPS = {
    "confusion": "crest",    # perceptually uniform
    "diverging": "icefire",  # for error or deviation plots
    "bars": "viridis",       # for accuracy and metrics
}
palette = sns.color_palette("viridis", n_colors=10).as_hex()
FIGSIZE = {
    "full": (6.4, 3.6),
    "half": (3.1, 1.8),
    "tall": (5.0, 4.0),
    "wide": (7.0, 3.5),
}

plt.rcParams.update({
    # --- Figure layout ---
    "figure.figsize": FIGSIZE["full"],      # 16:9 aspect ratio (full width)
    "figure.dpi": 200,                 # crisp display, lighter files than 300
    "savefig.dpi": 300,                # for exported plots (publication quality)

    # --- Font sizes ---
    "font.size": 10,                   # match Beamer base font
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,

    # --- Fonts ---
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Fira Sans"],
    "text.latex.preamble": r"""
        \usepackage{FiraSans}
        \usepackage{sfmath}
        \renewcommand*\familydefault{\sfdefault}
    """,

    # --- Axes and grid ---
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.linestyle": ":",
    "grid.alpha": 0.3,

    # --- Save options ---
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

plotly_template = dict(
    layout=dict(
        font=dict(family="Fira Sans", size=10, color="black"),
        title=dict(font=dict(size=11)),
        xaxis=dict(title_font=dict(size=10), tickfont=dict(size=9)),
        yaxis=dict(title_font=dict(size=10), tickfont=dict(size=9)),
        legend=dict(font=dict(size=9)),
        width=1280,   # full slide width (~16:9)
        height=720,   # full slide height
        coloraxis=dict(colorbar=dict(title="Importance")),  # optional default
    )
)


# -------------------------------------------------------------------------
# Setup
# -------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ['plane','car','bird','cat','deer','dog','frog','horse','ship','truck']

# Folder to save images
IMG_DIR = "images"
os.makedirs(IMG_DIR, exist_ok=True)

def savefig(name, ext="pdf", tight=True, transparent=True, dpi=300):
    """
    Save the current matplotlib figure in a clean, consistent format.

    Parameters
    ----------
    name : str
        Base filename (no extension).
    ext : str, optional
        File extension, e.g. 'pdf', 'png', or 'svg'. Default is 'pdf'.
    tight : bool, optional
        Apply tight_layout() before saving. Default is True.
    transparent : bool, optional
        Use transparent background (useful for LaTeX Beamer overlays). Default is False.
    dpi : int, optional
        Dots per inch for raster formats (e.g., PNG). Default is 300.
    """

    # Ensure output directory exists
    os.makedirs(IMG_DIR, exist_ok=True)

    # Build full file path
    path = os.path.join(IMG_DIR, f"{name}.{ext}")

    # Apply layout adjustments
    if tight:
        plt.tight_layout()

    # Save figure
    plt.savefig(path, dpi=dpi, bbox_inches="tight", transparent=transparent)

    # Close to free memory
    plt.close()

    # Print confirmation with size info
    try:
        size_kb = os.path.getsize(path) / 1024
        print(f"💾 Saved: {path} ({size_kb:.1f} KB)")
    except OSError:
        print(f"💾 Saved: {path}")

# -------------------------------------------------------------------------
# Load trained model
# -------------------------------------------------------------------------
model = torch.load("09 best model", weights_only=False)
model.to(device)
model.eval()

# -------------------------------------------------------------------------
# Load test data
# -------------------------------------------------------------------------
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010))
])
testset = torchvision.datasets.CIFAR10(
    root='datasets', train=False, download=True, transform=transform_test)
loader_test = torch.utils.data.DataLoader(
    testset, batch_size=256, shuffle=False, num_workers=4)

# -------------------------------------------------------------------------
# Helper: denormalize for visualization
# -------------------------------------------------------------------------
def denormalize(img):
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2023, 0.1994, 0.2010])
    img = img.permute(1,2,0).cpu().numpy()
    img = std * img + mean
    return np.clip(img, 0, 1)

# -------------------------------------------------------------------------
# Optuna Bayesian Seach Plots
# -------------------------------------------------------------------------
# Load study
study_name = "cifar10_hyperparam"
storage_name = f"sqlite:///{study_name}.db"
study = optuna.load_study(study_name=study_name, storage=storage_name)

# Plots
# Optimization history
def plot_optuna(palette, FIGSIZE, plotly_template, IMG_DIR, study):
    fig = optuna.visualization.plot_optimization_history(study)
    # fig.update_layout(**plotly_template["layout"])
    # fig.update_traces(marker=dict(color=palette))
    fig.write_image(os.path.join(IMG_DIR, "optimization_history.png"))

# --- Parameter importances (half slide) ---
    fig2 = optuna.visualization.plot_param_importances(study)

# Make a copy of the template and override width/height
    # layout_half = plotly_template["layout"].copy()
    # layout_half["width"] = int(FIGSIZE["half"][0] / FIGSIZE["full"][0] * plotly_template["layout"]["width"])
    # layout_half["height"] = int(FIGSIZE["half"][1] / FIGSIZE["full"][1] * plotly_template["layout"]["height"])

    # fig2.update_layout(**layout_half)
    # fig2.update_traces(marker=dict(color=palette))
    fig2.write_image(os.path.join(IMG_DIR, "param_importances.png"))

# --- Parallel coordinates (full slide) ---
    fig3 = optuna.visualization.plot_parallel_coordinate(study)
    # fig3.update_layout(**plotly_template["layout"])
    fig3.write_image(os.path.join(IMG_DIR, "parallel_coordinate.png"))

# -------------------------------------------------------------------------
# 1️⃣ Predictions Grid
# -------------------------------------------------------------------------
def show_predictions_grid(model, loader, rows=5, cols=8):
    """Show a compact grid of predictions vs true labels."""
    model.eval()
    imgs, labels = next(iter(loader))
    imgs, labels = imgs.to(device), labels.to(device)

    with torch.no_grad():
        preds = model(imgs).argmax(1)

    fig, axes = plt.subplots(
        nrows=rows,
        ncols=cols,
        figsize=FIGSIZE["full"],  # base 16:9 size
        constrained_layout=True,  # use full space efficiently
    )

    # Flatten axes array for easy iteration
    axes = axes.flatten()
    
    # Compute number of images to show
    num_imgs = min(rows * cols, len(imgs))

    for i in range(num_imgs):
        ax = axes[i]
        img = denormalize(imgs[i])
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])

        # Hide all spines
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Disable grid
        ax.grid(False)

        true, pred = CLASSES[labels[i]], CLASSES[preds[i]]
        color = "green" if pred == true else "red"
        ax.set_title(pred, color=color, fontsize=8.5, pad=2, fontweight="semibold")

    # Hide unused axes
    for ax in axes[num_imgs:]:
        ax.axis("off")

    # Adjust spacing
    plt.subplots_adjust(wspace=0.2, hspace=0.25)

    savefig("predictions_grid")

# -------------------------------------------------------------------------
# 2️⃣ Confusion Matrix & 3️⃣ Per-class Accuracy
# -------------------------------------------------------------------------
def show_confusion_matrix(model, loader):
    y_true, y_pred = [], []
    model.eval()

    with torch.no_grad():
        for x, y in tqdm(loader, desc="Evaluating"):
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(1)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    # --- Confusion matrix plot ---
    plt.figure(figsize=FIGSIZE["full"])  # 16:9 aspect ratio from your settings
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        cmap=CMAPS["bars"],  # consistent theme from your settings
        annot_kws={"size": 8},
        cbar_kws={"label": "Accuracy"},
        # square=True,
    )

    # --- Axis and labels ---
    plt.xlabel("Predicted label", labelpad=6)
    plt.ylabel("True label", labelpad=6)
    plt.xticks(rotation=40, ha="right")   # ✅ readable in 2-column mode
    plt.yticks(rotation=0, va="center")   # ✅ TODO solved

    # --- Layout and save ---
    plt.tight_layout(pad=0.5)
    savefig("confusion_matrix")

    # Per-class accuracy
    class_acc = cm.diagonal() / cm.sum(axis=1)
    class_acc = class_acc.astype(float)  # sicherstellen, dass float
    plt.figure(figsize=FIGSIZE["full"])
    sns.barplot(x=CLASSES, y=class_acc, palette=sns.color_palette(CMAPS["bars"], n_colors=len(CLASSES)))
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    savefig("class_accuracy")


# -------------------------------------------------------------------------
# Accuacries
# -------------------------------------------------------------------------
df = pd.read_csv("training_results.csv", converters={
        "train_accs": ast.literal_eval,
        "val_accs": ast.literal_eval,
        "lrs": ast.literal_eval
    })
df_expanded = pd.concat([
        pd.DataFrame({
            "epoch": range(1, len(r.train_accs)+1),
            "train_acc": r.train_accs,
            "val_acc": r.val_accs,
            "trial": r.trial,
            "lr": r.lrs, 
        })
        for _, r in df.iterrows()
    ], ignore_index=True)
best_trial = df.loc[df['val_acc'].idxmax(), 'trial']

def plot_accuracies(df):
    """
    Plots training and validation accuracy curves for all trials, 
    highlighting the best trial.
    
    Args:
        df (pd.DataFrame): DataFrame containing training results, with columns:
            ['trial', 'train_accs', 'val_accs', 'lrs', ...]
        save_path (str): File path to save the figure.
    """

    # All trials in faint lines
    sfig, ax = plt.subplots(1, 2, sharey=True)
    sns.lineplot(data=df_expanded, x="epoch", y="train_acc", hue="trial", alpha=0.3, legend=False, ax=ax[0])
    sns.lineplot(data=df_expanded, x="epoch", y="val_acc", hue="trial", alpha=0.3, legend=False, ax=ax[1])
    # highlight best trial
    sns.lineplot(data=df_expanded[df_expanded.trial==best_trial], x="epoch", y="train_acc", color="red", lw=2, ax=ax[0], label=f"Best trial")
    sns.lineplot(data=df_expanded[df_expanded.trial==best_trial], x="epoch", y="val_acc", color="red", lw=2, ax=ax[1], label=f"Best trial")

    ax[0].set_title("Training Accuracy")
    ax[1].set_title("Validation Accuracy")
    ax[0].set_xlabel("Epoch"); ax[0].set_ylabel("Accuracy")
    ax[1].set_xlabel("Epoch"); ax[1].set_ylabel("Accuracy")
    savefig("all_trials_accuracy")

def plot_accuracies_csv(csv_path):
    """
    Loads Optuna results from a CSV and plots training and validation accuracies for all trials.
    Highlights the best trial based on final validation accuracy.
    """
    # Load and parse lists
    df = pd.read_csv(csv_path)
    for col in ["train_accs", "val_accs", "lrs"]:
        df[col] = df[col].apply(ast.literal_eval)

    # Expand into long format
    records = []
    for _, row in df.iterrows():
        n_epochs = len(row["train_accs"])
        for epoch in range(n_epochs):
            records.append({
                "trial": row["trial"],
                "epoch": epoch + 1,
                "train_acc": row["train_accs"][epoch],
                "val_acc": row["val_accs"][epoch],
                "lr": row["lrs"][epoch],
                "val_acc_final": row["val_acc"],  # for identifying best
            })

    df_expanded = pd.DataFrame(records)

    # Identify best trial (highest final val_acc)
    best_trial = df_expanded.loc[df_expanded.groupby("trial")["val_acc_final"].first().idxmax(), "trial"]

    # --- Plot ---
    fig, ax = plt.subplots(1, 2, sharey=True)

    sns.lineplot(data=df_expanded, x="epoch", y="train_acc", hue="trial",
                 alpha=0.3, legend=False, ax=ax[0])
    sns.lineplot(data=df_expanded, x="epoch", y="val_acc", hue="trial",
                 alpha=0.3, legend=False, ax=ax[1])

    # Highlight best trial
    sns.lineplot(data=df_expanded[df_expanded.trial == best_trial],
                 x="epoch", y="train_acc", color="red", lw=2, ax=ax[0], label="Best trial")
    sns.lineplot(data=df_expanded[df_expanded.trial == best_trial],
                 x="epoch", y="val_acc", color="red", lw=2, ax=ax[1], label="Best trial")

    ax[0].set_title("Training Accuracy")
    ax[1].set_title("Validation Accuracy")
    for a in ax:
        a.set_xlabel("Epoch")
        a.set_ylabel("Accuracy")
    savefig("all_trials_accuracy")

# -------------------------------------------------------------------
# 2️⃣ show_learning_rate
# -------------------------------------------------------------------
def show_learning_rate():
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=df_expanded, x="epoch", y="lr", hue="trial", alpha=0.4, legend=False)
    sns.lineplot(data=df_expanded[df_expanded.trial == best_trial], x="epoch", y="lr",
             color="red", lw=2, label=f"Best trial")
    plt.title("Learning Rate Sched")
    plt.xlabel("Epoch")
    plt.ylabel("Learning rate")
    savefig("all_trials_lr")

# --------------------------------------------
# 3️⃣ Runtime comparison
# --------------------------------------------
def show_runtimes(df):
    plt.figure(figsize=(8, 4))
    sns.barplot(data=df.sort_values("runtime", ascending=False), x="trial", y="runtime", palette="viridis")
    plt.xlabel("Trial")
    plt.ylabel("Runtime (s)")
    savefig("runtime_per_trial")

# -------------------------------------------------------------------------
# Run all visualizations
# -------------------------------------------------------------------------
if __name__ == "__main__":
    print("📊 Generating and saving visualizations to /images ...")
    show_predictions_grid(model, loader_test)
    # show_confusion_matrix(model, loader_test)
    # plot_accuracies(df)
    # plot_accuracies_csv("results_merged.csv")
    # show_runtimes(df)
    # plot_optuna(palette, FIGSIZE, plotly_template, IMG_DIR, study)
    print("✅ All visualizations saved in the 'images' folder.")
