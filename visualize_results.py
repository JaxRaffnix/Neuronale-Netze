
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

sns.set_theme("talk")

# -------------------------------------------------------------------------
# Setup
# -------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ['plane','car','bird','cat','deer','dog','frog','horse','ship','truck']

# Folder to save images
IMG_DIR = "images"
os.makedirs(IMG_DIR, exist_ok=True)

def savefig(name, tight=True):
    """Save the current figure with a clean format."""
    path = os.path.join(IMG_DIR, f"{name}.png")
    if tight:
        plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"💾 Saved: {path}")

# -------------------------------------------------------------------------
# Load trained model
# -------------------------------------------------------------------------
model = torch.load("08 best model", weights_only=False)
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
# 1️⃣ Predictions Grid
# -------------------------------------------------------------------------
def show_predictions_grid(model, loader, n=25):
    model.eval()
    imgs, labels = next(iter(loader))
    imgs, labels = imgs.to(device), labels.to(device)
    with torch.no_grad():
        preds = model(imgs).argmax(1)

    fig, axes = plt.subplots(5, 5, figsize=(10,10))
    for i, ax in enumerate(axes.flat):
        img = denormalize(imgs[i])
        ax.imshow(img)
        ax.axis('off')
        true, pred = CLASSES[labels[i]], CLASSES[preds[i]]
        color = "green" if pred == true else "red"
        ax.set_title(f"{pred}", color=color, fontsize=9)
    fig.suptitle(f"{n} Model Predictions (green=correct, red=wrong)")
    savefig("predictions_grid")

# -------------------------------------------------------------------------
# 2️⃣ Confusion Matrix & 3️⃣ Per-class Accuracy
# -------------------------------------------------------------------------
def show_confusion_matrix(model, loader):
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Evaluating"):
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(1)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8,6))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES, annot_kws={"size": 8})
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Normalized Confusion Matrix")
    savefig("confusion_matrix")

    # Per-class accuracy
    class_acc = cm.diagonal() / cm.sum(axis=1)
    plt.figure(figsize=(12,6))
    sns.barplot(x=CLASSES, y=class_acc, palette="viridis")
    plt.title("Per-Class Accuracy")
    plt.ylim(0,1)
    plt.ylabel("Accuracy")
    savefig("class_accuracy")

# -------------------------------------------------------------------------
# 4️⃣ Feature Map Visualization
# -------------------------------------------------------------------------
def visualize_feature_maps(model, image_idx=0):
    model.eval()
    x, _ = next(iter(loader_test))
    img = x[image_idx:image_idx+1].to(device)

    activations = []
    hooks = []
    def hook_fn(module, inp, out):
        activations.append(out.detach().cpu())

    # Collect activations from first 3 Conv layers
    conv_layers = [m for m in model.modules() if isinstance(m, torch.nn.Conv2d)]
    for layer in conv_layers[:3]:
        hooks.append(layer.register_forward_hook(hook_fn))

    _ = model(img)
    for h in hooks: h.remove()

    fig, axes = plt.subplots(1, len(activations), figsize=(15,5))
    for i, feat in enumerate(activations):
        fmap = feat[0, :16].permute(1,2,0).mean(-1)
        axes[i].imshow(fmap, cmap='viridis')
        axes[i].set_title(f"Conv Layer {i+1}")
        axes[i].axis('off')
    plt.suptitle("Feature Maps from Different Layers")
    savefig("feature_maps")

# -------------------------------------------------------------------------
# 5️⃣ Grad-CAM
# -------------------------------------------------------------------------
def gradcam(model, img, target_class=None):
    model.eval()
    grads, activations = [], []

    def backward_hook(module, grad_in, grad_out):
        grads.append(grad_out[0].detach())
    def forward_hook(module, inp, out):
        activations.append(out.detach())

    last_conv = None
    for layer in reversed(list(model.modules())):
        if isinstance(layer, torch.nn.Conv2d):
            last_conv = layer
            break
    fh = last_conv.register_forward_hook(forward_hook)
    bh = last_conv.register_full_backward_hook(backward_hook)

    output = model(img)
    if target_class is None:
        target_class = output.argmax(1)
    loss = output[0, target_class]
    model.zero_grad()
    loss.backward()

    fh.remove(); bh.remove()
    act = activations[0][0]
    grad = grads[0][0]
    weights = grad.mean(dim=(1,2))
    cam = torch.zeros_like(act[0])
    for i, w in enumerate(weights):
        cam += w * act[i]
    cam = torch.clamp(cam, min=0)
    cam = cam / cam.max()

    img_np = denormalize(img[0])
    heatmap_resized = F.interpolate(
        cam.unsqueeze(0).unsqueeze(0), size=(img_np.shape[0], img_np.shape[1]), mode='bilinear', align_corners=False
    )[0, 0].cpu().numpy()
    heatmap = plt.cm.jet(heatmap_resized)[...,:3]
    overlay = 0.5 * heatmap + 0.5 * img_np
    plt.figure(figsize=(5,5))
    plt.imshow(overlay)
    plt.title(f"Grad-CAM (class={CLASSES[target_class]})")
    plt.axis('off')
    savefig("gradcam")

# -------------------------------------------------------------------------
# 6️⃣ t-SNE Embedding
# -------------------------------------------------------------------------
def show_tsne_embeddings(model, loader, num_samples=2000):
    model.eval()
    features, labels = [], []

    with torch.no_grad():
        for x, y in tqdm(loader, total=num_samples//loader.batch_size):
            x, y = x.to(device), y.to(device)
            # Extract features before classifier layers
            out = model[:-2](x) if isinstance(model, torch.nn.Sequential) else model(x)
            if out.ndim > 2:
                out = torch.flatten(out, 1)
            features.append(out.cpu())
            labels.append(y.cpu())
            if len(features)*loader.batch_size >= num_samples:
                break

    features = torch.cat(features)[:num_samples]
    labels = torch.cat(labels)[:num_samples]
    
    # t-SNE to 2D
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    emb = tsne.fit_transform(features)

    # Map numeric labels to class names
    label_names = [CLASSES[int(l)] for l in labels]

    plt.figure(figsize=(8,6))
    sns.scatterplot(x=emb[:,0], y=emb[:,1],
                    hue=label_names,  # Use class labels for hue
                    s=20, palette="tab10", legend='full')
    plt.title("t-SNE of Learned Representations")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # legend outside plot
    savefig("tsne_embeddings")
    plt.show()

# -------------------------------------------------------------------------
# Layer Archiutecture
# -------------------------------------------------------------------------
def show_layers(device, model, save_dir="images"):
    """
    Visualizes the model architecture in multiple ways:
    1. Full model (simplified, vector graphics)
    2. First convolutional block
    3. HiddenLayer graph (clean hierarchical)
    """
    x = torch.randn(1, 3, 32, 32).to(device)
    y = model(x)

    # -------------------------------
    # Full model (simplified)
    # -------------------------------
    params = {k: v for k, v in model.named_parameters()}
    dot = make_dot(y, params=params, show_attrs=False, show_saved=False)
    dot.format = 'svg'   # zoomable vector graphic
    dot.render(f'{save_dir}/model_graph', cleanup=True)

    # -------------------------------
    # First block only
    # -------------------------------
    block_out = model[:4](x)  # first Conv+BN+ReLU
    dot_block = make_dot(block_out, params=dict(model.named_parameters()), show_attrs=False, show_saved=False)
    dot_block.format = 'svg'
    dot_block.render(f'{save_dir}/reduced_graph', cleanup=True)

    # -------------------------------
    # HiddenLayer visualization
    # -------------------------------
    hl_graph = hl.build_graph(model, torch.zeros([1, 3, 32, 32])).build_dot()
    # hl_graph.save(f"{save_dir}/hl_model_graph", format="svg")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch

def draw_model_layers_zigzag(layers, layer_names=None, figsize=(14,4), savepath=None):
    """
    Draw a sequential model in a zig-zag horizontal layout.
    After MaxPool blocks, the next block goes one row below.
    """
    n = len(layers)
    fig, ax = plt.subplots(figsize=figsize)
    
    # coordinates
    x, y = 0.05, 0.75
    dx, dy = 0.08, 0.25
    spacing = 0.04  # horizontal spacing
    row_shift = 0.35  # vertical shift after MaxPool
    
    # Colors per type
    conv_color = "#a6cee3"
    pool_color = "#1f78b4"
    linear_color = "#b2df8a"
    act_color = "#fb9a99"
    dropout_color = "#fdbf6f"
    
    def get_color(layer_name):
        l = layer_name.lower()
        if "conv" in l: return conv_color
        if "pool" in l: return pool_color
        if "linear" in l: return linear_color
        if "relu" in l or "sigmoid" in l: return act_color
        if "dropout" in l: return dropout_color
        return "#cccccc"
    
    # Track previous coordinates for arrows
    prev_x, prev_y = x, y
    
    for i, layer in enumerate(layers):
        color = get_color(layer)
        
        rect = FancyBboxPatch((x, y-dy/2), dx, dy, linewidth=1,
                              edgecolor='black', facecolor=color,
                              boxstyle="round,pad=0.02")
        ax.add_patch(rect)
        
        # Label
        if layer_names is None:
            ax.text(x+dx/2, y, layer, ha='center', va='center', fontsize=9)
        else:
            ax.text(x+dx/2, y+0.05, layer, ha='center', va='bottom', fontsize=9, weight='bold')
            ax.text(x+dx/2, y-0.05, layer_names[i], ha='center', va='top', fontsize=8, color='gray')
        
        # Draw arrow from previous block
        if i > 0:
            ax.annotate('', xy=(x, y), xytext=(prev_x+dx, prev_y),
                        arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
        
        # If current layer is MaxPool, drop down one row
        if "pool" in layer.lower():
            x += dx + spacing
            prev_x, prev_y = x, y
            y -= row_shift
        else:
            prev_x, prev_y = x, y
            x += dx + spacing
    
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.axis('off')
    
    if savepath:
        plt.savefig(savepath, bbox_inches='tight')
    plt.show()

# -----------------------------------
# Example for your CIFAR-10 model
layers = [
    "Conv2d 3→64 5x5", "BatchNorm2d", "ReLU",
    "Conv2d 64→64 3x3", "BatchNorm2d", "ReLU", "MaxPool2d 2x2",
    "Conv2d 64→64 3x3", "BatchNorm2d", "ReLU", "MaxPool2d 2x2",
    "Flatten", "Linear 4096→128", "ReLU", "Dropout", "Linear 128→10"
]

output_shapes = [
    "32x32", "32x32", "32x32",
    "32x32", "32x32", "32x32", "16x16",
    "16x16", "16x16", "16x16", "8x8",
    "4096", "128", "128", "128", "10"
]


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

def plot_all_trials_accuracies(df):
    """
    Plots training and validation accuracy curves for all trials, 
    highlighting the best trial.
    
    Args:
        df (pd.DataFrame): DataFrame containing training results, with columns:
            ['trial', 'train_accs', 'val_accs', 'lrs', ...]
        save_path (str): File path to save the figure.
    """
    fig, ax = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    # All trials in faint lines
    sfig, ax = plt.subplots(1, 2, figsize=(13,5), sharey=True)
    sns.lineplot(data=df_expanded, x="epoch", y="train_acc", hue="trial", alpha=0.3, legend=False, ax=ax[0])
    sns.lineplot(data=df_expanded, x="epoch", y="val_acc", hue="trial", alpha=0.3, legend=False, ax=ax[1])
    # highlight best trial
    sns.lineplot(data=df_expanded[df_expanded.trial==best_trial], x="epoch", y="train_acc", color="red", lw=2, ax=ax[0])
    sns.lineplot(data=df_expanded[df_expanded.trial==best_trial], x="epoch", y="val_acc", color="red", lw=2, ax=ax[1])

    ax[0].set_title("Training Accuracy – all trials")
    ax[1].set_title("Validation Accuracy – all trials")
    ax[0].set_xlabel("Epoch"); ax[0].set_ylabel("Accuracy")
    ax[1].set_xlabel("Epoch"); ax[1].set_ylabel("Accuracy")
    savefig("all_trials_accuracy")


# -------------------------------------------------------------------
# 2️⃣ show_learning_rate
# -------------------------------------------------------------------
def show_learning_rate():
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=df_expanded, x="epoch", y="lr", hue="trial", alpha=0.4, legend=False)
    sns.lineplot(data=df_expanded[df_expanded.trial == best_trial], x="epoch", y="lr",
             color="red", lw=2, label=f"Best trial {best_trial}")
    plt.title("Learning Rate Schedules – all trials")
    plt.xlabel("Epoch")
    plt.ylabel("Learning rate")
    savefig("all_trials_lr")

# --------------------------------------------
# 3️⃣ Runtime comparison
# --------------------------------------------
def show_runtimes(df):
    plt.figure(figsize=(8, 4))
    sns.barplot(data=df.sort_values("runtime", ascending=False), x="trial", y="runtime", palette="viridis")
    plt.title("Runtime per Trial")
    plt.xlabel("Trial")
    plt.ylabel("Runtime [s]")
    savefig("runtime_per_trial")

# -------------------------------------------------------------------------
# Run all visualizations
# -------------------------------------------------------------------------
if __name__ == "__main__":
    print("📊 Generating and saving visualizations to /images ...")
    # show_predictions_grid(model, loader_test)
    # show_confusion_matrix(model, loader_test)
    # visualize_feature_maps(model)
    img, _ = next(iter(loader_test))
    # gradcam(model, img[0:1].to(device))
    # show_tsne_embeddings(model, loader_test)
    # show_layers(device, model)
    draw_model_layers_zigzag(layers, output_shapes, figsize=(14,4), savepath="images/model_zigzag.pdf")
    # plot_all_trials_accuracies(df)
    # show_runtimes(df)
    # show_learning_rate()
    print("✅ All visualizations saved in the 'images' folder.")
