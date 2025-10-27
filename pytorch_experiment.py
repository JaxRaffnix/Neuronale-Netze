# _________________________________________________
# Imports 

if __name__ == "__main__":

    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torchvision.datasets as dset
    import torchvision.transforms as T
    from torch.utils.data import DataLoader, SubsetRandomSampler
    from torch.amp import autocast, GradScaler
    import torch.nn.functional as F
    from torch.optim.lr_scheduler import StepLR
    from itertools import product
    import copy
    import math
    import numpy as np
    import optuna

    # _________________________________________________
    # Device

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    print("Using device:", device)

    # _________________________________________________
    # Constants

    NUM_TRAIN = 49000
    BATCH_SIZE = 256
    NUM_WORKERS = 12
    # MAX_ROUNDS = 3
    NUM_EPOCHS = 10  # per round
    NUM_TRIALS = 20

    RANDOM_SEED = 0
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # _________________________________________________
    # Load Dataset
    transform_train = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomCrop(32, padding=4),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    transform_val_test = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    cifar10_train = dset.CIFAR10('datasets', train=True, download=True, transform=transform_train)
    cifar10_val_test = dset.CIFAR10('datasets', train=True, download=True, transform=transform_val_test)
    cifar10_test = dset.CIFAR10('datasets', train=False, download=True, transform=transform_val_test)

    indices = np.arange(len(cifar10_train))
    np.random.shuffle(indices)
    train_idx = indices[:NUM_TRAIN]
    val_idx = indices[NUM_TRAIN:]

    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    loader_train = DataLoader(
        cifar10_train,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True
    )

    loader_val = DataLoader(
        cifar10_val_test,
        batch_size=BATCH_SIZE,
        sampler=val_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True
    )

    print(f"Train batches: {len(loader_train)}, Val batches: {len(loader_val)}")
    print(f"Results in {np.ceil(NUM_TRAIN / BATCH_SIZE)} iterations per epoch.")

    # _________________________________________________
    # Helper Functions

    def train_model(
        model,
        train_loader,
        val_loader,
        num_epochs=10,
        lr=1e-3,
        optimizer=None,
        criterion=None,
        device=None,
        print_every=20,
        patience=3,
        accumulation_steps=1,
        scheduler=None
    ):
        """
        Optimized training function with AMP, early stopping, gradient accumulation, 
        best model saving, and optional scheduler.
        
        Args:
            model: nn.Module
            train_loader: DataLoader for training
            val_loader: DataLoader for validation
            num_epochs: maximum number of epochs
            device: 'cuda' or 'cpu' (default autodetect)
            lr: learning rate
            print_every: iterations between logging loss
            patience: early stopping patience
            accumulation_steps: steps to accumulate gradients for effective larger batch size
            scheduler: optional LR scheduler
        Returns:
            best_model: model with highest validation accuracy
        """
        device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # model.to(device)
        # device_type = 'cuda' if device.type == 'cuda' else 'cpu'

        criterion = criterion or F.cross_entropy
        optimizer = optimizer or torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scaler = GradScaler()

        best_val_acc = -float('inf')
        best_model = None
        epochs_no_improve = 0

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0

            for i, (x, y) in enumerate(train_loader):
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad() if accumulation_steps == 1 else None

                with autocast("cuda"):
                    outputs = model(x)
                    loss = criterion(outputs, y) / accumulation_steps  # scale loss for accumulation

                scaler.scale(loss).backward()

                if (i + 1) % accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                
                running_loss += loss.item() * accumulation_steps
                if (i + 1) % print_every == 0:
                    print(f"Epoch {epoch+1}, Iter {i+1}, Avg Loss: {running_loss/print_every:.4f}")
                    running_loss = 0.0

            # Step scheduler at epoch end
            if scheduler is not None:
                scheduler.step()

            # Validation check
            val_acc = get_accuracy(val_loader, model, f"Epoch {epoch+1} Validation", device)

            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model = copy.deepcopy(model)
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping triggered after {epoch+1} epochs.")
                    break

        return best_model if best_model else model


    def get_accuracy(loader, model, name="Validation", device=device, debug=True):
        """
        Compute accuracy of the model on a given DataLoader.
        """
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                with autocast("cuda"):
                    scores = model(x)
                preds = scores.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        acc = correct / total
        if debug:
            print(f"{name} Accuracy: {100*acc:.2f}%")
        return acc

    # _________________________________________________
    # Neural Network

    def objective(trial):
        ch1 = trial.suggest_categorical("ch1", [16, 32, 64])
        ch2 = trial.suggest_categorical("ch2", [32, 64, 128])
        ch3 = trial.suggest_categorical("ch3", [64, 128, 256])
        lr = trial.suggest_loguniform("lr", 1e-4, 1e-2)        
        weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-3)
        
        # Build model
        model = nn.Sequential(
            nn.Conv2d(3, ch1, 5, padding=2),
            nn.BatchNorm2d(ch1),
            nn.ReLU(),

            nn.Conv2d(ch1, ch2, 3, padding=1),
            nn.BatchNorm2d(ch2),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(ch2, ch3, 3, padding=1),
            nn.BatchNorm2d(ch3),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Flatten(),
            nn.Linear(ch3 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 10)
        ).to(device)
                
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

        trained_model = train_model(
            model=model,
            train_loader=loader_train,
            val_loader=loader_val,
            num_epochs=NUM_EPOCHS,
            lr=lr,
            device=device,
            print_every=50,
            patience=2,
            scheduler=scheduler
        )
        
        val_acc = get_accuracy(loader_val, trained_model, device=device, debug=False)

        # print(f"Trial {trial+1}: ch1={ch1}, ch2={ch2}, lr={lr}, val_acc={100*val_acc:.2f}%")
        # trial += 1

        # # Track the best model
        # if val_acc > best_accuracy:
        #     best_accuracy = val_acc
        #     best_model = trained_model
        #     best_ch1, best_ch2, best_lr = ch1, ch2, lr
        #     print(f"✅ Found new best model: {100*val_acc:.2f}")
        
        return val_acc
    
    # _________________________________________________
    # Run Bayesian Search

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=NUM_TRIALS) 

    best_trial = study.best_trial
    print("Best hyperparameters:", best_trial.params)
    print(f"Best validation accuracy: {100*best_trial.value:.2f}%")

    # _________________________________________________
    # After the model has been trained, get test score

# Best hyperparameters: {'ch1': 64, 'ch2': 128, 'ch3': 64, 'lr': 0.00036203756027455733, 'weight_decay': 1.4247931835872181e-06}
# Best validation accuracy: 74.80%

    finished = False
    if finished:

        loader_test = DataLoader(
            cifar10_test,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            persistent_workers=True
        )
        print(f"Test batches: {len(loader_test)}")

        loader_test = DataLoader(cifar10_test, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
        get_accuracy(loader_test, best_model, name="Test")