from torchvision.datasets import OxfordIIITPet
from torchvision import transforms
from torch.utils.data import random_split, DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
import torch

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu" )
    if torch.cuda.is_available() == True:
        print("using gpu")
    else:
        print("using cpu")
    dataset = OxfordIIITPet(
        r"C:\Users\mega\Desktop\pet classification",
        "trainval",
        "category",
        download=True,
        transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))
        ])
    )
    print("Dataset length:", len(dataset))

    generator1 = torch.Generator().manual_seed(42)
    train_split, val_split = random_split(dataset, [3128, 552], generator=generator1)

    
    num_workers = 2

    train_loader = DataLoader(train_split, batch_size=32, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_split, batch_size=32, shuffle=False,
                            num_workers=num_workers, pin_memory=True, drop_last=False)

    # Quick verification
    train_features, train_labels = next(iter(train_loader))
    print(f"Train batch: X={train_features.size()} y={train_labels.size()}")
    val_features, val_labels = next(iter(val_loader))
    print(f"Val batch:   X={val_features.size()} y={val_labels.size()}")

    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Linear(in_features=512, out_features=37)
    model = model.to(device)

    # Freeze backbone, train only fc
    for p in model.parameters():
        p.requires_grad = False
    for p in model.fc.parameters():
        p.requires_grad = True

    criterion = CrossEntropyLoss()
    optimizer = AdamW(model.fc.parameters(), lr=1e-3)

    n_epochs = 12
    best_val_acc=0
    for epoch in range(1, n_epochs + 1):
        # -------- Train --------
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(images)  # logits (B, 37)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            bs = labels.size(0)
            train_loss_sum += loss.item() * bs
            preds = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += bs

        train_loss_avg = train_loss_sum / train_total
        train_acc = train_correct / train_total

        # -------- Validate --------
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                outputs = model(images)
                loss = criterion(outputs, labels)

                bs = labels.size(0)
                val_loss_sum += loss.item() * bs
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += bs

        val_loss_avg = val_loss_sum / val_total
        val_acc = val_correct / val_total
        
        if val_acc>best_val_acc:
            best_val_acc=val_acc
            torch.save(model.state_dict(),"best_model.pt")

        print(f"Epoch {epoch:02d}/{n_epochs} | "
              f"train_loss={train_loss_avg:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss_avg:.4f} val_acc={val_acc:.4f}")
    
if __name__ == "__main__":
    main()
