from torchvision.datasets import OxfordIIITPet
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
import torch

def main():
    device=torch.device("cuda")
    dataset=OxfordIIITPet(
        r"C:\Users\mega\Desktop\pet classification",
        "test",
        "category",
        transform=transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                    std=(0.229, 0.224, 0.225))
                                    ]),download=False)    
                
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Linear(in_features=512, out_features=37)

    model.load_state_dict(torch.load("best_model.pt", map_location=device))
    model = model.to(device)
    model.eval()

    test_loader = DataLoader(dataset, batch_size=32, shuffle=False,
                                num_workers=2, pin_memory=True, drop_last=False)
    criterion = CrossEntropyLoss()
    test_loss_sum = 0.0
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for images,labels in test_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs=model(images)
            loss = criterion(outputs, labels)
            bs = labels.size(0)
            test_loss_sum += loss.item() * bs

            preds = outputs.argmax(dim=1)           # (B,)
            test_correct += (preds == labels).sum().item()
            test_total += bs

    test_loss_avg = test_loss_sum / test_total
    test_acc = test_correct / test_total

    print(f"Test loss={test_loss_avg:.4f} | Test acc={test_acc:.4f} | N={test_total}")
if __name__ == "__main__":
    main()