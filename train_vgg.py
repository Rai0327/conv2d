import torch
import torch.nn as nn
import torchvision as tv
import torchvision.transforms as T
from quantized_vgg import quantized_vgg16
from torch_vgg import torch_vgg16
from tqdm import tqdm
import wandb

torch.backends.cudnn.benchmark = True

DATASETS = [
    "mnist",
    "cifar10",
    "places365",
]

ENTITY = "rahulaiyer"
PROJECT = "conv2d"
DEVICE = torch.device("cuda")

class Trainer:
    def __init__(self, quantize, dataset, epochs, batch_size, log):
        train_tf = T.Compose([
            T.Resize(224),
            T.RandomHorizontalFlip(),
            T.RandomCrop(224, padding=4),
            T.ToTensor(),
            T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]), # ImageNet norm
        ])
        val_tf = T.Compose([
            T.Resize(224),
            T.ToTensor(),
            T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]), # ImageNet norm
        ])

        if dataset == "mnist":
            train_tf = T.Compose([
                T.Grayscale(num_output_channels=3),
                T.Resize(224),
                T.RandomCrop(224, padding=4),
                T.ToTensor(),
                T.Normalize([0.1307]*3, [0.3081]*3),
            ])
            val_tf = T.Compose([
                T.Grayscale(num_output_channels=3),
                T.Resize(224),
                T.ToTensor(),
                T.Normalize([0.1307]*3, [0.3081]*3),
            ])

            self.model = quantized_vgg16(num_classes=10) if quantize else torch_vgg16(num_classes=10)
            self.train_dataset = tv.datasets.MNIST(root="./data", train=True, download=True, transform=train_tf)
            self.val_dataset = tv.datasets.MNIST(root="./data", train=False, download=True, transform=val_tf)
        elif dataset == "cifar10":
            train_tf = T.Compose([
                T.Resize(224),
                T.RandomHorizontalFlip(),
                T.RandomCrop(224, padding=4),
                T.ToTensor(),
                T.Normalize([0.4914,0.4822,0.4465],[0.2470,0.2435,0.2616]),
            ])
            val_tf = T.Compose([
                T.Resize(224),
                T.ToTensor(),
                T.Normalize([0.4914,0.4822,0.4465],[0.2470,0.2435,0.2616]),
            ])

            self.model = quantized_vgg16(num_classes=10) if quantize else torch_vgg16(num_classes=10)
            self.train_dataset = tv.datasets.CIFAR10(root="./data", train=True, download=True, transform=train_tf)
            self.val_dataset = tv.datasets.CIFAR10(root="./data", train=False, download=True, transform=val_tf)
        elif dataset == "places365":
            self.model = quantized_vgg16(num_classes=365) if quantize else torch_vgg16(num_classes=365)
            self.train_dataset = tv.datasets.Places365(root="./data", split="train-standard", download=True, transform=train_tf)
            self.val_dataset = tv.datasets.Places365(root="./data", split="val", download=True, transform=val_tf)
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")
        
        self.dataset = dataset
        self.epochs = epochs

        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        self.val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

        self.model.to(DEVICE)
        # after self.model.to(DEVICE)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4, nesterov=True)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
        self.max_acc = 0.0
        self.min_loss = float('inf')

        self.log = log
        if log:
            NAME = f"{'quantized_' if quantize else ''}vgg_{dataset}"
            wandb.init(entity=ENTITY, project=PROJECT, name=NAME)

    def train(self):
        for epoch in range(self.epochs):
            self.model.train()
            correct = 0
            total = 0
            tot_loss = 0.0
            for images, labels in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs}", unit="batch", leave=False):
                images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                tot_loss += loss.item()
                predicted = outputs.argmax(dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            train_acc = correct / total
            train_loss = tot_loss / len(self.train_loader)

            print(f"Epoch [{epoch+1}/{self.epochs}], Acc: {100 * train_acc:.2f}%, Loss: {train_loss:.4f}")

            val_acc, val_loss = self.validate()

            if self.log:
                wandb.log({
                    "epoch": epoch,
                    "train_accuracy": train_acc,
                    "train_loss": train_loss,
                    "val_accuracy": val_acc,
                    "val_loss": val_loss,
                })

            if val_acc > self.max_acc or val_loss < self.min_loss:
                self.max_acc = max(self.max_acc, val_acc)
                self.min_loss = min(self.min_loss, val_loss)
                # Save the model
                torch.save(self.model.state_dict(), f"vgg_{self.dataset}_epoch_{epoch}.pth")
                print(f"Model saved as vgg_{self.dataset}_epoch_{epoch}.pth")

            self.scheduler.step()

        if self.log:
            wandb.finish()

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        correct = 0
        total = 0
        tot_loss = 0.0
        for images, labels in tqdm(self.val_loader, desc="Validation", unit="batch", leave=False):
            images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            tot_loss += loss.item()
            predicted = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        acc = correct / total

        print(f'Validation Accuracy: {100 * acc:.2f}%, Validation Loss: {tot_loss / len(self.val_loader):.4f}')

        return acc, tot_loss / len(self.val_loader)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a VGG model")
    parser.add_argument("--quantize", action="store_true", help="Use quantized VGG model")
    parser.add_argument("--dataset", type=str, choices=DATASETS, required=True, help="Dataset to use for training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--log", action="store_true", help="Log training with wandb")

    args = parser.parse_args()

    trainer = Trainer(args.quantize, args.dataset, args.epochs, args.batch_size, args.log)
    trainer.train()
