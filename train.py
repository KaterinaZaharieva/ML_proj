import os
import cv2
import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from tqdm import tqdm
from dataset import CityscapesDataset


def main():
    DATASET_DIR = "datasets/cityscapes"

    train_img_root = os.path.join(DATASET_DIR, "leftImg8bit/train")
    train_mask_root = os.path.join(DATASET_DIR, "gtFine/train")

    image_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    mask_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 512), interpolation=0)
    ])

    train_dataset = CityscapesDataset(train_img_root, train_mask_root,
                                      image_transform, mask_transform)

    print("Total training images:", len(train_dataset))

    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size

    train_dataset, val_dataset = random_split(train_dataset,
                                              [train_size, val_size])

    train_loader = DataLoader(train_dataset,
                              batch_size=4,
                              shuffle=True,
                              num_workers=0,
                              pin_memory=True)

    val_loader = DataLoader(val_dataset,
                            batch_size=4,
                            shuffle=False,
                            num_workers=0,
                            pin_memory=True)

    print("Train:", len(train_dataset))
    print("Val:", len(val_dataset))

    # DataLoaders
    train_loader = DataLoader(train_dataset,
                              batch_size=4,
                              shuffle=True,
                              num_workers=0,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=4,
                            shuffle=False,
                            num_workers=0,
                            pin_memory=True)

    print(
        f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.segmentation.deeplabv3_resnet50(pretrained=True)
    model.classifier[4] = nn.Conv2d(256, 19, kernel_size=1)

    model.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler('cuda')  # mixed precision
    num_epochs = 2

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch in loop:
            inputs = batch['image'].to(device)
            masks = batch['mask'].to(device)

            optimizer.zero_grad()

            outputs = model(inputs)['out']
            loss = criterion(outputs, masks)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(train_loader)
        print(f"Train Loss: {epoch_loss:.4f}")

    torch.save(model.state_dict(), "deeplabv3_tenep.pt")
    print("Model saved.")


if __name__ == '__main__':
    main()
