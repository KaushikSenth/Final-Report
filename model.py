import torch
import torch.optim as optim
from tqdm import tqdm
from torchvision.models import vgg16, VGG16_Weights
from net import SurroundNet, SurroundNetLoss  # Ensure the correct import path
from data_load import train_loader  # Ensure correct import path

device = torch.device("mps")
weights_path = "/Users/kaushiksenthilkumar/.cache/torch/hub/checkpoints/vgg16-397923af.pth"
vgg = vgg16()
vgg.load_state_dict(torch.load(weights_path))
vgg = vgg.to(device)

model = SurroundNet().to(device)
criterion = SurroundNetLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 10
accumulation_steps = 2
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for i, (low_light_img, normal_light_img) in enumerate(tqdm(train_loader)):
        low_light_img = low_light_img.to(device, non_blocking=True)
        normal_light_img = normal_light_img.to(device, non_blocking=True)

        if (i % accumulation_steps == 0):
            optimizer.zero_grad()


        enhanced_img = model(low_light_img)
        loss = criterion(enhanced_img, normal_light_img, enhanced_img, low_light_img)
        loss = loss / accumulation_steps

        loss.backward()

        if (i + 1) % accumulation_steps == 0:
            optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

    scheduler.step()

torch.save(model.state_dict(), "surroundnet_lol.pth")
print("Model saved as 'surroundnet_lol.pth'.")

