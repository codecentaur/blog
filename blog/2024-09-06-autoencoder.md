---
slug: MNIST-autoencoder
title: MNIST Autoencoder
tags: [autoencoder, MNIST]
---
In this post, I’ll explain how I built an autoencoder to compress and reconstruct images from the MNIST dataset. Autoencoders are a type of neural network that learn to compress data into a smaller latent space and reconstruct the original data from that compressed representation. This post will take you through the key steps in the process of training the autoencoder, explaining the important concepts along the way.
<!--truncate-->


```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision
import numpy as np
import os

# Set seed for reproducibility
seed = 42
torch.manual_seed(seed)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
```

    Using device: cuda


### Data

The MNIST dataset contains 28x28 pixel images of handwritten digits (0-9). Each image has 784(28x28) pixels. In this project, I used separate datasets for training and validation. I didn’t hold out a testing set, as this was more of an experiment to understand the autoencoder's functionality.


```python
transform = transforms.Compose([
    transforms.ToTensor(),  # Scales pixel values between 0 and 1
])

# Download MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
valid_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

# Load data
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=64, shuffle=False)
```

## Model Architecture

 I experimented with a few different architectures, but I ended up with an autoencoder that compresses the input image (784 pixels) down to a 6-dimensional latent space. The architecture consists of an encoder, which progressively reduces the data through several layers, and a decoder, which mirrors the encoder to reconstruct the image.

Here’s a visual representation of the architecture:

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 320">
  <!-- Input Layer -->
  <rect x="10" y="100" width="180" height="100" fill="#1e3a8a" stroke="#ffffff" />
  <text x="100" y="155" text-anchor="middle" font-size="14" fill="#ffffff">Input Image (784)</text>
  
  <!-- Encoder Layers -->
  <rect x="210" y="112" width="80" height="76" fill="#065f46" stroke="#ffffff" />
  <text x="250" y="155" text-anchor="middle" font-size="14" fill="#ffffff">128</text>
  
  <rect x="310" y="120" width="60" height="60" fill="#065f46" stroke="#ffffff" />
  <text x="340" y="155" text-anchor="middle" font-size="14" fill="#ffffff">64</text>
  
  <rect x="390" y="134" width="40" height="32" fill="#065f46" stroke="#ffffff" />
  <text x="410" y="155" text-anchor="middle" font-size="14" fill="#ffffff">12</text>
  
  <!-- Latent Space -->
  <rect x="450" y="143" width="30" height="14" fill="#854d0e" stroke="#ffffff" />
  <text x="465" y="205" text-anchor="middle" font-size="16" fill="#ffffff">Latent (6)</text>
  
  <!-- Decoder Layers -->
  <rect x="500" y="134" width="40" height="32" fill="#9f1239" stroke="#ffffff" />
  <text x="520" y="155" text-anchor="middle" font-size="14" fill="#ffffff">12</text>
  
  <rect x="560" y="120" width="60" height="60" fill="#9f1239" stroke="#ffffff" />
  <text x="590" y="155" text-anchor="middle" font-size="14" fill="#ffffff">64</text>
  
  <rect x="640" y="112" width="80" height="76" fill="#9f1239" stroke="#ffffff" />
  <text x="680" y="155" text-anchor="middle" font-size="14" fill="#ffffff">128</text>
  
  <!-- Output Layer -->
  <rect x="740" y="100" width="180" height="100" fill="#1e3a8a" stroke="#ffffff" />
  <text x="830" y="155" text-anchor="middle" font-size="14" fill="#ffffff">Output Image (784)</text>
  
  <!-- Connecting Lines -->
  <line x1="190" y1="150" x2="210" y2="150" stroke="#ffffff" stroke-width="2" />
  <line x1="290" y1="150" x2="310" y2="150" stroke="#ffffff" stroke-width="2" />
  <line x1="370" y1="150" x2="390" y2="150" stroke="#ffffff" stroke-width="2" />
  <line x1="430" y1="150" x2="450" y2="150" stroke="#ffffff" stroke-width="2" />
  <line x1="480" y1="150" x2="500" y2="150" stroke="#ffffff" stroke-width="2" />
  <line x1="540" y1="150" x2="560" y2="150" stroke="#ffffff" stroke-width="2" />
  <line x1="620" y1="150" x2="640" y2="150" stroke="#ffffff" stroke-width="2" />
  <line x1="720" y1="150" x2="740" y2="150" stroke="#ffffff" stroke-width="2" />
  
  <!-- Labels -->
  <text x="300" y="230" text-anchor="middle" font-size="16" fill="#ffffff">Encoder</text>
  <text x="630" y="230" text-anchor="middle" font-size="16" fill="#ffffff">Decoder</text>
</svg>


```python
# Define the model
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 12),
            nn.ReLU(),
            nn.Linear(12, 6)  # Latent representation
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(6, 12),
            nn.ReLU(),
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28*28),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def encode(self, x):
        return self.encoder(x)

```

One variation I tried was scaling the input data between [-1, 1] instead of [0, 1]. I also changed the output activation function from Sigmoid to hyperbolic tangent (tanh). Sigmoid maps outputs to the [0, 1] range, while tanh maps them to [-1, 1]. Having the input and output ranges match helped the model learn faster. I'll plot the two activation functions below to show the difference.

In the end, I didn’t notice any change in performance, so I reverted to scaling the data between [0, 1], which is more common.


```python
def plot_sigmoid_tanh():
    # range of x values
    x = np.linspace(-6, 6, 100)

    # functions to plot
    sigmoid = 1 / (1 + np.exp(-x))
    tanh = np.tanh(x)

    plt.figure(figsize=(8, 5))
    plt.plot(x, sigmoid, label='Sigmoid', color='blue')
    plt.plot(x, tanh, label='Tanh', color='red')
    plt.title('Sigmoid vs Tanh Activation Functions')
    plt.xlabel('Input')
    plt.ylabel('Activation')
    plt.grid(True)
    plt.legend()
    plt.show()

# Plot the Sigmoid and Tanh functions
plot_sigmoid_tanh()
```


    
![png](2024-09-06/03a%20Autoencoder-linear_files/03a%20Autoencoder-linear_7_0.png)
    


### Training the Model

I used Mean Squared Error (MSE) as the loss function and the Adam optimizer. The model was trained for 50 epochs.


```python
def train(model, dataloader, device):
    model.train()
    total_loss = 0
    for images, _ in dataloader:
        images = images.view(images.size(0), -1).to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, images)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    average_loss = total_loss / len(dataloader)
    return average_loss

def validate(model, dataloader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.view(images.size(0), -1).to(device)
            outputs = model(images)
            loss = criterion(outputs, images)
            total_loss += loss.item()
    average_loss = total_loss / len(dataloader)
    return average_loss


# Create an instance of the model and move it to the device
autoencoder = Autoencoder().to(device)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.0005)

# Training Loop
train_losses = []
valid_losses = []
num_epochs = 50
for epoch in range(num_epochs):
    train_loss = train(autoencoder, train_loader, device)
    valid_loss = validate(autoencoder, valid_loader, device)
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
```

### Training and validation loss

Here is a plot of the training and validation loss. It shows that the model is learning and not overfitting.


```python
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(valid_losses, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
```


    
![png](2024-09-06/03a%20Autoencoder-linear_files/03a%20Autoencoder-linear_11_0.png)
    


Zooming in on the last 15 epocs we can see that training and validation are still decreasing although the rate of decrease has slowed down.


```python
plt.figure(figsize=(10, 5))
plt.plot(train_losses[-15:], label='Training Loss')
plt.plot(valid_losses[-15:], label='Validation Loss')
plt.title('Training and Validation Loss - Last 15 Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
```


    
![png](2024-09-06/03a%20Autoencoder-linear_files/03a%20Autoencoder-linear_13_0.png)
    


### Results

After training, I tested the autoencoder by passing images from the validation set and comparing the original and reconstructed images. Here’s a visual comparison.


```python
def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap='gray')
    plt.show()

# Display original images
images, labels = next(iter(valid_loader))
images = images.to(device)
print("Original Images")
imshow(torchvision.utils.make_grid(images[:5].cpu()))

# Display reconstructed images
images_flattened = images.view(images.size(0), -1)
outputs = autoencoder(images_flattened)
outputs = outputs.view(outputs.size(0), 1, 28, 28) # Reshape the outputs
print("Reconstructed Images")
imshow(torchvision.utils.make_grid(outputs[:5].cpu()))
```

    Original Images



    
![png](2024-09-06/03a%20Autoencoder-linear_files/03a%20Autoencoder-linear_15_1.png)
    


    Reconstructed Images



    
![png](2024-09-06/03a%20Autoencoder-linear_files/03a%20Autoencoder-linear_15_3.png)
    


### Saving to disk and restoring

This isn't lossless compression, however this is really good results given that the latent space representation is only six numbers!

In the code below I use the encoder and decoder separately to save and load the latent space representation to confirm everything is working as expected.

This was interesting and took a couple tries. At first I was just serializing the tensor to disk, however the tensor has metadata in it that inflated the size to around 2,000 bytes. I was able to remove the metadata below by detaching and converting to a numpy array.


```python
# Create a latent representation of the first image in the validation set
images, _ = next(iter(valid_loader))
images = images.view(images.size(0), -1).to(device)
latent_representation = autoencoder.encode(images[:1])

# Convert to numpy array to avoid saving tensor metadata to disk
latent_numpy = latent_representation.cpu().detach().numpy()

# Save to disk using a binary format
latent_numpy.tofile("latent_representation.bin")

# Print the size of the saved file
file_size = os.path.getsize("latent_representation.bin")
print(f"Size of the latent representations on disk: {file_size} bytes")
```

    Size of the latent representations on disk: 24 bytes


The compressed image ends up being twenty four bytes. That is our expected size because the latent representation is six floats at four bytes each.

The original input was 784(28x28) numbers. Each of those numbers would fit into a byte because they were in the range 0-255. That means that we got a 97% size reduction!

Let's reconstruct the saved image to confirm that the autoencoder is working as expected.


```python
# Display the original image
original_image = images[0].view(1, 28, 28).cpu()
print("Original Image")
imshow(torchvision.utils.make_grid(original_image))

# Load from disk
latent_size = 6
latent_representation = np.fromfile("latent_representation.bin", dtype=np.float32, count=latent_size)
latent_representation = torch.from_numpy(latent_representation).to(device).unsqueeze(0)

# Use the decoder to reconstruct the image
reconstructed = autoencoder.decoder(latent_representation)

# Display the reconstructed image
reconstructed_image = reconstructed.view(1, 28, 28).cpu()
print("Reconstructed Image")
imshow(torchvision.utils.make_grid(reconstructed_image))
```

    Original Image



    
![png](2024-09-06/03a%20Autoencoder-linear_files/03a%20Autoencoder-linear_20_1.png)
    


    Reconstructed Image



    
![png](2024-09-06/03a%20Autoencoder-linear_files/03a%20Autoencoder-linear_20_3.png)
    


### Conclusion

In this project, I built an autoencoder to compress and reconstruct MNIST images. The autoencoder achieved about 97% compression by reducing the input from 784 pixels to a 6-number latent space. This type of model has potential applications in data compression, anomaly detection, and data denoising. While this was a simple experiment, autoencoders are a powerful tool in many areas of machine learning.
