from pathlib import Path
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

"""
README FIRST

The below code is a template for the solution. You can change the code according
to your preferences, but the test_model function has to save the output of your 
model on the test data as it does in this template. This output must be submitted.

Replace the dummy code with your own code in the TODO sections.

We also encourage you to use tensorboard or wandb to log the training process
and the performance of your model. This will help you to debug your model and
to understand how it is performing. But the template does not include this
functionality.
Link for wandb:
https://docs.wandb.ai/quickstart/
Link for tensorboard: 
https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html
"""

# The device is automatically set to GPU if available, otherwise CPU
# If you want to force the device to CPU, you can change the line to
# device = torch.device("cpu")

# If you have a Mac consult the following link:
# https://pytorch.org/docs/stable/notes/mps.html

# It is important that your model and all data are on the same device.
device = torch.device("cuda:0" if torch.cuda.is_available() else "mps")
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_data(**kwargs):
    """
    Get the training and test data. The data files are assumed to be in the
    same directory as this script.

    Args:
    - kwargs: Additional arguments that you might find useful - not necessary

    Returns:
    - train_data_input: Tensor[N_train_samples, C, H, W]
    - train_data_label: Tensor[N_train_samples, C, H, W]
    - test_data_input: Tensor[N_test_samples, C, H, W]
    """
    train_data = np.load("train_data.npz")["data"]  # Shape (N, 28, 28)
    
    # Convert to tensor with channel dimension and normalize to [0, 1]
    train_data = torch.tensor(train_data, dtype=torch.float32) / 255.0  # Shape: (N, 1, 28, 28)
    
    # Load test data
    test_data_input = np.load("test_data.npz")["data"]
    test_data_input = torch.tensor(test_data_input, dtype=torch.float32) / 255.0  # Shape: (N_test, 1, 28, 28)
    
    # Create masked inputs
    def apply_mask(image):
        masked_image = image.clone()
        masked_image[:, 10:18, 10:18] = 0  # Apply mask on spatial dimensions
        return masked_image
    
    # Apply mask while preserving channels
    train_data_input = torch.stack([apply_mask(img) for img in train_data])
    train_data_label = train_data.clone()
    
    return train_data_input, train_data_label, test_data_input



import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split



import numpy as np
import torch

def get_data(**kwargs):
    """
    Get the training and test data. The data files are assumed to be in the
    same directory as this script.

    Args:
    - kwargs: Additional arguments that you might find useful - not necessary

    Returns:
    - train_data_input: Tensor[N_train_samples, C, H, W]
    - train_data_label: Tensor[N_train_samples, C, H, W]
    - test_data_input: Tensor[N_test_samples, C, H, W]
    """
    train_data = np.load("train_data.npz")["data"]  # Shape (N, 28, 28)
    
    # Convert to tensor with channel dimension and normalize to [0, 1]
    train_data = torch.tensor(train_data, dtype=torch.float32) / 255.0  # Shape: (N, 1, 28, 28)
    
    # Load test data
    test_data_input = np.load("test_data.npz")["data"]
    test_data_input = torch.tensor(test_data_input, dtype=torch.float32) / 255.0  # Shape: (N_test, 1, 28, 28)
    
    # Create masked inputs
    def apply_mask(image):
        masked_image = image.clone()
        masked_image[:, 10:18, 10:18] = 0  # Apply mask on spatial dimensions
        return masked_image
    
    # Apply mask while preserving channels
    train_data_input = torch.stack([apply_mask(img) for img in train_data])
    train_data_label = train_data.clone()
    
    return train_data_input, train_data_label, test_data_input

def train_model(train_data_input, train_data_label, **kwargs):
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    
    # Create dataset and split into train/validation
    dataset = TensorDataset(train_data_input, train_data_label)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    batch_size = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model and move to device
    model = kwargs.get('model', Model()).to(device)
    
    # Define loss functions
    huber_criterion = nn.SmoothL1Loss(beta=1.0, reduction='sum')  # Huber loss
    mse_criterion = nn.MSELoss(reduction='sum')  # For tracking MSE
    
    # Set up optimizer and scheduler
    optimizer = optim.SGD(model.parameters(), 
                         lr=0.01, 
                         momentum=0.9, 
                         weight_decay=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                    mode='min', 
                                                    factor=0.1,
                                                    patience=2,
                                                    verbose=True)
    
    # Training loop
    for epoch in range(20):
        model.train()
        train_mse = 0.0
        
        # Training phase
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Generate mask and crop to center 8x8
            mask = (inputs == 0).float().to(device)  # Shape: [batch_size, 1, 28, 28]
            center_mask = mask[:, :, 10:18, 10:18]  # Shape: [batch_size, 1, 8, 8]
            center_targets = targets[:, :, 10:18, 10:18]  # Shape: [batch_size, 1, 8, 8]
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)  # Shape: [batch_size, 1, 8, 8]
            
            # Compute Huber loss on 8x8 region
            #huber_loss = huber_criterion(outputs * center_mask, center_targets * center_mask) / (center_mask.sum() + 1e-8)
            mse_loss = mse_criterion(outputs * center_mask, center_targets * center_mask) / (center_mask.sum() + 1e-8)
            
            # Add gradient loss for edge sharpness
            outputs_squeezed = outputs.squeeze(1)  # Shape: [batch_size, 8, 8]
            targets_squeezed = center_targets.squeeze(1)  # Shape: [batch_size, 8, 8]
            mask_squeezed = center_mask.squeeze(1)  # Shape: [batch_size, 8, 8]
            
            grad_outputs_x = torch.abs(torch.gradient(outputs_squeezed, dim=2)[0])
            grad_outputs_y = torch.abs(torch.gradient(outputs_squeezed, dim=1)[0])
            grad_targets_x = torch.abs(torch.gradient(targets_squeezed, dim=2)[0])
            grad_targets_y = torch.abs(torch.gradient(targets_squeezed, dim=1)[0])
            
            grad_loss = (F.l1_loss(grad_outputs_x * mask_squeezed, grad_targets_x * mask_squeezed) +
                         F.l1_loss(grad_outputs_y * mask_squeezed, grad_targets_y * mask_squeezed)) / (mask_squeezed.sum() + 1e-8)
            
            # Total loss
            loss = mse_loss + 0.4 * grad_loss
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Calculate MSE for tracking
            with torch.no_grad():
                train_mse += mse_criterion(outputs * center_mask, center_targets * center_mask).item() / (center_mask.sum().item() + 1e-8) * inputs.size(0)
        
        # Calculate average training MSE
        train_mse /= len(train_dataset)
        
        # Validation phase
        model.eval()
        val_mse = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                # Generate mask and crop to center 8x8
                mask = (inputs == 0).float().to(device)
                center_mask = mask[:, :, 10:18, 10:18]
                center_targets = targets[:, :, 10:18, 10:18]
                
                # Forward pass
                outputs = model(inputs)
                
                # Calculate MSE on 8x8 region
                val_mse += mse_criterion(outputs * center_mask, center_targets * center_mask).item() / (center_mask.sum().item() + 1e-8) * inputs.size(0)
        
        val_mse /= len(val_dataset)
        
        # Update learning rate scheduler
        scheduler.step(val_mse)
        
        # Print progress
        print(f"Epoch [{epoch+1}/20] "
              f"Train MSE: {train_mse:.4f} - "
              f"Val MSE: {val_mse:.4f}")
    
    return model

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


import torch
import torch.nn as nn

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Downsample to 14x14
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Downsample to 7x7
        )
        
        # Decoder: Focus on 8x8 center
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),  # Output: [N, 1, 7, 7]
            nn.Upsample(size=(8, 8), mode='bilinear', align_corners=False),  # Upsample to 8x8
            nn.Tanh()  # Residual in [-1, 1]
        )

    def forward(self, x):
        # Input: x [N, 1, 28, 28]
        # Save input for center slicing
        input_image = x
        x = self.encoder(x)  # Shape: [N, 64, 7, 7]
        residual = self.decoder(x)  # Shape: [N, 1, 8, 8]
        
        # Combine with input's 8x8 center
        center_input = input_image[:, :, 10:18, 10:18]  # Shape: [N, 1, 8, 8]
        output = center_input + residual
        output = torch.clamp(output, 0, 1)  # Ensure output in [0, 1]
        
        return output



from pathlib import Path
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch

def test_model(model, test_data_input):
    """
    Uses your model to predict the outputs for the test data. Saves the outputs
    as a binary file. This file needs to be submitted.

    Args:
    - model: torch.nn.Module
    - test_data_input: Tensor
    """
    model.eval()
    model.to(device)

    with torch.no_grad():
        test_data_input = test_data_input.to(device)
        # Predict the output batch-wise
        test_data_output = torch.zeros_like(test_data_input)  # Shape: [N, 1, 28, 28]
        test_data_output[:, :, 10:18, 10:18] = 0  # Initialize center to 0
        batch_size = 64
        for i in tqdm(
            range(0, test_data_input.shape[0], batch_size),
            desc="Predicting test output",
        ):
            batch_input = test_data_input[i : i + batch_size]
            
            # Forward pass: Predict 8x8 residual
            output = model(batch_input)  # Shape: [batch_size, 1, 8, 8]
            
            # Place 8x8 output in the center
            test_data_output[i : i + batch_size, :, 10:18, 10:18] = output
        
        # Copy unmasked regions from input
        mask = (test_data_input == 0).float()
        test_data_output = test_data_input * (1 - mask) + test_data_output * mask

    # Ensure the output has the correct shape
    assert test_data_output.shape == test_data_input.shape, (
        f"Expected shape {test_data_input.shape}, but got "
        f"{test_data_output.shape}."
    )

    # Save the output
    test_data_output = test_data_output.cpu().numpy()
    save_data_clipped = np.clip(test_data_output * 255, 0, 255)  # Scale to [0, 255]
    save_data_uint8 = save_data_clipped.astype(np.uint8)
    save_data = np.zeros_like(save_data_uint8)
    save_data[:, :, 10:18, 10:18] = save_data_uint8[:, :, 10:18, 10:18]

    np.savez_compressed(
        "submit_this_test_data_output.npz", data=save_data)

    # Plot the output
    if True:
        if not Path("test_image_output").exists():
            Path("test_image_output").mkdir()
        for i in tqdm(range(20), desc="Plotting test images"):
            plt.subplot(1, 2, 1)
            plt.title("Test Input")
            plt.imshow(test_data_input[i].squeeze().cpu().numpy(), cmap="gray")
            plt.subplot(1, 2, 2)
            plt.imshow(test_data_output[i].squeeze(), cmap="gray")
            plt.title("Test Output")

            plt.savefig(f"test_image_output/image_{i}.png")
            plt.close()

def main():
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Load data
    try:
        train_data_input, train_data_label, test_data_input = get_data()
    except FileNotFoundError:
        print("Error: Could not find data files. Please ensure:")
        print("- train_data.npz and test_data.npz exist in the current directory")
        print("- The files are downloaded from the course resources")
        return

    model = train_model(train_data_input, train_data_label)
    test_model(model, test_data_input)

if __name__ == "__main__":
    main()