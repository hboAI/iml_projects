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
    where N_train_samples is the number of training samples, N_test_samples is
    the number of test samples, C is the number of channels (1 for grayscale),
    H is the height of the image, and W is the width of the image.
    """
    train_data = np.load("train_data.npz")["data"]  # Shape (N, 28, 28)
    
    # Convert to tensor with channel dimension
    train_data = torch.tensor(train_data, dtype=torch.float32)  # Now (N, 1, 28, 28)
    
    # Load test data
    test_data_input = np.load("test_data.npz")["data"]
    test_data_input = torch.tensor(test_data_input, dtype=torch.float32)
    
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
    batch_size = 10
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model and move to device
    model = kwargs.get('model', Model()).to(device)  # Allow custom model via kwargs
    
    # Define loss function
    mse_criterion = nn.MSELoss(reduction='sum')  # Sum for masked loss
    
    # Set up optimizer and scheduler
    optimizer = optim.SGD(model.parameters(), 
                         lr=0.1, 
                         momentum=0.9, 
                         weight_decay=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                    mode='min', 
                                                    factor=0.1,
                                                    patience=5,
                                                    verbose=True)
    
    # Training loop
    for epoch in range(10):
        model.train()
        train_mse = 0.0
        
        # Training phase
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Generate mask (masked pixels are 0)
            mask = (inputs == 0).float().to(device)  # Shape: [batch_size, 1, 28, 28]
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs, mask)  # Model takes inputs and mask
            
            # Compute loss on masked regions
            loss = mse_criterion(outputs * mask, targets * mask) / mask.sum()
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Calculate MSE for tracking
            with torch.no_grad():
                train_mse += mse_criterion(outputs * mask, targets * mask).item() / mask.sum().item() * inputs.size(0)
        
        # Calculate average training MSE
        train_mse /= len(train_dataset)
        
        # Validation phase
        model.eval()
        val_mse = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                # Generate mask
                mask = (inputs == 0).float().to(device)  # Shape: [batch_size, 1, 28, 28]
                
                # Forward pass
                outputs = model(inputs, mask)
                
                # Calculate MSE on masked regions
                val_mse += mse_criterion(outputs * mask, targets * mask).item() / mask.sum().item() * inputs.size(0)
        
        val_mse /= len(val_dataset)
        
        # Update learning rate scheduler
        scheduler.step(val_mse)
        
        # Print progress
        print(f"Epoch [{epoch+1}/10] "
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

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
        # Encoder
        self.conv1_pre_pool = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),  # Input: [image, mask]
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Decoder
        self.decoder_up1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU()
        )
        
        # Spatial attention for first skip connection
        self.attention1 = SpatialAttention(kernel_size=7)
        
        # Fusion convolution for first skip connection (14x14)
        self.fusion_conv1 = nn.Conv2d(32 + 32, 32, kernel_size=3, padding=1)
        self.fusion_relu1 = nn.ReLU()
        
        self.decoder_up2 = nn.Sequential(
            nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2)
        )
        
        # Spatial attention for second skip connection
        self.attention2 = SpatialAttention(kernel_size=7)
        
        # Fusion convolution for second skip connection (28x28)
        self.fusion_conv2 = nn.Conv2d(32 + 1, 1, kernel_size=3, padding=1)
        self.final_activation = nn.Tanh()  # Residual in [-1, 1]

    def forward(self, x_corr, mask):
        # Input: x_corr [N, 1, 28, 28], mask [N, 1, 28, 28]
        # Concatenate corrupted image and mask
        x = torch.cat([x_corr, mask], dim=1)  # Shape: [N, 2, 28, 28]
        
        # Encoder
        f0 = self.conv1_pre_pool(x)  # Shape: [N, 32, 28, 28]
        f1 = self.pool1(f0)  # Shape: [N, 32, 14, 14]
        f2 = self.conv2(f1)  # Shape: [N, 64, 7, 7]
        
        # Decoder
        d1 = self.decoder_up1(f2)  # Shape: [N, 32, 14, 14]
        
        # First skip connection with attention
        attn1 = self.attention1(f1)  # Shape: [N, 1, 14, 14]
        f1 = f1 * attn1  # Apply attention to enhance edges
        d1 = torch.cat([f1, d1], dim=1)  # Shape: [N, 32+32, 14, 14]
        
        # Fuse first skip connection
        d1 = self.fusion_conv1(d1)  # Shape: [N, 32, 14, 14]
        d1 = self.fusion_relu1(d1)
        
        # Second upsampling
        d2 = self.decoder_up2(d1)  # Shape: [N, 1, 28, 28]
        
        # Second skip connection with attention
        attn2 = self.attention2(f0)  # Shape: [N, 1, 28, 28]
        f0 = f0 * attn2  # Apply attention to enhance edges
        d2 = torch.cat([f0, d2], dim=1)  # Shape: [N, 32+1, 28, 28]
        
        # Predict residual
        residual = self.fusion_conv2(d2)  # Shape: [N, 1, 28, 28]
        residual = self.final_activation(residual)  # Residual in [-1, 1]
        
        # Combine with input
        output = x_corr + mask * residual
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
    as a binary file. This file needs to be submitted. This function does not
    need to be modified except for setting the batch_size value. If you choose
    to modify it otherwise, please ensure that the generating and saving of the
    output data is not modified.

    Args:
    - model: torch.nn.Module
    - test_data_input: Tensor
    """
    model.eval()
    model.to(device)

    with torch.no_grad():
        test_data_input = test_data_input.to(device)
        # Predict the output batch-wise to avoid memory issues
        test_data_output = []
        # Batch size for inference
        batch_size = 64
        for i in tqdm(
            range(0, test_data_input.shape[0], batch_size),
            desc="Predicting test output",
        ):
            # Generate mask for the batch
            batch_input = test_data_input[i : i + batch_size]
            mask = (batch_input == 0).float().to(device)  # Shape: [batch_size, 1, 28, 28]
            
            # Forward pass with input and mask
            output = model(batch_input, mask)
            test_data_output.append(output.cpu())
        test_data_output = torch.cat(test_data_output)

    # Ensure the output has the correct shape
    assert test_data_output.shape == test_data_input.shape, (
        f"Expected shape {test_data_input.shape}, but got "
        f"{test_data_output.shape}."
        "Please ensure the output has the correct shape."
        "Without the correct shape, the submission cannot be evaluated and "
        "will hence not be valid."
    )

    # Save the output
    test_data_output = test_data_output.numpy()
    # Ensure all values are in the range [0, 255]
    save_data_clipped = np.clip(test_data_output, 0, 255)
    # Convert to uint8
    save_data_uint8 = save_data_clipped.astype(np.uint8)
    # Loss is only computed on the masked area - so set the rest to 0 to save
    # space
    save_data = np.zeros_like(save_data_uint8)
    save_data[:, :, 10:18, 10:18] = save_data_uint8[:, :, 10:18, 10:18]

    np.savez_compressed(
        "submit_this_test_data_output.npz", data=save_data)

    # Plot the output if desired
    if True:
        # Create the output directory if it doesn't exist
        if not Path("test_image_output").exists():
            Path("test_image_output").mkdir()
        for i in tqdm(range(20), desc="Plotting test images"):
            # Show the test input and output side by side
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

