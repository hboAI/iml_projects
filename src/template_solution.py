import math
from pathlib import Path
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import keras

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
    train_data = torch.tensor(train_data, dtype=torch.float32) / 255.0 
    
    # Load test data
    test_data_input = np.load("test_data.npz")["data"]
    test_data_input = torch.tensor(test_data_input, dtype=torch.float32) / 255.0
    
    # Create masked inputs
    def apply_mask(image):
        masked_image = image.clone()
        masked_image[:, 10:18, 10:18] = 0  # Apply mask on spatial dimensions
        return masked_image
    
    # Apply mask while preserving channels
    train_data_input = torch.stack([apply_mask(img) for img in train_data])
    train_data_label = train_data.clone()
    
    return train_data_input, train_data_label, test_data_input


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1)
        )
    
    def forward(self, x):
        return x + self.conv(x) 

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
        # Enhanced Encoder
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            ResBlock(32),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 14x14
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            ResBlock(64),
            nn.MaxPool2d(kernel_size=2, stride=2) 
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, dilation=2, padding=2),  
            nn.MaxPool2d(kernel_size=2, stride=2)  # 3x3
        )
        
        # Progressive Decoder
        self.decoder_up2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2),  
            nn.ReLU()
        )
        
        self.decoder_up1 = nn.Sequential(
            nn.ConvTranspose2d(128, 32, kernel_size=2, stride=2), 
            nn.ReLU()
        )
        
        # attention
        self.attention = nn.Sequential(
            nn.Conv2d(32+32+1, 32, 1), 
            nn.Sigmoid()
        )
        
        # Final Decoder
        self.decoder_final = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            ResBlock(32),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Tanh()
        )
        
        self.crop = lambda x: x[:, :, 3:11, 3:11] 

    def forward(self, x):
        # Encoder
        e0 = x  # Original input
        f1 = self.conv1(x)  # 32@14x14
        f2 = self.conv2(f1)  # 64@7x7
        f3 = self.conv3(f2)  # 128@3x3
        
        # Decoder with multi-scale skips
        d2 = self.decoder_up2(f3)  # 64@7x7
        d2 = torch.cat([d2, f2], dim=1)  # 64+64=128@7x7
        
        d1 = self.decoder_up1(d2)  # 32@14x14
        e0_down = F.avg_pool2d(e0, 2)  # 1@14x14
        
        # Attention-guided fusion
        attn = self.attention(torch.cat([d1, f1, e0_down], dim=1))
        d1 = d1 * attn  # Weighted features
        
        # Final reconstruction
        residual = self.decoder_final(d1)
        residual = self.crop(residual)
        
        # Combine with input
        center_input = x[:, :, 10:18, 10:18]
        output = center_input + residual
        return torch.clamp(output, 0, 1)



#other schedulers were ass. found this in a kaggle repo for mnist
class OneCycleScheduler:
    """
    Implements 1Cycle Learning Rate and Momentum (beta1 for Adam) scheduling.
    - LR increases from lr_min to lr_max, then decreases to lr_min/100.
    - Beta1 decreases from beta1_max to beta1_min, then increases back.
    """
    def __init__(self, optimizer, lr_max, beta1_max, beta1_min, total_steps, div_factor=10, final_div_factor=100):
        self.optimizer = optimizer
        self.lr_max = lr_max
        self.lr_min = lr_max / div_factor
        self.final_lr = lr_max / final_div_factor
        self.beta1_max = beta1_max
        self.beta1_min = beta1_min
        self.total_steps = total_steps
        self.half_steps = total_steps // 2
        self.step_count = 0
        
    def step(self):
        self.step_count += 1
        # Compute interpolation factor (0 to 1 for first half, 1 to 0 for second half)
        if self.step_count <= self.half_steps:
            # Phase 1: Increase LR, decrease beta1
            frac = self.step_count / self.half_steps
            lr = self.lr_min + frac * (self.lr_max - self.lr_min)
            beta1 = self.beta1_max - frac * (self.beta1_max - self.beta1_min)
        else:
            # Phase 2: Decrease LR, increase beta1
            frac = (self.step_count - self.half_steps) / self.half_steps
            lr = self.lr_max - frac * (self.lr_max - self.final_lr)
            beta1 = self.beta1_min + frac * (self.beta1_max - self.beta1_min)
        

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            param_group['betas'] = (beta1, param_group['betas'][1])  # Update beta1, keep beta2

def train_model(train_data_input, train_data_label, **kwargs):

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    

    dataset = TensorDataset(train_data_input, train_data_label)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model and move to device
    model = kwargs.get('model', Model()).to(device)
    
    # Define loss functions
    #want to superposition loss functions
    huber_criterion = nn.SmoothL1Loss(beta=1.0, reduction='sum')  # Huber loss
    mse_criterion = nn.MSELoss(reduction='sum')  # For tracking MSE
    
    # maybe sgd also works
    optimizer = optim.Adam(
        model.parameters(),
        lr=0.0001,  # Initial LR, will be overridden by scheduler
        betas=(0.9, 0.999),  # Initial (beta1, beta2), beta1 will be overridden
        weight_decay=0.0001
    )
    

    total_epochs = 50
    steps_per_epoch = len(train_loader)
    total_steps = total_epochs * steps_per_epoch
    scheduler = OneCycleScheduler(
        optimizer,
        lr_max=3e-4,
        beta1_max=0.95,
        beta1_min=0.8,  # Wider momentum swing
        total_steps=total_epochs * len(train_loader),  # Ensure this matches
        div_factor=10,         # lr_min = lr_max / div_factor
        final_div_factor=100   # final_lr = lr_max / final_div_factor
    )
    
    # Training loop
    for epoch in range(total_epochs):
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
            huber_loss = huber_criterion(outputs * center_mask, center_targets * center_mask) / (center_mask.sum() + 1e-8)
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
            #loss = huber_loss + 0.1 * grad_loss
            loss = 0.7*huber_loss + 0.3*mse_loss + 0.1*grad_loss  # Adjusted weights
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update scheduler
            scheduler.step()
            
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
        
        # Print progress with current learning rate and beta1
        current_lr = optimizer.param_groups[0]['lr']
        current_beta1 = optimizer.param_groups[0]['betas'][0]
        print(f"Epoch [{epoch+1}/{total_epochs}] "
              f"Train MSE: {train_mse:.4f} - "
              f"Val MSE: {val_mse:.4f} - "
              f"LR: {current_lr:.6f} - "
              f"Beta1: {current_beta1:.4f}")
    
    return model


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

