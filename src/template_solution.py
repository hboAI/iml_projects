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
    model = Model().to(device)
    
    # Define loss functions
    criterion = nn.SmoothL1Loss(beta=1.0)  # Huber loss
    mse_criterion = nn.MSELoss()  # For tracking MSE
    
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
    for epoch in range(20):
        model.train()
        train_mse = 0.0
        
        # Training phase
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Calculate MSE for tracking
            with torch.no_grad():
                train_mse += mse_criterion(outputs, targets).item() * inputs.size(0)
        
        # Calculate average training MSE
        train_mse /= len(train_dataset)
        
        # Validation phase
        model.eval()
        val_mse = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                outputs = model(inputs)
                val_mse += mse_criterion(outputs, targets).item() * inputs.size(0)
        
        val_mse /= len(val_dataset)
        
        # Update learning rate scheduler
        scheduler.step(val_mse)
        
        # Print progress
        print(f"Epoch [{epoch+1}/20] "
              f"Train MSE: {train_mse:.4f} - "
              f"Val MSE: {val_mse:.4f}")
    
    return model


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
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Spatial attention
        self.spatial_attention = SpatialAttention(kernel_size=7)
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            
            nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)

        # Apply spatial attention
        attention = self.spatial_attention(x)
        x = x * attention

        x = self.decoder(x)
        return x

# Example usage:
# model = Model()
# optimizer = optim.Adam(model.parameters(), lr=0.1)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
# criterion = nn.MSELoss()

def test_model(model, test_data_input):
    """
    Uses your model to predict the ouputs for the test data. Saves the outputs
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
        # TODO: You can increase or decrease this batch size depending on your
        # memory requirements of your computer / model
        # This will not affect the performance of the model and your score
        batch_size = 64
        for i in tqdm(
            range(0, test_data_input.shape[0], batch_size),
            desc="Predicting test output",
        ):
            output = model(test_data_input[i : i + batch_size])
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

    # You can plot the output if you want
    # Set to False if you don't want to save the images
    if True:
        # Create the output directory if it doesn't exist
        if not Path("test_image_output").exists():
            Path("test_image_output").mkdir()
        for i in tqdm(range(20), desc="Plotting test images"):
            # Show the training and the target image side by side
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

