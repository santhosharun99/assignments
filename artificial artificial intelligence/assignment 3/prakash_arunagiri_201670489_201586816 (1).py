# Import required libraries

import os                    # Library to interact with the file system
import numpy as np           # Library for numerical calculations
import matplotlib.pyplot as plt  # Library for data visualization
import albumentations as A   # Library for augmentations
from sklearn.model_selection import train_test_split # Library for data splitting
import torch                 # Library for deep learning
import torch.nn as nn        # Library for neural network building
import torch.optim as optim  # Library for optimization algorithms
from torch.utils.data import Dataset, DataLoader  # Library for handling datasets and dataloaders
from albumentations.pytorch.transforms import ToTensorV2 # Library for data transformations
from torchvision import transforms  # Library for computer vision transforms
import cv2                   # Library for computer vision tasks
from PIL import Image        # Library for handling images
import torch.nn.functional as F   # Library for various functional interfaces of PyTorch
from sklearn.model_selection import ParameterGrid # Library for generating a grid of hyperparameters
import torchvision.models as models # Library for pretrained models

# Define the path to dataset
data_path = "Cam101"

# Load the training dataset
train_images = []
train_labels = []
for file in os.listdir(os.path.join(data_path, 'train')):
    if '_L.png' in file:
        continue
    img = np.array(Image.open(os.path.join(data_path, 'train', file)).convert('RGB')) # Convert to RGB
    label = np.array(Image.open(os.path.join(data_path, 'train', file[:-4]+'_L.png')).convert('RGB'))
    train_images.append(img)
    train_labels.append(label)

# Load the test dataset
test_images = []
test_labels = []
for file in os.listdir(os.path.join(data_path, 'test')):
    if '_L.png' in file:
        continue
    img = np.array(Image.open(os.path.join(data_path, 'test', file)).convert('RGB')) # Convert to RGB
    label = np.array(Image.open(os.path.join(data_path, 'test', file[:-4]+'_L.png')).convert('RGB'))
    test_images.append(img)
    test_labels.append(label)

# Threshold the labels to convert them into binary masks
label = cv2.threshold(label, 0, 1, cv2.THRESH_BINARY)[1]
train_labels.append(torch.tensor(label).long())

label = cv2.threshold(label, 0, 1, cv2.THRESH_BINARY)[1]
test_labels.append(torch.tensor(label).long())
# Print number of training and testing samples
print(f"Number of training samples: {len(train_images)}")
print(f"Number of testing samples: {len(test_images)}")

# Calculate mean of training images
mean = np.mean(train_images, axis=(0, 1))  # Calculate the mean value of the training images along the (0, 1) axis

# Visualize some samples
plt.figure(figsize=(10, 10))
for i in range(4):
    plt.subplot(2, 4, i+1)  # Create a subplot in the first row and plot the original image
    plt.imshow(train_images[i])
    plt.title('Original')
    plt.axis('off')
    plt.subplot(2, 4, i+5)  # Create a subplot in the second row and plot the label image
    plt.imshow(train_labels[i])
    plt.title('Label')
    plt.axis('off')
plt.show()

# Define the data transformation pipeline
transform = A.Compose([
    A.RandomCrop(width=256, height=256),
    A.Resize(width=128, height=128),
    A.HorizontalFlip(p=0.5),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), max_pixel_value=255.0),
    ToTensorV2(),
])

# Define the mean intersection over union function
def mean_iou(preds, targets):
    intersection = (preds & targets).float().sum((1, 2))  # Calculate the intersection of predicted and target masks
    union = (preds | targets).float().sum((1, 2))  # Calculate the union of predicted and target masks
    iou = (intersection + 1e-6) / (union + 1e-6)  # Calculate the IOU of predicted and target masks
    return iou.mean()  # Return the mean IOU

# Define the pixel accuracy function
def pixel_acc(preds, targets):
    preds = torch.as_tensor(preds)  # Convert the predicted masks to a tensor
    targets = torch.as_tensor(targets).long()  # Convert the target masks to a tensor
    correct = (preds == targets).float().sum()  # Calculate the number of correctly predicted pixels
    total = targets.numel()  # Calculate the total number of pixels
    acc = correct / total  # Calculate the pixel accuracy
    return acc  # Return the pixel accuracy

# Define the custom dataset class
class Cam101Dataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images  # List of images
        self.labels = labels  # List of corresponding labels
        self.transform = transform  # Transformation pipeline
    
    def __len__(self):
        return len(self.images)  # Return the length of the dataset
    
    def __getitem__(self, idx):
        image = self.images[idx]  # Get the image at the specified index
        label = self.labels[idx]  # Get the label at the specified index
        
        label = cv2.threshold(label, 0, 1, cv2.THRESH_BINARY)[1]  # Threshold the label to get binary mask
        
        label = np.argmax(label, axis=-1)  # Convert the label to an integer mask
        if self.transform:
            augmented = self.transform(image=image, mask=label)  # Apply the transformation pipeline to the image and mask
            image = augmented['image']  # Get the transformed image
            label = augmented['mask']  # Get the transformed mask
        return image, label  # Return the transformed image and mask

#DoubleConv neural network module
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        #Sequential convolutional neural network module with 2 sets of convolutional layers
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),  # 2D convolutional layer with specified input and output channels, kernel size, and padding
            nn.BatchNorm2d(out_channels),  # Batch normalization layer for normalization and stabilization
            nn.ReLU(inplace=True),  # ReLU activation function to introduce non-linearity
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),  # Second 2D convolutional layer with the same number of input and output channels, kernel size, and padding
            nn.BatchNorm2d(out_channels),  # Batch normalization layer
            nn.ReLU(inplace=True)  # ReLU activation function
        )

    # Forward pass of the DoubleConv module
    def forward(self, x):
        x = self.conv(x)  # Apply the sequential convolutional layers to the input
        return x


# Up neural network module
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        # If bilinear interpolation is to be used for upsampling
        if bilinear:
            # Define an nn.Upsample module with a scale factor of 2, bilinear interpolation mode, and corner alignment
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            # Define an nn.ConvTranspose2d module with a kernel size of 2 and stride of 2
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)  # Create an instance of DoubleConv module
        
    # Define the forward pass of the Up module
    def forward(self, x1, x2):
        x1 = self.up(x1)  # Upsample the input using either interpolation or transposed convolution
        diffY = x2.size()[2] - x1.size()[2]  # Calculate the difference in height between x2 and x1
        diffX = x2.size()[3] - x1.size()[3]  # Calculate the difference in width between x2 and x1

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])  # Pad x1 to match the spatial dimensions of x2
        x = torch.cat([x2, x1], dim=1)  # Concatenate x2 and x1 along the channel dimension
        x = self.conv(x)  # Apply the DoubleConv module to the concatenated tensor
        return x

#UNet neural network module
class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=32, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Encoder part of the UNet
        self.inc = DoubleConv(n_channels, 64)  # Initial convolutional layers
        self.down1 = DoubleConv(64, 128)  # Downsample 1
        self.down2 = DoubleConv(128, 256)  # Downsample 2
        self.down3 = DoubleConv(256, 512)  # Downsample 3
        self.down4 = DoubleConv(512, 512)  # Downsample 4

        # Decoder part of the UNet
        self.up1 = Up(1024, 256, bilinear)  # Upsample 1
        self.up2 = Up(512, 128, bilinear)  # Upsample 2
        self.up3 = Up(256, 64, bilinear)  # Upsample 3
        self.up4 = Up(128, n_classes, bilinear)  # Upsample 4

        self.outc = nn.Conv2d(n_classes, n_classes, kernel_size=1)  # Output convolutional layer

    # Forward pass of the UNet module
    def forward(self, x):
        x1 = self.inc(x)  # Apply initial convolutional layers
        x2 = self.down1(x1)  # Apply downsample 1
        x3 = self.down2(x2)  # Apply downsample 2
        x4 = self.down3(x3)  # Apply downsample 3
        x5 = self.down4(x4)  # Apply downsample 4

        x = self.up1(x5, x4)  # Apply upsample 1
        x = self.up2(x, x3)  # Apply upsample 2
        x = self.up3(x, x2)  # Apply upsample 3
        x = self.up4(x, x1)  # Apply upsample 4
        x = self.outc(x)  # Apply output convolutional layer

        return x

# SEGNET module
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),  # 3x3 convolutional layer
            nn.BatchNorm2d(out_channels),  # Batch normalization
            nn.ReLU(inplace=True),  # ReLU activation function
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),  # 3x3 convolutional layer
            nn.BatchNorm2d(out_channels),  # Batch normalization
            nn.ReLU(inplace=True)  # ReLU activation function
        )

    def forward(self, x):
        x = self.conv(x)  # Pass input through the convolutional layers
        return x


#Up module
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # Bilinear upsampling
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)  # Transposed convolution

        self.conv = DoubleConv(in_channels, out_channels)  # DoubleConv block

    def forward(self, x1, x2):
        x1 = self.up(x1)  # Upsample the input
        diffY = x2.size()[2] - x1.size()[2]  # Calculate the difference in height
        diffX = x2.size()[3] - x1.size()[3]  # Calculate the difference in width

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])  # Pad the input to match the size of the other input
        x = torch.cat([x2, x1], dim=1)  # Concatenate the two inputs along the channel dimension
        x = self.conv(x)  # Pass the concatenated input through the DoubleConv block
        return x



# SEGNet model
class SEGNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=32, bilinear=True):
        super(SEGNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Encoder
        self.conv1 = DoubleConv(n_channels, 64)  # First DoubleConv block
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # First max pooling layer
        self.conv2 = DoubleConv(64, 128)  # Second DoubleConv block
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # Second max pooling layer
        self.conv3 = DoubleConv(128, 256)  # Third DoubleConv block
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # Third max pooling layer
        self.conv4 = DoubleConv(256, 512)  # Fourth DoubleConv block
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)  # Fourth max pooling layer
        self.conv5 = DoubleConv(512, 512)  # Fifth DoubleConv block
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)  # Fifth max pooling layer

        # Decoder
        self.up4 = Up(1024, 256, bilinear)  # Fourth Up block
        self.up3 = Up(512, 128, bilinear)  # Third Up block
        self.up2 = Up(256, 64, bilinear)  # Second Up block
        self.up1 = Up(128, n_classes, bilinear)  # First Up block
        self.outc = nn.Conv2d(n_classes, n_classes, kernel_size=1)  # Output convolutional layer

    def forward(self, x):
        # Encoder
        x1 = self.conv1(x)  # Pass input through the first DoubleConv block
        x2 = self.conv2(self.pool1(x1))  # Pass the output of the first block through the second DoubleConv block and max pooling
        x3 = self.conv3(self.pool2(x2))  # Pass the output of the second block through the third DoubleConv block and max pooling
        x4 = self.conv4(self.pool3(x3))  # Pass the output of the third block through the fourth DoubleConv block and max pooling
        x5 = self.conv5(self.pool4(x4))  # Pass the output of the fourth block through the fifth DoubleConv block and max pooling

        # Decoder
        x = self.up4(x5, x4)  # Pass the output of the fifth block and the fourth block through the fourth Up block
        x = self.up3(x, x3)  # Pass the output of the previous step and the third block through the third Up block
        x = self.up2(x, x2)  # Pass the output of the previous step and the second block through the second Up block
        x = self.up1(x, x1)  # Pass the output of the previous step and the first block through the first Up block
        x = self.outc(x)  # Pass the final output through the output convolutional layer

        return x

# Define the DoubleConv block
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),  # 3x3 convolution
            nn.BatchNorm2d(mid_channels),  # Batch normalization
            nn.ReLU(inplace=True),  # ReLU activation function
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),  # 3x3 convolution
            nn.BatchNorm2d(out_channels),  # Batch normalization
            nn.ReLU(inplace=True)  # ReLU activation function
        )

    def forward(self, x):
        return self.double_conv(x)

# Define the Up block
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # Bilinear upsampling
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)  # Transposed convolution
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)  # Upsample the input
        if x1.size()[2] != x2.size()[2]:
            # If the spatial dimensions do not match, pad the input
            diff_h = x2.size()[2] - x1.size()[2]
            diff_w = x2.size()[3] - x1.size()[3]
            x1 = F.pad(x1, (diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2))
        x = torch.cat([x2, x1], dim=1)  # Concatenate the feature maps along the channel dimension
        x = self.conv(x)  # Apply DoubleConv operation
        return x


# Define the Pyramid Spatial Pooling (PSP) module
class PSPModule(nn.Module):
    def __init__(self, features, out_features=512, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []  # List to store stages of PSP
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])  # Create stages using _make_stage function
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)  # Bottleneck convolution
        self.relu = nn.ReLU(inplace=True)  # ReLU activation function

    def _make_stage(self, features, size):
        pool = nn.AdaptiveAvgPool2d(output_size=(size, size))  # Adaptive average pooling
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)  # 1x1 convolution
        return nn.Sequential(pool, conv)  # Sequential block

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)  # Height and width of input feature map
        out = [feats]  # List to store feature maps at each stage
        for stage in self.stages:
            pooled = stage(feats)  # Apply stage operation
            pooled = F.interpolate(pooled, size=(h, w), mode='bilinear', align_corners=True)  # Bilinear interpolation
            out.append(pooled)  # Append the pooled feature map
        out = torch.cat(out, 1)  # Concatenate the feature maps along the channel dimension
        out = self.bottleneck(out)  # Apply bottleneck convolution
        return self.relu(out)  # Apply ReLU activation

# Define the PSPNet model
class PSPNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=32, bilinear=True):
        super(PSPNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Encoder
        self.inc = DoubleConv(n_channels, 64)  # First convolutional block
        self.down1 = DoubleConv(64, 128)  # Second convolutional block
        self.down2 = DoubleConv(128, 256)  # Third convolutional block
        self.down3 = DoubleConv(256, 512)  # Fourth convolutional block
        self.down4 = DoubleConv(512, 512)  # Fifth convolutional block

        # Pyramid Spatial Pooling
        self.pyramid_pooling = PSPModule(512)

        # Decoder
        self.up1 = Up(1024, 256, bilinear)  # First upsampling block
        self.up2 = Up(512, 128, bilinear)  # Second upsampling block
        self.up3 = Up(256, 64, bilinear)  # Third upsampling block
        self.up4 = Up(128, n_classes, bilinear)  # Fourth upsampling block

        # Final output convolution
        self.outc = nn.Conv2d(n_classes, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)


        x5 = self.pyramid_pooling(x5)


        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)

        return x

# Create train and test datasets using the Cam101Dataset class and apply the transformation to the data
train_dataset = Cam101Dataset(train_images, train_labels, transform=transform)
test_dataset = Cam101Dataset(test_images, test_labels, transform=transform)

# Split the train dataset into train and validation datasets
train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [len(train_dataset)-int(len(train_dataset)*0.1), int(len(train_dataset)*0.1)])

# Define the batch size and create data loaders for train, validation, and test datasets
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# Define the different hyperparameters to be used in training the model
learning_rates = [1e-3, 1e-4, 1e-5]
optimizers = [optim.Adam, optim.SGD]
loss_functions = [nn.CrossEntropyLoss(), nn.BCEWithLogitsLoss()]

# Define a function to train the model using a given set of inputs, targets, criterion, optimizer, and device
# This function returns the average loss, accuracy, and IOU for the training data
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    train_loss = 0.0
    train_acc = 0.0
    train_iou = 0.0
    total_train_samples = 0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets.long())
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * inputs.size(0)
        train_acc += pixel_acc(outputs.argmax(dim=1), targets.squeeze(1)) * inputs.size(0)
        train_iou += mean_iou(outputs.argmax(dim=1), targets.squeeze(1)) * inputs.size(0)
        total_train_samples += inputs.size(0)

    train_loss /= total_train_samples
    train_acc /= total_train_samples
    train_iou /= total_train_samples

    return train_loss, train_acc, train_iou

# Define a function to test the model using a given set of inputs, targets, criterion, and device
# This function returns the average loss, accuracy, and IOU for the test data
def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    test_acc = 0.0
    test_iou = 0.0
    total_test_samples = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets.long())

            test_loss += loss.item() * inputs.size(0)
            test_acc += pixel_acc(outputs.argmax(dim=1), targets.squeeze(1)) * inputs.size(0)
            test_iou += mean_iou(outputs.argmax(dim=1), targets.squeeze(1)) * inputs.size(0)
            total_test_samples += inputs.size(0)

    test_loss /= total_test_samples
    test_acc /= total_test_samples
    test_iou /= total_test_samples

    return test_loss, test_acc, test_iou



# Set the number of epochs for training
num_epochs=1
# Check if GPU is available, if yes use it, else use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Define hyperparameters to try out
learning_rates = [1e-3, 5e-4]
optimizers = [optim.Adam, optim.SGD]
loss_functions = [nn.CrossEntropyLoss]
params = {'learning_rate': learning_rates, 'optimizer': optimizers, 'loss_function': loss_functions}

# Initialize variables for best model selection
best_model = None
best_score = float('-inf')

# Define the models to iterate over
models = [UNet(), SEGNet(), PSPNet()]

# Iterate over all models
for model in models:
    # Iterate over all hyperparameter combinations using cross-validation
    for p in ParameterGrid(params):
        learning_rate = p['learning_rate']
        optimizer = p['optimizer']
        loss_function = p['loss_function']

        # Define the loss function with given hyperparameters
        criterion = loss_function()
        # Define the optimizer with given hyperparameters
        optimizer = optimizer(model.parameters(), lr=learning_rate)
        # Define the learning rate scheduler
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        train_losses = []
        val_losses = []
        train_accs = []
        train_iou_scores = []
        val_accs = []
        val_iou_scores = []
        # Train model on training set
        for epoch in range(num_epochs):
            train_loss, train_acc, train_iou = train(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc, val_iou = test(model, val_loader, criterion, device)
            # Store the losses
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            # Store the evaluation metrics
            train_accs.append(train_acc)
            train_iou_scores.append(train_iou)
            val_accs.append(val_acc)
            val_iou_scores.append(val_iou)
            # Update the learning rate
            scheduler.step()
            # Print the training and validation metrics for each epoch
            print(f"Model: {model.__class__.__name__}, Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.5f}, Train IoU: {train_iou:.5f}, Train Acc: {train_acc:.5f}, Val Loss: {val_loss:.5f}, Val IoU: {val_iou:.5f}, Val Acc: {val_acc:.5f}")

        # Evaluate model on validation set
        val_loss, val_acc, val_iou = test(model, val_loader, criterion, device)
        score = val_iou  # Use IoU as evaluation metric
        # Plot the losses and evaluation metrics for training and validation
        epochs = range(num_epochs)
        plt.plot(epochs, train_losses, label='Training loss')
        plt.plot(epochs, val_losses, label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        epochs = range(num_epochs)
        plt.plot(epochs, train_accs, label='Training accuracy')
        plt.plot(epochs, train_iou_scores, label='Training IoU')
        plt.plot(epochs, val_accs, label='Validation accuracy')
        plt.plot(epochs, val_iou_scores, label='Validation IoU')
        plt.title('Training and validation evaluation metrics')
        plt.xlabel('Epoch')
        plt.ylabel('Evaluation metric')
        plt.legend()
        plt.show

                # Update best model
        if score > best_score:
            best_model = model
            best_score = score
            best_hyperparams = p
    with torch.no_grad():
        # get a random sample from the test dataset
        sample_idx = np.random.randint(len(test_images))
        image = test_images[sample_idx]
        label = test_labels[sample_idx]
        # apply the same transform used for training
        augmented = transform(image=image, mask=label)
        image = augmented['image']
        label = augmented['mask']
        # make predictions
        output = model(image.unsqueeze(0).to(device))
        pred = torch.argmax(output, dim=1).squeeze(0)
        # plot the results
        fig, ax = plt.subplots(1, 3, figsize=(10, 5))
        ax[0].imshow(img/255.0)
        ax[0].set_title('Original Image')
        ax[1].imshow(pred.cpu().squeeze(), cmap='gray')
        ax[1].set_title('Predicted Mask')
        ax[2].imshow(label.squeeze(), cmap='gray')
        ax[2].set_title('Ground Truth Mask')
        plt.show()

    # Test best model on test dataset
    test_loss, test_acc, test_iou = test(best_model, test_loader, criterion, device)
    print(f"Best hyperparameters: {best_hyperparams}, Test Loss: {test_loss:.5f}, Test IoU: {test_iou:.5f}, Test Acc: {test_acc:.5f}")


