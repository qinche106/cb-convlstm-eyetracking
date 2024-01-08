import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import os
import cv2
import pandas as pd
from convlstmbak import ConvLSTM
import tqdm
import tables
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from thop import profile
from scipy.ndimage import median_filter

pretrained = False
test_one =True
height = N = 60   # input y size
width = M = 80  # input x size
batch_size = 16
seq = 40
stride = 1
stride_val = 40
chunk_size = 500
num_epochs = 60
# interval=int((chunk_size-seq)/stride+1)

log_dir = 'log'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
plot_dir = 'plot'
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)


def normalize_data(data):

    # Convert the image data to a numpy array
    img_data = np.array(data)

    # Calculate mean and standard deviation
    mean = np.mean(img_data)
    std = np.std(img_data)

    # Check for constant images
    if std == 0:
        # print("Warning: constant image. Normalization may not be appropriate.")
        return img_data  # or handle in a different way if needed

    # Normalize the image
    normalized_img_data = (img_data - mean) / (std + 1e-10)

    return normalized_img_data

def create_samples(data, sequence, stride):
    num_samples = data.shape[0]

    chunk_num = num_samples // chunk_size

    # Create start indices for each chunk
    chunk_starts = np.arange(chunk_num) * chunk_size

    # For each start index, create the indices of subframes within the chunk
    within_chunk_indices = np.arange(sequence) + np.arange(0, chunk_size - sequence + 1, stride)[:, None]

    # For each chunk start index, add the within chunk indices to get the complete indices
    indices = chunk_starts[:, None, None] + within_chunk_indices[None, :, :]

    # Reshape indices to be two-dimensional
    indices = indices.reshape(-1, indices.shape[-1])

    subframes = data[indices]
    # sublabels = labels[indices]

    return subframes


class EventDataset(Dataset):
    def __init__(self, folder, target_dir, seq, stride):
        self.folder = sorted(folder)
        self.target_dir = target_dir
        self.seq = seq
        self.stride = stride
        self.target = self._concatenate_files()
        self.interval = int((chunk_size - self.seq) / self.stride + 1)
    def __len__(self):
        return len(self.folder) * self.interval  # assuming each file contains 100 samples

    def __getitem__(self, index):
        file_index = index // self.interval
        sample_index = index % self.interval

        file_path = self.folder[file_index]
        with tables.open_file(file_path, 'r') as file:
            sample = file.root.vector[sample_index]
            sample_resize = []
            for i in range(len(sample)):
                sample_resize.append(normalize_data(cv2.resize(sample[i,0], (int(width ), int(height )))))
            sample_resize = np.expand_dims(np.array(sample_resize), axis=1)

        label1 = self.target[index][:, 0]/M/(8)
        label2 = self.target[index][:, 1]/N/(8)
        # label = label1, label2
        label = np.concatenate([label1.reshape(-1, 1), label2.reshape(-1, 1)], axis=1)
        return torch.from_numpy(sample_resize), label

    def _concatenate_files(self):
        # Sort the file paths
        sorted_target_file_paths = sorted(self.target_dir)
        target = []
        for file_path in sorted_target_file_paths:
            with open(file_path, 'r') as target_file:
                lines = target_file.readlines()
                lines =lines[3::4]
            lines = [list(map(float, line.strip().split())) for line in lines]
            target.extend(lines)
        targets= np.array(torch.tensor(target))
        extended_labels = create_samples(targets, self.seq, self.stride)
        return torch.from_numpy(extended_labels)


def load_filenames(path):
    with open(path, 'r') as f:
        return [line.strip() for line in f.readlines()]


data_dir_train = "./dataset/data_ts_pro/train/"
data_dir_val = "./dataset/data_ts_pro/val/"
target_dir = "./dataset/pupil_st/label"

# Load filenames from the provided lists
train_filenames = load_filenames('train_files.txt')
val_filenames = load_filenames('val_files.txt')

# Get the data file paths and target file paths
data_train = [os.path.join(data_dir_train, f + '.h5') for f in train_filenames]
target_train = [os.path.join(target_dir, f + '.txt') for f in train_filenames]

data_val = [os.path.join(data_dir_val, f + '.h5') for f in val_filenames]
target_val = [os.path.join(target_dir, f + '.txt') for f in val_filenames]



# Create datasets
train_dataset = EventDataset(data_train, target_train, seq, stride)
val_dataset = EventDataset(data_val, target_val, seq, stride_val)


# Create dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

valid_dataloader_plt = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


class MyModel(nn.Module):
    def __init__(self, height, width, input_dim):
        super(MyModel, self).__init__()

        self.convlstm1 = ConvLSTM(input_dim=input_dim, hidden_dim=8, kernel_size=(3, 3), num_layers=1, batch_first=True)
        self.bn1 = nn.BatchNorm3d(8)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2))
        # #
        # self.convlstm1_rgb = ConvLSTM(input_dim=input_dim, hidden_dim=32, kernel_size=(3, 3), num_layers=1, batch_first=True)
        # self.bn1_rgb = nn.BatchNorm3d(32)
        # self.pool1_rgb = nn.MaxPool3d(kernel_size=(1, 2, 2))

        self.convlstm2 = ConvLSTM(input_dim=8, hidden_dim=16, kernel_size=(3, 3), num_layers=1, batch_first=True)
        self.bn2 = nn.BatchNorm3d(16)
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2))

        self.convlstm3 = ConvLSTM(input_dim=16, hidden_dim=32, kernel_size=(3, 3), num_layers=1, batch_first=True)
        self.bn3 = nn.BatchNorm3d(32)
        self.pool3 = nn.MaxPool3d(kernel_size=(1, 2, 2))

        self.convlstm4 = ConvLSTM(input_dim=32, hidden_dim=64, kernel_size=(3, 3), num_layers=1, batch_first=True)
        self.bn4 = nn.BatchNorm3d(64)
        self.pool4 = nn.MaxPool3d(kernel_size=(1, 2, 2))

        # self.convlstm5 = ConvLSTM(input_dim=64, hidden_dim=64, kernel_size=(3, 3), num_layers=1, batch_first=True)
        # self.bn5 = nn.BatchNorm3d(64)
        # self.pool5 = nn.MaxPool3d(kernel_size=(1, 2, 2))

        # self.lstm = nn.LSTM(input_size=2240, hidden_size=512, num_layers=1, batch_first=True)
        # self.conv3d = nn.Conv3d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0)

        self.fc1 = nn.Linear(960, 128)
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 2)


    def forward(self, x):
        x, _ = self.convlstm1(x)
        x = x[0].permute(0, 2, 1, 3, 4)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)


        x = x.permute(0, 2, 1, 3, 4)
        x, _ = self.convlstm2(x)
        x = x[0].permute(0, 2, 1, 3, 4)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = x.permute(0, 2, 1, 3, 4)
        x, _ = self.convlstm3(x)
        x = x[0].permute(0, 2, 1, 3, 4)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)

        x = x.permute(0, 2, 1, 3, 4)
        x, _ = self.convlstm4(x)
        x = x[0].permute(0, 2, 1, 3, 4)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.pool4(x)

        # x = x.permute(0, 2, 1, 3, 4)
        # x, _ = self.convlstm5(x)
        # x = x[0].permute(0, 2, 1, 3, 4)
        # x = self.bn5(x)
        # x = F.relu(x)
        # x = self.pool5(x)


        # Flatten and apply LSTM layer
        x_list=[]
        b, c, seq, h, w = x.size()
        for t in range(seq):
            data = x[:,:,t,:,:]
            data = data.reshape(b, -1)
            data = F.relu(self.fc1(data))
            data = self.drop(data)
            data = self.fc2(data)
            x_list.append(data)
        y = torch.stack(x_list, dim =0)
        y = y.permute(1, 0, 2)
        return y


input_dim = 1 # set as per your data
model = MyModel(height, width, input_dim)
print(model)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params}")

criterion = nn.SmoothL1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


if pretrained == True:
    print("restoring the pretrained model....")
    checkpoint = torch.load('checkpoint.pth')
    model = MyModel(height, width, input_dim)
    model.to(device)
    # load state dict from file or wherever it is
    loaded_state_dict = torch.load('checkpoint.pth')

    model.load_state_dict(loaded_state_dict['model_state_dict'])
    optimizer.load_state_dict(loaded_state_dict['optimizer_state_dict'])
    epoch = loaded_state_dict['epoch']
    loss = loaded_state_dict['loss']
    print(f"Loss: {loss:.4f}")

# Training loop
model.train()
best_val_loss = float('inf')  # Initialize with a large value
# a = 1
for epoch in range(num_epochs):
    running_loss = 0.0
    total_data = len(train_dataloader)
    for t, data in tqdm.tqdm(enumerate(train_dataloader, 0), total=total_data):
        images, targets = data
        images = images.to(device).float()
        targets = targets.to(device).float()

        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        prev_output = outputs
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    # Validation
    val_running_loss=0
    num_values = 0
    num_values_3 = 0
    num_values_5 = 0
    num_values_1 = 0
    tot_values=0
    model.eval()
    with torch.no_grad():
        for images, targets in valid_dataloader:
            images = images.to(device).float()
            targets = targets.to(device).float()
            outputs = model(images)
            val_loss = criterion(outputs, targets)
            dis = targets - outputs
            dis[:, :, 0] *= height
            dis[:, :, 1] *= width
            dist = torch.norm(dis, dim=-1)
            # dist_list.append(dist)
            num_values = num_values+ torch.sum(dist > 10)
            num_values_5 = num_values_5 + torch.sum(dist > 5)
            num_values_3 = num_values_3 + torch.sum(dist > 3)
            num_values_1 = num_values_1 + torch.sum(dist > 1)
            tot_values = tot_values+ dist.numel()
            val_running_loss += val_loss.item()
        val_epoch_loss = val_running_loss / len(valid_dataloader)
        err_rate = num_values/tot_values
        err_rate_3 = num_values_3/tot_values
        err_rate_5 = num_values_5/tot_values
        err_rate_1 = num_values_1/tot_values
        print(f"Validation Loss: {val_epoch_loss:.4f}")
        print(f"err_rate: {err_rate:.4f}")

        # File path
        file_path = os.path.join(log_dir, 'training_log.txt')
        with open(file_path, 'a') as f:
            f.write(f"Size {height}, Epoch {epoch}, Loss: {val_epoch_loss}, err_rate_1:{err_rate_1}, err_rate_3:{err_rate_3}, err_rate_5:{err_rate_5}  err: {err_rate} num_values: {num_values} tot_values: {tot_values}\n")
        # Save the model if it has the best validation loss so far
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            print("saving best model...")
            # torch.save(model.state_dict(), 'best_model.pth')

            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, 'checkpoint.pth')


        valid_dataloader_plt = DataLoader(val_dataset, batch_size=100, shuffle=False)

        for t, data in enumerate(valid_dataloader_plt):
            if t ==1:
                break
            images, targets = data
            frames_plot = images.to(device).float()
            target_plot = targets.to(device).float()
            t_l_list =[]
            t_r_list =[]
            o_l_list =[]
            o_r_list = []
            for i in range(len(frames_plot)):
                images = frames_plot[i].unsqueeze(0)
                # images = frames_plot[i].to(device)
                outputs = model(images)
                targets = target_plot[i]
                t_l =np.array(targets[:,0].cpu()).flatten()
                t_r =np.array(targets[:,1].cpu()).flatten()
                o_l =outputs.detach().cpu().numpy()[:,:,0].flatten()
                o_r =outputs.detach().cpu().numpy()[:,:,1].flatten()
                t_l_list.append(t_l)
                t_r_list.append(t_r)
                o_l_list.append(o_l)
                o_r_list.append(o_r)
            t_l_numpy = np.array(t_l_list).flatten()
            t_r_numpy = np.array(t_r_list).flatten()
            o_l_numpy = np.array(o_l_list).flatten()
            o_r_numpy = np.array(o_r_list).flatten()
            fig, (ax1, ax2) = plt.subplots(2, 1)

            # Plot l1 and l2 in the first subplot
            ax1.plot(t_l_numpy, label='target 1')
            ax1.plot(o_l_numpy, label='output 1')
            ax1.set_xlabel('x')
            ax1.set_ylabel('y')
            ax1.set_title('Plot of Line 1 and Line 2')
            ax1.set_ylim([0, 1])
            ax1.legend()

            # Plot l3 and l4 in the second subplot
            ax2.plot(t_r_numpy, label='target 2')
            ax2.plot(o_r_numpy, label='output 2')
            ax2.set_xlabel('x')
            ax2.set_ylabel('y')
            ax2.set_title('Plot of Line 3 and Line 4')
            ax2.set_ylim([0, 1])
            ax2.legend()

            # Adjust the spacing between subplots
            plt.subplots_adjust(hspace=0.5)

            picname=f'event_plot_{epoch}.png'
            plt.savefig(os.path.join(plot_dir, picname))
            plt.close()

            frames_plot = np.array(frames_plot.reshape(-1, frames_plot.shape[-2], frames_plot.shape[-1]).cpu())
            fig, axs = plt.subplots(4, 4, figsize=(10, 10))

            for i, ax in enumerate(axs.flatten()):
                # Plot the image
                ax.imshow(frames_plot[i], cmap='gray')  # Displaying it in grayscale for this example
                x, y = o_l_numpy[i] * width, o_r_numpy[i] * height
                ax.plot(x, y, 'ro')
                # Hide the axes
                ax.axis('off')

            # plt.show()
            picname2 = f'eye_plot_{epoch}.png'
            plt.savefig(os.path.join(plot_dir, picname2))
            plt.close()