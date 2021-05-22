import os
import sys
import glob
import shutil
import tarfile
import argparse

import numpy as np

from sklearn.model_selection import train_test_split

import cv2
import torch
from torch import nn, optim
from torchvision import datasets, transforms

from tqdm import tqdm

from config import Config
from network import Model

# Config Parsing
def get_config():
    parser = argparse.ArgumentParser(description = "Multi-layer perceptron")
    parser.add_argument("--epochs", default = 10, type = int)
    parser.add_argument("--batch_size", default = 256, type = int)
    parser.add_argument("--lr", default = 0.001, type = float)

    args = parser.parse_args()

    config = Config(
        EPOCHS = args.epochs,
        BATCH_SIZE = args.batch_size,
        LEARNING_RATE = args.lr,
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    )

    return config

'''
# notMNIST dataset
def loadData(BATCH_SIZE: int):
    data = []

    folders = os.listdir('./notMNIST_small')
    for label, folder in enumerate(folders):
        files = glob.glob(os.path.join('notMNIST_small', folder, '*.png'))
        
        for file in files:
            try:
                arr = cv2.imread(file, 0).astype(np.float) / 255.0
                tensor = transforms.ToTensor()(arr)
                data.append((tensor, label))
            except:
                pass
    train, test = train_test_split(data, test_size = 0.2, random_state = 42)
    
    train_iter = torch.utils.data.DataLoader(train, batch_size = BATCH_SIZE, shuffle = True)
    test_iter = torch.utils.data.DataLoader(test, batch_size = BATCH_SIZE, shuffle = True)
    
    return train_iter, test_iter
'''


# load and organize the file structure
def loadData():

    directory = os.listdir()

    if 'dataset' in directory:
        print('already loaded data..')
        return

    # 압축 해제
    if 'notMNIST_small' not in directory:
        fname = "notMNIST_small.tar.gz"
        tar = tarfile.open(fname, "r")
        tar.extractall()
        tar.close()
    
    # 모든 파일 로드
    dataset = []
    folders = os.listdir('./notMNIST_small')
    for label, folder in enumerate(folders):
        files = glob.glob(os.path.join('notMNIST_small', folder, '*.png'))
        dataset.extend(list(zip(files, folder * len(files))))
    dataset = np.array(dataset)
    
    # 데이터셋 별, 클래스 별 폴더 정리
    train, test = train_test_split(dataset, test_size = 0.2, stratify = dataset[:, -1])
    
    DATA_PATH = 'dataset'
    if not os.path.exists(DATA_PATH):
        os.mkdir(DATA_PATH)
        os.makedirs(os.path.join(DATA_PATH, 'train'))
        os.makedirs(os.path.join(DATA_PATH, 'test'))
        
    for folder in set(dataset[:, -1]):
        os.makedirs(os.path.join(DATA_PATH, "train", folder))
        os.makedirs(os.path.join(DATA_PATH, "test", folder))
    
    # 파일 이동
    for folder, data in [('train', train), ('test', test)]:
        for file in data:
            file_name = file[0].split('/')[-1]
            label = file[-1]
            from_path = file[0]
            to_path = os.path.join('dataset', folder, label, file_name)
            shutil.move(from_path, to_path)
            
    print(f'total files: {len(dataset)}')
    print(f'total labels: {len(set(dataset[:, -1]))}')
                

# make tensor iter dataset for pytorch
def makeData(BATCH_SIZE: int):
    
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
    # 테스트의 경우에는 변형성의 augmentation 테크닉을 적용하지는 않음
    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224), # 사진 자르기
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    DATASET_PATH = "dataset"
    # Pass transforms in here, then run the next cell to see how the transforms look
    # ImageFolder, 각 폴더별 정리된 데이터를 자동으로 라벨링 해줌 엄청나다!
    train_data = datasets.ImageFolder(DATASET_PATH + '/train', transform = train_transforms)
    test_data = datasets.ImageFolder(DATASET_PATH + '/test', transform = test_transforms)
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size = BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(train_data, batch_size = BATCH_SIZE, shuffle=True)
    
    return train_loader, test_loader


# Defining Model
def get_network(LEARNING_RATE: float, device: str):
    network = Model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(network.parameters(), lr=LEARNING_RATE)

    return network, criterion, optimizer


# Print Model Info
def print_modelinfo(model: nn.Module):
    total_params = 0
    for param_name, param in model.named_parameters():
        if param.requires_grad:
            total_params += len(param.reshape(-1))
    print(f"Number of Total Parameters: {total_params:,d}")


# Define help function
def test_eval(model: nn.Module, test_iter, batch_size: int, device: str):
    with torch.no_grad():
        test_loss = 0
        total = 0
        correct = 0
        for batch_img, batch_lab in test_iter:
            X = batch_img.view(-1, 3, 224, 224).to(device)
            Y = batch_lab.to(device)
            y_pred = model(X.float())
            _, predicted = torch.max(y_pred.data, 1)
            correct += (predicted == Y).sum().item()
            total += batch_img.size(0)
        val_acc = 100 * correct / total
    return val_acc


# Train MLP Model
def train_model(
    model: nn.Module, train_iter, test_iter, EPOCHS: int, BATCH_SIZE: int, device: str
):
    # Training Phase
    print_every = 1
    print("Start training !")
    # Training loop
    for epoch in range(EPOCHS):
        loss_val_sum = 0
        for batch_img, batch_lab in tqdm(train_iter):
            X = batch_img.view(-1, 3, 224, 224).type(torch.float).to(device)
            Y = batch_lab.to(device)

            # Inference & Calculate los
            y_pred = model.forward(X)
            loss = criterion(y_pred, Y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_val_sum += loss

        if ((epoch % print_every) == 0) or (epoch == (EPOCHS - 1)):
            # accr_val = M.test(x_test, y_test, batch_size)
            loss_val_avg = loss_val_sum / len(train_iter)
            accr_val = test_eval(model, test_iter, BATCH_SIZE, device)
            print(
                f"epoch:[{epoch+1}/{EPOCHS}] cost:[{loss_val_avg:.3f}] test_accuracy:[{accr_val:.3f}]"
            )
    print("Training Done !")


def test_model(model, test_iter, device: str):
    model.eval()
    mnist_test = test_iter.dataset

    n_sample = 64
    sample_indices = np.random.choice(len(mnist_test.targets), n_sample, replace=False)
    test_x = mnist_test.data[sample_indices]
    test_y = mnist_test.targets[sample_indices]

    with torch.no_grad():
        y_pred = model.forward(test_x.view(-1, 3, 224, 224).to(device))
#         y_pred = model.forward(test_x.view(-1, 28 * 28).type(torch.float).to(device))

    y_pred = y_pred.argmax(axis=1)

    plt.figure(figsize=(20, 20))

    for idx in range(n_sample):
        plt.subplot(8, 8, idx + 1)
        plt.imshow(test_x[idx], cmap="gray")
        plt.axis("off")
        plt.title(f"Predict: {y_pred[idx]}, Label: {test_y[idx]}")

    plt.show()


if __name__ == "__main__":
    print("PyTorch version:[%s]." % (torch.__version__))

    config = get_config()
    print("This code use [%s]." % (config.device))

    loadData()
    train_iter, test_iter = makeData(config.BATCH_SIZE)
    print("Preparing dataset done!")

    network, criterion, optimizer = get_network(config.LEARNING_RATE, config.device)
    print_modelinfo(network)

    train_model(
        network, train_iter, test_iter, config.EPOCHS, config.BATCH_SIZE, config.device
    )

#     test_model(network, test_iter, config.device)
