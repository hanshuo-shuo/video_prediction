import numpy as np
from gif_to_array import readGif
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

def sample_d(num):
    t_start = np.random.randint(0,10)
    t_end = np.random.randint(t_start, 10+t_start)

    gif_array = readGif(f"gif_real/test_{num}.gif")
    image_start = gif_array[t_start]
    image_end = gif_array[t_end]

    distance = t_end - t_start

    return image_start, image_end, distance


def train_set(n):
    train_set = []

    for i in range(n):
        num = np.random.randint(0,1000)
        image_start, image_end, distance = sample_d(num)
        sample = np.concatenate((image_start, image_end), axis=1)
        train_set.append([sample, distance])

    return train_set


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  # input shape (1, 64, 128)
            nn.Conv2d(
                in_channels=1,      # input height
                out_channels=16,    # n_filters
                kernel_size=5,      # filter size
                stride=1,           # filter movement/step
                padding=2,      # 如果想要 con2d 出来的图片长宽没有变化, padding=(kernel_size-1)/2 当 stride=1
            ),      # output shape (16, 64, 128)
            nn.ReLU(),    # activation
            nn.MaxPool2d(kernel_size=2),    # 在 2x2 空间里向下采样, output shape (16, 32, 64)
        )
        self.conv2 = nn.Sequential(  # input shape (16, 32, 64)
            nn.Conv2d(16, 32, 5, 1, 2),  # output shape  (32, 32, 64)
            nn.ReLU(),  # activation
            nn.MaxPool2d(2),  # output shape (32, 16, 32)
        )
        self.out2 = nn.Linear(32 * 16 * 32, 10) # fully connected layer, output 10 classes
        self.out = nn.Linear(10, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.reshape(x.size(0), -1)
        x = self.out2(x)
        output = self.out(x)
        return output


def train_and_val(n):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    a = train_set(n)
    train_x = []
    label_y = []
    for i in range(n):
        img = a[i][0]
        train_x.append(img)
        label = a[i][1]
        label_y.append(label)

    train_x = np.array(train_x)
    train_x = np.float32(train_x)
    label_y = np.array(label_y)
    label_y = np.float32(label_y)
    train_x, val_x, train_y, val_y = train_test_split(train_x, label_y, test_size=0.1)

    train_x = train_x.reshape(int(n * 0.9), 1, 64, 128)
    train_x = torch.from_numpy(train_x)

    train_y = train_y.reshape(int(n * 0.9), 1)
    train_y = train_y.astype(int)
    train_y = torch.from_numpy(train_y)
    val_x = val_x.reshape(int(0.1 * n), 1, 64, 128)
    val_x = torch.from_numpy(val_x)

    # converting the target into torch format
    val_y = val_y.reshape(int(n * 0.1), 1)
    val_y = val_y.astype(int)
    val_y = torch.from_numpy(val_y)

    x_train, y_train = Variable(train_x), Variable(train_y)
    x_val, y_val = Variable(val_x), Variable(val_y)

    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_val = x_val.to(device)
    y_val = y_val.to(device)

    return x_train, y_train, x_val, y_val

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    EPOCH = 10000
    BATCH_SIZE = 20
    LR = 0.001
    cnn = CNN().to(device)

    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)  # optimize all cnn parameters
    criterion = nn.CrossEntropyLoss()


    # clearing the Gradients of the model parameters
    optimizer.zero_grad()
    # empty list to store training losses
    train_losses = []
    # empty list to store validation losses
    val_losses = []

    # training and testing
    for epoch in range(100):
        x_train, y_train, x_val, y_val = train_and_val(BATCH_SIZE)
        for i in range(10):
            output_train = cnn(x_train)
            output_val = cnn(x_val)

            loss_train = criterion(output_train, y_train.float())
            loss_val = criterion(output_val, y_val.float())

            # computing the updated weights of all the model parameters
            loss_train.backward()
            optimizer.step()

        if epoch % 10 == 0:

            print('Epoch : ', epoch + 1, '\t', 'train_loss :', loss_train)
            print('Epoch : ', epoch + 1, '\t', 'val_loss :', loss_val)
            train_losses.append(loss_train)
            val_losses.append(loss_val)


    with torch.no_grad():
        plt.plot(train_losses, '-bx')
        plt.plot(val_losses, '-rx')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(['Training', 'Validation'])
        plt.title('MSE Loss vs. No. of epochs')
        plt.show()



def main1():
    BATCH_SIZE = 50
    train_loader = train_and_val(BATCH_SIZE)

    # for step, (x) in enumerate(train_loader):
    #     x1 = x[0]
    #     print("---")
    #     print(x1)
    #     print(step)




if __name__ == "__main__":
    main()