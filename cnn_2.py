import numpy as np
from gif_to_array import readGif
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
CHECKPOINT_PATH = './checkpoint.tar'
import wandb

PROJECT_NAME = 'cnn_prediction1_and_save_model'

run = wandb.init(project=PROJECT_NAME, resume=False)

def sample_d(num):
    t_start = np.random.randint(0,10)
    t_end = np.random.randint(t_start, 10+t_start)

    gif_array = readGif(f"gif_real/test_{num}.gif")
    image_start = gif_array[t_start]
    image_end = gif_array[t_end]

    distance = t_end - t_start
    if distance == 0:
        d = 0
    elif distance in [1,2,3]:
        d = 1
    elif distance in [4,5,6]:
        d = 2
    else:
        d = 3

    return image_start, image_end, d


def train_set(n):
    train_set = []

    for i in range(n):
        num = np.random.randint(0,10000)
        image_start, image_end, distance = sample_d(num)
        sample = np.concatenate((image_start, image_end), axis=1)
        train_set.append([sample, distance])

    return train_set


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.fc1 = nn.Linear(226432, 2048)
        self.fc2 = nn.Linear(2048, 256)
        self.fc3 = nn.Linear(256, 4)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.dropout2(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        output = F.log_softmax(x, dim=1)
        return output

class soft_CNN(nn.Module):
    def __init__(self):
        super(soft_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(119040, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, 4)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        y_hat = F.log_softmax(x, dim=1)
        return y_hat

def train_and_val(n):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    a = train_set(n)
    train_x = []
    label_y = []
    test_size = 0.5
    for i in range(n):
        img = a[i][0]
        train_x.append(img)
        label = a[i][1]
        label_y.append(label)

    train_x = np.array(train_x)
    train_x = np.float32(train_x)
    label_y = np.array(label_y)
    label_y = np.float32(label_y)
    train_x, val_x, train_y, val_y = train_test_split(train_x, label_y, test_size=test_size)

    train_x = train_x.reshape(int(n * test_size), 1, 64, 128)
    train_x = torch.from_numpy(train_x)

    train_y = train_y.astype(int)
    train_y = torch.from_numpy(train_y)
    val_x = val_x.reshape(int((1-test_size) * n), 1, 64, 128)
    val_x = torch.from_numpy(val_x)

    # converting the target into torch format
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
    EPOCH = 3
    BATCH_SIZE = 40
    LR = 0.0001
    cnn = CNN().to(device)
    soft_cnn = soft_CNN().to(device)

    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)  # optimize all cnn parameters
    criterion = nn.CrossEntropyLoss()  # the target label is not one-hotted


    # clearing the Gradients of the model parameters
    optimizer.zero_grad()
    # empty list to store training losses
    train_losses = []
    # empty list to store validation losses
    val_losses = []
    p_traines = []
    p_vales = []


    # training and testing
    for epoch in range(EPOCH):
        x_train, y_train, x_val, y_val = train_and_val(BATCH_SIZE)
        # train how many times for each data
        for i in range(1):
            print(y_train)
            output_train = cnn(x_train)
            print(output_train)
            output_val = cnn(x_val)
            loss_train = criterion(output_train, y_train)
            loss_val = criterion(output_val, y_val)

            # computing the updated weights of all the model parameters
            loss_train.backward()
            optimizer.step()

        if epoch % 1 == 0:

            print('Epoch : ', epoch + 1, '\t', 'train_loss :', loss_train)
            print('Epoch : ', epoch + 1, '\t', 'val_loss :', loss_val)
            train_losses.append(loss_train)
            val_losses.append(loss_val)
            wandb.log({'loss_train': loss_train, 'loss_val': loss_val})

            with torch.no_grad():
                output = cnn(x_train)
                softmax = torch.exp(output)
                prob = list(softmax.numpy())
                predictions = np.argmax(prob, axis=1)

                # accuracy on training set
                p_train = accuracy_score(y_train, predictions)

            with torch.no_grad():
                output = cnn(x_val)
                softmax = torch.exp(output)
                prob = list(softmax.numpy())
                predictions = np.argmax(prob, axis=1)

                p_val = accuracy_score(y_val, predictions)

            p_traines.append(p_train)
            p_vales.append(p_val)
            print("accuracy on the training set is:", p_train, "accuracy on the val set is:", p_val)
            wandb.log({'p_train': p_train, 'p_val': p_val})

            # Save our checkpoint loc
            torch.save({
                'optimizer_state_dict': optimizer.state_dict(),
                'model_state_dict': cnn.state_dict()
            }, CHECKPOINT_PATH)

            wandb.save(CHECKPOINT_PATH)




    with torch.no_grad():
        # plot Loss vs. No. of epochs
        plt.plot(train_losses, '-bx')
        plt.plot(val_losses, '-rx')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(['Training', 'Validation'])
        plt.title('Loss vs. No. of epochs')
        plt.legend()
        plt.savefig("Loss.jpg")
        plt.show()

        plt.plot(p_traines, '-bx')
        plt.plot(p_vales, '-rx')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.legend(['Training', 'Validation'])
        plt.title('accuracy vs. No. of epochs')
        plt.legend()
        plt.savefig("accuracy.jpg")



def main1():
    x_train, y_train, x_val, y_val = train_and_val(50)
    print(x_train)
    print(y_train)





if __name__ == "__main__":
    main()