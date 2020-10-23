import torch
import numpy as np
import os
import matplotlib.pyplot as plt

from model1_gpu import Model1
from dataset import Dataset


if __name__ == '__main__':
    root = 'digit_data'
    train_root = os.path.join(root, 'train')
    val_root = os.path.join(root, 'val')

    num_epoch = 200
    learning_rate = .001

    model = Model1(10, learning_rate)

    train_dset = Dataset(train_root)
    val_dset = Dataset(val_root)
    train_dset.load_numpy_data(augment=True)
    val_dset.load_numpy_data(augment=True)

    train_images, train_labels = train_dset.images, train_dset.labels
    val_images, val_labels = val_dset.images, val_dset.labels

    assert len(train_images) == len(train_labels)
    assert len(val_images) == len(val_labels)

    len_train_data = len(train_images)
    len_val_data = len(val_images)

    train_loss_list = []
    train_acc_list = []

    val_loss_list = []
    val_acc_list = []

    for epoch in range(num_epoch):
        print('[{}/{}] '.format(epoch + 1, num_epoch), end='')
        train_loss = 0
        train_acc = 0

        val_loss = 0
        val_acc = 0

        for i in range(len_train_data):
            x_, y_ = train_images[i], train_labels[i]
            x_, y_ = torch.Tensor(x_).cuda(), torch.Tensor(y_).cuda()
            output = model.forward(x_)
            loss = model.cross_entropy(output, y_).sum() / len(y_)
            model.backward(x_, output, y_)

            if torch.argmax(output) == torch.argmax(y_):
                train_acc += 1

            train_loss += loss

        for i in range(len_val_data):
            x_, y_ = val_images[i], val_labels[i]
            output = model.forward(x_)
            loss = model.cross_entropy(output, y_).sum() / len(y_)

            if torch.argmax(output) == torch.argmax(y_):
                val_acc += 1

            val_loss += loss

        train_loss_list.append(train_loss / len_train_data)
        train_acc_list.append(train_acc / len_train_data)

        val_loss_list.append(val_loss / len_val_data)
        val_acc_list.append(val_acc / len_val_data)

        print('<train_loss> {} <train_acc> {} <val_loss> {} <val_acc> {}'.format(train_loss_list[-1], train_acc_list[-1], val_loss_list[-1], val_acc_list[-1]))

    plt.figure(1)
    plt.title('Train/Validation Loss')
    plt.plot([i for i in range(num_epoch)], train_loss_list, 'r-', label='train')
    plt.plot([i for i in range(num_epoch)], val_loss_list, 'b-', label='val')
    plt.legend()

    plt.figure(2)
    plt.title('Train/Validation Accuracy')
    plt.plot([i for i in range(num_epoch)], train_acc_list, 'r-', label='train')
    plt.plot([i for i in range(num_epoch)], val_acc_list, 'b-', label='val')
    plt.legend()

    plt.show()
