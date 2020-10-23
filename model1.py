import numpy as np


class Model1:
    def __init__(self, n_class=10):
        self.learning_rate = .001
        self.n_class = n_class
        self.n_hidden = 2
        self.conv1 = np.zeros((4, 8, 8))
        self.conv2 = np.zeros((8, 4, 4))
        self.dense = np.zeros(8 * 4 * 4)
        self.classifier = np.zeros(n_class)
        # self.W_conv1 = np.random.random((4, 3, 3))
        self.W_conv1 = np.random.normal(0, .01, (4, 3, 3))
        # self.W_conv2 = np.random.random((8, 3, 3))
        self.W_conv2 = np.random.normal(0, .01, (8, 3, 3))
        # self.W_dense = np.random.random((8 * 4 * 4, 10))
        self.W_dense = np.random.normal(0, .01, (8 * 4 * 4, 10))
        self.padding = Model1.padding
        self.maxpool = Model1.maxpool
        self.relu = Model1.relu
        self.softmax = Model1.softmax
        self.loss = Model1.cross_entropy

    def forward(self, x):
        # Input
        # (Channel, H, W) : (1, 16, 16)

        # 1st hidden layer
        # (Channel, H, W) : (4, 8, 8)
        x = self.padding(x, (x.shape[1] + 1, x.shape[2] + 1))
        H1 = np.zeros((4, 16, 16))
        for c_i in range(x.shape[0]):
            for c_o in range(H1.shape[0]):
                for h in range(H1.shape[1]):
                    for w in range(H1.shape[2]):
                        for k1 in range(3):
                            for k2 in range(3):
                                H1[c_o, h, w] += x[c_i, h + k1, w + k2] * self.W_conv1[c_o, k1, k2]
        H1 = self.maxpool(H1)
        self.conv1 = H1 = self.relu(H1)

        # 2nd hidden layer
        # (Channel, H, W) : (8, 4, 4)
        H1 = self.padding(H1, (H1.shape[1] + 1, H1.shape[2] + 1))
        H2 = np.zeros((8, 8, 8))
        for c_i in range(H1.shape[0]):
            for c_o in range(H2.shape[0]):
                for h in range(H2.shape[1]):
                    for w in range(H2.shape[2]):
                        for k1 in range(3):
                            for k2 in range(3):
                                H2[c_o, h, w] += H1[c_i, h + k1, w + k2] * self.W_conv2[c_o, k1, k2]
        H2 = self.maxpool(H2)
        self.conv2 = H2 = self.relu(H2)

        # Classifier
        # (8 * 4 * 4) -> (10)
        self.dense = H2_flat = H2.reshape(-1)
        y_ = np.zeros(10)
        for i in range(H2_flat.shape[0]):
            for j in range(y_.shape[0]):
                y_[j] += H2_flat[i] * self.W_dense[i, j]
        self.classifier = y_
        y_ = self.softmax(y_)

        return y_

    def backward(self, input, output, predict):
        x_, y_, pred = input, output, predict

        D_out = Model1.cross_entropy_derv(y_, pred)

        D_classifier = np.zeros(self.classifier.shape)
        softmax_derv_mat = Model1.softmax_derv(self.classifier)
        for i in range(len(D_classifier)):
            D_classifier[i] = softmax_derv_mat[i].sum()

        D_W_dense = np.zeros(self.W_dense.shape)
        for m in range(self.W_dense.shape[0]):
            for n in range(self.W_dense.shape[1]):
                D_W_dense[m, n] = self.learning_rate * D_classifier[n] * self.dense[m]

        D_dense = np.zeros(self.dense.shape)
        for i in range(D_dense.shape[0]):
            for j in range(D_classifier.shape[0]):
                D_dense[i] += D_classifier[j] * self.W_dense[i, j]
            D_dense *= Model1.relu_derv(self.dense[i])

        self.W_dense -= D_W_dense

        D_conv2 = D_dense.reshape(self.conv2.shape)

        D_W_conv2 = np.zeros(self.W_conv2.shape)
        for m in range(self.W_conv2.shape[1]):
            for n in range(self.W_conv2.shape[2]):
                for c1 in range(self.conv1.shape[0]):
                    for c2 in range(self.conv2.shape[0]):
                        for i in range(self.conv2.shape[1]):
                            for j in range(self.conv2.shape[2]):
                                D_W_conv2[c2, m, n] += self.learning_rate * D_conv2[c2, i, j] * self.conv1[c1, i + m, j + n]

        W_conv2_flip = np.flip(self.W_conv2)
        D_conv2_pad = self.padding(D_conv2, (D_conv2.shape[1] + 3, D_conv2.shape[2] + 3), 'zero')
        D_conv1 = np.zeros(self.conv1.shape)
        for c1 in range(self.conv1.shape[0]):
            for c2 in range(self.conv2.shape[0]):
                for i in range(self.conv1.shape[1]):
                    for j in range(self.conv1.shape[2]):
                        for m in range(W_conv2_flip.shape[1]):
                            for n in range(W_conv2_flip.shape[2]):
                                D_conv1[c1, i, j] += D_conv2_pad[c2, i + m, j + n] * W_conv2_flip[c2, m, n]

        self.W_conv2 -= D_W_conv2

        D_conv1 *= Model1.relu_derv(self.conv1)

        D_W_conv1 = np.zeros(self.W_conv1.shape)
        for m in range(self.W_conv1.shape[1]):
            for n in range(self.W_conv1.shape[2]):
                for cx in range(x_.shape[0]):
                    for c1 in range(self.conv1.shape[0]):
                        for i in range(self.conv1.shape[1]):
                            for j in range(self.conv1.shape[2]):
                                D_W_conv1[c1, m, n] += self.learning_rate * D_conv1[c1, i, j] * x_[cx, i + m, j + n]

        self.W_conv1 -= D_W_conv1

    @staticmethod
    def padding(x, shape, mode='same'):
        if x.shape[1] < shape[0]:
            dif = shape[0] - x.shape[1]
            for i in range(dif * 2):
                if mode == 'same':
                    pad_ = np.array(x[:, 0, :]) if i % 2 == 0 else np.array(x[:, -1, :])
                    pad_ = np.expand_dims(pad_, 1)
                elif mode == 'zero':
                    pad_ = np.zeros((x.shape[0], 1, x.shape[2]))
                x = np.concatenate([pad_, x], axis=1) if i % 2 == 0 else np.concatenate([x, pad_], axis=1)
        if x.shape[2] < shape[1]:
            dif = shape[1] - x.shape[2]
            for i in range(dif * 2):
                if mode == 'same':
                    pad_ = np.array(x[:, :, 0]) if i % 2 == 0 else np.array(x[:, :, -1])
                    pad_ = np.expand_dims(pad_, 2)
                elif mode == 'zero':
                    pad_ = np.zeros((x.shape[0], x.shape[1], 1))
                x = np.concatenate([pad_, x], axis=2) if i % 2 == 0 else np.concatenate([x, pad_], axis=2)

        return x

    @staticmethod
    def maxpool(x, kernel=(2, 2), stride=2):
        h_out = int(np.floor(((x.shape[1] - kernel[0]) / stride) + 1))
        w_out = int(np.floor(((x.shape[2] - kernel[1]) / stride) + 1))
        out_ = np.zeros((x.shape[0], h_out, w_out))

        for c in range(x.shape[0]):
            for h in range(0, out_.shape[1]):
                for w in range(0, out_.shape[2]):
                    out_[c, h, w] = np.max(x[c, 2 * h:2 * h + kernel[0], 2 * w:2 * w + kernel[1]])

        return out_

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def relu_derv(x):
        return x > 0

    @staticmethod
    def softmax(x):
        up = np.exp(x)
        down = np.exp(x).sum()

        return up / down

    @staticmethod
    def softmax_derv(x):
        matrix = np.zeros((len(x), len(x)))
        for i in range(len(x)):
            for j in range(len(x)):
                matrix[i, j] = Model1.softmax(x[i]) * (1 - Model1.softmax(x[i])) if i == j else -1 * Model1.softmax(x[i]) * Model1.softmax(x[j])

        return matrix

    @staticmethod
    def cross_entropy(output, predict):
        y_, pred = output, predict

        return -1 * pred * np.log2(y_)

    @staticmethod
    def cross_entropy_derv(output, predict):
        return output - predict

