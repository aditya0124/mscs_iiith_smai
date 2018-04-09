
# coding: utf-8

# ### Q1.1 Implementing Forward-Pass on LeNet-5 Architecture using MNIST Dataset

# In[1]:


import os, sys
import numpy as np
from scipy import signal, misc
from mnist import MNIST
import warnings
warnings.filterwarnings('ignore')


# In[2]:


# Convolutional Layer
# Input Parameters:
# =================
#  layer_size: tuple consisting (depth, height, width)
#  kernel_size: tuple consisting (number_of_kernels, inp_depth, inp_height, inp_width)
#  ntup: tuple of number of nodes in previous layer and this layer
#  params: directory consists of pad_len and stride ... can be extended to pass anything else
class ConvLayer:
    def __init__(self, layer_size, kernel_size, ntup, **params):
        self.depth, self.height, self.width = layer_size
        self.pad = params.get('pad', 0)       # Default Padding = 0
        self.stride = params.get('stride', 1) # Default Stride = 1
        if self.pad < 0:
            print("Invalid padding: pad cannot be negative")
            sys.exit()

        f = np.sqrt(6)/np.sqrt(ntup[0] + ntup[1])
        epsilon = 1e-6
        self.kernel = np.random.uniform(-f, f + epsilon, kernel_size)
        self.bias = np.random.uniform(-f, f + epsilon, kernel_size[0])
        pass

    # Computes the forward pass of Conv Layer.
    # Notation:
    # =========
    # N = batch_size or number of images
    # H, W = Height and Width of input layer
    # D = Depth of input layer
    # K = Number of filters/kernels or depth of this conv layer
    # K_H, K_W = kernel height and Width
    #
    # X: Input data of shape (N, D, H, W)
    # kernel: Weights of shape (K, K_D, K_H, K_W)
    # bias: Bias of each filter.
    def forward(self, X):
        pad_len = self.pad
        stride = self.stride

        N, D, H, W = X.shape
        K, K_D, K_H, K_W = self.kernel.shape

        conv_h = (H - K_H + 2*pad_len) // stride + 1
        conv_w = (W - K_W + 2*pad_len) // stride + 1

        # feature map of a batch
        self.feature_map = np.zeros([N, K, conv_h, conv_w])

        X_padded = np.pad(X,
                          ((0,0), (0,0), (pad_len, pad_len), (pad_len, pad_len)),
                          'constant')

        # stride = 1
        # Rotate kernel by 180
        kernel_180 = np.rot90(self.kernel, 2, (2,3))
        for img in range(N):
            for conv_depth in range(K):
                for inp_depth in range(D):
                    self.feature_map[img, conv_depth] +=                     signal.convolve2d(X_padded[img, inp_depth],
                                      kernel_180[conv_depth, inp_depth],
                                      mode='valid')
                self.feature_map[img, conv_depth] += self.bias[conv_depth]

        return self.feature_map, np.sum(np.square(self.kernel))


# In[3]:


# Rectified Linear Unit (ReLU)
# Activation function after the Convolution Layer
class ReLULayer:
    def __init__(self):
        pass

    # Computes the forward pass of ReLU Layer.
    # X: Input data of any shape
    def forward(self, X):
        self.feature_map = np.maximum(X, 0)
        return self.feature_map, 0


# In[4]:


# Max Pooling Layer
# Only reduce dimensions of height and width by a factor.
# It does not put max filter on same input twice i.e. stride = factor = kernel_dimension
class MaxPoolLayer:
    def __init__(self, **params):
        self.factor = params.get('stride', 2)

    # Computes the forward pass of Max Pooling Layer.
    # Notation:
    # =========
    # N = batch_size or number of images
    # H, W = Height and Width of input layer
    # D = Depth of input layer
    #
    # X: Input data of shape (N, D, H, W)
    def forward(self, X):
        factor = self.factor
        N, D, H, W = X.shape
        #assert H%factor == 0 and W%factor == 0
        self.feature_map = X.reshape(N, D, H//factor, factor, W//factor, factor).max(axis=(3,5))
        #assert self.feature_map.shape == (N, D, H//factor, W//factor)
        return self.feature_map, 0


# In[5]:


# Fully Connected Layer
# layer_size: number of neurons/nodes in fc layer
# kernel_size: kernel of shape (nodes_l1 , nodes_l2)
# where,
#    nodes_l1: number of nodes in previous layer
#    nodes_l2: number of nodes in this fc layer
class FCLayer:
    def __init__(self, layer_size, kernel_size, **params):
        self.nodes = layer_size
        f = np.sqrt(6)/np.sqrt(kernel_size[0] + kernel_size[1])
        epsilon = 1e-6
        self.kernel = np.random.uniform(-f, f + epsilon, kernel_size)
        self.bias = np.random.uniform(-f, f + epsilon, kernel_size[1])
        pass

    # Computing the forward pass of FC Layer.
    # X: Input data of shape (N, nodes_l1)
    # kernel: Weight array of shape (nodes_l1, nodes_l2)
    # bias: Biases of shape (nodes_l2)
    def forward(self, X):
        kernel, bias = self.kernel, self.bias
        self.activations = np.dot(X, kernel) + bias
        #assert self.activations.shape == (X.shape[0], bias.shape[0])
        return self.activations, np.sum(np.square(self.kernel))


# In[6]:


# Sigmoid Layer Computations.
class SigmoidLayer:
    def __init__(self):
        pass

    # X: Input data of any shape
    def forward(self, X):
        self.feature_map = 1.0/(1.0 + np.exp(-X))
        return self.feature_map, 0


# In[7]:


# Softmax Layer Computations.
class SoftmaxLayer:
    def __init__(self):
        pass

    # Computing the forward pass of Softmax Layer.
    # Notation:
    # =========
    #    N: Batch size
    #    C: Number of nodes in Softmax Layer or classes
    #
    #    X: Input data of shape (N, C)
    #    Y: Final output of shape (N, C)
    def forward(self, X):
        dummy = np.exp(X)
        self.Y = dummy / np.sum(dummy, axis=1, keepdims=True)
        return self.Y, 0


# In[8]:


# Creates Lenet-5 architecture
# t_input:  True Training input of shape (N, Depth, Height, Width)
# t_output: True Training output of shape (N, Class_Label)
class LeNet5:
    def __init__(self, t_input, t_output, v_input, v_output):
        # Conv Layer-1
        conv1 = ConvLayer((6, 28, 28), (6, 1, 5, 5), (784, 4704), pad=2, stride=1)
        relu1 = ReLULayer()
        
        # Sub-sampling-1
        pool2 = MaxPoolLayer(stride=2)
        
        # Conv Layer-2
        conv3 = ConvLayer((16, 10, 10), (16, 6, 5, 5), (1176, 1600), pad=0, stride=1)
        relu3 = ReLULayer()
        
        # Sub-sampling-2
        pool4 = MaxPoolLayer(stride=2)
        
        # Fully Connected-1
        fc5 = FCLayer(120, (400, 120))
        sigmoid5 = SigmoidLayer()
        
        # Fully Connected-2
        fc6 = FCLayer(84, (120, 84))
        sigmoid6 = SigmoidLayer()
        
        # Final Output
        output = FCLayer(10, (84, 10))
        softmax = SoftmaxLayer()
        
        self.layers = [conv1, relu1,
                       pool2,
                       conv3, relu3,
                       pool4,
                       fc5, sigmoid5,
                       fc6, sigmoid6,
                       output, softmax]

        self.X = t_input
        self.Y = t_output
        self.Xv = v_input
        self.Yv = v_output

    # Create and save image of feature maps of conv and fc layers
    # X: Input an image of shape (1, 1, 28, 28)
    # layers: List of layers.
    @staticmethod
    def generate_feature_maps(X, layers, digit, batch_string):
        inp = X
        size = (224,224)
        misc.imsave("feature_maps/" + digit + "/" + "ainput_" + digit + batch_string + ".jpeg",
                    misc.imresize(X[0][0], size))
        conv_i = 1
        max_i = 1

        for layer in layers:
            if isinstance(layer, FCLayer) and len(inp.shape) == 4:
                inp, ws = layer.forward(inp.reshape(inp.shape[0],
                                                    inp.shape[1]*inp.shape[2]*inp.shape[3]))
            else:
                inp, ws = layer.forward(inp)

            if isinstance(layer, ReLULayer):
                for channel in range(inp.shape[1]):
                    misc.imsave("feature_maps/" + digit + "/" + "conv" + str(conv_i) + "_c" +
                                str(channel+1) + batch_string + ".jpeg",
                                misc.imresize(inp[0][channel], size))
                conv_i += 1

            if isinstance(layer, MaxPoolLayer):
                for channel in range(inp.shape[1]):
                    misc.imsave("feature_maps/" + digit + "/" + "maxpool" + str(max_i) + "_c" +
                                str(channel+1)+ batch_string + ".jpeg",
                                misc.imresize(inp[0][channel], size))
                max_i += 1

    # Computes final output of neural network by passing
    # output of one layer to another.
    # X: Input
    # layers: List of layers.
    # Out: Final output
    @staticmethod
    def feedForward(X, layers):
        inp = X
        wsum = 0
        for layer in layers:
            if isinstance(layer, FCLayer) and len(inp.shape) == 4:
                inp, ws = layer.forward(inp.reshape(inp.shape[0], inp.shape[1]*inp.shape[2]*inp.shape[3]))
            else:
                inp, ws = layer.forward(inp)
            wsum += ws
        return inp, wsum

    # Train the LeNet-5 (Only Forward Pass).
    # params: parameters including "batch_size" (can be extended)
    def lenet_train(self, **params):
        batch_size  = params.get("batch_size", 50) # Default 50
        print("Running LeNet-5 Forward-Pass on batch_size =", batch_size)
        
        X_train = self.X
        Y_train = self.Y
        assert X_train.shape[0] == Y_train.shape[0]
        
        num_batches = int(np.ceil(X_train.shape[0] / batch_size))
        X_batches = zip(np.array_split(X_train, num_batches, axis=0),
                        np.array_split(Y_train, num_batches, axis=0))

        for x, y in X_batches:
            predictions, weight_sum = LeNet5.feedForward(x, self.layers)

        pass


# In[9]:


class LoadMNISTData:
    lim = 256.0

    def __init__(self, data_path):
        self.path = data_path

    def loadData(self):
        mndata = MNIST(self.path)
        train_img, train_label = mndata.load_training()
        test_img, test_label = mndata.load_testing()
        self.train_img = np.asarray(train_img, dtype='float64') / LoadMNISTData.lim
        self.train_label = np.asarray(train_label)
        self.test_img = np.asarray(test_img, dtype='float64') / LoadMNISTData.lim
        self.test_label = np.asarray(test_label)

        print("\nTraining Images Size (train_img):         ", self.train_img.shape)
        print("Training Image Labels Size (train_label): ", self.train_label.shape)
        print("Test Images Size (test_img):              ", self.test_img.shape)
        print("Test Images Labels Size (test_label):     ", self.test_label.shape)


# In[10]:


print("Running Forward Pass on MNIST dataset using LeNet-5 Architecture")
cwd = os.getcwd()
dataset = LoadMNISTData(cwd)
print("\nLoading MNIST dataset ...")
dataset.loadData()
N = 50000

X_train = dataset.train_img[range(0, N)].reshape(N, 1, 28, 28)
Y_train = np.zeros((N, 10))
Y_train[np.arange(N), dataset.train_label[range(0, N)]] = 1

M = 10000
X_valid = dataset.train_img[N:].reshape(M, 1, 28, 28)
Y_valid = np.zeros((M, 10))
Y_valid[np.arange(M), dataset.train_label[N:]] = 1

print("\nTraining set:   ", X_train.shape, Y_train.shape)
print("Validation set: ", X_valid.shape, Y_valid.shape)

### Create LeNet5 object ###
print("\nCreating LeNet-5 layers ...")
cnn_lenet = LeNet5(X_train, Y_train, X_valid, Y_valid)

### Training LeNet5 ###
cnn_lenet.lenet_train(batch_size=128)

batch_string = "_batch_128"
print("Generating feature maps for the forward-pass")
### Visualize Feature Maps of conv layers for a image of a digit ###
for digit, index in zip(range(10),[1, 8, 16, 7, 2, 0, 18, 91, 31, 45]):
    cnn_lenet.generate_feature_maps(X_train[index].reshape(1, 1, 28, 28),
                                     cnn_lenet.layers,
                                     str(digit),
                                     batch_string)
print("Done !!!")

