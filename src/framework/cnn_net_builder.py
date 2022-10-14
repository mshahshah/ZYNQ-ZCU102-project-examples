import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from PIL import Image
import torch.nn.functional as F
import numpy as np
import torchvision.models as models
from numpy import savetxt
from numpy import asarray
from dnn_tools import *
from pathlib import Path
import time
import matplotlib as ml
import matplotlib.pyplot as plt
import csv
from collections import namedtuple


np.set_printoptions(suppress= True)
flag = False
model_flag = False

output_layers_test = {}
input_sample_test ={}
input_test_index = 0


QTensor = namedtuple('QTensor', ['tensor', 'scale', 'zero_point'])



if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

class LeNet(nn.Module):
    def __init__(self, lyrs=[{"fi":1, "fo":6, "k":5, "s":1},
                             {"fi":6, "fo":16, "k":5, "s":1},
                             {"fi":16*5*5, "fo":120, "k":1, "s":1},
                             {"fi":120, "fo":84, "k":1, "s":1},
                             {"fi":84, "fo":10, "k":1, "s":1}]):
        super(LeNet, self).__init__()
        self.model = nn.Module()
        for lyr_n, lyr in enumerate(lyrs):
            if lyr['type'].lower() == 'conv':
                self.model.add_module("conv{}".format(lyr_n), nn.Conv2d(in_channels=lyr["lyr_in"], out_channels=lyr["lyr_out"], kernel_size=lyr["w_ker"]))
            elif lyr['type'].lower() == 'fc':
                self.model.add_module("fc{}".format(lyr_n), nn.Linear(lyr["lyr_in"], lyr["lyr_out"]))
        # input channel = 1, output channel = 6, kernel_size = 5
        # input size = (32, 32), output size = (28, 28)
        #self.conv1 = nn.Conv2d(in_channels=lyrs[0]["fi"], out_channels=lyrs[0]["fo"], kernel_size=lyrs[0]["k"])
        # input channel = 6, output channel = 16, kernel_size = 5
        # input size = (14, 14), output size = (10, 10)
        #self.conv2 = nn.Conv2d(in_channels=lyrs[1]["fi"], out_channels=lyrs[1]["fo"], kernel_size=lyrs[1]["k"])
        # input dim = 16*5*5, output dim = 120
        #self.fc1 = nn.Linear(lyrs[2]["fi"], lyrs[2]["fo"])
        # input dim = 120, output dim = 84
        #self.fc2 = nn.Linear(lyrs[3]["fi"], lyrs[3]["fo"])
        # input dim = 84, output dim = 10
        #self.fc3 = nn.Linear(lyrs[4]["fi"], lyrs[4]["fo"])

    def forward(self, x):
        if flag :
            # pool size = 2
            # input size = (28, 28), output size = (14, 14), output channel = 6
            out_layer = {}
            x = F.max_pool2d(F.relu(self.model.conv1(x)), 2)
            out_layer['conv1']= x
            # pool size = 2
            # input size = (10, 10), output size = (5, 5), output channel = 16
            x = F.max_pool2d(F.relu(self.model.conv3(x)), 2)
            out_layer['conv2']= x
            # flatten as one dimension
            x = x.view(x.size()[0], -1)
            # input dim = 16*5*5, output dim = 120
            x = F.relu(self.model.fc5(x))
            out_layer['fc1']= x
            # input dim = 120, output dim = 84
            x = F.relu(self.model.fc6(x))
            out_layer['fc2']=x
            # input dim = 84, output dim = 10
            x = self.model.fc7(x)
            out_layer['fc3']= x
            return x, out_layer
        else:
            x = F.max_pool2d(F.relu(self.model.conv1(x)), 2)
            # pool size = 2
            # input size = (10, 10), output size = (5, 5), output channel = 16
            x = F.max_pool2d(F.relu(self.model.conv3(x)), 2)
            # flatten as one dimension
            x = x.view(x.size()[0], -1)
            # input dim = 16*5*5, output dim = 120
            x = F.relu(self.model.fc5(x))
            # input dim = 120, output dim = 84
            x = F.relu(self.model.fc6(x))
            # input dim = 84, output dim = 10
            x = self.model.fc7(x)
            return x


def train(model, optimizer, epoch, train_loader, log_interval):
    # State that you are training the model
    model.train()
    model.to(device) #Shah
    # define loss function
    loss_fn = torch.nn.CrossEntropyLoss()
    # Iterate over batches of data
    for batch_idx, (data, target) in enumerate(train_loader):
        # Wrap the input and target output in the `Variable` wrapper
        data, target = Variable(data), Variable(target)
        data = data.to(device) #Shah
        target = target.to(device) #Shah
        # Clear the gradients, since PyTorch accumulates them
        optimizer.zero_grad()
        # Forward propagation
        output = model(data)
        loss = loss_fn(output, target)
        # Backward propagation
        loss.backward()
        # Update the parameters(weight,bias)
        optimizer.step()
        # print log
        if batch_idx % log_interval == 0:
            print('Train set, Epoch {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),loss.item()))


def test(model, epoch, test_loader):
    # State that you are testing the model; this prevents layers e.g. Dropout to take effect
    model.eval()
    global input_test_index
    # Init loss & correct prediction accumulators
    test_loss = 0
    correct = 0
    # define loss function
    loss_fn = torch.nn.CrossEntropyLoss(size_average=False)
    # Iterate over data
    # Forward propagation
    if flag:
        for data, target in test_loader:
            if input_test_index ==100 :
                break
            data, target = Variable(data), Variable(target)
            data = data.to(device)#Shah
            target = target.to(device)#Shah
            #plt.imshow(data.numpy()[0][0])
            #plt.show()
            output,out_layer_test = model(data)
            input_sample_test ['test'+str(input_test_index)]= data
            output_layers_test['test'+str(input_test_index)]= out_layer_test
            input_test_index += 1
            test_loss += loss_fn(output, target).item()

            # Get the index of the max log-probability (the predicted output label)
            pred = np.argmax(output.cpu().data, axis=1)
            # If correct, increment correct prediction accumulator
            correct = correct + np.equal(pred, target.cpu().data).sum()
        test_loss /= input_test_index
        print('\nTest set with {} samples , Error:% {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        input_test_index, test_loss*100, correct, input_test_index,
        100. * correct / input_test_index))
        
    else :
        for data, target in test_loader:
            data, target = Variable(data), Variable(target)
            data = data.to(device)#Shah
            target = target.to(device)#Shah
            output = model(data)
            # Calculate & accumulate loss
            test_loss += loss_fn(output, target).item()

            # Get the index of the max log-probability (the predicted output label)
            pred = np.argmax(output.cpu().data, axis=1)

            # If correct, increment correct prediction accumulator
            correct = correct + np.equal(pred, target.cpu().data).sum()
        # Print log
        test_loss /= len(test_loader.dataset)
        print('\nTest set, Epoch {} , Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(epoch,
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return output_layers_test




def quantize_tensor(x, num_bits=8):
    qmin = 0.
    qmax = 2.**num_bits - 1.
    min_val, max_val = x.min(), x.max()

    scale = (max_val - min_val) / (qmax - qmin)

    initial_zero_point = qmin - min_val / scale

    zero_point = 0
    if initial_zero_point < qmin:
        zero_point = qmin
    elif initial_zero_point > qmax:
        zero_point = qmax
    else:
        zero_point = initial_zero_point

    zero_point = int(zero_point)
    q_x = zero_point + x / scale
    q_x.clamp_(qmin, qmax).round_()
    q_x = q_x.round().byte()
    return QTensor(tensor=q_x, scale=scale, zero_point=zero_point)


def dequantize_tensor(q_x):
    return q_x.scale * (q_x.tensor.float() - q_x.zero_point)


def calcScaleZeroPoint(min_val, max_val, num_bits=8):
    # Calc Scale and zero point of next
    qmin = 0.
    qmax = 2. ** num_bits - 1.

    scale_next = (max_val - min_val) / (qmax - qmin)

    initial_zero_point = qmin - min_val / scale_next

    zero_point_next = 0
    if initial_zero_point < qmin:
        zero_point_next = qmin
    elif initial_zero_point > qmax:
        zero_point_next = qmax
    else:
        zero_point_next = initial_zero_point

    zero_point_next = int(zero_point_next)

    return scale_next, zero_point_next


def quantizeLayer(x, layer, stat, scale_x, zp_x):
    # for both conv and linear layers
    W = layer.weight.data
    B = layer.bias.data

    # scale_x = x.scale
    # zp_x = x.zero_point
    w = quantize_tensor(layer.weight.data)
    b = quantize_tensor(layer.bias.data)

    layer.weight.data = w.tensor.float()
    layer.bias.data = b.tensor.float()

    ####################################################################
    # This is Quantisation !!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    scale_w = w.scale
    zp_w = w.zero_point

    scale_b = b.scale
    zp_b = b.zero_point

    scale_next, zero_point_next = calcScaleZeroPoint(min_val=stat['min'], max_val=stat['max'])

    # Perparing input by shifting
    X = x.float() - zp_x
    layer.weight.data = scale_x * scale_w * (layer.weight.data - zp_w)
    layer.bias.data = scale_b * (layer.bias.data + zp_b)

    # All int

    x = (layer(X) / scale_next) + zero_point_next

    x = F.relu(x)

    # Reset
    layer.weight.data = W
    layer.bias.data = B

    return x, scale_next, zero_point_next


def count_parameters(model):
    total_param = 0
    total_weight = 0
    i=0
    for name, param in model.named_parameters():
        i +=1
        if param.requires_grad:
            num_param = np.prod(param.size())
            if param.dim() > 1:
                print(name, ':', 'x'.join(str(x) for x in list(param.size())), '=', num_param)
            else:
                print(name, ':', num_param)
            if i<3 :
                total_weight += num_param
                if i==2:
                    print('Number of parameters :', total_weight,'\nWeight size :', total_weight*4,'\n')
            else:
                total_weight = num_param
                i=1
            total_param += num_param
    return total_param


def quantForward( model, x):#, stats):
    # Quantise before inputting into incoming layers
    x = quantize_tensor_act(x, stats['conv1'])

    x, scale_next, zero_point_next = quantizeLayer(x.tensor, model.model.conv1, stats['conv2'], x.scale, x.zero_point)

    x = F.max_pool2d(x, 2, 2)

    x, scale_next, zero_point_next = quantizeLayer(x, model.model.conv3, stats['fc1'], scale_next, zero_point_next)

    x = F.max_pool2d(x, 2, 2)

    x = x.view(-1, 4 * 4 * 50)

    x, scale_next, zero_point_next = quantizeLayer(x, model.model.fc5, stats['fc2'], scale_next, zero_point_next)

    # Back to dequant for final layer
    x = dequantize_tensor(QTensor(tensor=x, scale=scale_next, zero_point=zero_point_next))

    x = model.model.fc6(x)

    x = model.model.fc7(x)

    return F.log_softmax(x, dim=1)

def convert_to_fixed(float_weights):

    precision = 16
    utils = dnn_tools()
    fixed_weights= {}
    for key, value in float_weights.items():
        fixed_weights[key] = utils.float2int(float_weights[key],precision,'ceil')
    np.save('fixed_weights', fixed_weights)
    return fixed_weights

def plot_dist_weight(plt_dict):
    tmp = {}
    for index,(key, value) in enumerate(plt_dict.items()):
        tmp[key]= (value+1)*128
        #plt.subplot(2, 5, index+1)
        #plt.subplots_adjust(hspace=0.4, wspace=0.4)
        
        plt.hist(tmp[key].reshape(1,-1)[0],bins=10)
        plt.title(key)
        plt.ylabel('Probability')
        plt.xlabel('weights')
        plt.savefig('hist_'+str(key)+'.jpg')
        plt.close()

def plot_output(plt_out):
    tmp = {}
    for key, value in plt_out.items():
        tmp[key]= (value+1)*128
        plt.hist(tmp[key].detach().reshape(1,-1)[0],bins=10)
        plt.title(key)
        plt.ylabel('Probability')
        plt.xlabel('output')
        plt.savefig('output_'+str(key)+'.jpg')
        plt.close()

def save_csv(dict1,stri):
    #with open(stri+'.csv', 'w') as f:
    #if dict1.values() == weights.values() : stri='float_weight_'
    #elif dict1.items() == fixed_weights.items() : stri = 'fixed_weights'
    #elif dict1 == output_layers_test : stri = 'output_'
    dict1_tmp ={}
    for key in dict1.keys():
        if isinstance(dict1[key], torch.Tensor):
            dict1_tmp[key] = dict1[key].detach().numpy()
        elif not isinstance(dict1[key],np.ndarray):
            dict1_tmp[key] = np.array(dict1[key])
        else:
            dict1_tmp[key] = dict1[key]
        try:
            np.savetxt(stri+key+'.csv',dict1_tmp[key].reshape(100,-1),delimiter=',',fmt="%5f")
        except:
             np.savetxt(stri+key+'.csv',dict1_tmp[key].reshape(1,-1),delimiter=',',fmt="%5f")
            #for k in dict1[key].keys():
                #np.savetxt(stri+key+k+'.csv',dict1[key].detach.numpy.reshape(100,-1),delimiter=',',fmt="%5f")
                #f.write("{}:\n,{}\n".format(key,np.array_str(dict1[key])))
        #f.close()


    

def create_coe_file(path='', array=[1]):
    lines = []
    lines.append('memory_initialization_radix=10;')
    lines.append('memory_initialization_vector=')
    for i in array.flatten():
        lines.append(str(i))
    lines.append(';')

    np.savetxt(path,lines, fmt='%s')


class cnn_net_builder:
    def __init__(self,cfg):
        self.cfg = cfg
        self.utils = utils(cfg)
        self.momentum =  0.5
        self.learning_rate= 0.01
        self.random_seed = 1
        self.log_interval = 100
        self.n_epochs=1
        self.train_batch_size = 64
        self.test_batch_size = 20
        torch.backends.cudnn.enabled = False
        torch.manual_seed(self.random_seed)

    def selec_model(self,arch, lyrs):
        if arch == 'lenet':
            return LeNet(lyrs)


    def load_data(self):
        train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('files', train=True, download=True,
                                    transform= transforms.Compose([
                                    transforms.Resize((32, 32)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(
                                        (0.1307,), (0.3081,))
                                    ])),
        batch_size=self.train_batch_size, shuffle=True)

        test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('files', train=False, download=True,
                                    transform=transforms.Compose([
                                    transforms.Resize((32, 32)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(
                                        (0.1307,), (0.3081,))
                                    ])),
        batch_size=self.test_batch_size, shuffle=True)
        return (train_loader, test_loader)

    def save_test_images(self, test_loader):
        f_path = os.path.join(self.cfg.paths.test_files, 'test_images')
        np.save(f_path, np.array(test_loader.batch_sampler.sampler.data_source.test_data))

    def train_model(self, model, train_loader, test_loader):
        file_path = Path(self.cfg.paths.design_model + "/trained_model.pth")
        if file_path.is_file() and not self.cfg.design_setting.training_setting['retrain']:
            model.load_state_dict(torch.load(file_path, map_location=torch.device('cpu')))
            model.eval()
            model_flag = True
        else:
            if not os.path.exists(self.cfg.paths.design_model):
                os.mkdir(self.cfg.paths.design_model)
            model_flag = False

        optimizer = optim.SGD(model.parameters(), lr=self.learning_rate, momentum=self.momentum)
        print('\nNumber of parameters and size of weights for each layer:\n')
        print('Number of total trainable parameters =', count_parameters(model))

        if model_flag == False:
            for epoch in range(1, self.n_epochs + 1):
                if epoch < 2:
                    start_time_train = time.time()
                    train(model, optimizer, epoch, train_loader, log_interval=self.log_interval)
                    train_time = time.time() - start_time_train
                else:
                    train(model, optimizer, epoch, train_loader, log_interval=self.log_interval)
                test(model, epoch, test_loader)

            torch.save(model.state_dict(), file_path)

        return model

    def test_model(self, model, test_loader):
        flag = True
        model.to(device)
        start_time_test = time.time()
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                #o2 = quantForward(model, data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        accuracy_perc = 100. * correct / len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset), accuracy_perc))
        print("Test time for one sample : %s Milliseconds\n" % (((time.time() - start_time_test) / 100) * 1000))
        self.test_loss = test_loss
        self.accuracy_perc = accuracy_perc
        #plot_output(output_layers_test['test1'])



    def export_trained_weights(self, model):
        if not os.path.exists(self.cfg.paths.dnn_weights):
            os.mkdir(self.cfg.paths.dnn_weights)

        [fixed_weights,fload_weights] = self.extract_n_save_weights(model)
        self.save_fixed_point_csv_files(fixed_weights)
        self.save_floating_point_csv_files(fload_weights)
        #plot_dist_weight(weights)

    # save the weights in a dictionary and save the dictionary in a file
    def extract_n_save_weights(self,model):
        scale = 2 ** self.cfg.design_variable_types['ker_t']
        float_weights = {}
        fixed_wieghts = {}
        for name, param in model.named_parameters():
            label = name.replace('.', '')
            float_weights[label] = param.detach().cpu().numpy()
            fixed_wieghts[label] = float_weights[label].dot(scale).astype(int)
        np.save(self.cfg.files.trained_model_weights_fixed+'.npy', fixed_wieghts)
        np.save(self.cfg.files.trained_model_weights_float+'.npy', float_weights)
        return fixed_wieghts, float_weights

    def create_training_testing_report(self):
        fname = os.path.join(self.cfg.paths.test_files, "report_{}.txt".format(self.accuracy_perc))
        flines = ['Network under test: {}\n'.format(self.cfg.design_setting.design_model)]
        flines.append('Learning Rate     : {}'.format(self.learning_rate))
        flines.append('Log interval      : {}'.format(self.log_interval))
        flines.append('Number of epochs  : {}'.format(self.n_epochs))
        flines.append('Test_batch_size   : {}'.format(self.test_batch_size))
        flines.append('Train_batch_size  : {}'.format(self.train_batch_size))
        flines.append('Test_loss         : {}'.format(self.test_loss))
        flines.append('Accuracy %        : {}'.format(self.accuracy_perc))
        flines.append('Test_sample       : {}'.format(self.test_sample))
        flines.append('Predicted label   : {}'.format(self.predicted_label))
        self.utils.save_list_to_file(fname, flines)



    def save_fixed_point_csv_files(self, fixed_weights):
        flatten_weights = np.array([])
        for key in fixed_weights.keys():
            with open(os.path.join(self.cfg.paths.dnn_weights, '{}_fixed.txt'.format(key)),'w') as outfile:
                if fixed_weights[key].ndim == 1:
                    np.savetxt(outfile, fixed_weights[key], delimiter=",", fmt='%7d')
                    flatten_weights = np.append(flatten_weights, fixed_weights[key].flatten().astype(int))
                elif fixed_weights[key].ndim == 2:
                    np.savetxt(outfile, fixed_weights[key], delimiter=",", fmt='%7d')
                    flatten_weights = np.append(flatten_weights, fixed_weights[key].flatten().astype(int))
                    
                    np.savetxt(os.path.join(self.cfg.paths.dnn_weights, '{}_fixed1D.txt'.format(key)), fixed_weights[key].flatten(), delimiter=",", fmt='%7d')
                else:
                    for D1, data_sliceD1 in enumerate(fixed_weights[key]):
                        for D2, data_sliceD2 in enumerate(data_sliceD1):
                            flatten_weights = np.append(flatten_weights, data_sliceD2.flatten().astype(int))
                            outfile.write('Lyr {}:{} -> Array shape: {}\n'.format(D1, D2, data_sliceD2.shape))
                            np.savetxt(outfile, data_sliceD2, delimiter=",", fmt='%7d')
                            outfile.write('\n')
                outfile.close()

        with open(os.path.join(self.cfg.paths.test_files,'kernels.txt'), 'w') as outfile:
            np.savetxt(outfile, flatten_weights, delimiter=",", fmt='%7d')
            outfile.write('\n')
        create_coe_file(os.path.join(self.cfg.paths.test_files, 'kernels.coe'), array=flatten_weights.astype(int))
        print("PYTHON : All weights are saved in csv and npy format")

    def save_floating_point_csv_files(self, float_weights):
        flatten_weights = np.array([])
        for key in float_weights.keys():
            with open(os.path.join(self.cfg.paths.dnn_weights, '{}_float.txt'.format(key)),'w') as outfile:
                if float_weights[key].ndim == 1:
                    np.savetxt(outfile, float_weights[key], delimiter=",", fmt='%10.5f')
                    flatten_weights = np.append(flatten_weights, float_weights[key].flatten())
                elif float_weights[key].ndim == 2:
                    np.savetxt(outfile, float_weights[key], delimiter=",", fmt='%10.5f')
                    flatten_weights = np.append(flatten_weights, float_weights[key].flatten())
                else:
                    for D1, data_sliceD1 in enumerate(float_weights[key]):
                        for D2, data_sliceD2 in enumerate(data_sliceD1):
                            flatten_weights = np.append(flatten_weights, data_sliceD2.flatten())
                            outfile.write('Lyr {}:{} -> Array shape: {}\n'.format(D1,D2,data_sliceD2.shape))
                            np.savetxt(outfile, data_sliceD2, delimiter=",", fmt='%10.5f')
                            outfile.write('\n')
                outfile.close()
        
        with open(os.path.join(self.cfg.paths.test_files,'kernels_fload.txt'), 'w') as outfile:
            np.savetxt(outfile, flatten_weights, delimiter=",", fmt='%9.6f')
            outfile.write('\n')

        print("PYTHON : All weights are saved in csv and npy format")

    def test1pic(self,model, test_loader, sample):
        sampleDirName = os.path.join(self.cfg.paths.sim, 'sample' + str(sample))
        if os.path.exists(sampleDirName):
            shutil.rmtree(sampleDirName)
            os.mkdir(sampleDirName)
        else:
            os.mkdir(sampleDirName)

        fname = os.path.join(sampleDirName, 'TestImage')
        tdata = test_loader.dataset[sample][0]
        
        test_data = np.array(tdata)
        temp = abs(test_data).max()
        image = np.multiply(test_data, 250 / temp).astype(int)
        np.savetxt(fname+'.csv', image[0], delimiter=",", fmt='%5d')
        plt.imshow(image[0])
        plt.savefig(fname+'.jpg')
        device = torch.device("cpu")
        tdata.to(device)
        model.to(device)
        tstImage = Variable(tdata.unsqueeze(0))
        predicted_label = model(tstImage)
        layer_outputs = self.save_LeNet_intermediate_data(sampleDirName, model, tstImage)
        np.save(os.path.join(sampleDirName, 'TestImage_float.npy'), test_data[0])
        np.save(os.path.join(sampleDirName, 'TestImage_fixed.npy'), image)
        np.save(os.path.join(sampleDirName, 'layers_output.npy'), layer_outputs)

        np.savetxt(os.path.join(self.cfg.paths.test_files, 'in_data_float.txt'), test_data[0].flatten(), fmt='%9.5f')
        np.savetxt(os.path.join(self.cfg.paths.test_files, 'in_data.txt'), image.flatten(), fmt='%7d')

        np.savetxt(os.path.join(sampleDirName, 'predicted_label.txt'), predicted_label.detach().numpy(), delimiter=",", fmt='%5d')
        print("PYTHON : The predicted label at sample {} is {}".format(sample, predicted_label.argmax()))
        self.test_sample = sample
        self.predicted_label = predicted_label.argmax()
        create_coe_file(os.path.join(self.cfg.paths.test_files, 'in_data.coe'), array=image)

        return predicted_label



    def save_LeNet_intermediate_data(self,sampleDirName, model, tstImage):
        layer_outputs = {}
        layer_outputs['L0'] = tstImage
        layer_outputs['L1_C'] = model.model.conv1(layer_outputs['L0'])
        layer_outputs['L1_A'] = F.relu(layer_outputs['L1_C'])
        layer_outputs['L1_P'] = F.max_pool2d(layer_outputs['L1_A'], (2, 2))

        layer_outputs['L2_C'] = model.model.conv3(layer_outputs['L1_P'])
        layer_outputs['L2_A'] = F.relu(layer_outputs['L2_C'])
        layer_outputs['L2_P'] = F.max_pool2d(layer_outputs['L2_A'], (2, 2))

        layer_outputs['L3'] = model.model.fc5(layer_outputs['L2_P'].flatten())
        layer_outputs['L4'] = model.model.fc6(layer_outputs['L3'])
        layer_outputs['L5'] = model.model.fc7(layer_outputs['L4'])

        scale = 2 ** (self.cfg.design_variable_types['ker_t']-1)
        max_scale = 0
        for key in layer_outputs.keys():
            max_scale = max(layer_outputs[key].abs().max(),max_scale)
        for key in layer_outputs.keys():
            temp = layer_outputs[key].detach().numpy()
            values = temp.dot(scale/max_scale.detach().numpy()).astype(int)
            np.savetxt(os.path.join(sampleDirName, '{}.txt'.format(key)), values.flatten(), delimiter=",", fmt='%5d')
        return layer_outputs



if __name__ == '__main__':

    exit()





