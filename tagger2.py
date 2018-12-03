import helper_funcs2 as my_helps_2


import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D

from torch.utils.data import TensorDataset

import sys

###############################################################
# Function Name:
# Function input:
# Function output:none
# Function Action:
#
################################################################

NEW_LINE = "\n"
TAB = "\t"
# let's define our parameters:

learning_rate = 0.01
window_size = 5  # 2 neighbors + target word
DIM_HIDDEN_LAYER = 150
tags_number = 10
DIM_INPUT = 250  # window x vector_size = 5 x 50 =250
NUM_OF_EPOCHS = 3
VECTOR_EMBEDDINGS_DIM = 50

BATCH_NORMALIZATION_SIZE = 1024  # denotes the number of samples contained in each generated batch.


###############################################################
# Function Name:load_dev_data_set
# Function input:dev_file
# Function output:dev data set
# Function Action: loading the dev data set
#
################################################################
def load_dev_data_set(dev_file):


    windows_array, tags = my_helps_2.get_dev_data(dev_file)
    # Convert a list into a numpy array and set data-type to float32
    windows_array = np.asarray(windows_array, np.float32)

    # Convert a list into a numpy array and set data-type to int32
    tags = np.asarray(tags, np.int32)

    """
    Creates a Tensor from a numpy.ndarray
    The returned tensor and ndarray share the same memory. Modifications to the tensor will
     be reflected in the ndarray and vice versa. The returned tensor is not resizable.
    """
    tags = torch.from_numpy(tags)
    windows_array = torch.from_numpy(windows_array)

    # make sure the words and tag have same size, in order to pass it through TensorDataset
    tags = tags.type(torch.LongTensor)
    windows_array = windows_array.type(torch.LongTensor)

    data = TensorDataset(windows_array, tags)

    """
    shuffle is set to True, we will get a new order of exploration at each pass.
    Shuffling the order in which examples are fed to the classifier is helpful so
     that batches between epochs do not look alike. Doing so will eventually make our model more robust.
    """
    return DataLoader(batch_size=1, shuffle=True, dataset=data)


###############################################################
# Function Name:
# Function input:
# Function output:none
# Function Action:
#
################################################################
class NeuralNet(nn.Module):
    """
       Model A: Neural Network with one hidden layer, the first layer should have a size of 250 since each
       embedding vector contains 50 neurons and each window has 5 word
       should be followed by tanh activation function
       """

    ###############################################################
    # Function Name:
    # Function input:
    # Function output:none
    # Function Action:
    #
    ################################################################
    def __init__(self, input_size):
        super(NeuralNet, self).__init__()

        # create the E matrix
        # TODO: my_helps_2.
        self.E = nn.Embedding(my_helps_2.E.shape[0], my_helps_2.E.shape[1])
        self.E.weight.data.copy_(torch.from_numpy(my_helps_2.E))
        self.input_size = my_helps_2.E.shape[1] * window_size
        self.fc0 = nn.Linear(input_size, DIM_HIDDEN_LAYER)
        self.fc1 = nn.Linear(DIM_HIDDEN_LAYER, len(my_helps_2.Dictionary_of_classes))

    ###############################################################
    # Function Name:
    # Function input:
    # Function output:none
    # Function Action:
    #
    ################################################################
    def forward(self, x):
        x = self.E(x).view(-1, self.input_size)
        x = F.tanh(self.fc0(x))
        x = self.fc1(x)
        x_softmax = F.log_softmax(x, dim=1)
        return x_softmax


###############################################################
# Function Name:create_match_ploting
# Function input:m_dev_loss, m_precents_correct_dev
# Function output:none
# Function Action: the function uses plot library to plot graph shes the learning and scores of the model#
################################################################
def create_match_ploting(m_dev_loss, m_precents_correct_dev):


    line1, = plt.plot(m_dev_loss.keys(), m_dev_loss.values(), "red",label='Dev avg loss')

    #add a title to the plots
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=4)})

    #show the plot
    plt.show()
    line2, = plt.plot(m_precents_correct_dev.keys(),
                      m_precents_correct_dev.values(),label='Dev avg accuracy')

    # add a title to the plots
    plt.legend(handler_map={line2: HandlerLine2D(numpoints=4)})
    plt.show()


###############################################################
# Function Name:load_training_data_set
# Function input:train_set_file
# Function output:the training data set
# Function Action:get the train data
#
################################################################
def load_training_data_set(train_set_file):
    windows_array, tags = my_helps_2.get_train_data(train_set_file)
    # Convert a list into a numpy array and set data-type to float32
    windows_array = np.asarray(windows_array, np.float32)

    # Convert a list into a numpy array and set data-type to int32
    tags = np.asarray(tags, np.int32)

    """
    Creates a Tensor from a numpy.ndarray
    The returned tensor and ndarray share the same memory. Modifications to the tensor will
     be reflected in the ndarray and vice versa. The returned tensor is not resizable.
    """
    tags = torch.from_numpy(tags)
    windows_array = torch.from_numpy(windows_array)

    # make sure the words and tag have same size, in order to pass it through TensorDataset
    tags = tags.type(torch.LongTensor)
    windows_array = windows_array.type(torch.LongTensor)

    data = TensorDataset(windows_array, tags)

    """
    shuffle is set to True, we will get a new order of exploration at each pass.
    Shuffling the order in which examples are fed to the classifier is helpful so
     that batches between epochs do not look alike. Doing so will eventually make our model more robust.
    """
    return DataLoader(batch_size=BATCH_NORMALIZATION_SIZE, shuffle=True, dataset=data)






###############################################################
# Function Name:main
# Function input:
# Function output:none
# Function Action:
#
################################################################
def main(argv):
    # ner or pos (user input)
    folder_name_input = argv[0]
    # global learning_rate
    #
    # if folder_name_input == 'ner':
    #     learning_rate = 0.05

    # define a path for each dataset file
    path_test = folder_name_input + '/test'
    path_train = folder_name_input + '/train'
    path_dev = folder_name_input + '/dev'

    set_of_training = load_training_data_set(path_train)
    print(len(my_helps_2.Dictionary_of_words))

    set_of_dev = load_dev_data_set(path_dev)
    #
    dataset_test = load_test_set(path_test)
    # # done splitting
    my_neural_network_model = NeuralNet(input_size=DIM_INPUT)
    optimizer = optim.Adam(my_neural_network_model.parameters(), lr=learning_rate)
    #
    trainer = Part1Model(optimizer, set_of_training, my_neural_network_model, dataset_test, set_of_dev)
    trainer.Start_Action(folder_name_input)


###############################################################
# Function Name: get_y_tag
# Function input: model
# Function output:return the model prediction tag
# Function Action: the function return the prediction
# by getting  the index of the max log-probability
################################################################

def get_y_tag(model):
    return model.data.max(1, keepdim=True)[1]


###############################################################
# Function Name:print_message_each_epoch
# Function input:kind_of_set,length of set, loss of model, number
# of correct predictions of model and size of batch
# Function output:none
# Function Action:the function print a message to help the user
# follow the network progress
# Function Action:the function calculate the loss of the model
# for each epochs,and print write message
################################################################

def print_message_each_epoch(is_training_set, m_total, epoch, m_loss, m_success, size_of_batch, accuracy):
    if (is_training_set == 1):
        title = "Training Set"
    else:
        title = "Validation set"

    print('\n' + title + ': Epoch number is:{}  The average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(epoch,m_loss,m_success,m_total,accuracy))



class Part1Model(object):

    ###############################################################
    # Function Name: init function on the class - constractor
    # Function input: kind_of_optim, training_set, nn_model, test_data,dev_dataset
    # Function output:a model object with the next methods
    # Function Action:The __init__ method is roughly what represents a constructor in Python    #
    ################################################################
    def __init__(self, kind_of_optim, training_set, nn_model, test_data, dev_dataset):

        self.kind_of_optim = kind_of_optim
        self.training_set = training_set
        self.nn_model = nn_model
        self.test_data = test_data
        self.dev_dataset = dev_dataset

    ###############################################################
    # Function Name:Get_Representation_Of_Indexes_By_classes
    # Function input:lst_of_index_to_convert
    # Function output:return the class which the index is
    # Function Action:    return the class which the index is
    ################################################################

    def Get_Representation_Of_Indexes_By_classes(self, lst_of_index_to_convert):
        return [my_helps_2.Representation_Of_Indexes_By_classes[i] for i in lst_of_index_to_convert]



    ###############################################################
    # Function Name:train_neural_network
    # Function input:model, trainng and validation sets
    # Function output:none
    # Function Action:train on the training set and then test the
    # network on the test set. This has the network make predictions on data it has never seen
    ################################################################

    def train_neural_network(self, iter, m_loss_training):

        self.nn_model.train()
        train_loss = 0
        m_success = 0

        #looping the training set
        for data, labels in self.training_set:
            self.kind_of_optim.zero_grad()
            returned_model = self.nn_model(data)
            # call help function to compute the predicted y
            y_hat = get_y_tag(returned_model)
            m_success += y_hat.eq(labels.data.view_as(y_hat)).cpu().sum().item()
            # computing the loss
            loss = F.nll_loss(returned_model, labels)
            train_loss += loss
            # start back action
            loss.backward()
            # updating parameters
            self.kind_of_optim.step()

        train_loss /= (len(self.training_set))
        m_loss_training[iter] = train_loss
        length = len(self.training_set) * BATCH_NORMALIZATION_SIZE
        accuracy = 100. * m_success / (len(self.training_set) * BATCH_NORMALIZATION_SIZE)

        print_message_each_epoch(1, length, iter, train_loss, m_success, BATCH_NORMALIZATION_SIZE, accuracy)

    ###############################################################
    # Function Name:feedforward_the_dev_set
    # Function input:
    # Function output:none
    # Function Action:This is the forward propagation function
    # #feed the neural networks firts time, and use activition function
    # #to convert values to be between 0 to 1
    ################################################################
    def feedforward_the_dev_set(self, iter_number, dev_loss_dictionary,
                                dev_precents_of_accuracy_dictionary, nerOrPos):

        # let the model know to switch to eval mode by calling .eval() on the model
        self.nn_model.eval()
        # define varibles for loss, and number of correct prediction
        m_loss = 0
        m_success = 0
        m_count = 0
        for data, target in self.dev_dataset:
            # let the model know to switch to eval mode by calling .eval() on the model
            output = self.nn_model(data)
            m_loss += F.nll_loss(output, target, size_average=False).item()
            # call get_y_tag to find y_hat
            y_hat = get_y_tag(output)
            #check if the mapping type is ner so :
            if nerOrPos == 'ner':
                if my_helps_2.Representation_Of_Indexes_By_classes[y_hat.cpu().sum().item()] != 'O' or \
                        my_helps_2.Representation_Of_Indexes_By_classes[target.cpu().sum().item()] != 'O':
                    m_success += y_hat.eq(target.data.view_as(y_hat)).cpu().sum().item()
                    m_count += 1
            else:
                m_count += 1
                m_success += y_hat.eq(target.data.view_as(y_hat)).cpu().sum().item()

        m_loss /= len(self.dev_dataset)
        dev_loss_dictionary[iter_number] = m_loss
        accuracy = 100. * m_success / m_count
        dev_precents_of_accuracy_dictionary[iter_number] = accuracy

        print_message_each_epoch(0, m_count, iter_number, m_loss, m_success, 1, accuracy)

    ###############################################################
    # Function Name:forward_test_data
    # Function input:ner_or_pos
    # Function output:none
    # Function Action:This is the forward propagation function
    #     # #feed the neural networks firts time, and use activition function
    #     # #to convert values to be between 0 to 1
    ################################################################
    def forward_test_data(self, ner_or_pos):

        self.nn_model.eval()
        m_y_hats = []
        for data in self.test_data:
            the_model = self.nn_model(torch.LongTensor(data))
            #call get_y_tag to find y_hat

            pred = get_y_tag(the_model)

            #append y_hat in the file of preds
            m_y_hats.append(pred.item())

            m_y_hats = self.Get_Representation_Of_Indexes_By_classes(m_y_hats)
        #define the path to the test file - from where we'll get the data
        path_to_test_file = ner_or_pos + "/test"
        self.create_predictions_file(path_to_test_file, "test2." + ner_or_pos, m_y_hats)

    ###############################################################
    # Function Name:create_predictions_file
    # Function input:test_file_name, output_file_name, list_of_y_hats
    # Function output:none
    # Function Action:#save resluts into predictions file
    #
    ################################################################
    def create_predictions_file(self, test_file_name, output_file_name, list_of_y_hats):

        with open(test_file_name, 'r') as unseen_data, open(output_file_name, 'w') as preds_file:
            m_count = 0#set count to zero
            #read the lines of file
            m_lines_of_file = unseen_data.readlines()
            #do a for loop
            for m_cure_line in m_lines_of_file:
                #check if current line is a new line
                if m_cure_line == NEW_LINE:
                    #if so add this to the predictions file
                    preds_file.write(m_cure_line)
                else:
                    preds_file.write(m_cure_line.strip(NEW_LINE) +
                                     " " + list_of_y_hats[m_count] +
                                     NEW_LINE)
                    #update the count value!
                    m_count = 1 + m_count

        ###############################################################
        # Function Name:Start_Action
        # Function input:ner_or_pos - a info which tell what kind of dataset we are dealing with
        # Function output:none
        # Function Action:the function runing a loop (epoch) which in each iteration it train the
        # network and The function produces two graphs that describe the loss of the function and
        # its accuracy percentage for the training set and test set at each iteration
        ################################################################

    def Start_Action(self, ner_or_pos):
        m_precent_of_accuracy_on_dev = {}
        m_loss_training = {}
        m_loss_dev = {}
        # runing in a loop as number of epoch we defined
        for epoch in range(NUM_OF_EPOCHS):
            print "epoch number " + str(epoch)
            # call the tairn function to start the training progress
            self.train_neural_network(epoch, m_loss_training)
            self.feedforward_the_dev_set(epoch, m_loss_dev, m_precent_of_accuracy_on_dev, ner_or_pos)
        create_match_ploting(m_loss_dev, m_precent_of_accuracy_on_dev)
        # call the forward_test_data function to find the predictions for unseen data
        self.forward_test_data(ner_or_pos)



###############################################################
# Function Name:load_test_set
# Function input:the file to load contains the test data
# Function output:return the test data
# Function Action: get the test data
#
################################################################
def load_test_set(file_name):

#call help function to read the data
    x = my_helps_2.bring_test_data(file_name)
    return x

if __name__ == "__main__":
    main(sys.argv[1:])
