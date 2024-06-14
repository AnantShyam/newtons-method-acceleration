import torch 
import cifar 
import cifar_model
from cifar10_loader import CIFAR10Dataset  
from torch.nn import functional as F
import torch.nn as nn
import torch.optim as optim
import pyhessian
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy

# w vector = w .parameters
# torch.nn.utils.parameters_to_vector()

# compute eigendecomposition of hessian 
# take k positive eigenvalues, k <= # num of positive eigenvalues of hessian 

# Hessian = d x d = V D V^T

# d * k k * k k* d
# add some regularization 

# if no positive eigenvalues, take a gradient step and try again

def newton_method(model, data_loader, num_epochs, layers_to_be_updated):

    # layers_to_be_updated = indices of layers to run newton's method on
    
    train_accuracies = []

    for epoch in tqdm(range(num_epochs)):

        # test model at the start of each epoch
        train_accuracy = cifar.test_model(model, data_loader)
        print(train_accuracy)
        train_accuracies.append(train_accuracy)

        for inputs, labels in data_loader:
            hessian_class = pyhessian.hessian(model, F.cross_entropy, data=(inputs, labels), cuda=False)
            params, grads = pyhessian.utils.get_params_grad(model)

            names = [name for name, _ in model.state_dict().items()]
            for i in range(len(list(model.parameters()))):
                if i in layers_to_be_updated:
                    weight_vector = list(model.parameters())[i]
                    original_shape = weight_vector.shape
                    weight_vector = weight_vector.flatten()
                    weight_vector = weight_vector.reshape(len(weight_vector), 1)

                    gradient = grads[i]
                    eigenvalues, eigenvectors = hessian_class.eigenvalues(maxIter=1, top_n=1)
                    eigenvectors = eigenvectors[0]

                    eigenvalue = eigenvalues[0]
                    eigenvector = eigenvectors[i].flatten()
                    eigenvector = eigenvector.reshape(len(eigenvector), 1)
                    
                    gradient = gradient.flatten()
                    gradient = gradient.reshape(len(gradient), 1)

                    # approximate hessian
                    hessian_matrix_first_matrix = (eigenvalue) * (eigenvector @ eigenvector.T)
                    
                    tau = 10**-5 

                    # try and make hessian positive definite
                    hessian_matrix_first_matrix = hessian_matrix_first_matrix + (tau*torch.eye(len(hessian_matrix_first_matrix)))
                    hessian_matrix_first_matrix_inverse = torch.inverse(hessian_matrix_first_matrix)

                    alpha = 0.01

                    new_first_weight_vector = weight_vector - (alpha * (hessian_matrix_first_matrix_inverse @ gradient))
                    new_first_layer_weight_vector = new_first_weight_vector.reshape(original_shape)

                    model.state_dict()[names[i]].data += new_first_layer_weight_vector

                    print('done')

    return model, train_accuracies
    


if __name__ == "__main__":
    train_data_loader, test_data_loader = cifar.create_train_test_dataloaders(2000)
    initial_model = cifar_model.CIFAR10Net()
    #print(cifar.test_model(initial_model, train_data_loader))
    #train_newton_method(initial_model, train_data_loader, 1)


    # run newton's method on just first layer for 5 epochs and plot accuracies

    num_epochs = 4
    trained_model, train_accuracies = newton_method(initial_model, train_data_loader, num_epochs, [0])
    
    # plt.plot([i for i in range(1, num_epochs + 1)], train_accuracies)
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # plt.savefig('nonconvex_model_plots/newton_method_accuracies_test.png')

    #print(cifar.test_model(trained_model, train_data_loader))
    # torch.save(trained_model, 'model_weights/trained_model_newton_method_test.pt')
    # print('Accuracy: ')
    