import numpy as np 
import torch
import helpers
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy
import time 
import krylov

            

# give same initializations for all methods 

class A9A_Analysis:

    def __init__(self, X_train, y_train, X_test, y_test):
        self.X = X_train    
        self.y = y_train 
        self.X_test = X_test 
        self.y_test = y_test
        self.regularization = 0.01
        self.w_init = torch.load('model_weights/initial_weight_vector.pt')


    def l2_regularized_logistic_regression_loss(self,w):

        loss = 0
        n = len(self.y)
        for i in range(n):
            training_example, label = self.X[:, i], self.y[i]
            concat_tensor = torch.tensor([0, (-w.T @ training_example) * label])
            exp_term = torch.logsumexp(concat_tensor, 0)

            loss = loss + exp_term

        regularization_term = (self.regularization * torch.pow(torch.norm(w), 2)/2.0)
        total_loss = (loss/n) + regularization_term
        return total_loss

    # H = U (Sigma) V^T 
    # H = A B, A = first column of U, B = first column of V^T

    def gradient_descent(self):
        d, n = self.X.shape
        w = torch.randn(d)

        # compute the largest eigenvalue of Hessian at w = 0, hessian = 124 x 124
        # H = 1/n  X^T P X, P = 0.25 I when w = 0
        hessian_w_equals_0 = (1/n) * (self.X @ (0.25 * torch.eye(n)) @ self.X.T) 
        hessian_w_equals_0 += (self.regularization * torch.eye(len(hessian_w_equals_0)))


        # assuming all eigenvalues are essentially real
        eigenvalues_hessian_w_equals_0 = torch.view_as_real(torch.linalg.eig(hessian_w_equals_0)[0])
    
        largest_eigenvalue = torch.max(eigenvalues_hessian_w_equals_0)

        # should this be 2/largest_eigenvalue?
        alpha = 1/largest_eigenvalue
        loss_values = [float('inf')]

        tolerance = 10**(-16)
        while True:
            gradient = self.compute_gradient(w)
            w = w - (alpha * gradient)
            curr_loss = self.l2_regularized_logistic_regression_loss(w).item()

            print("acc", self.test_model(w), "loss", curr_loss)
            if abs(curr_loss - loss_values[-1]) <= tolerance:
                break
            else:
                loss_values.append(curr_loss)

        return w, loss_values[1:]
        
    def form_hessian(self, w):
        d = self.X.shape[0]
        hessian = torch.zeros(d, d)
        for i in range(d):
            # hard coded for train dataset
            y_i, x_i = self.y[i], self.X[:, i].reshape((124, 1))
            term1 = torch.sigmoid(-y_i * (w.T @ x_i)) * (1 - torch.sigmoid(-y_i * (w.T @ x_i)))
            hessian = hessian + (term1 * (x_i @ x_i.T))
        hessian = hessian/d

        hessian +=  (self.regularization * torch.eye(len(hessian)))
        return hessian

    def backtrack_line_search(self, w, update):
        init_alpha = 1.0
        rho = 0.5
        c = 0.001
        min_alpha = 0.05

        alpha = init_alpha
        while True:
            f_w_k = self.l2_regularized_logistic_regression_loss(w).item()
            f_w_k_alpha_update = self.l2_regularized_logistic_regression_loss(w + (alpha * update)).item()

            gradient = self.compute_gradient(w)
            c_alpha_gradient_update = (c * alpha) * (gradient.T @ update)

            if f_w_k >= (f_w_k_alpha_update - c_alpha_gradient_update.item()):
                return alpha
            elif alpha <= min_alpha:
                return min_alpha
            # if f_w_k_alpha_update <= (f_w_k + c_alpha_gradient_update).item():
            #     return alpha 
            
            alpha = rho * alpha 

    def newton_method_exact(self, num_epochs):
        #w = torch.rand(self.X.shape[0])
        w = torch.clone(self.w_init)
        loss_values = {}
        accuracy_values = {}

        start = time.time()
        for epoch in tqdm(range(num_epochs)):

            # compute the gradient 
            gradient = self.compute_gradient(w)

            # compute the hessian and inverting the hessian
            hessian_matrix = self.form_hessian(w)
            hessian_num_rows, hessian_num_columns = hessian_matrix.shape
            hessian_inverse = torch.inverse(hessian_matrix)

            # do backtracking line search - Armijo line search - algorithm 3.1 in nocedal and wright
            update = hessian_inverse @ gradient
            alpha = self.backtrack_line_search(w, update)

            loss_val = self.l2_regularized_logistic_regression_loss(w).item()
            accuracy_val = self.test_model(w)
            accuracy_values[epoch + 1] = accuracy_val
            w = w - (alpha * (update))

            #loss_val = self.l2_regularized_logistic_regression_loss(w).item()
            print(loss_val)
            print(accuracy_val)

            loss_values[epoch + 1] = loss_val
            

            #print('acc', self.test_model(w), 'loss val', loss_val)
        end = time.time()
        return w, loss_values, accuracy_values, end-start


    def compute_gradient(self, w, is_train=True):
        if is_train:
            return (torch.sigmoid(-self.y*((self.X.T)@w))*(-self.y*self.X)).mean(1) + (self.regularization * w)
        else:
            return (torch.sigmoid (-self.y_test * ((self.X_train.T) @ w)) * 
            (-self.y_test * self.X_test)).mean(1) + (self.regularization * w)


    def conjugate_residual(self, num_epochs):
        #w = torch.rand(self.X.shape[0])
        w = torch.clone(self.w_init)
        loss_values = {}
        accuracy_values = {}

        start = time.time()
        for epoch in range(num_epochs):
            gradient = self.compute_gradient(w)
            hessian_matrix = self.form_hessian(w)

            update = krylov.conjugate_residual(hessian_matrix, gradient)
            alpha = self.backtrack_line_search(w, update)

            accuracy_values[epoch + 1] = self.test_model(w)
            loss_values[epoch + 1] = self.l2_regularized_logistic_regression_loss(w).item()

            w = w - (alpha * update)
            
            
            #print(self.test_model(w))
        end = time.time()

        return w, loss_values, accuracy_values, end-start

    
    def gmres(self, num_epochs):
        #w = torch.rand(self.X.shape[0]) 
        w = torch.clone(self.w_init)
        loss_values = {}
        accuracy_values = {}

        start = time.time()
        for epoch in tqdm(range(num_epochs)):
            gradient = self.compute_gradient(w)
            hessian_matrix = self.form_hessian(w)

            gradient = gradient.numpy()
            hessian_matrix = hessian_matrix.numpy()

            update, _ = scipy.sparse.linalg.gmres(hessian_matrix, gradient, maxiter=5)
            update = torch.from_numpy(update)

            #alpha = 0.09 # do a line search here
            alpha = self.backtrack_line_search(w, update)
            accuracy_values[epoch + 1] = self.test_model(w)

            loss_val = self.l2_regularized_logistic_regression_loss(w).item()

            w = w - (alpha * update)
            #loss_val = self.l2_regularized_logistic_regression_loss(w).item()
            loss_values[epoch + 1] = loss_val
            
            #print("loss", self.l2_regularized_logistic_regression_loss(w).item(), "acc", self.test_model(w))
        end = time.time()
        return w, loss_values, accuracy_values, end-start

     



    def test_model(self, weight_vector, is_train=True):
        if is_train:
            accuracy = (np.sign((self.X.T)@weight_vector) == self.y).float().mean()
            return accuracy
        else:
            accuracy = (np.sign((self.X_test.T)@weight_vector) == self.y_test).float().mean()
            return accuracy
    

    def plot_losses_or_accuracies_all_newton_methods(self, filename, num_epochs, plot_losses=True):
        # newton_methods = {'Exact Newton': self.newton_method_exact, 'GMRES': 
        # self.gmres, 'Conjugate Residual': self.conjugate_residual}

        #newton_methods = {'Exact Newton' :self.newton_method_exact}
        #newton_methods={'GMRES': self.gmres}
        newton_methods={'Conjugate Residual':self.conjugate_residual}
        for newton_method_name, newton_method in newton_methods.items():
            _, loss_vals, accuracy_vals, _ = newton_method(num_epochs)
            epochs = [i for i in range(1, num_epochs + 1)]
            loss_values = [val for _, val in loss_vals.items()] 
            accuracy_values = [val for _, val in accuracy_vals.items()]
            print(accuracy_values)
            if plot_losses:
                plt.plot(epochs, loss_values, label=newton_method_name)
            else:
                plt.plot(epochs, accuracy_values, label=newton_method_name)
            

            plt.legend()
            plt.xlabel('Number of Epochs')
            if plot_losses:
                plt.ylabel('Loss Values')
                plt.title('Loss values vs Epochs')
            else:
                plt.ylabel('Accuracy Values')
                plt.title('Accuracy Values vs Epochs')
            plt.savefig(f'convex_model_plots/{newton_method_name}_{filename}')

    
    def measure_wall_clock_time_all_newton_methods(self, num_epochs):
        newton_methods = {'Exact Newton': self.newton_method_exact, 'GMRES': 
        self.gmres, 'Conjugate Residual': self.conjugate_residual}

        #newton_methods = {'Exact Newton': self.newton_method_exact}

        f = open('model_statistics/wall_clock_times.txt', 'w')
        f.write(f'{num_epochs} \n')
        wall_clock_times = {}
        for newton_method_name, newton_method in newton_methods.items():
            _, _, _, wall_clock_time = newton_method(num_epochs)
            wall_clock_times[newton_method_name] = wall_clock_time

        for newton_method_name, wall_clock_time in wall_clock_times.items():
            f.write(f'{newton_method_name}, {str(wall_clock_time)} \n')
            
        return wall_clock_times


    def plot_suboptimality_all_newton_methods(self, filename, num_epochs):
        # difference between Gradient Descent approximately converged loss and Newton's Method Loss over iterations 
        gradient_descent_loss = torch.load('model_statistics/gradient_descent_loss.pt')
        final_converged_loss = gradient_descent_loss[-1]

        # newton_methods = {'Exact Newton': self.newton_method_exact, 'GMRES': 
        # self.gmres, 'Conjugate Residual': self.conjugate_residual}

        #newton_methods = {'Exact Newton': self.newton_method_exact}
        #newton_methods = {'GMRES': self.gmres}
        newton_methods = {'Conjugate Residual': self.conjugate_residual}

        for newton_method_name, newton_method in newton_methods.items():
            _, loss_vals, _, _ = newton_method(num_epochs)
            epochs = [i for i in range(1, num_epochs + 1)]
            loss_differences = [abs(final_converged_loss - loss_val) for _, loss_val in loss_vals.items()]
            plt.plot(np.array(epochs), np.log(np.array(loss_differences)), label=newton_method_name)
            
            plt.legend()
            plt.xlabel('Number of Epochs')
            plt.ylabel('Suboptimality (Log Scale)')
            plt.savefig(f'convex_model_plots/{newton_method_name}_{filename}')
        
        #_, newton_method_loss_vals = newton_method(15)
         
        # torch.save(torch.tensor(list(newton_method_loss_vals.values())), 'model_training_information/biconjugate_loss.pt')
        # torch.save(torch.tensor(list(newton_method_loss_vals.values())), '../model_training_information/newton_method_loss_a9a.pt')
        
        # loss_differences = {i: abs(newton_method_loss_vals[i] - final_converged_loss) 
        #                         for i in range(1, len(newton_method_loss_vals))}

        # helpers.plot_curve(list(loss_differences.keys()), list(loss_differences.values()), 
        # 'Number of Epochs', 'Suboptimality', filename)


if __name__ == "__main__":
    a9a_dataset_train, labels_train = helpers.read_a9a_dataset('../data/a9a_train.txt')
    a9a_dataset_test, labels_test = helpers.read_a9a_dataset('../data/a9a_test.txt')

    #init_w = torch.rand(a9a_dataset_train.shape[0])
    #torch.save(init_w, 'model_weights/initial_weight_vector.pt')
    a9a = A9A_Analysis(a9a_dataset_train, labels_train, a9a_dataset_test, labels_test)
    # init_w = torch.rand(a9a.X.shape[0])
    
    #w, _, _, _ = a9a.newton_method_exact(10, init_w)
    #_, gradient_descent_loss_values = a9a.gradient_descent()
    # print(gradient_descent_loss_values[-1])#
    #torch.save(torch.tensor(gradient_descent_loss_values), 'model_statistics/gradient_descent_loss.pt')

    # w = torch.randn(124)
    # v = torch.randn(124)

    # print(a9a.hessian_vector_product(w, v))
    #wall_clock_times = a9a.measure_wall_clock_time_all_newton_methods(10)
    #print(wall_clock_times)

    #print(torch.autograd.functional.hessian(a9a.l2_regularized_logistic_regression_loss, torch.ones(124)))
    
    
    #a9a.plot_losses_or_accuracies_all_newton_methods('loss_vals.png', 10, True)
    #a9a.plot_losses_or_accuracies_all_newton_methods('accuracy_vals.png', 10, False)
    #a9a.plot_suboptimality_all_newton_methods('suboptimality.png', 10)

    #_ = a9a.conjugate_residual(10)
    # print(time)
    #print(a9a.test_model(w, False))
