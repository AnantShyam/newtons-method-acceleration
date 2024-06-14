import torch

def conjugate_residual(A, b):
    tolerance = 10**-1
    x = torch.rand(b.shape[0])
    residual = b - (A @ x)
    p = residual 

    while True:
        alpha = (residual.T @ A @ residual)/((A @ p).T @ (A @ p))
        new_x = x + (alpha * p) 
        if torch.norm(new_x - x) <= tolerance:
            return new_x 
        
        new_r = residual - (alpha * (A @ p))
        beta = (new_r.T @ A @ new_r)/(residual.T @ A @ residual)
        new_p = new_r + (beta * p)

        residual = new_r 
        p = new_p
        x = new_x














