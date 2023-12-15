import numpy as np

def calculate_covariance_matrix(normalized_pixels):
    return normalized_pixels.T @ normalized_pixels 

def calculate_eig_sorted(covariance_matrix):
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    idx = eigenvalues.argsort()[::-1]   
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:,idx]

    return eigenvalues, eigenvectors