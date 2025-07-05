import numpy as np

def get_dominant_eigenvalue_and_eigenvector(data, num_steps):
    """
    data: np.ndarray – symmetric diagonalizable real-valued matrix
    num_steps: int – number of power method steps

    Returns:
    eigenvalue: float – dominant eigenvalue estimation after `num_steps` steps
    eigenvector: np.ndarray – corresponding eigenvector estimation
    """
    ### YOUR CODE HERE
    n = data.shape[0]
    v_prev = np.random.random(size=(n,))
    value = None
    for _ in range(num_steps):
        w = data.dot(v_prev)
        v_cur = w / np.linalg.norm(w)
        value = v_cur.dot(data.dot(v_cur)) / v_cur.dot(v_cur)
        v_prev = v_cur
    
    return float(value), v_prev