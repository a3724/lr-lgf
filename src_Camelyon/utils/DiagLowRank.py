
import numpy as np
import jax.numpy as jnp
from scipy.sparse.linalg import svds
from scipy.linalg import svd
import scipy
from typing import Optional, List, Tuple
from .base_utils import GlobalRegistry, TrainState
import jax

class Diag_LowRank(object):

    def __init__(self, shape_weights: int, num_classes: int):
        """
        Initializes a Diag_LowRank object. The precision adjusted mean is initialized with zeros, the diagonal plus low rank is initialized with ones on the diagonal.

        Args:
            shape_weights (int): The shape of the vectorized weights.
            num_classes (int): The number of classes in the dataset (or the chosen rank).

        """
        config = GlobalRegistry.get_config()
        self.mPi = jnp.zeros((shape_weights, 1))
        self.Pi_t = [config.LAMBDA_INIT*jnp.ones((1, shape_weights)),
                     jnp.zeros((shape_weights, num_classes)), 
                     jnp.zeros((num_classes, num_classes))]


    def add_low(self, J: jnp.ndarray, H: jnp.ndarray, LAM_TASK, task_id, eps=1e-4):
        """
        Updates the Pi_t decomposition with the Jacobian and Hessian of the GGN. Computes the square root of the low-rank part and for each batch dimension of the Jacobian and Hessian. Then, uses the m-truncated SVD of the stacked matrix.
        The SVD is computed on:
        [U C^{1/2}, J_0 H_0^{1/2}, ..., J_b H_b^{1/2}]
        where `U` and `C` are the current low-rank components.

        Args:
                J (jnp.array): The Jacobian of the network output w.r.t weights, shape (b, d, c).
                H (jnp.array): The Hessian of the not-regularized loss w.r.t network output, shape (b, c, c).
                eps: A small value added th the Hessian for numeric`l stability.
        """
        config = GlobalRegistry.get_config()
        U = self.Pi_t[1]
        C = self.Pi_t[2]

        rank = C.shape[0]
        batch_size, output_dim, weight_dim = J.shape

        left_matrix = np.zeros((weight_dim, batch_size+1, rank))

        C_12 = scipy.linalg.sqrtm(C)
        left_matrix[:, 0 ,:] = U @ C_12

        #J = np.sqrt(LAM_TASK)* J
        H = LAM_TASK * H

        for b in range(batch_size):
            H_12_b = scipy.linalg.sqrtm(H[b]+1e-4*np.eye(output_dim))#+ eps*jnp.eye(output_dim))
            if np.allclose(np.imag(H_12_b), 0, atol=1e-10):
                H_12_b = np.real(H_12_b)
                vals, _ = np.linalg.eig(H[b]+1e-4*np.eye(output_dim))
                min_val = np.min(vals)
                if np.min(vals) < -1e-6:
                    print(f"Warning: The original matrix has negative eigenvalues. Minimum value: {min_val}")
            if np.iscomplexobj(H_12_b):
                print('Working with complex values')
                H_12_b = check_complex_influence(H[b], eps)
            left_matrix[:,b+1,:output_dim] = (1/np.sqrt(batch_size))*J[b].T @ H_12_b

        #new_Ul, new_Sl, _ = self.truncated_svd(left_matrix, m=(task_id+1)*rank)
        new_Ul, new_Sl, _ = self.truncated_svd(left_matrix, m=rank)

        self.Pi_t[1], self.Pi_t[2] = new_Ul, jnp.diag(new_Sl**2)

    def truncated_svd(self, M: np.ndarray, m: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes the truncated SVD of a matrix.

        Args:
            M (jnp.array): The input matrix of shape (weights_dim, batch_size+1, output_dim).
            m (int): The number of singular values to keep in the decomposition.

        Returns:
            U (jnp.array): Left singular vectors, shape (weights_dim, m).
            s (jnp.array): Singular values, shape (m,).
            Vt (jnp.array): Right singular vectors (transposed), shape (m, (batch_size+1) * output_dim).
        """
        weights_dim, batch_size_1, output_dim = M.shape
        M_reshaped = M.reshape(weights_dim, batch_size_1*output_dim)
        U, s, Vt = svds(np.array(M_reshaped) , k=m)

        sorted_indices = np.argsort(s)[::-1]
        U = U[:, sorted_indices]
        s = s[sorted_indices]
        Vt = Vt[sorted_indices, :]
        return U, s, Vt

    '''

    def compute_inv_sum_diag_dlr(self, Q: jnp.ndarray, Pts: Optional[List] = None, Pms: Optional[List] = None, t: Optional[int] = None) -> Optional[Tuple[List, List]]:
        """
        Computes the inverse sum of the diagonal low-rank object and updates the Pi_t components.
        Optionally appends intermediate decomposition steps to `Pts` and `Pms`.

        The equation:
            (Q + (Pi_t[0] + Pi_t[1]@Pi_t[2]@Pi_t[1].T)^{-1})^{-1} =
            = (new_Pi_t[0] +new_Pi_t[1]@new_Pi_t[2]@new_Pi_t[1].T)
            where:
            new_Pi_t[0] = (Q + (Pi_t[0])^{-1})^{-1}
            new_Pi_t[1] = (Q + (Pi_t[0])^{-1})^{-1} @ (Pi_t[0])^{-1} @ Pi_t[1]
            new_Pi_t[2] = ((Pi_t[2])^{-1} + self.Pi_t[1].T @ (Pi_t[0])^{-1} @ Pi_t[1] - (Pi_t[0])^{-1}.T @ Pi_t[1].T @ (Q + (Pi_t[0])^{-1})^{-1} @ (Pi_t[0])^{-1} @ Pi_t[1])^{-1}
        is computed using Woodbury.

        Args:
            Q (jnp.ndarray): The matrix to add to the diagonal component, shape (1, weight_dim).
            Pts (Optional[List]): A list of low-rank matrices from update step.
            Pms (Optional[List]): A list of low-rank matrices from predict step.
            t (Optional[int]): Which task.

        Returns:
            Optional[Tuple[List, List]]:
                - Pts (List): Updated list of low-rank matrices from update step.
                - Pms (List): Updated list of low-rank matrices from predict step.
                Returns `None` if neither `Pts` nor `Pms` is provided.
        """
        A_1 = (1/(self.Pi_t[0]))

        # Updated diagonal component
        L = (1/(Q + A_1)) 

        # Low-rank components
        Up = A_1.T * self.Pi_t[1]
        Cp = jnp.diag(1/jnp.diag(self.Pi_t[2])) + (self.Pi_t[1].T @ Up)

        # Updated low-rank components
        left = L.T * Up
        mid = (Cp - Up.T @ left)
        mid_inv = jnp.linalg.pinv(mid)

        self.Pi_t = [L, left, mid_inv]

        if Pts is not None:
            Pts.append([A_1, Up, jnp.linalg.pinv(Cp)])
            Pms.append([A_1+Q, Up, jnp.linalg.pinv(Cp)])
            return Pts, Pms
        
        return None
    '''

    from jax import config

    def compute_inv_sum_diag_dlr(self, Q: jnp.ndarray, Pts: Optional[List] = None, Pms: Optional[List] = None,
                                 t: Optional[int] = None) -> Optional[Tuple[List, List]]:
        """
        Computes the inverse sum of the diagonal low-rank object and updates the Pi_t components.
        Generalized implementation to handle various input scenarios, with calculations in 64-bit precision
        for this function only, and results converted back to 32-bit.
        """
        # Save the original config state
        original_x64_setting = jax.config.read("jax_enable_x64")

        # Enable 64-bit precision
        jax.config.update("jax_enable_x64", True)

        try:
            # Ensure inputs are in 64-bit precision
            Q_64 = Q.astype(jnp.float64)
            Pi_t_64 = [component.astype(jnp.float64) for component in self.Pi_t]

            # Diagonal component inverse
            A_1 = (1 / Pi_t_64[0])

            # Updated diagonal component
            L = (1 / (Q_64 + A_1))

            # Low-rank components
            Up = A_1.T * Pi_t_64[1]
            Cp = jnp.diag(1 / jnp.diag(Pi_t_64[2])) + (Pi_t_64[1].T @ Up)

            # Updated low-rank components
            left = L.T * Up
            mid = (Cp - Up.T @ left)
            mid_inv = jnp.linalg.pinv(mid)

            # Convert results back to 32-bit and update Pi_t
            self.Pi_t = [
                L.astype(jnp.float32),
                left.astype(jnp.float32),
                mid_inv.astype(jnp.float32)
            ]

            print('Sucessfully inverted')

            # Optionally append to tracking lists if provided
            if Pts is not None:
                Pts.append([
                    A_1.astype(jnp.float32),
                    Up.astype(jnp.float32),
                    jnp.linalg.pinv(Cp).astype(jnp.float32)
                ])
                Pms.append([
                    (A_1 + Q_64).astype(jnp.float32),
                    Up.astype(jnp.float32),
                    jnp.linalg.pinv(Cp).astype(jnp.float32)
                ])
                return Pts, Pms
        finally:
            # Restore the original config state
            jax.config.update("jax_enable_x64", original_x64_setting)

        return None

    def update_mPi(self, theta_star):
        """
        Updates the mean vector `mPi` using the given optimal parameter `theta_star` and the current `Pi_t`.

        Args:
            theta_star (jnp.ndarray): The optimal parameter vector, shape (weight_dim,).
        """
        self.mPi = theta_star[None,:] * self.Pi_t[0]

        # Compute low-rank contributions
        low_rank_term = theta_star[None, :] @ self.Pi_t[1]  # Shape: (1, rank)
        low_rank_term = low_rank_term @ self.Pi_t[2]        # Shape: (1, rank)
        low_rank_term = low_rank_term @ self.Pi_t[1].T      # Shape: (1, weight_dim)

        self.mPi += low_rank_term
        self.mPi = self.mPi.T

    def smooth(self, m_t, P_t, m_s, Q):  # update, predict+1, smooth+1
        Q_small = Q + 1e-8

        Q_1 = 1 / Q_small
        L = 1 / (P_t[0] * Q_1 + np.ones(Q.shape))
        left = np.ones(Q.shape) - L
        Up = L.T * P_t[1]
        Cp = jnp.diag(1 / jnp.diag(P_t[2])) + P_t[1].T * Q_1[None, :] * L @ P_t[1]
        Cpp = jnp.linalg.pinv(Cp)
        Vp = P_t[1].T * Q_1[None, :] * L
        G = [left, Up, Cpp, Vp]

        diff = (m_s - m_t)
        ms = m_t + (G[0] * diff)[0] + (G[1] @ G[2] @ G[3] @ diff)
        return ms


def check_complex_influence(mat, eps=1e-4, eps2=1e-3, eps3=1e-1):
    """
    Adjusts the matrix to ensure non-negative eigenvalues and returns its square root.

    Parameters:
    - mat_12: Precomputed square root of the matrix (not used in computation but returned if no adjustment is needed).
    - mat: The input matrix to be checked and adjusted.
    - eps: Small value added to the diagonal for the first adjustment step.
    - eps2: Larger value added to the diagonal if negative eigenvalues persist after the first adjustment.
    - eps3: Final adjustment value added to the diagonal if negative eigenvalues persist after the second adjustment.

    Returns:
    - The square root of the adjusted matrix without negative eigenvalues.
    """
    def check_eigenvalues(matrix):
        vals, _ = np.linalg.eig(matrix)
        return np.min(vals), vals

    # Step 1: Check the original matrix
    min_val, _ = check_eigenvalues(mat+1e-4*np.eye(mat.shape[0]))
    if min_val < 0:
        print(f"Warning: The original matrix has negative eigenvalues. Minimum value: {min_val}")

        # Step 2: Adjust with `eps`
        adjusted_mat = mat + eps * np.eye(mat.shape[0])
        min_val, _ = check_eigenvalues(adjusted_mat)
        if min_val < 0:
            print(f"Warning: Matrix with diagonal adjusted by {eps} has negative eigenvalues. Minimum value: {min_val}")

            # Step 3: Adjust with `eps2`
            adjusted_mat = mat + eps2 * np.eye(mat.shape[0])
            min_val, _ = check_eigenvalues(adjusted_mat)
            if min_val < 0:
                print(f"Warning: Matrix with diagonal adjusted by {eps2} has negative eigenvalues. Minimum value: {min_val}")

                # Step 4: Adjust with `eps3`
                adjusted_mat = mat + eps3 * np.eye(mat.shape[0])
                min_val, _ = check_eigenvalues(adjusted_mat)
                if min_val < 0:
                    raise ValueError(f"Matrix could not be adjusted to remove negative eigenvalues even with {eps3} added.")
                else:
                    print(f"Matrix with diagonal adjusted by {eps3} has no negative eigenvalues.")
            else:
                print(f"Matrix with diagonal adjusted by {eps2} has no negative eigenvalues.")
        else:
            print(f"Matrix with diagonal adjusted by {eps} has no negative eigenvalues. Minimal eigenvalue {min_val}")
    else: adjusted_mat = mat

    # Return the square root of the adjusted matrix
    sqrt_adjusted_mat = scipy.linalg.sqrtm(adjusted_mat)
    return sqrt_adjusted_mat


def smoother_run(theta_star, flt, thetas, list_of_Pts, Qs, batch_statss, cnn_trainer, list_of_eval_loaders, unflatten_func):
    config = GlobalRegistry.get_config()

    mss = np.zeros((config.T, theta_star.size))
    mss[-1, :] = theta_star
    for sid in range(config.T - 2, -1, -1):
        ms = flt.smooth(thetas[sid], list_of_Pts[sid], mss[sid + 1], Qs[sid])
        mss[sid] = ms
    lam_t = 1
    smooth_acc = np.zeros((config.T, config.T))
    for si in range(config.T):
        unflattened_m = unflatten_func(mss[si])
        state_new = TrainState.create(apply_fn=cnn_trainer.model.apply,
                                      params=unflattened_m,
                                      batch_stats=batch_statss[si],
                                      tx=cnn_trainer.state.tx)
        for ti in range(config.T):
            acc = cnn_trainer.eval_model(list_of_eval_loaders[ti], thetas[ti], flt, lam_t , extra_state=state_new)
            smooth_acc[si, ti] = acc

    return smooth_acc




