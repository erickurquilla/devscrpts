import numpy as np

def compute_rotation_matrix_from_basis_vectors(basis_new, basis_old):
    """
    Compute R^T that maps components from the OLD basis to the NEW basis.

    If v_old are components in basis_old and v_new are components in basis_new:
        v_new = R^T @ v_old

    Parameters
    ----------
    basis_new : array-like, shape (3, 3)
        New basis unit vectors expressed in OLD coordinates.
        Columns are [x_new, y_new, z_new].
    basis_old : array-like, shape (3, 3)
        Old basis unit vectors expressed in OLD coordinates.
        Columns are [x_old, y_old, z_old].
        (If your old basis is the standard Cartesian basis, pass np.eye(3).)

    Returns
    -------
    RT : ndarray, shape (3, 3)
        The transpose rotation matrix R^T that converts components OLD -> NEW.
    """
    Bn = np.asarray(basis_new, dtype=float)
    Bo = np.asarray(basis_old, dtype=float)

    if Bn.shape != (3, 3) or Bo.shape != (3, 3):
        raise ValueError("basis_new and basis_old must both be 3x3 arrays (columns are basis vectors).")

    # Orthonormality checks (optional but helpful)
    if not np.allclose(Bo.T @ Bo, np.eye(3), atol=1e-10):
        raise ValueError("basis_old is not orthonormal (Bo^T Bo != I).")
    if not np.allclose(Bn.T @ Bn, np.eye(3), atol=1e-10):
        raise ValueError("basis_new is not orthonormal (Bn^T Bn != I).")

    # R maps NEW components -> OLD components: v_old = R @ v_new
    # R = Bo^T @ Bn  (genqqeral form; if Bo = I, then R = Bn)
    R = Bo.T @ Bn

    # We return R^T so that v_new = R^T @ v_old
    RT = R.T
    return RT    

def rotate_theta_phi_from_old_to_new_basis(theta, phi, new_basis, old_basis):

    theta, phi = np.broadcast_arrays(theta, phi)
    orig_shape = theta.shape

    # Flatten for vectorized computation
    theta_flat = theta.ravel()
    phi_flat = phi.ravel()
    
    # Compute rhat vectors, shape (N, 3)
    x = np.sin(theta_flat) * np.cos(phi_flat)
    y = np.sin(theta_flat) * np.sin(phi_flat)
    z = np.cos(theta_flat)
    rhat = np.stack([x, y, z], axis=1)  # (N, 3)

    # Prepare rotated vectors array
    rhat_rot = np.zeros_like(rhat)    
    rotation_matrix = compute_rotation_matrix_from_basis_vectors(new_basis, old_basis)
    rhat_rot = rhat @ rotation_matrix.T

    # Extract new theta, phi for each rotated vector
    z_rot = rhat_rot[:, 2]
    y_rot = rhat_rot[:, 1]
    x_rot = rhat_rot[:, 0]
    theta_rotated = np.arccos(z_rot)
    phi_rotated = np.arctan2(y_rot, x_rot)
    phi_rotated = np.mod(phi_rotated, 2*np.pi)

    # Reshape back to input
    theta_rotated = theta_rotated.reshape(orig_shape)
    phi_rotated = phi_rotated.reshape(orig_shape)
    return theta_rotated, phi_rotated