import torch
from torch import FloatTensor, LongTensor, Tensor, Size, lerp, zeros_like
from torch.linalg import norm

from scipy.spatial.transform import Slerp
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation

import numpy as np
import math

from lse_nerf.utils import EPS

def matrix_to_tangent_vector(matrix):
    """
    Converts a 4x4 transformation matrix to its tangent vector representation in SO(3)xR^3.

    Args:
        matrix: A 4x4 homogeneous transformation matrix.

    Returns:
        A 6-element vector where the first 3 elements are the translation part and the
        last 3 elements are the so(3) tangent vector representation of the rotation.
    """
    # Ensure the matrix is a PyTorch tensor
    # matrix = torch.tensor(matrix, dtype=torch.float32)
    
    # Extract the translation part directly
    translation = matrix[:3, 3]

    # Extract the rotation matrix
    rotation_matrix = matrix[:3, :3]
    
    # Compute the axis-angle representation of the rotation
    angle = torch.acos((torch.trace(rotation_matrix) - 1) / 2)
    if angle.abs() < 1e-6:
        # Handle the case of very small angles where the direction is not well-defined
        axis = torch.tensor([0, 0, 1])  # Default to some arbitrary axis
    else:
        # For a non-trivial rotation, compute the axis of rotation
        axis = torch.tensor([
            rotation_matrix[2, 1] - rotation_matrix[1, 2],
            rotation_matrix[0, 2] - rotation_matrix[2, 0],
            rotation_matrix[1, 0] - rotation_matrix[0, 1],
        ]) / (2 * torch.sin(angle))
    
    # Scale the axis by the angle to get the so(3) representation
    so3_vector = axis * angle
    
    # Combine the translation and rotation parts into a single vector
    tangent_vector = torch.cat([translation, so3_vector])

    return tangent_vector


def slerp(v0, v1, t):
    """
    input:
        v0 (torch.tensor): (n, 4) start quaterion
        v1 (torch.tensor): (n, 4) end quaterion
        t (torch.tensor): (n,) time in [0, 1]

    return:
        interpolated quaterion
    """
    v0_normed = v0 / torch.norm(v0, dim=1, keepdim=True)
    v1_normed = v1 / torch.norm(v1, dim=1, keepdim=True)


    dot = torch.sum(v0_normed * v1_normed, dim=1, keepdim=True).clamp(-1.0 + EPS, 1.0 - EPS)
    dot_mag = torch.abs(dot)

    dot_neg = dot < 0
    v1_normed = torch.where(dot_neg, -v1_normed, v1_normed)
    dot = torch.where(dot_neg, -dot, dot)

    # Handle near-zero rotations
    is_near_zero = dot_mag.isnan() | (dot_mag > 0.9995)
    linear_rot = (1 - t) * v0_normed + t * v1_normed  # Linear interpolation for small angles
    
    # Handle rotations near 180 degrees
    # theta = torch.acos(dot) * t
    theta_0 = torch.acos(dot)     
    theta_t = theta_0 * t
    sin_theta_t = torch.sin(theta_t)
    sin_theta_0 = torch.sin(theta_0)

    # Prevent division by zero
    sin_theta_0 = torch.where(sin_theta_0 == 0, torch.ones_like(sin_theta_0), sin_theta_0)
    
    s0 = torch.sin(theta_0 - theta_t) / sin_theta_0
    s1 = sin_theta_t / sin_theta_0

    slerp_rot = s0 * v0_normed + s1 * v1_normed
    
    # Choose SLERP or linear interpolation based on the condition
    rot_interpolated = torch.where(is_near_zero.expand_as(slerp_rot), linear_rot, slerp_rot)

    return rot_interpolated


def vectorized_generalized_interpolation(control_poses, control_ts, interp_ts):
    control_poses = control_poses.float()
    control_ts = control_ts.float()
    interp_ts = interp_ts.float()

    indices = torch.searchsorted(control_ts, interp_ts, right=True)
    indices = torch.clamp(indices, 1, len(control_ts) - 1) - 1

    start_poses = control_poses[indices]
    end_poses = control_poses[indices + 1]
    start_ts = control_ts[indices]
    end_ts = control_ts[indices + 1]

    t = ((interp_ts - start_ts) / (end_ts - start_ts)).unsqueeze(-1)

    # Linear interpolation for translation
    trans_interpolated = (1 - t) * start_poses[:, :3] + t * end_poses[:, :3]

    # Enhanced SLERP for rotation handling edge cases
    v0 = start_poses[:, 3:]   # start rotation
    v1 = end_poses[:, 3:]     # end rotation

    rot_interpolated = slerp(v0, v1, t)

    interpolated_poses = torch.cat([trans_interpolated, rot_interpolated], dim=1)

    return interpolated_poses



def hom_exp_map_SO3xR3(tangent_vector):
    """Compute the exponential map of the direct product group `SO(3) x R^3`.

    This can be used for learning pose deltas on SE(3), and is generally faster than `exp_map_SE3`.

    Args:
        tangent_vector: Tangent vector; length-3 translations, followed by an `so(3)` tangent vector.
    Returns:
        [R|t] transformation matrices.
    """
    # code for SO3 map grabbed from pytorch3d and stripped down to bare-bones
    log_rot = tangent_vector[:, 3:]
    nrms = (log_rot * log_rot).sum(1)
    rot_angles = torch.clamp(nrms, 1e-4).sqrt()
    rot_angles_inv = 1.0 / rot_angles
    fac1 = rot_angles_inv * rot_angles.sin()
    fac2 = rot_angles_inv * rot_angles_inv * (1.0 - rot_angles.cos())
    skews = torch.zeros((log_rot.shape[0], 3, 3), dtype=log_rot.dtype, device=log_rot.device)
    skews[:, 0, 1] = -log_rot[:, 2]
    skews[:, 0, 2] = log_rot[:, 1]
    skews[:, 1, 0] = log_rot[:, 2]
    skews[:, 1, 2] = -log_rot[:, 0]
    skews[:, 2, 0] = -log_rot[:, 1]
    skews[:, 2, 1] = log_rot[:, 0]
    skews_square = torch.bmm(skews, skews)

    ret = torch.zeros(tangent_vector.shape[0], 4, 4, dtype=tangent_vector.dtype, device=tangent_vector.device)
    ret[:, 3, 3] = 1.0
    ret[:, :3, :3] = (
        fac1[:, None, None] * skews
        + fac2[:, None, None] * skews_square
        + torch.eye(3, dtype=log_rot.dtype, device=log_rot.device)[None]
    )

    # Compute the translation
    ret[:, :3, 3] = tangent_vector[:, :3]
    return ret



def exp_map_to_quat(v):
    """
    Converts a batch of exponential maps (rotation vectors) to quaternions using PyTorch.
    
    Parameters:
    - v: A PyTorch tensor of shape (n, 3), where each row is a 3-element rotation vector.
    
    Returns:
    - A PyTorch tensor of shape (n, 4), where each row is a 4-element quaternion.
    """
    # Compute the rotation angles (magnitudes of the rotation vectors)
    thetas = torch.norm(v, dim=1)
    
    # Compute the normalized rotation axes, handling division by zero for zero rotations
    n = torch.zeros_like(v)
    valid = thetas != 0
    n[valid] = v[valid] / thetas[valid].unsqueeze(1)
    
    # Compute the quaternion components
    quaternions = torch.zeros((v.shape[0], 4), dtype=v.dtype, device=v.device)
    quaternions[:, 0] = torch.cos(thetas / 2)  # w components
    sin_thetas_over_2 = torch.sin(thetas / 2)
    quaternions[valid, 1] = n[valid, 0] * sin_thetas_over_2[valid]  # x components
    quaternions[valid, 2] = n[valid, 1] * sin_thetas_over_2[valid]  # y components
    quaternions[valid, 3] = n[valid, 2] * sin_thetas_over_2[valid]  # z components
    
    return quaternions


def quat_to_rot_mat(quaternions):
    """
    Converts a batch of quaternions to rotation matrices using PyTorch.
    
    Parameters:
    - quaternions: A PyTorch tensor of shape (n, 4), representing a batch of quaternions.
    
    Returns:
    - A PyTorch tensor of shape (n, 3, 3), representing the corresponding rotation matrices.
    """
    # Extract quaternion components
    w, x, y, z = quaternions[:, 0], quaternions[:, 1], quaternions[:, 2], quaternions[:, 3]
    
    # Pre-compute squared components
    x2, y2, z2 = x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    
    # Initialize rotation matrices
    rotation_matrices = torch.empty((quaternions.size(0), 3, 3), dtype=quaternions.dtype, device=quaternions.device)
    
    # Fill in the rotation matrix components
    rotation_matrices[:, 0, 0] = 1.0 - 2.0 * (y2 + z2)
    rotation_matrices[:, 0, 1] = 2.0 * (xy - wz)
    rotation_matrices[:, 0, 2] = 2.0 * (xz + wy)
    rotation_matrices[:, 1, 0] = 2.0 * (xy + wz)
    rotation_matrices[:, 1, 1] = 1.0 - 2.0 * (x2 + z2)
    rotation_matrices[:, 1, 2] = 2.0 * (yz - wx)
    rotation_matrices[:, 2, 0] = 2.0 * (xz - wy)
    rotation_matrices[:, 2, 1] = 2.0 * (yz + wx)
    rotation_matrices[:, 2, 2] = 1.0 - 2.0 * (x2 + y2)
    
    return rotation_matrices


def exp_map_to_quat_map(exp_map):
    out = torch.empty(len(exp_map), 7, device=exp_map.device)
    out[:,:3] = exp_map[:,:3]
    out[:, 3:] = exp_map_to_quat(exp_map[:, 3:])
    return out


def quat_map_to_mtx(quat_map):
    quat = quat_map[:, 3:]
    rot = quat_to_rot_mat(quat)
    return torch.concat([rot, quat_map[:, :3][..., None]], axis = 2)


def unit_vector(data, axis=None, out=None):
    """Return ndarray normalized by length, i.e. Euclidean norm, along axis.
    >>> v0 = np.random.random(3)
    >>> v1 = unit_vector(v0)
    >>> np.allclose(v1, v0 / np.linalg.norm(v0))
    True
    >>> v0 = np.random.rand(5, 4, 3)
    >>> v1 = unit_vector(v0, axis=-1)
    >>> v2 = v0 / np.expand_dims(np.sqrt(np.sum(v0*v0, axis=2)), 2)
    >>> np.allclose(v1, v2)
    True
    >>> v1 = unit_vector(v0, axis=1)
    >>> v2 = v0 / np.expand_dims(np.sqrt(np.sum(v0*v0, axis=1)), 1)
    >>> np.allclose(v1, v2)
    True
    >>> v1 = np.empty((5, 4, 3))
    >>> unit_vector(v0, axis=1, out=v1)
    >>> np.allclose(v1, v2)
    True
    >>> list(unit_vector([]))
    []
    >>> list(unit_vector([1]))
    [1.0]
    """
    if out is None:
        data = np.array(data, dtype=np.float64, copy=True)
        if data.ndim == 1:
            data /= math.sqrt(np.dot(data, data))
            return data
    else:
        if out is not data:
            out[:] = np.array(data, copy=False)
        data = out
    length = np.atleast_1d(np.sum(data*data, axis))
    np.sqrt(length, length)
    if axis is not None:
        length = np.expand_dims(length, axis)
    data /= length
    if out is None:
        return data
    return None

def quaternion_slerp(quat0, quat1, fraction, spin=0, shortestpath=True):
    """Return spherical linear interpolation between two quaternions.
    >>> q0 = random_quaternion()
    >>> q1 = random_quaternion()
    >>> q = quaternion_slerp(q0, q1, 0)
    >>> np.allclose(q, q0)
    True
    >>> q = quaternion_slerp(q0, q1, 1, 1)
    >>> np.allclose(q, q1)
    True
    >>> q = quaternion_slerp(q0, q1, 0.5)
    >>> angle = math.acos(np.dot(q0, q))
    >>> np.allclose(2, math.acos(np.dot(q0, q1)) / angle) or \
        np.allclose(2, math.acos(-np.dot(q0, q1)) / angle)
    True
    """
    _EPS = 1e-6
    q0 = unit_vector(quat0[:4])  # [0.73673994, 0.57996866, 0.13155995, 0.32178034]
    q1 = unit_vector(quat1[:4])  # [ 0.06532506, -0.8060292 ,  0.42178851, -0.41005367]
    if fraction == 0.0:
        return q0
    if fraction == 1.0:
        return q1
    d = np.dot(q0, q1)  # -0.49580083080612775
    if abs(abs(d) - 1.0) < _EPS:
        return q0
    if shortestpath and d < 0.0:
        # invert rotation
        d = -d   # 0.49580083080612775
        np.negative(q1, q1)   # [-0.06532506,  0.8060292 , -0.42178851,  0.41005367]
    angle = math.acos(d) + spin * math.pi    # 1.0520395849865911
    if abs(angle) < _EPS:
        return q0
    isin = 1.0 / math.sin(angle)        # 1.151494
    q0 *= math.sin((1.0 - fraction) * angle) * isin  # [0.42595364, 0.33531474, 0.07606271, 0.18604055]
    q1 *= math.sin(fraction * angle) * isin # [-0.03776834,  0.46601392, -0.24386129,  0.23707667]
    q0 += q1 
    return q0 # [ 0.3881853 ,  0.80132866, -0.16779858,  0.42311722]

class SciSpline:
    def __init__(self, w2cs, ts) -> None:
        self.w2cs = w2cs 
        self.ts = ts

        self.Rs = w2cs[:,:3,:3]
        self.trans = w2cs[:,:3,3:]

        self.rot_interp = Slerp(self.ts, Rotation.from_matrix(self.Rs))
        self.trans_interp = interp1d(self.ts, self.trans, axis=0, kind="linear")
    
    def interpolate(self, ts):
        ts = np.clip(ts, self.ts[0] + 1e-6, self.ts[-1] - 1e-6)
        r, t = self.rot_interp(ts).as_matrix(), self.trans_interp(ts)
        return np.concatenate([r, t], axis=2)

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.reshape(-1)
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

def gen_data(n_rnd = 10, n_bins=2, max_t = 10):
    np.random.seed(0)
    rots = Rotation.random(n_rnd).as_matrix()
    trans = np.random.rand(n_rnd, 3, 1) * 4
    w2cs = np.concatenate([rots, trans], axis= 2)
    ts = np.arange(0, max_t, max_t/n_rnd)
    return ts.astype(np.float32), w2cs.astype(np.float32)

def to_qvecs(w2cs):
    ts = w2cs[:,:3,3]
    Rs = w2cs[:,:3,:3]

    qvecs = np.stack([rotmat2qvec(mat) for mat in Rs])
    return np.concatenate([ts, qvecs], axis = 1)

def to_rot(vec):
    qvecs = vec[:,3:]
    mats = np.stack([qvec2rotmat(qvec) for qvec in qvecs])

    return np.concatenate([mats, vec[:,:3][..., None]], axis = 2)

def test_slerp(n_rnd = 10, n_bins=2, max_t = 10):
    ts, w2cs = gen_data(n_rnd, n_bins, max_t)
    interp_ts = np.arange(0, max_t, max_t/(n_bins * n_rnd))
    interp_ts = interp_ts[interp_ts <= ts.max()]

    sspline = SciSpline(w2cs, ts)

    # A correct reference implementation
    qvecs = to_qvecs(w2cs)[:, 3:]
    o2 = qvec2rotmat(quaternion_slerp(qvecs[0], qvecs[1], 0.5))

    tcams = vectorized_generalized_interpolation(torch.from_numpy(to_qvecs(w2cs)), 
                                                 torch.from_numpy(ts), 
                                                 torch.from_numpy(interp_ts))
    tcams = to_rot(tcams)
    scams = sspline.interpolate(interp_ts)

    assert np.abs(tcams - scams).max() < 1e-5
    print("slerp test pass")



def test_tangent_convt():
    _, w2cs = gen_data()
    w2cs = torch.from_numpy(w2cs)
    assert torch.abs(w2cs - hom_exp_map_SO3xR3(torch.stack([matrix_to_tangent_vector(M) for M in w2cs]))[:,:3]).max() < 1e-5
    print('tangent, mtx conversion pass test')


def test_exp_quat_mx_convt():
    _, w2cs = gen_data()
    w2cs = torch.from_numpy(w2cs)
    exp_map = torch.stack([matrix_to_tangent_vector(M) for M in w2cs])[:,3:]
    recov_R = quat_to_rot_mat(exp_map_to_quat(exp_map))

    max_err = torch.abs(w2cs[:,:3,:3] - recov_R).max()
    assert max_err < 1e-5
    print("pass w2c -> exp_map -> quat -> w2cs", max_err)


def test_exp_map_interp(n_rnd = 10, n_bins=2, max_t = 10):
    ts, w2cs = gen_data(n_rnd, n_bins, max_t)
    # ts, w2cs = ts[:2], w2cs[:2]
    interp_ts = np.arange(0, max_t, max_t/(n_bins * n_rnd)).astype(np.float32)
    interp_ts = interp_ts[interp_ts <= ts.max()][:2]

    sspline = SciSpline(w2cs, ts)

    exp_map = torch.stack([matrix_to_tangent_vector(M) for M in torch.from_numpy(w2cs)]).float() #  [ 1.7544,  3.9535,  0.4082, -2.4320,  1.2727, -1.2373]
    exp_map.requires_grad_()
    quat_map = exp_map_to_quat_map(exp_map) # [ 1.7544,  3.9535,  0.4082,  0.0653, -0.8060,  0.4218, -0.4101]

    # [ 1.5081,  2.7042,  1.3445,  0.3882,  0.8013, -0.1678,  0.4231]
    tcams = vectorized_generalized_interpolation(quat_map, 
                                                 torch.from_numpy(ts),
                                                 torch.from_numpy(interp_ts))
    tcams = quat_map_to_mtx(tcams)
    scams = sspline.interpolate(interp_ts)

    assert np.abs(tcams.detach().numpy() - scams).max() < 1e-5
    print("pipeline test pass")

    # check if exp_map has grad
    loss = quat_map.sum()
    loss.backward()
    assert exp_map.grad is not None


if __name__ == "__main__":
    # test_tangent_convt()
    # test_slerp()
    # test_exp_quat_mx_convt()
    test_exp_map_interp()