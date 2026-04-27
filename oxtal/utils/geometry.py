import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation

from oxtal.model.utils import expand_at_dim


def angle_3p(a, b, c):
    """
    Calculate the angle between three points in a 2D space.

    Args:
        a (list or array-like): The coordinates of the first point.
        b (list or array-like): The coordinates of the second point.
        c (list or array-like): The coordinates of the third point.

    Returns:
        float: The angle in degrees (0, 180) between the vectors
               from point a to point b and point b to point c.
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ab = b - a
    bc = c - b

    dot_product = np.dot(ab, bc)

    norm_ab = np.linalg.norm(ab)
    norm_bc = np.linalg.norm(bc)

    cos_theta = np.clip(dot_product / (norm_ab * norm_bc + 1e-4), -1, 1)
    theta_radians = np.arccos(cos_theta)
    theta_degrees = np.degrees(theta_radians)
    return theta_degrees


def random_transform(
    points, max_translation=1.0, apply_augmentation=False, centralize=True
) -> np.ndarray:
    """
    Randomly transform a set of 3D points.

    Args:
        points (numpy.ndarray): The points to be transformed, shape=(N, 3)
        max_translation (float): The maximum translation value. Default is 1.0.
        apply_augmentation (bool): Whether to apply random rotation/translation on ref_pos

    Returns:
        numpy.ndarray: The transformed points.
    """
    if centralize:
        points = points - points.mean(axis=0)
    if not apply_augmentation:
        return points
    translation = np.random.uniform(-max_translation, max_translation, size=3)
    R = Rotation.random().as_matrix()
    transformed_points = np.dot(points + translation, R.T)
    return transformed_points


class DistOneHotCalculator:
    def __init__(
        self,
        dist_bin_min: float,
        dist_bin_max: float,
        num_bins: int,
    ):
        # TODO: BEFORE MERGING verify that the bin min/max values make sense
        self.bins = torch.linspace(
            dist_bin_min,
            dist_bin_max,
            steps=num_bins - 1,
        ).pow(2)

        self.squared_bin_max = dist_bin_max**2
        self.num_bins = num_bins

    def get_centre_dist_one_hot(self, atom_positions: torch.Tensor, input_feature_dict: dict):
        if self.bins.device != atom_positions.device:
            self.bins = self.bins.to(device=atom_positions.device)

        rep_atom_positions = None
        rep_atom_mask = input_feature_dict["distogram_rep_atom_mask"].to(dtype=torch.bool)
        match atom_positions.ndim:
            case 2:
                rep_atom_positions = atom_positions[rep_atom_mask]
            case 3:
                rep_atom_positions = atom_positions[:, rep_atom_mask]
            case _:
                raise ValueError(f"Had invalid atom_positions ndim of {atom_positions.ndim}")

        diffs = rep_atom_positions.unsqueeze(dim=-3) - rep_atom_positions.unsqueeze(dim=-2)
        squared_l2 = diffs.pow(2).sum(dim=-1)

        n_tokens = input_feature_dict["residue_index"].shape[-1]
        not_same_res_mask = ~torch.diag(
            torch.ones((n_tokens,), device=atom_positions.device, dtype=torch.bool)
        )

        if atom_positions.ndim == 3:
            not_same_res_mask = not_same_res_mask.unsqueeze(0).repeat(len(atom_positions), 1, 1)

        mask = (squared_l2 < self.squared_bin_max) & not_same_res_mask

        bins = self.bins
        while bins.ndim < squared_l2.ndim + 1:
            bins = bins.unsqueeze(dim=0)

        dist_bins = (bins < squared_l2.unsqueeze(dim=-1)).sum(dim=-1)

        one_hots = F.one_hot(dist_bins, num_classes=self.num_bins).to(dtype=atom_positions.dtype)

        return one_hots * mask[..., None]
