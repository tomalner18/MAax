import jax
from jax import numpy as jp
from worldgen.util.rotation import normalize_angles
from worldgen.util.geometry import raycast

def caught(origin_pts, threshold_dist=0.5):
    '''
    Computes whether agents are within a threhsold distance of each other in 2D space
    Args:
        origin_pts (jp.ndarray): array with shape (n_agents, 2) of agent x-y positions
    Returns:
        contact_mask (jp.ndarray): array with shape (n_agents, n_agents) of bools
    '''
    assert isinstance(origin_pts, jp.ndarray)
    # print(origin_pts.shape)
    assert origin_pts.shape[1] == 2

    #Initialise contact mask
    contact_mask = jp.zeros((origin_pts.shape[0], origin_pts.shape[0]), dtype=bool)
    # Populate contact_mask with whether agents are within threshold distance
    for i in range(origin_pts.shape[0]):
        for j in range(origin_pts.shape[0]):
            if i != j:
                contact_mask = contact_mask.at[i, j].set(jp.linalg.norm(origin_pts[i] - origin_pts[j]) < threshold_dist)
    return contact_mask
    


def in_cone2d(origin_pts, origin_angles, cone_angle, target_pts):
    '''
        Computes whether 2D points target_pts are in the cones originating from
            origin_pts at angle origin_angles with cone spread angle cone_angle.
        Args:
            origin_pts (jp.ndarray): array with shape (n_points, 2) of origin points
            origin_angles (jp.ndarray): array with shape (n_points,) of origin angles
            cone_angle (float): cone angle width
            target_pts (jp.ndarray): target points to check whether in cones
        Returns:
            jp.ndarray of bools. Each row corresponds to origin cone, and columns to
                target points
    '''
    assert isinstance(origin_pts, jp.ndarray)
    assert isinstance(origin_angles, jp.ndarray)
    assert isinstance(cone_angle, float)
    assert isinstance(target_pts, jp.ndarray)
    assert origin_pts.shape[0] == origin_angles.shape[0]
    assert len(origin_angles.shape) == 1, "Angles should only have 1 dimension"
    jp.seterr(divide='ignore', invalid='ignore')
    cone_vec = jp.array([jp.cos(origin_angles), jp.sin(origin_angles)]).T
    # Compute normed vectors between all pairs of agents
    pos_diffs = target_pts[None, ...] - origin_pts[:, None, :]
    norms = jp.sqrt(jp.sum(jp.square(pos_diffs), -1, keepdims=True))
    unit_diffs = pos_diffs / norms
    # Dot product between unit vector in middle of cone and the vector
    dot_cone_diff = jp.sum(unit_diffs * cone_vec[:, None, :], -1)
    angle_between = jp.arccos(dot_cone_diff)
    # Right now the only thing that should be nan will be targets that are on the origin point
    # This can only happen for the origin looking at itself, so just make this always true
    angle_between[jp.isnan(angle_between)] = 0.

    return jp.abs(normalize_angles(angle_between)) <= cone_angle
