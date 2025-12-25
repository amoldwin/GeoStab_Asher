import click
import torch

ATOM_N, ATOM_CA, ATOM_C, ATOM_O, ATOM_CB = 0, 1, 2, 3, 4


def NormVec(V):
    eps = 1e-10
    axis_x = V[:, 2] - V[:, 1]
    axis_x /= torch.norm(axis_x, dim=-1).unsqueeze(1) + eps
    axis_y = V[:, 0] - V[:, 1]
    axis_z = torch.cross(axis_x, axis_y, dim=1)
    axis_z /= torch.norm(axis_z, dim=-1).unsqueeze(1) + eps
    axis_y = torch.cross(axis_z, axis_x, dim=1)
    axis_y /= torch.norm(axis_y, dim=-1).unsqueeze(1) + eps
    Vec = torch.stack([axis_x, axis_y, axis_z], dim=1)
    return Vec


def QuaternionMM(q1, q2):
    a = q1[..., 0] * q2[..., 0] - (q1[..., 1:] * q2[..., 1:]).sum(-1)
    bcd = (
        torch.cross(q2[..., 1:], q1[..., 1:], dim=-1)
        + q1[..., 0].unsqueeze(-1) * q2[..., 1:]
        + q2[..., 0].unsqueeze(-1) * q1[..., 1:]
    )
    q = torch.cat([a.unsqueeze(-1), bcd], dim=-1)
    return q


def NormQuaternionMM(q1, q2):
    q = QuaternionMM(q1, q2)
    return q / torch.sqrt((q * q).sum(-1, keepdim=True))


def Rotation2Quaternion(r):
    a = torch.sqrt(r[..., 0, 0] + r[..., 1, 1] + r[..., 2, 2] + 1) / 2.0
    b = (r[..., 2, 1] - r[..., 1, 2]) / (4 * a)
    c = (r[..., 0, 2] - r[..., 2, 0]) / (4 * a)
    d = (r[..., 1, 0] - r[..., 0, 1]) / (4 * a)
    q = torch.stack([a, b, c, d], dim=-1)
    q = q / torch.sqrt((q * q).sum(-1, keepdim=True))
    return q


def NormQuaternion(q):
    q = q / torch.sqrt((q * q).sum(-1, keepdim=True))
    q = torch.sign(torch.sign(q[..., 0]) + 0.5).unsqueeze(-1) * q
    return q


def infer_atom_mask_from_pos14(pos14: torch.Tensor) -> torch.Tensor:
    finite = torch.isfinite(pos14).all(dim=-1)
    nonzero = (pos14.abs().sum(dim=-1) > 0)
    return finite & nonzero

def get_atom_mask_from_coord_dict(coord: dict, pos14: torch.Tensor) -> torch.Tensor:
    """
    Returns atom_mask: [L,14] bool.
    Handles masks stored as [L,14] or [L,14,3].
    """
    for k in ["atom14_mask", "pos14_mask", "mask14", "mask", "atom_mask", "atom_mask14"]:
        if k in coord:
            m = coord[k]

            # convert numeric to bool
            if m.dtype != torch.bool:
                m = m != 0

            # Some pipelines store a per-coordinate mask [L,14,3]
            if m.dim() == 3 and m.shape[-1] == 3 and m.shape[:2] == pos14.shape[:2]:
                # atom is valid only if all xyz coords are valid
                m = m.all(dim=-1)

            assert m.shape == pos14.shape[:2], f"{k} shape {m.shape} != {pos14.shape[:2]}"
            return m

    return infer_atom_mask_from_pos14(pos14)



def rbf_expand(d: torch.Tensor, centers: torch.Tensor, gamma: float) -> torch.Tensor:
    if d.dim() == 2:
        d = d.unsqueeze(-1)
    return torch.exp(-gamma * (d - centers) ** 2)


def compute_dmin_chunked(pos14: torch.Tensor, atom_mask: torch.Tensor, chunk: int = 64) -> torch.Tensor:
    device = pos14.device
    L, A, _ = pos14.shape
    dmin = torch.empty((L, L), device=device, dtype=pos14.dtype)

    pos_j = pos14[None, :, None, :, :]       # [1, L, 1, A, 3]
    mask_j = atom_mask[None, :, None, :]     # [1, L, 1, A]

    inf = torch.tensor(float("inf"), device=device, dtype=pos14.dtype)

    for i0 in range(0, L, chunk):
        i1 = min(L, i0 + chunk)
        pos_i = pos14[i0:i1, None, :, None, :]    # [c, 1, A, 1, 3]
        mask_i = atom_mask[i0:i1, None, :, None]  # [c, 1, A, 1]

        diff = pos_j - pos_i
        dist = torch.linalg.norm(diff, dim=-1)    # [c, L, A, A]
        valid = mask_i & mask_j
        dist = dist.masked_fill(~valid, inf)
        block = dist.amin(dim=(-1, -2))           # [c, L]
        block = torch.where(torch.isfinite(block), block, torch.full_like(block, 20.0))
        dmin[i0:i1] = block

    return dmin


def build_pair(
    coordinate_file: str,
    saved_folder: str,
    rbf_k: int = 16,
    rbf_min: float = 2.0,
    rbf_max: float = 12.0,
    dmin_chunk: int = 64,
):
    coord = torch.load(coordinate_file)
    print("pos14", coord["pos14"].shape, coord["pos14"].dtype)
    print("pos14_mask", coord["pos14_mask"].shape, coord["pos14_mask"].dtype)

    pos14 = coord["pos14"]
    assert pos14.shape[1:] == (14, 3)
    L = pos14.shape[0]

    atom_mask = get_atom_mask_from_coord_dict(coord, pos14)  # [L,14] bool
    assert atom_mask.shape == (pos14.shape[0], 14)
    
    # backbone frame + quats (same as original)
    rotation = NormVec(pos14[:, :3, :])  # [L,3,3]
    U, _, V = torch.svd(torch.eye(3, device=pos14.device).unsqueeze(0).permute(0, 2, 1) @ rotation)
    d = torch.sign(torch.det(U @ V.permute(0, 2, 1)))
    Id = torch.eye(3, device=pos14.device).repeat(L, 1, 1)
    Id[:, 2, 2] = d
    r = V @ (Id @ U.permute(0, 2, 1))  # [L,3,3]

    q = Rotation2Quaternion(r)  # [L,4]
    q_1 = torch.cat([q[..., 0].unsqueeze(-1), -q[..., 1:]], dim=-1)

    QAll = NormQuaternionMM(
        q.unsqueeze(1).repeat(1, L, 1),
        q_1.unsqueeze(0).repeat(L, 1, 1),
    )  # [L,L,4]
    QAll[..., 0][torch.isnan(QAll[..., 0])] = 1.0
    QAll[torch.isnan(QAll)] = 0.0
    QAll = NormQuaternion(QAll)

    # xyz_CA (existing)
    xyz_CA = torch.einsum(
        "a b i, a i j -> a b j",
        pos14[:, ATOM_CA].unsqueeze(0) - pos14[:, ATOM_CA].unsqueeze(1),
        r,
    )  # [L,L,3]

    # xyz_CB (new)
    ca = pos14[:, ATOM_CA]                    # [L,3]
    cb_exists = atom_mask[:, ATOM_CB]         # [L]
    cb = torch.where(cb_exists.unsqueeze(-1), pos14[:, ATOM_CB], ca)

    vec_cb = cb.unsqueeze(0) - ca.unsqueeze(1)  # [L,L,3]  (a,b) = CB[b] - CA[a]
    xyz_CB = torch.einsum("a b i, a i j -> a b j", vec_cb, r)  # [L,L,3]

    # dmin + RBF (new)
    dmin = compute_dmin_chunked(pos14, atom_mask, chunk=dmin_chunk)  # [L,L]
    centers = torch.linspace(rbf_min, rbf_max, rbf_k, device=pos14.device, dtype=pos14.dtype)

    if rbf_k > 1:
        delta = (rbf_max - rbf_min) / (rbf_k - 1)
        gamma = 1.0 / (delta * delta)
    else:
        gamma = 1.0

    dmin_rbf = rbf_expand(dmin, centers, gamma)  # [L,L,rbf_k]

    pair_feat = torch.cat([xyz_CA, QAll, xyz_CB, dmin_rbf], dim=-1)  # [L,L, 10+rbf_k]
    torch.save(pair_feat.detach().cpu().clone(), f"{saved_folder}/pair.pt")
    return pair_feat


@click.command()
@click.option("--coordinate_file", required=True, type=str)
@click.option("--saved_folder", required=True, type=str)
@click.option("--rbf_k", default=16, show_default=True, type=int)
@click.option("--rbf_min", default=2.0, show_default=True, type=float)
@click.option("--rbf_max", default=12.0, show_default=True, type=float)
@click.option("--dmin_chunk", default=64, show_default=True, type=int)
def cli(coordinate_file, saved_folder, rbf_k, rbf_min, rbf_max, dmin_chunk):
    build_pair(
        coordinate_file=coordinate_file,
        saved_folder=saved_folder,
        rbf_k=rbf_k,
        rbf_min=rbf_min,
        rbf_max=rbf_max,
        dmin_chunk=dmin_chunk,
    )


if __name__ == "__main__":
    cli()
