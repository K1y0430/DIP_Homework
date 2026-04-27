import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2


# =========================================================
# Config

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(SCRIPT_DIR, "data")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")
FIG_DIR = os.path.join(SCRIPT_DIR, "figures")
REPROJ_DIR = os.path.join(FIG_DIR, "reprojection")

IMAGE_DIR = os.path.join(DATA_DIR, "images")

IMAGE_SIZE = 1024
CX = IMAGE_SIZE / 2.0
CY = IMAGE_SIZE / 2.0

INIT_FOCAL = 1000.0
INIT_DIST = 2.5

NUM_ITERS = 4000
LR = 5e-3
PRINT_EVERY = 100

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================================================
# Utils

def ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(REPROJ_DIR, exist_ok=True)


def load_data(data_dir=DATA_DIR, device=DEVICE):

    points2d = np.load(os.path.join(data_dir, "points2d.npz"))
    colors = np.load(os.path.join(data_dir, "points3d_colors.npy"))

    view_names = sorted(points2d.keys())
    obs_xy_list = []
    vis_list = []

    for k in view_names:
        arr = points2d[k]  # (N, 3)
        obs_xy_list.append(arr[:, :2])
        vis_list.append(arr[:, 2])

    obs_xy = torch.tensor(np.stack(obs_xy_list), dtype=torch.float32, device=device)         # (V, N, 2)
    visibility = torch.tensor(np.stack(vis_list), dtype=torch.float32, device=device)         # (V, N)
    colors = torch.tensor(colors, dtype=torch.float32, device=device)                         # (N, 3)

    return obs_xy, visibility, colors, view_names


def euler_xyz_to_matrix(euler_angles):

    rx = euler_angles[:, 0]
    ry = euler_angles[:, 1]
    rz = euler_angles[:, 2]

    cx = torch.cos(rx)
    sx = torch.sin(rx)
    cy = torch.cos(ry)
    sy = torch.sin(ry)
    cz = torch.cos(rz)
    sz = torch.sin(rz)

    Rx = torch.zeros((euler_angles.shape[0], 3, 3), dtype=euler_angles.dtype, device=euler_angles.device)
    Ry = torch.zeros((euler_angles.shape[0], 3, 3), dtype=euler_angles.dtype, device=euler_angles.device)
    Rz = torch.zeros((euler_angles.shape[0], 3, 3), dtype=euler_angles.dtype, device=euler_angles.device)

    # Rx
    Rx[:, 0, 0] = 1.0
    Rx[:, 1, 1] = cx
    Rx[:, 1, 2] = -sx
    Rx[:, 2, 1] = sx
    Rx[:, 2, 2] = cx

    # Ry
    Ry[:, 0, 0] = cy
    Ry[:, 0, 2] = sy
    Ry[:, 1, 1] = 1.0
    Ry[:, 2, 0] = -sy
    Ry[:, 2, 2] = cy

    # Rz
    Rz[:, 0, 0] = cz
    Rz[:, 0, 1] = -sz
    Rz[:, 1, 0] = sz
    Rz[:, 1, 1] = cz
    Rz[:, 2, 2] = 1.0

    # XYZ convention
    R = Rz @ Ry @ Rx
    return R


def initialize_parameters(obs_xy, visibility, init_focal=INIT_FOCAL, init_dist=INIT_DIST, device=DEVICE):
    """
    Initialize:
        focal length
        Euler angles
        translations
        3D points
    """
    num_views, num_points, _ = obs_xy.shape

    # Use log_f so that focal = exp(log_f) > 0
    log_f = nn.Parameter(torch.tensor(math.log(init_focal), dtype=torch.float32, device=device))

    # Camera rotations: initialize near frontal views
    euler_angles = torch.zeros((num_views, 3), dtype=torch.float32, device=device)
    yaw_init = torch.linspace(-0.8, 0.8, num_views, device=device)  # roughly [-46°, +46°]
    euler_angles[:, 1] = yaw_init
    euler_angles = nn.Parameter(euler_angles)

    # Camera translations
    translations = torch.zeros((num_views, 3), dtype=torch.float32, device=device)
    translations[:, 2] = -init_dist
    translations = nn.Parameter(translations)

    # 3D points initialization from average 2D observations
    vis_sum = visibility.sum(dim=0).clamp(min=1.0)  # (N,)
    mean_u = (obs_xy[:, :, 0] * visibility).sum(dim=0) / vis_sum
    mean_v = (obs_xy[:, :, 1] * visibility).sum(dim=0) / vis_sum

    # Rough inverse projection under front view assumption
    X0 = (mean_u - CX) * init_dist / init_focal
    Y0 = -(mean_v - CY) * init_dist / init_focal
    Z0 = 0.02 * torch.randn_like(X0)

    points3d = torch.stack([X0, Y0, Z0], dim=-1)
    points3d = nn.Parameter(points3d)

    return log_f, euler_angles, translations, points3d


def project_points(points3d, R, T, focal, image_size=IMAGE_SIZE):

    cx = image_size / 2.0
    cy = image_size / 2.0

    # Expand points to all views
    P = points3d.unsqueeze(0).expand(R.shape[0], -1, -1)

    cam_xyz = torch.bmm(P, R.transpose(1, 2)) + T.unsqueeze(1)

    Xc = cam_xyz[..., 0]
    Yc = cam_xyz[..., 1]
    Zc = cam_xyz[..., 2]

    eps = 1e-8
    u = -focal * Xc / (Zc + eps) + cx
    v = focal * Yc / (Zc + eps) + cy

    pred_xy = torch.stack([u, v], dim=-1)  # (V, N, 2)
    return pred_xy, cam_xyz


def compute_loss(pred_xy, obs_xy, visibility, cam_xyz, points3d, euler_angles, translations):

    # Reprojection error
    diff = pred_xy - obs_xy
    sq_err = (diff ** 2).sum(dim=-1)   # (V, N)
    reproj_loss = (sq_err * visibility).sum() / (visibility.sum() + 1e-8)

    # Depth constraint: according to assignment, points should satisfy Zc < 0
    zc = cam_xyz[..., 2]
    depth_penalty = (F.relu(zc + 1e-3) ** 2 * visibility).sum() / (visibility.sum() + 1e-8)

    # Mild regularization to reduce drift
    center_reg = (points3d.mean(dim=0) ** 2).sum()
    point_reg = (points3d ** 2).mean()
    trans_reg = (translations ** 2).mean()

    # Slightly anchor the first camera
    euler_anchor = (euler_angles[0] ** 2).sum()
    trans_anchor = ((translations[0] - torch.tensor([0.0, 0.0, -INIT_DIST], device=translations.device)) ** 2).sum()

    loss = (
        reproj_loss
        + 1e-2 * depth_penalty
        + 1e-4 * center_reg
        + 1e-5 * point_reg
        + 1e-5 * trans_reg
        + 1e-4 * euler_anchor
        + 1e-4 * trans_anchor
    )

    stats = {
        "reproj_loss": reproj_loss.item(),
        "depth_penalty": depth_penalty.item(),
        "center_reg": center_reg.item(),
    }
    return loss, stats


def compute_final_reprojection_rmse(obs_xy, visibility, points3d, euler_angles, translations, log_f):
    with torch.no_grad():
        focal = torch.exp(log_f)
        R = euler_xyz_to_matrix(euler_angles)
        pred_xy, _ = project_points(points3d, R, translations, focal, IMAGE_SIZE)

        diff = pred_xy - obs_xy
        sq_err = (diff ** 2).sum(dim=-1)
        mse = (sq_err * visibility).sum() / (visibility.sum() + 1e-8)
        rmse = torch.sqrt(mse)
        return rmse.item()


def save_obj(path, points3d, colors):

    pts = points3d.detach().cpu().numpy()
    cols = colors.detach().cpu().numpy()

    if cols.max() > 1.0:
        cols = cols / 255.0

    with open(path, "w") as f:
        for p, c in zip(pts, cols):
            f.write(f"v {p[0]} {p[1]} {p[2]} {c[0]} {c[1]} {c[2]}\n")


def save_camera_params(path, focal, euler_angles, translations):
    np.savez(
        path,
        focal=np.array([focal], dtype=np.float32),
        euler_angles=euler_angles.detach().cpu().numpy(),
        translations=translations.detach().cpu().numpy(),
    )


def plot_loss_curve(loss_history, save_path):
    plt.figure(figsize=(7, 4.5))
    plt.plot(loss_history, linewidth=1.5)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Bundle Adjustment Optimization Loss")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def visualize_point_cloud(points3d, colors, save_path):
    
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    pts = points3d.detach().cpu().numpy()
    cols = colors.detach().cpu().numpy()

    if cols.max() > 1.0:
        cols = cols / 255.0

    # Sample points for faster plotting
    n = pts.shape[0]
    max_show = 5000
    if n > max_show:
        idx = np.random.choice(n, max_show, replace=False)
        pts = pts[idx]
        cols = cols[idx]

    # make axes roughly equal
    max_range = np.array([
        pts[:, 0].max() - pts[:, 0].min(),
        pts[:, 1].max() - pts[:, 1].min(),
        pts[:, 2].max() - pts[:, 2].min()
    ]).max() / 2.0

    mid_x = (pts[:, 0].max() + pts[:, 0].min()) * 0.5
    mid_y = (pts[:, 1].max() + pts[:, 1].min()) * 0.5
    mid_z = (pts[:, 2].max() + pts[:, 2].min()) * 0.5

    # save multiple viewpoints
    views = [
        (20, -60, "view1"),
        (20, 30, "view2"),
        (0, 90, "front"),
        (0, 0, "side"),
        (90, -90, "top"),
    ]

    base, ext = os.path.splitext(save_path)

    for elev, azim, name in views:
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=cols, s=1)

        ax.set_title(f"Reconstructed 3D Point Cloud ({name})")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        ax.view_init(elev=elev, azim=azim)

        plt.tight_layout()
        plt.savefig(f"{base}_{name}{ext}", dpi=200)
        plt.close()


def visualize_reprojection(obs_xy, visibility, points3d, euler_angles, translations, log_f, view_names, save_dir):

    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        focal = torch.exp(log_f)
        R = euler_xyz_to_matrix(euler_angles)
        pred_xy, _ = project_points(points3d, R, translations, focal, IMAGE_SIZE)

        obs_xy_np = obs_xy.detach().cpu().numpy()
        pred_xy_np = pred_xy.detach().cpu().numpy()
        vis_np = visibility.detach().cpu().numpy().astype(bool)

        selected_views = [0, 12, 25, 37, 49]

        for v in selected_views:
            key = view_names[v]
            img_path = os.path.join(IMAGE_DIR, f"{key}.png")

            if os.path.exists(img_path):
                img = cv2.imread(img_path)
            else:
                img = np.ones((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8) * 255

            vis = vis_np[v]
            obs = obs_xy_np[v][vis]
            pred = pred_xy_np[v][vis]

            max_show = 3000
            if len(obs) > max_show:
                idx = np.random.choice(len(obs), max_show, replace=False)
                obs = obs[idx]
                pred = pred[idx]

            # observed: green
            for p in obs:
                x, y = int(round(p[0])), int(round(p[1]))
                if 0 <= x < IMAGE_SIZE and 0 <= y < IMAGE_SIZE:
                    cv2.circle(img, (x, y), 2, (0, 255, 0), -1)

            # predicted: red
            for p in pred:
                x, y = int(round(p[0])), int(round(p[1]))
                if 0 <= x < IMAGE_SIZE and 0 <= y < IMAGE_SIZE:
                    cv2.circle(img, (x, y), 2, (0, 0, 255), -1)

            out_path = os.path.join(save_dir, f"{key}_reproj.png")
            cv2.imwrite(out_path, img)


def main():
    print(f"Using device: {DEVICE}")
    ensure_dirs()

    # 1. Load data
    obs_xy, visibility, colors, view_names = load_data(DATA_DIR, DEVICE)
    num_views, num_points, _ = obs_xy.shape
    print(f"Loaded {num_views} views and {num_points} points.")

    # 2. Initialize parameters
    log_f, euler_angles, translations, points3d = initialize_parameters(
        obs_xy, visibility, init_focal=INIT_FOCAL, init_dist=INIT_DIST, device=DEVICE
    )

    # 3. Optimizer
    optimizer = torch.optim.Adam(
        [log_f, euler_angles, translations, points3d],
        lr=LR
    )

    loss_history = []

    # 4. Optimization loop
    for it in range(NUM_ITERS):
        focal = torch.exp(log_f)
        R = euler_xyz_to_matrix(euler_angles)

        pred_xy, cam_xyz = project_points(points3d, R, translations, focal, IMAGE_SIZE)
        loss, stats = compute_loss(pred_xy, obs_xy, visibility, cam_xyz, points3d, euler_angles, translations)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

        if it % PRINT_EVERY == 0 or it == NUM_ITERS - 1:
            print(
                f"[{it:04d}/{NUM_ITERS}] "
                f"loss={loss.item():.6f} | "
                f"reproj={stats['reproj_loss']:.6f} | "
                f"depth={stats['depth_penalty']:.6f} | "
                f"f={focal.item():.4f}"
            )

    # 5. Save results
    final_focal = torch.exp(log_f).item()
    final_rmse = compute_final_reprojection_rmse(
        obs_xy, visibility, points3d, euler_angles, translations, log_f
    )

    print("\nOptimization finished.")
    print(f"Final focal length: {final_focal:.6f}")
    print(f"Final reprojection RMSE: {final_rmse:.4f} pixels")

    save_obj(os.path.join(OUTPUT_DIR, "reconstructed.obj"), points3d, colors)
    save_camera_params(
        os.path.join(OUTPUT_DIR, "camera_params.npz"),
        final_focal,
        euler_angles,
        translations
    )
    plot_loss_curve(loss_history, os.path.join(FIG_DIR, "loss_curve.png"))
    visualize_point_cloud(points3d, colors, os.path.join(FIG_DIR, "point_cloud.png"))
    visualize_reprojection(
        obs_xy,
        visibility,
        points3d,
        euler_angles,
        translations,
        log_f,
        view_names,
        REPROJ_DIR
    )

    print(f"Saved OBJ to: {os.path.join(OUTPUT_DIR, 'reconstructed.obj')}")
    print(f"Saved camera params to: {os.path.join(OUTPUT_DIR, 'camera_params.npz')}")
    print(f"Saved loss curve to: {os.path.join(FIG_DIR, 'loss_curve.png')}")
    print(f"Saved point cloud figure to: {os.path.join(FIG_DIR, 'point_cloud.png')}")
    print(f"Saved reprojection figures to: {REPROJ_DIR}")


if __name__ == "__main__":
    main()