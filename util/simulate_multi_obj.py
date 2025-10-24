import os

import h5py
import numpy as np
import torch
import warp as wp
from mpm_solver_warp.mpm_solver_warp import MPM_Simulator_WARP
from tqdm import tqdm

# Default configuration
DEFAULT_CONFIG = {
    "material_params": {
        "n_grid": 150,  # 150
        "grid_lim": 2.0,
        "material": "jelly",
        "E": 1e7,
        "nu": 0.3,
        "friction_angle": 0,
        "g": [0.0, -2.0, 0],
        "grid_v_damping_scale": 0.9999,
        "rpic_damping": 0.0,
        "density": 2000.0,
    },
    "time_params": {"substep_dt": 0.0001, "frame_dt": 0.03, "frame_num": 100},
}


def particle_position_to_ply(mpm_solver, filename):
    # position is (n,3)
    if os.path.exists(filename):
        os.remove(filename)
    position = mpm_solver.mpm_state.particle_x.numpy()
    num_particles = (position).shape[0]
    position = position.astype(np.float32)
    with open(filename, "wb") as f:  # write binary
        header = f"""ply
format binary_little_endian 1.0
element vertex {num_particles}
property float x
property float y
property float z
end_header
"""
        f.write(str.encode(header))
        f.write(position.tobytes())
        print("write", filename)


def particle_position_tensor_to_ply(position_tensor, filename):
    # position is (n,3)
    if os.path.exists(filename):
        os.remove(filename)
    position = position_tensor.clone().detach().cpu().numpy()
    num_particles = (position).shape[0]
    position = position.astype(np.float32)
    with open(filename, "wb") as f:  # write binary
        header = f"""ply
format binary_little_endian 1.0
element vertex {num_particles}
property float x
property float y
property float z
end_header
"""
        f.write(str.encode(header))
        f.write(position.tobytes())
        print("write", filename)


def save_data_at_frame(mpm_solver, dir_name, frame, save_to_ply=True, save_to_h5=False):
    os.umask(0)
    os.makedirs(dir_name, 0o777, exist_ok=True)

    fullfilename = dir_name + "/sim_" + str(frame).zfill(10) + ".h5"

    if save_to_ply:
        particle_position_to_ply(mpm_solver, fullfilename[:-2] + "ply")

    if save_to_h5:
        if os.path.exists(fullfilename):
            os.remove(fullfilename)
        newFile = h5py.File(fullfilename, "w")

        x_np = (
            mpm_solver.mpm_state.particle_x.numpy().transpose()
        )  # x_np has shape (3, n_particles)
        newFile.create_dataset("x", data=x_np)  # position

        currentTime = np.array([mpm_solver.time]).reshape(1, 1)
        newFile.create_dataset("time", data=currentTime)  # current time

        f_tensor_np = (
            mpm_solver.mpm_state.particle_F.numpy().reshape(-1, 9).transpose()
        )  # shape = (9, n_particles)
        newFile.create_dataset("f_tensor", data=f_tensor_np)  # deformation grad

        v_np = (
            mpm_solver.mpm_state.particle_v.numpy().transpose()
        )  # v_np has shape (3, n_particles)
        newFile.create_dataset("v", data=v_np)  # particle velocity

        C_np = (
            mpm_solver.mpm_state.particle_C.numpy().reshape(-1, 9).transpose()
        )  # shape = (9, n_particles)
        newFile.create_dataset("C", data=C_np)  # particle C
        print("save siumlation data at frame ", frame, " to ", fullfilename)


def get_particle_volume(pos, n_grid, dx, uniform=True):
    """
    Calculate the volume of particles

    Args:
        pos: Particle positions
        n_grid: Number of grid cells
        dx: Grid cell size
        uniform: Whether to use uniform volumes
    """
    if uniform:
        return torch.ones(pos.shape[0], device=pos.device) * (dx**3)
    else:
        # For non-uniform volumes, could implement more complex logic here
        return torch.ones(pos.shape[0], device=pos.device) * (dx**3)


def transform_objects_for_simulation(
    fg_pos_list, bg_pos, target_scale=0.4, eps=0.1, grid_lim=2.0
):
    """
    Transform all objects with the same scale and shift for simulation

    Steps:
    1. Concat all foreground objects to calculate bbox
    2. Zero-mean and scale so the longest side becomes target_scale (0.4)
    3. Apply same transformation to all fg and bg objects
    4. Shift all objects by [1,1,1]
    5. Filter background points outside valid range

    Args:
        fg_pos_list: List of foreground position tensors
        bg_pos: Background position tensor
        target_scale: Target length for longest side of bbox
        eps: Epsilon margin for valid range
        grid_lim: Grid limit

    Returns:
        Tuple of (transformed fg list, transformed bg, global_mean, global_scale)
    """
    device = fg_pos_list[0].device

    # 1. Concat all foreground objects
    all_fg_pos = torch.cat(fg_pos_list, dim=0)

    # 2. Calculate global bbox
    min_pos = torch.min(all_fg_pos, dim=0)[0]
    max_pos = torch.max(all_fg_pos, dim=0)[0]
    global_mean = (min_pos + max_pos) / 2.0

    # Calculate scale to make longest side target_scale (0.4)
    max_diff = torch.max(max_pos - min_pos)
    global_scale = target_scale / max_diff

    valid_bg_mask = torch.zeros(bg_pos.shape[0], dtype=torch.bool, device=device)
    for i in range(3):
        valid_bg_mask = (
            valid_bg_mask | (bg_pos[:, i] < min_pos[i]) | (bg_pos[:, i] > max_pos[i])
        )
    bg_pos = bg_pos[valid_bg_mask]

    # 3. Apply transformation to all foreground objects
    transformed_fg_list = []
    for pos in fg_pos_list:
        # Zero-mean and scale
        pos_transformed = (pos - global_mean) * global_scale
        # Shift by [1,1,1]
        pos_transformed = pos_transformed + torch.tensor([1.0, 1.0, 1.0], device=device)
        transformed_fg_list.append(pos_transformed)

    # 4. Apply same transformation to background
    bg_transformed = (bg_pos - global_mean) * global_scale
    bg_transformed = bg_transformed + torch.tensor([1.0, 1.0, 1.0], device=device)

    # 5. Filter background points outside valid range
    valid_mask = torch.ones(bg_transformed.shape[0], dtype=torch.bool, device=device)
    for i in range(3):
        valid_mask = (
            valid_mask
            & (bg_transformed[:, i] >= eps)
            & (bg_transformed[:, i] <= grid_lim - eps)
        )

    bg_transformed = bg_transformed[valid_mask]

    # Return transformed positions and transformation parameters
    return transformed_fg_list, bg_transformed, global_mean, global_scale


def densify_with_o3d(bg_pos, bg_covariance, sample_num=500000):
    import open3d as o3d

    # Ensure input is valid
    points = bg_pos.detach().cpu().numpy()
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Input points should be a Nx3 array.")
    if points.shape[0] == 0:
        return points
    if points.shape[0] < 3:
        # Not enough points to form a mesh; return original
        return points

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Estimate normals (required for surface reconstruction)
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = avg_dist * 2.0 if avg_dist > 0 else 0.1
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30)
    )

    # Try to create mesh via Ball-Pivoting
    mesh = None
    try:
        radii = o3d.utility.DoubleVector(
            [avg_dist * 0.5, avg_dist * 1.5, avg_dist * 2.5]
        )
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, radii
        )
        if mesh.is_empty() or len(mesh.triangles) == 0:
            mesh = None
    except Exception:
        mesh = None

    # If Ball-Pivoting failed, use Poisson reconstruction
    if mesh is None:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=8
        )
        # Crop the mesh to the original bounding box to remove spurious parts
        bbox = pcd.get_axis_aligned_bounding_box()
        mesh = mesh.crop(bbox)

    # Uniformly sample points on the mesh surface
    pcd_dense = mesh.sample_points_uniformly(number_of_points=sample_num)
    points_dense = np.asarray(pcd_dense.points)
    densified_pos = torch.tensor(points_dense, device=bg_pos.device).float()

    # Use the same covariance for all densified points
    densified_cov = bg_covariance[0].repeat(densified_pos.shape[0], 1).float()
    return densified_pos, densified_cov


def densify_with_marching_cubes(
    bg_pos, bg_covariance, grid_n, grid_lim, sample_num=500000
):
    """
    Densify background points using marching cubes and mesh sampling

    Args:
        bg_pos: Background positions tensor
        bg_covariance: Background covariance tensor
        grid_n: Number of grid cells
        grid_lim: Grid limit
        samples_per_cell: Number of samples per grid cell

    Returns:
        Tuple of (densified positions, densified covariance)
    """
    import mcubes
    import numpy as np
    import trimesh

    # Create density field
    grid_dx = grid_lim / grid_n
    density_field = np.zeros((grid_n, grid_n, grid_n))

    # Convert positions to grid indices
    grid_indices = (bg_pos.cpu().numpy() / grid_dx).astype(int)

    # Accumulate density
    for idx in grid_indices:
        if 0 <= idx[0] < grid_n and 0 <= idx[1] < grid_n and 0 <= idx[2] < grid_n:
            density_field[idx[0], idx[1], idx[2]] += 1

    # Apply marching cubes
    vertices, triangles = mcubes.marching_cubes(density_field, 0.5)

    # Scale vertices to original space
    vertices = vertices * grid_dx

    # Create trimesh object
    mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)

    samples, _ = trimesh.sample.sample_surface(mesh, sample_num)

    # # Ensure samples are within original bbox
    # samples = samples - np.mean(samples, axis=0) + original_center  # Center samples
    # scale = np.min(original_size / (np.max(samples, axis=0) - np.min(samples, axis=0)))
    # samples = (samples - original_center) * scale + original_center  # Scale to fit original bbox

    # Convert samples to tensor
    densified_pos = torch.tensor(samples, device=bg_pos.device).float()

    # Use the same covariance for all densified points
    densified_cov = bg_covariance[0].repeat(len(samples), 1).float()

    return densified_pos, densified_cov


def densify_with_voxel(bg_pos, bg_covariance, grid_n, grid_lim, sample_num=500000):
    """
    Densify background points using marching cubes and mesh-to-volume sampling

    Args:
        bg_pos: Background positions tensor
        bg_covariance: Background covariance tensor
        grid_n: Number of grid cells
        grid_lim: Grid limit
        sample_num: Number of samples to generate

    Returns:
        Tuple of (densified positions, densified covariance)
    """
    import mcubes
    import numpy as np
    import trimesh

    # Create density field
    grid_dx = grid_lim / grid_n
    density_field = np.zeros((grid_n, grid_n, grid_n))

    # Convert positions to grid indices
    grid_indices = (bg_pos.cpu().numpy() / grid_dx).astype(int)

    # Accumulate density
    for idx in grid_indices:
        if 0 <= idx[0] < grid_n and 0 <= idx[1] < grid_n and 0 <= idx[2] < grid_n:
            density_field[idx[0], idx[1], idx[2]] += 1

    # Apply marching cubes
    vertices, triangles = mcubes.marching_cubes(density_field, 0.5)

    # Scale vertices to original space
    vertices = vertices * grid_dx

    # Create trimesh object
    mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)

    # 为体素化定义适当的分辨率
    voxel_size = grid_lim / (2 * grid_n)  # 选择适当的体素大小

    # 将mesh转换为体积表示(voxels)
    voxel_grid = mesh.voxelized(pitch=voxel_size)

    # 获取体积内的点 - 这些是体素中心点
    voxel_centers = voxel_grid.points

    if len(voxel_centers) == 0:
        # 如果体素化失败，退回到表面采样
        samples, _ = trimesh.sample.sample_surface(mesh, sample_num)
    else:
        # 获取体素边长的一半，用于在体素内部随机采样
        half_size = voxel_size / 2.0

        # 计算需要从每个体素中采样的点数
        points_per_voxel = max(1, int(sample_num / len(voxel_centers)))
        all_samples = []

        # 从每个体素内部均匀采样
        for center in voxel_centers:
            # 在体素内部随机采样
            local_samples = np.random.uniform(
                -half_size, half_size, size=(points_per_voxel, 3)
            )
            # 将局部坐标转换为全局坐标
            global_samples = center + local_samples
            all_samples.append(global_samples)

        # 合并所有采样点
        samples = np.vstack(all_samples)

        # 如果采样点过多，随机选择指定数量的点
        if len(samples) > sample_num:
            indices = np.random.choice(len(samples), sample_num, replace=False)
            samples = samples[indices]

    # 将结果转换为张量
    densified_pos = torch.tensor(samples, device=bg_pos.device).float()

    # 为所有采样点使用相同的协方差
    densified_cov = bg_covariance[0].repeat(len(samples), 1).float()

    return densified_pos, densified_cov


def restore_positions(pos, global_mean, global_scale):
    """
    Restore positions to original space after simulation

    Args:
        pos: Position tensor
        global_mean: Global mean used for centering
        global_scale: Global scale used for scaling

    Returns:
        Restored positions
    """
    # Undo shift by [1,1,1]
    pos_unshifted = pos - torch.tensor([1.0, 1.0, 1.0], device=pos.device)
    # Undo scale and shift
    pos_restored = pos_unshifted / global_scale + global_mean
    return pos_restored


def simulate_multi(
    fg_pos_list,
    fg_covariance_list,
    bg_pos,
    bg_covariance,
    config=None,
    fg_material_configs=None,
    output_dir=None,
    force=None,
    use_bg=False,
):
    """
    Run MPM simulation with multiple foreground objects interacting with each other and a static background

    Args:
        fg_pos_list (list): List of torch.Tensor for foreground particle positions [(n1, 3), (n2, 3), ...]
        fg_covariance_list (list): List of torch.Tensor for foreground particle covariances [(n1, 6), (n2, 6), ...]
        bg_pos (torch.Tensor): Background particle positions (m, 3)
        bg_covariance (torch.Tensor): Background particle covariances (m, 6)
        config (dict, optional): Base configuration dictionary. If None, uses DEFAULT_CONFIG
        fg_material_configs (list, optional): List of material configurations for each foreground object
        output_dir (str, optional): Directory to save simulation results

    Returns:
        list of tuples: List of (pos_list, cov_list) for each foreground object, where each list contains states over time
    """
    # Initialize warp
    wp.init()
    wp.config.verify_cuda = True
    wp.config.mode = "debug"

    device = "cuda:0"

    # Use default config if none provided
    if config is None:
        config = DEFAULT_CONFIG.copy()

    # Get number of foreground objects
    num_fg_objects = len(fg_pos_list)

    # # Check if each foreground object has its own material configuration
    # if fg_material_configs is not None:
    #     config['material_params'] = {**config['material_params'], **fg_material_configs}
    #     print("update material configs", config['material_params'])

    # Move each foreground object to device
    for i in range(num_fg_objects):
        fg_pos_list[i] = fg_pos_list[i].to(device)
        fg_covariance_list[i] = fg_covariance_list[i].to(device)

    # Move background to device
    bg_pos = bg_pos.to(device)
    bg_covariance = bg_covariance.to(device)

    grid_n = config["material_params"]["n_grid"]
    grid_lim = config["material_params"]["grid_lim"]
    grid_dx = grid_lim / grid_n

    # Create opacity tensor for background (all ones)
    bg_opacity = torch.ones(bg_pos.shape[0], 1, device=device)

    # Transform positions using common scale and shift
    transformed_fg_list, transformed_bg, global_mean, global_scale = (
        transform_objects_for_simulation(
            fg_pos_list, bg_pos, target_scale=0.4, eps=0.15, grid_lim=grid_lim
        )
    )

    # Filter background opacities and covariances to match filtered positions
    valid_mask = torch.ones(bg_pos.shape[0], dtype=torch.bool, device=device)
    for i in range(3):
        bg_shifted = (bg_pos - global_mean) * global_scale + torch.tensor(
            [1.0, 1.0, 1.0], device=device
        )
        valid_mask = (
            valid_mask
            & (bg_shifted[:, i] >= 0.01)
            & (bg_shifted[:, i] <= grid_lim - 0.01)
        )

    bg_covariance = bg_covariance[valid_mask]

    print("before densify", transformed_bg.shape)

    # # Densify background points using marching cubes
    grid_n = config["material_params"]["n_grid"]
    grid_lim = config["material_params"]["grid_lim"]
    transformed_bg, bg_covariance = densify_with_marching_cubes(
        transformed_bg, bg_covariance, grid_n=50, grid_lim=grid_lim, sample_num=1000000
    )
    # transformed_bg, bg_covariance = densify_with_o3d(
    #     transformed_bg, bg_covariance, sample_num=1000000
    # )

    print("after densify", transformed_bg.shape)
    # import pdb
    # pdb.set_trace()

    # Track object indices

    object_indices = []

    if use_bg:
        start_idx = transformed_bg.shape[0]  # Start after background particles

        # Combine all particles: background first, then all foreground objects
        all_pos = [transformed_bg]
        all_covariance = [bg_covariance]

        print(transformed_bg.shape)
    else:
        start_idx = 0  # Start after background particles

        # Combine all particles: background first, then all foreground objects
        all_pos = []
        all_covariance = []

    # Add foreground objects and track their indices
    for i in range(num_fg_objects):
        end_idx = start_idx + transformed_fg_list[i].shape[0]
        object_indices.append((start_idx, end_idx))

        all_pos.append(transformed_fg_list[i])
        all_covariance.append(fg_covariance_list[i])

        start_idx = end_idx

    # Concatenate all particles
    pos = torch.cat(all_pos, dim=0)
    covariance = torch.cat(all_covariance, dim=0)

    # Get number of background particles
    num_bg = transformed_bg.shape[0] if use_bg else 0
    total_particles = pos.shape[0]

    # Initialize MPM solver
    mpm_solver = MPM_Simulator_WARP(total_particles)

    # Calculate particle volumes
    n_grid = config["material_params"]["n_grid"]
    dx = grid_lim / n_grid
    volume = get_particle_volume(
        pos, n_grid, dx, uniform=(config["material_params"]["material"] == "sand")
    )

    # Load initial data
    mpm_solver.load_initial_data_from_torch(
        pos, volume, covariance, n_grid=n_grid, grid_lim=grid_lim
    )

    # Set base material parameters

    # Set background particles as static (not affected by physics)
    mpm_solver.mpm_state.object_selection[:].fill_(1)  # First set all to dynamic
    mpm_solver.mpm_state.object_selection[num_bg:].fill_(0)  # Set background to static

    # Set particle selection to allow all particles to participate in simulation
    mpm_solver.mpm_state.particle_selection[:].fill_(
        0
    )  # All particles participate in simulation

    mpm_solver.set_parameters_dict(config["material_params"])

    # Set different material parameters for each foreground object
    if fg_material_configs is not None:
        for i, (start_idx, end_idx) in enumerate(object_indices):
            if i < len(fg_material_configs):
                fg_material_configs[i] = {
                    **config["material_params"],
                    **fg_material_configs[i],
                }
                # Get the bounding box of this object
                obj_pos = pos[start_idx:end_idx]
                min_pos = torch.min(obj_pos, dim=0)[0]
                max_pos = torch.max(obj_pos, dim=0)[0]
                center = (min_pos + max_pos) / 2.0
                size = (max_pos - min_pos) / 2.0

                # Add material parameters for this object
                additional_params = {
                    "point": center.cpu().numpy().tolist(),
                    "size": size.cpu().numpy().tolist(),
                    **fg_material_configs[i],
                }

                # Update material parameters for this object
                mpm_solver.set_parameters_dict(
                    {"additional_material_params": [additional_params]}
                )

    mpm_solver.add_bounding_box()
    mpm_solver.enforce_scene_velocity_translation(
        velocity=[0, 0, 0],
        start_time=0,
        end_time=1e3,
    )

    if force is not None:
        if isinstance(force, list) and isinstance(
            force[0], (list, tuple, np.ndarray, torch.Tensor)
        ):
            # force为list，每个物体一个力
            for i, (start_idx, end_idx) in enumerate(object_indices):
                if i < len(force):
                    # 计算物体的中心和范围
                    obj_pos = pos[start_idx:end_idx]
                    min_pos = torch.min(obj_pos, dim=0)[0]
                    max_pos = torch.max(obj_pos, dim=0)[0]
                    center = (min_pos + max_pos) / 2.0
                    size = (max_pos - min_pos) / 2.0

                    print(center, size)
                    mpm_solver.add_impulse_on_object(
                        force=force[i],
                        dt=config["time_params"]["substep_dt"],
                        point=center.cpu().numpy().tolist(),
                        size=size.cpu().numpy().tolist(),
                        num_dt=2,  # 2
                        start_time=0.05,
                    )
        else:
            # 单个力，作用于所有物体
            mpm_solver.add_impulse_on_object(
                force=force,
                dt=config["time_params"]["substep_dt"],
                num_dt=2,
                start_time=0.05,
            )

    # Finalize material parameters
    mpm_solver.finalize_mu_lam()

    # Add surface collider
    # if "surface_collider" in config:
    #     collider = config["surface_collider"]
    #     mpm_solver.add_surface_collider(
    #         point=collider["point"],
    #         normal=collider["normal"],
    #         surface=collider["surface"],
    #         friction=collider["friction"]
    #     )

    if not use_bg:
        # Add surface collider slightly below the minimum y value
        eps = 0.01  # Small offset to prevent penetration
        min_y = pos.min(0).values[1].item()
        mpm_solver.add_surface_collider(
            point=[0, min_y - eps, 0],  # Point on the surface, slightly below min y
            normal=[0.0, 1.0, 0.0],  # Normal vector pointing up
            surface="slip",  # Surface type (0 for static)
            friction=0.0,  # Friction coefficient
        )

    # Create output directory if specified
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        save_data_at_frame(
            mpm_solver, output_dir, 0, save_to_ply=True, save_to_h5=False
        )

    # Run simulation
    substep_dt = config["time_params"]["substep_dt"]
    frame_dt = config["time_params"]["frame_dt"]
    frame_num = config["time_params"]["frame_num"]
    step_per_frame = int(frame_dt / substep_dt)

    # Create lists to store each foreground object's states over time
    fg_results = []
    for _ in range(num_fg_objects):
        fg_results.append(([], []))  # (pos_list, cov_list) for each object

    for frame in tqdm(range(frame_num)):
        for step in range(step_per_frame):
            mpm_solver.p2g2p(frame, substep_dt, device=device)

        # Get current state for all particles
        current_pos = mpm_solver.export_particle_x_to_torch()
        current_cov = mpm_solver.export_particle_cov_to_torch().view(-1, 6)

        # Save each foreground object's data separately
        for i in range(num_fg_objects):
            start_idx, end_idx = object_indices[i]
            # Get positions in simulation space
            sim_pos = current_pos[start_idx:end_idx]

            # Transform back to original space
            orig_pos = restore_positions(sim_pos, global_mean, global_scale)

            fg_results[i][0].append(orig_pos.cpu())
            fg_results[i][1].append(current_cov[start_idx:end_idx].cpu())

        # Save frame if output directory specified
        if output_dir is not None:
            save_data_at_frame(
                mpm_solver, output_dir, frame + 1, save_to_ply=True, save_to_h5=False
            )

    return fg_results


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, help="Output directory")
    parser.add_argument(
        "--fg_ply_list", nargs="+", type=str, help="List of foreground PLY file paths"
    )
    parser.add_argument("--bg_ply", type=str, help="Background PLY file path")
    args = parser.parse_args()

    # Example: Load particles from PLY files or generate random ones
    if args.fg_ply_list and args.bg_ply:
        # Code to load from PLY files would go here
        raise NotImplementedError("Loading from PLY not yet implemented")
    else:
        # Example tensors for multiple foreground objects and background

        # Background particles (floor-like)
        n_bg_particles = 1000
        bg_pos = torch.rand(n_bg_particles, 3) * 2.0 - 1.0  # Positions in range [-1, 1]
        bg_pos[:, 1] = (
            bg_pos[:, 1] * 0.1 - 0.9
        )  # Flatten in Y dimension to create floor
        bg_covariance = torch.rand(n_bg_particles, 6) * 0.01

        # Create multiple foreground objects
        fg_pos_list = []
        fg_covariance_list = []

        # Foreground object 1 (left) - soft jelly
        n_fg_particles1 = 800
        fg_pos1 = (
            torch.rand(n_fg_particles1, 3) * 0.4 - 0.5
        )  # Positions around [-0.5, -0.1]
        fg_pos1[:, 1] += 0.5  # Move up in Y dimension
        fg_covariance1 = torch.rand(n_fg_particles1, 6) * 0.01
        fg_opacity1 = torch.ones(n_fg_particles1, 1)

        # Foreground object 2 (right) - harder material
        n_fg_particles2 = 600
        fg_pos2 = (
            torch.rand(n_fg_particles2, 3) * 0.4 + 0.1
        )  # Positions around [0.1, 0.5]
        fg_pos2[:, 1] += 0.3  # Move up in Y dimension
        fg_covariance2 = torch.rand(n_fg_particles2, 6) * 0.01
        fg_opacity2 = torch.ones(n_fg_particles2, 1)

        fg_pos_list = [fg_pos1, fg_pos2]
        fg_covariance_list = [fg_covariance1, fg_covariance2]

        # Define different material parameters for each foreground object
        fg_material_configs = [
            {
                "material": "jelly",  # Soft jelly material
                "E": 500,  # Lower Young's modulus (softer)
                "nu": 0.3,
                "density": 150.0,
            },
            {
                "material": "sand",  # Harder material
                "E": 2000,  # Higher Young's modulus (harder)
                "nu": 0.2,
                "density": 300.0,
            },
        ]

    # Run multi-object simulation with different material parameters
    fg_results = simulate_multi(
        fg_pos_list,
        fg_covariance_list,
        bg_pos,
        bg_covariance,
        fg_material_configs=fg_material_configs,
        output_dir=args.output,
    )

    print(f"Simulation complete for {len(fg_results)} foreground objects")
    for i, (pos_list, cov_list) in enumerate(fg_results):
        print(
            f"Object {i + 1}: {len(pos_list)} frames, {pos_list[0].shape[0]} particles per frame"
        )
