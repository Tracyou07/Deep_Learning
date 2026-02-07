#!/usr/bin/env python3
"""
Modified from CUT3R: https://github.com/CUT3R/CUT3R

Online Human-Scene Reconstruction Inference and Visualization Script

This script performs inference using the ARCroco3DStereo model and visualizes the
resulting 3D scene point clouds and SMPLX sequences with the SceneHumanViewer. 
Use the command-line arguments to adjust parameters 
such as the model checkpoint path, image sequence directory, image size, device, etc.

Example:
    python demo.py --model_path src/human3r.pth --size 512 \
        --seq_path examples/GoodMornin1.mp4 --subsample 1 --vis_threshold 2 \
        --downsample_factor 1 --use_ttt3r --reset_interval 100
"""

import os
import numpy as np
import torch
import time
import glob
import random
import cv2
import argparse
import tempfile
import shutil
from copy import deepcopy
from add_ckpt_path import add_path_to_dust3r
import imageio.v2 as iio
import roma
import json
import matplotlib.pyplot as plt

# Set random seed for reproducibility.
random.seed(42)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run 3D point cloud inference and visualization using ARCroco3DStereo."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="src/cut3r_512_dpt_4_64.pth",
        help="Path to the pretrained model checkpoint.",
    )
    parser.add_argument(
        "--seq_path",
        type=str,
        default="",
        help="Path to the directory containing the image sequence.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on (e.g., 'cuda' or 'cpu').",
    )
    parser.add_argument(
        "--size",
        type=int,
        default="512",
        help="Shape that input images will be rescaled to; if using 224+linear model, choose 224 otherwise 512",
    )
    parser.add_argument(
        "--vis_threshold",
        type=float,
        default=1.5,
        help="Visualization threshold for the viewer. Ranging from 1 to INF",
    )
    parser.add_argument(
        "--msk_threshold",
        type=float,
        default=0.1,
        help="Mask threshold. Ranging from 0 to 1",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./tmp",
        help="value for tempfile.tempdir",
    )
    parser.add_argument(
        "--save_smpl",
        action="store_true",
        help="Save smpl results.",
    )
    parser.add_argument(
        "--save_video",
        action="store_true",
        help="Save smpl video.",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Max frames to use. Default is None (use all images).",
    )
    parser.add_argument(
        "--subsample",
        type=int,
        default=1,
        help="Subsample factor for input images. Default is 1 (use all images).",
    )
    parser.add_argument(
        "--reset_interval", 
        type=int, 
        default=10000000
        )
    parser.add_argument(
        "--use_ttt3r",
        action="store_true",
        help="Use TTT3R.",
        default=False
    )
    parser.add_argument(
        "--downsample_factor",
        type=int,
        default=10,
        help="Point cloud downsample factor for the viewer",
    )
    parser.add_argument(
        "--smpl_downsample",
        type=int,
        default=1,
        help="SMPL sequence downsample factor for the viewer",
    )
    parser.add_argument(
        "--camera_downsample",
        type=int,
        default=1,
        help="Camera motion downsample factor for the viewer",
    )
    parser.add_argument(
        "--mask_morph",
        type=int,
        default=10,
        help="Mask morphology for the viewer",
    )
    return parser.parse_args()


def prepare_input(
    img_paths, 
    img_mask, 
    size, 
    raymaps=None, 
    raymap_mask=None, 
    revisit=1, 
    update=True, 
    img_res=None, 
    reset_interval=100
):
    """
    Prepare input views for inference from a list of image paths.

    Args:
        img_paths (list): List of image file paths.
        img_mask (list of bool): Flags indicating valid images.
        size (int): Target image size.
        raymaps (list, optional): List of ray maps.
        raymap_mask (list, optional): Flags indicating valid ray maps.
        revisit (int): How many times to revisit each view.
        update (bool): Whether to update the state on revisits.

    Returns:
        list: A list of view dictionaries.
    """
    # Import image loader (delayed import needed after adding ckpt path).
    from src.dust3r.utils.image import load_images, pad_image
    from dust3r.utils.geometry import get_camera_parameters

    images = load_images(img_paths, size=size)
    if img_res is not None:
        K_mhmr = get_camera_parameters(img_res, device="cpu") # if use pseudo K

    views = []
    if raymaps is None and raymap_mask is None:
        # Only images are provided.
        for i in range(len(images)):
            view = {
                "img": images[i]["img"],
                "ray_map": torch.full(
                    (
                        images[i]["img"].shape[0],
                        6,
                        images[i]["img"].shape[-2],
                        images[i]["img"].shape[-1],
                    ),
                    torch.nan,
                ),
                "true_shape": torch.from_numpy(images[i]["true_shape"]),
                "idx": i,
                "instance": str(i),
                "camera_pose": torch.from_numpy(
                    np.eye(4, dtype=np.float32)
                    ).unsqueeze(0),
                "img_mask": torch.tensor(True).unsqueeze(0),
                "ray_mask": torch.tensor(False).unsqueeze(0),
                "update": torch.tensor(True).unsqueeze(0),
                "reset": torch.tensor((i+1) % reset_interval == 0).unsqueeze(0),
            }
            if img_res is not None:
                view["img_mhmr"] = pad_image(view["img"], img_res)
                view["K_mhmr"] = K_mhmr
            views.append(view)
            if (i+1) % reset_interval == 0:
                overlap_view = deepcopy(view)
                overlap_view["reset"] = torch.tensor(False).unsqueeze(0)
                views.append(overlap_view)
    else:
        # Combine images and raymaps.
        num_views = len(images) + len(raymaps)
        assert len(img_mask) == len(raymap_mask) == num_views
        assert sum(img_mask) == len(images) and sum(raymap_mask) == len(raymaps)

        j = 0
        k = 0
        for i in range(num_views):
            view = {
                "img": (
                    images[j]["img"]
                    if img_mask[i]
                    else torch.full_like(images[0]["img"], torch.nan)
                ),
                "ray_map": (
                    raymaps[k]
                    if raymap_mask[i]
                    else torch.full_like(raymaps[0], torch.nan)
                ),
                "true_shape": (
                    torch.from_numpy(images[j]["true_shape"])
                    if img_mask[i]
                    else torch.from_numpy(np.int32([raymaps[k].shape[1:-1][::-1]]))
                ),
                "idx": i,
                "instance": str(i),
                "camera_pose": torch.from_numpy(
                    np.eye(4, dtype=np.float32)
                    ).unsqueeze(0),
                "img_mask": torch.tensor(img_mask[i]).unsqueeze(0),
                "ray_mask": torch.tensor(raymap_mask[i]).unsqueeze(0),
                "update": torch.tensor(img_mask[i]).unsqueeze(0),
                "reset": torch.tensor((i+1) % reset_interval == 0).unsqueeze(0),
            }
            if img_res is not None:
                view["img_mhmr"] = pad_image(view["img"], img_res)
                view["K_mhmr"] = K_mhmr
            if img_mask[i]:
                j += 1
            if raymap_mask[i]:
                k += 1
            views.append(view)
            if (i+1) % reset_interval == 0:
                overlap_view = deepcopy(view)
                overlap_view["reset"] = torch.tensor(False).unsqueeze(0)
                views.append(overlap_view)
        assert j == len(images) and k == len(raymaps)

    if revisit > 1:
        new_views = []
        for r in range(revisit):
            for i, view in enumerate(views):
                new_view = deepcopy(view)
                new_view["idx"] = r * len(views) + i
                new_view["instance"] = str(r * len(views) + i)
                if r > 0 and not update:
                    new_view["update"] = torch.tensor(False).unsqueeze(0)
                new_views.append(new_view)
        return new_views

    return views

def prepare_output(
        outputs, outdir, revisit=1, use_pose=True, 
        save_smpl=False, save_video=False, img_res=None, subsample=1):
    """
    Process inference outputs to generate point clouds and camera parameters for visualization.

    Args:
        outputs (dict): Inference outputs.
        revisit (int): Number of revisits per view.
        use_pose (bool): Whether to transform points using camera pose.
        save_smpl (bool): Whether to save smpl results.
        save_video (bool): Whether to save smpl video.

    Returns:
        tuple: (points, colors, confidence, camera parameters dictionary)
    """
    from src.dust3r.utils.camera import pose_encoding_to_camera
    from src.dust3r.post_process import estimate_focal_knowing_depth
    from src.dust3r.utils.geometry import geotrf, matrix_cumprod
    from src.dust3r.utils import SMPL_Layer, vis_heatmap, render_meshes
    from src.dust3r.utils.image import unpad_image
    from viser_utils import get_color

    # Only keep the outputs corresponding to one full pass.
    valid_length = len(outputs["pred"]) // revisit
    outputs["pred"] = outputs["pred"][-valid_length:]
    outputs["views"] = outputs["views"][-valid_length:]

    # delet overlaps: reset_mask=True outputs["pred"] and outputs["views"]
    reset_mask = torch.cat([view["reset"] for view in outputs["views"]], 0)
    shifted_reset_mask = torch.cat([torch.tensor(False).unsqueeze(0), reset_mask[:-1]], dim=0)
    outputs["pred"] = [
        pred for pred, mask in zip(outputs["pred"], shifted_reset_mask) if not mask]
    outputs["views"] = [
        view for view, mask in zip(outputs["views"], shifted_reset_mask) if not mask]
    reset_mask = reset_mask[~shifted_reset_mask]

    pts3ds_self_ls = [output["pts3d_in_self_view"] for output in outputs["pred"]]
    pts3ds_other = [output["pts3d_in_other_view"] for output in outputs["pred"]]
    conf_self = [output["conf_self"] for output in outputs["pred"]]
    conf_other = [output["conf"] for output in outputs["pred"]]
    pts3ds_self = torch.cat(pts3ds_self_ls, 0)

    # Recover camera poses.
    pr_poses = [
        pose_encoding_to_camera(pred["camera_pose"].clone()).cpu()
        for pred in outputs["pred"]
    ]

    # reset_mask = torch.cat([view["reset"] for view in outputs["views"]], 0)
    if reset_mask.any():
        pr_poses = torch.cat(pr_poses, 0)
        identity = torch.eye(4, device=pr_poses.device)
        reset_poses = torch.where(reset_mask.unsqueeze(-1).unsqueeze(-1), pr_poses, identity)
        cumulative_bases = matrix_cumprod(reset_poses)
        shifted_bases = torch.cat([identity.unsqueeze(0), cumulative_bases[:-1]], dim=0)
        pr_poses = torch.einsum('bij,bjk->bik', shifted_bases, pr_poses)
        # keeps only reset_mask=False pr_poses
        pr_poses = list(pr_poses.unsqueeze(1).unbind(0))

    R_c2w = torch.cat([pr_pose[:, :3, :3] for pr_pose in pr_poses], 0)
    t_c2w = torch.cat([pr_pose[:, :3, 3] for pr_pose in pr_poses], 0)

    if use_pose:
        transformed_pts3ds_other = []
        for pose, pself in zip(pr_poses, pts3ds_self):
            transformed_pts3ds_other.append(geotrf(pose, pself.unsqueeze(0)))
        pts3ds_other = transformed_pts3ds_other
        conf_other = conf_self

    # Estimate focal length based on depth.
    B, H, W, _ = pts3ds_self.shape
    pp = torch.tensor([W // 2, H // 2], device=pts3ds_self.device).float().repeat(B, 1)
    focal = estimate_focal_knowing_depth(pts3ds_self, pp, focal_mode="weiszfeld")

    colors = [
        0.5 * (output["img"].permute(0, 2, 3, 1) + 1.0) for output in outputs["views"]
    ]

    cam_dict = {
        "focal": focal.numpy(),
        "pp": pp.numpy(),
        "R": R_c2w.numpy(),
        "t": t_c2w.numpy(),
    }

    pts3ds_self_tosave = pts3ds_self  # B, H, W, 3
    depths_tosave = pts3ds_self_tosave[..., 2]
    pts3ds_other_tosave = torch.cat(pts3ds_other)  # B, H, W, 3
    conf_self_tosave = torch.cat(conf_self)  # B, H, W
    conf_other_tosave = torch.cat(conf_other)  # B, H, W
    colors_tosave = torch.cat(
        [
            0.5 * (output["img"].permute(0, 2, 3, 1) + 1.0)
            for output in outputs["views"]
        ]
    )  # [B, H, W, 3]
    cam2world_tosave = torch.cat(pr_poses)  # B, 4, 4
    intrinsics_tosave = (
        torch.eye(3).unsqueeze(0).repeat(cam2world_tosave.shape[0], 1, 1)
    )  # B, 3, 3
    intrinsics_tosave[:, 0, 0] = focal.detach()
    intrinsics_tosave[:, 1, 1] = focal.detach()
    intrinsics_tosave[:, 0, 2] = pp[:, 0]
    intrinsics_tosave[:, 1, 2] = pp[:, 1]

    # get SMPL parameters from outputs
    smpl_shape = [output.get(
        "smpl_shape", torch.empty(1,0,10))[0] for output in outputs["pred"]]
    smpl_rotvec = [roma.rotmat_to_rotvec(
        output.get(
            "smpl_rotmat", torch.empty(1,0,53,3,3))[0]) for output in outputs["pred"]]
    smpl_transl = [output.get(
        "smpl_transl", torch.empty(1,0,3))[0] for output in outputs["pred"]]
    smpl_expression = [output.get(
        "smpl_expression", [None])[0] for output in outputs["pred"]]
    smpl_id = [output.get(
        "smpl_id", torch.empty(1,0))[0] for output in outputs["pred"]]
    # smpl_loc = [output.get(
    #     "smpl_loc", torch.empty(1,0,2))[0] for output in outputs["pred"]]
    # K_mhmr = [output.get(
    #     "K_mhmr", torch.empty(1,0,3))[0] for output in outputs["views"]]
        
    if save_smpl:
        smpl_scores = [
            output["smpl_scores"][...,0] for output in outputs["pred"]]
        if img_res is not None:
            smpl_scores = [
                unpad_image(s, [H, W])[0] for s in smpl_scores]

    has_mask = "msk" in outputs["pred"][0]
    if has_mask:
        msks = [output["msk"][...,0] for output in outputs["pred"]]
        if img_res is not None:
            msks = [unpad_image(m, [H, W]) for m in msks]
    else:
        msks = [torch.zeros(1, H, W) for _ in range(B)]

    # SMPL layer
    smpl_layer = SMPL_Layer(type='smplx', 
                            gender='neutral', 
                            num_betas=smpl_shape[0].shape[-1], 
                            kid=False, 
                            person_center='head')
    smpl_faces = smpl_layer.bm_x.faces

    # os.makedirs(os.path.join(outdir, "depth"), exist_ok=True)
    # os.makedirs(os.path.join(outdir, "conf"), exist_ok=True)
    # os.makedirs(os.path.join(outdir, "color"), exist_ok=True)
    # os.makedirs(os.path.join(outdir, "camera"), exist_ok=True)

    all_verts = []
    for f_id in range(B):
        n_humans_i = smpl_shape[f_id].shape[0]
        
        if n_humans_i > 0:
            with torch.no_grad():
                smpl_out = smpl_layer(
                    smpl_rotvec[f_id], 
                    smpl_shape[f_id], 
                    smpl_transl[f_id], 
                    None, None, 
                    K=intrinsics_tosave[f_id].expand(n_humans_i, -1 , -1), 
                    expression=smpl_expression[f_id])
        
        depth = depths_tosave[f_id].numpy()
        conf = conf_self_tosave[f_id].numpy()
        color = colors_tosave[f_id].numpy()
        c2w = cam2world_tosave[f_id].numpy()
        intrins = intrinsics_tosave[f_id].numpy()

        if n_humans_i > 0:
            # transform smpl verts to world coordinates
            all_verts.append(geotrf(pr_poses[f_id], smpl_out['smpl_v3d'].unsqueeze(0))[0])
            pr_verts = [t.numpy() for t in smpl_out['smpl_v3d'].unbind(0)]
            pr_faces = [smpl_faces] * n_humans_i
        else:
            pr_verts = []
            pr_faces = []
            all_verts.append(torch.empty(0))

        if save_smpl:
            hm = vis_heatmap(colors_tosave[f_id], smpl_scores[f_id]).numpy()
            img_array_np = (color * 255).astype(np.uint8)
            smpl_rend = render_meshes(img_array_np.copy(), pr_verts, pr_faces,
                                        {'focal': intrins[[0,1],[0,1]], 
                                        'princpt': intrins[[0,1],[-1,-1]]},
                                        color=[get_color(i)/255 for i in smpl_id[f_id]])
            if has_mask:
                msk_array_np = vis_heatmap(colors_tosave[f_id], msks[f_id][0]).numpy()
                color_smpl = np.concatenate([
                    img_array_np, 
                    (msk_array_np * 255).astype(np.uint8), 
                    (hm * 255).astype(np.uint8), 
                    smpl_rend], 1)
            else:
                color_smpl = np.concatenate([
                    img_array_np, 
                    (hm * 255).astype(np.uint8), 
                    smpl_rend], 1)
        
        # np.save(os.path.join(outdir, "depth", f"{f_id:06d}.npy"), depth)
        # np.save(os.path.join(outdir, "conf", f"{f_id:06d}.npy"), conf)
        # iio.imwrite(
        #     os.path.join(outdir, "color", f"{f_id:06d}.png"),
        #     (color * 255).astype(np.uint8),
        # )
        # np.savez(
        #     os.path.join(outdir, "camera", f"{f_id:06d}.npz"),
        #     pose=c2w,
        #     intrinsics=intrins,
        # )

        # Save smpl results
        if save_smpl:
            os.makedirs(os.path.join(outdir, "color_smpl"), exist_ok=True)
            iio.imwrite(
                os.path.join(outdir, "color_smpl", f"{f_id:06d}.png"),
                color_smpl,
            )
            # os.makedirs(os.path.join(outdir, "smpl"), exist_ok=True)
            # np.savez(
            #     os.path.join(outdir, "smpl", f"{f_id:06d}.npz"),
            #     scores=smpl_scores[f_id].numpy(),
            #     msk=msks[f_id].numpy() if has_mask else None,
            #     shape=smpl_shape[f_id].numpy(),
            #     rotvec=smpl_rotvec[f_id].numpy(),
            #     transl=smpl_transl[f_id].numpy(),
            #     expression=smpl_expression[f_id].numpy() if smpl_expression[f_id] is not None else None
            # )

    if save_smpl and save_video:
        frames_dir = os.path.join(outdir, "color_smpl")
        video_path = os.path.join(outdir, "output_video.mp4")
        output_fps = 30 // subsample
        os.system(f'/usr/bin/ffmpeg -y -framerate {output_fps} -i "{frames_dir}/%06d.png" '
                f'-vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" '
                f'-vcodec h264 -preset fast -profile:v baseline -pix_fmt yuv420p '
                f'-movflags +faststart -b:v 5000k "{video_path}"')
    
    return (
        pts3ds_other,
        colors, 
        conf_other, 
        cam_dict, 
        all_verts, 
        smpl_faces,
        smpl_id,
        msks
    )

def parse_seq_path(p):
    if os.path.isdir(p):
        img_paths = sorted(glob.glob(f"{p}/*"))
        tmpdirname = None
    else:
        cap = cv2.VideoCapture(p)
        if not cap.isOpened():
            raise ValueError(f"Error opening video file {p}")
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if video_fps == 0:
            cap.release()
            raise ValueError(f"Error: Video FPS is 0 for {p}")
        frame_interval = 1
        frame_indices = list(range(0, total_frames, frame_interval))
        print(
            f" - Video FPS: {video_fps}, Frame Interval: {frame_interval}, Total Frames to Read: {len(frame_indices)}"
        )
        img_paths = []
        tmpdirname = tempfile.mkdtemp()
        for i in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break
            frame_path = os.path.join(tmpdirname, f"frame_{i}.jpg")
            cv2.imwrite(frame_path, frame)
            img_paths.append(frame_path)
        cap.release()
    return img_paths, tmpdirname


def run_inference(args):
    """
    Execute the full inference and visualization pipeline.

    Args:
        args: Parsed command-line arguments.
    """
    # Set up the computation device.
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available. Switching to CPU.")
        device = "cpu"

    # Add the checkpoint path (required for model imports in the dust3r package).
    add_path_to_dust3r(args.model_path)

    # Import model and inference functions after adding the ckpt path.
    from src.dust3r.inference import inference_recurrent_lighter
    from src.dust3r.model import ARCroco3DStereo
    from viser_utils import SceneHumanViewer

    # Prepare image file paths.
    img_paths, tmpdirname = parse_seq_path(args.seq_path)
    if not img_paths:
        print(f"No images found in {args.seq_path}. Please verify the path.")
        return
    
    if args.max_frames is not None:
        img_paths = img_paths[:args.max_frames]
    img_paths = img_paths[::args.subsample]

    print(f"Found {len(img_paths)} images in {args.seq_path}.")
    img_mask = [True] * len(img_paths)

    # Load and prepare the model.
    print(f"Loading model from {args.model_path}...")
    model = ARCroco3DStereo.from_pretrained(args.model_path).to(device)
    model.eval()

    # Prepare input views.
    print("Preparing input views...")
    img_res = getattr(model, 'mhmr_img_res', None)
    views = prepare_input(
        img_paths=img_paths,
        img_mask=img_mask,
        size=args.size,
        revisit=1,
        update=True,
        img_res=img_res,
        reset_interval=args.reset_interval
    )

    if tmpdirname is not None:
        shutil.rmtree(tmpdirname)

    # Run inference.
    print("Running inference...")
    start_time = time.time()
    outputs, _ = inference_recurrent_lighter(
        views, model, device, use_ttt3r=args.use_ttt3r)
    total_time = time.time() - start_time
    per_frame_time = total_time / len(views)
    print(
        f"Inference completed in {total_time:.2f} seconds (average {per_frame_time:.2f} s per frame)."
    )

    # Process outputs for loss computation.
    print("Preparing output for loss computation.")
    (
        pts3ds_other, 
        colors, 
        conf, 
        cam_dict, 
        all_smpl_verts, 
        smpl_faces,
        smpl_id,
        msks,
        ) = prepare_output(
        outputs, args.output_dir, 1, True, 
        args.save_smpl, args.save_video, img_res, args.subsample
    )

    device = all_smpl_verts[0].device

    body_mean_ground, feet_mean_ground = compute_mean_dist_series_ground(
        all_smpl_verts,
        pts3ds_other,
        person_idx=0,          # 看第一个人
        use_pseudo_feet=True,  # 用几何方法自动找脚
        foot_contact_ids=None  # 这时不用传 segmentation 的脚
    )

    print("Body->Ground mean dist:", body_mean_ground)
    print("Feet->Ground mean dist:", feet_mean_ground)

        # 选一帧 t0，比如 0
    t0 = 0
    verts_t = all_smpl_verts[t0]      # [P,V,3]
    body = verts_t[0]                 # [V,3]

    pt = pts3ds_other[t0][0].reshape(-1, 3)  # [M,3]

    print("body y min / max / mean:", body[:,1].min().item(),
                                    body[:,1].max().item(),
                                    body[:,1].mean().item())

    print("scene y min / max / mean:", pt[:,1].min().item(),
                                    pt[:,1].max().item(),
                                    pt[:,1].mean().item())





    # 先只看第一个人（person_idx=0）
    # lc1 = compute_lc1(all_smpl_verts, pts3ds_other,contact_ids=foot_contact_ids, delta_c1=0.01,   person_idx=0)
    # lc2 = compute_lc2(all_smpl_verts,              contact_ids=foot_contact_ids, delta_c2=0.0005, person_idx=0)

    # print(f"Lc1(debug, person 0): {lc1:.6f}")
    # print(f"Lc2(debug, person 0): {lc2:.6f}")



def compute_lc1(all_smpl_verts, pts3ds_other, contact_ids,
                delta_c1=0.01, person_idx=0, eps_ground=0.01):
    """
    Lc1: foot contact to ground loss（脚底 vs 地面）
    
    all_smpl_verts: list[T], 每个元素 Tensor[P, V, 3]
    pts3ds_other :  list[T], 每个元素 Tensor[1, H, W, 3]
    contact_ids  : 1D LongTensor[K], 脚部顶点 index（在 V 维度上）
    delta_c1     : hinge 阈值 (作用在 squared distance 上)
    person_idx   : 取第几个人
    eps_ground   : 估计 ground_height 的偏移 (单位与坐标系一致, 一般 ~0.01)
    """
    lc1 = None  # 用 None 来判断有没有累加过
    T = len(all_smpl_verts)

    for t in range(T):
        verts_t = all_smpl_verts[t]      # [P, V, 3]
        pts_t   = pts3ds_other[t]        # [1, H, W, 3]

        # --- 1. 取某一个人 ---
        assert verts_t.ndim == 3, f"expect [P,V,3], got {verts_t.shape}"
        P, V, _ = verts_t.shape
        if person_idx >= P:
            # 这一帧没有这个人
            continue

        body_verts = verts_t[person_idx]       # [V, 3]

        # contact_ids 可能有越界，先过滤
        valid_ids = contact_ids[contact_ids < V]
        if valid_ids.numel() == 0:
            continue

        # 只用脚部点
        xh = body_verts[valid_ids]             # [K, 3]

        # --- 2. 展平场景点云 ---
        assert pts_t.ndim == 4 and pts_t.shape[0] == 1, f"expect [1,H,W,3], got {pts_t.shape}"
        pts_cloud = pts_t[0].reshape(-1, 3)    # [M, 3]

        if xh.numel() == 0 or pts_cloud.numel() == 0:
            continue

        # --- 2.1 从场景点云中选 ground points ---
        # 这里假设第 2 维 (y) 是竖直方向
        y = pts_cloud[:, 1]
        ground_height = y.min() + eps_ground          # 略高于最低点，防止极端噪声
        mask_ground = y < ground_height + 0.02        # 再给一点厚度 2cm
        ground_points = pts_cloud[mask_ground]        # [Mg, 3]

        if ground_points.numel() == 0:
            # 这一帧没法估计地面
            continue

        # --- 3. 最近邻距离（脚部点到地面点） ---
        dists = torch.cdist(xh, ground_points)        # [K, Mg]
        min_dists, nn_idx = dists.min(dim=1)          # [K]

        xs = ground_points[nn_idx]                    # [K, 3] ✅

        d2 = ((xh - xs) ** 2).sum(dim=-1)             # [K]
        hinge = torch.clamp(d2 - delta_c1, min=0.0)   # [K]
        hinge_sum = hinge.sum()

        # 打印一下每帧统计方便观察
        # 这里用 mean_dist ≈ sqrt( E[d^2] )
        mean_dist = d2.mean().sqrt().item()
        mean_hinge = hinge.mean().item()
        print(f"frame {t}: mean_dist={mean_dist:.6f}, mean_hinge={mean_hinge:.6e}, "
              f"ground_points={ground_points.shape[0]}")

        if lc1 is None:
            lc1 = hinge_sum
        else:
            lc1 = lc1 + hinge_sum

    if lc1 is None:
        return 0.0

    return lc1.item()


def compute_lc2(all_smpl_verts, contact_ids ,delta_c2=0.0005, person_idx=0):
    """
    Debug 版 Lc2：某一个人的所有顶点都算 contact。

    all_smpl_verts: list[T]，每个元素 Tensor[P, V, 3]
    """
    lc2 = 0.0
    T = len(all_smpl_verts)

    for t in range(T - 1):
        v0 = all_smpl_verts[t]      # [P, V, 3]
        v1 = all_smpl_verts[t + 1]  # [P, V, 3]

        assert v0.ndim == 3 and v1.ndim == 3, f"expect [P,V,3], got {v0.shape}, {v1.shape}"

        P0, V0, _ = v0.shape
        P1, V1, _ = v1.shape
        if person_idx >= P0 or person_idx >= P1:
            # 其中一帧没有这个人，跳过
            continue

        body0 = v0[person_idx]      # [V, 3]
        body1 = v1[person_idx]      # [V, 3]

        K = min(body0.shape[0], body1.shape[0])
        xh_t  = body0[contact_ids]       # [K,3]
        xh_t1 = body1[contact_ids]       # [K,3]


        d2 = ((xh_t - xh_t1) ** 2).sum(dim=-1)  # [K]
        hinge = torch.clamp(d2 - delta_c2, min=0.0)

        lc2 += hinge.sum()

        hinge = torch.clamp(d2 - delta_c2, min=0.0)

        print("frame pair", t, "mean_step =", d2.mean().sqrt().item(),
              "mean_hinge =", hinge.mean().item())

    return lc2.item()


def get_pseudo_feet_ids_from_frame(body_verts, k_per_foot=150, candidate_ratio=0.1):
    """
    根据几何信息，从单帧 body_verts 中自动选出左右脚底顶点 index。

    body_verts: Tensor[V,3]
    k_per_foot: 每只脚选多少个最低的顶点
    candidate_ratio: 先从最低的多少比例顶点中做左右脚划分
    """
    V = body_verts.shape[0]
    y = body_verts[:, 1]   # 竖直方向
    x = body_verts[:, 0]   # 左右方向

    # 1. 先选出 y 最低的一批候选，比如 10% 顶点
    num_candidates = max(200, int(candidate_ratio * V))
    _, cand_ids = torch.topk(-y, num_candidates)   # y 最小 → -y 最大
    cand_verts = body_verts[cand_ids]              # [Nc,3]

    # 2. 在候选中按 x 分左右脚（粗暴：按 x 的中位数分）
    x_cand = cand_verts[:, 0]
    x_mid = x_cand.median()
    left_mask  = x_cand < x_mid
    right_mask = ~left_mask

    left_ids_all  = cand_ids[left_mask]
    right_ids_all = cand_ids[right_mask]

    feet_ids = []

    # 3. 对左脚：再按 y 选出最低的 k_per_foot 个点
    if left_ids_all.numel() > 0:
        _, left_sel = torch.topk(-y[left_ids_all],
                                 k=min(k_per_foot, left_ids_all.numel()))
        feet_ids.append(left_ids_all[left_sel])

    # 4. 对右脚：同理
    if right_ids_all.numel() > 0:
        _, right_sel = torch.topk(-y[right_ids_all],
                                  k=min(k_per_foot, right_ids_all.numel()))
        feet_ids.append(right_ids_all[right_sel])

    if len(feet_ids) == 0:
        return None

    feet_ids = torch.cat(feet_ids, dim=0).unique()
    return feet_ids

def build_ground_height_map(
    pts_cloud,
    grid_size=0.05,          # 网格边长 5cm，可调
    y_percentile=5.0,        # 取每个格子内 y 的第 5 百分位，抗噪一点
    y_outlier_thresh=0.5     # 过滤掉非常离谱的点（比如比整体 min 高很多）
):
    """
    输入:
        pts_cloud: Tensor [M,3], (x,y,z)，假设 y 是竖直方向
    输出:
        ground_dict: 字典 {(ix, iz): y_ground}
        grid_size  : 网格大小（后面查地面时要用）
    """
    device = pts_cloud.device
    x = pts_cloud[:, 0]
    y = pts_cloud[:, 1]
    z = pts_cloud[:, 2]

    # 1) 大范围过滤极端异常值（比如漂在天上的点）
    #   用整体的低百分位 + 阈值简单裁一下
    y_global_min = torch.quantile(y, 0.01)
    mask = y < (y_global_min + y_outlier_thresh)
    x = x[mask]
    y = y[mask]
    z = z[mask]

    if x.numel() == 0:
        return {}, grid_size

    # 2) 把 (x,z) 映射到网格索引 (ix, iz)
    ix = torch.floor(x / grid_size).long()
    iz = torch.floor(z / grid_size).long()

    # 3) 根据 (ix,iz) 聚合 y 值
    #   用字典收集每个格子的 y 列表（如果数据量大可以优化成排序/segment）
    ground_dict = {}
    for ixi, izi, yi in zip(ix.tolist(), iz.tolist(), y.tolist()):
        key = (ixi, izi)
        if key not in ground_dict:
            ground_dict[key] = []
        ground_dict[key].append(yi)

    # 4) 对每个格子，取 y 的低百分位（比如第 5 百分位）当地面高度
    for key, ys in ground_dict.items():
        ys_tensor = torch.tensor(ys)
        y0 = torch.quantile(ys_tensor, y_percentile / 100.0).item()
        ground_dict[key] = y0

    return ground_dict, grid_size

def query_ground_height(vert, ground_dict, grid_size, default=None, k_search=1):
    """
    vert       : Tensor [3], (x,y,z)
    ground_dict: {(ix,iz): y_ground}
    grid_size  : 同上
    default    : 查不到时返回的默认地面高度（可以用全局最小 y 等）
    k_search   : 如果 exact cell 没有地面, 可以扩展到邻近 k_search 范围内找最近的格子
    """
    x, y, z = vert[0].item(), vert[1].item(), vert[2].item()
    ix = int(np.floor(x / grid_size))
    iz = int(np.floor(z / grid_size))
    key = (ix, iz)

    if key in ground_dict:
        return ground_dict[key]

    # 如果这一格没有点（比如场景稀疏），可以搜一下周围邻居格子
    if k_search > 0:
        best_y = None
        for dx in range(-k_search, k_search + 1):
            for dz in range(-k_search, k_search + 1):
                kk = (ix + dx, iz + dz)
                if kk in ground_dict:
                    yg = ground_dict[kk]
                    if best_y is None or yg < best_y:
                        best_y = yg
        if best_y is not None:
            return best_y

    return default

def compute_mean_dist_series_ground(
    all_smpl_verts,
    pts3ds_other,
    person_idx=0,
    use_pseudo_feet=True,
    foot_contact_ids=None,
    grid_size=0.05,          # 地面高度图网格大小 (m)，默认 5cm
    y_percentile=5.0,        # 每个格子里取 y 的第 5 百分位作为地面高度
    y_outlier_thresh=0.5,    # 过滤离整体地面太高的 outlier 点
    k_search=1,              # 查不到格子时的邻居范围
    pseudo_k_per_foot=150,   # 几何找脚：每只脚选多少个点
    pseudo_candidate_ratio=0.1 # 几何找脚：候选最低点比例
):
    """
    计算每一帧：
      - 全身顶点 到 “局部地面高度” 的平均距离
      - 脚部顶点 到 “局部地面高度” 的平均距离

    地面用 height-map 表示，支持楼梯 / 斜坡。

    参数:
        all_smpl_verts: list[T]，每个元素是 Tensor[P, V, 3]
        pts3ds_other  : list[T]，每个元素是 Tensor[1, H, W, 3]
        person_idx    : 选第几个人
        use_pseudo_feet: True 时用几何方法自动找脚；False 时用 foot_contact_ids
        foot_contact_ids: 1D LongTensor[K]，如果不用 pseudo feet 就用这份
        grid_size     : 地面高度图的网格边长 (米)
        y_percentile  : 每格取第多少百分位的 y 作为地面高度
        y_outlier_thresh: 过滤极端高点的阈值
        k_search      : 查某顶点脚下格子时，如该格无数据，就在邻居 [-k,+k] 搜
        pseudo_k_per_foot: 几何脚提取参数
        pseudo_candidate_ratio: 几何脚提取参数

    返回:
        mean_all  : list[T]，每帧全身平均“离地”距离
        mean_feet : list[T]，每帧脚部平均“离地”距离
    """

    assert len(all_smpl_verts) == len(pts3ds_other), \
        "all_smpl_verts 和 pts3ds_other 长度必须一致"

    device = all_smpl_verts[0].device
    T = len(all_smpl_verts)

    # ========== 辅助函数 1：构建 per-frame 地面高度图 ==========
    def build_ground_height_map(pts_cloud):
        """
        pts_cloud: Tensor[M,3] (x,y,z)
        返回: (ground_dict, grid_size)
            ground_dict[(ix,iz)] = y_ground
        """
        if pts_cloud.numel() == 0:
            return {}, grid_size

        x = pts_cloud[:, 0]
        y = pts_cloud[:, 1]
        z = pts_cloud[:, 2]

        # 1) 全局过滤极端高点：整体 y 的 1% 分位 + 阈值
        y_global_min = torch.quantile(y, 0.01)
        mask = y < (y_global_min + y_outlier_thresh)
        x = x[mask]
        y = y[mask]
        z = z[mask]

        if x.numel() == 0:
            return {}, grid_size

        # 2) 映射到 (ix,iz) 网格
        ix = torch.floor(x / grid_size).long().cpu().numpy()
        iz = torch.floor(z / grid_size).long().cpu().numpy()
        y_np = y.cpu().numpy()

        # 3) 按格子聚合 y
        tmp_dict = {}
        for ixi, izi, yi in zip(ix, iz, y_np):
            key = (int(ixi), int(izi))
            if key not in tmp_dict:
                tmp_dict[key] = []
            tmp_dict[key].append(float(yi))

        # 4) 每格取第 y_percentile 百分位
        ground_dict = {}
        q = y_percentile / 100.0
        for key, ys in tmp_dict.items():
            ys_arr = torch.tensor(ys)
            y0 = torch.quantile(ys_arr, q).item()
            ground_dict[key] = y0

        return ground_dict, grid_size

    # ========== 辅助函数 2：批量查询一堆顶点的“局部地面高度” ==========
    def query_ground_heights_batch(verts, ground_dict, grid_size, k_search=1):
        """
        verts      : Tensor[N,3]
        ground_dict: {(ix,iz): y_ground}
        返回:
            ground_heights: list[float or None]，长度为 N
        """
        if len(ground_dict) == 0 or verts.numel() == 0:
            return [None] * verts.shape[0]

        # 默认地面高度：整个 height-map 的最小 y_ground
        default_ground = min(ground_dict.values())

        x = verts[:, 0].cpu().numpy()
        z = verts[:, 2].cpu().numpy()

        ix_arr = np.floor(x / grid_size).astype(np.int64)
        iz_arr = np.floor(z / grid_size).astype(np.int64)

        ground_h_list = []
        for ix_i, iz_i in zip(ix_arr, iz_arr):
            key = (int(ix_i), int(iz_i))
            y_ground = None

            if key in ground_dict:
                y_ground = ground_dict[key]
            elif k_search > 0:
                # 在邻居 [-k,k]×[-k,k] 里找最近的地面格子
                best_y = None
                for dx in range(-k_search, k_search + 1):
                    for dz in range(-k_search, k_search + 1):
                        kk = (int(ix_i + dx), int(iz_i + dz))
                        if kk in ground_dict:
                            yg = ground_dict[kk]
                            if best_y is None or yg < best_y:
                                best_y = yg
                if best_y is not None:
                    y_ground = best_y

            if y_ground is None:
                # 实在找不到就用全局最小高度兜底
                y_ground = default_ground

            ground_h_list.append(float(y_ground))

        return ground_h_list

    # ========== 第一步：如果要用 pseudo feet，就先用第一帧找到脚 idx ==========
    feet_ids = None

    if use_pseudo_feet:
        for t in range(T):
            verts_t = all_smpl_verts[t]  # [P,V,3]
            if verts_t.ndim != 3:
                continue
            P, V, _ = verts_t.shape
            if person_idx >= P:
                continue
            body_verts = verts_t[person_idx]  # [V,3]
            if body_verts.numel() == 0:
                continue

            from math import ceil
            # 用你之前写好的几何脚提取函数
            pseudo_feet_ids = get_pseudo_feet_ids_from_frame(
                body_verts,
                k_per_foot=pseudo_k_per_foot,
                candidate_ratio=pseudo_candidate_ratio
            )
            if pseudo_feet_ids is not None and pseudo_feet_ids.numel() > 0:
                feet_ids = pseudo_feet_ids.long().to(device)
                print(f"[compute_mean_dist_series_ground] use pseudo feet, #verts = {feet_ids.numel()}")
                break

        if feet_ids is None:
            print("[compute_mean_dist_series_ground] Warning: pseudo feet not found, will only输出 body mean")
    else:
        if foot_contact_ids is not None:
            feet_ids = foot_contact_ids.long().to(device)
            print(f"[compute_mean_dist_series_ground] use given foot_contact_ids, #verts = {feet_ids.numel()}")
        else:
            print("[compute_mean_dist_series_ground] Warning: no foot_contact_ids provided, will only输出 body mean")

    # ========== 第二步：逐帧构建地面 + 计算 body / feet 距离 ==========
    mean_all = []
    mean_feet = []

    for t in range(T):
        verts_t = all_smpl_verts[t]   # [P,V,3]
        pts_t   = pts3ds_other[t]     # [1,H,W,3]

        if not (torch.is_tensor(verts_t) and torch.is_tensor(pts_t)):
            mean_all.append(float('nan'))
            mean_feet.append(float('nan'))
            continue

        assert verts_t.ndim == 3, f"expect [P,V,3], got {verts_t.shape}"
        assert pts_t.ndim == 4 and pts_t.shape[0] == 1, f"expect [1,H,W,3], got {pts_t.shape}"

        P, V, _ = verts_t.shape
        if person_idx >= P:
            mean_all.append(float('nan'))
            mean_feet.append(float('nan'))
            continue

        body_verts = verts_t[person_idx]        # [V,3]
        pts_cloud  = pts_t[0].reshape(-1, 3)    # [M,3]

        if body_verts.numel() == 0 or pts_cloud.numel() == 0:
            mean_all.append(float('nan'))
            mean_feet.append(float('nan'))
            continue

        # 2.1 构建该帧的 ground height map
        ground_dict, _ = build_ground_height_map(pts_cloud)

        if len(ground_dict) == 0:
            mean_all.append(float('nan'))
            mean_feet.append(float('nan'))
            continue

        # 2.2 body: 对所有顶点算 “y_body - y_ground(x,z)”
        body_ground_heights = query_ground_heights_batch(
            body_verts, ground_dict, grid_size, k_search=k_search
        )
        body_y = body_verts[:, 1].cpu().numpy()
        d_all = []

        for yb, yg in zip(body_y, body_ground_heights):
            d_all.append(float(yb - yg))

        mean_all.append(np.mean(d_all) if len(d_all) > 0 else float('nan'))

        # 2.3 feet: 用 feet_ids 子集算同样的东西
        if feet_ids is not None and feet_ids.numel() > 0:
            valid_ids = feet_ids[feet_ids < V]
            if valid_ids.numel() > 0:
                feet_verts = body_verts[valid_ids]  # [K,3]
                feet_ground_heights = query_ground_heights_batch(
                    feet_verts, ground_dict, grid_size, k_search=k_search
                )
                feet_y = feet_verts[:, 1].cpu().numpy()
                d_feet = []
                for yf, yg in zip(feet_y, feet_ground_heights):
                    d_feet.append(float(yf - yg))
                mean_feet.append(np.mean(d_feet) if len(d_feet) > 0 else float('nan'))
            else:
                mean_feet.append(float('nan'))
        else:
            mean_feet.append(float('nan'))

    return mean_all, mean_feet

def main():
    args = parse_args()
    if not args.seq_path:
        print(
            "No inputs found! Please use our gradio demo if you would like to iteractively upload inputs."
        )
        return
    else:
        run_inference(args)


if __name__ == "__main__":
    main()
