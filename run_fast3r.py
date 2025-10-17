import torch
from fast3r.dust3r.utils.image import load_images
from fast3r.dust3r.inference_multiview import inference
from fast3r.models.fast3r import Fast3R
from fast3r.models.multiview_dust3r_module import MultiViewDUSt3RLitModule
import argparse
import numpy as np
from matplotlib import cm
from fast3r.dust3r.utils.device import to_numpy
from fast3r.viz.viser_visualizer import detect_sky_mask, safe_color_conversion, generate_ply_bytes
import os
import shutil
import cv2

def extract_frames_from_video(video_path, frame_interval_sec=1, tmp_dir="./tmp_imgs"):
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir)

    vs = cv2.VideoCapture(video_path)
    fps = vs.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * frame_interval_sec)
    image_paths = []

    count = 0
    frame_idx = 0
    while True:
        gotit, frame = vs.read()
        if not gotit:
            break
        count += 1
        if count % frame_interval == 0:
            image_path = os.path.join(tmp_dir, f"{frame_idx:06d}.png")
            cv2.imwrite(image_path, frame)
            image_paths.append(image_path)
            frame_idx += 1

    vs.release()
    return sorted(image_paths)

def gather_image_paths(input_path, tmp_dir="./tmp"):
    """
    支持:
    - 单图片
    - 多图片（逗号分隔）
    - 文件夹
    - 视频文件
    所有图片都会被复制/保存到 tmp_dir/images 下。
    """
    exts_img = [".png", ".jpg", ".jpeg"]
    exts_vid = [".mp4", ".mov", ".avi", ".mkv"]
    tmp_dir = os.path.join(tmp_dir, "images")
    # 清空临时目录
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir, exist_ok=True)

    # 多图情况（逗号分隔）
    if "," in input_path:
        paths = [p.strip() for p in input_path.split(",") if os.path.exists(p.strip())]
        image_paths = []
        for i, src_path in enumerate(paths):
            ext = os.path.splitext(src_path)[1]
            dst_path = os.path.join(tmp_dir, f"{i:06d}{ext}")
            shutil.copy2(src_path, dst_path)
            image_paths.append(dst_path)
        return sorted(image_paths)

    # 视频文件
    if any(input_path.lower().endswith(ext) for ext in exts_vid):
        print(f"[INFO] Detected video input: {input_path}")
        return extract_frames_from_video(input_path, tmp_dir=tmp_dir)

    # 文件夹
    if os.path.isdir(input_path):
        imgs = [
            os.path.join(input_path, f)
            for f in sorted(os.listdir(input_path))
            if any(f.lower().endswith(ext) for ext in exts_img)
        ]
        if not imgs:
            raise ValueError(f"No valid image found in directory: {input_path}")

        image_paths = []
        for i, src_path in enumerate(imgs):
            ext = os.path.splitext(src_path)[1]
            dst_path = os.path.join(tmp_dir, f"{i:06d}{ext}")
            shutil.copy2(src_path, dst_path)
            image_paths.append(dst_path)
        return sorted(image_paths)

    # 单图片
    if os.path.isfile(input_path) and any(input_path.lower().endswith(ext) for ext in exts_img):
        ext = os.path.splitext(input_path)[1]
        dst_path = os.path.join(tmp_dir, f"000000{ext}")
        shutil.copy2(input_path, dst_path)
        return [dst_path]

    raise ValueError(f"Invalid input path: {input_path}")

# --- Setup ---
# Load the model from Hugging Face
def run_fast3r(args):
    min_conf_thr_percentile = args.conf_thres  # Percentile threshold for filtering points based on confidence
    mask_sky = False
    model = Fast3R.from_pretrained(args.model_path)  # If you have networking issues, try pre-download the HF checkpoint dir and change the path here to a local directory
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Create a lightweight lightning module wrapper for the model.
    # This provides functions to estimate camera poses, evaluate 3D reconstruction, etc.
    lit_module = MultiViewDUSt3RLitModule.load_for_inference(model)

    # Set model to evaluation mode
    model.eval()
    lit_module.eval()

    # --- Load Images ---
    # Provide a list of image file paths. Images can come from different cameras and aspect ratios.
    image_paths = gather_image_paths(args.input_path)
    images = load_images(image_paths, size=512, verbose=True)

    # --- Run Inference ---
    # The inference function returns a dictionary with predictions and view information.
    output_dict, profiling_info = inference(
        images,
        model,
        device,
        dtype=torch.float32,  # or use torch.bfloat16 if supported
        verbose=True,
        profiling=True,
    )

    # --- Estimate Camera Poses ---
    # This step estimates the camera-to-world (c2w) poses for each view using PnP.
    poses_c2w_batch, estimated_focals = MultiViewDUSt3RLitModule.estimate_camera_poses(
        output_dict['preds'],
        niter_PnP=100,
        focal_length_estimation_method='first_view_from_global_head'
    )
    try:
        for pred in output_dict['preds']:
            for k, v in pred.items():
                if isinstance(v, torch.Tensor):
                    pred[k] = v.cpu()
        for view in output_dict['views']:
            for k, v in view.items():
                if isinstance(v, torch.Tensor):
                    view[k] = v.cpu()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"Warning: {e}")
        
    f = estimated_focals[0][0]
    H, W = output_dict['views'][0]['img'].shape[-2:]
    cx, cy = W / 2, H / 2
    intrinsic = np.array([[f, 0, cx],
                [0, f, cy],
                [0, 0, 1]])
    # poses_c2w_batch is a list; the first element contains the estimated poses for each view.
    camera_poses = poses_c2w_batch[0]

    lit_module.align_local_pts3d_to_global(
        preds=output_dict['preds'],
        views=output_dict['views'],
        min_conf_thr_percentile=min_conf_thr_percentile
    )

    num_frames = len(output_dict['preds'])
    all_points, all_colors = [], []

    for i in range(num_frames):
        pred = output_dict['preds'][i]
        view = output_dict['views'][i]

        # 图像 RGB [-1,1] 转换为 numpy
        img_rgb = to_numpy(view['img'].cpu().squeeze().permute(1, 2, 0))
        not_sky_mask = detect_sky_mask(img_rgb).flatten().astype(np.int8) if mask_sky else np.ones(img_rgb.size // 3, np.int8)

        # 取点云与置信度
        pts3d = to_numpy(pred['pts3d_local_aligned_to_global'].cpu().squeeze()).reshape(-1, 3)
        conf = to_numpy(pred['conf_local'].cpu().squeeze()).flatten()
        rgb_flat = img_rgb.reshape(-1, 3)

        # 置信度排序并滤除低置信度
        sort_idx = np.argsort(-conf)
        sorted_pts = pts3d[sort_idx]
        sorted_rgb = rgb_flat[sort_idx]
        sorted_conf = conf[sort_idx]
        sorted_not_sky = not_sky_mask[sort_idx]

        # 基于百分位选取高置信度点
        keep_n = int(len(sorted_conf) * (100 - min_conf_thr_percentile) / 100)
        keep_n = max(1, keep_n)
        sorted_pts = sorted_pts[:keep_n]
        sorted_rgb = sorted_rgb[:keep_n]
        sorted_not_sky = sorted_not_sky[:keep_n]

        # 如果启用 mask_sky，过滤天空点
        if mask_sky:
            valid_mask = sorted_not_sky > 0
            sorted_pts = sorted_pts[valid_mask]
            sorted_rgb = sorted_rgb[valid_mask]

        # 颜色归一化 [-1,1] → [0,255]
        colors_uint8 = safe_color_conversion(sorted_rgb)

        all_points.append(sorted_pts)
        all_colors.append(colors_uint8)

    # 合并所有帧
    all_points = np.concatenate(all_points, axis=0)
    all_colors = np.concatenate(all_colors, axis=0)

    predictions = {}
    predictions["images"] = np.array([to_numpy(view['img'].cpu().squeeze().permute(1, 2, 0)) for view in output_dict['views']])
    predictions["extrinsic"] = np.array(camera_poses)[:, :3, :] # (N, 3, 4)
    predictions["intrinsic"] = intrinsic # (3, 3)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    prediction_save_path = os.path.join("./tmp", "predictions_fast3r.npz")

    np.savez(prediction_save_path, **predictions)
    # 生成 PLY 文件
    ply_bytes = generate_ply_bytes(all_points, all_colors)
    with open(args.output_path, "wb") as f:
        f.write(ply_bytes)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True,
                        help="Path to input (video / image / multi-image comma-separated / folder)")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to save output file")
    parser.add_argument("--model_path", type=str, default="/data2/zxy/workspace/models/Fast3R_ViT_Large_512",
                        help="Path to pretrained Fast3R model")
    parser.add_argument("--conf_thres", type=float, default=30.0, help="Confidence threshold for point cloud filtering")
    parser.add_argument("--device", type=str, default="cuda:0",  help="Compute device (cuda or cpu)")
    args = parser.parse_args()

    run_fast3r(args)