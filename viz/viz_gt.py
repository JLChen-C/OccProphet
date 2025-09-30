# Developed by Junyi Ma
# Cam4DOcc: Benchmark for Camera-Only 4D Occupancy Forecasting in Autonomous Driving Applications
# https://github.com/haomo-ai/Cam4DOcc

from tqdm import tqdm
import pickle
import numpy as np
from mayavi import mlab
from tqdm import trange
import os
from xvfbwrapper import Xvfb
from torch.utils.data import Dataset, DataLoader
from nuscenes import NuScenes
import shutil
import argparse
import psutil

# export QT_QPA_PLATFORM='offscreen' 
mlab.options.offscreen = True

def mkdir_or_exists(target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)
    return target_dir

def parse_args():
    parser = argparse.ArgumentParser(description='Visualization')
    parser.add_argument('--pred_dir', default=None, help='path of the directory of prediction files')
    parser.add_argument('--save_dir', default=None, help='path of the directory of visualization files')
    parser.add_argument('--n_past_frames', type=int, default=3, help='number of past frames to show')
    parser.add_argument('--n_future_frames', type=int, default=4, help='number of future frames to show')
    parser.add_argument('--start_frame', type=int, default=2, help='start frame of images')
    parser.add_argument('--end_frame', type=int, default=23, help='end frame of images')
    parser.add_argument('--use_lyft', action="store_true", help='whether using lyft-level5 dataset')
    parser.add_argument('--show_by_frame', action="store_true", help='whether showing frame by frame')
    parser.add_argument('--not_show_time_change', action="store_true", help='whether not showing time changes')
    args = parser.parse_args()
    return args

def viz_occ(occ, occ_mo, save_path, voxel_size, show_occ, show_time_change):

    vdisplay = Xvfb(width=1, height=1)
    vdisplay.start()

    mlab.figure(size=(800,800), bgcolor=(1,1,1))

    plt_plot_occ = mlab.points3d(
        occ[:, 0] * voxel_size,
        occ[:, 1] * voxel_size,
        occ[:, 2] * voxel_size,
        occ[:, 3],
        colormap="viridis",
        scale_factor=voxel_size - 0.05 * voxel_size,
        mode="cube",
        opacity=0.9,
        vmin=1,
    )
    colors_occ = np.array(
        [
            [152, 251, 152, 255],
            [152, 251, 152, 255],
            [152, 251, 152, 255],
            [152, 251, 152, 255],
            [152, 251, 152, 255],
        ]
    ).astype(np.uint8)    
    plt_plot_occ.glyph.scale_mode = "scale_by_vector"
    plt_plot_occ.module_manager.scalar_lut_manager.lut.table = colors_occ

    plt_plot_mov = mlab.points3d(
        occ_mo[:, 0] * voxel_size,
        occ_mo[:, 1] * voxel_size,
        occ_mo[:, 2] * voxel_size,
        occ_mo[:, 3],
        colormap="viridis",
        scale_factor=voxel_size - 0.05 * voxel_size,
        mode="cube",
        opacity=0.9,
        vmin=1,
    )
    if show_time_change:
        colors_occ_mo = np.array(
            [
                [255, 70, 255, 255],
                [255, 110, 255, 255],
                [255, 150, 255, 255],
                [255, 190, 255, 255],
                [255, 250, 250, 255],
            ]
        ).astype(np.uint8)
    else:
        # colors_occ_mo = np.array(
        #     [
        #         [220, 20, 60, 255],
        #         [255, 127, 80, 255],
        #         [0, 0, 230, 255],
        #         [255, 158, 0, 255],
        #         [233, 150, 70, 255],
        #         [47, 79, 79, 255],
        #         [255, 99, 71, 255],
        #         [175, 0, 75, 255],
        #         [255, 61, 99, 255],
        #     ]
        # ).astype(np.uint8)

        colors_occ_mo = np.array(
            [
                [255, 70, 255, 255],
                [255, 127, 80, 255],
                [0, 0, 230, 255],
                [255, 158, 0, 255],
                [233, 150, 70, 255],
                [47, 79, 79, 255],
                [255, 99, 71, 255],
                [175, 0, 75, 255],
                [255, 61, 99, 255],
            ]
        ).astype(np.uint8)
    plt_plot_mov.glyph.scale_mode = "scale_by_vector"
    plt_plot_mov.module_manager.scalar_lut_manager.lut.table = colors_occ_mo

    mlab.savefig(save_path)
    vdisplay.stop()


def parallel_exec(func, num_workers, num_samples, batch_size=1):

    class FunctionalDataset(Dataset):
        
        def __init__(self, func):
            self.func = func
            self.num_samples = num_samples

        def __getitem__(self, index):
            return self.func(index)
        
        def __len__(self):
            return self.num_samples

    dataset = FunctionalDataset(func)
    dataloader = DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=False, collate_fn=lambda x: x)

    results = []
    for batch in tqdm(dataloader):
        results += batch
    return results


def main():

    default_n_future_frames = 4

    args = parse_args()

    # show_time_change = True
    show_time_change = not args.not_show_time_change

    # nuscocc_path = "../data/nuScenes-Occupancy/"
    # cam4docc_path = "../data/cam4docc/GMO/segmentation/"

    nuscocc_path = "../data/nuscenes/nuScenes-Occupancy/"
    cam4docc_path = "../data/cam4docc/GMO/segmentation/"
    if args.pred_dir is None:
        pred_dir = "../work_dirs/pretrained/OCFNet_in_Cam4DOcc_V1.1/results/"
    else:
        pred_dir = arg.pred_dir

    if args.save_dir is None:
        if args.show_by_frame:
            save_dir = mkdir_or_exists('./GMO/GT_by_frame_corrected')
        else:
            save_dir = mkdir_or_exists('./GMO/GT_corrected')
        if args.n_future_frames != default_n_future_frames:
            save_dir += mkdir_or_exists(f'_{args.n_future_frames}f')
            cam4docc_path = cam4docc_path.replace('cam4docc', f'cam4docclt_{args.n_past_frames}f-{args.n_future_frames}f')
        image_save_dir = mkdir_or_exists('./images')
        image_seq_save_dir = mkdir_or_exists('./image_seqs')
    else:
        save_root = os.path.abspath(args.save_dir)
        if args.show_by_frame:
            save_dir = mkdir_or_exists(os.path.join(save_root, 'GMO', 'GT_by_frame'))
        else:
            save_dir = mkdir_or_exists(os.path.join(save_root, 'GMO', 'GT'))
        if args.n_future_frames != default_n_future_frames:
            save_dir += mkdir_or_exists(f'_{args.n_past_frames}f-{args.n_future_frames}f')
            cam4docc_path = cam4docc_path.replace('cam4docc', f'cam4docclt_{args.n_past_frames}f-{args.n_future_frames}f')
        image_save_dir = mkdir_or_exists(os.path.join(save_root, 'images'))
        image_seq_save_dir = mkdir_or_exists(os.path.join(save_root, 'image_seqs'))
    print(f'save_dir: {save_dir}')
    print(f'image_save_dir: {image_save_dir}')
    print(f'image_seq_save_dir: {image_seq_save_dir}')

    nusc = NuScenes(version='v1.0-trainval', dataroot="../data/nuscenes", verbose=False)

    # segmentation_files = os.listdir(cam4docc_path)

    segmentation_files = os.listdir(pred_dir)
    segmentation_files.sort(key=lambda x: (x.split("_")[1]))
    # index = 0

    # segmentation_files = segmentation_files[::10]
    # segmentation_files = segmentation_files[::5]
    # # for occlusion
    # segmentation_files = ["2ca15f59d656489a8b1a0be4d9bead4e_f7246b6b5cce46f2bfa4290d5b9d713d.npz"]
    # for occlusion
    # segmentation_files = ["2ca15f59d656489a8b1a0be4d9bead4e_08fe0286b676413e9ce8ffb746a9ea90.npz"]
    for file_ in tqdm(segmentation_files):
    # def job(index):
    #     file_ = segmentation_files[index]

        scene_token = file_.split("_")[0]
        lidar_token = file_.split("_")[1]

        ##
        cam_save_dir = mkdir_or_exists(os.path.join(image_save_dir, file_[:-4]))

        cam_seq_save_dir = mkdir_or_exists(os.path.join(image_seq_save_dir, file_[:-4]))
        ##
        sample_tokens = nusc.field2token('sample', 'scene_token', scene_token)

        # save single-frame camera images
        # for sample_token in sample_tokens:
        sample_token = sample_tokens[args.start_frame]
        sample = nusc.get('sample', sample_token)
        for cam_view in ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']:
        # if sample['data']['CAM_FRONT'] != '':
        #     cam_sample_token = sample['data']['CAM_FRONT']
            cam_sample_token = sample['data'][cam_view]
            # cam_sample = nusc.get('sample_data', cam_sample_token)

            image_path = nusc.get_sample_data_path(cam_sample_token)
            # image_save_path = os.path.join(cam_save_dir, image_path)
            image_save_path = os.path.join(cam_save_dir, cam_view + '.jpg')
            # print(f'cam_sample_token: {cam_sample_token}, image_path: {image_path}, image_save_path: {image_save_path}')
            if (not os.path.isfile(image_save_path)) or (not os.access(image_save_path, os.R_OK)):
                shutil.copy(image_path, image_save_path)

        # save sequence of camera images
        # for sample_token in sample_tokens:
        for fi in range(args.end_frame):
            sample_token = sample_tokens[fi]
            sample = nusc.get('sample', sample_token)

            cam_fi_save_dir = mkdir_or_exists(os.path.join(cam_seq_save_dir, str(fi)))
            for cam_view in ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']:
                cam_sample_token = sample['data'][cam_view]
                image_path = nusc.get_sample_data_path(cam_sample_token)
                image_save_path = os.path.join(cam_fi_save_dir, cam_view + '.jpg')
                if (not os.path.isfile(image_save_path)) or (not os.access(image_save_path, os.R_OK)):
                    shutil.copy(image_path, image_save_path)

        gt_file = nuscocc_path+"scene_"+scene_token+"/occupancy/"+lidar_token[:-4]+".npy"
        gt_occ_semantic =  np.load(gt_file,allow_pickle=True)
        gt_occ_semantic = gt_occ_semantic[gt_occ_semantic[:, -1]!=0]
        gt_occ_semantic = gt_occ_semantic[::2]
        gt_occ_semantic_refine = np.zeros_like(gt_occ_semantic)
        gt_occ_semantic_refine[:, 0] = gt_occ_semantic[:, 2]
        gt_occ_semantic_refine[:, 1] = gt_occ_semantic[:, 1]
        gt_occ_semantic_refine[:, 2] = gt_occ_semantic[:, 0]
        gt_occ_semantic_refine[:, 3] = 1

        gt_mo_semantic =  np.load(cam4docc_path+file_,allow_pickle=True)['arr_0']

        if args.show_by_frame:
            for t in range(0, args.n_past_frames + args.n_future_frames):
                gt_mo_cur = gt_mo_semantic[t]
                gt_mo_cur = np.array(gt_mo_cur)
                # gt_mo_cur = gt_mo_cur[::2]
                # if show_time_change:
                #     gt_mo_cur[:, -1] = int(t + 1)
                # gt_mo_semantic_to_draw = np.concatenate((gt_mo_semantic_to_draw, gt_mo_cur))
                gt_mo_semantic_to_draw = gt_mo_cur

                save_dir_ = os.path.join(save_dir, file_[:-4])
                if not os.path.exists(save_dir_):
                    os.makedirs(save_dir_, exist_ok=True)
                save_path = os.path.join(save_dir_, f"{t}.png")
                if (not os.path.isfile(save_path)) or (not os.access(save_path, os.R_OK)):
                    viz_occ(gt_occ_semantic_refine, gt_mo_semantic_to_draw, save_path, voxel_size=0.2, show_occ=True, show_time_change=False)
        else:
            gt_mo_semantic_to_draw = np.zeros((0, args.n_futre_frames))
            for t in range(0, args.n_future_frames):
                # wrong
                # gt_mo_cur = gt_mo_semantic[t]
                gt_mo_cur = gt_mo_semantic[t + args.n_past_frames]
                gt_mo_cur = np.array(gt_mo_cur)
                gt_mo_cur = gt_mo_cur[::2]
                if show_time_change:
                    gt_mo_cur[:, -1] = int(t + 1)
                gt_mo_semantic_to_draw = np.concatenate((gt_mo_semantic_to_draw, gt_mo_cur))

            save_path = os.path.join(save_dir, file_[:-4] + ".png")
            if (not os.path.isfile(save_path)) or (not os.access(save_path, os.R_OK)):
                viz_occ(gt_occ_semantic_refine, gt_mo_semantic_to_draw, save_path, voxel_size=0.2, show_occ=True, show_time_change=show_time_change)

        # index += 1

        memory_usage = psutil.virtual_memory()
        # percentage of the used memory
        if memory_usage.percent > 90:
            break

    # N_JOBS = 1
    # num_samples = len(segmentation_files)
    # parallel_exec(job, num_workers=N_JOBS, num_samples=num_samples, batch_size=1)

if __name__ == "__main__":
    main()
