from tqdm import tqdm
import pickle
import numpy as np
from mayavi import mlab
from tqdm import trange
import os
from xvfbwrapper import Xvfb
from torch.utils.data import Dataset, DataLoader
import argparse
import psutil

mlab.options.offscreen = True

def parse_args():
    parser = argparse.ArgumentParser(description='Visualization')
    parser.add_argument('--pred_dir', help='path of the directory of prediction files')
    parser.add_argument('--save_dir', help='path of the directory of visualization files')
    parser.add_argument('--n_frames', type=int, default=4, help='number of future frames to show')
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

    args = parse_args()

    # show_time_change = True
    show_time_change = not args.not_show_time_change

    # nuscocc_path = "../data/nuScenes-Occupancy/"
    # pred_dir = "../data/cam4docc/results/"
    nuscocc_path = "../data/nuscenes/nuScenes-Occupancy/"
    # lyftocc_path = "../data/cam4docc/GMO_lyft_val/segmentation"
    # main result
    # pred_dir = "../work_dirs/fastocc4d/unet/FastOcc4D_V1.1_r34-dconv-gcb_448x800_unet_dl3_inprojks3_w7_hks3_condgen_normact_4gpus/results/"
    # Cam4DOcc
    # pred_dir = "../work_dirs/pretrained/OCFNet_in_Cam4DOcc_V1.1/results/"
    # ablation studies
    # pred_dir = "../work_dirs/fastocc4d/unet/ablation/FastOcc4D_V1.1_r18_448x800_unet_dl3_inprojks3_w7_hks3_condgen_normact_no_preunet_condpred_postunet_4gpus/results/"
    pred_dir = args.pred_dir
    
    if args.show_by_frame:
        save_dir = pred_dir.replace('results', 'results_by_frame_png')
    else:
        save_dir = pred_dir.replace('results', 'results_png')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    print(f'pred_dir: {pred_dir}')
    print(f'save_dir: {save_dir}')

    segmentation_files = os.listdir(pred_dir)
    segmentation_files.sort(key=lambda x: (x.split("_")[1]))
    # index = 0

    # segmentation_files = segmentation_files[::10]
    segmentation_files = segmentation_files[::5]
    # segmentation_files = ["2ca15f59d656489a8b1a0be4d9bead4e_2d0c1cca52b74244a6cde586d6818d0f.npz",
        # "2f56eb47c64f43df8902d9f88aa8a019_468c21f5bd5140959918a522a0bde997.npz"]
    # # for occlusion
    # segmentation_files = ["2ca15f59d656489a8b1a0be4d9bead4e_f7246b6b5cce46f2bfa4290d5b9d713d.npz"]
    # for occlusion
    # segmentation_files = ["2ca15f59d656489a8b1a0be4d9bead4e_08fe0286b676413e9ce8ffb746a9ea90.npz"]
    for file_ in tqdm(segmentation_files):
    # def job(index):
    #     file_ = segmentation_files[index]

        scene_token = file_.split("_")[0]
        lidar_token = file_.split("_")[1]
        # gt_file = nuscocc_path+"scene_"+scene_token+"/occupancy/"+lidar_token[:-4]+".npy"
        # gt_occ_semantic =  np.load(gt_file,allow_pickle=True)
        # gt_occ_semantic = gt_occ_semantic[gt_occ_semantic[:, -1]!=0]
        # gt_occ_semantic = gt_occ_semantic[::2]
        # gt_occ_semantic_refine = np.zeros_like(gt_occ_semantic)
        # gt_occ_semantic_refine[:, 0] = gt_occ_semantic[:, 2]
        # gt_occ_semantic_refine[:, 1] = gt_occ_semantic[:, 1]
        # gt_occ_semantic_refine[:, 2] = gt_occ_semantic[:, 0]
        # gt_occ_semantic_refine[:, 3] = 1
        if args.use_lyft:
            gt_file = lyftocc_path + scene_token + "_" + lidar_token + ".npz"
            gt_occ_semantic =  np.load(gt_file, allow_pickle=True)['arr_0']
        else:
            gt_file = nuscocc_path + "scene_" + scene_token + "/occupancy/" + lidar_token[:-4] + ".npy"
            gt_occ_semantic =  np.load(gt_file, allow_pickle=True)
        gt_occ_semantic = gt_occ_semantic[gt_occ_semantic[:, -1]!=0]
        gt_occ_semantic = gt_occ_semantic[::2]
        gt_occ_semantic_refine = np.zeros_like(gt_occ_semantic)
        gt_occ_semantic_refine[:, 0] = gt_occ_semantic[:, 2]
        gt_occ_semantic_refine[:, 1] = gt_occ_semantic[:, 1]
        gt_occ_semantic_refine[:, 2] = gt_occ_semantic[:, 0]
        gt_occ_semantic_refine[:, 3] = 1

        pred_mo_semantic =  np.load(pred_dir+file_,allow_pickle=True)['arr_0']

        if args.show_by_frame:
            for t in range(0, args.n_frames):
                pred_mo_cur = pred_mo_semantic[t]
                pred_mo_cur = np.array(pred_mo_cur)
                # pred_mo_cur = pred_mo_cur[::2]
                # if show_time_change:
                #     pred_mo_cur[:, -1] = int(t + 1)
                # pred_mo_semantic_to_draw = np.concatenate((pred_mo_semantic_to_draw, pred_mo_cur))
                pred_mo_semantic_to_draw = pred_mo_cur

                save_dir_ = os.path.join(save_dir, file_[:-4])
                if not os.path.exists(save_dir_):
                    os.makedirs(save_dir_, exist_ok=True)
                # save_path = os.path.join(save_dir_, f"{t+1}.png")
                save_path = os.path.join(save_dir_, f"{t}.png")
                if (not os.path.isfile(save_path)) or (not os.access(save_path, os.R_OK)):
                    viz_occ(gt_occ_semantic_refine, pred_mo_semantic_to_draw, save_path, voxel_size=0.2, show_occ=True, show_time_change=False)
        else:
            # pred_mo_semantic_to_draw=np.zeros((0,4))
            # for t in range(0,4):
            pred_mo_semantic_to_draw = np.zeros((0, args.n_frames))
            for t in range(0, args.n_frames):
                pred_mo_cur = pred_mo_semantic[t]
                pred_mo_cur = np.array(pred_mo_cur)
                pred_mo_cur = pred_mo_cur[::2]
                if show_time_change:
                    pred_mo_cur[:, -1] = int(t + 1)
                pred_mo_semantic_to_draw = np.concatenate((pred_mo_semantic_to_draw, pred_mo_cur))

            save_path = os.path.join(save_dir, file_[:-4] + ".png")
            if (not os.path.isfile(save_path)) or (not os.access(save_path, os.R_OK)):
                viz_occ(gt_occ_semantic_refine, pred_mo_semantic_to_draw, save_path, voxel_size=0.2, show_occ=True, show_time_change=show_time_change)

        # index += 1

        memory_usage = psutil.virtual_memory()
        # percentage of the used memory
        if memory_usage.percent > 90:
            break

    # N_JOBS = 2
    # num_samples = len(segmentation_files)
    # parallel_exec(job, num_workers=N_JOBS, num_samples=num_samples, batch_size=1)

if __name__ == "__main__":
    main()
    # export QT_QPA_PLATFORM='offscreen' 