import argparse
import numpy as np
import torch

from models.cnn import CNN, CNN_Pose, CNN_Ego, CNN_Ego_Pose
from logging import getLogger

logger = getLogger('main')


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--in_data', type=str)
    parser.add_argument('--nb_train', type=int, default=-1)
    parser.add_argument('--nb_jobs', type=int, default=8)
    parser.add_argument('--nb_splits', type=int, default=5)
    parser.add_argument('--eval_split', type=int, default=0)

    parser.add_argument('--model', type=str, default="cnn")
    parser.add_argument('--input_len', type=int, default=10)
    parser.add_argument('--offset_len', type=int, default=10)
    parser.add_argument('--pred_len', type=int, default=10)
    parser.add_argument('--inter_list', type=int, nargs="*", default=[])
    parser.add_argument('--last_list', type=int, nargs="*", default=[])
    parser.add_argument('--channel_list', type=int, nargs="*", default=[])
    parser.add_argument('--deconv_list', type=int, nargs="*", default=[])
    parser.add_argument('--ksize_list', type=int, nargs="*", default=[])
    parser.add_argument('--dc_ksize_list', type=int, nargs="*", default=[])
    parser.add_argument('--pad_list', type=int, nargs="*", default=[])

    parser.add_argument('--nb_iters', type=int, default=10000)
    parser.add_argument('--iter_snapshot', type=int, default=1000)
    parser.add_argument('--iter_display', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--optimizer', type=str, default="adam")
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--lr_step_list', type=float, nargs="*", default=[])
    parser.add_argument('--momentum', type=float, default=0.99)
    parser.add_argument('--resume', type=str, default="")

    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--save_model', action='store_true')
    parser.add_argument('--root_dir', type=str, default="outputs")
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=960)
    parser.add_argument('--nb_grids', type=int, default=6)
    parser.add_argument('--seed', type=int, default=1701)
    parser.add_argument('--ego_type', type=str, default="sfm")

    args = parser.parse_args()
    if args.gpu >= 0:
        args.device = torch.device('cuda:{}'.format(args.gpu))
    else:
        args.device = torch.device('cpu')
    return args


def get_model(args):
    mean = np.array([640., 476.23620605, 88.2875590389])
    std = np.array([227.59802246, 65.00177002, 52.7303319245])
    if "scale" not in args.model:
        mean, std = mean[:2], std[:2]

    mean = torch.from_numpy(mean).float()
    std = torch.from_numpy(std).float()

    logger.info("Mean: {}, std: {}".format(mean, std))
    if args.model == "cnn" or args.model == "cnn_scale":
        model = CNN(mean, std, args.device, args.channel_list, args.deconv_list, args.ksize_list,
                    args.dc_ksize_list, args.inter_list, args.last_list, args.pad_list)
    elif args.model == "cnn_pose" or args.model == "cnn_pose_scale":
        model = CNN_Pose(mean, std, args.device, args.channel_list, args.deconv_list, args.ksize_list,
                         args.dc_ksize_list, args.inter_list, args.last_list, args.pad_list)
    elif args.model == "cnn_ego" or args.model == "cnn_ego_scale":
        model = CNN_Ego(mean, std, args.device, args.channel_list, args.deconv_list, args.ksize_list,
                        args.dc_ksize_list, args.inter_list, args.last_list, args.pad_list, args.ego_type)
    elif args.model == "cnn_ego_pose" or args.model == "cnn_ego_pose_scale":
        model = CNN_Ego_Pose(mean, std, args.device, args.channel_list, args.deconv_list, args.ksize_list,
                             args.dc_ksize_list, args.inter_list, args.last_list, args.pad_list, args.ego_type)
    else:
        logger.info("Invalid argument: model={}".format(args.model))
        exit(1)

    if args.resume != "":
        model.load_state_dict(torch.load(args.resume))

    return model


def write_prediction(pred_dict, batch, pred_y):
    for idx in range(len(pred_y)):
        sample = tuple(batch[j][idx] if isinstance(batch[j], (list, tuple, torch.Tensor)) else batch[j] for j in range(len(batch)))
        past, ground_truth, pose, vid, frame, pid, flipped, egomotion, scale, mag, size = sample

        if isinstance(vid, torch.Tensor):
            vid = vid.item()
        if isinstance(frame, torch.Tensor):
            frame = frame.item()
        if isinstance(pid, torch.Tensor):
            pid = pid.item()
        if isinstance(flipped, torch.Tensor):
            flipped = flipped.item()
        if isinstance(scale, torch.Tensor):
            scale = scale.item()
        if isinstance(mag, torch.Tensor):
            mag = mag.item()
        if isinstance(size, torch.Tensor):
            size = size.tolist()

        frame, pid = str(frame), str(pid)
        vid = str(vid)

        if isinstance(pred_y, torch.Tensor):
            pred_np = pred_y[idx].detach().cpu().numpy()
        else:
            pred_np = pred_y[idx]
        if isinstance(ground_truth, torch.Tensor):
            ground_truth = ground_truth.detach().cpu().numpy()
        if isinstance(pose, torch.Tensor):
            pose = pose.detach().cpu().numpy()
        pose = np.array(pose)

        err = float(np.linalg.norm(pred_np - ground_truth, axis=1)[-1])

        front_cnt = 0
        hip_dists = []
        for t in range(pose.shape[0]):
            ps = pose[t]
            if float(ps[11, 0, 0]) - float(ps[8, 0, 0]) > 0:
                front_cnt += 1
            hip_dists.append(np.abs(float(ps[11, 0, 0]) - float(ps[8, 0, 0])))
        hip_dist = float(np.mean(hip_dists))
        front_ratio = front_cnt / pose.shape[0]

        if hip_dist < 0.25:
            traj_type = 2
        elif front_ratio > 0.75:
            traj_type = 0
        elif front_ratio < 0.25:
            traj_type = 1
        else:
            traj_type = 3

        if vid not in pred_dict:
            pred_dict[vid] = {}
        if frame not in pred_dict[vid]:
            pred_dict[vid][frame] = {}

        result = [vid, frame, pid, flipped, pred_np.tolist(), None, None, err, traj_type]
        pred_dict[vid][frame][pid] = result