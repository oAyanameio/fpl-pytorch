# pytorch版本训练代码，训练结束后会做evalutaion，所以没有再单独做evaluation了

# The main training file, afer calling run.py to generate commands.
# This file will be called to start the main training parts.
# Look at the command file in gen_scripts for input parameters.

# python -u train_cv.py --root_dir experiments/5fold_proposed_only_240303_130532
# --in_data dataset/id_list_20_180521_173817_20_sp5_2_5.joblib --nb_iters 17000
# --iter_snapshot 1000 --optimizer adam --height 960 --batch_size 64 --save_model
# --nb_train -1 --pred_len 10 --channel_list 32 64 128 128 --deconv_list 256 128 64 32
# --ksize_list 3 3 3 3 --inter_list 256 --input_len 10 --lr_step_list 5000 10000 15000
# --model cnn_ego_pose_scale --nb_splits 5 --eval_split 4 --gpu 0

import os
import time
import json
import joblib
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
from utils.dataset import SceneDatasetCV
from utils.evaluation import Evaluator
from utils.summary_logger import SummaryLogger
from utils.generic import get_args, get_model, write_prediction
from mllogger import MLLogger

logger = MLLogger(init=False)


def concat_examples_pytorch(batch, device, data_idxs):
    batch_array = []
    # batch 11*batch_size
    for idx in data_idxs:
        batch_array.append(batch[idx].to(device))
    return batch_array


def validate_and_report(model, valid_loader, args, train_eval, valid_eval, summary, save_dir, epochs, st):
    args_dict = vars(args).copy()
    args_dict.pop('device', None)
    prediction_dict = {
        "arguments": args_dict,
        "predictions": {}
    }
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():  # 在评估时不计算梯度
        # 实现验证逻辑
        for batch in valid_loader:
            inputs = concat_examples_pytorch(batch, args.device, data_idxs)
            loss, pred_y = model(inputs)
            valid_eval.update(loss.item(), pred_y, batch)

            # 处理batch中的每个样本
            write_prediction(prediction_dict["predictions"], batch, pred_y)

        message_str = "Train loss {} / ADE {} / FDE {}, valid loss {} / ADE {} / FDE {}, elapsed time: {} (s)"
        logger.info(message_str.format(train_eval("loss"), train_eval("ade"), train_eval("fde"),
                                       valid_eval("loss"), valid_eval("ade"), valid_eval("fde"), time.time() - st))

        train_eval.update_summary(summary, epochs, ["loss", "ade", "fde"])
        valid_eval.update_summary(summary, epochs, ["loss", "ade", "fde"])

        predictions = prediction_dict["predictions"]
        pred_list = [[pred for vk, v_dict in sorted(predictions.items())
                      for fk, f_dict in sorted(v_dict.items())
                      for pk, pred in sorted(f_dict.items()) if pred[8] == idx] for idx in range(4)]

        error_rates = [np.mean([pred[7] for pred in preds]) for preds in pred_list]
        logger.info("Towards {} / Away {} / Across {} / Other {}".format(*error_rates))

        prediction_path = os.path.join(save_dir, "prediction.json")
        with open(prediction_path, "w") as f:
            json.dump(prediction_dict, f)


if __name__ == "__main__":
    args = get_args()

    torch.manual_seed(args.seed)
    start = time.time()

    logger.initialize(args.root_dir)
    logger.info(vars(args))
    save_dir = logger.get_savedir()
    logger.info("Written to {}".format(save_dir))
    summary = SummaryLogger(args, logger, os.path.join(args.root_dir, "summary.csv"))
    summary.update("finished", 0)

    data_dir = "data"
    data = joblib.load(args.in_data)
    traj_len = data["trajectories"].shape[1]

    # Load training data
    train_splits = list(filter(lambda x: x != args.eval_split, range(args.nb_splits)))
    valid_split = args.eval_split + args.nb_splits

    train_dataset = SceneDatasetCV(data, args.input_len, args.offset_len, args.pred_len,
                                   args.width, args.height, data_dir, train_splits, args.nb_train,
                                   True, "scale" in args.model, args.ego_type)
    logger.info(train_dataset.X.shape)
    valid_dataset = SceneDatasetCV(data, args.input_len, args.offset_len, args.pred_len,
                                   args.width, args.height, data_dir, valid_split, -1,
                                   False, "scale" in args.model, args.ego_type)
    logger.info(valid_dataset.X.shape)

    # X: input, Y: output, poses, egomotions
    # dataset.py __getitem__函数返回7个变量，选其中4个作为模型输入给forward
    data_idxs = [0, 1, 2, 7]
    if data_idxs is None:
        logger.info("Invalid argument: model={}".format(args.model))
        exit(1)

    model = get_model(args).to(args.device)  # 确保get_model返回的是PyTorch模型

    optimizer = optim.Adam(model.parameters(), weight_decay=1e-4)
    scheduler = MultiStepLR(optimizer, milestones=args.lr_step_list, gamma=0.5)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.nb_jobs)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.nb_jobs)

    train_eval = Evaluator("train", args)
    valid_eval = Evaluator("valid", args)

    logger.info("Training...")
    train_eval.reset()
    st = time.time()

    epochs = 28
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()  # 清除旧的梯度
            # 将数据和标签移动到相应设备
            inputs = concat_examples_pytorch(batch, args.device, data_idxs)
            # 前向传播
            loss, pred_y = model(inputs)
            # 反向传播和优化
            loss.backward()  # 计算梯度
            optimizer.step()  # 更新参数
            # 更新训练评估
            train_eval.update(loss.item(), pred_y, batch)
        scheduler.step()  # 更新学习率
        msg = "Epoch {} / {} : train loss {} / ADE {} / FDE {}"
        logger.info(msg.format(epoch + 1, epochs, train_eval("loss"), train_eval("ade"), train_eval("fde")))
        # if epoch % 2 == 1:
        #     validate_and_report(model, valid_loader, args)
        #     print("-------------------")
    logger.info("Validation...")
    validate_and_report(model, valid_loader, args, train_eval, valid_eval, summary, save_dir, epochs, st)
    if args.save_model:
        torch.save(model.state_dict(), os.path.join(save_dir, "fpl-pytorch.pth"))
    summary.update("finished", 1)
    summary.write()
    logger.info("Elapsed time: {} (s), Saved at {}".format(time.time() - start, save_dir))
