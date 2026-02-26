# transform_on_690datasets.py
# !/usr/bin/env	python3
"""
train network using pytorch
author Long-Chen Shen
"""

import sys
import os
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from train_api_worst import train
from test_api_worst import eval_training
from conf import settings
from utils import (
    get_network,
    get_training_dataloader,
    get_valid_dataloader,
    get_test_dataloader,
    best_auc_weights,
    save_best_result_attack,
)


def main(args):
    # Load the dataset
    dataset_path = args.dataset_path
    dataset_list = sorted(
        [
            d
            for d in os.listdir(dataset_path)
            if os.path.isdir(os.path.join(dataset_path, d))
        ]
    )

    # Fine-tune the pre-trained model on TFs
    for dataset_ in dataset_list:
        patience = settings.PATIENCE_TRANSFORM
        net = get_network(args)

        # Acquire training, validation and test set
        dataset = os.path.join(dataset_path, dataset_)
        dna_training_loader = get_training_dataloader(
            path=dataset, num_workers=0, batch_size=args.b, shuffle=False
        )  # originall True
        dna_valid_loader = get_valid_dataloader(
            path=dataset, num_workers=0, batch_size=args.b, shuffle=False
        )

        # Define model training
        loss_function = nn.CrossEntropyLoss()
        softmax_output = nn.Softmax(dim=1)
        optimizer = optim.SGD(
            params=net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4
        )
        train_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=settings.MILESTONES_TRANSFORM, gamma=0.8
        )

        # Create path for new checkpoints
        checkpoint_path_new = os.path.join(
            settings.CHECKPOINT_TRANSFER_PATH, args.net, settings.TIME_NOW, dataset_
        )

        # Use tensorboard
        if not os.path.exists(settings.LOG_TRANSFER_DIR):
            os.mkdir(settings.LOG_TRANSFER_DIR)

        #  Record the epoch
        df_path = os.path.join(
            settings.LOG_TRANSFER_DIR, args.net, settings.TIME_NOW, dataset_
        )
        if not os.path.exists(df_path):
            os.makedirs(df_path)
        df_file = os.path.join(df_path, "df_log.pickle")
        if not os.path.isfile(df_file):
            df_ = pd.DataFrame(
                columns=[
                    "epoch",
                    "lr",
                    "train_loss",
                    "train_acc",
                    "valid_loss",
                    "valid_acc",
                    "valid_auc",
                ]
            )
            df_.to_pickle(df_file)
            print("log DataFrame created!")

        # Create model_weights folder to save model
        if not os.path.exists(checkpoint_path_new):
            os.makedirs(checkpoint_path_new)
        checkpoint_path_new = os.path.join(
            checkpoint_path_new, "{net}-{epoch}-{type}.pth"
        )

        best_auc = 0.0
        best_result_str = ""
        resume_epoch = 0
        if args.resume:
            best_weights = best_auc_weights(args.checkpoint_path)
            if best_weights:
                weights_path = os.path.join(args.checkpoint_path, best_weights)
                print("found best auc weights file:{}".format(weights_path))
                print("load best training file to test auc...")
                net.load_state_dict(torch.load(weights_path, map_location=args.device))
                loss_valid, acc_valid, best_auc, pred_result_valid = eval_training(
                    net,
                    dna_valid_loader,
                    loss_function,
                    softmax_output,
                    args,
                    train_after=False,
                )
                # save best result
                best_result_str = acc_valid
                save_best_result_attack(df_path, pred_result_valid, acc_valid)
                print("best valid auc is {:0.4f}".format(best_auc))

        for epoch in range(1, settings.EPOCH_TRANSFORM + 1):
            if args.resume:
                if epoch <= resume_epoch:
                    continue

            output_interval = settings.OUTPUT_INTERVAL_TRANSFER
            log_dic = train(
                net,
                dna_training_loader,
                optimizer,
                loss_function,
                epoch,
                args,
                output_interval,
            )

            if epoch > args.warm:
                train_scheduler.step()

            loss_valid, acc_valid, auc_valid, pred_result_valid = eval_training(
                net,
                dna_valid_loader,
                loss_function,
                softmax_output,
                args,
                epoch=epoch,
                df_file=df_file,
                log_dic=log_dic,
                train_after=True,
            )

            # start to save best performance model after learning rate decay to 0.01
            if best_auc < auc_valid:
                weights_path = checkpoint_path_new.format(
                    net=args.net, epoch=epoch, type="best"
                )
                print("saving weights file to {}".format(weights_path))
                torch.save(net.state_dict(), weights_path)
                best_auc = auc_valid
                patience = settings.PATIENCE_TRANSFORM
                # save best result
                save_best_result_attack(df_path, pred_result_valid, acc_valid)
                best_result_str = acc_valid
                continue

            patience -= 1
            if patience == 0:
                print("The best:", dataset_, best_result_str)
                print("The end!")
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Dataset arguments
    parser.add_argument(
        "--dataset_path", type=str, required=True, help="Path to the dataset directory"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to a checkpoint folder to resume training from",
    )
    parser.add_argument("-net", type=str, default="maresnet", help="net type")
    parser.add_argument("-b", type=int, default=64, help="batch size for dataloader")
    parser.add_argument("-warm", type=int, default=0, help="warm up training phase")
    parser.add_argument("-lr", type=float, default=0.0004, help="initial learning rate")
    parser.add_argument("-resume", action="store_true", help="resume training")
    parser.add_argument("-depth", type=int, help="WideResNet depth")
    parser.add_argument("-widen_factor", type=int, help="WideResNet widen factor")

    # Attack method params
    parser.add_argument("--attack", type=str, default="attacks.RandomCrop")
    parser.add_argument("--seq_length", type=int, default=90)
    parser.add_argument("--attack_batch", type=int, default=64)
    parser.add_argument("--n_try", type=int)
    parser.add_argument("--loss", type=str, default="xe")

    args = parser.parse_args()
    args.with_cuda = True
    cuda_condition = torch.cuda.is_available() and args.with_cuda
    args.device = torch.device("cuda:0" if cuda_condition else "cpu")

    main(args)
