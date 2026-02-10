"""
train_api
author Long-Chen Shen
"""

import os
import time
from ood.attacks import Attack
import ood.util as util


def make_attack(class_name, model, args) -> Attack:
    attack_cls = util.load_class(class_name)
    return attack_cls(model, **vars(args))


def train(
    net,
    dna_training_loader,
    optimizer,
    loss_function,
    epoch,
    args,
    output_interval,
    is_tensorboard=False,
    writer=None,
):
    start = time.time()
    total_loss = 0.0
    correct = 0.0

    attack = make_attack(args.attack, net, args)

    for batch_index, item in enumerate(dna_training_loader):
        dna_seqs = item["seq"].to(args.device).float()
        labels = item["label"].to(args.device)

        x_batch_attacked, y_batch_attacked = attack(dna_seqs, labels)
        net.train()
        optimizer.zero_grad()
        outputs = net(x_batch_attacked)
        loss = loss_function(outputs, y_batch_attacked)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x_batch_attacked.size(0)
        _, pred = outputs.max(1)
        correct += pred.eq(y_batch_attacked).sum().item()

        if batch_index % output_interval == 0 and batch_index != 0:
            current_item = batch_index * args.b + len(x_batch_attacked)
            print(
                "Training Epoch: {epoch} [{trained_samples}/{total_samples}]"
                "\t Average loss: {:0.4f}"
                "\tAccuracy: {:.4f}"
                "\tLR: {:0.6f}".format(
                    total_loss / current_item,
                    correct / current_item,
                    optimizer.param_groups[0]["lr"],
                    epoch=epoch,
                    trained_samples=current_item,
                    total_samples=len(dna_training_loader.dataset),
                )
            )

    if is_tensorboard:
        for name, param in net.named_parameters():
            layer, attr = os.path.splitext(name)
            attr = attr[1:]
            writer.add_histogram("{}/{}".format(layer, attr), param, epoch)

    finish = time.time()
    print(
        "----------- epoch {} training time consumed: {:.2f}s -----------".format(
            epoch, finish - start
        )
    )
    print(
        "Training Epoch: {epoch}\tAverage loss: {:0.4f}\tAccuracy: {:.4f}\tLR: {:0.6f}".format(
            total_loss / len(dna_training_loader.dataset),
            correct / len(dna_training_loader.dataset),
            optimizer.param_groups[0]["lr"],
            epoch=epoch,
        )
    )
    log_dic = {
        "epoch": epoch,
        "train_loss": total_loss / len(dna_training_loader.dataset),
        "train_acc": correct / len(dna_training_loader.dataset),
        "lr": optimizer.param_groups[0]["lr"],
    }
    return log_dic
