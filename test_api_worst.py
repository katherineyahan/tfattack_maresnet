"""
test_api
author Long-Chen Shen
"""
import time
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_curve, auc
from modified_attack_fine_tune.attacks import Attack
import modified_attack_fine_tune.util as util

def make_attack(class_name, model, args) -> Attack:
    attack_cls = util.load_class(class_name)
    return attack_cls(model, **vars(args))

@torch.no_grad()
def eval_training(net, dna_valid_loader, loss_function, softmax_output,
                  args, epoch=0, df_file=None, log_dic=None, train_after=False):
    print()
    print('============== Evaluating Network Start ==============')
    start = time.time()
    net.eval()
    # valid evaluating
    loss_valid, acc_valid, auc_valid, pred_result_valid = eval_model_valid(net=net, dataloader=dna_valid_loader,
                                                                     loss_function=loss_function,
                                                                     softmax_output=softmax_output,
                                                                     args=args)

    finish = time.time()
    print(' Valid set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, AUC: {:.4f}, Time consumed:{:.2f}s'.format(
        epoch,
        loss_valid,
        acc_valid,
        auc_valid,
        finish - start
    ))
    print('=============== Evaluating Network End ===============')
    print()
    if log_dic is not None and train_after:
        log_dic['valid_loss'] = loss_valid
        log_dic['valid_acc'] = acc_valid
        log_dic['valid_auc'] = auc_valid

        df = pd.read_pickle(df_file)
        df = pd.concat([df, pd.DataFrame([log_dic])], ignore_index=True)
        df.to_pickle(df_file)

    return loss_valid, acc_valid, auc_valid, pred_result_valid


def auc_computing(real, pred_numerics):
    for i in range(len(pred_numerics)):
        if np.isnan(pred_numerics[i]):
            pred_numerics[i] = 0.5
    fpr, tpr, thresholds = roc_curve(real, pred_numerics)
    roc_auc = auc(fpr, tpr)
    return roc_auc

def eval_model_valid(net, dataloader, loss_function, softmax_output, args):
    loss_all = 0.0
    correct = 0.0
    prob_all = []
    label_all = []
    attack = make_attack(args.attack, net, args)
    for item in dataloader:
        dna_seqs = item['seq'].to(args.device).float()
        labels = item['label'].to(args.device)

        x_val, y_val = attack(dna_seqs, labels)
        outputs = net(x_val)
        loss = loss_function(outputs, y_val)
        prob = softmax_output(outputs)
        loss_all += loss.item() * x_val.size(0)

        _, pred = outputs.max(1)
        prob_all.extend(prob[:, 1].cpu().numpy())
        label_all.extend(y_val.cpu().numpy())

        correct += pred.eq(y_val).sum().item()
    avg_loss = loss_all / len(dataloader.dataset)
    eval_acc = correct / len(dataloader.dataset)
    eval_auc = auc_computing(label_all, prob_all)
    return avg_loss, eval_acc, eval_auc, prob_all

@torch.no_grad()
def eval_model_test(net, dataloader, loss_function, softmax_output, args):
    loss_all = 0.0
    correct = 0.0
    prob_all = []
    label_all = []
    net.eval()
    attack = make_attack(args.attack, net, args)
    for item in dataloader:
        dna_seqs = item['seq'].to(args.device).float()
        labels = item['label'].to(args.device)

        x_test, y_test = attack(dna_seqs, labels)
        outputs = net(x_test)
        loss = loss_function(outputs, y_test)
        prob = softmax_output(outputs)
        loss_all += loss.item() * x_test.size(0)

        _, pred = outputs.max(1)
        prob_all.extend(prob[:, 1].cpu().numpy())
        label_all.extend(y_test.cpu().numpy())

        correct += pred.eq(y_test).sum().item()
    avg_loss = loss_all / len(dataloader.dataset)
    eval_acc = correct / len(dataloader.dataset)
    eval_auc = auc_computing(label_all, prob_all)
    return eval_acc, eval_auc, prob_all
