"""
train_api
author Long-Chen Shen
"""
import os
import time
from attacks import Attack
import util as util

def make_attack(class_name, model, args) -> Attack:
    attack_cls = util.load_class(class_name)
    return attack_cls(model, **vars(args))

def train(net, dna_training_loader, optimizer, loss_function,
          epoch, args, output_interval, attack_method,
          is_tensorboard=False, writer=None):

    start = time.time()
    total_loss = 0.0
    correct = 0.0

    attack = make_attack(args.attack, net, args)
    net.train()
    for batch_index, item in enumerate(dna_training_loader):
        dna_seqs = item['seq'].to(args.device).float()
        labels = item['label'].to(args.device)

        #perform the attack
        xbatcha, ybatcha = attack(dna_seqs, labels)
        optimizer.zero_grad() #clear previous gradients
        outputs = net(xbatcha) #pass DNA sequences through the model
        loss = loss_function(outputs, ybatcha)
        loss.backward() #compute gradients via backpropagation
        optimizer.step() #updates model parameters using the optimizer
        total_loss += loss.item() * xbatcha.size(0)
        _, pred = outputs.max(1)
        correct += pred.eq(ybatcha).sum().item()

        if batch_index % output_interval == 0 and batch_index != 0:
            current_item = batch_index * args.b + len(xbatcha)
            print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]'
                  '\t Average loss: {:0.4f}'
                  '\tAccuracy: {:.4f}'
                  '\tLR: {:0.6f}'.format(
                    total_loss / current_item,
                    correct / current_item,
                    optimizer.param_groups[0]['lr'],
                    epoch=epoch,
                    trained_samples=current_item,
                    total_samples=len(dna_training_loader.dataset)
                    ))

    if is_tensorboard:
        for name, param in net.named_parameters():
            layer, attr = os.path.splitext(name)
            attr = attr[1:]
            writer.add_histogram("{}/{}".format(layer, attr), param, epoch)

    finish = time.time()
    print('----------- epoch {} training time consumed: {:.2f}s -----------'.format(epoch, finish - start))
    print('Training Epoch: {epoch}\tAverage loss: {:0.4f}\tAccuracy: {:.4f}\tLR: {:0.6f}'.format(
        total_loss / len(dna_training_loader.dataset),
        correct / len(dna_training_loader.dataset),
        optimizer.param_groups[0]['lr'],
        epoch=epoch,
    ))
    log_dic = {
        "epoch": epoch,
        "train_loss": total_loss / len(dna_training_loader.dataset),
        "train_acc": correct / len(dna_training_loader.dataset),
        "lr": optimizer.param_groups[0]['lr']
    }
    return log_dic
