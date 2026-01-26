from abc import ABC
from abc import abstractmethod
#import tensorflow as tf
import torch
import numpy as np
import time
import torch.nn as nn

class Attack(ABC):
    def __init__(self, model, **kwargs) -> None:
        self.model = model

    @abstractmethod
    def __call__(self, x, y):
        pass

    @abstractmethod
    def get_name(self):
        pass


class Shuffle(Attack):

    def __init__(self, model, seed=9, **kwargs) -> None:
        super().__init__(model, **kwargs)
        self.seed = seed

    def get_name(self):
        return "Shuffle"

    def __call__(self, x, y):
        rnd = np.random.RandomState(self.seed)
        x_shuffle = np.array(x, copy=True)
        seq_length = x.shape[1]
        for i in range(x.shape[0]):
            idxs = rnd.permutation(seq_length)
            x_shuffle[i] = x[i][idxs]
        return x_shuffle, y


class Uniform(Attack):

    def __init__(self, model, seed=9, **kwargs) -> None:
        super().__init__(model, **kwargs)
        self.seed = seed

    def get_name(self):
        return "Uniform"

    def __call__(self, x, y):
        rnd = np.random.RandomState(self.seed)
        x_uni = np.array(x, copy=True, dtype=np.float32)
        shape = x[0].shape
        for i in range(x_uni.shape[0]):
            x_uni[i] = rnd.uniform(0, 1, shape)
        return x_uni, y


class GenSeq(Attack):

    def __init__(self, model, seed=9, **kwargs) -> None:
        super().__init__(model, **kwargs)
        self.seed = seed

    def get_name(self):
        return "GenSeq"

    def __call__(self, x, y):
        rnd = np.random.RandomState(self.seed)
        x_gen = np.zeros_like(x, dtype=np.float32)
        sample_shape = x[0].shape
        seq_idxs = np.arange(0, sample_shape[0])
        for i in range(x_gen.shape[0]):
            random_ids = rnd.randint(0, 4, sample_shape[0])
            x_gen[i, seq_idxs, random_ids] = 1
            # print(np.mean(x_gen[i], axis=0))
        return x_gen, y


class MiddleCrop(Attack):

    def __init__(self, model, seq_length, **kwargs) -> None:
        super().__init__(model, **kwargs)
        self.seq_length = seq_length

    def get_name(self):
        return "MiddleCrop"

    def __call__(self, x, y):
        dim_to_crop = x.shape[2] - self.seq_length
        if dim_to_crop == 0:
            return x, y
        elif dim_to_crop < 0:
            raise Exception('seq length must be lower or equal to {0}'.format(x.shape[1]))
        return x[:, :, dim_to_crop // 2:-dim_to_crop // 2], y


class RandomCrop(Attack):

    def __init__(self, model, seq_length, **kwargs) -> None:
        super().__init__(model, **kwargs)
        self.seq_length = seq_length

    def get_name(self):
        return "RandomCrop"

    def __call__(self, x, y):
        x_crop = torch.zeros((x.shape[0], x.shape[1], self.seq_length), dtype=x.dtype, device=x.device)

        for i in range(x.shape[0]):
            max_start = x.shape[2] - self.seq_length
            if max_start < 0:
                raise ValueError(f"Sequence too short: {x.shape[2]} < {self.seq_length}")
            start = torch.randint(0, max_start + 1, (1,)).item()
            x_crop[i] = x[i, :, start:start + self.seq_length]

        return x_crop, y


class WorstCrop(Attack):

    def __init__(self, model, seq_length, loss='zero-one', attack_batch=64, n_try=None, debug=False,
                 seed=9,
                 **kwargs) -> None:
        super().__init__(model, **kwargs)
        self.rnd = np.random.RandomState(seed)
        self.n_try = n_try
        self.seq_length = seq_length
        self.batch_size = attack_batch
        self.debug = debug
        if loss == 'zero-one':
            self.loss = lambda p, y: (p.max(1)[1] != y).float()

        elif loss == 'mse':
            self.loss = lambda p, y: torch.sum((p - y) ** 2)
        elif loss == 'xe':
            # Numerically unstable
            # self.loss = lambda p, y: -np.sum(np.log(p) * y, axis=1)
            # self.loss = lambda p, y: -np.log(np.sum(p * y, axis=1))
            eps = 1e-4
            # negative log likelihood
            criterion = nn.NLLLoss(reduction='none')
            # self.loss = lambda p, y: -np.log(np.maximum(np.sum(p * y, axis=1), eps))
            # store loss function to be used later when p and y are available
            self.loss = lambda p, y: criterion(torch.log(p.clamp_min(eps)), y)


        elif loss == 'bce':
            eps = 1e-4
            #self.loss = lambda p, y: np.mean(np.where(y, -np.log(np.maximum(p, eps)), -np.log(np.maximum(1 - p, eps))),
             #                                axis=1)
            self.loss = lambda p, y: torch.mean(
                torch.where(
                    y.bool(),
                    -torch.log(torch.clamp(p, min=eps)),
                    -torch.log(torch.clamp(1 - p, min=eps))
                ),
                dim=1
            )

    def get_name(self):
        return "WorstCrop"

    def pred_slides(self, x, shifts, requires_grad=True): #get model predictions for each crop

        was_training = self.model.training

        self.model.eval()
        dim_to_crop = x.shape[2] - self.seq_length
        x_slides = []
        softmax_output = nn.Softmax(dim=1)
        for i in shifts:
            x_slides.append(x[:, :, i:x.shape[2] - (dim_to_crop - i)])
        x_all = torch.cat(x_slides, dim=0).to("cpu")

        with torch.set_grad_enabled(requires_grad): # don't need gradient torch.no_grad, see if there's a difference of loss in eval or train mode
            logits_all = self.model(x_all)
            pred_all = softmax_output(logits_all)
            #pred_labels = torch.argmax(pred_all, dim=1)

       # pred_all = self.model(np.concatenate(x_slides, axis=0), self.batch_size)
        pred_slds = []
        for i in range(shifts.shape[0]):
            pred_slds.append(pred_all[i * x.shape[0]:(i + 1) * x.shape[0]])

        if was_training:
            self.model.train()
        return pred_slds

    def __call__(self, x, y):
        dim_to_crop = x.shape[2] - self.seq_length
        #x_adv = x[:, :, :-dim_to_crop] # remove the last dim_to_crop nucleotides
        shifts = np.arange(dim_to_crop + 1) if self.n_try is None else self.rnd.choice(np.arange(dim_to_crop + 1),
                                                                                       self.n_try, replace=False) # how many shifts to try,
        x_adv = x[:, :, shifts[0]:x.shape[2] - (dim_to_crop - shifts[0])]
        preds = self.pred_slides(x, shifts, requires_grad=self.model.training)
        l_val = self.loss(preds[0], y)
        n = dim_to_crop + 1 if self.n_try is None else self.n_try
        if self.debug:
            print('Progress: {:.3f} {:.5f}'.format(1 / n, np.mean(l_val)), end='\r')
        for idx, i in enumerate(shifts[1:]):
            if self.debug:
                time.sleep(1)
                print('Progress: {:.3f} {:.5f}'.format((idx + 2) / n, np.mean(l_val)), end='\r')
            x_adv_i = x[:, :, i:x.shape[2] - (dim_to_crop - i)]
            pred_i = preds[idx + 1]
            l_vali = self.loss(pred_i, y)
            improved = l_vali > l_val
            if torch.any(improved):
                #x_adv = tf.where(tf.reshape(improved, (improved.shape[0], 1, 1)), x_adv_i, x_adv)
                mask = improved.view(-1, 1, 1) # reshape 1D list of booleans into a 3D structure to apply it to tensor
                x_adv = torch.where(mask, x_adv_i, x_adv)
                l_val[improved] = l_vali[improved]
        if self.debug:
            print('')
        return x_adv, y

