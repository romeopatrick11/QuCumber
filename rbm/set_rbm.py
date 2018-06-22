from rbm import RBM
import torch
import numpy as np


class SetRBMWeightEvolution:
    def __init__(self, zeta, total_epochs, seed=1234):
        self.zeta = zeta
        self.total_epochs = total_epochs
        self.rand_state = np.random.RandomState(seed)

    def __call__(self, rbm, epoch):
        # Cut weak connections
        bounds = np.array([self.zeta / 2., 1. - (self.zeta / 2.)])*100
        pt = np.percentile(rbm.weights.data.detach().cpu().numpy(), bounds)
        pt = tuple(pt)

        weight_mask = ((pt[0] > rbm.weights.data).to(device=rbm.device)
                       | (rbm.weights.data > pt[1]).to(device=rbm.device))

        rbm.weights.data = rbm.weights.mul(weight_mask.to(dtype=torch.double))

        # Grow new connections
        if epoch < self.total_epochs:
            # Get the indices of the nonzero weights
            non_zero_weight_indices = np.transpose(np.nonzero(
                weight_mask.detach().cpu().numpy()))

            num_deleted_cnxns = len(np.unique(non_zero_weight_indices[:, 0]))
            k = rbm.num_visible - num_deleted_cnxns

            # Randomly select k of them
            rand_indices = self.rand_state.randint(
                0, len(non_zero_weight_indices), size=k)
            idx = np.transpose(non_zero_weight_indices[rand_indices])

            # Reinitialize weights corresponding to the selected indices
            rbm.weights[idx] = (torch.randn(k) / np.sqrt(rbm.num_visible))


class SetRBM(RBM):
    def __init__(self, num_visible, num_hidden, epsilon, zeta,
                 gpu=True, seed=1234):
        super(SetRBM, self).__init__(num_visible=num_visible,
                                     num_hidden=num_hidden,
                                     gpu=gpu,
                                     seed=seed)
        self.epsilon = epsilon
        self.zeta = zeta
        mask_prob = 1.
        mask_prob -= ((num_visible + num_hidden) * epsilon
                      / (num_visible * num_hidden))
        mask_prob = 0. if mask_prob < 0 else mask_prob
        weight_mask = ((torch.rand_like(self.weights) >= mask_prob)
                       .to(dtype=torch.double))

        self.weights.data = self.weights.mul(weight_mask)

    def __repr__(self):
        return ("SetRBM(num_visible={}, num_hidden={}, gpu={})"
                .format(self.num_visible, self.num_hidden, self.gpu))

    def train(self, data, epochs, batch_size,
              k=10, persistent=False,
              lr=1e-3, momentum=0.0,
              method='sgd', l1_reg=0.0, l2_reg=0.0,
              initial_gaussian_noise=0.0, gamma=0.55,
              callbacks=[], progbar=False,
              **kwargs):
        callbacks = callbacks + [
            SetRBMWeightEvolution(self.zeta, epochs, seed=self.seed)
        ]
        super(SetRBM, self).train(
            data, epochs, batch_size,
            k=k, persistent=persistent,
            lr=lr, momentum=momentum,
            method=method, l1_reg=l1_reg, l2_reg=l2_reg,
            initial_gaussian_noise=initial_gaussian_noise, gamma=gamma,
            callbacks=callbacks, progbar=progbar,
            **kwargs)
