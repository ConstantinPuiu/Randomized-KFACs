import math

import torch
import torch.optim as optim

from kfac_utils_for_vgg16_bn import (ComputeCovA, ComputeCovG)
from kfac_utils_for_vgg16_bn import update_running_stat

#import ipdb

import numpy as np

def X_reg_inverse_M(U,D,M,lambdda):
    # X = UDU^T; want to compute (X + lambda I)^{-1}M
    # X is low rank! X is square: X is either AA^T or GG^T
    # This is actually for G as G sits before
    # the damping here is adaptive! - it adjusts based on the amxium eigenvalue !
    lambdda = lambdda * torch.max(D)
    #### effective computations :
    U_T_M = torch.matmul(U.T, M)
    U_times_reg_D_times_U_T_M = torch.matmul( U * ( 1/(D + lambdda) - 1/lambdda), U_T_M)
    return U_times_reg_D_times_U_T_M + (1/lambdda) * M
    
def M_X_reg_inverse(U,D,M,lambdda):
    # X = UDU^T; want to compute (X + lambda I)^{-1}M
    # X is low rank! X is square: X is either AA^T or GG^T
    # This is actually for A as A sits after M
    # the damping here is adaptive! - it adjusts based on the amxium eigenvalue !
    lambdda = lambdda * torch.max(D)
    #### effective computations :
    M_times_U_times_reg_D_times_U_T = M @ ( U * ( 1/(D + lambdda) - 1/lambdda) ) @ U.T
    return M_times_U_times_reg_D_times_U_T + (1/lambdda) * M
    
def srevd_lowrank(M, oversampled_rank, target_rank, niter, start_matrix = None):
    #generate random matrix
    devicce =  torch.device('cuda:0')
    n = M.shape[0]
    if start_matrix == None:
        Omega = torch.randn(size = [n, oversampled_rank], device = devicce)
    else:
        Omega = torch.hstack([start_matrix,torch.randn(size = [n,oversampled_rank - start_matrix.shape[1]], device = devicce)])
    
    #multiply to produce subspace
    for i in range(niter):
        Omega = M @ Omega
    
    # QR
    Omega, _ = torch.linalg.qr(Omega, mode='reduced',out=None)
    M = Omega.T @ M @ Omega
    
    #get small space eigh
    D, U = torch.linalg.eigh(M, UPLO='L',  out=None) # EIGENVALUE ARE IN ASCENDING ORDER!
    Omega = Omega @ U #store in omega cause that's bigger! (i.e. automatically delete the bigger one to optimize memory)
    
    return D[-target_rank:] + 0.0, Omega[:,-target_rank:] + 0.0 # OMEGA IS u - overwritten for efficiency

class SRE_KFACOptimizer(optim.Optimizer):
    def __init__(self,
                 model,
                 lr_function = lambda epoch_n, iteration_n: 0.1,
                 momentum=0.9,
                 stat_decay=0.95,
                 damping=0.001,
                 kl_clip=0.001,
                 weight_decay=0,
                 TCov=10,
                 TInv=100,
                 batch_averaged=True,
                 rsvd_rank = 220,
                 oversampling_parameter = 10,
                 rsvd_niter = 3):
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        
        self.epoch_number = 1
        self.lr_function = lr_function
        #self.lr = self.lr_function(self.epoch_number, 0)
        
        defaults = dict(lr = self.lr_function(self.epoch_number, 0), 
                        momentum=momentum, damping=damping,
                        weight_decay=weight_decay)
      
        super(SRE_KFACOptimizer, self).__init__(model.parameters(), defaults)
        self.CovAHandler = ComputeCovA()
        self.CovGHandler = ComputeCovG()
        self.batch_averaged = batch_averaged

        self.known_modules = {'Linear', 'Conv2d'}

        self.modules = []
        self.grad_outputs = {}

        self.model = model
        self._prepare_model()

        self.steps = 0
        
        self.U_aa, self.D_aa, self.V_aa = {}, {}, {}
        self.U_gg, self.D_gg, self.V_gg = {}, {}, {}
        
        self.m_aa, self.m_gg = {}, {}
        self.Q_a, self.Q_g = {}, {}
        self.d_a, self.d_g = {}, {}
        self.stat_decay = stat_decay

        self.kl_clip = kl_clip
        self.TCov = TCov
        self.TInv = TInv
        
        #rsvd_params
        self.rsvd_rank = rsvd_rank
        self.total_rsvd_rank = oversampling_parameter + rsvd_rank
        self.rsvd_niter = rsvd_niter
        
        # debug quantities
        self.matrix_save_counter = 0

    def _save_input(self, module, input):
        if torch.is_grad_enabled() and self.steps % self.TCov == 0:
            aa = self.CovAHandler(input[0].data, module)
            # Initialize buffers
            if self.steps == 0:
                self.m_aa[module] = torch.diag(aa.new(aa.size(0)).fill_(1))
                
            #ipdb.set_trace(context = 7)
            
            update_running_stat(aa, self.m_aa[module], self.stat_decay)

    def _save_grad_output(self, module, grad_input, grad_output):
        # Accumulate statistics for Fisher matrices
        if self.acc_stats and self.steps % self.TCov == 0:
            gg = self.CovGHandler(grad_output[0].data, module, self.batch_averaged)
            # Initialize buffers
            if self.steps == 0:
                self.m_gg[module] = torch.diag(gg.new(gg.size(0)).fill_(1))
            
            #ipdb.set_trace(context = 7)
            
            update_running_stat(gg, self.m_gg[module], self.stat_decay)

    def _prepare_model(self):
        count = 0
        print(self.model)
        print("=> We keep following layers in KFAC. ")
        for module in self.model.modules():
            classname = module.__class__.__name__
            # print('=> We keep following layers in KFAC. <=')
            if classname in self.known_modules:
                self.modules.append(module)
                module.register_forward_pre_hook(self._save_input)
                module.register_backward_hook(self._save_grad_output)
                print('(%s): %s' % (count, module))
                count += 1

    def _update_inv(self, m):
        """Do eigen decomposition for computing inverse of the ~ fisher.
        :param m: The layer
        :return: no returns.
        """
        eps = 1e-10  # for numerical stability
        #self.d_a[m], self.Q_a[m] = torch.symeig( self.m_aa[m] + 0.01* torch.eye(self.m_aa[m].shape[0], device = torch.device('cuda:0'))  , eigenvectors=True)
        oversampled_rank = min(self.m_aa[m].shape[0], self.total_rsvd_rank)
        actual_rank = min(self.m_aa[m].shape[0], self.rsvd_rank)
        self.d_a[m], self.Q_a[m] = srevd_lowrank(M = self.m_aa[m], oversampled_rank = oversampled_rank, target_rank = actual_rank, niter = self.rsvd_niter, start_matrix = None)
        
        #self.d_g[m], self.Q_g[m] = torch.symeig( self.m_gg[m] , eigenvectors=True) # computes the eigen decomposition of bar_A and G matrices
        oversampled_rank = min(self.m_gg[m].shape[0], self.total_rsvd_rank)
        actual_rank = min(self.m_gg[m].shape[0], self.rsvd_rank)
        self.d_g[m], self.Q_g[m] = srevd_lowrank(M = self.m_gg[m], oversampled_rank = oversampled_rank, target_rank = actual_rank, niter = self.rsvd_niter, start_matrix = None)

        self.d_a[m].mul_((self.d_a[m] > eps).float())
        self.d_g[m].mul_((self.d_g[m] > eps).float())

    @staticmethod
    def _get_matrix_form_grad(m, classname):
        """
        :param m: the layer
        :param classname: the class name of the layer
        :return: a matrix form of the gradient. it should be a [output_dim, input_dim] matrix.
        """
        if classname == 'Conv2d':
            p_grad_mat = m.weight.grad.data.view(m.weight.grad.data.size(0), -1)  # n_filters * (in_c * kw * kh)
        else:
            p_grad_mat = m.weight.grad.data
        if m.bias is not None:
            p_grad_mat = torch.cat([p_grad_mat, m.bias.grad.data.view(-1, 1)], 1)
        return p_grad_mat

    def _get_natural_grad(self, m, p_grad_mat, damping):
        """
        :param m:  the layer
        :param p_grad_mat: the gradients in matrix form
        :return: a list of gradients w.r.t to the parameters in `m`
        """
        # p_grad_mat is of output_dim * input_dim
        # inv((ss')) p_grad_mat inv(aa') = [ Q_g (1/R_g) Q_g^T ] @ p_grad_mat @ [Q_a (1/R_a) Q_a^T]
        
        v1 = X_reg_inverse_M(U = self.Q_g[m], D = self.d_g[m], M = p_grad_mat, lambdda = damping)
        v = M_X_reg_inverse(U = self.Q_a[m], D = self.d_a[m], M = v1, lambdda = damping)
        
        '''v1 = self.Q_g[m].t() @ p_grad_mat @ self.Q_a[m]
        v2 = v1 / (self.d_g[m].unsqueeze(1) * self.d_a[m].unsqueeze(0) + damping)
        v = self.Q_g[m] @ v2 @ self.Q_a[m].t()'''
        
        if m.bias is not None:
            # we always put gradient w.r.t weight in [0]
            # and w.r.t bias in [1]
            v = [v[:, :-1], v[:, -1:]]
            v[0] = v[0].view(m.weight.grad.data.size())
            v[1] = v[1].view(m.bias.grad.data.size())
        else:
            v = [v.view(m.weight.grad.data.size())]

        return v

    def _kl_clip_and_update_grad(self, updates, lr):

        for m in self.modules:
            v = updates[m]
            if m.bias is not None:
                concat_V = torch.cat( (v[0].flatten(), v[1].flatten() ) )
                numel_v = torch.numel(concat_V)
                nu = min(1, self.kl_clip/(torch.norm( concat_V, p=2 )/math.sqrt(numel_v)))
            else:
                nu = min(1, self.kl_clip/(torch.norm(v[0], p = 2)/math.sqrt(torch.numel(v[0]))))
            m.weight.grad.data.copy_(v[0])
            m.weight.grad.data.mul_(nu)
            if m.bias is not None:
                m.bias.grad.data.copy_(v[1])
                m.bias.grad.data.mul_(nu)

    def _step(self, closure):
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1, d_p)
                    d_p = buf
                if weight_decay != 0 and self.steps >= 20 * self.TCov: # movedD weight decay AFTER momentum. 
                # it used to be before which meant it meant you got momentum on weight decay.
                    d_p.add_(weight_decay, p.data)

                p.data.add_(-group['lr'], d_p)

    def step(self, epoch_number, error_savepath, closure = None):
        self.lr = self.lr_function(epoch_number, self.steps)
        for g in self.param_groups:
            g['lr'] = self.lr_function(epoch_number, self.steps)
            
        group = self.param_groups[0]
        lr = group['lr']
        damping = group['damping']
        updates = {}
        for m in self.modules:
            classname = m.__class__.__name__
            if self.steps % self.TInv == 0:
                self._update_inv(m)
            p_grad_mat = self._get_matrix_form_grad(m, classname)
            v = self._get_natural_grad(m, p_grad_mat, damping)
            updates[m] = v
        self._kl_clip_and_update_grad(updates, lr)

        self._step(closure)
        self.steps += 1