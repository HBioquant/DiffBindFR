# Copyright (c) MDLDrugLib. All rights reserved.
import torch
from torch.optim.optimizer import Optimizer

from .builder import OPTIMIZERS

try:
    # triton v2.0 required
    import triton
    import triton.language as tl
    @triton.autotune(configs = [
        triton.Config({'BLOCK_SIZE': 128}, num_warps = 4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps = 8),
    ], key = ['n_elements'])
    @triton.jit
    def update_fn_kernel(
          p_ptr,
          grad_ptr,
          exp_avg_ptr,
          lr,
          wd,
          beta1,
          beta2,
          n_elements,
          BLOCK_SIZE, # tl.constexpr
    ):
        pid = tl.program_id(axis = 0)

        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)

        mask = offsets < n_elements

        # offsetted pointers
        offset_p_ptr = p_ptr + offsets
        offset_grad_ptr = grad_ptr + offsets
        offset_exp_avg_ptr = exp_avg_ptr + offsets

        # load
        p = tl.load(offset_p_ptr, mask = mask)
        grad = tl.load(offset_grad_ptr, mask = mask)
        exp_avg = tl.load(offset_exp_avg_ptr, mask = mask)

        # stepweight decay
        p = p * (1 - lr * wd)

        # diff between momentum running average and grad
        diff = exp_avg - grad

        # weight update
        update = diff * beta1 + grad

        # torch.sign
        can_update = update != 0
        update_sign = tl.where(update > 0, -lr, lr)

        p = p + update_sign * can_update

        # decay the momentum running average coefficient
        exp_avg = diff * beta2 + grad

        # store new params and momentum running average coefficient
        tl.store(offset_p_ptr, p, mask = mask)
        tl.store(offset_exp_avg_ptr, exp_avg, mask = mask)

    def update_fn_triton(
        p: torch.Tensor,
        grad: torch.Tensor,
        exp_avg: torch.Tensor,
        lr: float,
        wd: float,
        beta1: float,
        beta2: float
    ):
        assert all([t.is_cuda for t in (p, grad, exp_avg)])
        n_elements = p.numel()

        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

        update_fn_kernel[grid](
            p,
            grad,
            exp_avg,
            lr,
            wd,
            beta1,
            beta2,
            n_elements
        )
except ImportError:
    update_fn_triton = None


def update_fn(p, grad, exp_avg, lr, wd, beta1, beta2):
    # stepweight decay
    p.data.mul_(1 - lr * wd)

    # weight update
    update = exp_avg.clone().mul_(beta1).add(grad, alpha=1 - beta1).sign_()
    p.add_(update, alpha=-lr)

    # decay the momentum running average coefficient
    exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)


@OPTIMIZERS.register_module()
class Lion(Optimizer):
    """Implements Lion algorithm."""
    def __init__(
            self,
            params,
            lr = 1e-4,
            betas = (0.9, 0.99),
            weight_decay = 0.0,
            use_triton = False
    ):
      """
      Initialize the hyperparameters.
      Args:
        params (iterable): iterable of parameters to optimize or dicts defining
          parameter groups
        lr (float, optional): learning rate (default: 1e-4)
        betas (Tuple[float, float], optional): coefficients used for computing
          running averages of gradient and its square (default: (0.9, 0.99))
        weight_decay (float, optional): weight decay coefficient (default: 0)
        use_triton:
      """

      if not 0.0 <= lr:
          raise ValueError('Invalid learning rate: {}'.format(lr))
      if not 0.0 <= betas[0] < 1.0:
          raise ValueError('Invalid beta parameter at index 0: {}'.format(betas[0]))
      if not 0.0 <= betas[1] < 1.0:
          raise ValueError('Invalid beta parameter at index 1: {}'.format(betas[1]))
      defaults = dict(lr = lr, betas = betas, weight_decay = weight_decay)
      super().__init__(params, defaults)
      self.update_fn = update_fn

      if use_triton:
          self.update_fn = update_fn_triton if update_fn_triton is not None else update_fn

    @torch.no_grad()
    def step(self, closure = None):
        """
        Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        Returns:
            the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad, lr, wd, beta1, beta2, state = p.grad, group['lr'], group['weight_decay'], *group['betas'], self.state[p]

                # init state - exponential moving average of gradient values
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)

                exp_avg = state['exp_avg']

                self.update_fn(
                    p,
                    grad,
                    exp_avg,
                    lr,
                    wd,
                    beta1,
                    beta2
                )

        return loss