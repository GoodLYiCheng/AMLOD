import torch
from torch.optim.optimizer import Optimizer
import math

class DiagOCP(Optimizer):
    """
    DiagOCP (Diagonal Optimal Conjugate Projection) 通用 PyTorch 优化器插件。
    该优化器通过 Hutchinson 方法估计 Hessian 矩阵对角线信息，实现自适应二阶优化。
    
    参数 (基于 AMLOD 基础配置):
        params (iterable): 模型参数迭代器。
        lr (float): 学习率 (默认: 0.05，对应 OCP M=alpha*I)。
        betas (tuple): (beta1, beta2) (默认: (0.9, 0.999))。
        eps (float): 对角正则化项 (默认: 1e-8)。
        weight_decay (float): 权重衰减系数 (默认: 0.0005)。
        update_each (int): 每隔多少个 step 更新一次 Hessian (默认: 4)。
        n_samples (int): Hutchinson 采样次数 (默认: 1)。
        hessian_clip (float): Hessian 对角线下界 (默认: 1e-4)。
        hessian_max (float): Hessian 对角线上界 (默认: 1e2)。
        delta_clip (float): 参数更新步长裁剪 (默认: 10.0)。
        gamma_max (float): 二阶加速比率上限 (默认: 100.0)。
    """
    def __init__(
        self,
        params,
        lr=0.05,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0005,
        update_each=4,
        n_samples=1,
        hessian_clip=1e-4,
        hessian_max=1e2,
        delta_clip=10.0,
        gamma_max=100.0,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1 parameter: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2 parameter: {betas[1]}")
        
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            update_each=update_each,
            n_samples=n_samples,
            hessian_clip=hessian_clip,
            hessian_max=hessian_max,
            delta_clip=delta_clip,
            gamma_max=gamma_max,
        )
        super(DiagOCP, self).__init__(params, defaults)
        self._generators = {}

    def _get_generator(self, device):
        """为特定设备获取或创建随机数生成器。"""
        if device not in self._generators:
            self._generators[device] = torch.Generator(device).manual_seed(2147483647)
        return self._generators[device]

    def need_hessian_update(self):
        """判断当前 step 是否需要计算 Hessian。"""
        group0 = self.param_groups[0]
        update_each = group0.get("update_each", 4)
        
        current_step = 0
        for group in self.param_groups:
            for p in group['params']:
                if p in self.state and 'step' in self.state[p]:
                    current_step = self.state[p]['step']
                    break
            if current_step > 0: break
        
        next_step = current_step + 1
        return (next_step == 1) or (next_step % update_each == 0)

    def _hutchinson_hessian_diag(self, params, grads, n_samples=1):
        """Hutchinson 方法估计 Hessian 对角线。"""
        hessian_diag = [torch.zeros_like(p) for p in params]

        for _ in range(n_samples):
            zs = [
                torch.randint(0, 2, p.size(), device=p.device, dtype=p.dtype, 
                              generator=self._get_generator(p.device)) * 2.0 - 1.0
                for p in params
            ]
            
            try:
                h_zs = torch.autograd.grad(
                    grads, params, grad_outputs=zs,
                    retain_graph=True, only_inputs=True, allow_unused=True
                )
            except RuntimeError:
                h_zs = [g * z if g is not None else None for g, z in zip(grads, zs)]

            for h, h_z, z in zip(hessian_diag, h_zs, zs):
                if h_z is not None:
                    hv = h_z * z
                    hv = torch.where(torch.isfinite(hv), hv, torch.zeros_like(hv))
                    h.add_(hv.abs()) 
        
        return [h / n_samples for h in hessian_diag]

    @torch.no_grad()
    def step(self, closure=None):
        """执行一步参数更新。"""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        params_with_grad = []
        grads = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    params_with_grad.append(p)
                    grads.append(p.grad)

        if not params_with_grad:
            return loss

        first_p = params_with_grad[0]
        state0 = self.state[first_p]
        if "step" not in state0:
            state0["step"] = 0
            
        current_step_cnt = state0["step"] + 1
        update_each = self.param_groups[0].get("update_each", 4)
        compute_hessian = (current_step_cnt == 1) or (current_step_cnt % update_each == 0)

        param_to_hess = {}
        if compute_hessian:
            with torch.enable_grad():
                hess_values = self._hutchinson_hessian_diag(params_with_grad, grads, 
                                                           self.param_groups[0].get("n_samples", 1))
            for p, h in zip(params_with_grad, hess_values):
                param_to_hess[id(p)] = h

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            wd = group.get("weight_decay", 0.0)
            eps = group.get("eps", 1e-8)
            h_clip = group.get("hessian_clip", 1e-4)
            h_max = group.get("hessian_max", 1e2)
            d_clip = group.get("delta_clip", 10.0)
            g_max = group.get("gamma_max", 100.0)

            for p in group["params"]:
                if p.grad is None: continue
                
                state = self.state[p]
                if "exp_avg" not in state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_hess_diag"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]
                exp_hess = state["exp_hess_diag"]
                state["step"] += 1
                step = state["step"]

                if id(p) in param_to_hess:
                    h = param_to_hess[id(p)]
                    h = torch.clamp(h, min=h_clip, max=h_max)
                    exp_hess.mul_(beta2).add_(h, alpha=1 - beta2)
                
                exp_avg.mul_(beta1).add_(p.grad, alpha=1 - beta1)

                bias_corr1 = 1 - beta1 ** step
                bias_corr2 = 1 - beta2 ** step
                m_hat = exp_avg / bias_corr1
                d_hat = exp_hess / bias_corr2

                denom = d_hat + eps  
                gamma = 1.0 / denom
                gamma = torch.clamp(gamma, max=g_max)

                delta = lr * m_hat * gamma
                delta = torch.clamp(delta, -d_clip, d_clip)

                if wd != 0:
                    p.mul_(1 - lr * wd)

                p.add_(delta, alpha=-1)

        return loss
