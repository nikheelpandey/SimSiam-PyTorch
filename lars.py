import torch 
import torchvision
from torch.optim.optimizer import Optimizer 
import torch.nn as nn 

# paper : https://paperswithcode.com/method/lars 
# source: https://github.com/nikheelpandey/TAUP-PyTorch/blob/master/lars.py



class LARS(Optimizer):
    def __init__(self, 
                 named_modules, 
                 lr,
                 momentum=0.9,
                 trust_coef=1e-3,
                 weight_decay=1.5e-6,
                exclude_bias_from_adaption=True):

        defaults = dict(momentum=momentum,
                lr=lr,
                weight_decay=weight_decay,
                 trust_coef=trust_coef)
        parameters = self.exclude_from_model(named_modules, exclude_bias_from_adaption)
        super(LARS, self).__init__(parameters, defaults)

    def _use_weight_decay(self, group):
        return False if group['name'] == 'exclude' else True
    def _do_layer_adaptation(self, group):
        return False if group['name'] == 'exclude' else True


    def exclude_from_model(self, named_modules, exclude_bias_from_adaption=True):
        base = [] 
        exclude = []
        for name, module in named_modules:
            if type(module) in [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]:
                for name2, param in module.named_parameters():
                    exclude.append(param)
            else:
                for name2, param in module.named_parameters():
                    if name2 == 'bias':
                        exclude.append(param)
                    elif name2 == 'weight':
                        base.append(param)
                    else:
                        pass # non leaf modules 
        return [{
            'name': 'base',
            'params': base
            },{
            'name': 'exclude',
            'params': exclude
        }] if exclude_bias_from_adaption == True else [{
            'name': 'base',
            'params': base+exclude 
        }]



    @torch.no_grad() 
    def step(self):
        for group in self.param_groups: 
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            lr = group['lr']
            trust_coef = group['trust_coef']

            for p in group['params']:
                if p.grad is None:
                    continue
                global_lr = lr
                velocity = self.state[p].get('velocity', 0)  

                if self._use_weight_decay(group):
                    p.grad.data += weight_decay * p.data 

                trust_ratio = 1.0 
                if self._do_layer_adaptation(group):
                    w_norm = torch.norm(p.data, p=2)
                    g_norm = torch.norm(p.grad.data, p=2)
                    trust_ratio = trust_coef * w_norm / g_norm if w_norm > 0 and g_norm > 0 else 1.0 
                scaled_lr = global_lr * trust_ratio 
                next_v = momentum * velocity + scaled_lr * p.grad.data 
                update = next_v
                p.data = p.data - update