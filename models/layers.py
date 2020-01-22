import torch.nn as nn
import torch.nn.functional as F

# Works !!
class Conv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)
        default = {'stride':1,'padding':0,'dilation':1,'groups':1}
        self.kwargs = {key: kwargs[key] if key in kwargs.keys() else default[key] for key in default.keys()}

    def forward(self,x,state_dict=None,prefix=None):
        if state_dict is None:
            return super().forward(x)
        else:
            try:
                return F.conv2d(x,state_dict[prefix+'.weight'],state_dict[prefix+'.bias'],**self.kwargs)
            except KeyError:
                # No bias
                return F.conv2d(x,state_dict[prefix+'.weight'],**self.kwargs)

class BatchNorm2d(nn.BatchNorm2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)
        self.kwargs = kwargs

    def forward(self,x,state_dict=None,prefix=None):
        if state_dict is None:
            return super().forward(x)
        else:
            return F.batch_norm(x,
                state_dict[prefix+'.running_mean'],
                state_dict[prefix+'.running_var'],
                state_dict[prefix+'.weight'],
                state_dict[prefix+'.bias'],
                **self.kwargs)

class Linear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)

    def forward(self,x,state_dict=None,prefix=None):
        if state_dict is None:
            return super().forward(x)
        else:
            try:
                return F.linear(x,state_dict[prefix+'.weight'],state_dict[prefix+'.bias'])
            except KeyError:
                return F.linear(x,state_dict[prefix+'.weight'])

class Sequential(nn.Sequential):
    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)

    def forward(self,x,state_dict=None,prefix=None):
        if state_dict is None:
            return super().forward(x)
        else:
            for i,module in enumerate(self):
                try:
                    x = module(x,state_dict,prefix=f'{prefix}.{i}')
                except:
                    # For when the prefix is empty, i.e. Sequential object is main model
                    x = module(x,state_dict,prefix=f'{i}')
            return x

