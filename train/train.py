# from fastai.basic_data import *
from fastai.vision import *
from fastai.callbacks import *
from collections import OrderedDict
from fastai.vision.learner import cnn_config

from MetaAI.data import MetaDataBunch
from MetaAI.models import Net
from MetaAI.train.meta_sgd import *
from MetaAI.train.reptile import *


fit_fns = {'meta_sgd':meta_sgd_fit,'reptile':reptile_fit,'maml':None}
# Make a functional model
# Wrap that model in a MetaModel
# Make a MetaLearner from MetaModel

class MetaLearner(Learner):
    # def __init__(self, data, model, **kwargs):
    #     super(MetaLearner, self).__init__(data.train_tasks[0], model, **kwargs)
    
    @classmethod
    def from_model(cls, data, model, mode=None, **kwargs):
        model = MetaModel(model)
        if mode not in fit_fns.keys(): raise NotImplementedError
        if mode == 'meta_sgd': model.init_lr_params()
        learn = Learner(data.train_tasks[0], model, **kwargs)
        learn.meta_databunch = data
        model_params = list(model.parameters()) + list(model.task_lr.values()) if mode=='meta_sgd' else list(model.parameters())
        learn.opt = torch.optim.Adam(model_params)
        Learner.meta_fit = partial(meta_fit, fit_fn = fit_fns[mode])
        return learn

    "Not yet implemented"
    @classmethod
    def from_arch(cls,data:MetaDataBunch, base_arch:Callable, cut:Union[int,Callable]=None, pretrained:bool=True,
                lin_ftrs:Optional[Collection[int]]=None, ps:Floats=0.5, custom_head:Optional[nn.Module]=None,
                split_on:Optional[SplitFuncOrIdxList]=None, bn_final:bool=False, init=nn.init.kaiming_normal_,
                concat_pool:bool=True, **kwargs:Any)->Learner:
        # meta = cnn_config(base_arch)
        model= MetaModel(create_cnn_model(base_arch,data.train_tasks[0].c, cut, pretrained, lin_ftrs, ps=ps, custom_head=custom_head,
            bn_final=bn_final, concat_pool=concat_pool))
        model.init_lr_params()
        learn = Learner(data.train_tasks[0], model, **kwargs)
        # learn.split(split_on or meta['split'])
        # if pretrained: learn.freeze()
        # if init: apply_init(model[1], init)
        learn.meta_databunch = data
        model_params = list(model.parameters()) + list(model.task_lr.values())
        learn.opt = torch.optim.Adam(model_params)
        Learner.meta_fit = partial(meta_fit, fit_fn = fit_fns[mode])
        return learn

class MetaModel(nn.Module):
    def __init__(self, model=Net(3,5)):
        super(MetaModel, self).__init__()
        self.meta_learner = model

    def forward(self, X, adapted_params=None):
        if adapted_params == None:
            # self.load_state_dict(self.initial_weights)
            out = self.meta_learner(X)
        else:
            # self.load_state_dict(adapted_params)
            out = self.meta_learner(X,adapted_params)
        return out

    def cloned_state_dict(self):
        """
        Only returns state_dict of meta_learner (not task_lr)
        """
        cloned_state_dict = {
            key: val.clone()
            for key, val in self.state_dict().items()
        }
        return cloned_state_dict

    def init_lr_params(self):
        self.task_lr = OrderedDict()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        for key, val in self.named_parameters():
            # self.task_lr[key] = 1e-3 * torch.ones_like(val, requires_grad=True)
            self.task_lr[key] = nn.Parameter(
                1e-3 * torch.ones_like(val, requires_grad=True, device=device))
            # if torch.cuda.is_available():
            #     self.task_lr[key] = self.task_lr[key].cuda()
            # self.initial_weights = self.cloned_state_dict()

def meta_fit(self,fit_fn,epochs:int, lr:Union[Floats,slice]=defaults.lr,
    wd:Floats=None, callbacks:Collection[Callback]=None,**kwargs)->None:
    "Fit the model on this learner with `lr` learning rate, `wd` weight decay for `epochs` with `callbacks`."
    lr = self.lr_range(lr)
    if wd is None: wd = self.wd
    if not isinstance(self.opt, OptimWrapper):    
        self.opt = OptimWrapper(self.opt,lr,wd)
    callbacks = [cb(self) for cb in self.callback_fns + listify(defaults.extra_callback_fns)] + listify(callbacks)
    self.cb_fns_registered = True
    fit_fn(epochs, self, metrics=self.metrics, callbacks=self.callbacks+callbacks,**kwargs) 


