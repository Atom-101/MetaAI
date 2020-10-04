from fastai.basic_data import *
from fastai.vision import *
from fastai.callbacks import *
from collections import OrderedDict
from fastai.vision.learner import cnn_config

from MetaAI.train.utils import tensor_splitter
from MetaAI.data import MetaDataBunch


class ReptileTrainUtils():
    "Encapsulates methods needed to train a learner using Reptile algorithm"
    @classmethod
    def train_single_task(cls, learn, xb, yb, inner_lr, k, cb_handler,**kwargs):
        # set model to training mode
        learn.model.train()
        cb_handler.set_dl(data.train_dl)
        original_state_dict = learn.model.cloned_state_dict()
        optim = torch.optim.SGD(learn.model.parameters(),lr=inner_lr)
        if cb_handler: xb,yb = cb_handler.on_batch_begin(xb,yb)
        for _ in range(k):
            ypred = learn.model(xb)
            loss = learn.loss_func(ypred,yb.cuda())/xb.shape[0]
            loss.backward()
            optim.step()
            optim.zero_grad()
        adapted_state_dict = learn.model.cloned_state_dict()
        learn.model.load_state_dict(original_state_dict)
        return loss,adapted_state_dict
    
    @classmethod
    def meta_update_batch(cls, learn, adapted_state_dict,epsilon,cb_handler):
        for k,param in learn.model.named_parameters():
            if param.grad is None:
                param.grad = torch.zeros(param.size(),requires_grad=True).cuda()
            param.grad.data.zero_()
            param.grad.data.add_(param.data - adapted_state_dict[k].data)
        learn.opt.step()
        learn.opt.zero_grad()
        
    @classmethod
    def meta_validate_batch(cls,learn,xb,yb,adapted_state_dict):
        learn.model.eval()
        orig_state_dict = learn.model.cloned_state_dict()
        learn.model.load_state_dict(adapted_state_dict)
        yb = yb.cuda()
        y_pred = learn.model(xb)
        loss = learn.loss_func(ypred,yb)
        learn.model.load_state_dict(orig_state_dict)
        del(adapted_state_dict)
        gc.collect()
        acc = (y_pred.argmax(1).detach().cpu() == yb.cpu()).numpy().sum()/yb.shape[0]
        return loss,acc

def average_dicts(av_dict,ad_dict,num_acc):
    if av_dict is None:
        av_dict = ad_dict.copy()
        for k in ad_dict.keys():
            av_dict[k] = av_dict[k]/num_acc
        return av_dict
    else:
        for k in ad_dict.keys():
            av_dict[k] = av_dict[k] + ad_dict[k]/num_acc
        return av_dict

def reptile_fit(epochs:int, learn:Learner, callbacks:Optional[CallbackList]=None, metrics:OptMetrics=None,
                k=5,inner_lr=1e-3,epsilon=1e-3,num_acc=8)->None:
    cb_handler = CallbackHandler(callbacks, metrics)
    pbar = master_bar(range(epochs))
    cb_handler.on_train_begin(epochs,pbar=pbar,metrics=metrics)
    # max_acc = 0
    # cb_handler.set_dl(learn.data.train_dl)

    exception=False
    try:
        for epoch in pbar:    
            cb_handler.on_epoch_begin()
            cb_handler.set_dl(learn.meta_databunch.train_tasks)
            orig = learn.model.cloned_state_dict()
            avg_state_dict = None
            for i,(xb,yb) in enumerate(progress_bar(learn.meta_databunch.train_tasks.train_dl,parent=pbar)):
                loss,adapted_state_dict = ReptileTrainUtils.train_single_task(learn,xb,yb,inner_lr,k,cb_handler)
                cb_handler.on_backward_begin(loss)
                avg_state_dict = average_dicts(avg_state_dict,adapted_state_dict,num_acc)
                if i%num_acc==num_acc-1 or i==len(learn.meta_databunch.train_tasks)-1:
                    ReptileTrainUtils.meta_update_batch(learn,avg_state_dict,epsilon,cb_handler)
                    del(avg_state_dict)
                    avg_state_dict = None
                cb_handler.on_batch_end(loss)
                del(adapted_state_dict)
                gc.collect()    
            if not cb_handler.skip_validate:
                val_loss,accuracies = reptile_validate(learn, cb_handler=cb_handler,pbar=pbar,k=k,inner_lr=inner_lr)
            else:
                val_loss = None
            if cb_handler.on_epoch_end(val_loss): break
            print(np.array(accuracies).mean())
    except Exception as e:
        exception = e
        raise
    finally: cb_handler.on_train_end(exception) 

def reptile_validate(learn,average=True,cb_handler=None,pbar=None,k=5,inner_lr=1e-3,ind=None):
    val_losses,accuracies=[],[]
    for task in progress_bar(learn.meta_databunch.valid_tasks,parent=pbar):
        xb,yb = list(task.train_dl)[0]
        idx=ind if ind else tensor_splitter(yb,learn.ways,learn.shots,train=False)
        loss,adapted_state_dict= ReptileTrainUtils.train_single_task(learn,xb[ind],yb[ind],inner_lr,k,cb_handler)
        idx = [i for i in range(xb.shape[0]) if i not in idx]
        query_val_loss,acc = ReptileTrainUtils.meta_validate_batch(learn,xb[idx],yb[idx],adapted_state_dict)    
        val_losses.append(query_val_loss.detach().cpu().numpy())
        accuracies.append(np.array(acc))
        learn.model.load_state_dict(original_state_dict)
        if cb_handler and cb_handler.on_batch_end(val_losses[-1]): break
    if average: return np.stack(val_losses).mean(),accuracies
    else: return val_losses,accuracies
