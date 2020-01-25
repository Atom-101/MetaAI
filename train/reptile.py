from fastai.basic_data import *
from fastai.vision import *
from fastai.callbacks import *
from collections import OrderedDict
from fastai.vision.learner import cnn_config

from MetaAI.data import MetaDataBunch


class ReptileTrainUtils():
    "Encapsulates methods needed to train a learner using Reptile algorithm"
    @classmethod
    def train_single_task(cls, learn, data, inner_lr, k, cb_handler,**kwargs):
        # set model to training mode
        learn.model.train()
        # learn.model.cuda()
        # learn.opt.zero_grad_()
        cb_handler.set_dl(data.train_dl)
        original_state_dict = learn.model.cloned_state_dict()
        optim = torch.optim.SGD(learn.model.parameters(),lr=inner_lr)
        # adapted_params = OrderedDict()
        for _ in range(k):
            for xb,yb in data.train_dl:
                xb,yb = cb_handler.on_batch_begin(xb,yb)
                ypred = learn.model(xb)
                loss = learn.loss_func(ypred,yb)
                loss.backward()
                optim.step()
                optim.zero_grad()       
            # for key, val in learn.model.named_parameters():
            #     adapted_params[key] = val - inner_lr * val.grad
            #     adapted_state_dict[key] = adapted_params[key]
        
        # adapted_state_dict = learn.model.cloned_state_dict()
        # learn.model.load_state_dict(original_state_dict)
        # gc.collect()
        # return loss,adapted_state_dict
        return loss,original_state_dict
    
    @classmethod
    def meta_update_batch(cls, learn, original_state_dict,epsilon,cb_handler):
        "Evaluate model trained on a task and return loss"        
        # for key, val in learn.model.named_parameters():
        #     val.grad.data.zero_()
        #     val.grad.data = epsilon * (val - adapted_state_dict[key])        
        # if learn.opt is not None:
        #     if not cb_handler.on_backward_end(): learn.opt.step()
        #     if not cb_handler.on_step_end():     learn.opt.zero_grad()
        trained_state_dict = learn.model.state_dict()
        learn.model.load_state_dict(
            {
                name:
                original_state_dict[name]+epsilon*(trained_state_dict[name]-original_state_dict[name])
                for name in original_state_dict
            }
        )

    @classmethod
    def evaluate_trained_model(cls,learn,task):
        learn.model.eval()
        for xb,yb in task.valid_dl:
            ypred = learn.model(xb)
            loss = learn.loss_func(ypred,yb)
        acc = (ypred.argmax(1).detach().cpu() == yb.cpu()).numpy().sum()/yb.shape[0]
        return loss,acc

def reptile_fit(epochs:int, learn:Learner, callbacks:Optional[CallbackList]=None, metrics:OptMetrics=None,k=5,inner_lr=1e-3,epsilon=1e-3)->None:
    cb_handler = CallbackHandler(callbacks, metrics)
    pbar = master_bar(range(epochs))
    cb_handler.on_train_begin(epochs,pbar=pbar,metrics=metrics)
    # cb_handler.set_dl(learn.data.train_dl)

    exception=False
    try:
        for epoch in pbar:    
            cb_handler.on_epoch_begin()
            for task in progress_bar(learn.meta_databunch.train_tasks,parent=pbar):
                loss,original_state_dict = ReptileTrainUtils.train_single_task(learn,task,inner_lr,k,cb_handler)
                cb_handler.on_backward_begin(loss)
                ReptileTrainUtils.meta_update_batch(learn,original_state_dict,epsilon,cb_handler)
                cb_handler.on_batch_end(loss)
            # accuracy = []
            if not cb_handler.skip_validate:
                val_loss,accuracies = reptile_validate(learn, cb_handler=cb_handler,pbar=pbar,k=k,inner_lr=inner_lr)
            else:
                val_loss = None
            if cb_handler.on_epoch_end(val_loss): break
            print(np.mean(np.array(accuracies)))
    except Exception as e:
        exception = e
        raise
    finally: cb_handler.on_train_end(exception) 

def reptile_validate(learn,average=True,cb_handler=None,pbar=None,k=5,inner_lr=1e-3):
    val_losses,accuracies=[],[]
    for task in progress_bar(learn.meta_databunch.valid_tasks,parent=pbar):
        query_train_loss,original_state_dict= ReptileTrainUtils.train_single_task(learn,task,inner_lr,k,cb_handler)
        query_val_loss,acc = ReptileTrainUtils.evaluate_trained_model(learn,task)
        val_losses.append(query_val_loss)
        accuracies.append(acc)
        learn.model.load_state_dict(original_state_dict)
        if cb_handler and cb_handler.on_batch_end(val_losses[-1]): break
    if average: return to_np(torch.stack(val_losses)).mean(),accuracies
    else: return val_losses,accuracies