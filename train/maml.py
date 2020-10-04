from fastai.basic_data import *
from fastai.vision import *
from fastai.callbacks import *
from collections import OrderedDict
from fastai.vision.learner import cnn_config

from MetaAI.train.utils import tensor_splitter
from MetaAI.data import MetaDataBunch

class MamlTrainUtils():
    "Encapsulates methods needed to train a learner using MetaSGD algorithm"
    @classmethod
    def train_single_task(cls, learn, data, inner_lr, cb_handler, ind=None, train=True):
        # set model to training mode
        learn.model.train()
        grads = None
        loss = 0
        if cb_handler: cb_handler.set_dl(data.train_dl)
        adapted_state_dict = learn.model.cloned_state_dict()
        def zero_grad(params):
            for p in params:
                if p.grad is not None:
                    p.grad.zero_()
        zero_grad(learn.model.parameters()) 
        for i,(xb,yb) in enumerate(data.train_dl):
            idx=ind if ind else tensor_splitter(yb,learn.ways,learn.shots,train=train)
            xb,yb = xb[idx],yb[idx]
            if cb_handler: xb,yb = cb_handler.on_batch_begin(xb,yb)
            for _ in range(5):
                ypred = learn.model(xb)
                loss = learn.loss_func(ypred,yb.cuda())/xb.shape[0]
                grads = torch.autograd.grad(loss, adapted_state_dict.values(),create_graph=learn.mode=='fomaml')
                for (key, val), grad in zip(learn.model.named_parameters(), grads):
                    adapted_state_dict[key] = val - inner_lr * grad
        return adapted_state_dict,idx
    
    @classmethod
    def meta_update_batch(cls, learn, eval_bundle, cb_handler):
        "Evaluate model trained on a task and return loss"
        meta_loss = 0.0
        learn.model.train()
        for trained_dict,val_dl,idx in eval_bundle:
            loss_task = 0.0
            cb_handler.set_dl(val_dl)
            for xb,yb in val_dl:
                idx = [i for i in range(xb.shape[0]) if i not in idx]
                xb,yb = xb[idx],yb[idx]
                if cb_handler: xb,yb = cb_handler.on_batch_begin(xb.contiguous(),yb.contiguous())
                y_pred = learn.model(xb,trained_dict,'meta_learner')
                loss_task += learn.loss_func(y_pred,yb.cuda())
            loss_task /= len(idx)
            meta_loss += loss_task
        meta_loss /= len(eval_bundle)
        
        if learn.opt is not None:
            meta_loss,skip_bwd = cb_handler.on_backward_begin(meta_loss)
            if not skip_bwd:                     meta_loss.backward()
            if not cb_handler.on_backward_end(): learn.opt.step()
            if not cb_handler.on_step_end():     learn.opt.zero_grad()
        return meta_loss
    
    @classmethod
    def meta_validate_batch(cls, learn, eval_bundle, cb_handler):
        meta_loss = 0.0
        acc = 0.0
        learn.model.eval()
        with torch.no_grad():
            for trained_dict,val_dl,idx in eval_bundle:
                loss_task,task_acc = 0.,0.
                if cb_handler: cb_handler.set_dl(val_dl)
                for xb,yb in val_dl:
                    idx = [i for i in range(xb.shape[0]) if i not in idx]
                    xb,yb = xb[idx],yb[idx]
                    if cb_handler: xb,yb = cb_handler.on_batch_begin(xb.contiguous(),yb.contiguous())
                    y_pred = learn.model(xb,trained_dict,'meta_learner')
                    task_acc += (y_pred.argmax(1).detach().cpu() == yb.cpu()).numpy().sum()/yb.shape[0]
                    loss_task += learn.loss_func(y_pred,yb)
                loss_task /= len(val_dl.items)
                meta_loss += loss_task
                acc += task_acc 
            meta_loss /= len(eval_bundle)
            acc /= len(eval_bundle)
        return meta_loss,acc

def maml_fit(epochs:int,learn:Learner,support_train_lr:float=1e-2,callbacks:Optional[CallbackList]=None,metrics:OptMetrics=None,
            outer_batch_size:int=5)->None:
    cb_handler = CallbackHandler(callbacks, metrics)
    pbar = master_bar(range(epochs))
    cb_handler.on_train_begin(epochs,pbar=pbar,metrics=metrics)
    # max_acc = 0
    # cb_handler.set_dl(learn.data.train_dl)

    exception=False
    try:
        for epoch in pbar:    
            cb_handler.on_epoch_begin()
            meta_train_bundle = []
            # Training
            for i,task in enumerate(progress_bar(learn.meta_databunch.train_tasks,parent=pbar)):
                trained_state_dict,idx = MamlTrainUtils.train_single_task(learn,task,1e-3,cb_handler,train=True)
                meta_train_bundle.append((trained_state_dict,task.trtain_dl,idx))
                if i%outer_batch_size == outer_batch_size-1 or i == len(learn.meta_databunch.train_tasks)-1:
                    loss = MamlTrainUtils.meta_update_batch(learn,meta_train_bundle,cb_handler)
                    meta_train_bundle = []
                    gc.collect()
                    cb_handler.on_batch_end(loss)           
            # Validation         
            if not cb_handler.skip_validate:
                val_losses,accuracy = maml_validate(learn,outer_batch_size,support_train_lr,cb_handler,pbar)
            val_loss = np.stack(val_losses).mean()
            if cb_handler.on_epoch_end(val_loss): break
            print(np.array(accuracy).mean())
    except Exception as e:
        exception = e
        raise
    finally: cb_handler.on_train_end(exception)

def maml_validate(learn,outer_batch_size=5,support_train_lr=1e-2,cb_handler=None,pbar=None):
    meta_eval_bundle = []
    val_losses,accuracy = [],[]
    for i,task in enumerate(progress_bar(learn.meta_databunch.valid_tasks,parent=pbar)):
        trained_state_dict,idx = MamlTrainUtils.train_single_task(learn,task,1e-3,cb_handler,train=False)
        meta_eval_bundle.append((trained_state_dict,task.train_dl,idx))
        if i%outer_batch_size == outer_batch_size-1 or i == len(learn.meta_databunch.valid_tasks)-1:
            val_loss,acc = MamlTrainUtils.meta_validate_batch(learn,meta_eval_bundle,cb_handler)            
            val_losses.append(val_loss)
            accuracy.append(np.array(acc))
            meta_eval_bundle = []
            if cb_handler and cb_handler.on_batch_end(val_losses[-1]): break
    return val_losses,accuracy
