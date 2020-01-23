from fastai.basic_data import *
from fastai.vision import *
from fastai.callbacks import *
from collections import OrderedDict
from fastai.vision.learner import cnn_config

from MetaAI.data import MetaDataBunch

class MetaSGDTrainUtils():
    "Encapsulates methods needed to train a learner using MetaSGD algorithm"
    @classmethod
    def train_single_task(cls, learn, data, cb_handler, **kwargs):
        # set model to training mode
        learn.model.train()
        # learn.model.cuda()
        # forward pass all batches in task. and accumulate gradients(doesn't work with BatchNormalization)
        grads = None
        loss = 0
        cb_handler.set_dl(data.train_dl)
        for i,(xb,yb) in enumerate(data.train_dl):
            xb,yb = cb_handler.on_batch_begin(xb,yb)
            ypred = learn.model(xb)
            # raise ValueError()
            loss += learn.loss_func(ypred,yb)
        
        loss /= len(data.train_dl.items)
    
        # loss.backward(create_graph=True)
            # if not grads:    
            #     grads = torch.autograd.grad(loss, learn.model.parameters(), create_graph=True)
            # else:
            #     grads = [grads[i] + torch.autograd.grad(loss, learn.model.parameters(), retain_graph=True)[i] for i in range(len(grads))]
        def zero_grad(params):
            for p in params:
                if p.grad is not None:
                    p.grad.zero_()
        zero_grad(learn.model.parameters())        
        grads = torch.autograd.grad(loss, learn.model.parameters()) if kwargs['train']==False \
            else torch.autograd.grad(loss, learn.model.parameters(), create_graph=True)
        # perform manual gradient updates
        adapted_state_dict = learn.model.cloned_state_dict()
        adapted_params = OrderedDict()
        for (key, val), grad in zip(learn.model.named_parameters(), grads):
            task_lr = learn.model.task_lr[key]
            adapted_params[key] = val - task_lr * grad
            adapted_state_dict[key] = adapted_params[key]
        # for key, val in learn.model.named_parameters():
        #     # Also we only need single update of inner gradient update
        #     task_lr = learn.model.task_lr[key]
        #     adapted_params[key] = val - task_lr * val.grad
        #     adapted_state_dict[key] = adapted_params[key]

        return adapted_state_dict
    
    @classmethod
    def meta_update_batch(cls, learn, eval_bundle, cb_handler):
        "Evaluate model trained on a task and return loss"
        meta_loss = 0.0
        
        for trained_dict,val_dl in eval_bundle:
            loss_task = 0.0
            cb_handler.set_dl(val_dl)
            for xb,yb in val_dl:
                ids = np.random.shuffle(np.arange(yb.shape[0]))
                xb,yb = cb_handler.on_batch_begin(xb[ids,...].squeeze().contiguous(),yb[ids].squeeze().contiguous())
                y_pred = learn.model(xb,trained_dict,'meta_learner')
                loss_task += learn.loss_func(y_pred,yb)
            loss_task /= len(val_dl.items)
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
        with torch.no_grad():
            for trained_dict,val_dl in eval_bundle:
                loss_task = 0.0
                cb_handler.set_dl(val_dl)
                task_acc = 0
                for xb,yb in val_dl:
                    xb,yb = cb_handler.on_batch_begin(xb,yb)
                    y_pred = learn.model(xb,trained_dict,'meta_learner')
                    task_acc += (y_pred.argmax(-1)==yb).float().mean()
                    loss_task += learn.loss_func(y_pred,yb)
                loss_task /= len(val_dl.items)
                meta_loss += loss_task
                acc += task_acc 
            meta_loss /= len(eval_bundle)
            acc /= len(eval_bundle)
        return meta_loss,acc

def meta_sgd_fit(epochs:int, learn:Learner, callbacks:Optional[CallbackList]=None, metrics:OptMetrics=None,outer_batch_size=5)->None:
    cb_handler = CallbackHandler(callbacks, metrics)
    pbar = master_bar(range(epochs))
    cb_handler.on_train_begin(epochs,pbar=pbar,metrics=metrics)
    # cb_handler.set_dl(learn.data.train_dl)

    exception=False
    try:
        for epoch in pbar:    
            cb_handler.on_epoch_begin()
            meta_train_bundle = []
            # Training
            for i,task in enumerate(progress_bar(learn.meta_databunch.train_tasks,parent=pbar)):
                trained_state_dict = MetaSGDTrainUtils.train_single_task(learn,task,cb_handler,train=True)
                meta_train_bundle.append((trained_state_dict,task.valid_dl))
                if i%outer_batch_size == 1 or i == len(learn.meta_databunch.train_tasks)-1:
                    loss = MetaSGDTrainUtils.meta_update_batch(learn,meta_train_bundle,cb_handler)
                    meta_train_bundle = []
                    gc.collect()
                    cb_handler.on_batch_end(loss)
            
            # Validation
            
            if not cb_handler.skip_validate:
                val_losses,accuracy = meta_sgd_validate(learn,outer_batch_size,cb_handler,pbar)
            #     for i,task in enumerate(progress_bar(learn.meta_databunch.valid_tasks,parent=pbar)):
            #         trained_state_dict = MetaSGDTrainUtils.train_single_task(learn,task,cb_handler,train=False)
            #         meta_eval_bundle.append((trained_state_dict,task.valid_dl))
            #         if i%outer_batch_size == 1:
            #             val_loss,acc = MetaSGDTrainUtils.meta_validate_batch(learn,meta_eval_bundle,cb_handler)            
            #             val_losses.append(val_loss)
            #             accuracy.append(np.array(acc))
            #             if cb_handler and cb_handler.on_batch_end(val_losses[-1]): break
            
            val_loss = to_np(torch.stack(val_losses)).mean()
            if cb_handler.on_epoch_end(val_loss): break
            print(np.array(accuracy).mean())
    except Exception as e:
        exception = e
        raise
    finally: cb_handler.on_train_end(exception)

def meta_sgd_validate(learn,outer_batch_size=5,cb_handler=None,pbar=None):
    meta_eval_bundle = []
    val_losses,accuracy = [],[]
    for i,task in enumerate(progress_bar(learn.meta_databunch.valid_tasks,parent=pbar)):
        trained_state_dict = MetaSGDTrainUtils.train_single_task(learn,task,cb_handler,train=False)
        meta_eval_bundle.append((trained_state_dict,task.valid_dl))
        if i%outer_batch_size == 1 or i == len(learn.meta_databunch.valid_tasks)-1:
            val_loss,acc = MetaSGDTrainUtils.meta_validate_batch(learn,meta_eval_bundle,cb_handler)            
            val_losses.append(val_loss)
            accuracy.append(np.array(acc))
            if cb_handler and cb_handler.on_batch_end(val_losses[-1]): break
    return val_losses,accuracy