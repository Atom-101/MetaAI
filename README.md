# Meta.AI
This is a deep learning library specialised for meta-learning. It uses fastai(v1) and Pytorch. Currently the meta learning DataBunch only supports from_df() functionality. 

# Demo
```python
from MetaAI.models import *
from MetaAI.train import *
from MetaAI.data import *

from fastai.vision import *
from fastai.callbacks import *
import pandas as pd

data = MetaDataBunch(path='../Omniglot/Data/images_background',
                     df=pd.read_csv('../Omniglot/Omniglot.csv'),
                     label_col='class',
                     bs=5,
                     val_bs=95,
                     size=32,
                     ways=5,
                     shots=1)

model = resnet.resnet18()
model.fc = layers.Linear(512,5)
learn = MetaLearner.from_model(data=data,
                               model=model,
                               mode='meta_sgd',
                               loss_func=nn.CrossEntropyLoss(reduction='sum'),
                               callback_fns=[ShowGraph,
                                             partial(ReduceLROnPlateauCallback,patience=3,factor=0.1,min_delta=5e-3)
                                            ])
                                            
learn.meta_fit(epochs=20,lr=1e-3,outer_batch_size=16)                                            
```
## Results 
Task performed was Omniglot 5-way 1-shot. All models were trained for 20 epochs with meta lr = 1e-3. All models used 3 channel 32x32 images as input.

|Model|Meta-SGD|MAML|
|---|---|---|
|Default Net|94.2|79.5|
|Resnet18|71.9|51.9|
|Resnet18 (pretrained)|53.4|26.1

*MAML has a low accuracy because it needs more epochs to converge. It has a slower convergence speed than Meta-SGD.
