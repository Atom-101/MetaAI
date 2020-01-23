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
