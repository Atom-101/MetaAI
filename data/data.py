from fastai.basic_data import *
from fastai.vision import *
import pandas as pd

class MetaDataBunch():
    "Each MetaDataBunch has a list of train and validation tasks. Each task is an ImageDataBunch with fixed number of train examples."
    #changes made to basic_data.py line 118-119
    def __init__(self,path:PathOrStr, df:pd.DataFrame, folder:PathOrStr=None, label_delim:str=None, valid_pct:float=0.2,
                seed:int=None, fn_col:IntsOrStrs=0, label_col:IntsOrStrs=1, suffix:str='', shots=5, ways=20,**kwargs:Any)->'MetaDataBunch':
        
        if seed:
            np.random.seed(seed)
        sampling = np.arange(df[label_col].nunique())
        np.random.shuffle(sampling)
        df[label_col] = pd.factorize(df[label_col],sort=True)[0]

        def chunk(l,n):
            return [l[i:i+n] for i in range(0, len(l)-len(l)%n, n)]
        

        def split_meta(self,shots,col):
            "Split the items giving the train_dl fixed number of examples of each class, while remaining go to valid_dl"
            df = self.inner_df.set_index(self.inner_df.columns[col]).reset_index()
            train_df = pd.concat([g.sample(shots,replace=False) for _,g in df.groupby(label_col)])
            train_idx = train_df.index.values
            valid_idx = np.setdiff1d(arange_of(self.items),train_idx)
            return self.split_by_idxs(train_idx,valid_idx)
        ImageList.split_meta = split_meta

        def produce_databunch(df):
            "Make a databunch from df using split_meta"
            db = ImageList.from_df(df,path=path,folder=folder,suffix=suffix, cols=fn_col)
            db = (db.split_meta(shots,fn_col)
                    .label_from_df(label_delim=label_delim, cols=label_col))
            db = ImageDataBunch.create_from_ll(db,**kwargs).normalize(imagenet_stats)
            return db


        sampling = chunk(sampling,ways)
        df_list = [df[df[label_col].isin(l)] for l in sampling]   
        
        meta_train_dfs = df_list[:-int(valid_pct*len(df_list))]
        meta_valid_dfs = df_list[-int(valid_pct*len(df_list)):]

        
        self.train_tasks = [produce_databunch(d) for d in meta_train_dfs]
        self.valid_tasks = [produce_databunch(d) for d in meta_valid_dfs]
        # self.path = self.train_tasks[0].path
        # self.device = self.train_tasks[0].device
        # self.loss_func = self.train_tasks[0].loss_func


