import importlib
import dill
import os
import copy
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
import random
import torch
import sys

if "src" not in os.getcwd():
    os.chdir("src/")

import dataloader
import train
import model as tcn_model
import utils
import pprint

        
class TCNBinaryClassifier:
    def __init__(self, tcn_layers, mlp_layers, epoch, kernel_size, 
                 batch_size=32, device='cpu', dropout_mlp=0.5, dropout_tcn=0.2, 
                 lr=0.0002, lr_decay_freq=5, model_dir=None, output_size=1, pad_after=True, 
                 patience=5, pos_weight=1, print_freq=200, seed=12345, weight_decay=0, 
                 default_embed_dim=10, vocabs=None, verbose=1, feature_embed_dims=None, 
                 warm_start=True, min_length=1, mlp_batch_norm=True, clip_grad=10.0):
        """ Initialize TCN binary classifier
        
        Parameters
        ----------
        tcn_layers: list or list of list
            Number of neurons at each TCN layer. Can be a list or a list of list

            e.g. [100, 100, 100, 100] if using single stack
                 [[100, 100], [100, 100]] if using stacked tcn

            TCN_v1 is built using single stack. Since each TemporalConvNet block
                comes with residual connections, along with fixed geometric dilution of
                2**i at layer  i. The advantage of using stacked TCN is to 
                    1) incorporate additional non-linearity to by stacking TemporalConvNet without 
                        residual connections in between and 
                    2) have more densely connected convolutional connections

        
        mlp_layers: list of int
            number of neurons for each hidden layers
        
        
        epoch: int
            number of epoches to train
        
        
        kernel_size: int or list of int
            kernel size, or list of kernel sizes for each stack of tcn.

            int if using single stack. e.g.  5
            list of int if using stacked tcn. e.g. [5, 7]

            If provided a list of kernel sizes, assert(len(kernel_size) == len(num_channels)
 
        
        batch_size: int, default = 32
            batch size
        
        
        device: str, {‘cpu’, ‘cuda:0’}, default = ‘cpu’
                Whether running on CPU or single GPU
        
        
        dropout_mlp: float in [0,1], default = 0.5
            dropout probability for MLP
            recommend for customer risk model: 0.5
        
        
        dropout_tcn: float in [0, 1], default = 0.2
            dropout probability for TCN
            recommend for customer risk model: 0.2
        
        
        lr: float, default = 0.0001
            learning rate
            The current implementation uses the Adam optimizer. 
            Please choose accordingly.
        
        
        lr_decay_freq: int, default = 5
            how many epoches before reducing learning rate by half.
            Set it to be larger than epoch to avoid learning rate decay.
        
        
        model_dir: str, default = None
            directory to save the model.
            Always provide this because this enables early stopping and logging.
        
        
        output_size: int, default = 1
            output dimension. For the current TCN binary classification implementation,
            always set to 1 since the criterion used is BCEwithLogitLoss, which for each
            data point, takes scalar output.
        
        
        pad_after: bool, default = True
            Whether to pad the data points in the front or the end. 

            Always pad_after when dealing with current TCN implementation, 
            but may be helpful to pad in the front when using 
                Sequential Variational Autoencoder
        
        
        patience: int, default = 1
            Number of validations to wait before early stopping
            
            Number of validation refers the number of time the model is validated,
            which means calling "train.valid_epoch". In this implementaion, the model
            validates once every <print_freq> number of batches and once at the 
            end of every epoch.
        
        
        pos_weight: int, default = 1
            weight for the positive class
            
        
        print_freq: int, default = 200
            frequency to print training progress.
            print every <print_freq> number of batches
            
        
        seed: int, default = 12345
            random seed
        
        
        weight_decay: int, default = 0
            weight_decay, L2 normalization. 
            
        
        default_embed_dim: int, default = 10
            default dimension for categorical features. However, try to avoid using this!
            always explicitly define embedding dimensions to avoid bugs
        
        
        vocabs: dict of dict, None-able, default = None
            If provided, must have keys = {'vocab_lens', 'w2i_dicts', 'i2w_dicts'}
            If provided, the current dataset loads the vocabularies 
                from the provided dictionary.
     
        
        verbose: float, default = 1
            verbosity of training.
            - 0: print virtually nothing.
            
            - 1: print parameters and performance every <print_freq> number of batches
            
            - 2: print detailed analysis such as precision-recall summary at each
                print_freq along with info listed above
        
        
        feature_embed_dims: dict
            Dictionary with key = categorical features and
                            value = dimension of the embedding
        
        
        warm_start: bool, default = True
            When calling fit, whether to train with the previously allocated model.
            
            - True: if the model is initialized, calling .fit will train that model.
            - Flase: regardless whether a has been intialized, initialize a new one.
            
        
        min_length: int, default = 1
            minimum sequence length of the data to be considered.
        
        
        mlp_batch_norm: bool, default = True
            Whether to use batch norm in MLP
        
        
        clip_grad: float, default = 10.0
            Gradient clipping
        
        
        Returns
        ----------
        None
        """
        self.batch_size = batch_size
        self.device = device
        self.dropout_tcn = dropout_tcn
        self.dropout_mlp = dropout_mlp
        self.epoch = epoch
        self.mlp_layers = mlp_layers
        self.kernel_size = kernel_size
        self.tcn_layers = tcn_layers
        self.lr = lr
        self.lr_decay_freq = lr_decay_freq
        self.model_dir = model_dir
        self.pad_after = pad_after 
        self.patience = patience
        self.print_freq = print_freq 
        self.weight_decay = weight_decay
        self.output_size = output_size
        self.pos_weight = pos_weight
        self.seed = seed
        self.default_embed_dim = default_embed_dim
        self.vocabs = vocabs
        self.feature_embed_dims = feature_embed_dims
        self.log_path = os.path.join(model_dir, 'log.txt')
        self.verbose = verbose
        self.warm_start = warm_start
        self.min_length = min_length
        self.mlp_batch_norm = mlp_batch_norm        
        self.clip_grad = clip_grad
         
             
    def df_to_list(self, df, sort_by=[], group_by=[], return_index=False, min_length=1):
        """ Convert pd.DataFrame to list of np.arrays
        
        Parameters
        ----------
        df: np.DataFrame
            the dataframe to be converted
            
        
        sort_by: list
            sort the df by the columns in sort_by to maintain ordering
        
        
        group_by: list
            group the df by the columns. Each array in the resulting
            list is a group in the form of np.array.
            
            e.g. in the Customer Risk Model, we group by borrower id
            for the transactions dataframe so each array contains the 
            past transactions of an individual borrower.
        
        
        return_index: bool, default = False
            For each datapoint, whether to return index along with the np.array
            
        
        min_length: int, default = 1
            minimum sequence length of the data to be considered.
        
        
        Returns
        ----------
        data: list of np.arrays
            contains the data in df
        """
        
        # helper function to extract data from dataframe after groupby
        def fn(df_, data_):    
            """
            The reason to convert to dictionary first and then to list
            is because when calling groupby, pandas creates an extra copy of 
            the first group to determine the data structure. This creates an
            additional copy for the first datapoint, which is not a good idea.
            """
            # extract the identifier from group_by
            uid = df_[group_by].iloc[0][group_by[0]] 
            if len(sort_by) > 0:
                df_tmp = df_.sort_values(sort_by)
            else:
                df_tmp = df_

            d = df_tmp.values

            # abandon the datum if it's length < min_length
            if len(d) < min_length:
                return None

            # add datum to the dictionary of data
            if return_index:
                i = df_tmp.index.to_list()
                data_[uid] = (d, i)
            else:
                data_[uid] = d
            return None

        data = {}
        if len(group_by) > 0:
            df.groupby(group_by).apply(lambda x: fn(x, data))
        else:
            raise NotImplementedError
            df.apply(lambda x: fn(x, data))

        if return_index:
            data, index = list(zip(*data.values()))
            return data, index

        data = list(data.values())
        return data
       
        
    def get_features(self, df, target=[], meta=[]):
        """ Given dataframe, obtain lists of features needed
        
        Parameters
        ----------
        df: pd.DataFrame
            input dataframe
            
        
        target: list of str
            target column. Almost always ['target']
        
        
        meta: list of str
            columns of meta-data
            
        
        Returns
        ----------
        selected_features: list of str
            columns the TCN Binary Classifier will use as features
        
        
        features_num: list of str
            numerical feature columns of the selected_features
            
        
        features_cat: list of str
            categorical featuer columns of the selected_features
            
        
        features_idx: dict
            col2i: dict
                column to index mapping for df
        
        """
        ############################################################
        #                   Type conversion
        ############################################################

        types = df[df.columns[~df.columns.isin(target+meta)]].dtypes
        for col_name, col_type in types.iteritems():
            if col_type == bool:
                df[col_name] = df[col_name].astype(float)

        ############################################################
        #                 Get features by type
        ############################################################

        features_cat = filter(lambda x: not np.issubdtype(x[1], np.number), types.iteritems())
        features_cat = sorted(list(map(lambda x: x[0], features_cat)))
        # target and meta should have already been removed. but just to be sure
        features_num = sorted(list(set(types.index) - set(features_cat) - set(target) - set(meta))) 
        selected_features = df.columns.to_list()
        features_idx = dict(zip(selected_features, range(len(selected_features))))

        return selected_features, features_num, features_cat, features_idx
    
    
    def process_data(self, X_train, target_col, X_valid=None, sort_by_col=[], group_by=[], meta=[], stnry=[]):
        """ Process input dataframes into model-specific datasets.
        
        Parameters
        ----------
        X_train: pd.DataFrame
            dataframe containing training data
            
        
        target_col: str
            column of the target
            
            
        X_valid: pd.DataFrame
            dataframe containing validation data
            
        
        sort_by_col: list
            sort the df by the columns in sort_by to maintain ordering
        
        
        group_by: list
            group the df by the columns. Each array in the resulting
            list is a group in the form of np.array.
            
            e.g. in the Customer Risk Model, we group by borrower id
            for the transactions dataframe so each array contains the 
            past transactions of an individual borrower.
            
            
        meta: list of str
            columns of meta-data
        
        
        stnry: list of str
            columns of the stnry features
        
        
        Returns
        ----------
        data: dict
            Processed data object for TCN. With the following fields:
            
            - train: list of np.arrays
                obtained by calling df_to_list on X_train.
                each element in the list is a np.array containing a sequence of data
                
            
            - test: list of np.arrays
                obtained by calling df_to_list on X_valid
                each element in the list is a np.array containing a sequence of data
                
                
            - features_num: list of str
                numerical features of the current dataset
            
            
            - features_cat: list of str
                categorical features of the current dataset
            
            
            - features_target: list of str
                column of the target variable
            
            
            - features_meta: list of str
                columns of meta-data
            
            
            - features_idx: dict
                col2i: dict
                    column to index mapping for df

            
            - features_stnry: list of str
                features to feed concatnate with output of TCN and feed into MLP
                Treat as stationary features
                
            
        """
        data = {}    
        target = [target_col]
        meta = list(set(sort_by_col + group_by + meta))
        selected_features, features_num, features_cat, features_idx = self.get_features(X_train, target, meta)
        
        train_data = self.df_to_list(X_train, 
                                sort_by=sort_by_col, 
                                group_by=group_by, 
                                return_index=False, 
                                min_length=self.min_length)
        data['train'] = train_data
        
        if X_valid is not None:
            test_data = self.df_to_list(X_valid, 
                                   sort_by=sort_by_col, 
                                   group_by=group_by, 
                                   return_index=False, 
                                   min_length=self.min_length)
            data['test'] = test_data
        
        ############################################################
        #                   build data object
        ############################################################

        # index
        data['features_num'] = features_num
        data['features_cat'] = features_cat
        data['features_target'] = target
        data['features_meta'] = meta
        data['features_idx'] = features_idx
        data['features_stnry'] = stnry
        
        # make sure all features in stationary is a part of the source_features
        assert(len(set(stnry) - set(features_num+features_cat)) == 0)
            
        print("data processed!")
        return data

    
    def process_data_inference(self, X_train, sort_by_col=[], group_by=[], meta=[], stnry=[], target_col=None):
        """ Process input dataframe into model-specific datasets.
        
        Difference from process_data is this function only process train
        dataframe and it does not require target column 
        
        
        Parameters
        ----------
        X_train: pd.DataFrame
            dataframe containing training data
            
        
        target_col: str
            column of the target (if present)
            
        
        sort_by_col: list
            sort the df by the columns in sort_by to maintain ordering
        
        
        group_by: list
            group the df by the columns. Each array in the resulting
            list is a group in the form of np.array.
            
            e.g. in the Customer Risk Model, we group by borrower id
            for the transactions dataframe so each array contains the 
            past transactions of an individual borrower.
            
            
        meta: list of str
            columns of meta-data
        
        
        stnry: list of str
            columns of the stnry features
        
        
        Returns
        ----------
        data: dict
            Processed data object for TCN. With the following fields:
            
            - train: list of np.arrays
                obtained by calling df_to_list on X_train.
                each element in the list is a np.array containing a sequence of data
                
            
            - index: list of np.arrays
                index/identifier of the input data, will be used to identify
                input data and their predictions
                
                
            - features_num: list of str
                numerical features of the current dataset
            
            
            - features_cat: list of str
                categorical features of the current dataset
            
            
            - features_meta: list of str
                columns of meta-data
            
            
            - features_idx: dict
                col2i: dict
                    column to index mapping for df
                    
            
            - features_stnry: list of str
                features to feed concatnate with output of TCN and feed into MLP
                Treat as stationary features
        """
        data = {}    
        meta = list(set(sort_by_col + group_by + meta))
        target_col = [] if target_col is None else [target_col]
        selected_features, features_num, features_cat, features_idx = self.get_features(X_train, target_col, meta)
        train_data, index = self.df_to_list(X_train, sort_by=sort_by_col, 
                                       group_by=group_by, return_index=True, min_length=self.min_length)
        
        data['train'] = train_data
        data['index'] = index

        # index
        data['features_num'] = features_num
        data['features_cat'] = features_cat
        data['features_meta'] = meta
        data['features_idx'] = features_idx
        data['features_stnry'] = stnry
    
        print("data processed!")
        return data

    
    def get_loader(self, data):
        """ create dataloaders using provided data object
        
        Parameters
        ----------
        data: dict
            The dictionary that contains all required information 
                to construct dataset

            required keys: {'train', 'features_num', 'features_cat', 
                            'features_meta', 'features_stnry', 'features_idx'}
            optional keys: {'test'}

            - train: training data, list of np.array, with each array being a datum 
                (e.g. a sequence of transactions for user_i)

            - test: testing data, optional, 
                list of np.array, each array being a datum 
                (e.g. a sequence of transactions for user_i)

            - features_num: list of str
                numerical features of the current dataset

            - features_cat: list of str
                categorical features of the current dataset

            Note: features_num + features_cat contains all features TCN has access to.

            - features_meta: list of str
                meta data columns of the current dataset

            - features_stnry: list of str
                stationary features of the current dataset

            - features_idx: dict
                column to index mapping
                
        
        Returns
        ----------
        (train_loader, test_loader, has_stnry): (dataloader, dataloader, bool)

            - train_loader: dataloader for training data, each batch contains 
                (src, tgt, seq_len, meta, stnry)
                 torch.utils.data.DataLoader


            - test_loader: dataloader for testing data, can be None.
                 torch.utils.data.DataLoader

                 If provided, each batch contains (src, tgt, seq_len, meta, stnry)

        or 

        (train_loader, has_stnry): (data_loader, bool)
            dataloader for training data, each batch contains 
            (src, tgt, seq_len, meta, stnry)
            torch.utils.data.DataLoader
        
        """
        has_stnry = (len(data['features_stnry']) > 0)
        loaders = dataloader.get_loader(copy.deepcopy(data), batch_size=self.batch_size, 
                                        pad_after=self.pad_after, vocabs=self.vocabs, 
                                        inference=False, has_stnry=has_stnry)

        if len(loaders) == 2:
            train_loader, test_loader = loaders
        else:
            train_loader = loaders
            test_loader = None
            
        if self.verbose > 0:
            print("train loader length: {}".format(len(train_loader)))
            
        # set vocabs for later uses
        self.vocabs = {'vocab_lens': train_loader.dataset.vocab_lens,
                       'w2i_dicts': train_loader.dataset.w2i_dicts,
                       'i2w_dicts': train_loader.dataset.i2w_dicts}
        return train_loader, test_loader, has_stnry
    
        
    def set_vocabs(self, vocabs):
        """  
        set vocabularies
        
        Parameters
        ----------
        vocabs: dict of dict
            If provided, must have keys = {'vocab_lens', 'w2i_dicts', 'i2w_dicts'}
            If provided, the current dataset loads the vocabularies from 
                the provided dictionary.
        
        Returns
        ----------
        None
        """
        self.vocabs = vocabs
        
        
    def get_inference_loader(self, data):
        """  
        
        Parameters
        ----------
        data: dict
            The dictionary that contains all required information 
                to construct dataset

            required keys: {'train', 'features_num', 'features_cat', 
                            'features_meta', 'features_stnry', 'features_idx'}

            - train: training data, list of np.array, with each array being a datum 
                (e.g. a sequence of transactions for user_i)

            - features_num: list of str
                numerical features of the current dataset

            - features_cat: list of str
                categorical features of the current dataset

            Note: features_num + features_cat contains all features TCN has access to.

            - features_meta: list of str
                meta data columns of the current dataset

            - features_stnry: list of str
                stationary features of the current dataset

            - features_idx: dict
                column to index mapping
        
        
        Returns
        ----------
        (train_loader, has_stnry): (data_loader, bool)
            dataloader for inference data, each batch contains 
            (src, tgt, seq_len, meta, stnry)
            torch.utils.data.DataLoader
            
        """
        has_stnry = (len(data['features_stnry']) > 0)
        if self.vocabs is None:
            print("""No vocabs for categorical features provided... 
                     this is equivalent to randomly embed categorical features""")
        train_loader = dataloader.get_loader(copy.deepcopy(data), batch_size=self.batch_size, 
                                             pad_after=self.pad_after, vocabs=self.vocabs, 
                                             inference=True, has_stnry=has_stnry)
        return train_loader, has_stnry
        
    
    def prep_model_params(self, data, train_loader, test_loader):
        """ aggregate parameters for the low level TCN model.
                => model.TCNBinClfBase
        
        Parameters
        ----------
        data: dict
            The dictionary that contains all required information 
                to construct dataset

            required keys: {'train', 'features_num', 'features_cat', 
                            'features_meta', 'features_stnry', 'features_idx'}

            - train: training data, list of np.array, with each array being a datum 
                (e.g. a sequence of transactions for user_i)

            - test: testing data, optional, 
                list of np.array, each array being a datum 
                (e.g. a sequence of transactions for user_i)

            - features_num: list of str
                numerical features of the current dataset

            - features_cat: list of str
                categorical features of the current dataset

            Note: features_num + features_cat contains all features TCN has access to.

            - features_meta: list of str
                meta data columns of the current dataset

            - features_stnry: list of str
                stationary features of the current dataset

            - features_idx: dict
                column to index mapping
                
        train_loader: dataloader for training data, each batch contains (src, tgt, seq_len, meta, stnry)
            torch.utils.data.DataLoader


        test_loader: dataloader for testing data, can be None.
            torch.utils.data.DataLoader

            If provided, each batch contains (src, tgt, seq_len, meta, stnry)
        
        
        Returns
        ----------
        params: dict
            all required params. See implementation for detail.
        """
        assert(hasattr(self, 'vocabs'))
        
        params = {
            'features_num': data['features_num'],
            'features_cat': data['features_cat'],
            'features_stnry': data['features_stnry'],
            'src_col2i': train_loader.src_col2i,
            'stnry_col2i': train_loader.stnry_col2i,
            'embed_dims': self.feature_embed_dims,
            'vocab_lens': self.vocabs['vocab_lens'],
            'tcn_layers': self.tcn_layers, 
            'mlp_layers': self.mlp_layers,
            'output_size': self.output_size, 
            'dropout_tcn': self.dropout_tcn,
            'dropout_mlp': self.dropout_mlp,
            'kernel_size': self.kernel_size,
            'device': self.device,
            'mlp_batch_norm': self.mlp_batch_norm,
        } 
        return params
       
        
    def process_feature_embed_dims(self, data, feature_embed_dims, default_embed_dim):
        """ Process the embedding dimensions of the categorical features
        
        This function assigns default embedding dimension for unspecified
        categorical features.
        
        Parameters
        ----------
        data: dict
            must contain features_cat for this function to work
        
        
        feature_embed_dims: dict
            Dictionary with key = categorical features and
                            value = dimension of the embedding

            
        default_embed_dim: int, default = 10
            default dimension for categorical features. However, try to avoid using this!
            always explicitly define embedding dimensions to avoid bugs
        
        
        Returns
        ----------
        feature_embed_dims: dict
            Processed dictionary with key = categorical features and
                            value = dimension of the embedding
        """
        if feature_embed_dims is None:
            print("""No feature embedding dimensions provided... 
            use default embedding dim: {}""".format(default_embed_dim))
            features_embed_dims = {}
            for f in data['features_cat']:
                feature_embed_dims[f] = default_embed_dim
                
        elif set(data['features_cat']) != set(feature_embed_dims.keys()):
            print("""unmatched embeding dimensions. Setting missing embed dims to default embedding dim""")
            for f in data['features_cat']:
                if f not in feature_embed_dims:
                    print("setting embed dim to {} for feature: {}".format(default_embed_dim, f))
                    feature_embed_dims[f] = default_embed_dim
        return feature_embed_dims
        
        
       
    def fit(self, data, feature_embed_dims=None, warm_start=True):    
        """ After initializing the TCNBinaryClassifier object
        fit the model with provided data. 
        
        Parameters
        ----------
        data: dict
            The dictionary that contains all required information 
                to construct dataset

            required keys: {'train', 'features_num', 'features_cat', 
                            'features_meta', 'features_stnry', 'features_idx'}
            optional keys: {'test'}

            - train: training data, list of np.array, with each array being a datum 
                (e.g. a sequence of transactions for user_i)

            - test: testing data, optional, 
                list of np.array, each array being a datum 
                (e.g. a sequence of transactions for user_i)

            - features_num: list of str
                numerical features of the current dataset

            - features_cat: list of str
                categorical features of the current dataset

            Note: features_num + features_cat contains all features TCN has access to.

            - features_meta: list of str
                meta data columns of the current dataset

            - features_stnry: list of str
                stationary features of the current dataset

            - features_idx: dict
                column to index mapping
            
        
        feature_embed_dims: dict
            Dictionary with key = categorical features and
                            value = dimension of the embedding
                            
        
        warm_start: bool, default = True
            When calling fit, whether to train with the previously allocated model.
            
            - True: if the model is initialized, calling .fit will train that model.
            - Flase: regardless whether a has been intialized, initialize a new one.
            
        
        Returns
        ----------
        None
        """
        random.seed(self.seed)
        train_loader, test_loader, self.has_stnry = self.get_loader(data)
        if warm_start and hasattr(self, 'model'):
            # if warm_start, just keep on training
            self.model = train.train(self, self.model, train_loader, test_loader)
        else:
            if feature_embed_dims is not None:
                self.feature_embed_dims = feature_embed_dims
                
            assert(hasattr(self, 'feature_embed_dims'))
            # process embedding dimensions for categorical features
            self.feature_embed_dims = self.process_feature_embed_dims(data, 
                                          self.feature_embed_dims, 
                                          self.default_embed_dim)

            self.params = self.prep_model_params(data, train_loader, test_loader)
            self.model = tcn_model.TCNBinClfBase(self.params)

            print("saving vocabularies ... ")
            # save vocabularies
            torch.save(self.vocabs, '{}/vocabs.pt'.format(self.model_dir))

            ##################################################################
            #                          Training
            ##################################################################

            # run training
            self.model = train.train(self, self.model, train_loader, test_loader)

    
    def predict_proba(self, data, return_sequences=False, return_df=True):
        """ Make predictions using provided data object
        
        Parameters
        ----------
        data: dict
            data object for inference
    

            required keys: {'train', 'features_num', 'features_cat', 
                            'features_meta', 'features_stnry', 'features_idx'}

            - train: training data, list of np.array, with each array being a datum 
                (e.g. a sequence of transactions for user_i)

            - features_num: list of str
                numerical features of the current dataset

            - features_cat: list of str
                categorical features of the current dataset

            Note: features_num + features_cat contains all features TCN has access to.

            - features_meta: list of str
                meta data columns of the current dataset

            - features_stnry: list of str
                stationary features of the current dataset

            - features_idx: dict
                column to index mapping
                
        
        return_sequences: bool
            True:  to return outputs from all time steps (many-to-many prediction) 
                shape = data_size x sequence_length
                
            False: to return the prediction at the last time step (many-to-one prediction) 
                shape = data_size
        
        
        return_df: bool
            True to obtain a dataframe with predicted probabilities and meta data, call
                inference_formatter.get_inference(<meta_data_as_list>)
                
            False to obtain only the sequence of predictions, call:
                inference_formatter.get_inference(None)
        
        
        Returns
        ----------
        result: pd.DataFrame or np.array
        """
        loader, self.has_stnry = self.get_inference_loader(data)
        
        if not hasattr(self, 'model'):
            raise ValueError("don't yet have model")
        inference_formatter = train.inference(self, self.model, loader, return_sequences)
        if return_df:
            result = inference_formatter.get_inference(data['features_meta'])
        else:
            result = inference_formatter.get_inference(None)
        return result
        
    
    def set_params(self, **params):
        """ A function to set attributes
        
        Parameters
        ----------
        params: **kwargs
            kwargs to set self.k = v
            
            
        Returns
        ----------
        None
        """
        for k,v in params.items():
            if k == 'model_dir':
                self.log_path = os.path.join(v, 'log.txt')
            setattr(self, k, v)
            
            
    def effective_history(self):
        """  
        Compute the effective history of self.model (TCN)
        
        Parameters
        ----------
        None
        
        A model must be initialized => at least call fit for an epoch. or initialize
        model as shown in the fit function.  Line 899 - 909
        
        Returns
        ----------
        eh: int
            effective history
        """
        dilation=2
        if isinstance(self.tcn_layers[0], int) and isinstance(self.kernel_size, int):
            return (self.kernel_size-1)*(dilation**len(self.tcn_layers))
        
        assert(len(self.tcn_layers) == len(self.kernel_size))
        eh = 1
        for i in range(len(self.tcn_layers)):
            k = self.kernel_size[i]
            l = len(self.tcn_layers[i])
            eh += (k-1)*(dilation**l) - 1
        return eh
    
    
    def count_parameters(self):
        """  
        Compute the number of parameters in self.model
        
        Parameters
        ----------
        None
        
        A model must be initialized => at least call fit for an epoch. or initialize
        model as shown in the fit function.  Line 899 - 909
        
        Returns
        ----------
        result: int
            number of parameters in the model
        """
        if hasattr(self, 'model'):
            return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("No model initiated")
        return None
            
            
    def save(self, model_dir): 
        """ save model to model_dir/tcn.pth using torch.
        
        Will clear many attribute objects to save memory.
        
        Parameters
        ----------
        model_dir: str
            path to the model directory
        
        Returns
        ----------
            None
        """
        # temporarily remove all information
        if hasattr(self, 'criterion'):
            self.criterion.reset_test()
            criterion_tmp = copy.deepcopy(self.criterion)
            self.criterion.reset_train()
        if hasattr(self, 'train_metrics'):
            train_metrics_tmp = copy.deepcopy(self.train_metrics)
            self.train_metrics.reset()
        if hasattr(self, 'test_metrics'):
            self.test_metrics.reset()
        if hasattr(self, 'valid_formatter'):
            self.valid_formatter.reset()
        if hasattr(self, 'valid_formatter_target'):
            self.valid_formatter_target.reset()
        if hasattr(self, 'inference_formatter'):
            self.inference_formatter.reset()
            
        path = os.path.join(model_dir, 'tcn.pth')
        torch.save(self, path)
        
        # restore train_metrics
        if hasattr(self, 'train_metrics'):
            self.train_metrics = train_metrics_tmp
        if hasattr(self, 'criterion'):
            self.criterion = criterion_tmp
