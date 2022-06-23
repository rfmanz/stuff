import re
import dill
import numpy as np
import torch
import copy
from torch import nn, LongTensor
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

class MyDataset(Dataset):
    
    def __init__(self, data_object, is_train, inference, 
                 train_dataset=None, load_vocabs=None):        
        """ Dataset object for TCNs
        
        Parameters
        ----------
        data_object: dict
            The dictionary that contains all required information 
            to construct dataset
                
            required keys: {'train', 'features_num', 'features_cat', 
                            'features_meta', 'features_stnry', 'features_idx'}
            optional keys: {'test'}
            
            - train: training data, list of np.array, each array being a datum 
                (e.g. a sequence of transactions for user_i)
                
            - test: testing data, optional, 
                list of np.array, each array being a datum 
                (e.g. a sequence of transactions for user_i)
                
            - features_num: list of str
                numerical features of the current dataset
            
            - features_cat: list of str
                categorical features of the current dataset
                
            Note: features_num + features_cat contains all features TCN uses.
            
            - features_meta: list of str
                meta data columns of the current dataset
                
            - features_stnry: list of str
                stationary features of the current dataset
                
            - features_idx: dict
                column to index mapping
        
        
        is_train: bool
            Whether the current dataset is the training data.
        
        
        inference: bool
            Whether the current dataset is for inference.
            
        
        train_dataset: dataloader.MyDataset, None-able
            If provided, the current dataset loads the vocabularies 
                from the train_dataset.
            
            
        load_vocabs: dict of dict, None-able
            If provided, must have keys = {'vocab_lens', 'w2i_dicts', 'i2w_dicts'}
            If provided, the current dataset loads the vocabularies from 
                the provided dictionary.
            
        
        Returns
        ----------
        None
        
        
        Modifies
        ----------
        data_object: dict
            The modified data_object now has two additional keys: 
                {'src_col2i', 'stnry_col2i'}
            
            - src_col2i: mapping of features to index for self.source 
                (dynamic/time-series data)
                
            - stnry_col2i: mapping of features to index for self.stnry 
                (stationary data)
        
        """
        self.is_train = is_train
        self.inference = inference
        
        if is_train or inference:
            # data is a list of arrays
            self.data = data_object['train']
        else:
            # data is a list of arrays
            self.data = data_object['test']

        self.features_num = data_object['features_num']    
        self.features_cat = data_object['features_cat']     
        self.features_meta = data_object['features_meta']    
        self.features_stnry = data_object['features_stnry']
        self.col2i = copy.deepcopy(data_object['features_idx'])
        self.source_features = sorted(list(set(self.features_num + self.features_cat) - set(self.features_stnry)))
        self.len = len(self.data)
        
        if load_vocabs is not None:
            print("loading vocabs....")
            self.vocab_lens = load_vocabs['vocab_lens']
            self.w2i_dicts = load_vocabs['w2i_dicts']
            self.i2w_dicts = load_vocabs['i2w_dicts']
        elif is_train:
            if inference:
                print("""making inference with randomly generated embeddings. 
                         Please provide vocab for consistent results.""")
                
            # initializing vocabularies for categorical features
            self.vocab_lens, self.w2i_dicts, self.i2w_dicts = self.build_vocab(self.data, self.features_cat, self.col2i)
        else:
            self.vocab_lens, self.w2i_dicts, self.i2w_dicts = self.load_vocab(train_dataset)
        
        
        # encode data and segment by diff feature/column types.
        self.data = self.encode_source(self.data, self.features_cat, self.col2i, self.w2i_dicts)
        self.source, self.src_col2i = self.select_cols(self.data, self.source_features, self.col2i) # train 
        self.meta, _ = self.select_cols(self.data, self.features_meta, self.col2i)
        self.stnry_col2i = None
        if len(self.features_stnry) > 0:
            self.stnry, self.stnry_col2i = self.select_cols(self.data, self.features_stnry, self.col2i)

        data_object['src_col2i'] = self.src_col2i
        data_object['stnry_col2i'] = self.stnry_col2i
        if not inference:
            self.features_target = data_object['features_target']
            self.target, _ = self.select_cols(self.data, self.features_target, self.col2i) # list of targets
    
    
    def build_vocab(self, data, features_cat, col2i):
        """ Build vocabulary for categorical features in data
        
        Parameters
        ----------
        data: list of np.arrays
            The time-series data to feed TCN. For the list of np.arrays, 
                each element array contains a sequence of features. 
            
            data[i] has shape: sequence_length x num_features
            
        
        features_cat: list of str
            categorical features of the current dataset
            
        
        col2i: dict
            column to index mapping for data.
            
        
        Returns
        ----------
        vocab_lens: dict
            Dictionary with key = categorical features and  
                            value = number of distinct categories for that feature
            e.g {'feature_A': 3, 'feature_B': 5}
        
        
        w2i_dicts: dict of dict, for nn.Embedding
            Nested dictionary containing the <word-to-index> mapping 
                for each categorical feature.
            e.g. {'feature_A': {'a': 0, 'b': 1, 'c': 2}, 
                  'feature_B': {'aa': 0, 'bb': 1, 'cc': 2, 'dd': 3, 'ee': 4}}
        
        
        w2i_dicts: dict of dict, for nn.Embedding
            Nested dictionary containing the <index-to-word> mapping 
                for each categorical feature.
                
            e.g. {'feature_A': {0: 'a', 1: 'b', 2: 'c'}, 
                  'feature_B': {0: 'aa', 1: 'bb', 2: 'cc', 3: 'dd', 4: 'ee'}}
        """
        vocab_lens = {}
        w2i_dicts = {}
        i2w_dicts = {}
        for f in features_cat:
            col = col2i[f]
            vocab_iter = (map(lambda d_array: d_array[:, col], data))
            vocab = set()
            for v in vocab_iter:
                vocab = vocab.union(v)
            is_None = None in vocab
                                        
            vocab = list(vocab)
            vocab = sorted(list(filter(lambda b:b is not None, vocab))) # sort the vocab list
            
            if is_None:
                vocab = np.append(vocab, 'None')
            
            vocab_lens[f] = len(vocab) + 3
            w2i_dicts[f] = dict(zip(vocab, range(3, len(vocab)+3)))
            w2i_dicts[f]['<pad>'] = 0  # padded token
            w2i_dicts[f]['<unk>'] = 1  # unknown token
            w2i_dicts[f]['<eos>'] = 2  # eos token... not useful in TCN scenario

            i2w_dicts[f] = dict(zip(range(3, len(vocab)+3), vocab))
            i2w_dicts[f][0] = '<pad>'
            i2w_dicts[f][1] = '<unk>'
            i2w_dicts[f][2] = '<eos>'
            
            print('feature ({}) vocabulary size: {} '.format(f, vocab_lens[f]))
        
        return vocab_lens, w2i_dicts, i2w_dicts
            
            
    def load_vocab(self, train_dataset):
        """ Load vocabularies from given <train_dataset>
        
        Parameters
        ----------
        train_dataset: dataloader.MyDataset
            A MyDataset object. Should contain the following three attributes:
                 {'vocab_lens', 'w2i_dicts', 'i2w_dicts'}
            
    
        Returns
        ----------
        vocab_lens: dict
            Dictionary with key = categorical features and  
                            value = number of distinct categories for that feature
            e.g {'feature_A': 3, 'feature_B': 5}
        
        
        w2i_dicts: dict of dict, for nn.Embedding
            Nested dictionary containing the <word-to-index> mapping 
                for each categorical feature.
                
            e.g. {'feature_A': {'a': 0, 'b': 1, 'c': 2}, 
                  'feature_B': {'aa': 0, 'bb': 1, 'cc': 2, 'dd': 3, 'ee': 4}}
        
        
        i2w_dicts: dict of dict, for nn.Embedding
            Nested dictionary containing the <index-to-word> mapping 
                for each categorical feature.
                
            e.g. {'feature_A': {0: 'a', 1: 'b', 2: 'c'}, 
                  'feature_B': {0: 'aa', 1: 'bb', 2: 'cc', 3: 'dd', 4: 'ee'}}
        """
        vocab_lens = train_dataset.vocab_lens
        w2i_dicts = train_dataset.w2i_dicts
        i2w_dicts = train_dataset.i2w_dicts
        return vocab_lens, w2i_dicts, i2w_dicts
    
    
    def encode_source(self, data, features_cat, col2i, w2i_dicts):
        """ Encode categorical features based on provided w2i_dicts 
            (word-to-index mapping)
        
        This function encodes the data object in place to save memory.
        
        Parameters
        ----------
        data: list of np.arrays
            The time-series data to feed TCN. For the list of np.arrays, 
                each element array contains a sequence of features. 
            
            data[i] has shape: sequence_length x num_features
            
            data[i] may contain any type of data 
                e.g. numerical, str, datetime... before encoding
            
        
        features_cat: list of str
            categorical features of the current dataset
        
        
        col2i: dict
            column to index mapping for data.
        
        
        w2i_dicts: dict of dict, for nn.Embedding
            Nested dictionary containing the <word-to-index> mapping 
                for each categorical feature.
                
            e.g. {'feature_A': {'a': 0, 'b': 1, 'c': 2}, 
                  'feature_B': {'aa': 0, 'bb': 1, 'cc': 2, 'dd': 3, 'ee': 4}}
        
        
        Returns
        ----------
        data: list of np.arrays
            The embedded data. This list of np.arrays now only contain 
                numerical values.
        """
        
        n_unk = 0
        
        def w2i(feature_w2i, token, vocab, n_unk):
            if token not in vocab:
                n_unk += 1
            return feature_w2i[token] if token in vocab else feature_w2i['<unk>']
        
        for col in features_cat:
            col_idx = col2i[col]
            vocab = w2i_dicts[col].keys()
            mapper = np.vectorize(lambda x: w2i(w2i_dicts[col], x, vocab, n_unk))
            
            for datum in data:
                # for memory purpose, we will modify data object inplace
                datum[:, col_idx] = mapper(datum[:, col_idx])
                
        if not self.is_train:
            print('num unknown tokens in test set: ',n_unk)
        
        return data
            
    
    def select_cols(self, data, features, col2i):
        """ Given list of features and the column to index mapping, extract
        the columns from data and corresponding column to index mapping.                     
                                    
        Parameters
        ----------
        data: list of np.arrays
            The time-series data to feed TCN. For the list of np.arrays, 
                each element array contains a sequence of features. 
            
            data[i] has shape: sequence_length x num_features
            
            
        features: list of str
            list of features to include.
            
            
        col2i: dict
            column to index mapping for data.
            
            
        Returns
        ----------
        result: list of np.arrays
            data with only selected columns
            
            
        sub_col2i: dict
            column to index mapping within result
        
        
        Example:
        ----------
        e.g.data[0] = np.array([[ 1,  2,  3],
                                [20, 21, 22],
                                [33, 13, 33]])
            features = ['A', 'C']
            col2i = {'A': 0, 'B': 1, 'C': 2}
            
        result:
            data[0] = np.array([[ 1, 3],
                                [20,22],
                                [33,33]])
            sub_col2i = {'A': 0, 'C': 2}
        """
        
        cols = [col2i[f] for f in features]
        result = list(map(lambda d_array: d_array[:, cols], data))
        sub_col2i = dict(zip(features, range(len(features)))) # col2i mapping within 
        return result, sub_col2i
        
        
    def __getitem__(self, index):
        """ get item given index
        
        Parameters
        ----------
        index: int
            index of extracted datum
            
        
        Returns
        ----------
        datum: tuple - (list, list, list, list/None) or (list, list, list/None)
            returns the index_th sequence, separated into:
            
            
            for training and validation
            - (source[index], target[index], meta[index], and stnry[index]/None) 
                
                
            for inference             
            - (source[index], meta[index], and stnry[index]/None) 
        """
        
        if self.inference: # when encoding 
            stnry = self.stnry[index] if len(self.features_stnry) > 0 else None
            datum = (self.source[index], self.meta[index], stnry)
        else: # when training and validating
            stnry = self.stnry[index] if len(self.features_stnry) > 0 else None
            datum = (self.source[index], self.target[index], self.meta[index], stnry)
        return datum
    
    
    def __len__(self):
        return self.len
        
        
        
def collate_fn_pad(data, pad_after=True, has_stnry=False):
    """ Collate function with padding
    info: https://pytorch.org/docs/stable/data.html#dataloader-collate-fn
          https://discuss.pytorch.org/t/how-to-use-collate-fn/27181
    
    Parameters
    ----------
    data: list of np.arrays for a BATCH
        The time-series data to feed TCN. For the list of np.arrays, 
            each element array contains a sequence of features. 
        
        data[i] has shape: sequence_length x num_features
            
    
    pad_after: bool
        Whether to pad the data points in the front or the end. 
        
        Always pad_after when dealing with current TCN implementation, 
        but may be helpful to pad in the front when using 
            Sequential Variational Autoencoder
        
        
    has_stnry: bool
        Whether there is stationary data to pad
        
        
    Returns
    ----------
    src_seq: torch.Tensor
        batched source sequences
        shape: batch_size x sequence_length_max x feature_dim
    
    tgt_seq: torch.Tensor
        batched target sequences
        shape: batch_size x sequence_length_max x 1
        
    
    seq_lengths: torch.Tensor
        batched sequence lengths
        length: batch_size
    
    
    meta: list 
        meta data for each data point
        meta[i].shape = n_meta_features x seq_length
        length: batch_size
    
    
    stnry_seq: torch.Tensor, None-able, 
        batched stationary features
        shape: batch_size x sequence_length_max x feature_dim
    """
    
    
    # revert seq_tensor for target_tensor
    source, target, meta, stnry = tuple(map(list, zip(*data)))
    seq_lengths = LongTensor(list(map(len, source)))
                        
    # pad source seq in the front and target in the end, since we reconstruct in the reverse order
    src_seq, _ = pad(source, seq_lengths, pad_after=pad_after)
    tgt_seq, _ = pad(target, seq_lengths, pad_after=pad_after)
    stnry_seq = None
    if has_stnry:
        stnry_seq, _ = pad(stnry, seq_lengths, pad_after=pad_after)

    return src_seq, tgt_seq, seq_lengths, meta, stnry_seq
    

def collate_fn_pad_inference(data, pad_after=True, has_stnry=False):    
    """ Collate function with padding for inference 
    info: https://pytorch.org/docs/stable/data.html#dataloader-collate-fn
          https://discuss.pytorch.org/t/how-to-use-collate-fn/27181
          
    difference from the previous function: collate_fn_pad
    No need to return tgt_seq
    
    Parameters
    ----------
    data: list of np.arrays for a BATCH
        The time-series data to feed TCN. For the list of np.arrays, each element array
        contains a sequence of features. 
        
        data[i] has shape: sequence_length x num_features
            
    
    pad_after: bool
        Whether to pad the data points in the front or the end. 
        
        Always pad_after when dealing with current TCN implementation, 
        but may be helpful to pad in the front when using Sequential Variational Autoencoder
        
        
    has_stnry: bool
        Whether there is stationary data to pad
        
        
    Returns
    ----------
    src_seq: torch.Tensor, batch_size x sequence_length_max x feature_dim
        batched, padded source sequences
    

    seq_lengths: torch.Tensor, batch_size
        batched, padded sequence lengths
        
    
    meta: list, 
        unpadded meta data for each data point
        len(meta) = batch_size
        meta[i].shape = n_meta_features x seq_length
        
    
    stnry_seq: torch.Tensor, None-able, batch_size x sequence_length_max x feature_dim
        batched, padded stationary features
    """
    # revert seq_tensor for target_tensor
    source, meta, stnry = tuple(map(list, zip(*data)))
    seq_lengths = LongTensor(list(map(len, source)))
    
    # pad source seq in the front and target in the end, since we reconstruct in the reverse order
    src_seq, _ = pad(source, seq_lengths, pad_after=pad_after)
    stnry_seq = None
    if has_stnry:
        stnry_seq, _ = pad(stnry, seq_lengths, pad_after=pad_after)

    return src_seq, seq_lengths, meta, stnry_seq
    

def pad(seq, seq_lengths, pad_after=True): # batch_size x sequence_length_max x feature_dim 
    """ Pad sequence into a torch tensor
    
    Parameters
    ----------
    seq: list of np.array
        a batch of unpadded sequences
        seq[i].shape = sequence_length x num_features
        
    
    seq_lengths: list of int, len(seq_lengths) = batch_size
        sequences_length for each data point in seq
        
    
    pad_after: bool
        Whether to pad the data points in the front or the end. 
        
        Always pad_after when dealing with current TCN implementation, 
        but may be helpful to pad in the front when using 
            Sequential Variational Autoencoder
        
        
    Returns
    ----------
    seq_tensor: torch.Tensor
        batched, padded sequence
        shape: batch_size x sequence_length_max x feature_dim

    seq_lengths: torch.Tensor, batch_size
        batched, padded sequence lengths
    """

    max_seq_len = seq_lengths.max()
    seq_tensor = Variable(torch.zeros((len(seq), max_seq_len, len(seq[0][0])))).long()
    # pad input tensor
    for idx, seq in enumerate(seq):
        seq_len = seq_lengths[idx]
        if pad_after:
            seq_tensor[idx, :seq_len] = LongTensor(np.asarray(seq).astype(int))
        else: 
            # pad before
            seq_tensor[idx, max_seq_len-seq_len:] = LongTensor(np.asarray(seq).astype(int))
    return seq_tensor, seq_lengths
    

def get_loader(data_object, batch_size=32, pad_after=True, vocabs=None, inference=False, has_stnry=False):
    """
    Create dataloader(s)
    
    If both 'train' and 'test' are available in the data_object, 
    this function process the train first by building the dataset, 
    convert the categorical data into numerical indices, 
    and build vocabularies. Then it process the test data similarly 
    except for using training data's vocabulary to maintain the ordering. 
    In the end, this function builds a train dataloader and a test dataloader.
    
    If only 'train' is provided or the data is for inference, this function 
    only process and return the train dataloader.
    
    Parameters
    ----------
    data_object: dict
        The dictionary that contains all required information 
            to construct dataset
            
        required keys: {'train', 'features_num', 'features_cat', 
                        'features_meta', 'features_stnry', 'features_idx'}
        optional keys: {'test'}
            
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
    
    
    batch_size: int
        Batch size
    
    
    pad_after: bool
        Whether to pad the data points in the front or the end. 
        
        Always pad_after when dealing with current TCN implementation, 
        but may be helpful to pad in the front when using 
            Sequential Variational Autoencoder
        
    
    vocabs: dict of dict, None-able
        If provided, must have keys = {'vocab_lens', 'w2i_dicts', 'i2w_dicts'}
        If provided, the current dataset loads the vocabularies 
            from the provided dictionary.
    
    
    inference: bool
        Whether the current dataset is for inference.
        
    
    has_stnry: bool
        Whether there is stationary data to pad
    
    
    Returns
    ----------
    (train_loader, test_loader): tuple of dataloaders
    
        - train_loader: dataloader for training data, each batch contains 
            (src, tgt, seq_len, meta, stnry)
             torch.utils.data.DataLoader
        
        
        - test_loader: dataloader for testing data, can be None.
             torch.utils.data.DataLoader

             If provided, each batch contains (src, tgt, seq_len, meta, stnry)
    
    or 
    
    train_loader: dataloader for training data, each batch contains 
        (src, tgt, seq_len, meta, stnry)
        torch.utils.data.DataLoader
    
    """
    
    
    train_dataset = MyDataset(data_object, is_train=not inference, 
                              load_vocabs=vocabs, inference=inference)   
    if inference:
        collate_wrapper = lambda d: collate_fn_pad_inference(d, pad_after=pad_after, has_stnry=has_stnry)
    else:
        collate_wrapper = lambda d: collate_fn_pad(d, pad_after=pad_after, has_stnry=has_stnry)
    
    print("shuffling dataset: ", not inference)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=(not inference),
                              collate_fn=collate_wrapper)
    train_loader.src_col2i = train_dataset.src_col2i
    train_loader.stnry_col2i = train_dataset.stnry_col2i
    
    if 'test' in data_object and not inference:
        test_dataset = MyDataset(data_object, is_train=False, 
                                 train_dataset=train_dataset, inference=inference)
        test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             collate_fn=collate_wrapper)
        test_loader.src_col2i = test_dataset.src_col2i
        test_loader.stnry_col2i = test_dataset.stnry_col2i

        return train_loader, test_loader    
    return train_loader
