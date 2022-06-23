import time
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_util

from torch import nn, LongTensor
from torch.autograd import Variable
from torch.nn import LSTM
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

from tcn import TemporalConvNet


class Encoder_TCN(nn.Module):
    def __init__(self, input_size,
                 num_channels, kernel_size, dropout):        
        """ Encoder TCN module. 

        Sequential encoder with TCN as the main building block
        
        Parameters
        ----------
        input_size: int
            Length of the input vector to TCN at each time step


        output_size: int
            Length of the output vector of TCN at each time step
        
        
        num_channels: list or list of list
            Number of neurons at each layer. Can be a list or a list of list

            e.g. [100, 100, 100, 100] if using single stack
                 [[100, 100], [100, 100]] if using stacked tcn

            TCN_v1 is built using single stack. Since each TemporalConvNet block
                comes with residual connections, along with fixed geometric dilution of
                2**i at layer  i. The advantage of using stacked TCN is to 
                    1) incorporate additional non-linearity to by stacking TemporalConvNet without 
                        residual connections in between and 
                    2) have more densely connected convolutional connections
        

        kernel_size: int or list of int
            kernel size, or list of kernel sizes for each stack of tcn.

            int if using single stack. e.g.  5
            list of int if using stacked tcn. e.g. [5, 7]

            If provided a list of kernel sizes, assert(len(kernel_size) == len(num_channels)


        dropout: float in [0, 1]
            dropout probability, recommend for customer risk model: 0.2


        Returns
        ----------
        None
        """
        super(Encoder_TCN, self).__init__()   
   
        if isinstance(num_channels[0], int) and isinstance(kernel_size, int):
            self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
            
        elif isinstance(num_channels[0], list) and isinstance(kernel_size, list):
            # there is a kernel_size for each stack
            assert(len(num_channels) == len(kernel_size))
            layers = []
            for i in range(len(num_channels)):
                input_size_ = input_size if i == 0 else num_channels[i-1][-1]     
                layers.append(TemporalConvNet(input_size_, 
                                              num_channels[i], 
                                              kernel_size=kernel_size[i],
                                              dropout=dropout))
                
            self.tcn = nn.Sequential(*layers)
                   
            
    def forward(self, inputs):
        """ Forward computation 
        
        Parameters
        ----------
        Inputs: tensor
            inputs have dimension: batch_size x senquence_length x num_features.   (N, L_in, C_in) 	
                

        Returns
        ----------
        o: tensor
            output of TCN, must have dimension: batch_size x senquence_length x num_features.  
        """
        # TemporalConvNet inputs have to have dimension (N, C_in, L_in)
        o = self.tcn(inputs.transpose(1,2))  # input should have dimension (N, C, L)
        return o.transpose(2,1)
      
    
######################################################
#                  Wrapper Module
###################################################### 


class MLP_block(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, batch_norm=False, dropout=1, out_layer=False):
        """ Multi-layer perceptron block wrapper for 
            batch normalization, fully connected layer, relu activation, and dropout.

        
        Parameters
        ----------
        in_dim: int
            Input dimension to the fully connected layer.


        out_dim: int
            Output dimension of the fully connected layer.


        bias: bool
            Whether to include bias.


        batch_norm: bool
            Whether to use Batch Normalization.


        dropout: float in [0, 1]
            dropout probability, recommend for customer risk model: 0.5
            

        out_layer: bool
            Whether the allocated MLP_block is the output layer. 
            
            If True: there is no ReLU or dropout attached after the fully connected layer.

        
        Returns
        ----------
        None
        """
        super(MLP_block, self).__init__()
        if batch_norm:
            self.bn = nn.BatchNorm1d(in_dim)
        self.linear = nn.Linear(in_dim, out_dim, bias)
        self.out_layer = out_layer
        if not out_layer:
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout)
        
        
    def forward(self, x):
        """ Forward computation
        
        Parameters
        ----------
        x: tensor
            input tensor
        
        Returns
        ----------
        out: tensor
            output tensor
        """
        # x has input size: batch x length x features
        out = self.bn(x.transpose(1,2)).transpose(2,1) if hasattr(self, 'bn') else x        
        out = self.linear(out)
        out = self.dropout(self.relu(out)) if not self.out_layer else out
        return out


class MLP(nn.Module):
    def __init__(self, in_dim, mlp_layers, out_dim, dropout=1, device='cpu', batch_norm=False):
        """ Multi-layer Perceptron 
        
        Parameters
        ----------
        in_dim: int
            Input dimension to the Multi-layer perceptron.


        mlp_layers: list of int
            number of neurons for each hidden layers.


        out_dim: int
            Output dimension of the MLP.


        dropout: float in [0, 1], default = 1
            dropout probability, recommend for customer risk model: 0.5


        device: str, {‘cpu’, ‘cuda:0’}, default=‘cpu’
            Whether running on CPU or single GPU


        batch_norm: bool, default=‘False’
            Whether to use Batch Normalization.


        Returns
        ----------
        None
        """
        super(MLP, self).__init__()
        
        cuda_available = torch.cuda.is_available()
        self.device = device
        
        in_layer_dims = list(zip([in_dim]+mlp_layers, mlp_layers+[out_dim])) # set latent_dim to 2
        self.encoder = self.build_encoder(in_layer_dims, batch_norm, dropout)
        
        
    def build_encoder(self, in_layer_dims, batch_norm, dropout):
        """ Construct MLP using MLP_block ’s
        
        Parameters
        ----------
        in_layer_dims: list of (int, int)
            The list of (input_dim, output_dim) to each MLP layers (MLP_block)
            e.g.  [(100, 200), (200, 200), (200, 5)]


        batch_norm: bool, default=‘False’
            Whether to use Batch Normalization.


        dropout: float in [0, 1], default = 1
            dropout probability, recommend for customer risk model: 0.5


        Returns
        ----------
        encoder: nn.Sequential model
            MLP model built with nn.Sequential.
        """
        encoder_layer = []

        for idx, (in_dim, out_dim) in enumerate(in_layer_dims):
            is_out_layer = (idx == len(in_layer_dims) - 1)
            encoder_layer.append(MLP_block(in_dim, 
                                           out_dim, 
                                           batch_norm=batch_norm, 
                                           dropout=dropout, 
                                           out_layer=is_out_layer)) 
            
        encoder = nn.Sequential(*encoder_layer)      
        return encoder
    
    
    def forward(self, inputs):
        """ Forward pass
        
        Parameters
        ----------
        Inputs: tensor
            input to the MLP


        Returns
        ----------
        out: tensor
            output from the MLP
        
        """
        out = self.encoder(inputs)
        return out
    
    
    
class TCNBinClfBase(nn.Module):
    
    def __init__(self, params):
               
        """
        @param features: list of features to use
        @param embed_dim: embedding dimension of the input tokens
        @param vocab_lens: vocab size of each categorical features
        @param mlp_layers: list of ints, hidden layers of mlp
        @param latent_dim: dim of the latent vector, the "embedded information"
        @param dropout: dropout
        @param tcn_layers: layers of TCN
        @param mlp_batch_norm: append batch norm 1d for MLP if true
        """
        """ Encoder TCN module. 
        
        Parameters
        ----------
        params: dict
            This dictionary contains all required parameters to initialize this function.
        
            This is a clear example of TECHNICAL DEBT...I should have used kwargs
            instead of a dictionary. Now I do not have enough time to modify and validate
            this function.
            
            keys:
            
            - device: str, {‘cpu’, ‘cuda:0’}, default=‘cpu’
                Whether running on CPU or single GPU
            
            
            - features_num: list of str
                numerical features of the current dataset
            
            
            - features_cat: list of str
                categorical features of the current dataset
            
            
            - features_stnry: list of str
                features to feed concatnate with output of TCN and feed into MLP
                Treat as stationary features
                
                
            - src_col2i: mapping of features to index for self.source 
                (dynamic/time-series data)
                
                
            - stnry_col2i: mapping of features to index for self.stnry 
                (stationary data)
                
                
            - vocab_lens: dict
                Dictionary with key = categorical features and 
                                value = number of distinct categories for that feature
                e.g {'feature_A': 3, 'feature_B': 5}
            
            
            - embed_dims: dict
                Dictionary with key = categorical features and
                                value = dimension of the embedding
            
            
            - tcn_layers: list or list of list
                Number of neurons at each TCN layer. Can be a list or a list of list

                e.g. [100, 100, 100, 100] if using single stack
                     [[100, 100], [100, 100]] if using stacked tcn

                TCN_v1 is built using single stack. Since each TemporalConvNet block
                    comes with residual connections, along with fixed geometric dilution of
                    2**i at layer  i. The advantage of using stacked TCN is to 
                        1) incorporate additional non-linearity to by stacking TemporalConvNet without 
                            residual connections in between and 
                        2) have more densely connected convolutional connections


            - kernel_size: int or list of int
                kernel size, or list of kernel sizes for each stack of tcn.

                int if using single stack. e.g.  5
                list of int if using stacked tcn. e.g. [5, 7]

                If provided a list of kernel sizes, assert(len(kernel_size) == len(num_channels)


            - dropout_tcn: float in [0, 1]
                dropout probability for TCN
                recommend for customer risk model: 0.2
                
                
            - mlp_layers: list of int
                number of neurons for each hidden layers.


            - output_size: int
                Output dimension of the MLP.


            - dropout_mlp: float in [0, 1], default = 1
                dropout probability for MLP
                recommend for customer risk model: 0.5


            - mlp_batch_norm: bool, default=‘False’
                Whether to use Batch Normalization.

        
        Returns
        ----------
        None
        
        
        Modify
        ----------
        params now contains two additional k,v pairs:
        
        src_input_dim: int
            input dimention to TCN after including Embedding dimension 
                for categorical variables.
                
        stnry_input_dim: int
            input dimension to MLP after including stationary features.
            
        """
        super(TCNBinClfBase, self).__init__()
        cuda_available = torch.cuda.is_available()
        self.device = params['device']
    
        if self.device != 'cpu':
            self.device = 'cuda:0'
        elif self.device == 'cpu' and cuda_available:
            print("cuda is available, you should use GPU")

        # set attributes
        self.params = params
        self.features_num = params['features_num']
        self.features_cat = params['features_cat']
        self.features_stnry = params['features_stnry']
        self.src_col2i = params['src_col2i']
        self.stnry_col2i = params['stnry_col2i']
  
        self.vocab_lens = params['vocab_lens']
        self.embed_dims = params['embed_dims']
        
        # set embeddings for categorical features
        for f in list(self.vocab_lens.keys()):
            setattr(self, 'embeds_'+f, nn.Embedding(self.vocab_lens[f], self.embed_dims[f]))
                
        src_input_dim = 0
        stnry_input_dim = 0
        
        for f in self.features_num:
            if f in self.features_stnry:
                stnry_input_dim += 1
            else: 
                src_input_dim += 1
        for f in self.features_cat:
            if f in self.features_stnry:
                stnry_input_dim += self.embed_dims[f]
            else:
                src_input_dim += self.embed_dims[f]
                
        
        self.params['src_input_dim'] = src_input_dim
        self.params['stnry_input_dim'] = stnry_input_dim
        seq_enc_param = {'input_size': self.params['src_input_dim'], 
                         'num_channels': params['tcn_layers'], 
                         'kernel_size': params['kernel_size'],  # k = 4 -> history scope: (3-1) * 2^(4-1) = 32
                         'dropout': params['dropout_tcn']}

        if isinstance(params['tcn_layers'][-1], int):
            # if tcn_layers is a 1d list -> tcn_v1
            in_dim_mlp = params['tcn_layers'][-1] + self.params['stnry_input_dim']
        elif isinstance(params['tcn_layers'][-1][-1], int):
            # if tcn_layers is a 2d list -> tcn_v2 = stacked tcn
            in_dim_mlp = params['tcn_layers'][-1][-1] + self.params['stnry_input_dim']
            
        mlp_param = {'in_dim': in_dim_mlp,
                     'mlp_layers': params['mlp_layers'],
                     'out_dim': params['output_size'],
                     'dropout': params['dropout_mlp'],
                     'batch_norm': params['mlp_batch_norm']}
        
        self.seq_encoder = Encoder_TCN(**seq_enc_param)
        self.linear = MLP(**mlp_param)

        print("\nInitializing a new model\n")
        
        
    def embed_inputs(self, inputs, col2i):
        """ Embed inputs with Embedding 
        
        Inputs contain both numerical features and categorical features with
            integer-indexed categories, and we need to convert the categories 
            into embedding vectors, and the concatnate with the numerical 
            data to construct input to TCN.
            
        
            
        Inputs has dim: batch x seq_len x features 
        
        Embedded inputs has dim: batch x seq_len x features_dim_w_embedding
        
        
        Parameters
        ----------
        inputs: tensor
            Input tensor containing both numerical features and categorical 
                features with integer-indexed categories
                
            shape: batch_size x seq_len x features 
            
            
        col2i: dict
            column to index mapping for inputs
            
            
        Returns
        ----------
        inputs: tensor
            the embedded inputs 
            
            shape: batch_size x seq_len x features_dim_w_embedding
        """
        
        b, s, d = inputs.shape
        inputs_l = []
        
        for feature in col2i:
            i = col2i[feature]
            if feature in self.features_cat:
                embedding = getattr(self, 'embeds_'+feature)
                inputs_l.append(embedding(inputs[:, :, i].long()))
                
            elif feature in self.features_num: 
                inputs_l.append(inputs[:, :, i].reshape(b, s, 1).float())
        
        inputs = torch.cat(tuple(inputs_l), dim=2)
        return inputs
    

    def forward(self, inputs, stnry=None): 
        """ Forward pass for TCN Binary Classifier 
        
        Parameters
        ----------
        inputs: tensor
            Input tensor to TCN containing both numerical features 
                and categorical features with integer-indexed categories
                
            shape: batch_size x seq_len x features 
            
        
        stnry: tensor, None-able
            Input tensor to MLP containing both numerical features 
                and categorical features with integer-indexed categories
            
            shape: batch_size x seq_len x features 
            
            HAVE NOT THOROUGHLY TESTED WHEN THE STATNIONRY FEATURE CONTAINS
            CATEGORICAL VARIABLES. It works for numerical stationary features
        
        
        Returns
        ----------
        out: tensor
            output of the model

        """
        out = self.embed_inputs(inputs, self.src_col2i)
        out = self.seq_encoder(out)
        if stnry is not None:
            stnry = self.embed_inputs(stnry, self.stnry_col2i)
            out = torch.cat([out, stnry], dim=2)
            out = self.linear(out)
        else:
            out = self.linear(out)

        return out