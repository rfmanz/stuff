import tensorflow as tf
from tensorflow.keras import layers, losses, metrics, callbacks, utils
from tensorflow.keras import Sequential, Input


def get_mlp_clf(num_features, num_classes, nhids=[50], 
                activation="relu", dropout=0, output_bias=None):
    layers_ = []
    
    output_bias = None
    if output_bias:
        output_bias = tf.keras.initializers.Constant(output_bias)
    
    # input shape
    layers_.append(Input(shape=(num_features,)))
    
    # intermediate layers
    for nhid in nhids:
        layers_.append(layers.Dense(nhid, activation=activation))
        if dropout > 0:
            layers_.append(layers.Dropout(dropout))
    
    # build network
    layers_.append(layers.Dense(num_classes, activation="softmax",
                                bias_initializer=output_bias))
    model = Sequential(layers_)
    return model
    
def get_default_callbacks():
    es = callbacks.EarlyStopping(monitor='val_loss',
                                 patience=3)
    tb = callbacks.TensorBoard()
    
    callbacks = [es, tb]
    return callbacks

# need target_col, weight_col

def get_initial_bias(target, weight=None):
    """
    compute initial bias for imbalanced binary data
    
    given target and weight array, produce initial bias
    
    ref: https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#optional_set_the_correct_initial_bias
    """
    import numpy as np
    import math
    
    if weight is None:
        weight = np.ones_like(target)
    assert(sorted(list(np.unique(target))) == [0,1])
    pos_w = weight[target.astype(bool)].sum()
    neg_w = weight[~target.astype(bool)].sum()
    b = math.log(pos_w/neg_w)
    return b


###########################
#   TabNet
###########################

from tabnet import TabNet


class TabNetClassifier(tf.keras.Model):

    def __init__(self, feature_columns,
                 num_classes,
                 num_features=None,
                 feature_dim=64,
                 output_dim=64,
                 num_decision_steps=5,
                 relaxation_factor=1.5,
                 sparsity_coefficient=1e-5,
                 norm_type='group',
                 batch_momentum=0.98,
                 virtual_batch_size=None,
                 num_groups=1,
                 epsilon=1e-5,
                 output_bias:float=None,
                 **kwargs):
        """
        """
        super(TabNetClassifier, self).__init__(**kwargs)

        self.num_classes = num_classes

        self.tabnet = TabNet(feature_columns=feature_columns,
                             num_features=num_features,
                             feature_dim=feature_dim,
                             output_dim=output_dim,
                             num_decision_steps=num_decision_steps,
                             relaxation_factor=relaxation_factor,
                             sparsity_coefficient=sparsity_coefficient,
                             norm_type=norm_type,
                             batch_momentum=batch_momentum,
                             virtual_batch_size=virtual_batch_size,
                             num_groups=num_groups,
                             epsilon=epsilon,
                             **kwargs)
        
        if output_bias:
            output_bias = tf.keras.initializers.Constant([output_bias, 
                                                          output_bias])
        self.clf = tf.keras.layers.Dense(num_classes, activation='softmax', 
                                         bias_initializer=output_bias, name='classifier')

    def call(self, inputs, training=None):
        self.activations = self.tabnet(inputs, training=training)
        out = self.clf(self.activations)

        return out

    def summary(self, *super_args, **super_kwargs):
        super().summary(*super_args, **super_kwargs)
        self.tabnet.summary(*super_args, **super_kwargs)