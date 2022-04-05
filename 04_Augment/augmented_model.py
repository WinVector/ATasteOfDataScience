

# build our neural net factory
# this one implements an augmented logistic regression.


import pandas
import numpy
from functools import partial

# set up Keras imports, this can be brittle
# no longer import keras, import tensorflow.keras
from tensorflow.keras import Model
from tensorflow.keras.layers import Concatenate, Dense, Input, Multiply, Softmax


def mk_model_template(
    *, 
    simple_model: bool = False,
    black_box_model: bool = False,
    n_hidden_models: int = 0,
    encode_dim: int,
    ):
    """
    This is an augmented logistic model.
    It predicts a 2-vector [p_class, p_class_augmented] or a 4-vector [p_class, p_class_augmented, certainty, uncertainty].
    The augmentation is, smoothing to 0.5 by the degree of ambiguousness
    """
    inputs = Input(shape=(encode_dim, ), name='inputs')
    # model0: basic logistic regression
    logistic_model0 = Dense(1, name='logistic_model0', activation='sigmoid')(inputs)
    if simple_model:
        results = logistic_model0
    else:
        # mix: sensor fusion style selection between logistic_model0 and 0.5
        half_value = Dense(1, name='half_value')(logistic_model0)  # set weights to: multiply by zero and add 0.5 to get 0.5
        hidden_models = [Dense(1, name=f'hidden_logistic_model_{i}', activation='sigmoid')(inputs) for i in range(n_hidden_models)]
        sub_models = [logistic_model0, half_value] + hidden_models
        n_sub_models = len(sub_models)
        models_layer = Concatenate(name='models_layer')(sub_models)
        # build weighed fusion of sub-models
        fuse_link = Dense(n_sub_models, name='fuse_link')(inputs)
        fuse_weights = Softmax(name='fuse_weights')(fuse_link)
        fused_prediction = Multiply(name='fused_prediction')([models_layer, fuse_weights])
        fused_model = Dense(1, name='fused_model')(fused_prediction)  # will set weights non-trainable to get effect we want
        if black_box_model:
            results = fused_model
        else:
            results = Concatenate(name='results')([fused_model, logistic_model0])
        # https://keras.io/api/models/model/
        logistic_model = Model(inputs, results)
        # make max sum add with weight 1 and no bias
        fused_model_layer = logistic_model.get_layer('fused_model')
        fused_model_layer.set_weights([numpy.array([[1.0]] * n_sub_models), numpy.array([0.0])])
        fused_model_layer.trainable = False
        # form constant 0.5 as alternate model
        half_value_layer = logistic_model.get_layer('half_value')
        half_value_layer.set_weights([numpy.array([[0.0]]), numpy.array([0.5])])
        half_value_layer.trainable = False
    logistic_model.compile(
        optimizer='rmsprop',
        loss='binary_crossentropy')
    return logistic_model


def augment_training_output_template(
    v,
    *, 
    simple_model: bool = False,
    black_box_model: bool = False,
    ):
    """Augment training target"""
    if black_box_model or simple_model:
        return [v]
    return [v, v]


def unwrap_predictions_template(
    preds: pandas.DataFrame,
    *, 
    simple_model: bool = False,
    black_box_model: bool = False,
    ) -> pandas.DataFrame:
    preds_frame = pandas.DataFrame(preds).copy()
    if black_box_model or simple_model:
        assert preds_frame.shape[1] == 1
        # just raw prediction or block box prediction
        preds_frame.columns = ['prediction']
        preds_frame['prediction_augmented'] = preds_frame['prediction']
    else:
        assert preds_frame.shape[1] == 2
        # augmented prediction and basic prediction
        preds_frame.columns = ['prediction_augmented', 'prediction']
    return preds_frame



def mk_modeling_fns(
    *, 
    simple_model: bool = False,
    black_box_model: bool = False,
    n_hidden_models: int = 0,):
    return (
        partial(mk_model_template, 
            simple_model=simple_model,
            black_box_model=black_box_model,
            n_hidden_models=n_hidden_models),
        partial(augment_training_output_template, 
            simple_model=simple_model,  
            black_box_model=black_box_model),
        partial(unwrap_predictions_template, 
            simple_model=simple_model, 
            black_box_model=black_box_model),
    )

