
# Define custom losses here and add them to the global_loss_list dict (important!)
import tensorflow as tf
global_loss_list = {}


def my_loss(y_true, y_pred):
    '''Finds the square difference between the genjet pt and the jet pt obtained by summing the products of constituent pt and network output.
    The loss is then found by the mean of the square errors for all jets.'''
    
    pt_correction_factor = y_pred[:,:,0]
    consti_pt = y_pred[:,:,1]
    consti_pt_corrected = consti_pt*pt_correction_factor
    jet_pt = tf.reduce_sum(consti_pt_corrected, axis=1)
    
    loss = (jet_pt-y_true)**2
    
    return tf.reduce_mean(loss)


global_loss_list['my_loss'] = my_loss
