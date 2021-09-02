
# Define custom losses here and add them to the global_loss_list dict (important!)
import tensorflow as tf
global_loss_list = {}


def my_loss(y_true, y_pred):
    '''Finds the square difference between the genjet pt and the jet pt obtained by summing the products of constituent pt and network output.
    The loss is then found by the mean of the square errors for all jets.'''
    # note that network output is (batch size, # jets, # units in final dense layer+1)
    # the +1 is due to the PF pt passed thoruh the network, which will always be the last element
    pt_correction_factor = tf.reduce_mean(y_pred[:,:,0:-1], axis=2)
    consti_pt = y_pred[:,:,-1]
    consti_pt_corrected = consti_pt*pt_correction_factor
    jet_pt = tf.reduce_sum(consti_pt_corrected, axis=1)
    
    loss = ((jet_pt-y_true)/y_true)**2
    
    return loss


global_loss_list['my_loss'] = my_loss
