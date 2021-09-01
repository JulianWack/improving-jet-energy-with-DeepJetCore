from DeepJetCore.TrainData import TrainData, fileTimeOut
from DeepJetCore import SimpleArray

import numpy as np
import uproot
from matching import matching


class TrainData_jet(TrainData):
    def __init__(self):
        TrainData.__init__(self)
        # no class member is mandatory
        self.description = "This is a TrainData example file. Having a description string is not a bad idea (but not mandatory), e.g. for describing the array structure."
        #define any other (configuration) members that seem useful
        self.someusefulemember = "something you might need later"
            
        
    def convertFromSourceFile(self, filename, weighterobjects, istraining):
        '''Converts data from input ROOT file to network inputs and desired output. These are [feature_array, consti_pts_array] and [truth_array] respectively.
        The feature array is an ndarray of shape (# jets, largest # of constituents in a jet, 4), detailing 4 attributes of every constituent of every jet.
        To later compute the jet pt (using the constituent pt * correction factor which we aim to determine through the DNN for every particle) in the loss function,
        the (# jets, largest # of constituents in a jet) ndarray calles consti_pts_array is also returned. Note that to assure the second dimension is always the same,
        padding with zeros is used. Also not that only recojets which have a corresponding genjet are considered. The pt of the genjets in stored in truth_array.
        '''
        
        max_idx = 50
        print('Reading {} events from {}'.format(max_idx, filename))
        event_tree = uproot.open(filename)['Events']
        

        pts = event_tree['PF_pt'].array(entry_stop=max_idx, library='np')
        etas = event_tree['PF_eta'].array(entry_stop=max_idx, library='np')
        phis = event_tree['PF_phi'].array(entry_stop=max_idx, library='np')
        masses = event_tree['PF_mass'].array(entry_stop=max_idx, library='np')

        recojet_pts = event_tree['Jet_pt'].array(entry_stop=max_idx, library='np')
        recojet_etas = event_tree['Jet_eta'].array(entry_stop=max_idx, library='np')
        recojet_phis = event_tree['Jet_phi'].array(entry_stop=max_idx, library='np')
        recojet_masses = event_tree['Jet_mass'].array(entry_stop=max_idx, library='np')
        #raw_pt_factor = 1-event_tree['Jet_rawFactor'].array(entry_stop=max_idx, library='np')
        recojet_nconsti = event_tree['Jet_nConstituents'].array(entry_stop=max_idx, library='np')
        recojet_consti = event_tree['PF_jetsIdx'].array(entry_stop=max_idx, library='np')

        genjet_pts = event_tree['GenJet_pt'].array(entry_stop=max_idx, library='np')
        genjet_etas = event_tree['GenJet_eta'].array(entry_stop=max_idx, library='np')
        genjet_phis = event_tree['GenJet_phi'].array(entry_stop=max_idx, library='np')
        genjet_masses = event_tree['GenJet_mass'].array(entry_stop=max_idx, library='np')
        
        
        
        def truth_vals(match, genjet_pt):
            '''Returnes array with elements being associated genjet_pt when matching !=-1.'''
            recojet_mask = np.where(match!=-1)[0] # gives recojet idx for which genjet partner exists
            genjet_mask = match[recojet_mask] # gives genjet idx for which recojet partner exists
            truth = np.zeros(len(match))
            truth[recojet_mask] = genjet_pt[genjet_mask]

            return truth
        
        
        # find jets and fill feature and truth array
        features, truth = [], []
        consti_pts = []
        max_nconsti = 0
        for i in range(max_idx):
            reco_gen_match = matching(recojet_etas[i], recojet_phis[i], genjet_etas[i], genjet_phis[i])
            truth.extend(truth_vals(reco_gen_match, genjet_pts[i]))
            
            # update maximum number of jet constituents
            max_nconsti_this_event = max(recojet_nconsti[i])
            if max_nconsti_this_event > max_nconsti:
                max_nconsti = max_nconsti_this_event
                
            for j in range(len(recojet_nconsti[i])): # iterate through all recojets
                PF_props = np.ndarray(shape=(recojet_nconsti[i][j],4), dtype='float32', order='C')
                PF_idxs = np.where(recojet_consti[i]==j)[0]
                for k, PF_idx in enumerate(PF_idxs):
                    PF_props[k][0] = pts[i][PF_idx]
                    PF_props[k][1] = pts[i][PF_idx]/recojet_pts[i][j]
                    PF_props[k][2] = recojet_etas[i][j] - etas[i][PF_idx]
                    dphi = recojet_phis[i][j] - phis[i][PF_idx]
                    PF_props[k][3] = (dphi+np.pi)%(2*np.pi)-np.pi
                
                features.append(PF_props)
                consti_pts.append(PF_props[:,0])
                
                
        # make mask to only condsider jets for which recojet, genjet pair exists
        keep_mask = np.where(np.array(truth)!=0, True, False)
        # pad all elements of features to get same length. Also pad the arrays containing the pt of the clustered PF with zeros
        feature_array = np.empty((len(features),max_nconsti,4), dtype='float32', order='C')
        consti_pts_array = np.empty((len(features),max_nconsti), dtype='float32', order='C')
        for i, (fea, c) in enumerate(zip(features, consti_pts)):
            dn = max_nconsti - len(fea)
            if dn > 0:
                feature_array[i] = np.vstack((fea, np.zeros((dn,4), dtype='float32', order='C')))
                consti_pts_array[i] = np.concatenate((c, np.zeros(dn)))
            else:
                feature_array[i] = fea
                consti_pts_array[i] = c
                
        feature_array = feature_array[keep_mask]
        consti_pts_array = np.expand_dims(consti_pts_array[keep_mask], axis=2)
        
        # make truth array
        truth_array = np.array(truth, dtype='float32', order='C')
        truth_array = truth_array[keep_mask]
        #truth_array = np.expand_dims(truth_array[keep_mask], axis=1)
        
        
        print('feature_array',feature_array.shape)
        #self.nsamples = len(features)
        
        #returns a list of feature arrays, a list of truth arrays and a list of weight arrays
        return [SimpleArray(feature_array,name="features"), SimpleArray(consti_pts_array,name="consti_pt")], [SimpleArray(truth_array,name="truth")], []
        #return [feature_array, consti_pts_array], [truth_array], [] 
     
       
    def writeOutPrediction(self, predicted, features, truth, weights, outfilename, inputfile):
        '''Defines how to write out the prediction. Here will be a list of probabilities.'''
        from root_numpy import array2root
        out = np.core.records.fromarrays(predicted[0].transpose(), names='prob_isA, prob_isB, prob_isC')
        
        array2root(out, outfilename, 'tree')
