import numpy as np
from numpy.lib.recfunctions import append_fields
from numpy.lib.recfunctions import recursive_fill_fields
import uproot
import pyjet
import numba

from node_class import Node
import sum4 as sum4

import mpmath
np_cosh = np.frompyfunc(mpmath.cosh, 1, 1)
np_sinh = np.frompyfunc(mpmath.sinh, 1, 1)
np_log = np.frompyfunc(mpmath.log, 1, 1)



def rapidity(pt, eta, m):
    '''Accepts np.array or scalars for all inputs and returns rapidity in the format of the input'''
    return np_log((np.sqrt(m**2+pt**2*np_cosh(eta)**2) + pt*np_sinh(eta))/np.sqrt(m**2+pt**2))


@numba.vectorize
def anti_kt_dist(pt_i, pt_j, y_i, y_j, phi_i, phi_j, R):
    dphi = phi_i - phi_j
    dR2 = (y_i-y_j)**2 + ((dphi+np.pi)%(2*np.pi)-np.pi)**2
    
    return min(1/pt_i**2, 1/pt_j**2)*dR2/R**2
    
    
@numba.jit(nopython=True)
def keep_mask(pairs, del_idx):
    '''Returns bool array where elements are True if neither of the elements of in pairs is del_idx.
    Hence, apply as mask to extract values to keep'''
    a, b = pairs[0] != del_idx, pairs[1] != del_idx
    return np.logical_and(a,b)
    

    
def anti_kt_clustering(pts, etas, phis, ms, pv_labels, R, ptmin):
    '''Returns 2 lists. The first contains pt eta phi mass 4 vectors (as structured array) of found jets of the single event passed.
    The second list contains lists of the indicies of clustered PF candidates.'''
    jet = [] # 4 vectors of clustered jets
    
    # changing dtype of array elements to mpf for higher precision calculation
    pts = np.array([mpmath.mpf(str(p)) for p in pts])
    etas = np.array([mpmath.mpf(str(e)) for e in etas])
    phis = np.array([mpmath.mpf(str(p)) for p in phis])
    ms = np.array([mpmath.mpf(str(m)) for m in ms])
    
    # make structured array (pt, eta, phi, m, y, node) for every particle in event
    # collection of structured particles stored in event
    ys = rapidity(pts, etas, ms)
    id_nodes = [Node(i, is_PF=True) for i in range(len(pts))]
    event_4vect = np.column_stack((pts, etas, phis, ms)).astype('float64')
    event = np.core.records.fromarrays(event_4vect.transpose(), dtype=pyjet.DTYPE_PTEPM)
    event = append_fields(event, 'y', data=ys, dtypes='f8').data
    # make new data type to add field with Node instances
    dt = event.dtype.descr + [('id_node', 'O')]
    aux = np.empty(event.shape, dt) 
    event = recursive_fill_fields(event, aux)
    event['id_node'] = id_nodes
    
    chs_mask = np.where(pv_labels>0)[0]
    event = event[chs_mask]
    
    
    # Find distances for all initial particles. Afterwards only update values effected by merging or deleting particles
    pairs = np.triu_indices(len(event),k=1) # tuple of 2 arrays: values i and j indicies for upper triangular elements
    anti_kt_distance = anti_kt_dist(event['pT'][pairs[0]], event['pT'][pairs[1]], event['y'][pairs[0]], event['y'][pairs[1]], event['phi'][pairs[0]], event['phi'][pairs[1]], R)
    beam_distance = 1/event['pT']**2


    while len(event) > 1:   
        # Dealing with degenerate minima
        # choose particle pair with max pt
        min_anti_kt_d = np.min(anti_kt_distance)
        anti_kt_dmin_pair_idxs = np.where(anti_kt_distance==min_anti_kt_d)[0]
        if len(anti_kt_dmin_pair_idxs) > 1:
            pts_potential_pairs = event['pT'][pairs[0][anti_kt_dmin_pair_idxs]] + event['pT'][pairs[1][anti_kt_dmin_pair_idxs]]
            chosen_pair_id = np.argmax(pts_potential_pairs)
            min_id = anti_kt_dmin_pair_idxs[chosen_pair_id]
        else:
            min_id = anti_kt_dmin_pair_idxs[0]
            
        min_ij = [pairs[0][min_id], pairs[1][min_id]] # guarentees smaller index always first  
       
        # choose particle with larger mass
        min_beam_d = np.min(beam_distance)
        beam_dmin_idxs = np.where(beam_distance==min_beam_d)[0]
        if len(beam_dmin_idxs) > 1:
            chosen_min = np.argmax(event['mass'][beam_dmin_idxs])
            min_idx = beam_dmin_idxs[chosen_min]
        else:
            min_idx = beam_dmin_idxs[0]

            
        
        if (min_beam_d < min_anti_kt_d): # declare particle as jet if pt>=ptmin
            if event[min_idx]['pT'] >= ptmin:
                jet.append(event[min_idx])
            event = np.delete(event, min_idx, axis=0)
            # update pairs and distances
            particles_to_keep = keep_mask(pairs, min_idx)
            pairs = np.triu_indices(len(event),k=1)
            anti_kt_distance = anti_kt_distance[particles_to_keep] 
            beam_distance = np.delete(beam_distance, min_idx)
        else: # merge two particles together
            # find new 4 vector
            p1, p2 = event[min_ij[0]].tolist()[:4], event[min_ij[1]].tolist()[:4]
            aux = np.column_stack((p1,p2))
            new_vec4 = sum4.np_sum4vec(aux[0], aux[1], aux[2], aux[3])
            new_y = rapidity(new_vec4[0], new_vec4[1], new_vec4[3])
            # make new node
            parent_node1, parent_node2 = event['id_node'][min_ij[0]], event['id_node'][min_ij[1]] 
            new_node = Node(label=parent_node1.label) # smaller of the two parent labels
            new_node.add_parents(parent_node1, parent_node2)
            # change particle with smaller id to merged one
            event[min_ij[0]] = tuple(np.concatenate((new_vec4, [new_y, new_node])))
            
            # updates distances due to merge
            beam_distance[min_ij[0]] = 1/new_vec4[0]**2
            mask = keep_mask(pairs, min_ij[0])
            new_d_needed = np.invert(mask)
            p, q = pairs[0][new_d_needed], pairs[1][new_d_needed]
            # Only compute new anti-kt distance for pairs effected by merge of two particles. The remaining distances remain unchanged 
            # Needs improvement as distance to particle that will be deletet is still computed.
            anti_kt_distance[new_d_needed] = anti_kt_dist(event['pT'][p], event['pT'][q], event['y'][p], event['y'][q], event['phi'][p], event['phi'][q], R)

            # delete particle with smaller id
            event = np.delete(event, min_ij[1], axis=0)
            particles_to_keep = keep_mask(pairs, min_ij[1])
            pairs = np.triu_indices(len(event),k=1)
            anti_kt_distance = anti_kt_distance[particles_to_keep] 
            beam_distance = np.delete(beam_distance, min_ij[1])


    # deal with last particle
    if event[0]['pT'] >= ptmin:
        jet.append(event[0])
    event = np.delete(event, 0, axis=0)
            

    # deal with case when no jets exist
    if len(jet) == 0:
        return [], []     
    
    # TypeError occurs for event 67, 74, 138. Problematic is only the last element: TypeError: '<' not supported between instances of 'Node' and 'Node'
    try:
        jet = np.sort(np.array(jet), order=['pT', 'mass'])[::-1]
    except TypeError:
        print("-----\nSort error\n-----")
        jet_pts = [jet[0] for jet in jet]
        sorting_idxs = np.argsort(jet_pts)
        jet = np.array(jet)[sorting_idxs][::-1]
    

    # find PF constituents of every jet
    jets = np.empty(len(jet), dtype=pyjet.DTYPE_PTEPM)
    jet_consti = []
    for i, jet in enumerate(jet):
        jet_consti.append(np.sort(jet['id_node'].get_leaf_labels()))
        jets[i] = jet[['pT', 'eta', 'phi', 'mass']]
        
    
    return jets, jet_consti

