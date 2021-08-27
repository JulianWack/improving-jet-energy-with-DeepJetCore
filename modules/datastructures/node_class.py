class Node():
    '''Tree where a node represents a particle/pseudojet in the clustering process. 
    Leaf nodes represent PF candidates of an event, s.t. their label is the index of the PF candidate.
    The parents if a node are the two particles/pseudojets of the previous clustering stage which have been merged to form a new particle.'''
    
    def __init__(self, label, is_PF = False):
        self.label = label
        self.is_leaf = is_PF
        self.parents = []
        
    def add_parents(self, parent_node1, parent_node2):
        self.parents = [parent_node1, parent_node2]
        
    def get_leaf_labels(self):
        '''Return PF candiates which have been clustered to form this Node'''
        leaf_labels = []
        self._collect_leaf_nodes(self, leaf_labels)
        return leaf_labels
    
    def _collect_leaf_nodes(self, node, leaf_labels):
        if node.is_leaf:
            leaf_labels.append(node.label)
        for n in node.parents:
            self._collect_leaf_nodes(n, leaf_labels)

