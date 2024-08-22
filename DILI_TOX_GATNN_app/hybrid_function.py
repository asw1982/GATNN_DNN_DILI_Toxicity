# -*- coding: utf-8 -*-
# import packages
# general tools
import pandas as pd
import numpy as np
# RDkit
from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
# RDkit
from rdkit.Chem import MACCSkeys
from rdkit.Chem import rdMolDescriptors
from rdkit import DataStructs

# Pytorch and Pytorch Geometric
import torch

import torch.nn as nn
from torch.nn import Linear
import torch.optim as optim
import torch.nn.functional as F # activation function
from torch.utils.data import Dataset
from torch.utils.data import DataLoader as V_Loader # dataset management
import torchvision.datasets as datasets #bank of dataset
import torchvision.transforms as transforms #can create pipeline to preprocess da

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as G_Loader 
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import BatchNorm


import statistics
#from prettytable import PrettyTable

# performances visualization 
#import matplotlib.pyplot as plt

# PREPARE THE DATASET 
# DATASETS ARE SEPARATED INTO 3 DATA : DATA_TRAIN, DATA_VALIDATION, DATA_TEST (INDEPENDENT DATA_SET)

# THESE ARE MODULES USED TO GENERATE GRAPH STRUCTURED DATASET 
#==============================================================================================================================
def compute_fingerprint_MACCS(one_smiles_data):
    """
    Compute ECFP2 & PubChem fingerprint features for a list 
    of SMILES strings

    Parameters
    ----------
    smiles_list: List[str]
        The list of SMILES strings.

    Returns
    -------
    np.ndarray
        Returns a 2D numpy array, where each row corrsponds
        to the fingerprints of a SMILES strings in order.
    """
    molecular_mols = Chem.MolFromSmiles(one_smiles_data)
    # Initialize an array to store ECFP2 & PubChem fingerprint features
   
    one_fingerp =MACCSkeys.GenMACCSKeys(molecular_mols)
    list_one_fingerp = list(one_fingerp)
    del list_one_fingerp[0]
    
    return list_one_fingerp

def one_hot_encoding(x, permitted_list):
    """
    Maps input elements x which are not in the permitted list to the last element
    of the permitted list.
    """
    if x not in permitted_list:
        x = permitted_list[-1]
    binary_encoding = [int(boolean_value) for boolean_value in list(map(lambda s: x == s, permitted_list))]
    return binary_encoding

def get_atom_features(atom, 
                      use_chirality = True, 
                      hydrogens_implicit = True):
    """
    Takes an RDKit atom object as input and gives a 1d-numpy array of atom features as output.
    """
    # define list of permitted atoms
    
    permitted_list_of_atoms =  ['C','N','O','S','F','Si','P','Cl','Br','Mg','Na','Ca','Fe','As','Al','I', 'B','V','K','Tl','Yb','Sb','Sn','Ag','Pd','Co','Se','Ti','Zn', 'Li','Ge','Cu','Au','Ni','Cd','In','Mn','Zr','Cr','Pt','Hg','Pb','Unknown']
    
    if hydrogens_implicit == False:
        permitted_list_of_atoms = ['H'] + permitted_list_of_atoms
    
    # compute atom features
    
    atom_type_enc = one_hot_encoding(str(atom.GetSymbol()), permitted_list_of_atoms)
    
    n_heavy_neighbors_enc = one_hot_encoding(int(atom.GetDegree()), [0, 1, 2, 3, 4, "MoreThanFour"])
    
    formal_charge_enc = one_hot_encoding(int(atom.GetFormalCharge()), [-3, -2, -1, 0, 1, 2, 3, "Extreme"])
    
    hybridisation_type_enc = one_hot_encoding(str(atom.GetHybridization()), ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER"])
    
    is_in_a_ring_enc = [int(atom.IsInRing())]
    
    is_aromatic_enc = [int(atom.GetIsAromatic())]
    
    atomic_mass_scaled = [float((atom.GetMass() - 10.812)/116.092)]
    
    vdw_radius_scaled = [float((Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()) - 1.5)/0.6)]
    
    covalent_radius_scaled = [float((Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum()) - 0.64)/0.76)]
    atom_feature_vector = atom_type_enc + n_heavy_neighbors_enc + formal_charge_enc + hybridisation_type_enc + is_in_a_ring_enc + is_aromatic_enc + atomic_mass_scaled + vdw_radius_scaled + covalent_radius_scaled
                                    
    if use_chirality == True:
        chirality_type_enc = one_hot_encoding(str(atom.GetChiralTag()), ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW", "CHI_OTHER"])
        atom_feature_vector += chirality_type_enc
    
    if hydrogens_implicit == True:
        n_hydrogens_enc = one_hot_encoding(int(atom.GetTotalNumHs()), [0, 1, 2, 3, 4, "MoreThanFour"])
        atom_feature_vector += n_hydrogens_enc
    return np.array(atom_feature_vector)

def get_bond_features(bond, 
                      use_stereochemistry = True):
    """
    Takes an RDKit bond object as input and gives a 1d-numpy array of bond features as output.
    """
    permitted_list_of_bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
    bond_type_enc = one_hot_encoding(bond.GetBondType(), permitted_list_of_bond_types)
    
    bond_is_conj_enc = [int(bond.GetIsConjugated())]
    
    bond_is_in_ring_enc = [int(bond.IsInRing())]
    
    bond_feature_vector = bond_type_enc + bond_is_conj_enc + bond_is_in_ring_enc
    
    if use_stereochemistry == True:
        stereo_type_enc = one_hot_encoding(str(bond.GetStereo()), ["STEREOZ", "STEREOE", "STEREOANY", "STEREONONE"])
        bond_feature_vector += stereo_type_enc
    return np.array(bond_feature_vector)

def create_pytorch_geometric_graph_data_list_from_smiles_and_labels(x_smiles, y):
    """
    Inputs:
    
    x_smiles = [smiles_1, smiles_2, ....] ... a list of SMILES strings
    y = [y_1, y_2, ...] ... a list of numerial labels for the SMILES strings (such as associated pKi values)
    
    Outputs:
    
    data_list = [G_1, G_2, ...] ... a list of torch_geometric.data.Data objects which represent labeled molecular graphs that can readily be used for machine learning
    
    """
    
    data_list = []
    list_smiles_error= []
    num_error = 0
    for (smiles, y_val) in zip(x_smiles, y):
        
        try :
        # convert SMILES to RDKit mol object
            mol = Chem.MolFromSmiles(smiles)

        # get feature dimensions
            n_nodes = mol.GetNumAtoms()
            n_edges = 2*mol.GetNumBonds()
            unrelated_smiles = "O=O"
            unrelated_mol = Chem.MolFromSmiles(unrelated_smiles)
            n_node_features = len(get_atom_features(unrelated_mol.GetAtomWithIdx(0)))
            n_edge_features = len(get_bond_features(unrelated_mol.GetBondBetweenAtoms(0,1)))

        # construct node feature matrix X of shape (n_nodes, n_node_features)
            X = np.zeros((n_nodes, n_node_features))

            for atom in mol.GetAtoms():
                X[atom.GetIdx(), :] = get_atom_features(atom)
            
            X = torch.tensor(X, dtype = torch.float)
        
        # construct edge index array E of shape (2, n_edges)
            (rows, cols) = np.nonzero(GetAdjacencyMatrix(mol))
            torch_rows = torch.from_numpy(rows.astype(np.int64)).to(torch.long)
            torch_cols = torch.from_numpy(cols.astype(np.int64)).to(torch.long)
            E = torch.stack([torch_rows, torch_cols], dim = 0)
        
        # construct edge feature array EF of shape (n_edges, n_edge_features)
            EF = np.zeros((n_edges, n_edge_features))
        
            for (k, (i,j)) in enumerate(zip(rows, cols)):
            
                EF[k] = get_bond_features(mol.GetBondBetweenAtoms(int(i),int(j)))
        
            EF = torch.tensor(EF, dtype = torch.float)
        
        # construct label tensor
            y_tensor = torch.tensor(np.array([y_val]), dtype = torch.float)
        
        # construct Pytorch Geometric data object and append to data list
            data_list.append(Data(x = X, edge_index = E, edge_attr = EF, y = y_tensor))
        except: 
            num_error = num_error+1
            print (num_error)
            list_smiles_error.append(smiles)
    return data_list

from typing import Tuple, List
def compute_fingerprint_MACCS2(smiles_list: List[str]) -> np.ndarray:
    """
    Compute ECFP2 & PubChem fingerprint features for a list 
    of SMILES strings

    Parameters
    ----------
    smiles_list: List[str]
        The list of SMILES strings.

    Returns
    -------
    np.ndarray
        Returns a 2D numpy array, where each row corrsponds
        to the fingerprints of a SMILES strings in order.
    """
    molecular_mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]
    # Initialize an array to store ECFP2 & PubChem fingerprint features
    features = np.zeros((len(smiles_list), 166), dtype=np.int32)

    for i, mol in enumerate(molecular_mols):
        one_fingerp =MACCSkeys.GenMACCSKeys(mol)
        list_one_fingerp = list(one_fingerp)
        del list_one_fingerp[0]
        numerical_representation = np.array(list_one_fingerp,dtype=np.int64)
        features[i] = numerical_representation

    return features


# CREATE THE GRAPH NEURAL NETWORK MODEL 
# THE PARAMETER WHICH ARE USED IN THIS MODEL => HIDDEN CHANNEL , NUM_LAYER , DROP_OUT PERCENTAGE 
#==========================================================================
class modelA(torch.nn.Module):
    def __init__(self, hidden_channels1,hidden_channels2, num_node_features,heads1,heads2
                 ,dropout_rateA,dropout_rateB, dropout_rateC,dense_layer1):
        super(modelA, self).__init__()
        
        torch.manual_seed(12345)
        self.conv1 = GATConv(num_node_features, hidden_channels1,heads1)
        self.conv2 = GATConv(hidden_channels1*heads1,hidden_channels2, heads2)
        
        self.bn1 = BatchNorm (hidden_channels1*heads1)
        self.bn2 = BatchNorm (hidden_channels2*heads2)
        
        self.dropoutA = dropout_rateA
        self.dropoutB = dropout_rateB
        self.dropoutC = dropout_rateC
        
        self.lin1 = Linear(hidden_channels2*heads2,dense_layer1)
        #self.lin1 = Linear(hidden_channels2*heads2,dense_layer1)
        self.lin2=  Linear(dense_layer1,1)
        
    def forward(self, x, edge_index, batch):

        x = self.conv1(x, edge_index)
        
        x = F.dropout(x, p=self.dropoutA , training=self.training)
        x = x.relu()
        x = self.bn1(x)
        x = self.conv2(x, edge_index)
        
        x = F.dropout(x, p=self.dropoutB , training=self.training)
        x = x.relu()
        x = self.bn2(x)  
        
        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        x = self.lin1(x)
        x = F.dropout(x, p=self.dropoutC , training=self.training)
        x = x.relu()
        x = self.lin2(x)
        return torch.sigmoid(x)


class modelB(torch.nn.Module):
    def __init__(self, input_features, output_features,dropout_rateB1,dropout_rateB2,dropout_rateB3,  
                 dense_layer1,dense_layer2, dense_layer3):
        super(modelB, self).__init__()
        self.lin1 = nn.Linear(input_features,dense_layer1)
      
        self.lin2 = nn.Linear(int(dense_layer1), dense_layer2)
        self.lin3 = nn.Linear(int(dense_layer2), dense_layer3)
        self.lin4 = nn.Linear(int(dense_layer3), output_features)
        
        self.bn1 = nn.BatchNorm1d(int(dense_layer1))
        self.bn2 = nn.BatchNorm1d(int(dense_layer2))
        self.bn3 = nn.BatchNorm1d(int(dense_layer3))
        self.dropoutB1 = dropout_rateB1
        self.dropoutB2 = dropout_rateB2
        self.dropoutB3 = dropout_rateB3
        
    def forward(self, x):
        x = self.lin1(x)
        x = self.bn1(x)
       
        x = F.dropout(x, p= self.dropoutB1, training=self.training)
        x = x.relu()
  #      
        x = self.lin2(x)
        x = self.bn2(x)   
        x = F.dropout(x, p= self.dropoutB2, training=self.training)
        x = x.relu()
  #      
        x = self.lin3(x)
        x = self.bn3(x)   
        x = F.dropout(x, p= self.dropoutB3, training=self.training)

        x = x.relu()
        x = self.lin4(x)
        return torch.sigmoid(x)        
    
class Combined_model(nn.Module):
    def __init__(self, modelA, modelB, total_input_features, num_classes,dropout_rate_C, dense_layer_C):
        super(Combined_model, self).__init__()
        self.model_1 = modelA
        self.model_2 = modelB
        
        
        self.lin1 = Linear(total_input_features,int(dense_layer_C))
        self.lin2 = Linear(int(dense_layer_C),num_classes)
        
        self.dropout_C = dropout_rate_C
        
    def forward(self, x1,edge_index, batch, x2):
        xa = self.model_1(x1, edge_index, batch) # x1 node features in graph
        xb= self.model_2(x2)                     # x2 is a vector features 
  #      xc= self.model_3(x3)                     # x2 is a vector features 
        
        x = torch.cat((xa, xb), dim=1)
  #      x = torch.cat((x, xc), dim=1)
         # 3. Apply a final classifier
        x = self.lin1(x)
        x = F.dropout(x, p= self.dropout_C, training=self.training)
        x = x.relu()
        x = self.lin2(x)
        return torch.sigmoid(x)


# Not count the performance metrics only the outcome of prediction
def test_1(g_loader,f_loader,combined_model):
    combined_model.eval()
    list_pred =[]
    list_targets =[]
    correct = 0
    for data_X1, data_X2 in zip(g_loader,f_loader): # Iterate in batches over the training dataset.
        out = combined_model(data_X1.x, data_X1.edge_index, data_X1.batch, 
                             torch.tensor(data_X2, dtype=torch.float32))  # Perform a single forward pass.
        out_1 = out[:,0] 
        list_pred.append(out_1.item())
        list_targets.append(data_X1.y.item())
    return list_pred, list_targets 


# create empty model with the hyperparameter 
nCV = 10

hyper_param_d = {'dropout_rateB1': 0.19029920767006717,
               'dropout_rateB2': 0.12142154817498285,
               'dropout_rateB3': 0.30892320125476364,
               'dense_layer1'  : 180,
               'dense_layer2'  : 96,
               'dense_layer3'  : 40,
               'learning_rate' : 0.00014475405687104653,
               'weight_decay'  : 0.0001839631656485908}

hyper_param_g = {'hidden_channels1': 82,
                 'hidden_channels2': 82,
                 'num_node_features': 79,
                 'heads1':9,
                 'heads2':9,
                 'dropout_rateA':0.20184714781996305,
                 'dropout_rateB':0.26157496401430025,
                 'dropout_rateC':0.12826012269932247,
                 'dense_layer1':64}

hyper_param_c = {'total_input_features':2,
                 'num_classes':1,
                 'dropout_rate_C':0.3054920980974637,
                 'dense_layer_C':16}


# load the combined model 
# load the combined model 
k =10
list_modelA = []
list_modelB = []
list_modelC = []

# load model B
for i in range(k):
    
    input_features    = 166 # length of feature data vector 
    output_features   = 1
    
    dropout_rateB1 =hyper_param_d['dropout_rateB1']
    dropout_rateB2 =hyper_param_d['dropout_rateB2']
    dropout_rateB3 =hyper_param_d['dropout_rateB3']
    dense_layer1 = hyper_param_d['dense_layer1']
    dense_layer2 = hyper_param_d['dense_layer2']
    dense_layer3 = hyper_param_d['dense_layer3']
 
        
    model_b= modelB(input_features, output_features,dropout_rateB1,dropout_rateB2,dropout_rateB3,  
                 dense_layer1,dense_layer2, dense_layer3)
#    PATH = '0.39289model_fingerp'+ str(i)+'.pth'
#    model_b.load_state_dict(torch.load(PATH))
    
    list_modelB.append(model_b)

# load model A
for i in range(k):
    hidden_channels1=hyper_param_g['hidden_channels1']
    hidden_channels2=hyper_param_g['hidden_channels2']
    num_node_features=hyper_param_g['num_node_features']
    heads1=hyper_param_g['heads1']
    heads2=hyper_param_g['heads2']
    dropout_rateA=hyper_param_g['dropout_rateA']
    dropout_rateB=hyper_param_g['dropout_rateB']
    dropout_rateC=hyper_param_g['dropout_rateC']
    dense_layer1=hyper_param_g['dense_layer1']
    
    model_a  = modelA(hidden_channels1,hidden_channels2, num_node_features,heads1,heads2
                 ,dropout_rateA,dropout_rateB, dropout_rateC,dense_layer1)
   
    list_modelA.append(model_a)
    
# load model C (combined model)
for i in range(k):
    
    model_a  = list_modelA[i]
        
    model_b  = list_modelB[i] # model b is pretrained model trainable = off 
    
    total_input_features=hyper_param_c['total_input_features']
    num_classes=hyper_param_c['num_classes']
    dropout_rate_C=hyper_param_c['dropout_rate_C']
    dense_layer_C=hyper_param_c[ 'dense_layer_C']
    
    combined_model = Combined_model(model_a, model_b, total_input_features, num_classes,dropout_rate_C, dense_layer_C) 
    
    PATH = '0.75144model_hybrid'+ str(i)+'.pth'
    combined_model.load_state_dict(torch.load(PATH))
    
    list_modelC.append(combined_model)
# test the model
#=======================================================================

def smiles_to_DILI(smiles_string):
    
    y = [3] # random value 
    x_smiles = [smiles_string]
    data_list_graph_test = create_pytorch_geometric_graph_data_list_from_smiles_and_labels(x_smiles, y)
    
    data_list_vector_test = compute_fingerprint_MACCS2(x_smiles)
    
    test_loader1 = G_Loader(dataset = data_list_graph_test, batch_size = 1)
    test_loader2 = V_Loader(dataset = data_list_vector_test, batch_size = 1)
#    print(data_list_vector_test)
    
    nCV= 10 # ten crossfold validation 
    list_fold_pred =[]
    list_fold_targets =[]
    
    for i, hybrid_model in enumerate(list_modelC):
        #test_acc = test(test_loader,gnn_model)
        list_pred,_ = test_1(test_loader1, test_loader2,hybrid_model)
        list_fold_pred.append(list_pred[0])
    mean_pred = statistics.mean(list_fold_pred)
    if mean_pred > 0.5 :
        return 'DILI Toxic (' + str(mean_pred) +')'
    else :
        return 'Non-DILI Toxic (' + str(mean_pred) +')' 