import os
import pdb
import random
import torch
import numpy as np
import rdkit
import rdkit.Chem
from rdkit import Chem
from rdkit.Geometry import Point3D
from rdkit import Geometry


# should redesign to make exactly analogous to ROCS_shape_overlap
def shape_align(reference, query, shaep_path = '../../software', remove_files = True, ID = ''):
    
    if not os.path.exists('shaep_objects_temp'):
        os.makedirs('shaep_objects_temp')
    
    job_number = random.randint(0, 10000000)
    
    Chem.MolToMolFile(reference, f'shaep_objects_temp/mol_ref_shaep_{ID}_{job_number}.mol')
    Chem.MolToMolFile(query, f'shaep_objects_temp/mol_query_shaep_{ID}_{job_number}.mol')
    
    os.system(f"{shaep_path}/shaep --onlyshape -q shaep_objects_temp/mol_ref_shaep_{ID}_{job_number}.mol shaep_objects_temp/mol_query_shaep_{ID}_{job_number}.mol -s shaep_objects_temp/query_mol_shaep_{ID}_{job_number}.sdf shaep_objects_temp/shapesimilarity_shaep_{ID}_{job_number}.txt >/dev/null 2>&1")

    suppl = rdkit.Chem.rdmolfiles.ForwardSDMolSupplier(f'shaep_objects_temp/query_mol_shaep_{ID}_{job_number}.sdf')
    mol = next(suppl)
    
    if remove_files:
        os.system(f'rm shaep_objects_temp/mol_ref_shaep_{ID}_{job_number}.mol shaep_objects_temp/mol_query_shaep_{ID}_{job_number}.mol shaep_objects_temp/query_mol_shaep_{ID}_{job_number}.sdf shaep_objects_temp/shapesimilarity_shaep_{ID}_{job_number}.txt shaep_objects_temp/shapesimilarity_shaep_{ID}_{job_number}_hits.txt')
    
    rocs = get_ROCS(torch.as_tensor(mol.GetConformer().GetPositions()), torch.as_tensor(reference.GetConformer().GetPositions()))
    
    return mol, float(mol.GetProp('Similarity_shape')), rocs.item()


def ESP_shape_align(reference, query, shaep_path = '../software', temp_path = '../temp/', remove_files = True, ID = ''):
    
    if not os.path.exists(temp_path+'shaep_objects_temp'):
        os.makedirs(temp_path+'shaep_objects_temp')
    
    job_number = random.randint(0, 10000000)
    
    Chem.MolToMolFile(reference, f'{temp_path}shaep_objects_temp/mol_ref_shaep_{ID}_{job_number}.mol')
    Chem.MolToMolFile(query, f'{temp_path}shaep_objects_temp/mol_query_shaep_{ID}_{job_number}.mol')
    
    os.system(f"{shaep_path}/shaep -q {temp_path}shaep_objects_temp/mol_ref_shaep_{ID}_{job_number}.mol {temp_path}shaep_objects_temp/mol_query_shaep_{ID}_{job_number}.mol -s {temp_path}shaep_objects_temp/query_mol_shaep_{ID}_{job_number}.sdf {temp_path}shaep_objects_temp/shapesimilarity_shaep_{ID}_{job_number}.txt >/dev/null 2>&1")
    
    suppl = rdkit.Chem.rdmolfiles.ForwardSDMolSupplier(f'{temp_path}shaep_objects_temp/query_mol_shaep_{ID}_{job_number}.sdf')
    mol = next(suppl)
    
    if remove_files:
        os.system(f'rm {temp_path}shaep_objects_temp/mol_ref_shaep_{ID}_{job_number}.mol {temp_path}shaep_objects_temp/mol_query_shaep_{ID}_{job_number}.mol {temp_path}shaep_objects_temp/query_mol_shaep_{ID}_{job_number}.sdf {temp_path}shaep_objects_temp/shapesimilarity_shaep_{ID}_{job_number}.txt {temp_path}shaep_objects_temp/shapesimilarity_shaep_{ID}_{job_number}_hits.txt')
    
    rocs = get_ROCS(torch.as_tensor(mol.GetConformer().GetPositions()), torch.as_tensor(reference.GetConformer().GetPositions()))
    
    return mol, rocs.item()

def VAB_2nd_order(centers_1, centers_2, alphas_1, alphas_2, prefactors_1, prefactors_2):
    R2 = (torch.cdist(centers_1, centers_2)**2.0).T
    prefactor1_prod_prefactor2 = prefactors_1 * prefactors_2.unsqueeze(1)
    alpha1_prod_alpha2 = alphas_1 * alphas_2.unsqueeze(1)
    alpha1_sum_alpha2 = alphas_1 + alphas_2.unsqueeze(1)

    VAB_2nd_order = torch.sum(np.pi**(1.5) * prefactor1_prod_prefactor2 * torch.exp(-(alpha1_prod_alpha2 / alpha1_sum_alpha2) * R2) / (alpha1_sum_alpha2**(1.5)))
    return VAB_2nd_order

def shape_tanimoto(centers_1, centers_2, alphas_1, alphas_2, prefactors_1, prefactors_2):
    VAA = VAB_2nd_order(centers_1, centers_1, alphas_1, alphas_1, prefactors_1, prefactors_1)
    VBB = VAB_2nd_order(centers_2, centers_2, alphas_2, alphas_2, prefactors_2, prefactors_2)
    VAB = VAB_2nd_order(centers_1, centers_2, alphas_1, alphas_2, prefactors_1, prefactors_2)
    return VAB / (VAA + VBB - VAB)

def get_ROCS(centers_1, centers_2, prefactor = 0.8, alpha = 0.81):
    #centers_1 = torch.tensor(centers_1)
    #centers_2 = torch.tensor(centers_2)
    prefactors_1 = torch.ones(centers_1.shape[0]) * prefactor
    prefactors_2 = torch.ones(centers_2.shape[0]) * prefactor
    alphas_1 = torch.ones(prefactors_1.shape[0]) * alpha
    alphas_2 = torch.ones(prefactors_2.shape[0]) * alpha

    tanimoto = shape_tanimoto(centers_1, centers_2, alphas_1, alphas_2, prefactors_1, prefactors_2)
    return tanimoto
