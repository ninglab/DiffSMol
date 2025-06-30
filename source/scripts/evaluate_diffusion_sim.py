import argparse
import os
import pdb
import pickle
import numpy as np
from rdkit import Chem
from rdkit import RDLogger
import torch
from tqdm.auto import tqdm
from glob import glob
from collections import Counter
import warnings
from datasets.shape_mol_dataset import ShapeMolDataset
from datasets.shape_mol_data import FOLLOW_BATCH
import utils.transforms as trans

from utils import eval_bond_length
from utils import eval_atom_type
from utils import analyze
from utils import misc
from utils import scoring_func
from utils import reconstruct
from datasets import get_dataset
from utils.docking import QVinaDockingTask
from utils import similarity
from multiprocessing import Pool
IF_LOAD_ROCS = True
try:
    from utils.openeye_utils import ROCS_shape_overlap
except:
    print("cannot load openeye ROCS, use Shaep to calculate similarity")
    IF_LOAD_ROCS = False
import time
import pandas as pd
from utils.evaluation import compare_with_ref, Local3D

data_metrics = pd.read_csv("../data/MOSES2/metrics/test_metric_50k_mols2.csv")
with open("../data/MOSES2/metrics/test_metric_50k_mols2_local3d.pkl", 'rb') as f:
    data_local3d_dict = pickle.load(f)

def get_jsds_with_data(values, data):
    metric_list = list(values.keys())
    all_jsds = {}
    for metric in metric_list:
        value_list = values[metric]
        ref_list = data[metric]
        jsd = compare_with_ref(value_list, ref_list)
        if jsd is None or np.isnan(jsd): continue
        all_jsds[metric] = jsd
    return all_jsds

def print_dict(d, logger):
    for k, v in d.items():
        if v is not None:
            logger.info(f'{k}:\t{v:.4f}')
        else:
            logger.info(f'{k}:\tNone')


def print_ring_ratio(all_ring_sizes, logger):
    for ring_size in range(3, 10):
        n_mol = 0
        for counter in all_ring_sizes:
            if ring_size in counter:
                n_mol += 1
        logger.info(f'ring size: {ring_size} ratio: {n_mol / len(all_ring_sizes):.3f}')

def get_ref_similarity_shaep(eval_tuple):
    mols, ref = eval_tuple
    results = []
    shaep_align_mols, shaep_shape_simROCS = similarity.calculate_shaep_shape_sim(mols, ref)
    print(shaep_shape_simROCS)
    sims = similarity.tanimoto_sim_pairwise(mols)
    for mol, shaep_align_mol, shaep_shape_simrocs in \
        zip(mols, shaep_align_mols, shaep_shape_simROCS):
        try:
            smiles = Chem.MolToSmiles(shaep_align_mol)
            tanimoto_sim = similarity.tanimoto_sim(mol, ref)
        except:
            tanimoto_sim = -1
            smiles = None

        results.append({
            'smiles': smiles,
            'align_mol': shaep_align_mol,
            'tanimoto_sim': tanimoto_sim,
            'shape_sim': shaep_shape_simrocs,
        })
    return results, sims

def get_ref_similarity_rocs(eval_tuple):
    mols, ref = eval_tuple
    results = []
    scores = ROCS_shape_overlap(mols, ref)
    sims = similarity.tanimoto_sim_pairwise(mols)

    for _, rocs_sim, align_mol in scores:
        try:
            smiles = Chem.MolToSmiles(align_mol)
            tanimoto_sim = similarity.tanimoto_sim(align_mol, ref)
        except:
            tanimoto_sim = -1
            smiles = None
        results.append({
            'smiles': smiles,
            'align_mol': align_mol,
            'tanimoto_sim': tanimoto_sim,
            'shape_sim': rocs_sim
        })
    return results, sims

def calculate_local3d_metrics(mols, logger):
    local3d = Local3D()
    local3d.get_predefined()
    #logger.info(f'Computing local 3d - bond lengths metric...')
    lengths = local3d.calc_frequent(mols, type_='length', parallel=True)
    #logger.info(f'Computing local 3d - bond angles metric...')
    angles = local3d.calc_frequent(mols, type_='angle', parallel=True)
    #logger.info(f'Computing local 3d - dihedral angles metric...')
    dihedrals = local3d.calc_frequent(mols, type_='dihedral', parallel=True)
    
    length_jsds = get_jsds_with_data(lengths, data_local3d_dict['lengths'])
    angle_jsds = get_jsds_with_data(angles, data_local3d_dict['angles'])
    dihedral_jsds = get_jsds_with_data(dihedrals, data_local3d_dict['dihedral'])

    mean_length_jsd = np.mean(list(length_jsds.values()))
    mean_angle_jsd = np.mean(list(angle_jsds.values()))
    mean_dihedral_jsd = np.mean(list(dihedral_jsds.values()))
    string = "length_jsd: \n"
    for metric in length_jsds:
        string += "%s %.6f\n" % (metric, length_jsds[metric])
    
    string += "angle_jsd: \n"
    for metric in angle_jsds:
        string += "%s %.6f\n" % (metric, angle_jsds[metric])

    string += "dihedral_jsd: \n"
    for metric in dihedral_jsds:
        string += "%s %.6f\n" % (metric, dihedral_jsds[metric])
    logger.info(string)
    
    return mean_length_jsd, mean_angle_jsd, mean_dihedral_jsd

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('sample_path', type=str)
    parser.add_argument('testset_path', type=str)
    parser.add_argument('--verbose', type=eval, default=False)
    parser.add_argument('--eval_step', type=int, default=-1)
    parser.add_argument('--eval_start', type=int, default=0)
    parser.add_argument('--eval_num_examples', type=int, default=None)
    parser.add_argument('--basic_mode', type=eval, default=True)
    parser.add_argument('--save', type=eval, default=True)
    parser.add_argument('--docking', type=eval, default=False)
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--shape_sim_type', type=str, default="rocs")
    parser.add_argument('--atom_enc_mode', type=str, default='add_aromatic')
    parser.add_argument('--use_bond', type=eval, default=False)
    args = parser.parse_args()
    
    result_path = os.path.join(args.sample_path, 'eval_sim_usebond%s_results' % (args.use_bond))
    os.makedirs(result_path, exist_ok=True)
    logger = misc.get_logger('evaluate', log_dir=result_path)
    if not args.verbose:
        RDLogger.DisableLog('rdApp.*')
    
    # Load generated data
    results_fn_list = glob(os.path.join(args.sample_path, '*result_*.pt'))
    results_fn_list = sorted([(int(os.path.basename(result_fn)[:-3].split('_')[-1]), result_fn) for result_fn in results_fn_list], key=lambda x: x[0])
    if args.eval_num_examples is not None:
        results_fn_list = results_fn_list[args.eval_start:args.eval_start + args.eval_num_examples]
    num_examples = len(results_fn_list)
    
    logger.info(f'Load generated data done! {num_examples} examples in total.')

    test_data = pickle.load(open(args.testset_path, 'rb'))['rdkit_mol_cistrans_stereo']
    test_idx_file = open("../data/MOSES2/index_map.txt", 'r')
    test_idx_dict = {}
    for line in test_idx_file.readlines():
        idxs = line.strip().split(":")
        test_idx_dict[int(idxs[0])] = int(idxs[1])
    test_idx_file.close()

    num_samples = 0
    all_mol_stable, all_atom_stable, all_n_atom = 0, 0, 0
    n_recon_success, n_eval_success, n_complete = 0, 0, 0
    all_results = []
    all_pair_dist, all_bond_dist = [], []
    all_atom_types = Counter()
    success_pair_dist, success_atom_types = [], Counter()
    complete_mol_2dsims, complete_mol_3dsims = [], []
    all_smiles, complete_mol_2ddivs = [], []
    
    all_evalsim_tuples = []
    all_generated_mols = []
    for example_idx, r_name in tqdm(results_fn_list, desc='Eval'):
        try:
            r = torch.load(r_name, map_location='cuda' if torch.cuda.is_available() else 'cpu')  # ['data', 'pred_ligand_pos', 'pred_ligand_v', 'pred_ligand_pos_traj', 'pred_ligand_v_traj']
        except Exception as e:
            print(f'failed to load {r_name} due to error: {e}')
            continue
        cond_mol = test_data[test_idx_dict[example_idx]]

        all_pred_ligand_pos = r['pred_ligand_pos_traj']  # [num_samples, num_steps, num_atoms, 3]
        # all_pred_ligand_pos = r['pred_ligand_pos']
        all_pred_ligand_v = r['pred_ligand_v_traj']
        all_pred_bonds = r['pred_ligand_bond'] if 'pred_ligand_bond' in r else []
        num_samples += len(all_pred_ligand_pos)
        
        results, complete_mols, complete_smiles = [], [], []
        
        for sample_idx, (pred_pos, pred_v) in enumerate(zip(all_pred_ligand_pos, all_pred_ligand_v)):
            pred_pos, pred_v = pred_pos[args.eval_step], pred_v[args.eval_step]
            
            # stability check
            pred_atom_type = trans.get_atomic_number_from_index(pred_v, mode=args.atom_enc_mode)

            all_atom_types += Counter(pred_atom_type)

            r_stable = analyze.check_stability(pred_pos, pred_atom_type)
            all_mol_stable += r_stable[0]
            all_atom_stable += r_stable[1]
            all_n_atom += r_stable[2]

            pair_dist = eval_bond_length.pair_distance_from_pos_v(pred_pos, pred_atom_type)
            all_pair_dist += pair_dist

            #if len(all_pred_bonds) > 0:
            #    try:
            #        pred_bonds = all_pred_bonds[sample_idx]
            #    except:
            #        print('cannot get bond for %d-th generated mol for %d sample' % (sample_idx, example_idx))
            #        pred_bonds = None
            #else:
            pred_bonds = all_pred_bonds[sample_idx] if args.use_bond else None
            # reconstruction
            try:
                pred_aromatic = trans.is_aromatic_from_index(pred_v, mode=args.atom_enc_mode)
                mol = reconstruct.reconstruct_from_generated(pred_pos, pred_atom_type, pred_aromatic,
                                                             basic_mode=args.basic_mode, pred_bond=pred_bonds)
                smiles = Chem.MolToSmiles(mol)
            except reconstruct.MolReconsError:
                if args.verbose:
                    logger.warning('Reconstruct failed %s' % f'{example_idx}_{sample_idx}')
                continue
            n_recon_success += 1

            if '.' in smiles:
                continue
            n_complete += 1
            complete_mols.append(mol)
            all_smiles.append(smiles)
            all_generated_mols.append(mol)
            # chemical and docking check
            try:
                # qed / logp / SA / lipniz / ringsize check
                chem_results = scoring_func.get_chem(mol)
                # print(chem_results)
                if args.docking:
                    vina_task = QVinaDockingTask.from_generated_mol(
                        mol, r['data'].ligand_filename, protein_root=args.protein_root)
                    vina_results = vina_task.run_sync()
                else:
                    vina_results = None
                n_eval_success += 1
            except:
                if args.verbose:
                    logger.warning('Evaluation failed for %s' % f'{example_idx}_{sample_idx}')
                continue

            # now we only consider complete molecules as success
            bond_dist = eval_bond_length.bond_distance_from_mol(mol)
            all_bond_dist += bond_dist

            success_pair_dist += pair_dist
            success_atom_types += Counter(pred_atom_type)

            # now we calculate the 3d shape similarity between complete molecules and condition molecules
            
            results.append({
                'mol': mol,
                'smiles': smiles,
                # 'gen_results': r,
                'ligand_filename': r_name,
                'pred_pos': pred_pos,
                'pred_v': pred_v,
                'chem_results': chem_results,
                #'vina': vina_results
            })

        all_results.append(results)
        all_evalsim_tuples.append((complete_mols, cond_mol))

        # ============ evaluate uniqueness and diversity ============
        #sims = similarity.tanimoto_sim_pairwise(complete_mols)
        #complete_mol_2ddivs.append(sims)
    
    """
    torch.save({
        'nums': [all_mol_stable, all_atom_stable, n_recon_success, n_eval_success, n_complete],
        'all_smiles': all_smiles,
        'all_results': all_results,
        'all_evalsim_tuples': all_evalsim_tuples,
        'complete_mol_2ddivs': complete_mol_2ddivs,
    }, os.path.join(result_path, f'metrics_{args.eval_step}_{args.basic_mode}.pt'))
    """
    get_ref_similarity_shaep((complete_mols, cond_mol))

    complete_mol_2ddivs = []
    with Pool(processes=args.num_workers) as pool:
        func = get_ref_similarity_shaep if args.shape_sim_type == 'shaep' or (not IF_LOAD_ROCS) else get_ref_similarity_rocs
        for i, (results, sims) in tqdm(enumerate(pool.imap(func, all_evalsim_tuples))):
            complete_mol_2ddivs.append(sims)
            for j in range(len(all_results[i])):
                all_results[i][j].update(results[j])
    
    print(len(all_generated_mols))
    mean_length_jsd, mean_angle_jsd, mean_dihedral_jsd = calculate_local3d_metrics(all_generated_mols, logger)
    logger.info(f'Evaluate done! {num_samples} samples in total.')
    
    fraction_mol_stable = all_mol_stable / num_samples
    fraction_atm_stable = all_atom_stable / all_n_atom
    fraction_recon = n_recon_success / num_samples
    fraction_eval = n_eval_success / num_samples
    fraction_complete = n_complete / num_samples
    fraction_uniq = len(set(all_smiles)) / n_complete
    # diversity
    pairwise_sims = [(np.sum(sims)-sims.shape[0]) / (sims.shape[0] * (sims.shape[0] - 1)) for sims in complete_mol_2ddivs]
    avg_pairwise_sims = np.mean(pairwise_sims)
    std_pairwise_sims = np.std(pairwise_sims)

    ref_tanimoto_sims = [np.mean([element['tanimoto_sim'] for element in results if element['tanimoto_sim'] >= 0]) for results in all_results]
    avg_ref_tanimoto_sims = np.mean(ref_tanimoto_sims)
    std_ref_tanimoto_sims = np.std(ref_tanimoto_sims)
    
    ref_rocs_sims = [np.mean([element['shape_sim'] for element in results if element['shape_sim'] >= 0]) for results in all_results]
    avg_ref_rocs_sims = np.mean(ref_rocs_sims)
    std_ref_rocs_sims = np.std(ref_rocs_sims)
    
    #avg_ref_max_rocs_sims = np.mean([np.max([element['rocs_sim'] for element in results if element['rocs_sim'] >= 0]) for results in all_results])
    max_rocs_sims, max_rocs_tanimoto_sims = [], []
    for results in all_results:
        max_idx = 0
        for i, element in enumerate(results):
            if element['shape_sim'] > results[max_idx]['shape_sim']:
                max_idx = i

        max_rocs_sims.append(results[max_idx]['shape_sim'])
        max_rocs_tanimoto_sims.append(results[max_idx]['tanimoto_sim'])
    avg_ref_max_rocs_sims = np.mean(max_rocs_sims)
    std_ref_max_rocs_sims = np.std(max_rocs_sims)
    avg_ref_max_tanimoto_sims = np.mean(max_rocs_tanimoto_sims)
    std_ref_max_tanimoto_sims = np.std(max_rocs_tanimoto_sims)

    validity_dict = {
        'mol_stable': fraction_mol_stable,
        'atm_stable': fraction_atm_stable,
        'recon_success': fraction_recon,
        'eval_success': fraction_eval,
        'complete': fraction_complete,
        'uniq_over_complete': fraction_uniq,
        'avg_pairwise_sims': avg_pairwise_sims,
        'std_pairwise_sims': std_pairwise_sims,
        'avg_ref_tanimoto_sims': avg_ref_tanimoto_sims,
        'std_ref_tanimoto_sims': std_ref_tanimoto_sims,
        'avg_ref_rocssims': avg_ref_rocs_sims,
        'std_ref_rocssims': std_ref_rocs_sims,
        'avg_ref_max_rocssims': avg_ref_max_rocs_sims,
        'std_ref_max_rocssims': std_ref_max_rocs_sims,
        'avg_ref_max_tanimoto_sims': avg_ref_max_tanimoto_sims,
        'std_ref_max_tanimoto_sims': std_ref_max_tanimoto_sims,
        'avg_length_jsds': mean_length_jsd,
        'avg_angle_jsds': mean_angle_jsd,
        'avg_dihedral_jsds': mean_dihedral_jsd
    }
    print_dict(validity_dict, logger)

    logger.info('Number of reconstructed mols: %d, complete mols: %d, evaluated mols: %d' % (
        n_recon_success, n_complete, n_eval_success))
    
    c_bond_length_profile = eval_bond_length.get_bond_length_profile(all_bond_dist)
    c_bond_length_dict = eval_bond_length.eval_bond_length_profile(c_bond_length_profile)
    logger.info('JS bond distances of complete mols: ')
    print_dict(c_bond_length_dict, logger)

    pair_length_profile = eval_bond_length.get_pair_length_profile(all_pair_dist)
    js_metrics = eval_bond_length.eval_pair_length_profile(pair_length_profile)
    logger.info('JS pair distances:  ')
    print_dict(js_metrics, logger)

    atom_type_js = eval_atom_type.eval_atom_type_distribution(all_atom_types)
    logger.info('Atom type JS: %.4f' % atom_type_js)

    success_pair_length_profile = eval_bond_length.get_pair_length_profile(success_pair_dist)
    success_js_metrics = eval_bond_length.eval_pair_length_profile(success_pair_length_profile)
    logger.info('\nSuccess JS pair distances:  ')
    print_dict(success_js_metrics, logger)

    atom_type_js = eval_atom_type.eval_atom_type_distribution(success_atom_types)
    logger.info('Atom type JS: %.4f' % atom_type_js)

    if args.save:
        eval_bond_length.plot_distance_hist(pair_length_profile,
                                            metrics=js_metrics,
                                            save_path=os.path.join(result_path, f'pair_dist_hist_{args.eval_step}.png'))

    logger.info('Number of reconstructed mols: %d, complete mols: %d, evaluated mols: %d' % (
        n_recon_success, n_complete, num_samples))
    mean_qed = np.mean([np.mean([r['chem_results']['qed'] for r in results]) for results in all_results])
    mean_sa = np.mean([np.mean([r['chem_results']['sa'] for r in results]) for results in all_results])
    if args.docking:
        mean_vina = np.mean([np.mean([r['vina'][0]['affinity'] for r in results]) for results in all_results])
    else:
        mean_vina = 0.
    logger.info('QED: %.3f SA: %.3f Vina: %.3f' % (mean_qed, mean_sa, mean_vina))

    # check ring distribution
    print_ring_ratio([r['chem_results']['ring_size'] for results in all_results for r in results], logger)

    torch.save({
        'stability': validity_dict,
        'bond_length': all_bond_dist,
        'atom_type': all_atom_types,
        'all_results': all_results,
        'pairwise_div': complete_mol_2ddivs,
    }, os.path.join(result_path, f'metrics_{args.eval_step}_{args.basic_mode}_{args.eval_start}_{args.eval_num_examples}.pt'))
