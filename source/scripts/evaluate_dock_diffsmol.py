import argparse
import os
from utils import misc, reconstruct, transforms
import numpy as np
from rdkit import Chem
from rdkit import RDLogger
import torch
from tqdm.auto import tqdm
from glob import glob
from collections import Counter
import pdb
from utils import eval_atom_type, scoring_func, analyze, eval_bond_length
from utils.docking_qvina import QVinaDockingTask
from utils.docking_vina import VinaDockingTask
from utils import similarity
from multiprocessing import Pool
from functools import partial

def get_diversity(mols):
    sims = similarity.tanimoto_sim_pairwise(mols)
    return sims

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

def evaluate_single_mol(pred_pos, pred_v, center=None, args=None):
    pred_pos, pred_v = pred_pos[args.eval_step], pred_v[args.eval_step]
    
    pred_pos = pred_pos + center
    # stability check
    pred_atom_type = transforms.get_atomic_number_from_index(pred_v, mode=args.atom_enc_mode)
    r_stable = analyze.check_stability(pred_pos, pred_atom_type)
    pair_dist = eval_bond_length.pair_distance_from_pos_v(pred_pos, pred_atom_type)
    
    # reconstruction
    try:
        pred_aromatic = transforms.is_aromatic_from_index(pred_v, mode=args.atom_enc_mode)
        mol = reconstruct.reconstruct_from_generated(pred_pos, pred_atom_type, pred_aromatic)
        smiles = Chem.MolToSmiles(mol)
    except reconstruct.MolReconsError:
        return {'mol': None, 'reconstruct': False}
    
    if '.' in smiles:
        return {'mol': None, 'reconstruct': True, 'connect': False}
    
    # chemical and docking check
    try:
        chem_results = scoring_func.get_chem(mol)
        if args.docking_mode == 'qvina':
            vina_task = QVinaDockingTask.from_generated_mol(
                mol, r['data'].ligand_filename, protein_root=args.protein_root)
            vina_results = vina_task.run_sync()
        elif args.docking_mode in ['vina_score', 'vina_dock']:
            vina_task = VinaDockingTask.from_generated_mol(
                mol, r['data'].ligand_filename, protein_root=args.protein_root)
            score_only_results = vina_task.run(mode='score_only', exhaustiveness=args.exhaustiveness)
            #print(score_only_results)
            score_vina_results.append(score_only_results[0]['affinity'])
            minimize_results = vina_task.run(mode='minimize', exhaustiveness=args.exhaustiveness)
            #print(minimize_results[0]['affinity'])
            min_vina_results.append(minimize_results[0]['affinity'])
            vina_results = {
                'score_only': score_only_results,
                'minimize': minimize_results
            }
            if args.docking_mode == 'vina_dock':
                docking_results = vina_task.run(mode='dock', exhaustiveness=args.exhaustiveness)
                vina_results['dock'] = docking_results
        else:
            vina_results = None
    except Exception as e:
        print(e)
        return {'mol': None, 'reconstruct': True, 'connect': True}

    # now we only consider complete molecules as success
    bond_dist = eval_bond_length.bond_distance_from_mol(mol)

    result = {
        'mol': mol,
        'smiles': smiles,
        'ligand_filename': r['data'].ligand_filename,
        'pred_pos': pred_pos,
        'pred_v': pred_v,
        'chem_results': chem_results,
        'vina': vina_results,
        'rstable': r_stable,
        'bond_dist': bond_dist,
        'pair_dist': pair_dist,
        'atom_type': Counter(pred_atom_type),
    }
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('sample_path', type=str)
    parser.add_argument('--verbose', type=eval, default=False)
    parser.add_argument('--eval_step', type=int, default=-1)
    parser.add_argument('--eval_id', type=int, default=-1)
    parser.add_argument('--eval_num_examples', type=int, default=None)
    parser.add_argument('--save', type=eval, default=True)
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--protein_root', type=str, default='../data/')
    parser.add_argument('--atom_enc_mode', type=str, default='add_aromatic')
    parser.add_argument('--docking_mode', type=str, choices=['qvina', 'vina_score', 'vina_dock', 'none'])
    parser.add_argument('--exhaustiveness', type=int, default=16)
    args = parser.parse_args()

    result_path = os.path.join(args.sample_path, 'eval_results')
    os.makedirs(result_path, exist_ok=True)
    logger = misc.get_logger('evaluate', log_dir=result_path)
    if not args.verbose:
        RDLogger.DisableLog('rdApp.*')

    # Load generated data
    results_fn_list = glob(os.path.join(args.sample_path, '*result_*.pt'))
    
    results_fn_list = sorted(results_fn_list, key=lambda x: int(os.path.basename(x)[:-3].split('_')[-1]))
    if args.eval_num_examples is not None:
        results_fn_list = results_fn_list[:args.eval_num_examples]
    if args.eval_id != -1:
        results_fn_list = [results_fn_list[args.eval_id]]
    
    num_examples = len(results_fn_list)
    logger.info(f'Load generated data done! {num_examples} examples in total.')

    num_samples = 0
    all_mol_stable, all_atom_stable, all_n_atom = 0, 0, 0
    n_recon_success, n_eval_success, n_complete = 0, 0, 0
    results = []
    all_bond_dist = []
    all_atom_types = Counter()
    success_pair_dist, success_atom_types = [], Counter()
    
    all_evalsim_mols = []
    for example_idx, r_name in enumerate(tqdm(results_fn_list, desc='Eval')):
        try:
            r = torch.load(r_name, map_location='cpu')  # ['data', 'pred_ligand_pos', 'pred_ligand_v', 'pred_ligand_pos_traj', 'pred_ligand_v_traj']
        except Exception as e:
            print(r_name)
            print("error occurred when loading %d sample %s" % (example_idx, e))
            continue
        all_pred_ligand_pos = r['pred_ligand_pos_traj']  # [num_samples, num_steps, num_atoms, 3]
        all_pred_ligand_v = r['pred_ligand_v_traj']
        num_samples += len(all_pred_ligand_pos)
        
        center = (r['data'].point_cloud_center + r['data'].ligand_center).numpy()
        score_vina_results = []
        min_vina_results = []
        complete_mols = []
        
        pred_poses, pred_vs = [], []
        for sample_idx, (pred_pos, pred_v) in enumerate(zip(all_pred_ligand_pos, all_pred_ligand_v)):
            pred_poses.append(pred_pos)
            pred_vs.append(pred_v)
        
        tmp = evaluate_single_mol(pred_poses[0], pred_vs[0], center=center, args=args)
        
        with Pool(args.num_workers) as p:
            # testset_results = p.starmap(partial(eval_single_datapoint, args=args),
            #                             zip(test_index[:args.eval_num_examples], list(range(args.eval_num_examples))))
            for result in tqdm(p.starmap(partial(evaluate_single_mol, center=center, args=args),
                        zip(pred_poses, pred_vs)), desc='Overall Eval'):
                if result['mol'] is None:
                    if result['reconstruct']:
                        n_recon_success += 1
                    if 'connect' in result and result['connect']:
                        n_complete += 1
                else:
                    n_recon_success += 1
                    n_complete += 1
                    n_eval_success += 1

                    r_stable = result['rstable']
                    bond_dist = result['bond_dist']
                    pair_dist = result['pair_dist']
                    atom_type = result['atom_type']
                    
                    all_mol_stable += r_stable[0]
                    all_atom_stable += r_stable[1]
                    all_n_atom += r_stable[2]
                    all_bond_dist += bond_dist
                    success_pair_dist += pair_dist
                    success_atom_types += Counter(atom_type)
                    results.append(result)
    
    #complete_mol_2ddivs = []
    #with Pool(processes=min(10, len(all_evalsim_mols))) as pool:
    #    for i, sims in tqdm(enumerate(pool.imap(get_diversity, all_evalsim_mols))):
    #        complete_mol_2ddivs.append(sims)

    #pairwise_sims = [(np.sum(sims)-sims.shape[0]) / (sims.shape[0] * (sims.shape[0] - 1)) for sims in complete_mol_2ddivs]
    #avg_pairwise_sims = np.mean(pairwise_sims)
    #median_pairwise_sims = np.median(pairwise_sims)
    fraction_mol_stable = all_mol_stable / num_samples
    fraction_atm_stable = all_atom_stable / all_n_atom
    fraction_recon = n_recon_success / num_samples
    fraction_eval = n_eval_success / num_samples
    fraction_complete = n_complete / num_samples
    validity_dict = {
        'mol_stable': fraction_mol_stable,
        'atm_stable': fraction_atm_stable,
        'recon_success': fraction_recon,
        'eval_success': fraction_eval,
        'complete': fraction_complete,
    }
    print_dict(validity_dict, logger)

    c_bond_length_profile = eval_bond_length.get_bond_length_profile(all_bond_dist)
    c_bond_length_dict = eval_bond_length.eval_bond_length_profile(c_bond_length_profile)
    logger.info('JS bond distances of complete mols: ')
    print_dict(c_bond_length_dict, logger)

    success_pair_length_profile = eval_bond_length.get_pair_length_profile(success_pair_dist)
    success_js_metrics = eval_bond_length.eval_pair_length_profile(success_pair_length_profile)
    print_dict(success_js_metrics, logger)

    try:
        atom_type_js = eval_atom_type.eval_atom_type_distribution(success_atom_types)
    except:
        atom_type_js = 1
    logger.info('Atom type JS: %.4f' % atom_type_js)

    if args.save:
        eval_bond_length.plot_distance_hist(success_pair_length_profile,
                                            metrics=success_js_metrics,
                                            save_path=os.path.join(result_path, f'pair_dist_hist_{args.eval_step}.png'))

    logger.info('Number of reconstructed mols: %d, complete mols: %d, evaluated mols: %d' % (
        n_recon_success, n_complete, len(results)))

    qed = [r['chem_results']['qed'] for r in results]
    sa = [r['chem_results']['sa'] for r in results]
    logger.info('QED:   Mean: %.3f Median: %.3f' % (np.mean(qed), np.median(qed)))
    logger.info('SA:    Mean: %.3f Median: %.3f' % (np.mean(sa), np.median(sa)))
    logger.info('mol stable: Mean: %.3f' % fraction_mol_stable)
    logger.info('atm_stable: Mean: %.3f' % fraction_atm_stable)

    if args.docking_mode == 'qvina':
        vina = [r['vina'][0]['affinity'] for r in results]
        logger.info('Vina:  Mean: %.3f Median: %.3f' % (np.mean(vina), np.median(vina)))
    elif args.docking_mode in ['vina_dock', 'vina_score']:
        vina_score_only = [r['vina']['score_only'][0]['affinity'] for r in results]
        vina_min = [r['vina']['minimize'][0]['affinity'] for r in results]
        logger.info('Vina Score:  Mean: %.3f Median: %.3f' % (np.mean(vina_score_only), np.median(vina_score_only)))
        logger.info('Vina Min  :  Mean: %.3f Median: %.3f' % (np.mean(vina_min), np.median(vina_min)))
        if args.docking_mode == 'vina_dock':
            vina_dock = [r['vina']['dock'][0]['affinity'] for r in results]
            logger.info('Vina Dock :  Mean: %.3f Median: %.3f' % (np.mean(vina_dock), np.median(vina_dock)))

    # check ring distribution
    print_ring_ratio([r['chem_results']['ring_size'] for r in results], logger)

    if args.eval_id != -1:
        path = os.path.join(result_path, f'metrics_{args.eval_step}_{args.eval_id}_dock.pt')
    else:
        path = os.path.join(result_path, f'metrics_{args.eval_step}_{args.eval_num_examples}_dock.pt')

    torch.save({
        'stability': validity_dict,
        'bond_length': all_bond_dist,
        'all_results': results
    }, path)
