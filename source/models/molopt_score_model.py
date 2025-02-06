"""
the newest version about how to incorporate shape embedding with electrostatic features.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum, scatter_mean, scatter_softmax, scatter_min
import numpy as np
from tqdm.auto import tqdm
import random
from models.common import ShiftedSoftplus, MLP, GaussianSmearing, find_closest_points, compute_bond_angle, compute_torsion_angle
from models.uni_transformer import UniTransformerO2TwoUpdateGeneral
from models.diffusion import *
from sklearn.neighbors import KDTree
import pdb

def get_refine_net(refine_net_type, config):
    if refine_net_type == 'uni_o2':
        refine_net = UniTransformerO2TwoUpdateGeneral(
            num_blocks=config.num_blocks,
            num_layers=config.num_layers,
            scalar_hidden_dim=config.scalar_hidden_dim,
            vec_hidden_dim=config.vec_hidden_dim,
            shape_dim=config.shape_dim,
            shape_latent_dim=config.shape_latent_dim,
            n_heads=config.n_heads,
            k=config.knn,
            edge_feat_dim=config.edge_feat_dim,
            num_r_gaussian=config.num_r_gaussian,
            num_node_types=config.num_node_types,
            act_fn=config.act_fn,
            norm=config.norm,
            cutoff_mode=config.cutoff_mode,
            r_feat_mode=config.r_feat_mode,
            r_max=config.r_max,
            x2h_out_fc=config.x2h_out_fc,
            atom_enc_mode=config.atom_enc_mode,
            shape_type=config.shape_type,
            sync_twoup=config.sync_twoup,
            pred_bond_type=config.pred_bond_type,
            use_shape_vec_mul=config.use_shape_vec_mul,
            use_residue=config.use_residue,
            time_emb_dim=config.time_emb_dim
        )
    else:
        raise ValueError(refine_net_type)
    return refine_net

def get_distance(pos, edge_index):
    return (pos[edge_index[0]] - pos[edge_index[1]]).norm(dim=-1)

def to_torch_const(x):
    x = torch.from_numpy(x).float()
    x = nn.Parameter(x, requires_grad=False)
    return x

def center_pos(ligand_pos, batch_ligand, mode='none'):
    if mode == 'none':
        offset = 0.0
    elif mode == 'center':
        offset = scatter_mean(ligand_pos, batch_ligand, dim=0)
        ligand_pos = ligand_pos - offset[batch_ligand]
    else:
        raise NotImplementedError
    return ligand_pos, offset


# %% categorical diffusion related
def index_to_log_onehot(x, num_classes):
    assert x.max().item() < num_classes, f'Error: {x.max().item()} >= {num_classes}'
    x_onehot = F.one_hot(x, num_classes)
    log_x = torch.log(x_onehot.float().clamp(min=1e-30))
    return log_x


def log_onehot_to_index(log_x):
    return log_x.argmax(1)


def categorical_kl(log_prob1, log_prob2):
    kl = (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=1)
    return kl


def log_categorical(log_x_start, log_prob):
    return (log_x_start.exp() * log_prob).sum(dim=1)


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    KL divergence between normal distributions parameterized by mean and log-variance.
    """
    kl = 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2) + (mean1 - mean2) ** 2 * torch.exp(-logvar2))
    return kl.sum(-1)


def log_normal(values, means, log_scales):
    var = torch.exp(log_scales * 2)
    log_prob = -((values - means) ** 2) / (2 * var) - log_scales - np.log(np.sqrt(2 * np.pi))
    return log_prob.sum(-1)


def log_sample_categorical(logits):
    uniform = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
    sample_index = (gumbel_noise + logits).argmax(dim=-1)
    # sample_onehot = F.one_hot(sample, self.num_classes)
    # log_sample = index_to_log_onehot(sample, self.num_classes)
    return sample_index


def log_1_min_a(a):
    return np.log(1 - np.exp(a) + 1e-40)


def log_add_exp(a, b):
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))

# %%

def dynamic_threshold(x0, p):
    s = torch.quantile(x0, p)
    x0 = torch.clip(x0, min=-s, max=s)
    return x0

def reference_threshold(x0, x0_cond, p):
    s = torch.max(torch.abs(x0_cond)) * p
    x0 = torch.clip(x0, min=-s, max=s)
    return x0

def rescale(x0, x0_cond, p):
    std_x0 = torch.std(x0)
    std_x0_cond = torch.std(x0_cond)
    rescale_ratio = std_x0_cond / std_x0
    x0_rescale = x0 * rescale_ratio
    x0 = p * x0_rescale + (1-p) * x0
    return x0

def threshold_CFG(x0, x0_cond, threshold_type, threshold_args, bounds=None):
    if threshold_type == 'reference_threshold':
        p = threshold_args.get('p', 1.1)
        x0 = reference_threshold(x0, x0_cond, p)
    elif threshold_type == 'dynamic_threshold':
        p = threshold_args.get('p', 0.995)
        x0 = dynamic_threshold(x0, p)
    elif threshold_type == 'rescale':
        p = threshold_args.get('p', 0.7)
        x0 = rescale(x0, x0_cond, p)
    elif threshold_type is not None:
        raise ValueError("undefined thresholding strategy: expect one of (reference_threshold, dynamic_threshold, rescale, none) " + \
                         "but get %s" % (threshold_type))
    
    if bounds is not None:
        x0 = torch.clamp(x0, min=bounds[:,0], max=bounds[:, 1])
    return x0

# Time embedding
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

# Model
class ScorePosNet3D(nn.Module):

    def __init__(self, config, ligand_atom_feature_dim, ligand_bond_feature_dim):
        super().__init__()
        self.config = config

        # variance schedule
        self.denoise_type = config.denoise_type  # ['diffusion', 'score_matching']
        self.model_mean_type = config.model_mean_type  # ['noise', 'C0']
        self.loss_v_weight = config.loss_v_weight
        self.loss_weight_type = config.loss_weight_type

        self.v_mode = config.v_mode
        self.v_net_type = getattr(config, 'v_net_type', 'mlp')

        self.pred_bond_type = config.pred_bond_type
        if self.pred_bond_type:
            self.loss_bond_type = nn.CrossEntropyLoss(reduce=False)
            self.loss_bond_weight = config.loss_bond_weight

        self.use_bond_dist_loss = config.use_bond_dist_loss
        self.loss_bond_dist_weight = config.loss_bond_dist_weight
        self.use_bond_angle_loss = config.use_bond_angle_loss
        self.loss_bond_angle_weight = config.loss_bond_angle_weight
        self.use_torsion_angle_loss = config.use_torsion_angle_loss
        self.loss_torsion_angle_weight = config.loss_torsion_angle_weight
        self.loss_torsion_angle_type = config.loss_torsion_angle_type

        self.sample_time_method = config.sample_time_method  # ['importance', 'symmetric']
        self.loss_pos_type = config.loss_pos_type  # ['mse', 'kl']
        print(f'Loss pos mode {self.loss_pos_type} applied!')

        betas = get_beta_schedule(
            num_diffusion_timesteps=config.num_diffusion_timesteps,
            **config.schedule_pos
        )
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        if self.loss_weight_type == 'noise_level':
            snr_values = alphas_cumprod / (1-alphas_cumprod)
            self.loss_pos_step_weight = to_torch_const(np.clip(config.loss_pos_min_weight + snr_values, None, config.loss_pos_max_weight))

        self.betas = to_torch_const(betas)
        self.num_timesteps = self.betas.size(0)
        self.alphas_cumprod = to_torch_const(alphas_cumprod)
        self.alphas_cumprod_prev = to_torch_const(alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = to_torch_const(np.sqrt(alphas_cumprod))
        self.sqrt_one_minus_alphas_cumprod = to_torch_const(np.sqrt(1. - alphas_cumprod))
        self.sqrt_recip_alphas_cumprod = to_torch_const(np.sqrt(1. / alphas_cumprod))
        self.sqrt_recipm1_alphas_cumprod = to_torch_const(np.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.posterior_mean_c0_coef = to_torch_const(betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.posterior_mean_ct_coef = to_torch_const(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))
        
        # log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_var = to_torch_const(posterior_variance)
        # self.posterior_logvar = to_torch_const(np.log(np.maximum(posterior_variance, 1e-10)))
        self.posterior_logvar = to_torch_const(np.log(np.append(self.posterior_var[1], self.posterior_var[1:])))

        betas_v = get_beta_schedule(
            num_diffusion_timesteps=config.num_diffusion_timesteps,
            **config.schedule_v
        )
        alphas_v = 1. - betas_v

        log_alphas_v = np.log(alphas_v)
        log_alphas_cumprod_v = np.cumsum(log_alphas_v)
        
        self.log_alphas_v = to_torch_const(log_alphas_v)
        self.log_one_minus_alphas_v = to_torch_const(log_1_min_a(log_alphas_v))
        self.log_alphas_cumprod_v = to_torch_const(log_alphas_cumprod_v)
        self.log_one_minus_alphas_cumprod_v = to_torch_const(log_1_min_a(log_alphas_cumprod_v))

        # model definition
        self.scalar_hidden_dim = config.scalar_hidden_dim
        self.vec_hidden_dim = config.vec_hidden_dim
        self.num_classes = ligand_atom_feature_dim
        
        ###### to test ######
        # center pos
        self.center_pos_mode = config.center_pos_mode  # ['none', 'center']

        # time embedding
        self.time_emb_dim = config.time_emb_dim
        if self.time_emb_dim > 0:
            self.time_emb = nn.Sequential(
                SinusoidalPosEmb(self.time_emb_dim),
                nn.Linear(self.time_emb_dim, self.time_emb_dim * 2),
                nn.SiLU(),
                nn.Linear(self.time_emb_dim * 2, self.time_emb_dim)
            )
            self.ligand_atom_emb = nn.Linear(ligand_atom_feature_dim + self.time_emb_dim + int(self.v_mode=='tomask'), self.scalar_hidden_dim)
        else:
            self.ligand_atom_emb = nn.Linear(ligand_atom_feature_dim + int(self.v_mode=='tomask'), self.scalar_hidden_dim)
            
        self.refine_net_type = config.model_type
        self.refine_net = get_refine_net(self.refine_net_type, config)

        print(f'v net type: {self.v_net_type}')
        if self.v_net_type == 'mlp':
            self.v_inference = nn.Sequential(
                nn.Linear(self.scalar_hidden_dim, self.scalar_hidden_dim),
                ShiftedSoftplus(),
                nn.Linear(self.scalar_hidden_dim, ligand_atom_feature_dim + int(self.v_mode=='tomask')),
            )
        elif self.v_net_type == 'attention':
            self.v_distance_expansion = GaussianSmearing(0., 10., num_gaussians=config.num_r_gaussian)
            norm = True
            act_fn = 'relu'
            kv_input_dim = self.scalar_hidden_dim * 2 + config.num_r_gaussian
            self.v_n_heads = 16
            self.vk_func = MLP(kv_input_dim, self.scalar_hidden_dim, self.scalar_hidden_dim, norm=norm, act_fn=act_fn)
            self.vv_func = MLP(kv_input_dim, self.scalar_hidden_dim, self.scalar_hidden_dim, norm=norm, act_fn=act_fn)
            self.vq_func = MLP(self.scalar_hidden_dim, self.scalar_hidden_dim, self.scalar_hidden_dim, norm=norm, act_fn=act_fn)
            self.v_inference = nn.Sequential(
                nn.Linear(self.scalar_hidden_dim * 2, self.scalar_hidden_dim),
                ShiftedSoftplus(),
                nn.Linear(self.scalar_hidden_dim, ligand_atom_feature_dim + int(self.v_mode=='tomask')),
            )
        else:
            raise NotImplementedError
        self.cond_mask_prob = config.cond_mask_prob

    def forward(self, ligand_pos_perturbed, ligand_v_perturbed, 
                batch_ligand, ligand_shape, time_step=None, ligand_bond_index=None, 
                ligand_bond_type=None, pred_bond=False, if_test=False, return_all=False):
        """
        f(x0, v0 | xt, vt): predicts the original position and atom type from noisy samples at step t
        """
        batch_size = batch_ligand.max().item() + 1

        ligand_v_perturbed = F.one_hot(ligand_v_perturbed, self.num_classes+int(self.v_mode=='tomask')).float()
        
        # time embedding
        if self.time_emb_dim > 0:
            time_feat = self.time_emb(time_step)[batch_ligand]
            ligand_feat = torch.cat([ligand_v_perturbed, time_feat], -1)
        else:
            ligand_feat = ligand_v_perturbed
            time_feat = None

        ligand_emb = self.ligand_atom_emb(ligand_feat)
        
        outputs = self.refine_net(ligand_v_perturbed, ligand_emb, ligand_pos_perturbed, batch_ligand, 
                                  ligand_shape, ligand_bond_index=ligand_bond_index, 
                                  ligand_bond_type=ligand_bond_type, pred_bond=pred_bond, if_test=if_test, return_all=return_all)
        final_pos, final_h = outputs['x'], outputs['h']
        final_v = self.v_inference(final_h)

        preds = {
            'pred_ligand_pos': final_pos,
            'pred_ligand_h': final_h,
            'pred_ligand_v': final_v,
        }
        if self.pred_bond_type and pred_bond:
            preds.update({
                'bond_pred': outputs['bond_pred'],
                'bond_gt': outputs['bond_gt'],
                'edge_index': outputs['edge_index'],
                'all_bond_loss': outputs['all_bond_loss']
            })

        if return_all:
            final_all_pos, final_all_h = outputs['all_x'], outputs['all_h']
            final_all_ligand_pos = [pos for pos in final_all_pos]
            final_all_ligand_v = [self.v_inference(h) for h in final_all_h]
            preds.update({
                'layer_pred_ligand_pos': final_all_ligand_pos,
                'layer_pred_ligand_v': final_all_ligand_v
            })
        return preds


    def q_v_pred_one_timestep(self, log_vt_1, t, batch):
        """
        forward diffusion process: q(vt | vt-1)
        """
        log_alpha_t = extract(self.log_alphas_v, t, batch)
        log_1_min_alpha_t = extract(self.log_one_minus_alphas_v, t, batch)

        if self.v_mode == 'uniform':
            # alpha_t * vt-1 + (1 - alpha_t) 1 / K
            log_probs = log_add_exp(
                log_vt_1 + log_alpha_t,
                log_1_min_alpha_t - np.log(self.num_classes)
            )
        elif self.v_mode == 'tomask':
            # alpha_t * vt for category at step t and (1 - alpha_t) for mask
            log_probs = log_vt_1 + log_alpha_t
            log_probs[:, -1] = log_1_min_alpha_t.squeeze(1)
        else:
            raise ValueError("undefined v_mode: %s (expect uniform or tomask)" % (self.v_mode))  
        return log_probs

    def q_v_pred(self, log_v0, t, batch):
        """
        forward diffusion process: q(vt | v0)
        """
        log_cumprod_alpha_t = extract(self.log_alphas_cumprod_v, t, batch)
        log_1_min_cumprod_alpha = extract(self.log_one_minus_alphas_cumprod_v, t, batch)

        if self.v_mode == 'uniform':
            # cum_alpha_t * v0 + (1 - cum_alpha_t) 1 / K
            log_probs = log_add_exp(
                log_v0 + log_cumprod_alpha_t,
                log_1_min_cumprod_alpha - np.log(self.num_classes)
            )
        elif self.v_mode == 'tomask':
            # cum_alpha_t * v0 
            # (1 - cum_alpha_t) for mask
            log_probs = log_v0 + log_cumprod_alpha_t
            log_probs[:, -1] = log_1_min_cumprod_alpha.squeeze(1)
        else:
            raise ValueError("undefined v_mode: %s (expect uniform or tomask)" % (self.v_mode))
        return log_probs

    def q_v_sample(self, log_v0, t, batch, num_classes):
        """
        backward generative process q(vt)
        """
        log_qvt_v0 = self.q_v_pred(log_v0, t, batch)
        sample_index = log_sample_categorical(log_qvt_v0)
        log_sample = index_to_log_onehot(sample_index, num_classes)
        
        return sample_index, log_sample

    # atom type generative process
    def q_v_posterior(self, log_v0, log_vt, t, batch):
        # q(vt-1 | vt, v0) = q(vt | vt-1, x0) * q(vt-1 | x0) / q(vt | x0)
        t_minus_1 = t - 1
        # Remove negative values, will not be used anyway for final decoder
        t_minus_1 = torch.where(t_minus_1 < 0, torch.zeros_like(t_minus_1), t_minus_1)
        log_qvt1_v0 = self.q_v_pred(log_v0, t_minus_1, batch)
        unnormed_logprobs = log_qvt1_v0 + self.q_v_pred_one_timestep(log_vt, t, batch)
        log_vt1_given_vt_v0 = unnormed_logprobs - torch.logsumexp(unnormed_logprobs, dim=-1, keepdim=True)
        return log_vt1_given_vt_v0

    def kl_v_prior(self, log_x_start, batch):
        num_graphs = batch.max().item() + 1
        log_qxT_prob = self.q_v_pred(log_x_start, t=[self.num_timesteps - 1] * num_graphs, batch=batch)
        log_half_prob = -torch.log(self.num_classes * torch.ones_like(log_qxT_prob))
        kl_prior = categorical_kl(log_qxT_prob, log_half_prob)
        kl_prior = scatter_mean(kl_prior, batch, dim=0)
        return kl_prior

    def _predict_x0_from_eps(self, xt, eps, t, batch):
        pos0_from_e = extract(self.sqrt_recip_alphas_cumprod, t, batch) * xt - \
                      extract(self.sqrt_recipm1_alphas_cumprod, t, batch) * eps
        return pos0_from_e

    def q_pos_posterior(self, x0, xt, t, batch):
        # Compute the mean and variance of the diffusion posterior q(x_{t-1} | x_t, x_0)
        pos_model_mean = extract(self.posterior_mean_c0_coef, t, batch) * x0 + \
                         extract(self.posterior_mean_ct_coef, t, batch) * xt
        return pos_model_mean

    def kl_pos_prior(self, pos0, batch):
        num_graphs = batch.max().item() + 1
        a_pos = extract(self.alphas_cumprod, [self.num_timesteps - 1] * num_graphs, batch)  # (num_ligand_atoms, 1)
        pos_noise = torch.zeros_like(pos0)
        pos_noise.normal_()
        pos_perturbed = a_pos.sqrt() * pos0 + (1.0 - a_pos).sqrt() * pos_noise
        pos_prior = torch.randn_like(pos_perturbed)
        kl_prior = torch.mean((pos_perturbed - pos_prior) ** 2)
        return kl_prior

    def sample_time(self, num_graphs, device):
        time_step = torch.randint(
            0, self.num_timesteps, size=(num_graphs // 2 + 1,), device=device)
        time_step = torch.cat(
            [time_step, self.num_timesteps - time_step - 1], dim=0)[:num_graphs]
        pt = torch.ones_like(time_step).float() / self.num_timesteps
        return time_step, pt

    def compute_pos_Lt(self, pos_model_mean, x0, xt, t, batch):
        # fixed pos variance
        pos_log_variance = extract(self.posterior_logvar, t, batch)
        pos_true_mean = self.q_pos_posterior(x0=x0, xt=xt, t=t, batch=batch)
        kl_pos = normal_kl(pos_true_mean, pos_log_variance, pos_model_mean, pos_log_variance)
        kl_pos = kl_pos / np.log(2.)

        decoder_nll_pos = -log_normal(x0, means=pos_model_mean, log_scales=0.5 * pos_log_variance)
        assert kl_pos.shape == decoder_nll_pos.shape
        mask = (t == 0).float()[batch]
        loss_pos = scatter_mean(mask * decoder_nll_pos + (1. - mask) * kl_pos, batch, dim=0)
        # print('kl pos: ', kl_pos, 'nll pos: ', decoder_nll_pos, 'loss pos: ', loss_pos)
        return loss_pos

    def compute_v_Lt(self, log_v_model_prob, log_v0, log_v_true_prob, t, batch):
        kl_v = categorical_kl(log_v_true_prob, log_v_model_prob)  # [num_atoms, ]
        decoder_nll_v = -log_categorical(log_v0, log_v_model_prob)  # L0
        assert kl_v.shape == decoder_nll_v.shape
        mask = (t == 0).float()[batch]
        loss_v = scatter_mean(mask * decoder_nll_v + (1. - mask) * kl_v, batch, dim=0)
        # print('kl v: ', kl_v, 'nll v: ', decoder_nll_v, 'loss v: ', loss_v)
        return loss_v

    def compute_bond_angle_loss(self, n0_dst, pred_pos, gt_pos, bond_index, batch_bond):
        src, dst = bond_index

        # Calculate ground truth angles
        gt_pos_ji = gt_pos[src] - gt_pos[dst]
        gt_pos_j0 = gt_pos[n0_dst] - gt_pos[dst]

        gt_angle = compute_bond_angle(gt_pos_ji, gt_pos_j0)

        # Calculate predicted angles
        pred_pos_ji = pred_pos[src] - pred_pos[dst]
        pred_pos_j0 = pred_pos[n0_dst] - pred_pos[dst]

        pred_angle = compute_bond_angle(pred_pos_ji, pred_pos_j0)
        
        angle_mse = scatter_mean(((gt_angle - pred_angle) ** 2), batch_bond, dim=0)
        return angle_mse

    def compute_torsion_angle_loss(self, n0_src, n1_src, n0_dst, n1_dst, pred_pos, gt_pos, bond_index, batch_bond, torsion_type='one'):
        src, dst = bond_index

        gt_pos_ji =  gt_pos[dst] - gt_pos[src]
        pred_pos_ji = pred_pos[dst] - pred_pos[src]
        """
        # Calculate ground truth angles
        gt_pos_ji = gt_pos[src] - gt_pos[dst]
        gt_pos_j0 = gt_pos[n0_dst] - gt_pos[dst]
        gt_pos_j1 = gt_pos[n1_dst] - gt_pos[dst]
        gt_torsion_angle = compute_torsion_angle(gt_pos_ji, gt_pos_j0, gt_pos_j1)

        # Calculate predicted angles
        pred_pos_ji = pred_pos[src] - pred_pos[dst]
        pred_pos_j0 = pred_pos[n0_dst] - pred_pos[dst]
        pred_pos_j1 = pred_pos[n1_dst] - pred_pos[dst]
        pred_torsion_angle = compute_torsion_angle(pred_pos_ji, pred_pos_j0, pred_pos_j1)
        #print(gt_torsion_angle)
        #print(pred_torsion_angle)
        #gt_torsion_angle - - pred_torsion_angle
        torsion_angle_mse = scatter_mean(((gt_torsion_angle - pred_torsion_angle) ** 2), batch_bond, dim=0)
        #print(torsion_angle_mse)
        """
        # Calculate ground truth angles
        src_ref_mask = n0_src == dst
        src_ref = torch.clone(n0_src)
        src_ref[src_ref_mask] = n1_src[src_ref_mask]
        
        dst_ref_mask = n0_dst == src
        dst_ref = torch.clone(n0_dst)
        dst_ref[dst_ref_mask] = n1_dst[dst_ref_mask]
        
        gt_pos_src_ref = gt_pos[src_ref] - gt_pos[dst]
        gt_pos_dst_ref = gt_pos[dst_ref] - gt_pos[dst]
        gt_torsion_angle1 = compute_torsion_angle(gt_pos_ji, gt_pos_src_ref, gt_pos_dst_ref)

        pred_pos_src_ref = pred_pos[src_ref] - pred_pos[dst]
        pred_pos_dst_ref = pred_pos[dst_ref] - pred_pos[dst]
        pred_torsion_angle1 = compute_torsion_angle(pred_pos_ji, pred_pos_src_ref, pred_pos_dst_ref)
        
        angle_diff_1 = (gt_torsion_angle1 - pred_torsion_angle1)
        angle_diff_2 = 2 * torch.pi - (gt_torsion_angle1 - pred_torsion_angle1)
        angle_diff, _ = torch.min(torch.abs(torch.stack([angle_diff_1, angle_diff_2], dim=1)), dim=1)
        torsion_angle_mse = scatter_mean((angle_diff ** 2), batch_bond, dim=0)
        #torsion_angle_mse = (torsion_angle_mse + torsion_angle_mse1) / 2
        
        return torsion_angle_mse

    def get_diffusion_loss(self, ligand_pos, ligand_v, batch_ligand, ligand_bond_index, ligand_bond_type, ligand_shape=None, time_step=None, eval_mode=False):
        num_graphs = batch_ligand.max().item() + 1 
        ligand_pos, _ = center_pos(ligand_pos, batch_ligand, mode=self.center_pos_mode)
        
        # 1. sample noise levels
        if time_step is None:
            time_step, pt = self.sample_time(num_graphs, ligand_pos.device)
        else:
            pt = torch.ones_like(time_step).float() / self.num_timesteps
        a = self.alphas_cumprod.index_select(0, time_step)
        
        # 2. perturb pos and v
        a_pos = a[batch_ligand].unsqueeze(-1)  
        pos_noise = torch.zeros_like(ligand_pos)
        pos_noise.normal_()
        
        ligand_pos_perturbed = a_pos.sqrt() * ligand_pos + (1.0 - a_pos).sqrt() * pos_noise  # pos_noise * std
        if self.v_mode == 'uniform':
            # Vt = a * V0 + (1-a) / K
            log_ligand_v0 = index_to_log_onehot(ligand_v, self.num_classes)
            ligand_v_perturbed, log_ligand_vt = self.q_v_sample(log_ligand_v0, time_step, batch_ligand, self.num_classes)
        elif self.v_mode == 'tomask':
            # Vt = a * V0 + (1-a) * mask
            log_ligand_v0 = index_to_log_onehot(ligand_v, self.num_classes+1)
            ligand_v_perturbed, log_ligand_vt = self.q_v_sample(log_ligand_v0, time_step, batch_ligand, self.num_classes+1)
        else:
            raise ValueError
        
        # mask ligand shape
        ligand_shape = ligand_shape.view(num_graphs, -1, 3)
        
        if not eval_mode:
            cond_mask_probs = torch.ones(num_graphs) * (1 - self.cond_mask_prob)
            cond_mask = torch.bernoulli(cond_mask_probs).unsqueeze(1).to(ligand_shape.device)
            if len(ligand_shape.shape) == 3: cond_mask = cond_mask.unsqueeze(1)
            ligand_shape = cond_mask * ligand_shape
        
        # 3. forward-pass NN, feed perturbed pos and v, output noise
        preds = self(
            ligand_pos_perturbed=ligand_pos_perturbed,
            ligand_v_perturbed=ligand_v_perturbed,
            batch_ligand=batch_ligand,
            ligand_shape=ligand_shape,
            time_step=time_step,
            ligand_bond_index=ligand_bond_index,
            ligand_bond_type=ligand_bond_type,
            pred_bond=True
        )

        pred_ligand_pos, pred_ligand_v = preds['pred_ligand_pos'], preds['pred_ligand_v']
        
        # atom type
        log_ligand_v_recon = F.log_softmax(pred_ligand_v, dim=-1)
        log_v_model_prob = self.q_v_posterior(log_ligand_v_recon, log_ligand_vt, time_step, batch_ligand)
        log_v_true_prob = self.q_v_posterior(log_ligand_v0, log_ligand_vt, time_step, batch_ligand)
        
        kl_v = self.compute_v_Lt(log_v_model_prob=log_v_model_prob, log_v0=log_ligand_v0,
                                 log_v_true_prob=log_v_true_prob, t=time_step, batch=batch_ligand)
        
        
        loss_pos_weight = None
        if self.loss_weight_type == 'noise_level':
            loss_pos_weight = self.loss_pos_step_weight.index_select(0, time_step)
        
        # unweighted  
        target, pred = ligand_pos, pred_ligand_pos
        loss_pos = scatter_mean(((pred - target) ** 2).sum(-1), batch_ligand, dim=0)

        if self.loss_weight_type == 'uniform':
            loss_pos = torch.mean(loss_pos)
        elif self.loss_weight_type == 'noise_level':
            loss_pos = torch.mean(loss_pos_weight * loss_pos)
        
        loss_v = torch.mean(kl_v)
        
        loss = loss_pos + loss_v * self.loss_v_weight

        if self.pred_bond_type:
            pred_bond_losses = torch.stack(preds['all_bond_loss'], dim=1)
            pred_bond_types, gt_bond_types = preds['bond_pred'], preds['bond_gt']
            if self.loss_weight_type == 'noise_level':
                loss_bond_all = loss_pos_weight.unsqueeze(1) * pred_bond_losses
            loss_bond_aux = torch.mean(loss_bond_all[:, :-1])
            loss_bond_final = torch.mean(loss_bond_all[:, -1])
            loss += self.loss_bond_weight * (loss_bond_aux + loss_bond_final)
        else:
            loss_bond_aux, loss_bond_final, pred_bond_types, gt_bond_types = 0.0, 0.0, [], []
        
        pred_bond_dist = get_distance(pred_ligand_pos, ligand_bond_index)
        gt_bond_dist = get_distance(target, ligand_bond_index)
        batch_ligand_bond = torch.index_select(batch_ligand, 0, ligand_bond_index[0, :])
        loss_bond_dist = scatter_mean(((pred_bond_dist - gt_bond_dist) ** 2), batch_ligand_bond, dim=0)
        if self.loss_weight_type == 'noise_level':
            loss_bond_dist = loss_pos_weight * loss_bond_dist
        loss_bond_dist = torch.mean(loss_bond_dist)

        if self.use_bond_dist_loss:
            loss += self.loss_bond_dist_weight * loss_bond_dist
        #else:
        #    pred_bond_dist, gt_bond_dist, loss_bond_dist = None, None, 0.0

        src, dst = ligand_bond_index        
        n0, n1 = find_closest_points(gt_bond_dist, ligand_bond_index)
        
        n0_dst = torch.index_select(n0, 0, dst)
        n1_dst = torch.index_select(n1, 0, dst)

        loss_bond_angle = self.compute_bond_angle_loss(n0_dst, pred_ligand_pos, target, ligand_bond_index, batch_ligand_bond)
        if self.loss_weight_type == 'noise_level':
            loss_bond_angle = loss_pos_weight * loss_bond_angle
        loss_bond_angle = torch.mean(loss_bond_angle)

        n0_src = torch.index_select(n0, 0, src)
        n1_src = torch.index_select(n1, 0, src)

        loss_torsion_angle = self.compute_torsion_angle_loss(n0_src, n1_src, n0_dst, n1_dst, pred_ligand_pos, target, ligand_bond_index, 
                                                        batch_ligand_bond, torsion_type=self.loss_torsion_angle_type)
        
        if self.loss_weight_type == 'noise_level':
            loss_torsion_angle = loss_pos_weight * loss_torsion_angle
        loss_torsion_angle = torch.mean(loss_torsion_angle)

        if self.use_bond_angle_loss:
            loss += self.loss_bond_angle_weight * loss_bond_angle
            
        if self.use_torsion_angle_loss:
            loss += self.loss_torsion_angle_weight * loss_torsion_angle
        
        return {
            'loss_pos': loss_pos,
            'loss_v': loss_v,
            'loss_bond_aux': loss_bond_aux,
            'loss_bond_final': loss_bond_final,
            'loss_bond_dist': loss_bond_dist,
            'loss_bond_angle': loss_bond_angle,
            'loss_torsion_angle': loss_torsion_angle,
            'loss': loss,
            'x0': ligand_pos,
            'ligand_pos_perturbed': ligand_pos_perturbed,
            'ligand_v_perturbed': ligand_v_perturbed,
            'pred_ligand_pos': pred_ligand_pos,
            'pred_ligand_v': pred_ligand_v,
            'pred_bond_type': pred_bond_types,
            'gt_bond_type': gt_bond_types,
            'ligand_v_recon': F.softmax(pred_ligand_v, dim=-1)
        }

    @torch.no_grad()
    def sample_diffusion(self, init_ligand_pos, init_ligand_v, batch_ligand, ligand_shape, 
                         num_steps=None, center_pos_mode=None, use_mesh_data=None, use_pointcloud_data=None, 
                         use_pocket_data=None, grad_step=500, pred_bond=False, guide_stren=0, bounds=None, 
                         protein_pos=None, protein_ligand_dist_threshold=None):

        if num_steps is None:
            num_steps = self.num_timesteps
        num_graphs = batch_ligand.max().item() + 1
        print('sample center pos mode: ', center_pos_mode)

        if self.cond_mask_prob == 0:
            assert guide_stren == 0

        init_ligand_pos, offset = center_pos(init_ligand_pos, batch_ligand, mode=center_pos_mode)
        
        if self.config.shape_mode is not None:
            ligand_shape = ligand_shape.view(num_graphs, -1, 3)
        
        pos_traj, v_traj = [], []
        pos_cond_traj, v_cond_traj = [], []
        pos_uncond_traj, v_uncond_traj = [], []
        v0_pred_traj, vt_pred_traj = [], []
        ligand_pos, ligand_v = init_ligand_pos, init_ligand_v
        pred_bonds = None
        # time sequence
        time_seq = list(reversed(range(self.num_timesteps - num_steps, self.num_timesteps)))
        for i in tqdm(time_seq, desc='sampling', total=len(time_seq)):
            t = torch.full(size=(num_graphs,), fill_value=i, dtype=torch.long, device=ligand_pos.device)
            with torch.no_grad():
                if guide_stren != -1.0:
                    preds_with_cond = self(
                        ligand_pos_perturbed=ligand_pos,
                        ligand_v_perturbed=ligand_v,
                        batch_ligand=batch_ligand,
                        ligand_shape=ligand_shape,
                        if_test=True,
                        pred_bond=True if self.pred_bond_type and pred_bond and i == 0 else False,
                        time_step=t
                    )
            
            if self.pred_bond_type and pred_bond and i == 0:
                pred_bonds = torch.argmax(preds_with_cond['bond_pred'], dim=-1).view(-1, 1)
                pred_edge_indices = preds_with_cond['edge_index'].transpose(0, 1)
                pred_bonds = torch.cat([pred_edge_indices, pred_bonds], dim=1)

            preds = {}
            if use_mesh_data != None:
                pred_ligand_pos = preds_with_cond['pred_ligand_pos']
                if i > grad_step:
                    pred_ligand_pos = mesh_shape_guidance(use_mesh_data, pred_ligand_pos)

                preds['pred_ligand_pos'] = pred_ligand_pos
                preds['pred_ligand_v'] = preds_with_cond['pred_ligand_v']

            elif use_pointcloud_data is not None:
                pred_ligand_pos = preds_with_cond['pred_ligand_pos']
                if i > grad_step:
                    pred_ligand_pos = pointcloud_shape_guidance(use_pointcloud_data, pred_ligand_pos)

                preds['pred_ligand_pos'] = pred_ligand_pos
                preds['pred_ligand_v'] = preds_with_cond['pred_ligand_v']
                """
                elif use_grad:
                    preds_ligand_pos = preds_with_cond['pred_ligand_pos']
                    if i > grad_step:
                        for j in range(ligand_shape.shape[0]):
                            single_ligand_idxs = torch.where(batch_ligand == j)
                            single_ligand_pos = preds_ligand_pos[single_ligand_idxs].unsqueeze(0)
                            single_ligand_pos = torch.autograd.Variable(single_ligand_pos, requires_grad=True).cuda()
                            single_ligand_shape = ligand_shape[j, :, :].unsqueeze(0)
                            
                            pos_dists = shape_AE.generator(single_ligand_pos, single_ligand_shape)
                            pos_dists = torch.clip(pos_dists, max=0.5) - 0.5
                            neg_pos_dists = torch.mean(pos_dists)
                            if neg_pos_dists == 0: continue
                            single_grad = torch.autograd.grad(neg_pos_dists, single_ligand_pos)[0]
                            grad_pos_idxs = single_ligand_idxs
                            preds_ligand_pos[grad_pos_idxs] = preds_ligand_pos[grad_pos_idxs] - grad_lr * pos_dists.unsqueeze(2).repeat(1, 1, 3) * single_grad

                    preds['pred_ligand_pos'] = preds_ligand_pos
                    preds['pred_ligand_v'] = preds_with_cond['pred_ligand_v']
                    
                    pos_cond_traj.append(preds_with_cond['pred_ligand_pos'])
                    v_cond_traj.append(preds_with_cond['pred_ligand_v'])
                """
            elif self.cond_mask_prob > 0 and guide_stren != 0.0:
                mask_ligand_shape = torch.zeros_like(ligand_shape).to(ligand_shape.device)
                with torch.no_grad():
                    preds_without_cond = self(
                        ligand_pos_perturbed=ligand_pos,
                        ligand_v_perturbed=ligand_v,
                        batch_ligand=batch_ligand,
                        ligand_shape=mask_ligand_shape,
                        time_step=t,
                    )
                preds = preds_without_cond
                pos_cond_traj.append(preds_without_cond['pred_ligand_pos'])
                v_cond_traj.append(preds_without_cond['pred_ligand_v'])
            else:
                preds = preds_with_cond
                
                #preds['pred_ligand_v'] = preds_with_cond['pred_ligand_v']
            
            #if use_pocket_data != None:
            #    pocket_grad_step = use_pocket_data[-2]
            #    pred_ligand_pos = preds['pred_ligand_pos']
            #    if i > pocket_grad_step:
            #        pred_ligand_v = torch.argmax(preds['pred_ligand_v'], dim=1)
            #        pred_ligand_pos = pocket_guidance(use_pocket_data, pred_ligand_pos, pred_ligand_v, step=i)
            #        preds['pred_ligand_pos'] = pred_ligand_pos

            pos_cond_traj.append(preds_with_cond['pred_ligand_pos'])
            v_cond_traj.append(preds_with_cond['pred_ligand_v'])
            
            if self.v_mode == 'tomask':
                #mask the mask atom
                preds['pred_ligand_v'][:, -1] = -1.e5

            # Compute posterior mean and variance
            with torch.no_grad():
                pos0_from_e = preds['pred_ligand_pos']
                v0_from_e = preds['pred_ligand_v']
                
                pos_model_mean = self.q_pos_posterior(x0=pos0_from_e, xt=ligand_pos, t=t, batch=batch_ligand)
                pos_log_variance = extract(self.posterior_logvar, t, batch_ligand)
                # no noise when t == 0
                nonzero_mask = (1 - (t == 0).float())[batch_ligand].unsqueeze(-1)
                ligand_pos_next = pos_model_mean + nonzero_mask * (0.5 * pos_log_variance).exp() * torch.randn_like(
                    ligand_pos)

                ligand_pos = ligand_pos_next

                log_ligand_v_recon = F.log_softmax(v0_from_e, dim=-1)
                log_ligand_v = index_to_log_onehot(ligand_v, self.num_classes+int(self.v_mode=='tomask'))
                log_model_prob = self.q_v_posterior(log_ligand_v_recon, log_ligand_v, t, batch_ligand)
                ligand_v_next = log_sample_categorical(log_model_prob)

                if use_pocket_data != None:
                    pocket_grad_step = use_pocket_data[-2]
                    if i > pocket_grad_step:
                        ligand_pos = pocket_guidance(use_pocket_data, ligand_pos_next, ligand_v_next, step=i)

                v0_pred_traj.append(log_ligand_v_recon.clone().cpu())
                vt_pred_traj.append(log_model_prob.clone().cpu())
                ligand_v = ligand_v_next

                if center_pos_mode != 'none':
                    ori_ligand_pos = ligand_pos + offset[batch_ligand]
                else:
                    ori_ligand_pos = ligand_pos
                
            pos_traj.append(ori_ligand_pos.clone().cpu())
            v_traj.append(ligand_v.clone().cpu())
    
        if center_pos_mode != 'none':
            ligand_pos = ligand_pos + offset[batch_ligand]
        
        if pred_bond:

            return {
                'pos': ligand_pos,
                'v': ligand_v,
                'bond': pred_bonds,
                'pos_traj': pos_traj,
                'pos_cond_traj': pos_cond_traj,
                'pos_uncond_traj': pos_uncond_traj,
                'v_traj': v_traj,
                'v_cond_traj': v_cond_traj,
                'v_uncond_traj': v_uncond_traj,
                'v0_traj': v0_pred_traj,
                'vt_traj': vt_pred_traj
            }
        else:
            return {
                'pos': ligand_pos,
                'v': ligand_v,
                'bond': pred_bonds,
                'pos_traj': pos_traj,
                'pos_cond_traj': pos_cond_traj,
                'pos_uncond_traj': pos_uncond_traj,
                'v_traj': v_traj,
                'v_cond_traj': v_cond_traj,
                'v_uncond_traj': v_uncond_traj,
                'v0_traj': v0_pred_traj,
                'vt_traj': vt_pred_traj
            }

def pointcloud_shape_guidance(use_pointcloud_data, pred_ligand_pos, k=3, ratio=0.2):
    """
    apply additional point cloud shape guidance
    """
    point_clouds, kdtree, radius = use_pointcloud_data
    pred_ligand_pos_numpy = np.array(pred_ligand_pos.cpu())
    dists, point_idxs = kdtree.query(pred_ligand_pos_numpy, k=k)
    
    faratom_idxs = np.where(np.mean(dists, axis=1)>radius)[0]
    faratom_points = pred_ligand_pos_numpy[faratom_idxs, :]
    faratom_point_idxs = point_idxs[faratom_idxs, :]

    changed_faratom_idxs = []
    changed_faratom_points = []
    
    j = 0
    while len(faratom_idxs) > 0 and j < 5:
        # change outmesh_points
        faratom_nearest_points = np.mean(point_clouds[faratom_point_idxs, :], axis=1)
        distance_dir = faratom_points - faratom_nearest_points
        distance_scalar = np.expand_dims(np.random.random(faratom_points.shape[0]) * (0.8-ratio) + ratio, 1)
        new_faratom_points = faratom_points - distance_scalar * distance_dir

        dists, point_idxs = kdtree.query(new_faratom_points, k=3)
        
        contained = (np.mean(dists, axis=1)<radius)
        changed_faratom_idxs.extend(faratom_idxs[contained])
        changed_faratom_points.extend(new_faratom_points[contained, :])

        faratom_idxs = faratom_idxs[~contained]
        faratom_points = new_faratom_points[~contained]
        faratom_point_idxs = point_idxs[~contained, :]
        j += 1

    if j == 5:
        changed_faratom_idxs.extend(faratom_idxs)
        changed_faratom_points.extend(faratom_points)

    changed_faratom_idxs = torch.LongTensor(np.array(changed_faratom_idxs))
    pred_ligand_pos[changed_faratom_idxs, :] = torch.FloatTensor(np.array(changed_faratom_points)).cuda()

    return pred_ligand_pos

def mesh_shape_guidance(use_mesh_data, pred_ligand_pos, k=3, ratio=0.5):
    mesh, point_clouds, kdtree = use_mesh_data
    pred_ligand_pos_numpy = np.array(pred_ligand_pos.cpu())
    contained = mesh.contains(pred_ligand_pos_numpy)
    dists, point_idxs = kdtree.query(pred_ligand_pos_numpy)

    within_mesh_points = pred_ligand_pos_numpy[contained & (dists.squeeze(1) > 0.4), :]
    new_kdtree = KDTree(within_mesh_points)

    changed_outmesh_idxs = []
    changed_outmesh_points = []
    outmesh_idxs = np.where(~contained | (dists.squeeze(1) < 0.2))[0]
    outmesh_points = pred_ligand_pos_numpy[outmesh_idxs, :]
    j = 0
    while len(outmesh_idxs) > 0 and j < 5:
        # change outmesh_points
        dists, point_idxs = new_kdtree.query(outmesh_points, k=3)
        distance_dir = outmesh_points - np.mean(within_mesh_points[point_idxs, :], axis=1)
        distance_scalar = np.expand_dims(np.random.random(outmesh_points.shape[0]) * 0.8 + 0.2, 1)
        new_outmesh_points = outmesh_points - distance_scalar * distance_dir
        contained = mesh.contains(new_outmesh_points)

        dists1, _ = kdtree.query(new_outmesh_points)
        outmesh_idx_bool = contained & (dists1.squeeze(1) > 0.2)
        changed_outmesh_idxs.extend(outmesh_idxs[outmesh_idx_bool])
        changed_outmesh_points.extend(new_outmesh_points[outmesh_idx_bool, :])

        outmesh_idxs = outmesh_idxs[~outmesh_idx_bool]
        outmesh_points = new_outmesh_points[~outmesh_idx_bool]
        j += 1

    changed_outmesh_idxs = torch.LongTensor(np.array(changed_outmesh_idxs))
    pred_ligand_pos[changed_outmesh_idxs, :] = torch.FloatTensor(np.array(changed_outmesh_points)).cuda()
    return pred_ligand_pos
                              
def extract(coef, t, batch):
    out = coef[t][batch]
    return out.unsqueeze(-1)

def pocket_guidance2(use_pocket_data, pred_ligand_pos, pred_ligand_v, step=0, k=3, ratio=0.5):
    """
    apply additional point cloud shape guidance
    """
    pocket_atom_pos, pocket_atom_elems, kdtree, protein_ligand_mat, _ = use_pocket_data
    pred_ligand_pos_numpy = np.array(pred_ligand_pos.cpu())
    pred_ligand_v_numpy = np.array(pred_ligand_v.cpu())
    dists, k_point_idxs = kdtree.query(pred_ligand_pos_numpy, k=k)
    
    closeatom_idxs = set()
    for i in range(k):
        close_point_idxs = k_point_idxs[:, i]
        close_protein_elems = pocket_atom_elems[close_point_idxs]
        close_protein_ligand_threshold = protein_ligand_mat[pred_ligand_v_numpy, close_protein_elems]
        closeatom_idxs.update(np.where(dists[:, i] < close_protein_ligand_threshold)[0].tolist())
    closeatom_idxs = np.array(list(closeatom_idxs))
    if len(closeatom_idxs) == 0: return pred_ligand_pos
    print("num closeatom idx %d" % (len(closeatom_idxs)))
    closeatom_points = pred_ligand_pos_numpy[closeatom_idxs, :]
    closeatom_point_idxs = k_point_idxs[closeatom_idxs, :]
    changed_closeatom_idxs = []
    changed_closeatom_points = []
    
    j = 0
    while len(closeatom_idxs) > 0 and j < 5:
        # change outmesh_points
        try:
            closeatom_nearest_points = np.mean(pocket_atom_pos[closeatom_point_idxs, :], axis=1)
        except:
            pdb.set_trace()
        distance_dir = closeatom_points - closeatom_nearest_points
        distance_val = np.sqrt(np.sum(distance_dir ** 2, axis=1))
        unit_dir = distance_dir / distance_val.reshape(-1, 1)
        distance_scalar = np.expand_dims(np.random.random(closeatom_points.shape[0]) * (1-ratio) + ratio, 1)
        new_closeatom_points = closeatom_points + distance_scalar * unit_dir
        
        dists, k_point_idxs = kdtree.query(new_closeatom_points, k=k)
        
        faratom_idxs = np.zeros_like(k_point_idxs)
        for i in range(k):
            far_point_idxs = k_point_idxs[:, i]
            far_protein_elems = pocket_atom_elems[far_point_idxs]
            far_protein_ligand_threshold = protein_ligand_mat[pred_ligand_v_numpy[closeatom_idxs], far_protein_elems]
            faratom_idxs[np.where(dists[:, i] > far_protein_ligand_threshold)[0], i] = 1
        far_atom = (np.sum(faratom_idxs, axis=1) == 3)
        changed_closeatom_idxs.extend(closeatom_idxs[far_atom])
        changed_closeatom_points.extend(new_closeatom_points[far_atom, :])

        closeatom_idxs = closeatom_idxs[~far_atom]
        closeatom_points = new_closeatom_points[~far_atom]
        closeatom_point_idxs = k_point_idxs[~far_atom, :]
        j += 1

    if j == 5:
        changed_closeatom_idxs.extend(closeatom_idxs)
        changed_closeatom_points.extend(closeatom_points)
    #print("step %d: %d close atoms" % (step, len(closeatom_idxs)))
    changed_closeatom_idxs = torch.LongTensor(np.array(changed_closeatom_idxs))
    if len(changed_closeatom_idxs) > 0:
        pred_ligand_pos[changed_closeatom_idxs, :] = torch.FloatTensor(np.array(changed_closeatom_points)).cuda()
    return pred_ligand_pos

def pocket_guidance(use_pocket_data, pred_ligand_pos, pred_ligand_v, step=0, k=3):
    """
    apply additional point cloud shape guidance
    """
    pocket_atom_pos, pocket_atom_elems, kdtree, protein_ligand_mat, _, ratio = use_pocket_data
    pred_ligand_pos_numpy = np.array(pred_ligand_pos.cpu())
    pred_ligand_v_numpy = np.array(pred_ligand_v.cpu())
    dists, k_point_idxs = kdtree.query(pred_ligand_pos_numpy, k=k)
    
    closeatom_idxs = set()
    closeatom_dists = np.zeros((pred_ligand_pos.shape[0]))
    for i in range(k):
        close_point_idxs = k_point_idxs[:, i]
        close_protein_elems = pocket_atom_elems[close_point_idxs]
        close_protein_ligand_threshold = protein_ligand_mat[pred_ligand_v_numpy, close_protein_elems]
        closeatom_idx = np.where(dists[:, i] < close_protein_ligand_threshold)[0]
        threshold_dist = close_protein_ligand_threshold[closeatom_idx] - dists[closeatom_idx, i]
        try:
            closeatom_dists[closeatom_idx] = np.max([closeatom_dists[closeatom_idx], threshold_dist], axis=0)
        except:
            pdb.set_trace()
        closeatom_idxs.update(closeatom_idx.tolist())
    closeatom_idxs = np.array(list(closeatom_idxs))

    if len(closeatom_idxs) == 0: return pred_ligand_pos
    try:
        closeatom_dists = closeatom_dists[closeatom_idxs]
    except:
        pdb.set_trace()
    closeatom_points = pred_ligand_pos_numpy[closeatom_idxs, :]
    closeatom_point_idxs = k_point_idxs[closeatom_idxs, :]
    #print("step %d: %d close atoms" % (step, len(closeatom_idxs)))
    changed_closeatom_idxs = []
    changed_closeatom_points = []
    
    j = 0
    while len(closeatom_idxs) > 0 and j < 5:
        # change outmesh_points
        try:
            closeatom_nearest_points = np.mean(pocket_atom_pos[closeatom_point_idxs, :], axis=1)
        except:
            pdb.set_trace()
        distance_dir = closeatom_points - closeatom_nearest_points
        distance_val = np.sqrt(np.sum(distance_dir ** 2, axis=1)).reshape(-1, 1)
        unit_dir = distance_dir / distance_val
        distance_scalar = closeatom_dists + np.random.random(len(closeatom_dists)) * ratio
        #if step < 100: pdb.set_trace()
        new_closeatom_points = closeatom_points + distance_scalar.reshape(-1, 1) * unit_dir
        
        dists, k_point_idxs = kdtree.query(new_closeatom_points, k=k)
        
        faratom_idxs = np.zeros_like(k_point_idxs)
        closeatom_dists = np.zeros((len(k_point_idxs)))
        for i in range(k):
            far_point_idxs = k_point_idxs[:, i]
            far_protein_elems = pocket_atom_elems[far_point_idxs]
            far_protein_ligand_threshold = protein_ligand_mat[pred_ligand_v_numpy[closeatom_idxs], far_protein_elems]
            far_idxs = np.where(dists[:, i] > far_protein_ligand_threshold)[0]
            faratom_idxs[far_idxs, i] = 1
            close_idxs = np.where(dists[:, i] < far_protein_ligand_threshold)[0]
            close_dist = far_protein_ligand_threshold[close_idxs] - dists[close_idxs, i]
            closeatom_dists[close_idxs] = np.max([closeatom_dists[close_idxs], close_dist], axis=0)

        far_atom = (np.sum(faratom_idxs, axis=1) == 3)
        changed_closeatom_idxs.extend(closeatom_idxs[far_atom])
        changed_closeatom_points.extend(new_closeatom_points[far_atom, :])

        closeatom_idxs = closeatom_idxs[~far_atom]
        closeatom_dists = closeatom_dists[~far_atom]
        closeatom_points = new_closeatom_points[~far_atom]
        closeatom_point_idxs = k_point_idxs[~far_atom, :]
        j += 1

    #print("still step %d: %d close atoms" % (step, len(closeatom_idxs)))
    if j == 5:
        changed_closeatom_idxs.extend(closeatom_idxs)
        changed_closeatom_points.extend(closeatom_points)
    #print("step %d: %d close atoms" % (step, len(closeatom_idxs)))
    changed_closeatom_idxs = torch.LongTensor(np.array(changed_closeatom_idxs))
    if len(changed_closeatom_idxs) > 0:
        pred_ligand_pos[changed_closeatom_idxs, :] = torch.FloatTensor(np.array(changed_closeatom_points)).cuda()
    return pred_ligand_pos
