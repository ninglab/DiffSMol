import numpy as np
from math import ceil, pi
from utils.tfbio_data import make_grid, ROTATIONS
import random
import copy
from rdkit.Chem import rdMolTransforms
from rdkit import Geometry
from utils.tfbio_data import rotation_matrix
import utils.misc as misc

import oddt
import oddt.surface
from oddt import toolkit
from oddt.shape import electroshape

import torch
from rdkit import Chem
from models.shape_modelAE import IM_AE
from models.shape_pointcloud_modelAE import PointCloud_AE
from torch.autograd import Variable
from pytorch3d.structures.meshes import Meshes
from pytorch3d.ops import sample_points_from_meshes
import trimesh
import pdb
from sklearn.neighbors import KDTree

# van der Waals radius
ATOM_RADIUS = {
    'C': 1.908,
    'F': 1.75,
    'Cl': 1.948,
    'Br': 2.22,
    'I': 2.35,
    'N': 1.824,
    'O': 1.6612,
    'P': 2.1,
    'S': 2.0,
    'Si': 2.2 # not accurate
}

# atomic number
ATOMIC_NUMBER = {
    'C': 6,
    'F': 9,
    'Cl': 17,
    'Br': 35,
    'I': 53,
    'N': 7,
    'O': 8,
    'P': 15,
    'S': 16,
    'Si': 14
}

ATOMIC_NUMBER_REVERSE = {v: k for k, v in ATOMIC_NUMBER.items()}

def get_atom_stamp(grid_resolution, max_dist):
    # atom stamp is a sphere which radius equal to atom van der Waals radius
    def _get_atom_stamp(symbol):
        box_size = ceil(2 * max_dist // grid_resolution + 1)

        x, y, z = np.indices((box_size, box_size, box_size))
        x = x * grid_resolution + grid_resolution / 2
        y = y * grid_resolution + grid_resolution / 2
        z = z * grid_resolution + grid_resolution / 2

        mid = (box_size // 2, box_size // 2, box_size // 2)
        mid_x = x[mid]
        mid_y = y[mid]
        mid_z = z[mid]
        
        sphere = (x - mid_x)**2 + (y - mid_y)**2 + (z - mid_z)**2 \
            <= ATOM_RADIUS[symbol]**2
        sphere = sphere.astype(int)
        sphere[sphere > 0] = ATOMIC_NUMBER[symbol]
        return sphere

    atom_stamp = {}
    for symbol in ATOM_RADIUS:
        atom_stamp[symbol] = _get_atom_stamp(symbol)
    return atom_stamp

def get_atom_prop(atom, prop_name):
    if atom.HasProp(prop_name):
        return atom.GetProp(prop_name)
    else:
        return None

def get_binary_features(mol, confId, without_H):
    coords = []
    features = []
    confermer = mol.GetConformer(confId)
    for atom in mol.GetAtoms():
        if atom.HasProp('mask') and get_atom_prop(atom, 'mask') == 'true':
            continue
        idx = atom.GetIdx()
        syb = atom.GetSymbol()
        if without_H and syb == 'H':
            continue
        coord = list(confermer.GetAtomPosition(idx))
        coords.append(coord)
        features.append(atom.GetAtomicNum())
    coords = np.array(coords)
    features = np.array(features)
    features = np.expand_dims(features, axis=1)
    return coords, features

def get_voxel_shape(mol, atom_stamp, grid_resolution, max_dist, confId=-1, without_H=True, by_coords=False, coords=None, features=None):
    # expand each atom point to a sphere
    if not by_coords:
        coords, features = get_binary_features(mol, confId, without_H)
    grid, atomic2grid = make_grid(coords, features, grid_resolution, max_dist)
    shape = np.zeros(grid[0, :, :, :, 0].shape)
    for tup in atomic2grid:
        atomic_number = int(tup[0])
        stamp = atom_stamp[ATOMIC_NUMBER_REVERSE[atomic_number]]
        for grid_ijk in atomic2grid[tup]:
            i = grid_ijk[0]
            j = grid_ijk[1]
            k = grid_ijk[2]

            x_left = i - stamp.shape[0] // 2 if i - stamp.shape[0] // 2 > 0 else 0
            x_right = i + stamp.shape[0] // 2 if i + stamp.shape[0] // 2 < shape.shape[0] else shape.shape[0] - 1
            x_l = i - x_left
            x_r = x_right - i

            y_left = j - stamp.shape[1] // 2 if j - stamp.shape[1] // 2 > 0 else 0
            y_right = j + stamp.shape[1] // 2 if j + stamp.shape[1] // 2 < shape.shape[1] else shape.shape[1] - 1
            y_l = j - y_left
            y_r = y_right - j

            z_left = k - stamp.shape[2] // 2 if k - stamp.shape[2] // 2 >0 else 0
            z_right = k + stamp.shape[2] // 2 if k + stamp.shape[2] // 2 < shape.shape[2] else shape.shape[2] - 1
            z_l = k - z_left
            z_r = z_right - k

            mid = stamp.shape[0] // 2
            shape_part =  shape[x_left: x_right + 1, y_left: y_right + 1, z_left: z_right + 1]
            stamp_part = stamp[mid - x_l: mid + x_r + 1, mid - y_l: mid + y_r + 1, mid - z_l: mid + z_r + 1]

            shape_part += stamp_part
    shape[shape > 0] = 1
    return shape


def get_voxel_shape(mol, atom_stamp, grid_resolution, max_dist, confId=-1, without_H=True, by_coords=False, coords=None, features=None):
    # expand each atom point to a sphere
    if not by_coords:
        coords, features = get_binary_features(mol, confId, without_H)
    grid, atomic2grid = make_grid(coords, features, grid_resolution, max_dist)
    shape = np.zeros(grid[0, :, :, :, 0].shape)
    for tup in atomic2grid:
        atomic_number = int(tup[0])
        stamp = atom_stamp[ATOMIC_NUMBER_REVERSE[atomic_number]]
        for grid_ijk in atomic2grid[tup]:
            i = grid_ijk[0]
            j = grid_ijk[1]
            k = grid_ijk[2]

            x_left = i - stamp.shape[0] // 2 if i - stamp.shape[0] // 2 > 0 else 0
            x_right = i + stamp.shape[0] // 2 if i + stamp.shape[0] // 2 < shape.shape[0] else shape.shape[0] - 1
            x_l = i - x_left
            x_r = x_right - i

            y_left = j - stamp.shape[1] // 2 if j - stamp.shape[1] // 2 > 0 else 0
            y_right = j + stamp.shape[1] // 2 if j + stamp.shape[1] // 2 < shape.shape[1] else shape.shape[1] - 1
            y_l = j - y_left
            y_r = y_right - j

            z_left = k - stamp.shape[2] // 2 if k - stamp.shape[2] // 2 >0 else 0
            z_right = k + stamp.shape[2] // 2 if k + stamp.shape[2] // 2 < shape.shape[2] else shape.shape[2] - 1
            z_l = k - z_left
            z_r = z_right - k

            mid = stamp.shape[0] // 2
            shape_part =  shape[x_left: x_right + 1, y_left: y_right + 1, z_left: z_right + 1]
            stamp_part = stamp[mid - x_l: mid + x_r + 1, mid - y_l: mid + y_r + 1, mid - z_l: mid + z_r + 1]

            shape_part += stamp_part
    shape[shape > 0] = 1
    return shape

def sample_augment(sample, rotation_bin, confId=-1):
    sample = copy.deepcopy(sample)
    conformer = sample.GetConformer(confId)

    rot = random.choice(range(rotation_bin))
    rotation_mat = ROTATIONS[rot]

    # rotation the molecule
    rotation = np.zeros((4, 4))
    rotation[:3, :3] = rotation_mat
    rdMolTransforms.TransformConformer(conformer, rotation)
    
    return sample

def get_grid_coords(coords, max_dist, grid_resolution):
    grid_coords = (coords + max_dist) / grid_resolution
    grid_coords = grid_coords.round().astype(int)
    return grid_coords

def get_rotation_bins(sp, rp):
    mid = sp // 2
    sr = 1.0 / sp

    face1 = []
    for y in range(sp):
        for z in range(sp):
            face1.append(np.array([0.5, (y - mid) * sr, (z - mid) * sr]))
    face2 = []
    for x in range(sp):
        for y in range(sp):
            face2.append(np.array([(x - mid) * sr, (y - mid) * sr, 0.5]))
    face3 = []
    for x in range(sp):
        for z in range(sp):
            face3.append(np.array([(x - mid) * sr, 0.5, (z - mid) * sr]))
    
    face_point = face1 + face2 + face3
    
    rotation_mat_bin = [rotation_matrix(np.array((1, 1, 1)), 0)]
    for p in face_point:
        for t in range(1, rp):
            axis = p
            theta = t * pi / (rp / 2)
            rotation_mat_bin.append(rotation_matrix(axis, theta))
    rotation_mat_bin = np.stack(rotation_mat_bin, axis=0)

    return rotation_mat_bins

def get_mesh(mol, center_pos=False, confId=-1, scaling=1.0, probe_radius=1.4):
    if center_pos:
        conformer = mol.GetConformer(confId)
        center = rdMolTransforms.ComputeCentroid(conformer)
        if max(np.abs([center.x, center.y, center.z])) > 0.1:
            pos = np.array(conformer.GetPositions(), dtype=np.float64)
            offset = np.mean(pos, axis=0)
            pos = pos - offset
            for i in range(pos.shape[0]):
                point = Geometry.Point3D(pos[i, 0], pos[i, 1], pos[i, 2])
                conformer.SetAtomPosition(i, point)

    oddtconf = Chem.MolToMolBlock(mol, confId=confId)
    oddtconftool = toolkit.readstring('sdf', oddtconf)
    verts, faces = oddt.surface.generate_surface_marching_cubes(oddtconftool, scaling=scaling, probe_radius=probe_radius)
    return verts, faces

def get_pointcloud_from_mol(poses, confId=-1, N=20, var = 1./(12.*1.7)):
    point_clouds = []
    for i in range(poses.shape[0]):
        points = np.random.multivariate_normal(poses[i, :], [[var, 0, 0], [0, var, 0], [0, 0, var]], size=N, check_valid='warn', tol=1e-8)
        point_clouds.append(points)
    point_clouds = np.concatenate(point_clouds, axis=0)
    return point_clouds

def get_pointcloud_from_mesh(mesh, num_samples, return_mesh=False):
    mesh = Meshes(verts=[torch.FloatTensor(mesh[0].copy())], faces=[torch.FloatTensor(mesh[1].copy())])
    point_clouds = sample_points_from_meshes(mesh, num_samples)
    if not return_mesh:
        return point_clouds
    else:
        return point_clouds, mesh

def build_voxel_shapeAE_model(config, device='cpu'):
    #if not torch.cuda.is_available(): device = 'cpu'
    ckpt = torch.load(config.checkpoint, map_location=device)
    model = IM_AE(
        ckpt['config'].model
    ).to(device)
    model.load_state_dict(ckpt['model'], strict=False if 'train_config' in config else True)
    for variable in model.parameters():
        variable.detach_()
    return model

def build_point_shapeAE_model(config, device='cpu', detach=True):
    #if not torch.cuda.is_available(): device = 'cpu'
    ckpt = torch.load(config.checkpoint, map_location=device)
    model = PointCloud_AE(
        ckpt['config'].model
    ).to(device)
    model.load_state_dict(ckpt['model'], strict=False if 'train_config' in config else True)
    if detach:
        for variable in model.parameters():
            variable.detach_()
    return model

def get_electro_shape_emb(mols):
    shape_embs = []
    for mol in mols:
        oddtconf = Chem.MolToMolBlock(mol)
        oddtconftool = oddt.toolkit.readstring('sdf', oddtconf)
        oddtconftool.calccharges()
        shape_emb = electroshape(oddtconftool)
        shape_embs.append(shape_emb)
    return shape_embs

def get_voxelAE_shape_emb(mols, model, atom_stamp, grid_resolution=0.5, max_dist=11, shape_parallel=False, batch_size=32):
    batch_voxels = []
    for i in range(0, len(mols) + batch_size - 1, batch_size):
        batch_mols = mols[i:min(i+batch_size, len(mols))]
        if len(batch_mols) == 0: break
        batch_voxel = []
        for mol in batch_mols:
            voxel = get_voxel_shape(mol, atom_stamp, 
                                    grid_resolution=grid_resolution, 
                                    max_dist=max_dist)
            voxel = torch.from_numpy(voxel).to(torch.float32)
            batch_voxel.append(voxel)
        
        #if not torch.cuda.is_available(): device = 'cpu'
        #else: device = 'cuda'
        
        batch_voxel = torch.stack(batch_voxel).unsqueeze(1)
        batch_voxels.append(batch_voxel)
    
    zs = model.encode(batch_voxels)
    
    return zs

def get_pointAE_shape_emb(mols, model, point_cloud_samples, config, shape_parallel=False, batch_size=32):
    batch_point_clouds = []
    batch_point_cloud_centers = []
    batch_bounds = []
    remove_idxs = []
    for i in range(0, len(mols) + batch_size - 1, batch_size):
        batch_mols = mols[i:min(i+batch_size, len(mols))]
        if len(batch_mols) == 0: break
        batch_point_cloud = []
        batch_point_cloud_center = []
        batch_bound = []

        for k, mol in enumerate(batch_mols):
            try:
                mesh = get_mesh(mol)
            except Exception as e:
                print("cannot generate mesh for %d-th molecule %s due to the error: %s" % (i+k, Chem.MolToSmiles(mol), e))
                remove_idxs.append(i + k)
                continue
            point_cloud, mesh_ = get_pointcloud_from_mesh(mesh, point_cloud_samples, return_mesh=True)
            point_cloud = point_cloud.squeeze(0)
            point_cloud_center = torch.mean(point_cloud, dim=0)
            point_cloud = point_cloud - point_cloud_center
            batch_point_cloud.append(point_cloud)
            batch_point_cloud_center.append(point_cloud_center)
            bound = mesh_.get_bounding_boxes().squeeze(0)
            bound = bound.transpose(1, 0) - point_cloud_center
            batch_bound.append(bound.transpose(1, 0))
            
        #if not torch.cuda.is_available(): device = 'cpu'
        #else: device = 'cuda'
        if len(batch_point_cloud) == 0:
            continue
        
        batch_point_cloud = torch.stack(batch_point_cloud).unsqueeze(1)
        batch_point_cloud_center = torch.stack(batch_point_cloud_center)      
        batch_point_clouds.append(batch_point_cloud)
        batch_point_cloud_centers.append(batch_point_cloud_center)
        batch_bound = torch.stack(batch_bound, axis=0)
        batch_bounds.append(batch_bound)

    if len(batch_point_clouds) == 0:
        return (remove_idxs, None, batch_bounds, batch_point_clouds, batch_point_cloud_centers)
    
    if shape_parallel:
        zs = model.encode(batch_point_clouds)
    else:
        batch_point_clouds = torch.cat(batch_point_clouds, dim=0).to('cuda')
        zs = model.encoder(batch_point_clouds).detach().cpu()

    batch_bounds = torch.cat(batch_bounds, dim=0)
    batch_point_cloud_centers = torch.cat(batch_point_cloud_centers, dim=0)
    return (remove_idxs, zs, batch_bounds, batch_point_clouds, batch_point_cloud_centers)