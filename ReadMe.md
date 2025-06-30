# Generating 3D Small Binding Molecules Using Shape-Conditioned Diffusion models

This is the implementation of our DiffSMol model. This paper has been accepted by the journal "Nature Machine Intelligence".

This paper is an extended implementation of our workshop paper published in NeurIPS@GenBio 2023 workshop: https://openreview.net/pdf?id=JWfvMT43pZ


### Dependencies

#### Install via Conda and Pip

```
conda create -n diffsmol python=3.9 -y
conda activate diffsmol
 
conda install pytorch=1.11.0 cudatoolkit=11.5 -c pytorch -c conda-forge
conda install -c pytorch3d pytorch3d
 
conda install -c conda-forge rdkit openbabel tensorboard pyyaml easydict python-lmdb numpy=1.23.5 scipy=1.9.3 scikit-learn=1.1.3 scikit-image oddt
 
pip install torch-scatter==2.0.9 torch-sparse==0.6.13 torch-cluster==1.6.0 torch-spline-conv==1.2.1 -f https://data.pyg.org/whl/torch-1.11.0+cu115.html
pip install torch-geometric==2.0.4
 
pip install meeko==0.1.dev3 scipy pdb2pqr vina==1.2.2
pip install git+https://github.com/Valdes-Tresanco-MS/AutoDockTools_py3
 
pip install trimesh
```

As we use openeye to calculate the similarity values, the license of openeye might be required in order to run our code. Please check the website of openeye (https://www.eyesopen.com/academic-licensing) about how to get their license.

You can also use shaep (https://users.abo.fi/mivainio/shaep/index.php) for free to calculate shaep similarities as suggested by (https://github.com/keiradams/SQUID). 

Due to the limitations of file size, we provide our generated data and our processed training data at the link https://zenodo.org/records/14855728.

### Training

#### Shape Embedding from SDF

Please use the command below under the directory "source" to train the shape autoencoder for shape embeddings:
```
python -m scripts.train_shapeAE ../config/shape/se_VNDGCNN_hidden256_latent128_eclayer4_dclayer2_signeddist_pc512.yml --logdir <path to save trained models>
```
Please check the config file "se_VNDGCNN_hidden256_latent128_eclayer4_dclayer2_signeddist_pc512.yml" for available configs.

Please note that if no processed dataset exists in the data directory, the above script will first preprocess the molecules from the training set and save them. The preprocessing of shape dataset can take a few hours. Unfortunately, we are unable to share our processed dataset due to its substantial size of over 20GB.


#### Molecule Generative Diffusion model

Please use the command below to train the diffusion model:
```
python -m scripts.train_diffusion ../config/training/diff_pos0_10_pos1.e-7_0.01_6_v001_bondTrue_scalar128_vec32_layer8.yml --logdir <path to save trained models>
```
Same as above, the above script will first preprocess the dataset if no processed dataset exists. Preprocessing the dataset for the diffusion model could take ~15 hours on our GPU. The longer time is mainly due to the prior calculation of shape embeddings using the trained shape model. The size of the processed dataset is ~ 30GB.

If you decide to train the shape embedding model by yourself, please update the path of shape model checkpoint in the config file.


### Trained models

We provided our trained models in the directory "trained_models".

Please use the command below to test the trained diffusion model for shape-conditioned molecule generation:

```
python -m scripts.sample_diffusion_no_pocket ../config/sampling/SMG/sample_diff_pos0_10_pos1.e-7_0.01_6_v001_bondTrue_scalar128_vec32_layer8_with_guidance.yml --data_id <index of molecule> --result_path ./result/with_guide/
```

```
python -m scripts.sample_diffusion_no_pocket ../config/sampling/SMG/sample_diff_pos0_10_pos1.e-7_0.01_6_v001_bondTrue_scalar128_vec32_layer8_no_guidance.yml --data_id <index of molecule> --result_path ./result/without_guide/
```

Please use the command below to test the trained diffusion model for pocket-conditioned molecule generation:

```
python -m scripts.sample_diffusion_with_pocket ../config/sampling/PMG/sample_diff_pos0_10_pos1.e-7_0.01_6_v001_scalar128_vec32_layer8_with_pocket_guidance_with_shape_guidance.yml --data_id <index of molecule> --result_path ./result/pocket_with_guide/
```

```
python -m scripts.sample_diffusion_with_pocket ../config/sampling/PMG/sample_diff_pos0_10_pos1.e-7_0.01_6_v001_scalar128_vec32_layer8_with_pocket_guidance_no_shape_guidance.yml --data_id <index of molecule> --result_path ./result/pocket_without_guide/
```

Here, "data_id" denotes the index of molecules. Please note that lmdb, which we used to save processed dataset, uses string order of indices to reindex the molecules. Please check "index_map.txt" in data directory to find the mapping between the value of data_id and its corresponding index in the test dataset.

### Results

We provided our results for PMG and SMG reported in the paper at the link https://zenodo.org/records/14855728. The results can be loaded with torch.load("filename") function. 

### Evaluation

We provided code to evaluate the quality of generated molecules.

For pocket-conditioned molecule generation:
```
python -m scripts.evaluate_new_dock_shapemol ./result/pocket_with_guide/ --docking_mode vina_dock --protein_root <test protein data path>
```

To evaluate the docking quality, please download the test data provided by TargetDiff (https://github.com/guanjq/targetdiff) into the test protein data path. Please note that docking is a very time-consuming task, so we also provide the option to evaluate all the generated molecules for each single pocket by adding the option "--eval_id <id>". Since the latest versions of some dependency packages, such as meeko, are not compatible to the evaluation code, please download the required package based on the provided version in the environment file.

For shape-conditioned molecule generation:
```
python -m scripts.evaluate_diffusion_sim ./result/with_guide/ ./data/MOSES2/MOSES2_test_mol.pkl
```



