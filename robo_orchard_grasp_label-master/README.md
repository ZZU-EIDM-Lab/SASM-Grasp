# Generate Grasp Label
This repository provides a set of tools to generate grasp labels for objects in a simulated environment. The process involves converting mesh models to USD format, generating cluster scenes, and creating both object-level and scene-level grasp labels.

## Installation
Linux (Tested on Ubuntu 22.04), Python >= 3.10 (Tested on 3.10), torch >= 2.5.1


For cluster scene generation, you need to install the [NVIDIA Isaac Lab](https://isaac-sim.github.io/IsaacLab/v2.0.2/) environment, which is a prerequisite for running the simulation. The code is tested on the NVIDIA Isaac Lab 2.0.2 with Isaac Sim 4.5.0. Please refer to the [Isaac Lab installation guide](https://isaac-sim.github.io/IsaacLab/v2.0.2/source/setup/installation/pip_installation.html) for more details.


## Prerequest
Before label generation, please make sure you have prepared the following files:
- `models`: The mesh models, which should be in `.obj` or `.ply` format.
- `table`: The table usd file, which should be in `.usd` format.
- `T_cam_to_table.npz`: The camera-to-table transformation matrices that enable observations from 256 predefined viewpoints distributed across a quarter-spherical trajectory, as in [GraspNet-1Billion](https://graspnet.net/)

## Usages
Step1: convert the mesh model into USD format and generate the object_params.
```bash
python3 examples/mesh_to_usd.py --in_path './data/models' --out_path './data/usds/'
```

Step2: generate cluster scene using the generated USD files.
```bash
python3 examples/generate_sim_scene.py \
    --num_objects 6 \
    --asset_dir './data/usds' \
    --save_dir './data/scenes' \
    --table_asset_path './data/table/desk_top/eggere/usd/eggere.usd' \
    --t_cam_to_table_path './data/T_cam_to_table.npz' \
    --scene_id 0 \
    --enable_cameras \
    --headless
```

Step3: generate object-level grasp label for each mesh model.
```bash
# generate object sdf
python3 examples/generate_object_sdf.py --input_file './data/models/SCHOOL_BUS/meshes/model.obj'
# generate object label
python3 examples/generate_object_label.py \
    --model_dir './data/model' \
    --output_dir './data/grasp_labels'
```

Step4: generate scene-level grasp label for generated cluster scene.
```bash
python3 examples/generate_scene_label.py \
    --scene_dir './data/scenes' \
    --model_dir './data/models'\
    --grasp_label_dir './data/grasp_labels'\
    --output_dir './data' \
```
Step5: generate economic grasp label for each cluster scene.
```bash
git clone https://github.com/iSEE-Laboratory/EconomicGrasp.git
python3 dataset/generate_economic.py
```

## Dataset Structure
```text
data
└──models # mesh models
|   ├──SCHOOL_BUS
|   |   ├──meshes
|   |   |   ├──model.obj
|   |   |   ├──model.mtl
|   |   |   ├──model.sdf
|   |   |   ├──texture.png
|   |   |   └──...
|   |   └──...
|   └──...
└──usds # converted usd files
|   ├──SCHOOL_BUS
|   |   ├──textures
|   |   ├──model_converted.usd
|   |   └──...
|   └──...
└──grasp_labels # object-level grasp label
|   ├──SCHOOL_BUS_labels.npz
|   └──...
└──scene_points # cluster scene points
|   ├──scene_0000.npy
|   └──...
└──scenes # scene-level grasp label
|   ├──scene_0000
|   |   ├──rgb
|   |   ├──depth
|   |   ├──label
|   |   ├──poses # object poses in camera frame
|   |   ├──meta.npz
|   |   ├──object_name_list.json
|   |   ├──T_cam_to_world.npy
|   |   └──...
|   └──...
└──collision_label
|   ├──scene_0000
|   |   ├──SCHOOL_BUS_collision.npy
|   |   └──...
|   └──...
└──graspness
|   ├──scene_0000
|   |   ├──0000.npy
|   |   ├──...
|   |   └──0255.npy
|   └──...
└──economic_grasp_label
|   ├──scene_0000_labels.npz
|   └──...
└──table # table usd
└──T_cam_to_table.npz # camera pose w.r.t. table
```

## Label Format
### grasp_labels
The object-level grasp labels of each mesh model are stored in `.npz` files.

```python
>>> import numpy as np
>>> grasp_labels = np.load('data/grasp_labels/SCHOOL_BUS_labels.npz')
>>> grasp_labels.files
['points', 'offsets', 'collision', 'scores']
```

`points`: shape (Nm, 3), where Nm is the number of sampled points of the mesh model.
`offsets`: shape (Nm, num_views, num_angles, num_depths, 3), records the in-plane rotation, depth and width of the grasp.
`collision`: shape (Nm, num_views, num_angles, num_depths), indicates whether the grasp is collision-free.
`scores`: shape (Nm, num_views, num_angles, num_depths), indicates the grasp quality score, the lower the better.


### instance-normalized graspness
The instance-normalized graspness scores of the cluster scenes for each camera frame are stored in `.npy` files.

```python
>>> import numpy as np
>>> graspness = np.load('data/graspness/scene_0000/0000.npy')
>>> graspness.shape
(Ns, 1)
```
where Ns is the number of sampled points in the scene.

### economic_grasp_label
The economic grasp labels of the cluster scenes are stored in `.npz` files, which share the same format as the one used in the [EconomicGrasp](https://github.com/iSEE-Laboratory/EconomicGrasp).
