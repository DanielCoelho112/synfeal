# Synfeal: A Data-Driven Simulator for Camera Localization

<p align="center">
<img src="docs/img/logo.png" data-canonical-src="docs/img/logo.png" width="507" height="209" />
</p>


<p align="center">
<img src="docs/img/synfeal.png" data-canonical-src="docs/img/synfeal.png" width="422" height="1100" />
</p>



**Synfeal** (*Synthetic from Real*) synthesizes large localization datasets free of inconsistencies based on realistic 3D reconstructions of the environment.
For instructions on how to install and use, **check the Documentation**.

For a demonstration of the data collection procedure, see the following video:

<p align="center">
<a href="https://www.youtube.com/watch?v=sRxalb6BoFs&ab/">https://www.youtube.com/watch?v=sRxalb6BoFs&ab/</a>
</p> 


# Documentation

## Requirements


    pip3 install -r /synfeal/requirements.txt

    mkdir -p $HOME/datasets/localbot
    mkdir -p $HOME/models/localbot
    mkdir -p $HOME/results/localbot
    mkdir -p $HOME/models_3d/localbot

Add to zsh or bash file:

    export GAZEBO_MODEL_PATH="`rospack find synfeal_bringup`/models:${GAZEBO_MODEL_PATH}":${HOME}/models_3d/localbot
    export SYNFEAL_DATASET=${HOME}
    export PYTHONPATH="$PYTHONPATH:/home/user/catkin_ws/src/synfeal"
Where /home/user/catkin_ws/src/synfeal is the location of synfeal. 

## How to launch the system?

First launch the texture mesh:

    roslaunch synfeal_bringup bringup_mesh.launch world:=santuario.world

Then launch the virtual camera:

    roslaunch synfeal_bringup bringup_camera.launch

## How to collect the data?

Run:

    rosrun synfeal_collection data_collector -nf 100000 -m <mode> -mc 'santuario.yaml' -s name_of_the_dataset

where mode can be one option of the following: **interactive**, **random**, **path**.

Adapt the yaml file according to your needs.

Activate the flag **-f** just to collect the RGB image and the corresponding pose.

In case you want to add objects moving through the scene add the flag **-obj**. If no object database is available or if you want to place the objects randomly add the flag **-ro**.

If you want to add variations in the light conditions add the flag -uvl and in case the variations in the light's conditions use the flag **-rl**.

In case you want to replicate the camera movement of a previous dataset, run:

```
 rosrun synfeal_collection data_collector -nf 100000 -m repeat -mc 'santuario.yaml' -s name_of_the_dataset -s_prev name_of_the_previous_dataset
```

## How to process the dataset?

Run:

    cd synfeal/process_dataset/scripts && ./process_dataset -d name_of_the_dataset -s '_processed' -fi 0.5 -pts 1000

Or, in the case of when **-f** is activated, run:

    cd synfeal/process_dataset/scripts && ./create_statistics -d name_of_the_dataset

Then validate the dataset with:

    ./validate_dataset -d name_of_the_dataset


## How to train the algorithms?

If you want to use the PoseNet with Beta Loss run:

    ./rgb_training -fn posenet_beta_lab024_50k -mn posenetgooglenet -train_set seq41 -test_set seq42 -n_epochs 300 -batch_size 45  -loss 'BetaLoss(100)' -c -im 'PoseNetGoogleNet(True,0.8)' -lr_step_size 60 -lr 1e-4 -lr_gamma 0.5 -wd 1e-2 -gpu 2

For informations on how to use other models, see https://github.com/DanielCoelho112/synfeal/blob/main/models/readme.md#L5

If you want to add data augmentation containing Random Erasing and Color Jitter add the flag **-augm**.


## How to produce the results?

Run:

    ./produce_results/scripts/rgb_produce_results -test_set 'seq22_p0r20' -rf <results_folder> -mp <model_name>

## How to visualize the results?

Run:

    roslaunch synfeal_visualization visualize_results.launch results_folder:=synfeal0 idx_max:=100 mesh_name:=room_024
