# Synfeal: Data-Driven Simulator for Camera Localization


![localbot_gazebo](docs/img/synfeal.png)




### Add to .bashrc:

```
export GAZEBO_MODEL_PATH="`rospack find localbot_gazebo`/models:${GAZEBO_MODEL_PATH}"
```

# Requirements

Due to the docker image used, you must use torchvision==0.7.0:

    pip install torchvision==0.7.0



# Running the simulation

To launch the gazebo with the room_024.world run:

    roslaunch localbot_gazebo localbot.launch

To spawn the robotic system into the gazebo run:

    roslaunch localbot_bringup bringup.launch

# Collecting the data

In each frame we store:

- **frame-X.pcd** - The point cloud w.r.t. the rgb_frame
- **frame-X.pose.txt** - The 4x4 transformation matrix from world to rgb_frame
- **frame-X.rgb.png** - The rgb image

For collecting each frame, we have four options:

1.  **interactive**

        rosrun localbot_core data_collector --mode interactive --seq seq110

Here we have to manually drag the interactive marker and click on another marker to save each frame of the dataset.

2.  **automatic_random**

        rosrun localbot_core data_collector --mode automatic_random --seq seq110 -nf 1000 -uvl 

In this case, in each frame, it is generated a random pose. Here we do not have any movement between poses.

video:

https://uapt33090-my.sharepoint.com/:v:/g/personal/danielsilveiracoelho_ua_pt/ERR9giFm8ZREiKSNzqZQNaABZ38q4M1HU_Zte4d_4e1Zxg?e=oGkjRv

3.  **automatic_path** with random poses 

        rosrun localbot_core data_collector --mode automatic_path --seq seq110 -ns 20  -nf 1000

In this case, it is generated a random pose, and then the camera moves towards that pose in a continuous movement. As soon as the pose is reached, another pose is generated, and so on... One hyperparameter that must be defined is the number of steps (n_steps). Basically, it is the number of iterations between the initial pose and the final pose. 

video: 

https://uapt33090-my.sharepoint.com/:v:/g/personal/danielsilveiracoelho_ua_pt/Ef34nF86NIVEsGxx6OH2jE8BbOZAVPP9-_WT9ltrG2xr0g?e=xPEihX


4.  **automatic_path** with the desired pose 

        rosrun localbot_core data_collector --mode automatic_path --seq seq110 --ns 20 --destination_pose '1,4,0,0.5,0.2,1.5'

In this case, we have to define where we want the model to go, and then a path is generated towards that pose. 



# Processing the dataset

Run:

    rosrun localbot_localization process_dataset -d 'seq44' -s '_processed' -fi 0.5 -pts 1000 -igp


# Validating the datasets

Every collected dataset MUST be validated.

Run:

    rosrun localbot_localization validate_dataset -d 'seq_depth_test_depth_stat_local_norm'

# Merging different datasets

It is possible to merge different datasets. All datasets should be considered valid before merging...

Run:

    rosrun localbot_localization merge_dataset -dataset1 'seq1_v' -dataset2 'seq1_v' -merged_dataset 'merged_dataset'


# Visualizing the training set

Run:

    roslaunch localbot_localization visualize_train_set.launch train_set:=seq110


# Training the models

**PointNet**

    rosrun localbot_localization point_net_training -fn test1 -mn pointnetv0 -train_set seq110 -test_set seq111 -n_epochs 5 -batch_size 2 -loss 'DynamicLoss()'

**DepthNet**

    rosrun localbot_localization depth_training -c -fn model_depth -mn depthv0 -train_set 'seq_test0_d_v' -test_set 'seq_test0_d_v'

**PoseNet**

    ./localbot_localization/scripts/training/posenet_training -fn posenet_googlenet_no_trained_ADAM_wd5_lr4_randomcrop_new_images -mn posenetgooglenet -train_set seq3dpii_v2 -test_set seq4dpii_v2 -n_epochs 150 -batch_size 32  -loss 'DynamicLoss()' -c -im 'PoseNetGoogleNet(True,0.5)' -lr_step_size 60 -lr 1e-4 -lr_gamma 0.5 -wd 1e-5 -gpu 0


# Producing the results

**PointNet**

    rosrun localbot_localization point_net_produce_results -test_set seq111 -mp test1 -rf results_folder

**DepthNet**

    rosrun localbot_localization depth_net_produce_results -test_set 'seq1d_p_global_norm' -mp 'depth_low_global_norm' -rf results15

**PoseNet**

    rosrun localbot_localization posenet_produce_results -test_set 'seq22_rtab0' -mp 'posenet_googlenet_rtab0_lr4_wd2_dropout75' -rf results52

# Visualizing the results

Run:

    roslaunch localbot_localization visualize_results.launch results_folder:=results19 frame_selection_function:="'pos[idx]>1.5 and rot[idx]>0.2'"
    roslaunch localbot_localization visualize_results.launch results_folder:=results19 idx_max:=10


# Extras:
## Using SfM - MVG to collect data

divide the dataset in batches

    rosrun localbot_localization batch_images -f seq20_images -b 500

    cd ~/PhD/localization/openMVG_Build/software/SfM

change the file sfm_batches.sh and then

    ./sfm_batches.sh

    
    roscd localbot_localization/scripts/mvg

change the file mvg_run_merge.sh and then

    ./mvg_run_merge.sh


Delete _tmp, and change _c_new to _c.

Then look into the results, and when the results are bad (really bad), start rtab from there.
Do it all over again.


## Using RTAB and TORO

    rtabmap  (choose right database and choose dataset)  (0 for rtab default and 4 for toro)

    roscd localbot_localization/scripts/rtab

change file rtab_run_merge and then

    ./rtab_run_merge.sh

Delete _tmp, and change _c_new to _c.





