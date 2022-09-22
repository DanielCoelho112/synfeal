# Synfeal: A Data-Driven Simulator for Camera Localization

<p align="center">
<img src="docs/img/logo.png" data-canonical-src="docs/img/logo.png" width="507" height="209" />
</p>


<p align="center">
<img src="docs/img/synfeal.png" data-canonical-src="docs/img/synfeal.png" width="422" height="1100" />
</p>



**Synfeal** (*Synthetic from Real*) synthesizes large localization datasets free of inconsistencies based on realistic 3D reconstructions of the environment.
For instructions on how to install and use, **check the documentation** (work in progress).

For a demonstration of the data collection procedure, see the following video:

<p align="center">
<a href="https://www.youtube.com/watch?v=sRxalb6BoFs&ab/">https://www.youtube.com/watch?v=sRxalb6BoFs&ab/</a>
</p> 


# Documentation

## Requirements

## How to collect data?

First launch the texture mesh:

    roslaunch synfeal_bringup bringup_mesh.launch world:=santuario.world

Then launch the virtual camera:

    roslaunch synfeal_bringup bringup_camera.launch


## How to train the algorithms?

## How to visualize the results?