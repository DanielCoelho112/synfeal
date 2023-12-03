## Build the image


    docker build . -t danielc11/synfeal:latest

    docker push danielc11/synfeal:latest

    docker pull danielc11/synfeal:latest

## Spawn the container - interactive

    docker run -it --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --network=host -v /home/danc/synfeal:/root/synfeal   -v /results/synfeal:/root/models/synfeal -v /datasets/synfeal:/root/datasets/synfeal danielc11/synfeal bash

## Spawn the container - dettached

    docker run -d -it --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --network=host -v /home/danc/synfeal:/root/synfeal   -v /results/synfeal:/root/models/synfeal -v /datasets/synfeal:/root/datasets/synfeal danielc11/synfeal bash


## Attach to a running container

    docker attach <container_id>

## Dettach from running container

    press CRTL-p & CRTL-q