## Build the image


    docker build . -t danielc11/synfeal:latest

    docker push danielc11/synfeal:latest

    docker pull danielc11/synfeal:latest

## Spawn the container - interactive

    docker run -it --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --name pytorch_t10 --network=host -v /home/DanielCoelho11/models/localbot:/root/models/localbot -v /home/DanielCoelho11/datasets/localbot:/root/datasets/localbot danielc11/synfeal bash

## Spawn the container - dettached

    docker run -d --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --name pytorch_t10 --network=host -v /home/DanielCoelho11/models/localbot:/root/models/localbot -v /home/DanielCoelho11/datasets/localbot:/root/datasets/localbot danielc11/synfeal bash -c "cd synfeal && git pull"

## Attach to a running container

    docker attach <container_id>

## Dettach from running container

    press CRTL-p & CRTL-q