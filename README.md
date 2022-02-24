# localization_end_to_end

**Local**ization **B**ot (localbot) is a conceptual robot designed to test end-to-end localization algorithms. The system contains the following sensor:

- **hand_camera** - A RGB-D camera


### Add to .bashrc:

```
export GAZEBO_MODEL_PATH="`rospack find localbot_gazebo`/models:${GAZEBO_MODEL_PATH}"
```

# Running the simulation

To open the gazebo with the room_024.world run:

    roslaunch localbot_gazebo localbot.launch

To spawn the robotic system into the gazebo run:

    roslaunch localbot_bringup bringup.launch