rosrun localbot_localization rtab_convert_poses -i 'poses_seq19_4.txt' -o 'seq19_rtab4'

rosrun localbot_localization rtab_synthesize_poses -d 'seq19' -f 'seq19_rtab4'

rosrun localbot_localization rtab_convert_to_initial_frame -f 'seq19_rtab4'

rosrun localbot_localization rtab_scale_transformations -f 'seq19_rtab4' -d 'seq19'

rosrun localbot_localization rtab_convert_to_world_frame -f 'seq19_rtab4' -d 'seq19'

rosrun localbot_localization rtab_move_poses -f 'seq19_rtab4' -d 'seq19_rtab'

rosrun localbot_localization rtab_produce_comparison -d 'seq19' -f 'seq19_rtab4' -rf 'seq19_rtab4'