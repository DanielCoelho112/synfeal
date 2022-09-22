rosrun localbot_localization rtab_convert_poses -i 'seq20_2064_0.txt' -o 'seq20_rtab0_t'

rosrun localbot_localization rtab_synthesize_poses -d 'seq20' -f 'seq20_rtab0_t'

rosrun localbot_localization rtab_convert_to_initial_frame -f 'seq20_rtab0_t'


rosrun localbot_localization rtab_convert_to_world_frame -f 'seq20_rtab0_t' -d 'seq20'

#rosrun localbot_localization rtab_move_poses -f 'seq19_rtab2' -d 'seq19_rtab'

rosrun localbot_localization rtab_produce_comparison -d 'seq20' -f 'seq20_rtab0_t' -rf 'seq20_rtab0_t'