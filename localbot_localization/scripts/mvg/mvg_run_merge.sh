rosrun localbot_localization mvg_convert_poses -f 'seq20_b8500-9000' -o 'seq20_tmp' 

rosrun localbot_localization mvg_synthesize_poses -d 'seq20' -f 'seq20_tmp' -s 8500

rosrun localbot_localization mvg_convert_to_initial_frame -f 'seq20_tmp' -s 8500

rosrun localbot_localization mvg_scale_transformations -d 'seq20' -f 'seq20_tmp' -s 8500

rosrun localbot_localization mvg_merge_poses -c 'seq20_c' -t 'seq20_tmp'

rosrun localbot_localization mvg_produce_comparison -d 'seq20' -f 'seq20_c_new' -rf 'seq20_c_new'

