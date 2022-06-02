rosrun localbot_localization rtab_convert_poses -i 'seq20_1993_4000.txt' -o 'seq20_rtab0_tmp' -s 1993

rosrun localbot_localization rtab_synthesize_poses -d 'seq20' -f 'seq20_rtab0_tmp' -s 1993

rosrun localbot_localization rtab_convert_to_initial_frame -f 'seq20_rtab0_tmp' -s 1993

rosrun localbot_localization rtab_merge_poses -c 'seq20_rtab0_c' -t 'seq20_rtab0_tmp'

rosrun localbot_localization rtab_produce_comparison -d 'seq20' -f 'seq20_rtab0_c_new' -rf 'seq20_rtab0_c_new'

# see where the error starts to get too big, and run rtabmap there.
# rename seq20_rtab0_c_new to seq20_rtab0_c