num_gpus=7
n_trajs=100
result_root="./results/motivation/contrast"

policies=("openvla")
checkpoints=("pretrained/openvla-7b")
search_opts="by point_tracking alpha 0.8"

tasks=(
    # "google_robot_pick_coke_can_bright"
    # "google_robot_pick_coke_can_dark"
    # "google_robot_pick_coke_can_drawer_variant"
    "google_robot_pick_coke_can_light_variant"
    # "google_robot_pick_coke_can_table_red_variant"
    # "google_robot_pick_coke_can_table_blue_variant"
    # "google_robot_pick_coke_can_table_gray_variant"
    # "google_robot_pick_coke_can_table_paper_variant"
    # "google_robot_pick_coke_can_table_stone_variant"
    # "google_robot_pick_coke_can_table_stone2_variant"
    # "google_robot_pick_coke_can_table_stone3_variant"
)

for i in "${!policies[@]}"; do
    for task in "${tasks[@]}"; do
        echo "Running inference for ${policies[$i]} on $task"

        python parallel_inference.py \
            --contrast \
            --n-trajs $n_trajs \
            --num-gpus $num_gpus \
            --result-root $result_root \
            --policy ${policies[$i]} \
            --checkpoint ${checkpoints[$i]} \
            --task $task \
            --search-opts $search_opts
    done
done
