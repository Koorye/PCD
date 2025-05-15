export XLA_PYTHON_CLIENT_PREALLOCATE=false

num_gpus=7
n_trajs=100
result_root="./results/default/contrast"

# search_opts="by gt alpha 0.2 num_repeats 32"
# search_opts="by point_tracking alpha 0.4 num_repeats 32"
# search_opts="by point_tracking,box_tracking,sed_tracking alpha 0.2 num_repeats 24"
search_opts="by sed_tracking alpha 1.0 num_repeats 24"

policies=("octo")
checkpoints=("pretrained/octo-base")

tasks=(
    "google_robot_pick_coke_can"
    "google_robot_move_near"
    "google_robot_close_drawer"
    "google_robot_open_drawer"
    "widowx_carrot_on_plate"
    "widowx_spoon_on_towel"
    "widowx_put_eggplant_in_basket"
    "widowx_stack_cube"
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
