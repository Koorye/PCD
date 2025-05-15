num_gpus=6
result_root="./results/default/baseline"

policies=("octo")
checkpoints=("pretrained/octo-base")
tasks=(
    "google_robot_pick_coke_can"
    "google_robot_move_near"
    "google_robot_close_drawer"
    "google_robot_open_drawer"
    "widowx_put_eggplant_in_basket"
    "widowx_spoon_on_towel"
    "widowx_carrot_on_plate"
    "widowx_stack_cube"
)

for i in "${!policies[@]}"; do
    for task in "${tasks[@]}"; do
        echo "Running inference for ${policies[$i]} on $task"

        python parallel_inference.py \
            --num-gpus $num_gpus \
            --result-root $result_root \
            --policy ${policies[$i]} \
            --checkpoint ${checkpoints[$i]} \
            --task $task
    done
done
