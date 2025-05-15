import gymnasium as gym
import mani_skill2_real2sim.envs

ENVIRONMENTS = [
    "google_robot_pick_coke_can",
    "google_robot_pick_pepsi_can",
    "google_robot_pick_red_bull_can",
    "google_robot_pick_blue_plastic_bottle",
    "google_robot_pick_apple",
    "google_robot_pick_orange",
    "google_robot_pick_sponge",
    "google_robot_pick_horizontal_coke_can",
    "google_robot_pick_vertical_coke_can",
    "google_robot_pick_standing_coke_can",
    "google_robot_pick_object",
    "google_robot_move_near_v0",
    "google_robot_move_near_v1",
    "google_robot_move_near",
    "google_robot_open_drawer",
    "google_robot_open_top_drawer",
    "google_robot_open_middle_drawer",
    "google_robot_open_bottom_drawer",
    "google_robot_close_drawer",
    "google_robot_close_top_drawer",
    "google_robot_close_middle_drawer",
    "google_robot_close_bottom_drawer",
    "google_robot_place_in_closed_drawer",
    "google_robot_place_in_closed_top_drawer",
    "google_robot_place_in_closed_middle_drawer",
    "google_robot_place_in_closed_bottom_drawer",
    "google_robot_place_apple_in_closed_top_drawer",
    "widowx_spoon_on_towel",
    "widowx_carrot_on_plate",
    "widowx_stack_cube",
    "widowx_put_eggplant_in_basket",

    # light
    "google_robot_pick_coke_can_dark",
    "google_robot_move_near_dark",
    "google_robot_open_drawer_dark",
    "google_robot_close_drawer_dark",
    "google_robot_place_apple_in_closed_top_drawer_dark",
    
    "widowx_carrot_on_plate_dark",
    "widowx_put_eggplant_in_basket_dark",
    "widowx_spoon_on_towel_dark",
    "widowx_stack_cube_dark",

    "google_robot_pick_coke_can_bright",
    "google_robot_move_near_bright",
    "widowx_carrot_on_plate_bright",
    "widowx_put_eggplant_in_basket_bright",
    "widowx_spoon_on_towel_bright",
    "widowx_stack_cube_bright",

    # variant aggregation
    "google_robot_pick_coke_can_variant",
    "google_robot_pick_object_variant",
    "google_robot_move_near_variant",
    "google_robot_open_drawer_variant",
    "google_robot_close_drawer_variant",
    "widowx_spoon_on_towel_variant",
    "widowx_carrot_on_plate_variant",
    "widowx_stack_cube_variant",
    "widowx_put_eggplant_in_basket_variant",

    "google_robot_pick_coke_can_drawer_variant",
    "google_robot_pick_coke_can_light_variant",
    "google_robot_pick_coke_can_table_red_variant",
    "google_robot_pick_coke_can_table_blue_variant",
    "google_robot_pick_coke_can_table_gray_variant",
    "google_robot_pick_coke_can_table_paper_variant",
    "google_robot_pick_coke_can_table_stone_variant",
    "google_robot_pick_coke_can_table_stone2_variant",
    "google_robot_pick_coke_can_table_stone3_variant",
]

ENVIRONMENT_MAP = {
    "google_robot_pick_coke_can": ("GraspSingleOpenedCokeCanInScene-v0", {}),
    "google_robot_pick_pepsi_can": ("GraspSingleOpenedPepsiCanInScene-v0", {}),
    "google_robot_pick_red_bull_can": ("GraspSingleOpenedRedBullCanInScene-v0", {}),
    "google_robot_pick_blue_plastic_bottle": ("GraspSingleBluePlasticBottleInScene-v0", {}),
    "google_robot_pick_apple": ("GraspSingleAppleInScene-v0", {}),
    "google_robot_pick_orange": ("GraspSingleOrangeInScene-v0", {}),
    "google_robot_pick_sponge": ("GraspSingleSpongeInScene-v0", {}),
    "google_robot_pick_horizontal_coke_can": (
        "GraspSingleOpenedCokeCanInScene-v0",
        {"lr_switch": True},
    ),
    "google_robot_pick_vertical_coke_can": (
        "GraspSingleOpenedCokeCanInScene-v0",
        {"laid_vertically": True},
    ),
    "google_robot_pick_standing_coke_can": (
        "GraspSingleOpenedCokeCanInScene-v0",
        {"upright": True},
    ),
    "google_robot_pick_object": ("GraspSingleRandomObjectInScene-v0", {}),
    "google_robot_move_near": ("MoveNearGoogleBakedTexInScene-v1", {}),
    "google_robot_move_near_v0": ("MoveNearGoogleBakedTexInScene-v0", {}),
    "google_robot_move_near_v1": ("MoveNearGoogleBakedTexInScene-v1", {}),
    "google_robot_open_drawer": ("OpenDrawerCustomInScene-v0", {}),
    "google_robot_open_top_drawer": ("OpenTopDrawerCustomInScene-v0", {}),
    "google_robot_open_middle_drawer": ("OpenMiddleDrawerCustomInScene-v0", {}),
    "google_robot_open_bottom_drawer": ("OpenBottomDrawerCustomInScene-v0", {}),
    "google_robot_close_drawer": ("CloseDrawerCustomInScene-v0", {}),
    "google_robot_close_top_drawer": ("CloseTopDrawerCustomInScene-v0", {}),
    "google_robot_close_middle_drawer": ("CloseMiddleDrawerCustomInScene-v0", {}),
    "google_robot_close_bottom_drawer": ("CloseBottomDrawerCustomInScene-v0", {}),
    "google_robot_place_in_closed_drawer": ("PlaceIntoClosedDrawerCustomInScene-v0", {}),
    "google_robot_place_in_closed_top_drawer": ("PlaceIntoClosedTopDrawerCustomInScene-v0", {}),
    "google_robot_place_in_closed_middle_drawer": ("PlaceIntoClosedMiddleDrawerCustomInScene-v0", {}),
    "google_robot_place_in_closed_bottom_drawer": ("PlaceIntoClosedBottomDrawerCustomInScene-v0", {}),
    "google_robot_place_apple_in_closed_top_drawer": (
        "PlaceIntoClosedTopDrawerCustomInScene-v0", 
        {"model_ids": "baked_apple_v2"}
    ),
    "widowx_spoon_on_towel": ("PutSpoonOnTableClothInScene-v0", {}),
    "widowx_carrot_on_plate": ("PutCarrotOnPlateInScene-v0", {}),
    "widowx_stack_cube": ("StackGreenCubeOnYellowCubeBakedTexInScene-v0", {}),
    "widowx_put_eggplant_in_basket": ("PutEggplantInBasketScene-v0", {}),
    
    # light
    "google_robot_pick_coke_can_dark": ("GraspSingleCokeCanDarkerInScene-v0", {}),
    "google_robot_move_near_dark": ("MoveNearGoogleBakedTexDarkerInScene-v1", {}),
    "google_robot_open_drawer_dark": ("OpenDrawerCustomDarkerInScene-v0", {}),
    "google_robot_close_drawer_dark": ("CloseDrawerCustomDarkerInScene-v0", {}),
    "google_robot_place_apple_in_closed_top_drawer_dark": (
        "PlaceIntoClosedTopDrawerCustomDarkerInScene-v0",
        {"model_ids": "apple"}
    ),
    "widowx_carrot_on_plate_dark": ("PutCarrotOnPlateDarkerInScene-v0", {}),
    "widowx_put_eggplant_in_basket_dark": ("PutEggplantInBasketDarkerScene-v0", {}),
    "widowx_spoon_on_towel_dark": ("PutSpoonOnTableClothDarkerInScene-v0", {}),
    "widowx_stack_cube_dark": ("StackGreenCubeOnYellowCubeDarkerInScene-v0", {}),

    "google_robot_pick_coke_can_bright": ("GraspSingleCokeCanBrighterInScene-v0", {}),
    "google_robot_move_near_bright": ("MoveNearGoogleBakedTexBrighterInScene-v1", {}),
    "widowx_carrot_on_plate_bright": ("PutCarrotOnPlateBrighterInScene-v0", {}),
    "widowx_put_eggplant_in_basket_bright": ("PutEggplantInBasketBrighterScene-v0", {}),
    "widowx_spoon_on_towel_bright": ("PutSpoonOnTableClothBrighterInScene-v0", {}),
    "widowx_stack_cube_bright": ("StackGreenCubeOnYellowCubeBrighterInScene-v0", {}),
    
    # variant aggregation
    "google_robot_pick_coke_can_variant": ("GraspSingleOpenedCokeCanInSceneVariant-v0", {}),
    "google_robot_pick_object_variant": ("GraspSingleRandomObjectInSceneVariant-v0", {}),
    "google_robot_move_near_variant": ("MoveNearGoogleBakedTexInSceneVariant-v1", {}),
    "google_robot_open_drawer_variant": ("OpenDrawerCustomInSceneVariant-v0", {}),
    "google_robot_close_drawer_variant": ("CloseDrawerCustomInSceneVariant-v0", {}),
    "widowx_spoon_on_towel_variant": ("PutSpoonOnTableClothInSceneVariant-v0", {}),
    "widowx_carrot_on_plate_variant": ("PutCarrotOnPlateInSceneVariant-v0", {}),
    "widowx_stack_cube_variant": ("StackGreenCubeOnYellowCubeBakedTexInSceneVariant-v0", {}),
    "widowx_put_eggplant_in_basket_variant": ("PutEggplantInBasketSceneVariant-v0", {}),
    
    "google_robot_pick_coke_can_drawer_variant": ("GraspSingleOpenedCokeCanDrawerVariantInScene-v0", {}),
    "google_robot_pick_coke_can_light_variant": ("GraspSingleOpenedCokeCanLightVariantInScene-v0", {}),
    "google_robot_pick_coke_can_table_red_variant": ("GraspSingleOpenedCokeCanTableRedVariantInScene-v0", {}),
    "google_robot_pick_coke_can_table_blue_variant": ("GraspSingleOpenedCokeCanTableBlueVariantInScene-v0", {}),
    "google_robot_pick_coke_can_table_gray_variant": ("GraspSingleOpenedCokeCanTableGrayVariantInScene-v0", {}),
    "google_robot_pick_coke_can_table_paper_variant": ("GraspSingleOpenedCokeCanTablePaperVariantInScene-v0", {}),
    "google_robot_pick_coke_can_table_stone_variant": ("GraspSingleOpenedCokeCanTableStoneVariantInScene-v0", {}),
    "google_robot_pick_coke_can_table_stone2_variant": ("GraspSingleOpenedCokeCanTableStone2VariantInScene-v0", {}),
    "google_robot_pick_coke_can_table_stone3_variant": ("GraspSingleOpenedCokeCanTableStone3VariantInScene-v0", {}),
}


def make(task_name):
    """Creates simulated eval environment from task name."""
    assert task_name in ENVIRONMENTS, f"Task {task_name} is not supported. Environments: \n {ENVIRONMENTS}"
    env_name, kwargs = ENVIRONMENT_MAP[task_name]
    kwargs["prepackaged_config"] = True
    env = gym.make(env_name, obs_mode="rgbd", **kwargs)
    return env
