#!/usr/bin/env bash

# Shared configuration for the first HIL v2 dataset round.
DATASET_REPO_R1="AlvinAi/insert_carrot_into_the_hole_hil_v2_r1"
DATASET_ROOT_R1="/scratch/u6pw/ql337.u6pw/data/lerobot/AlvinAi/insert_carrot_into_the_hole_hil_v2_r1"

VALUE_TAG_R1="pistar06_v2_r1_n50_r30"
VALUE_OUTPUT_R1="/scratch/u6pw/ql337.u6pw/projects/Evo-RL/outputs/value_train/pistar06_insert_carrot_v2_r1"
VALUE_INFER_OUTPUT_R1="/scratch/u6pw/ql337.u6pw/projects/Evo-RL/outputs/value_infer/pistar06_insert_carrot_v2_r1_n50_r30"
VALUE_CHECKPOINT_R1="$VALUE_OUTPUT_R1/continued_from_000500_6581542"

ACP_VALUE_FIELD_R1="complementary_info.value_${VALUE_TAG_R1}"
ACP_ADVANTAGE_FIELD_R1="complementary_info.advantage_${VALUE_TAG_R1}"
ACP_INDICATOR_FIELD_R1="complementary_info.acp_indicator_${VALUE_TAG_R1}"

# The dataset keeps its native camera names; Pi0.5 expects these model names.
PI05_RENAME_MAP_R1='{"observation.images.front":"observation.images.base_0_rgb","observation.images.side":"observation.images.left_wrist_0_rgb"}'
