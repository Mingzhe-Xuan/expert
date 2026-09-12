#!/usr/bin/env bash

select_backbone_venv() {
  if [[ $# -ne 1 || ! $1 =~ ^[0-9]+$ ]]; then
    echo "select_backbone_venv requires one non-negative array index" >&2
    return 2
  fi
  local array_index="$1"
  case $((array_index % 4)) in
    0)
      : "${EXPERT_MACE_VENV:?Set EXPERT_MACE_VENV to the recorded MACE/core environment}"
      EXPERT_SELECTED_VENV="${EXPERT_MACE_VENV}"
      ;;
    1)
      : "${EXPERT_GRACE_VENV:?Set EXPERT_GRACE_VENV to the recorded GRACE environment}"
      EXPERT_SELECTED_VENV="${EXPERT_GRACE_VENV}"
      export EXPERT_GRACE_TF_DEVICE="${EXPERT_GRACE_TF_DEVICE:-cpu}"
      ;;
    2)
      : "${EXPERT_DPA4_VENV:?Set EXPERT_DPA4_VENV to the recorded DPA4 environment}"
      EXPERT_SELECTED_VENV="${EXPERT_DPA4_VENV}"
      ;;
    3)
      : "${EXPERT_EQUIFORMERV2_VENV:?Set EXPERT_EQUIFORMERV2_VENV to the recorded EquiformerV2 environment}"
      EXPERT_SELECTED_VENV="${EXPERT_EQUIFORMERV2_VENV}"
      ;;
  esac
  export EXPERT_SELECTED_VENV
}
