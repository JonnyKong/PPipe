#!/bin/bash
trap 'echo "Error on line $LINENO: $BASH_COMMAND"' ERR
set -e

DATA_DIR=./data
OUTPUT_DIR=./outputs
DURATION=30000000

function check_and_confirm_overwrite() {
    local parent_dir="$1"

    # Check if the parent directory exists
    if [[ ! -d "$parent_dir" ]]; then
        return 0
    fi

    # Check if the parent directory is non-empty
    if [[ -n "$(ls -A "$parent_dir" 2>/dev/null)" ]]; then
        while true; do
            read -p "The directory '$parent_dir' is not empty. Overwrite? (yes/no): " response
            case "$response" in
            [Yy][Ee][Ss] | [Yy])
                rm -rf "$parent_dir" && mkdir -p "$parent_dir"
                return 0
                ;;
            [Nn][Oo] | [Nn])
                exit 1
                ;;
            *)
                echo "Invalid input. Please type 'yes' or 'no'."
                ;;
            esac
        done
    fi

    return 0
}

function main_results_maf19() {
    PLANS_DIR=${OUTPUT_DIR}/plans/maf19
    LOGS_DIR=outputs/cluster-logs/maf19
    check_and_confirm_overwrite ${LOGS_DIR}

    python scripts/run_sim_in_batch.py multi_dnn_maf19 \
        ${DATA_DIR}/models/block-timing-tf32 \
        ${OUTPUT_DIR}/prepartition_mappings \
        ${PLANS_DIR} ${LOGS_DIR}
    python scripts/parse_cluster_sim.py multi_dnn_maf19 \
        --plans-dir ${PLANS_DIR} --logs-dir ${LOGS_DIR} --duration ${DURATION}
}

function main_results_maf21() {
    PLANS_DIR=${OUTPUT_DIR}/plans/maf21
    LOGS_DIR=outputs/cluster-logs/maf21
    check_and_confirm_overwrite ${LOGS_DIR}

    python scripts/run_sim_in_batch.py multi_dnn_maf21 \
        ${DATA_DIR}/models/block-timing-tf32 \
        ${OUTPUT_DIR}/prepartition_mappings \
        ${PLANS_DIR} ${LOGS_DIR}
    python scripts/parse_cluster_sim.py multi_dnn_maf21 \
        --plans-dir ${PLANS_DIR} --logs-dir ${LOGS_DIR} --duration ${DURATION}
}

function ablation_results_maf19() {
    PLANS_DIR=${OUTPUT_DIR}/plans/ablation
    LOGS_DIR=outputs/cluster-logs/ablation_maf19
    python scripts/run_sim_in_batch.py ablation_maf19 \
        ${DATA_DIR}/models/block-timing-tf32 \
        ${OUTPUT_DIR}/prepartition_mappings \
        ${PLANS_DIR} ${LOGS_DIR}
    python scripts/parse_cluster_sim.py ablation_maf19 \
        --plans-dir ${PLANS_DIR} --logs-dir ${LOGS_DIR} --duration ${DURATION}
}

"$@"
