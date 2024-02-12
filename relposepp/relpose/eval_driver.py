"""
A few ways to use this script:

- To generate a list of evaluation job commands to run:
    python relpose/eval_driver.py --list_jobs --output_text_path [path]

- To read all evaluation results and print a nicely formatted report:
    python relpose/eval_driver.py --list_jobs --output_text_path [path] --results_dir [directory]

- To run a single evaluation job for a model given the mode of evaluation, number of frames,
  and CO3D category and order index:
    python relpose/eval_driver.py --checkpoint_path weights/relposepp --mode pairwise \
        --num_frames 2 --category apple --sample_num 0

Note that to run the coordinate ascent rotation evaluation must be run before the translation evaluation modes.
"""

import argparse
import json
import os

import numpy as np

from dataset.co3d_v2 import TEST_CATEGORIES, TRAINING_CATEGORIES
from eval.eval_rotation import evaluate_category_rotation
from eval.eval_translation import evaluate_category_translation

DEFAULT_CHECKPOINT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights/normalize_camera_0723_1203_LR1e-05_N8_RandomNTrue_B128_AMP_TROURS_DDP"
)

# Change these arguments to suit your needs
OUTPUT_DIRECTORIES = [DEFAULT_CHECKPOINT_PATH]
MODES = ["pairwise", "coordinate_ascent", "cc", "t"]
NUM_FRAMES = [2, 3, 4, 5, 6, 7, 8]
CATEGORIES = TEST_CATEGORIES


def list_jobs(output_path):
    with open(output_path, "w") as f:
        for checkpoint in OUTPUT_DIRECTORIES:
            for mode in MODES:
                for num_frames in NUM_FRAMES:
                    for category in CATEGORIES:
                        command = f"python3 relpose/eval_driver.py --checkpoint_path {checkpoint} --mode {mode} --num_frames {num_frames} --category {category}\n"
                        f.write(command)


def checkpoint_report(output_path, results_dir, sample_num):
    with open(output_path, "w") as f:
        for mode in MODES:
            for num_frames in NUM_FRAMES:
                scores = []
                for category in CATEGORIES:
                    f = open(
                        f"{results_dir}/eval/{mode}-{num_frames:03d}-sample{sample_num}/{category}.json"
                    )
                    results = json.load(f)
                    angular_errors = []
                    for d in results.values():
                        angular_errors.extend(d["angular_errors"])
                    acc = np.mean(np.array(angular_errors) < 15)
                    scores.append(acc)

                avg = np.mean(np.array(scores))
                f.write(f"{mode} {num_frames}: {avg}")


def get_parser():
    parser = argparse.ArgumentParser()

    # Arguments for listing jobs and creating final summarizing report
    parser.add_argument("--output_text_path", type=str)
    parser.add_argument("--results_dir", type=str)
    parser.add_argument("--list_jobs", action="store_true")
    parser.add_argument("--report", action="store_true")

    # Arguments for evaluating on one category
    parser.add_argument("--checkpoint_path", type=str, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--num_frames", type=int)
    parser.add_argument("--mode", type=str)
    parser.add_argument("--category", type=str)
    parser.add_argument("--sample_num", type=int, default=0)
    return parser

def full_evaluation(checkpoint_path, num_frames):
    # 20*C(N)(N-1)
    angular_errors_pairwise = evaluate_category_rotation(
        checkpoint_path=checkpoint_path,
        dataset_type='finetune',
        category='gso_test',
        mode='pairwise',
        num_frames=num_frames,
        force=True,
        use_pbar=True,
    )

    angular_errors_coordinate_ascent = evaluate_category_rotation(
        checkpoint_path=checkpoint_path,
        dataset_type='finetune',
        category='gso_test',
        mode='coordinate_ascent',
        num_frames=num_frames,
        force=True,
        use_pbar=True,
    )

    trans_errors_cc = evaluate_category_translation(
        checkpoint_path=checkpoint_path,
        dataset_type='finetune',
        category='gso_test',
        mode='cc',
        num_frames=num_frames,
        force=True,
        use_pbar=True,
    )

    trans_errors_t = evaluate_category_translation(
        checkpoint_path=checkpoint_path,
        dataset_type='finetune',
        category='gso_test',
        mode='t',
        num_frames=num_frames,
        force=True,
        use_pbar=True,
    )

    return {
        "angular_errors_pairwise"         : (angular_errors_pairwise),
        "angular_errors_coordinate_ascent": (angular_errors_coordinate_ascent),
        "trans_errors_cc"                 : (trans_errors_cc),
        "trans_errors_t"                  : (trans_errors_t),
    }

if __name__ == "__main__":
    checkpoint_path = 'ckpts_refine_objaverse/ver_nonormalize/0812_0954_LR1e-05_N8_RandomNTrue_B64_AMP_TROURS_DDP'
    error_info = full_evaluation(checkpoint_path, num_frames=3)
    print(error_info)