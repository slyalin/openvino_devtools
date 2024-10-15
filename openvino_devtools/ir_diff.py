#!/usr/bin/env python3

"""
This script is designed to compare XML files in two directories and report differences in <layer> tags.
It scans the directories, collects operation counts from the XML files, and prints any differences
in the operations count between the reference and target files. The script also includes an option
to filter out files containing 'tokenizer' in their names.

./ir_diff.py [--filter-tokenizer | --no-filter-tokenizer] <reference_directory> <target_directory>
"""
import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)


def get_irs(directory: Path) -> Iterator[Path]:
    """Get all XML files in a directory recursively"""
    if directory.is_dir():
        return directory.rglob('*.xml')
    else:
        raise NotADirectoryError(f"{directory} is not a directory")


def collect_ops(path: Path) -> Dict[str, int]:
    """Collect operation counts from an XML file"""
    ops = defaultdict(int)
    try:
        with path.open("r") as file:
            for line in file.readlines():
                if '<layer ' in line:
                    if 'type="' not in line:
                        raise ValueError(f'Unexpected line {line}')
                    op = line.split('type="')[1].split('"')[0]
                    if not op:
                        raise ValueError(f'Unexpected line {line}')
                    ops[op] += 1
    except FileNotFoundError:
        logger.error(f"File not found: {path}")
    except ValueError as ve:
        logger.error(f"Value error in file {path}: {ve}")
    except Exception as e:
        logger.error(f"Unexpected error processing file {path}: {e}")
    return ops


def print_diff(reference_path: Path, target_path: Path) -> None:
    """Print the differences between two sets of operations in .xml file and sizes of .bin files"""
    # .xml file difference
    reference, target = collect_ops(reference_path), collect_ops(target_path)

    keys = set(reference.keys()).union(target.keys())
    no_diff = True
    for key in sorted(keys):
        if reference[key] != target[key]:
            if no_diff:
                logger.info("Op difference detected")
                logger.info(f"{'op_name':25} {'ref':^7} {'target':^7} {'r-t':^7}")
                logger.info(f"{'-' * 49}")
            logger.info(f"{key:25} {reference[key]:7} {target[key]:7} {reference[key] - target[key]:7}")
            no_diff = False

    if no_diff:
        logger.info("No op difference")
    else:
        reference_count = 0
        for op, count in reference.items():
            reference_count += count
        target_count = 0
        for op, count in target.items():
            target_count += count
        logger.info(f"{'-'*49}")
        logger.info(f"{'Total':25} {reference_count:7} {target_count:7} {reference_count - target_count:7}")

    # .bin file difference
    reference_bin, target_bin = reference_path.with_suffix('.bin'), target_path.with_suffix('.bin')
    if not reference_bin.exists():
        logger.info(f".bin file doesn't exist in reference dir: {str(reference_bin)}")
    if not target_bin.exists():
        logger.info(f".bin file doesn't exist in target dir: {str(target_bin)}")
    if reference_bin.exists() and target_bin.exists():
        ref_bin_size = reference_bin.stat().st_size
        tar_bin_size = target_bin.stat().st_size
        if ref_bin_size == tar_bin_size:
            logger.info(f".bin file sizes are equal: {ref_bin_size} bytes")
        else:
            logger.info(
                f".bin file sizes differ: reference bin: {ref_bin_size} bytes, target bin: {tar_bin_size} bytes")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Compare XML files in two directories and report differences in <layer> tags.\n\n"
                    "This script scans two directories containing XML files and compares number of operations of different types "
                    "defined within them. It reports any differences in the operations between the reference "
                    "and target files. By default, the script filters out files containing 'tokenizer' "
                    "in their names, but this behavior can be controlled using the --filter-tokenizer or "
                    "--no-filter-tokenizer options.")
    parser.add_argument('reference', type=Path,
                        help="Path to the reference directory containing XML files or an XML file.")
    parser.add_argument('target', type=Path, help="Path to the target directory containing XML files or an XML file.")
    parser.add_argument('--filter-tokenizer', action=argparse.BooleanOptionalAction, default=True,
                        help="Filter out tokenizer files (default: True). Use --no-filter-tokenizer to include them.")
    return parser.parse_args()


def main():
    try:
        args = parse_arguments()

        reference = args.reference
        target = args.target
        filter_tokenizer = args.filter_tokenizer
        logger.info(f"Reference: {reference}")
        logger.info(f"Target: {target}")
        logger.info(f"Filter tokenizer files: {filter_tokenizer}")

        if reference.is_dir() and target.is_dir():
            reference_ir_paths = get_irs(reference)
            for reference_path in reference_ir_paths:
                if filter_tokenizer and 'tokenizer' in str(reference_path):
                    continue
                logger.info("=" * 100)
                model = reference_path.relative_to(reference)
                target_path = target / model
                if not target_path.exists():
                    logger.info(f"Model exists only in reference dir: {model}")
                else:
                    logger.info(f"Diff: {model}")
                    print_diff(reference_path, target_path)
            logger.info("=" * 100)
        elif reference.is_file() and reference.suffix == '.xml' and target.is_file() and target.suffix == '.xml':
            print_diff(reference, target)
        else:
            logger.error(f"Unexpected paths passed as reference {str(reference)} and target {str(target)}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")


if __name__ == '__main__':
    main()
