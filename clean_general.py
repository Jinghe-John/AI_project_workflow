#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
JSON File Cleaner
Traverses a directory, finds files matching a given suffix (default: *_model.json),
retains only the 'type' and 'content' fields, and writes cleaned files to an output directory.

Usage:
    python clean_json.py -i <input_dir> -o <output_dir>
    python clean_json.py --input <input_dir> --output <output_dir>

Examples:
    python clean_json.py -i /data/input/ -o /data/output/
    python clean_json.py --input ./en_output_1124 --output ./cleaned_output
"""

import os
import json
import argparse
from tqdm import tqdm
from datetime import datetime
import pandas as pd


def clean_json_content(data):
    """
    Recursively clean JSON data, keeping only 'type' and 'content' fields.
    """
    if isinstance(data, list):
        cleaned = []
        for item in data:
            if isinstance(item, dict):
                cleaned_item = {}
                if 'type' in item:
                    cleaned_item['type'] = item['type']
                if 'content' in item:
                    cleaned_item['content'] = item['content']
                if cleaned_item:  # Only append if at least one field was found
                    cleaned.append(cleaned_item)
            elif isinstance(item, list):
                cleaned.append(clean_json_content(item))
        return cleaned
    elif isinstance(data, dict):
        # If the top level is a dict, recurse into nested lists/dicts
        cleaned = {}
        for key, value in data.items():
            if isinstance(value, (list, dict)):
                cleaned[key] = clean_json_content(value)
            else:
                cleaned[key] = value
        return cleaned
    else:
        return data


def find_model_json_files(input_dir, suffix):
    """
    Walk the directory tree and return all file paths matching the given suffix.
    """
    json_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(suffix):
                json_files.append(os.path.join(root, file))
    return json_files


def process_json_files(input_dir, output_dir, suffix, keep_structure):
    """
    Process all matching JSON files: clean and write to the output directory.

    Returns a list of record dicts describing the outcome of each file.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("Scanning directory for matching files...")
    json_files = find_model_json_files(input_dir, suffix)
    print(f"Found {len(json_files)} file(s) to process.")

    if not json_files:
        return []

    records = []

    for file_path in tqdm(json_files, desc="Cleaning JSON files", unit="file"):
        record = {
            'Source Path':      file_path,
            'Source Filename':  os.path.basename(file_path),
            'Output Filename':  '',
            'Output Path':      '',
            'Status':           '',
            'Error':            '',
            'Processed At':     datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            cleaned_data = clean_json_content(data)

            # Derive the output filename by stripping the suffix and appending .json
            # e.g. 2014_09_12_9_model.json -> 2014_09_12_9.json
            original_filename = os.path.basename(file_path)
            new_filename = original_filename.replace(suffix, '.json')

            if keep_structure:
                # Mirror the source directory structure under output_dir
                relative_path = os.path.relpath(os.path.dirname(file_path), input_dir)
                output_subdir = os.path.join(output_dir, relative_path)
                os.makedirs(output_subdir, exist_ok=True)
                output_path = os.path.join(output_subdir, new_filename)
            else:
                # Flatten: place all output files in output_dir directly
                output_path = os.path.join(output_dir, new_filename)

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(cleaned_data, f, ensure_ascii=False, indent=2)

            record['Output Filename'] = new_filename
            record['Output Path']     = output_path
            record['Status']          = 'Success'

        except json.JSONDecodeError as e:
            record['Status'] = 'Failed'
            record['Error']  = f'JSON parse error: {e}'
        except Exception as e:
            record['Status'] = 'Failed'
            record['Error']  = str(e)

        records.append(record)

    return records


def save_report(records, output_dir, report_name=None):
    """
    Save a processing report (detailed records + summary) to an Excel file.
    """
    df = pd.DataFrame(records)

    total   = len(records)
    success = sum(1 for r in records if r['Status'] == 'Success')
    failed  = total - success

    if report_name:
        report_filename = report_name if report_name.endswith('.xlsx') else f"{report_name}.xlsx"
    else:
        report_filename = f"cleaning_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"

    report_path = os.path.join(output_dir, report_filename)

    with pd.ExcelWriter(report_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Detailed Records', index=False)

        summary_df = pd.DataFrame({
            'Metric': ['Total Files', 'Succeeded', 'Failed', 'Success Rate'],
            'Value':  [
                total,
                success,
                failed,
                f'{success / total * 100:.2f}%' if total > 0 else '0%',
            ],
        })
        summary_df.to_excel(writer, sheet_name='Summary', index=False)

    print(f"\nReport saved to: {report_path}")
    print(f"Results  —  Total: {total}  |  Succeeded: {success}  |  Failed: {failed}")

    return report_path


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description='JSON File Cleaner — retains only "type" and "content" fields.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  %(prog)s -i /data/input/ -o /data/output/
  %(prog)s --input ./en_output_1124 --output ./cleaned_output
  %(prog)s -i ./input -o ./output --suffix "_model.json" --flat
  %(prog)s -i ./input -o ./output --report my_report.xlsx
        '''
    )

    parser.add_argument(
        '-i', '--input',
        type=str, required=True,
        help='Input directory path (required).'
    )
    parser.add_argument(
        '-o', '--output',
        type=str, required=True,
        help='Output directory path (required).'
    )
    parser.add_argument(
        '-s', '--suffix',
        type=str, default='_model.json',
        help='File suffix to match (default: _model.json).'
    )
    parser.add_argument(
        '--flat',
        action='store_true',
        help='Write all output files to a single directory instead of mirroring the source structure.'
    )
    parser.add_argument(
        '-r', '--report',
        type=str, default=None,
        help='Custom report filename (default: cleaning_report_<timestamp>.xlsx).'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    input_dir      = args.input
    output_dir     = args.output
    suffix         = args.suffix
    keep_structure = not args.flat
    report_name    = args.report

    print("=" * 60)
    print("JSON File Cleaner")
    print("=" * 60)
    print(f"Input directory  : {input_dir}")
    print(f"Output directory : {output_dir}")
    print(f"File suffix      : {suffix}")
    print(f"Keep structure   : {'yes' if keep_structure else 'no (flat)'}")
    print("=" * 60)

    if not os.path.exists(input_dir):
        print(f"Error: input directory does not exist — {input_dir}")
        return

    records = process_json_files(input_dir, output_dir, suffix, keep_structure)

    if records:
        save_report(records, output_dir, report_name)
    else:
        print("No matching files were found; nothing to process.")

    print("\nDone.")


if __name__ == "__main__":
    main()
