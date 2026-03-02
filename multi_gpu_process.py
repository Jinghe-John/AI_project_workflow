#!/usr/bin/env python3
"""
Multi-GPU Parallel PDF Processing
Distributes PDF folders across N GPU workers and records results to a CSV file.

Usage:
    Set INPUT_ROOT, OUTPUT_DIR, CSV_FILE, and NUM_GPUS, then run:
        python process_pdfs_multi_gpu.py
"""

import csv
import multiprocessing as mp
import os
import queue
import subprocess
import time
from collections import defaultdict
from datetime import datetime
from multiprocessing import Queue
from pathlib import Path

# ============================================================
# Configuration
# ============================================================
INPUT_ROOT = "/path/to/your/pdf/input/directory"
OUTPUT_DIR = "pdf_output"
CSV_FILE   = "pdf_processing_results.csv"
NUM_GPUS   = 4          # Number of available GPUs (e.g. A800s)

PROCESS_TIMEOUT  = 3600  # Max seconds per folder before killing the subprocess
WORKER_STAGGER   = 2     # Seconds to stagger worker process launches
PROGRESS_EVERY   = 10    # Print an overall progress update every N completed folders

# CSV column headers
CSV_HEADERS = ["filename", "full_path", "is_corrupted", "error_message", "process_time"]


# ============================================================
# PDF Discovery
# ============================================================

def find_pdf_folders(root_dir: str) -> dict[str, list[str]]:
    """
    Walk root_dir and return a mapping of folder path → list of PDF filenames
    for every folder that contains at least one PDF.
    """
    pdf_folders: dict[str, list[str]] = defaultdict(list)
    for root, _dirs, files in os.walk(root_dir):
        pdfs = [f for f in files if f.lower().endswith(".pdf")]
        if pdfs:
            pdf_folders[root] = pdfs
    return pdf_folders


# ============================================================
# Folder Processor  (runs inside a worker process)
# ============================================================

def process_folder(
    folder_path: str,
    pdf_files: list[str],
    output_dir: str,
    gpu_id: int,
    env: dict,
) -> list[tuple]:
    """
    Invoke MinerU on every PDF in folder_path and return a list of result tuples:
        (filename_no_ext, full_path, success: bool, error_msg: str, avg_duration: float)
    """
    results: list[tuple] = []
    print(f"[GPU {gpu_id}] Processing folder: {folder_path}  ({len(pdf_files)} PDF(s))")

    cmd = ["mineru", "-p", folder_path, "-o", output_dir, "-b", "vlm-vllm-engine"]
    start_time = datetime.now()

    try:
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        output_lines: list[str] = []
        error_lines: dict[str, list[str]] = defaultdict(list)

        for line in process.stdout:
            output_lines.append(line)
            line_lower = line.lower()
            if "error" in line_lower or "failed" in line_lower:
                for pdf_file in pdf_files:
                    if pdf_file.replace(".pdf", "") in line:
                        error_lines[pdf_file].append(line.strip())

        return_code = process.wait(timeout=PROCESS_TIMEOUT)

        duration = (datetime.now() - start_time).total_seconds()
        avg_time = duration / len(pdf_files)

        for pdf_file in pdf_files:
            stem      = pdf_file[:-4] if pdf_file.endswith(".pdf") else pdf_file
            full_path = os.path.join(folder_path, pdf_file)
            out_path  = os.path.join(output_dir, stem)

            if os.path.exists(out_path) and os.listdir(out_path):
                results.append((stem, full_path, True, "", avg_time))
            else:
                msg = "\n".join(error_lines.get(pdf_file, ["Output not generated"]))
                results.append((stem, full_path, False, msg[:500], 0.0))

        success_count = sum(1 for r in results if r[2])
        print(
            f"[GPU {gpu_id}] ✓ Done: {folder_path} — "
            f"{success_count}/{len(pdf_files)} succeeded  ({duration:.1f}s)"
        )

    except subprocess.TimeoutExpired:
        process.kill()
        print(f"[GPU {gpu_id}] ✗ Timeout: {folder_path}")
        for pdf_file in pdf_files:
            stem      = pdf_file[:-4] if pdf_file.endswith(".pdf") else pdf_file
            full_path = os.path.join(folder_path, pdf_file)
            results.append((stem, full_path, False, "Timeout", 0.0))

    except Exception as exc:
        print(f"[GPU {gpu_id}] ✗ Error: {folder_path} — {exc}")
        for pdf_file in pdf_files:
            stem      = pdf_file[:-4] if pdf_file.endswith(".pdf") else pdf_file
            full_path = os.path.join(folder_path, pdf_file)
            results.append((stem, full_path, False, str(exc), 0.0))

    return results


# ============================================================
# Worker Process  (one per GPU)
# ============================================================

def worker_process(
    gpu_id: int,
    folder_queue: Queue,
    result_queue: Queue,
    output_dir: str,
) -> None:
    """
    Long-running worker that pulls folder tasks from folder_queue,
    processes them, and pushes result lists into result_queue.
    A None sentinel in folder_queue signals this worker to exit.
    """
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"]      = str(gpu_id)
    env["OMP_NUM_THREADS"]           = "1"
    env["ORT_DISABLE_THREAD_AFFINITY"] = "1"

    print(f"[GPU {gpu_id}] Worker started.")

    while True:
        try:
            item = folder_queue.get(timeout=5)
        except queue.Empty:
            continue

        if item is None:  # Sentinel — time to exit
            print(f"[GPU {gpu_id}] Received exit signal — shutting down.")
            break

        folder_path, pdf_files = item
        try:
            results = process_folder(folder_path, pdf_files, output_dir, gpu_id, env)
            result_queue.put(results)
        except Exception as exc:
            print(f"[GPU {gpu_id}] Worker error while processing {folder_path}: {exc}")
            break

    print(f"[GPU {gpu_id}] Worker finished.")


# ============================================================
# CSV Writer Helper
# ============================================================

def open_csv_writer(csv_path: str):
    """Open a CSV file for writing and return (file_handle, csv.writer)."""
    fh = open(csv_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(fh)
    writer.writerow(CSV_HEADERS)
    return fh, writer


def write_batch_to_csv(
    writer,
    results: list[tuple],
) -> tuple[int, int]:
    """
    Write a batch of result tuples to the CSV.
    Returns (success_count, fail_count) for this batch.
    """
    success = fail = 0
    for filename, full_path, ok, error_msg, duration in results:
        writer.writerow([
            filename,
            full_path,
            "FALSE" if ok else "TRUE",
            error_msg,
            f"{duration:.1f}s",
        ])
        if ok:
            success += 1
        else:
            fail += 1
    return success, fail


# ============================================================
# Progress Reporter
# ============================================================

def print_progress(
    processed_folders: int,
    total_folders: int,
    processed_pdfs: int,
    total_pdfs: int,
    total_success: int,
    total_failed: int,
    elapsed_seconds: float,
) -> None:
    """Print a formatted progress snapshot to stdout."""
    avg_speed     = processed_pdfs / elapsed_seconds if elapsed_seconds > 0 else 0.0
    remaining     = total_pdfs - processed_pdfs
    eta_seconds   = remaining / avg_speed if avg_speed > 0 else 0.0

    print(f"\n{'=' * 60}")
    print(f"  Folders : {processed_folders}/{total_folders}  ({processed_folders / total_folders * 100:.1f}%)")
    print(f"  PDFs    : {processed_pdfs}/{total_pdfs}  ({processed_pdfs / total_pdfs * 100:.1f}%)")
    print(f"  Success : {total_success}  |  Failed: {total_failed}")
    print(f"  Speed   : {avg_speed:.2f} PDF/s  ({avg_speed * 60:.1f} PDF/min)")
    print(f"  ETA     : {eta_seconds / 3600:.1f} hour(s)")
    print(f"{'=' * 60}")


# ============================================================
# Main
# ============================================================

def main() -> None:
    print("=" * 60)
    print("MinerU — Multi-GPU Parallel PDF Processing")
    print("=" * 60)
    print(f"  Input directory : {INPUT_ROOT}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  CSV output      : {CSV_FILE}")
    print(f"  GPUs            : {NUM_GPUS}")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Discover PDFs ────────────────────────────────────────
    print("\nScanning for PDF folders...")
    pdf_folders    = find_pdf_folders(INPUT_ROOT)
    total_folders  = len(pdf_folders)
    total_pdfs     = sum(len(f) for f in pdf_folders.values())

    print(f"  Folders with PDFs : {total_folders}")
    print(f"  Total PDF files   : {total_pdfs}")
    print(f"  Folders per GPU   : ~{total_folders // NUM_GPUS}\n")

    if total_pdfs == 0:
        print("No PDF files found — nothing to do.")
        return

    # ── Build task & result queues ───────────────────────────
    folder_queue: Queue = Queue()
    result_queue: Queue = Queue()

    for folder_path, pdf_files in sorted(pdf_folders.items()):
        folder_queue.put((folder_path, pdf_files))

    # One None sentinel per worker signals it to exit after the queue is drained
    for _ in range(NUM_GPUS):
        folder_queue.put(None)

    # ── Launch GPU workers ───────────────────────────────────
    print(f"Launching {NUM_GPUS} GPU worker process(es)...\n")
    processes: list[mp.Process] = []
    for gpu_id in range(NUM_GPUS):
        p = mp.Process(
            target=worker_process,
            args=(gpu_id, folder_queue, result_queue, OUTPUT_DIR),
        )
        p.start()
        processes.append(p)
        time.sleep(WORKER_STAGGER)

    # ── Collect results ──────────────────────────────────────
    csv_fh, csv_writer = open_csv_writer(CSV_FILE)
    total_success = total_failed = processed_folders = 0
    overall_start = datetime.now()

    print("Collecting results...\n")

    try:
        while processed_folders < total_folders:
            try:
                results = result_queue.get(timeout=10)
            except queue.Empty:
                if not any(p.is_alive() for p in processes):
                    print("All worker processes have exited.")
                    break
                continue

            batch_ok, batch_fail = write_batch_to_csv(csv_writer, results)
            total_success    += batch_ok
            total_failed     += batch_fail
            processed_folders += 1
            csv_fh.flush()

            if processed_folders % PROGRESS_EVERY == 0:
                elapsed = (datetime.now() - overall_start).total_seconds()
                print_progress(
                    processed_folders, total_folders,
                    total_success + total_failed, total_pdfs,
                    total_success, total_failed, elapsed,
                )

    except KeyboardInterrupt:
        print("\n\nInterrupted by user — terminating all workers...")
        for p in processes:
            p.terminate()

    # ── Teardown ─────────────────────────────────────────────
    print("\nWaiting for all workers to finish...")
    for p in processes:
        p.join(timeout=10)
        if p.is_alive():
            p.terminate()

    csv_fh.close()

    total_duration = (datetime.now() - overall_start).total_seconds()

    # ── Final summary ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Processing complete.")
    print("=" * 60)
    print(f"  GPUs used        : {NUM_GPUS}")
    print(f"  Total folders    : {total_folders}")
    print(f"  Total PDFs       : {total_pdfs}")
    success_pct = total_success / total_pdfs * 100 if total_pdfs else 0
    failed_pct  = total_failed  / total_pdfs * 100 if total_pdfs else 0
    print(f"  Succeeded        : {total_success}  ({success_pct:.1f}%)")
    print(f"  Failed           : {total_failed}  ({failed_pct:.1f}%)")
    print(f"  Total time       : {total_duration / 3600:.2f} hour(s)")
    print(f"  Average speed    : {total_pdfs / total_duration:.2f} PDF/s")
    print(f"  Theoretical speedup: ~{NUM_GPUS}x")
    print(f"\n  Results saved to : {CSV_FILE}")
    print("=" * 60)


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
