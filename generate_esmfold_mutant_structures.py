import os
import time
import argparse
import pickle
from pathlib import Path

import torch
import esm
from Bio import SeqIO
from typing import Optional

def sanitize_sequence(seq: str) -> str:
    """
    Substitute non-standard amino acids:
    - 'U' (selenocysteine) -> 'C'
    - 'O' (pyrrolysine) -> 'K'
    """
    seq = seq.replace("U", "C").replace("O", "K")
    # Optionally handle other rare AAs:
    # seq = seq.replace("B", "D").replace("Z", "E").replace("X", "A")
    return seq

def format_hms(seconds: float) -> str:
    """Convert seconds to H:MM:SS or M:SS."""
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h:d}:{m:02d}:{s:02d}"
    else:
        return f"{m:d}:{s:02d}"

def create_batched_job_dataset(
    jobs, max_tokens_per_batch: int
):
    batch_headers = []
    batch_seqs = []
    batch_pdbs = []
    batch_pkls = []
    num_tokens = 0

    for header, seq, pdb_path, pkl_path in jobs:
        # -- SANITIZE sequences before batching! --
        seq = sanitize_sequence(seq)

        seq_len = len(seq)
        if num_tokens > 0 and (num_tokens + seq_len > max_tokens_per_batch):
            yield batch_headers, batch_seqs, batch_pdbs, batch_pkls
            batch_headers, batch_seqs = [], []
            batch_pdbs, batch_pkls = [], []
            num_tokens = 0

        batch_headers.append(header)
        batch_seqs.append(seq)
        batch_pdbs.append(pdb_path)
        batch_pkls.append(pkl_path)
        num_tokens += seq_len

    if batch_headers:
        yield batch_headers, batch_seqs, batch_pdbs, batch_pkls

def main(
    data_parent: str,
    num_samples: Optional[int] = None,
    max_tokens_per_batch: int = 1024,
    num_recycles: Optional[int] = None,
    chunk_size: Optional[int] = None,
    cpu_only: bool = False,
    start_from_longest: bool = False,
    array_ID: Optional[int] = None,
    array_N: Optional[int] = None,
):
    print("CUDA available:", torch.cuda.is_available(), flush=True)

    model = esm.pretrained.esmfold_v1()
    model = model.eval()
    if chunk_size is not None:
        model.set_chunk_size(chunk_size)

    if cpu_only or not torch.cuda.is_available():
        device = "cpu"
        model.cpu()
    else:
        device = "cuda"
        model.cuda()

    print(f"Loaded ESMFold model on {device}", flush=True)

    jobs = []
    data_parent_path = Path(data_parent)

    for sample_name in os.listdir(data_parent_path):
        sample_dir = data_parent_path / sample_name
        mut_dir = sample_dir / "mut_data"
        mut_fasta = mut_dir / "mut.fasta"
        mut_pdb = mut_dir / "mut_esmf.pdb"
        mut_pkl = mut_dir / "mut_esmf.pkl"

        if mut_fasta.exists():
            if mut_pkl.exists():
                continue

            record = SeqIO.read(str(mut_fasta), "fasta")
            header = record.id
            seq = str(record.seq)
            # -- SANITIZE sequence for rare residues --
            seq = sanitize_sequence(seq)
            jobs.append((header, seq, str(mut_pdb), str(mut_pkl)))

    if num_samples is not None:
        jobs = jobs[:num_samples]

    jobs.sort(key=lambda x: len(x[1]), reverse=start_from_longest)

    # Partition jobs by SLURM array variables, if requested
    total_jobs_orig = len(jobs)
    if array_ID is not None and array_N is not None:
        # Only process jobs where i % array_N == array_ID
        jobs = [job for i, job in enumerate(jobs) if i % array_N == array_ID]
        print(f"SLURM array partitioning: array_ID {array_ID}, array_N {array_N}, assigned jobs: {len(jobs)} (out of {total_jobs_orig})", flush=True)

    total_jobs = len(jobs)
    print(f"Will predict {total_jobs} mut structures (missing pkl files).", flush=True)

    if total_jobs == 0:
        print("No mut.fasta files needing pkl generation found; exiting.", flush=True)
        return

    completed = 0
    start_time = time.time()

    for headers, sequences, pdb_paths, pkl_paths in create_batched_job_dataset(
        jobs, max_tokens_per_batch=max_tokens_per_batch
    ):
        batch_size = len(sequences)
        if batch_size == 0:
            continue
        filtered = []
        for h, s, pdb, pkl in zip(headers, sequences, pdb_paths, pkl_paths):
            if os.path.exists(pkl):
                completed += 1
                continue
            filtered.append((h, s, pdb, pkl))
        if not filtered:
            continue
        headers, sequences, pdb_paths, pkl_paths = zip(*filtered)
        headers, sequences = list(headers), list(sequences)
        pdb_paths, pkl_paths = list(pdb_paths), list(pkl_paths)
        batch_size = len(sequences)

        try:
            output = model.infer(sequences, num_recycles=num_recycles)
        except RuntimeError as e:
            msg = str(e)
            if msg.startswith("CUDA out of memory"):
                print(
                    f"[WARN] CUDA OOM on batch of size {batch_size} "
                    f"(max_tokens_per_batch={max_tokens_per_batch}). "
                    f"Consider lowering --max_tokens_per_batch.",
                    flush=True,
                )
                continue
            else:
                print(
                    f"[ERROR] Unexpected RuntimeError on batch {headers}: {e}",
                    flush=True,
                )
                continue

        output = {key: value.cpu() for key, value in output.items()}
        pdb_strings = model.output_to_pdb(output)
        plddts = output.get("plddt", None)
        mean_plddt = output.get("mean_plddt", None)
        ptm = output.get("ptm", None)

        for idx, (header, seq, pdb_str, pdb_path, pkl_path) in enumerate(
            zip(headers, sequences, pdb_strings, pdb_paths, pkl_paths)
        ):
            if not os.path.exists(pdb_path):
                with open(pdb_path, "w") as f:
                    f.write(pdb_str)
                print(f"Wrote {pdb_path}", flush=True)

            if plddts is not None and not os.path.exists(pkl_path):
                plddt_vec = plddts[idx].numpy()
                with open(pkl_path, "wb") as f:
                    pickle.dump({"plddt": plddt_vec}, f)
                print(f"Wrote {pkl_path}", flush=True)

            completed += 1
            elapsed = time.time() - start_time
            avg_per_job = elapsed / max(completed, 1)
            remaining = total_jobs - completed
            eta = remaining * avg_per_job
            pct = 100.0 * completed / total_jobs if total_jobs > 0 else 100.0
            mean_pl = mean_plddt[idx].item() if mean_plddt is not None else float("nan")
            ptm_val = ptm[idx].item() if ptm is not None else float("nan")

            print(
                f"[{completed}/{total_jobs}] ({pct:5.1f}%) "
                f"| Elapsed: {format_hms(elapsed)} "
                f"| ETA: {format_hms(eta)} "
                f"| len={len(seq)} "
                f"| mean pLDDT={mean_pl:0.1f} pTM={ptm_val:0.3f} "
                f"| {header}",
                flush=True,
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_parent", required=True)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--max_tokens_per_batch", type=int, default=1024)
    parser.add_argument("--num_recycles", type=int, default=None)
    parser.add_argument("--chunk_size", type=int, default=None)
    parser.add_argument("--cpu_only", action="store_true")
    parser.add_argument("--start_from_longest", action="store_true")
    # Array job parameters
    parser.add_argument("--ID", type=int, default=None, help="SLURM array job index (0-based)")
    parser.add_argument("--N", type=int, default=None, help="Total number of array jobs")

    args = parser.parse_args()

    # Get SLURM_ARRAY_TASK_ID from environment if --ID not provided
    if args.ID is None:
        slurm_id = os.environ.get("SLURM_ARRAY_TASK_ID", None)
        if slurm_id is not None:
            args.ID = int(slurm_id)

    main(
        data_parent=args.data_parent,
        num_samples=args.num_samples,
        max_tokens_per_batch=args.max_tokens_per_batch,
        num_recycles=args.num_recycles,
        chunk_size=args.chunk_size,
        cpu_only=args.cpu_only,
        start_from_longest=args.start_from_longest,
        array_ID=args.ID,
        array_N=args.N
    )