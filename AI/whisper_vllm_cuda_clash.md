# Whisper vs vLLM CUDA clash

On EXCLUSIVE_PROCESS GPUs, Whisper and vLLM cannot share the same GPU
process context simultaneously.

## Symptom

`marquis-extract qa` fails with CUDA errors when it tries to run both
Whisper (for transcript preparation) and vLLM (for QA answering) in the
same process on an exclusive GPU.

## Workaround

Run transcript preparation as a separate step that exits and releases the
GPU before QA starts:

```bash
# Step 1: prepare transcripts (uses Whisper, caches to data.transcripts)
marquis-extract prepare-transcripts data.transcripts=outputs/transcripts.json ...

# Step 2: QA answering (uses vLLM, reads cached transcripts)
marquis-extract qa data.transcripts=outputs/transcripts.json ...
```

The `prepare-transcripts` subcommand caches `{video_id: transcript}` to a
JSON file, exits, and frees the GPU. The `qa` subcommand then reads from
the cache instead of invoking Whisper.

## SLURM implications

In a single-GPU SBATCH job, chain the two steps sequentially in the script.
Do not attempt to run them as parallel srun steps on the same GPU.
