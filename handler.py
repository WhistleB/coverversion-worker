"""
Seed-VC Singing Voice Conversion Worker for RunPod Serverless.

Accepts a song audio + user voice reference (30s),
runs zero-shot singing voice conversion via Seed-VC,
returns the converted audio.
"""

import os
import sys
import base64
import tempfile
import time
import subprocess
import traceback

import requests
import runpod
import torchaudio

# ── Constants ────────────────────────────────────────────────────
SEED_VC_DIR = "/app/seed-vc"
INFERENCE_SCRIPT = os.path.join(SEED_VC_DIR, "inference.py")


def download_file(url: str, dest_path: str):
    """Download a file from URL to local path."""
    print(f"[Download] {url}")
    resp = requests.get(url, stream=True, timeout=300)
    resp.raise_for_status()
    with open(dest_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    size_mb = os.path.getsize(dest_path) / (1024 * 1024)
    print(f"[Download] Done: {size_mb:.1f} MB")


def run_seed_vc(source_path: str, target_path: str, output_dir: str,
                pitch_shift: int = 0, diffusion_steps: int = 25):
    """
    Run Seed-VC inference via subprocess.

    source_path: Song audio file (to be converted)
    target_path: User's voice reference audio (~30s)
    output_dir: Directory to save output
    pitch_shift: Semitone shift (-12 to 12)
    diffusion_steps: Quality/speed tradeoff (10=fast, 25=balanced, 50=best)
    """
    cmd = [
        "python", INFERENCE_SCRIPT,
        "--source", source_path,
        "--target", target_path,
        "--output", output_dir,
        "--diffusion-steps", str(diffusion_steps),
        "--length-adjust", "1.0",
        "--inference-cfg-rate", "0.7",
        "--f0-condition", "True",
        "--auto-f0-adjust", "True",
        "--semi-tone-shift", str(pitch_shift),
        "--fp16", "True",
    ]

    print(f"[Inference] CMD: {' '.join(cmd)}")
    start = time.time()

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=600,
        cwd=SEED_VC_DIR,
    )

    elapsed = time.time() - start
    print(f"[Inference] Finished in {elapsed:.1f}s, exit code: {result.returncode}")

    if result.stdout:
        print(f"[Inference] STDOUT (last 1000 chars):\n{result.stdout[-1000:]}")
    if result.stderr:
        print(f"[Inference] STDERR (last 1000 chars):\n{result.stderr[-1000:]}")

    if result.returncode != 0:
        raise RuntimeError(f"Seed-VC failed (exit {result.returncode}): {result.stderr[-500:]}")

    # Find the output wav file
    wav_files = [f for f in os.listdir(output_dir) if f.endswith(".wav")]
    if not wav_files:
        raise RuntimeError(f"No output .wav found in {output_dir}. Files: {os.listdir(output_dir)}")

    output_path = os.path.join(output_dir, wav_files[0])
    print(f"[Inference] Output: {output_path}")
    return output_path


def handler(job):
    """RunPod Serverless handler — main entry point for each job."""
    job_input = job["input"]

    task_id = job_input.get("task_id", "unknown")
    song_url = job_input["song_url"]
    voice_url = job_input["voice_url"]
    pitch_shift = int(job_input.get("pitch_shift", 0))
    diffusion_steps = int(job_input.get("diffusion_steps", 25))

    print(f"[Job] task_id={task_id}, pitch={pitch_shift}, steps={diffusion_steps}")

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            # ── Stage 1: Download ────────────────────────────────
            runpod.serverless.progress_update(job, {
                "task_id": task_id, "stage": "downloading", "progress": 0.1
            })

            song_path = os.path.join(tmpdir, "song_input.wav")
            voice_path = os.path.join(tmpdir, "voice_ref.wav")
            output_dir = os.path.join(tmpdir, "output")
            os.makedirs(output_dir, exist_ok=True)

            download_file(song_url, song_path)
            download_file(voice_url, voice_path)

            # ── Stage 2: Get song info ───────────────────────────
            runpod.serverless.progress_update(job, {
                "task_id": task_id, "stage": "preprocessing", "progress": 0.2
            })

            song_info = torchaudio.info(song_path)
            song_duration = song_info.num_frames / song_info.sample_rate
            print(f"[Job] Song duration: {song_duration:.1f}s")

            # ── Stage 3: Seed-VC inference ───────────────────────
            runpod.serverless.progress_update(job, {
                "task_id": task_id, "stage": "converting", "progress": 0.3
            })

            output_path = run_seed_vc(
                song_path, voice_path, output_dir,
                pitch_shift=pitch_shift,
                diffusion_steps=diffusion_steps
            )

            # ── Stage 4: Prepare result ──────────────────────────
            runpod.serverless.progress_update(job, {
                "task_id": task_id, "stage": "finishing", "progress": 0.9
            })

            output_info = torchaudio.info(output_path)
            output_duration = output_info.num_frames / output_info.sample_rate

            # Read output and encode as base64 for direct return
            # (For production, upload to S3 instead)
            with open(output_path, "rb") as f:
                audio_base64 = base64.b64encode(f.read()).decode("utf-8")

            output_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            print(f"[Job] Output: {output_duration:.1f}s, {output_size_mb:.1f} MB")

            return {
                "task_id": task_id,
                "status": "success",
                "duration": round(output_duration, 2),
                "output_audio_base64": audio_base64,
                "output_format": "wav",
                "sample_rate": output_info.sample_rate,
            }

        except Exception as e:
            traceback.print_exc()
            return {
                "task_id": task_id,
                "status": "error",
                "error": str(e),
            }


# ── Start RunPod Serverless Worker ───────────────────────────────
if __name__ == "__main__":
    print("[Init] Seed-VC RunPod Worker starting...")
    print(f"[Init] Inference script: {INFERENCE_SCRIPT}")
    print(f"[Init] Script exists: {os.path.exists(INFERENCE_SCRIPT)}")
    runpod.serverless.start({"handler": handler})
