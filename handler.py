"""
Seed-VC Cover Song Worker for RunPod Serverless.

Full pipeline:
  1. Separate vocals from instrumental (demucs)
  2. Convert vocals with Seed-VC (zero-shot)
  3. Mix converted vocals + original instrumental
  4. Upload and return result URL
"""

import os
import sys
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


def upload_file(file_path: str, filename: str) -> str:
    """Upload file to tmpfiles.org and return direct download URL."""
    print(f"[Upload] Uploading {filename} ({os.path.getsize(file_path) / 1024 / 1024:.1f} MB)...")
    with open(file_path, "rb") as f:
        resp = requests.post(
            "https://tmpfiles.org/api/v1/upload",
            files={"file": (filename, f, "audio/wav")},
            timeout=120,
        )
    resp.raise_for_status()
    data = resp.json()
    if data.get("status") != "success":
        raise RuntimeError(f"Upload failed: {data}")
    url = data["data"]["url"].replace("tmpfiles.org/", "tmpfiles.org/dl/")
    print(f"[Upload] Done: {url}")
    return url


def separate_vocals(song_path: str, output_dir: str):
    """Separate vocals and instrumental using demucs."""
    print(f"[Demucs] Separating vocals...")
    cmd = [
        "python", "-m", "demucs",
        "-n", "htdemucs",
        "--two-stems", "vocals",
        "-o", output_dir,
        song_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"Demucs failed: {result.stderr[-300:]}")

    song_name = os.path.splitext(os.path.basename(song_path))[0]
    separated_dir = os.path.join(output_dir, "htdemucs", song_name)
    vocals_path = os.path.join(separated_dir, "vocals.wav")
    instrumental_path = os.path.join(separated_dir, "no_vocals.wav")

    if not os.path.exists(vocals_path):
        raise RuntimeError(f"Vocals not found: {os.listdir(separated_dir) if os.path.exists(separated_dir) else 'dir missing'}")

    print(f"[Demucs] Done.")
    return vocals_path, instrumental_path


def run_seed_vc(source_path: str, target_path: str, output_dir: str,
                pitch_shift: int = 0, diffusion_steps: int = 25):
    """Run Seed-VC inference via subprocess."""
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

    print(f"[Inference] Starting Seed-VC...")
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd=SEED_VC_DIR)
    elapsed = time.time() - start

    print(f"[Inference] Done in {elapsed:.1f}s, exit={result.returncode}")
    if result.stderr:
        for line in result.stderr.split('\n'):
            if 'RTF' in line:
                print(f"[Inference] {line.strip()}")

    if result.returncode != 0:
        raise RuntimeError(f"Seed-VC failed: {result.stderr[-300:]}")

    wav_files = [f for f in os.listdir(output_dir) if f.endswith(".wav")]
    if not wav_files:
        raise RuntimeError(f"No output .wav found")

    return os.path.join(output_dir, wav_files[0])


def mix_audio(vocals_path: str, instrumental_path: str, output_path: str):
    """Mix converted vocals with original instrumental using ffmpeg."""
    print(f"[Mix] Mixing vocals + instrumental...")
    cmd = [
        "ffmpeg", "-y",
        "-i", vocals_path,
        "-i", instrumental_path,
        "-filter_complex",
        "[0:a]volume=1.0[v];[1:a]volume=1.0[i];[v][i]amix=inputs=2:duration=longest",
        "-ac", "2", "-ar", "44100",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg mix failed: {result.stderr[-300:]}")
    print(f"[Mix] Done.")


def handler(job):
    """RunPod Serverless handler — full cover song pipeline."""
    job_input = job["input"]

    task_id = job_input.get("task_id", "unknown")
    song_url = job_input["song_url"]
    voice_url = job_input["voice_url"]
    pitch_shift = int(job_input.get("pitch_shift", 0))
    diffusion_steps = int(job_input.get("diffusion_steps", 25))

    print(f"\n{'='*60}")
    print(f"[Job] task_id={task_id}, pitch={pitch_shift}, steps={diffusion_steps}")
    print(f"{'='*60}")

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            total_start = time.time()

            # ── Stage 1: Download ────────────────────────────────
            runpod.serverless.progress_update(job, {
                "task_id": task_id, "stage": "downloading", "progress": 0.05
            })

            song_path = os.path.join(tmpdir, "song_input.wav")
            voice_path = os.path.join(tmpdir, "voice_ref.wav")
            vc_output_dir = os.path.join(tmpdir, "vc_output")
            demucs_output_dir = os.path.join(tmpdir, "demucs_output")
            os.makedirs(vc_output_dir, exist_ok=True)

            t = time.time()
            download_file(song_url, song_path)
            download_file(voice_url, voice_path)
            download_time = time.time() - t

            song_info = torchaudio.info(song_path)
            song_duration = song_info.num_frames / song_info.sample_rate
            print(f"[Job] Song: {song_duration:.1f}s, Download: {download_time:.1f}s")

            # ── Stage 2: Vocal separation ────────────────────────
            runpod.serverless.progress_update(job, {
                "task_id": task_id, "stage": "separating", "progress": 0.1
            })

            t = time.time()
            vocals_path, instrumental_path = separate_vocals(song_path, demucs_output_dir)
            separation_time = time.time() - t
            print(f"[Job] Separation: {separation_time:.1f}s")

            # ── Stage 3: Voice conversion ────────────────────────
            runpod.serverless.progress_update(job, {
                "task_id": task_id, "stage": "converting", "progress": 0.3
            })

            t = time.time()
            converted_vocals_path = run_seed_vc(
                vocals_path, voice_path, vc_output_dir,
                pitch_shift=pitch_shift,
                diffusion_steps=diffusion_steps
            )
            conversion_time = time.time() - t
            print(f"[Job] Conversion: {conversion_time:.1f}s")

            # ── Stage 4: Mix ─────────────────────────────────────
            runpod.serverless.progress_update(job, {
                "task_id": task_id, "stage": "mixing", "progress": 0.85
            })

            t = time.time()
            final_output = os.path.join(tmpdir, "final_cover.wav")
            mix_audio(converted_vocals_path, instrumental_path, final_output)
            mix_time = time.time() - t

            # ── Stage 5: Upload ──────────────────────────────────
            runpod.serverless.progress_update(job, {
                "task_id": task_id, "stage": "uploading", "progress": 0.95
            })

            output_info = torchaudio.info(final_output)
            output_duration = output_info.num_frames / output_info.sample_rate
            output_size_mb = os.path.getsize(final_output) / (1024 * 1024)

            t = time.time()
            output_url = upload_file(final_output, f"cover_{task_id}.wav")
            upload_time = time.time() - t

            total_time = time.time() - total_start

            print(f"\n[Job] === SUMMARY ===")
            print(f"[Job] Download:   {download_time:.1f}s")
            print(f"[Job] Separation: {separation_time:.1f}s")
            print(f"[Job] Conversion: {conversion_time:.1f}s")
            print(f"[Job] Mix:        {mix_time:.1f}s")
            print(f"[Job] Upload:     {upload_time:.1f}s")
            print(f"[Job] TOTAL:      {total_time:.1f}s")
            print(f"[Job] Output:     {output_duration:.1f}s, {output_size_mb:.1f} MB")

            return {
                "task_id": task_id,
                "status": "success",
                "output_url": output_url,
                "duration": round(output_duration, 2),
                "download_time": round(download_time, 2),
                "separation_time": round(separation_time, 2),
                "conversion_time": round(conversion_time, 2),
                "mix_time": round(mix_time, 2),
                "upload_time": round(upload_time, 2),
                "total_time": round(total_time, 2),
                "output_format": "wav",
                "sample_rate": output_info.sample_rate,
                "size_mb": round(output_size_mb, 2),
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
    print("[Init] Seed-VC Cover Song Worker v3 (clean subprocess)")
    print(f"[Init] Inference script exists: {os.path.exists(INFERENCE_SCRIPT)}")
    runpod.serverless.start({"handler": handler})
