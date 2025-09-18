import os
import subprocess
import threading
import time
from pathlib import Path

import matplotlib.pyplot as plt
import psutil
import pynvml

SAMPLE_INTERVAL = 0.5

keep_sampling = True
cpu_samples = []
gpu_samples = []
timestamps = []


def get_gpu_utilization():
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # First GPU
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        return util.gpu  # Percentage
    except pynvml.NVMLError as e:
        print("GPU sampling error:", e)
        return 0.0


def sample_usage():
    while keep_sampling:
        cpu = psutil.cpu_percent(interval=None)  # Non-blocking
        gpu = get_gpu_utilization()
        cpu_samples.append(cpu)
        gpu_samples.append(gpu)
        timestamps.append(time.time())
        time.sleep(SAMPLE_INTERVAL)


def timed_input(prompt: str, timeout: int = 5, default: str = "yes"):
    user_input = [default]

    def ask():
        try:
            user_input[0] = input(prompt)
        except EOFError:
            pass  # in case input() fails in some environments

    thread = threading.Thread(target=ask)
    thread.daemon = True
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        print(f"\nNo input received in seconds. Defaulting to '{default}'.")
    return user_input[0]


def main(script_path_raw: str, model: str):
    global keep_sampling

    pynvml.nvmlInit()

    sampling_thread = threading.Thread(target=sample_usage)
    sampling_thread.start()

    kaggle = timed_input(
        "Are you in kaggle? Answer yes if yes", timeout=5, default="yes"
    )
    start_time = time.time()
    if kaggle == "yes":
        if model == "DLinear":
            script_path: Path = (
                Path("/content")
                / "overtourism_prediction"
                / model
                / "scripts"
                / "EXP-LongForecasting"
                / "DLinear"
                / "arena_2020_dlinear.sh"
            )
            subprocess.call(["bash", script_path])

        if model == "PatchTST":
            script_path: Path = (
                Path("/content")
                / "overtourism_prediction"
                / model
                / "scripts"
                / "PatchTST"
                / "arena_2020_patchtst_train.sh"
            )
            subprocess.call(["bash", script_path])

        if model == "Informer2020":
            script_path: Path = (
                Path("/content")
                / "overtourism_prediction"
                / model
                / "scripts"
                / "arena_2020_informer_train.sh"
            )
            subprocess.call(["bash", script_path])

        if model == "TimeMixer":
            script_path: Path = (
                Path("/content")
                / "overtourism_prediction"
                / model
                / "scripts"
                / "long_term_forecast"
                / "arena_2020_timemixer_train.sh"
            )
            subprocess.call(["bash", script_path])

    elif kaggle != "yes":
        subprocess.call(["bash", script_path_raw])
        print(script_path_raw)

    else:
        raise ValueError("Invalid input")

    end_time = time.time()
    keep_sampling = False
    sampling_thread.join()

    total_time = end_time - start_time
    print(f"\n✅ Total time to train {model}: {total_time:.2f} seconds")

    avg_cpu = sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0
    avg_gpu = sum(gpu_samples) / len(gpu_samples) if gpu_samples else 0
    print(f"📊 Average CPU usage: {avg_cpu:.2f}%")
    print(f"📊 Average GPU usage: {avg_gpu:.2f}%")

    rel_timestamps = [t - timestamps[0] for t in timestamps]

    plt.figure(figsize=(10, 5))
    plt.plot(rel_timestamps, cpu_samples, label="CPU Usage (%)", color="blue")
    plt.plot(rel_timestamps, gpu_samples, label="GPU Usage (%)", color="orange")
    plt.xlabel("Time (s)")
    plt.ylabel("Utilization (%)")
    plt.title(f"CPU and GPU Utilization Over Time for {model}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # plt.show()
    if kaggle == "yes":
        plt.savefig(f"/content/overtourism_prediction/scripts/img/{model}_arena.png")
    if kaggle != "yes":
        plt.savefig(f"img/{model}_arena.png")
    pynvml.nvmlShutdown()


def get_abs_path(script_path: str):
    cur_dir = os.getcwd().strip("scripts")
    new_path = os.path.join(cur_dir, script_path)
    return new_path


if __name__ == "__main__":
    main(
        get_abs_path(
            "DLinear/scripts/EXP-LongForecasting/DLinear/arena_2020_dlinear.sh"
        ),
        "DLinear",
    )
    # main(get_abs_path(
    #     'PatchTST/scripts/PatchTST/giulietta_patchtst.sh'),
    #     'PatchTST')
    # main(get_abs_path(
    #     'Informer2020/scripts/giulietta_informer.sh'),
    #     'Informer2020')

    # main(
    #    get_abs_path(
    #        "TimeMixer/scripts/long_term_forecast/giulietta_2020_timemixer_train.sh"
    #    ),
    #    "TimeMixer",
    # )
