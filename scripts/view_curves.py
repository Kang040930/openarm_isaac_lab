#!/usr/bin/env python3
"""终端高清实时显示训练曲线 (需要 Kitty 终端) — Ctrl+C 退出"""

import sys
import time
import glob
import tempfile
import subprocess
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def latest_logdir(base):
    dirs = sorted(glob.glob(f"{base}/*/"))
    return dirs[-1] if dirs else None


def get_data(logdir):
    ea = EventAccumulator(logdir)
    ea.Reload()
    data = {}
    for tag in ea.Tags()["scalars"]:
        events = ea.Scalars(tag)
        data[tag] = ([e.step for e in events], [e.value for e in events])
    return data


def draw_chart(data):
    """用 matplotlib 生成高清子图 PNG"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor("#1e1e1e")

    plots = [
        (0, 0, ["Train/mean_reward", "Episode_Reward/lifting_object",
                 "Episode_Reward/object_goal_tracking"],
         "Rewards", ["#4ecdc4", "#ffe66d", "#ff6b6b"]),
        (0, 1, ["Loss/value_function", "Loss/surrogate", "Loss/entropy"],
         "Losses", ["#45b7d1", "#96ceb4", "#f7dc6f"]),
        (1, 0, ["Perf/total_fps", "Perf/collection time", "Perf/learning_time"],
         "Performance", ["#a29bfe", "#fd79a8", "#00cec9"]),
        (1, 1, ["Metrics/object_pose/position_error",
                 "Metrics/object_pose/orientation_error"],
         "Errors", ["#e17055", "#00b894"]),
    ]

    for row, col, tags, title, colors in plots:
        ax = axes[row, col]
        ax.set_facecolor("#2d2d2d")
        ax.set_title(title, color="white", fontsize=12, fontweight="bold")
        ax.tick_params(colors="gray", labelsize=8)
        ax.grid(True, alpha=0.3, color="gray")
        ax.spines["bottom"].set_color("gray")
        ax.spines["left"].set_color("gray")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        for tag, color in zip(tags, colors):
            if tag in data and data[tag][0]:
                steps, vals = data[tag]
                label = tag.split("/")[-1]
                ax.plot(steps, vals, color=color, linewidth=1.2,
                        alpha=0.9, label=label)
        ax.legend(loc="best", fontsize=7, facecolor="#2d2d2d",
                  edgecolor="gray", labelcolor="white")

    plt.tight_layout(pad=3)
    buf = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    plt.savefig(buf.name, dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)
    return buf.name


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else "logs/rsl_rl/openarm_bi_lift"
    print(f"Watching: {base}  |  Ctrl+C to quit")
    print(f"Note: requires Kitty terminal for inline image display\n")

    last_png = None

    while True:
        try:
            logdir = latest_logdir(base)
            if logdir is None:
                print(f"\rWaiting for logs...   ", end="", flush=True)
                time.sleep(2)
                continue

            data = get_data(logdir)
            if not data:
                print(f"\rNo data yet...        ", end="", flush=True)
                time.sleep(2)
                continue

            png_path = draw_chart(data)

            if last_png:
                subprocess.run(["kitty", "+kitten", "icat", "--clear",
                                "--stdin", "no"], input=b"", capture_output=True)

            subprocess.run(["kitty", "+kitten", "icat",
                            "--place", "0x0@0x0",
                            "--scale-up",
                            png_path], check=False)

            if last_png and last_png != png_path:
                import os
                os.unlink(last_png)
            last_png = png_path

            time.sleep(5)

        except KeyboardInterrupt:
            print("\nDone.")
            if last_png:
                import os
                os.unlink(last_png)
            break
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Retrying in 3s...")
            time.sleep(3)


if __name__ == "__main__":
    main()
