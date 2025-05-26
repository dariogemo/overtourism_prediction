from pathlib import Path
import subprocess
from run_all_time_metric import get_abs_path, timed_input


def main(script_path, model):
    kaggle = timed_input(
        "Are you in colab? Answer yes if yes", timeout=5, default="yes"
    )
    train_inp = timed_input(
        "Do you want to train the model or just do prediction?\n[1 - train and prediction]\n[2 - prediction]\n",
        timeout=5,
        default="1",
    )

    if train_inp == "1":
        if kaggle == "yes":
            if model == "DLinear":
                script_path = (
                    Path("/content")
                    / "overtourism_prediction"
                    / model
                    / "scripts"
                    / "EXP-LongForecasting"
                    / "DLinear"
                    / "giulietta_2020_dlinear.sh"
                )
                script_train = script_path.with_name(script_path.stem + "_train.sh")
                subprocess.call(["bash", str(script_train)])
                script_test = script_path.with_name(script_path.stem + "_test.sh")
                subprocess.call(["bash", script_test])

            if model == "PatchTST":
                script_path = (
                    Path("/content")
                    / "overtourism_prediction"
                    / model
                    / "scripts"
                    / "PatchTST"
                    / "giulietta_2020_patchtst.sh"
                )
                script_train = script_path.with_name(script_path.stem + "_train.sh")
                subprocess.call(["bash", str(script_train)])
                script_test = script_path.with_name(script_path.stem + "_test.sh")
                subprocess.call(["bash", script_test])

            if model == "Informer2020":
                script_path = (
                    Path("/content")
                    / "overtourism_prediction"
                    / model
                    / "scripts"
                    / "giulietta_2020_informer.sh"
                )
                script_train = script_path.with_name(script_path.stem + "_train.sh")
                subprocess.call(["bash", str(script_train)])
                script_test = script_path.with_name(script_path.stem + "_test.sh")
                subprocess.call(["bash", script_test])

        elif kaggle != "yes":
            script_path = Path(script_path)
            script_train = script_path.with_name(script_path.stem + "_train.sh")
            subprocess.call(["bash", str(script_train)])
            script_test = script_path.with_name(script_path.stem + "_test.sh")
            subprocess.call(["bash", script_test])

    if train_inp == "2":
        if kaggle == "yes":
            if model == "DLinear":
                script_path = (
                    Path("/content")
                    / "overtourism_prediction"
                    / model
                    / "scripts"
                    / "EXP-LongForecasting"
                    / "DLinear"
                    / "giulietta_2020_dlinear.sh"
                )
                script_path = str(script_path).strip(".sh") + "_test.sh"
                subprocess.call(["bash", script_path])

            if model == "PatchTST":
                script_path = (
                    Path("/content")
                    / "overtourism_prediction"
                    / model
                    / "scripts"
                    / "PatchTST"
                    / "giulietta_2020_patchtst.sh"
                )
                script_path = str(script_path).strip(".sh") + "_test.sh"
                subprocess.call(["bash", script_path])

            if model == "Informer2020":
                script_path = (
                    Path("/content")
                    / "overtourism_prediction"
                    / model
                    / "scripts"
                    / "giulietta_2020_informer.sh"
                )
                script_path = str(script_path).strip(".sh") + "_test.sh"
                subprocess.call(["bash", script_path])

        elif kaggle != "yes":
            script_path = Path(str(script_path).strip(".sh") + "_test.sh")
            subprocess.call(["bash", script_path])

    else:
        pass


if __name__ == "__main__":
    main(
        get_abs_path(
            # "DLinear/scripts/EXP-LongForecasting/DLinear/giulietta_2020_dlinear.sh"
            # "PatchTST/scripts/PatchTST/giulietta_2020_patchtst.sh"
            "Informer2020/scripts/giulietta_2020_informer.sh"
        ),
        # "DLinear",
        # "PatchTST",
        "Informer2020",
    )
