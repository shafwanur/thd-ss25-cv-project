import re
import shutil
import subprocess
from itertools import chain, dropwhile
from pathlib import Path
from typing import Callable, ParamSpec

import requests


def prepare_tmp_output(path: Path):
    save_dir = path.parent
    save_dir.mkdir(parents=True, exist_ok=True)

    tmp_filepath = save_dir / (path.name + ".tmp")
    shutil.rmtree(tmp_filepath, ignore_errors=True)
    return tmp_filepath


P = ParamSpec("P")


def mk(fn: Callable[P, None], *args: P.args, **kwargs: P.kwargs):
    output: Path = kwargs["output"]
    if output.exists():
        print(f"{output}: already written")
        return
    tmp_output = prepare_tmp_output(output)
    kwargs["output"] = tmp_output
    try:
        fn(*args, **kwargs)
    except BaseException as exc:
        shutil.rmtree(tmp_output, ignore_errors=True)
        raise exc
    _ = tmp_output.rename(output)


def keep_file(fn: Callable[P, None]):
    def f(*args: P.args, **kwargs: P.kwargs):
        mk(fn, *args, **kwargs)

    return f


@keep_file
def download_file(*, url: str, output: Path):
    written = 0

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(output, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                written += f.write(chunk)
                print(f"{output}: written {(written // 100000) / 10}MB")


def lines(*, input: Path):
    with input.open() as i:
        for line in i:
            yield line


def filtered_lines(*, input: Path, filter: Callable[[str], object]):
    lines_read = 0
    lines_written = 0
    with input.open() as i:
        for line in i:
            lines_read += 1
            if filter(line):
                lines_written += 1
                yield line
            if lines_read % 100000 == 0:
                print(f"filtering {input}: {lines_read} -> {lines_written}")
    print(f"filtered {input}: {lines_read} -> {lines_written}")


@keep_file
def filter_file(*, input: Path, output: Path, filter: str):
    pattern = re.compile(filter)

    with output.open("w") as o:
        for line in filtered_lines(input=input, filter=pattern.search):
            _ = o.write(line)


@keep_file
def prepare_image_ids(*, data: Path, output: Path):
    splits = ["train", "validation", "test"]
    filtered = chain(
        *map(
            lambda split: map(
                lambda x: split + "/" + x.split(",")[0] + "\n",
                dropwhile(
                    lambda x: x.startswith("ImageID,"),
                    lines(input=data / split / "box.csv"),
                ),
            ),
            splits,
        )
    )

    with output.open("w") as o:
        for line in filtered:
            _ = o.write(line)


@keep_file
def download_images(*, output: Path, downloader: Path, ids: Path):
    import sys
    _ = subprocess.run([sys.executable, downloader, "--download_folder", output, ids])


@keep_file
def prepare_data(*, filtered: Path, output: Path, label2id: dict[str, int]):
    splits = ["train", "validation", "test"]

    for split in splits:
        images = output / "images" / split
        images.mkdir(parents=True)
        labels = output / "labels" / split
        labels.mkdir(parents=True)

        with (filtered / split / "box.csv").open() as f:
            for line in dropwhile(lambda x: x.startswith("ImageID,"), f):
                # ImageID,Source,LabelName,Confidence,XMin,XMax,YMin,YMax,IsOccluded,IsTruncated,IsGroupOf,IsDepiction,IsInside
                id, _, label, _, xmin, xmax, ymin, ymax, is_occluded, is_truncated, is_group_of, is_depiction, is_inside, *_ = line.split(",")
                if int(is_group_of) != 0:
                    continue

                xmin = float(xmin)
                xmax = float(xmax)
                ymin = float(ymin)
                ymax = float(ymax)

                label_id = label2id[label]

                target = images / f"{id}.jpg"
                if not target.exists():
                    _ = shutil.copy(filtered / "images" / f"{id}.jpg", target)

                with (labels / f"{id}.txt").open("a") as f:
                    xcenter = (xmin + xmax) / 2
                    ycenter = (ymin + ymax) / 2
                    width = xmax - xmin
                    height = ymax - ymin
                    _ = f.write(f"{label_id} {xcenter} {ycenter} {width} {height}\n")


if __name__ == "__main__":
    data = Path("./data")

    original = data / "original"

    download_file(
        url="https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv",
        output=original / "train" / "box.csv",
    )
    download_file(
        url="https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv",
        output=original / "validation" / "box.csv",
    )
    download_file(
        url="https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv",
        output=original / "test" / "box.csv",
    )

    download_file(
        url="https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv",
        output=original / "meta" / "class-name.csv",
    )

    name_of_interest = re.compile(r"(egg)|(chicken)", re.IGNORECASE)
    labels_of_interest: list[str] = []
    names_of_interest: list[str] = []

    for line in filtered_lines(
        input=original / "meta" / "class-name.csv", filter=name_of_interest.search
    ):
        label, display = line.split(",")
        labels_of_interest.append(label.strip())
        names_of_interest.append(display.strip())

    label_map = {label: idx for idx, label in enumerate(labels_of_interest)}

    filtered = data / "filtered"

    for split in ["train", "validation", "test"]:
        filter_file(
            input=original / split / "box.csv",
            output=filtered / split / "box.csv",
            filter="|".join(
                map(
                    lambda x: f"({x})",
                    labels_of_interest + ["^ImageID,"],
                )
            ),
        )

    prepare_image_ids(data=filtered, output=filtered / "ids.txt")

    download_file(
        url="https://raw.githubusercontent.com/openimages/dataset/master/downloader.py",
        output=data / "downloader.py",
    )

    download_images(
        output=filtered / "images",
        downloader=data / "downloader.py",
        ids=filtered / "ids.txt",
    )

    prepared = data / "prepared"
    if prepared.exists():
        shutil.rmtree(prepared)
    prepare_data(filtered=filtered, output=prepared, label2id=label_map)

    with (data / "desc.yaml").open("w") as f:
        _ = f.write(f"path: {data / 'prepared'}\n")
        _ = f.write(f"train: images/train\n")
        _ = f.write(f"val: images/validation\n")
        _ = f.write(f"test: images/test\n")
        _ = f.write(f"names:\n")
        for idx, name in enumerate(names_of_interest):
            _ = f.write(f"    {idx}: {name}\n")
