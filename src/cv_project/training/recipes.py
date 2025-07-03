import re
import shutil
import subprocess
import sys
from itertools import chain, dropwhile
from pathlib import Path

from .lazy import force, lazy
from .output_keeper import DIR, FILE
from .recipe_utils import (
    download_file,
    filter_file,
    filtered_lines,
    lines,
    output,
    run_prediction,
)
from .recipe_utils import (
    download_video_sample as download_video_sample,
)

SPLITS = ["train", "validation", "test"]

downloader = download_file(
    url="https://raw.githubusercontent.com/openimages/dataset/master/downloader.py",
    checksum="614947a4f8d8063292a5e5583c30b29e8b636de283949014b7a13ea5535cdb0b",
)

class_desc = download_file(
    url="https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv",
    checksum="1839e0e7e84130ae281f7f67413768601b031581c0c42e7fc17527b8e2a99aa9",
)

original_boxes = lazy(dict[str, Path])
original_boxes["train"] = download_file(
    url="https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv",
    checksum="dfc9637907a6b105f87e435bac91a5ee9b29af3ff8391168f86c1d63879786b6",
)
original_boxes["validation"] = download_file(
    url="https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv",
    checksum="d8bbd59410af14835d7733165a7bb8a3f0213981b22dd5077b0b9f7878991ff2",
)
original_boxes["test"] = download_file(
    url="https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv",
    checksum="fe22e579b9453875601576859d14ef3304058165de139c66b725e599202bd7b1",
)


def get_labels_of_interest(class_names_file: Path):
    name_of_interest = re.compile(r"(egg)|(chicken)", re.IGNORECASE)
    labels: list[str] = []
    names: list[str] = []

    for line in filtered_lines(input=class_names_file, filter=name_of_interest.search):
        label, display = line.split(",")
        labels.append(label.strip())
        names.append(display.strip())

    return labels, names


labels_names = lazy(get_labels_of_interest, class_desc)
labels_of_interest = labels_names[0]
names_of_interest = labels_names[1]


box_filter = lazy(
    re.compile,
    lazy(
        "|".join,
        lazy(map, lambda x: f"({x})", labels_of_interest + ["^ImageID,"]),
    ),
)

filtered_boxes = lazy(dict[str, Path])
for split in SPLITS:
    filtered_boxes[split] = filter_file(input=original_boxes[split], filter=box_filter)


@output.keep
def prepare_image_ids(boxes: dict[str, Path], output: Path = FILE, _v: int = 1):
    filtered = chain(
        *map(
            lambda split: map(
                lambda x: split + "/" + x.split(",")[0] + "\n",
                dropwhile(
                    lambda x: x.startswith("ImageID,"),
                    lines(input=boxes[split]),
                ),
            ),
            SPLITS,
        )
    )

    added = set[str]()

    with output.open("w") as o:
        for line in filtered:
            if line not in added:
                _ = o.write(line)
                added.add(line)


filtered_ids = prepare_image_ids(filtered_boxes)


@output.keep
def download_images(*, output: Path = DIR, downloader: Path, ids: Path):
    _ = subprocess.run(
        [sys.executable, downloader, "--download_folder", output, ids], check=True
    )


filtered_images = download_images(
    downloader=downloader,
    ids=filtered_ids,
)


@output.keep
def prepare_data(
    *,
    filtered: dict[str, Path],
    filtered_images: Path,
    output: Path = DIR,
    label2id: dict[str, int],
    drop_groups: bool,
):
    splits = ["train", "validation", "test"]

    for split in splits:
        images = output / "images" / split
        images.mkdir(parents=True)
        labels = output / "labels" / split
        labels.mkdir(parents=True)

        with filtered[split].open() as f:
            for line in dropwhile(lambda x: x.startswith("ImageID,"), f):
                # ImageID,Source,LabelName,Confidence,XMin,XMax,YMin,YMax,IsOccluded,IsTruncated,IsGroupOf,IsDepiction,IsInside
                id, _, label, _, xmin, xmax, ymin, ymax, _, _, is_group_of, _, _, *_ = (
                    line.split(",")
                )
                if drop_groups and is_group_of == "1":
                    continue

                xmin = float(xmin)
                xmax = float(xmax)
                ymin = float(ymin)
                ymax = float(ymax)

                label_id = label2id[label]

                target = images / f"{id}.jpg"
                if not target.exists():
                    _ = shutil.copy(filtered_images / f"{id}.jpg", target)

                with (labels / f"{id}.txt").open("a") as f:
                    xcenter = (xmin + xmax) / 2
                    ycenter = (ymin + ymax) / 2
                    width = xmax - xmin
                    height = ymax - ymin
                    _ = f.write(f"{label_id} {xcenter} {ycenter} {width} {height}\n")


@output.keep
def mk_desc(
    *,
    output: Path = DIR,
    prepared: Path,
    class_names: list[str],
):
    with (output / "desc.yaml").open("w") as f:
        _ = f.write(f"path: {prepared}\n")
        _ = f.write(f"train: images/train\n")
        _ = f.write(f"val: images/validation\n")
        _ = f.write(f"test: images/test\n")
        _ = f.write(f"names:\n")
        for idx, name in enumerate(class_names):
            _ = f.write(f"    {idx}: {name}\n")


label_map: dict[str, int] = lazy(
    lambda x: {label: idx for idx, label in enumerate(x)}, labels_of_interest
)


prepared_data_with_groups = prepare_data(
    filtered=filtered_boxes,
    filtered_images=filtered_images,
    label2id=label_map,
    drop_groups=False,
)

prepared_with_groups = mk_desc(
    prepared=prepared_data_with_groups,
    class_names=names_of_interest,
)

prepared_data_without_groups = prepare_data(
    filtered=filtered_boxes,
    filtered_images=filtered_images,
    label2id=label_map,
    drop_groups=True,
)

prepared_without_groups = mk_desc(
    prepared=prepared_data_without_groups,
    class_names=names_of_interest,
)


@output.keep
def train_(*, desc: Path, output: Path = DIR, epochs: int):
    from ultralytics import YOLO

    model = YOLO("yolo11n.pt")
    _results = model.train(data=desc, epochs=epochs, imgsz=640, project=output)


def train(*, groups: bool = True, epochs: int):
    if groups:
        desc = prepared_with_groups
    else:
        desc = prepared_without_groups
    return train_(desc=desc / "desc.yaml", epochs=epochs)


def test(*, groups: bool = True, epochs: int):
    model = train(groups=groups, epochs=epochs)
    data = prepared_data_with_groups
    return run_prediction(model / "train/weights/best.pt", images=data / "images/test")


def make(expr: str, dry: bool = False, trace: bool = False):
    no_compute = output.no_compute
    trace_hashing = output.trace_hashing
    try:
        output.no_compute = dry
        output.trace_hashing = trace

        res = force(eval(expr))
        return res
    finally:
        output.no_compute = no_compute
        output.trace_hashing = trace_hashing


def main():
    from argparse import ArgumentParser

    parser = ArgumentParser()
    _ = parser.add_argument("--dry", action="store_true")
    _ = parser.add_argument("--trace", action="store_true")
    _ = parser.add_argument("expr")
    args = parser.parse_args()

    return make(expr=args.expr, dry=args.dry, trace=args.trace)


if __name__ == "__main__":
    main()
