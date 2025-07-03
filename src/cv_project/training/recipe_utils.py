import re
from collections.abc import Callable, Iterator
from hashlib import sha256
from pathlib import Path
from typing import cast

import requests

from .output_keeper import DIR, FILE, Store
from .utils import eprint

DATA = Path("./data")
output = Store(DATA)


@output.keep
def download_file(*, url: str, output: Path = FILE, checksum: str):
    written = 0

    h = sha256()

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(output, "wb") as f:
            it = cast(Iterator[bytes], r.iter_content(chunk_size=8192))

            n_chunks = 0

            for chunk in it:
                n_chunks += 1
                written += f.write(chunk)
                h.update(chunk)
                if n_chunks % 100 == 0:
                    eprint(f"\r{output}: written {(written // 100000) / 10}MB", end="")
        eprint(f"\r{output}: written {(written // 100000) / 10}MB")

    digest = h.hexdigest()
    if digest != checksum:
        raise RuntimeError("checksum")


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
            if lines_read % 10000 == 0:
                eprint(f"\rfiltering {input}: {lines_read} -> {lines_written}", end="")
    eprint(f"\rfiltered {input}: {lines_read} -> {lines_written}")


@output.keep
def filter_file(*, input: Path, output: Path = FILE, filter: re.Pattern[str]):
    with output.open("w") as o:
        for line in filtered_lines(input=input, filter=filter.search):
            _ = o.write(line)


@output.keep
def run_prediction(model: Path, images: Path, output: Path = DIR):
    from pathlib import Path

    import cv2
    from ultralytics import YOLO
    from ultralytics.utils.plotting import Annotator

    m = YOLO(model)

    results = m.predict(source=images)

    for r in results:
        # https://stackoverflow.com/questions/75324341/yolov8-get-predicted-bounding-box#75332799
        annotator = Annotator(r.orig_img)
        boxes = r.boxes
        assert boxes is not None
        for box in boxes:
            b = box.xyxy[0]
            c = box.cls
            annotator.box_label(b, m.names[int(c)])
        img = annotator.result()

        relpath = Path(r.path).absolute().relative_to(images.absolute())
        assert relpath != Path()
        save_name = output / relpath
        save_name.parent.mkdir(parents=True, exist_ok=True)
        eprint(f"\rsaving {save_name}", end="")
        ok = cv2.imwrite(str(save_name), img)
        if not ok:
            eprint(f": failed?")
    eprint("\r")


@output.keep
def download_video_sample(url: str, output: Path = DIR):
    import yt_dlp

    # --- Formatting
    video_dl_options = {
        "format": "bestvideo[width<=640][vcodec^=avc][ext=mp4]/best[ext=mp4]/best",
        "outtmpl": str(output / "video.mp4"),
    }

    with yt_dlp.YoutubeDL(video_dl_options) as ydl:
        _ = ydl.download([url])
