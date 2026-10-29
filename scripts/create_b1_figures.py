#!/usr/bin/env python3
"""Create deterministic B1 figures using valid rows from an image manifest."""

from __future__ import annotations

import argparse
import csv
import math
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


COLORS = [(78, 121, 167), (242, 142, 43), (225, 87, 89), (118, 183, 178), (89, 161, 79), (237, 201, 72)]
BACKGROUND = (255, 255, 255)
TEXT = (31, 41, 55)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--clean-manifest", type=Path, help="Optional clean_manifest.csv for before/after chart.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples-per-class", type=int, default=4)
    return parser.parse_args()


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    name = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    try:
        return ImageFont.truetype(name, size=size)
    except OSError:
        return ImageFont.load_default()


def load_valid_rows(path: Path) -> dict[str, list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["is_valid"] == "True" and row["class_name"]:
            grouped[row["class_name"]].append(row)
    return {name: sorted(items, key=lambda item: item["relative_path"]) for name, items in sorted(grouped.items())}


def centered_text(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str, text_font: ImageFont.ImageFont, fill: tuple[int, int, int] = TEXT) -> None:
    left, top, right, bottom = box
    bbox = draw.textbbox((0, 0), text, font=text_font)
    x = left + (right - left - (bbox[2] - bbox[0])) / 2
    y = top + (bottom - top - (bbox[3] - bbox[1])) / 2
    draw.text((x, y), text, font=text_font, fill=fill)


def draw_class_distribution(grouped: dict[str, list[dict[str, str]]], output: Path) -> None:
    width, height = 1400, 900
    image = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(image)
    title_font, label_font, value_font = font(38, True), font(22), font(20, True)
    draw.text((70, 36), "SOHAS Classification — Class distribution", font=title_font, fill=TEXT)
    plot = (130, 145, 1340, 735)
    left, top, right, bottom = plot
    counts = [len(grouped[name]) for name in grouped]
    max_count = max(counts)
    for tick in range(6):
        value = max_count * tick / 5
        y = bottom - (bottom - top) * tick / 5
        draw.line((left, y, right, y), fill=(220, 224, 230), width=1)
        label = f"{value:,.0f}"
        bbox = draw.textbbox((0, 0), label, font=label_font)
        draw.text((left - 16 - (bbox[2] - bbox[0]), y - 12), label, font=label_font, fill=TEXT)
    draw.line((left, top, left, bottom), fill=TEXT, width=2)
    draw.line((left, bottom, right, bottom), fill=TEXT, width=2)
    names = list(grouped)
    slot = (right - left) / len(names)
    bar_width = slot * 0.58
    for index, name in enumerate(names):
        count = len(grouped[name])
        x_center = left + slot * (index + 0.5)
        bar_top = bottom - (count / max_count) * (bottom - top)
        draw.rounded_rectangle((x_center - bar_width / 2, bar_top, x_center + bar_width / 2, bottom), radius=8, fill=COLORS[index % len(COLORS)])
        centered_text(draw, (int(x_center - bar_width / 2), int(bar_top - 40), int(x_center + bar_width / 2), int(bar_top - 6)), f"{count:,}", value_font)
        centered_text(draw, (int(x_center - slot / 2), bottom + 18, int(x_center + slot / 2), bottom + 53), name, label_font)
    draw.text((left, 790), "Class label", font=label_font, fill=TEXT)
    image.save(output)


def draw_before_after_distribution(grouped: dict[str, list[dict[str, str]]], clean_manifest: Path, output: Path) -> None:
    """Draw valid-image counts before cleaning versus rows included after cleaning."""
    with clean_manifest.open(encoding="utf-8", newline="") as handle:
        clean_rows = list(csv.DictReader(handle))
    before = {name: len(rows) for name, rows in grouped.items()}
    after = {name: sum(row["class_name"] == name and row["is_in_clean_dataset"] == "True" for row in clean_rows) for name in before}
    width, height = 1400, 900
    image = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(image)
    title_font, label_font, value_font = font(35, True), font(20), font(16, True)
    draw.text((65, 35), "Class distribution — before and after exact-duplicate cleaning", font=title_font, fill=TEXT)
    left, top, right, bottom = 135, 155, 1340, 720
    max_count = max(before.values())
    for tick in range(6):
        value = max_count * tick / 5
        y = bottom - (bottom - top) * tick / 5
        draw.line((left, y, right, y), fill=(220, 224, 230), width=1)
        label = f"{value:,.0f}"
        box = draw.textbbox((0, 0), label, font=label_font)
        draw.text((left - 14 - (box[2] - box[0]), y - 11), label, font=label_font, fill=TEXT)
    draw.line((left, top, left, bottom), fill=TEXT, width=2)
    draw.line((left, bottom, right, bottom), fill=TEXT, width=2)
    names = list(before)
    slot = (right - left) / len(names)
    bar_width = slot * 0.28
    for idx, name in enumerate(names):
        center = left + slot * (idx + 0.5)
        for offset, count, color, label in ((-bar_width * 0.65, before[name], (78, 121, 167), "before"), (bar_width * 0.65, after[name], (89, 161, 79), "after")):
            x0, x1 = center + offset - bar_width / 2, center + offset + bar_width / 2
            y = bottom - count / max_count * (bottom - top)
            draw.rounded_rectangle((x0, y, x1, bottom), radius=6, fill=color)
            centered_text(draw, (int(x0 - 15), int(y - 29), int(x1 + 15), int(y - 3)), f"{count:,}", value_font)
        centered_text(draw, (int(center - slot / 2), bottom + 18, int(center + slot / 2), bottom + 46), name, label_font)
    draw.rectangle((left, 775, left + 20, 795), fill=(78, 121, 167))
    draw.text((left + 28, 772), "Before cleaning (valid images)", font=label_font, fill=TEXT)
    draw.rectangle((left + 335, 775, left + 355, 795), fill=(89, 161, 79))
    draw.text((left + 363, 772), "After cleaning (is_in_clean_dataset=True)", font=label_font, fill=TEXT)
    image.save(output)


def draw_samples(grouped: dict[str, list[dict[str, str]]], dataset_root: Path, output: Path, seed: int, samples_per_class: int) -> int:
    names = list(grouped)
    cell_w, cell_h, label_w = 285, 210, 155
    header_h, footer_h = 85, 18
    width = label_w + cell_w * samples_per_class + 35
    height = header_h + cell_h * len(names) + footer_h
    canvas = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 25), f"Random valid-image samples per class (seed={seed})", font=font(28, True), fill=TEXT)
    failures = 0
    for row_idx, name in enumerate(names):
        selected = random.Random(f"{seed}:{name}").sample(grouped[name], k=min(samples_per_class, len(grouped[name])))
        y0 = header_h + row_idx * cell_h
        draw.rectangle((0, y0, width, y0 + cell_h), fill=(248, 250, 252) if row_idx % 2 == 0 else BACKGROUND)
        centered_text(draw, (10, y0, label_w - 10, y0 + cell_h), name, font(20, True))
        for col_idx, row in enumerate(selected):
            x0 = label_w + col_idx * cell_w + 8
            try:
                with Image.open(dataset_root / row["relative_path"]) as source:
                    thumb = ImageOps.contain(source.convert("RGB"), (cell_w - 24, cell_h - 48))
                thumb_x = x0 + (cell_w - 8 - thumb.width) // 2
                thumb_y = y0 + 10 + (cell_h - 51 - thumb.height) // 2
                canvas.paste(thumb, (thumb_x, thumb_y))
                filename = Path(row["relative_path"]).name
                centered_text(draw, (x0, y0 + cell_h - 33, x0 + cell_w - 8, y0 + cell_h - 5), filename[:34], font(11))
            except OSError:
                failures += 1
                draw.rectangle((x0 + 8, y0 + 12, x0 + cell_w - 16, y0 + cell_h - 42), outline=(225, 87, 89), width=3)
                centered_text(draw, (x0, y0 + 70, x0 + cell_w - 8, y0 + 120), "Could not reopen", font(14), (225, 87, 89))
    canvas.save(output)
    return failures


def draw_resolution_scatter(grouped: dict[str, list[dict[str, str]]], output: Path) -> None:
    width, height = 1400, 900
    image = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(image)
    title_font, label_font = font(36, True), font(21)
    draw.text((85, 30), "Image resolution distribution (log scale)", font=title_font, fill=TEXT)
    left, top, right, bottom = 150, 125, 1325, 740
    draw.rectangle((left, top, right, bottom), outline=TEXT, width=2)
    all_rows = [row for rows in grouped.values() for row in rows]
    min_w, max_w = min(float(r["width"]) for r in all_rows), max(float(r["width"]) for r in all_rows)
    min_h, max_h = min(float(r["height"]) for r in all_rows), max(float(r["height"]) for r in all_rows)
    lx0, lx1, ly0, ly1 = map(math.log10, (min_w, max_w, min_h, max_h))
    for index, (name, rows) in enumerate(grouped.items()):
        color = COLORS[index % len(COLORS)]
        for row in rows:
            x = left + (math.log10(float(row["width"])) - lx0) / (lx1 - lx0) * (right - left)
            y = bottom - (math.log10(float(row["height"])) - ly0) / (ly1 - ly0) * (bottom - top)
            draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=color)
    for value in (10, 100, 1000):
        if min_w <= value <= max_w:
            x = left + (math.log10(value) - lx0) / (lx1 - lx0) * (right - left)
            draw.line((x, top, x, bottom), fill=(226, 232, 240), width=1)
            centered_text(draw, (int(x - 35), bottom + 10, int(x + 35), bottom + 38), str(value), label_font)
        if min_h <= value <= max_h:
            y = bottom - (math.log10(value) - ly0) / (ly1 - ly0) * (bottom - top)
            draw.line((left, y, right, y), fill=(226, 232, 240), width=1)
            bbox = draw.textbbox((0, 0), str(value), font=label_font)
            draw.text((left - 15 - (bbox[2] - bbox[0]), y - 12), str(value), font=label_font, fill=TEXT)
    centered_text(draw, (left, 772, right, 816), "Width (pixels, log scale)", label_font)
    y_label = Image.new("RGBA", (330, 35), (255, 255, 255, 0))
    y_label_draw = ImageDraw.Draw(y_label)
    y_label_draw.text((0, 0), "Height (pixels, log scale)", font=label_font, fill=TEXT)
    image.paste(y_label.rotate(90, expand=True), (38, 380), y_label.rotate(90, expand=True))
    legend_x, legend_y = 915, 785
    for index, name in enumerate(grouped):
        x = legend_x + (index % 3) * 145
        y = legend_y + (index // 3) * 38
        draw.ellipse((x, y + 4, x + 16, y + 20), fill=COLORS[index % len(COLORS)])
        draw.text((x + 23, y), name, font=font(16), fill=TEXT)
    image.save(output)


def main() -> int:
    args = parse_args()
    if args.samples_per_class < 1:
        raise SystemExit("--samples-per-class must be at least 1")
    grouped = load_valid_rows(args.manifest)
    if not grouped:
        raise SystemExit("No valid labeled images found in manifest")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    draw_class_distribution(grouped, args.output_dir / "class_distribution.png")
    if args.clean_manifest:
        draw_before_after_distribution(grouped, args.clean_manifest, args.output_dir / "class_distribution_before_after_cleaning.png")
    failures = draw_samples(grouped, args.dataset_root, args.output_dir / "sample_images.png", args.seed, args.samples_per_class)
    draw_resolution_scatter(grouped, args.output_dir / "image_resolution.png")
    print(f"Classes plotted: {len(grouped)}")
    print(f"Samples requested per class: {args.samples_per_class}; seed: {args.seed}")
    print(f"Sample files that could not be reopened: {failures}")
    filenames = ["class_distribution.png", "sample_images.png", "image_resolution.png"]
    if args.clean_manifest:
        filenames.append("class_distribution_before_after_cleaning.png")
    for filename in filenames:
        print(args.output_dir / filename)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
