"""Record a GitHub star count and render the weekly observations for the README."""

import argparse
import csv
from datetime import date, datetime, timezone
from pathlib import Path


ASSETS = Path(__file__).resolve().parents[1] / "assets"
DATA = ASSETS / "star-history.csv"
CHART = ASSETS / "star-history.svg"


def read_observations(path):
    with path.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != ["date", "stars"]:
            raise ValueError("Expected CSV columns date,stars")
        rows = [(date.fromisoformat(row["date"]), int(row["stars"])) for row in reader]
    if any(count < 0 for _, count in rows):
        raise ValueError("Star counts must be nonnegative")
    if any(a[0] >= b[0] for a, b in zip(rows, rows[1:])):
        raise ValueError("Observations must be in ascending order with one per UTC date")
    return rows


def render(rows):
    if not rows:
        raise ValueError("The chart needs at least one observation")
    first, last = rows[0][0], rows[-1][0]
    days = max((last - first).days, 1)
    peak = max(1, max(count for _, count in rows))
    left, right, top, bottom = 64, 684, 78, 180
    points = [(left + (day - first).days / days * (right - left),
               bottom - count / peak * (bottom - top)) for day, count in rows]
    latest_x, latest_y = points[-1]
    path = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    count = rows[-1][1]
    label = "star" if count == 1 else "stars"
    end_label = (f'\n      <text x="{right}" y="207" text-anchor="end">{last}</text>'
                 if first != last else "")
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 720 226" role="img" aria-labelledby="title desc">
  <title id="title">GitHub star history</title>
  <desc id="desc">{len(rows)} observations from {first} to {last}. Latest count: {count} {label}.</desc>
  <rect width="720" height="226" rx="12" fill="#f6f8fa"/>
  <g font-family="system-ui, sans-serif" fill="#24292f">
    <text x="32" y="42" font-size="22" font-weight="600">GitHub stars</text>
    <text x="684" y="42" font-size="18" text-anchor="end">{count} {label}</text>
    <path d="M {left} {top} H {right} M {left} {bottom} H {right}" fill="none" stroke="#afb8c1"/>
    <g font-size="13" fill="#424a53">
      <text x="51" y="{top + 4}" text-anchor="end">{peak}</text>
      <text x="51" y="{bottom + 4}" text-anchor="end">0</text>
      <text x="{left}" y="207">{first}</text>{end_label}
    </g>
    <polyline points="{path}" fill="none" stroke="#175ba7" stroke-width="3" stroke-linejoin="round"/>
    <circle cx="{latest_x:.1f}" cy="{latest_y:.1f}" r="4.5" fill="#175ba7"/>
  </g>
</svg>
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--record", type=int, metavar="STARS", help="Record today's UTC count")
    mode.add_argument("--check", action="store_true", help="Check the CSV and rendered SVG")
    args = parser.parse_args()
    rows = read_observations(DATA)
    if args.check:
        if CHART.read_text(encoding="utf-8") != render(rows):
            parser.exit(1, "Star history SVG differs from the observations.\n")
        return

    today = datetime.now(timezone.utc).date()
    if args.record < 0 or (rows and rows[-1][0] > today):
        parser.error("Expected a nonnegative count and no observations after today")
    if rows and rows[-1][0] == today:
        rows.pop()
    rows.append((today, args.record))
    svg = render(rows)
    with DATA.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(["date", "stars"])
        writer.writerows(rows)
    CHART.write_text(svg, encoding="utf-8")


if __name__ == "__main__":
    main()
