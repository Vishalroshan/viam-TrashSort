# viam-TrashSort

A robot arm that clears a table: it photographs what's on the surface, asks
Claude which items are trash, locates each one in 3D, and picks the trash up.

Built on a [Viam](https://www.viam.com/) machine — an xArm with a wrist-mounted
depth camera and a gripper.

## How it works

Finding *where* things are and deciding *what* they are come from different
places, each doing what it is good at:

- **Where** — the machine's `obstacles-pointcloud` vision service removes the
  table plane and clusters what's left into one point set per object.
- **What** — Claude sees the photo plus a copy with a numbered box drawn around
  each cluster, then names and classifies each number. It never has to produce
  coordinates, which it does only approximately.

Positions are reported in the `world` frame. The camera rides on the arm, so
camera-frame coordinates shift whenever the arm moves and aren't a stable
description of where anything is.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env     # then fill it in
```

`.env` needs Viam machine credentials and an Anthropic API key; see
[.env.example](.env.example) for each variable and where to get it. It is
gitignored — keep credentials out of the source files.

## Scripts

| Script | What it does |
| --- | --- |
| `table_scan.py` | Scan the table and write `table_objects.json`: every object, its trash classification, and its world-frame position. The foundation the others build on. |
| `table_cleanup.py` | Scan, then pick up everything classified as trash and drop it in a fixed location. Supports `--dry-run`. |
| `cleanup_ui.py` | PyQt5 chat window — scan, then say what to remove in plain language ("get rid of all the trash except the paper"). |
| `table_orbit_scan.py` | Orbit the arm around one object, fusing depth from several viewpoints into a dense point cloud (`orbit_cloud.ply`). |
| `table_reconstruct.py` | Orbit scan + Poisson surface reconstruction + render, as a library for `cleanup_ui.py`. |
| `viam_move_to_centroid.py` | Early standalone pick-and-place against a simple green-block segmenter. Kept for reference. |

Each file's module docstring documents its own flags and outputs in detail.

## Running

```bash
.venv/bin/python table_scan.py                  # scan only
.venv/bin/python table_scan.py --image test.jpeg  # classify a photo, no 3D
.venv/bin/python table_cleanup.py --dry-run     # scan, print the plan, move nothing
.venv/bin/python table_cleanup.py               # scan and clear the table
.venv/bin/python cleanup_ui.py                  # chat UI
```

`table_cleanup.py` has no confirmation prompt — it scans and immediately acts on
whatever it classified as trash. Ctrl-C stops the arm where it is.

## Notes

Objects whose depth signal is too sparse to trust are skipped and reported
rather than blindly grasped. Clear plastic and shiny metal are the usual
offenders under IR depth sensing; the water bottle on this table is the running
example.

Scan outputs (`frame*.jpg`, `table_objects*.json`, `*.ply`) are generated
artifacts and are gitignored.
