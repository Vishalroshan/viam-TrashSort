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

## Layout

```
trashsort/          the shared modules
  table_scan.py         scan and classify — the foundation the rest build on
  table_cleanup.py      pick up and bin whatever was classified as trash
  table_orbit_scan.py   orbit one object for a dense point cloud
  table_reconstruct.py  Poisson reconstruction + render (library for the UI)
  paths.py              where generated files go
cleanup_ui.py       PyQt5 chat window (entry point)
examples/           standalone reference scripts
output/             generated frames, scans, clouds and meshes (gitignored)
```

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env     # then fill it in
```

`.env` needs Viam machine credentials and an Anthropic API key; see
[.env.example](.env.example) for each variable and where to get it. It is
gitignored — keep credentials out of the source files.

## Running

Run from the repo root:

```bash
.venv/bin/python -m trashsort.table_scan                 # scan only
.venv/bin/python -m trashsort.table_scan --image photo.jpg  # classify a photo, no 3D
.venv/bin/python -m trashsort.table_cleanup --dry-run    # scan, print the plan, move nothing
.venv/bin/python -m trashsort.table_cleanup              # scan and clear the table
.venv/bin/python -m trashsort.table_orbit_scan           # orbit the last scan's first trash object
.venv/bin/python cleanup_ui.py                           # chat UI
```

`table_cleanup` has no confirmation prompt — it scans and immediately acts on
whatever it classified as trash. Ctrl-C stops the arm where it is.

Each module's docstring documents its own flags and outputs in detail.

## Output

Everything generated lands in `output/`, resolved relative to the repo rather
than your shell's working directory, so it goes to the same place wherever you
run from. `TRASHSORT_OUTPUT_DIR` redirects it; the `--out` and `--objects` flags
override individual paths.

`--image` takes any local photo. Sample images aren't versioned here.

## Notes

Objects whose depth signal is too sparse to trust are skipped and reported
rather than blindly grasped. Clear plastic and shiny metal are the usual
offenders under IR depth sensing; the water bottle on this table is the running
example.
