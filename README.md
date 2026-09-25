# viam-TrashSort

An intelligent robotic arm that autonomously identifies, sorts, and removes trash from a tabletop using visual perception, contextual reasoning, and 3D object localization.
The system captures images of the workspace and reasons about whether an object should be classified as trash based on its appearance and context. For example, a sealed soda can is not trash, an open can may still contain an unfinished drink, while a crushed can is a strong indicator of waste. Once trash is identified, the robot localizes each object in 3D, picks it up, and deposits it into a designated bin.

Beyond autonomous trash removal, the system provides an interactive user interface for object manipulation and workspace organization.

- **Interactive Object Rearrangement:** When objects are too close together for reliable grasping, users can draw trajectories directly on the UI, guiding the robotic arm to push objects apart and create sufficient clearance.
- **Language-Guided Manipulation:** Users can interact with the robot through a natural-language chat interface to specify which objects to pick up.
- **Waypoint-Based Placement:** Once the trash has been cleared, users can reorganize their workspace by selecting target placement locations through the UI, allowing the robot to pick and place objects as desired.
The project combines visual reasoning, 3D perception, robotic manipulation, and human-in-the-loop control to transform a cluttered tabletop into an organized workspace.

Built on a [Viam](https://www.viam.com/) machine — an xArm with a wrist-mounted
depth camera and a gripper.

## How it works

The system separates 3D object localization from semantic understanding, allowing each component to focus on what it does best.

- **Where — 3D Perception:** The robot's obstacles-pointcloud vision service processes the captured point cloud, removes the table plane, and clusters the remaining points into individual objects. Each cluster provides the spatial information needed for robotic manipulation.
  
- **What — Visual Reasoning:** Claude receives the original scene image alongside an annotated version containing numbered bounding boxes corresponding to the detected clusters. It identifies each object and determines whether it should be classified as trash based on its appearance and context. By associating semantic labels with numbered clusters, the system avoids relying on the approximate spatial coordinates produced by a vision-language model.

**Coordinate Frames**
All object positions are expressed in the world coordinate frame. Since the camera is mounted on the robotic arm, its position and orientation change as the arm moves. Transforming detected object positions from the camera frame into the world frame provides a consistent spatial reference for object localization, grasp planning, and manipulation.

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
