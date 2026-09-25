"""Tabletop tidying robot: scan a table, classify what is on it, clear the trash.

The modules here are meant to be run as scripts:

    python -m trashsort.table_scan
    python -m trashsort.table_cleanup --dry-run
    python -m trashsort.table_orbit_scan

table_reconstruct is a library used by cleanup_ui.py, not an entry point.
"""
