import runpy

try:
    runpy.run_path(
        "/content/remote_storybook_accessories_v5_good_enough.py",
        run_name="__main__",
    )
except SystemExit as exc:
    if exc.code not in (None, 0):
        raise
