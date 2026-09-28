# Marks `app/` as a Python package so `from app.xxx import yyy` works
# both when run as `uvicorn app.main:app` and when imported directly
# by the test suite / batch scripts.
