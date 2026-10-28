"""Launch the local FastAPI server for the browser-based interface."""

import uvicorn


def main() -> None:
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
