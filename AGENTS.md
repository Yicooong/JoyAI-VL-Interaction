# Repository Guidelines

## Project Structure & Module Organization

Core runtime components live under `services/`: `webui/` serves the browser UI and video transport, `webinfer/` connects to the VLM, and `asr/`, `tts/`, and `background-agent/` provide optional adapters. Cross-service launchers are in `services/scripts/`; component-specific launchers belong in each service's `scripts/` directory. Installation, dependency constraints, model downloads, and environment verification are under `install/`. Architecture and deployment documentation lives in `doc/`, static project artwork in `img/`, and concurrency tooling in `benchmarks/video_concurrency/`.


## Coding Style & Naming Conventions

Use four-space indentation and Python type hints where they clarify service boundaries. Follow `snake_case` for functions and modules, `PascalCase` for classes, and `UPPER_SNAKE_CASE` for constants. WebUI Python uses Black and Ruff with a 100-character line length; run `black services/` and `ruff check services/` before submitting broad Python changes. Keep shell scripts POSIX-friendly where practical and preserve the existing SPDX license headers. Update both English and `.zh-CN.md` documentation when changing user-facing behavior.

## Testing Guidelines

Tests use pytest and pytest-asyncio. Name files `test_*.py`, classes `Test*`, and functions `test_*`. Add focused regression coverage for session cleanup, concurrent video streams, transport changes, and adapter behavior. Hardware/model-dependent checks should remain in `install/tests/` or benchmarks and clearly document GPU and model prerequisites.

## Commit & Pull Request Guidelines

Recent history uses short, imperative summaries such as `增加多路视频` and `add benchmarks scripts`. Keep each commit focused and state the affected subsystem. Pull requests should explain the problem and solution, list verification commands, link relevant issues, and include screenshots or recordings for WebUI changes. Call out configuration, model-download, port, or CUDA compatibility impacts explicitly.

## Security & Configuration Tips

Never commit API keys, credentials, model weights, generated certificates, or local environment files. Prefer documented environment variables such as `WEBRTC_TRANSPORT`; keep defaults compatible with the installation constraints in `install/constraints.txt`.
