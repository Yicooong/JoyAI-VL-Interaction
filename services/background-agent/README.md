# StreamingHarness Codex API

> 中文文档: [README.zh-CN.md](README.zh-CN.md)

Local FastAPI wrapper around the system `codex` CLI for background tasks.

## Run

The recommended path is to start it from the repository root with the component script:

```bash
./services/background-agent/scripts/run.sh
```

`run.sh` prefers the shared environment `services/.venv` created by the install script. If that environment does not exist, it falls back to `uv run` development mode:

```bash
cd services/background-agent
./scripts/run.sh
```

The WebUI background client uses `http://127.0.0.1:8079` by default. Override it with:

```bash
export BACKGROUND_AGENT_API_URL=http://127.0.0.1:8079
```

`run.sh` uses `<repo>/agent-workspace` as the default Codex workspace and creates it on startup. Override runtime paths with environment variables:

```bash
CODEX_HOME=/path/to/codex-home CODEX_API_WORKSPACE=/path/to/repo ./scripts/run.sh
```

## Behavior

- Uses `codex-home/config.toml` and `codex-home/auth.json` in the service directory by default.
- Runs `codex exec` with `--dangerously-bypass-approvals-and-sandbox`, `--search`, `--json`, and `--ephemeral`.
- Caps parallel subagents with `agents.max_threads`; default is `6`.
- Binds to localhost by default. Do not expose this service to untrusted networks.

## Health Check

```bash
curl http://127.0.0.1:8079/health
```
