# Tools, Codex and authentication

## Inspected setup

Setup inspection found an empty repository on unborn `main`, with `origin` already set to the requested GitHub repository. No existing files or commits needed preservation. Accessible local AGENTS files from `auto_apply_job_bot` and `Projects/tradingBot` informed the concise communication, preservation, secret-exclusion, scoped verification and honest fixture/live-test conventions. No credentials, domain-specific instructions or unrelated project files were copied.

Installed Codex CLI: **0.159.3**. `codex --version` and its installed binary confirmed `.agents/skills` support. [Official skill documentation](https://developers.openai.com/codex/skills/) describes repository discovery in `.agents/skills` from the working directory through the Git root. This repository uses that supported location; it does not invent a skill-registration file. Invoke a skill by its name or let Codex discover it. Restart Codex if newly added skills do not appear.

`codex mcp list` was inspected. Existing enabled entries included OpenAI documentation and computer-use capabilities; no GitHub or geospatial MCP was configured. Unrelated integrations were left untouched. [Codex supports](https://developers.openai.com/codex/mcp/) project-scoped MCP settings in `.codex/config.toml` for trusted projects, but this project needs no new MCP configuration.

## GitHub

Git and `gh` provide the needed repository, commit, push and CI operations using existing keyring authentication. `gh auth status` succeeded outside the local sandbox. The sandbox initially reported authentication failure because it could not use the same keyring access; that was not an expired account token.

[GitHub's official MCP server](https://github.com/github/github-mcp-server) was reviewed. It adds no necessary capability for this foundation over the authenticated Git/gh workflow, so no redundant MCP, token or wrapper was installed. If later project work needs agent-accessible issue/PR tools, evaluate the official maintained server then. Do not commit an authentication token or assume an untested connection works.

For a new teammate:

```sh
gh auth login --hostname github.com
gh auth status
```

## Data and Earth Engine

EuroSAT uses the publisher-linked Zenodo archive through normal HTTPS/Python APIs, with publisher MD5 verification and extracted-file SHA-256 manifests. Scratch training needs no account. Pretrained weights use torchvision's official loader and local cache. The dataset download command has an offline `--archive PATH` option for an already obtained official ZIP.

Earth Engine is optional and uses its maintained Python API. Install `uv sync --locked --extra nepal`, register an Earth Engine-enabled Cloud project, then run `uv run earthengine authenticate`. Put the project ID in a private copy of `configs/nepal.yaml`; `.env.example` also documents `EE_PROJECT`. User OAuth credentials live outside the repo. No Earth Engine account/project was configured or live export verified during setup. Do not store service-account JSON in Git.

## Environment notes

`uv.lock` records exact transitive versions; local validation used uv 0.9.26, Python 3.12.12 and Apple silicon CPU. Linux defaults to official CPU Torch wheels for lightweight CI. Optional GPU environments should retain the Torch/torchvision pair, record the chosen CUDA wheel index and save their resolved dependency list with results.

The managed macOS sandbox blocks some keyring, network and uv system-configuration access. Those installation/authentication operations required normal user-approved execution outside the sandbox. Matplotlib can build a slow first-use font cache; the README cache variables keep it project-local. These are environment restrictions, not model or dataset failures.
