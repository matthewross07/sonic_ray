# AI attribution

**What this is.** I built `sonicray` with AI. I set the goals, supplied the data and the film designs, reviewed the results and made the decisions. Claude wrote most of the code and the documentation under my direction, and ran the measurements.

| | |
|---|---|
| Model | Anthropic **Claude Sonnet 5.5** (`claude-sonnet-5-5`) |
| Tool | Claude Code (desktop app, Code tab), running locally with file, shell and browser tools |
| Effort setting | **high** (I confirm this; the model cannot read its own effort setting from inside a session) |
| Period | October 2026 |
| Git | Commits that Claude wrote carry a `Co-Authored-By: Claude Sonnet 5.5` trailer |

**What we verified, and how.** Everything this repository calls "measured" ran on one machine (Apple M4 Pro, macOS 15.7) during development. Claude timed the 162 layer renders and the full film composites,
compared GPU frames with the CPU compositor using `tools/compare_reference.py`, and tested the install into a fresh R library.
**What we did not verify.** Other Macs, other macOS versions, and the long-term behaviour of the Metal driver.
Where a number is an estimate, the text says so. Please report problems as issues.

**Prior work.** I follow the ideas of Tyler Morgan-Wall's rayshader and rayrender (see ACKNOWLEDGEMENTS.md). This repository includes no code from those projects.
