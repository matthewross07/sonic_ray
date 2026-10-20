# AI attribution

**What this is.** `sonicray` was designed and written with AI assistance. The human author is Matthew Ross (OpenCurrent), who set the goals, supplied the data and film
designs, reviewed the results and made the decisions. The AI wrote most of the code and documentation under that direction.

| | |
|---|---|
| Model | Anthropic **Claude Sonnet 5.5** (`claude-sonnet-5-5`) |
| Tool | Claude Code (desktop app, Code tab), running locally with file, shell and browser tools |
| Effort setting | **"high"**, as stated by the author. (The model cannot verify its own effort setting from inside a session; treat this line as the author's statement.) |
| Period | October 2026 |
| Git | AI-written commits carry a `Co-Authored-By: Claude Sonnet 5.5` trailer |

**What was verified (and how).** Everything quoted as "measured" in this repository was run on one machine (Apple M4 Pro, macOS 15.7) during development:
layer timings from rendering the 162 layers; compositor timings from full film renders; GPU-vs-CPU image parity from `tools/compare_reference.py`; install tested into a fresh R library.
**What was not verified.** Other Macs, other macOS versions, a machine without the Python packages pre-installed, the Intel OIDN path on other chips, and long-term behaviour of the Metal driver.
Where a number is an estimate it says so. Please report problems as issues.

**Prior work.** The ideas follow Tyler Morgan-Wall's rayshader/rayrender (see ACKNOWLEDGEMENTS.md). No code from those projects is included.
