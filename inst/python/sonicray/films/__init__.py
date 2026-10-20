# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Matthew Ross. Written with AI assistance (Claude Sonnet 5.5); see AI_ATTRIBUTION.md.
"""Stories. A story is any class with  __init__(comp), frame_spec(t) -> FrameSpec, .total (seconds), .keyframes [(t, label)]."""
from .network import NetworkStory
from .poudre import PoudreStory

STORIES = {"poudre": PoudreStory, "network": NetworkStory}
