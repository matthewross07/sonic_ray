"""Stories. A story is any class with  __init__(comp), frame_spec(t) -> FrameSpec, .total (seconds), .keyframes [(t, label)]."""
from .network import NetworkStory
from .poudre import PoudreStory

STORIES = {"poudre": PoudreStory, "network": NetworkStory}
