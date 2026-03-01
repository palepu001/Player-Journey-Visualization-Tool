from __future__ import annotations

from pathlib import Path

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

DATA_PATH = PROJECT_ROOT / "player_data"
MINIMAP_PATH = PROJECT_ROOT / "minimaps"

# -----------------------------------------------------------------------------
# Core UI constants
# -----------------------------------------------------------------------------
IMAGE_SIZE = 1024
DATE_PLACEHOLDER = "-- Select Date --"
MAP_PLACEHOLDER = "-- Select Map --"

# -----------------------------------------------------------------------------
# Map calibration
# These are the values from the working game.py
# -----------------------------------------------------------------------------
MAP_CONFIG = {
    "AmbroseValley": {
        "label": "Ambrose Valley",
        "image": "AmbroseValley_Minimap.png",
        "scale": 900.0,
        "origin_x": -370.0,
        "origin_z": -473.0,
    },
    "GrandRift": {
        "label": "Grand Rift",
        "image": "GrandRift_Minimap.png",
        "scale": 581.0,
        "origin_x": -290.0,
        "origin_z": -290.0,
    },
    "Lockdown": {
        "label": "Lockdown",
        "image": "Lockdown_Minimap.jpg",
        "scale": 1000.0,
        "origin_x": -500.0,
        "origin_z": -500.0,
    },
}

# Optional alias map for older code paths
MINIMAP_FILES = {
    "AmbroseValley": "AmbroseValley_Minimap.png",
    "GrandRift": "GrandRift_Minimap.png",
    "Lockdown": "Lockdown_Minimap.jpg",
}

# -----------------------------------------------------------------------------
# Event groups
# These must match game.py, or the paths and counts will differ
# -----------------------------------------------------------------------------
MOVEMENT_EVENTS = {"position", "botposition"}
KILL_EVENTS = {"kill", "botkill"}
DEATH_EVENTS = {"killed", "botkilled", "killedbystorm"}
LOOT_EVENTS = {"loot"}

# -----------------------------------------------------------------------------
# Plot styling
# -----------------------------------------------------------------------------
PLAYER_TYPE_COLORS = {
    "Human": "#32CD32",
    "Bot": "#1E90FF",
}

MATCH_COLOR_PALETTE = [
    "#32CD32",
    "#1E90FF",
    "#FF6347",
    "#FFD700",
    "#BA55D3",
    "#FF69B4",
    "#00CED1",
    "#FFA500",
    "#7B68EE",
    "#20B2AA",
    "#DC143C",
    "#8A2BE2",
]

EVENT_MARKER_STYLE_RAW = {
    "kill": {"symbol": "x", "color": "#FF4D4F", "size": 10},
    "botkill": {"symbol": "x", "color": "#FF8C00", "size": 10},
    "killed": {"symbol": "diamond", "color": "#FFD700", "size": 10},
    "botkilled": {"symbol": "diamond", "color": "#00BFFF", "size": 10},
    "killedbystorm": {"symbol": "diamond", "color": "#C0C0C0", "size": 10},
    "loot": {"symbol": "square", "color": "#BA55D3", "size": 9},
}

# -----------------------------------------------------------------------------
# Focused event mapping
# These are the exact labels used in the working game.py
# -----------------------------------------------------------------------------
EVENT_LABEL_TO_RAW = {
    "Human killed human": "kill",
    "Human killed by human": "killed",
    "Human killed bot": "botkill",
    "Human killed by bot": "botkilled",
    "Killed by storm": "killedbystorm",
    "Loot": "loot",
}

FOCUS_EVENT_OPTIONS = [
    "All Events",
    "Human killed human",
    "Human killed by human",
    "Human killed bot",
    "Human killed by bot",
    "Killed by storm",
    "Loot",
]