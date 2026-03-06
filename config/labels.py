"""
Label-to-ID and label-to-text mapping for the gesture dataset.
All IDs are 0-based (class index) for use in models and datasets.
"""

LABEL_TO_TEXT = {
    "D0X": "Non-gesture",
    "B0A": "Pointing with one finger",
    "B0B": "Pointing with two fingers",
    "G01": "Click with one finger",
    "G02": "Click with two fingers",
    "G03": "Throw up",
    "G04": "Throw down",
    "G05": "Throw left",
    "G06": "Throw right",
    "G07": "Open twice",
    "G08": "Double click with one finger",
    "G09": "Double click with two fingers",
    "G10": "Zoom in",
    "G11": "Zoom out",
}

# Ordered list of labels; index in this list is the class ID (0-based)
LABELS = [
    "D0X", "B0A", "B0B", "G01", "G02", "G03", "G04", "G05",
    "G06", "G07", "G08", "G09", "G10", "G11",
]

# 0-based class ID -> label code
ID_TO_LABEL = {i: lab for i, lab in enumerate(LABELS)}

# label code -> 0-based class ID (for model / dataset)
LABEL_TO_ID = {lab: i for i, lab in enumerate(LABELS)}

NUM_CLASSES = len(LABELS)
