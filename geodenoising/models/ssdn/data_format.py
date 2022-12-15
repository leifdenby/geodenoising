from enum import Enum, auto


class DataFormat:
    BHWC = "BHWC"
    BWHC = "BWHC"
    BCHW = "BCHW"
    BCWH = "BCWH"
    HWC = "HWC"
    WHC = "WHC"
    CHW = "CHW"
    CWH = "CWH"


class DataDim(Enum):
    BATCH = auto()
    CHANNEL = auto()
    WIDTH = auto()
    HEIGHT = auto()


DATA_FORMAT_DIM_INDEX = {}
