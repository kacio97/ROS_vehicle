import time
from gz.msgs10.twist_pb2 import *
from gz.msgs10.stringmsg_pb2 import *
from gz.msgs10.laserscan_pb2 import *
from gz.msgs10.vector3d_pb2 import *
from gz.transport13 import *

class Vehicle_movement:
    def __init__(self) -> None:
        self.position_x = 0.0
        self.position_z = 0.0

    

    