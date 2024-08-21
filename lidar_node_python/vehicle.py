from vehicle_movement import Movement
from lidar_sensor import Lidar


class Vehicle:
    def __init__(self) -> None:

        print(f'[DEBUG] STARTING LIDAR')
        self.lidar = Lidar()
        self.lidar.start_lidar()
        print(f'[DEBUG] LIDAR STARTED')

        self.vehicle_movement = Movement(self.lidar)



        
    