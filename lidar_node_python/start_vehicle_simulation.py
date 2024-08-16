from lidar_sensor import Lidar
from steer_vehicle import Vehicle_movement

if __name__ == "__main__":
    print(f'[DEBUG] STARTING LIDAR')
    lidar_instance = Lidar()
    lidar_instance.start_lidar()
    print(f'[DEBUG] LIDAR STARTED')
    steering = Vehicle_movement(lidar_instance)

    
