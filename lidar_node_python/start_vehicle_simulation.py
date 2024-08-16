

from lidar_sensor import Lidar


if __name__ == "__main__":
    print(f'[DEBUG] STARTING LIDAR')
    lidar_instance = Lidar()
    lidar_instance.start_lidar()
    print(f'[DEBUG] LIDAR STARTED')
