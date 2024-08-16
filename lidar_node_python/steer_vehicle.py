import time
from gz.msgs10.twist_pb2 import *
from gz.msgs10.stringmsg_pb2 import *
from gz.msgs10.laserscan_pb2 import *
from gz.msgs10.vector3d_pb2 import *
from gz.transport13 import *

from pynput import keyboard

#                  #
# GLOBAL VARIABLES #
# INT32 VALUES     #
up_arrow = 16777235
down_arrow = 16777237
left_arrow = 16777234
right_arrow = 16777236


class Vehicle_movement:
    def __init__(self, lidar_instance) -> None:
        print(f'[DEBUG] INIT STEERING OF VEHICLE')
        self.node_instance = Node()
        self.lidar = lidar_instance

        try:
            print(f'[DEBUG] STEERING OF VEHICLE INITIALIZED')
            with keyboard.Listener(on_press=self._on_press, on_release=self._on_release) as listener: listener.join()       
        except:
            print(f'[DEBUG] INITIALIZATION STEERING OF VEHICLE FAILED')

    def _on_press(self, key):
        try:
            if key == keyboard.Key.up:
                print(f"{str(key)}")
                self.move_vehicle('forward', up_arrow)
            elif key == keyboard.Key.down:
                print(f"{str(key)}")
                self.move_vehicle('backward', down_arrow)
            elif key == keyboard.Key.left:
                print(f"{str(key)}")
                self.move_vehicle('left', left_arrow)
            elif key == keyboard.Key.right:
                print(f"{str(key)}")
                self.move_vehicle('right', right_arrow)
        except AttributeError:
            pass

    def _on_release(self, key):
        if key == keyboard.Key.esc:
        # Zatrzymaj nasłuchiwanie po naciśnięciu klawisza ESC
            return False

    def _check_if_region_is_safe(self, direction):
        if direction == 'forward':
            return self.lidar.get_is_safe_front_side_of_vehicle()
        elif direction == 'backward':
            return self.lidar.get_is_safe_rear_side_of_vehicle()
        elif direction == 'left':
            return self.lidar.get_is_safe_left_side_of_vehicle()
        elif direction == 'right':
            return self.lidar.get_is_safe_right_side_of_vehicle()
        

    def move_vehicle(self, direction, key_value):
        print(f'[DEBUG] - START TO MOVE VEHICLE {direction}')
                 
        message = Twist()
        topic = '/cmd_vel'
        publisher = self.node_instance.advertise(topic, Twist)

        if direction == 'forward' and self._check_if_region_is_safe(direction):

            message.angular.z = 0.0
            message.linear.x = 0.5
        elif direction == 'backward' and self._check_if_region_is_safe(direction):

            message.angular.z = 0.0
            message.linear.x = -0.5
        elif direction == 'left' and self._check_if_region_is_safe(direction):
       
            message.angular.z = 0.5
            message.linear.x = 0.0
        elif direction == 'right' and self._check_if_region_is_safe(direction):

            message.angular.z = -0.5
            message.linear.x = 0.0
        else:
            print(f'[DEBUG] UNKNOWN MESSAGE: {direction}')
    
        if publisher.publish(message):
            print(f'[INFO] Successfully send {message}')
        else:
            print(f'[ERROR] Failed to send {message} message')

    