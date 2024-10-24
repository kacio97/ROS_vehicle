from time import sleep
from gz.msgs10.twist_pb2 import *
from gz.msgs10.stringmsg_pb2 import *
from gz.msgs10.laserscan_pb2 import *
from gz.msgs10.vector3d_pb2 import *
from gz.transport13 import *

from pynput import keyboard
from voice_commands import Voice_movement
import threading


class Movement:
    def __init__(self,lidar) -> None:
        print(f'[DEBUG] INIT STEERING OF VEHICLE')
        self.node_instance = Node()
        self.lidar = lidar

        self.voice_command = Voice_movement()
        self.cmd_thread = threading.Thread(target=self.voice_command.listen_and_predict_command)
        self.cmd_thread.start()

        self.cmd_move = threading.Thread(target=self._command_listener)
        self.cmd_move.start()
        try:
            print(f'[DEBUG] STEERING OF VEHICLE BY KEYBOARD INITIALIZED')
            with keyboard.Listener(on_press=self._on_press, on_release=self._on_release) as listener: listener.join()       
        except:
            print(f'[DEBUG] INITIALIZATION STEERING OF VEHICLE BY KEYBOARD FAILED')
   


    def _command_listener(self):
        current_direction = 'stop'
        while True:
            # print(f'[DEBUG] voice command result {self.voice_command.current_direction}')
            if current_direction != self.voice_command.current_direction:
                current_direction = self.voice_command.current_direction
                print(f'[DEBUG] Current direction {current_direction}')
                self.move_vehicle(str(current_direction))
            # else:
            #     self.move_vehicle(str(current_direction))
            sleep(0.2)
       
    def _on_press(self, key):
        try:
            if key == keyboard.Key.up:
                print(f"[DEBUG] {str(key)}")
                self.move_vehicle('forward')
            elif key == keyboard.Key.down:
                print(f"[DEBUG] {str(key)}")
                self.move_vehicle('backward')
            elif key == keyboard.Key.left:
                print(f"[DEBUG] {str(key)}")
                self.move_vehicle('left')
            elif key == keyboard.Key.right:
                print(f"[DEBUG] {str(key)}")
                self.move_vehicle('right')
            elif key == keyboard.Key.ctrl_r:
                print(f"[DEBUG] {str(key)}")
                self.move_vehicle('stop')
        except AttributeError:
            pass

    def _on_release(self, key):
        if key == keyboard.Key.esc:
            return False

    def _check_if_region_is_safe(self, direction):
        # PL
        # Metoda sprawdzajaca czy dana strona pojazdu jest bezpieczna od przeszkod
        # dodatkowa forma sprawdzenia przy wysylaniu wiadomosci do pojazdu kiedy ma sie poruszac
        # ENG
        # Method to check if a given side of the vehicle is safe from obstacles
        # An additional form of verification when sending a message to the vehicle for movement
        if direction == 'forward':
            return self.lidar.get_is_safe_front_side_of_vehicle()
        elif direction == 'backward':
            return self.lidar.get_is_safe_rear_side_of_vehicle()
        elif direction == 'left':
            return self.lidar.get_is_safe_left_side_of_vehicle()
        elif direction == 'right':
            return self.lidar.get_is_safe_right_side_of_vehicle()
        

    def move_vehicle(self, direction):
        # PL
        # Poruszanie sie pojazdem ale pod warunkiem ze pojazdowi nie zagraza kolizja
        # ENG
        # Moves vehicle if zone is safe 
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
        elif direction == 'stop':
            message.angular.z = 0.0
            message.linear.x = 0.0
        else:
            print(f'[DEBUG] UNKNOWN MESSAGE: {direction}')
    
        if publisher.publish(message):
            print(f'[INFO] Successfully send {message}')
        else:
            print(f'[ERROR] Failed to send {message} message')
