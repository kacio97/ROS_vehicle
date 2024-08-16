import time
import threading

from gz.msgs10.twist_pb2 import *
from gz.msgs10.stringmsg_pb2 import *
from gz.msgs10.laserscan_pb2 import *
from gz.msgs10.vector3d_pb2 import *
from gz.transport13 import *


class Lidar:
    def __init__(self) -> None:
        self.topic = '/lidar'
        self.lidar_node = Node()
        self.data_listener = None
        self.msgs = None

        # Storage for particles, each side of vehicle 640 / 4 = 160 particles per side
        self.left_side = None
        self.right_side = None
        self.front_side = None
        self.rear_side = None

        # Flags uses to check if vehicle is on safe distance from objects
        self.is_on_safe_position_left = False
        self.is_on_safe_position_right = False
        self.is_on_safe_position_front = False
        self.is_on_safe_position_rear = False



    def start_lidar(self):
        if self.lidar_node.subscribe(LaserScan, self.topic, self.collect_lidar_data):
            print(f'[INFO] Subcribed {self.topic} topic')
        else:
            raise f'[ERROR] Error during topic {self.topic} subscription'
        
        self.data_listener = threading.Thread(target=self.listen_for_data)
        # self.data_listener.daemon = True # Zapewni to zamknięcie wątku po zakończeniu programu głównego.
        self.data_listener.start()
        
    
    def listen_for_data(self):
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        print("[INFO] Listening for LIDAR data finished")

    
    def collect_lidar_data(self, msg: LaserScan):
        # ============================
        # DEBUG PART
        # ============================
        # f = open('res2.txt', 'w')
        # for i in range(len(msg.ranges)):
        #     f.write(f'| {i} |{msg.ranges[i]}\n')
        #     print(f'| {i} |{msg.ranges[i]}')
        # f.close()
        # ============================
        
        # self.particles_analyzer = threading.Thread(target=self.analyze_particles(msg.ranges))
        # # self.particles_analyzer.daemon = True # Zapewni to zamknięcie wątku po zakończeniu programu głównego.
        # self.particles_analyzer.start()

        self.analyze_particles(msg.ranges)

        # for _msgs in msg.ranges:
            # print(f'Range - {_msgs}\n')
        

    def analyze_particles(self, ranges):
        # Moj pojazd posiada 640 wiazek lasera do analizy tak wiec
        # trzeba podzielic 640 / 4 strony pojazdu co daje nam okolo 160 czastek na strone

        # Na podstawie zebranych probek z Gazebo
        # PRÓBKI Z LIDAR
        # 0-80 tył lewy 561 - 640 tył prawy
        # 81-240 prawy bok
        # 241 - 400 środek
        # 401-560 lewy bok
        
        self.right_side = ranges[80:240]
        self.left_side = ranges[400:560]
        self.front_side = ranges[240:400]
        self.rear_side = ranges[-80:] + ranges[:80]

        
        self.is_on_safe_position_left = self.analyze_side_of_vehicle(self.left_side, 'left')
        print(f'[DEBUG] left side is safe: {self.is_on_safe_position_left}')

        self.is_on_safe_position_right = self.analyze_side_of_vehicle(self.right_side, 'right')
        print(f'[DEBUG] right side is safe: {self.is_on_safe_position_right}')

        self.is_on_safe_position_front = self.analyze_side_of_vehicle(self.front_side, 'front')
        print(f'[DEBUG] front side is safe: {self.is_on_safe_position_front}')

        self.is_on_safe_position_rear = self.analyze_side_of_vehicle(self.rear_side, 'rear')
        print(f'[DEBUG] rear side is safe: {self.is_on_safe_position_rear}')

        print(f'[DEBUG] PARTICLES ANALYZED')  

    
    def analyze_side_of_vehicle(self, particles, side):
        for particle in particles:
            if particle == 'inf':
                continue
            if side == 'rear' or side == 'front' and float(particle) <= 1.5:
                print(f'[DEBUG] Distance on {side} is closer than 0.5')
                return False         
            elif float(particle) <= 1:
                print(f'[DEBUG] Distance on {side} is closer than 0.5')
                return False                  
        print(f'[DEBUG] Distance on {side} is bigger than 0.5')        
        return True


def stop_vehicle():
    print(f'[DEBUG] - IN STOP VEHICLE INITIALIZE')
    node = Node()
    twist_message = Twist()
    topic = '/cmd_vel'
    node_publisher = node.advertise(topic, Twist)

    twist_message.angular.x = 0.0
    twist_message.angular.z = 0.0
    twist_message.linear.x = 0.0
    twist_message.linear.z = 0.0
   
    twist_message
    print(f'[DEBUG] - IN STOP VEHICLE PUBLISH')

    if node_publisher.publish(twist_message):
        print(f'[INFO] Successfully send {twist_message}')
    else:
        print(f'[ERROR] Failed to send {twist_message} message')
