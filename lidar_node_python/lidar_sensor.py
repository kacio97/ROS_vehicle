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

        self._blocked_front = False
        self._blocked_rear = False
        self._blocked_left = False
        self._blocked_right = False

        # Storage for particles, each side of vehicle 640 / 4 = 160 particles per side
        self.left_side = None
        self.right_side = None
        self.front_side = None
        self.rear_side = None

        # Flags uses to check if vehicle is on safe distance from objects
        self._is_safe_left_side_of_vehicle = False
        self._is_safe_right_side_of_vehicle = False
        self._is_safe_front_side_of_vehicle = False
        self._is_safe_rear_side_of_vehicle = False



    def start_lidar(self):
        # PL
        # Uruchamiamy instancje lidara ktory subskrybuje sie na zadany temat
        # wewnatrz przekazujemy mu 3 parametry w zasadzie najwazniejszy to 'LaserScan'
        # - LaserScan zawiera informacje o odległościach mierzonych przez skaner laserowy, 
        # kąty oraz inne parametry związane z detekcją obiektów.
        # - topic to temat jaki wpisany jest do modelu robota dla fragmentu z LIDAREM
        # - collect_lidar_data (callback function) to metoda która jest wywoływana po odebraniu wiadomości
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
    
    #         #
    # GETTERS #
    #         #

    def get_is_safe_left_side_of_vehicle(self):
        return self._is_safe_left_side_of_vehicle
    
    def get_is_safe_right_side_of_vehicle(self):
        return self._is_safe_right_side_of_vehicle

    def get_is_safe_front_side_of_vehicle(self):
        return self._is_safe_front_side_of_vehicle
    
    def get_is_safe_rear_side_of_vehicle(self):
        return self._is_safe_rear_side_of_vehicle
    
    def get_is_stopped(self):
        return self._is_stopped
    

    #         #
    # SETTERS #
    #         #

    def set_is_stopped(self, value):
        self._is_stopped = value
    
    def collect_lidar_data(self, msg: LaserScan):
        self.analyze_particles(msg.ranges)
        

    def analyze_particles(self, ranges):
        # PL
        # Moj pojazd posiada 640 wiazek lasera do analizy tak wiec
        # trzeba podzielic 640 / 4 strony pojazdu co daje nam okolo 160 czastek na strone
        # Na podstawie zebranych probek z Gazebo
        # PRÓBKI Z LIDAR
        # 0-80 tył lewy 561 - 640 tył prawy
        # 81-240 prawy bok
        # 241 - 400 środek
        # 401-560 lewy bok

        # ENG
        # My vehicle has 640 laser beams for analysis, so:
        # We need to divide 640 by 4 sides of the vehicle, which gives us approximately 160 beams per side.
        # Based on the collected samples from Gazebo:
        # LIDAR SAMPLES:
        # 0-80: rear left, 561-640: rear right
        #  81-240: right side
        # 241-400: center
        # 401-560: left side
        
        self.right_side = ranges[80:240]
        self.left_side = ranges[400:560]
        self.front_side = ranges[240:400]
        self.rear_side = ranges[-80:] + ranges[:80]

        
        self._is_safe_left_side_of_vehicle = self.analyze_side_of_vehicle(self.left_side, 'left')
        # print(f'[DEBUG] left side is safe: {self._is_safe_left_side_of_vehicle}')

        self._is_safe_right_side_of_vehicle = self.analyze_side_of_vehicle(self.right_side, 'right')
        # print(f'[DEBUG] right side is safe: {self._is_safe_right_side_of_vehicle}')

        self._is_safe_front_side_of_vehicle = self.analyze_side_of_vehicle(self.front_side, 'front')
        # print(f'[DEBUG] front side is safe: {self._is_safe_front_side_of_vehicle}')

        self._is_safe_rear_side_of_vehicle = self.analyze_side_of_vehicle(self.rear_side, 'rear')
        # print(f'[DEBUG] rear side is safe: {self._is_safe_rear_side_of_vehicle}')

        if self._is_safe_front_side_of_vehicle:
            self._blocked_front = False
        elif not self._is_safe_front_side_of_vehicle and self._blocked_front == False:
            self._blocked_front = True
            self.stop_vehicle()
        
        if self._is_safe_rear_side_of_vehicle:
            self._blocked_rear = False
        elif not self._is_safe_rear_side_of_vehicle and self._blocked_rear == False:
            self._blocked_rear = True
            self.stop_vehicle()

        if self._is_safe_left_side_of_vehicle:
            self._blocked_left = False
        elif not self._is_safe_left_side_of_vehicle and self._blocked_left == False:
            self._blocked_left = True
            self.stop_vehicle()

        if self._is_safe_right_side_of_vehicle:
            self._blocked_right = False
        elif not self._is_safe_right_side_of_vehicle and self._blocked_right == False:
            self._blocked_right = True
            self.stop_vehicle()
        

        # print(f'[DEBUG] PARTICLES ANALYZED')  

    
    def analyze_side_of_vehicle(self, particles, side):
        # PL
        # Ta metoda analizuje czasteczki pobrane z lidara
        # nastepnie decyduje czy pojazd znajduje sie
        # w bezpiecznej odleglosci czy nie
        # odlegosc <= 2 jest pogladowa i warto by kiedys to zoptymalizowac
        # pod kadtem prawidlowego decydowania czy pojazd jest bezpieczny
        # ENG
        # This method analyzing particles collected from lidar
        # and decide if distance is safe or not
        # @return boolean
        for particle in particles:
            if particle == 'inf':
                continue        
            if float(particle) <= 2:
                # print(f'[DEBUG] Distance on {side} is not safe {particle}')
                return False       
        # print(f'[DEBUG] Distance on {side} is safe {particle}')        
        return True

    def stop_vehicle(self):
        # PL
        # Zatrzymaj pojazd jesli dystans dla wybranej strony pojazdu 
        # jest mniejszy niz ~1.5 metra
        # ENG
        # Stops vehicle if distance in exact side is closer than ~1.5 meter
        print(f'[DEBUG] - IN STOP VEHICLE INITIALIZE')
        node = Node()
        message = Twist()
        topic = '/cmd_vel'
        publisher = node.advertise(topic, Twist)

        message.angular.x = 0.0
        message.angular.z = 0.0
        message.linear.x = 0.0
        message.linear.z = 0.0
    
        print(f'[DEBUG] - IN STOP VEHICLE PUBLISH')

        if publisher.publish(message):
            print(f'[INFO] Successfully send {message}')
        else:
            print(f'[ERROR] Failed to send {message} message')