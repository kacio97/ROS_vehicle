import time
from gz.msgs10.twist_pb2 import *
from gz.msgs10.stringmsg_pb2 import *
from gz.msgs10.laserscan_pb2 import *
from gz.msgs10.vector3d_pb2 import *
from gz.transport13 import *


def stop_vehicle():
    print(f'DEBUG - IN STOP VEHICLE INITIALIZE')
    node = Node()
    twist_message = Twist()
    topic = '/cmd_vel'
    node_publisher = node.advertise(topic, Twist)

    twist_message.angular.x = 0.0
    twist_message.angular.z = 0.0
    twist_message.linear.x = 0.0
    twist_message.linear.z = 0.0
   
    twist_message
    print(f'DEBUG - IN STOP VEHICLE PUBLISH')

    if node_publisher.publish(twist_message):
        print(f'Successfully send {twist_message}')
    else:
        print(f'Failed to send {twist_message} message')



def collect_lidar_data(msg: LaserScan):
    for _msg in msg.ranges:
        if str(_msg) == 'inf':
           continue

        print(f'Range - {_msg}')

        if float(_msg) < 1.0:
           stop_vehicle()

    

if __name__ == "__main__":
    topic_subscription = '/lidar'
    node = Node()

    if node.subscribe(LaserScan, topic_subscription, collect_lidar_data):
        print(f'Subcribed {topic_subscription} topic')
    else:
        raise f'Error during topic {topic_subscription} subscription'

    # wait for shutdown
    try:
      while True:
        time.sleep(1)
    except KeyboardInterrupt:
      pass
    print("Done")