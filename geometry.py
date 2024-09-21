import utm
import pyproj


import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2  , PointField
from sensor_msgs_py import point_cloud2 as pc2
from sensor_msgs.msg import CompressedImage
import std_msgs.msg
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import cv2
from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry

FIELDS_XYZ = [
    PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)
]

zone_number = 17
zone_letter = "T"
GNSS_origin = [42.3005934157, -83.699283188811]

UTM_offset = [-277497.10, -4686518.71, 0.0]

# x: 136.69110318587627
#       y: 25.73287988640368
#       z: 0.0
# x= 136.69110318587627
# y= 25.73287988640368
# z= 0.0


def utm_local_to_global(x, y):
    utm_x = x - UTM_offset[0]
    utm_y = y - UTM_offset[1]

    return utm_x, utm_y

def utm_global_to_local(utm_x, utm_y):
    x = utm_x + UTM_offset[0]
    y = utm_y + UTM_offset[1]

    return x, y

def latlon_to_xy(lat_center, lon_center, lat_point, lon_point):
    # Define a Transverse Mercator projection with the center point as the origin
    projection = pyproj.Proj(proj='tmerc', lat_0=lat_center, lon_0=lon_center, ellps='WGS84')
    
    # Convert the point lat/lon to x, y coordinates relative to the center point
    x, y = projection(lon_point, lat_point)
    
    return x, y

def xy_to_latlon(lat_center, lon_center, x, y):
    # Define the same Transverse Mercator projection with the center point as the origin
    projection = pyproj.Proj(proj='tmerc', lat_0=lat_center, lon_0=lon_center, ellps='WGS84')
    
    # Convert x, y coordinates back to lat/lon using the inverse of the projection
    lon_point, lat_point = projection(x, y, inverse=True)
    
    return lat_point, lon_point

def utm_to_carla(utm_x, utm_y):
    lat, lon = utm.to_latlon(utm_x, utm_y, zone_number, zone_letter)
    local_x, local_y = latlon_to_xy(GNSS_origin[0], GNSS_origin[1], lat, lon)

    return local_x, local_y

def carla_to_utm(x, y):
    lat, lon = xy_to_latlon(GNSS_origin[0], GNSS_origin[1], x, -y)
    utm_x, utm_y, _, _ = utm.from_latlon(lat, lon)

    return utm_x, utm_y



# Read ego info from /localization/pose_twist_fusion_filter/kinematic_state,
# extract (x,y) position, use function utm_local_to_global to convert that 
# to a global UTM coordinate, then use function utm_to_carla to convert that 
# to carla local coordinate.

# x1,y1 = utm_local_to_global(x, y)
# x2,y2 = utm_to_carla(x1, y1)
# print(x1,y1, "utm local to global")
# print(x2,y2, "utm to carla")
#get rostopic realtime and show the results as point cloud output
class PerceptionNode(Node):

    def __init__(self):

        super().__init__('perception_node_2')
        #/localization/pose_twist_fusion_filter/kinematic_state
        self.sub_localization = self.create_subscription(Odometry, "/localization/pose_twist_fusion_filter/kinematic_state", self.callback, 10)
     
        self.pub_localization = self.create_publisher(PointCloud2, "/perception/localization", 10)
        


    def callback(self, msg):
            print(msg)
            
            x = msg.pose.pose.position.x
            y = msg.pose.pose.position.y
            z = msg.pose.pose.position.z

            

            x1,y1 = utm_local_to_global(x, y)
            x2,y2 = utm_to_carla(x1, y1)  #vehicle position in carla world

            print(x1,y1, "utm local to global")
            print(x2,y2, "utm to carla")

            p3d = [(x2, y2, z)]

            header = std_msgs.msg.Header()
            header.stamp = self.get_clock().now().to_msg()
            header.frame_id = "frame"
            pointcloud = pc2.create_cloud_xyz32(header, p3d)
            # Print x, y, z values
            for point in p3d:
                x, y, z = point  # Assuming p3d is a list of (x, y, z)
                print(f"x: {x}, y: {y}, z: {z}")
            
            self.pub_localization.publish(pointcloud)


if __name__ == '__main__':
    rclpy.init()
    node_pereption_2 = PerceptionNode()
    rclpy.spin(node_pereption_2)




