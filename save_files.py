
import rosbag

# Open the bag file
bag_path ="/home/avalocal/mcity/rosbag2_2024_09_15-14_12_43_0.db3"
bag = rosbag.Bag(bag_path)

# Specify the topic you want to read
topic_name = '/localization/pose_twist_fusion_filter/kinematic_state'

# Open a text file to write the data
with open('output.txt', 'w') as f:
    for topic, msg, t in bag.read_messages(topics=[topic_name]):
        # Write the message data to the file
        f.write(str(msg) + '\n')

# Close the bag file
bag.close()
