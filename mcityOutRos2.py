import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2  , PointField
from sensor_msgs_py import point_cloud2 as pc2
from sensor_msgs.msg import CompressedImage
import std_msgs.msg
import torch
from models.modelMulti import GLPDepth
from labels import labels
import torch.backends.cudnn as cudnn
import numpy as np
import cv2
import argparse
from labels import labels
from numpy.matlib import repmat

#for clustering
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

FIELDS_XYZ = [
    PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)
]

ckpt_dir='/home/avalocal/thesis23/Mcity2.0_demo/epoch_40_model.ckpt'

def load_glp_depth_model(args, device):
    model = GLPDepth(args=args).to(device)
    load_model(ckpt_dir, model)
    model.eval()
    return model

def load_model(ckpt, model, optimizer=None):
    ckpt_dict = torch.load(ckpt, map_location='cpu')
    state_dict = ckpt_dict['model']
    weights = {key[len('module.'):] if key.startswith('module.') else key: value for key, value in state_dict.items()}
    model.load_state_dict(weights)
    if optimizer is not None:
        optimizer_state = ckpt_dict['optimizer']
        optimizer.load_state_dict(optimizer_state)

def get_color_mask(mask, labels, id_type='id'):

    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    if id_type == 'id':
        for label in labels:
            color_mask[mask == label.id] = label.color
    elif id_type == 'trainId':
        for label in labels:
            color_mask[mask == label.trainId] = label.color

    return color_mask

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_depth", type=float, default=1000, help="Maximum depth value")
    parser.add_argument("--backbone", type=str, default="swin_base_v2", help="Backbone model")
    parser.add_argument("--depths", type=list, default=[2, 2, 18, 2], help="Number of layers in each stage")
    parser.add_argument("--num_filters", type=list, default=[32, 32, 32], help="Number of filters in each stage")
    parser.add_argument("--deconv_kernels", type=list, default=[2, 2, 2], help="Kernel size for deconvolution")
    parser.add_argument("--window_size", type=list, default=[30, 30, 30, 15], help="Window size for MIM")
    parser.add_argument("--pretrain_window_size", type=list, default=[12, 12, 12, 6], help="Window size for pretraining")
    parser.add_argument("--use_shift", type=list, default=[True, True, False, False], help="Use shift operation")
    parser.add_argument("--shift_size", type=int, default=16, help="Shift size")
    parser.add_argument("--save_visualization", type=bool, default=False, help="Save visualization")
    parser.add_argument("--flip_test", type=bool, default=False, help="Flip test")
    parser.add_argument("--shift_window_test", type=bool, default=False, help="Shift window test")
    parser.add_argument("--num_classes", type=int, default=20, help="Number of classes")
    parser.add_argument("--drop_path_rate", type=float, default=0.3, help="Drop path rate")
    parser.add_argument("--pretrained", type=str, default="/home/avalocal/thesis23/MIM-Depth-Estimation/weights/swin_v2_base_simmim.pth", help="Pretrained weights")
    parser.add_argument("--save_model", type=bool, default=False, help="Save model")
    parser.add_argument("--crop_h", type=int, default=480, help="Crop height")
    parser.add_argument("--crop_w", type=int, default=480, help="Crop width")
    parser.add_argument("--layer_decay", type=float, default=0.9, help="Layer decay")
    parser.add_argument("--use_checkpoint", type=bool, default=True, help="Use checkpoint")
    parser.add_argument("--num_deconv", type=int, default=3, help="Number of deconvolution layers")
    return parser.parse_args()

class PerceptionNode(Node):

    def __init__(self, args):
        super().__init__('perception_node')

        self.sub_image = self.create_subscription(CompressedImage, "/camera/image_color/compressed", self.callback, 10)
        self.pub_image = self.create_publisher(Image, "/perception/image", 10)
        self.pub_depth = self.create_publisher(Image, "/perception/depth", 10)
        self.pub_seg = self.create_publisher(Image, "/perception/segmentation", 10)
        self.pub_pointcloud = self.create_publisher(PointCloud2, "/perception/pointcloud", 10)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cudnn.benchmark = True
        self.model = load_glp_depth_model(args, self.device)
        self.model.eval()


        w, h = 608, 320
        # w,h = 1200,600
        FOV = 90
        k = np.identity(3)
        k[0, 2] = w / 2
        k[1, 2] = h / 2
        k[0, 0] = k[1, 1] = w / (2 * np.tan(FOV * np.pi / 360)) #=w/2
        self.invK = np.linalg.inv(k)


    def callback(self, image):
        print("Received image")

        img = np.frombuffer(image.data, dtype=np.uint8)
        img = cv2.imdecode(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (608, 320))
        im = img.transpose(2, 0, 1) #3, 320, 608
        im = torch.from_numpy(im).to(self.device).float() / 255
        
        #publish image
        image = Image()
        image.header = image.header
        image.height = img.shape[0]
        image.width = img.shape[1]
        image.encoding = "bgr8"
        image.step = img.shape[1] * 3
        image.data = img.tobytes()
        self.pub_image.publish(image)
        

        if self.model is None:
            raise ValueError("Model not loaded")
        

        with torch.no_grad():
            pred = self.model(im.unsqueeze(0).to(self.device))
            pred_depth = pred['pred_d'].squeeze(0).squeeze(0)
            pred_depth = pred_depth.cpu().numpy()
            pred_depth = (pred_depth * 100).astype(np.uint16) #depth in cm, #320, 608, #uint16
            pred_seg = torch.argmax(pred['pred_seg'], dim=1)  #320, 608, #uint8
            pred_seg =pred_seg.squeeze(0).cpu().numpy().astype(np.uint8)  #320, 608, #uint8            
            pred_seg_colored = get_color_mask(pred_seg, labels, id_type='trainId')
        
        print(pred_depth.shape, pred_seg.shape, pred_seg_colored.shape, "shapes")

        #binary map for cars
        car_class = 14
        car_mask = pred_seg == car_class
        not_car_mask = pred_seg != car_class
        pred_seg[car_mask] = 1
        pred_seg[not_car_mask] = 0

        non_zero = np.column_stack(np.where(pred_seg == 1))
        db = DBSCAN(eps=5, min_samples=50).fit(non_zero)
        labels2 = db.labels_
        unique_labels = set(labels2)
        unique_labels.remove(-1)

        car_centers = []
        car_depths_mean = []
        
        for label in unique_labels:
            class_points = non_zero[labels2 == label]
            num_clusters = class_points.shape[0]            
            car_centers.append(np.mean(class_points, axis=0)) #u, v
            
            car_depths_mean.append(np.mean(pred_depth[class_points[:, 0], class_points[:, 1]]))

        car_centers = np.array(car_centers)
        car_depths_mean = np.array(car_depths_mean)
        

        # show car centers on pred_seg with red dots
        for center in car_centers:
            cv2.circle(pred_seg_colored, (int(center[1]), int(center[0])), 5, (255, 0, 0), -1)
          
        p2d_cars = np.array([car_centers[:, 0], car_centers[:, 1], np.ones_like(car_centers[:, 0])])
        p3d_cars = np.dot(self.invK, p2d_cars) * car_depths_mean
        p3d_cars = p3d_cars.T
        print(p3d_cars.shape, "p3d_cars shape")
        p3d = p3d_cars


        #camera.get_transform().get_inverse_matrix() and here is the result: [[-0.12058158218860626, 0.9921349287033081, -0.03359150141477585, 30.274852752685547], [-0.9926763772964478, -0.12075912207365036, -0.0033001673873513937, 10.892216682434082], [-0.007330691441893578, 0.03294754773378372, 0.999430239200592, -236.77127075195312], [0.0, 0.0, 0.0, 1.0]]
        
        # p_inv = np.array([[-0.12058158218860626, 0.9921349287033081, -0.03359150141477585, 30.274852752685547], [-0.9926763772964478, -0.12075912207365036, -0.0033001673873513937, 10.892216682434082], [-0.007330691441893578, 0.03294754773378372, 0.999430239200592, -236.77127075195312], [0.0, 0.0, 0.0, 1.0]]
        # )

        # p3d = np.column_stack((p3d, np.ones(p3d.shape[0])))
        # p3d = np.dot(p_inv, p3d.T).T
        # p3d = p3d[:, :3]


        x= p3d[:, 0]
        y= p3d[:, 2]
        z= p3d[:, 1]
        p3d = np.column_stack((x, y, z))
              
        
        
        publish_Image = True
        if publish_Image:
            # Publish depth image
            depth_image = Image()
            depth_image.header = image.header
            depth_image.height = pred_depth.shape[0]
            depth_image.width = pred_depth.shape[1]
            depth_image.encoding = "16UC1"
            depth_image.step = pred_depth.shape[1] * 2
            depth_image.data = pred_depth.tobytes()
            self.pub_depth.publish(depth_image)
            # Publish segmentation image
            
            seg_image = Image()
            seg_image.header = image.header
            seg_image.height = pred_seg_colored.shape[0]
            seg_image.width = pred_seg_colored.shape[1]
            seg_image.encoding = "rgb8"
            seg_image.step = pred_seg_colored.shape[1] * 3
            seg_image.data = pred_seg_colored.tobytes()
            self.pub_seg.publish(seg_image)

        publish_pointcloud = True
        if publish_pointcloud:
            # Publish pointcloud
            header = std_msgs.msg.Header()
            header.stamp = self.get_clock().now().to_msg()
            header.frame_id = "frame"
            pointcloud = pc2.create_cloud_xyz32(header, p3d)
            # Print x, y, z values
            for point in p3d:
                x, y, z = point  # Assuming p3d is a list of (x, y, z)
                print(f"x: {x}, y: {y}, z: {z}")
            
            self.pub_pointcloud.publish(pointcloud)




    

def main(args=None):
    rclpy.init(args=args)
    args = parse_opt()
    perception_node = PerceptionNode(args)
    rclpy.spin(perception_node)
    perception_node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
