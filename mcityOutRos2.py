import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2  , PointField
from sensor_msgs.msg import CompressedImage
import torch
from models.modelMulti import GLPDepth
from labels import labels
import torch.backends.cudnn as cudnn
import numpy as np
import cv2
import argparse

#for clustering
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

ckpt_dir='/home/avalocal/thesis23/SwinMTL/logs/2024-09-01_13-25-34_mcity_4_swin_v2_base_simmim_deconv3_32_2_480_480_00005_3e-06_09_005_500_30_30_30_15_2_2_18_2/epoch_100_model.ckpt'

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
        w, h = 608, 320
        FOV = 90
        k = np.identity(3)
        k[0, 2] = w / 2
        k[1, 2] = h / 2
        k[0, 0] = k[1, 1] = w / (2 * np.tan(FOV * np.pi / 360))
        self.invK = np.linalg.inv(k)


     

    def callback(self, image):
        print("Received image")

        img = np.frombuffer(image.data, dtype=np.uint8)
        #bgr to rgb
        img = cv2.imdecode(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (608, 320))
        print(img.shape, " ", img.dtype, " img info") #(320, 608, 3)
        im = img.transpose(2, 0, 1) #3, 320, 608
        im = torch.from_numpy(im).to(self.device).float() / 255
        #make it bgr
        # im = im[[2, 1, 0], :, :]
        

        if self.model is None:
            raise ValueError("Model not loaded")
        
        #publish image
        image = Image()
        image.header = image.header
        image.height = img.shape[0]
        image.width = img.shape[1]
        image.encoding = "bgr8"
        image.step = img.shape[1] * 3
        image.data = img.tobytes()
        self.pub_image.publish(image)

        with torch.no_grad():
            pred = self.model(im.unsqueeze(0).to(self.device))
            pred_depth = pred['pred_d'].squeeze(0).squeeze(0)
            pred_depth = pred_depth.cpu().numpy()
            pred_depth = (pred_depth * 1000).astype(np.uint16)
            pred_seg = torch.argmax(pred['pred_seg'], dim=1)
            pred_seg = pred_seg.squeeze(0).cpu().numpy().astype(np.uint8)     
        
            print(pred_depth.shape, pred_seg.shape)
            print(pred_depth.dtype, pred_seg.dtype)
            print(pred_depth.min(), pred_depth.max(), "depth min max")
            print(pred_seg.min(), pred_seg.max(), "seg min max") 
            print(np.unique(pred_seg), "unique seg")

        #get inary map of just car(1: car, 0: not car)
        car_idx = pred_seg == 14
        not_car_idx = pred_seg != 14
        car_seg = np.zeros_like(pred_seg)
        car_seg[car_idx] = 1
        car_seg[not_car_idx] = 0
        non_zero_idx_car = np.column_stack(np.where(car_seg == 1))
        db = DBSCAN(eps=5, min_samples=50).fit(non_zero_idx_car)
        labels = db.labels_
        unique_labels = set(labels)
        unique_labels.remove(-1)

        centeroids = []
        for label in unique_labels:
            cluster = non_zero_idx_car[labels == label]
            # pca = PCA(n_components=2)
            # pca.fit(cluster)
            centeroids.append(np.mean(cluster, axis=0))

            # centeroids.append(pca.mean_)
        centeroids = np.array(centeroids)
        print(centeroids, "centeroids")
        depth_values = []
        for center in centeroids:
            depth_values.append(pred_depth[int(center[0]), int(center[1])])
 
        #u, v, 1
        points3d = np.column_stack((centeroids, np.ones(centeroids.shape[0])))
        points3d = np.dot(self.invK, points3d.T).T
        points3d = points3d * np.array(depth_values)[:, np.newaxis]

        print(points3d, "3d points")
        publish_pointcloud = True
        if publish_pointcloud:
            #publish pointcloud
            pointcloud = PointCloud2()
            pointcloud.header = image.header
            pointcloud.height = 1
            pointcloud.width = points3d.shape[0]
            pointcloud.fields = [PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
                                PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
                                PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1)]
            pointcloud.is_bigendian = False
            pointcloud.point_step = 12
            pointcloud.row_step = 12 * points3d.shape[0]
            pointcloud.is_dense = True
            pointcloud.data = points3d.astype(np.float32).tobytes()
            self.pub_pointcloud.publish(pointcloud)






        
        publish_Image = False
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
            seg_image.height = pred_seg.shape[0]
            seg_image.width = pred_seg.shape[1]
            seg_image.encoding = "mono8"
            seg_image.step = pred_seg.shape[1]
            seg_image.data = pred_seg.tobytes()
            self.pub_seg.publish(seg_image)

    

def main(args=None):
    rclpy.init(args=args)
    args = parse_opt()
    perception_node = PerceptionNode(args)
    rclpy.spin(perception_node)
    perception_node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
