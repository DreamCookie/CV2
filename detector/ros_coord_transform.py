import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener
import yaml
from pathlib import Path
import numpy as np
import tf_transformations as tr

class RosCoordTransformer(Node):
    """
    Динамический трансформер координат eye-in-hand через ROS2-TF
    """
    def __init__(self,
                 config_path: Path,
                 robot_frame: str = 'base_link',
                 table_frame: str = 'table_link',
                 camera_frame: str = 'camera_link'):
        super().__init__('ros_coord_transformer')
        self.robot_frame = robot_frame
        self.table_frame = table_frame
        self.camera_frame = camera_frame

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        with config_path.open('r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        self.scale_mm_per_px = cfg['contour'].get('scale_mm_per_px', 1.0)

    def pixels_to_robot(self, cx: float, cy: float):
        X_table = cx * self.scale_mm_per_px
        Y_table = cy * self.scale_mm_per_px
        pt_table = np.array([X_table/1000.0, Y_table/1000.0, 0.0, 1.0])

        try:
            now = rclpy.time.Time()
            trans = self.tf_buffer.lookup_transform(
                self.robot_frame,
                self.table_frame,
                now,
                timeout=rclpy.duration.Duration(seconds=0.1)
            )
            q = trans.transform.rotation
            t = trans.transform.translation
            T = np.eye(4)
            T[:3,:3] = tr.quaternion_matrix([q.x,q.y,q.z,q.w])[:3,:3]
            T[:3,3] = [t.x,t.y,t.z]

            pt_robot = T.dot(pt_table)
            return pt_robot[:3] * 1000.0
        except Exception as e:
            self.get_logger().warn(f'TF lookup failed: {e}')
            return None
