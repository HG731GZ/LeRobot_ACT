import cv2

from UR_Utils.URRealtimeClient import URRealtimeClient
from UR_Utils.URScriptClient import URScriptClient
from UR_Utils.GripperController import GripperController
from UR_Utils.RealSenseCamera import Camera
import scripts.ur_action_to_pose
from scripts.ur5e_act_realtime_inference import UR5eACTRealtimeInference
import time
import numpy as np
import cv2

CAMERA_RESOLUTION = (1280, 720)  # 相机分辨率，RGB和深度统一设定
CAMERA_FPS = 30  # 相机帧率
URIP = '192.168.3.15'

predictor = UR5eACTRealtimeInference(
    policy_path="../outputs/train/ur5e_act_a6000",
    device="cuda",
    inference_interval_s=0.1,
)

Camera1 = Camera('d435i', resolution=CAMERA_RESOLUTION, fps=CAMERA_FPS)
time.sleep(0.2)
Camera2 = Camera('d455', resolution=CAMERA_RESOLUTION, fps=CAMERA_FPS)

URScriptClient = URScriptClient(URIP, auto_connect=True)
URRealtimeClient = URRealtimeClient(URIP, auto_connect=True)
GripperController = GripperController(port=URIP + f":{54321}", slave_id=1, connection_type="tcp", debug=False)
GripperController.start(interval=0.05)

# 第一次获取状态确保全部不为None
UR_states = URRealtimeClient.get_latest_state()
gripper_fb = GripperController.feedback
trycount = 0
while (UR_states is None) or (gripper_fb is None) or trycount < 50:
    UR_states = URRealtimeClient.get_latest_state()
    gripper_fb = GripperController.feedback
    time.sleep(0.1)
    trycount = trycount + 1

GripperController.move(0.99, speed=20, accel=10, force=20)
print("开始ACT测试！")
time.sleep(1)
# 正式开始循环
while True:
    UR_states = URRealtimeClient.get_latest_state()
    UR_tcp_pose = UR_states.tcp_pose
    camera1_image = Camera1.get_rgb_frame().image
    camera2_image = Camera2.get_rgb_frame().image
    camera1_show = cv2.cvtColor(camera1_image, cv2.COLOR_RGB2BGR)
    camera2_show = cv2.cvtColor(camera2_image, cv2.COLOR_RGB2BGR)

    cv2.imshow("camera1", camera1_show)
    cv2.imshow("camera2", camera2_show)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    gripper_fb = GripperController.feedback
    gripper = [gripper_fb.open, gripper_fb.current]
    next_tcp_pose = predictor.predict_next_tcp_pose(
        camera1_image,
        camera2_image,
        UR_tcp_pose,
        gripper,
    )
    URScriptClient.movel(next_tcp_pose[:6], a=0.1, v=0.1, frame='base_abs')
    GripperController.set_target_position(next_tcp_pose[6])
    print(next_tcp_pose)

    time.sleep(0.1)

cv2.destroyAllWindows()
