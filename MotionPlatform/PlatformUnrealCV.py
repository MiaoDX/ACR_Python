from MotionPlatform.PlatformBase import PlatformBase
from Utils.POSE3 import Pose3
from UnrealCVBase.UnrealCVEnv import UnrealCVEnv


class PlatformUnrealCV(PlatformBase):
    def __init__(self, unreal_env: UnrealCVEnv, X: Pose3=Pose3()):
        super(PlatformBase, self).__init__()
        self.unrealCvEnv = unreal_env
        self.X = X  # hand eye relative pose, just for suppose, in virtual environment, it does not exist in fact

    def open(self) -> bool:
        self.unrealCvEnv.connect()
        return self.unrealCvEnv.isConnected()

    def close(self) -> bool:
        self.unrealCvEnv.disconnect()
        return not self.unrealCvEnv.isConnected()

    # def rotate(self, roll, yaw, pitch) -> bool:
    #     self.unrealCvEnv.rotate_cam(roll=roll, yaw=yaw, pitch=pitch)
    #     return True
    #
    # def translate(self, x, y, z)->bool:
    #     self.unrealCvEnv.translate_cam(x=x, y=y, z=z)
    #     return True


    # Recommend to use for all relative motions
    # AX = XB, thus A = X B X^-1
    def movePose(self, movingPose: Pose3):
        eyeMotion = self.X.compose(movingPose.compose(self.X.inverse()))
        self.unrealCvEnv.move_relative_pose_eye(eyeMotion)
        return True

    def autoHorizon(self) -> bool:
        raise NotImplementedError

    def stopAutoHorizon(self) -> bool:
        raise NotImplementedError

    def goHome(self) -> bool:
        self.unrealCvEnv.resetPose()
        return True
