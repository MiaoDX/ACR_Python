
from Camera.CameraBase import CameraBase
from MotionPlatform.PlatformBase import PlatformBase
from RelativePose import ComputePnP as pnp
from Utils.POSE3 import Pose3
import numpy as np
import os
import cv2
import json


class ACRD(object):
    def __init__(self, camera, platform):
        self.camera = camera  # type: CameraBase
        self.pnp = pnp.PnP_CV()
        self.platform = platform  # type: PlatformBase

        # damp the computed rotation and translation
        self.dampRatio = 1

        # stop condition parameters
        self.stopAngle = 0.1
        self.stopTrans = 0.2
        #self.stopAFD = 0.2
        self.stopAFD = 1.0
        self.maxStep = 15

        # current condition and data
        self.step = 0
        self.curPose = Pose3()
        self.curAFD = 0

        # reference image
        self.refImage = np.ndarray([1, 1, 3], np.uint8)
        self.refImage_depth = np.ndarray([1, 1, 3], np.uint16)
        # dir to storage image
        self.data_dir = ""

    def initSettings(self, data_dir, refImage: np.ndarray, refImage_depth: np.ndarray):
        self.step = 0
        self.curPose = Pose3.from6D([100, 100, 100, 100, 100, 100])  # set initial pose to be a large value
        self.curAFD = 100

        self.data_dir = data_dir
        self.refImage = refImage
        self.refImage_depth = refImage_depth

    def openAll(self):
        self.camera.open()
        self.platform.open()

    @classmethod
    def rtsize(cls, pose: Pose3):
        t, axis, angle = pose.to_t_aixsAngle()
        ts = np.linalg.norm(t, 2)
        return ts, angle

    def stopCondition(self):
        t, angle = self.rtsize(self.curPose)
        if (t < self.stopTrans and angle < self.stopAngle) or self.curAFD < self.stopAFD or self.step >= self.maxStep:
            return True
        return False

    def computeAFD(cls, match_points_ref, match_points_cur):
        error = match_points_ref - match_points_cur
        error2 = np.linalg.norm(error,2, axis=1)
        AFD = np.sum(error2) / len(error2)
        return AFD

    def dumpPose(self, pose):
        motion = pose.toCenter6D()
        rots = np.array(motion[3:]) * self.dampRatio   # sequence: rz, ry, rx
        trans = np.array(motion[:3]) * self.dampRatio  # sequence: tx, ty, tz
        return Pose3.fromCenter6D(np.append(trans, rots))

    @classmethod
    def writeInfo(cls, directory, stepNum, cur_image, cur_image_depth, cur_pose, cur_AFD, cur_image_warp, cur_image_depth_warp):
        if not os.path.exists(directory):
            os.mkdir(directory)

        img_path = os.path.join(directory, "rgb_{}.png".format(stepNum))
        img_depth_path = os.path.join(directory, "depth_{}.png".format(stepNum))
        cv2.imwrite(img_path, cur_image)
        cv2.imwrite(img_depth_path, cur_image_depth)

        pose_file = os.path.join(directory, "pose_{}.json".format(stepNum))
        with open(pose_file, 'w') as f:
            pose_dic = {"curPose": cur_pose.toSE3().tolist(), "cur_AFD": cur_AFD}
            pose_json = json.dumps(pose_dic, indent=2, separators=(',', ': '))
            f.write(pose_json)

        img_warp_path = os.path.join(directory, "rgb_warp_{}.png".format(stepNum))
        img_depth_warp_path = os.path.join(directory, "depth_warp_{}.png".format(stepNum))
        cv2.imwrite(img_warp_path, cur_image_warp)
        cv2.imwrite(img_depth_warp_path, cur_image_depth_warp)

    @classmethod
    def writeHandInfo(cls, directory, stepNum, rots, trans):
        pose_file = os.path.join(directory, "handInfo_{}.json".format(stepNum))
        with open(pose_file, 'w') as f:
            pose_dic = {"trans": trans.tolist(), "rots": rots.tolist()}
            pose_json = json.dumps(pose_dic, indent=2, separators=(',', ': '))
            f.write(pose_json)


    def relocation(self):
        while True:
            image, image_depth = self.camera.getImage()
            pose_ref2cur, match_points_ref, match_points_cur = self.pnp.getPoseRef2Cur(self.refImage, image, image_depth,
                                                                        self.camera.cameraCalibration.getK())

            pose_cur2ref = pose_ref2cur.inverse()

            image_warp, image_depth_warp, warped_mask = cv2.rgbd.warpFrame(image, np.float32(image_depth),
                                                                           None, pose_cur2ref.toSE3(),
                                                                           self.camera.cameraCalibration.getK(), None)
            image_depth_warp = np.uint16(image_depth_warp)

            self.curAFD = self.computeAFD(match_points_ref, match_points_cur)
            self.writeInfo(self.data_dir, self.step, image, image_depth, pose_cur2ref, self.curAFD, image_warp, image_depth_warp)

            if self.stopCondition():
                break

            self.curPose = pose_ref2cur
            # moving platform
            dumpMotion = self.dumpPose(pose_cur2ref)
            self.platform.movePose(dumpMotion)

            # self.writeHandInfo(self.data_dir, self.step, rots, trans)

            self.step = self.step + 1



def test1():
    from Camera.ZEDCamera import ZEDCamera
    from MotionPlatform.Platform5Axis import Platform5Axis

    camera = ZEDCamera()  # type: CameraBase
    platform = Platform5Axis(url='http://127.0.0.1:10241/')
    myACR = ACRD(camera=camera, platform=platform)
    myACR.openAll()

    # myACR.platform.autoHorizon()
    # myACR.platform.goHome()
    directory = "C:/Code/bird/AFGCD-master/data/test5"
    if not os.path.exists(directory):
        os.mkdir(directory)
    img_path = os.path.join(directory, "rgb_ref.png")
    img_depth_path = os.path.join(directory, "depth_ref.png")
    ref_image = cv2.imread(img_path)
    ref_image_depth = cv2.imread(img_depth_path, cv2.IMREAD_UNCHANGED)

    myACR.initSettings(data_dir=directory, refImage= ref_image, refImage_depth=ref_image_depth)

    myACR.relocation()

def test2():

    from Camera.ZEDCamera import ZEDCamera
    from MotionPlatform.Platform5Axis import Platform5Axis

    camera = ZEDCamera()  # type: CameraBase
    platform = Platform5Axis(url='http://127.0.0.1:10241/')

    myACR = ACRD(camera=camera, platform=platform)

    myACR.openAll()

    myACR.camera.setParameters({'GAIN':40, 'EXPOSURE': 100, 'SATURATION':4, 'BRIGHTNESS':4, 'CONTRAST':4})

    # myACR.platform.autoHorizon()
    # myACR.platform.goHome()

    # myACR.platform.translate(10, 8, 11)
    # myACR.platform.rotate(3, 1, 2)

    ref_image, ref_image_depth = myACR.camera.getImage()
    rots = np.array([-2, -1, -3])  # z, y, x
    trans = np.array([15, 0, -15])  # x, y, z

    myACR.platform.translate(trans[0], trans[1], trans[2])
    myACR.platform.rotate(rots[2], rots[1], rots[0])


    directory = "D:/temp/acr"
    if not os.path.exists(directory):
        os.mkdir(directory)

    myACR.writeHandInfo(directory, -1, rots, trans)

    img_path = os.path.join(directory, "rgb_ref.png")
    img_depth_path = os.path.join(directory, "depth_ref.png")
    cv2.imwrite(img_path, ref_image)
    cv2.imwrite(img_depth_path, ref_image_depth)

    myACR.initSettings(data_dir=directory, refImage= ref_image, refImage_depth=ref_image_depth)
    input('Press to continue...')
    myACR.relocation()


def test3():
    from Camera.UnrealCVCamera import UnrealCVCamera, CameraCalibration
    from MotionPlatform.PlatformUnrealCV import PlatformUnrealCV
    from UnrealCVBase.UnrealCVEnv import UnrealCVEnv

    # initPose = Pose3().from6D(np.array([-500, 500, -1000, 0, 0, 0]))  # sofa
    initPose = Pose3().fromCenter6D(np.array([0, -1000, -1000, 0, 0, 0]))  # sofa

    unrealbase = UnrealCVEnv(init_pose=initPose)
    camera = UnrealCVCamera(unreal_env=unrealbase, cameraCalib=CameraCalibration())  # type: CameraBase
    HandEyeX = Pose3().fromCenter6D([10, 10, 10, -5, 5, -5])
    platform = PlatformUnrealCV(unreal_env=unrealbase, X=HandEyeX)
    myACR = ACRD(camera=camera, platform=platform)
    myACR.openAll()

    ref_image, ref_image_depth = myACR.camera.getImage()

    directory = "H:/projects/graduation_project_codebase/ACR_Python/ACR_RUN/PnP_Aided/acr_test3"
    assert not os.path.exists(directory)
    os.makedirs(directory)

    # myACR.writeHandInfo(directory, -1, rots, trans)

    img_path = os.path.join(directory, "rgb_ref.png")
    img_depth_path = os.path.join(directory, "depth_ref.png")
    cv2.imwrite(img_path, ref_image)
    cv2.imwrite(img_depth_path, ref_image_depth)

    #pose = Pose3.fromCenter6D([30, 20, 10, 1.2, 0, -1.3])
    pose = Pose3.fromCenter6D([100, -50, 150, 5, -15, -10])
    platform.movePose(movingPose=pose)

    myACR.initSettings(data_dir=directory, refImage=ref_image, refImage_depth=ref_image_depth)
    input('Press to continue...')
    myACR.relocation()

def test_warp_and_shift():
    from Camera.UnrealCVCamera import UnrealCVCamera, CameraCalibration
    from MotionPlatform.PlatformUnrealCV import PlatformUnrealCV
    from UnrealCVBase.UnrealCVEnv import UnrealCVEnv

    #initPose = Pose3().fromCenter6D(np.array([-210, -1620, 1300, 0, 0, 0]))  # wall
    initPose = Pose3().fromCenter6D(np.array([-1400, -1620, 1300, 0, 0, 0]))  # wall, the left

    unrealbase = UnrealCVEnv(init_pose=initPose)
    camera = UnrealCVCamera(unreal_env=unrealbase, cameraCalib=CameraCalibration())  # type: CameraBase
    # HandEyeX = Pose3().fromCenter6D([10, 10, 10, -5, 5, -5])
    HandEyeX = Pose3()
    platform = PlatformUnrealCV(unreal_env=unrealbase, X=HandEyeX)
    myACR = ACRD(camera=camera, platform=platform)
    myACR.openAll()

    ref_image, ref_image_depth = myACR.camera.getImage()

    # ratio = 1/4.0
    ratio = 1/3.0
    # ratio = 2/5.0
    use_warp=True # False for shift

    if use_warp:
        step = int(1400*ratio)  # distance, Distance to wall=700mm, FOV=90°, so, image width=1400mm
    else:
        # step = 128  # pixel
        step = int(640 * ratio)


    #dir_name = "wall/wall_{}_{:.2f}".format("warp" if use_warp else "shift", ratio)
    dir_name = "wall/wall_{}_{:.2f}".format("warp" if use_warp else "shift", ratio)

    base_dir = "H:/projects/graduation_project_codebase/ACR_Python/ACR_RUN/PnP_Aided/"+dir_name
    # directory = "H:/projects/graduation_project_codebase/ACR_Python/ACR_RUN/PnP_Aided/acr_test_warp_wall_600_2"
    assert not os.path.exists(base_dir)
    os.makedirs(base_dir)

    # myACR.writeHandInfo(directory, -1, rots, trans)

    img_path = os.path.join(base_dir, "rgb_ref.png")
    img_depth_path = os.path.join(base_dir, "depth_ref.png")
    cv2.imwrite(img_path, ref_image)
    cv2.imwrite(img_depth_path, ref_image_depth)


    for i in range(8):

        # GIVEN ref_image, ref_image_depth
        work_dir = base_dir+'/'+str(i)+'/'
        assert not os.path.exists(work_dir)
        os.makedirs(work_dir)


        # TWEAKing the reference images
        if use_warp:
            rp = Pose3().fromCenter6D([step, 0, 0, 0, 0, 0])
            # K = np.float32([[320, 0, 320], [0, 320, 240], [0, 0, 1]])

            image_warp, image_depth_warp, warped_mask = cv2.rgbd.warpFrame(ref_image, np.float32(ref_image_depth),
                                                                           None, rp.toSE3(),
                                                                           camera.getCameraCalibration().getK(), None)
        else: # shift
            M = np.float32([[1,0,-step],[0,1,0]])
            h, w = ref_image.shape[:2]
            #using above translation matrix , we shift the image to 20 pixel right and 40 pixel to down
            image_warp = cv2.warpAffine(ref_image, M, (w, h)) # we get shifted image
            image_depth_warp = cv2.warpAffine(ref_image_depth, M, (w, h))





        img_path = os.path.join(work_dir, "rgb_ref.png")
        img_depth_path = os.path.join(work_dir, "depth_ref.png")
        cv2.imwrite(img_path, ref_image)
        cv2.imwrite(img_depth_path, ref_image_depth)
        image_depth_warp = np.uint16(image_depth_warp)
        img_path = os.path.join(work_dir, "rgb_ref_transform.png")
        img_depth_path = os.path.join(work_dir, "depth_ref_transform.png")
        cv2.imwrite(img_path, image_warp)
        cv2.imwrite(img_depth_path, image_depth_warp)

        #pose = Pose3.fromCenter6D([30, 20, 10, 1.2, 0, -1.3])
        pose = Pose3.fromCenter6D([100, -50, 150, 5, -5, -10])
        platform.movePose(movingPose=pose)

        #myACR.initSettings(data_dir=directory, refImage=ref_image, refImage_depth=ref_image_depth)
        myACR.initSettings(data_dir=work_dir, refImage=image_warp, refImage_depth=image_depth_warp)
        # input('Press to continue...')
        myACR.relocation()

        ref_image, ref_image_depth = myACR.camera.getImage() # re-assign new image


        location = list(np.random.random_integers(-5, 5, 3))
        rotation = list(np.random.random_integers(-20, 20, 3) / 10.0)
        rp_6d = location + rotation
        print(rp_6d)
        rp = Pose3.fromCenter6D(rp_6d)
        myACR.platform.movePose(rp)
        random_rgb, _ = myACR.camera.getImage()
        img_path = os.path.join(work_dir, "rgb_random_after_ACR.png")
        cv2.imwrite(img_path, random_rgb)
        myACR.platform.movePose(rp.inverse())

if __name__ == "__main__":
    # test1()
    # test2()
    #stress_test()
    # test3()
    test_warp_and_shift()
