from Utils.POSE3 import Pose3
import subprocess
import os
import json
from Feature2D.SIFTFeature import *
from pytypes import override
import cv2
import typing


class PnPBase(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self) -> None:
        pass

    def getPoseRef2Cur(self, refImg : np.ndarray, curImg: np.ndarray, curImg_depth:np.ndarray, K: np.ndarray) -> (Pose3, np.ndarray, np.ndarray):
        pose = Pose3()
        p_ref, p_cur = None, None
        return pose, p_ref, p_cur

class PnP_CV(PnPBase):
    def __init__(self):
        super(PnP_CV, self).__init__()

    def get3dPoint(cls, imgpt, depth, K):
        imgpt_homo = np.matrix([imgpt[0], imgpt[1], 1]).T
        K_i = np.matrix(K).I
        pt3d = (K_i * imgpt_homo * depth).ravel()
        return pt3d

    def get3dPoints(cls, ref_pts, cur_pts, curImg_depth, K):
        # get valid data
        valid_ref_pts = []
        valid_cur_pts = []
        valid_cur_depth = []
        for i in range(0, len(ref_pts)):
            x = int(cur_pts[i, 0])
            y = int(cur_pts[i, 1])
            if curImg_depth[y, x] == 0: # invalid depth
                continue
            valid_ref_pts.append(ref_pts[i])
            valid_cur_pts.append(cur_pts[i])
            valid_cur_depth.append(curImg_depth[y, x]) # note: first y:height , then x:width

        # conver 2d image points to 3d points
        valid_cur_pts3d = []
        for i in range(0, len(valid_cur_pts)):
            pt3d = cls.get3dPoint(valid_cur_pts[i], valid_cur_depth[i], K)
            valid_cur_pts3d.append(pt3d)

        return np.float32(valid_ref_pts), np.float32(valid_cur_pts), np.float32(valid_cur_pts3d)

    # @override
    def getPoseRef2Cur(self, refImg: np.ndarray, curImg: np.ndarray, curImg_depth: np.ndarray, K: np.ndarray):
        ref_pts, cur_pts = SIFTFeature.detectAndMatch(image1=refImg,image2=curImg)
        ref_pts_2d, cur_pts_2d, cur_pts_3d = self.get3dPoints(ref_pts, cur_pts, curImg_depth, K)

        # solvePnPRansac(A, B) -> Pose_B = Relative_Pose_A2B * Pose_A
        retval, rvec, tvec, inlier = cv2.solvePnPRansac(cur_pts_3d, ref_pts_2d, K, None, flags=cv2.SOLVEPNP_EPNP)

        if retval == True:
            R, jacobian = cv2.Rodrigues(rvec)
            pose_cur2ref = Pose3.fromRt(R, tvec)

            pose_ref2cur = pose_cur2ref.inverse()
            return pose_ref2cur, ref_pts_2d, cur_pts_2d
        else:
            return Pose3(), ref_pts_2d, cur_pts_2d

def testPnP_CV():
    print("Test PnP in OpenCV:")
    pnp = PnP_CV()
    im_dir = ""
    refImg = cv2.imread(os.path.join(im_dir, '1_ref.png'))
    curImg = cv2.imread(os.path.join(im_dir, '1_cur.png'))
    curImg_depth = cv2.imread(os.path.join(im_dir, '1_cur_depth.png'), cv2.IMREAD_UNCHANGED)

    leftK = np.float32([[320, 0, 320], [0, 320, 240], [0, 0, 1]])
    pose, p_ref, p_cur = pnp.getPose(refImg, curImg, curImg_depth, leftK)


    print("Pose computed using PnP:")
    pose.display()
    print("ground-truth Pose:")
    pose_gt.display()


if __name__ == "__main__":
    # testPnP_MVG()
    testPnP_CV()

