import abc
from Utils.POSE3 import Pose3
import numpy as np


class PlatformBase(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def open(self) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def close(self) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def rotate(self, roll, yaw, pitch) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def translate(self, x, y, z)->bool:
        raise NotImplementedError

    @abc.abstractmethod
    def movePose(self, movingPose: Pose3):
        raise NotImplementedError

    def autoHorizon(self) -> bool:
        raise NotImplementedError

    def stopAutoHorizon(self) -> bool:
        raise NotImplementedError

    def goHome(self) -> bool:
        raise  NotImplementedError






