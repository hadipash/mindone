from abc import ABC, abstractmethod

import av
import cv2
import numpy as np


class _VideoReader(ABC):
    @abstractmethod
    def __enter__(self) -> "_VideoReader":
        ...

    @abstractmethod
    def __exit__(self, *args):
        ...

    @abstractmethod
    def __len__(self) -> int:
        ...

    @abstractmethod
    def get_frames(self, num: int = 0, start_pos: int = 0, step: int = 1) -> np.ndarray:
        ...


class OpenCVVideoReader(_VideoReader):
    """
    Extracts information about a video and reads frames in batches using OpenCV.
    Must be used with a context manager.

    Args:
        video_path (str): Path to the video file.

    Attributes:
        shape (Tuple[int, int]): The shape (width, height) of the video.
        fps (float): The frame rate of the video.

    Raises:
        IOError: If the video cannot be opened.

    Examples:
        >>> with OpenCVVideoReader("video.mp4") as reader:
        ...     width, height = reader.shape
        ...     fps = reader.fps
        ...     total_frames = len(reader)
        ...     frames = reader.get_frames(num=10, start_pos=10, step=2)
    """

    def __init__(self, video_path: str):
        self._video_path = video_path
        self._cap = None
        self.shape = (0, 0)
        self.fps = 0

    def __enter__(self) -> "OpenCVVideoReader":
        self._cap = cv2.VideoCapture(self._video_path, apiPreference=cv2.CAP_FFMPEG)
        if not self._cap.isOpened():
            raise IOError(f"Video {self._video_path} cannot be opened.")
        self.shape = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self._cap.get(cv2.CAP_PROP_FPS)
        return self

    def __exit__(self, *args):
        self._cap.release()

    def __len__(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def get_frames(self, num: int = 0, start_pos: int = 0, step: int = 1) -> np.ndarray:
        """
        Fetches a sequence of frames from the video starting at a specified position with a specified step.

        Parameters:
            num: The number of frames to fetch.
            start_pos: The frame index to start fetching from. Default: 0.
            step: The interval at which frames are fetched from the video: a step of N returns every Nth frame.
                  Default: 1 (no frame skipping).

        Returns:
            np.ndarray: An array containing the fetched frames.

        Raises:
            ValueError: If the requested number of frames exceeds the video length.
            RuntimeError: If the requested number of frames cannot be fetched.
        """
        min_len = (num - 1) * step + 1
        if len(self) < min_len:
            raise ValueError(f"Number of frames to fetch ({min_len}) must be less than video length ({len(self)}).")

        if start_pos:
            start_pos = min(start_pos, len(self) - min_len)
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, start_pos)

        frames = []
        i = start_pos
        ret, frame = self._cap.read()
        while ret and len(frames) < num:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if step > 1:
                i += step
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = self._cap.read()

        if len(frames) != num:
            raise RuntimeError(f"Failed to read {num} frames from {self._video_path}.")

        return np.stack(frames)


class PyAVVideoReader(_VideoReader):
    def __init__(self, video_path: str):
        self._video_path = video_path

    def __enter__(self) -> "PyAVVideoReader":
        self._container = av.open(self._video_path, metadata_errors="ignore")
        self._stream = self._container.streams.video[0]
        self.shape = self._stream.width, self._stream.height
        self.fps = float(self._stream.average_rate)
        return self

    def __exit__(self, *args):
        self._container.close()

    def __len__(self) -> int:
        return self._stream.frames

    def get_frames(self, num: int = 0, start_pos: int = 0, step: int = 1) -> np.ndarray:
        start_offset = int(num / self._stream.average_rate / self._stream.time_base)
        self._container.seek(start_offset, backward=True, any_frame=True)

        frames = []
        for i, frame in enumerate(self._container.decode(video=0)):
            if len(frames) == num:
                break
            elif not i % step:  # TODO: is there a more efficient way to do read batch of frames?
                frames.append(frame.to_rgb().to_ndarray())

        if len(frames) != num:
            raise RuntimeError(f"Failed to read {num} frames from {self._video_path}.")

        return np.stack(frames)
