from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
)
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.io import load_from_json
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.cameras.cameras import CameraType

from lse_nerf.lse_cameras import EdCameras as Cameras
from lse_nerf.lse_cameras import HardCamType

from dataclasses import dataclass, field
from pathlib import Path
from typing import Type
import os.path as osp
import numpy as np
import torch
import glob
import warnings
import json
import functools
from PIL import Image
from tqdm import tqdm

from lse_nerf.lse_dataset import EventFrameDataset, ColorDataset
from lse_nerf.utils import gbconfig


def load_from_json(filename):
    """Load a dictionary from a JSON filename.

    Args:
        filename: The filename to load from.
    """
    if not osp.exists(filename):
        CONSOLE.print(f"[yellow] {filename} does not exist")
        return

    assert osp.splitext(filename)[-1] == ".json", f"{osp.basename(filename)} is not json"
    with open(filename, encoding="UTF-8") as file:
        return json.load(file)


def cv_to_working(R):
    """
    R = opencv world to cam (4x4)

    return: gl cam to world
    """
    c2w = np.copy(R)
    mtx, pos = R[:3,:3], R[:3,3]

    pos = -mtx.T@pos
    right, up, forward = mtx
    mtx = np.stack([right, -up, -forward])
    c2w[:3,:3] = mtx.T
    c2w[:3,3] = pos

    return c2w
    

@dataclass
class CameraDataparserOutputs(DataparserOutputs):
    dataset_cls: InputDataset = InputDataset
    appearance_ids: list = None
    msk: np.ndarray = None
    dM: torch.Tensor = None

@dataclass
class CameraParserConfig(DataParserConfig):
    scale_factor: float = 1.0
    scene_scale: float = 1.0

@dataclass
class CameraParser(DataParser):

    def __init__(self, config: CameraParserConfig):
        super().__init__(config=config)
        self.data = config.data
        self.scale_factor = 1
        self.cam_translation = self._load_camera_transform()
        self.cam_data_json = self._load_all_cam_json()
        self.metadata = self._load_metadata() 
        self.appearance_ids = self._create_appearance_ids(self.metadata)
    
    def _load_msk(self, data_idxs=None):
        msk = None
        msk_f = osp.join(self.data, "msk.npy")
        if osp.exists(msk_f):
            msk = np.load(msk_f)
            if data_idxs is not None:
                msk = np.stack([msk[i] for i in data_idxs])
        
        return msk
    
    def _load_camera_transform(self):
        transform_f = osp.join(self.data, "camera_transform.json")
        if osp.exists(transform_f):
            with open(transform_f, "r") as f:
                transforms = json.load(f)
            return np.array(transforms["translation"])
        else:
            return None
    
    def _create_appearance_ids(self, metadata:dict):
        keys = sorted(list(metadata.keys()))
        return [metadata[k]["appearance_id"] for k in keys]

    def _load_all_cam_json(self, cam_dir = None, idxs = None):
        if cam_dir is None:
            cam_dir = osp.join(self.data, "camera")

        if not osp.exists(cam_dir):
            return None

        fs = sorted(glob.glob(osp.join(cam_dir, "*.json")))
        if idxs is not None:
            tmp_fs = []
            for i in idxs:
                if i < len(fs):
                    tmp_fs.append(fs[i])
                else:
                    # NOTE: if i==len(fs); its fine
                    CONSOLE.print(f"[yellow] warning {i} out of index of list of size {len(fs)}")
            fs = tmp_fs

        return [load_from_json(f) for f in fs]

    def _load_metadata(self):
        meta_f = osp.join(self.data, "metadata.json")
        meta = load_from_json(meta_f)
        new_meta = {}
        for k, v in meta.items():
            try:
                img_id = int(k)
            except Exception as e:
                CONSOLE.print(f"[yellow] {str(e)}")
                continue

            v["img_id"] = img_id
            new_meta[img_id] = v
        return new_meta

    def _format_camera(self, data, cam_type, calc_dm=False):
        """
        cam_type (int) = 0: rgb, 1: evs
        """
        mtxs = np.tile(np.eye(4)[None], (len(data), 1, 1)).astype(np.float32)
        ori_mtxs = np.zeros((len(data), 4, 4), dtype=np.float32)
        times = None
        for i, datum in enumerate(data):
            mtx_ori = np.array(datum["orientation"])
            pos = np.array(datum["position"]).reshape(3, 1) # position in world space, T_c2w
            if self.cam_translation is not None:
                pos = pos + self.cam_translation
                

            w2c = np.concatenate([mtx_ori, -mtx_ori@pos], axis=1)
            w2c = np.concatenate([w2c, np.array([[0, 0, 0, 1]])], 0)
            ori_mtxs[i] = np.copy(w2c)


            c2w = cv_to_working(w2c)

            # corrected
            mtxs[i, :3,:4] = c2w[:3,:4]

            if datum.get("t") is not None:
                if times is None:
                    times = [float(datum["t"])]
                else:
                    times.append(float(datum["t"]))
        
        dM = None
        if load_from_json(osp.join(self.data, "metadata.json")).get("colmap_scale") is not None:
            dM = self._get_rel_cam(ori_mtxs)
        

        mtxs[:,:3, 3] = mtxs[:,:3, 3] * self.config.scale_factor

        datum = data[0]
        cx, cy = datum["principal_point"]
        w, h = datum["image_size"]
        k1,k2,k3 = datum["radial_distortion"]
        p1, p2 = datum["tangential_distortion"]
        distortion = torch.tensor((k1, k2, k3, 0, p1, p2))
        cams = Cameras(
            camera_to_worlds=torch.from_numpy(mtxs)[:, :3, :4],
            fx=datum["focal_length"],
            fy=datum["focal_length"],
            cx=cx,
            cy=cy,
            width = w,
            height=h,
            distortion_params=None if distortion.sum() == 0 else distortion, #(k1,k2,k3,k4,p1,p2)
            camera_type=CameraType.PERSPECTIVE,
            times=torch.tensor(times, dtype=torch.float32) if times is not None else None
        )
        cams.set_hard_cam_type(cam_type)

        if calc_dm:
            return cams, dM
        return cams
    
    def _get_rel_cam(self, mtx:np.ndarray):
        """
        mtx: (n, 4, 4) opencv homegeneous camera to world cameras

        return:
            dR (4,4): relative extrinsics in opengl space
                - R_{ev gl c2w} = R_{rgb gl c2w} @ dR
        """
        Mrs = mtx

        # colmap_scale = np.loadtxt(osp.join(osp.dirname(self.data), "colmap_scale.txt"))
        colmap_scale = load_from_json(osp.join(self.data, "metadata.json"))["colmap_scale"]
        relcam_f = osp.join(osp.dirname(self.data), "rel_cam.json")
        with open(relcam_f, "r") as f:
            data = json.load(f)
            R = np.array(data["R"])
            T = np.array(data["T"]) * colmap_scale
        dM = np.concatenate([R, T.reshape(-1,1)], axis=1)
        dM = np.concatenate([dM, np.array([[0, 0, 0, 1]])], 0)

        Mes = np.stack([dM@Mr for Mr in Mrs])

        Mrgs = np.stack([cv_to_working(m) for m in Mrs])  # rgb c2w extrinsics in gl
        Megs = np.stack([cv_to_working(m) for m in Mes])  # evs c2w extrinsics in gl

        Mrgs[:,:3,3] = Mrgs[:,:3,3] * self.config.scale_factor  # apply config.scale_factor
        Megs[:,:3,3] = Megs[:,:3,3] * self.config.scale_factor

        dr1 = np.linalg.inv(Mrgs[0])@Megs[0]

        if len(Mrgs) > 1:
            dr2 = np.linalg.inv(Mrgs[5])@Megs[5]
            assert (np.abs(dr1 - dr2) < 1e-6).all(), "gl relative extrinsics calculated wrong!"

        return torch.tensor(dr1).float()

    
    def _get_scene_box(self):
        sc = self.config.scene_scale
        return SceneBox(aabb=torch.tensor([[-sc, -sc, -sc], 
                                           [sc, sc, sc]], dtype=torch.float32))


    def get_max_appearence_id(self):
        app_ids = [v["appearance_id"] for (k,v) in self.metadata.items()]
        return max(app_ids) + 1

@dataclass
class EventDataParserConfig(CameraParserConfig):
    _target: Type = field(default_factory=lambda: Events)
    data: Path = Path("")
    scale_factor: float = 1.0
    """Directory specifying location of data."""
    downscale_factor: int = 1
    """how much to downscale images; option is obselete"""
    e_thresh:str = None
    """use this e_thresh instead of one saved in data set"""
    dataset_cls: EventFrameDataset = EventFrameDataset
    """event data initialization class; DO NOT TOUCH"""

    event_type:str = None

    def __post_init__(self):
        if type(self.e_thresh) is str:
            if self.e_thresh.lower() == "none":
                self.e_thresh = None
            else:
                self.e_thresh = float(self.e_thresh)
                
        if (self.event_type is not None) and (self.event_type.lower() == "none"):
            self.event_type = None


@dataclass
class EventsparserOutputs(CameraDataparserOutputs):
    events: np.ndarray = None
    e_thresh: float = None
    prev_cameras: Cameras = None
    next_cameras: Cameras = None

@dataclass
class Events(CameraParser):
    config: EventDataParserConfig

    def __init__(self, config:EventDataParserConfig):
        if config.event_type is not None:
            data_path = str(config.data)
            config.data = Path(osp.join(osp.dirname(data_path), config.event_type))
        super().__init__(config=config)
        self.scale_factor = config.scale_factor
    
    def create_formatted_camera(self, idxs):
        
        prev_cam_dir, next_cam_dir = osp.join(self.data, "prev_camera"), osp.join(self.data, "next_camera")

        if osp.exists(prev_cam_dir):
            prev_cam_json, next_cam_json = self._load_all_cam_json(prev_cam_dir, idxs=idxs), self._load_all_cam_json(next_cam_dir, idxs=idxs)

            return self._format_camera(prev_cam_json, cam_type=HardCamType.EVS), \
                self._format_camera(prev_cam_json, cam_type=HardCamType.EVS), \
                self._format_camera(next_cam_json, cam_type=HardCamType.EVS)
        else:
            return self._format_camera(self.cam_data_json, cam_type=HardCamType.EVS), None, None
    
    
    def _load_events(self, idxs):
        src_events = np.load(self.data/"eimgs"/f"eimgs_1x.npy", "r")
        
        events = np.zeros((len(idxs), *src_events.shape[1:]), dtype=src_events.dtype)
        for i, idx in tqdm(enumerate(idxs), desc="loading events", total=len(idxs)):
            events[i] = src_events[idx]

        events = events[..., None]
        return events

    def _generate_dataparser_outputs(self, split: str = "train") -> DataparserOutputs:
        if split != "train":
            warnings.warn(f"event camera dataset support only, split is {split}")

        dataset_meta = load_from_json(osp.join(self.data, "dataset.json"))
        data_idxs = sorted(int(e) for e in dataset_meta["train_ids"])

        cameras, prev_cameras, next_cameras = self.create_formatted_camera(idxs = data_idxs)
        events = self._load_events(data_idxs)
        appearance_ids = [self.appearance_ids[idx] for idx in data_idxs]

        scene_box =  self._get_scene_box()
        scene_data = load_from_json(osp.join(self.data, "scene.json"))

        e_thresh = 0.2
        if (scene_data is not None) and scene_data.get("e_thresh") is not None:
            e_thresh = scene_data["e_thresh"]

        if self.config.e_thresh is not None: # overwrite dataset e_thresh if provided
            e_thresh = self.config.e_thresh
        
        msk = self._load_msk()
        if self.config.event_type == "decam_set":
            e_thresh = 1

        return EventsparserOutputs(
            image_filenames=None,
            events = events,
            cameras=cameras,
            scene_box=scene_box,
            dataparser_scale=self.scale_factor,
            e_thresh = e_thresh,
            dataset_cls = self.config.dataset_cls,
            appearance_ids=appearance_ids,
            msk=msk,
            prev_cameras=prev_cameras,
            next_cameras=next_cameras
        )


@dataclass
class ColorDataParserConfig(CameraParserConfig):
    _target: Type = field(default_factory=lambda: Color)
    data: Path = Path("data/ShakeCarpet1_formatted/colcam_set") #Path("data/dnerf/lego")
    scale_factor: float = 1.0
    """How much to scale the origin by"""
    downscale_factor: int = 1
    """how much to downscale images; option is obselete"""
    image_type: str = "gamma"
    """one of [linear, gamma]; option is obselete"""
    quality: str = "clear"
    """one of [clear, blur]; option is obselete"""
    dataset_cls: InputDataset = ColorDataset
    """dataset to create; DON'T TOUCH"""
    use_gray: bool = False
    """use grayscale images or not"""

    def __post_init__(self):
        self.dataset_cls = functools.partial(ColorDataset, use_gray=self.use_gray)


@dataclass
class Color(CameraParser):
    config:ColorDataParserConfig

    def __init__(self, config:ColorDataParserConfig):
        super().__init__(config=config)
        self.split_dic:dict = {"train":"train_ids", "test":"val_ids", "val": "val_ids"} # TODO: make a val ids list
        self.scale_factor = config.scale_factor
        self.all_img_fs = sorted(glob.glob(osp.join(self.data, "rgb", "1x", "*.[pj][np]g")))
    
    def _get_img_dir(self, *nargs):
        dir_prefix = ""

        for e in nargs:
            if e is not None and e != "":
                dir_prefix = dir_prefix + f"{e}_"

        base_dir = osp.dirname(self.data)
        colcam_dir = osp.join(base_dir, dir_prefix + "colcam_set")
        if osp.exists(colcam_dir):
            return colcam_dir
        else:
            CONSOLE.print("[yellow] colcam quality and image_type provided but dataset does not exist, loading default colcam_set")
            return osp.join(base_dir, "colcam_set") #self.data


    def _generate_dataparser_outputs(self, split: str = "train", spec_data_idxs = None) -> DataparserOutputs:
        quality = self.config.quality if split == "train" else "clear"   # eval, val on clear
        img_dir = self._get_img_dir(quality, self.config.image_type)

        self.config.data = img_dir
        self.__init__(self.config)

        self.dataset_meta = load_from_json(osp.join(self.data,"dataset.json"))

        if split == "train" and gbconfig.IS_EVAL and self.dataset_meta.get("half_train_ids") is not None:
            id_key = "half_train_ids"
        else:
            split = "val" if (gbconfig.IS_EVAL and not gbconfig.DO_PRETRAIN) else split     # NeRF model frozen, only optimize on train cameras
            id_key = self.split_dic[split]

        img_fs = sorted(glob.glob(osp.join(img_dir, "rgb", "1x", "*.[pj][np]g")))  # jpg or png
        data_idxs = sorted(int(e) for e in self.dataset_meta[id_key]) if spec_data_idxs is None else spec_data_idxs
        data_idxs = [idx for idx in data_idxs if idx < len(img_fs) - 1]

        cam_data = self.cam_data_json 
        cam_data = [cam_data[idx] for idx in data_idxs]
        appearance_ids = [self.appearance_ids[idx] for idx in data_idxs]
        msk = self._load_msk(data_idxs)
 

        CONSOLE.print(f"[yellow] loading {split} images from {img_dir}")
        CONSOLE.print(f"[yellow] number of {split} images: {len(data_idxs)}")

        img_fs = sorted(glob.glob(osp.join(img_dir, "rgb", "1x", "*.[pj][np]g")))
        img_fs = [img_fs[idx] for idx in data_idxs]
        scene_box = self._get_scene_box() 

        cameras, dM = self._format_camera(cam_data, cam_type=HardCamType.RGB, calc_dm=True)
        return CameraDataparserOutputs(
            image_filenames=img_fs,
            cameras=cameras,
            alpha_color=None,
            scene_box=scene_box,
            dataparser_scale=self.scale_factor,
            dataset_cls = self.config.dataset_cls,
            appearance_ids=appearance_ids,
            msk=msk,
            dM=dM
        )
    
    def get_num_train(self):
        return len(self.dataset_meta[self.split_dic["train"]])
    
    
    def get_train_ids(self):
        return sorted(int(e) for e in self.dataset_meta["train_ids"] if int(e) < len(self.all_img_fs) - 1)

    def get_all_cameras(self):
        full_traj_dir = osp.join(self.data, "full_camera")

        if osp.exists(full_traj_dir):
            cam_data = self._load_all_cam_json(full_traj_dir)
        else:
            cam_data = self.cam_data_json[:-1]
            
        return self._format_camera(cam_data, cam_type=HardCamType.RGB)
    

    def get_train_ts(self):
        data_idxs = sorted(int(e) for e in self.dataset_meta["train_ids"])
        img_fs = sorted(glob.glob(osp.join(self.config.data, "rgb", "1x", "*.[pj][np]g")))
        data_idxs = [idx for idx in data_idxs if idx < len(img_fs) - 1]
        cam_data = [self.cam_data_json[idx] for idx in data_idxs]

        if cam_data[0].get("t") is None:
            return None
        
        return torch.tensor([cam_datum["t"] for cam_datum in cam_data], dtype=torch.float32)

    def get_img(self, idx=0):
        return np.array(Image.open(self.all_img_fs[idx]))