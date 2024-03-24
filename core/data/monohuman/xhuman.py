from cgi import print_directory
from curses import keyname
from email.policy import strict
from http.client import NON_AUTHORITATIVE_INFORMATION
from ntpath import join
from operator import imod
import os
import pickle
from pickletools import optimize
import random
from select import select
from sys import path
from tabnanny import verbose
from tkinter.messagebox import NO
from turtle import color
from typing import KeysView
from unicodedata import name
from cv2 import norm
from core.utils.network_util import MotionBasisComputer
from torch import optim
import os.path as osp
import imageio


import numpy as np
import cv2
import torch
import torch.utils.data

import trimesh

from smplx_modified.body_models import SMPLX
import torch
import math
from typing import List, Any
from typing import NamedTuple
from pathlib import Path
import copy
from plyfile import PlyData
from PIL import Image



from core.utils.image_util import load_image
from core.utils.body_util import \
    body_pose_to_body_RTs, \
    get_canonical_global_tfms, \
    approx_gaussian_bone_volumes
from core.utils.file_util import list_files, split_path
from core.utils.camera_util import \
    apply_global_tfm_to_camera, \
    get_rays_from_KRT, \
    rays_intersect_3d_bbox

from configs import cfg
from third_parties.smpl.smpl_numpy import SMPL
from tqdm import trange
MODEL_DIR = 'third_parties/smpl/models'
from core.nets.monohuman.network import Network



def get_bound_2d_mask(bounds, K, pose, H, W):
    corners_3d = get_bound_corners(bounds)
    corners_2d = project(corners_3d, K, pose)
    corners_2d = np.round(corners_2d).astype(int)
    corners_2d[:,0] = np.clip(corners_2d[:,0], 0, W)
    corners_2d[:,1] = np.clip(corners_2d[:,1], 0, H)
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [corners_2d[[0, 1, 3, 2, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[4, 5, 7, 6, 4]]], 1) # 4,5,7,6,4
    cv2.fillPoly(mask, [corners_2d[[0, 1, 5, 4, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[2, 3, 7, 6, 2]]], 1)
    cv2.fillPoly(mask, [corners_2d[[0, 2, 6, 4, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[1, 3, 7, 5, 1]]], 1)
    return mask

def get_bound_corners(bounds):
    min_x, min_y, min_z = bounds[0]
    max_x, max_y, max_z = bounds[1]
    corners_3d = np.array([
        [min_x, min_y, min_z],
        [min_x, min_y, max_z],
        [min_x, max_y, min_z],
        [min_x, max_y, max_z],
        [max_x, min_y, min_z],
        [max_x, min_y, max_z],
        [max_x, max_y, min_z],
        [max_x, max_y, max_z],
    ])
    return corners_3d

def project(xyz, K, RT):
    """
    xyz: [N, 3]
    K: [3, 3]
    RT: [3, 4]
    """
    xyz = np.dot(xyz, RT[:, :3].T) + RT[:, 3:].T
    xyz = np.dot(xyz, K.T)
    xy = xyz[:, :2] / xyz[:, 2:]
    return xy

def load_smpl_param(path):
    smpl_params = dict(np.load(str(path)))
    if "thetas" in smpl_params:
        smpl_params["body_pose"] = smpl_params["thetas"][..., 3:]
        smpl_params["global_orient"] = smpl_params["thetas"][..., :3]
    return {
        "betas": smpl_params["betas"].astype(np.float32).reshape(1, 10),
        "body_pose": smpl_params["body_pose"].astype(np.float32),
        "global_orient": smpl_params["global_orient"].astype(np.float32),
        "transl": smpl_params["transl"].astype(np.float32),
    }

def get_camera_extrinsics_zju_mocap_refine(view_index, val=False, camera_view_num=36):
    def norm_np_arr(arr):
        return arr / np.linalg.norm(arr)

    def lookat(eye, at, up):
        zaxis = norm_np_arr(at - eye)
        xaxis = norm_np_arr(np.cross(zaxis, up))
        yaxis = np.cross(xaxis, zaxis)
        _viewMatrix = np.array([
            [xaxis[0], xaxis[1], xaxis[2], -np.dot(xaxis, eye)],
            [yaxis[0], yaxis[1], yaxis[2], -np.dot(yaxis, eye)],
            [-zaxis[0], -zaxis[1], -zaxis[2], np.dot(zaxis, eye)],
            [0       , 0       , 0       , 1     ]
        ])
        return _viewMatrix
    
    def fix_eye(phi, theta):
        camera_distance = 3
        return np.array([
            camera_distance * np.sin(theta) * np.cos(phi),
            camera_distance * np.sin(theta) * np.sin(phi),
            camera_distance * np.cos(theta)
        ])

    if val:
        eye = fix_eye(np.pi + 2 * np.pi * view_index / camera_view_num + 1e-6, np.pi/2 + np.pi/12 + 1e-6).astype(np.float32) + np.array([0, 0, -0.8]).astype(np.float32)
        at = np.array([0, 0, -0.8]).astype(np.float32)

        extrinsics = lookat(eye, at, np.array([0, 0, -1])).astype(np.float32)
    return extrinsics

class CameraInfo(NamedTuple):
    uid: int
    pose_id: int
    R: np.array
    T: np.array
    K: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    bkgd_mask: np.array
    bound_mask: np.array
    width: int
    height: int
    smpl_param: dict
    world_vertex: np.array
    world_bound: np.array
    big_pose_smpl_param: dict
    big_pose_world_vertex: np.array
    big_pose_world_bound: np.array
    bbox: np.array
    smplx_model_out: Any




class Dataset(torch.utils.data.Dataset):
    def __init__(
            self, 
            dataset_path,
            index_a,
            index_b,
            keyfilter=None,
            maxframes=-1,
            bgcolor=None,
            ray_shoot_mode='image',
            skip=1,
            data_type='train',
            **_):

        print('[Dataset Path]', dataset_path) 

        self.ray_shoot_mode = ray_shoot_mode

        self.smpl_model = SMPL(sex='neutral', model_dir=MODEL_DIR)

        self.dataset_path = dataset_path

        data_root = dataset_path

        all_views = [os.path.join(data_root, 'train', p) for p in os.listdir(os.path.join(data_root, 'train'))] + \
                        [os.path.join(data_root, 'test', p) for p in os.listdir(os.path.join(data_root, 'test'))]
        all_views = sorted(all_views)
        train_view = [all_views[0]]
        # train_view = [6]
        test_view = copy.deepcopy(all_views)
        test_view.remove(train_view[0])

        split = data_type

        if split == 'train':
            views = train_view
        elif split == "progress":
            views = test_view[:1]
        elif split == "test":
            views = test_view

        print(f"{split} views are: {views}")




        pose_start = 0
        if split == 'train':
            pose_interval = 1
            num_img = os.listdir(osp.join(data_root, views[0], 'render', 'image'))
            pose_num = len(num_img)
            pose_interval = 50 # for debug
        elif split == 'test':
            pose_start = 0
            pose_interval = 5
            pose_num = 20
        elif split == 'progress':
            pose_start = 0
            pose_interval = 5
            pose_num = 20

        image_zoom_ratio = 1

        self.image_zoom_ratio = image_zoom_ratio

        cams_dict = {}
        ims_dict = {}
        ims_name_dict = {}

        for view in views:
            # root = osp.join(data_root, view)
            root = view
            cam_numpy = np.load(os.path.join(view, 'render', 'cameras.npz'), allow_pickle=True)

            extr = cam_numpy['extrinsic']
            intr = cam_numpy['intrinsic']
            cams = {
                'K': intr,
                'R': extr[:,:3,:3],
                'T': extr[:,:3,3:4],
            }
            cams_dict[view] = cams

            ims_list = sorted(os.listdir(os.path.join(root, 'render/image')))
            ims = np.array([
                np.array(os.path.join(root, 'render/image', im))
                for im in ims_list[pose_start:pose_start + pose_num * pose_interval][::pose_interval]
            ])
            pose_num = len(ims)
            ims_dict[view] = ims

            img_name_list = sorted(os.listdir(os.path.join(root, 'render', 'image')))
            # img_name_list = [img_name
            #                  for img_name in img_name_list[pose_start:pose_start + pose_num * pose_interval][::pose_interval]]
            ims_name_dict[view] = img_name_list

        SMPLX_PKL_PATH = "/home/tw554/GART/models/smplx"
        smplx_zoo = {
            'male': SMPLX(model_path=f'{SMPLX_PKL_PATH}/SMPLX_MALE.pkl', ext='pkl',
                            use_face_contour=True, flat_hand_mean=False, use_pca=False,
                            num_betas=10, num_expression_coeffs=10),
            'female': SMPLX(model_path=f'{SMPLX_PKL_PATH}/SMPLX_FEMALE.pkl', ext='pkl',
                        use_face_contour=True, flat_hand_mean=False, use_pca=False,
                        num_betas=10, num_expression_coeffs=10),
        }
        with open(os.path.join(data_root, 'gender.txt'), 'r') as f:
            gender = f.readline()
        gender = gender.strip()
        smplx_model = smplx_zoo[gender]
        # '/home/hh29499/Datasets/X_Human/00016/mean_shape_smplx.npy'
        smpl_param_path = os.path.join(data_root, 'mean_shape_smplx.npy')
        template_shape = np.load(smpl_param_path)
        # SMPL in canonical space
        big_pose_smpl_param = {}
        big_pose_smpl_param['R'] = np.eye(3).astype(np.float32)
        big_pose_smpl_param['Th'] = np.zeros((1, 3)).astype(np.float32)
        big_pose_smpl_param['global_orient'] = np.zeros((1, 3)).astype(np.float32)
        big_pose_smpl_param['betas'] = np.zeros((1, 10)).astype(np.float32)
        big_pose_smpl_param['betas'] = template_shape[None].astype(np.float32)
        big_pose_smpl_param['body_pose'] = np.zeros((1, 63)).astype(np.float32)
        big_pose_smpl_param['jaw_pose'] = np.zeros((1, 3)).astype(np.float32)
        big_pose_smpl_param['left_hand_pose'] = np.zeros((1, 45)).astype(np.float32)
        big_pose_smpl_param['right_hand_pose'] = np.zeros((1, 45)).astype(np.float32)
        big_pose_smpl_param['leye_pose'] = np.zeros((1, 3)).astype(np.float32)
        big_pose_smpl_param['reye_pose'] = np.zeros((1, 3)).astype(np.float32)
        big_pose_smpl_param['expression'] = np.zeros((1, 10)).astype(np.float32)
        big_pose_smpl_param['transl'] = np.zeros((1, 3)).astype(np.float32)
        big_pose_smpl_param['body_pose'][0, 2] = 45 / 180 * np.array(np.pi)
        big_pose_smpl_param['body_pose'][0, 5] = -45 / 180 * np.array(np.pi)
        big_pose_smpl_param['body_pose'][0, 20] = -30 / 180 * np.array(np.pi)
        big_pose_smpl_param['body_pose'][0, 23] = 30 / 180 * np.array(np.pi)
        big_pose_smpl_param_tensor = {}
        for key in big_pose_smpl_param.keys():
            big_pose_smpl_param_tensor[key] = torch.from_numpy(big_pose_smpl_param[key])
        body_model_output = smplx_model(
            global_orient=big_pose_smpl_param_tensor['global_orient'],
            betas=big_pose_smpl_param_tensor['betas'],
            body_pose=big_pose_smpl_param_tensor['body_pose'],
            jaw_pose=big_pose_smpl_param_tensor['jaw_pose'],
            left_hand_pose=big_pose_smpl_param_tensor['left_hand_pose'],
            right_hand_pose=big_pose_smpl_param_tensor['right_hand_pose'],
            leye_pose=big_pose_smpl_param_tensor['leye_pose'],
            reye_pose=big_pose_smpl_param_tensor['reye_pose'],
            expression=big_pose_smpl_param_tensor['expression'],
            transl=big_pose_smpl_param_tensor['transl'],
            return_full_pose=True,
        )
        big_pose_smpl_param['poses'] = body_model_output.full_pose.detach()
        big_pose_smpl_param['shapes'] = np.concatenate([big_pose_smpl_param['betas'], big_pose_smpl_param['expression']],
                                                    axis=-1)
        big_pose_xyz = np.array(body_model_output.vertices.detach()).reshape(-1, 3).astype(np.float32)

        self.t_joints = body_model_output.original_joints[0].numpy()


        big_pose_min_xyz = np.min(big_pose_xyz, axis=0)
        big_pose_max_xyz = np.max(big_pose_xyz, axis=0)
        big_pose_min_xyz -= 0.05
        big_pose_max_xyz += 0.05
        big_pose_world_bound = np.stack([big_pose_min_xyz, big_pose_max_xyz], axis=0)


        idx = 0
        cam_infos = []
        novel_view_vis = False
        white_background = False
        for pose_index in range(pose_num):
            # pose_index = 100
            for view in views:

                if novel_view_vis:
                    view_index_look_at = view
                    view = 0

                # Load image, mask, K, D, R, T
                # try:
                # image_path = os.path.join(path, str(ims_dict[view][pose_index]).replace('\\', '/'))
                image_path = str(ims_dict[view][pose_index]).replace('\\', '/')
                # except:
                #     a = 1
                #     print('error')
                if 'train' in image_path:
                    split_flag = 'train'
                else:
                    split_flag = 'test'
                image_name = ims_dict[view][pose_index].split('.')[0]
                image = np.array(imageio.imread(image_path).astype(np.float32) / 255.)

                # msk_path = image_path.replace('image', 'mask').replace('jpg', 'png')
                # msk = imageio.imread(msk_path)
                # msk = (msk != 0).astype(np.uint8)

                msk_path = image_path.replace('image', 'mask_new').replace('jpg', 'png')
                msk_alpha = imageio.imread(msk_path)
                msk_alpha = msk_alpha / 255.
                # msk = (msk_alpha != 0).astype(np.uint8)
                msk = (msk_alpha > 0.3).astype(np.uint8)

                if not novel_view_vis:
                    K = np.array(cams_dict[view]['K'])
                    # D = np.array(cams['D'])
                    # R = np.array(cams_dict[view_index]['R'][int(image_name.split('_')[-1])-1])
                    # T = np.array(cams_dict[view_index]['T'][int(image_name.split('_')[-1])-1]) #/ 1000.
                    R = np.array(cams_dict[view]['R'][ims_name_dict[view].index(image_name.split('/')[-1]+'.png')])
                    T = np.array(cams_dict[view]['T'][ims_name_dict[view].index(image_name.split('/')[-1]+'.png')])  # / 1000.
                    # image = cv2.undistort(image, K)
                    # msk = cv2.undistort(msk, K)
                else:
                    raise NotImplementedError()
                    pose = np.matmul(np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]),
                                    get_camera_extrinsics_zju_mocap_refine(view_index_look_at, val=True))
                    R = pose[:3, :3]
                    T = pose[:3, 3].reshape(-1, 1)
                    cam_ind = cam_inds[pose_index][view]
                    K = np.array(cams['K'][cam_ind])

                # image[msk == 0] = 1 if white_background else 0
                image = image * msk_alpha[..., None]

                # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
                w2c = np.eye(4)
                w2c[:3, :3] = R
                w2c[:3, 3:4] = T

                # get the world-to-camera transform and set R, T
                R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
                T = w2c[:3, 3]

                # Reduce the image resolution by ratio, then remove the back ground
                ratio = image_zoom_ratio
                # ratio = 1
                if ratio != 1.:
                    H, W = int(image.shape[0] * ratio), int(image.shape[1] * ratio)
                    image = cv2.resize(image, (W, H), interpolation=cv2.INTER_AREA)
                    msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)
                    K[:2] = K[:2] * ratio

                image = Image.fromarray(np.array(image * 255.0, dtype=np.byte), "RGB")

                focalX = K[0, 0]
                focalY = K[1, 1]
                def focal2fov(focal, pixels):
                    return 2*math.atan(pixels/(2*focal))
                FovX = focal2fov(focalX, image.size[0])
                FovY = focal2fov(focalY, image.size[1])

                # load smplx data 'mesh-f00001_smplx'
                id = os.path.basename(image_path).split('.')[0].split('_')[1]
                vertices_path = os.path.join(view,
                                            'SMPLX', 'mesh-f'+id[1:]+'_smplx.ply')
                vert_data = PlyData.read(vertices_path)
                vertices = vert_data['vertex']
                xyz = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T

                smpl_param_path = os.path.join(view,
                                            'SMPLX', 'mesh-f'+id[1:]+'_smplx.pkl')
                with open(smpl_param_path, 'rb') as f:
                    smpl_param = pickle.load(f)

                ###
                # load smpl data
                smpl_param = {
                    'global_orient': np.expand_dims(smpl_param['global_orient'].astype(np.float32), axis=0),
                    'transl': np.expand_dims(smpl_param['transl'].astype(np.float32), axis=0),
                    'body_pose': np.expand_dims(smpl_param['body_pose'].astype(np.float32), axis=0),
                    'jaw_pose': np.expand_dims(smpl_param['jaw_pose'].astype(np.float32), axis=0),
                    'betas': np.expand_dims(smpl_param['betas'].astype(np.float32), axis=0),
                    'expression': np.expand_dims(smpl_param['expression'].astype(np.float32), axis=0),
                    'leye_pose': np.expand_dims(smpl_param['leye_pose'].astype(np.float32), axis=0),
                    'reye_pose': np.expand_dims(smpl_param['reye_pose'].astype(np.float32), axis=0),
                    'left_hand_pose': np.expand_dims(smpl_param['left_hand_pose'].astype(np.float32), axis=0),
                    'right_hand_pose': np.expand_dims(smpl_param['right_hand_pose'].astype(np.float32), axis=0),
                    }
                smpl_param['R'] = np.eye(3).astype(np.float32)
                smpl_param['Th'] = smpl_param['transl'].astype(np.float32)
                smpl_param_tensor = {}
                for key in smpl_param.keys():
                    smpl_param_tensor[key] = torch.from_numpy(smpl_param[key])
                body_model_output = smplx_model(
                    global_orient=smpl_param_tensor['global_orient'],
                    betas=smpl_param_tensor['betas'],
                    body_pose=smpl_param_tensor['body_pose'],
                    jaw_pose=smpl_param_tensor['jaw_pose'],
                    left_hand_pose=smpl_param_tensor['left_hand_pose'],
                    right_hand_pose=smpl_param_tensor['right_hand_pose'],
                    leye_pose=smpl_param_tensor['leye_pose'],
                    reye_pose=smpl_param_tensor['reye_pose'],
                    expression=smpl_param_tensor['expression'],
                    transl=smpl_param_tensor['transl'],
                    return_full_pose=True,
                )
                smpl_param['poses'] = body_model_output.full_pose.detach()
                smpl_param['shapes'] = np.concatenate([smpl_param['betas'], smpl_param['expression']], axis=-1)

                # from nosmpl.vis.vis_o3d import vis_mesh_o3d
                # vertices = body_model_output.vertices.squeeze()
                # faces = smplx_model.faces.astype(np.int32)
                # vis_mesh_o3d(vertices.detach().cpu().numpy(), faces)
                # vis_mesh_o3d(xyz, faces)
                ###

                # obtain the original bounds for point sampling
                min_xyz = np.min(xyz, axis=0)
                max_xyz = np.max(xyz, axis=0)
                min_xyz -= 0.05
                max_xyz += 0.05
                world_bound = np.stack([min_xyz, max_xyz], axis=0)

                # xy = get_2dkps(xyz, K, w2c[:3], image.size[1], image.size[0])
                # import matplotlib.pyplot as plt
                # plt.figure(figsize=(6.4, 3.6))
                # plt.scatter(xy[:, 0].tolist(), xy[:, 1].tolist())
                # plt.show()

                # get bounding mask and bcakground mask
                bound_mask = get_bound_2d_mask(world_bound, K, w2c[:3], image.size[1], image.size[0])
                bound_mask = Image.fromarray(np.array(bound_mask * 255.0, dtype=np.byte))

                try:
                    bkgd_mask = Image.fromarray(np.array(msk * 255.0, dtype=np.byte))
                except:
                    bkgd_mask = Image.fromarray(np.array(msk[:,:,0] * 255.0, dtype=np.byte))

                bkgd_mask = np.array(bkgd_mask) / 255.

                bbox = self.skeleton_to_bbox(body_model_output.original_joints[0].numpy())


                cam_infos.append(CameraInfo(uid=idx, pose_id=pose_index, R=R, T=T, K=K, FovY=FovY, FovX=FovX, image=np.array(image),
                                            image_path=image_path, image_name=image_name, bkgd_mask=bkgd_mask,
                                            bound_mask=bound_mask, width=image.size[0], height=image.size[1],
                                            smpl_param=smpl_param, world_vertex=xyz, world_bound=world_bound,
                                            big_pose_smpl_param=big_pose_smpl_param, big_pose_world_vertex=big_pose_xyz,
                                            big_pose_world_bound=big_pose_world_bound,
                                            bbox=bbox,
                                            smplx_model_out=body_model_output,
                                            ))

                idx += 1


            # break
                
        bgcolor = np.zeros(3)

        self.cam_infos: List[CameraInfo] = cam_infos


        self.index_a = index_a
        self.index_b = min(index_b, len(self.cam_infos)-1)
        self.keyfilter = keyfilter
        self.bgcolor = bgcolor

        img_a = self.cam_infos[self.index_a].image
        img_a = (img_a / 255.).astype('float32')
        img_b = self.cam_infos[self.index_b].image
        img_b = (img_b / 255.).astype('float32')
        self.src_img = np.array([img_a, img_b])

        self.canonical_joints = self.cam_infos[0].smplx_model_out.original_joints[0].numpy()
        self.canonical_bbox = self.skeleton_to_bbox(self.canonical_joints)





        self.in_K = []
        self.in_E = []

        self.in_dst_poses = []
        self.in_dst_tposes_joints = []

        self.in_index = [self.index_a, self.index_b]

        for in_idx in self.in_index:


            K_ = self.cam_infos[in_idx].K.copy()
            K_[:2] *= cfg.resize_img_scale

            E_ = np.eye(4)
            E_[:3,:3] = self.cam_infos[in_idx].R
            E_[:3,3] = self.cam_infos[in_idx].T
            pose_ = self.cam_infos[in_idx].smpl_param['poses']
            tpose_joints_ = self.t_joints

            self.in_K.append(K_.astype('float32'))
            self.in_E.append(E_.astype('float32'))

            self.in_dst_poses.append(pose_.numpy().reshape([-1,3]))
            self.in_dst_tposes_joints.append(tpose_joints_)
        



        if 'motion_weights_priors' in keyfilter:
            self.motion_weights_priors = \
                approx_gaussian_bone_volumes(
                    self.canonical_joints,   
                    self.canonical_bbox['min_xyz'],
                    self.canonical_bbox['max_xyz'],
                    grid_size=cfg.mweight_volume.volume_size).astype('float32')
            





    def load_canonical_joints(self):
        cl_joint_path = os.path.join(self.dataset_path, 'canonical_joints.pkl')
        with open(cl_joint_path, 'rb') as f:
            cl_joint_data = pickle.load(f)
        canonical_joints = cl_joint_data['joints'].astype('float32')
        canonical_bbox = self.skeleton_to_bbox(canonical_joints)

        return canonical_joints, canonical_bbox

    def load_train_cameras(self):
        with open(os.path.join(self.dataset_path, 'cameras.pkl'), 'rb') as f:
            cameras = pickle.load(f)
        return cameras

    @staticmethod
    def skeleton_to_bbox(skeleton):
        min_xyz = np.min(skeleton, axis=0) - cfg.bbox_offset
        max_xyz = np.max(skeleton, axis=0) + cfg.bbox_offset

        return {
            'min_xyz': min_xyz,
            'max_xyz': max_xyz
        }

    def load_train_mesh_infos(self):

        with open(os.path.join(self.dataset_path, 'mesh_infos.pkl'), 'rb') as f:
            mesh_infos = pickle.load(f)
        for frame_name in mesh_infos.keys():
            bbox = self.skeleton_to_bbox(mesh_infos[frame_name]['joints'])
            mesh_infos[frame_name]['bbox'] = bbox
        return mesh_infos

    def load_train_frames(self):
        img_paths = list_files(os.path.join(self.dataset_path, 'images'),
                            exts=['.png'])
        return [split_path(ipath)[1] for ipath in img_paths]
    
    def query_dst_skeleton(self, idx):
        return {
            'poses': self.cam_infos[idx].smpl_param['poses'].numpy().astype('float32').reshape([-1, 3]),
            'dst_tpose_joints': \
                self.t_joints.astype('float32'),
            'bbox': self.cam_infos[idx].bbox.copy(),
            'Rh': self.cam_infos[idx].smpl_param['R'].astype('float32'),
            'Th': self.cam_infos[idx].smpl_param['Th'].astype('float32')
        }

    @staticmethod
    def select_rays(select_inds, rays_o, rays_d, ray_img, near, far):
        rays_o = rays_o[select_inds]
        rays_d = rays_d[select_inds]
        ray_img = ray_img[select_inds]
        near = near[select_inds]
        far = far[select_inds]
        return rays_o, rays_d, ray_img, near, far
    
    def get_patch_ray_indices(
            self, 
            N_patch, 
            ray_mask, 
            subject_mask, 
            bbox_mask,
            patch_size, 
            H, W):

        assert subject_mask.dtype == np.bool
        assert bbox_mask.dtype == np.bool

        bbox_exclude_subject_mask = np.bitwise_and(
            bbox_mask,
            np.bitwise_not(subject_mask)
        )

        list_ray_indices = []
        list_mask = []
        list_xy_min = []
        list_xy_max = []

        total_rays = 0
        patch_div_indices = [total_rays]
        for _ in range(N_patch):
            # let p = cfg.patch.sample_subject_ratio
            # prob p: we sample on subject area
            # prob (1-p): we sample on non-subject area but still in bbox
            if np.random.rand(1)[0] < cfg.patch.sample_subject_ratio:
                candidate_mask = subject_mask
            else:
                candidate_mask = bbox_exclude_subject_mask

            ray_indices, mask, xy_min, xy_max = \
                self._get_patch_ray_indices(ray_mask, candidate_mask, 
                                            patch_size, H, W)

            assert len(ray_indices.shape) == 1
            total_rays += len(ray_indices)

            list_ray_indices.append(ray_indices)
            list_mask.append(mask)
            list_xy_min.append(xy_min)
            list_xy_max.append(xy_max)
            
            patch_div_indices.append(total_rays)

        select_inds = np.concatenate(list_ray_indices, axis=0)
        patch_info = {
            'mask': np.stack(list_mask, axis=0),
            'xy_min': np.stack(list_xy_min, axis=0),
            'xy_max': np.stack(list_xy_max, axis=0)
        }
        patch_div_indices = np.array(patch_div_indices)

        return select_inds, patch_info, patch_div_indices


    def _get_patch_ray_indices(
            self, 
            ray_mask, 
            candidate_mask, 
            patch_size, 
            H, W):

        assert len(ray_mask.shape) == 1
        assert ray_mask.dtype == np.bool
        assert candidate_mask.dtype == np.bool

        valid_ys, valid_xs = np.where(candidate_mask)

        # determine patch center
        select_idx = np.random.choice(valid_ys.shape[0], 
                                      size=[1], replace=False)[0]
        center_x = valid_xs[select_idx]
        center_y = valid_ys[select_idx]

        # determine patch boundary
        half_patch_size = patch_size // 2
        x_min = np.clip(a=center_x-half_patch_size, 
                        a_min=0, 
                        a_max=W-patch_size)
        x_max = x_min + patch_size
        y_min = np.clip(a=center_y-half_patch_size,
                        a_min=0,
                        a_max=H-patch_size)
        y_max = y_min + patch_size

        sel_ray_mask = np.zeros_like(candidate_mask)
        sel_ray_mask[y_min:y_max, x_min:x_max] = True

        #####################################################
        ## Below we determine the selected ray indices
        ## and patch valid mask

        sel_ray_mask = sel_ray_mask.reshape(-1)
        inter_mask = np.bitwise_and(sel_ray_mask, ray_mask)
        select_masked_inds = np.where(inter_mask)

        masked_indices = np.cumsum(ray_mask) - 1
        select_inds = masked_indices[select_masked_inds]
        
        inter_mask = inter_mask.reshape(H, W)

        return select_inds, \
                inter_mask[y_min:y_max, x_min:x_max], \
                np.array([x_min, y_min]), np.array([x_max, y_max])

    def load_image(self, frame_name, bg_color):

        imagepath = os.path.join(self.image_dir, '{}.png'.format(frame_name))
        
        orig_img = np.array(load_image(imagepath))

        maskpath = os.path.join(self.dataset_path,
                                'masks',
                                '{}.png'.format(frame_name))

        alpha_mask = np.array(load_image(maskpath))

        # undistort image
        if frame_name in self.cameras and 'distortions' in self.cameras[frame_name]:
            K = self.cameras[frame_name]['intrinsics']
            D = self.cameras[frame_name]['distortions']
            orig_img = cv2.undistort(orig_img, K, D)
            alpha_mask = cv2.undistort(alpha_mask, K, D)

        alpha_mask = alpha_mask / 255.
        img = alpha_mask * orig_img + (1.0 - alpha_mask) * bg_color[None, None, :]
        if cfg.resize_img_scale != 1.:
            img = cv2.resize(img, None, 
                                fx=cfg.resize_img_scale,
                                fy=cfg.resize_img_scale,
                                interpolation=cv2.INTER_LANCZOS4)
            alpha_mask = cv2.resize(alpha_mask, None, 
                                    fx=cfg.resize_img_scale,
                                    fy=cfg.resize_img_scale,
                                    interpolation=cv2.INTER_LINEAR)
                                
        return img, alpha_mask


    def get_total_frames(self):
        return len(self.cam_infos)

    def sample_patch_rays(self, img, H, W,
                          subject_mask, bbox_mask, ray_mask,
                          rays_o, rays_d, ray_img, near, far):

        select_inds, patch_info, patch_div_indices = \
            self.get_patch_ray_indices(
                N_patch=cfg.patch.N_patches, 
                ray_mask=ray_mask, 
                subject_mask=subject_mask, 
                bbox_mask=bbox_mask,
                patch_size=cfg.patch.size, 
                H=H, W=W)

        rays_o, rays_d, ray_img, near, far = self.select_rays(
            select_inds, rays_o, rays_d, ray_img, near, far)
        
        targets = []
        for i in range(cfg.patch.N_patches):
            x_min, y_min = patch_info['xy_min'][i] 
            x_max, y_max = patch_info['xy_max'][i]
            targets.append(img[y_min:y_max, x_min:x_max])
        target_patches = np.stack(targets, axis=0) # (N_patches, P, P, 3)

        patch_masks = patch_info['mask']  # boolean array (N_patches, P, P)

        return rays_o, rays_d, ray_img, near, far, \
                target_patches, patch_masks, patch_div_indices

    def __len__(self):
        return self.get_total_frames()

    def __getitem__(self, idx):
        results = {
            'frame_name': self.cam_infos[idx].image_name
        }


        dst_skel_info = self.query_dst_skeleton(idx)
        dst_bbox = dst_skel_info['bbox']
        dst_poses = dst_skel_info['poses'].reshape([-1])
        dst_tpose_joints = dst_skel_info['dst_tpose_joints']

        if self.bgcolor is None:
            bgcolor = (np.random.rand(3) * 255.).astype('float32')
        else:
            bgcolor = np.array(self.bgcolor, dtype='float32')


        camera_info: CameraInfo = self.cam_infos[idx]
        img, alpha = camera_info.image, camera_info.bkgd_mask
        alpha = alpha[..., None]

        H, W = img.shape[0:2]
        src_img = self.src_img

        # poses = camera_info.smpl_param['poses']
        # betas = camera_info.smpl_param['shapes']
        joints = camera_info.smplx_model_out.original_joints[0].numpy()

        K = camera_info.K
        K[:2] *= cfg.resize_img_scale

        R = camera_info.R
        T = camera_info.T
        E = np.eye(4)
        E[:3,:3] = R
        E[:3, 3] = T

        
        rays_o, rays_d = get_rays_from_KRT(H, W, K, R, T)
        ray_img = img.reshape(-1, 3) 
        rays_o = rays_o.reshape(-1, 3) # (H, W, 3) --> (N_rays, 3)
        rays_d = rays_d.reshape(-1, 3)

        # (selected N_samples, ), (selected N_samples, ), (N_samples, )
        near, far, ray_mask = rays_intersect_3d_bbox(dst_bbox, rays_o, rays_d)
        rays_o = rays_o[ray_mask]
        rays_d = rays_d[ray_mask]
        ray_img = ray_img[ray_mask]

        near = near[:, None].astype('float32')
        far = far[:, None].astype('float32')

        if self.ray_shoot_mode == 'image':
            pass
        elif self.ray_shoot_mode == 'patch':
            rays_o, rays_d, ray_img, near, far, \
            target_patches, patch_masks, patch_div_indices = \
                self.sample_patch_rays(img=img, H=H, W=W,
                                       subject_mask=alpha[:, :, 0] > 0.,
                                       bbox_mask=ray_mask.reshape(H, W),
                                       ray_mask=ray_mask,
                                       rays_o=rays_o, 
                                       rays_d=rays_d, 
                                       ray_img=ray_img, 
                                       near=near, 
                                       far=far)
        else:
            assert False, f"Ivalid Ray Shoot Mode: {self.ray_shoot_mode}"
    
        batch_rays = np.stack([rays_o, rays_d], axis=0) 

        if 'rays' in self.keyfilter:
            results.update({
                'img_width': W,
                'img_height': H,
                'ray_mask': ray_mask,
                'rays': batch_rays,
                'near': near,
                'far': far,
                'bgcolor': bgcolor, 
                'ori_img': img,
                'src_imgs':src_img,
                'joints': joints,
                'canonical_joints': self.canonical_joints
                })

            if self.ray_shoot_mode == 'patch':
                results.update({
                    'patch_div_indices': patch_div_indices,
                    'patch_masks': patch_masks,
                    'target_patches': target_patches})

        if 'target_rgbs' in self.keyfilter:
            results['target_rgbs'] = ray_img

        if 'motion_bases' in self.keyfilter:
            dst_Rs, dst_Ts = body_pose_to_body_RTs(
                    dst_poses, dst_tpose_joints
                )
            cnl_gtfms = get_canonical_global_tfms(
                            self.canonical_joints)
            results.update({
                'dst_Rs': dst_Rs,
                'dst_Ts': dst_Ts,
                'cnl_gtfms': cnl_gtfms
            })

            in_dst_Rs = []
            in_dst_Ts = []
            for i in range(len(self.in_dst_poses)):
                dst_Rs_, dst_Ts_ = body_pose_to_body_RTs(
                    self.in_dst_poses[i].reshape([-1]), self.in_dst_tposes_joints[i]
                )
                in_dst_Rs.append(dst_Rs_)
                in_dst_Ts.append(dst_Ts_)
            results.update(
                {
                    'in_dst_Rs': np.array(in_dst_Rs),
                    'in_dst_Ts': np.array(in_dst_Ts)
                }
            )
        #print('len---in---dataset', len(in_dst_Rs))
        if 'motion_weights_priors' in self.keyfilter:
            results['motion_weights_priors'] = self.motion_weights_priors.copy()

        # get the bounding box of canonical volume
        if 'cnl_bbox' in self.keyfilter:
            min_xyz = self.canonical_bbox['min_xyz'].astype('float32')
            max_xyz = self.canonical_bbox['max_xyz'].astype('float32')
            results.update({
                'cnl_bbox_min_xyz': min_xyz,
                'cnl_bbox_max_xyz': max_xyz,
                'cnl_bbox_scale_xyz': 2.0 / (max_xyz - min_xyz)
            })
            assert np.all(results['cnl_bbox_scale_xyz'] >= 0)

        if 'dst_posevec_69' in self.keyfilter:
            # 1. ignore global orientation
            # 2. add a small value to avoid all zeros
            dst_posevec_69 = dst_poses[3:] + 1e-2
            results.update({
                'dst_posevec': dst_posevec_69,
            })

        in_dst_posevec = []
        for posevec in self.in_dst_poses:
            in_dst_posevec_69 = posevec.reshape([-1])[3:] + 1e-2
            in_dst_posevec.append(in_dst_posevec_69)
        results.update({
            'in_dst_posevec': np.array(in_dst_posevec)
        })

        results.update({

            'in_K': np.array(self.in_K),
            'in_E': np.array(self.in_E),
            'E': E,
            'K': K
        }
        )
        return results






