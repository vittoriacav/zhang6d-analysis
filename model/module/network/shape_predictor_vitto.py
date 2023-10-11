from absl import flags
import torch
import torch.nn as nn
import numpy as np

from nerf import models, run_network
from nerf.nerf_helpers import positional_encoding
from scipy.spatial.transform import Rotation as R
import torch.nn.functional as F
import trimesh
import os
from model.module.mesh import CanonicalMesh


flags.DEFINE_bool('no_deform', False, 'do not predict deformation')
flags.DEFINE_bool('rot_enc', False, 'Rotate by R from the encoder the canonical mesh')
flags.DEFINE_bool('rot_enc_post', False, 'Rotate by R from the encoder the instance mesh')
flags.DEFINE_bool('rot_sampling', False, 'Test different rotations of the canonical mesh and keep the best one')
flags.DEFINE_integer('rot_manual', 0, 'Rotate by manual rotation the canonical mesh')
flags.DEFINE_float('deform_ratio', 1., 'deform ratio')

class ShapePredictor(nn.Module):
    def __init__(self, opts):
        super(ShapePredictor, self).__init__()
        self.shapenerf = models.CondNeRFModel(
            num_layers=2,
            num_encoding_fn_xyz=0,
            num_encoding_fn_dir=0,
            include_input_xyz=True,
            include_input_dir=False,
            use_viewdirs=False,
            out_channel=3,
            codesize=opts.codedim
        )
        nerf_param = sum(p.numel() for p in self.shapenerf.parameters())
        print(f"NUMBER OF PARAMETERS SHAPE PREDICTOR: {nerf_param}")
        self.iters = 0
        self.no_deform = opts.no_deform
        self.deform_ratio = opts.deform_ratio
        self.rot_sampling = opts.rot_sampling
        self.rot_enc = opts.rot_enc
        self.rot_enc_post = opts.rot_enc_post
        self.rot_manual = opts.rot_manual

        self.opts = opts

    
    def forward(self, mean_v, shape_code, rotation_enc):
        """
        mean_v  --> shape = torch.Size([16, 642, 3])
            16 = batch size
            643 = nub of vertices (same number across all meshes of the same category)
            3 = [x, y, z] for each vertex
        
        rotation_enc --> shape = torch.Size([16, 3, 3])
            16 = batch size
            [3,3] = 3x3 rotation matrix
        """
        
        size_v = mean_v.shape
        bsz = size_v[0]
        nvert = size_v[1]

        # Test different rotations of the canonical mesh and keep the best one
        if self.rot_sampling:
            print("Rotating mesh using different rotations and selecting the best one")
            # define list of rotations around z that you want to test
            degs = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
            
            best_deg = 0
            best_loss = float('inf')
            best_pred_v = 0

            for deg in degs:
                rdeg = R.from_euler('z', deg, degrees=True)
                rdeg_matrix = torch.cuda.FloatTensor(rdeg.as_matrix())

                # repeating and reshaping the same rotation matrix that will be applied to:
                #   - all vertices in each instance
                #   - all instances in the batch --> !! probably wrong (if having multiple rotations)!!
                rdeg_matrix = rdeg_matrix.repeat(bsz, nvert, 1)
                rdeg_matrix = rdeg_matrix.reshape(bsz, nvert, 3, 3)
                
                mean_v = mean_v.reshape(bsz, nvert, size_v[2], 1)
                # apply rotation
                mean_v = rdeg_matrix @ mean_v 
                mean_v = mean_v.reshape(bsz, nvert, size_v[2])

                # mV = mean_v
                if self.no_deform:
                    return mean_v
                else:
                    # ΔV = shape_delta
                    shape_delta = run_network(
                        self.shapenerf, 
                        mean_v.detach(), 
                        None, 
                        131072, 
                        None, 
                        None,
                        code=shape_code
                    )[:,:,:-1]
                    
                    #sort of normalization i guess
                    shape_delta -= shape_delta.mean(1, keepdims=True)
                    # instance mesh = V = mV + ΔV
                    pred_v = mean_v + shape_delta * self.deform_ratio

                    # smooth L1 loss --> measure of similarity between canonical mesh (mV) and the final instance mesh (V)
                    loss = F.smooth_l1_loss(pred_v, mean_v, reduction='mean')
                    
                # keep track of the best performing one
                if loss < best_loss:
                    best_loss = loss
                    best_deg = deg
                    best_pred_v = pred_v
                                               
            return best_pred_v

        # Rotate by R from the encoder the canonical mesh (rotation_enc)
        elif self.rot_enc:
            print("Rotating mesh using R|t from encoder")
            # this for loop is needed to transform the shape of the tensor from the encoder
            # from (16, 3, 3)
            # to   (16, 642, 3, 3) --> apply the rotation matrix of each instance to all vertices
            rotation_vert = []
            for sub in rotation_enc:
                vert = sub.repeat(nvert,1)
                vert = vert.reshape(nvert,3,3)
                rotation_vert.append(vert)
            final_rotation = torch.stack(rotation_vert, 0)

            mean_v = mean_v.reshape(bsz, nvert, size_v[2], 1)
            # apply rotation
            mean_v = final_rotation @ mean_v 
            mean_v = mean_v.reshape(bsz, nvert, size_v[2])

            if self.no_deform:
                return mean_v
            else:
                shape_delta = run_network(
                    self.shapenerf, 
                    mean_v.detach(), 
                    None, 
                    131072, 
                    None, 
                    None,
                    code=shape_code
                )[:,:,:-1]

                shape_delta -= shape_delta.mean(1, keepdims=True)
                pred_v = mean_v + shape_delta * self.deform_ratio
                return pred_v

        elif self.rot_manual != 0:
            deg = self.rot_manual
            print("Rotating mesh of {} degree".format(deg))

            rdeg = R.from_euler('z', deg, degrees=True)
            rdeg_matrix = torch.cuda.FloatTensor(rdeg.as_matrix())

            # repeating and reshaping the same rotation matrix that will be applied to:
            #   - all vertices in each instance
            #   - all instances in the batch --> !! probably wrong (if having multiple rotations)!!
            rdeg_matrix = rdeg_matrix.repeat(bsz, nvert, 1)
            rdeg_matrix = rdeg_matrix.reshape(bsz, nvert, 3, 3)

            mean_v = mean_v.reshape(bsz, nvert, size_v[2], 1)
            # apply rotation
            mean_v = rdeg_matrix @ mean_v 
            mean_v = mean_v.reshape(bsz, nvert, size_v[2])
           
            if self.no_deform:
                return mean_v
            else:
                shape_delta = run_network(
                    self.shapenerf, 
                    mean_v.detach(), 
                    None, 
                    131072, 
                    None, 
                    None,
                    code=shape_code
                )[:,:,:-1]

                shape_delta -= shape_delta.mean(1, keepdims=True)
                pred_v = mean_v + shape_delta * self.deform_ratio
                return pred_v           
        elif self.rot_enc_post:
            print("Rotation after shape prediction")
            if self.no_deform:
                return mean_v
            else:
                shape_delta = run_network(
                    self.shapenerf, 
                    mean_v.detach(), 
                    None, 
                    131072, 
                    None, 
                    None,
                    code=shape_code
                )[:,:,:-1]

                shape_delta -= shape_delta.mean(1, keepdims=True)
                pred_v = mean_v + shape_delta * self.deform_ratio

                rotation_vert = []
                for sub in rotation_enc:
                    vert = sub.repeat(nvert,1)
                    vert = vert.reshape(nvert,3,3)
                    rotation_vert.append(vert)
                final_rotation = torch.stack(rotation_vert, 0)

                pred_v = pred_v.reshape(bsz, nvert, size_v[2], 1)
                # apply rotation
                pred_v = final_rotation @ pred_v 
                pred_v = pred_v.reshape(bsz, nvert, size_v[2])

                return pred_v
        #############################################################

        # base implementation of Kaifeng
        else:
            print("No rotation on the mesh, base implementation of Kaifeng")
            if self.no_deform:
                print("No mesh deformation")
                return mean_v
            else:
                shape_delta = run_network(
                    self.shapenerf, 
                    mean_v.detach(), 
                    None, 
                    131072, 
                    None, 
                    None,
                    code=shape_code
                )[:,:,:-1]

                shape_delta -= shape_delta.mean(1, keepdims=True)
                pred_v = mean_v + shape_delta * self.deform_ratio
                return pred_v