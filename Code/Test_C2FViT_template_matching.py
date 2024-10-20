import os
from argparse import ArgumentParser

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from C2FViT_model import C2F_ViT_stage, AffineCOMTransform, CustomAffineCOMTransform, Center_of_mass_initial_pairwise, CustomCenter_of_mass_initial_pairwise
from Functions import save_img, load_4D, min_max_norm, pad_to_shape, crop_image, update_affine, reorient_image
from tqdm import tqdm


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--modelpath", type=str,
                        dest="modelpath",
                        default='../Model/C2FViT_affine_COM_template_matching_stagelvl3_116000.pth',
                        help="Pre-trained Model path")
    parser.add_argument("--savepath", type=str,
                        dest="savepath", default='../Result',
                        help="path for saving images")
    parser.add_argument("--fixed", type=str,
                        dest="fixed", default='../Data/MNI152_T1_1mm_brain_pad_RSP.nii.gz',
                        help="fixed image")
    parser.add_argument("--moving", type=str,
                        dest="moving", default=None,
                        help="moving image")
    parser.add_argument("--folder", type=str,
                        dest="folder", default=None,
                        help="folder containing moving images")
    parser.add_argument("--com_initial", type=bool,
                        dest="com_initial", default=True,
                        help="True: Enable Center of Mass initialization, False: Disable")
    parser.add_argument("--crop", action='store_true', 
                        help="Crop the output images to 160*224*192")
    parser.add_argument("--output_RAS", action='store_true', 
                        help="Convert the output image into RAS coordinates")
    parser.add_argument("--Eval", action='store_true', 
                        help="Used later to evaluate the results")
    parser.add_argument("--VBM", action='store_true', 
                        help="Perform VBM analysis")
    parser.add_argument("--RAS", action='store_true', 
                        help="True: The image is in RAS coordinates during testing, False: LIA coordinates")    
    opt = parser.parse_args()

    savepath = opt.savepath
    fixed_path = opt.fixed
    moving_path = opt.moving
    folder_path = opt.folder
    com_initial = opt.com_initial
    crop = opt.crop
    output_RAS = opt.output_RAS
    eval_flag = opt.Eval
    VBM = opt.VBM
    RAS = opt.RAS
    
    if not os.path.isdir(savepath):
        os.mkdir(savepath)

    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")

    model = C2F_ViT_stage(img_size=128, patch_size=[3, 7, 15], stride=[2, 4, 8], num_classes=12,
                          embed_dims=[256, 256, 256],
                          num_heads=[2, 2, 2], mlp_ratios=[2, 2, 2], qkv_bias=False, qk_scale=None, drop_rate=0.,
                          attn_drop_rate=0., norm_layer=nn.Identity,
                          depths=[4, 4, 4], sr_ratios=[1, 1, 1], num_stages=3, linear=False).to(device)

    print(f"Loading model weight {opt.modelpath} ...")
    model.load_state_dict(torch.load(opt.modelpath))
    model.eval()

    affine_transform = CustomAffineCOMTransform().cuda()
    init_center = CustomCenter_of_mass_initial_pairwise()

    fixed_base = os.path.basename(fixed_path)
    fixed_img_nii = nib.load(fixed_path)
    fixed_header, fixed_affine = fixed_img_nii.header, fixed_img_nii.affine
    fixed_img = fixed_img_nii.get_fdata()
    
    # Ensure the image size is 256x256x256, pad if necessary
    target_shape = (256, 256, 256)
    if fixed_img.shape != target_shape:
        fixed_img = pad_to_shape(fixed_img, target_shape)
        
    fixed_img = np.reshape(fixed_img, (1,) + fixed_img.shape)

    # If fixed img is MNI152 altas, do windowing (contrast stretching)
    if fixed_base == "MNI152_T1_1mm_brain_pad_RSP.nii.gz" or fixed_base == "MNI152_T1_1mm_brain_pad_RSP_RAS.nii.gz":
        fixed_img = np.clip(fixed_img, a_min=2500, a_max=np.max(fixed_img))

    fixed_img = min_max_norm(fixed_img)
    fixed_img = torch.from_numpy(fixed_img).float().to(device).unsqueeze(dim=0)

    def process_moving_image(moving_img_path, header, affine, Eval=False):
        moving_base = os.path.basename(moving_img_path)
        moving_img = load_4D(moving_img_path, RAS)
         
        moving_img = min_max_norm(moving_img)
        moving_img = torch.from_numpy(moving_img).float().to(device).unsqueeze(dim=0)

        with torch.no_grad():
            if com_initial:
                moving_img, init_flow = init_center(moving_img, fixed_img)
    
            X_down = F.interpolate(moving_img, scale_factor=0.5, mode="trilinear", align_corners=True)
            Y_down = F.interpolate(fixed_img, scale_factor=0.5, mode="trilinear", align_corners=True)
    
            warpped_x_list, y_list, affine_para_list = model(X_down, Y_down)
          #rigid
            #affine_para_list[-1][0, 11] = 0
            #affine_para_list[-1][0, 10] = 0
            #affine_para_list[-1][0, 9] = 0
            #affine_para_list[-1][0, 8] = 0
            #affine_para_list[-1][0, 7] = 0
            #affine_para_list[-1][0, 6] = 0
            X_Y, affine_matrix = affine_transform(moving_img, affine_para_list[-1])
            
            X_Y_cpu = X_Y.data.cpu().numpy()[0, 0, :, :, :]
            
            if Eval:
                #ABIDE_50
                moving_seg = load_4D(moving_img_path.replace("ABIDE_NoAffine", "ABIDE_aseg").replace("_tbet.nii.gz", "_aseg.nii.gz"), RAS)
                #CC359_60
                #moving_seg = load_4D(moving_img_path.replace("CC359_60", "CC359_60_aseg").replace(".nii.gz", "_aseg.nii.gz"), RAS)
                #VBM
                #moving_seg = load_4D(moving_img_path.replace("raw", "GM").replace(".nii.gz", "_cgw_pve1.nii.gz"), RAS)
                
                moving_seg = torch.from_numpy(moving_seg).float().to(device).unsqueeze(dim=0)
                
                mode = "bilinear" if VBM else "nearest"

                moving_seg = F.grid_sample(moving_seg, init_flow, mode=mode, align_corners=True)
                F_X_Y = F.affine_grid(affine_matrix, moving_seg.shape, align_corners=True)
                moving_seg = F.grid_sample(moving_seg, F_X_Y, mode=mode, align_corners=True).cpu().numpy()[0, 0, :, :, :]
                
            if output_RAS:
                X_Y_cpu_nii = nib.nifti1.Nifti1Image(X_Y_cpu, affine, header=header)
                X_Y_cpu_nii = reorient_image(X_Y_cpu_nii, ('R', 'A', 'S'))
                
                if Eval:
                    moving_seg_nii = nib.nifti1.Nifti1Image(moving_seg, affine, header=header)
                    moving_seg_nii = reorient_image(moving_seg_nii, ('R', 'A', 'S'))
                    moving_seg = moving_seg_nii.get_fdata()
                    
                header, affine = X_Y_cpu_nii.header, X_Y_cpu_nii.affine
                X_Y_cpu = X_Y_cpu_nii.get_fdata()
                
            if crop:
                X_Y_cpu, crop_slices = crop_image(X_Y_cpu, target_shape=(160, 224, 192))
                affine = update_affine(affine, crop_slices)
                
                if Eval:
                    moving_seg, crop_slices = crop_image(moving_seg, target_shape=(160, 224, 192))
                    
            save_img(X_Y_cpu, f"{savepath}/{moving_base}", header=header, affine=affine)
            if Eval:
                if VBM:
                    if not os.path.isdir(savepath + "/GM"):
                        os.mkdir(savepath + "/GM")
                        
                    moving_base = moving_base.replace(".nii.gz", "_cgw_pve1.nii.gz")
                    save_img(moving_seg, f"{savepath}/GM/{moving_base}", header=header, affine=affine)
                else:
                    moving_base = moving_base.replace(".nii.gz", "_aseg.nii.gz")
                    save_img(moving_seg, f"{savepath}/{moving_base}", header=header, affine=affine)    
            
    if folder_path:
        files = [f for f in os.listdir(folder_path) if f.endswith('.nii.gz')]
        for file_name in tqdm(files, desc="Processing images"):
            process_moving_image(os.path.join(folder_path, file_name), fixed_header, fixed_affine, eval_flag)
    else:
        process_moving_image(moving_path, fixed_header, fixed_affine, eval_flag)
        
    print("Result saved to :", savepath)