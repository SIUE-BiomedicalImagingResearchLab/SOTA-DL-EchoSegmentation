# -*- coding: utf-8 -*-

"""
usage example:
python MedSAM_Inference.py -i assets/img_demo.png -o ./ --box "[95,255,190,350]"

"""
import numpy as np
import pandas as pd
import os, json, cv2, random, yaml, csv
import matplotlib.pyplot as plt
from skimage.measure import regionprops
from scipy.spatial.distance import directed_hausdorff
join = os.path.join
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import monai
from segment_anything import SamPredictor, sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide

def get_bbox_from_mask(mask):
    '''Returns a bounding box from a mask'''
    nLbls = np.max(mask)
    boxes = []
    for jj in range(nLbls):
        y_indices, x_indices = np.where(mask == jj + 1)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        # add perturbation to bounding box coordinates
        H, W = mask.shape
        x_min = max(0, x_min - np.random.randint(0, 20))
        x_max = min(W, x_max + np.random.randint(0, 20))
        y_min = max(0, y_min - np.random.randint(0, 20))
        y_max = min(H, y_max + np.random.randint(0, 20))

        box = np.array([x_min, y_min, x_max, y_max])
        boxes.append(box)

    return np.asarray(boxes)

# visualization functions
# source: https://github.com/facebookresearch/segment-anything/blob/main/notebooks/predictor_example.ipynb
# change color to avoid red and green
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251 / 255, 252 / 255, 30 / 255, 0.6])
    h, w = mask.shape[-2:]

    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor="blue", facecolor=(0, 0, 0, 0), lw=2))

def dice_similarity(a, b):
    # a = np.array(a / 255, dtype=np.uint8)
    # b = np.array(b / 255, dtype=np.uint8)

    if a.size == 0 or b.size == 0:
        return []

    dice = np.sum(b[a == 1]) * 2.0 / (np.sum(a) + np.sum(b))
    return dice

def Hausdorff_Distance(u, v):
    # u and v must be two 2-D arrays of coordinates
    if not (u is None and v is None):
        HD = max(directed_hausdorff(u, v)[0], directed_hausdorff(v, u)[0]) / 8.885960105352543 #bModeDistPxPerMM
        return HD

def list_contours(mask):
    if np.max(mask) == 0:
        return [(10, 10), (10, 20), (20, 20), (20, 10)]

    if mask is not None:
        grayscale = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        grayscale[np.where(grayscale != 0)] = 255
        # Get contour points
        non_zero_coordinates = np.nonzero(grayscale)

        return [(y, x) for y, x in zip(non_zero_coordinates[0], non_zero_coordinates[1])]


def metric_info_string(metric, lm_m, lv_m, right_m, average_m):
    return "Average %s: %.3f\n" \
           "Left Myocardium %s: %.3f\n" \
           "Left Ventricle %s: %.3f\n" \
           "Right Ventricle %s: %.3f" % (metric, average_m, metric, lm_m, metric, lv_m, metric, right_m)


def Compute_MasknMetrics(med_sam, gt_data):
    lab_mask = np.zeros((256, 256)).astype(np.uint8)
    lab_mask = med_sam[0, :, :] + med_sam[1, :, :]  # add first LV + LE
    if np.max(lab_mask) > 2:  # fix LV, LE overlap (2 + 1)
        lab_mask[lab_mask > 2] = 2
    lab_mask = lab_mask + med_sam[2, :, :]          # add RE
    if np.max(lab_mask) > 3:  # fix RV, LE overlap (3 + 1)
        lab_mask[lab_mask > 3] = 1
    # Extrac objects properties
    props = regionprops(np.asarray(lab_mask, np.uint8))
    props_gt = regionprops(gt_data)

    # Compute DSC between MFP and ground truth
    lm_dsc = dice_similarity((lab_mask == 1).astype(np.uint8), (gt_data == 1).astype(np.uint8))
    lv_dsc = dice_similarity((lab_mask == 2).astype(np.uint8), (gt_data == 2).astype(np.uint8))
    right_dsc = dice_similarity((lab_mask == 3).astype(np.uint8), (gt_data == 3).astype(np.uint8))
    average_dsc = np.average([lm_dsc, lv_dsc, right_dsc])

    # Compute HD between MFP and ground truth take care of unpredicted labels
    # if a labels is not predicted dsc is computed agins a square located in the middle of image
    missLbl = 0
    bModeAreaPx2PerMm2 = 8.885960105352543
    if lm_dsc == 0:
        lm_hd = Hausdorff_Distance([[100, 100], [100, 156], [156, 156], [156, 100]], props_gt[0].coords.tolist())
        lm_csa = abs(0 - props_gt[0].area) / bModeAreaPx2PerMm2
        missLbl += 1  # increase missed lbl counter
    else:
        lm_hd = Hausdorff_Distance(props[0].coords.tolist(), props_gt[0].coords.tolist())
        lm_csa = abs(props[0].area - props_gt[0].area) / bModeAreaPx2PerMm2

    if lv_dsc == 0:
        lv_hd = Hausdorff_Distance([[100, 100], [100, 156], [156, 156], [156, 100]], props_gt[1].coords.tolist())
        lv_csa = abs(0 - props_gt[1].area) / bModeAreaPx2PerMm2
        missLbl += 1  # increase missed lbl counter
    else:
        lv_hd = Hausdorff_Distance(props[1 - missLbl].coords.tolist(), props_gt[1 - missLbl].coords.tolist())
        lv_csa = abs(props[1 - missLbl].area - props_gt[1 - missLbl].area) / bModeAreaPx2PerMm2

    if right_dsc == 0:
        right_hd = Hausdorff_Distance([[100, 100], [100, 156], [156, 156], [156, 100]], props_gt[2].coords.tolist())
        right_csa = abs(0 - props_gt[2].area) / bModeAreaPx2PerMm2
        missLbl += 1
    else:
        right_hd = Hausdorff_Distance(props[2 - missLbl].coords.tolist(), props_gt[2 - missLbl].coords.tolist())
        right_csa = abs(props[2 - missLbl].area - props_gt[2 - missLbl].area) / bModeAreaPx2PerMm2

    average_hd = np.average([lm_hd, lv_hd, right_hd])

    # Compute CSA between MFP and ground truth
    average_csa = np.average([lm_csa, lv_csa, right_csa])

    return lab_mask, lm_dsc, lv_dsc, right_dsc, average_dsc, lm_hd, lv_hd, right_hd, average_hd, lm_csa, lv_csa, right_csa, average_csa


def saveExcels(output_directory, filename, model_vs_ground_truth_dsc, lm_dsc_list, lv_dsc_list, right_dsc_list, model_vs_ground_truth_hd,
               lm_hd_list, lv_hd_list, right_hd_list, model_vs_ground_truth_csa, lm_csa_list, lv_csa_list, right_csa_list, listlista):
    # Compute average DSC among all test cases
    model_gt_average = np.average(model_vs_ground_truth_dsc)
    average_lm = np.average(lm_dsc_list)
    average_lv = np.average(lv_dsc_list)
    average_right = np.average(right_dsc_list)

    # Compute average hd among all test cases
    model_gt_average_hd = np.average(model_vs_ground_truth_hd)
    average_lm_hd = np.average(lm_hd_list)
    average_lv_hd = np.average(lv_hd_list)
    average_right_hd = np.average(right_hd_list)

    # Compute average DSC among all test cases
    model_gt_average_csa = np.average(model_vs_ground_truth_csa)
    average_lm_csa = np.average(lm_csa_list)
    average_lv_csa = np.average(lv_csa_list)
    average_right_csa = np.average(right_csa_list)

    temp = {}
    extra_list = [0] * len(lm_csa_list)
    caseDescription = filename
    temp['Case'] = [caseDescription] + listlista
    temp['LM_DSC'] = [average_lm] + lm_dsc_list
    temp['LV_DSC'] = [average_lv] + lv_dsc_list
    temp['RV_DSC'] = [average_right] + right_dsc_list
    temp['DSC_AVG'] = [model_gt_average] + extra_list

    temp['LM_HD'] = [average_lm_hd] + lm_hd_list
    temp['LV_HD'] = [average_lv_hd] + lv_hd_list
    temp['RV_HD'] = [average_right_hd] + right_hd_list
    temp['DSC_HD'] = [model_gt_average_hd] + extra_list

    temp['LM_CSA'] = [average_lm_csa] + lm_csa_list
    temp['LV_CSA'] = [average_lv_csa] + lv_csa_list
    temp['RV_CSA'] = [average_right_csa] + right_csa_list
    temp['DSC_CSA'] = [model_gt_average_csa] + extra_list

    df = pd.DataFrame(temp)

    df.to_excel(output_directory + '/' + filename)


# prepare SAM model
model_type = 'vit_b'
checkPsam = "work_dir/MedSAM/medsam_vit_b.pth"
checkpoint = 'work_dir/SAM/sam_vit_b_01ec64.pth'
checkpointT = 'work_dir/medsam_model_BIRL.pth'
ts_img_path = "../data/Experiment/Testing/imgs"
ts_gt_path = '../data/Experiment/Testing/gts'
output_directory = "../data/Experiment/Segmentation Output/BIRLmedSAM"
device = 'cuda:0'

#%% compare the segmentation results between the original SAM model and the fine-tuned model
# load the original SAM model
ori_sam_model = sam_model_registry[model_type](checkpoint=checkpoint).to(device)
ori_sam_predictor = SamPredictor(ori_sam_model)
# predict the segmentation using original MedSam
Msam_model = sam_model_registry[model_type](checkpoint=checkPsam).to(device)
med_sam_predictor = SamPredictor(Msam_model)
# predict the segmentation mask using the fine-tuned model
sam_model = sam_model_registry[model_type](checkpoint=checkpointT).to(device)
new_sam_predictor = SamPredictor(sam_model)

# Metrics vbles BIRL MedSam
model_vs_ground_truth_dsc = []
lm_dsc_list = []
lv_dsc_list = []
right_dsc_list = []

model_vs_ground_truth_csa = []
lm_csa_list = []
lv_csa_list = []
right_csa_list = []

model_vs_ground_truth_hd = []
lm_hd_list = []
lv_hd_list = []
right_hd_list = []

# Metrics vbles MedSAM
mSAM_vs_ground_truth_dsc = []
mSAMlm_dsc_list = []
mSAMlv_dsc_list = []
mSAMright_dsc_list = []

mSAM_vs_ground_truth_csa = []
mSAMlm_csa_list = []
mSAMlv_csa_list = []
mSAMright_csa_list = []

mSAM_vs_ground_truth_hd = []
mSAMlm_hd_list = []
mSAMlv_hd_list = []
mSAMright_hd_list = []

# Metrics vbles SAM
SAM_vs_ground_truth_dsc = []
SAMlm_dsc_list = []
SAMlv_dsc_list = []
SAMright_dsc_list = []

SAM_vs_ground_truth_csa = []
SAMlm_csa_list = []
SAMlv_csa_list = []
SAMright_csa_list = []

SAM_vs_ground_truth_hd = []
SAMlm_hd_list = []
SAMlv_hd_list = []
SAMright_hd_list = []

# Loop over the images in the input folder, Segment images and save object level information into a csv file
jr = 0      # parser.get_default("--data_path")
listlista = os.listdir(ts_img_path)
for image_filename in listlista:
    image_data = np.load(join(ts_img_path, image_filename))
    # read ground truth (gt should have the same name as the image) and simulate a bounding box
    gt_data = np.load(join(ts_gt_path, image_filename))
    bbox_raw = get_bbox_from_mask(gt_data)

    # preprocess: cut-off and max-min normalization
    lower_bound, upper_bound = np.percentile(image_data, 0.5), np.percentile(image_data, 99.5)
    image_data_pre = np.clip(image_data, lower_bound, upper_bound)
    image_data_pre = (image_data_pre - np.min(image_data_pre)) / (
                np.max(image_data_pre) - np.min(image_data_pre)) * 255.0
    image_data_pre[image_data == 0] = 0
    image_data_pre = np.uint8(image_data_pre)

    # predict the segmentation mask using the original SAM model
    ori_sam_predictor.set_image(image_data_pre)
    # predict the segmentation mask using the BIRL SAM model
    new_sam_predictor.set_image(image_data_pre)
    # predict the segmentation mask using the medSAM model
    med_sam_predictor.set_image(image_data_pre)

    med_sam = np.zeros((3, 1024, 1024))
    BIRL_sam = np.zeros((3, 1024, 1024))
    SAM_sam = np.zeros((3, 1024, 1024))
    for jj in range(np.max(gt_data)):
        # SAM prediction
        SAM_sam[jj], _, _ = ori_sam_predictor.predict(point_coords=None, box=bbox_raw[jj], multimask_output=False)
        SAM_sam[jj] = np.asarray(SAM_sam[jj] * (jj + 1), np.uint8)
        # medSAM prediction
        med_sam[jj], _, _ = med_sam_predictor.predict(point_coords=None, box=bbox_raw[jj], multimask_output=False)
        med_sam[jj] = np.asarray(med_sam[jj] * (jj + 1), np.uint8)
        # BIRLmedSAM prediction
        BIRL_sam[jj], _, _ = new_sam_predictor.predict(point_coords=None, box=bbox_raw[jj], multimask_output=False)
        BIRL_sam[jj] = np.asarray(BIRL_sam[jj] * (jj + 1), np.uint8)

    lab_mask, lm_dsc, lv_dsc, right_dsc, average_dsc, lm_hd, lv_hd, right_hd, average_hd, lm_csa, lv_csa, right_csa, average_csa = Compute_MasknMetrics(
        BIRL_sam, gt_data)
    # Sore metrics for excel saving BIRL
    model_vs_ground_truth_dsc.append(average_dsc)
    lm_dsc_list.append(lm_dsc)
    lv_dsc_list.append(lv_dsc)
    right_dsc_list.append(right_dsc)
    # Compute HD between MFP and ground truth take care of unpredicted labels
    model_vs_ground_truth_hd.append(average_hd)
    lm_hd_list.append(lm_hd)
    lv_hd_list.append(lv_hd)
    right_hd_list.append(right_hd)
    # Compute CSA between MFP and ground truth
    model_vs_ground_truth_csa.append(average_csa)
    lm_csa_list.append(lm_csa)
    lv_csa_list.append(lv_csa)
    right_csa_list.append(right_csa)

    SAM_mask, Slm_dsc, Slv_dsc, Srv_dsc, Savg_dsc, Slm_hd, Slv_hd, Srv_hd, Savg_hd, Slm_csa, Slv_csa, Srv_csa, Savg_csa = Compute_MasknMetrics(
        med_sam, gt_data)
    # Sore metrics for excel saving SAM
    SAM_vs_ground_truth_dsc.append(Savg_dsc)
    SAMlm_dsc_list.append(Slm_dsc)
    SAMlv_dsc_list.append(Slv_dsc)
    SAMright_dsc_list.append(Srv_dsc)
    # Compute HD between MFP and ground truth take care of unpredicted labels
    SAM_vs_ground_truth_hd.append(Savg_hd)
    SAMlm_hd_list.append(Slm_hd)
    SAMlv_hd_list.append(Slv_hd)
    SAMright_hd_list.append(Srv_hd)
    # Compute CSA between MFP and ground truth
    SAM_vs_ground_truth_csa.append(Savg_csa)
    SAMlm_csa_list.append(Slm_csa)
    SAMlv_csa_list.append(Slv_csa)
    SAMright_csa_list.append(Srv_csa)

    med_mask, Mlm_dsc, Mlv_dsc, Mrv_dsc, Mavg_dsc, Mlm_hd, Mlv_hd, Mrv_hd, Mavg_hd, Mlm_csa, Mlv_csa, Mrv_csa, Mavg_csa = Compute_MasknMetrics(
        SAM_sam, gt_data)
    # Sore metrics for excel saving medSAM
    mSAM_vs_ground_truth_dsc.append(Mavg_dsc)
    mSAMlm_dsc_list.append(Mlm_dsc)
    mSAMlv_dsc_list.append(Mlv_dsc)
    mSAMright_dsc_list.append(Mrv_dsc)
    # Compute HD between MFP and ground truth take care of unpredicted labels
    mSAM_vs_ground_truth_hd.append(Mavg_hd)
    mSAMlm_hd_list.append(Mlm_hd)
    mSAMlv_hd_list.append(Mlv_hd)
    mSAMright_hd_list.append(Mrv_hd)
    # Compute CSA between MFP and ground truth
    mSAM_vs_ground_truth_csa.append(Mavg_csa)
    mSAMlm_csa_list.append(Mlm_csa)
    mSAMlv_csa_list.append(Mlv_csa)
    mSAMright_csa_list.append(Mrv_csa)

    plt.ioff()
    fig = plt.figure(figsize=(20, 8))
    fig.suptitle(image_filename.split('.')[0])
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(image_data)
    ax1.set_title("Frame", fontsize=16)
    ax1.grid(False)  # Changed from .grid(b=None)
    ax1.text(0.5, -0.25, metric_info_string(metric='CSA', lm_m=Slm_csa, lv_m=Slv_csa, right_m=Srv_csa, average_m=Savg_csa),
             size=12, ha="center", transform=ax1.transAxes)

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.set_title("Ground truth", fontsize=16)
    ax2.imshow(gt_data)
    ax2.grid(False)
    ax2.text(0.5, -0.25, metric_info_string(metric='HD', lm_m=Slm_hd, lv_m=Slv_hd, right_m=Srv_hd, average_m=Savg_hd),
             size=12, ha="center", transform=ax2.transAxes)

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.set_title('Predicted by \n' + "SAM Model", fontsize=16)  # ('Predicted by \n' + AI_name, fontsize=16)
    ax3.imshow(SAM_mask)
    ax3.grid(False)
    ax3.text(0.5, -0.25, metric_info_string(metric='DSC', lm_m=Slm_dsc, lv_m=Slv_dsc, right_m=Srv_dsc, average_m=Savg_dsc),
             size=12, ha="center", transform=ax3.transAxes)
    print('Saving %s' % (output_directory + '/ResultsS_%d.png' % jr))
    plt.savefig(output_directory + '/Results/ResultsS_%d.png' % jr)  # plt.show()
    plt.close(fig)

    fig = plt.figure(figsize=(20, 8))
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(image_data)
    ax1.set_title("Frame", fontsize=16)
    ax1.grid(False)  # Changed from .grid(b=None)
    ax1.text(0.5, -0.25,
             metric_info_string(metric='CSA', lm_m=Mlm_csa, lv_m=Mlv_csa, right_m=Mrv_csa, average_m=Mavg_csa),
             size=12, ha="center", transform=ax1.transAxes)

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.set_title("Ground truth", fontsize=16)
    ax2.imshow(gt_data)
    ax2.grid(False)
    ax2.text(0.5, -0.25,
             metric_info_string(metric='HD', lm_m=Mlm_hd, lv_m=Mlv_hd, right_m=Mrv_hd, average_m=Mavg_hd),
             size=12, ha="center", transform=ax2.transAxes)

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.set_title('Predicted by \n' + "medSAM Model", fontsize=16)  # ('Predicted by \n' + AI_name, fontsize=16)
    ax3.imshow(med_mask)
    ax3.grid(False)
    ax3.text(0.5, -0.25,
             metric_info_string(metric='DSC', lm_m=Mlm_dsc, lv_m=Mlv_dsc, right_m=Mrv_dsc, average_m=Mavg_dsc),
             size=12, ha="center", transform=ax3.transAxes)
    print('Saving %s' % (output_directory + '/ResultsM_%d.png' % jr))
    plt.savefig(output_directory + '/Results/ResultsM_%d.png' % jr)  # plt.show()
    plt.close(fig)

    fig = plt.figure(figsize=(20, 8))
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(image_data)
    ax1.set_title("Frame", fontsize=16)
    ax1.grid(False)  # Changed from .grid(b=None)
    ax1.text(0.5, -0.25,
             metric_info_string(metric='CSA', lm_m=lm_csa, lv_m=lv_csa, right_m=right_csa, average_m=average_csa),
             size=12, ha="center", transform=ax1.transAxes)

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.set_title("Ground truth", fontsize=16)
    ax2.imshow(gt_data)
    ax2.grid(False)
    ax2.text(0.5, -0.25,
             metric_info_string(metric='HD', lm_m=lm_hd, lv_m=lv_hd, right_m=right_hd, average_m=average_hd),
             size=12, ha="center", transform=ax2.transAxes)

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.set_title('Predicted by \n' + "BIRL-medSAM", fontsize=16)  # ('Predicted by \n' + AI_name, fontsize=16)
    ax3.imshow(lab_mask)
    ax3.grid(False)
    ax3.text(0.5, -0.25,
             metric_info_string(metric='DSC', lm_m=lm_dsc, lv_m=lv_dsc, right_m=right_dsc, average_m=average_dsc),
             size=12, ha="center", transform=ax3.transAxes)

    print('Saving %s' % (output_directory + '/ResultsB_%d.png' % jr))
    plt.savefig(output_directory + '/Results/ResultsB_%d.png' % jr)  # plt.show()
    plt.close(fig)
    jr += 1

print(".::: Segmentation of all images completed and metrics computed :::.")
# save excel files
saveExcels(output_directory, 'BIRLmedSAM Metrics.xlsx', model_vs_ground_truth_dsc, lm_dsc_list, lv_dsc_list, right_dsc_list,
           model_vs_ground_truth_hd, lm_hd_list, lv_hd_list, right_hd_list, model_vs_ground_truth_csa, lm_csa_list, lv_csa_list, right_csa_list, listlista)
saveExcels(output_directory, 'medSAM Metrics.xlsx', mSAM_vs_ground_truth_dsc, mSAMlm_dsc_list, mSAMlv_dsc_list, mSAMright_dsc_list, mSAM_vs_ground_truth_hd,
           mSAMlm_hd_list, mSAMlv_hd_list, mSAMright_hd_list, mSAM_vs_ground_truth_csa, mSAMlm_csa_list, mSAMlv_csa_list, mSAMright_csa_list, listlista)
saveExcels(output_directory, 'SAM Metrics.xlsx', SAM_vs_ground_truth_dsc, SAMlm_dsc_list, SAMlv_dsc_list, SAMright_dsc_list, SAM_vs_ground_truth_hd,
           SAMlm_hd_list, SAMlv_hd_list, SAMright_hd_list, SAM_vs_ground_truth_csa, SAMlm_csa_list, SAMlv_csa_list, SAMright_csa_list, listlista)
print('.::: Stop here :::.')