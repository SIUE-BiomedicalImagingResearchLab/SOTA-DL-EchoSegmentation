import numpy as np
import pandas as pd
import os, json, cv2, random, yaml, csv
from skimage.measure import regionprops, label
from scipy.spatial.distance import directed_hausdorff
from matplotlib import pyplot as plt

# import some common detectron2 utilities
import torch, detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.utils.visualizer import ColorMode

# show system info
os.system('nvcc --version')
TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
print("detectron2:", detectron2.__version__)


def dice_similarity(a, b):
    # Check for empty images
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


def metric_info_string(metric, background_m, lm_m, lv_m, right_m, average_m):
    return "Average %s: %.3f\n" \
           "Background %s: %.3f\n" \
           "Left Myocardium %s: %.3f\n" \
           "Left Ventricle %s: %.3f\n" \
           "Right Ventricle %s: %.3f" % (metric, average_m, metric, background_m, metric, lm_m, metric, lv_m, metric, right_m)


# Train on a custom dataset (https://colab.research.google.com/drive/1iy9MPSS5KITmL6DB74mNPjCos2dr7Gov?authuser=3#scrollTo=M2yUBzSPFPAS)
''' Import the necessary function to register datasets in the COCO format. Let us register both the training and 
validation datasets. Please note that we are working with training (and validation) data that is is the coco format 
where we have a single JSON file that describes all the annotations from all training images. Here, we are naming our 
training data as 'my_dataset_train' and the validation data as 'my_dataset_val'. '''
#register_coco_instances("my_dataset_train", {}, "../data/Experiment/Training/Train_annotations.json",
 #                       "../data/Experiment/Training/pngs")
register_coco_instances("my_dataset_val", {}, "../data/Experiment/Validation/Val_annotations.json",
                        "../data/Experiment/Validation/pngs")

''' extract the metadata and dataset dictionaries for both training and validation datasets. These can be used later 
for other purposes, like visualization, model training, evaluation, etc. We will see a visualization example right away. '''
#train_metadata = MetadataCatalog.get("my_dataset_train")
#train_dataset_dicts = DatasetCatalog.get("my_dataset_train")
val_metadata = MetadataCatalog.get("my_dataset_val")
val_dataset_dicts = DatasetCatalog.get("my_dataset_val")

# Visualize some random samples from Train or Val pngs
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
for ax, d in zip(axs.ravel(), random.sample(val_dataset_dicts, 4)):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=val_metadata, scale=0.5)
    vis = visualizer.draw_dataset_dict(d)

    ax.imshow(vis.get_image()[:, :, ::-1])
    ax.axis('off')  # Turn off the axes
    ax.set_title(d["file_name"].split('/')[-1])

plt.show()
plt.pause(10)
plt.close()

# Train
''' Now we are ready to train a Mask R-CNN model using the Detectron2 library. We start by setting up a configuration 
file (.cfg) for the model. The configuration file contains many details including the output directory path, 
training dataset information, pre-trained weights, base learning rate, maximum number of iterations, etc.'''
dev = "cpu" # cpu or cuda

cfg = get_cfg()
cfg.OUTPUT_DIR = "../Data/SegmentOutput/Detec2Mdl"
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("my_dataset_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
# uncomment this for training
#cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 1000    # 1000 iterations seems good enough for this dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256   # Default is 512, using 256 for this dataset.
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # We have 3 classes.
cfg.MODEL.DEVICE = dev # cpu or cuda
# NOTE: this config means the number of classes, without the background. Do not use num_classes+1 here.

#os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
# trainer = DefaultTrainer(cfg) #Create an instance of of DefaultTrainer with the given congiguration
# trainer.resume_or_load(resume=True) #Load a pretrained model if available (resume training) or start training from scratch if no pretrained model is available
#
# trainer.train() #Start the training process
#
# # Save the configuration to a config.yaml file
# config_yaml_path = cfg.OUTPUT_DIR + "/config.yaml"
# with open(config_yaml_path, 'w') as file:
#     yaml.dump(cfg, file)

# Inference should use the config with parameters that are used in training
# cfg now already contains everything we've set previously. We changed it a little bit for inference:
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.85   # set a custom testing threshold
predictor = DefaultPredictor(cfg)
# Verify segmentation on random validation images
# for d in random.sample(val_dataset_dicts, 1):    #select number of images for display
#     im = cv2.imread(d["file_name"])
#     outputs = predictor(im)
#     v = Visualizer(im[:, :, ::-1],
#                    metadata=val_metadata,
#                    scale=1.5,
#                    instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
#     )
#     out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
#     cv2.imshow('fIRST PREDICTION', out.get_image()[:, :, ::-1])
#     cv2.waitKey()
#     cv2.destroyAllWindows()

# Check average precision and recall. (Need more validation data than just 2 images with handful of annotations)
# another equivalent way to evaluate the model is to use `trainer.test`
# evaluator = COCOEvaluator("my_dataset_val", output_dir="./output")
# val_loader = build_detection_test_loader(cfg, "my_dataset_val")
# print(inference_on_dataset(predictor.model, val_loader, evaluator))

# Load a new image and segment it.
# new_im = cv2.imread("../data/Experiment/Testing/pngs/MF0307POST_5_ES_35_Julie.png")
# outputs = predictor(new_im)
#
# # We can use `Visualizer` to draw the predictions on the image.
# v = Visualizer(new_im[:, :, ::-1], metadata=train_metadata)
# out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
# cv2.imshow('Prediction on test data', out.get_image()[:, :, ::-1])

# Process multiple images in a directory and save the results in an output directory
# Directory path to the input images folder
input_images_directory = "../data/Experiment/Testing"
# Output directory where the segmented images will be saved
output_directory = cfg.OUTPUT_DIR + "/Results"  # Replace this with the path to your desired output directory
# Output directory where the CSV file will be saved
output_csv_path = cfg.OUTPUT_DIR + "/output_objects.csv"

# Metrics vbles
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
# Open the CSV file for writing
with open(output_csv_path, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)

    # Write the header row in the CSV file
    csvwriter.writerow(["File Name", "Class Name", "Object Number", "Area", "Centroid", "BoundingBox"])  # Add more columns as needed for other properties
    # Loop over the images in the input folder, Segment images and save object level information into a csv file
    jr = 0
    listlista = os.listdir(input_images_directory + "/pngs")
    for image_filename in listlista:
        image_path = os.path.join(input_images_directory, "pngs", image_filename)
        new_im = cv2.imread(image_path)
        gt = np.load(input_images_directory + "/gt_mS/" + image_filename.replace('.png', '.npy'))

        # Perform prediction on the new image
        outputs = predictor(new_im)  # Format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format

        # Convert the predicted mask to a binary mask
        mask = outputs["instances"].pred_masks.to("cpu").numpy().astype(np.uint8)
        lab_mask = np.zeros((256, 256)).astype(np.uint8)
        labeled_mask = np.zeros((3, 256, 256), dtype=np.uint8)

        if mask.shape[0] != 0:
            # Get the predicted class labels
            class_labels = outputs["instances"].pred_classes.to("cpu").numpy()
            Clabs = class_labels + 1

            # Debugging: print class_labels and metadata.thing_classes
            # print("Class Labels:", class_labels)
            print("Thing Classes:", train_metadata.thing_classes)
            print(outputs["instances"].scores)

            # select the largest object per label and create a 1 band mask
            # get duplicates lbls and indexes
            uniEle, idx = np.unique(Clabs, return_inverse=True)
            counts = np.bincount(idx)
            dups = uniEle[counts > 1]
            unis = uniEle[counts == 1]
            if len(unis > 0): # save not repeated labels
                for iddx in range(len(unis)):
                    labeled_mask[iddx] = mask[np.where(Clabs == unis[iddx])]

            if len(dups) > 0:
                areas = []
                dupsIdx = [np.where(Clabs == d)[0].tolist() for d in dups]
                #props = regionprops(label(mask+1))
                for iddx in range(len(dupsIdx)):
                    for idxx in range(len(dupsIdx[iddx])):
                        areas.append(regionprops(mask[dupsIdx[0][idxx]])[0].area)

                    fIdx = np.asarray(np.where(areas == max(areas)))[0][0]
                    labeled_mask[len(unis) + iddx] = mask[dupsIdx[iddx][fIdx]]
                    areas = []
                mask = labeled_mask
                Clabs = uniEle

            for lbl in Clabs:
                #get index of RV label (3)
                idx = np.where(Clabs == lbl)
                if len(idx) > 0 and lbl == 3:   # RV
                    labeled_mask[idx, :, :] = mask[idx, :, :] * 3
                elif len(idx) > 0:
                    labeled_mask[idx, :, :] = mask[idx, :, :]

            lab_mask = labeled_mask[0, :, :] + labeled_mask[1, :, :] + labeled_mask[2, :, :]
            if np.max(lab_mask) > 3:    # fix RV, LE overlap (3 + 1)
                lab_mask[lab_mask > 3] = 1

            # Write the object-level information to the CSV file
            props = regionprops(lab_mask)
            for i, prop in enumerate(props):
                object_number = i + 1  # Object number starts from 1
                area = prop.area
                centroid = prop.centroid
                bounding_box = prop.bbox

                # Check if the corresponding class label exists
                if i < len(class_labels):
                    class_label = class_labels[i]
                    class_name = train_metadata.thing_classes[class_label]
                else:
                    # If class label is not available (should not happen), use 'Unknown' as class name
                    class_name = 'Unknown'

                # Write the object-level information to the CSV file
                csvwriter.writerow([image_filename, class_name, object_number, area, centroid,
                                    bounding_box])  # Add more columns as needed for other properties

        # Extrac objects properties
        props = regionprops(lab_mask)
        props_gt = regionprops(gt)

        # Compute DSC between MFP and ground truth
        lm_dsc = dice_similarity((lab_mask == 1).astype(np.uint8), (gt == 1).astype(np.uint8))
        lv_dsc = dice_similarity((lab_mask == 2).astype(np.uint8), (gt == 2).astype(np.uint8))
        right_dsc = dice_similarity((lab_mask == 3).astype(np.uint8), (gt == 3).astype(np.uint8))
        average_dsc = np.average([lm_dsc, lv_dsc, right_dsc])
        model_vs_ground_truth_dsc.append(average_dsc)
        lm_dsc_list.append(lm_dsc)
        lv_dsc_list.append(lv_dsc)
        right_dsc_list.append(right_dsc)

        # Compute HD between MFP and ground truth take care of unpredicted labels
        # if a labels is not predicted dsc is computed agins a square located in the middle of image
        missLbl = 0
        bModeAreaPx2PerMm2 = 8.885960105352543
        if lm_dsc == 0:
            lm_hd = Hausdorff_Distance([[100, 100], [100, 156], [156, 156], [156, 100]], props_gt[0].coords.tolist())
            lm_csa = abs(0 - props_gt[0].area) / bModeAreaPx2PerMm2
            missLbl += 1    # increase missed lbl counter
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
        model_vs_ground_truth_hd.append(average_hd)
        lm_hd_list.append(lm_hd)
        lv_hd_list.append(lv_hd)
        right_hd_list.append(right_hd)

        # Compute CSA between MFP and ground truth
        average_csa = np.average([lm_csa, lv_csa, right_csa])
        model_vs_ground_truth_csa.append(average_csa)
        lm_csa_list.append(lm_csa)
        lv_csa_list.append(lv_csa)
        right_csa_list.append(right_csa)

        # We can use `Visualizer` to draw the predictions on the image.
        v = Visualizer(new_im[:, :, ::-1], metadata=train_metadata)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        # Create the output filename with _result extension
        # result_filename = os.path.splitext(image_filename)[0] + "_result.png"
        # output_path = os.path.join(output_directory, result_filename)

        plt.ioff()
        fig = plt.figure(figsize=(20, 8))
        fig.suptitle(image_filename.split('.')[0])
        ax1 = fig.add_subplot(1, 3, 1)
        ax1.imshow(new_im)
        ax1.set_title("Frame", fontsize=16)
        ax1.grid(False)  # Changed from .grid(b=None)
        ax1.text(0.5, -0.25, metric_info_string(metric='CSA', background_m=0, lm_m=lm_csa,
                                                lv_m=lv_csa, right_m=right_csa, average_m=average_csa),
                 size=12, ha="center", transform=ax1.transAxes)

        ax2 = fig.add_subplot(1, 3, 2)
        ax2.set_title("Ground truth", fontsize=16)
        ax2.imshow(gt)
        ax2.grid(False)
        ax2.text(0.5, -0.25, metric_info_string(metric='HD', background_m=0, lm_m=lm_hd,
                                                lv_m=lv_hd, right_m=right_hd, average_m=average_hd),
                 size=12, ha="center", transform=ax2.transAxes)

        ax3 = fig.add_subplot(1, 3, 3)
        ax3.set_title('Predicted by \n' + "BIRL-Detec2", fontsize=16)  # ('Predicted by \n' + AI_name, fontsize=16)
        ax3.imshow(out.get_image()[:, :, ::-1]) #lab_mask)
        ax3.grid(False)
        ax3.text(0.5, -0.25, metric_info_string(metric='DSC', background_m=0, lm_m=lm_dsc,
                                                lv_m=lv_dsc, right_m=right_dsc, average_m=average_dsc),
                 size=12, ha="center", transform=ax3.transAxes)
        print('Saving %s' % (output_directory + '/Results_%d.png' % jr))

        plt.savefig(output_directory + '/Results_%d.png' % jr)  # plt.show()
        plt.close(fig)
        jr += 1
        # Save the segmented image
        #cv2.imwrite(output_path, out.get_image()[:, :, ::-1])

print(".::: Segmentation of all images completed and Object-level information saved to CSV file :::.")
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
caseDescription = "Detectron2 Segmentation"
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

df.to_excel(output_directory[:-len(output_directory.split('/')[-1])] + 'Detec2BIRL Metrics.xlsx')

''' Saving binary (actually multinary) images for each class for further processing. Here, for each input image we will 
save n images corresponding to the number of classes. In our example, we will save 4 images for each image corresponding 
to the 4 classes. Each of these images will contain objects numbered 1, 2, 3, etc. - basically instance segmentation 
like images. These images can be used for further downstream processing.'''
# Loop over the images in the input folder
# for image_filename in os.listdir(input_images_directory):
#     image_path = os.path.join(input_images_directory, image_filename)
#     new_im = cv2.imread(image_path)
#
#     # Perform prediction on the new image
#     outputs = predictor(new_im)  # Format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
#
#     # Create a dictionary to store the mask for each class with unique integer labels
#     class_masks = {class_name: torch.zeros_like(outputs["instances"].pred_masks[0], dtype=torch.uint8, device=torch.device("cuda:0"))
#                    for class_name in train_metadata.thing_classes}
#
#     # Assign a unique integer label to each object in the mask
#     for i, pred_class in enumerate(outputs["instances"].pred_classes):
#         class_name = train_metadata.thing_classes[pred_class]
#         class_masks[class_name] = torch.where(outputs["instances"].pred_masks[i].to(device=torch.device("cuda:0")),
#                                               i + 1,
#                                               class_masks[class_name])
#
#     # Save the masks for each class with unique integer labels
#     for class_name, class_mask in class_masks.items():
#         # Convert the tensor to a NumPy array and then to a regular (CPU) array
#         class_mask_np = class_mask.cpu().numpy()
#
#         # Create the output filename with _class_name_result.png extension
#         class_filename = os.path.splitext(image_filename)[0] + f"_{class_name}_result.png"
#         class_output_path = os.path.join(output_directory, class_filename)
#
#         # Save the image with unique integer labels
#         cv2.imwrite(class_output_path, class_mask_np.astype(np.uint8))
#
# print(".::: Tasks completed :::.")