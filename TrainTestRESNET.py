import pandas
import pandas as pd

# from Experiment.plotROC import plotROC

from Training.UnetModel.mfpUnet import *
from Training.UnetModel.pre_processing import *
from Training.UnetModel.resnet import ResnetBuilder
from Training.UnetModel.trainModel import *
import matplotlib.pyplot as plt
import numpy as np
# from tensorflow_core.python.keras.utils import plot_model
from Training.UnetModel.unet import unet
from PostProcessing.postProcessing import *  # extract_tissue_labels
from Preprocessing.ParametricImageGeneration.functions import *  # To do postProc at once
from util.util import getCaseDescription
from datetime import datetime

from scipy.spatial.distance import directed_hausdorff

def make_loss_plot(training_log_file=None, saving_path=None):
    if not training_log_file:
        return

    df = pandas.read_csv(training_log_file)

    val_loss = tuple(df['val_loss'])
    train_loss = tuple(df['loss'])
    val_acc = tuple(df['val_accuracy'])
    train_acc = tuple(df['accuracy'])
    val_dsc = tuple(df['val_dice_coef'])
    train_dsc = tuple(df['dice_coef'])

    # Loss plot
    fig, ax = plt.subplots(nrows=3, ncols=1)
    ax[0].plot(train_loss)
    ax[0].plot(val_loss)
    ax[0].set_title('Model loss', fontsize=14)
    ax[0].legend(['Train', 'Validation'], loc='upper right')
    ax[0].set_ylabel('Loss')
    #ax[0].set_xlabel('Epoch')
    ax[0].set_xticklabels([])
    ax[0].grid(color='lightgray')

    # Accuracy plot
    ax[1].plot(train_acc)
    ax[1].plot(val_acc)
    ax[1].set_title('Model Accuracy', fontsize=14)
    ax[1].legend(['Train', 'Validation'], loc='lower right')
    ax[1].set_ylabel('Accuracy')
    #ax[1].set_xlabel('Epoch')
    ax[1].set_xticklabels([])
    ax[1].grid(color='lightgray')

    # DSC plot
    ax[2].plot(train_dsc)
    ax[2].plot(val_dsc)
    ax[2].set_title('Model DSC', fontsize=14)
    ax[2].legend(['Train', 'Validation'], loc='lower right')
    ax[2].set_ylabel('DSC')
    ax[2].set_xlabel('Epoch')
    ax[2].grid(color='lightgray')

    plt.tight_layout()

    if saving_path:
        plt.savefig(saving_path + '/' + 'plots.png')
    else:
        plt.show()

    plt.close()

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
        HD = max(directed_hausdorff(u, v)[0], directed_hausdorff(v, u)[0]) / bModeDistPxPerMM
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


def resDim(del_lv):
    ddel_lm = np.uint8(cv2.resize(del_lv[:, :, 0], (201, 596), interpolation=cv2.INTER_LINEAR))
    ddel_lm[ddel_lm > 0] = 255
    ddel_lm = find_largest_contour(ddel_lm)
    ddel_lm = convert_mask_from_polar_to_cartesian(ddel_lm)

    colored_delineated = np.uint8(np.zeros((600, 800, 3)))
    for ii in range(3):
        colored_delineated[:, :, ii] = ddel_lm

    return colored_delineated


def printOutResults(model, modelPath, frameDataPath, maskDataPath, loss_log=None, savingPath='./Results',
                    training_detail=None, selected_band=None, z_score=True, saveTracings=0):
    batch_size = len(os.listdir(frameDataPath))
    batch_img, batch_mask = test_generator(frameDataPath=frameDataPath, maskDataPath=maskDataPath,
                                           z_score=z_score, selected_band=selected_band)

    model.load_weights(modelPath)

    ultrasoundTopLevelPath = config['TRACED_DATA_PROCESSING']['ULTRASOUND_DATA_PATH']
    AI_Name = config['SEGMENTATION_TRAINING_SETTINGS']['AI_NAME']
    # check/create directories
    if saveTracings == 1 or saveTracings == 2:
        if not os.path.exists(savingPath + '/' + 'Results'):
            os.makedirs(savingPath + '/' + 'Results')

    if not os.path.exists(savingPath):
        os.makedirs(savingPath)

    # plot model summary
    # plot_model(model, to_file=savingPath + '/' + 'model.png',
    #            show_shapes=True, show_layer_names=True)

    # save info of training settings
    with open(savingPath + '/' + "validation_info.txt", 'w') as f:
        for key, value in training_detail.items():
            f.write('%s:%s\n' % (key, value))

    if loss_log:
        make_loss_plot(training_log_file=loss_log, saving_path=savingPath)

    testing_list = os.listdir(frameDataPath)

    model_vs_ground_truth_dsc = []
    background_dsc_list = []
    lm_dsc_list = []
    lv_dsc_list = []
    right_dsc_list = []

    model_vs_ground_truth_csa = []
    lm_csa_list = []
    lv_csa_list = []
    right_csa_list = []

    model_vs_ground_truth_hd = []
    bg_csa_list = []
    lm_hd_list = []
    lv_hd_list = []
    right_hd_list = []
    #pred_prob = []

    for i in range(0, batch_size):
        pred_mfp = model.predict(np.array([batch_img[i]]))[0]
        #pred_prob.append(pred_mfp)

        gt = onehot_to_rgb(batch_mask[i], ID_TO_CODE)
        gt_background, gt_lm, gt_lv, gt_right = extract_tissue_labels(gt, get_background=True)

        mfp = onehot_to_rgb(pred_mfp, ID_TO_CODE)
        mfp_background, mfp_lm, mfp_lv, mfp_right = extract_tissue_labels(mfp, get_background=True)

        # Compute DSC between MFP and ground truth
        background_dsc = dice_similarity(mfp_background, gt_background)
        lm_dsc = dice_similarity(mfp_lm, gt_lm)
        lv_dsc = dice_similarity(mfp_lv, gt_lv)
        right_dsc = dice_similarity(mfp_right, gt_right)
        average_dsc = np.average([background_dsc, lm_dsc, lv_dsc, right_dsc])
        model_vs_ground_truth_dsc.append(average_dsc)

        background_dsc_list.append(background_dsc)
        lm_dsc_list.append(lm_dsc)
        lv_dsc_list.append(lv_dsc)
        right_dsc_list.append(right_dsc)

        # extract contours from GT
        congt_lm = find_largest_contour(gt_lm)
        congt_lm = list_contours(congt_lm)
        # delineate left vent
        congt_lv = find_largest_contour(gt_lv)
        congt_lv = list_contours(congt_lv)
        # delineate right vent
        congt_rv = find_largest_contour(gt_right)
        congt_rv = list_contours(congt_rv)

        # extract contours and delineated images from MFP
        del_lm = find_largest_contour(mfp_lm)
        con_lm = list_contours(del_lm)
        # delineate left vent
        del_lv = find_largest_contour(mfp_lv)
        con_lv = list_contours(del_lv)
        # delineate right vent
        del_rv = find_largest_contour(mfp_right)
        con_rv = list_contours(del_rv)
        combined = mfp_lm + mfp_right
        Ccon_rv = find_largest_contour(combined)
        # combine to get RV
        final_delineated_right_vent = np.uint8(cv2.bitwise_and(Ccon_rv, del_rv))

        # Compute HD between MFP and ground truth
        lm_hd = Hausdorff_Distance(congt_lm, con_lm)
        lv_hd = Hausdorff_Distance(congt_lv, con_lv)
        right_hd = Hausdorff_Distance(congt_rv, con_rv)
        average_hd = np.average([lm_hd, lv_hd, right_hd])
        model_vs_ground_truth_hd.append(average_hd)

        lm_hd_list.append(lm_hd)
        lv_hd_list.append(lv_hd)
        right_hd_list.append(right_hd)

        # Compute CSA between MFP and ground truth
        bg_csa = abs(np.count_nonzero(mfp_background) - np.count_nonzero(gt_background)) / bModeAreaPx2PerMm2
        lm_csa = abs(np.count_nonzero(mfp_lm) - np.count_nonzero(gt_lm)) / bModeAreaPx2PerMm2
        lv_csa = abs(np.count_nonzero(mfp_lv) - np.count_nonzero(gt_lv)) / bModeAreaPx2PerMm2
        right_csa = abs(np.count_nonzero(mfp_right) - np.count_nonzero(gt_right)) / bModeAreaPx2PerMm2
        average_csa = np.average([bg_csa, lm_csa, lv_csa, right_csa])
        model_vs_ground_truth_csa.append(average_csa)

        bg_csa_list.append(bg_csa)
        lm_csa_list.append(lm_csa)
        lv_csa_list.append(lv_csa)
        right_csa_list.append(right_csa)

        if saveTracings == 1 or saveTracings == 2:
            plt.ioff()
            fig = plt.figure(figsize=(20, 8))
            fig.suptitle(testing_list[i].split('.')[0])
            ax1 = fig.add_subplot(1, 3, 1)
            ax1.imshow(batch_img[i][:, :, 0])
            ax1.set_title("Frame", fontsize=16)
            ax1.grid(False)  # Changed from .grid(b=None)
            ax1.text(0.5, -0.25, metric_info_string(metric='CSA', background_m=bg_csa, lm_m=lm_csa,
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
            ax3.set_title('Predicted by \n' + AI_Name, fontsize=16)  # ('Predicted by \n' + AI_name, fontsize=16)
            ax3.imshow(mfp)
            ax3.grid(False)
            ax3.text(0.5, -0.25, metric_info_string(metric='DSC', background_m=background_dsc, lm_m=lm_dsc,
                                                 lv_m=lv_dsc, right_m=right_dsc, average_m=average_dsc),
                                                 size=12, ha="center", transform=ax3.transAxes)
            print('Saving %s' % (savingPath + '/Results/Results_%d.png' % i))

            plt.savefig(savingPath + '/Results/Results_%d.png' % i)  # plt.show()
            plt.close(fig)

        if saveTracings == 2 or saveTracings == 3:
            # Save predicted tracings as NRRD file
            # convert to BMode images
            del_lm = resDim(del_lm)
            del_lv = resDim(del_lv)
            final_delineated_right_vent = resDim(final_delineated_right_vent)
            # label_codes = [(0, 0, 0), (128, 174, 128), (141, 93, 137), (181, 85, 57)]
            colored_delineated_left_myo = color_image(del_lm, color=LABEL_CODES[1][::-1])
            colored_delineated_left_vent = color_image(del_lv, color=[0, 25, 255][::-1])
            colored_final_delineated_right_vent = color_image(final_delineated_right_vent, color=LABEL_CODES[3][::-1])

            # Load original background image
            subjectID, prepost, image_number, phase, sliceIndex = getInfosFromFileName(testing_list[i].split('.')[0])
            prepost = prepost.replace('_', '-')  # UPDATED TO KEEP ORIGINAL US DATA FOLDER STRUCTURE
            dataToBeSaved, header = nrrd.read(ultrasoundTopLevelPath + '/%s/%s/Traced Contour/%s/' % \
                                              (subjectID, prepost, phase) + testing_list[i].split('.')[0] + '.nrrd')

            # Update data with contours from AI
            dataToBeSaved[1, :, :, 0:3] = colored_delineated_left_vent[:, :, ::-1]  # Left Endo
            dataToBeSaved[1, :, :, 3] = dataToBeSaved[1, :, :, 2]  # 4th band fix
            dataToBeSaved[2, :, :, 0:3] = colored_delineated_left_myo[:, :, ::-1]  # Left Epi
            dataToBeSaved[2, :, :, 3] = dataToBeSaved[2, :, :, 2]  # 4th band fix
            temp = dataToBeSaved[2, :, :, 3]
            temp[temp > 120] = 255
            dataToBeSaved[3, :, :, 0:3] = colored_final_delineated_right_vent[:, :, ::-1]  # Right Epi
            dataToBeSaved[3, :, :, 3] = dataToBeSaved[3, :, :, 2]  # 4th band fix
            temp = dataToBeSaved[3, :, :, 3]
            temp[temp > 50] = 255

            customFieldMap = {'Layer Names': 'string list', 'Frame Name': 'string', 'Frame Index': 'int',
                              'Tracing Infos': 'string list', 'Cardiac Phase': 'string', 'Color space': 'string',
                              'tgc': 'int list', 'masterGain': 'int', 'noiseReduction': 'string',
                              'dynamicRange': 'int list'
                              }
            header['Layer Names'] = ['Background', 'LeftEndocardium', 'LeftEpicardium', 'RightVentricle']

            # check if there is a previous observer and remove it
            try:
                obs = len(testing_list[i].split('.')[0].split('_')[4])
            except:
                obs = 0

            if obs > 0:
                filename = testing_list[i].split('.')[0][0:-obs]

            newPath = ultrasoundTopLevelPath + '/%s/%s/Traced Contour/%s/%s%s' % \
                      (subjectID, prepost, phase, filename, AI_Name) + '.nrrd'

            nrrd.write(newPath, dataToBeSaved, header=header, custom_field_map=customFieldMap)

            print('Prediction done with %s and AI tracings saved' % filename)

    # Compute average DSC among all test cases
    model_gt_average = np.average(model_vs_ground_truth_dsc)
    average_background = np.average(background_dsc_list)
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
    average_background_csa = np.average(bg_csa_list)
    average_lm_csa = np.average(lm_csa_list)
    average_lv_csa = np.average(lv_csa_list)
    average_right_csa = np.average(right_csa_list)

    temp = {}
    extra_list = [0] * len(background_dsc_list)
    caseDescription = getCaseDescription()
    temp['Case'] = [caseDescription] + testing_list
    temp['BG_DSC'] = [average_background] + background_dsc_list
    temp['LM_DSC'] = [average_lm] + lm_dsc_list
    temp['LV_DSC'] = [average_lv] + lv_dsc_list
    temp['RV_DSC'] = [average_right] + right_dsc_list
    temp['DSC_AVG'] = [model_gt_average] + extra_list

    temp['LM_HD'] = [average_lm_hd] + lm_hd_list
    temp['LV_HD'] = [average_lv_hd] + lv_hd_list
    temp['RV_HD'] = [average_right_hd] + right_hd_list
    temp['DSC_HD'] = [model_gt_average_hd] + extra_list

    temp['BG_CSA'] = [average_background_csa] + bg_csa_list
    temp['LM_CSA'] = [average_lm_csa] + lm_csa_list
    temp['LV_CSA'] = [average_lv_csa] + lv_csa_list
    temp['RV_CSA'] = [average_right_csa] + right_csa_list
    temp['DSC_CSA'] = [model_gt_average_csa] + extra_list

    df = pd.DataFrame(temp)

    df.to_excel(savingPath + '/' + AI_Name + ' Metrics.xlsx', engine='xlsxwriter')


if __name__ == '__main__':
    # check if GPU setup is complete
    # Running option: 1 - Training, 2 - Print out prediction results
    runningOption = 2
    saveTra = 1         # to save plots and tracings after predictions of test set
    # 1 = Only plot, # 2 = plot and Tracing, # 3 = only tracing
    plotMdl = 0         # plot model graph
    checkGpuAvailibility()
    train_data_path = '../../data/Experiment/Training'
    validation_data_path = '../../data/Experiment/Validation'
    test_data_path = '../../data/Experiment/Testing'

    z_score = config['SEGMENTATION_TRAINING_SETTINGS']['Z_SCORE']

    batch_size = config['SEGMENTATION_TRAINING_SETTINGS']['BATCH_SIZE']
    num_filters = config['SEGMENTATION_TRAINING_SETTINGS']['NUM_FILTERS']
    dilation_rate = config['SEGMENTATION_TRAINING_SETTINGS']['DILATION_RATE']
    dropout_rate = config['SEGMENTATION_TRAINING_SETTINGS']['DROPOUT_RATE']
    num_epochs = config['SEGMENTATION_TRAINING_SETTINGS']['NUM_EPOCHS']

    model_options = config['SEGMENTATION_TRAINING_SETTINGS']['MODEL_OPTIONS']
    selected_model_index = config['SEGMENTATION_TRAINING_SETTINGS']['SELECTED_MODEL']
    model_type = model_options[selected_model_index]
    AI_name = config['SEGMENTATION_TRAINING_SETTINGS']['AI_NAME']

    log_file_path = '../Data/SegmentOutput/%s/%s_training.log' % (AI_name, model_type.lower())

    model_output_path = '../Data/SegmentOutput/%s' % (AI_name)

    initial_learning_rate = config['SEGMENTATION_TRAINING_SETTINGS']['INITIAL_LEARNING_RATE']
    learning_rate_patience = config['SEGMENTATION_TRAINING_SETTINGS']['PATIENCE']
    early_stopping_patience = config['SEGMENTATION_TRAINING_SETTINGS']['EARLY_STOP']
    learning_rate_drop = config['SEGMENTATION_TRAINING_SETTINGS']['LEARNING_RATE_DROP']
    validate_augment = config['SEGMENTATION_TRAINING_SETTINGS']['VALIDATION_AUGMENT']
    transpose = config['SEGMENTATION_TRAINING_SETTINGS']['TRANSPOSE']
    batch_normalized = config['SEGMENTATION_TRAINING_SETTINGS']['BATCH_NORMALIZATION']
    buffer_size = config['SEGMENTATION_TRAINING_SETTINGS']['BUFFER_SIZE']
    selected_band = config['SEGMENTATION_TRAINING_SETTINGS']['SELECTED_BAND']
    kernel_size = config['SEGMENTATION_TRAINING_SETTINGS']['KERNEL_SIZE']

    selectedBands = config['SEGMENTATION_TRAINING_SETTINGS']['SELECTED_BAND']
    remap = config['SEGMENTATION_TRAINING_SETTINGS']['REMAP']
    noiseReduction = config['SEGMENTATION_TRAINING_SETTINGS']['NOISE_REDUCTION']
    enhancement = config['SEGMENTATION_TRAINING_SETTINGS']['ENHANCEMENT']

    if not os.path.exists(model_output_path):
        os.makedirs(model_output_path)

    modelName = '%s.hdf5' % model_type

    input_shape = (IMAGE_HEIGHT, IMAGE_WIDTH, len(selectedBands))
    if selected_model_index == 0:
        """" MFP - Unet"""
        model = MFP_Unet(input_size=input_shape,
                         n_filters=num_filters,
                         dilation_rate=dilation_rate,
                         batch_normalized=batch_normalized,
                         dropout_rate=dropout_rate,
                         transpose=transpose,
                         kernel_size=kernel_size)

    elif selected_model_index == 1:
        model = unet(num_filters=32, input_size=input_shape,
                     dropout_rate=dropout_rate, dilation_rate=dilation_rate)

    elif selected_model_index == 2:
        model = ResnetBuilder.build_resnet_50(input_shape=input_shape, num_outputs=NUM_CLASSES) #34 50 101 152

    elif selected_model_index == 3:
        model = ResnetBuilder.build_resnet_101(input_shape=input_shape, num_outputs=NUM_CLASSES)

    elif selected_model_index == 4:
        model = ResnetBuilder.build_resnet_152(input_shape=input_shape, num_outputs=NUM_CLASSES)

    if runningOption == 1:
        if plotMdl:
            from tensorflow.keras.utils import plot_model
            plot_model(model, to_file=model_output_path + '/' + 'model.png', show_shapes=True, show_layer_names=True)

        num_train_file = len(os.listdir(train_data_path + '/' + 'masks'))
        num_val_file = len(os.listdir(validation_data_path + '/' + 'masks'))

        steps_per_epoch = num_train_file // batch_size * 4
        validation_steps = num_val_file // batch_size

        callbacks = get_callbacks(model_file=model_output_path + '/' + modelName,
                                  initial_learning_rate=initial_learning_rate,
                                  learning_rate_patience=learning_rate_patience,
                                  early_stopping_patience=early_stopping_patience,
                                  logging_file=log_file_path)

        print('Generator for training is running')
        timeStart = datetime.now()
        trainGenerator = data_generator(top_level_frames_path=train_data_path + '/' + 'frames',
                                        top_level_masks_path=train_data_path + '/' + 'masks',
                                        z_score=z_score,
                                        selected_band=selectedBands, )

        trainGenerator = trainGenerator.shuffle(buffer_size).repeat().batch(batch_size)
        timeEnd = datetime.now()
        print('Train generator time: ', timeEnd - timeStart)

        print('Generator for validation is running')
        timeStart = datetime.now()
        validationGenerator = data_generator(top_level_frames_path=validation_data_path + '/' + 'frames',
                                             top_level_masks_path=validation_data_path + '/' + 'masks',
                                             augment=validate_augment,
                                             selected_band=selectedBands,
                                             z_score=z_score, )
        # noiseReduction=noiseReduction)    # 230525 Property is not defined in function declaration

        validationGenerator = validationGenerator.batch(batch_size)
        timeEnd = datetime.now()
        print('Validation generator time: ', timeEnd - timeStart)

        trainModel(model=model,
                   trainGenerator=trainGenerator,
                   validationGenerator=validationGenerator,
                   validation_steps=validation_steps,
                   num_epochs=num_epochs,
                   steps_per_epoch=steps_per_epoch,
                   modelName=model_output_path + '/' + modelName,
                   callbacks=callbacks,
                   initial_learning_rate=initial_learning_rate)

        # save info of training settings
        training_detail = config['SEGMENTATION_TRAINING_SETTINGS']
        with open(model_output_path + '/' + "training_info.txt", 'w') as f:
            for key, value in training_detail.items():
                f.write('%s:%s\n' % (key, value))

    elif runningOption == 2:
        topLevelDir = None
        frameDataPath = test_data_path + '/' + 'frames'
        maskDataPath = test_data_path + '/' + 'masks'
        printOutResults(model=model,
                        modelPath=model_output_path + '/' + modelName,
                        frameDataPath=frameDataPath,
                        maskDataPath=maskDataPath,
                        loss_log=log_file_path,
                        savingPath=model_output_path,
                        training_detail=config['SEGMENTATION_TRAINING_SETTINGS'],
                        selected_band=selectedBands, saveTracings=saveTra)
