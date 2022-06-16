import os
import numpy as np
import pandas as pd
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import f1_score, accuracy_score, auc, roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix
from scipy import interp
import cv2
from collections import Counter
from PIL import Image

def save_variable(var, filename):
    pickle_f = open(filename, 'wb')
    pickle.dump(var, pickle_f)
    pickle_f.close()
    return filename


def load_variable(filename):
    pickle_f = open(filename, 'rb')
    var = pickle.load(pickle_f)
    pickle_f.close()
    return var


def plot_confusion_matrix(y_true, y_pred, classes, normalize=True, title=None, cmap=plt.cm.Blues, fig_name="Confusion_matrix.png"):
    plt.close('all')
    plt.figure(figsize=(8, 8), dpi=400)
    if not title:
        if normalize:
            title = 'Normalized confusion matrix of independent test results'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true.argmax(axis=1), y_pred.argmax(axis=1))

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.close('all')
    plt.style.use("ggplot")
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig('HEAL_Workspace/figures/{}'.format(fig_name), dpi=400)
    return ax


def plot_roc_curve(pred_y, test_y, class_label, n_classes, fig_name="roc_auc.png", title="ROC curve of HEAL"):
    colors = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000", "#66CC99", "#999999"]
    plt.close('all')
    plt.style.use("ggplot")
    plt.figure(figsize=(8, 8), dpi=400)
    for i in range(n_classes):
        _tmp_pred = pred_y
        _tmp_label = test_y
        _fpr, _tpr, _ = roc_curve(_tmp_label[:, i], _tmp_pred[:, i])
        _auc = auc(_fpr, _tpr)
        plt.plot(_fpr, _tpr, color=colors[i],
                 label=r'%s ROC (AUC = %0.3f)' % (class_label[i], _auc), lw=2, alpha=.9)
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.savefig(fig_name, dpi=400)
    plt.close('all')


def cv_result_detect(result_list):
    jobids = []
    cv_ids = []
    single_ids = []

    for tmp in result_list:
        _jobid = str(tmp.split("/")[-1])[0:19]
        #print("Prefix {}".format(_jobid))
        jobids.append(_jobid)
    unique_jobids = list(set(jobids))
    #print(unique_jobids)
    test_res = None
    for _id in unique_jobids:
        for _res in result_list:
            if _id in _res:
                test_res = _res
                break

        fold_num = 0
        for i in range(10):
            tmp_fold = "fold{}".format(i)
            tmp_res = re.sub('fold[0-9]', tmp_fold, test_res)
            if tmp_res in result_list:
                fold_num += 1

        if fold_num > 2:
            cv_ids.append(_id)
        else:
            single_ids.append(_id)

    return cv_ids, single_ids


def patient_level_results(tile_result_dict):
    preds = tile_result_dict["preds"]
    labels = tile_result_dict["labels"]
    img_paths = tile_result_dict["sample_path"]
    patient_list = []
    for img in img_paths:
        print(img)
#         patient_id = img.split("/")[3]
        patient_id = img.split("/")[3].split("_")[0]
        print(patient_id)
        patient_list.append(patient_id)
    patient_list = list(set(patient_list))

    new_patient_list = []
    new_patient_labels = np.array([])
    new_patient_preds = np.array([])

    for _p in patient_list:
        _p_idx = [index for index, value in enumerate(img_paths) if _p in value]
        _p_preds = [preds[index] for index in _p_idx]
        _p_labels = [labels[index] for index in _p_idx]
        _p_preds = np.array(_p_preds)

        new_patient_list.append(_p)
        #new_patient_labels.append(list(_p_labels[0]))
        new_patient_labels = np.vstack([new_patient_labels, _p_labels[0]]) if new_patient_labels.size else _p_labels[0]
        new_patient_preds = np.vstack([new_patient_preds, _p_preds.mean(axis=0)]) if new_patient_preds.size else _p_preds.mean(axis=0)
        #new_patient_preds.append(list(_p_preds.mean(axis=0)))

    patient_results = {"preds": new_patient_preds, "labels": new_patient_labels, "sample_path": new_patient_list}
    return patient_results, new_patient_preds, new_patient_labels


def plot_single_roc_confusion_matrix(results, class_cate, n_class, _work_mode):
    for res in results:
        base_name = str(res.split("/")[-1]).split(".")[0]
        fig_name = "HEAL_Workspace/figures/{}_tile_level_ROC.png".format(base_name)
        p_fig_name = "HEAL_Workspace/figures/{}_patient_level_ROC.png".format(base_name)

        model_base_name = base_name.split("_")[2]
        title = "ROC curve of {}".format(model_base_name)
        p_title = "ROC curve of {} (Patient level)".format(model_base_name)

        result_dict = load_variable(res)
        preds = result_dict["preds"]
        labels = result_dict["labels"]
        patient_res, p_preds, p_labels = patient_level_results(result_dict)

        plot_roc_curve(preds, labels, class_cate, n_class, fig_name=fig_name, title=title)
        plot_roc_curve(p_preds, p_labels, class_cate, n_class, fig_name=p_fig_name, title=p_title)
        patient_res_path = "HEAL_Workspace/logs/{}_patient_level.out".format(base_name)
        save_variable(patient_res, patient_res_path)

        if not _work_mode:
            plot_confusion_matrix(labels, preds, class_cate, fig_name="{}_tile_level_confusion_matrix.png".format(base_name))
            plot_confusion_matrix(p_labels, p_preds, class_cate, fig_name="{}_patient_level_confusion_matrix.png".format(base_name))
    return


def plot_roc_ms(cv_result_list, class_cate, n_classes, fig_name):
    colors = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000", "#66CC99", "#999999"]
    class_label = class_cate
    base_fpr = np.linspace(0, 1, 1001)
    plt.figure(figsize=(8, 8), dpi=400)
    for i in range(n_classes):
        fprs = []
        tprs = []
        aucs = []
        for res in cv_result_list:
            _tmp_res = load_variable(res)
            _tmp_pred = _tmp_res["preds"]
            _tmp_label = _tmp_res["labels"]
            _fpr, _tpr, _ = roc_curve(_tmp_label[:, i], _tmp_pred[:, i])
            _auc = auc(_fpr, _tpr)
            _tpr = interp(base_fpr, _fpr, _tpr, period=1000)
            _tpr[0] = 0.0
            fprs.append(_fpr)
            tprs.append(_tpr)
            aucs.append(_auc)
        _mean_tpr = np.mean(tprs, axis=0)
        #_mean_fpr = np.mean(fprs, axis=0)
        _mean_auc = auc(base_fpr, _mean_tpr)
        _std_auc = np.std(aucs)
        plt.plot(base_fpr, _mean_tpr, color=colors[i],
                 label=r'%s Mean ROC (AUC = %0.3f $\pm$ %0.3f)' % (class_label[i], _mean_auc, _std_auc),
                 lw=2, alpha=.9)

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(_mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(_mean_tpr - std_tpr, 0)
        plt.fill_between(base_fpr, tprs_lower, tprs_upper, color=colors[i], alpha=.2)
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Mean ROC curves on 10-fold cross-validation test')
    plt.legend(loc="lower right")
    plt.savefig(fig_name, dpi=400)
    plt.close('all')


def plot_cv_roc(cv_results, result_list, class_cate, n_classes):
    for cv_result in cv_results:
        # tile level
        _idx = [index for index, value in enumerate(result_list) if cv_result in value]
        _fold_path = [result_list[index] for index in _idx]
        base_name = (str(_fold_path[0].split("/")[-1]).split(".")[0]).split("_")[0:3]
        sep = "-"
        base_name = sep.join(base_name)
        fig_name = "HEAL_Workspace/figures/{}_tile_level_CVROC.png".format(base_name)
        plot_roc_ms(_fold_path, class_cate, n_classes, fig_name)

        # patient level
        patient_paths = []
        for _root, _dirs, _files in os.walk("HEAL_Workspace/logs"):
            for _file in _files:
                if str(_file).startswith("Results") and str(_file).endswith("out"):
                    result_path = os.path.join(_root, _file)
                    patient_paths.append(result_path)

        _p_fold_path = [patient_paths[index] for index in _idx]
        fig_name = "HEAL_Workspace/figures/{}_patient_level_CVROC.png".format(base_name)
        plot_roc_ms(_p_fold_path, class_cate, n_classes, fig_name)
    return


def get_row_col(_img_path):
    _col_list = []
    _row_list = []
    # print(_img_path_df.iloc[:, 0])
    for _img_p in _img_path:
        _info = _img_p.split('/')[-1]
        _info = _info.split('.')[0]
        # print(_img_p)
        _col, _row = _info.split('_')
        _col_list.append(int(_col))
        _row_list.append(int(_row))
    return _col_list, _row_list


def concat_img(patient_id, _class_cate, _class_dict, UNIT_SIZE, col_list, row_list, y_pred, label, img_dirs):
    try:
        file_name = img_dirs[0].split('/')[-3]
    except Exception as e:
        print(img_dirs)
        print(e)
    _pred_class_index = y_pred.argmax(axis=1)
    print("Plotting heatmap of %s" % patient_id)
    print("Total number of tiles is %d" % len(_pred_class_index))

    voting = Counter(_pred_class_index)

    true_label = []
    
    
    for i in range(len(label)):
        if int(label[i]) == 1:
            true_label.append(_class_cate[i])
            
    print("True label is %s" % true_label)

    for ele in true_label:
        _con_img = Image.new('RGBA', (UNIT_SIZE * max(col_list), UNIT_SIZE * max(row_list)), color=(255, 255, 255))
        for _i, _idx in enumerate(_pred_class_index):
            # print(_i,_idx,_pred_class_index)
            # print(img_dirs[_i])
            img = cv2.imread(img_dirs[_i])
            img_r = img[:, :, 0]
            img_g = img[:, :, 1]
            img_b = img[:, :, 2]
            red = np.full((UNIT_SIZE, UNIT_SIZE), 255)
            green = np.full((UNIT_SIZE, UNIT_SIZE), 255)
            blue = np.full((UNIT_SIZE, UNIT_SIZE), 255)
                #print(ele)
            if _idx == _class_dict[ele]:
            #if _idx == _class_dict[true_label]:
                tmp_r = red
                tmp_g = img_g
                tmp_b = img_b
                tmp_a = np.full((UNIT_SIZE, UNIT_SIZE), 255 * y_pred[_i, _idx])
            else:
                tmp_r = img_r
                tmp_g = img_g
                tmp_b = blue
                tmp_a = np.full((UNIT_SIZE, UNIT_SIZE), 255 * y_pred[_i, _idx])

            tmp_r = Image.fromarray(np.uint8(tmp_r), mode="L")
            tmp_g = Image.fromarray(np.uint8(tmp_g), mode="L")
            tmp_b = Image.fromarray(np.uint8(tmp_b), mode="L")
            tmp_a = Image.fromarray(np.uint8(tmp_a), mode="L")
            tmp_img = Image.merge("RGBA", (tmp_r, tmp_g, tmp_b, tmp_a))
            _con_img.paste(tmp_img, (col_list[_i] * UNIT_SIZE, row_list[_i] * UNIT_SIZE, (col_list[_i] + 1) * UNIT_SIZE, (row_list[_i] + 1) * UNIT_SIZE))
        plt.close("all")
        plt.style.use("ggplot")
        #matplotlib.rcParams['font.family'] = "Arial"
        plt.rcParams["axes.grid"] = False
        plt.title("Predicted as %s" % (true_label), fontsize=12)
        plt.axis('off')
        #plt.title(ele, )
        sc = plt.imshow(_con_img, vmin=0, vmax=1, cmap="bwr")
        cbar = plt.colorbar(sc, ticks=[1, 0])
        cbar.ax.set_yticklabels(["%s"%ele, "Not-%s"%ele])
        #plt.show()
        plt.savefig('HEAL_Workspace/figures/%s_%s_concat_%s_%f.png' %
                    (patient_id, ele, _class_cate[voting.most_common(1)[0][0]], voting.most_common(1)[0][1] / len(_pred_class_index)), dpi=300)

    return


def plot_conc_heatmap(result_list, img_size, _class_number, _class_cate, _class_dict):
    for res in result_list:
        result_dict = load_variable(res)
        preds = result_dict["preds"]
        labels = result_dict["labels"]
        img_paths = result_dict["sample_path"]
        patient_list = []
        for img in img_paths:
            patient_id = img.split("/")[3]
            patient_list.append(patient_id)
        patient_list = list(set(patient_list))

        for p_res in patient_list:
            _p_idx = [index for index, value in enumerate(img_paths) if p_res in value]
            _p_preds = [preds[index] for index in _p_idx]
            _p_preds = np.array(_p_preds)
            _p_labels = [labels[index] for index in _p_idx]
            _p_paths = [img_paths[index] for index in _p_idx]

            col_list, row_list = get_row_col(_p_paths)
            concat_img(p_res, _class_cate, _class_dict, img_size, col_list, row_list, _p_preds, _p_labels[0], _p_paths)
    return


def data_visualisation(_tile_info):
    conf_dict = load_variable("HEAL_Workspace/outputs/parameter.conf")
    _work_mode = conf_dict["Mode"]
    _class_cate = list(conf_dict["Classes"])
    print("[INFO] Label information: {}".format(_class_cate))
    _class_number = conf_dict["Class_number"]
    img_size = _tile_info[0]
    _class_idx = []
    for i in range(_class_number):
        _class_idx.append(i)
    _class_dict = dict(zip(_class_cate, _class_idx))

    result_list = []
    for _root, _dirs, _files in os.walk("HEAL_Workspace/outputs"):
        for _file in _files:
            if str(_file).startswith("Results") and str(_file).endswith("out"):
                result_path = os.path.join(_root, _file)
                result_list.append(result_path)
                
    print("[INFO] Plotting tile- and patient-level ROC and Confusion matrix ...")
    plot_single_roc_confusion_matrix(result_list, _class_cate, _class_number, _work_mode)
    cv_res, _ = cv_result_detect(result_list)
    
    print("[INFO] Plotting ROC and Confusion matrix of K-fold cross-validation ...")
    plot_cv_roc(cv_res, result_list, _class_cate, _class_number)
    
    #print("[INFO] Plotting concatenated heatmap ...")
    #plot_conc_heatmap(result_list, img_size, _class_number, _class_cate, _class_dict)

