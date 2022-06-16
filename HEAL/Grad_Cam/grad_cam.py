import glob
import os
import torch
from PIL import Image
from torchvision import models, transforms
from torchvision.utils import save_image
from HEAL.Grad_Cam.cam import SmoothGradCAMpp
from HEAL.Grad_Cam.visualize import visualize, reverse_normalize
from HEAL.Independent_test.independent_test import load_variable
from HEAL.Training.train import get_model

def get_target_layer(_model_name, model):
    if _model_name == "ResNet50":
        target_layer = model.module.layer4[2].conv3
        return target_layer
    elif _model_name == "ResNet18":
        target_layer = model.module.layer4[1].conv2
        return target_layer
    elif _model_name == "ResNet34":
        target_layer = model.module.layer4[2].conv2
        return target_layer
    elif _model_name == "WideResNet50":
        target_layer = model.module.layer4[2].conv3
        return target_layer
    elif _model_name == "ResNet101":
        target_layer = model.module.layer4[2].conv3
        return target_layer
    elif _model_name == "WideResNet101":
        target_layer = model.module.layer4[2].conv3
        return target_layer
    elif _model_name == "ResNet152":
        target_layer = model.module.layer4[2].conv3
        return target_layer
    ###vgg family###
    elif _model_name == "Vgg16":
        target_layer = model.module.features[28]
        return target_layer
    elif _model_name == "Vgg16_BN":
        target_layer = model.module.features[40]
        return target_layer
    elif _model_name == "Vgg19":
        target_layer = model.module.features[34]
        return target_layer
    elif _model_name == "Vgg19_BN":
        target_layer = model.module.features[34]
        return target_layer
    ###alexnet###
    elif _model_name == "AlexNet":
        target_layer = model.module.features[10]
        return target_layer
    ###densenet161###
    elif _model_name == "DenseNet161":
        target_layer = model.module.features.denseblock4.denselayer24.conv2
        return target_layer
    ###InceptionV3###
    elif _model_name == "InceptionV3":
        target_layer = model.module.Mixed_7c.branch_pool.conv
        return target_layer
    ###ShuffleNetV2###
    elif _model_name == "ShuffleNetV2":
        target_layer = model.module.conv5[0]
        return target_layer
    ###MobileNetV2###
    elif _model_name == "MobileNetV2":
        target_layer = model.module.features[18][0]
        return target_layer
    ###GoogleNet###
    elif _model_name == "GoogleNet":
        target_layer = model.module.inception5b.branch4[1].conv
        return target_layer
    ###MNASNet###
    elif _model_name == "MNASNET":
        target_layer = model.module.layers[14]
        return target_layer


def grad_cam():
    if not os.path.exists("HEAL_Workspace/figures/grad-cam"):
        os.mkdir("HEAL_Workspace/figures/grad-cam")
    conf_dict = load_variable("HEAL_Workspace/outputs/parameter.conf")
    _work_mode = conf_dict["Mode"]
    _class_cate = conf_dict["Classes"]
    _class_number = conf_dict["Class_number"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Get the latest model
    list_of_models = glob.glob('HEAL_Workspace/models/*.pt')  # * means all if need specific format then *.csv
    latest_model = max(list_of_models, key=os.path.getctime)
    print(latest_model)
    model_name = str(latest_model.split("/")[-1]).split("_")[-2]
    model, _ = get_model(model_name, _class_number, _work_mode)
    #print(model)

    # Get the latest output file
    list_of_outs = glob.glob('HEAL_Workspace/outputs/Results*.out')  # * means all if need specific format then *.csv
    latest_out = max(list_of_outs, key=os.path.getctime)
    latest_results = load_variable(latest_out)
    img_list = latest_results["sample_path"]

    for img in img_list:
        img_base = str(img.split("/")[-1]).split(".")[0]
        patient_id = str(img.split("/")[-3])
        image = Image.open(img)
        data_transforms = transforms.Compose([transforms.ToTensor()])
        image_t = data_transforms(image)
        image_t = image_t.unsqueeze(0).to(device)
        model.load_state_dict(torch.load(latest_model, map_location=device))
        model.to(device)
        model.eval()

        target_layer = get_target_layer(model_name, model)

        wrapped_model = SmoothGradCAMpp(model, target_layer, n_samples=25, stdev_spread=0.15)

        cam, idx = wrapped_model(image_t)
        imgtt = reverse_normalize(image_t)

        heatmap = visualize(imgtt.cpu().detach(), cam.cpu().detach())

        save_image(heatmap, "HEAL_Workspace/figures/grad-cam/{}_{}_{}_{}.png".format(model_name, img_base, patient_id, _class_cate[idx]))

    return
