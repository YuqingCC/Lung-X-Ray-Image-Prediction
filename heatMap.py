
import numpy as np
import cv2
import torch
import model_construction

from matplotlib import pyplot as plt
from torchvision import models, transforms
from torch.nn import functional as F
from torch import topk
'''
This file get the class activation map for the input image
The program need the full path of the image and saved trained model to run
'''


def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam


def show_cam(CAMs, width, height, orig_image, class_idx, save_name):
    for i, cam in enumerate(CAMs):
        heatmap = cv2.applyColorMap(cv2.resize(cam, (width, height)), cv2.COLORMAP_JET)
        result = heatmap * 0.5 + orig_image * 0.5
        # put class label text on the result
        cv2.putText(result, str(int(class_idx[i])), (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('CAM', result/255.)
        cv2.waitKey(0)
        cv2.imwrite(f"output/CAM_{save_name}.jpg", result)


def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())


if __name__ == "__main__":
    # read single image
    image_path = 'Covid19-dataset/test/Viral Pneumonia/0120.jpeg'
    image = cv2.imread(image_path)
    orig_image = image.copy()
    # single_image = cv2.resize(single_image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, _ = image.shape

    transforms = transforms.Compose(
        [transforms.ToPILImage(),
         transforms.Resize((224, 224)),
         transforms.ToTensor(),
         ])
    a = transforms(img=image)
    print(a.shape)
    # add batch dimension
    image_tensor = a.unsqueeze(0)

    test_model = model_construction.model_ResNN(n_class=3, layer_list=[3, 4, 6, 3])
    model_state = torch.load(f'./output/model/model_res{10}.pth')
    test_model.load_state_dict(model_state['model_state_dict'], strict=False)

    features_blobs = []

    test_model._modules.get('layer3').register_forward_hook(hook_feature)
    # get the softmax weight
    params = list(test_model.parameters())
    weight_softmax = np.squeeze(params[-2].data.numpy())
    print(weight_softmax)

    # get the predictions
    out = test_model(image_tensor)
    probs = F.softmax(out, 1).data.squeeze()
    class_idx = topk(probs, 1)[1].int()
    print(class_idx)

    # generate class activation mapping for the top1 prediction
    CAMs = returnCAM(features_blobs[0], weight_softmax, class_idx)
    save_name = "Pneumonia_cam"
    # show and save the results
    show_cam(CAMs, width, height, orig_image, class_idx, save_name)






