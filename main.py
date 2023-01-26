from itertools import cycle

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import glob
from sklearn.preprocessing import label_binarize
import data_processing
from tqdm import tqdm
import model_construction
from torch.utils.data import DataLoader
import json
import ast
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix, auc, roc_curve


def model_data_saver(result):
    with open('./output/result_res.txt', 'w') as data:
        data.write(str(result))


def plot_result(history_dict, label=None, x_axis=None, y_axis=None, title=None, store=False, path=None):
    data_to_plot = history_dict[label]
    plt.figure(figsize=(8, 6))
    plt.plot(data_to_plot, '-o')
    if x_axis is not None:
        plt.xlabel(x_axis)
    if y_axis is not None:
        plt.ylabel(y_axis)
    if title is not None:
        plt.title(title)
    if store:
        plt.savefig(path)
    plt.show()


def plot_roc_curve(y_test, y_pred):
    n_classes = len(np.unique(y_test))
    y_test = label_binarize(y_test, classes=np.arange(n_classes))
    y_pred = label_binarize(y_pred, classes=np.arange(n_classes))

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure(figsize=(10, 5))
    lw = 2
    plt.plot(fpr["micro"], tpr["micro"],
             label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
             color="deeppink", linestyle=":", linewidth=4, )

    plt.plot(fpr["macro"], tpr["macro"],
             label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
             color="navy", linestyle=":", linewidth=4, )

    colors = cycle(["aqua", "darkorange", "darkgreen", "yellow", "blue"])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label="ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc[i]), )

    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) curve")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # load dataImage
    train_path = "Covid19-dataset/train"
    test_path = "Covid19-dataset/test"
    train_data_set = data_processing.load_data(train_path)
    print(train_data_set)
    print(train_data_set.class_to_idx)
    test_data_set = data_processing.load_data(test_path, test=True)
    print(test_data_set)
    print(test_data_set.class_to_idx)

    # Visualize images
    data_processing.visualize_image(train_data_set)

    # Create data loader
    batch_size = 40
    train_loader = DataLoader(train_data_set, batch_size=batch_size, shuffle=True)
    for x, y in train_loader:
        print(f'Image Shape :{x.shape}')
        print(f'label :{y.shape}')
        break
    test_loader = DataLoader(test_data_set, batch_size=batch_size, shuffle=True)
    for x, y in test_loader:
        print(f'Image Shape :{x.shape}')
        print(f'label :{y.shape}')
        break

    # parameter for ResNet model
    Res_34 = [3, 4, 6, 3]
    total_epoch = 10
    learning_rate = 0.0001

    # build and train the model
    model = model_construction.model_ResNN(n_class=3, layer_list=Res_34)
    res_result = model_construction.training(model, train_loader, learning_rate, total_epoch)
    # save training outcome
    model_data_saver(res_result)

    # plot training result
    with open('./output/result_res.txt', 'r') as content_file:
        res_result = content_file.read()
    res_result = ast.literal_eval(res_result)

    plot_result(res_result, label="Loss", x_axis="epoch", y_axis="loss", title="training loss vs. epoch",
                store=True, path="./output/resNet_tr_loss")
    plot_result(res_result, label="Accuracy", x_axis="epoch", y_axis="accuracy", title="accuracy vs. epoch",
                store=True, path="./output/resNet_tr_acc")

    '''
    Res Net with pre-train weights from ImageNet
    Attention: the model do not use in final evaluation, since somehow the accuracy for the pre-trained model does not have significant improvement
    '''
    # model = model_construction.resNet_pretrain()
    # res_result = model_construction.training(model, train_loader, learning_rate, total_epoch)

    # calculate testing accuracy
    test_result = dict.fromkeys(["Accuracy"])
    test_acc = []
    for i in range(total_epoch):
        # load the model
        model_to_evl = model_construction.model_ResNN(n_class=3, layer_list=Res_34)
        model_state = torch.load(f'./output/model/model_res{i+1}.pth')
        model_to_evl.load_state_dict(model_state['model_state_dict'], strict=False)
        model_to_evl.eval()
        if i == 9 or i == 0:
            print("Model's state_dict:")
            for param_tensor in model_to_evl.state_dict():
                print(param_tensor, "\t", model_to_evl.state_dict()[param_tensor].size())
        test_epoch = model_construction.testing(model=model_to_evl, data_loader=test_loader)
        test_acc.append(test_epoch)
        print(f'Epoch {i + 1}/{total_epoch} | Acc: {test_epoch:.4f}')
    test_result["Accuracy"] = test_acc
    plot_result(test_result, label="Accuracy", x_axis="epoch", y_axis="accuracy", title="accuracy vs. epoch",
                store=True, path="./output/resNet_test_acc")

    # Model evaluation
    print("======================== Model evaluation ========================")
    all_data_path = "Covid19-dataset/all"
    all_image = data_processing.load_data(all_data_path, test=True)
    print(all_image)
    print(all_image.class_to_idx)
    all_image_loader = DataLoader(all_image, batch_size=batch_size, shuffle=True)

    last_model = model_construction.model_ResNN(n_class=3, layer_list=Res_34)
    model_state = torch.load(f'./output/model/model_res{10}.pth')
    last_model.load_state_dict(model_state['model_state_dict'], strict=False)
    last_model.eval()
    y_pred, y_true, = [], []
    with torch.no_grad():
        for i, data in tqdm(enumerate(all_image_loader), total=len(all_image_loader)):
            image, label = data
            outputs = last_model(image)
            # get the prediction
            _, predictions = torch.max(outputs.data, 1)
            y_pred.append(predictions.numpy())
            y_true.append(label.numpy())
        y_pred = np.concatenate(y_pred)
        y_true = np.concatenate(y_true)
    print(y_pred.shape)
    print(y_true.shape)
    c = confusion_matrix(y_true, y_pred, labels=range(3))
    print(c)
    print(classification_report(y_true, y_pred, labels=range(3)))
    plot_roc_curve(y_true, y_pred)

























