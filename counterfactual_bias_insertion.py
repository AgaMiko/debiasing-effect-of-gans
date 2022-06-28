import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from efficientnet_pytorch import EfficientNet
from torch_transforms import get_transforms
from tqdm import tqdm
import os
import numpy as np
import statistics
import csv
from sklearn.metrics import f1_score, recall_score, precision_score
from model import Net
import glob
import argparse


def get_args_parser():
    parser = argparse.ArgumentParser()
    # paths and dirs
    parser.add_argument(
        "--test_data_dir",
        type=str,
        required=True,
        help="path to directory with test images"
    )
    parser.add_argument(
        "--models_dir",
        type=str,
        required=True,
        help="path to directory with models to test"
    )
    parser.add_argument(
        "--cbi_masks_dir",
        type=str,
        required=True,
        help="path to directory with augmentation masks for CBI experiments"
    )
    parser.add_argument(
        "--augmentation_names",
        type=str,
        default=["frame", "hair-short", "hair-medium", "hair-dense", "ruler"],
        nargs='+',
        help="path to directory with augmentation masks for CBI experiments"
    )
    parser.add_argument(
        "--save_results_path",
        type=str,
        default='statistics.csv',
        help="path to .csv statistics"
    )
    # general settings
    parser.add_argument('-b', '--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for cbi testing (default: 16)')
    parser.add_argument(
        '--gpu', type=int, default=0, metavar='GPU',
        help='GPU number to use (default: 0)')
    return parser


def make_prediction(batch, model):
    """
    Calculates predictions for given batch of in puts.
    """
    images, labels = batch
    outputs = model(images)
    prob = torch.sigmoid(outputs).item()
    predicted = 1 if prob > 0.5 else 0
    return prob, predicted, labels.item()


def switched_score(pred_labels, aug_labels):
    """
    Calculates number of switched predictions.

    This function gets predicted labels for image without any inserted bias,
    and compares it with prediction of the same input but with inserted bias.
    Number of predictions that changed (switched) class is counted.
    """
    switched = 0
    ben_to_mal = 0
    mal_to_ben = 0
    for pred, aug in zip(pred_labels, aug_labels):
        if pred != aug:
            switched += 1
            if pred == "benign":
                ben_to_mal += 1
            elif pred == "malignant":
                mal_to_ben += 1
    print("Switched classes", switched, "out of", len(
        pred_labels), "--", switched/len(pred_labels)*100, "%")
    print("Switched benign to malignant", ben_to_mal)
    print("Switched malignant to benign", mal_to_ben)
    return switched, ben_to_mal, mal_to_ben


def main(args):
    header = ['model', 'aug_type', 'aug_number',
              'mean', 'median', 'max', 'min', 'switched', 'mal_to_ben', 'ben_to_mal',
              'f1', 'f1_aug', 'recall', 'recall_aug',
              'precision', 'precision_aug']
    models_to_explain = glob.glob(os.path.join(args.models_dir, "*.pth"))
    with open(args.save_results_path, 'w', encoding='UTF-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

    # for each model from models_to_explain run Counterfactual Experiments
    # run aug_list for each model
    # each aug will be run with different settings
    for model_name in models_to_explain:
        img_size = 256
        _, test_transform = get_transforms(img_size)
        test_set = datasets.ImageFolder(root=args.test_data_dir,
                                        transform=test_transform)
        test_loader = DataLoader(
            test_set, batch_size=args.batch_size, shuffle=False, num_workers=4)
        test_dict = test_set.class_to_idx

        # load trained classification model
        arch = EfficientNet.from_pretrained('efficientnet-b2')
        model = Net(arch=arch)
        model.load_state_dict(torch.load(args.models_dir + model_name,
                                         map_location=torch.device("cuda"))
                              )
        model.eval()
        model = model.to("cuda")
        # counterfactual experiments start
        for aug_nr, aug_type in enumerate(args.augmentation_names):
            mask_dir = os.path.join(args.cbi_masks_dir, aug_type)

            mask_list = glob.glob(mask_dir + "*")
            for aug_name in mask_list:
                aug_transform, _ = get_transforms(
                    img_size, type_aug=args.augmentation_names[aug_nr], aug_p=1.0, mask_list=[aug_name])
                aug_set = datasets.ImageFolder(root=args.test_data_dir,
                                               transform=aug_transform,
                                               )
                aug_loader = DataLoader(
                    aug_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
                classes = list(aug_set.class_to_idx.keys())

                prob_diff = list()
                org_labels = list()
                pred_labels = list()
                aug_labels = list()
                preds = list()
                preds_aug = list()
                ind = 0

                for i, (batch, batch_aug) in tqdm(enumerate(zip(test_loader, aug_loader))):
                    probability, predicted_class, labels = make_prediction(
                        batch, model)
                    probability_aug, predicted_class_aug, labels = make_prediction(
                        batch_aug, model)

                    preds.append(predicted_class)
                    preds_aug.append(predicted_class_aug)

                    org_labels.append(labels)
                    prob_diff.append(np.abs(probability-probability_aug))

                    pred_labels.append(classes[predicted_class])
                    aug_labels.append(classes[predicted_class_aug])

                print("===============SUMMARY===============")
                print(model_name)
                print("Mean difference", statistics.mean(prob_diff))
                print("Median difference", statistics.median(prob_diff))
                print("Max difference", np.max(prob_diff))
                print("Min difference", np.min(prob_diff))

                f1 = f1_score(org_labels, preds, zero_division=1)
                f1_aug = f1_score(org_labels, preds_aug, zero_division=1)
                recall = recall_score(org_labels, preds, zero_division=1)
                recall_aug = recall_score(
                    org_labels, preds_aug, zero_division=1)
                precision = precision_score(org_labels, preds, zero_division=1)
                precision_aug = precision_score(
                    org_labels, preds_aug, zero_division=1)
                switched, ben_to_mal, mal_to_ben = switched_score(
                    pred_labels, aug_labels)

                # counterfactual experiments end
                print("Saving results to file...")
                data = [model_name, args.augmentation_names[aug_nr], aug_name.split("/")[-1], statistics.mean(prob_diff),
                        statistics.median(prob_diff), np.max(prob_diff),
                        np.min(prob_diff), switched, mal_to_ben, ben_to_mal,
                        f1, f1_aug, recall, recall_aug, precision, precision_aug]

                with open(args.save_results_path, 'a', encoding='UTF-8', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(data)


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
