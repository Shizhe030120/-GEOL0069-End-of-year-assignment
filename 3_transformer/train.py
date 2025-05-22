import os
import math
import argparse
import matplotlib.pyplot as plt
from collections import defaultdict

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns  # for drawing confusion matrix

from my_dataset import MyDataSet
from vit_model import vit_base_patch16_224 as create_model
from utils import read_split_data, train_one_epoch, evaluate

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if not os.path.exists("./weights"):
        os.makedirs("./weights")

    tb_writer = SummaryWriter()

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}

    # Instantiate training dataset
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    # Instantiate validation dataset
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    model = create_model(num_classes=args.num_classes).to(device)

    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)
        # Delete unnecessary weights
        del_keys = ['head.weight', 'head.bias']
        for k in del_keys:
            del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))

    if args.freeze_layers:
        for name, para in model.named_parameters():
            # Freeze all weights except head and pre_logits
            if "head" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=5E-5)

    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # Used to save loss and accuracy for each epoch
    metrics = defaultdict(lambda: defaultdict(list))

    # Store all prediction values and true labels for validation set
    all_preds = []
    all_labels = []

    for epoch in range(args.epochs):
        # train
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)

        scheduler.step()

        # validate
        val_loss, val_acc, epoch_preds, epoch_labels = evaluate(
            model=model,
            data_loader=val_loader,
            device=device,
            epoch=epoch,
            return_preds=True  # Modify evaluate method to support returning prediction values and true labels
        )

        # Collect all predictions and true labels from validation set
        all_preds.extend(epoch_preds)
        all_labels.extend(epoch_labels)

        # Record loss and accuracy for each epoch
        metrics["train"]["loss"].append(train_loss)
        metrics["train"]["accuracy"].append(train_acc)
        metrics["val"]["loss"].append(val_loss)
        metrics["val"]["accuracy"].append(val_acc)

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)
        if epoch == 19:
            torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch))

    # Classification report
    print("\nClassification Report:")
    report = classification_report(epoch_labels, epoch_preds, target_names=[str(c) for c in val_dataset.classes], digits=2)
    print(report)

    # Draw classification report as table
    plt.figure(figsize=(10, 10))
    plt.text(0.01, 0.01, report, {'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.title("Classification Report")
    plt.savefig("classification_report.png")  # Save classification report image
    plt.show()

    # Confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(15, 15))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=val_dataset.classes, yticklabels=val_dataset.classes)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")  # Save confusion matrix image
    plt.show()

    # Draw loss and accuracy curves
    plt.figure(figsize=(12, 6))

    # Draw loss curve
    plt.subplot(1, 2, 1)
    plt.plot(range(1, args.epochs + 1), metrics["train"]["loss"], label="Train Loss")
    plt.plot(range(1, args.epochs + 1), metrics["val"]["loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss vs Epochs")
    plt.grid(True)
    plt.legend()

    # Draw accuracy curve
    plt.subplot(1, 2, 2)
    plt.plot(range(1, args.epochs + 1), metrics["train"]["accuracy"], label="Train Accuracy")
    plt.plot(range(1, args.epochs + 1), metrics["val"]["accuracy"], label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Epochs")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig("metrics_curve.png")
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.01)

    # Root directory of the dataset
    parser.add_argument('--data-path', type=str,
                        default="/root/autodl-tmp/1200120012001200/data_set/data")
    parser.add_argument('--model-name', default='', help='create model name')

    # Pre-trained weights path, set to empty string if you don't want to load
    parser.add_argument('--weights', type=str, default='',
                        help='initial weights path')
    # Whether to freeze weights
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)