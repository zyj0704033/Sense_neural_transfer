import argparse
import os
import sys
import time

import numpy as np
import torch
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

import utils
from transformer_net import TransformerNet
from vgg import Vgg16
from vgg import maskpooling


def check_paths(args):
    try:
        if not os.path.exists(args.save_model_dir):
            os.makedirs(args.save_model_dir)
        if args.checkpoint_model_dir is not None and not (os.path.exists(args.checkpoint_model_dir)):
            os.makedirs(args.checkpoint_model_dir)
    except OSError as e:
        print(e)
        sys.exit(1)


def train(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    transform = transforms.Compose([
	transforms.Resize(args.image_size),                                                                                   
	transforms.CenterCrop(args.image_size), 
	transforms.ToTensor(),
    transforms.Lambda(lambda x: x.mul(255))
    ])
    #train_dataset = datasets.ImageFolder(args.dataset, transform)
    color_sets = [[[0,255,0]],[[0,0,255]],[[255,0,0]],[[255,255,255]],[[0,255,255]]]
    train_dataset = utils.Mydataset(args.image_dir,args.mask_dir,color_sets,transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,num_workers=16,drop_last=True)

    transformer = TransformerNet()
    optimizer = Adam(transformer.parameters(), args.lr)
    mse_loss = torch.nn.MSELoss()

    vgg = Vgg16(requires_grad=False)
    style_transform = transforms.Compose([
	transforms.Resize(args.image_size),                                                                                   
	transforms.CenterCrop(args.image_size), 
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    style = utils.load_image(args.style_image, size=args.style_size)
    style = style_transform(style)
    style = style.repeat(args.batch_size, 1, 1, 1)
    style_mask = utils.load_image(args.style_mask, size=args.style_size)
    style_masks = []
    for color in color_sets:
        style_masks.append(utils.masktransform(style_mask,color))


    if torch.cuda.device_count()>1:
        transformer = torch.nn.DataParallel(transformer)
	#vgg = torch.nn.DataParallel(vgg)
    MaskPooling = maskpooling()

    if args.cuda:
        transformer.cuda()
        vgg.cuda()
        style = style.cuda()
        MaskPooling.cuda()

    style_v = Variable(style)
    style_v = utils.normalize_batch(style_v)
    features_style = vgg(style_v)

    #get the style mask features
    features_masks = [[] for _ in style_masks]
    #feature_masks = 6*[ 4*[1*1*w*h] ]
    smi = 0
    for sm in style_masks:
        sm_t = style_transform(sm)
        sm_t = sm_t.repeat(1,1,1,1)
        sm_v = Variable(sm_t)
        if args.cuda:
            sm_v = sm_v.cuda()
        features_masks[smi].append(MaskPooling(sm_v))
        smi += 1



    features_stylemulmasks = [[] for _ in features_style]
    for fsi in range(len(features_style)):
        for msi in range(len(features_masks)):
            b,ch,w,h = features_style.shape
            fm = features_masks[msi][fsi]
            fm = fm.repeat(b,ch,1,1)
            features_stylemulmasks[fsi].append(features_style*fm)
    #features_stylemulmasks 4*[ 6*[b*ch*w*h] ]
    gram_stylemulmask = [[utils.gram_matrix(y) for y in feature_smm] for feature_smm in features_stylemulmasks]
    #gram_stylemulmask 4*[ 6*[b*ch*ch] ]
    for e in range(args.epochs):
        transformer.train()
        agg_content_loss = 0.
        agg_style_loss = 0.
        count = 0
        for batch_id, image_and_mask in enumerate(train_loader):
            x = image_and_mask[0]  #image b*ch*w*h
            content_mask = image_and_mask[1] #list 6*[(b*1*w*h)]
            n_batch = len(x)
            count += n_batch
            optimizer.zero_grad()
            x = Variable(x)
            if args.cuda:
                x = x.cuda()

            y = transformer(x)

            y = utils.normalize_batch(y)
            x = utils.normalize_batch(x)

            features_y = vgg(y)
            features_x = vgg(x)

            content_loss = args.content_weight * mse_loss(features_y[1], features_x[1])

            style_loss = 0.

            #*****style loss*******
            content_mask_features = []
            for cm in content_mask:
                cm_v = Variable(cm)
                if args.cuda:
                    cm_v = cm_v.cuda()
                content_mask_features.append(MaskPooling(cm_v))
                #content_mask_features 6*[ 4[] ]
            ki = 0
            for ft_y, gm_s in zip(features_y, gram_stylemulmask):
                for mi in range(len(content_mask_features)):
                    b,ch,w,h = ft_y.shape
                    conmaski = content_mask_features[mi][ki].repeat(1,ch,1,1)
                    ft_y_m = ft_y * conmaski
                    gm_y = utils.gram_matrix(ft_y_m)
                    #style_loss += mse_loss(gm_y, gm_s[:n_batch, :, :])
                    style_loss += mse_loss(gm_y,gm_s[mi][:n_batch,:,:]) 
                ki += 1
            style_loss *= args.style_weight

            total_loss = content_loss + style_loss
            total_loss.backward()
            optimizer.step()

            agg_content_loss += content_loss.data[0]
            agg_style_loss += style_loss.data[0]

            if (batch_id + 1) % args.log_interval == 0:
                mesg = "{}\tEpoch {}:\t[{}/{}]\tcontent: {:.6f}\tstyle: {:.6f}\ttotal: {:.6f}".format(
                    time.ctime(), e + 1, count, len(train_dataset),
                                  agg_content_loss / (batch_id + 1),
                                  agg_style_loss / (batch_id + 1),
                                  (agg_content_loss + agg_style_loss) / (batch_id + 1)
                )
                print(mesg)

            if args.checkpoint_model_dir is not None and (batch_id + 1) % args.checkpoint_interval == 0:
                transformer.eval()
                if args.cuda:
                    transformer.cpu()
                ckpt_model_filename = "ckpt_epoch_" + str(e) + "_batch_id_" + str(batch_id + 1) + ".pth"
                ckpt_model_path = os.path.join(args.checkpoint_model_dir, ckpt_model_filename)
                torch.save(transformer.state_dict(), ckpt_model_path)
                if args.cuda:
                    transformer.cuda()
                transformer.train()

    # save model
    transformer.eval()
    if args.cuda:
        transformer.cpu()
    save_model_filename = "epoch_" + str(args.epochs) + "_" + str(time.ctime()).replace(' ', '_') + "_" + str(
        args.content_weight) + "_" + str(args.style_weight) + ".model"
    save_model_path = os.path.join(args.save_model_dir, save_model_filename)
    torch.save(transformer.state_dict(), save_model_path)

    print("\nDone, trained model saved at", save_model_path)


def stylize(args):
    content_image = utils.load_image(args.content_image, scale=args.content_scale)
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0)
    if args.cuda:
        content_image = content_image.cuda()
    content_image = Variable(content_image, volatile=True)

    style_model = TransformerNet()
    style_model.load_state_dict(torch.load(args.model))
    if args.cuda:
        style_model.cuda()
    output = style_model(content_image)
    if args.cuda:
        output = output.cpu()
    output_data = output.data[0]
    utils.save_image(args.output_image, output_data)


def main():
    main_arg_parser = argparse.ArgumentParser(description="parser for fast-neural-style")
    subparsers = main_arg_parser.add_subparsers(title="subcommands", dest="subcommand")

    train_arg_parser = subparsers.add_parser("train", help="parser for training arguments")
    train_arg_parser.add_argument("--epochs", type=int, default=2,
                                  help="number of training epochs, default is 2")
    train_arg_parser.add_argument("--batch-size", type=int, default=32,
                                  help="batch size for training, default is 32")
    # train_arg_parser.add_argument("--dataset", type=str, required=True,
    #                               help="path to training dataset, the path should point to a folder "
    #                                    "containing another folder with all the training images")
    train_arg_parser.add_argument("--image_dir", type=str, required=True)
    train_arg_parser.add_argument("--mask_dir", type=str, required=True)
    train_arg_parser.add_argument("--style-image", type=str, default="images/style-images/mosaic.jpg",
                                  help="path to style-image")
    train_arg_parser.add_argument("--style_mask", type=str, required=True)
    train_arg_parser.add_argument("--save-model-dir", type=str, required=True,
                                  help="path to folder where trained model will be saved.")
    train_arg_parser.add_argument("--checkpoint-model-dir", type=str, default=None,
                                  help="path to folder where checkpoints of trained models will be saved")
    train_arg_parser.add_argument("--image-size", type=int, default=256,
                                  help="size of training images, default is 256 X 256")
    train_arg_parser.add_argument("--style-size", type=int, default=None,
                                  help="size of style-image, default is the original size of style image")
    train_arg_parser.add_argument("--cuda", type=int, required=True,
                                  help="set it to 1 for running on GPU, 0 for CPU")
    train_arg_parser.add_argument("--seed", type=int, default=42,
                                  help="random seed for training")
    train_arg_parser.add_argument("--content-weight", type=float, default=1e5,
                                  help="weight for content-loss, default is 1e5")
    train_arg_parser.add_argument("--style-weight", type=float, default=1e10,
                                  help="weight for style-loss, default is 1e10")
    train_arg_parser.add_argument("--lr", type=float, default=1e-3,
                                  help="learning rate, default is 1e-3")
    train_arg_parser.add_argument("--log-interval", type=int, default=500,
                                  help="number of images after which the training loss is logged, default is 500")
    train_arg_parser.add_argument("--checkpoint-interval", type=int, default=2000,
                                  help="number of batches after which a checkpoint of the trained model will be created")

    eval_arg_parser = subparsers.add_parser("eval", help="parser for evaluation/stylizing arguments")
    eval_arg_parser.add_argument("--content-image", type=str, required=True,
                                 help="path to content image you want to stylize")
    eval_arg_parser.add_argument("--content-scale", type=float, default=None,
                                 help="factor for scaling down the content image")
    eval_arg_parser.add_argument("--output-image", type=str, required=True,
                                 help="path for saving the output image")
    eval_arg_parser.add_argument("--model", type=str, required=True,
                                 help="saved model to be used for stylizing the image")
    eval_arg_parser.add_argument("--cuda", type=int, required=True,
                                 help="set it to 1 for running on GPU, 0 for CPU")

    args = main_arg_parser.parse_args()

    if args.subcommand is None:
        print("ERROR: specify either train or eval")
        sys.exit(1)
    if args.cuda and not torch.cuda.is_available():
        print("ERROR: cuda is not available, try running on CPU")
        sys.exit(1)

    if args.subcommand == "train":
        check_paths(args)
        train(args)
    else:
        stylize(args)


if __name__ == "__main__":
    main()
