import argparse
import itertools
import os

from tqdm import tqdm

from constants import ROOT_DIR
from logger import setup_logger
from network.style_net import StyleTransferNetwork


def cli():
    parser=argparse.ArgumentParser(description='Run test samples')
    parser.add_argument("--test-dir","-td",default=os.path.join(ROOT_DIR,"test_samples"), type=str, help='Path to images directory')
    parser.add_argument("--output-dir","-od",default=os.path.join(ROOT_DIR,"generated"), type=str, help='Path to output directory')
    parser.add_argument("--log_level","-l",default="DEBUG", type=str, help='Log level')
    parser.add_argument("--num_iter","-n",default=3e2, type=int, help='Number of iterations')
    parser.add_argument("--learning_rate","-lr",default=0.02, type=float, help='Learning rate')
    parser.add_argument("--content_weight","-cw",default=1e4, type=float, help='Content weight')
    parser.add_argument("--style_weight","-sw",default=1e2, type=float, help='Style weight')
    return parser.parse_args()

def main(args):
    network=StyleTransferNetwork(num_iter=args.num_iter,lr=args.learning_rate)
    test_dir = args.test_dir
    images=os.listdir(test_dir)
    content_images,styles_images=[],[]
    for image in images:
        if not os.path.isfile(os.path.join(test_dir,image)):
            continue
        if "_style" in image:
            styles_images.append(os.path.join(test_dir,image))
        else:
            content_images.append(os.path.join(test_dir,image))
    pbar=tqdm(itertools.product(content_images,styles_images),desc="Generating images")
    for content_image,style_image in pbar:
        network.transfer(content_image,style_image,args.output_dir,args.content_weight,args.style_weight,lr=args.learning_rate)
        break

if __name__=="__main__":
    args=cli()
    setup_logger(args)
    main(args)