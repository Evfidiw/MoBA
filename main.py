import os
import torch
import argparse
import random
import numpy as np
from transformers import CLIPProcessor
from PIL import ImageFile

from model import Model
from trainer import MyTrainer, set_logger
from data_set import MyDataset


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='1', type=str, help='device number')
    parser.add_argument('--model', default='CLIP', type=str, help='the model name', choices=['CLIP'])
    parser.add_argument('--model_path', default='CLIP-vit', type=str, help='the model path')
    parser.add_argument('--fusion', default='att', type=str, help='the fusion method', choices=['add', 'concat', 'gate', 'att'])
    parser.add_argument('--text_name', default='text_json_final', type=str, help='the text data folder name')
    parser.add_argument('--simple_linear', default=False, type=bool, help='linear implementation choice')
    parser.add_argument('--num_train_epochs', default=10, type=int, help='number of train epoched')
    parser.add_argument('--num_validation_steps', default=315, type=int, help='number of dev steps')
    parser.add_argument('--early_stop', default=5, type=int, help='early_stop')
    parser.add_argument('--train_batch_size', default=32, type=int, help='batch size in train phase')
    parser.add_argument('--dev_batch_size', default=32, type=int, help='batch size in dev phase')
    parser.add_argument('--label_number', default=2, type=int, help='the number of classification labels')
    parser.add_argument('--text_size', default=512, type=int, help='text hidden size')
    parser.add_argument('--image_size', default=768, type=int, help='image hidden size')
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--optimizer_name", type=str, default='adam', help="use which optimizer to train the model.")
    parser.add_argument('--learning_rate', default=5e-4, type=float, help='learning rate for modules expect CLIP')
    parser.add_argument('--clip_learning_rate', default=1e-6, type=float, help='learning rate for CLIP')
    parser.add_argument('--max_len', default=50, type=int, help='max len of text based on CLIP')
    parser.add_argument('--layers', default=2, type=int, help='number of transformer layers')
    parser.add_argument('--num_experts', default=8, type=int, help='number of experts')
    parser.add_argument('--max_grad_norm', default=5.0, type=float, help='grad clip norm')
    parser.add_argument('--weight_decay', default=0.05, type=float, help='weight decay')
    parser.add_argument('--warmup_proportion', default=0.2, type=float, help='warm up proportion')
    parser.add_argument('--dropout_rate', default=0.1, type=float, help='dropout rate')
    parser.add_argument('--output_dir', default='output_dir/', type=str, help='the output path')
    parser.add_argument('--limit', default=None, type=int, help='the limited number of training examples')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument("--train", action='store_true', help="Include to perform training")
    parser.add_argument("--test", action='store_true', help="Include to perform testing")
    return parser.parse_args()


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    args = set_args()
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    train_data = MyDataset(mode='train', text_name=args.text_name, limit=None)
    dev_data = MyDataset(mode='valid', text_name=args.text_name, limit=None)
    test_data = MyDataset(mode='test', text_name=args.text_name, limit=None)

    if args.model == 'CLIP':
        processor = CLIPProcessor.from_pretrained(args.model_path)
        model = Model(args)
    else:
        raise RuntimeError('Error model name!')
    model.to(device)

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    logger = set_logger(args, 'my_model', args.output_dir)

    trainer = MyTrainer(args, processor)
    trainer.train(args, model, processor, device, train_data, dev_data, test_data)
