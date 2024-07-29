import os
import pdb
import sys
import logging
from tqdm import tqdm, trange
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn import metrics
from transformers import Trainer, TrainingArguments
from transformers.trainer_callback import EarlyStoppingCallback
from data_set import MyDataset


class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


def set_logger(args, name, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if args.text_name == "text_json_final":
        mmsd2_dir = os.path.join(output_dir, "MMSD2")
        if not os.path.exists(mmsd2_dir):
            os.makedirs(mmsd2_dir)
        log_file_path = os.path.join(mmsd2_dir, "training.log")
    elif args.text_name == "text_json_clean":
        mmsd_dir = os.path.join(output_dir, "MMSD")
        if not os.path.exists(mmsd_dir):
            os.makedirs(mmsd_dir)
        log_file_path = os.path.join(mmsd_dir, "training.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        file_handler = logging.FileHandler(log_file_path, mode='w')
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                           datefmt='%m/%d/%Y %H:%M:%S')
        file_handler.setFormatter(file_formatter)

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_formatter = logging.Formatter('%(message)s')
        stream_handler.setFormatter(stream_formatter)

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
    return logger


class MyTrainer():
    def __init__(self, args, processor):
        self.args = args
        self.processor = processor

    def training(self, args, model, device, train_data, dev_data, processor):
        if not os.path.exists(args.output_dir):
            os.mkdir(args.output_dir)

        train_loader = DataLoader(dataset=train_data,
                                  batch_size=args.train_batch_size,
                                  collate_fn=MyDataset.collate_fn,
                                  shuffle=True)
        total_steps = int(len(train_loader) * args.num_train_epochs)
        model.to(device)

        from transformers.optimization import get_linear_schedule_with_warmup
        clip_params = list(map(id, model.model.parameters()))
        base_params = filter(lambda p: id(p) not in clip_params, model.parameters())
        optimizer = torch.optim.AdamW([
            {"params": base_params},
            {"params": model.model.parameters(), "lr": args.clip_learning_rate}
        ], lr=args.learning_rate, weight_decay=args.weight_decay)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=int(args.warmup_proportion * total_steps),
                                                    num_training_steps=total_steps)
        logger = logging.getLogger('my_model')

        trainable_params = model.calculate_trainable_params()
        logger.info(f"Total trainable parameters: {trainable_params}")

        logger.info('------------------ Begin Training! ------------------')
        validation_step = args.num_validation_steps
        early_stopping = EarlyStopping(patience=args.early_stop, verbose=True)

        for i_epoch in trange(0, int(args.num_train_epochs), desc="Epoch", disable=False):
            sum_loss = 0.
            sum_step = 0
            steps_since_validation = 0

            iter_bar = tqdm(train_loader, desc="Iter (loss=X.XXX)", disable=False)
            model.train()
            for step, batch in enumerate(iter_bar):
                text_list, image_list, label_list, id_list = batch
                inputs = processor(text=text_list, images=image_list, padding='max_length', truncation=True,
                                   max_length=args.max_len, return_tensors="pt").to(device)
                labels = torch.tensor(label_list).to(device)
                loss, score = model(inputs, labels=labels)
                sum_loss += loss.item()
                sum_step += 1
                steps_since_validation += 1

                iter_bar.set_description("Iter (loss=%5.3f)" % loss.item())
                loss.backward()
                optimizer.step()
                if args.optimizer_name == 'adam':
                    scheduler.step()
                optimizer.zero_grad()

                if steps_since_validation >= validation_step or step == len(train_loader) - 1:
                    dev_loss, dev_acc, dev_f1, dev_precision, dev_recall = self.evaluate_acc_f1(args, model,
                                                                                                device, dev_data,
                                                                                                processor, mode='dev')
                    logger.info(
                        f"epoch is: {i_epoch + 1},\n"
                        f"dev loss is: {dev_loss:.4f},\n"
                        f"dev_acc is: {dev_acc:.4f},\n"
                        f"dev_f1 is: {dev_f1:.4f},\n"
                        f"dev_precision is: {dev_precision:.4f},\n"
                        f"dev_recall is: {dev_recall:.4f}."
                    )
                    early_stopping(dev_loss)
                    if early_stopping.early_stop:
                        print("Early stopping")
                        break
                    steps_since_validation = 0
            if early_stopping.early_stop:
                logger.info("Early stopping")
                break
        logger.info('------------------ Train done! ------------------')

        if args.text_name == "text_json_final":
            model_path_to_save = os.path.join(args.output_dir, "MMSD2")
        elif args.text_name == "text_json_clean":
            model_path_to_save = os.path.join(args.output_dir, "MMSD")
        model_to_save = (model.module if hasattr(model, "module") else model)
        torch.save(model_to_save.state_dict(), os.path.join(model_path_to_save, 'model.pt'))
        torch.cuda.empty_cache()

    def testing(self, args, model, device, test_data, processor):
        if args.text_name == "text_json_final":
            load_path = os.path.join(args.output_dir, "MMSD2")
        elif args.text_name == "text_json_clean":
            load_path = os.path.join(args.output_dir, "MMSD")
        model_path = os.path.join(load_path, 'model.pt')
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            sys.exit(1)
        model.load_state_dict(torch.load(model_path))

        logger = logging.getLogger('my_model')
        logger.info('------------------ Begin Testing! ------------------')
        model.eval()
        test_loss, test_acc, test_f1, test_precision, test_recall = self.evaluate_acc_f1(args, model,
                                                                                         device, test_data,
                                                                                         processor, mode='test',
                                                                                         macro=True)
        _, _, test_f1_, test_precision_, test_recall_ = self.evaluate_acc_f1(args, model, device,
                                                                             test_data, processor, mode='test')
        logger.info(
            f"Test done! \n "
            f"test_acc: {test_acc:.4f},\n"
            f"marco_test_f1: {test_f1:.4f},\n"
            f"marco_test_precision: {test_precision:.4f},\n"
            f"macro_test_recall: {test_recall:.4f},\n"
            f"micro_test_f1: {test_f1_:.4f},\n"
            f"micro_test_precision: {test_precision_:.4f},\n"
            f"micro_test_recall: {test_recall_:.4f}"
        )

    def evaluate_acc_f1(self, args, model, device, data, processor, mode, macro=False, pre=None):
        data_loader = DataLoader(data, batch_size=args.dev_batch_size, collate_fn=MyDataset.collate_fn, shuffle=False)
        n_correct, n_total = 0, 0
        t_targets_all, t_outputs_all = None, None

        model.eval()
        sum_loss = 0.
        sum_step = 0
        with torch.no_grad():
            for i_batch, t_batch in enumerate(tqdm(data_loader, desc="Evaluating")):
                text_list, image_list, label_list, id_list = t_batch
                inputs = processor(text=text_list, images=image_list, padding='max_length',
                                   truncation=True, max_length=args.max_len, return_tensors="pt").to(device)
                labels = torch.tensor(label_list).to(device)

                t_targets = labels
                loss, t_outputs = model(inputs, labels=labels)
                sum_loss += loss.item()
                sum_step += 1

                outputs = torch.argmax(t_outputs, -1)

                n_correct += (outputs == t_targets).sum().item()
                n_total += len(outputs)

                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = outputs
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, outputs), dim=0)
        logger = logging.getLogger(__name__)
        if mode == 'test':
            logger.info("test loss: {:.4f}".format(sum_loss / sum_step))
        else:
            logger.info("dev loss: {:.4f}".format(sum_loss / sum_step))
        final_loss = sum_loss / sum_step
        if pre != None:
            with open(pre, 'w', encoding='utf-8') as fout:
                predict = t_outputs_all.cpu().numpy().tolist()
                label = t_targets_all.cpu().numpy().tolist()
                for x, y, z in zip(predict, label):
                    fout.write(str(x) + str(y) + z + '\n')
        if not macro:
            acc = n_correct / n_total
            f1 = metrics.f1_score(t_targets_all.cpu(), t_outputs_all.cpu())
            precision = metrics.precision_score(t_targets_all.cpu(), t_outputs_all.cpu())
            recall = metrics.recall_score(t_targets_all.cpu(), t_outputs_all.cpu())
        else:
            acc = n_correct / n_total
            f1 = metrics.f1_score(t_targets_all.cpu(), t_outputs_all.cpu(), labels=[0, 1], average='macro')
            precision = metrics.precision_score(t_targets_all.cpu(), t_outputs_all.cpu(), labels=[0, 1],
                                                average='macro')
            recall = metrics.recall_score(t_targets_all.cpu(), t_outputs_all.cpu(), labels=[0, 1], average='macro')
        return final_loss, acc, f1, precision, recall

    def train(self, args, model, processor, device, train_data, dev_data, test_data):
        if args.train:
            self.training(args, model, device, train_data, dev_data, processor)
        if args.test:
            self.testing(args, model, device, test_data, processor)
        if not args.train and not args.test:
            print("No action specified. Please use --train and/or --test.")
