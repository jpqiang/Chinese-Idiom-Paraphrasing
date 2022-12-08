import math
from functools import partial
import jieba
import torch
import logging
import sacrebleu
import numpy as np
import argparse
import os
from transformers import SchedulerType
from accelerate import Accelerator
from datasets import load_dataset
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AdamW,
    DataCollatorForSeq2Seq,
    get_scheduler,
    set_seed,
    BertTokenizer,
    T5ForConditionalGeneration
)


class T5PegasusTokenizer(BertTokenizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pre_tokenizer = partial(jieba.cut, HMM=False)

    def _tokenize(self, text, *arg, **kwargs):
        split_tokens = []
        for text in self.pre_tokenizer(text):
            if text in self.vocab:
                split_tokens.append(text)
            else:
                split_tokens.extend(super()._tokenize(text))
        return split_tokens


def model_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a Seq2Seq task")
    parser.add_argument(
        "--model_name", type=str, default='fnlp/bart-base-chinese',
    )
    parser.add_argument(
        "--train_file", type=str, default='data/parabank2.json',
        help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default='data/eval.json',
        help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--ignore_pad_token_for_loss",
        type=bool,
        default=True,
        help="Whether to ignore the tokens corresponding to " "padded labels in the loss computation or not.",
    )
    parser.add_argument(
        "--max_source_length",
        type=int,
        default=1024,
        help="The maximum total input sequence length after "
             "tokenization.Sequences longer than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", type=bool, default=False, help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--max_target_length",
        type=int,
        default=1024,
        help="The maximum total sequence length for target text after "
             "tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."
             "during ``evaluate`` and ``predict``.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default='steps', help="Where to store the final model.")
    parser.add_argument("--save_every", type=int, default=-1, help="To save the model every certain number of steps.")
    parser.add_argument(
        "--log_file", type=str, default=None, help="log file."
    )
    parser.add_argument("--early_stop", type=int, default=5, help="early stop if score not increase.")
    parser.add_argument(
        "--num_beams",
        type=int,
        default=5,
        help="Number of beams to use for evaluation. This argument will be "
             "passed to ``model.generate``, which is used during ``evaluate`` and ``predict``.",
    )
    args = parser.parse_args()

    # Sanity checks
    if args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a dataset name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    # Set seed
    args.seed = 42
    return args


def postprocess_text(preds, labels):
    preds = [pred.replace(" ", "") for pred in preds]
    labels = [label.replace(" ", "") for label in labels]
    return preds, labels


class MODEL:
    def __init__(self, checkpoint):
        self.model = T5ForConditionalGeneration.from_pretrained(checkpoint)
        self.tokenizer = T5PegasusTokenizer.from_pretrained(checkpoint)

    def train(self, args):
        logger = logging.getLogger(__name__)
        logging.basicConfig(
            filename=args.log_file,
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        # Initialize the accelerator. We will let the accelerator handle device placement for us
        # in this example
        accelerator = Accelerator()
        set_seed(args.seed)

        data_files = {
            'train': args.train_file,
            'validation': args.validation_file
        }
        extension = args.train_file.split('.')[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)

        # Preprocessing the datasets
        # First we tokenize all the texts
        column_names = raw_datasets['train'].column_names
        text_column, summary_column = column_names[0], column_names[1]

        # Temporarily set max_target_length for training
        padding = True

        def preprocess_function(examples):
            inputs = examples[text_column]
            targets = examples[summary_column]
            inputs = [inp for inp in inputs]
            model_inputs = self.tokenizer(
                inputs,
                return_token_type_ids=False,
                max_length=args.max_source_length,
                padding=padding,
                truncation=True
            )

            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(
                    targets,
                    return_token_type_ids=False,
                    max_length=args.max_target_length,
                    padding=padding,
                    truncation=True
                )

            # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
            # padding in the loss.
            if padding == "max_length" and args.ignore_pad_token_for_loss:
                labels["input_ids"] = [
                    [(l if l != self.tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
                ]

            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        processed_datasets = raw_datasets.map(
            preprocess_function, batched=True, remove_columns=column_names,
            load_from_cache_file=False
        )

        train_dataset = processed_datasets["train"]
        eval_dataset = processed_datasets["validation"]

        label_pad_token_id = -100 if args.ignore_pad_token_for_loss else self.tokenizer.pad_token_id
        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            model=self.model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if accelerator.use_fp16 else None,
        )

        train_dataloader = DataLoader(
            train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
        )
        eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

        # Optimizer
        # Split weights in two groups, one with weight decay and the other not.
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

        # Prepare everything with our `accelerator`.
        model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
            self.model, optimizer, train_dataloader, eval_dataloader
        )

        # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
        # shorter in multiprocess)

        # Scheduler and math around the number of training steps.
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        if args.max_train_steps is None:
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        else:
            args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=args.max_train_steps,
        )

        total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * \
                           args.gradient_accumulation_steps

        print("***** Running training *****")
        print(f"  Num examples = {len(train_dataset)}")
        print(f"  Num Epochs = {args.num_train_epochs}")
        print(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        print(f"  Total optimization steps = {args.max_train_steps}")
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
        completed_steps = 0
        best_bleu_score = 0.0
        for epoch in range(args.num_train_epochs):
            model.train()
            for step, batch in enumerate(train_dataloader):
                outputs = model(**batch)
                loss = outputs.loss
                loss = loss / args.gradient_accumulation_steps
                accelerator.backward(loss)
                if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)
                    completed_steps += 1

                if args.save_every > 0:
                    if completed_steps % args.save_every == 0:
                        out_dir = f'{args.output_dir}/{completed_steps}'
                        os.makedirs(out_dir, exist_ok=True)
                        accelerator.wait_for_everyone()
                        unwrapped_model = accelerator.unwrap_model(model)
                        unwrapped_model.save_pretrained(out_dir, save_function=accelerator.save)

                if completed_steps >= args.max_train_steps:
                    break

            model.eval()
            gen_kwargs = {
                "max_length": args.max_target_length,
                "num_beams": args.num_beams,
            }
            logger.info("***** start evaluating *****")
            predict_sys = []
            predict_ref = []
            for step, batch in enumerate(eval_dataloader):
                with torch.no_grad():
                    generated_tokens = accelerator.unwrap_model(model).generate(
                        batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        **gen_kwargs,
                    )
                    generated_tokens = accelerator.pad_across_processes(
                        generated_tokens, dim=1, pad_index=self.tokenizer.pad_token_id
                    )
                    # labels = batch["labels"]
                    # if not args.pad_to_max_length:
                    # If we did not pad to max length, we need to pad the labels too
                    labels = accelerator.pad_across_processes(
                        batch["labels"], dim=1,
                        pad_index=self.tokenizer.pad_token_id
                    )

                    generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()
                    labels = accelerator.gather(labels).cpu().numpy()

                    if args.ignore_pad_token_for_loss:
                        # Replace -100 in the labels as we can't decode them.
                        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)

                    decoded_preds = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                    decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
                    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

                    predict_sys.extend(decoded_preds)
                    predict_ref.extend(decoded_labels)

            predict_ref = [predict_ref]
            bleu_score = sacrebleu.corpus_bleu(predict_sys, predict_ref, tokenize='zh').score
            logging.info(f"  current score = {bleu_score}")

            if bleu_score > best_bleu_score:
                best_bleu_score = bleu_score
                early_stop = args.early_stop

                logging.info(
                    "========== Save Best Model For Epoch: {} , Bleu is {} ==========".format(epoch+1, bleu_score))

                file_path = os.path.join(args.output_dir, str(epoch + 1))
                os.makedirs(file_path, exist_ok=True)

                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(file_path, save_function=accelerator.save)

            else:
                early_stop -= 1
                logging.info("Early Stop Left: {}".format(early_stop))

            if early_stop == 0:
                logging.info("-------- Early Stop ! --------")
                break


def main():
    args = model_args()
    model = MODEL(args.model_name)
    model.train(args)


if __name__ == '__main__':
    main()
