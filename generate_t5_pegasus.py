#!/usr/bin/env python
# coding=utf-8
import argparse
from functools import partial
from tqdm import tqdm
import jieba
import torch
import logging
import sacrebleu
import numpy as np
from accelerate import Accelerator
from datasets import load_dataset
from torch.utils.data.dataloader import DataLoader
from transformers import (
    DataCollatorForSeq2Seq,
    set_seed,
    BertTokenizer,
    T5ForConditionalGeneration
)


def postprocess_text(preds, labels):
    preds = [pred.replace(" ", "") for pred in preds]
    labels = [label.replace(" ", "") for label in labels]
    return preds, labels


def gen_args():
    parser = argparse.ArgumentParser(description="generation for t5 pegasus")
    parser.add_argument(
        "--model_name", type=str, default='fnlp/bart-base-chinese',
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
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=815,
    )
    parser.add_argument("--output_path", type=str, default='steps', help="Where to store the final model.")
    parser.add_argument(
        "--log_file", type=str, default=None, help="log file."
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=5,
        help="Number of beams to use for evaluation. This argument will be "
             "passed to ``model.generate``, which is used during ``evaluate`` and ``predict``.",
    )
    args = parser.parse_args()

    # Sanity checks
    if args.validation_file is None:
        raise ValueError("Need either a dataset name or a validation file.")
    else:
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

    return args


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


def write_gen_file(predict_sys, out_file):
    with open(out_file, 'a+') as f:
        for line in predict_sys:
            f.write(line + '\n')
    print('write complete!')


class MODEL:
    def __init__(self, checkpoint):
        self.model = T5ForConditionalGeneration.from_pretrained(checkpoint)
        self.tokenizer = T5PegasusTokenizer.from_pretrained(checkpoint)

    def predict(self, args):
        logger = logging.getLogger(__name__)
        logging.basicConfig(
            filename=args.log_file,
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )

        accelerator = Accelerator()
        set_seed(args.seed)

        data_files = {'validation': args.validation_file}
        extension = args.validation_file.split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)

        column_names = raw_datasets['validation'].column_names
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

        eval_dataset = processed_datasets["validation"]
        label_pad_token_id = -100 if args.ignore_pad_token_for_loss else self.tokenizer.pad_token_id
        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            model=self.model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if accelerator.use_fp16 else None,
        )

        eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
        model, eval_dataloader = accelerator.prepare(
            self.model, eval_dataloader
        )

        model.eval()
        gen_kwargs = {
            "max_length": args.max_target_length,
            "num_beams": args.num_beams,
        }
        logger.info("***** start evaluating *****")
        predict_sys = []
        predict_ref = []
        for step, batch in enumerate(tqdm(eval_dataloader)):
            with torch.no_grad():
                generated_tokens = accelerator.unwrap_model(model).generate(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    **gen_kwargs,
                )
                generated_tokens = accelerator.pad_across_processes(
                    generated_tokens, dim=1, pad_index=self.tokenizer.pad_token_id
                )

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

        # predict_ref = [predict_ref]
        bleu_score = sacrebleu.corpus_bleu(predict_sys, [predict_ref], tokenize='zh').score
        logging.info(f"  current score = {bleu_score}")

        write_gen_file(predict_sys, args.output_path)


def main():
    args = gen_args()
    model = MODEL(args.model_name)
    model.predict(args)


if __name__ == '__main__':
    main()
