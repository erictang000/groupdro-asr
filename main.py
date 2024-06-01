import os
import datasets 
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch

from dataclasses import dataclass
from safetensors import safe_open
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from jiwer import wer

from utils import *
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from seq2seq_dro_trainer import Seq2SeqDROTrainer
from argparse import ArgumentParser

def run(args):
    all_paths, all_sentences = get_data(args.dataset_path)

    LANG = args.processor_language
    MODEL_ID = args.model_id
    processor = WhisperProcessor.from_pretrained(MODEL_ID, language=LANG)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID).to("cuda")

    languages = args.languages

    # Prepare the data for Hugging Face datasets
    train_data = {
        "audio": [],
        "sentence": [],
    }
    if args.use_group_dro:
        train_data["group_idx"] = []

    test_data = {
        "audio": [],
        "sentence": [],
    }
    if args.use_group_dro:
        test_data["group_idx"] = []
    group_str_to_idx = {language: idx for idx, language in enumerate(languages)}

    for language in languages: 
        for path, sentence in zip(all_paths[args.train_set][language], all_sentences[args.train_set][language]):
            train_data["audio"].append(path)
            train_data["sentence"].append(sentence)
            if args.use_group_dro:
                train_data["group_idx"].append(group_str_to_idx[language])
        for path, sentence in zip(all_paths["10mintest"][language], all_sentences["10mintest"][language]):
            test_data["audio"].append(path)
            test_data["sentence"].append(sentence)
            if args.use_group_dro:
                test_data["group_idx"].append(group_str_to_idx[language])

    # Create a Hugging Face dataset
    train_dataset = Dataset.from_dict(train_data)
    test_dataset = Dataset.from_dict(test_data)

    # Preprocess the dataset
    def preprocess(batch):
        audio = [load_audio(path) for path in batch["audio"]]
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt").to("cuda")
        labels = processor.tokenizer(text=batch["sentence"], return_tensors="pt", padding=True).input_ids
        inputs["labels"] = labels.to("cuda")
        return inputs

    train_set = train_dataset.map(preprocess, batched=True, batch_size=args.train_batch_size).with_format("torch")
    test_set = test_dataset.map(preprocess, batched=True, batch_size=args.train_batch_size).with_format("torch")
    train_dataloader = DataLoader(train_set, batch_size=args.train_batch_size, shuffle=True, num_workers=args.num_workers)
    test_dataloader = DataLoader(test_set, batch_size=args.train_batch_size, shuffle=False, num_workers=args.num_workers)

    training_args = Seq2SeqTrainingArguments(
        output_dir="./",  # change to a repo name of your choice
        per_device_train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,  # increase by 2x for every 2x decrease in batch size
        learning_rate=args.learning_rate,
        warmup_steps=args.num_warmup_steps,
        max_steps=args.steps,
        gradient_checkpointing=True,
        fp16=True,
        evaluation_strategy="steps",
        per_device_eval_batch_size=args.eval_batch_size,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=1000,
        eval_steps=args.eval_frequency,
        logging_steps=100,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
    )

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
        use_group_dro=args.use_group_dro
    )

    # Evaluate WER
    def compute_metrics(batch):
        global wer
        label_ids = batch.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_ids = batch.predictions
        
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        pred_str = [remove_punctuation(x).lower().strip() for x in pred_str]
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
        
        wer_score = wer(label_str, pred_str)
        return {"wer": wer_score}

    if args.use_group_dro:
        group_idx_to_str = {idx: language for idx, language in enumerate(languages)}
        def group_str_fn(group_idx):
            return group_idx_to_str[group_idx]
        trainer = Seq2SeqDROTrainer(
            n_groups=len(languages),
            group_counts=torch.LongTensor([len(all_paths["10mintrain"][language]) for language in languages]),
            group_str_fn=group_str_fn,
            alpha=args.alpha,
            args=training_args,
            model=model,
            train_dataset=train_set,
            eval_dataset=test_set,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            tokenizer=processor.feature_extractor,
        )
    else:
        trainer = Seq2SeqTrainer(
            args=training_args,
            model=model,
            train_dataset=train_set,
            eval_dataset=test_set,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            tokenizer=processor.feature_extractor,
        )

    if args.steps > 0:
        trainer.train()

    ## Print final per language eval metrics
    all_wer_scores = []
    average_wers = {}
    for language in languages: 
        test_data = {
            "audio": [],
            "sentence": []
        }
        for path, sentence in zip(all_paths["10mintest"][language], all_sentences["10mintest"][language]):
            test_data["audio"].append(path)
            test_data["sentence"].append(sentence)

        test_dataset = Dataset.from_dict(test_data)

        test_set = test_dataset.map(preprocess, batched=True, batch_size=args.eval_batch_size).with_format("torch")
        test_dataloader = DataLoader(test_set, batch_size=args.eval_batch_size)
        
        model.generation_config.language = LANG
        model.generation_config.task = "transcribe"
        model.generation_config.forced_decoder_ids = None

        # Compute WER for the entire dataset
        wer_scores = []

        # Use tqdm to wrap your dataloader to show a progress bar
        for batch in tqdm(test_dataloader, desc="Processing batches"):
            wer_scores.append(compute_wer(model, processor, batch, language=LANG))

        average_wer = np.mean(wer_scores)
        print(f"{language} Average WER: {average_wer}")
        average_wers[language] = average_wer

        all_wer_scores.extend(wer_scores)
    print(f"Overall Average WER: {np.mean(all_wer_scores)}")

    model_name = MODEL_ID.split("/")[-1] + "_" + LANG + "_" + args.train_set + "_" + "_".join(languages) + "_WER_" + str(np.mean(all_wer_scores)) + '_' + str(args.steps) + '_group_dro_' + str(args.use_group_dro) + ".pt"
    torch.save(model.state_dict(), os.path.join(args.output_dir, model_name))

    # save average wers
    with open(os.path.join(args.output_dir, model_name.replace(".pt", ".txt")), "w") as f:
        for language, avg_wer in average_wers.items():
            f.write(f"{language} {avg_wer}\n")

if __name__ == "__main__":
    parser = ArgumentParser()

    # training arguments
    parser.add_argument('--dataset_path', default='/vision/u/eatang/ml_superb/eighth_version/', type=str)
    parser.add_argument('--train_batch_size', default=64, type=int)
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int)
    parser.add_argument('--eval_batch_size', default=64, type=int)
    parser.add_argument('--learning_rate', default=1e-4, type=float)
    parser.add_argument('--num_warmup_steps', default=50, type=int)
    parser.add_argument('--steps', default=300, type=int)
    parser.add_argument('--eval_frequency', default=100, type=int)
    parser.add_argument('--processor_language', default="swahili", type=str)
    parser.add_argument('--model_id', default="openai/whisper-tiny", type=str)
    parser.add_argument('--languages', nargs='+', help='languages to include in train/test set', default=["common", "xho", "swa"])
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--train_set', default="10mintrain", type=str, choices=["10mintrain", "1htrain"])
    parser.add_argument('--output_dir', default="./checkpoints/", type=str)

    ## Group DRO
    parser.add_argument('--alpha', default=0.2, type=float)
    parser.add_argument('--use_group_dro', default=True, type=lambda x: (str(x).lower() == 'true'))

    args = parser.parse_args()

    run(args)