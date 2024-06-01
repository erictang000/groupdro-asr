from transformers import WhisperProcessor,WhisperForConditionalGeneration, Wav2Vec2Processor, Wav2Vec2ForCTC
import torchaudio
from datasets import Dataset
import torch
import os
import numpy as np
from jiwer import wer
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from collections import defaultdict
import re

# Function to load and preprocess audio files
def load_audio(path):
    speech, _ = torchaudio.load(path)
    return speech.squeeze().numpy()

# Function to decode model predictions
def decode_predictions(pred_ids):
    pred_ids = pred_ids.cpu().numpy()
    pred_str = processor.batch_decode(pred_ids)
    return pred_str

# Evaluate WER
def compute_wer(model, processor, batch, language="swahili"):
    inputs = {key: batch[key].to("cuda") for key in batch if key != "audio" and key != "sentence"}

    with torch.no_grad():
        pred_ids = model.generate(inputs["input_features"], language=language)
    
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    pred_str = [remove_punctuation(x).lower().strip() for x in pred_str]
    label_str = processor.batch_decode(batch["labels"].cpu().numpy(), skip_special_tokens=True)
    
    wer_score = wer(label_str, pred_str)
    return wer_score

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int
    use_group_dro: bool = False

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        if self.use_group_dro:
            batch["group_idx"] = torch.LongTensor([feature["group_idx"] for feature in features])
        return batch

def remove_punctuation(text):
        return re.sub(r'[^\w\s]', '', text)

def get_data(dataset_path):
    languages = []

    sources = os.listdir(dataset_path)

    for source in sources:
        if source[0] != ".":
            languages.extend(os.listdir(os.path.join(dataset_path, source)))
            
    languages = set([x for x in languages if '.' not in x])

    all_paths = {}
    all_sentences = {}
    for duration in ["10min", "1h"]:
        for split in ["train", "test"]:
            language_to_paths = defaultdict(list)
            language_to_sentences = defaultdict(list)
            for language in languages:
                for source in sources:
                    source_lang_path = os.path.join(dataset_path, source, language)
                    if os.path.exists(os.path.join(source_lang_path, f'transcript_{duration}_{split}.txt')):
                        with open(os.path.join(source_lang_path, f'transcript_{duration}_{split}.txt'), 'r') as file:
                            lines = [line.rstrip() for line in file]
                            sentences = []
                            paths = []
                            for line in lines:
                                sentence = " ".join(re.split(r'[ \t]+', line)[2:])
                                sentence = remove_punctuation(sentence).lower().strip()
                                if len(sentence) <= 1:
                                    continue
                                if len(re.split(r'[ \t]+', line)[0]) > 0:
                                    sentences.append(sentence)
                                    paths.append(os.path.join(source_lang_path, 'wav', re.split(r'[ \t]+', line)[0] + '.wav'))

                            language_to_paths[language].extend(paths)
                            language_to_sentences[language].extend(sentences)
            all_paths[duration + split] = language_to_paths
            all_sentences[duration + split] = language_to_sentences
    return all_paths, all_sentences


def get_data_group_sources(dataset_path, language="eng"):
    languages = []

    sources = os.listdir(dataset_path)

    for source in sources:
        if source[0] != ".":
            languages.extend(os.listdir(os.path.join(dataset_path, source)))
            
    languages = set([x for x in languages if '.' not in x])
    if language not in languages:
        raise ValueError(f"Language {language} not found in dataset")

    all_paths = {}
    all_sentences = {}
    for duration in ["10min", "1h"]:
        for split in ["train", "test"]:
            sources_to_paths = defaultdict(list)
            sources_to_sentences = defaultdict(list)
            for source in sources:
                source_lang_path = os.path.join(dataset_path, source, language)
                if os.path.exists(os.path.join(source_lang_path, f'transcript_{duration}_{split}.txt')):
                    with open(os.path.join(source_lang_path, f'transcript_{duration}_{split}.txt'), 'r') as file:
                        lines = [line.rstrip() for line in file]
                        sentences = []
                        paths = []
                        for line in lines:
                            sentence = " ".join(re.split(r'[ \t]+', line)[2:])
                            sentence = remove_punctuation(sentence).lower().strip()
                            if len(sentence) <= 1:
                                continue
                            if len(re.split(r'[ \t]+', line)[0]) > 0:
                                sentences.append(sentence)
                                paths.append(os.path.join(source_lang_path, 'wav', re.split(r'[ \t]+', line)[0] + '.wav'))

                        sources_to_paths[source].extend(paths)
                        sources_to_sentences[source].extend(sentences)
            all_paths[duration + split] = sources_to_paths
            all_sentences[duration + split] = sources_to_sentences
    return all_paths, all_sentences
