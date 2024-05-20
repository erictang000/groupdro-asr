from evaluate import evaluator
from datasets import load_dataset

model_name = "AntonyG/fine-tune-wav2vec2-large-xls-r-1b-sw"

# Load data and evaluator
task_evaluator = evaluator("automatic-speech-recognition")
tsn_test = load_dataset("tolulope/ml-superb-subset", name="tsn", split="test[:100]")


# temp fix - from https://github.com/huggingface/evaluate/issues/437
task_evaluator.PIPELINE_KWARGS.pop('truncation', None)
assert 'truncation' not in task_evaluator.PIPELINE_KWARGS

# Compute WER
results = task_evaluator.compute(
    model_or_pipeline=model_name,
    data=tsn_test,
    input_column="audio",
    label_column="sentence",
    metric="wer",
)
results