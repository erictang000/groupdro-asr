# cs224s-project
Group DRO ASR

### Setup
In order to run experiments on Group-DRO for languages, refer to `bash.sh`. For experiments on Group-DRO for groupings based on data source, refer to `bash_source.sh`.

Comment out `~/miniconda3/lib/python3.10/site-packages/transformers/generation/utils.py:1542` -> `self._validate_model_kwargs(model_kwargs.copy())` in order to allow the custom Seq2SeqDROTrainer to pass in group_idxs as an argument for computing loss at validation time!

