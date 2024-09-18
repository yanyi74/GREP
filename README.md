# GREP
## Requirements

Packages listed below are required.

- Python (tested on 3.7.2)
- CUDA (tested on 11.3)
- PyTorch (tested on 1.11.0)
- Transformers (tested on 4.18.0)
- spacy (tested on 2.3.7) 
- numpy (tested on 1.21.6)
- opt-einsum (tested on 3.3.0)
- ujson (tested on 5.3.0)
- tqdm (tested on 4.64.0)

## Datasets

Our experiments include the [DocRED](https://github.com/thunlp/DocRED) and [Re-DocRED](https://github.com/tonytan48/Re-DocRED) datasets. The expected file structure is as follows:

```
AA
 |-- dataset
 |    |-- docred
 |    |    |-- train_annotated.json
 |    |    |-- dev.json
 |    |    |-- test.json
 |    |    |-- train_revised.json
 |    |    |-- dev_revised.json
 |    |    |-- test_revised.json
 |-- meta
 |    |-- rel2id.json
 |    |-- rel_info.json
```

## Training
### DocRED
Models trained with and without evidence supervision are required for inference stage cross fusion (ISCF).
```bash
# Model without evidence supervision
bash scripts/run_bert.bash ${name} 0 ${seed} # For Bert
bash scripts/run_roberta.bash ${name} 0 ${seed} # For RoBERTa
# Model with evidence supervision
bash scripts/run_bert.bash ${name} ${lambda} ${seed} # For Bert
bash scripts/run_roberta.bash ${name} ${lambda} ${seed} # For RoBERTa
```
where ${name} represents the identifier for your training, ${lambda} is the weight of evidence supervision, and ${seed} is the random seed.
For example, to train a Bert-based model with evidence supervision on DocRED with a random seed 65, run: ``bash scripts/run_bert.bash bert 0.1 65``


### Re-DocRED
For the Re-DocRED dataset, you should use the model trained on DocRED with evidence supervision to generate token importance distributions.
```bash
bash scripts/infer_redoc_roberta.bash ${load_dir} 
```
Then, you can train the model on Re-DocRED:
```bash
# Model without evidence supervision
bash scripts/run_roberta_revised.bash ${name} ${teacher_signal_dir} 0 ${seed} # for RoBERTa
# Model with evidence supervision
bash scripts/run_roberta_revised.bash ${name} ${teacher_signal_dir} ${lambda} ${seed} # for RoBERTa
```

## Evaluation

### Dev set
The  inference stage cross fusion strategy is applied:
```bash
bash scripts/iscf_bert.bash ${model_dir} ${model_evi_dir} dev # for BERT
bash scripts/iscf_roberta.bash ${model_dir} ${model_evi_dir} dev # for RoBERTa
```
A threshold that minimizes the cross-entropy loss of RE on development set is saved in `${model_evi_dir}`.

### Test set

With `${model_evi_dir}/thresh` available, you can obtain the final predictions on the test set:
```bash
bash scripts/iscf_bert.bash ${model_dir} ${model_evi_dir}  test # for BERT
bash scripts/iscf_roberta.bash ${model_dir} ${model_evi_dir}  test # for RoBERTa

