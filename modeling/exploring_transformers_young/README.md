# Finetuning Transformers

The notebook finetune_transformers.ipynb is designed to efficiently finetune one of four pre-trained transformer networks using our augmented audio (anchor, positive, negative) triplets. The four networks considered are 

    - Wav2VecBertModel from https://huggingface.co/docs/transformers/v4.44.1/en/model_doc/wav2vec2-bert
    - Wav2Vec2BertForSequenceClassification model from https://huggingface.co/docs/transformers/v4.44.1/en/model_doc/wav2vec2-bert#transformers.Wav2Vec2BertForSequenceClassification
    - HubertForSequenceClassification model from https://huggingface.co/docs/transformers/v4.44.1/en/model_doc/hubert#transformers.HubertForSequenceClassification
    - ASTModel from https://huggingface.co/docs/transformers/en/model_doc/audio-spectrogram-transformer#transformers.ASTModel.

## Data preparation

The raw audio data is fed into AutoFeatureExtractor or AutoProcessor as appropriate to produce the expected data type and format each model expects as input. The function make_train_loaders() takes as input a model object and produces one DataLoader object for training and one DataLoader object for validation containing correctly-formatted data for the input model.

## Finetuning

The function setup_model() freezes the early layers of any of the four model classes in order to finetune the later layers.