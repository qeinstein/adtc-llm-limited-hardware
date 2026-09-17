This private dataset is the durable checkpoint store for falcon-production-v1.
Trainer checkpoints are uploaded as versioned archives during training. The base
Falcon model is reloaded from its pinned Hugging Face revision and is never
stored in this dataset.
