[ECCV DriveX 2026] CMD: Class Margin Dispersion for Post-Hoc Misclassification Detection
==============================================================================================

Our implementation builds upon and extends the code provided by the Relative Uncertainty (Rel-U) project.
https://github.com/edadaltocg/relative-uncertainty

Key Contributions
-----------------
- Novel uncertainty estimator: Introduces Class Margin Dispersion (CMD), a training-free, post-hoc method for estimating predictive uncertainty.
- Richer logit-space information: CMD measures the variability of nonlinear margins between the predicted class and competing classes, rather than relying only on softmax confidence or a single margin value.
- Architecture-agnostic evaluation: Demonstrates effectiveness across both CNNs and transformer-based models, showing that CMD is not tied to a specific model architecture.
- Strong empirical performance: CMD improves key uncertainty metrics, including AUROC, FPR@95TPR, and AURC, across autonomous-driving datasets (KITTI nuImages) and Tiny-ImageNet.
- Implements post-hoc misclassification detection baselines (MSP, ODIN, Entropy, Doctor, REL-U, LogitGap, CMD (ours)).

Installation
------------
1. Clone the repository:
   - git clone <repo_url>

2. Install dependencies and Prepare datasets:
   - pip install -r requirements.txt

3. Optional environment variables:
   - DATA_DIR: path for datasets 

Usage
-----
1. Evaluate a post-hoc method (MSP, ODIN, etc.):
   python main.py --model_name densenet121_tinyimagenet --method msp --style ce --batch_size 128 --seed 1 --checkpoint path/to/checkpoint

2. Train and evaluate BILD (learnable constrained):
   python train.py --model_name resnet34_tinyimagenet --style ce --batch_size 128 --seed 1

Arguments
---------
- --model_name: Model architecture, e.g., densenet121_tinyimagenet, fastvit_tinyimagenet
- --method: Detection method (msp, odin, doctor, cmd, ...)
- --batch_size: Batch size
- --r: Ratio for dataset splitting
- --lbd: Lambda for learnable methods
- --seed: Random seed
- --temperature: Softmax temperature
- --magnitude: Input perturbation magnitude
- --lr: Learning rate (for learnable methods)

Evaluation
----------
The framework computes the following metrics:

- Model accuracy
- FPR95@TPR
- AUROC
- AURC

Results are saved as CSV files.

Project Structure
-----------------
- utils/methods: Implementation of CMD and baseline detectors
- utils: Dataset loaders, model utilities, evaluation functions, helper scripts
- main.py: Main evaluation and detectors training script
- train/ : Train and Evaluate convolutional neural networks on different training regime (CE, MixUp, RegMixUp)
- results/: Output directory for evaluation metrics

Miscellaneous
-------------
- Plots presented in the paper and the code to generate them are included in this repo.