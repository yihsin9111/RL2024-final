# RL2024-final
Project : Human-to-Robot Video-Based Adaptation for Imitation Learning
Team : Secure Shell

### Build Environment

To set up the environment for this project, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/tonyzhaozh/act.git
    ```

2. Follow the steps in the `tonyzhaozh/act` repository to set up the environments.

3. Install the required Python packages:
    ```bash
    pip install Quaternion transformers ultralytics mediapipe
    ```

4. Place the files in `pandas/assets` under `act/assets`.

5. Place `build_dataset.py` and `build_dataset_h2o.py` under `act`.

### Steps to Build 3D Skeleton for Imitation Learning

The below steps will help build the 3D skeleton of human and object in order to do further imitation learning:

1. Turn the video into frames:
    ```bash
    python video2frames.py --input <video dir> --output <frames dir>
    ```

2. Turn the series of frames into 3D skeleton:
    ```bash
    python frames2skeleton.py --input <frames dir> --output <skeletons dir>
    ```

3. Turn the 3D skeleton into robot arms dataset for ACT model:
    - For general dataset:
      ```bash
      python build_dataset.py --input <skeletons dir> --output <dataset dir>
      ```
    - For H2O dataset:
      ```bash
      python build_dataset_h2o.py --input <skeletons dir> --output <dataset dir>
      ```
### Train & Inference
```bash
python3 imitate_episodes.py \
--task_name <task_name> \
--ckpt_dir <ckpt dir> \
--policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 32 --dim_feedforward 3200 \
--num_epochs 8000  --lr 1e-5 \
--seed 0
```

### Validation
```bash
python3 imitate_episodes.py \
--task_name <task_name> \
--ckpt_dir <ckpt dir> \
--policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 32 --dim_feedforward 3200 \
--num_epochs 8000  --lr 1e-5 \
--seed 0 \
--eval \
--onscreen_render
```
### dataset
our data extracted from H2O dataset can be found at : https://drive.google.com/file/d/1ZiYyKMYzEYbIx4ipVxvymGh2E7vNTM4q/view?usp=sharing
