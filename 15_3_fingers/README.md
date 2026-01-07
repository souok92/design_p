# CNN+LSTM for 3-Finger Angle Prediction (6 Joints)

This project is a copy of the `10_thumbless_model`, adapted to predict the angles of only 3 fingers (Index, Middle, Ring), resulting in 6 output joints.

## Model
- **Architecture**: CNN (ResNet-18) + LSTM
- **Outputs**: 6 (Index_MCP, Index_PIP, Middle_MCP, Middle_PIP, Ring_MCP, Ring_PIP)

## Key Changes
- The project is based on the CNN+LSTM architecture from `10_thumbless_model`.
- The `data/dataset.py` script is modified to load only 6 angle columns from the CSV files.
- The model configuration (`configs/*.yaml`) is updated to reflect `num_outputs: 6`.
- The `train.py` and `evaluate.py` scripts have been updated to dynamically handle the number of outputs.

## Usage
To train the model, run the `train.py` script with one of the config files:
```bash
python train.py --config configs/config_sj.yaml
```

To evaluate a trained model, run the `evaluate.py` script:
```bash
python evaluate.py --config configs/config_sj.yaml --checkpoint checkpoints/sj/{exp_name}/best_model.pth
```