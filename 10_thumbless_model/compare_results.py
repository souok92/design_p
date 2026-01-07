"""
Compare multiple experiment results
"""
import os
import json
import argparse
from tabulate import tabulate


def load_experiment_info(exp_dir):
    """Load experiment info from directory"""
    info_path = os.path.join(exp_dir, 'experiment_info.json')
    if os.path.exists(info_path):
        with open(info_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def compare_experiments(results_dir, person):
    """Compare all experiments for a person"""
    person_dir = os.path.join(results_dir, person)

    if not os.path.exists(person_dir):
        print(f"No results found for {person}")
        return

    # Load all experiments
    experiments = []
    exp_dirs = [d for d in os.listdir(person_dir)
                if os.path.isdir(os.path.join(person_dir, d))]

    for exp_name in sorted(exp_dirs):
        exp_dir = os.path.join(person_dir, exp_name)
        info = load_experiment_info(exp_dir)
        if info:
            experiments.append({
                'name': exp_name,
                'info': info
            })

    if not experiments:
        print(f"No valid experiments found for {person}")
        return

    # Prepare comparison table
    headers = ['Experiment', 'Timestamp', 'Seq Len', 'Hidden', 'Layers',
               'LR', 'Loss', 'MAE', 'RMSE', 'Epoch']
    rows = []

    for exp in experiments:
        info = exp['info']
        rows.append([
            exp['name'][:30],  # Truncate long names
            info['timestamp'].split()[1],  # Time only
            info['model_config']['sequence_length'],
            info['model_config']['lstm_hidden_size'],
            info['model_config']['lstm_num_layers'],
            f"{info['training_config']['learning_rate']:.0e}",
            info['training_config']['loss_type'],
            f"{info['results']['overall_mae']:.3f}",
            f"{info['results']['overall_rmse']:.3f}",
            info.get('checkpoint_epoch', 'N/A')
        ])

    # Print comparison table
    print(f"\n{'='*120}")
    print(f"Experiment Comparison: {person.upper()}")
    print(f"{'='*120}")
    print(tabulate(rows, headers=headers, tablefmt='grid'))

    # Find best experiment
    best_mae_idx = min(range(len(experiments)),
                       key=lambda i: experiments[i]['info']['results']['overall_mae'])
    best_exp = experiments[best_mae_idx]

    print(f"\n{'='*120}")
    print(f"Best Experiment (Lowest MAE): {best_exp['name']}")
    print(f"{'='*120}")
    print(f"MAE: {best_exp['info']['results']['overall_mae']:.4f}")
    print(f"RMSE: {best_exp['info']['results']['overall_rmse']:.4f}")
    print(f"Config: {os.path.join(person_dir, best_exp['name'], 'experiment_config.yaml')}")
    print(f"{'='*120}\n")


def main():
    parser = argparse.ArgumentParser(description='Compare experiment results')
    parser.add_argument('--results_dir', type=str, default='./results',
                        help='Results directory')
    parser.add_argument('--person', type=str, choices=['sj', 'tk', 'all'],
                        default='all', help='Person to compare')
    args = parser.parse_args()

    if args.person == 'all':
        for person in ['sj', 'tk']:
            compare_experiments(args.results_dir, person)
    else:
        compare_experiments(args.results_dir, args.person)


if __name__ == '__main__':
    main()
