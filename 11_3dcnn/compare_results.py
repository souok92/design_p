import os
import json
import argparse
import pandas as pd


def find_experiments(results_dir):
    """Find all experiment results"""
    experiments = []
    
    if not os.path.exists(results_dir):
        return experiments
    
    for person in os.listdir(results_dir):
        person_dir = os.path.join(results_dir, person)
        if not os.path.isdir(person_dir):
            continue
            
        for exp_name in os.listdir(person_dir):
            exp_dir = os.path.join(person_dir, exp_name)
            info_file = os.path.join(exp_dir, 'experiment_info.json')
            metrics_file = os.path.join(exp_dir, 'metrics.txt')
            
            if os.path.exists(info_file):
                with open(info_file, 'r', encoding='utf-8') as f:
                    info = json.load(f)
                info['exp_dir'] = exp_dir
                info['person'] = person
                
                # Parse metrics if available
                if os.path.exists(metrics_file):
                    metrics = parse_metrics(metrics_file)
                    info['metrics'] = metrics
                
                experiments.append(info)
    
    return experiments


def parse_metrics(metrics_file):
    """Parse metrics from text file"""
    metrics = {}
    
    with open(metrics_file, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('=') or line.startswith('-') or line.startswith('Angle'):
            continue
        if line.startswith('PER-ANGLE'):
            continue
        
        parts = line.split()
        if len(parts) >= 4:
            name = parts[0]
            try:
                mae = float(parts[1])
                rmse = float(parts[2])
                metrics[name] = {'MAE': mae, 'RMSE': rmse}
            except ValueError:
                continue
    
    return metrics


def compare_experiments(experiments, sort_by='overall_mae'):
    """Compare experiments and create summary"""
    
    if not experiments:
        print("No experiments found!")
        return None
    
    # Create comparison table
    rows = []
    for exp in experiments:
        config = exp.get('config', {})
        data_config = config.get('data', {})
        model_config = config.get('model', {})
        
        # Get overall MAE
        metrics = exp.get('metrics', {})
        overall = metrics.get('Overall', {})
        
        row = {
            'Experiment': exp.get('exp_name', 'unknown'),
            'Person': exp.get('person', 'N/A'),
            'Model': model_config.get('model_type', 'r3d'),
            'Seq Len': data_config.get('sequence_length', 'N/A'),
            'Stride': data_config.get('stride', 1),
            'MAE (°)': overall.get('MAE', exp.get('best_val_loss', 999)),
            'RMSE (°)': overall.get('RMSE', 'N/A'),
            'Epoch': exp.get('final_epoch', 'N/A'),
        }
        rows.append(row)
    
    # Sort by metric
    if sort_by == 'overall_mae':
        rows.sort(key=lambda x: x['MAE (°)'] if isinstance(x['MAE (°)'], (int, float)) else 999)
    elif sort_by == 'overall_rmse':
        rows.sort(key=lambda x: x['RMSE (°)'] if isinstance(x['RMSE (°)'], (int, float)) else 999)
    
    # Print table
    df = pd.DataFrame(rows)
    
    print("\n" + "="*100)
    print("EXPERIMENT COMPARISON (3D CNN)")
    print("="*100)
    print(df.to_string(index=False))
    print("="*100)
    
    # Best result
    if rows:
        best = rows[0]
        print(f"\n★ Best Result: {best['Experiment']}")
        print(f"   MAE: {best['MAE (°)']}°, RMSE: {best['RMSE (°)']}°")
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Compare 3D CNN experiment results')
    parser.add_argument('--results_dir', type=str, default='./results', help='Results directory')
    parser.add_argument('--sort_by', type=str, default='overall_mae', 
                        choices=['overall_mae', 'overall_rmse'], help='Sort metric')
    parser.add_argument('--output', type=str, default=None, help='Output CSV file')
    args = parser.parse_args()
    
    experiments = find_experiments(args.results_dir)
    df = compare_experiments(experiments, args.sort_by)
    
    if args.output and df is not None:
        df.to_csv(args.output, index=False)
        print(f"\nComparison saved to {args.output}")


if __name__ == '__main__':
    main()
