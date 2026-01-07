import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

# MATLAB 스타일 설정
plt.style.use('default')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 16
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.linestyle'] = ':'
plt.rcParams['grid.linewidth'] = 0.5

# 현재 스크립트가 있는 디렉토리를 기준으로 작업
script_dir = Path(__file__).parent
output_dir = script_dir / 'correlation_results'
output_dir.mkdir(exist_ok=True)

# 데이터 로딩
print("Loading data...")
files = ['sj_1.csv', 'sj_2.csv', 'tk.csv']
dfs = []

for file in files:
    df = pd.read_csv(script_dir / file)
    dfs.append(df)
    print(f"  {file}: {len(df)} rows")

# 데이터 합치기
combined_df = pd.concat(dfs, ignore_index=True)
print(f"\nTotal rows: {len(combined_df)}")

# Index 손가락 데이터만 추출
pip_col = 'Index_PIP_Flex'
dip_col = 'Index_DIP_Flex'

data = combined_df[[pip_col, dip_col]].copy()
data_clean = data.dropna()
print(f"Rows after removing NaN: {len(data_clean)}")

pip_data = data_clean[pip_col]
dip_data = data_clean[dip_col]

# 상관계수 계산
r, p_value = stats.pearsonr(pip_data, dip_data)

# DIP/PIP 비율 계산
ratio_data = dip_data / pip_data.replace(0, np.nan)
ratio_data_clean = ratio_data.dropna()

ratio_mean = ratio_data_clean.mean()
ratio_std = ratio_data_clean.std()
ratio_median = ratio_data_clean.median()

print("\n" + "="*60)
print("Index Finger - DIP-PIP Correlation Analysis")
print("="*60)
print(f"Correlation coefficient (r): {r:.4f}")
print(f"P-value: {p_value:.2e}")
print(f"DIP/PIP Ratio - Mean: {ratio_mean:.3f}, Median: {ratio_median:.3f}, Std: {ratio_std:.3f}")

# 시각화 1: 산점도
print("\nGenerating scatter plot...")
fig, ax = plt.subplots(figsize=(10, 8))

ax.scatter(pip_data, dip_data, alpha=0.6, s=40,
          color='#0072BD', edgecolors='none', marker='o', label='Data points')

# 회귀선
z = np.polyfit(pip_data, dip_data, 1)
p_line = np.poly1d(z)
x_line = np.linspace(pip_data.min(), pip_data.max(), 100)
ax.plot(x_line, p_line(x_line), 'r-', linewidth=2.5, label='Regression line')

ax.set_xlabel('Index PIP Flex (degrees)', fontsize=18, fontweight='bold')
ax.set_ylabel('Index DIP Flex (degrees)', fontsize=18, fontweight='bold')
ax.set_title(f'Index Finger: DIP vs PIP Flexion Angle\nr = {r:.4f}, p = {p_value:.2e}',
            fontsize=20, fontweight='bold', pad=20)
ax.grid(True, linestyle=':', alpha=0.3)
ax.set_axisbelow(True)
ax.legend(fontsize=15, loc='upper left')

# MATLAB 스타일 박스
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)

# DIP/PIP 비율 정보
textstr = f'DIP/PIP Ratio:\nMean: {ratio_mean:.3f}\nStd: {ratio_std:.3f}\nMedian: {ratio_median:.3f}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='black', linewidth=1.5)
ax.text(0.95, 0.05, textstr, transform=ax.transAxes, fontsize=16,
        verticalalignment='bottom', horizontalalignment='right', bbox=props)

plt.tight_layout()
plt.savefig(output_dir / 'index_scatter_plot.png', dpi=300, bbox_inches='tight')
print(f"Saved: {output_dir / 'index_scatter_plot.png'}")
plt.close()

# 시각화 2: 비율 히스토그램
print("Generating ratio histogram...")
fig, ax = plt.subplots(figsize=(10, 7))

ax.hist(ratio_data_clean, bins=40, color='#EDB120', alpha=0.7, edgecolor='black', linewidth=0.8)
ax.axvline(ratio_mean, color='red', linestyle='--', linewidth=2.5, label=f'Mean: {ratio_mean:.3f}')
ax.axvline(ratio_median, color='green', linestyle='--', linewidth=2.5, label=f'Median: {ratio_median:.3f}')
ax.axvline(1.0, color='blue', linestyle=':', linewidth=2.5, label='1:1 ratio')

ax.set_xlabel('DIP/PIP Ratio', fontsize=18, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=18, fontweight='bold')
ax.set_title('Index Finger: DIP/PIP Ratio Distribution', fontsize=20, fontweight='bold', pad=20)
ax.legend(fontsize=15, loc='upper right')
ax.grid(True, linestyle=':', alpha=0.3, axis='y')
ax.set_axisbelow(True)

# MATLAB 스타일 박스
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)

plt.tight_layout()
plt.savefig(output_dir / 'index_ratio_distribution.png', dpi=300, bbox_inches='tight')
print(f"Saved: {output_dir / 'index_ratio_distribution.png'}")
plt.close()

print("\n" + "="*60)
print("Index finger analysis complete!")
print(f"Results saved to: {output_dir}")
print("="*60)
