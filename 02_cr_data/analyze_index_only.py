import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

# MATLAB 스타일 설정
plt.style.use('default')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.0
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

# 결측치 제거
data_clean = data.dropna()
print(f"Rows after removing NaN: {len(data_clean)}")

pip_data = data_clean[pip_col]
dip_data = data_clean[dip_col]

# 상관계수 계산
print("\n" + "="*60)
print("Index Finger - DIP-PIP Correlation Analysis")
print("="*60)

# Pearson 상관계수 및 p-value
r, p_value = stats.pearsonr(pip_data, dip_data)

# DIP/PIP 비율 계산 (0으로 나누기 방지)
ratio_data = dip_data / pip_data.replace(0, np.nan)
ratio_data_clean = ratio_data.dropna()

# 통계 계산
pip_mean = pip_data.mean()
pip_std = pip_data.std()
pip_median = pip_data.median()
pip_min = pip_data.min()
pip_max = pip_data.max()

dip_mean = dip_data.mean()
dip_std = dip_data.std()
dip_median = dip_data.median()
dip_min = dip_data.min()
dip_max = dip_data.max()

ratio_mean = ratio_data_clean.mean()
ratio_std = ratio_data_clean.std()
ratio_median = ratio_data_clean.median()

print(f"\nCorrelation coefficient (r): {r:.4f}")
print(f"P-value: {p_value:.2e}")
print(f"Significance: {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'}")
print(f"\nPIP Statistics:")
print(f"  Mean: {pip_mean:.2f}, Std: {pip_std:.2f}")
print(f"  Median: {pip_median:.2f}")
print(f"  Range: [{pip_min:.2f}, {pip_max:.2f}]")
print(f"\nDIP Statistics:")
print(f"  Mean: {dip_mean:.2f}, Std: {dip_std:.2f}")
print(f"  Median: {dip_median:.2f}")
print(f"  Range: [{dip_min:.2f}, {dip_max:.2f}]")
print(f"\nDIP/PIP Ratio:")
print(f"  Mean: {ratio_mean:.3f}, Std: {ratio_std:.3f}")
print(f"  Median: {ratio_median:.3f}")

# 결과를 텍스트 파일로 저장
with open(output_dir / 'index_correlation_results.txt', 'w') as f:
    f.write("Index Finger - DIP-PIP Correlation Analysis Results\n")
    f.write("="*60 + "\n\n")
    f.write(f"Total samples: {len(data_clean)}\n")
    f.write(f"Files analyzed: {', '.join(files)}\n\n")
    f.write(f"Correlation coefficient (r): {r:.4f}\n")
    f.write(f"P-value: {p_value:.2e}\n\n")
    f.write(f"PIP Statistics:\n")
    f.write(f"  Mean: {pip_mean:.2f}, Std: {pip_std:.2f}\n")
    f.write(f"  Median: {pip_median:.2f}\n")
    f.write(f"  Range: [{pip_min:.2f}, {pip_max:.2f}]\n\n")
    f.write(f"DIP Statistics:\n")
    f.write(f"  Mean: {dip_mean:.2f}, Std: {dip_std:.2f}\n")
    f.write(f"  Median: {dip_median:.2f}\n")
    f.write(f"  Range: [{dip_min:.2f}, {dip_max:.2f}]\n\n")
    f.write(f"DIP/PIP Ratio:\n")
    f.write(f"  Mean: {ratio_mean:.3f}, Std: {ratio_std:.3f}\n")
    f.write(f"  Median: {ratio_median:.3f}\n")

print(f"\nResults saved to: {output_dir / 'index_correlation_results.txt'}")

# 시각화: 큰 산점도 (MATLAB 스타일)
print("\nGenerating scatter plot...")
fig, ax = plt.subplots(figsize=(10, 8))

# 산점도
ax.scatter(pip_data, dip_data, alpha=0.6, s=40,
          color='#0072BD', edgecolors='none', marker='o', label='Data points')

# 회귀선
z = np.polyfit(pip_data, dip_data, 1)
p_line = np.poly1d(z)
x_line = np.linspace(pip_data.min(), pip_data.max(), 100)
ax.plot(x_line, p_line(x_line), 'r-', linewidth=2.5, label='Regression line')

# 제목 및 레이블
ax.set_xlabel('Index PIP Flex (degrees)', fontsize=15, fontweight='bold')
ax.set_ylabel('Index DIP Flex (degrees)', fontsize=15, fontweight='bold')
ax.set_title(f'Index Finger: DIP vs PIP Flexion Angle\nr = {r:.4f}, p = {p_value:.2e}',
            fontsize=16, fontweight='bold', pad=20)
ax.grid(True, linestyle=':', alpha=0.3)
ax.set_axisbelow(True)
ax.legend(fontsize=12, loc='upper left')

# MATLAB 스타일 박스
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)

# DIP/PIP 비율 정보를 그래프에 추가 (우측 하단)
textstr = f'DIP/PIP Ratio:\nMean: {ratio_mean:.3f}\nStd: {ratio_std:.3f}\nMedian: {ratio_median:.3f}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='black', linewidth=1.5)
ax.text(0.95, 0.05, textstr, transform=ax.transAxes, fontsize=13,
        verticalalignment='bottom', horizontalalignment='right', bbox=props)

plt.tight_layout()
plt.savefig(output_dir / 'index_scatter_plot.png', dpi=300, bbox_inches='tight')
print(f"Saved: {output_dir / 'index_scatter_plot.png'}")
plt.close()

# 시각화 2: 히스토그램 (PIP, DIP 분포)
print("Generating distribution histograms...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# PIP 히스토그램
axes[0].hist(pip_data, bins=30, color='#0072BD', alpha=0.7, edgecolor='black', linewidth=0.8)
axes[0].axvline(pip_mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {pip_mean:.2f}')
axes[0].axvline(pip_median, color='green', linestyle='--', linewidth=2, label=f'Median: {pip_median:.2f}')
axes[0].set_xlabel('PIP Flex (degrees)', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Frequency', fontsize=13, fontweight='bold')
axes[0].set_title('Index PIP Flexion Distribution', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, linestyle=':', alpha=0.3, axis='y')
axes[0].set_axisbelow(True)
axes[0].spines['top'].set_visible(True)
axes[0].spines['right'].set_visible(True)

# DIP 히스토그램
axes[1].hist(dip_data, bins=30, color='#D95319', alpha=0.7, edgecolor='black', linewidth=0.8)
axes[1].axvline(dip_mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {dip_mean:.2f}')
axes[1].axvline(dip_median, color='green', linestyle='--', linewidth=2, label=f'Median: {dip_median:.2f}')
axes[1].set_xlabel('DIP Flex (degrees)', fontsize=13, fontweight='bold')
axes[1].set_ylabel('Frequency', fontsize=13, fontweight='bold')
axes[1].set_title('Index DIP Flexion Distribution', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, linestyle=':', alpha=0.3, axis='y')
axes[1].set_axisbelow(True)
axes[1].spines['top'].set_visible(True)
axes[1].spines['right'].set_visible(True)

plt.suptitle('Index Finger Joint Angle Distributions', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(output_dir / 'index_distributions.png', dpi=300, bbox_inches='tight')
print(f"Saved: {output_dir / 'index_distributions.png'}")
plt.close()

# 시각화 3: 비율 히스토그램
print("Generating ratio histogram...")
fig, ax = plt.subplots(figsize=(10, 6))

ax.hist(ratio_data_clean, bins=40, color='#EDB120', alpha=0.7, edgecolor='black', linewidth=0.8)
ax.axvline(ratio_mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {ratio_mean:.3f}')
ax.axvline(ratio_median, color='green', linestyle='--', linewidth=2, label=f'Median: {ratio_median:.3f}')
ax.axvline(1.0, color='blue', linestyle=':', linewidth=2, label='1:1 ratio')
ax.set_xlabel('DIP/PIP Ratio', fontsize=13, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=13, fontweight='bold')
ax.set_title('Index Finger: DIP/PIP Ratio Distribution', fontsize=15, fontweight='bold')
ax.legend(fontsize=12)
ax.grid(True, linestyle=':', alpha=0.3, axis='y')
ax.set_axisbelow(True)
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)

plt.tight_layout()
plt.savefig(output_dir / 'index_ratio_distribution.png', dpi=300, bbox_inches='tight')
print(f"Saved: {output_dir / 'index_ratio_distribution.png'}")
plt.close()

# 통합 시각화
print("Generating combined visualization...")
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.35)

# 상단: 큰 산점도
ax1 = fig.add_subplot(gs[0, :])
ax1.scatter(pip_data, dip_data, alpha=0.6, s=35, color='#0072BD', edgecolors='none', marker='o')
z = np.polyfit(pip_data, dip_data, 1)
p_line = np.poly1d(z)
x_line = np.linspace(pip_data.min(), pip_data.max(), 100)
ax1.plot(x_line, p_line(x_line), 'r-', linewidth=2)
ax1.set_xlabel('Index PIP Flex (degrees)', fontsize=13, fontweight='bold')
ax1.set_ylabel('Index DIP Flex (degrees)', fontsize=13, fontweight='bold')
ax1.set_title(f'DIP vs PIP Flexion Angle\nr = {r:.4f}, p = {p:.2e}', fontsize=14, fontweight='bold')
ax1.grid(True, linestyle=':', alpha=0.3)
ax1.set_axisbelow(True)
ax1.spines['top'].set_visible(True)
ax1.spines['right'].set_visible(True)

textstr = f'DIP/PIP Ratio: {ratio_mean:.3f} ± {ratio_std:.3f}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.7, edgecolor='black', linewidth=1.2)
ax1.text(0.95, 0.05, textstr, transform=ax1.transAxes, fontsize=12,
        verticalalignment='bottom', horizontalalignment='right', bbox=props)

# 중간 좌측: PIP 분포
ax2 = fig.add_subplot(gs[1, 0])
ax2.hist(pip_data, bins=30, color='#0072BD', alpha=0.7, edgecolor='black', linewidth=0.6)
ax2.axvline(pip_mean, color='red', linestyle='--', linewidth=1.5)
ax2.set_xlabel('PIP Flex (degrees)', fontsize=11)
ax2.set_ylabel('Frequency', fontsize=11)
ax2.set_title(f'PIP Distribution\nMean: {pip_mean:.2f}°', fontsize=12, fontweight='bold')
ax2.grid(True, linestyle=':', alpha=0.3, axis='y')
ax2.set_axisbelow(True)
ax2.spines['top'].set_visible(True)
ax2.spines['right'].set_visible(True)

# 중간 우측: DIP 분포
ax3 = fig.add_subplot(gs[1, 1])
ax3.hist(dip_data, bins=30, color='#D95319', alpha=0.7, edgecolor='black', linewidth=0.6)
ax3.axvline(dip_mean, color='red', linestyle='--', linewidth=1.5)
ax3.set_xlabel('DIP Flex (degrees)', fontsize=11)
ax3.set_ylabel('Frequency', fontsize=11)
ax3.set_title(f'DIP Distribution\nMean: {dip_mean:.2f}°', fontsize=12, fontweight='bold')
ax3.grid(True, linestyle=':', alpha=0.3, axis='y')
ax3.set_axisbelow(True)
ax3.spines['top'].set_visible(True)
ax3.spines['right'].set_visible(True)

# 하단 좌측: 비율 분포
ax4 = fig.add_subplot(gs[2, 0])
ax4.hist(ratio_data_clean, bins=40, color='#EDB120', alpha=0.7, edgecolor='black', linewidth=0.6)
ax4.axvline(ratio_mean, color='red', linestyle='--', linewidth=1.5)
ax4.axvline(1.0, color='blue', linestyle=':', linewidth=1.5)
ax4.set_xlabel('DIP/PIP Ratio', fontsize=11)
ax4.set_ylabel('Frequency', fontsize=11)
ax4.set_title(f'Ratio Distribution\nMean: {ratio_mean:.3f}', fontsize=12, fontweight='bold')
ax4.grid(True, linestyle=':', alpha=0.3, axis='y')
ax4.set_axisbelow(True)
ax4.spines['top'].set_visible(True)
ax4.spines['right'].set_visible(True)

# 하단 우측: 통계 요약
ax5 = fig.add_subplot(gs[2, 1])
ax5.axis('off')

stats_text = "Statistical Summary\n" + "="*30 + "\n\n"
stats_text += f"Samples: {len(data_clean)}\n"
stats_text += f"Files: {len(files)}\n\n"
stats_text += f"Correlation:\n"
stats_text += f"  r = {r:.4f}\n"
stats_text += f"  p = {p_value:.2e}\n\n"
stats_text += f"PIP (degrees):\n"
stats_text += f"  Mean: {pip_mean:.2f} ± {pip_std:.2f}\n"
stats_text += f"  Median: {pip_median:.2f}\n"
stats_text += f"  Range: [{pip_min:.2f}, {pip_max:.2f}]\n\n"
stats_text += f"DIP (degrees):\n"
stats_text += f"  Mean: {dip_mean:.2f} ± {dip_std:.2f}\n"
stats_text += f"  Median: {dip_median:.2f}\n"
stats_text += f"  Range: [{dip_min:.2f}, {dip_max:.2f}]\n\n"
stats_text += f"DIP/PIP Ratio:\n"
stats_text += f"  Mean: {ratio_mean:.3f} ± {ratio_std:.3f}\n"
stats_text += f"  Median: {ratio_median:.3f}"

ax5.text(0.1, 0.95, stats_text, transform=ax5.transAxes,
         fontsize=10, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='#F0F0F0', alpha=0.6,
                  edgecolor='black', linewidth=1))

fig.suptitle('Index Finger: Complete DIP-PIP Analysis',
            fontsize=16, fontweight='bold')

plt.savefig(output_dir / 'index_complete_analysis.png', dpi=300, bbox_inches='tight')
print(f"Saved: {output_dir / 'index_complete_analysis.png'}")
plt.close()

print("\n" + "="*60)
print("Index finger analysis complete!")
print(f"All results saved to: {output_dir}")
print("="*60)
