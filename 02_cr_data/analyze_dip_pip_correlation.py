import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

# 설정
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 300

# 데이터 경로
data_dir = Path('.')
output_dir = data_dir / 'correlation_results'
output_dir.mkdir(exist_ok=True)

# 데이터 로딩
print("Loading data...")
files = ['sj_1.csv', 'sj_2.csv', 'tk.csv']
dfs = []

for file in files:
    df = pd.read_csv(data_dir / file)
    dfs.append(df)
    print(f"  {file}: {len(df)} rows")

# 데이터 합치기
combined_df = pd.concat(dfs, ignore_index=True)
print(f"\nTotal rows: {len(combined_df)}")

# 필요한 컬럼 추출
fingers = ['Index', 'Middle', 'Ring']
columns_needed = []
for finger in fingers:
    columns_needed.extend([f'{finger}_PIP_Flex', f'{finger}_DIP_Flex'])

data = combined_df[columns_needed].copy()

# 결측치 제거
data_clean = data.dropna()
print(f"Rows after removing NaN: {len(data_clean)}")

# 상관계수 계산
print("\n" + "="*60)
print("Correlation Analysis Results")
print("="*60)

correlations = {}
for finger in fingers:
    pip_col = f'{finger}_PIP_Flex'
    dip_col = f'{finger}_DIP_Flex'

    pip_data = data_clean[pip_col]
    dip_data = data_clean[dip_col]

    # Pearson 상관계수 및 p-value
    r, p_value = stats.pearsonr(pip_data, dip_data)

    correlations[finger] = {
        'r': r,
        'p_value': p_value,
        'pip_data': pip_data,
        'dip_data': dip_data,
        'pip_mean': pip_data.mean(),
        'pip_std': pip_data.std(),
        'dip_mean': dip_data.mean(),
        'dip_std': dip_data.std()
    }

    print(f"\n{finger} Finger:")
    print(f"  Correlation coefficient (r): {r:.4f}")
    print(f"  P-value: {p_value:.2e}")
    print(f"  Significance: {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'}")
    print(f"  PIP - Mean: {pip_data.mean():.2f}, Std: {pip_data.std():.2f}")
    print(f"  DIP - Mean: {dip_data.mean():.2f}, Std: {dip_data.std():.2f}")

# 결과를 텍스트 파일로 저장
with open(output_dir / 'dip_pip_correlation_results.txt', 'w') as f:
    f.write("DIP-PIP Correlation Analysis Results\n")
    f.write("="*60 + "\n\n")
    f.write(f"Total samples: {len(data_clean)}\n")
    f.write(f"Files analyzed: {', '.join(files)}\n\n")

    for finger in fingers:
        f.write(f"\n{finger} Finger:\n")
        f.write(f"  Correlation coefficient (r): {correlations[finger]['r']:.4f}\n")
        f.write(f"  P-value: {correlations[finger]['p_value']:.2e}\n")
        f.write(f"  PIP - Mean: {correlations[finger]['pip_mean']:.2f}, Std: {correlations[finger]['pip_std']:.2f}\n")
        f.write(f"  DIP - Mean: {correlations[finger]['dip_mean']:.2f}, Std: {correlations[finger]['dip_std']:.2f}\n")

print(f"\nResults saved to: {output_dir / 'dip_pip_correlation_results.txt'}")

# 시각화 1: 산점도 with regression line
print("\nGenerating scatter plots...")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('DIP vs PIP Flexion Angle - Scatter Plots with Regression Lines',
             fontsize=24, fontweight='bold', y=1.02)

for idx, finger in enumerate(fingers):
    ax = axes[idx]
    pip_data = correlations[finger]['pip_data']
    dip_data = correlations[finger]['dip_data']
    r = correlations[finger]['r']
    p = correlations[finger]['p_value']

    # 산점도
    ax.scatter(pip_data, dip_data, alpha=0.3, s=10, color='steelblue')

    # 회귀선
    z = np.polyfit(pip_data, dip_data, 1)
    p_line = np.poly1d(z)
    x_line = np.linspace(pip_data.min(), pip_data.max(), 100)
    ax.plot(x_line, p_line(x_line), "r-", linewidth=2, label=f'Regression line')

    # 제목 및 레이블
    ax.set_xlabel(f'{finger} PIP Flex (degrees)', fontsize=22, fontweight='bold')
    ax.set_ylabel(f'{finger} DIP Flex (degrees)', fontsize=22, fontweight='bold')
    ax.set_title(f'{finger}\nr = {r:.4f}, p = {p:.2e}', fontsize=24, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

plt.tight_layout()
plt.savefig(output_dir / 'dip_pip_scatter_plots.png', dpi=300, bbox_inches='tight')
print(f"Saved: {output_dir / 'dip_pip_scatter_plots.png'}")
plt.close()

# 시각화 2: 상관계수 바 차트
print("Generating correlation bar chart...")
fig, ax = plt.subplots(figsize=(10, 6))

r_values = [correlations[finger]['r'] for finger in fingers]
colors = ['#2E86AB', '#A23B72', '#F18F01']

bars = ax.bar(fingers, r_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

# 각 막대 위에 값 표시
for idx, (bar, r_val) in enumerate(zip(bars, r_values)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{r_val:.4f}',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.set_ylabel('Pearson Correlation Coefficient (r)', fontsize=13, fontweight='bold')
ax.set_xlabel('Finger', fontsize=13, fontweight='bold')
ax.set_title('DIP-PIP Correlation Coefficients by Finger', fontsize=15, fontweight='bold')
ax.set_ylim(0, 1.0)
ax.grid(True, axis='y', alpha=0.3)
ax.axhline(y=0.7, color='red', linestyle='--', alpha=0.5, label='Strong correlation threshold')
ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Moderate correlation threshold')
ax.legend(loc='lower right')

plt.tight_layout()
plt.savefig(output_dir / 'dip_pip_correlation_barchart.png', dpi=300, bbox_inches='tight')
print(f"Saved: {output_dir / 'dip_pip_correlation_barchart.png'}")
plt.close()

# 시각화 3: 통계 정보 테이블
print("Generating statistics table...")
fig, ax = plt.subplots(figsize=(12, 6))
ax.axis('tight')
ax.axis('off')

# 테이블 데이터 준비
table_data = []
headers = ['Finger', 'Correlation (r)', 'P-value', 'PIP Mean±SD', 'DIP Mean±SD', 'Samples']

for finger in fingers:
    corr = correlations[finger]
    row = [
        finger,
        f"{corr['r']:.4f}",
        f"{corr['p_value']:.2e}",
        f"{corr['pip_mean']:.2f}±{corr['pip_std']:.2f}",
        f"{corr['dip_mean']:.2f}±{corr['dip_std']:.2f}",
        f"{len(data_clean)}"
    ]
    table_data.append(row)

# 테이블 생성
table = ax.table(cellText=table_data, colLabels=headers,
                cellLoc='center', loc='center',
                colWidths=[0.12, 0.15, 0.15, 0.18, 0.18, 0.12])

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.5)

# 헤더 스타일
for i in range(len(headers)):
    table[(0, i)].set_facecolor('#2E86AB')
    table[(0, i)].set_text_props(weight='bold', color='white')

# 행 색상 교차
for i in range(1, len(fingers) + 1):
    for j in range(len(headers)):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#E8F4F8')
        else:
            table[(i, j)].set_facecolor('#FFFFFF')

plt.title('DIP-PIP Correlation Statistics Summary',
         fontsize=15, fontweight='bold', pad=20)
plt.savefig(output_dir / 'dip_pip_statistics_table.png', dpi=300, bbox_inches='tight')
print(f"Saved: {output_dir / 'dip_pip_statistics_table.png'}")
plt.close()

# 통합 시각화 (선택사항 - 모든 정보를 하나의 이미지에)
print("Generating combined visualization...")
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

# 상단: 산점도 3개
for idx, finger in enumerate(fingers):
    ax = fig.add_subplot(gs[0, idx])
    pip_data = correlations[finger]['pip_data']
    dip_data = correlations[finger]['dip_data']
    r = correlations[finger]['r']
    p = correlations[finger]['p_value']

    ax.scatter(pip_data, dip_data, alpha=0.3, s=8, color='steelblue')

    z = np.polyfit(pip_data, dip_data, 1)
    p_line = np.poly1d(z)
    x_line = np.linspace(pip_data.min(), pip_data.max(), 100)
    ax.plot(x_line, p_line(x_line), "r-", linewidth=2)

    ax.set_xlabel(f'{finger} PIP (°)', fontsize=10, fontweight='bold')
    ax.set_ylabel(f'{finger} DIP (°)', fontsize=10, fontweight='bold')
    ax.set_title(f'{finger}\nr = {r:.4f}', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)

# 하단 좌측: 상관계수 바 차트
ax_bar = fig.add_subplot(gs[1, :2])
r_values = [correlations[finger]['r'] for finger in fingers]
colors = ['#2E86AB', '#A23B72', '#F18F01']
bars = ax_bar.bar(fingers, r_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

for bar, r_val in zip(bars, r_values):
    height = bar.get_height()
    ax_bar.text(bar.get_x() + bar.get_width()/2., height,
               f'{r_val:.4f}',
               ha='center', va='bottom', fontsize=11, fontweight='bold')

ax_bar.set_ylabel('Correlation (r)', fontsize=11, fontweight='bold')
ax_bar.set_xlabel('Finger', fontsize=11, fontweight='bold')
ax_bar.set_title('Correlation Coefficients', fontsize=12, fontweight='bold')
ax_bar.set_ylim(0, 1.0)
ax_bar.grid(True, axis='y', alpha=0.3)

# 하단 우측: 통계 정보
ax_stats = fig.add_subplot(gs[1, 2])
ax_stats.axis('off')

stats_text = "Statistical Summary\n" + "="*25 + "\n\n"
stats_text += f"Total Samples: {len(data_clean)}\n"
stats_text += f"Files: {len(files)}\n\n"

for finger in fingers:
    corr = correlations[finger]
    stats_text += f"{finger}:\n"
    stats_text += f"  r = {corr['r']:.4f}\n"
    stats_text += f"  p = {corr['p_value']:.2e}\n"
    stats_text += f"  PIP: {corr['pip_mean']:.1f}±{corr['pip_std']:.1f}°\n"
    stats_text += f"  DIP: {corr['dip_mean']:.1f}±{corr['dip_std']:.1f}°\n\n"

ax_stats.text(0.1, 0.95, stats_text, transform=ax_stats.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

fig.suptitle('DIP-PIP Correlation Analysis - Complete Overview',
            fontsize=16, fontweight='bold', y=0.98)

plt.savefig(output_dir / 'dip_pip_complete_analysis.png', dpi=300, bbox_inches='tight')
print(f"Saved: {output_dir / 'dip_pip_complete_analysis.png'}")
plt.close()

print("\n" + "="*60)
print("Analysis complete!")
print(f"All results saved to: {output_dir}")
print("="*60)
