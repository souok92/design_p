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
output_dir = script_dir / 'mcp_correlation_results'
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

# 3개 손가락 설정
fingers = ['Index', 'Middle', 'Ring']
colors = ['#0072BD', '#D95319', '#EDB120']

# MCP 컬럼명
mcp_cols = {
    'Index': 'Index_MCP_Flex',
    'Middle': 'Middle_MCP_Flex',
    'Ring': 'Ring_MCP_Flex'
}

pip_cols = {
    'Index': 'Index_PIP_Flex',
    'Middle': 'Middle_PIP_Flex',
    'Ring': 'Ring_PIP_Flex'
}

dip_cols = {
    'Index': 'Index_DIP_Flex',
    'Middle': 'Middle_DIP_Flex',
    'Ring': 'Ring_DIP_Flex'
}

# 상관계수 계산
correlations_mcp_dip = {}
correlations_mcp_pip = {}

print("\n" + "="*60)
print("MCP-DIP and MCP-PIP Correlation Analysis")
print("="*60)

for finger in fingers:
    mcp_col = mcp_cols[finger]
    pip_col = pip_cols[finger]
    dip_col = dip_cols[finger]

    # MCP-DIP 상관계수
    data_mcp_dip = combined_df[[mcp_col, dip_col]].copy()
    data_mcp_dip_clean = data_mcp_dip.dropna()

    mcp_data = data_mcp_dip_clean[mcp_col]
    dip_data = data_mcp_dip_clean[dip_col]

    r_mcp_dip, p_mcp_dip = stats.pearsonr(mcp_data, dip_data)

    # DIP/MCP 비율
    ratio_mcp_dip = dip_data / mcp_data.replace(0, np.nan)
    ratio_mcp_dip_clean = ratio_mcp_dip.dropna()

    correlations_mcp_dip[finger] = {
        'r': r_mcp_dip,
        'p_value': p_mcp_dip,
        'ratio_mean': ratio_mcp_dip_clean.mean(),
        'ratio_std': ratio_mcp_dip_clean.std(),
        'ratio_median': ratio_mcp_dip_clean.median(),
        'mcp_data': mcp_data,
        'dip_data': dip_data
    }

    # MCP-PIP 상관계수
    data_mcp_pip = combined_df[[mcp_col, pip_col]].copy()
    data_mcp_pip_clean = data_mcp_pip.dropna()

    mcp_data_pip = data_mcp_pip_clean[mcp_col]
    pip_data = data_mcp_pip_clean[pip_col]

    r_mcp_pip, p_mcp_pip = stats.pearsonr(mcp_data_pip, pip_data)

    # PIP/MCP 비율
    ratio_mcp_pip = pip_data / mcp_data_pip.replace(0, np.nan)
    ratio_mcp_pip_clean = ratio_mcp_pip.dropna()

    correlations_mcp_pip[finger] = {
        'r': r_mcp_pip,
        'p_value': p_mcp_pip,
        'ratio_mean': ratio_mcp_pip_clean.mean(),
        'ratio_std': ratio_mcp_pip_clean.std(),
        'ratio_median': ratio_mcp_pip_clean.median(),
        'mcp_data': mcp_data_pip,
        'pip_data': pip_data
    }

    print(f"\n{finger} Finger:")
    print(f"  MCP-DIP: r = {r_mcp_dip:.4f}, p = {p_mcp_dip:.2e}, DIP/MCP = {ratio_mcp_dip_clean.mean():.3f} ± {ratio_mcp_dip_clean.std():.3f}")
    print(f"  MCP-PIP: r = {r_mcp_pip:.4f}, p = {p_mcp_pip:.2e}, PIP/MCP = {ratio_mcp_pip_clean.mean():.3f} ± {ratio_mcp_pip_clean.std():.3f}")

# 시각화 1: MCP-DIP 산점도
print("\nGenerating MCP-DIP scatter plots...")
fig, axes = plt.subplots(1, 3, figsize=(24, 7))

for idx, finger in enumerate(fingers):
    ax = axes[idx]
    data_dict = correlations_mcp_dip[finger]

    mcp_data = data_dict['mcp_data']
    dip_data = data_dict['dip_data']
    r = data_dict['r']
    p_value = data_dict['p_value']
    ratio_mean = data_dict['ratio_mean']
    ratio_std = data_dict['ratio_std']
    ratio_median = data_dict['ratio_median']

    ax.scatter(mcp_data, dip_data, alpha=0.6, s=40,
              color=colors[idx], edgecolors='none', marker='o', label='Data points')

    # 회귀선
    z = np.polyfit(mcp_data, dip_data, 1)
    p_line = np.poly1d(z)
    x_line = np.linspace(mcp_data.min(), mcp_data.max(), 100)
    ax.plot(x_line, p_line(x_line), 'r-', linewidth=2.5, label='Regression line')

    ax.set_xlabel(f'{finger} MCP Flex (degrees)', fontsize=18, fontweight='bold')
    ax.set_ylabel(f'{finger} DIP Flex (degrees)', fontsize=18, fontweight='bold')
    ax.set_title(f'{finger}: DIP vs MCP\nr = {r:.4f}, p = {p_value:.2e}',
                fontsize=20, fontweight='bold', pad=20)
    ax.grid(True, linestyle=':', alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(fontsize=15, loc='upper left')

    # MATLAB 스타일 박스
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)

    # 비율 정보
    textstr = f'DIP/MCP Ratio:\nMean: {ratio_mean:.3f}\nStd: {ratio_std:.3f}\nMedian: {ratio_median:.3f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.text(0.95, 0.05, textstr, transform=ax.transAxes, fontsize=16,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)

plt.tight_layout()
plt.savefig(output_dir / 'mcp_dip_scatter_plots.png', dpi=300, bbox_inches='tight')
print(f"Saved: {output_dir / 'mcp_dip_scatter_plots.png'}")
plt.close()

# 시각화 2: MCP-PIP 산점도
print("Generating MCP-PIP scatter plots...")
fig, axes = plt.subplots(1, 3, figsize=(24, 7))

for idx, finger in enumerate(fingers):
    ax = axes[idx]
    data_dict = correlations_mcp_pip[finger]

    mcp_data = data_dict['mcp_data']
    pip_data = data_dict['pip_data']
    r = data_dict['r']
    p_value = data_dict['p_value']
    ratio_mean = data_dict['ratio_mean']
    ratio_std = data_dict['ratio_std']
    ratio_median = data_dict['ratio_median']

    ax.scatter(mcp_data, pip_data, alpha=0.6, s=40,
              color=colors[idx], edgecolors='none', marker='o', label='Data points')

    # 회귀선
    z = np.polyfit(mcp_data, pip_data, 1)
    p_line = np.poly1d(z)
    x_line = np.linspace(mcp_data.min(), mcp_data.max(), 100)
    ax.plot(x_line, p_line(x_line), 'r-', linewidth=2.5, label='Regression line')

    ax.set_xlabel(f'{finger} MCP Flex (degrees)', fontsize=18, fontweight='bold')
    ax.set_ylabel(f'{finger} PIP Flex (degrees)', fontsize=18, fontweight='bold')
    ax.set_title(f'{finger}: PIP vs MCP\nr = {r:.4f}, p = {p_value:.2e}',
                fontsize=20, fontweight='bold', pad=20)
    ax.grid(True, linestyle=':', alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(fontsize=15, loc='upper left')

    # MATLAB 스타일 박스
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)

    # 비율 정보
    textstr = f'PIP/MCP Ratio:\nMean: {ratio_mean:.3f}\nStd: {ratio_std:.3f}\nMedian: {ratio_median:.3f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.text(0.95, 0.05, textstr, transform=ax.transAxes, fontsize=16,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)

plt.tight_layout()
plt.savefig(output_dir / 'mcp_pip_scatter_plots.png', dpi=300, bbox_inches='tight')
print(f"Saved: {output_dir / 'mcp_pip_scatter_plots.png'}")
plt.close()

# 시각화 3: 상관계수 비교 바 차트
print("Generating correlation comparison bar chart...")
fig, ax = plt.subplots(figsize=(12, 8))

x = np.arange(len(fingers))
width = 0.35

r_values_mcp_dip = [correlations_mcp_dip[f]['r'] for f in fingers]
r_values_mcp_pip = [correlations_mcp_pip[f]['r'] for f in fingers]

bars1 = ax.bar(x - width/2, r_values_mcp_dip, width, label='MCP-DIP',
               color='#0072BD', alpha=0.8, edgecolor='black', linewidth=1.2)
bars2 = ax.bar(x + width/2, r_values_mcp_pip, width, label='MCP-PIP',
               color='#D95319', alpha=0.8, edgecolor='black', linewidth=1.2)

# 값 표시
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=14, fontweight='bold')

ax.set_xlabel('Finger', fontsize=18, fontweight='bold')
ax.set_ylabel('Correlation Coefficient (r)', fontsize=18, fontweight='bold')
ax.set_title('MCP Correlation Comparison: MCP-DIP vs MCP-PIP', fontsize=20, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(fingers, fontsize=16)
ax.legend(fontsize=16, loc='lower right')
ax.grid(True, linestyle=':', alpha=0.3, axis='y')
ax.set_axisbelow(True)
ax.set_ylim([0, 1.0])

# MATLAB 스타일 박스
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)

plt.tight_layout()
plt.savefig(output_dir / 'mcp_correlation_comparison.png', dpi=300, bbox_inches='tight')
print(f"Saved: {output_dir / 'mcp_correlation_comparison.png'}")
plt.close()

# 시각화 4: 비율 비교 바 차트
print("Generating ratio comparison bar chart...")
fig, ax = plt.subplots(figsize=(12, 8))

ratio_means_mcp_dip = [correlations_mcp_dip[f]['ratio_mean'] for f in fingers]
ratio_stds_mcp_dip = [correlations_mcp_dip[f]['ratio_std'] for f in fingers]
ratio_means_mcp_pip = [correlations_mcp_pip[f]['ratio_mean'] for f in fingers]
ratio_stds_mcp_pip = [correlations_mcp_pip[f]['ratio_std'] for f in fingers]

bars1 = ax.bar(x - width/2, ratio_means_mcp_dip, width,
               yerr=ratio_stds_mcp_dip, label='DIP/MCP',
               color='#0072BD', alpha=0.8, edgecolor='black', linewidth=1.2,
               capsize=5, error_kw={'linewidth': 2})
bars2 = ax.bar(x + width/2, ratio_means_mcp_pip, width,
               yerr=ratio_stds_mcp_pip, label='PIP/MCP',
               color='#D95319', alpha=0.8, edgecolor='black', linewidth=1.2,
               capsize=5, error_kw={'linewidth': 2})

# 값 표시
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=14, fontweight='bold')

ax.set_xlabel('Finger', fontsize=18, fontweight='bold')
ax.set_ylabel('Ratio (mean ± std)', fontsize=18, fontweight='bold')
ax.set_title('MCP Joint Angle Ratios', fontsize=20, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(fingers, fontsize=16)
ax.legend(fontsize=16, loc='upper right')
ax.grid(True, linestyle=':', alpha=0.3, axis='y')
ax.set_axisbelow(True)
ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=2, alpha=0.5, label='1:1 ratio')

# MATLAB 스타일 박스
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)

plt.tight_layout()
plt.savefig(output_dir / 'mcp_ratio_comparison.png', dpi=300, bbox_inches='tight')
print(f"Saved: {output_dir / 'mcp_ratio_comparison.png'}")
plt.close()

# 텍스트 결과 저장
print("Saving text results...")
with open(output_dir / 'mcp_correlation_results.txt', 'w', encoding='utf-8') as f:
    f.write("="*60 + "\n")
    f.write("MCP-DIP and MCP-PIP Correlation Analysis Results\n")
    f.write("="*60 + "\n\n")

    f.write("MCP-DIP Correlations:\n")
    f.write("-"*60 + "\n")
    for finger in fingers:
        data = correlations_mcp_dip[finger]
        f.write(f"{finger} Finger:\n")
        f.write(f"  Correlation coefficient (r): {data['r']:.4f}\n")
        f.write(f"  P-value: {data['p_value']:.2e}\n")
        f.write(f"  DIP/MCP Ratio - Mean: {data['ratio_mean']:.3f}, Median: {data['ratio_median']:.3f}, Std: {data['ratio_std']:.3f}\n\n")

    f.write("\nMCP-PIP Correlations:\n")
    f.write("-"*60 + "\n")
    for finger in fingers:
        data = correlations_mcp_pip[finger]
        f.write(f"{finger} Finger:\n")
        f.write(f"  Correlation coefficient (r): {data['r']:.4f}\n")
        f.write(f"  P-value: {data['p_value']:.2e}\n")
        f.write(f"  PIP/MCP Ratio - Mean: {data['ratio_mean']:.3f}, Median: {data['ratio_median']:.3f}, Std: {data['ratio_std']:.3f}\n\n")

print(f"Saved: {output_dir / 'mcp_correlation_results.txt'}")

print("\n" + "="*60)
print("MCP correlation analysis complete!")
print(f"Results saved to: {output_dir}")
print("="*60)
