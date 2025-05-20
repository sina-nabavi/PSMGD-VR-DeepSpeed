
# 导入第三方模块
import numpy as np
import matplotlib.pyplot as plt

# 中文和负号的正常显示
#plt.rcParams['font.sans-serif'] = 'Microsoft YaHei'
#plt.rcParams['axes.unicode_minus'] = False

# 使用ggplot的绘图风格
plt.style.use('ggplot')
#aaa=np.transpose(result)
#data=aaa
# 构造数据
start=1
row=0
column=0


import os
import numpy as np
import re

log_dir = "./PSMGDVR/baselines/radar/"  # Replace with your actual log folder path
result_dict = {}

for filename in os.listdir(log_dir):
    if filename.startswith("output_weighting") and filename.endswith(".txt"):
        method_name = filename.split("_")[-1].split('.')[0]
        with open(os.path.join(log_dir, filename), 'r') as f:
            content = f.read()

            # Fix regex patterns to handle optional spaces
            seg_match = re.search(r"segmentation:\s*\[([\d\.eE+-]+)", content)
            depth_match = re.search(r"depth:\s*\[([\d\.eE+-]+)", content)
            normal_match = re.search(r"normal:\s*\[.*?,.*?,.*?,.*?,\s*([\d\.eE+-]+)\]", content)

            if seg_match and depth_match and normal_match:
                seg_val = round(float(seg_match.group(1)), 2)
                depth_val = round(1 - float(depth_match.group(1)), 2)
                normal_val = round(float(normal_match.group(1)), 2)
                values = [seg_val, depth_val, normal_val, seg_val]
                result_dict[f"values_{method_name}"] = values


feature = ['task 1','task 2','task 3']

N = 3

# 设置雷达图的角度，用于平分切开一个圆面
angles=np.linspace(0, 2*np.pi, N, endpoint=False)



angles=np.concatenate((angles,[angles[0]]))

# 绘图
fig=plt.figure()
ax = fig.add_subplot(111, polar=True)

line_styles = [
    'b-', 'g-', 'r-', 'c-', 'm-', 'y-', 'k-', 'b--', 'g--', 'r--',
    'c--', 'm--', 'y--', 'k--', 'b-.', 'g-.', 'r-.', 'c-.', 'm-.',
    'y-.', 'k-.', 'b:', 'g:', 'r:', 'c:'
]

for i, (method_name, values) in enumerate(result_dict.items()):
    style = line_styles[i % len(line_styles)]
    ax.plot(angles, values, style, linewidth=2, label=method_name)
    ax.fill(angles, values, alpha=0.25)

# 添加每个特征的标签
ax.set_thetagrids(angles[0:3] * 180/np.pi, feature,fontsize=15)
# 设置雷达图的范围
#ax.set_ylim(0,5)
# 添加标题
plt.title('NYU2_100epochs')

# 添加网格线
ax.grid(True)
# 设置图例
plt.legend(loc = 'lower left',fontsize=10, borderaxespad=-9)
# 显示图形
plt.savefig('radar-map-nyu2.pdf',bbox_inches='tight')