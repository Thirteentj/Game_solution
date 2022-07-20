import math
import numpy as np
import pandas as pd
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="whitegrid")
# 以下两句防止中文显示为窗格
plt.rcParams["font.sans-serif"]=["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

#对计算结果可视化
filepath = r'D:\BaiduSyncdisk\waymo-od\result\xianxia_scoring_result_cal_type_all.xlsx'
df1 = pd.read_excel(filepath,sheet_name='风险偏好')
df2 = pd.read_excel(filepath,sheet_name='敏感程度')

# 设置窗口的大小
f, ax = plt.subplots(figsize=(11, 6))

# 绘制小提琴图
sns.violinplot(data=df1, palette="Set3", bw=.2, cut=1, linewidth=1)

# 设置轴显示的范围
#ax.set(ylim=(-.7, 1.05))
# 去除上下左右的边框（默认该函数会取出右上的边框）
sns.despine(left=True, bottom=True)
# plt.savefig('violin.png',dpi=150)
