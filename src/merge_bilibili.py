import pandas as pd
import glob
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', '')

# 读取文件夹里所有 B站数据文件
files = glob.glob(DATA_DIR + 'b站热门视频*.csv')
files.sort()  # 按文件名排序（日期从旧到新）

print('找到的文件:')
for f in files:
    print(' -', f)

# 全部读进来，合并
frames = []
for f in files:
    df = pd.read_csv(f, encoding='utf-8-sig')
    # 从文件名提取日期，比如 20260726
    df['数据日期'] = f.split('b站热门视频')[-1].replace('.csv', '')
    frames.append(df)

merged = pd.concat(frames, ignore_index=True)
print('\n合并前总数:', len(merged))

# 先按日期排序，确保最新日期排在最后
merged = merged.sort_values('数据日期')
# 清洗
print('空值数量:')
print(merged.isnull().sum().to_string())
print('播放量为 0:', (merged['播放量'] == 0).sum())
print('时长小于 10 秒:', (merged['时长秒'] < 10).sum())

# 观察异常值：超过 2 小时(7200秒)的视频
long_videos = merged[merged['时长秒'] > 7200]
print('时长超过 2 小时的视频数:', len(long_videos))
print(long_videos[['标题', 'UP主', '分区', '时长秒']].head(10).to_string())

# 按 BV 号去重，保留最后一个（最新日期的数据）
unique = merged.drop_duplicates(subset='BV号', keep='last').reset_index(drop=True)
print('去重后唯一视频:', len(unique))
print('覆盖日期:', unique['数据日期'].unique().tolist())
print('保留的数据日期分布:')
print(unique['数据日期'].value_counts().to_string())

# 保存汇总
unique.to_csv(DATA_DIR + 'b站热门数据_汇总.csv', index=False, encoding='utf-8-sig')
print('\n已保存: b站热门数据_汇总.csv')
