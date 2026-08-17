"""
B站热门视频流量规律分析 - 主脚本
数据管线：三周CSV -> 合并去重 -> 清洗+特征工程 -> 四维分析 -> 逻辑回归
运行：python src/analysis.py
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # 脚本所在目录
DATA_DIR = os.path.join(BASE_DIR, 'data', '')
OUT_DIR = os.path.join(BASE_DIR, 'reports', '')


def merge_dedup():
    """合并三周数据并按 BV号 去重（保留最新日期）"""
    files = sorted(glob.glob(DATA_DIR + 'b站热门视频*.csv'))
    frames = []
    for f in files:
        df = pd.read_csv(f, encoding='utf-8-sig')
        df['数据日期'] = f.split('b站热门视频')[-1].replace('.csv', '')
        frames.append(df)
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.sort_values('数据日期')
    unique = merged.drop_duplicates(subset='BV号', keep='last').reset_index(drop=True)
    unique.to_csv(DATA_DIR + 'b站热门数据_汇总.csv', index=False, encoding='utf-8-sig')
    print('[1] 合并去重完成:', len(unique), '条（原始', len(merged), '条）')
    return unique


def clean_and_feature(df):
    """清洗 + 特征工程"""
    # 类型转换
    df['发布时间'] = pd.to_datetime(df['发布时间'])
    df['数据日期'] = pd.to_datetime(df['数据日期'], format='%Y%m%d')
    num_cols = ['播放量', '点赞数', '评论数', '弹幕数', '收藏数', '投币数', '分享数', '时长秒']
    for col in num_cols:
        df[col] = pd.to_numeric(df[col])

    # 清洗：剔除非正值和超长视频（>10小时）
    before = len(df)
    df = df[df['播放量'] > 0]
    df = df[df['时长秒'] > 0]
    df = df[df['发布时间'].notna()]
    df = df[df['数据日期'].notna()]
    df = df[df['时长秒'] <= 36000]
    print('[2] 清洗完成:', before, '->', len(df), '条（剔除', before - len(df), '条）')

    # 发布天数（精确到小时）：数据日期 - 发布时间
    df['发布小时'] = (df['数据日期'] - df['发布时间']).dt.total_seconds() / 3600
    df['发布天数'] = df['发布小时'] / 24
    # 剔除发布时间晚于采集日的异常样本
    df = df[df['发布小时'] > 0]
    # 剔除不满24小时的新视频：播放累积不足、日均播放被小分母高估、爆款标签不可靠
    df = df[df['发布小时'] >= 24]
    print('[2.5] 剔除不满24h后条数:', len(df))

    # 互动率
    df['点赞率'] = df['点赞数'] / df['播放量']
    df['投币率'] = df['投币数'] / df['播放量']
    df['收藏率'] = df['收藏数'] / df['播放量']
    df['评论率'] = df['评论数'] / df['播放量']
    df['弹幕率'] = df['弹幕数'] / df['播放量']
    df['分享率'] = df['分享数'] / df['播放量']
    df['三连率'] = (df['点赞数'] + df['投币数'] + df['收藏数']) / df['播放量']
    df['币赞比'] = df['投币数'] / df['点赞数']
    df['弹幕密度'] = df['弹幕数'] / df['时长秒']

    # 日均播放量：消除发布天数差异（老视频累积播放更多，不公平）
    df['日均播放'] = df['播放量'] / df['发布天数']

    # 播放分层（按日均播放量分位，避免偏袒老视频）
    q30 = df['日均播放'].quantile(0.30)
    q90 = df['日均播放'].quantile(0.90)
    q99 = df['日均播放'].quantile(0.99)

    def level(p):
        if p >= q99:
            return '超级爆款'
        if p >= q90:
            return '头部爆款'
        if p >= q30:
            return '腰部普通'
        return '尾部低播放'

    df['播放分层'] = df['日均播放'].apply(level)
    df['是否爆款'] = df['播放分层'].isin(['头部爆款', '超级爆款']).astype(int)

    df.to_csv(DATA_DIR + 'b站数据_清洗后.csv', index=False, encoding='utf-8-sig')
    print('[3] 特征工程完成，已保存清洗后数据')
    return df


def describe_analysis(df):
    """模块1：整体描述性统计"""
    p = df['播放量']
    print('\n[模块1] 播放量分布')
    print('均值:', round(p.mean()), '中位数:', round(p.median()),
          '比值:', round(p.mean() / p.median(), 2))
    print('分层占比:', df['播放分层'].value_counts(normalize=True).round(4).to_dict())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(p, bins=50, color='skyblue', edgecolor='black')
    axes[0].set_title('播放量分布（原始）')
    axes[1].hist(np.log10(p + 1), bins=50, color='orange', edgecolor='black')
    axes[1].set_title('播放量分布（取对数）')
    fig.tight_layout()
    fig.savefig(OUT_DIR + '图0_播放量分布.png', dpi=150)
    print('已保存: 图0_播放量分布.png')


def partition_analysis(df):
    """模块2-1：分区维度"""
    valid_parts = df['分区'].value_counts()[df['分区'].value_counts() >= 5].index
    ps = df[df['分区'].isin(valid_parts)]
    stats = ps.groupby('分区').agg(
        视频数=('BV号', 'count'),
        播放中位数=('播放量', 'median'),
        播放均值=('播放量', 'mean'),
        三连率=('三连率', 'mean'),
    ).round(2)
    top_count = ps[ps['播放分层'].isin(['头部爆款', '超级爆款'])].groupby('分区').size()
    stats['爆款数'] = top_count
    stats['爆款率'] = (stats['爆款数'] / stats['视频数']).round(3)
    stats = stats.sort_values('播放中位数', ascending=False)

    print('\n[模块2-1] 分区播放中位数 Top5:')
    print(stats.head(5)[['视频数', '播放中位数', '三连率', '爆款率']].to_string())
    print('爆款率 Top5:')
    print(stats.sort_values('爆款率', ascending=False).head(5)[['视频数', '播放中位数', '爆款率']].to_string())
    return stats


def up_analysis(df):
    """模块2-2：UP主维度"""
    up = df.groupby('UP主')['播放量'].sum().sort_values(ascending=False)
    cum = up.cumsum() / up.sum()
    print('\n[模块2-2] UP主集中度')
    for p in [0.5, 0.8, 0.9]:
        n = (cum <= p).sum()
        print('前%d个UP主（占%d%%）贡献%.0f%%播放' % (n, round(n / len(up) * 100), p * 100))

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(up) + 1), cum, color='darkred')
    plt.axhline(0.5, linestyle='--', color='gray', alpha=0.6)
    plt.title('UP主播放量集中度曲线')
    plt.xlabel('UP主序号（按播放降序）')
    plt.ylabel('累计播放占比')
    plt.tight_layout()
    plt.savefig(OUT_DIR + '图2_UP主集中度.png', dpi=150)
    print('已保存: 图2_UP主集中度.png')


def corr_analysis(df):
    """模块3：相关性与互动质量"""
    print('\n[模块3] 互动率与播放量相关（Spearman）')
    rate_cols = ['点赞率', '评论率', '弹幕率', '收藏率', '投币率', '分享率']
    corr_rates = df[['播放量'] + rate_cols].corr(method='spearman').loc['播放量']
    print(corr_rates.round(3).to_string())
    print('弹幕密度与播放量相关:', round(df['播放量'].corr(df['弹幕密度'], method='spearman'), 3))

    hit = df[df['是否爆款'] == 1]
    normal = df[df['是否爆款'] == 0]
    print('爆款弹幕密度均值:', round(hit['弹幕密度'].mean(), 2),
          '普通弹幕密度均值:', round(normal['弹幕密度'].mean(), 2))


def model_analysis(df):
    """阶段3：逻辑回归爆款预测（满1日数据训练/验证 + 新视频预测演示）"""
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, classification_report

    feats = ['点赞率', '评论率', '弹幕率', '收藏率', '投币率', '分享率',
             '三连率', '币赞比', '弹幕密度']
    top_parts = ['特摄', '小剧场', '搞笑', '影视剪辑', '出行']
    df['高爆款分区'] = df['分区'].isin(top_parts).astype(int)
    feats += ['高爆款分区']

    work = df.replace([np.inf, -np.inf], np.nan).dropna(subset=feats + ['是否爆款'])
    X = work[feats]
    y = work['是否爆款']

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=0.3, random_state=42)

    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_train, y_train)

    print('\n[建模] 逻辑回归特征权重（越大越促进爆款）')
    coef = pd.DataFrame({'特征': feats, '系数': model.coef_[0].round(3)})
    print(coef.sort_values('系数', ascending=False).to_string(index=False))
    print('测试集准确率:', round(accuracy_score(y_test, model.predict(X_test)), 3))

    # ============ 新视频预测演示 ============
    # 用训练好的模型预测"不满24小时刚发布"的视频是否可能成为爆款
    new_path = DATA_DIR + 'b站数据_新视频.csv'
    if os.path.exists(new_path):
        new_df = pd.read_csv(new_path, encoding='utf-8-sig')
        new_df['发布小时'] = (pd.to_datetime(new_df['数据日期']) - pd.to_datetime(new_df['发布时间'])).dt.total_seconds() / 3600
        new_df = new_df[new_df['发布小时'] > 0]
        # 计算新视频的互动特征
        new_df['点赞率'] = new_df['点赞数'] / new_df['播放量']
        new_df['投币率'] = new_df['投币数'] / new_df['播放量']
        new_df['收藏率'] = new_df['收藏数'] / new_df['播放量']
        new_df['评论率'] = new_df['评论数'] / new_df['播放量']
        new_df['弹幕率'] = new_df['弹幕数'] / new_df['播放量']
        new_df['分享率'] = new_df['分享数'] / new_df['播放量']
        new_df['三连率'] = (new_df['点赞数'] + new_df['投币数'] + new_df['收藏数']) / new_df['播放量']
        new_df['币赞比'] = new_df['投币数'] / new_df['点赞数']
        new_df['弹幕密度'] = new_df['弹幕数'] / new_df['时长秒']
        new_df['高爆款分区'] = new_df['分区'].isin(top_parts).astype(int)

        new_clean = new_df.replace([np.inf, -np.inf], np.nan).dropna(subset=feats)
        if len(new_clean) > 0:
            X_new = scaler.transform(new_clean[feats])
            new_clean['预测概率'] = model.predict_proba(X_new)[:, 1]
            top_new = new_clean.nlargest(10, '预测概率')
            print('\n[建模] 新视频（不满24h）爆款潜力预测 Top10:')
            print(top_new[['标题', 'UP主', '分区', '播放量', '预测概率']].round(3).to_string())
            print('注：新视频预测为方向参考，其爆款标签不可靠（播放未充分累积），未做准确率评估')
    return model


if __name__ == '__main__':
    df_raw = merge_dedup()
    df = clean_and_feature(df_raw)
    describe_analysis(df)
    partition_analysis(df)
    up_analysis(df)
    corr_analysis(df)
    model_analysis(df)
    print('\n全部完成。报告见 reports/B站项目分析报告.html')
