import pandas as pd
import requests
import time
import os
from datetime import datetime

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0',
    'Referer': 'https://www.bilibili.com/',
}

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')

def collect_one_day():
    all_data = []
    seen_bv = set()
    for pn in range(1, 14):
        url = f'https://api.bilibili.com/x/web-interface/popular?pn={pn}&ps=50'
        try:
            r = requests.get(url, headers=headers, timeout=10)
            data = r.json()['data']['list']
            for v in data:
                bv = v['bvid']
                if bv not in seen_bv:
                    seen_bv.add(bv)
                    all_data.append({
                        '标题': v['title'],
                        'UP主': v['owner']['name'],
                        '分区': v['tname'],
                        '播放量': v['stat']['view'],
                        '点赞数': v['stat']['like'],
                        '评论数': v['stat']['reply'],
                        '弹幕数': v['stat']['danmaku'],
                        '收藏数': v['stat']['favorite'],
                        '投币数': v['stat']['coin'],
                        '分享数': v['stat']['share'],
                        '时长秒': v.get('duration', 0),
                        '发布时间': v['pubdate'] if isinstance(v.get('pubdate'), str) else str(v.get('pubdate', '')),
                        'BV号': bv,
                    })
        except Exception as e:
            print(f'第{pn}页失败: {e}')
        time.sleep(2)
    return all_data

if __name__ == '__main__':
    data = collect_one_day()
    date_str = datetime.now().strftime('%Y%m%d')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filepath = os.path.join(OUTPUT_DIR, f'b站热门视频{date_str}.csv')
    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False, encoding='utf-8-sig')
    print(f'采集完成: {len(df)} 条')
    print(f'已保存: {filepath}')
