import pandas as pd
import os
import jieba
import re
 
jieba.load_userdict('custom_dict.txt')

# 加载停用词表
with open('hit_stopwords.txt', 'r', encoding='utf-8') as f:
    stopwords = set([line.strip() for line in f.readlines()])

# 自定义停用词
custom_filter_words = {'手机','评论','测评','doge','偷笑','觉得','一下','一点',
                       '左右','客观','系列','真的','害羞','可能','已经','请问','现在','看看',
                       '准备', '知道', '没有', '感觉','不是', '出来', '存在', '用到','之前', '最好', '不够', '考虑','汗颜', '石化', '飞吻',
                       '微笑', '反正', '朋友','月份', '每天', '出来', '唯一', '尤其',
                       # '一年','后面','非常','确实','两个','一个','系统',
                       # '其实','红米','基本上'
                       }

# 读取专业词典中的词汇，创建专业词集合
with open('custom_dict.txt', 'r', encoding='utf-8') as f:
    # 假设文件中每行格式为：词语 [词频] [词性]
    professional_words = set(line.split()[0] for line in f if line.strip())


def tokenize_and_filter(text):
    if not isinstance(text, str):
        return []

    # 预处理文本
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', '', text)

    words = jieba.cut(text)
    filtered = []

    for word in words:
        word = word.strip()
        if len(word) <= 1:  # 过滤单字
            continue
        if word in professional_words:  # 专业词汇直接保留（跳过停用词检查）
            filtered.append(word)
            continue
        if word in stopwords or word in custom_filter_words:  # 普通词汇检查停用词
            continue
        if re.match(r'^[a-zA-Z0-9]+$', word):  # 过滤纯字母数字
            continue
        filtered.append(word)

    return filtered

# 笔记数据预处理
folder_path = '笔记数据'
note_files = [f for f in os.listdir(folder_path) if f.endswith('.xlsx')]
# 合并所有笔记文件
all_notes = pd.DataFrame()
for file in note_files:
    file_path = os.path.join(folder_path, file)
    df = pd.read_excel(file_path)
    all_notes = pd.concat([all_notes, df], ignore_index=True)
# 去重
all_notes.drop_duplicates(inplace=True)
# 输出总条数
print("合并后笔记总数为：", len(all_notes))
all_notes['标题分词结果'] = all_notes['标题'].apply(tokenize_and_filter)
all_notes['内容分词结果'] = all_notes['内容'].apply(tokenize_and_filter)
# 输出总条数
print(f"完成分词处理，共处理 {len(all_notes)} 条笔记")
# 保存为 Excel 文件
all_notes.to_excel('cleaned_notes.xlsx', index=False)
print("已保存清洗后的笔记数据为 cleaned_notes.xlsx")

# 笔记评论预处理
folder_path = '笔记评论'
comment_files = [f for f in os.listdir(folder_path) if f.endswith('.xlsx')]
# 合并所有评论文件
all_comments = pd.DataFrame()
for file in comment_files:
    file_path = os.path.join(folder_path, file)
    df = pd.read_excel(file_path)
    all_comments = pd.concat([all_comments, df], ignore_index=True)
# 去重
all_comments.drop_duplicates(inplace=True)
# 输出总条数
print("合并后评论总数为：", len(all_comments))
# 删除为空或纯空格的评论
before = len(all_comments)
all_comments.dropna(subset=['内容'], inplace=True)  # 替换为你的评论列名
all_comments['内容'] = all_comments['内容'].astype(str).str.strip()
all_comments = all_comments[all_comments['内容'] != '']
print(f"删除空评论后剩余：{len(all_comments)} 条，删除了 {before - len(all_comments)} 条")
# 删除长度小于等于5的评论
before = len(all_comments)
all_comments = all_comments[all_comments['内容'].str.len() > 5]
print(f"删除短评论后剩余：{len(all_comments)} 条，删除了 {before - len(all_comments)} 条")
# 对评论列进行处理，生成一个新列 '分词结果'
all_comments['分词结果'] = all_comments['内容'].apply(tokenize_and_filter)
print(f"完成分词处理，共处理 {len(all_comments)} 条评论")
# 保存为 Excel 文件
all_comments.to_excel('cleaned_comments.xlsx', index=False)
print("已保存清洗后的评论数据为 cleaned_comments.xlsx")