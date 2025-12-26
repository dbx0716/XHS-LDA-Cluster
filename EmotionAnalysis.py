import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gensim.corpora import Dictionary
from gensim.models import LdaModel, CoherenceModel
from sklearn.model_selection import train_test_split
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
from wordcloud import WordCloud
import seaborn as sns
import matplotlib.colors as mcolors
from collections import defaultdict
import os

os.environ['PYTHONIOENCODING'] = 'utf-8'
# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


# ---------------------- 步骤1：读取并合并两个文件的分词结果 ----------------------
def load_and_merge_data():
    # 读取笔记数据
    notes_df = pd.read_excel('cleaned_notes.xlsx', sheet_name='Sheet1')
    # 读取评论数据
    comments_df = pd.read_excel('cleaned_comments.xlsx', sheet_name='Sheet1')

    # 提取并合并分词结果
    notes_words = notes_df['内容分词结果'].tolist() + notes_df['标题分词结果'].tolist()
    comments_words = comments_df['分词结果'].tolist()

    # 合并所有分词结果到一个列表
    all_words_list = notes_words + comments_words
    return all_words_list


# ---------------------- 步骤2：解析原始数据为二维列表 ----------------------
def parse_to_gensim_format(string_list):
    """将字符串列表转换为二维词语列表"""
    parsed_documents = []
    for string in string_list:
        # 使用正则表达式提取单引号内的词语（支持中文、字母、数字）
        words = re.findall(r"'([\u4e00-\u9fffa-zA-Z0-9]+)'", string)
        parsed_documents.append(words)
    return parsed_documents


# ---------------------- 步骤3：创建词典和文档-词频矩阵 ----------------------
def prepare_gensim_input(documents):
    """生成gensim所需的词典和文档-词频矩阵"""
    # 创建词典
    dictionary = Dictionary(documents)

    # 过滤低频词（保留至少出现2次的词语）
    dictionary.filter_extremes(no_below=2)

    # 生成文档-词频矩阵（BoW格式）
    corpus = [dictionary.doc2bow(words) for words in documents]

    return dictionary, corpus


# ---------------------- 步骤4：训练LDA模型并计算评估指标 ----------------------
def train_lda_and_evaluate(corpus, dictionary, documents, max_topics=10, test_size=0.2):
    """训练LDA模型并评估最佳主题数（包括困惑度和一致性分数）"""
    # 分割训练集和测试集
    train_corpus, test_corpus = train_test_split(corpus, test_size=test_size, random_state=42)

    perplexity_scores = []
    coherence_scores = []

    for num_topics in range(2, max_topics + 1):
        # 训练LDA模型
        lda_model = LdaModel(
            corpus=train_corpus,  # 使用训练集训练模型
            id2word=dictionary,
            num_topics=num_topics,
            random_state=42,
            passes=10,
            chunksize=100
        )

        # 计算困惑度（值越低越好）
        perplexity = lda_model.log_perplexity(test_corpus)
        perplexity_scores.append(perplexity)

        # 计算一致性分数（c_v指标，值越高越好）
        coherence_model = CoherenceModel(
            model=lda_model,
            texts=documents,
            dictionary=dictionary,
            coherence='c_v'
        )
        coherence_score = coherence_model.get_coherence()
        coherence_scores.append(coherence_score)

        print(f"主题数 {num_topics} | 困惑度: {perplexity:.4f} | 一致性分数: {coherence_score:.4f}")

    # 绘制评估指标曲线
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 困惑度曲线（左Y轴）
    color = 'tab:red'
    ax1.set_xlabel('主题数')
    ax1.set_ylabel('困惑度 (log)', color=color)
    ax1.plot(range(2, max_topics + 1), perplexity_scores, 'o-', color=color, label='困惑度')
    ax1.tick_params(axis='y', labelcolor=color)

    # 一致性分数曲线（右Y轴）
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('一致性分数 (c_v)', color=color)
    ax2.plot(range(2, max_topics + 1), coherence_scores, 'o-', color=color, label='一致性分数')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('主题数 vs 困惑度和一致性分数')
    fig.tight_layout()
    # 保存图片
    plt.savefig('topic_evaluation.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 返回最佳主题数（基于一致性分数，通常更可靠）
    best_num_topics = np.argmax(coherence_scores) + 2
    print(f"根据一致性分数，最佳主题数为: {best_num_topics}")

    # 使用最佳主题数重新训练模型
    print(f"使用最佳主题数 {best_num_topics} 重新训练模型...")
    best_model = LdaModel(
        corpus=corpus,  # 使用完整语料库训练最终模型
        id2word=dictionary,
        num_topics=best_num_topics,
        random_state=42,
        passes=20,  # 增加迭代次数提高稳定性
        chunksize=100
    )

    return best_num_topics, best_model


# ---------------------- 步骤5：输出主题关键词 ----------------------
def print_topics(lda_model, dictionary, num_words=10):
    """打印每个主题的关键词"""
    print("\n各主题的关键词分布:")
    for topic_idx, topic in enumerate(lda_model.print_topics(num_words=num_words)):
        print(f"主题 {topic_idx + 1}: {topic[1]}")


# ---------------------- 步骤6：可视化LDA主题模型 ----------------------
def visualize_lda(lda_model, corpus, dictionary, documents):
    """可视化LDA主题模型，包括pyLDAvis交互图、关键词云图和主题热力图"""
    num_topics = lda_model.num_topics

    # 1. 使用pyLDAvis可视化主题分布
    print("\n正在生成pyLDAvis可视化...")
    vis_data = gensimvis.prepare(lda_model, corpus, dictionary)
    pyLDAvis.save_html(vis_data, f'lda_visualization_{num_topics}topics.html')
    print(f"已生成LDA主题可视化文件: lda_visualization_{num_topics}topics.html")

    # 2. 生成主题关键词云图
    print("正在生成关键词云图...")
    cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # 主题颜色

    cloud = WordCloud(
        font_path='SimHei.ttf',  # 确保中文显示正常
        background_color='white',
        width=2500,
        height=1800,
        max_words=15,  # 每个主题显示15个关键词
        colormap='tab10',
        prefer_horizontal=1.0
    )

    topics = lda_model.show_topics(formatted=False)

    # 计算需要的行数和列数
    n_cols = min(3, num_topics)  # 最多3列
    n_rows = (num_topics + n_cols - 1) // n_cols  # 向上取整

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), sharex=True, sharey=True)
    axes = axes.flatten() if num_topics > 1 else [axes]

    for i, ax in enumerate(axes):
        if i < num_topics:
            topic_words = dict(topics[i][1])
            cloud.generate_from_frequencies(topic_words, max_font_size=300)
            ax.imshow(cloud)
            ax.set_title(f'主题 {i + 1}', fontdict=dict(size=16))
            ax.axis('off')
        else:
            ax.axis('off')  # 隐藏空白子图

    plt.tight_layout()
    plt.savefig(f'topic_wordclouds_{num_topics}topics.png', bbox_inches='tight')
    plt.close()
    print(f"已生成主题关键词云图: topic_wordclouds_{num_topics}topics.png")

    # 3. 生成主题-词语热力图
    print("正在生成主题热力图...")
    topic_word_matrix = np.zeros((num_topics, len(dictionary)))

    for topic_id in range(num_topics):
        for word_id, prob in lda_model.get_topic_terms(topic_id, topn=len(dictionary)):
            topic_word_matrix[topic_id, word_id] = prob

    # 获取每个主题的前10个关键词
    top_words_idx = np.argsort(-topic_word_matrix, axis=1)[:, :10]
    top_words = [[dictionary[id] for id in topic] for topic in top_words_idx]

    # 创建一个只包含这些关键词的新矩阵
    selected_words = list(set([word for topic in top_words for word in topic]))
    word_indices = [dictionary.token2id[word] for word in selected_words]
    heatmap_data = topic_word_matrix[:, word_indices]

    # 转置矩阵以便词语在y轴，主题在x轴
    heatmap_data = heatmap_data.T

    plt.figure(figsize=(5 + num_topics, 8))  # 根据主题数量调整宽度
    sns.heatmap(heatmap_data, annot=False, cmap='YlGnBu',
                xticklabels=[f"主题 {i + 1}" for i in range(num_topics)],
                yticklabels=selected_words)
    plt.title(f'主题-关键词热力图 ({num_topics}个主题)')
    plt.tight_layout()
    plt.savefig(f'topic_heatmap_{num_topics}topics.png', bbox_inches='tight')
    plt.close()
    print(f"已生成主题热力图: topic_heatmap_{num_topics}topics.png")


# ---------------------- 主函数 ----------------------
if __name__ == "__main__":
    # 原始数据
    print("正在加载数据...")
    raw_data = load_and_merge_data()

    # 解析数据
    print("正在解析数据...")
    documents = parse_to_gensim_format(raw_data)

    # 准备输入
    print("正在准备词典和语料库...")
    dictionary, corpus = prepare_gensim_input(documents)

    # 训练模型并评估（使用最佳主题数）
    print("正在训练LDA模型并寻找最佳主题数...")
    best_num_topics, best_model = train_lda_and_evaluate(corpus, dictionary, documents, max_topics=10)

    # 输出主题关键词
    print_topics(best_model, dictionary, num_words=10)

    # 可视化（使用最佳主题数的模型）
    print("\n开始生成可视化...")
    visualize_lda(best_model, corpus, dictionary, documents)

    print("\n全部完成！请查看生成的可视化文件：")
    print(f"- LDA主题交互图: lda_visualization_{best_num_topics}topics.html")
    print(f"- 关键词云图: topic_wordclouds_{best_num_topics}topics.png")
    print(f"- 主题热力图: topic_heatmap_{best_num_topics}topics.png")