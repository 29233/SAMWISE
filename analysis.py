import re


def matches_template(template, sentence):
    """
    检查句子是否符合给定的模板

    参数:
    template (str): 模板字符串，包含方括号占位符（如 [object.1]）
    sentence (str): 待检测的句子

    返回:
    bool: 是否匹配模板
    """
    # 将模板中的占位符替换为正则表达式通配符 (.*?)
    # 使用非贪婪匹配以避免过度匹配
    pattern = re.sub(r'\[(.*?)\]', r'(.*?)', template)

    # 添加字符串起始和结束锚点，确保完全匹配
    pattern = f'^{pattern}$'

    # 检查句子是否匹配模板
    return re.match(pattern, sentence) is not None


def find_matching_sentences(sentences, template):
    """
    从句子列表中找出符合模板的句子

    参数:
    sentences (list): 字符串列表，包含待检测的句子
    template (str): 模板字符串

    返回:
    list: 符合模板的句子列表
    """
    return [s for s in sentences if matches_template(template, s)]


# 使用示例
if __name__ == "__main__":
    # 示例模板
    template = "The [object.1] appearing [duration] than [a_an] [object.2]."

    # 待检测的句子列表
    sentences = [
        "The car appearing faster than a bike.",
        "The car appearing faster than a bike",
        "The car appearing faster than a bike. This is a test.",
        "The car appearing faster than a bike. ",
        "The car appearing faster than an airplane.",
        "The cat appearing slower than a mouse.",
        "The car appearing faster than a bike. This is a test.",
        "The car appearing faster than a bike. and more text."
    ]

    # 检测匹配的句子
    matching_sentences = find_matching_sentences(sentences, template)

    # 输出结果
    print("符合模板的句子:")
    for s in matching_sentences:
        print(f"- {s}")