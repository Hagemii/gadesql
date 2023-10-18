import os
import re
import string
import subprocess
import nltk.corpus

env_path = '/data/ratsql/torch1.13'
activate_this = os.path.join(env_path, 'bin/activate_this.py')
exec(open(activate_this).read(), {'__file__': activate_this})

STOPWORDS = set(nltk.corpus.stopwords.words('english'))
PUNKS = set(a for a in string.punctuation)


# schema linking, similar to IRNet
# 计算问题中每个词语与schema中每个列和表格的匹配程度
# 将匹配的片段和列/表格标记为
# "CEM"（列的完全匹配）、"TEM"（表格的完全匹配）、"CPM"（列的部分匹配）或"TPM"（表格的部分匹配），
# 最终返回一个字典，包含两个键值对，分别表示自然语言问题中的词语和列/表格的匹配程度。
def compute_schema_linking(question, column, table):
    def partial_match(x_list, y_list):
        x_str = " ".join(x_list)
        y_str = " ".join(y_list)
        if x_str in STOPWORDS or x_str in PUNKS:
            return False
        if re.match(rf"\b{re.escape(x_str)}\b", y_str):
            assert x_str in y_str
            return True
        else:
            return False

    def exact_match(x_list, y_list):
        x_str = " ".join(x_list)
        y_str = " ".join(y_list)
        if x_str == y_str:
            return True
        else:
            return False

    q_col_match = dict()
    q_tab_match = dict()

    col_id2list = dict()
    for col_id, col_item in enumerate(column):
        if col_id == 0:
            continue
        col_id2list[col_id] = col_item

    tab_id2list = dict()
    for tab_id, tab_item in enumerate(table):
        tab_id2list[tab_id] = tab_item

    # 5-gram
    n = 5
    while n > 0:
        for i in range(len(question) - n + 1):
            n_gram_list = question[i:i + n]
            n_gram = " ".join(n_gram_list)
            if len(n_gram.strip()) == 0:
                continue
            # exact match case
            for col_id in col_id2list:
                if exact_match(n_gram_list, col_id2list[col_id]):
                    for q_id in range(i, i + n):
                        q_col_match[f"{q_id},{col_id}"] = "CEM"
            for tab_id in tab_id2list:
                if exact_match(n_gram_list, tab_id2list[tab_id]):
                    for q_id in range(i, i + n):
                        q_tab_match[f"{q_id},{tab_id}"] = "TEM"

            # partial match case
            for col_id in col_id2list:
                if partial_match(n_gram_list, col_id2list[col_id]):
                    for q_id in range(i, i + n):
                        if f"{q_id},{col_id}" not in q_col_match:
                            q_col_match[f"{q_id},{col_id}"] = "CPM"
            for tab_id in tab_id2list:
                if partial_match(n_gram_list, tab_id2list[tab_id]):
                    for q_id in range(i, i + n):
                        if f"{q_id},{tab_id}" not in q_tab_match:
                            q_tab_match[f"{q_id},{tab_id}"] = "TPM"
        n -= 1
    return {"q_col_match": q_col_match, "q_tab_match": q_tab_match}

# 计算每个词语在表格中对应的单元格的匹配程度

def compute_cell_value_linking(tokens, schema):
    def isnumber(word):
        try:
            float(word)
            return True
        except:
            return False

    def db_word_match(word, column, table, db_conn):
        cursor = db_conn.cursor()

        p_str = f"select {column} from {table} where {column} like '{word} %' or {column} like '% {word}' or " \
                f"{column} like '% {word} %' or {column} like '{word}'"
        try:
            cursor.execute(p_str)
            p_res = cursor.fetchall()
            if len(p_res) == 0:
                return False
            else:
                return p_res
        except Exception as e:
            return False

    num_date_match = {}
    cell_match = {}

    for q_id, word in enumerate(tokens):
        if len(word.strip()) == 0:
            continue
        if word in STOPWORDS or word in PUNKS:
            continue

        num_flag = isnumber(word)

        CELL_MATCH_FLAG = "CELLMATCH"

        for col_id, column in enumerate(schema.columns):
            if col_id == 0:
                assert column.orig_name == "*"
                continue

            # word is number 
            if num_flag:
                if column.type in ["number", "time"]:  # TODO fine-grained date
                    num_date_match[f"{q_id},{col_id}"] = column.type.upper()
            else:
                ret = db_word_match(word, column.orig_name, column.table.orig_name, schema.connection)
                if ret:
                    # print(word, ret)
                    cell_match[f"{q_id},{col_id}"] = CELL_MATCH_FLAG

    cv_link = {"num_date_match": num_date_match, "cell_match": cell_match}
    return cv_link

def generate_question_syntax_tree(question):
    def get_sdp(question):
        cmd = ['python', '/data/ratsql/rat-sql/sdp_tree/sdp.py', question]
        output_bytes = subprocess.check_output(cmd)
        output_str = output_bytes.decode('utf-8').strip()
        output_dict = eval(output_str)
        return output_dict
    
    doc = get_sdp(question)
    tree_link = {"sdp_tree": doc}
    return tree_link