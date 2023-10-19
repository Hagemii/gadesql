# import os
# import subprocess
# import sys

# def get_sdp(input_string):
#     # 设置要使用的虚拟环境路径
#     env_path = '/data/ratsql/torch1.13'

#     # 激活虚拟环境
#     activate_this = os.path.join(env_path, 'bin/activate_this.py')
#     exec(open(activate_this).read(), {'__file__': activate_this})

#     import hanlp
#     HanLP = hanlp.load(hanlp.pretrained.mtl.UD_ONTONOTES_TOK_POS_LEM_FEA_NER_SRL_DEP_SDP_CON_MMINILMV2L12)
#     test = input_string
#     doc = HanLP(test, tasks='sdp')
#     q_match = dict()
#     return doc
#     for i, item in enumerate(doc['sdp/dm']):
#         if len(item) !=0:
#             for elem in item:
#                 j, _ = elem
#                 if(j < i):
#                     q_match[f"{i},{j}"] = "FOR"
#                 if(j > i):
#                     q_match[f"{i},{j}"] = "BAC"
#     return q_match

# if __name__ == '__main__':
#     args = sys.argv[1]
#     doc = get_sdp(args)
#     print(doc)
