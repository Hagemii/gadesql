
import subprocess
import os
# import hanlp
text = "who bought a red apple from the grocery store"
# env_path = '/data/ratsql/torch1.13'
# activate_this = os.path.join(env_path, 'bin/activate_this.py')
# exec(open(activate_this).read(), {'__file__': activate_this})

# HanLP = hanlp.load(hanlp.pretrained.mtl.UD_ONTONOTES_TOK_POS_LEM_FEA_NER_SRL_DEP_SDP_CON_MMINILMV2L12) 
# q_match = dict()
# doc = HanLP(text, tasks='sdp')
# for i, item in enumerate(doc['sdp/dm']):
#     if len(item) !=0:
#         for elem in item:
#             j, _ = elem
#             if(j < i):
#                 q_match[f"{i},{j}"] = "FOR"
#             if(j > i):
#                 q_match[f"{i},{j}"] = "BAC"

# print(q_match)

cmd = ['python', '/data/ratsql/rat-sql/sdp_tree/sdp.py', text]
output_bytes = subprocess.check_output(cmd)
output_str = output_bytes.decode('utf-8').strip()
output_dict = eval(output_str)
print(output_bytes)

# import torch
# print(torch.__version__)

# import subprocess

# text = "Give the average number of working horses on farms with more than 5000 total horses"
# output_bytes = subprocess.check_output(['source ','/data/ratsql/torch1.13','&&','python', '/data/ratsql/rat-sql/spd_test.py', text])


# print(output_bytes)

# import torch
# print(torch.__version__)





# from sdp_tree import sdp

# result = sdp.get_sdp("Give the average number of working horses on farms with more than 5000 total horses")

# print(result)

# import torch
# print(torch.__version__)