
import subprocess
import os
import torch

text = "How many heads of the departments are older than 56 ?"
cmd = ['python', '/data/ratsql/rat-sql/sdp_tree/sdp.py', text]
output_bytes = subprocess.check_output(cmd)

print(output_bytes)
