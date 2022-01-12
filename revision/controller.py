# 
# This code comes as-is, and without guarantees
# 
import os
import sys
start_range = int(sys.argv[1])
end_range = int(sys.argv[2])
print(start_range, end_range)
for i in range(start_range, end_range):
    # number of samples (default 15, meaning 15 for every class)
    # optimization iterations used (default 2000)
    # optimization samples used in every iteration (default 6)
    # seed used (incremental)
    os.system("python3 qhack_revision.py 15 2000 6 "+str(i))
