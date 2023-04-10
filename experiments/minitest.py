import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from mpl_toolkits.mplot3d import Axes3D

# file_s = "./coverage1/#curve/connectivity.pkl"
file_c = "./coverage1/#curve/coverage_rate.pkl"
file_r = "./coverage1/#curve/rewards.pkl"
file_d = "./coverage1/#curve/done_steps.pkl"



with open(file_c, "rb") as fp:
    conn = pickle.load(fp)

num = len(conn)




# step = 1000
# for i in range()





# with open(file_d, "wb") as fp:
#     pickle.dump(conn, fp)

step = 1000
avg, avg_s = np.array([]), np.array([])
for i in range(0, num, step):
    avg = np.append(avg, np.mean(conn[i:i + step]))
    avg_s = np.append(avg_s, np.std(conn[i:i + step]))


plt.plot(avg, label="conn")
plt.plot(avg_s, label="conn_s")
# plt.fill_between(list(range(len(avg))), avg - avg_s, avg + avg_s, alpha=0.3)
# plt.ylim([0, 1.1])
plt.legend()
plt.show()



