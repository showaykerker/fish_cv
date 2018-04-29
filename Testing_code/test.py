import numpy as np
fx = np.random.normal(0, 1, (3,3))
fy = np.random.normal(0, 1, (3,3))
ft = np.random.normal(0, 1, (3,3))
Sumfx2 = sum([px*px for px in fx.flatten()])
Sumfy2 = sum([py*py for py in fy.flatten()])
Sumfxfy = sum([px*py for px, py in zip(fx.flatten(), fy.flatten())])
n_Sumfxft = -1 * sum([px*pt for px, pt in zip(fx.flatten(), ft.flatten())])
n_Sumfyft = -1 * sum([py*pt for py, pt in zip(fy.flatten(), ft.flatten())])

T1 = np.array([[Sumfx2, Sumfxfy],[Sumfxfy, Sumfy2]])**-1
T2 = np.array([[n_Sumfxft],[n_Sumfyft]])
u, v = T1.dot(T2).flatten()
print(u, v)
