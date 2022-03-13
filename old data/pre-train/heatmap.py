import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

aupr1 = [[0.673063752, 0.047143372, 0.050302477, 0.049099804, 0.048517292, 0.050323559],
         [0.049344053, 0.802825132, 0.047734022, 0.046376227, 0.046456375, 0.051070369],
         [0.04825979, 0.045885939, 0.655794158, 0.050486475, 0.047477785, 0.047036856],
         [0.048694472, 0.048957552, 0.0488165, 0.634075361, 0.047259868, 0.070185734],
         [0.047238009, 0.049242232, 0.048798771, 0.046152722, 0.681881519, 0.04469432],
         [0.049018331, 0.047889596, 0.047803141, 0.073930557, 0.047358028, 0.812241207]]

auroc1 = [[0.933063994, 0.501333432, 0.51296616, 0.503124195, 0.497047639, 0.510871221],
          [0.498335103, 0.962778004, 0.50540473, 0.496254179, 0.494893271, 0.49198603],
          [0.506515567, 0.488700877, 0.931616569, 0.515630957, 0.501430474, 0.498696005],
          [0.509275861, 0.500186905, 0.506851322, 0.926568883, 0.495963351, 0.547974211],
          [0.500182878, 0.500018414, 0.499947539, 0.48147949, 0.914795406, 0.478522297],
          [0.511238536, 0.506973915, 0.495755608, 0.539627959, 0.495742534, 0.962045431]]

aupr2 = [[0.643556123, 0.555339897, 0.524182339, 0.468117214, 0.594499025, 0.445457693],
         [0.406181354, 0.735712377, 0.359947786, 0.386695684, 0.558044184, 0.44030939],
         [0.441829528, 0.424754547, 0.621532731, 0.423734723, 0.496803405, 0.409155284],
         [0.427685972, 0.519280398, 0.46687784, 0.723562303, 0.532654813, 0.311388611],
         [0.541760014, 0.584196009, 0.411292322, 0.307421301, 0.755154692, 0.470763484],
         [0.373076655, 0.492283663, 0.449725622, 0.215213866, 0.536723845, 0.813728843]]

auroc2 = [[0.905414299, 0.925085877, 0.875093057, 0.9009248, 0.927050863, 0.929599685],
          [0.826376317, 0.962552021, 0.803827259, 0.8672096, 0.888619235, 0.923487464],
          [0.831193145, 0.892048983, 0.908964898, 0.8839136, 0.895557958, 0.901598837],
          [0.818915793, 0.872311402, 0.821862015, 0.941952, 0.875192974, 0.85883297],
          [0.881450327, 0.912438895, 0.839750693, 0.8346464, 0.945759577, 0.922731953],
          [0.832731183, 0.86508951, 0.870234591, 0.7971456, 0.86388895, 0.981490734]]

X = ['GM12878', 'HeLa-S3', 'HUVEC', 'IMR90', 'K562', 'NHEK']
y = pd.DataFrame(auroc2, index=[x for x in X], columns=[x for x in X])  ##
ax = sns.heatmap(y, annot=True, vmin=0, vmax=1, cmap="Blues")
# ax.set_title('AUPR value of cell line-specific models on different cell lines')
# ax.set_title('AUROC value of cell line-specific models on different cell lines')
# ax.set_title('AUPR value of cross-cell line models on different cell lines')  ##
# ax.set_title('AUROC value of cross-cell line models on different cell lines')
ax.set_xlabel('Test cell line', fontsize = 12)
ax.set_ylabel('Training cell line', fontsize = 12)
plt.show()