import numpy as np
import matplotlib.pyplot as plt
import os, sys
from scipy.interpolate import interp1d

elo  = []
game = []
nblocks = []
name = sys.argv[1]

for filename in sorted(os.listdir(os.getcwd()+"/%s_training_model" % name)):
  try:
    tmp_nblocks = int(filename.split("_")[-3])
  except:
    continue
  filename = os.getcwd() + "/%s_training_model/" % name + filename
  print filename
  try:
    elo.append(np.loadtxt("%s/elo.txt" % filename))
    game.append(np.loadtxt("%s/game.txt" % filename))
    nblocks.append(tmp_nblocks)
  except:
    pass
print "elo, game, nblocks"
print np.array([elo, game, nblocks]).T
x = np.linspace(min(game), max(game), 1000)
y = interp1d(game, elo)(x)
plt.scatter(game, elo)
plt.plot(x, y)
plt.xlabel("game")
plt.ylabel("elo")
plt.grid()

# add a line if nblocks change.
idx_nblocks_changed = np.append([1], np.diff(nblocks))
for i in range(len(idx_nblocks_changed)):
  if idx_nblocks_changed[i]:
    plt.axvline(game[i], label="nblocks %i" % nblocks[i])
plt.legend()
plt.tight_layout()
plt.show()

