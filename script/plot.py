import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import interp1d

elo  = []
game = []

for filename in os.listdir(os.getcwd()+"/reversi_training_model"):
  filename = os.getcwd() + "/reversi_training_model/" + filename
  print filename
  try:
    elo.append(np.loadtxt("%s/elo.txt" % filename))
    game.append(np.loadtxt("%s/game.txt" % filename))
  except:
    pass
print elo, game
x = np.linspace(min(game), max(game), 1000)
y = interp1d(game, elo)(x)
plt.scatter(game, elo)
plt.plot(x, y)
plt.xlabel("game")
plt.ylabel("elo")
plt.grid()
plt.show()

