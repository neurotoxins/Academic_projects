from __future__ import division
from pymongo import MongoClient
from progressbar import ProgressBar, Bar, Percentage, FormatLabel, ETA
import numpy as np

client = MongoClient()
db = client.dota
matches = db.bigdata
np.set_printoptions(threshold=np.nan)

heroesTotal = 110
featuresTotal = heroesTotal * 2

# The label vector, Y, is a bit vector indicating raduant won(1) or lost(-1)
matchesTotal = matches.count()

# Initialize training matrix
X = np.zeros((matchesTotal, featuresTotal), dtype=np.int8)

# Initialize training label vector
Y = np.zeros(matchesTotal, dtype=np.int8)

widgets = [FormatLabel('Processed: %(value)d/%(max)d matches. '), ETA(), Percentage(), ' ', Bar()]
pbar = ProgressBar(widgets=widgets, maxval=matchesTotal).start()

for i, match in enumerate(matches.find()):
    pbar.update(i)
    Y[i] = 1 if match['radiant_win'] else 0
    players = match['players']
    for player in players:
        hero_id = player['hero_id'] - 1

        # If the left-most bit of player_slot is set,
        # this player is on dire, so push the index accordingly
        player_slot = player['player_slot']
        if player_slot >= 128:
            hero_id += heroesTotal

        X[i, hero_id] = 1

pbar.finish()

print "Generating train and test sets."
indices = np.random.permutation(matchesTotal)
test_indices = indices[0:matchesTotal/10]
train_indices = indices[matchesTotal/10:matchesTotal]

X_test = X[test_indices]
Y_test = Y[test_indices]

X_train = X[train_indices]
Y_train = Y[train_indices]

print "Saving output file ..."
np.savez_compressed('lr_test_%d.npz' % len(test_indices), X=X_test, Y=Y_test)
np.savez_compressed('lr_train_%d.npz' % len(train_indices), X=X_train, Y=Y_train)