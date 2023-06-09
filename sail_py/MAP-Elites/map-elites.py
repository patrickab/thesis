import numpy as np


def mapElites(fitnessFunction, map, p, d):
    # View Initial Map
    h = []
    if p['display']['illu']:
        fig = plt.figure(2)
        fig.clf()
        h = fig.add_subplot(111, projection='3d')
        h.set_xlabel('Feature 1')
        h.set_ylabel('Feature 2')
        h.set_zlabel('Fitness')
        h.set_title('Illumination Fitness')

    # MAP-Elites
    iGen = 1
    percImproved = []
    while iGen <= p['nGens']:
        # 1) Create and Evaluate Children
        # Create children which satisfy geometric constraints for validity
        nMissing = p['nChildren']
        children = []

        while nMissing > 0:
            indPool = createChildren(map, nMissing, p, d)
            validFunction = lambda genomes: d['validate'](genomes, d)
            validChildren, _, nMissing = getValidInds(indPool, validFunction, nMissing)
            children.extend(validChildren)

        fitness, values = fitnessFunction(children)

        # 2) Add Children to Map
        replaced, replacement = nicheCompete(children, fitness, map, d)
        map = updateMap(replaced, replacement, map, fitness, children, values, d['extraMapValues'])

        # Improvement Stats
        percImproved.append(len(replaced) / p['nChildren'])

        # View Illuminatiom Progress
        if p['display']['illu'] and iGen % p['display']['illuMod'] == 0:
            h.clear()
            x, y, z = zip(*map['fitness'])
            h.scatter(x, y, z, c=z, cmap='viridis')
            plt.pause(0.001)

        iGen += 1
        if iGen % (2 ** 5) == 0:
            print('\tIllumination Generation:', iGen, '- Improved:', percImproved[-1] * 100, '%')

    if percImproved[-1] > 0.05:
        print('Warning: MAP-Elites finished while still making improvements (>5% / generation)')

    return map, percImproved, h
