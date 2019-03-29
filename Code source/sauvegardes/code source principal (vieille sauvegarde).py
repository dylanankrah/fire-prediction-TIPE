# indexation : 0 = lieu vide, 1 = arbre, 2 = arbre en feu, 3 = zone en cendres A RAJOUTER TRES VITE !
# manque de jolies couleurs QQ (trouver LA colormap des familles) (ou coloriser tout ça après coup aussi ça marche bien)
# cf. propagation directionnelle pour les bails ajoutés (vent selon une direction, humidité, phénomène d'étincelle)

# 0) BIBLIOTHEQUES 

from matplotlib.pyplot import matshow
import matplotlib as mpl
import matplotlib.pylab as plt
import matplotlib.animation as animation
import random
import numpy as np
from math import *
from scipy import misc
from PIL import Image

# 1) FONCTIONS UTILES (a)
 
def bool_p(p):
    # renvoie True avec une probabilite p et False avec une probabilité 1-p
    return random.random() <= p
             
def matgen(n,m,d):
    # cree une forest de dimensions n*m avec des arbres places aléatoirements à une densité d (facteur de percolation par site ici)
    forest = np.zeros((n,m))
    for i in range(n):
        for j in range(m):
            if bool_p(d):
                forest[i,j] = 1.
            else:
                forest[i,j] = 0.
    return forest
 
# 1) FONCTIONS UTILES (b)

p_l = 1. # proba de percolation par lien, modifiée par l'utilisateur en cas de besoin
w_dir = 'null' # direction éventuelle du vent
 
""" 
def burn_spot_R(forest): # nécessite d'être fix... pb de float ?
    # met le feu à un arbre aléatoirement
    n,m = forest.shape
    x=random.randint(0,n)
    y=random.randint(0,m)
    while(forest[x,y] != 1.):
        x=random.randint(0,n)
        y=random.randint(0,m)
    forest[x,y] = 2.F
    return(forest) """

def start_R(forest): # alternative, renvoie un couple (i,j) tel que la case soit verte (au hasard)
    n,m = forest.shape
    x=random.randint(0,n-1)
    y=random.randint(0,m-1)
    while(forest[x,y] != 1.):
        x=random.randint(0,n-1)
        y=random.randint(0,m-1)
    return(x,y)

def burn_spot(forest,i,j): # met le feu à l'arbre en position (i,j)
    if forest[i,j] == 1.:
        forest[i,j] = 2. # on fout le feu la case d'indice (i,j)
    return forest
 
""" def neighbors_from(x,y,forest): # renvoie un voisinage à 4 ou 8 cases dans la mesure du possible, du lieu en question
    # n,m = forest.shape
    # return [(x, y + 1 if y + 1 < SIZE else 0), (x, y - 1), (x + 1 if x + 1 < SIZE else 0, y),(x - 1, y)]
    """
    
def nextToFire(forest,i,j): # existence d'un arbre en feu au voisinage de l'arbre (i,j)
                            # tenter d'utiliser neighbors_from plutôt ? revoir le voisinage
    """ c'est aussi ici qu'intervient le phénomène du vent, de l'humidité, etc..."""
    n,m=forest.shape
    r = random.random() # peut-être rajouter une petite valeur ici pour compenser la valeur un peu faible en 0 de la fonction
    if forest[i,j] == 1.:
        if (i > 0 and forest[i - 1,j] == 2.): # si la case du bas est en feu
            # if (r <= north):
            return True
        if (i < n - 1 and forest[i + 1,j] == 2.): # si la case du haut est en feu
            # if (r <= south):
            return True
        if (j > 0 and forest[i,j - 1] == 2.): # si la case de gauche est en feu
            # if (r <= west):
            return True
        if (j < m - 1 and forest[i,j + 1] == 2.): # si la case de droite est en feu
            # if (r <= east):
            return True
    return False
    
""" Autre procédé de vérification pour les alentours...

    if forest[i,j] == 1.:
        for y in range(max(0,i-1),min(n,i+2)):
            if forest[y,j] == 2.:
                return True
        for x in range(max(0,j-1),min(m,j+2)):
            if forest[i,x] == 2.:
                return True
    return False
"""

def wfunc(x): # renvoie la valeur (à 10^-6 près) de la fonction de distribution pour x (angle) donné entre
    if (x < -pi - 0.000001 or x > pi + 0.000001):
        return 0
    return (((1/sqrt(2*pi))*exp(-(x/2)**2))/0.973679)
 
def propagateFire(forest):
    """  les arbres qui peuvent bruler autour d'un arbre en feu prennent feu
    rq. : on se place dans un cadre de percolation par site, la probabilité variante est celle de densité de placement, pas celle d'ouverture des liens dans L^d...
    ainsi, un arbre à proximité du feu prend systématiquement feu mskn
    pour tenir compte du phénomène de percolation par lien, implémenter une probabilité que le lien entre l'arbre en question et le voisin soit ouvert // FAIT !"""
    
    n,m=forest.shape # presque toujours une matrice carrée en fait
    for i in range(n): # percolation par site, on vérifie simplement s'il y a le feu dans un voisinage direct
        for j in range(m):
            if nextToFire(forest,i,j):
                 if random.random() <= p_l: # intervention de la percolation par lien ici
                        forest[i,j] = 2.
    
    return forest
 
def stillOnFire(forest): # vérifie s'il existe encore un arbre susceptible de cramer
    n,m=forest.shape
    for i in range(n):
        for j in range(m):
            if nextToFire(forest,i,j):
                return True
    return False
                 
def burnForest(forest,i,j): # démarre le feu aux coordonnées (i,j) et propage le feu jusqu'à ce que ça ne soit plus possible
    forest = burn_spot(forest,i,j)
    while stillOnFire(forest):
        forest = propagateFire(forest)
    return forest

def count(forest,f): # compte le nombre de zones indicées "f"
    n,m = forest.shape
    C = 0 # compteur de zones 
    for k in range(n):
        for i in range(m):
            if forest[k,i] == f:
                C+=1
    return C

# 2.1) COEUR DU PROGRAMME (+ STATS)

north, south, east, west = 0,0,0,0 # pour le vent...

def simulation(n,m,d,p,mode,w): # (n,m = dimensions, d,p = densité et proba de lien, mode = animé(1) ou non(0), w = direction du vent(W,S,E,N) ou pas ('null'))
    # RQ : avec un paramètre de vent nul on remarque qu'on a quand même une sorte de direction de propagation "aléatoire"
    p_l = p # percolation par lien définie
    w_dir = w
    """
    # on définit les valeurs de vérifications dans nextToFire selon la direction du vent que l'utilisateur renseigne (on rajoutera plus tard les NE,NW,SE,SW)
    if (w_dir == 'null'): # s'il y a pas de vent
        north = 1.
        east = 1.
        west = 1.
        south = 1. # on fait comme si de rien était lorsqu'on va lancer nextToFire (on aura toujours r <= 1 donc nextToFire fonctionne comme avant)
    elif (w_dir == 'N'): # si le vent souffle vers le nord
        north = 0.9,
        south = 0.1
        west = 0.5
        east = 0.5 # il y a vraisemblablement plus de chances qu'une case prenne feu sachant qu'une case en feu se situe en dessous d'elle plutôt qu'au dessus
    elif (w_dir == 'S'):
        north = 0.1,
        south = 0.9
        west = 0.5
        east = 0.5
    elif (w_dir == 'E'):
        north = 0.5,
        south = 0.5
        west = 0.1
        east = 0.9
    elif (w_dir == 'W'):
        north = 0.5,
        south = 0.5
        west = 0.9
        east = 0.1
    """
    forest = matgen(n,m,d) 
    green = count(forest,1.) # nb d'arbres à l'état initial
    void = count(forest,0.) # nb de zones vide au départ
    
    i,j = start_R(forest) # on pourrait fixer le point de départ du feu de manière volontaire, cf. ReTreeval
    if (mode == 1):
        forest = animate(forest,i,j) # procedé de percolation animé
    elif (mode == 0): # pour effectuer des centaines d'essais, mieux vaux désactiver l'animation
        forest = animate_nofilm(forest,i,j) # sans image
    
    bnt = count(forest,2.) # nb d'arbres brûlés
    bntPA = bnt / (n*m) # proportion brûlé / total
    bntPR = bnt / green # proportion brûlé / nb d'arbres qu'il y avait au début mskn
    
    return([bnt,bntPA,bntPR]) # certaines expériences à d > 0.5 sont plutôt étranges...
    #FIXED


def stat_density(n,m,p): # STAT : proba par sites (paramètre variant : densité d)
    d = 0.1 # d va évoluer de 0.1 à 0.95 par pas de 0.05
    result=[] # matrice qui contiendra les listes (proba, bnt, bntPA et PR)
    while (d <= 0.95):
        interm=[] # contient les résultats intermédiaires dont on va prendre la moyenne à la fin
        k = 1
        while (k <= 100): # on fait 100 essais par valeur de densité d
            interm.append(simulation(n,m,d,p,0))
            k+=1
        # ici, interm contient [[brulé à l'essai 1, proportions à l'essai 1],[brulé à l'essai 2, proportions à l'essai 2],...]
        bntM = 0
        bntPAM = 0
        bntPRM = 0
        for k in range(len(interm)):
            bntM += interm[k][0]
            bntPAM += interm[k][1]
            bntPRM += interm[k][2]
        bntM /= len(interm)
        bntPAM /= len(interm)
        bntPRM /= len(interm)
        
        result.append([d,bntM,bntPAM,bntPRM]) # result contient un tableau avec densité (variante), nb de brulés et proportions
        d += 0.05
        
    return(result)

""" def stat_areasonfire(n,m,p): # STAT : en percolation par sites, évalue le nb de zones en feu en fonction du nb d'étapes
    d = 0.6 # d évolue cette fois de 0.6 à 0.9 par pas de 0.05
    result=[] # la liste va contenir cette fois le nombre d'arbres en feu à chaque étape en moyenne sur 20 essais cette fois
    while (d <= 0.95):
        interm=[]
        k = 1
        while (k <= 100):
            F = matgen(n,m,d)
            interm.append """
 
# 2.2) ANIMATION 
""" RQ: ajouter cmap=plt.cm.nom_de_la_map pour les couleurs dans matshow """
 
def animate(forest,i,j):
    fig = plt.figure() # nouvelle figure
    film = []
    # Initialisation
    forestOnFire = burn_spot(forest,i,j)
    film.append([matshow(forestOnFire, fignum=False, animated=True)])
    plt.draw()
     
    # Propagation
    while stillOnFire(forest):
        forestOnFire = propagateFire(burn_spot(forest,i,j))
        film.append([matshow(forestOnFire, fignum=False, animated=True)])
        plt.draw()
     
    # Animation
    ani = animation.ArtistAnimation(fig, film, interval=100, blit=True, repeat=False)
    
    plt.draw()
    plt.show()
    
    return(forest)
    
def animate_nofilm(forest,i,j): # pour les stats
    forestOnFire = burn_spot(forest,i,j)
    while stillOnFire(forest):
        forestOnFire = propagateFire(burn_spot(forest,i,j))
        
    return (forest)

# 3) ReTreeval ?

def visualize(matrix): # pour visualiser une matrice rapidement (détecter une erreur)
    plt.matshow(matrix)
    plt.show()
    
def getmatrix(picname): # renvoie la matrice RGBA de l'image réelle
    return(np.array(Image.open(picname))) 

def creatematrix(picname):
    forest_R = getmatrix(picname) # la matrice "foret" contient maintenant les données RGB pures de l'image réelle
                                # mais il faudrait "filtrer" les pixels verts (arbres) du reste (potentiellement rien pour l'instant)
    """ tous les pixels dont les coordonnées sont dans un certain ensemble (un "green range") voient leur valeur alpha changer (devenir égale à 1), et les autres
    pixels porteront la valeur alpha égale à 0 (ici alpha ne désigne plus du tout une histoire de transparence mais simplement d'appartenance à une
    catégorie de couleur. On doit donc définir le lieu colorimétrique du vert forêt en clair 
    
    Pour l'instant, on se contente d'un lieu défini de la manière suivante : R entre Ra et Rb, B entre Ba et Bb, G entre Ga et Gb (on pourrait prendre plus large s'il
    n'y a pas d'eau, il faudrait process l'image au préalable sur un logiciel type Photoshop par exemple """
    
    n,m,w = forest_R.shape
    forest = np.zeros((n,m))
    for i in range (n):
        for j in range (m):
            if forest_R[i][j][0] >= 0 and forest_R[i][j][0] <= 130 and forest_R[i][j][1] >= 45 and forest_R[i][j][1] <= 180 and forest_R[i][j][2] >= 0 and forest_R[i][j][2] <= 120 :
                forest[i,j] = 1.
            else:
                forest[i,j] = 0.
    return(forest)
    
def retreeval(file,p,mode,w): # file = l'image de l'arbre, pas de densité de présence d puisque préfixée par l'image, mais toujours un choix sur la percolation par lien, mode : animé(1) ou non(0), w : vent, etc...

    w_dir = w
    p_l = p
    forest = creatematrix(file)
    n,m = forest.shape
    
    green = count(forest,1.) # nb d'arbres à l'état initial
    void = count(forest,0.) # nb de zones vide au départ
    
    i,j = start_R(forest) # ici on démarre le feu de manière aléatoire dans la forêt réelle, on pourrait renseigner i et j dans les paramètres du programme
    if (mode == 1):
        forest = animate(forest,i,j) # procedé de percolation animé
    elif (mode == 0): # pour effectuer des centaines d'essais, mieux vaux désactiver l'animation
        forest = animate_nofilm(forest,i,j) # sans image
        
    bnt = count(forest,2.) # nb d'arbres brûlés
    bntPA = bnt / (n*m) # proportion brûlé / total
    bntPR = bnt / green # proportion brûlé / nb d'arbres qu'il y avait au début mskn
    
    return([bnt,bntPA,bntPR]) #PB : on dirait que p_l n'a aucun effet...
    
    