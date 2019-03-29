""" indexation sur le reste du programme : 0. = zone vide, 1. = zone verte, 2 = zone en feu, 3 = zone en cendres
    * la colormap XCMAP définie globalement porte les couleurs proches de celles qu'on pourrait trouver en forêt, pour mieux visualiser """

""" 0) BIBLIOTHEQUES """ 

from matplotlib.pyplot import matshow
import matplotlib as mpl
import matplotlib.pylab as plt
import matplotlib.animation as animation
import random
import numpy as np
from math import *
from scipy import misc
from PIL import Image
from time import *

""" 1) FONCTIONS UTILES (a) """
 
def bool_p(p): # renvoie True avec une probabilite p et False avec une probabilité 1-p
    return random.random() <= p

def bool_p2(p): # fait le travail inverse
    return random.random() >= p
             
def matgen(n,m,d): # créer une matrice de dimensions (n,m) avec des zones vertes places aléatoirements suivant une densité d (facteur de percolation par site ici)

    forest = np.zeros((n,m))
    for i in range(n):
        for j in range(m):
            if bool_p(d):
                forest[i,j] = 1.
            else:
                forest[i,j] = 0.
    return forest
 
""" 1) FONCTIONS UTILES (b) """

XCMAP = mpl.colors.ListedColormap(['#7c5804','#009b14','#cc4e00','#000000']) # colormap "forêt" utilisée plus tard
                                                                             # on renseigne (0,1,2,3) dans cet ordre, en hex
w_dir = 'null' # direction éventuelle du vent

def start_R(forest): # renvoie un couple (i,j) tel que la case soit verte, au hasard
    n,m = forest.shape
    x=random.randint(0,n-1)
    y=random.randint(0,m-1)
    while(forest[x,y] != 1.):
        x=random.randint(0,n-1)
        y=random.randint(0,m-1)
    return(x,y)

def burn_spot(forest,i,j): # on démarre le feu en position (i,j) (sous réserve que cette zone soit verte au préalable)
    if forest[i,j] == 1.:
        forest[i,j] = 2.
    return forest
    
def nextToFire(forest,i,j): # renvoie un booléen caractérisant l'existence d'un arbre en feu au voisinage (à quatre directions) de l'arbre (i,j)
                            # RQ : le vent interviendrait ici...
                            
    n,m=forest.shape
    # r = random.random() || (pour le vent) peut-être rajouter une petite valeur ici pour compenser la valeur un peu faible en 0 de la fonction
    if forest[i,j] == 1.:
        if (i > 0 and forest[i - 1,j] == 2.):
            # if (r <= north):
            return True
        if (i < n - 1 and forest[i + 1,j] == 2.):
            # if (r <= south):
            return True
        if (j > 0 and forest[i,j - 1] == 2.):
            # if (r <= west):
            return True
        if (j < m - 1 and forest[i,j + 1] == 2.):
            # if (r <= east):
            return True
    return False
    
""" Autre procédé pour cette fonction ?...
    if forest[i,j] == 1.:
        for y in range(max(0,i-1),min(n,i+2)):
            if forest[y,j] == 2.:
                return True
        for x in range(max(0,j-1),min(m,j+2)):
            if forest[i,x] == 2.:
                return True
    return False
"""

def WVAL(x): # renvoie la valeur (à 10^-6 près) de la fonction de distribution pour x (angle) donné dans (-pi , pi)
             # utilisée uniquement pour l'implémentation du vent
             
    if (x < -pi - 0.000001 or x > pi + 0.000001):
        return 0
    return (((1/sqrt(2*pi))*exp(-(x/2)**2))/0.973679)
 
def propagateFire(forest,p_l,hMap): # la fonction reçoit la matrice à un instant t et la transforme en son image "aléatoire" à t + 1
                                    # les arbres au voisinage d'un arbre en feu prennent systématiquement feu si la probabilité de percolation p_l est fixée à 1.0, sinon celle-ci intervient aussi lors de cette étape
                                    # on stockera temporairement la liste des zones en feu avant d'effectuer les changements, pour pouvoir visualiser la propagation proprement
                                    
    n,m=forest.shape 
    currentFire = []
    for i in range(n):
        for j in range(m):
            if forest[i,j] == 2. :
                currentFire.append([i,j])
                
    # si on ne prend pas en compte le phénomène d'humidité            
    if (hMap == 'null'):
        for i in range(n): 
            for j in range(m):
                if nextToFire(forest,i,j): # percolation par site, on vérifie le voisinage direct
                    if bool_p(p_l): # intervention de la percolation par lien ici
                            forest[i,j] = 2.
        for elt in currentFire:
            i,j = elt[0],elt[1]
            forest[i,j] = 3.
            
        return forest
    
    # autrement, s'il existe une hMap :
    for i in range(n): 
        for j in range(m):
            if nextToFire(forest,i,j):
                if bool_p(p_l): 
                    if bool_p2(hMap[i,j]): # ici intervient l'humidité de la hMap
                        forest[i,j] = 2.
    
    
    for elt in currentFire:
        i,j = elt[0],elt[1]
        forest[i,j] = 3.
    
    return forest
 
def stillOnFire(forest): # vérifie s'il y a encore une zone verte susceptible de prendre feu
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

def count(forest,f): # compte le nombre de zones indicées "f" dans la matrice de la forêt
    n,m = forest.shape
    C = 0
    for k in range(n):
        for i in range(m):
            if forest[k,i] == f:
                C+=1
    return C

""" 2.1) COEUR DU PROGRAMME (+ STATS) """

north, south, east, west = 0,0,0,0 # pour le vent...

def simulation(n,m,d,p,mode,wind,hMap): # (n,m = dimensions, d,p = densité et proba de lien, mode = animé(1) ou non(0), wind = direction du vent(W,S,E,N) ou pas ('null'), hMap = la matrice d'humidité (ou pas : 'null'))
                                        # RQ : avec un paramètre de vent nul on remarque qu'on a quand même une sorte de direction de propagation "aléatoire"
                                        
    p_l = p # on définit la probabilité sur les arêtes
    w_dir = wind # et le vent qui nous sera utile plus tard
    
    """ # CETTE SECTION CONCERNE L'IMPLEMENTATION (POSSIBLEMENT) DU VENT
    
        #on définit les valeurs de vérifications dans nextToFire selon la direction du vent que l'utilisateur renseigne (on rajoutera plus tard les NE,NW,SE,SW)
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
        east = 0.1 """
        
    forest = matgen(n,m,d) 
    green = count(forest,1.) # comptage des zones à l'instant initial
    void = count(forest,0.) 
    
    i,j = start_R(forest) # le point de départ est désigné aléatoirement
    if (mode == 1):
        forest = animate(forest,i,j,p_l,hMap) # animation de la propagation
    elif (mode == 0): # pour effectuer des centaines d'essais, mieux vaux désactiver l'animation
        forest = animate_nofilm(forest,i,j,p_l,hMap) # sans image
    
    bnt = count(forest,2.) + count(forest,3.) # nb d'arbres brûlés au total
    bntPA = bnt / (n*m) # proportion brûlé / total
    bntPR = bnt / green # proportion brûlé / nb d'arbres qu'il y avait au début
    
    return([bnt,bntPA,bntPR]) # retourne quelques valeurs caractéristiques. Certaines expériences à d > 0.5 sont plutôt étranges... => problème réglé

# les programmes suivants seront utiles pour faire un relevé statistique (cf. relevés). Il serait intéressant de voir le relevé stat à p_l variable aussi, puis avec d et p_l variable (deux dimensions ?)

def stat_density(n,m,p): # paramètre variant : densité d, à p_l fixé

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
        
        result.append([d,bntM,bntPAM,bntPRM]) # result contient un tableau avec densité d (variante), nb de brulés et proportions
        d += 0.05
        
    return(result)

# def stat_link(n,m,d):
# def stat_dual(n,m):

""" 2.2) ANIMATION """ 
 
def animate(forest,i,j,p_l,hMap): # fonction principale qui va permettre d'animer le procédé
    fig = plt.figure()
    film = []
        
    # Initialisation
    forestOnFire = burn_spot(forest,i,j)
    film.append([matshow(forestOnFire, fignum=False, animated=True, cmap=XCMAP)])
    plt.draw()
     
    # Propagation
    while stillOnFire(forest):
        forestOnFire = propagateFire(burn_spot(forest,i,j),p_l,hMap)
        film.append([matshow(forestOnFire, fignum=False, animated=True, cmap=XCMAP)])
        plt.draw()
     
    # Animation
    ani = animation.ArtistAnimation(fig, film, interval=150, blit=True, repeat=False) # RQ : la valeur interval peut être modifiée pour le plaisir des yeux, et repeat=False remplacé par repeat_delay = X (en ms)
    
    plt.draw()
    plt.show()
    
    return(forest)
    
def animate_nofilm(forest,i,j,p_l,hMap): # variante sans animation ni image (utilisée pour le module statistique précédent)
    forestOnFire = burn_spot(forest,i,j)
    while stillOnFire(forest):
        forestOnFire = propagateFire(burn_spot(forest,i,j),p_l,hMap)
    return (forest)

""" 3) ReTreeval ? """

def visualize(matrix): # pour visualiser une matrice rapidement (permet de détecter une erreur aussi)
    plt.matshow(matrix) # on peut toujours changer de cmap ici aussi
    plt.show()
    
def getmatrix(picname): # renvoie la matrice RGBA associée l'image réelle
    return(np.array(Image.open(picname))) 

def creatematrix(picname): # cette partie a pour but de former la matrice discrète (position des arbres) à partir d'une image réelle

    forest_R = getmatrix(picname) # la matrice "foret" contient maintenant les données RGB pures de l'image réelle

    """ Afin de filtrer les pixels, on s'intéresse aux conditions sur les composantes RGB de chacun d'entre eux. Tous les pixels dont les               coordonnées sont dans un certain "produit d'intervalles" (un "green range") voient leur valeur changer (devenir égale à 1), et les autres
pixels porteront la valeur égale à 0 (on doit définir le lieu colorimétrique du vert adapté à la situaiton en clair)
    Pour l'instant, on se contente d'un lieu défini de la manière suivante : R entre Ra et Rb, B entre Ba et Bb, G entre Ga et Gb (on pourrait prendre plus large s'il n'y a pas d'eau, aussi il faudrait travailler l'image au préalable sur un logiciel type Photoshop par exemple """
    
    n,m,w = forest_R.shape
    forest = np.zeros((n,m))
    for i in range (n):
        for j in range (m):
            if forest_R[i][j][0] >= 0 and forest_R[i][j][0] <= 110 and forest_R[i][j][1] >= 0 and forest_R[i][j][1] <= 180 and forest_R[i][j][2] >= 0 and forest_R[i][j][2] <= 110 : # si le pixel en question est dans le vert
                forest[i,j] = 1.
            else: # autrement la case devient zone vide
                forest[i,j] = 0.
    return(forest)
    
def retreeval(file,p,mode,w): # file = l'image de l'arbre, (pas de densité de présence d puisque préfixée par l'image) choix sur la percolation par lien p, mode : animé(1) ou non(0), w : vent, etc...
                              # on aimerait pouvoir fixer le point de départ de manière volontaire ?
    w_dir = w
    p_l = p
    forest = creatematrix(file)
    n,m = forest.shape
    green = count(forest,1.) # nb d'arbres à l'état initial
    void = count(forest,0.) # nb de zones vide au départ
    i,j = start_R(forest) # ici on démarre le feu de manière aléatoire dans la forêt réelle, on pourrait renseigner i et j dans les paramètres du programme
    if (mode == 1):
        forest = animate(forest,i,j,p_l,'null') # animé
    elif (mode == 0):
        forest = animate_nofilm(forest,i,j,p_l,'null') # ou pas
        
    bnt = count(forest,2.) + count(forest,3.) # nb d'arbres brûlés
    bntPA = bnt / (n*m) # proportion brûlé / total
    bntPR = bnt / green # proportion brûlé / nb d'arbres qu'il y avait au début
    
    return([bnt,bntPA,bntPR])

# 4) Gestion de l'humidité, du vent ?

# ** pour l'humidité, mappage continu (type continuousmapG() pour le générer, ou une image en alpha pour une utilisation directe) d'un coefficient d'humidité, introduire différents types d'arbres, etc... pour le vent, cf site et la gaussienne, INTEGRER LES NOMBRES COMPLEXES ???, finir par une touche de physique stat

# ** pour le vent et l'humidité, faire un schéma à part où on ne prend pas compte de la percolation (i.e. matrice à paramètre (1,1))

def rand_matgen(n,m): # générer une matrice à coefficients dans [0;1] totalement aléatoire
    matrix = np.zeros((n,m))
    for k in range(n):
        for j in range(m):
            matrix[j,k] = random.random()
    return matrix
    
def hMapGen(n,m,hMin,hMax,lCT,mode): # le but de cette fonction est de créer une sorte de carte d'humidité de manière à peu près continue ("lisse"), et aléatoire (entre deux bornes). n,m : dimensions, (hMin,hMax) : valeurs minimales et maximales des coefficients dans la matrice (ces deux valeurs sont dans [0;1]), lCT est le décalage maximal entre deux cases en contact
    # on peut partir d'une matrice générée aléatoirement entre hMin et hMax puis l'affiner, ou créer la matrice de proche en proche
    
    if mode == 'continuous': # continuité totale (à faire...)
        return(matrix)
    
    elif mode == 'column': # continuité en colonne
        matrix = np.zeros((n,m))
        # génération de la première ligne
        matrix[0,0] = (hMax - hMin)*random.random() + hMin
        for k in range(1,m):
            c = random.random()
            x = random.random()
            if matrix[0,k-1] * (1 + c*lCT) >= hMax:
                matrix[0,k] = matrix[0,k-1] * (1 - c*lCT)
            elif matrix[0,k-1] * (1 - c*lCT) <= hMin:
                matrix[0,k] = matrix[0,k-1] * (1 + c*lCT)
            elif x < 0.5:
                matrix[0,k] = matrix[0,k-1] * (1 + c*lCT)
            elif x >= 0.5:
                matrix[0,k] = matrix[0,k-1] * (1 - c*lCT)
        # génération du reste
        for k in range(m):
            for j in range(1,n):
                c = random.random()
                x = random.random()
                if matrix[j-1,k] * (1 + c*lCT) >= hMax:
                    matrix[j,k] = matrix[j-1,k] * (1 - c*lCT)
                elif matrix[j-1,k] * (1 - c*lCT) <= hMin:
                    matrix[j,k] = matrix[j-1,k] * (1 + c*lCT)
                elif x < 0.5:
                    matrix[j,k] = matrix[j-1,k] * (1 + c*lCT)
                elif x >= 0.5:
                    matrix[j,k] = matrix[j-1,k] * (1 - c*lCT)
        # afin de visualiser plus correctement (avoir moins de contraste dans l'image), on force la présence de valeurs extrêmes
        matrix[0,0] = 0
        matrix[n-1,m-1] = 1
        
    elif mode == 'diag': #continuité diagonale
        matrix = np.zeros((n,m))
        # génération de la première ligne
        matrix[0,0] = (hMax - hMin)*random.random() + hMin
        for k in range(1,m):
            c = random.random()
            x = random.random()
            if matrix[0,k-1] * (1 + c*lCT) >= hMax:
                matrix[0,k] = matrix[0,k-1] * (1 - c*lCT)
            elif matrix[0,k-1] * (1 - c*lCT) <= hMin:
                matrix[0,k] = matrix[0,k-1] * (1 + c*lCT)
            elif x < 0.5:
                matrix[0,k] = matrix[0,k-1] * (1 + c*lCT)
            elif x >= 0.5:
                matrix[0,k] = matrix[0,k-1] * (1 - c*lCT)
        
        # génération de la première colonne
        for k in range(1,n):
            c = random.random()
            x = random.random()
            if matrix[k-1,0] * (1 + c*lCT) >= hMax:
                matrix[k,0] = matrix[k-1,0] * (1 - c*lCT)
            elif matrix[0,k-1] * (1 - c*lCT) <= hMin:
                matrix[k,0] = matrix[k-1,0] * (1 + c*lCT)
            elif x < 0.5:
                matrix[k,0] = matrix[k-1,0] * (1 + c*lCT)
            elif x >= 0.5:
                matrix[k,0] = matrix[k-1,0] * (1 - c*lCT)
                
        # génération du reste
        for k in range(1,n):
            for j in range(1,m):
                c = random.random()
                x = random.random()
                if ((matrix[j-1,k] + matrix[j-1,k-1] + matrix[j,k-1]) / 3) * (1 + c*lCT) >= hMax:
                    matrix[j,k] = ((matrix[j-1,k] + matrix[j-1,k-1] + matrix[j,k-1]) / 3) * (1 - c*lCT)
                elif ((matrix[j-1,k] + matrix[j-1,k-1] + matrix[j,k-1]) / 3) * (1 - c*lCT) <= hMin:
                    matrix[j,k] = ((matrix[j-1,k] + matrix[j-1,k-1] + matrix[j,k-1]) / 3) * (1 + c*lCT)
                elif x < 0.5:
                    matrix[j,k] = ((matrix[j-1,k] + matrix[j-1,k-1] + matrix[j,k-1]) / 3) * (1 + c*lCT)
                elif x >= 0.5:
                    matrix[j,k] = ((matrix[j-1,k] + matrix[j-1,k-1] + matrix[j,k-1]) / 3) * (1 - c*lCT)
                
        return (matrix)        
    