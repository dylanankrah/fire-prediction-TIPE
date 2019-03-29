""" * Note : dans tout le code, le terme "densité" fait référence au concept de percolation par site (par opposition à la percolation par lien)
    * indexation sur le reste du programme : 0. = zone vide, 1. = zone verte, 2 = zone en feu, 3 = zone en cendres
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
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm

""" 0) DONNEES UTILES (issues du module stat) """

# data contient les données de la matrice 3D finale, cf. partie 2.1) (STAT_DualView) (notepad++ pour retirer les retours de ligne)
# cf fichier relevé stat pour reprendre les listes interminables !

""" 1) FONCTIONS UTILES (a) """
 
def bInf(p): # renvoie True avec une probabilite p et False avec une probabilité 1-p
    return random.random() <= p

def bSup(p): # fait le travail inverse
    return random.random() >= p
             
def MatrixGen(n,m,d): # créer une matrice de dimensions (n,m) avec des zones vertes places aléatoirements suivant une densité d (facteur de percolation par site ici)

    forest = np.zeros((n,m))
    for i in range(n):
        for j in range(m):
            if bInf(d):
                forest[i,j] = 1.
            else:
                forest[i,j] = 0.
    return forest
 
""" 1) FONCTIONS UTILES (b) """

XCMAP = mpl.colors.ListedColormap(['#7c5804','#23cf5f','#f4013e','#000000']) # colormap "forêt" utilisée plus tard
                                                                             # on renseigne (0,1,2,3) dans cet ordre, en hex

def RandomEntryPoint(forest): # renvoie un couple (i,j) tel que la case soit verte, au hasard
    n,m = forest.shape
    x=random.randint(0,n-1)
    y=random.randint(0,m-1)
    while(forest[x,y] != 1.):
        x=random.randint(0,n-1)
        y=random.randint(0,m-1)
    return(x,y)

def StartFire(forest,i,j): # on démarre le feu en position (i,j) (sous réserve que cette zone soit verte au préalable)
    if forest[i,j] == 1.:
        forest[i,j] = 2.
    return forest
    
def hasFirePotential(forest,i,j): # renvoie un booléen caractérisant l'existence d'un arbre en feu au voisinage (à quatre directions) de l'arbre (i,j)
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
 
def Propagate(forest,p_l,hMap,wind): # la fonction reçoit la matrice à un instant t et la transforme en son image "aléatoire" à t + 1
                                         # les arbres au voisinage d'un arbre en feu prennent systématiquement feu si la probabilité de percolation p_l est fixée à 1.0, sinon celle-ci intervient aussi lors de cette étape
                                         # on stockera temporairement la liste des zones en feu avant d'effectuer les changements, pour pouvoir visualiser la propagation proprement
                                    
    n,m=forest.shape 
    currentFire = []
    forestupdate = np.copy(forest) # on retournera forestupdate
    
    # on stocke la liste des cases en feu à l'instant t
    for i in range(n):
        for j in range(m):
            if forest[i,j] == 2. :
                currentFire.append([i,j])
                
    # si on ne prend pas en compte le phénomène d'humidité ni le vent      
    if (hMap == 'null') and (wind == 'null'):
        for i in range(n): 
            for j in range(m):
                if (hasFirePotential(forest,i,j) and forest[i][j] == 1.): # percolation par site, on vérifie le voisinage direct
                    if bInf(p_l): # intervention de la percolation par lien ici
                            forestupdate[i,j] = 2.
        # enfin, on met les cases en feu à l'instant t dans l'état brûlé à l'instant t+1                    
        for elt in currentFire:
            i,j = elt[0],elt[1]
            forestupdate[i,j] = 3.
            
        return forestupdate
    
    # autrement, s'il existe une hMap mais pas de vent :
    elif (hMap != 'null') and (wind == 'null'):
        for i in range(n): 
            for j in range(m):
                if (hasFirePotential(forest,i,j) and forest[i][j] == 1.):
                    if bInf(p_l): 
                        if bSup(hMap[i,j]): # ici intervient l'humidité de la hMap
                            forestupdate[i,j] = 2.
        # enfin, on met les cases en feu à l'instant t dans l'état brûlé à l'instant t+1                    
        for elt in currentFire:
            i,j = elt[0],elt[1]
            forestupdate[i,j] = 3.
            
        return forestupdate
        
    # si on prend compte du vent, mais pas de l'humidité :
    elif (hMap == 'null') and (wind != 'null'):
        for i in range(n): 
            for j in range(m):
                if (hasFirePotential(forest,i,j) and forest[i][j] == 1.):
                    if bInf(p_l):
                        if Winder(GetNeighbors(forest,i,j,2.),[i,j],wind): # là, faire le get voisinage en feu, puis aviser de comment choisir à partir de ça si oui ou non (i,j) prend feu
                            forestupdate[i,j] = 2.
        # enfin, on met les cases en feu à l'instant t dans l'état brûlé à l'instant t+1                    
        for elt in currentFire:
            i,j = elt[0],elt[1]
            forestupdate[i,j] = 3.
            
        return forestupdate
    
    # si on prend compte des deux à la fois :
    elif (hMap != 'null') and (wind != 'null'):
        for i in range(n): 
            for j in range(m):
                if hasFirePotential(forest,i,j):
                    if bInf(p_l): 
                        if bSup(hMap[i,j]): # ici intervient l'humidité de la hMap
                           if Winder(GetNeighbors(forest,i,j,2.),[i,j],wind):
                                forestupdate[i,j] = 2.
        # enfin, on met les cases en feu à l'instant t dans l'état brûlé à l'instant t+1                    
        for elt in currentFire:
            i,j = elt[0],elt[1]
            forestupdate[i,j] = 3.
            
        return forestupdate

        
def StillOnFire(forest): # vérifie s'il y a encore une zone verte susceptible de prendre feu
    n,m=forest.shape
    for i in range(n):
        for j in range(m):
            if forest[i][j] == 2.: # ou si hasFirePotential(i,j)
                return True
    return False
                 
""" def burnForest(forest,i,j): # démarre le feu aux coordonnées (i,j) et propage le feu jusqu'à ce que ça ne soit plus possible
        forest = StartFire(forest,i,j)
        while StillOnFire(forest):
            forest = Propagate(forest,wind)
        return forest
    """

def IndexCount(forest,f): # compte le nombre de zones indicées "f" dans la matrice de la forêt
    n,m = forest.shape
    C = 0
    for k in range(n):
        for i in range(m):
            if forest[k,i] == f:
                C+=1
    return C

""" 2.1) COEUR DU PROGRAMME (+ STATS) """

def Simulate(n,m,d,p,mode,wind,hMap): # (n,m = dimensions, d,p = densité et proba de lien, mode = animé(1) ou non(0), wind = direction du vent(W,S,E,N) ou pas ('null'), hMap = la matrice d'humidité (ou pas : 'null'))
                                        # RQ : avec un paramètre de vent nul on remarque qu'on a quand même une sorte de direction de propagation "aléatoire"
                                        # RQ #2 : pour voir l'impact de l'humidité, on va générer une hMap qu'on visualise, puis on l'envoie en paramètre dans la Simulate et on compare (cf. dossier)
                                        
    p_l = p # on définit la probabilité sur les arêtes
    w_dir = wind # et le vent qui nous sera utile plus tard
    
    forest = MatrixGen(n,m,d) 
    green = IndexCount(forest,1.) # comptage des zones à l'instant initial
    void = IndexCount(forest,0.) 
    
    i,j = RandomEntryPoint(forest) # le point de départ est désigné aléatoirement
    if (mode == 1):
        forest = Animate(forest,i,j,p_l,hMap,w_dir) # animation de la propagation
    elif (mode == 0): # pour effectuer des centaines d'essais, mieux vaux désactiver l'animation
        forest = AnimateNF(forest,i,j,p_l,hMap,w_dir) # sans image
    
    bnt = IndexCount(forest,2.) + IndexCount(forest,3.) # nb d'arbres brûlés au total
    bntPA = bnt / (n*m) # proportion brûlé / total
    bntPR = bnt / green # proportion brûlé / nb d'arbres qu'il y avait au début
    
    return([bnt,bntPA,bntPR]) # retourne quelques valeurs caractéristiques. Certaines expériences à d > 0.5 sont plutôt étranges... => problème réglé

# les programmes suivants seront utiles pour faire un relevé statistique (cf. relevés) à une, puis deux variables

def STAT_Density(n,m,p): # paramètre variant : densité d, à p_l fixé (typiquement 1 dans le relevé)

    C = 0

    d = 0.1 # d va évoluer de 0.1 à 0.95 par pas de 0.05
    result=[] # matrice qui contiendra les listes (proba, bnt, bntPA et PR)
    while (d <= 0.95):
        interm=[] # contient les résultats intermédiaires dont on va prendre la moyenne à la fin
        k = 1
        while (k <= 100): # on fait 100 essais par valeur de densité d
            interm.append(Simulate(n,m,d,p,0,'null','null'))
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
        
        result.append([p,bntM,bntPAM,bntPRM]) # result contient un tableau avec densité d (variante), nb de brulés et proportions
        d += 0.05
        C += 1
        print(C)
        
    return(result)

def STAT_Link(n,m): # paramètre variant : probabilité sur les arêtes p_l = p, à d fixé = 1

    p = 0.1 # p va évoluer de 0.1 à 0.95 par pas de 0.05
    result=[] # matrice qui contiendra les listes (proba, bnt, bntPA et PR)
    while (p <= 0.95):
        interm=[] # contient les résultats intermédiaires dont on va prendre la moyenne à la fin
        k = 1
        while (k <= 100): # on fait 100 essais par valeur de densité p
            interm.append(Simulate(n,m,1,p,0,'null','null'))
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
        
        result.append([p,bntM,bntPAM,bntPRM]) # result contient un tableau avec densité p_l (variante), nb de brulés et proportions
        
        p += 0.05
        
    return(result)
    
def STAT_Dual(n,m): # les deux paramètres vont varier, le but est de tracer (si D désigne les proportions) D(d,p_l) de [0;1]^2 dans [0;1]

    C = 0 # compteur d'actualisation
    
    p = 0.01
    result=[] # matrice qui contiendra des listes de la forme (d,p_l,bnt,bntPA,bntPR)
    while (p <= 1.01):
        d = 0.01
        while (d <= 1.01):
            interm=[]
            k = 1
            while (k <= 5): # on fait 5 essais par couple (d,p_l)
                interm.append(Simulate(n,m,d,p,0,'null','null'))
                k+=1
            # ici, interm contient [[brulé à l'essai 1, proportions à l'essai 1],[brulé à l'essai 2, proportions à l'essai 2],...] pour p donné
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
            
            result.append([d,p,bntM,bntPAM,bntPRM])
            
            C += 1
            print(C) # vérification que le programme tourne effectivement
            
            d += 0.01
        p += 0.01
    
    # result contient donc [0.0,0.0,les proportions],...,[1.0,1.0,les proportions]
    return(result)

def STAT_DualView(): # cette fonction a pour but de représenter la proportion d'arbres brûlés en fonction des deux paramètres de probabilité à la fois, il s'agit donc d'un dessin 3D d'une fonction à deux variables D(d,p_l)
                # les coordonnées les plus parlantes sont les deux premières (évidentes, elles jouent le rôle de X et Y) et la dernière (plus parlante puisque c'est le ratio brûlé / nb. d'arbres au départ)
                # voir dossier pour résultats et courbe 3D
    
    X,Y,Z=[],[],[]
    X,Y = np.linspace(0.01,1.0,num=50),np.linspace(0.01,1.0,num=50)
    X,Y = np.meshgrid(X,Y)

    Z = np.zeros((len(X),len(Y)))
    for i in range(len(X)):
        for j in range(len(Y)):
            Z[i,j] = data100[i*len(X)+j][4]
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(X, Y, Z, rstride = 1, cstride = 1, cmap=plt.cm.plasma, linewidth=0, antialiased=False)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    
    ax.set_zlim(0, 1.00)
    ax.set_xlabel('Site')
    ax.set_ylabel('Lien')
    ax.set_zlabel('Proportion')

    plt.show()
    
""" 2.2) ANIMATION """ 
 
def Animate(forest,i,j,p_l,hMap,wind): # fonction principale qui va permettre d'animer le procédé
    fig = plt.figure()
    film = []
        
    # Initialisation
    forestOnFire = StartFire(forest,i,j)
    film.append([matshow(forestOnFire, fignum=False, animated=True, cmap=XCMAP)])
    plt.draw()
     
    # Propagation
    while StillOnFire(forestOnFire): # EN CAS D'ERREUR : remplacer forestOnFire par forest
        forestOnFire = Propagate(forestOnFire,p_l,hMap,wind) # ICI StartFire(forest,i,j) à la place de forestOnFire, ou plus dernièrement forest au lieu de onFire
        film.append([matshow(forestOnFire, fignum=False, animated=True, cmap=XCMAP)])
        plt.draw()
     
    # Animation
    ani = animation.ArtistAnimation(fig, film, interval=100, blit=True, repeat=False) # RQ : la valeur interval peut être modifiée pour le plaisir des yeux, et repeat=False remplacé par repeat_delay = X (en ms)
    
    plt.draw()
    plt.show()
    
    return(forestOnFire)
    
def AnimateNF(forest,i,j,p_l,hMap,wind): # variante sans animation ni image (utilisée pour le module statistique précédent), obsolète !! A mettre à jour en prenant l'exemple du dessus
    forestOnFire = StartFire(forest,i,j)
    while StillOnFire(forestOnFire):
        forestOnFire = Propagate(forestOnFire,p_l,hMap,wind)
    return (forestOnFire)

""" 3) ReTreeval ? """

def VisualizeM(matrix): # pour visualiser une matrice rapidement (permet de détecter une erreur aussi)
    plt.matshow(matrix, cmap = plt.cm.jet_r) # on peut toujours changer de cmap ici aussi, le jet_r assure bleu = humide, rouge = aride
    plt.show()
    
def GetMatrix(picname): # renvoie la matrice RGBA associée l'image réelle
    return(np.array(Image.open(picname))) 

def ConvertToMatrix(picname): # cette partie a pour but de former la matrice discrète (position des arbres) à partir d'une image réelle

    forest_R = GetMatrix(picname) # la matrice "foret" contient maintenant les données RGB pures de l'image réelle

    """ Afin de filtrer les pixels, on s'intéresse aux conditions sur les composantes RGB de chacun d'entre eux. Tous les pixels dont les coordonnées sont dans un certain "produit d'intervalles" (un "green range") voient leur valeur changer (devenir égale à 1), et les autres
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
    
def ReTreeval(file,p,mode,hMap,w): # file = l'image de l'arbre, (pas de densité de présence d puisque préfixée par l'image) choix sur la percolation par lien p, mode : animé(1) ou non(0), w : vent, etc...
                              # on aimerait pouvoir fixer le point de départ de manière volontaire, donc rajouter un i,j
    w_dir = w
    p_l = p
    forest = ConvertToMatrix(file)
    n,m = forest.shape
    green = IndexCount(forest,1.) # nb d'arbres à l'état initial
    void = IndexCount(forest,0.) # nb de zones vide au départ
    
    # au lieu de mettre RandomEntryPoint ici, on pourrait rajouter i,j = input(),input() pour l'utilisateur qui est supposé connaître les coordonnées du départ
    i,j = RandomEntryPoint(forest) # ici on démarre le feu de manière aléatoire dans la forêt réelle, on pourrait renseigner i et j dans les paramètres du programme
    if (mode == 1):
        forest = Animate(forest,i,j,p_l,hMap,w_dir) # animé
    elif (mode == 0):
        forest = AnimateNF(forest,i,j,p_l,hMap,w_dir) # ou pas
        
    bnt = IndexCount(forest,2.) + IndexCount(forest,3.) # nb d'arbres brûlés
    bntPA = bnt / (n*m) # proportion brûlé / total
    bntPR = bnt / green # proportion brûlé / nb d'arbres qu'il y avait au début
    
    return([bnt,bntPA,bntPR])

""" 4) Gestion de l'humidité, du vent """

# ** pour l'humidité, mappage continu (type continuousmapG() pour le générer, ou une image en alpha pour une utilisation directe) d'un coefficient d'humidité, introduire différents types d'arbres, etc... pour le vent, cf site et la gaussienne, INTEGRER LES NOMBRES COMPLEXES ???, finir par une touche de physique stat / NS

# ** pour le vent et l'humidité, faire un schéma à part où on ne prend pas compte de la percolation (i.e. matrice à paramètre (1,1)) !!

""" 4.1) Humidité """

def rMatrixGen(n,m): # générer une matrice à coefficients dans [0;1] totalement aléatoire
    matrix = np.zeros((n,m))
    for k in range(n):
        for j in range(m):
            matrix[j,k] = random.random()
    return matrix
    
def hMapGen(n,m,hMin,hMax,lCT,mode): # le but de cette fonction est de créer une sorte de carte d'humidité de manière à peu près continue ("lisse"), et aléatoire (entre deux bornes). n,m : dimensions, (hMin,hMax) : valeurs minimales et maximales des coefficients dans la matrice (ces deux valeurs sont dans [0;1]), lCT est le décalage maximal entre deux cases en contact
                                     # on peut partir d'une matrice générée aléatoirement entre hMin et hMax puis l'affiner, ou créer la matrice de proche en proche
                                     # on pourrait aussi utiliser une carte prégénérée par un autre logiciel de manière similaire, voir ConvertToHMap
                                     # RAPPEL : des bornes effectives après plusieurs essais, sont des bornes de type 0.2 à 0.8 avec un contraste de 0.05
                                     # RAPPEL #2 : ce programme est un essai simple, par la suite on utilise principalement ConvertToHMap (+ logique, + pratique, etc...)
                                     
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
        
    elif mode == 'diag': # continuité diagonale
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

def weighAverage(pixel): # retourne le gris correspondant à un pixel
    return(0.299*pixel[0] + 0.587*pixel[1] + 0.114*pixel[2])
    
def ConvertToHMap(picname): # cette fonction va convertir une image issue de Photoshop par exemple en une hMap utilisable pour le programme
                       # on supposera que l'image est déjà en noir et blanc : noir = aride, blanc = humide dans l'image de départ process par un externe
                       
    hMap_255 = GetMatrix(picname)
    n,m,w = hMap_255.shape
    hMap_0 = np.zeros((n,m))
    for k in range(n):
        for j in range(m):
            hMap_0[k,j] = weighAverage(hMap_255[k,j])/255
    
    print(n,m)
    return(hMap_0)

""" 4.2) Vent """
# ici, on placera des fonctions utiles pour le vent

def Winder(neighbors,pos,wind): # renvoie True ou False selon que le vent est propice à brûler la position i,j (pos) sachant le vent et l'entourage
                                # plusieurs versions de cette fonction sont écrites et seule celle qui n'est pas en commentaire sera retenue...

    val = 0 # Winder va renvoyer True si val dépasse une certaine valeur (typiquement 1, voir tests). val va augmenter plus ou moins à cause de la contribution de chacune des cases dans le voisinage direct, selon la direction du vent
    crit = 0.03 # valeur critique que val va devoir dépasser, aux alentours de 0.03 - 0.05
    
    # ce qu'il faut paramétrer ici, c'est les valeurs ajoutées aux randoms pour que ça ait l'air naturel
    if (wind == 'S'):
        for neigh in neighbors: # pour chaque voisin de la position en question
            # sachant qu'alors le vent vient du dessus
            if (neigh[0] < pos[0]) and (neigh[1] == pos[1]) : # si le voisin est pile en dessous
                val += random.random() # en clair, y a beaucoup de chances que val dépasse 1 à cause de cette contribution seule déjà (grâce au 0.5)
            elif ((neigh[0] < pos[0]) and (neigh[1] < pos[1])) or ((neigh[0] < pos[0]) and (neigh[1] > pos[1])) : # sinon s'il est en bas à gauche ou droite
                val += random.random()/4
            elif ((neigh[0] == pos[0]) and (neigh[1] < pos[1])) or ((neigh[0] == pos[0]) and (neigh[1] > pos[1])) : # sinon s'il est à gauche ou à droite
                val += random.random()/16
            elif ((neigh[0] > pos[0]) and (neigh[1] < pos[1])) or ((neigh[0] > pos[0]) and (neigh[1] > pos[1])) : # sinon s'il est en haut à gauche ou droite
                val += random.random()/16
            elif ((neigh[0] > pos[0]) and (neigh[1] < pos[1])) or ((neigh[0] > pos[0]) and (neigh[1] > pos[1])) : # s'il est pile au dessus
                val += random.random()/32
        return(val >= crit)
        
    elif (wind == 'N'):
        for neigh in neighbors: # pour chaque voisin de la position en question
            # sachant qu'alors le vent vient du dessous
            if (neigh[0] < pos[0]) and (neigh[1] == pos[1]) : # si le voisin est pile en dessous
                val += random.random()/32
            elif ((neigh[0] < pos[0]) and (neigh[1] < pos[1])) or ((neigh[0] < pos[0]) and (neigh[1] > pos[1])) : # sinon s'il est en bas à gauche ou droite
                val += random.random()/16
            elif ((neigh[0] == pos[0]) and (neigh[1] < pos[1])) or ((neigh[0] == pos[0]) and (neigh[1] > pos[1])) : # sinon s'il est à gauche ou à droite
                val += random.random()/16
            elif ((neigh[0] > pos[0]) and (neigh[1] < pos[1])) or ((neigh[0] > pos[0]) and (neigh[1] > pos[1])) : # sinon s'il est en haut à gauche ou droite
                val += random.random()/4
            elif ((neigh[0] > pos[0]) and (neigh[1] < pos[1])) or ((neigh[0] > pos[0]) and (neigh[1] > pos[1])) : # s'il est pile au dessus
                val += random.random()
        return(val >= crit)
        
    """    
    elif (wind == 'E'):
        for neigh in neighbors: # pour chaque voisin de la position en question
            # sachant qu'alors le vent vient de la gauche
            if (neigh[0] < pos[0]) and (neigh[1] == pos[1]) : # si le voisin est pile en dessous
                val += random.random()/32
            elif ((neigh[0] < pos[0]) and (neigh[1] < pos[1])) or ((neigh[0] < pos[0]) and (neigh[1] > pos[1])) : # sinon s'il est en bas à gauche ou droite
                val += random.random()/16
            elif ((neigh[0] == pos[0]) and (neigh[1] < pos[1])) or ((neigh[0] == pos[0]) and (neigh[1] > pos[1])) : # sinon s'il est à gauche ou à droite
                val += random.random()/16
            elif ((neigh[0] > pos[0]) and (neigh[1] < pos[1])) or ((neigh[0] > pos[0]) and (neigh[1] > pos[1])) : # sinon s'il est en haut à gauche ou droite
                val += random.random()/4
            elif ((neigh[0] > pos[0]) and (neigh[1] < pos[1])) or ((neigh[0] > pos[0]) and (neigh[1] > pos[1])) : # s'il est pile au dessus
                val += random.random()
        return(val >= crit
        )
    elif (wind == 'W'):
        return(True) """
        
    # elif les directions intermédiaires NW,NE,SW,SE
    
def GetNeighbors(forest,i,j,type): # renvoie la liste du 4-voisinage direct d'une case, portant le type donné

    neighbors = []
    n,m = forest.shape
    for y in range(max(0,i-1),min(n,i+2)):
        if forest[y,j] == type:
            neighbors.append([y,j])
    for x in range(max(0,j-1),min(m,j+2)):
        if forest[i,x] == type:
            neighbors.append([i,x])
            
    return(neighbors)

""" 5) Aspect physique, complémentaire thermodynamique """
