# à rajouter à la simulation : le caractère anisotrope dû au vent, à l'humidité de certains sites, aux éventuelles étincelles que le feu peut produire
# ainsi qu'un module de cartographie simple (PIL - RGB)

    # <<<<<<<<<<<< 4) TEST VENT DIRECTION EST >>>>>>>>>>>>
 
def proba(p_0):
    p_0=100*p_0
    p_1= (100-p_0)/2.
    assert p_0%2==0 # ici on veut que a soit pair pour que la somme des probas fasse bien 1
    assert int(p_0)+2*int(p_1) == 100
    liste = int(p_0)*["p_0"]+int(p_1)*["p_11"]+int(p_1)*["p_12"]
    return choice(liste)
     
def peutBrulerVentEst(foret,i,j,p_0):
    n,m=foret.shape # n et m respectivement le nombre de lignes et de colonnes
    if foret[i,j]==1.:
        # direction droite
        if proba(p_0)=="p_0":
            for x in range(max(0,j-1),j):
                if foret[i,x]==2.:
                    return True
        #diagonale haute
        if proba(p_0)=="p_11":
            for hy in range(i,min(n,i+1)):
                for hx in range(max(0,j-1),j):
                    if foret[hy,hx]==2.:
                        return True
        # diagonale basse
        if proba(p_0)=="p_12":
            for by in range(max(0,i-1),i):
                for bx in range(max(0,j-1),j):
                    if foret[by,bx]==2.:
                        return True
    return False
 
def propageFeuVentEst(foret,p_0):
    n,m=foret.shape # n et m respectivement le nombre de lignes et de colonnes
    c=np.copy(foret)
    for i in range(n):
        for j in range(m):
            if peutBrulerVentEst(c,i,j,p_0):
                foret[i,j]=2.
    return foret
 
def auFeuVentEst(foret,p_0):
    "verifie si au moins un arbre non en feu peut bruler"
    n,m=foret.shape
    for i in range(n):
        for j in range(m):
            if peutBrulerVentEst(foret,i,j,p_0):
                return True
    return False
 
def metFeuForetVentEst(foret,p_0):
    while auFeuVentEst(foret,p_0):
        foret = propageFeuVentEst(foret,p_0)
    return foret
 
def animationFeuVentEst(foret,i,j):
    fig = plt.figure() # nouvelle figure
    film = []
     
    # Initialisation
    film.append([matshow(foret, fignum=False, animated=True)])
    plt.draw() # mise a jour en temps reel du contenu des figures
     
    # Propagation
    while auFeuVentEst(foret):
        foret = propageFeuVentEst(foret)
        film.append([matshow(foret, fignum=False, animated=True)])
        plt.draw() # mise a jour en temps reel du contenu des figures
     
    # Animation
    ani = animation.ArtistAnimation(fig, film, interval=100, blit=True, repeat_delay=100)
     
    plt.draw() # mise a jour en temps reel du contenu des figures
    plt.show()