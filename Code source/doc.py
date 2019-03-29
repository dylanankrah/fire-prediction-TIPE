""" * Note : ce fichier a pour intérêt de recenser les paramètres demandés par les fonctions appelées plus particulièrement que les autres.
    * Note #2 : galal
    """

""" Simulate : (n,m,d,p,mode,wind,hMap) """
# n,m : dimensions de la matrice / forêt ; d,p : probabilités sur les arêtes et les sommets ; mode : animé (1) ou non (0), wind : paramètre de vent ('N','S','W','E') ou pas ('null'), hMap : la matrice d'humidité en paramètre, ou pas ('null')

""" VisualizeM : (matrix) """
# seul paramètre à renseigner : la matrice à visualiser

""" GetMatrix : (picname) """
# unique paramètre : l'adresse de l'image à convertir en forêt

""" ReTreeval : (file,p,mode,hMap,w) """
# file : adresse de l'image qui va servir de forêt ; p : probabilité de lien ; mode : animé (1) ou pas (0) ; hMap : la matrice d'humidité en paramètre (ou pas : 'null') ; w : paramètre de vent ('N','S','W','E') ou pas ('null')

""" ConvertToHMap : (picname) """
# unique paramètre : l'adresse de l'image à convertir en hMap
