
16/09/2025
- Detecció de matricules en segons que imatge aconseguit.
- model YOLO i detecció de contorns
- En algunes imatges detecta fatal on es troba la matricula.
TODO:
    * les matricules son regions blanques que utilitzi aixo per identificar fons clars --> VA MOLT MALAMENT AMB AQUEST AJUST image8
    * segons la posició, on es troba del cotxe, acostumen a estar en la part inferior de davant (només pasarem fotos de la part de davant) --> no accepta variabilitat d'imatges
    * aspect ratio més petit, afectará si tenim matricules petites?

19/09/2025
- Tenemos la deteccion de la matricula hecha

TODO:
    * Segmentar los caracteres de la matricula uno por uno
    * Transformació a grisos para quitar la parte de la E azul (COLOR_BGR2HSV)!
    * Utilitzar otzu, fa un histograma on la imatge casi es binaria
    * cv.findcontours en opencv
    * Usar un adaptative threshold


23/09

TODO:
    * Operacions de fitlratge: Erosio i dilatació  (opening)
    * Despres fer un closing (dilatacio i erosio), para que si hay huecos vacios en el numero se enganche
    * que hi hagi regions una dintre de l'altre es pq quan faig el cv.findcontours
    * si tinc la x,y,h,a si la x o algun d'aquests es == 0, es a dir, toca el borde de la matricula, 
    eliminar este contorno porque no sera un caracter que nos interesa
    * altra opcio: en el otzu la k restarle numero pequeño

NEXT WEEK:
    * buscar la font de la matricula española: font2u --> para identificar el caracter
    * amb aquesta font generar imatges per fer el clasificador 