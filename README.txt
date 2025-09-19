
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
