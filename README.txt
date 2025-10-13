### GRUP 3 - REPTE MATRICULES ###
Adriana, Paula, Lian i Martina

Aquest projecte té com a finalitat desenvolupar un sistema capaç de detectar i reconèixer automàticament 
les matrícules de vehicles a partir d'imatges. 

DESCRIPCIÓ GENERAL
El sistema es basa en tres etapes principals:
    1. Detecció de la matricula: es fa servir un model de detecció d'objectes YOLOv8 per localitzar la regió de la matrícula dins la imatge. 
    2. Preprocessament i segmentació: es fan servir tècniques de processament de imatge per tractar la imatge, eliminar el soroll i segmentar-la.
    3. Reconeixement de caràcters: s'utilitza EasyOCR per llegir els caràcters i convertir-los en text. 

FASES DEL PROJECTE
    * Preparació i anàlisi de dades: recollida d’imatges amb matrícules (BD de kaggle per l'entrenament i imatges nostres pel tets).
    
    * Detecció d’àrees de matrícula: entrenament d’un model YOLO preentrenat per identificar les zones de la matrícula dins cada imatge.
    
    * Preprocessament i segmentació: un cop la matrícula ha estat aillada, l'objectiu es transformar la imatge en una versió binària neta que
    faciliti l'aillament i posterior reconeixement de caràcters. 
            - Convertim la imatge RGB a escala HSV per neutralitzar els efectes del color i eliminar la banda blava europea.
            - S'aplica un filtre gaussià per suavitzar la imatge i reduir sorolls petits (ombres, taques, reflexos) sense deformar la forma dels caracters. 
            - S'utiliza el threshold OTSU, que calcula automàticament el llindar optim per a cada imatge i ens permet obtenir millor resultats per binaritzar la imatge. 
            - Invertim la imatge binària perque els caracters siguin blancs i el fons negre. 
            - Un cop obtinguda la imatge binaritzada, es realitza la segmentació dels caràcters utilitzant la funció findContours d’OpenCV, 
              que permet identificar contorns tancats com a possibles candidats a caràcters.
        
        També es van provar tècniques com closing i opening per omplir forats interns dels caràcters. Tot i això, durant la fase de test es va comprovar que 
        introduïen distorsions i reduïen la precisió del reconeixement, motiu pel qual finalment es van descartar.

        Per millorar la fiabilitat d’aquesta detecció, s’apliquen diverses heurístiques de filtratge:

            - Filtratge per àrea i alçada mínima: per eliminar petites taques o elements no desitjats.
            - Filtratge per ràtio d’aspecte: per assegurar-se que els contorns detectats s’ajusten a la forma típica d’un caràcter.
            - Comprovació de posició: es descarten els contorns que toquin la vora de la matrícula, per evitar confondre marcs o elements externs amb caràcters.
    
    * Reconeixement del text (OCR): aplicació d’EasyOCR per extreure els caràcters detectats.
    
    * Correcció i validació: implementació de funcions per corregir errors i validar el format final del text reconegut.
        Per millorar la precisió, s’apliquen correccions automàtiques basades en confusions habituals entre lletres i números 
        (per exemple, confondre “O” amb “0” o “I” amb “1”), i es comprova la coherència amb el format habitual de les matrícules espanyoles.
    
    * Avaluació dels resultats: ús de mètriques (WER/CER o la matriu de confusió) per quantificar el rendiment del sistema.
    
    * Visualització i anàlisi final: representació gràfica dels resultats i anàlisi d’errors per identificar possibles millores.


S’ha escollit aquest enfocament modular (detecció + OCR) perquè permet aprofitar els punts forts de cadascuna de les tecnologies. 
YOLO és molt eficient en la detecció d’objectes amb diferents mides i orientacions, mentre que EasyOCR és flexible i senzill d’integrar 
per al reconeixement de text.


##################################################################################################################################################################

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
- Preprocessament de les imatges i segmentació de la matricula fitlratge

TODO:
   * Operacions de fitlratge: Erosio i dilatació  (opening) 
   * Despres fer un closing (dilatacio i erosio), para que si hay huecos vacios en el numero se enganche
   * si tinc la x,y,h,a si la x o algun d'aquests es == 0, es a dir, toca el borde de la matricula, 
    eliminar este contorno porque no sera un caracter que nos interesa
    * altra opcio: en el otzu la k restarle numero pequeño

NEXT WEEK:
    * buscar la font de la matricula española: font2u --> para identificar el caracter
    * amb aquesta font generar imatges per fer el clasificador 

03/10/2025
Per la validació:
   - Hem de mirar lo be que va amb les nostres matricules, si no tenim groundtruth serà més dificil. 
   - Com només tenim 1 metode poca cosa tenim a comparar. 

06/10/2025
    * Implementació OCR con tesseract --> funciona molt malament 
    * Implementació de easyOCR:
        - confundeix 6 -> 5
                    M -> V
                    M -> H
                    7 -> 9
                    W -> R
                    X -> Y 
                    imatge 27 molt malament (renault)
                    6 -> 8



ASPECTES IMPORTANTS PRESENTACIÓ:
* Quedi clar les diferents etapes, primer localització, detecció, segmentació...
* Hem intentat X, no funcionaven, ho hem canviat per X... (tesseract per OCR no funciona be)
* Añadir la validación con boxplots, matriu de confusió, ver porque fallan las cosas. WER i CER 
* VALIDACIÓ PER CADA ETAPA
* HACER VALIDACIO PER CARACTERS, DE 200 M'HA DETECTAT BE 190 EXEMPLE