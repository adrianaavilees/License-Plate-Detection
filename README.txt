### GRUP 3 - REPTE MATRICULES ###
Adriana, Paula, Lian i Martina


DESCRIPCIÓ DEL PROJECTE

El projecte té com a objectiu crear un sistema de Reconeixement Automàtic de Matrícules per detectar i llegir matrícules de vehicles, pensat per poder ser utilitzat pel control d’entrada/sortida de vehicles en parkings.

En aquest projecte ens hem centrat en matrícules espanyoles amb el sistema actual amb format XXXX YYY (sent XXXX un nombre sequencial de 4 xifres i YYY tres lletres consonants).

El sistema es basa en tres etapes principals:
    1. Detecció de la matricula: es fa servir un model de detecció d'objectes YOLOv8 per localitzar la regió de la matrícula dins la imatge. 
    2. Preprocessament i segmentació: es fan servir tècniques de processament de imatge per tractar la imatge, eliminar el soroll i segmentar-la.
    3. Reconeixement de caràcters: s'utilitza EasyOCR per llegir els caràcters i convertir-los en text. 




GUIA D'EXECUCIÓ

SVM: Per tal de testejar el model de SVM, modificar les configuració del fitxer svm_main.py i executar-lo. Això permetrà fer prediccions per un conjunt d’imatges de vehicles.

Cal modificar les següents rutes:
IMAGE_DIR: Ruta a la carpeta que conté les imatges de vehicles dels quals es vol llegir la matrícula.
SAVE_DIR: Ruta al directori on es volen guardar els resultats i les prediccions (retall de la imatge on es detecta la matrícula, caràcters segmentats i predicció final)
MODEL_PATH: Ruta al model de localització de matrícules que es vol utilitzar.
SVM_DIGITS: Ruta al model de clasificació de nombres que es vol utilitzar per predir els primers 4 caràcters de la matrícula.
SVM_LETTERS: Ruta al model de clasificació de lletres que es vol utilitzar per predir els darrers 3 caràcters de la matrícula.


OCR:





DESENVOLUPAMENT TÈCNIC

La implementació del model es pot dividir en 3 etapes diferenciades, la localització de la matrícula, la segmentació dels caràcters, i el reconeixement d’aquests.


	1. Preparació i anàlisi de dades
	S'ha fet una recollida d'imatges amb matrícules (BD de Kaggle per l'entrenament i imatges pròpies pels tests)

	2. Detecció d’àrees de matrícula
	L’objectiu d’aquesta fase és localitzar la posició de les matrícules dins una imatge, és a dir, detectar on es troba la matrícula.
	
	Per aquesta tasca s’ha utilitzat YOLOv8, un algoritme d'Aprenentatge Profund de detecció d’objectes en imatges basat en xarxes neuronals convolucionals. Concretament s’ha decidit utilitzar com a punt de 	partida el model preentrenat yolov8s.pt, la versió petita de YOLOv8, i posteriorment s’ha reentrenat amb un dataset propi annotat específicament per la detecció de matrícules. 
	
	Per entrenar el model s’ha utilitzat un dataset 4000 matrícules ja annotat, obtingut de Kaggle. Aquest dataset consta d’imatges de cotxes de tots els països, no només Espanya. No obstant, això no ha estat un problema, ja que en aquesta part tan sols interessava la detecció de les matrícules, no la lectura, i les característiques visuals d’aquestes son similars a nivell internacional.
	
	Els models entrenats s’han guardat en fitxers de format <nom_del_fitxer>.pt. Es poden trobar en el codi dins la carpeta models/ (best.pt, best_license_plate.pt)

	3. Segmentació de caràcters
	Un cop la matrícula ha estat aillada, l'objectiu es transformar la imatge en una versió binària neta que faciliti l'aillament i posterior reconeixement de caràcters. 
	            - Convertim la imatge RGB a escala HSV per neutralitzar els efectes del color i eliminar la banda blava europea.
	            - S'aplica un filtre gaussià per suavitzar la imatge i reduir sorolls petits (ombres, taques, reflexos) sense deformar la forma dels caracters. 
	            - S'utiliza el threshold OTSU, que calcula automàticament el llindar optim per a cada imatge i ens permet obtenir millor resultats per binaritzar la imatge. 
	            - Invertim la imatge binària perque els caracters siguin blancs i el fons negre. 
	            - Un cop obtinguda la imatge binaritzada, es realitza la segmentació dels caràcters utilitzant la funció findContours d’OpenCV, 
	              que permet identificar contorns tancats com a possibles candidats a caràcters.
	        
	També es van provar tècniques com closing i opening per omplir forats interns dels caràcters. Tot i això, durant la fase de test es va comprovar que introduïen distorsions i reduïen la precisió del reconeixement, motiu pel qual finalment es van descartar.
	
	Per millorar la fiabilitat d’aquesta detecció, s’apliquen diverses heurístiques de filtratge:
	            - Filtratge per àrea i alçada mínima: per eliminar petites taques o elements no desitjats.
	            - Filtratge per ràtio d’aspecte: per assegurar-se que els contorns detectats s’ajusten a la forma típica d’un caràcter i eliminar així elements de la matrícula que no es desitjen pel reconeixement.
	            - Comprovació de posició: es descarten els contorns que toquin la vora de la matrícula, per evitar confondre marcs o elements externs amb caràcters.



	4. Reconeixement de caràcters
	En aquesta fase s’han implementat dues aproximacions diferents per comparar rendiment i flexibilitat: una basada en SVM i una altra basada en OCR (EasyOCR).


		(1) Reconeixement amb OCR (EasyOCR)
		Aplicació d'EasyOCR per extreure els caràcters detectats.
		S’ha escollit aquest enfocament modular (detecció + OCR) perquè permet aprofitar els punts forts de cadascuna de les tecnologies. 
		YOLO és molt eficient en la detecció d’objectes amb diferents mides i orientacions, mentre que EasyOCR és flexible i senzill d’integrar 
		per al reconeixement de text.



		(2) SVM (Support Vector Machine)
		Per aquesta solució s’han entrenat classificadors SVM (Support Vector Machine) independents per als números i per a les lletres. 
		
		Els models s’han creat amb dos conjunts de dades separats:
			* Un per als números (0-9)
			* Un altre per a les lletres (BCDFGHJKLMNPQRSTVWXYZ)
	
		Aquests conjunts de dades utilitzats han estat datasets propis. El procés de generació d’aquests conjunts de dades s’ha realitzat seguint el següent procés:
			1. En primer lloc s’han fet fotos reals de matrícules de varis vehicles.
			2. Seguidament s’ha utilitzat el model yolov8 entrenat prèviament a la fase de detecció, per extreure automàticament el retall de la matrícula dins de cada imatge.
			3. Sobre cada retall, s’ha aplicat l’algoritme de segmentació desenvolupat, que retorna els caràcters (numeros, lletres)
			4. Aquests retalls obtinguts s’han annotat manualment, assignant a cada imatge el nom corresponent al caràcter que representa.
			5. Finalment, s’han aplicat tècniques de data augmentation per augmentar la quantitat d’exemples i millorar la robustesa del model.



	5. Correcció i validació
	Implementació de funcions per corregir errors i validar el format final del text reconegut.
	Per millorar la precisió, s’apliquen correccions automàtiques basades en confusions habituals entre lletres i números 
	(per exemple, confondre “O” amb “0” o “I” amb “1”), i es comprova la coherència amb el format habitual de les matrícules espanyoles.


	6. Avaluació dels resultats
	S'han utilitzat mètriques (WER/CER o la matriu de confusió) per quantificar el rendiment del sistema.


	7. Visualització i anàlisi final
	S'ha realitzat una representació gràfica dels resultats i anàlisi d’errors per identificar possibles millores.





##################################################################################################################################################################


DIARI DE DESENVOLUPAMENT

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
07/10/2025
	ASPECTES IMPORTANTS PER LA PRESENTACIÓ:
		* Quedi clar les diferents etapes, primer localització, detecció, segmentació...
		* Hem intentat X, no funcionaven, ho hem canviat per X... (tesseract per OCR no funciona be)
		* Añadir la validación con boxplots, matriu de confusió, ver porque fallan las cosas. WER i CER 
		* VALIDACIÓ PER CADA ETAPA
		* HACER VALIDACIO PER CARACTERS, DE 200 M'HA DETECTAT BE 190 EXEMPLE


ESTRUCTURA DEL CODI

/models/
 ├── best.pt                      # Model yolov8 reentrenat per la detecció de matrícules versió 0
 ├── best_license_plate.pt  # Model yolov8 reentrenat per la detecció de matrícules versió 1
/v0/SVM/
	/auxiliar_svm/
    ├── cropper.py	# Classe PlatePredictor amb la funció predict_and_crop_image() que localitza la matrícula
    ├── segmentator.py	# Classe Segmentator amb la funció segment_characters() que retorna els numeros i caràcters segmentats
    ├── plate_reader.py	# Classe PlateReader amb la funció predict_plate() que donada una imatge retorna la predicció del valor de la matrícula
       	/svm_models/
    ├── svm_classifier_letters.py	# Codi que entrena el model de SVM per reconeixer lletres i el guarda al fitxer svm_letters.pkl 
    ├── svm_classifier_numbers.py # Codi que entrena el model SVM per reconeixer números i el guarda al fitxer svm_digits.pkl
    ├── svm_digits.pkl	# model svm entrenat per classificar les lletres
    ├── svm_letters.pkl	# model svm entrenat per classificar els numeros
    ├── svm_main.py	# Script principal per provar el reconeixement
/OCR/
 ├── ground_truth.json		# Arxiu JSON que vincula cada imatge amb la seva matrícula per tal de saber quin és el resultat esperat. 
 ├── REPTE_MATRICULES_VERSIO_FINAL.ipynb	# Conté la verdsió definitiva i final de les funcions (ja netes) per fer el reconeixement de la matrícula juntament amb la validació del model. 
 ├── license_plate_detector.py	# Classe LicensePlateDetector que agrupa totes les funcions per detectar, segmentar i reconéixer la matrícula. 
 ├── main.ipynb	# Conté la funció principal de detecció, segmentació i reconeixement.
 ├── main_validation.ipynb	# Conté la funció principal de detecció, segmentació i reconeixement, incloint-hi tota la part de validació del model. 
 ├── ocr_evaluation_results.csv # Fitxer CSV que conté els resultats del procés de reconeixement en quant a la matrícula que figura al ground truth i la predicció del model entrenat. 
 ├── segmentation.py	# Classe LicensePlateDetector que agrupa totes les funcions per detectar, segmentar i reconéixer la matrícula (amb algunes petites variacions)

