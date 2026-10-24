# Compteur d'argent marocain (MAD)

Application Python de détection de pièces et de billets marocains. Elle reprend le flux du notebook `moroccan_dirham_detector.ipynb` sous forme de scripts réutilisables : entraînement et comparaison de modèles YOLO, évaluation, détection sur image/vidéo/webcam et export ONNX.

> **Aucun modèle entraîné n'est fourni.** Les poids et les résultats sont produits après préparation d'un dataset annoté et exécution de l'entraînement. Un détecteur compte des objets dans une image ; les images seules rangées par valeur ne suffisent pas à l'entraîner.

## Classes prises en charge

| Classe | Valeur |
|---|---:|
| `coin_10c` | 0,10 DH |
| `coin_20c` | 0,20 DH |
| `coin_50c` | 0,50 DH |
| `coin_1dh` | 1 DH |
| `coin_2dh` | 2 DH |
| `coin_5dh` | 5 DH |
| `coin_10dh` | 10 DH |
| `note_20dh` | 20 DH |
| `note_50dh` | 50 DH |
| `note_100dh` | 100 DH |
| `note_200dh` | 200 DH |

## 1. Installation

Python 3.10 ou 3.11 est recommandé. Depuis ce dossier, créez un environnement virtuel :

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Pour utiliser un GPU NVIDIA, installez d'abord la version de PyTorch correspondant à votre version de CUDA en suivant le sélecteur officiel de PyTorch, puis installez `requirements.txt`. Sans GPU, le projet utilise le CPU ; l'entraînement peut alors être très lent.

Vérifiez l'installation :

```powershell
python -m mad_counter --help
```

## 2. Préparer les données (indispensable avant l'entraînement)

Le détecteur attend un dataset **YOLO Object Detection** : chaque objet visible doit avoir une boîte englobante et une classe. Utilisez Roboflow, CVAT ou un outil équivalent pour annoter les pièces et billets, puis exportez le dataset au format YOLO.

Placez les fichiers ainsi, en conservant les paires image/annotation :

```text
dataset/
  data.yaml
  train/
    images/
    labels/
  valid/
    images/
    labels/
  test/
    images/
    labels/
```

Le fichier `dataset/data.yaml` fourni est un modèle de configuration. Les images et les fichiers `.txt` de labels correspondants ne sont pas inclus ; ajoutez-les dans les dossiers ci-dessus. Chaque ligne d'un label YOLO doit contenir `class_id x_center y_center width height`, avec les coordonnées normalisées entre 0 et 1. Les identifiants de classe doivent suivre l'ordre présent dans `dataset/data.yaml`.

Le dossier de données source (`dataset/Moroccan Money dataset`, organisé par valeur) contient des photos isolées, pas des scènes réelles annotées. La commande `train-detector` crée des boîtes initiales couvrant chaque photo et des scènes synthétiques de quatre objets pour entraîner, valider et tester le comptage. Les images sources, les jeux de données générés et les images de démonstration restent locaux et ne sont pas inclus dans Git.

> **Limite importante :** les boîtes couvrant les photos et les scènes composites sont des annotations approximatives/générées, pas des boîtes tracées à la main. Le détecteur reconnaît plusieurs objets sur ces exemples, mais n'est pas validé sur des photos naturelles contenant plusieurs monnaies. Pour un comptage fiable, annotez de vraies scènes avec CVAT ou Roboflow, puis utilisez `train` avec le dataset YOLO ci-dessus.

### Entraîner un classificateur de test sur ces photos

En attendant les annotations de détection, les photos classées par valeur permettent d'entraîner un classificateur distinct. Un modèle de test entraîné est fourni dans `models/mad_classifier.pt`. Il prédit la valeur d'un objet isolé dans une photo recadrée ; il **ne détecte pas plusieurs objets** et ne remplace pas `best_mad_detector.pt`.

Évaluation du modèle fourni sur 67 images de test dédupliquées (split réservé, 11 classes) : **77,61 % top-1** et **98,51 % top-5**. Ces scores mesurent la classification de photos isolées, pas la détection ni le comptage dans une scène.

```powershell
python -m mad_counter train-classifier --source "..\dataset\Moroccan Money dataset" --device cpu --imgsz 128
```

Le jeu est dédupliqué par empreinte de fichier avant une séparation entraînement/validation/test. La précision top-1 sur le jeu de test réservé est affichée, puis le modèle est écrit dans `models/mad_classifier.pt`. L'entraînement commence avec les poids classification pré-entraînés Ultralytics `yolo11n-cls.pt`, qui sont téléchargés la première fois.

Testez une photo contenant une seule pièce ou un seul billet :

```powershell
python -m mad_counter classify --weights models/mad_classifier.pt --source chemin\vers\photo_recadree.jpg
```

Les détails de l'entraînement et les métriques sont dans [models/README.md](./models/README.md) et [models/mad_classifier_metrics.json](./models/mad_classifier_metrics.json). Les métriques du classificateur ne mesurent pas la capacité à compter dans une scène réelle.

### Entraîner le détecteur multiobjets avec le jeu de démonstration

Depuis la racine du dépôt :

```powershell
python -m mad_counter train-detector --source "..\dataset\Moroccan Money dataset" --device cpu
```

Cette commande crée un jeu YOLO isolé, répartit les sources sans doublons entre entraînement/validation/test, entraîne sur des photos individuelles et des scènes composites de quatre objets, puis évalue 72 scènes synthétiques de test. Le seuil de confiance par défaut `--conf 0.40` est réglé pour équilibrer détections manquées et doublons sur ce test synthétique ; adaptez-le après annotation de vraies scènes. Le total additionne la valeur des objets détectés dans l'image courante ; un total vidéo n'est pas cumulé entre les frames. Le meilleur checkpoint est enregistré dans `models/best_mad_detector.pt`, les métriques dans `models/mad_detector_metrics.json` et un exemple annoté dans `output/multiobject_demo.jpg` (local uniquement).

Pour tester le total sur une photo de plusieurs objets :

```powershell
python -m mad_counter predict --weights models/best_mad_detector.pt --source chemin\vers\photo.jpg
```

Le test synthétique est utile pour valider le fonctionnement du comptage et la chaîne de détection, mais **ne remplace pas** l'annotation et l'évaluation sur des scènes réelles.

## 3. Entraîner et comparer les modèles

Depuis la racine de ce projet :

```powershell
python -m mad_counter train --data dataset/data.yaml
```

Par défaut, le programme entraîne `yolov8n.pt`, `yolov8s.pt`, `yolo11n.pt` et `yolo11s.pt`, puis les compare sur le split de validation par mAP50-95. Il copie le meilleur modèle dans `models/best_mad_detector.pt` et écrit les métriques dans `runs/model_comparison.csv`.

Paramètres courants :

```powershell
python -m mad_counter train --data dataset/data.yaml --epochs 60 --imgsz 640 --batch 16 --device 0
python -m mad_counter train --data dataset/data.yaml --models yolov8n.pt yolo11n.pt
```

`--device auto` (valeur par défaut) choisit automatiquement le GPU ou le CPU. Utilisez `--device cpu` pour forcer le CPU, ou `--device 0` pour le premier GPU NVIDIA.

## 4. Détecter et compter

Sur une image (résultat annoté par défaut dans `output/`) :

```powershell
python -m mad_counter predict --weights models/best_mad_detector.pt --source chemin\vers\image.jpg
```

Sur la webcam numéro 0 :

```powershell
python -m mad_counter predict --weights models/best_mad_detector.pt --source 0
```

Sur une vidéo :

```powershell
python -m mad_counter predict --weights models/best_mad_detector.pt --source chemin\vers\video.mp4 --output output\video_annotee.mp4
```

Réglez le seuil de confiance avec `--conf` (entre 0 et 1), par exemple `--conf 0.5`. La somme affichée correspond aux objets détectés **dans l'image/instant courant** ; elle n'est pas cumulée d'une image vidéo à l'autre.

## 5. Évaluer et exporter

```powershell
python -m mad_counter evaluate --weights models/best_mad_detector.pt --data dataset/data.yaml
python -m mad_counter export --weights models/best_mad_detector.pt --format onnx
```

L'évaluation affiche précision, rappel, mAP50 et mAP50-95 sur la validation. L'export crée le format demandé à partir des poids sélectionnés.

Lancez les tests unitaires du projet avec :

```powershell
python -m unittest discover -s tests
```

## Arborescence

```text
mad_counter/          Code d'entraînement, de classification, d'inférence et interface CLI
dataset/data.yaml     Configuration YOLO et ordre des classes
models/               Classificateur de démonstration, détecteur entraîné et métriques
runs/                 Journaux, résultats et comparaison (créés localement)
output/               Prédictions annotées (créées localement)
```

## Conseils pour la qualité

- Visez plusieurs centaines d'objets annotés par classe, avec des éclairages, arrière-plans, orientations et faces variés.
- Incluez les séries anciennes et récentes ainsi que des exemples partiellement cachés.
- Vérifiez l'équilibre des classes et les métriques par classe après entraînement.
- Un total erroné peut venir d'une détection manquée ou multiple ; adaptez le dataset et le seuil de confiance avant tout usage opérationnel.

## Licence et données

Ce dépôt ne contient pas de poids pré-entraînés ni de dataset d'entraînement annoté. Vérifiez les licences des images et modèles que vous utilisez. Les résultats dépendent du dataset et ne sont pas garantis.
