# Modèle de classification de test

`mad_classifier.pt` est un checkpoint Ultralytics YOLO11n-cls entraîné par fine-tuning sur les photos classées par valeur du dataset source fourni à côté du projet. Il accepte une image recadrée contenant une seule pièce ou un seul billet.

## Évaluation

Le dataset a été dédupliqué par SHA-256 avant séparation pour éviter que des copies exactes se retrouvent dans plusieurs splits :

| Split | Images |
|---|---:|
| Entraînement | 215 |
| Validation | 54 |
| Test réservé | 67 |

Sur le split de test, la précision top-1 est de **77,61 %** et la précision top-5 de **98,51 %**. Les mesures détaillées sont disponibles dans `mad_classifier_metrics.json`. Il s'agit d'un petit jeu de classification ; ces scores ne représentent pas les performances en détection multi-objets ou en comptage caméra.

## Utilisation

Depuis la racine du projet :

```powershell
python -m mad_counter classify --weights models/mad_classifier.pt --source chemin\vers\photo_recadree.jpg
```

Le modèle est un artefact binaire de 3,2 Mo. Les dépendances requises figurent dans `../requirements.txt`.
