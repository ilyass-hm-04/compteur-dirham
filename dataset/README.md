# Données d'entraînement

Le fichier `data.yaml` définit les 11 classes et les chemins attendus. Ajoutez un dataset exporté au format YOLO Object Detection dans les dossiers `train/`, `valid/` et `test/`, chacun contenant `images/` et `labels/`.

Pour chaque image, le label `.txt` correspondant contient une ligne par objet :

```text
class_id x_center y_center width height
```

Les coordonnées sont normalisées entre 0 et 1. Le `class_id` correspond aux clés numériques de `data.yaml`, de 0 à 10. Les dossiers de données eux-mêmes sont ignorés par Git afin de ne pas versionner de grandes images ou des données dont la licence n'a pas été vérifiée.

Les photos classées par valeur ne sont pas des annotations de détection ; il faut dessiner une boîte autour de chaque pièce/billet et exporter les labels YOLO avant l'entraînement.
