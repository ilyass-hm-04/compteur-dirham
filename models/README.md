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

# Détecteur multiobjets de démonstration

`best_mad_detector.pt` est un détecteur YOLO11n entraîné à partir des photos classées par valeur. Il repère plusieurs objets dans une image et additionne leurs valeurs en dirhams pour afficher le total de la frame. `mad_detector_metrics.json` contient les paramètres et les résultats mesurés.

## Test synthétique

| Mesure | Résultat |
|---|---:|
| Images uniques sources après déduplication | 336 |
| Scènes synthétiques test réservées | 72, quatre objets par scène |
| Précision détection mAP50 | 79,71 % |
| mAP50–95 | 68,90 % |
| Scènes au compte exact (seuil 0,40) | 40 / 72 |
| Erreur absolue moyenne du total (seuil 0,40) | 34,83 DH |

Sur un exemple synthétique, le modèle a détecté les quatre objets et calculé `280,00 DH`. Les performances varient selon les images.

**Avertissement :** les annotations de départ couvrent le cadre entier des images source. Les validations et tests multiobjets utilisent des composites artificiels, et non des scènes naturelles annotées manuellement. Ces résultats ne garantissent pas la précision du comptage réel. Pour utiliser ce système avec de l'argent en situation réelle, ajoutez des boîtes annotées sur des scènes réelles et mesurez les erreurs de total sur des exemples indépendants.

## Tester une scène locale

```powershell
python -m mad_counter predict --weights models/best_mad_detector.pt --source chemin\vers\photo.jpg
```

Le projet conserve également le classificateur à objet unique `mad_classifier.pt` ; il ne remplace pas ce détecteur. Les métriques correspondantes sont enregistrées dans `mad_detector_metrics.json`.
