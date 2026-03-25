# Analyse de cytométrie en flux — Guide complet

Ce document décrit **l'ensemble des étapes d'une analyse de cytométrie en flux**, depuis l'acquisition sur le cytomètre jusqu'à l'interprétation statistique. Il ne se limite pas à ce que fait FLtower aujourd'hui : il couvre ce que fait un logiciel comme FlowJo, et sert de feuille de route pour les développements futurs.

Il est destiné aux **biologistes utilisateurs** et aux **développeurs** qui contribuent au projet.

---

## 1. La cytométrie en flux en 2 minutes

La cytométrie en flux fait passer des **cellules une par une** dans un (ou plusieurs) faisceau(x) laser. Chaque cellule produit des signaux :

| Signal | Ce qu'il mesure | Exemple de canal |
|---|---|---|
| **FSC** (Forward Scatter) | Taille de la cellule | FSC-A, FSC-H |
| **SSC** (Side Scatter) | Granularité / complexité interne | SSC-A, SSC-H |
| **Fluorescence** | Intensité d'un marqueur fluorescent | BL1-H (GFP), YL2-H (RFP), VL1-H (BFP) |

Chaque signal a deux variantes essentielles :
- **-A** (Area) : aire du pulse, proportionnelle à la quantité totale de signal
- **-H** (Height) : hauteur du pic, proportionnelle à l'intensité maximale instantanée

Un fichier **`.fcs`** (Flow Cytometry Standard) enregistre ces valeurs pour chaque **événement** (= chaque passage devant le laser). Un fichier contient typiquement **10 000 à 100 000 événements**.

### Le cas multi-couleurs (2, 3 couleurs ou plus)

Les cytomètres modernes possèdent **plusieurs lasers** et **plusieurs détecteurs**. Par exemple :

| Laser | Détecteur | Fluorochrome typique | Usage courant |
|---|---|---|---|
| Bleu (488nm) | BL1-H | GFP, FITC | Marqueur vert |
| Jaune (561nm) | YL2-H | mCherry, RFP, PE | Marqueur rouge |
| Violet (405nm) | VL1-H | BFP, Pacific Blue | Marqueur bleu |

Avec 3 couleurs, on peut suivre **3 gènes rapporteurs** en parallèle dans la même cellule — par exemple un circuit génétique avec un activateur (GFP), un répresseur (RFP) et un marqueur de sélection (BFP).

**Enjeu** : plus on a de couleurs, plus les spectres d'émission se chevauchent, et plus la **compensation** devient critique (voir étape 4).

---

## 2. Du cytomètre au logiciel : le workflow complet

Voici les **10 étapes** d'une analyse complète, de l'acquisition à la publication. Dans FlowJo, l'utilisateur les fait manuellement, puits par puits. L'objectif d'un outil comme FLtower est de les **automatiser pour une plaque entière**.

```
 ╔══════════════════════════════════════════════════════════════════╗
 ║                    PIPELINE D'ANALYSE COMPLET                    ║
 ╠══════════════════════════════════════════════════════════════════╣
 ║                                                                  ║
 ║   Cytomètre → Fichiers .fcs (1 par puits, 96 fichiers)          ║
 ║        │                                                         ║
 ║        ▼                                                         ║
 ║   ┌─────────────────────────────────┐                            ║
 ║   │  1. Lecture & Validation FCS    │                            ║
 ║   └────────────┬────────────────────┘                            ║
 ║                ▼                                                  ║
 ║   ┌─────────────────────────────────┐                            ║
 ║   │  2. Contrôle qualité (QC)       │                            ║
 ║   └────────────┬────────────────────┘                            ║
 ║                ▼                                                  ║
 ║   ┌─────────────────────────────────┐                            ║
 ║   │  3. Transformation des données  │  logicle / arcsinh         ║
 ║   └────────────┬────────────────────┘                            ║
 ║                ▼                                                  ║
 ║   ┌─────────────────────────────────┐                            ║
 ║   │  4. Compensation spectrale      │  (si multi-couleurs)      ║
 ║   └────────────┬────────────────────┘                            ║
 ║                ▼                                                  ║
 ║   ┌─────────────────────────────────┐                            ║
 ║   │  5. Gating hiérarchique         │                            ║
 ║   │    5a. Debris gate (FSC/SSC)    │                            ║
 ║   │    5b. Singlet gate (SSC-H/A)   │                            ║
 ║   │    5c. Gate de viabilité        │  (optionnel)               ║
 ║   └────────────┬────────────────────┘                            ║
 ║                ▼                                                  ║
 ║   ┌─────────────────────────────────┐                            ║
 ║   │  6. Analyse de fluorescence     │                            ║
 ║   │    6a. Histogrammes (1 canal)   │                            ║
 ║   │    6b. Scatter plots (2 canaux) │                            ║
 ║   │    6c. 3D / multi-paramétrique  │  (>2 canaux)              ║
 ║   └────────────┬────────────────────┘                            ║
 ║                ▼                                                  ║
 ║   ┌─────────────────────────────────┐                            ║
 ║   │  7. Statistiques par puits      │  GM, médiane, CV, %       ║
 ║   └────────────┬────────────────────┘                            ║
 ║                ▼                                                  ║
 ║   ┌─────────────────────────────────┐                            ║
 ║   │  8. Vue plaque (Heatmap)        │  Vision globale 96 puits  ║
 ║   └────────────┬────────────────────┘                            ║
 ║                ▼                                                  ║
 ║   ┌─────────────────────────────────┐                            ║
 ║   │  9. Statistiques de réplicats   │  Mean ± Std, tests stats  ║
 ║   └────────────┬────────────────────┘                            ║
 ║                ▼                                                  ║
 ║   ┌─────────────────────────────────┐                            ║
 ║   │ 10. Rapport & Reproductibilité  │  PDF, provenance, export  ║
 ║   └─────────────────────────────────┘                            ║
 ║                                                                  ║
 ╚══════════════════════════════════════════════════════════════════╝
```

---

## 3. Chaque étape en détail

---

### Étape 1 — Lecture & Validation des fichiers FCS

**Ce qu'on fait** : On ouvre chaque fichier `.fcs`, on extrait les données brutes (un tableau : lignes = événements, colonnes = canaux), et on lit les **métadonnées** (canaux disponibles, voltages, nom de l'échantillon, date d'acquisition, matrice de compensation embarquée).

**Pourquoi** : Le fichier FCS est le standard universel (FCS 2.0, 3.0, 3.1). Il contient non seulement les données mais aussi des informations critiques sur les conditions d'acquisition.

**Enjeux** :

| Problème | Conséquence |
|---|---|
| Fichier corrompu | Perte du puits |
| Canaux attendus absents | Analyse impossible sur ce canal |
| Version FCS non supportée | Erreur de parsing |
| Nom de fichier mal formé | Impossible d'identifier la position du puits sur la plaque |

**Ce que fait FlowJo** : Lecture robuste de toutes les versions FCS, affichage des métadonnées, détection automatique des canaux.

**Recommandation** : Valider au chargement que les canaux requis par la configuration existent dans le fichier. Alerter clairement si un fichier est manquant ou illisible.

---

### Étape 2 — Contrôle qualité (QC)

**Ce qu'on fait** : Avant toute analyse, on vérifie la **qualité des données brutes** :

- **Nombre d'événements suffisant** — un puits avec 50 événements n'est pas exploitable
- **Stabilité temporelle** — le signal ne doit pas dériver au cours de l'acquisition (détectable en traçant l'intensité vs. le temps)
- **Valeurs saturées (margin events)** — événements où le signal atteint le maximum du détecteur (ex: 262143 pour un ADC 18 bits). Ces cellules sont « hors échelle » et leur vraie intensité est inconnue
- **Valeurs nulles ou négatives** — artefacts électroniques courants, surtout sur les canaux de fluorescence

```
    Signal
    ──────── max détecteur (262144) ► événement saturé = perdu
    │
    │  xxxxxxxxx         x       population normale
    │  xxxxxxxxx      xxxx
    │  xxxxxxxxxxx  xxxxxx
    │  xxxxxxxxxxxxxxxx
    ──────── 0 ──────────────────── Temps d'acquisition
    │  xx                         ► valeurs négatives = artefact
```

**Enjeu** : Sans QC, on peut calculer des statistiques sur des données aberrantes sans s'en rendre compte. Un puits avec un problème d'acquisition fausse les moyennes du triplicat.

**Ce que fait FlowJo** : Affichage du nombre d'événements, détection visuelle par l'utilisateur. Certains plugins ajoutent un QC automatique (flowAI, PeacoQC).

**Recommandation** : Calculer et afficher automatiquement pour chaque puits : nombre total d'événements, % de margin events, % de valeurs négatives. Drapeau si un puits est en dessous d'un seuil configurable.

---

### Étape 3 — Transformation des données

**Ce qu'on fait** : On transforme les valeurs brutes pour les rendre **visualisables et analysables**. C'est une étape mathématique fondamentale, souvent invisible pour l'utilisateur mais critique.

#### Pourquoi ne pas simplement utiliser le log ?

Les données de fluorescence varient sur **5 à 6 ordres de grandeur** (de ~1 à ~250 000). L'échelle logarithmique est naturelle pour les visualiser. Mais le log a un problème : **log(0) = -∞** et **log(négatif) = impossible**.

Or en cytométrie, les valeurs proches de zéro et légèrement négatives sont **fréquentes et normales** — elles correspondent à des cellules non fluorescentes dont le signal fluctue autour du bruit de fond.

#### Les transformations standards

| Transformation | Gère les valeurs ≤ 0 ? | Usage |
|---|---|---|
| **log10** | Non — crash ou clip | Simple mais imprécis près de zéro |
| **logicle** (bi-exponentielle) | Oui | Standard FlowJo. Linéaire près de zéro, log loin de zéro |
| **arcsinh** (sinus hyperbolique inverse) | Oui | Alternative plus simple. Paramètre = cofacteur |
| **hyperlog** | Oui | Variante moins courante |

```
  Valeur transformée
    │
    │            ╱ log classique (ne fonctionne pas ici)
    │           ╱
    │         ╱╱╱╱ logicle (transition douce)
    │       ╱╱
    │     ╱╱
    │────╱───────── zone linéaire (près de zéro)
    │  ╱
    └──────────────── Valeur brute
   -100  0  100          10000        100000
```

**Enjeu majeur** : Si on utilise `log` + `clip(lower=1)` (ce que fait le code actuel), toutes les cellules avec une fluorescence entre -100 et 1 sont **écrasées à la même valeur**. On perd la résolution dans la population négative. La logicle résout ce problème en gardant une échelle linéaire près de zéro.

**Ce que fait FlowJo** : Logicle par défaut pour la fluorescence, linéaire pour FSC/SSC.

**Recommandation** : Implémenter la transformation logicle ou arcsinh. Le paramètre de transition (largeur de la zone linéaire) doit être configurable. L'utilisateur doit pouvoir choisir la transformation par canal.

---

### Étape 4 — Compensation spectrale

**Ce qu'on fait** : On corrige le **chevauchement spectral** entre les fluorochromes.

#### Le problème

Chaque fluorochrome émet de la lumière sur une plage de longueurs d'onde, pas sur une seule. Le signal détecté dans le canal « vert » contient un peu de lumière du fluorochrome « rouge » — et inversement.

```
  Émission
    │
    │      GFP          RFP
    │     ╱╲            ╱╲
    │    ╱  ╲    ╱╲    ╱  ╲
    │   ╱    ╲  ╱  ╲  ╱    ╲
    │  ╱      ╲╱    ╲╱      ╲
    └──────────────────────────── λ (nm)
         500    550    600    650
              │          │
         Détecteur    Détecteur
           BL1          YL2
              │          │
              ▼          ▼
         GFP + un     RFP + un
         peu de RFP   peu de GFP
```

Avec **2 couleurs**, c'est gérable. Avec **3 couleurs**, les chevauchements se multiplient et la compensation devient indispensable :

| | Détecté dans BL1 (vert) | Détecté dans YL2 (rouge) | Détecté dans VL1 (bleu) |
|---|---|---|---|
| **GFP émet** | **Signal vrai** | Fuite (spillover) | Fuite |
| **RFP émet** | Fuite | **Signal vrai** | Fuite |
| **BFP émet** | Fuite | Fuite | **Signal vrai** |

La **matrice de compensation** corrige ces fuites. Elle est souvent **embarquée dans le fichier FCS** (champ `$SPILLOVER` ou `$COMP`) si le biologiste l'a calibrée sur le cytomètre.

**Quand c'est nécessaire** :
- **Toujours** si on utilise 2+ fluorochromes dont les spectres se chevauchent
- Peut être omis si le cytomètre a déjà appliqué la compensation avant l'export (indiqué dans les métadonnées)

**Enjeu** : Sans compensation, une cellule fortement GFP+ apparaît artificiellement RFP+. Les pourcentages dans les quadrants sont faux.

**Ce que fait FlowJo** : Lecture automatique de la matrice, compensation interactive, visualisation avant/après.

**Recommandation** : Lire la matrice `$SPILLOVER` du FCS. Appliquer la compensation si elle n'est pas déjà faite. Afficher un avertissement si aucune matrice n'est trouvée et que >1 canal de fluorescence est analysé. Permettre à l'utilisateur de fournir sa propre matrice.

---

### Étape 5 — Gating hiérarchique

Le **gating** est l'opération la plus importante en cytométrie. C'est l'équivalent du « tri » des données : on sélectionne des sous-populations d'événements en appliquant des filtres successifs.

#### 5a. Debris gate (FSC-A vs SSC-A)

**Ce qu'on fait** : On trace la taille (FSC) contre la granularité (SSC) et on exclut les **débris** — petites particules qui ne sont pas des cellules intactes.

```
    SSC-A (granularité)
    │
    │     x            x
    │   xxxxx      cellules
    │  xxxxxxx     intactes ─── on GARDE ✓
    │   xxxxx     ╱
    │     x      ╱
    │  ─ ─ ─ ─╱─ ─ ─ ─ ─
    │  . . .   débris
    │  . . . .            ─── on EXCLUT ✗
    └──────────────────── FSC-A (taille)
```

**Pourquoi** : Les débris cellulaires (morceaux de membrane, agrégats protéiques) ont une fluorescence non spécifique qui pollue l'analyse. C'est la **première étape de gating dans FlowJo** — et FLtower ne la fait pas actuellement.

**Enjeu** : Sans debris gate, les populations « OFF » (Q3) sont artificiellement gonflées par des débris non fluorescents.

#### 5b. Singlet gate (SSC-H vs SSC-A)

**Ce qu'on fait** : On garde uniquement les **cellules individuelles** (singlets) en éliminant les doublets.

```
    SSC-H
    │        ╱
    │       ╱ ligne ratio = 1
    │    xxxx╱xxx
    │   xxxxxxxx  ← singlets (ratio SSC-H/SSC-A ≈ 1)
    │    xxxx╱xxx
    │       ╱
    │   xx ╱        ← doublets (ratio s'écarte)
    │     ╱ xx
    └──────────── SSC-A
```

**Comment** : On calcule le rapport SSC-H / SSC-A. Pour un singlet, le pulse a une forme régulière et le rapport est proche de 1. Pour un doublet, le pulse est allongé et le rapport s'écarte.

**Seuils typiques** : ratio entre 0.7 et 1.3 (à ajuster selon le cytomètre et le type cellulaire).

**Pourquoi c'est critique** : Un doublet de deux cellules (une GFP+, une RFP+) sera détecté comme **une cellule double-positive** (GFP+ ET RFP+). Cela crée des faux positifs dans le quadrant Q1.

**Variante** : On peut aussi utiliser **FSC-H vs FSC-A** comme gate supplémentaire ou alternatif.

#### 5c. Gate de viabilité (optionnel)

Certaines expériences utilisent un **colorant de viabilité** (ex : DAPI, propidium iodide, LIVE/DEAD). Les cellules mortes absorbent le colorant et deviennent fluorescentes dans un canal spécifique. On les exclut.

#### La hiérarchie des gates

L'ordre compte. Chaque gate s'applique sur les événements restants du gate précédent :

```
  Tous les événements (ex: 50 000)
    │
    ├─ Debris gate → 42 000 restants (84%)
    │   │
    │   ├─ Singlet gate → 38 000 restants (90% des cellules)
    │   │   │
    │   │   ├─ [Gate viabilité] → 36 000 restants (95%)
    │   │   │   │
    │   │   │   └─ Analyse de fluorescence sur 36 000 cellules propres
```

**Enjeu de traçabilité** : À chaque étape, on doit documenter **combien d'événements** passent le gate. Si 80% sont éliminés, c'est un signal d'alarme sur la qualité de l'échantillon.

**Ce que fait FlowJo** : Arbre de gating interactif avec compteurs à chaque niveau. L'utilisateur voit l'effet de chaque gate.

**Recommandation** : Implémenter un gating hiérarchique avec :
- Un compteur à chaque niveau
- Un résumé dans le rapport (tableau : événements totaux → post-debris → post-singlet → analysés)
- Tous les seuils configurables dans le fichier de paramètres

---

### Étape 6 — Analyse de fluorescence

Une fois les cellules filtrées (debris out, singlets only), on analyse les **canaux de fluorescence**.

#### 6a. Histogrammes (1 canal)

**Ce qu'on fait** : On trace la **distribution** d'un canal (ex: GFP) pour une population.

```
    Nombre de cellules
    │
    │        ╭───╮
    │       ╱     ╲        Population GFP+
    │     ╱╱       ╲╲
    │ ───╱───────────╲──────
    │   ╱               ╲
    └─────────────────────── Fluorescence (logicle)
       GFP-    │    GFP+
             Gate
```

**Métriques clés** :
- **Médiane** : valeur qui coupe la population en 2 (50%)
- **Moyenne géométrique (GM)** : tendance centrale adaptée aux distributions log-normales
- **CV (coefficient de variation)** : dispersions relative = σ/μ × 100
- **rCV (robust CV)** : rSD / médiane × 100 — plus résistant aux outliers
- **% dans un gate** : proportion de cellules entre deux seuils

**Pourquoi la GM et pas la moyenne arithmétique** :
Les données de fluorescence suivent une **distribution log-normale** — elles sont symétriques en échelle log. La moyenne arithmétique est tirée vers le haut par les cellules très brillantes. La GM est plus représentative de la tendance centrale.

```
  Exemple (3 cellules : 100, 1000, 10000)
  Moyenne arithmétique = 3700    ← tirée par le 10000
  Moyenne géométrique  = 1000   ← au centre en échelle log ✓
```

#### 6b. Scatter plots 2D (2 canaux)

**Ce qu'on fait** : On croise deux canaux de fluorescence. On divise en **quadrants** pour classer les cellules.

```
    Canal Y (ex: RFP)
    │
    │  Q2 (Y+ X-)  │  Q1 (X+ Y+)
    │   RFP only    │   double positif
    │               │
    ├───────────────┼──────── seuil Y
    │               │
    │  Q3 (X- Y-)   │  Q4 (X+ Y-)
    │   négatif/OFF │   GFP only
    │               │
    └───────────────┴──────── Canal X (ex: GFP)
                    seuil X
```

**Métriques clés** : % de cellules dans chaque quadrant, GM et médiane par quadrant.

**Types de visualisation** :
- **Dot plot** : un point par cellule. Clair pour <10 000 événements.
- **Density plot (hexbin/heatmap)** : couleur = densité de cellules. Préférable pour >10 000 événements.
- **Contour plot** : lignes de niveau de densité. Bon pour superposer des conditions.

#### 6c. Analyse multi-paramétrique (≥3 canaux)

Avec 3 couleurs, on a **3 combinaisons de scatter plots** possibles :

| Plot | Canaux | Ce qu'on visualise |
|---|---|---|
| GFP vs RFP | BL1-H vs YL2-H | Expression des deux rapporteurs |
| GFP vs BFP | BL1-H vs VL1-H | Activateur vs marqueur de sélection |
| RFP vs BFP | YL2-H vs VL1-H | Répresseur vs marqueur de sélection |

On peut aussi envisager :
- **Ternary plots** (diagrammes ternaires) pour visualiser les 3 canaux simultanément
- **Clustering automatique** (FlowSOM, Phenograph) pour identifier des sous-populations sans gates manuels — mais c'est hors scope pour un outil simple

**Recommandation** : Supporter N canaux de fluorescence configurables (pas seulement 2). Chaque combinaison de 2 canaux peut être un scatter plot. Les histogrammes sont générés pour chaque canal individuellement.

---

### Étape 7 — Statistiques par puits

Pour chaque puits et chaque configuration de plot, on calcule :

| Métrique | Définition | Usage |
|---|---|---|
| **Nombre d'événements** | Cellules après tous les gates | QC — trop peu = problème |
| **Médiane** | 50e percentile | Tendance centrale (robuste) |
| **Geometric Mean (GM)** | $\exp(\frac{1}{n}\sum \ln x_i)$ | Tendance centrale (standard cytométrie) |
| **CV** | $\frac{\sigma}{\mu} \times 100\%$ | Dispersion relative |
| **rCV** | $\frac{rSD}{\text{médiane}} \times 100\%$ | Dispersion robuste (résiste aux outliers) |
| **% dans gate/quadrant** | proportion de cellules | Classification des populations |
| **MFI** | Mean/Median Fluorescence Intensity | Terme générique pour GM ou médiane |

Toutes ces métriques sont exportées dans des **fichiers CSV** — un par configuration de plot.

---

### Étape 8 — Vue plaque 96 puits (Heatmap)

**Ce qu'on fait** : On projette une **métrique** (ex: médiane GFP, % Q3) sur le plan de la plaque. Chaque puits est coloré selon la valeur.

```
      1     2     3     4     5     6     7     8     9    10    11    12
  A  🟢   🟢   🟢   🔴   🔴   🔴   🟡   🟡   🟡   ⚫   ⚫   ⚫
  B  🟢   🟢   🟢   🔴   🔴   🔴   🟡   🟡   🟡   ⚫   ⚫   ⚫
  C  🟡   🟡   🟡   🟡   🟡   🟡   🟡   🟡   🟡   🟡   🟡   🟡
  ...
  H  🔴   🔴   🔴   🟢   🟢   🟢   🟡   🟡   🟡   🟢   🟢   🟢
      ──────────  ──────────  ──────────  ──────────
      triplicat 1 triplicat 2 triplicat 3 triplicat 4
```

**Pourquoi c'est indispensable** : Cette vue permet de repérer en 1 seconde :

| Pattern | Cause probable |
|---|---|
| Gradient gauche→droite | Effet de bord, temps d'incubation variable |
| Un puits aberrant dans un triplicat | Erreur de pipetage, bulle |
| Toute une rangée aberrante | Problème systématique (pipette multicanaux) |
| Pas de différence entre conditions | Expérience n'a pas fonctionné |

**Avantage clé sur FlowJo** : FlowJo analyse puits par puits. Il n'offre **aucune vue d'ensemble** de la plaque. C'est LE point fort d'un outil comme FLtower.

---

### Étape 9 — Statistiques de réplicats

**Ce qu'on fait** : Les puits sont regroupés par **réplicats** (typiquement 3 puits = 1 condition biologique en triplicat). Pour chaque groupe :

- **Moyenne** des réplicats
- **Écart-type (SD)** = variabilité technique
- **Coefficient de variation (CV)** = SD/Moyenne × 100%

**Pourquoi** : Une mesure unique n'est jamais fiable en biologie. Les triplicats permettent de distinguer un **vrai signal biologique** d'un **artefact technique**.

**Conventions de layout** :

| Convention | Description |
|---|---|
| **Colonnes par 3** | Colonnes 1-2-3 = triplicat 1, 4-5-6 = triplicat 2, etc. (4 conditions par rangée) |
| **Rangées identiques** | Chaque rangée = une construction/souche différente |
| **Layout libre** | L'utilisateur fournit une table de correspondance puits→condition |

**Enjeu** : La convention de groupement doit être claire et configurable. Si l'utilisateur arrange ses triplicats différemment (ex: par rangée au lieu de par colonne), le calcul est faux.

**Perspectives** : Pour des analyses plus poussées, on pourrait ajouter des **tests statistiques** (t-test, ANOVA, Mann-Whitney) pour comparer les conditions. Ces tests nécessitent au minimum des triplicats.

---

### Étape 10 — Rapport & Reproductibilité

Un résultat scientifique n'a de valeur que s'il est **reproductible**. Le rapport doit contenir tout ce qui est nécessaire pour que quelqu'un d'autre puisse reproduire exactement la même analyse.

**Contenu indispensable** :

| Élément | Pourquoi |
|---|---|
| Version du logiciel | Pour retrouver le code exact utilisé |
| Configuration complète (paramètres) | Tous les seuils, gates, transformations |
| Hash SHA256 des fichiers d'entrée | Prouve qu'on a analysé les bons fichiers |
| Versions des dépendances clés | numpy, scipy, etc. peuvent changer les résultats numériques |
| Date et heure | Horodatage de l'analyse |
| Résumé du gating | % filtré à chaque étape, par puits |
| Fichiers CSV complets | Données exploitables dans d'autres outils (R, Excel, etc.) |

**Formats de rapport** :
- **PDF** : pour partager, imprimer, archiver
- **HTML** (interactif) : pour explorer les résultats avec des plots zoomables
- **CSV** : pour l'exploitation programmatique (R, Python, Excel)

**Recommandation** : Le rapport PDF doit être autonome — tout ce qu'il faut pour interpréter les résultats doit être dedans, sans avoir à ouvrir d'autres fichiers.

---

## 4. FLtower vs FlowJo — positionnement

| Critère | FlowJo | FLtower (objectif) |
|---|---|---|
| **Prix** | ~7 500 USD/licence | Gratuit, open-source |
| **Interface** | GUI interactive | CLI + rapport automatique |
| **Analyse par puits** | Oui (manuelle) | Oui (automatisée) |
| **Vue plaque 96 puits** | Non | **Oui — avantage clé** |
| **Batch processing** | Limité (copier des gates) | **Natif — c'est le principe** |
| **Triplicats** | Manuel (export + Excel) | **Automatique** |
| **Compensation** | Oui (interactive) | À implémenter |
| **Transformation logicle** | Oui (par défaut) | À implémenter |
| **Gating interactif** | Oui (drag & drop) | Non (seuils dans fichier config) |
| **Reproductibilité** | Faible (dépend de l'utilisateur) | **Forte (automatisée, versionnée)** |
| **Multi-couleurs (≥3)** | Oui | En cours |
| **Extensible** | Plugins limités | **Code Python, intégrable dans un pipeline** |

**Positionnement stratégique** : FLtower **ne remplace pas FlowJo** pour l'exploration interactive d'un échantillon. Il le **remplace pour l'analyse standardisée de plaques entières**, là où FlowJo est lent, manuel et non reproductible.

---

## 5. Glossaire complet

| Terme | Définition |
|---|---|
| **FCS** | Flow Cytometry Standard — format de fichier universel (versions 2.0, 3.0, 3.1) |
| **Événement** | Un signal enregistré par le cytomètre (1 cellule, 1 débris, ou 1 doublet) |
| **Gate / Gating** | Filtre pour sélectionner une sous-population. Peut être un rectangle, un polygone, un seuil 1D, ou un quadrant |
| **Singlet** | Cellule individuelle (par opposition à un doublet/agrégat) |
| **Doublet** | Deux cellules collées passant ensemble devant le laser. Fausse les mesures |
| **FSC** | Forward Scatter — proportionnel à la taille. Variantes : FSC-A (aire), FSC-H (hauteur) |
| **SSC** | Side Scatter — proportionnel à la granularité. Variantes : SSC-A, SSC-H |
| **Spillover** | Fuite de fluorescence d'un fluorochrome dans le détecteur d'un autre |
| **Compensation** | Correction mathématique du spillover via une matrice de compensation |
| **Logicle** | Transformation bi-exponentielle standard en cytométrie — linéaire près de zéro, log loin de zéro |
| **Arcsinh** | Alternative à la logicle : $f(x) = \text{arcsinh}(x/\text{cofacteur})$ |
| **GFP** | Green Fluorescent Protein — protéine fluorescente verte, marqueur courant |
| **RFP** | Red Fluorescent Protein (mCherry, DsRed) — marqueur rouge |
| **BFP** | Blue Fluorescent Protein (TagBFP, mTagBFP2) — marqueur bleu |
| **GM (Geometric Mean)** | Moyenne géométrique — $\exp(\frac{1}{n}\sum \ln x_i)$ — tendance centrale pour données log-normales |
| **MFI** | Mean/Median Fluorescence Intensity — terme générique pour la fluorescence typique d'une population |
| **Médiane** | 50e percentile — sépare la population en deux moitiés égales |
| **CV** | Coefficient of Variation — $\sigma / \mu \times 100\%$ — dispersion relative |
| **rCV** | Robust CV — basé sur la médiane et l'IQR, résiste aux outliers |
| **Quadrant** | Division d'un scatter 2D en 4 zones par 2 seuils orthogonaux |
| **Triplicat** | 3 répétitions de la même condition. Minimum pour une statistique fiable |
| **Plaque 96 puits** | Support standard : 8 rangées (A–H) × 12 colonnes. Format compatible pipettes multicanaux |
| **FMO** | Fluorescence Minus One — contrôle où on omet un fluorochrome pour calibrer les gates |
| **Margin event** | Événement dont le signal est au min (0) ou max (262143) du détecteur — valeur réelle inconnue |
| **ADC** | Analog-to-Digital Converter — convertisseur de signal. Résolution typique : 18 bits (0–262143) |
