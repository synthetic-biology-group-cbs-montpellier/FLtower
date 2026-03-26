# Guide de configuration — `parameters.json`

Ce fichier définit les plots que FLtower génère pour chaque puits de votre plaque 96.
Il est lu au lancement et validé automatiquement. En cas d'erreur, un message explicite
vous indique le champ problématique.

## Structure générale

Le fichier contient une ou plusieurs configurations de plot, chacune identifiée par un
nom libre (ex: `plots_config_1`, `gfp_scatter`, etc.) :

```json
{
    "plots_config_1": { ... },
    "plots_config_2": { ... }
}
```

Chaque configuration est soit un **scatter plot**, soit un **histogramme**.

---

## Scatter plot

Un scatter plot affiche deux canaux l'un contre l'autre (ex: GFP vs RFP).

```json
{
    "mon_scatter": {
        "type": "scatter",
        "x_param": "BL1-H",
        "y_param": "YL2-H",
        "x_scale": "log",
        "y_scale": "log",
        "xlim": [1, 200000],
        "ylim": [1, 200000],
        "cmap": "inferno",
        "gridsize": 100,
        "scatter_type": "density",
        "quadrant_gates": {
            "x": 2600,
            "y": 2000
        },
        "96well_plots": [
            {"metric": "Q3_Percentage", "title": "OFF cells Percentage"}
        ],
        "triplicate_plots": [
            {"metric": "Q3_Percentage", "title": "OFF cells Percentage"}
        ]
    }
}
```

### Champs obligatoires

| Champ | Description |
|---|---|
| `type` | Doit être `"scatter"` |
| `x_param` | Nom du canal FCS pour l'axe X (ex: `"BL1-H"`, `"FSC-A"`) |
| `y_param` | Nom du canal FCS pour l'axe Y (ex: `"YL2-H"`, `"SSC-A"`) |

### Champs optionnels

| Champ | Défaut | Description |
|---|---|---|
| `x_scale` | `"linear"` | Échelle de l'axe X : `"linear"` ou `"log"` |
| `y_scale` | `"linear"` | Échelle de l'axe Y : `"linear"` ou `"log"` |
| `xlim` | auto | Limites de l'axe X, ex: `[1, 200000]` |
| `ylim` | auto | Limites de l'axe Y, ex: `[1, 200000]` |
| `cmap` | `"viridis"` | Palette de couleurs matplotlib (ex: `"inferno"`, `"plasma"`, `"coolwarm"`) |
| `gridsize` | `100` | Résolution du hexbin pour les density plots. Plus élevé = plus fin. Doit être > 0 |
| `scatter_type` | `"scatter"` | `"scatter"` (points individuels) ou `"density"` (hexbin) |
| `quadrant_gates` | auto (médiane) | Position des gates de quadrant : `{"x": valeur, "y": valeur}`. Si absent, la médiane est utilisée |
| `96well_plots` | `[]` | Liste de métriques à afficher en vue 96 puits (voir ci-dessous) |
| `triplicate_plots` | `[]` | Liste de métriques à afficher en vue triplicats (voir ci-dessous) |

### Métriques disponibles pour le scatter

Les quadrants sont numérotés dans le sens trigonométrique :
- **Q1** : haut-droite (X ≥ gate, Y ≥ gate) — double positif
- **Q2** : haut-gauche (X < gate, Y ≥ gate)
- **Q3** : bas-gauche (X < gate, Y < gate) — double négatif
- **Q4** : bas-droite (X ≥ gate, Y < gate)

Métriques utilisables dans `96well_plots` et `triplicate_plots` :
- `Q1_Percentage`, `Q2_Percentage`, `Q3_Percentage`, `Q4_Percentage`
- `Global_{canal}_Median` (ex: `Global_BL1-H_Median`)

---

## Histogramme

Un histogramme affiche la distribution d'un seul canal.

```json
{
    "mon_histo": {
        "type": "histogram",
        "x_param": "BL1-H",
        "x_scale": "log",
        "xlim": [1, 200000],
        "color": "seagreen",
        "kde": true,
        "gates": [[10, 800], [800, 100000]],
        "96well_plots": [
            {"metric": "Global_Median", "title": "GFP Median"}
        ],
        "triplicate_plots": [
            {"metric": "Global_Median", "title": "GFP-Median"}
        ]
    }
}
```

### Champs obligatoires

| Champ | Description |
|---|---|
| `type` | Doit être `"histogram"` |
| `x_param` | Nom du canal FCS (ex: `"BL1-H"`) |

### Champs optionnels

| Champ | Défaut | Description |
|---|---|---|
| `x_scale` | `"linear"` | Échelle de l'axe X : `"linear"` ou `"log"` |
| `xlim` | auto | Limites de l'axe X, ex: `[1, 200000]` |
| `color` | `"blue"` | Couleur matplotlib (ex: `"seagreen"`, `"coral"`, `"#FF5733"`) |
| `kde` | `false` | Superposer une courbe de densité (kernel density estimation) |
| `gates` | aucun | Liste de gates intervalles, ex: `[[10, 800], [800, 100000]]`. Chaque gate = `[min, max]` |
| `96well_plots` | `[]` | Liste de métriques à afficher en vue 96 puits |
| `triplicate_plots` | `[]` | Liste de métriques à afficher en vue triplicats |

### Métriques disponibles pour l'histogramme

- `Global_Median` — médiane de tout le canal
- Métriques par gate si des gates sont définies

---

## Vues 96 puits et triplicats

Les champs `96well_plots` et `triplicate_plots` définissent quelles métriques
seront visualisées en heatmap 96 puits et en barplot de triplicats.

Chaque entrée a deux champs obligatoires :

```json
{"metric": "Q3_Percentage", "title": "OFF cells Percentage"}
```

| Champ | Description |
|---|---|
| `metric` | Nom exact de la métrique calculée par FLtower |
| `title` | Titre affiché sur le plot et dans le nom de fichier |

---

## Exemple complet

Voici un fichier typique pour une expérience GFP + RFP :

```json
{
    "plots_config_1": {
        "type": "scatter",
        "x_param": "BL1-H",
        "y_param": "YL2-H",
        "x_scale": "log",
        "y_scale": "log",
        "xlim": [1, 200000],
        "ylim": [1, 200000],
        "cmap": "inferno",
        "gridsize": 100,
        "scatter_type": "density",
        "quadrant_gates": {"x": 2600, "y": 2000},
        "96well_plots": [
            {"metric": "Q3_Percentage", "title": "OFF cells Percentage"},
            {"metric": "Q2_Percentage", "title": "RFP cells Percentage"},
            {"metric": "Q4_Percentage", "title": "GFP cells Percentage"}
        ],
        "triplicate_plots": [
            {"metric": "Q3_Percentage", "title": "OFF cells Percentage"},
            {"metric": "Q2_Percentage", "title": "RFP cells Percentage"},
            {"metric": "Q4_Percentage", "title": "GFP cells Percentage"}
        ]
    },
    "plots_config_2": {
        "type": "histogram",
        "x_param": "BL1-H",
        "x_scale": "log",
        "xlim": [1, 200000],
        "color": "seagreen",
        "kde": true,
        "gates": [[10, 800], [800, 100000]],
        "96well_plots": [
            {"metric": "Global_Median", "title": "GFP Histogram-Median"}
        ],
        "triplicate_plots": [
            {"metric": "Global_Median", "title": "GFP-Median"}
        ]
    },
    "plots_config_3": {
        "type": "histogram",
        "x_param": "YL2-H",
        "x_scale": "log",
        "xlim": [1, 200000],
        "color": "coral",
        "kde": true,
        "gates": [[10, 800], [800, 100000]],
        "96well_plots": [
            {"metric": "Global_Median", "title": "RFP Histogram-Median"}
        ],
        "triplicate_plots": [
            {"metric": "Global_Median", "title": "RFP-Median"}
        ]
    }
}
```

## Trouver les noms de canaux

Les noms de canaux (ex: `BL1-H`, `YL2-H`, `FSC-A`, `SSC-A`) dépendent de votre
cytomètre. Pour les trouver, ouvrez un fichier `.fcs` dans FlowJo ou utilisez :

```python
import fcsparser
meta, data = fcsparser.parse("votre_fichier.fcs")
print(list(data.columns))
```

## Erreurs courantes

| Erreur | Cause | Solution |
|---|---|---|
| `"type" is not "scatter" or "histogram"` | Type de plot inconnu | Vérifier l'orthographe |
| `"y_param" field required` | Scatter sans axe Y | Ajouter `"y_param"` ou changer `type` en `"histogram"` |
| `"x_scale" input should be "linear" or "log"` | Échelle invalide | Utiliser `"linear"` ou `"log"` uniquement |
| `"gridsize" should be greater than 0` | gridsize à 0 ou négatif | Mettre une valeur positive (100 recommandé) |
| `"title" field required` in 96well_plots | Entrée incomplète | Chaque entrée doit avoir `"metric"` ET `"title"` |
