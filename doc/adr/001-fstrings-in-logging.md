# ADR-001 : Utilisation de f-strings dans les appels logging

**Date** : 2026-03-26  
**Statut** : Accepté  

## Contexte

Le module `logging` de Python recommande le style `%` (lazy formatting) pour différer
l'interpolation des chaînes et éviter un coût de formatage lorsque le message est filtré
par le niveau du logger. pylint signale les f-strings dans les loggers via la règle
`W1203 (logging-fstring-interpolation)`.

## Décision

Nous utilisons les **f-strings** dans tous les appels `logger.*()` de FLtower.

## Justification

1. **Le file handler est toujours à DEBUG** — chaque message est systématiquement formaté
   et écrit dans le fichier de log, rendant l'évaluation paresseuse sans effet.
2. **Temps d'exécution court** (~1 minute) — le gain de performance du lazy formatting est
   négligeable dans ce contexte.
3. **Lisibilité** — les f-strings sont plus faciles à lire et à maintenir que le style `%`
   avec des arguments positionnels séparés.

## Conséquences

- Désactiver la règle pylint `W1203` dans la configuration si pylint est activé.
- Si un jour le file handler n'est plus systématiquement à DEBUG (ex : mode production
  sans fichier de log), réévaluer cette décision.
