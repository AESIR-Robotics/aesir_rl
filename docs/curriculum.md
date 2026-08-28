# Curriculum de longitud de mision — probado y RECHAZADO

Se implemento, se entreno 900 iteraciones y se quito. Este documento deja el
registro: el diagnostico que lo motivo era CORRECTO y quedo confirmado, pero la
implementacion fallo por un parametro mal elegido, y para cuando eso se vio SAC
ya resolvia el mismo problema sin necesitar curriculum.

## El diagnostico que lo motivo (esto sigue siendo cierto)

Evaluando la politica entrenada en maze con las metas AGRUPADAS por longitud de
ruta (60 misiones por bin):

| ruta (waypoints) | exito |
|-----------------:|------:|
| 2-6   | **70.0%** |
| 7-12  | 33.3% |
| 13-20 | 0.0% |
| 21-35 | 0.0% |
| 36-80 | 0.0% |

Y la distribucion que recibia en ENTRENAMIENTO, replicando el sorteo de
`reset()` (400 muestras):

| | p10 | mediana | p90 |
|---|---:|---:|---:|
| waypoints por mision | 15 | **32** | 53 |

**Rutas de <= 8 waypoints en entrenamiento: 0.0%.** Ni una.

O sea que la politica sabia hacer al 70% una cosa que nunca se le pedia, y el
100% de sus episodios caian en el rango donde acertaba el 0%. Con eso
`GOAL_BONUS` (1000 puntos) no aparecia en NINGUN retorno de maze, y no habia
gradiente que empujara a completar la mision -- solo a acumular waypoints y no
caerse. La politica optimizaba lo unico que podia cobrar.

Otra firma del mismo problema: la politica moria en el waypoint ~5 con una
regularidad que no dependia en nada de la dificultad de la meta (3.9 / 5.8 /
5.5 / 5.1 / 5.0 waypoints alcanzados para rutas de 2-6 hasta 36-80).

## Que se implemento

Acotar la meta sorteada a rutas de <= H waypoints, con H auto-regulado por
pista segun su exito reciente. Al llegar H al tope de la pista el curriculum es
transparente: es exactamente el entrenamiento de siempre.

El filtro usaba el COSTE del arbol de Dijkstra que `plan()` ya cachea (gratis,
sin replanificar), y el tope se VERIFICABA planificando y reintentando, porque
la recta calibrada (`waypoints ~= 0.1229*coste + 4.71`, R2=0.965) es buena en
promedio pero se pasa en el rango bajo -- pidiendo H=12 devolvia mediana 19.

Verificado antes de entrenar: tope exacto con **100% de cumplimiento** en
H = 8, 10, 13, 18, 24, 32, 45, 57; cadencia de un ajuste por ventana de 50
episodios (`8 -> 10.4 -> 13.5 -> 17.6 -> 22.8`); 15-30 ms por reset.

```
CURRICULUM_START_WP = 8      CURRICULUM_UP      = 1.30
CURRICULUM_WINDOW   = 50     CURRICULUM_DOWN    = 1.20
CURRICULUM_UP_AT    = 0.60   CURRICULUM_DOWN_AT = 0.20
```

## Por que fallo

`CURRICULUM_UP_AT = 0.60`. El horizonte solo subia si el exito superaba el 60%,
y las pistas se estancan entre el 20% y el 55%: **caen en la banda muerta entre
`DOWN_AT` y `UP_AT`, donde no sube ni baja, y se congelan para siempre.**

Resultado tras 900 iteraciones (24 envs, 4 pistas, desde cero):

| pista | exito | H alcanzado | tope | `hidden_512` sin curriculum |
|---|---:|---:|---:|---:|
| flat    | **94.7%** | **60** (graduo) | 60 | 83.7% |
| ramps   | 60.6% | 44 | 60 | **80.7%** |
| steps1m | 21.7% | **8** (nunca subio) | 60 | **32.6%** |
| maze    | 34.8% | **18** | 57 | 3.2% |

Solo flat graduo, y mejoro. `steps1m` se quedo en H=8 las 900 iteraciones
enteras -- entrenando misiones triviales, sin llegar nunca a la tarea real, y
acabo PEOR que sin curriculum (21.7% contra 32.6%).

Y el 34.8% de maze **no es comparable** con el 3.2% de `hidden_512`: esta medido
sobre un tercio del rango de misiones. Era exactamente el riesgo que se anoto
antes de lanzarlo -- "un curriculum que nunca gradua es simplemente una tarea
mas facil" -- por eso el log imprimia H junto al success rate.

## Lo que si demostro

**maze salto a 43.4% en el bloque 0**, inmediatamente, sin entrenar nada nuevo.
Eso confirmo el diagnostico de arriba: la politica siempre pudo con maze, y el
bloqueo era que nunca se le pedia una mision a su alcance.

Ese resultado es lo que justifico probar SAC, que ataca el mismo problema en la
raiz: con buffer de repeticion los pocos exitos de maze se re-muestrean miles de
veces en vez de tirarse tras 10 epocas. SAC llego a **60-65% en maze sobre la
distribucion COMPLETA** en 100 iteraciones, sin curriculum. Con eso el
curriculum paso de ser una solucion a ser un parche a un problema de aprendizaje
que ya estaba resuelto por otra via.

## Si alguna vez se retoma

Dos arreglos, ninguno probado:

1. **`CURRICULUM_UP_AT` a ~0.35**, para que las pistas que se estancan en el
   20-55% sigan graduando.
2. **Condicion de graduacion por estancamiento**: si H no sube en N ventanas,
   subirlo igual. Sin eso, cualquier umbral fijo puede volver a congelar una
   pista que se asiente justo por debajo.

Y la leccion general, que aplica a cualquier curriculum futuro: **el log tiene
que imprimir el horizonte junto al exito.** Un 8% con H=10 y un 8% con H al tope
son resultados opuestos, y sin esa columna no se distinguen.
