# Castigo por sostener los flippers extendidos — probado y RECHAZADO

Termino de reward que se implemento, se entreno y se quito. Este documento deja
el registro: la geometria que lo motivo SIGUE SIENDO VALIDA y explica por que
maze falla, pero la forma del castigo era mala y hundio el entrenamiento.

## Lo que se implemento

```python
castigo = FLIPPER_IDLE_W * media_i( max(0, sin(theta_i)) )   # si no hay borde atacable
        = 0.0                                                 # si lo hay
```

Con `FLIPPER_IDLE_W = 1.0`. Complemento exacto de `flipper_terrain_bonus`, que
devuelve `0.0` cuando `flipper_edge["actionable"]` esta vacio. La primera version
usaba `max(0, |MOUNT_x| + FLIPPER_L*sin(theta) - CHASSIS_FRONT_X)` (lo que la
punta sobresale del chasis, con zona muerta hasta 21.5 grados); se simplifico a
`max(0, sin(theta))` a peticion del usuario.

## Por que parecia buena idea (esta parte sigue en pie)

Los flippers giran en el plano x-z: **no ensanchan el robot** (ancho constante
0.551 m a cualquier angulo) pero **lo alargan** de 0.815 a 1.223 m. Como el
planificador traza la ruta con la huella plegada, extenderlos invalida el plan.
Medido reconstruyendo el `TrackMap` del maze con la huella real de cada angulo:

| theta | largo | R circunscrito | area alcanzable | rotable |
|------:|------:|---------------:|----------------:|--------:|
|   0 deg | 0.815 m | 0.492 m | **75.9 m2** | 27.7 m2 |
|  15 deg | 0.849 m | 0.506 m | 75.1 m2 | 22.3 m2 |
|  30 deg | 0.945 m | 0.547 m | 61.5 m2 | 20.8 m2 |
|  45 deg | 1.081 m | 0.607 m | 56.3 m2 | 12.2 m2 |
|  60 deg | 1.176 m | 0.649 m | **5.4 m2** | 11.5 m2 |
|  75 deg | 1.223 m | 0.671 m | 5.0 m2 | 9.3 m2 |

Entre 45 y 60 grados el laberinto **deja de existir**: el robot queda encerrado
en un bolsillo de 5 m2. Eso es el 84% de "atascado" de maze. Y con la huella real
en reposo el maze es MAS transitable (75.9 m2) que lo que asume el planificador
(67.4 m2), asi que el robot si puede pasar — como sostuvo el usuario todo el
tiempo.

Nada cobraba por sostenerlos fuera: `flipper_terrain_bonus` solo premia
extenderlos junto a terreno trepable, y `flipper_jerk` solo castiga MOVERLOS.

## Por que fallo

El error fue dimensionar el termino en su estado FINAL (flippers plegados ->
cuesta ~0) sin mirar el CAMINO hasta ahi. Bajo la exploracion inicial, la Beta
de los flippers muestrea casi uniforme en `[FLIPPER_MIN_RAD, FLIPPER_MAX_RAD]` =
[-63, 151] grados, con media 44 grados:

| estado de la politica | castigo/paso | por episodio de 500 pasos |
|---|---:|---:|
| exploracion inicial (Beta ~ uniforme) | **0.502** | **-251** |
| ya aprendida a plegarlos | 0.074 | -37 |

**En la iteracion 0 cuesta -251 por episodio: exactamente `FALL_PENALTY`.** Y es
la senal mas facil de todo el reward — presente en cada paso de cada pista, sin
varianza y sin necesidad de navegar, mientras que `WP_BONUS` exige aprender a
moverse. La politica aprende "no muevas los flippers" antes que nada, las 4
dimensiones Beta colapsan, y las pistas que NECESITAN flippers no se recuperan.

Medido en entrenamiento (24 envs, iter ~320), contra `maze_seed1` sin el termino:

| | con el castigo | sin el castigo |
|---|---:|---:|
| entropia | **-2.1** | +1.478 (iter 0) |
| ramps | 3% | 71% |
| steps1m | 2% | 27% |
| flat | 4-5% | 94% |
| maze | 0% | 0% |

`pi = -0.03` y `v = 0.02` confirman que ya no quedaba gradiente: colapso
prematuro a una politica determinista.

## Que se aprende de esto

1. **Dimensionar un termino de reward por su coste bajo la politica ALEATORIA,
   no bajo la politica convergida.** Un castigo que es gratis al final puede ser
   el termino dominante al principio, y es entonces cuando decide que se aprende.
2. Un castigo denso, sin varianza y disponible en todos los estados compite en
   desventaja injusta contra un bonus esparso que exige una conducta compleja.
3. La geometria de la tabla de arriba **no esta refutada**: los flippers
   extendidos si hunden el area navegable del maze. Lo refutado es atacarlo con
   un castigo denso en el reward.

## Si alguna vez se retoma

Haria falta que el coste no aparezca hasta que la politica ya sepa navegar
(peso en rampa, o activarlo solo tras cierto success rate), o atacarlo fuera del
reward: por ejemplo que el planificador use la huella del angulo ACTUAL de los
flippers en vez de la plegada, para que la ruta sea factible con lo que el robot
esta haciendo de verdad.
