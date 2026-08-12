# σ dependiente del estado: RECHAZADA

> **Estado: revertida del codigo.** Este documento es el registro completo —
> que era, como estaba implementada, que se midio y por que se descarto. Si
> alguien quiere reproducirla, aqui esta todo; ya no queda ni una linea en el
> repo (recuperable con `git show <commit>:rl_ws/utils/measure_sigma.py`).

**Resumen en una linea:** el mecanismo funciona (la red aprende la σ correcta
por pista, `d = +1.5`) pero el resultado es peor (−23 pp en `flat` por
convergencia prematura) y **no mueve `pallets` en absoluto**. Lo valioso del
experimento no es σ: es lo que salio al buscar por que `pallets` no se movia
(§7).

---

## 1. Que era

En PPO la politica emite una distribucion: la red calcula μ (la accion que
quiere) y σ (el ruido encima), y lo que el robot ejecuta es `a ~ Normal(μ, σ)`.
σ **es** la exploracion.

σ era —y vuelve a ser— un parametro global:

```python
self.log_std_vw = nn.Parameter(torch.full((2,), log_std_init))
```

Dos numeros que el gradiente ajusta. No depende de la observacion: **la misma σ
en flat, ramps, steps1m y pallets**, en todos los instantes.

La variante probada la predecia desde el estado:

```python
self.actor_logstd = nn.Linear(hidden, 2)
nn.init.zeros_(self.actor_logstd.weight)              # pesos 0
nn.init.constant_(self.actor_logstd.bias, log_std_init)  # bias = -0.5
```

Con ese init, en la iteracion 0 σ vale `exp(-0.5) = 0.6065` en TODOS los
estados — identico a la version global, asi que las dos condiciones arrancaban
en el mismo punto y toda diferencia posterior era aprendizaje. Verificado:
sobre una red recien inicializada el medidor daba `d = 0.00` en las 4 pistas.

En `forward()`:
```python
log_std_vw = torch.clamp(self.actor_logstd(z), C.LOG_STD_MIN, C.LOG_STD_MAX)
std_vw = log_std_vw.exp()          # (batch,2), ya no un vector global expandido
```

Coste: +512 parametros sobre 239 533 (+0.21%).

## 2. Por que se probo

Medido sobre los checkpoints de E8 (σ global, iteracion 250):

| semilla | σ_v (m/s) | σ_ω (rad/s) |
|---|---|---|
| e8_seed1 | 0.165 | **1.129** |
| e8_seed2 | 0.177 | **1.261** |
| e8_seed3 | 0.143 | **1.042** |

`W_REF_RADPS = 0.76` es el giro maximo **real alcanzable**. El ruido de
exploracion en el giro era **~1.5× el rango entero**.

Las pistas piden cosas opuestas: `pallets` necesita **precision** (cruzar entre
palos con poco espacio), `steps1m` necesita **vigor** (empujar contra el
escalon). Un solo numero no puede ser las dos cosas, y pallets es 1 de 4
pistas: pierde la votacion.

La observacion que lo motivaba: entrenando **solo con pallets** el robot llega
a ~95%; con las cuatro pistas cae a ~0%.

## 3. El metodo de medicion (`measure_sigma.py`, eliminado)

Prediccion falsable: **σ(pallets) < σ(steps1m)**.

Rollout con la politica sobre `VecMujocoEnv`; σ se lee de `p["std_vw"]` en cada
paso y se agrupa por la pista de cada env (`VecMujocoEnv` reparte round-robin y
cada env vive toda la corrida en la suya, asi que la etiqueta es fija). Se
reporta media por pista y **Cohen's d** entre pallets y steps1m — con n de
miles cualquier diferencia sale "significativa", asi que lo que importa es el
tamaño, no un p-valor.

Criterio fijado ANTES de correr: `|d| < 0.2` → la red no diferencia, hipotesis
muerta. `|d| > 0.8` → efecto grande, seguir.

## 4. Tanda 1 — `ENT_COEF = 0.005` (la del sistema)

σ **se fue al techo** (`LOG_STD_MAX` → σ=1.0) en las tres pistas dificiles:

| iter | σ_v(pallets) | σ_v(steps1m) | d |
|---|---|---|---|
| 50 | 0.710 | 0.951 | **+0.89** ✓ |
| 250 | 0.980 | 0.981 | **−0.08** ✗ |

En la iteracion 50 la prediccion se cumplia con fuerza; para la 250 se habia
evaporado.

**Causa: el bonus de entropia.** Con σ global, subirla cuesta en *todos* los
estados a la vez, asi que el gradiente de politica empuja en contra donde
importa. Con una cabeza por estado, la red sube σ **selectivamente** donde la
ventaja es plana — la mayoria de los estados — y cobra `ENT_COEF` casi gratis.
El bonus paso de empujoncito a fuerza dominante.

| entropia media | it 0-50 | it 100-150 | it 200-250 |
|---|---|---|---|
| e8 (σ global, ent=.005) | 1.179 | −1.036 | −3.340 |
| σ por estado (ent=.005) | 1.843 | 0.293 | **−1.316** |

Resultado: sin efecto en ninguna pista (todo dentro del rango de las semillas
base).

## 5. Tanda 2 — `ENT_COEF = 0.001`

Bajar la entropia **si** arreglo el mecanismo. σ en la iteracion 250:

| pista | σ_v | σ_ω (rad/s) |
|---|---|---|
| flat | 0.057 | 0.52 |
| ramps | 0.237 | 1.47 |
| pallets | 0.270 | 2.04 |
| steps1m | 0.713 | 3.92 |

`d = +1.46` (σ_v) y `+1.23` (σ_ω) para pallets vs steps1m, en la direccion
predicha. El orden `flat < ramps < pallets < steps1m` es exactamente el
sensato. **El mecanismo queda confirmado.**

Pero el resultado (ventana 200-250, contra las 3 semillas de E8):

| pista | base media | σ por estado | rango base | |
|---|---|---|---|---|
| **flat** | 94.6 | **71.7** | [92.7, 97.1] | **FUERA** |
| ramps | 81.0 | 82.9 | [66.9, 88.4] | dentro |
| steps1m | 39.8 | 39.0 | [34.6, 44.1] | dentro |
| pallets | 0.0 | 0.0 | [0.0, 0.0] | dentro |

`flat` no se estanca — **regresa**: sube a 76.5% en la iteracion 100 y baja a
~70%. En paralelo σ_v en flat se derrumba 0.519 → 0.173 → 0.057. Es
**convergencia prematura**: flat es la pista facil, la recompensa llega rapido,
σ colapsa y la politica ya no puede salir de donde quedo.

## 6. Veredicto

Rechazada. Cuesta 23 pp en `flat` por un mecanismo que funciona pero no sirve
para lo que se queria: **`pallets` sigue en 0.0 exacto** en las dos tandas y en
las tres semillas base. σ no era el cuello de botella.

Se conserva como resultado documentado, no como parte del sistema. `ENT_COEF`
se bajo a 0.001 para esta tanda; **con σ global eso no esta validado** (la base
E8 se corrio con 0.005) — decidir aparte.

## 7. Lo que SI salio de aqui: por que falla `pallets`

Buscando por que pallets no se movia, la respuesta estaba en los logs, que ya
registran motivo de terminacion y waypoint alcanzado. Midiendo las rutas:

| pista | n_waypoints (30 resets) | largo ruta | ¿varia? |
|---|---|---|---|
| flat | 20.9 ± 9.8 | 9.3 m | **random cada reset** |
| ramps | 20.5 ± 7.5 | 9.2 m | **random** |
| steps1m | 22.1 ± 8.2 | 10.1 m | **random** |
| **pallets** | **74.0 ± 0.0** | **18.1 m** | **FIJA, identica siempre** |

Terminaciones en pallets: 55% "atascado (sin progreso)", 30% "toco el piso",
14% "caida", **0% meta**. Avanza en promedio hasta el waypoint 29 de 74 → **39%
de la ruta**; flat llega a 15.3 de ~21 → **~73%**.

**`pallets` es la unica pista sin randomizacion**: misma salida, misma meta, los
mismos 74 waypoints en cada uno de los ~5000 episodios. Las otras tres sortean
spawn y meta, asi que a veces les toca una ruta de 3 waypoints — trivial — y de
ahi arrancan.

Eso explica las dos observaciones a la vez:

- **pallets sola = 95%:** los 14 envs atacan esa unica ruta, ~4× mas intentos;
  tarde o temprano uno la completa, se cobra `GOAL_BONUS` y bootstrapea.
- **mezclada = 0%:** solo ~3 de 14 envs en pallets, sobre una ruta 2× mas larga.
  En ~5000 episodios **nunca** se cobro el `GOAL_BONUS` ni una vez. Sin ese
  evento no hay de donde aprender el final.

No es imprecision, no es capacidad y no es exploracion: es que `pallets` es una
tarea **fija, larga y todo-o-nada** que no puede bootstrapear con un cuarto de
los envs.

**Siguiente paso:** randomizar spawn y meta en `pallets` como las otras tres.
Convierte la ruta de 74 waypoints en un curriculo natural. Es el mismo `kind`
con inicio/meta sorteados que `maze` necesita, asi que el trabajo sirve para las
dos y esta en el camino critico igual.

## 8. Efecto colateral, ya revertido

Mientras la variante estuvo activa, los checkpoints con σ global no cargaban.
Ahora es al reves: `test_base.py` detecta `actor_logstd.weight` y avisa. Los
checkpoints de `runs/sigma_seed1/` son de la variante rechazada y **no cargan**;
su log si sirve.

Se cambio tambien el default de `--resume` a **vacio (desde cero)**. Antes
apuntaba a `checkpoints_base/fast_iter00150.pt` y arrancaba en caliente **sin
avisar** — veneno para un barrido de semillas. Ese cambio **se queda**: no tenia
nada que ver con σ y era un bug latente.
