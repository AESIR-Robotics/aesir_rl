# El gate de flippers: diseño, implementación y por qué se eliminó

> **Estado: ELIMINADO del código.** Este documento es el registro completo —
> qué era, cómo estaba implementado en cada archivo, qué se midió y qué se
> decidió. Si alguien quiere reproducir la variante con gate, aquí está todo lo
> necesario para reconstruirla; ya no queda ni una línea en el repo.
>
> Commit de eliminación: ver `git log -- rl_ws/base_training/config.py`.
> El código vivo está en el historial hasta ese commit.

---

## 1. Qué era

Una **séptima dimensión de acción**, `action[6]`, muestreada de una **Bernoulli**,
que decidía si la política controlaba los flippers o si estos iban a una pose de
reposo fija:

```
action[6] = 1.0  ->  los flippers siguen action[2:6]  (control normal)
action[6] = 0.0  ->  los flippers van a FLIPPER_HOME_RAD; action[2:6] se IGNORA
```

La acción del sistema era por tanto `ACT_DIM = 7`:

| índice | qué | distribución |
|---|---|---|
| `[0:2]` | `v`, `ω` | Normal, recortada a `[-1,1]` |
| `[2:6]` | flipper ×4 | Beta, soporte ya en `[0,1]` |
| `[6]` | **gate** | **Bernoulli**, `0.0`/`1.0` exacto |

## 2. Por qué se diseñó así

La motivación era razonable y es la que se le ocurriría a cualquiera:

- **Los flippers no deberían estar continuamente actuados.** En terreno plano no
  hay nada que trepar; tener cuatro articulaciones moviéndose todo el rato mete
  temblor, gasta energía y puede desestabilizar al robot.
- **Recuperar el comportamiento de la fase 1.** Antes de que los flippers fueran
  controlables, el robot cruzaba `flat` con ellos recogidos y lo hacía bien. El
  gate le daba a la política la opción de *volver* a ese modo cuando le
  conviniera, en vez de tener que aprender a emitir cuatro ángulos ≈ 0 y
  mantenerlos.
- **Menos que aprender en plano.** Con el gate a 0, cuatro dimensiones de acción
  dejan de importar: la política solo tiene que acertar un bit en lugar de
  cuatro continuas.

La elección de **Bernoulli** (en vez de un escalar continuo umbralizado en 0, que
fue el primer intento) sí fue correcta y por una razón concreta: la entropía de
una gaussiana no está acotada y puede crecer sin límite si el reward no la
contrarresta; la Bernoulli tiene entropía máxima `log(2)` **por construcción**.
Ese razonamiento sigue siendo válido — el problema no era la distribución, era
la existencia misma del gate.

## 3. Cómo estaba implementado

Cinco puntos de contacto. Este es el detalle exacto, por si hay que rehacerlo.

### `config.py`

```python
USE_FLIPPER_GATE = False          # flag; True = variante con gate
ACT_DIM = 7 if USE_FLIPPER_GATE else 6
FLIPPER_HOME_RAD = 0.0            # pose de reposo cuando el gate decide reposo
```

`FLIPPER_HOME_RAD` **se conserva** tras la eliminación: lo siguen usando
`base_env.flipper_targets` (rama `CONTROL_FLIPPERS=False`), el ancla al reposo
del reward de terreno y `base_ros_env.stop_robot`.

### `base_env.flipper_targets` — el único punto donde el gate tenía efecto real

```python
def flipper_targets(action):
    if not CONTROL_FLIPPERS:
        return None
    if USE_FLIPPER_GATE and float(action[6]) < 0.5:
        return np.full(4, FLIPPER_HOME_RAD, dtype=np.float32)
    u = np.clip(np.asarray(action[2:6], dtype=np.float32), 0.0, 1.0)
    return FLIPPER_MIN_RAD + u * (FLIPPER_MAX_RAD - FLIPPER_MIN_RAD)
```

Compartida por los dos backends (MuJoCo directo y bridge ROS2), así que el gate
se comportaba igual en simulación y en el robot.

### `ppo_cnn_extractor.CNNActorCritic` y `ppo.MLPActorCritic`

Cabeza extra en el tronco compartido:

```python
self.actor_gate = nn.Linear(hidden, 1) if act_dim == 7 else None
...
p["gate_logit"] = self.actor_gate(z).squeeze(-1)          # en forward()
d_gate = Bernoulli(logits=params["gate_logit"])           # en _dists()
```

y en `act_batch` / `act` / `evaluate`, el log-prob y la entropía sumaban el
término del gate:

```python
raw_gate = d_gate.sample()
logp = logp + d_gate.log_prob(raw_gate)                   # muestreo
...
logp = logp + d_gate.log_prob(actions[:, 6]).unsqueeze(-1) # evaluate
entropy = entropy + d_gate.entropy()
```

La columna del gate se concatenaba **tanto** al tensor `raw` (el que ve el
log-prob) **como** al `action` ejecutado — al ser `{0,1}` exacto no había
desajuste entre ambos, a diferencia de la Normal recortada.

### `sac.to_env_action`

SAC **nunca** tuvo gate en su actor: su política es una tanh-gaussiana
reparametrizable y una Bernoulli no encaja ahí. Rellenaba la columna a mano:

```python
out[..., 6] = 1.0                 # gate SIEMPRE ON
```

### `test_base.py`

El `act_dim` se deducía del checkpoint (presencia de `actor_gate.weight`) para
poder evaluar políticas de ambas variantes, y la acción determinista tomaba la
moda de la Bernoulli:

```python
cols.append((p["gate_logit"] > 0.0).float().unsqueeze(-1))
```

---

## 4. La primera señal: SAC

Antes del experimento controlado ya había un aviso. El primer SAC con gate dio
**1.54% de éxito** frente al 37.5% del PPO de la época. El diagnóstico:

> Con el gate umbralizado, cuando valía 0 `flipper_targets` ignoraba
> `action[2:6]` por completo → **5 de las 7 dimensiones no afectaban la
> transición**. SAC actualiza el actor con `dQ/da`, así que ese gradiente era
> ruido puro en el 71% de las dimensiones. PPO lo tolera porque nunca deriva
> respecto a la acción; SAC no.

Se quitó el gate de SAC y el problema desapareció. En ese momento se interpretó
como una incompatibilidad *específica de SAC*, no como un problema del gate.
Fue una lectura incompleta: el mecanismo — dimensiones que a veces no afectan a
nada — perjudica a cualquier algoritmo, solo que PPO lo paga más despacio.

## 5. El experimento (E8)

**Diseño:** 3 semillas × 250 iteraciones por condición, idénticas en todo salvo
`USE_FLIPPER_GATE`. Métrica: tasa de éxito promediada sobre la ventana de
iteraciones **200–250** (evita el ruido de la fase temprana y no cherry-pickea
un pico). Pistas de entrenamiento: `[flat, ramps, steps1m, pallets]`.

**Resultados por semilla** (% de éxito, ventana 200–250):

| condición | semilla | flat | ramps | **steps1m** | pallets | global |
|---|---|---|---|---|---|---|
| **con gate** | 1 | 95.8 | 82.3 | **27.6** | 1.4 | 53.9 |
| | 2 | 92.5 | 71.5 | **19.9** | 0.0 | 45.8 |
| | 3 | 93.2 | 79.6 | **17.7** | 0.1 | 48.2 |
| **sin gate** | 1 | 92.7 | 88.4 | **44.1** | 0.0 | 55.1 |
| | 2 | 94.0 | 66.9 | **40.7** | 0.0 | 52.9 |
| | 3 | 97.1 | 87.8 | **34.6** | 0.0 | 57.0 |

**Agregado:**

| pista | con gate | sin gate | Δ |
|---|---|---|---|
| flat | 93.8 | 94.6 | +0.8 (n.s.) |
| ramps | 77.8 | 81.0 | +3.2 (n.s.) |
| **steps1m** | **21.8** | **39.8** | **+18.0** |
| pallets | 0.5 | 0.0 | — (ambas ≈ 0) |
| global | 49.3 | 55.0 | +5.7 |

**Significancia en `steps1m`:** t = +4.42, df = 4, p ≈ 0.011 (Welch). Lo más
convincente no es el p-valor con n=3, sino que **los rangos no se solapan**:
con gate `[17.7, 19.9, 27.6]`, sin gate `[34.6, 40.7, 44.1]`. La peor semilla
sin gate supera a la mejor con gate por 7 puntos.

**Ninguna otra pista empeora.** No es un intercambio; es una mejora limpia en la
pista que más depende de los flippers.

## 6. Por qué el gate hacía daño

La explicación que encaja con lo medido — y con la señal previa de SAC:

1. **Rompe la asignación de crédito de la Beta.** En los pasos con el gate a 0,
   las cuatro dimensiones de flipper reciben gradiente de política aunque **no
   hayan afectado a nada**. El log-prob las incluye, la ventaja las multiplica.
   Es ruido inyectado directamente en la cabeza que más importa en `steps1m`.
2. **La discretización llega en el peor momento.** Trepar un escalón exige
   flippers *bien colocados*, no *encendidos*. El gate obliga a la política a
   acertar un bit antes de que sus cuatro ángulos tengan efecto — y ese bit se
   aprende con una señal de recompensa que llega muchos pasos después.
3. **El beneficio prometido era gratis sin el gate.** "Flippers recogidos" se
   expresa perfectamente comandando `θ ≈ 0` en `action[2:6]`. El gate no añadía
   capacidad expresiva: **solo quitaba gradiente**.

El error de diseño, en una frase: se introdujo un mecanismo discreto para
resolver un problema (temblor en plano) que la parametrización continua ya
resolvía sola, y el coste fue arruinar el aprendizaje donde de verdad hacía falta.

## 7. Decisión

- **El sistema es de 6 dimensiones y no tiene gate.** `ACT_DIM = 6`, fijo, sin
  flag.
- **El código se elimina por completo**, incluidas las ramas condicionales que
  lo mantenían opcional. Un flag que nunca se va a activar es deuda: obliga a
  cada archivo a soportar dos arquitecturas y a cada lector a preguntarse cuál
  está viva.
- **Para el paper**, el gate deja de ser una fila de la tabla de ablaciones y
  pasa a ser un **resultado negativo**: un diseño intuitivamente atractivo que
  cuesta 18 puntos en trepado de escalones. Es la parte más útil de C2 para
  quien vaya a construir algo parecido.
- Los datos crudos se conservan en `runs/seed{1,2,3}.log` (con gate) y
  `runs/e8_seed{1,2,3}.log` (sin gate), y las figuras en `figs/`.

## 8. Advertencia sobre estos números

Son **preliminares**. Se midieron con `ACTIVE_TRACKS = [flat, ramps, steps1m,
pallets]`. El conjunto final será `[flat, ramps, steps1m, maze]` con `pallets`
en validación, así que la línea base cambia y hay que repetir la tanda entera.

La dirección del efecto, sin embargo, es sólida: rangos sin solape en la pista
más sensible, más un mecanismo explicativo que también predijo el fallo de SAC.
No se espera que se invierta. Lo que puede cambiar es la magnitud.

Si al repetir con `maze` el gate volviera a parecer atractivo, este documento
tiene todo lo necesario para reimplementarlo — pero la barra debería ser alta.
