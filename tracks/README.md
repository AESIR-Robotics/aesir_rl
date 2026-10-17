# Pistas

Cada carpeta contiene todo lo de una pista. `models/` solo guarda el robot y el
botón, que se comparten entre pistas. Las rutas se declaran en una sola tabla:
`TRACK_DEFS` en `rl_ws/base_training/config.py`.

```
tracks/<pista>/
  scene.xml           robot + terreno; es lo que carga el simulador (columna xml)
  terrain.xml         solo el terreno, sin robot; de aquí sale el octomap (columna terrain)
  occupied_map.bt     octomap para el heatmap (columna bt)
  inflated_map.bt     octomap dilatado
  meshes/             mallas STL del terreno
  source/             CAD/URDF original y generadores del terreno
  preview.png         captura de referencia (si existe)
```

`pallets` además tiene `pallets.xml` y `door.xml`. Como en `steps2`, su mapa de navegación
sale del XML: las tarimas (`fatal_pallet*`) son la zona transitable (columna `drivable_prefix`)
y los postes y la puerta son pared (columna `wall_prefix`).
`_archive/` guarda modelos que ningún código usa (la versión vieja de `steps` como losa sólida).

## Añadir una pista
1. Crear `tracks/<pista>/` con `terrain.xml` y `scene.xml`. `scene.xml` incluye
   `../../models/aesir_mujoco_robot.xml` y `terrain.xml`; las mallas se referencian
   como `meshes/<archivo>.stl`, relativas a `terrain.xml`.
2. Añadir la fila en `TRACK_DEFS`.
3. Generar los octomaps: `python3 rl_ws/octomap_from_mujoco.py --track <pista>`.

Los XML se cargan siempre con ruta absoluta (así lo hace `config.py`): MuJoCo 3.10
falla con un `include` que empieza por `../` si la escena se abre con ruta relativa.

## Advertencias conocidas
- `steps2/source/gen_steps2.py` genera un suelo plano infinito, pero `steps2/terrain.xml`
  tiene un arreglo hecho a mano (suelo como caja recortada al circuito). Volver a
  ejecutar el generador deshace ese arreglo.
- `maze/occupied_map.bt` e `inflated_map.bt` no coinciden con lo que genera hoy
  `octomap_from_mujoco.py` a partir de `maze/terrain.xml`. Ya era así antes de la
  reorganización; regenerarlos cambiaría el heatmap que ve la política.
