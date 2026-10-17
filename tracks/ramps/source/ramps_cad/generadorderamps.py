import math
import os
import random
import xml.etree.ElementTree as ET


# =========================
# CONFIGURACION
# =========================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_XML = os.path.join(SCRIPT_DIR, "template.xml")
OUTPUT_XML = os.path.join(SCRIPT_DIR, "mapa_con_rampas.xml")


# cantidad de rampas
NUM_RAMPAS = 1190

# altura encima de la plataforma
Z = 0.12


# tamaño aproximado de la rampa en metros (según el STL escalado a 0.001)
RAMP_SIZE_X = 0.6
RAMP_SIZE_Y = 0.6


# tamaño de la plataforma calculado en base al patrón de rampas
# para que las rampas ocupen toda la superficie sin salir del mapa.
cols = int(math.sqrt(NUM_RAMPAS))
rows = math.ceil(NUM_RAMPAS / cols)
PLATAFORMA_X = (cols * RAMP_SIZE_X) / 2.0
PLATAFORMA_Y = (rows * RAMP_SIZE_Y) / 2.0


# rotaciones permitidas
ROTACIONES = [
    0,
    90,
    180,
    270
]



# =========================
# CARGAR XML
# =========================

tree = ET.parse(INPUT_XML)

root = tree.getroot()

worldbody = root.find("worldbody")



# =========================
# GENERAR POSICIONES
# =========================

# Distribución rectangular exacta para que no queden huecos ni solapamientos.
step_x = RAMP_SIZE_X
step_y = RAMP_SIZE_Y

# Desplazamos la grilla para que el borde de la primera rampa quede alineado
# con el borde de la plataforma y las siguientes se apoyen directamente.
offset_x = -PLATAFORMA_X + (step_x / 2.0)
offset_y = -PLATAFORMA_Y + (step_y / 2.0)

posiciones = []
for index in range(NUM_RAMPAS):
    col = index % cols
    row = index // cols
    x = offset_x + col * step_x
    y = offset_y + row * step_y
    posiciones.append((x, y))



# =========================
# CREAR RAMPAS
# =========================

for i,(x,y) in enumerate(posiciones):


    rot = random.choice(ROTACIONES)


    body = ET.Element(
        "body",
        {
            "name": f"rampa_{i}",
            "pos": f"{x:.3f} {y:.3f} {Z}",
            "euler": f"90 {rot} 0"
        }
    )


    ET.SubElement(
        body,
        "geom",
        {
            "name": f"rampa_geom_{i}",
            "type": "mesh",
            "mesh": "rampa_mesh",
            "material": "mat_plataforma",
            "condim": "3",
            "friction": "1 0.005 0.0001"
        }
    )


    worldbody.append(body)



# =========================
# GUARDAR
# =========================

tree.write(
    OUTPUT_XML,
    encoding="utf-8",
    xml_declaration=True
)


print(
    f"Mapa generado: {OUTPUT_XML}"
)

print(
    f"Rampas creadas: {NUM_RAMPAS}"
)