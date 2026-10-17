import random

def generate_stepfield_xml(filename="random_stepfields.xml", grid_size=15):
    """
    Genera un mundo de MuJoCo con un patrón de Stepfields aleatorio
    montado sobre la plataforma gigante de AESIR.
    grid_size: Número de celdas por lado (ej. 15x15 bloques).
    """
    
    # 1. Definir dimensiones de los stepfields (mitades para MuJoCo)
    # Base de 297mm -> 0.1485m
    box_xy = 0.1485 
    step = 0.297 # Distancia entre centros
    
    # Altura de la superficie de la plataforma gigante
    # pos_z (0.06) + size_z (0.06) = 0.12m
    platform_top_z = 0.12
    
    # 2. Plantilla base (Tu XML original)
    xml_lines = [
        '<mujoco model="plataforma_gigante_aesir">',
        '  <compiler angle="degree"/>',
        '  <option gravity="0 0 -9.81"/>',
        '  <asset>',
        '    <texture type="skybox" builtin="gradient" rgb1="0.4 0.6 0.8" rgb2="0 0 0" width="512" height="512"/>',
        '    <texture name="tex_piso" type="2d" builtin="checker" rgb1="0.15 0.1 0.1" rgb2="0.08 0.05 0.05" width="512" height="512"/>',
        '    <material name="mat_piso" texture="tex_piso" texrepeat="20 20" texuniform="true" reflectance="0.1"/>',
        '    <texture name="tex_plataforma" type="2d" builtin="flat" rgb1="0.8 0.4 0.1" rgb2="0.1 0.15 0.2" width="512" height="512"/>',
        '    <material name="mat_plataforma" texture="tex_plataforma" texrepeat="40 40" texuniform="true" reflectance="0.1"/>',
        '  </asset>',
        '  <worldbody>',
        '    <light directional="true" pos="0 0 10" dir="0 0 -1"/>',
        '    <geom name="suelo" type="plane" size="40 40 0.1" material="mat_piso" condim="3" friction="1 0.005 0.0001"/>',
        '    <geom name="plataforma_gigante" type="box" size="10 10 0.06" pos="0 0 0.06" material="mat_plataforma" condim="3" friction="1 0.005 0.0001"/>',
        '    <geom name="virtual_obstacle" type="box" size="0.3 0.3 0.3" pos="0 0 -5" rgba="0.85 0.1 0.1 1" condim="3" friction="1 0.005 0.0001"/>'
    ]

    # 3. Generar la cuadrícula aleatoria centrada
    offset = (grid_size * step) / 2.0

    for r in range(grid_size):
        for c in range(grid_size):
            # Calcular (x, y) centrado
            x = (c * step) - offset + (step / 2.0)
            y = (r * step) - offset + (step / 2.0)
            
            rand_val = random.random()
            
            # Distribución: 20% vacío, 40% bajos (15cm), 40% altos (30cm)
            if rand_val < 0.20:
                continue # Dejar la plataforma lisa visible
            elif rand_val < 0.60:
                # Obstáculo de 15cm
                h_half = 0.15 / 2.0
                z_pos = platform_top_z + h_half
                rgba = "1 0.6 0.1 1" # Naranja
                name = f"step_15_{r}_{c}"
            else:
                # Obstáculo de 30cm
                h_half = 0.30 / 2.0
                z_pos = platform_top_z + h_half
                rgba = "0.9 0.1 0.1 1" # Rojo
                name = f"step_30_{r}_{c}"

            geom_str = (
                f'    <geom name="{name}" type="box" size="{box_xy} {box_xy} {h_half}" '
                f'pos="{x:.4f} {y:.4f} {z_pos:.4f}" rgba="{rgba}" '
                f'condim="3" friction="1 0.005 0.0001"/>'
            )
            xml_lines.append(geom_str)

    # 4. Cerrar la jerarquía del XML
    xml_lines.append('  </worldbody>')
    xml_lines.append('</mujoco>')

    # 5. Escribir el archivo
    with open(filename, 'w') as f:
        f.write('\n'.join(xml_lines))
        
    print(f"✅ Mapa '{filename}' generado exitosamente con una cuadrícula de {grid_size}x{grid_size}.")

if __name__ == "__main__":
    # Puedes subir grid_size (ej. 30) para llenar más espacio de la plataforma gigante
    generate_stepfield_xml(grid_size=64)