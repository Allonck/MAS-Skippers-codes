import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import argparse
import sys
import re

def parse_xml_sequencer_robust(xml_file):
    """Analiza el XML aislando la definición de recetas del bloque sequence.

    Args:
        xml_file (str): Ruta al archivo XML del secuenciador.

    Returns:
        tuple: Contiene tres diccionarios (delays, states, recipes).
            - delays (dict): Mapeo de variables de tiempo a sus valores enteros.
            - states (dict): Mapeo de nombres de estados a conjuntos de señales activas.
            - recipes (dict): Mapeo de recetas a listas de pasos (estado y retardo).
    """
    with open(xml_file, 'r', encoding='utf-8') as f:
        xml_content = f.read()
    
    # Remover cualquier declaración XML
    xml_content = re.sub(r'<\?xml.*?\?>', '', xml_content, flags=re.DOTALL)
    wrapped_xml = f"<root>{xml_content}</root>"
    root = ET.fromstring(wrapped_xml)
    
    # Limpiar namespaces de todo el árbol
    for elem in root.iter():
        if '}' in elem.tag:
            elem.tag = elem.tag.split('}', 1)[1]
            
    delays, states, recipes = {}, {}, {}
    
    for var in root.findall('.//var'):
        name, val = var.get('name'), var.get('val')
        if name and val and val.isdigit():
            delays[name] = int(val)
            
    for state in root.findall('.//state'):
        name, val = state.get('name'), state.get('val')
        if name and val and ('|' in val or not val.strip().isdigit()):
            clean_components = set(c.strip() for c in val.split('|') if c.strip())
            states[name] = clean_components
            
    # CORRECCIÓN: Buscar recetas SÓLO dentro del bloque <recipes> explícito
    for recipes_block in root.findall('.//recipes'):
        for recipe_elem in recipes_block.findall('recipe'):
            name = recipe_elem.get('name')
            if name:
                steps = [{'state': step.get('state'), 'delay': step.get('delay')} 
                         for step in recipe_elem.findall('step')]
                recipes[name] = steps
                
    return delays, states, recipes

def generate_waveforms(target_recipes, recipes, states, delays, target_signals):
    """Genera las series temporales lógicas para las señales especificadas.
    
    Args:
        target_recipes (list): Lista de nombres de recetas en orden de ejecución.
        recipes (dict): Diccionario de recetas.
        states (dict): Diccionario de estados.
        delays (dict): Diccionario de retardos.
        target_signals (dict): Mapeo del nombre de la señal.
        
    Returns:
        dict: Tiempos y valores unificados para graficar en formato step.
    """
    waveforms = {sig: {'time': [], 'val': []} for sig in target_signals}
    current_time = 0
    
    for recipe_name in target_recipes:
        if recipe_name not in recipes:
            print(f"-> ADVERTENCIA: La receta '{recipe_name}' no existe en el XML.")
            continue
            
        for step in recipes[recipe_name]:
            state_name = step['state']
            delay_val = delays.get(step['delay'], 1)
            active_components = states.get(state_name, set())
            
            for sig, components in target_signals.items():
                is_active = any(comp in active_components for comp in components)
                level = 1 if is_active else 0
                
                waveforms[sig]['time'].append(current_time)
                waveforms[sig]['val'].append(level)
                
            current_time += delay_val
            
    # Añadir el punto temporal final para cerrar el último estado de la gráfica
    for sig in target_signals:
        if waveforms[sig]['time']:
            waveforms[sig]['time'].append(current_time)
            waveforms[sig]['val'].append(waveforms[sig]['val'][-1])
            
    return waveforms

def plot_phase_diagram(waveforms):
    total_puntos = sum(len(data['time']) for data in waveforms.values())
    if total_puntos == 0:
        sys.exit("\nERROR CRÍTICO: No se extrajeron datos de tiempo.")

    # Invertir el orden de iteración para graficar H3 arriba y Video abajo
    signals_order = list(waveforms.keys())

    fig, axs = plt.subplots(len(signals_order), 1, figsize=(10, 8), sharex=True)
    fig.subplots_adjust(hspace=0)

    for i, clock in enumerate(signals_order):
        ax = axs[i]
        data = waveforms[clock]
        
        ax.plot(data['time'], data['val'], color='black', drawstyle='steps-post', linewidth=1.5)
        
        # Estilos visuales
        ax.set_ylabel(clock, rotation=0, ha='right', va='center', fontsize=12, labelpad=15)
        ax.set_ylim(-0.2, 1.2)
        ax.set_yticks([]) 
        ax.set_xticks([]) 
        
        for spine in ['top', 'right', 'bottom', 'left']:
            ax.spines[spine].set_visible(False)

    axs[-1].spines['bottom'].set_visible(False)
    
    #axs[-1].set_xlabel('Tiempo de Ejecución (unidades arbitrarias del secuenciador)', fontsize=12)
    
    # Guarda en PNG con alta resolución y sin márgenes
    plt.savefig('waveforms.png', bbox_inches='tight', dpi=300)
    
    # Guarda en PDF en formato vectorial y sin márgenes
    plt.savefig('waveforms.pdf', bbox_inches='tight')

    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Genera diagramas de fase del secuenciador LTA (MAS-CCD).")
    parser.add_argument("-f", "--file", type=str, required=True, help="Ruta al archivo XML")
    parser.add_argument("-p", "--pixels", type=int, default=1, help="Número de píxeles a leer (fases)")
    args = parser.parse_args()
    
    # Mapeo alineado a la Fig. 2 del paper MAS-CCD, aislando los bits de conmutación.
    signals_map = {
        'H1': ['H1A', 'H1B'],
        'H2': ['H2C'],
        'H3': ['H3A', 'H3B'],
        'SW': ['SWB'],        # Summing Well (Activo en ALTO)
        'OG': ['OGB'],        # Output Gate (Activo en BAJO, el pulso se crea al apagar OGB)
        'RG': ['RGB'],        # Reset Gate (Activo en BAJO, el pulso se crea al apagar RGB)
        'PS': ['DGB'], # Pixel Separation
        'DG': ['DGA'], # Dump Gate
        'Int': ['HD1', 'HD2'] 
    }
    
    delays, states, recipes = parse_xml_sequencer_robust(args.file)
    
    # Secuencia MAS-CCD: Transferir carga entre etapas -> Horizontal -> Muestreo Skipper
    secuencia_lectura = ['transfer', 'horizontal', 'skipperQIS'] * args.pixels
    
    waveforms = generate_waveforms(secuencia_lectura, recipes, states, delays, signals_map)
    plot_phase_diagram(waveforms)
