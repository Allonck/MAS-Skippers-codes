import graphviz

def create_vertical_columns_diagram():
    # 1. Configuración Global: Orientación Vertical (Top-Bottom)
    # 'newrank' permite alinear nodos a través de diferentes clusters
    dot = graphviz.Digraph('MASSKIP_Vertical', comment='Pipeline Vertical Towers', format='png')
    dot.attr(rankdir='TB', splines='ortho', newrank='true', nodesep='0.6', ranksep='0.5')
    
    # Estilos
    dot.attr('node', shape='rect', style='filled', fontname='Helvetica', fontsize='10', height='0.5')
    dot.attr('edge', fontname='Helvetica', fontsize='9', color='#555555')

    # =========================================================
    # COLUMNA 1: REDUCCIÓN (AZUL)
    # =========================================================
    with dot.subgraph(name='cluster_ccd') as c:
        c.attr(label='1. REDUCCIÓN (mas-ccd)', style='rounded', color='#1565c0', bgcolor='#e3f2fd')
        c.attr('node', fillcolor='#bbdefb', color='#1565c0')
        
        # Nodos Verticales
        c.node('Raw', 'Imágenes Raw\n(16 Extensiones)', shape='cylinder', fillcolor='#e0e0e0')
        c.node('Overscan', '1. Corrección Overscan')
        c.node('Masters', '2. Crear Masters\n(Bias, Dark, Flat)')
        c.node('FlatNorm', '3. Norm. Flat\n(Polinomio 2D)')
        c.node('PreProc', '4. Calibración\n(Bias, Dark, Flat)', shape='diamond', style='filled,rounded')
        c.node('Cosmic', '5. L.A.Cosmic')
        c.node('Combine', '6. Combinar 16 Ch')
        c.node('Gain', '7. Conversión ADU -> e-\n(Ganancia Efectiva)')
        c.node('Reduced', 'Imágenes Reducidas\n(comb_*.fits)', shape='note', fillcolor='#f3e5f5', color='#7b1fa2')

        # Conexiones Verticales (TB por defecto)
        c.edge('Raw', 'Overscan')
        c.edge('Overscan', 'Masters')
        c.edge('Masters', 'FlatNorm')
        c.edge('FlatNorm', 'PreProc')
        c.edge('PreProc', 'Cosmic')
        c.edge('Cosmic', 'Combine')
        c.edge('Combine', 'Gain')
        c.edge('Gain', 'Reduced')

    # =========================================================
    # COLUMNA 2: APILADO (VERDE)
    # =========================================================
    with dot.subgraph(name='cluster_stack') as c:
        c.attr(label='2. APILADO (mas-stack)', style='rounded', color='#2e7d32', bgcolor='#e8f5e9')
        c.attr('node', fillcolor='#c8e6c9', color='#2e7d32')
        
        c.node('RefSel', '1. Sel. Referencia')
        c.node('Detect', '2. Detección Estrellas')
        c.node('AlignCheck', '¿Suficientes Estrellas?', shape='diamond', fillcolor='#fff9c4', color='#fbc02d')
        
        # Ramas paralelas para alineación
        c.node('Astroalign', 'A: Geometría\n(Triangulación)', style='dashed')
        c.node('HistShift', 'B: Histograma\n(Shift +/- 50px)', style='dashed')
        
        c.node('Trans', '4. Transformación')
        c.node('Stack', '5. Combinar\n(Median/SigmaClip)')
        c.node('Coadd', 'Master Deep Field\n(DeepcombLAE.fits)', shape='note', fillcolor='#f3e5f5', color='#7b1fa2')

        c.edge('RefSel', 'Detect')
        c.edge('Detect', 'AlignCheck')
        c.edge('AlignCheck', 'Astroalign', label='> 3 Stars')
        c.edge('AlignCheck', 'HistShift', label='< 3 Stars')
        c.edge('Astroalign', 'Trans')
        c.edge('HistShift', 'Trans')
        c.edge('Trans', 'Stack')
        c.edge('Stack', 'Coadd')

    # =========================================================
    # COLUMNA 3: FOTOMETRÍA (NARANJA)
    # =========================================================
    with dot.subgraph(name='cluster_phot') as c:
        c.attr(label='3. FOTOMETRÍA (mas-phot)', style='rounded', color='#ef6c00', bgcolor='#fff3e0')
        c.attr('node', fillcolor='#ffe0b2', color='#ef6c00')
        
        c.node('SourceDet', '1. Detección Fuentes')
        c.node('Aperture', '2. Fotometría Apertura')
        c.node('LocalBack', '3. Resta Fondo Local')
        c.node('ZeroPoint', '4. Calib. ZeroPoint')
        c.node('Catalog', '5. Generar Catálogo')
        c.node('FinalTable', 'Tabla Final\n(.csv / .fits)', shape='folder', fillcolor='#f3e5f5', color='#7b1fa2')

        c.edge('SourceDet', 'Aperture')
        c.edge('Aperture', 'LocalBack')
        c.edge('LocalBack', 'ZeroPoint')
        c.edge('ZeroPoint', 'Catalog')
        c.edge('Catalog', 'FinalTable')

    # =========================================================
    # TRUCO DE ALINEACIÓN (EL SECRETO)
    # =========================================================
    # 1. Conectar las cabeceras con flechas invisibles para forzar el orden Izquierda -> Derecha
    dot.edge('Raw', 'RefSel', style='invis')
    dot.edge('RefSel', 'SourceDet', style='invis')
    
    # 2. Forzar que las cabeceras estén en el mismo nivel jerárquico (rank)
    dot.body.append('{ rank=same; Raw; RefSel; SourceDet }')

    # =========================================================
    # CONEXIONES INTER-COLUMNAS
    # =========================================================
    # Usamos constraint='false' para que estas flechas no alteren la posición vertical de las columnas
    
    # De Reduced (Col 1 Abajo) a Stack (Col 2 Arriba)
    dot.edge('Reduced', 'RefSel', label='Deep Mode', color='#2e7d32', penwidth='2.0', constraint='false')
    
    # De Reduced (Col 1 Abajo) a Phot (Col 3 Arriba)
    dot.edge('Reduced', 'SourceDet', label='Standard Mode', color='#ef6c00', penwidth='2.0', constraint='false')
    
    # De Coadd (Col 2 Abajo) a Phot (Col 3 Arriba)
    dot.edge('Coadd', 'SourceDet', color='#7b1fa2', constraint='false')

    # Renderizar
    output_path = dot.render('pipeline_diagram_vertical', view=False)
    print(f"✅ Diagrama de torres verticales generado: {output_path}")

if __name__ == '__main__':
    create_vertical_columns_diagram()