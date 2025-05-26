import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Tuple, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuración de página
st.set_page_config(
    page_title="Optimizador de Programación Lineal",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS mejorados
st.markdown("""
    <style>
    /* Tema principal */
    .main > div {
        padding-top: 2rem;
    }
    
    /* Botones personalizados */
    .stButton > button {
        width: 100%;
        border-radius: 12px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        border: none;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px 0 rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px 0 rgba(102, 126, 234, 0.4);
    }
    
    /* Tarjetas de métricas */
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px 0 rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    
    /* Encabezados */
    .section-header {
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 1.5rem;
        margin-bottom: 1rem;
    }
    
    /* Alertas personalizadas */
    .custom-success {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Inputs mejorados */
    .stNumberInput > div > div > input {
        border-radius: 8px;
        border: 2px solid #e0e6ed;
        transition: border-color 0.3s ease;
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 1px #667eea;
    }
    
    /* Sidebar personalizado */
    .css-1d391kg {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    </style>
""", unsafe_allow_html=True)

class LinearOptimizer:
    """Clase para manejar la optimización lineal"""
    
    def __init__(self):
        self.constraints = []
        self.objective_coeffs = []
        self.variables = []
        
    def add_constraint(self, coeffs: List[float], operator: str, rhs: float):
        """Agregar una restricción al modelo"""
        self.constraints.append({
            'coeffs': coeffs,
            'operator': operator,
            'rhs': rhs
        })
    
    def set_objective(self, coeffs: List[float], maximize: bool = True):
        """Establecer función objetivo"""
        self.objective_coeffs = coeffs if maximize else [-c for c in coeffs]
        self.maximize = maximize
    
    def solve_graphical(self) -> Tuple[Optional[Tuple[float, float]], Optional[float], List[Tuple[float, float]]]:
        """Resolver usando método gráfico"""
        if len(self.objective_coeffs) != 2:
            return None, None, []
        
        # Encontrar puntos extremos
        vertices = self._find_vertices()
        
        if not vertices:
            return None, None, []
        
        # Evaluar función objetivo en cada vértice
        best_vertex = None
        best_value = None
        
        for vertex in vertices:
            value = sum(c * x for c, x in zip(self.objective_coeffs, vertex))
            if best_value is None or (self.maximize and value > best_value) or (not self.maximize and value < best_value):
                best_value = value
                best_vertex = vertex
        
        # Ajustar valor si estamos minimizando
        if not self.maximize:
            best_value = -best_value
            
        return best_vertex, best_value, vertices
    
    def _find_vertices(self) -> List[Tuple[float, float]]:
        """Encontrar vértices de la región factible"""
        vertices = []
        
        # Puntos en los ejes
        vertices.extend([(0, 0)])
        
        # Intersecciones con ejes
        for constraint in self.constraints:
            coeffs = constraint['coeffs']
            rhs = constraint['rhs']
            
            # Intersección con eje X (y=0)
            if coeffs[0] != 0:
                x_intercept = rhs / coeffs[0]
                if x_intercept >= 0:
                    vertices.append((x_intercept, 0))
            
            # Intersección con eje Y (x=0)
            if coeffs[1] != 0:
                y_intercept = rhs / coeffs[1]
                if y_intercept >= 0:
                    vertices.append((0, y_intercept))
        
        # Intersecciones entre restricciones
        for i in range(len(self.constraints)):
            for j in range(i + 1, len(self.constraints)):
                vertex = self._line_intersection(
                    self.constraints[i]['coeffs'],
                    self.constraints[i]['rhs'],
                    self.constraints[j]['coeffs'],
                    self.constraints[j]['rhs']
                )
                if vertex and vertex[0] >= 0 and vertex[1] >= 0:
                    vertices.append(vertex)
        
        # Filtrar vértices factibles
        feasible_vertices = []
        for vertex in vertices:
            if self._is_feasible(vertex):
                feasible_vertices.append(vertex)
        
        # Eliminar duplicados
        unique_vertices = []
        for vertex in feasible_vertices:
            is_duplicate = False
            for existing in unique_vertices:
                if abs(vertex[0] - existing[0]) < 1e-6 and abs(vertex[1] - existing[1]) < 1e-6:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_vertices.append(vertex)
        
        return unique_vertices
    
    def _line_intersection(self, coeffs1: List[float], rhs1: float, coeffs2: List[float], rhs2: float) -> Optional[Tuple[float, float]]:
        """Encontrar intersección entre dos líneas"""
        try:
            det = coeffs1[0] * coeffs2[1] - coeffs1[1] * coeffs2[0]
            if abs(det) < 1e-10:  # Líneas paralelas
                return None
            
            x = (rhs1 * coeffs2[1] - rhs2 * coeffs1[1]) / det
            y = (coeffs1[0] * rhs2 - coeffs2[0] * rhs1) / det
            
            return (x, y)
        except:
            return None
    
    def _is_feasible(self, point: Tuple[float, float]) -> bool:
        """Verificar si un punto es factible"""
        x, y = point
        
        if x < -1e-6 or y < -1e-6:  # No negatividad
            return False
        
        for constraint in self.constraints:
            coeffs = constraint['coeffs']
            operator = constraint['operator']
            rhs = constraint['rhs']
            
            value = coeffs[0] * x + coeffs[1] * y
            
            if operator == '<=' and value > rhs + 1e-6:
                return False
            elif operator == '>=' and value < rhs - 1e-6:
                return False
            elif operator == '=' and abs(value - rhs) > 1e-6:
                return False
        
        return True

def create_plotly_graph(optimizer: LinearOptimizer, x_range: Tuple[float, float] = (0, 20), y_range: Tuple[float, float] = (0, 20)):
    """Crear gráfico interactivo con Plotly"""
    fig = go.Figure()
    
    x = np.linspace(x_range[0], x_range[1], 1000)
    
    # Colores para las restricciones
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
    
    # Graficar restricciones
    for i, constraint in enumerate(optimizer.constraints):
        coeffs = constraint['coeffs']
        rhs = constraint['rhs']
        operator = constraint['operator']
        
        if coeffs[1] != 0:
            y_line = (rhs - coeffs[0] * x) / coeffs[1]
            
            # Línea de restricción
            fig.add_trace(go.Scatter(
                x=x, y=y_line,
                mode='lines',
                name=f'R{i}: {coeffs[0]}x₀ + {coeffs[1]}x₁ {operator} {rhs}',
                line=dict(color=colors[i % len(colors)], width=3),
                hovertemplate=f'<b>Restricción {i}</b><br>x₀: %{{x}}<br>x₁: %{{y}}<extra></extra>'
            ))
    
    # Región factible
    vertices = optimizer._find_vertices()
    if vertices:
        # Ordenar vértices para formar el polígono
        if len(vertices) > 2:
            center = (sum(v[0] for v in vertices) / len(vertices), 
                     sum(v[1] for v in vertices) / len(vertices))
            vertices = sorted(vertices, key=lambda v: np.arctan2(v[1] - center[1], v[0] - center[0]))
        
        if len(vertices) >= 3:
            vertices_array = np.array(vertices + [vertices[0]])  # Cerrar el polígono
            
            fig.add_trace(go.Scatter(
                x=vertices_array[:, 0],
                y=vertices_array[:, 1],
                fill='toself',
                fillcolor='rgba(102, 126, 234, 0.3)',
                line=dict(color='rgba(102, 126, 234, 0.8)', width=2),
                name='Región Factible',
                hovertemplate='<b>Región Factible</b><br>x₀: %{x}<br>x₁: %{y}<extra></extra>'
            ))
        
        # Marcar vértices
        for i, vertex in enumerate(vertices):
            fig.add_trace(go.Scatter(
                x=[vertex[0]], y=[vertex[1]],
                mode='markers+text',
                marker=dict(size=12, color='#FF4757', symbol='circle', 
                           line=dict(color='white', width=2)),
                text=[f'P{i}'],
                textposition="top center",
                name=f'Vértice P{i} ({vertex[0]:.2f}, {vertex[1]:.2f})',
                hovertemplate=f'<b>Vértice P{i}</b><br>x₀: {vertex[0]:.3f}<br>x₁: {vertex[1]:.3f}<extra></extra>'
            ))
    
    # Configuración del gráfico
    fig.update_layout(
        title=dict(
            text='<b>Análisis Gráfico de Programación Lineal</b>',
            x=0.5,
            font=dict(size=20, color='#2c3e50')
        ),
        xaxis=dict(
            title='<b>x₀</b>',
            range=x_range,
            gridcolor='lightgray',
            gridwidth=1,
            zeroline=True,
            zerolinecolor='black',
            zerolinewidth=2
        ),
        yaxis=dict(
            title='<b>x₁</b>',
            range=y_range,
            gridcolor='lightgray',
            gridwidth=1,
            zeroline=True,
            zerolinecolor='black',
            zerolinewidth=2
        ),
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.05
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        hovermode='closest'
    )
    
    return fig

def main():
    # Título principal con emoji y gradiente
    st.markdown("""
        <div style='text-align: center; margin-bottom: 2rem;'>
            <h1 style='background: linear-gradient(90deg, #667eea, #764ba2); 
                       -webkit-background-clip: text; -webkit-text-fill-color: transparent; 
                       font-size: 3rem; font-weight: 800; margin-bottom: 0.5rem;'>
                🧮 Optimizador de Programación Lineal
            </h1>
            <p style='color: #666; font-size: 1.2rem; margin: 0;'>
                Herramienta avanzada para resolver problemas de optimización lineal
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar para configuración
    with st.sidebar:
        st.markdown("<div class='section-header'>⚙️ Configuración</div>", unsafe_allow_html=True)
        
        # Método de solución
        metodo = st.radio(
            "**Método de Solución**",
            ["🔍 Método Gráfico", "📊 Método Simplex"],
            help="El método gráfico funciona solo para 2 variables"
        )
        
        # Tipo de optimización
        tipo_optimizacion = st.radio(
            "**Tipo de Optimización**",
            ["📈 Maximizar", "📉 Minimizar"]
        )
        
        st.markdown("---")
        
        # Configuración de variables
        st.markdown("<div class='section-header'>🔢 Variables</div>", unsafe_allow_html=True)
        x0_label = st.text_input("**Nombre de x₀:**", value="Producto A", key="x0_label")
        x1_label = st.text_input("**Nombre de x₁:**", value="Producto B", key="x1_label")
        
        st.markdown("---")
        
        # Función objetivo
        st.markdown("<div class='section-header'>🎯 Función Objetivo</div>", unsafe_allow_html=True)
        c0 = st.number_input(f"**Coeficiente de {x0_label}:**", value=3.0, step=0.1, key="c0")
        c1 = st.number_input(f"**Coeficiente de {x1_label}:**", value=2.0, step=0.1, key="c1")
        
        # Mostrar función objetivo
        operador = "Max" if "Maximizar" in tipo_optimizacion else "Min"
        st.markdown(f"""
            <div class='metric-card'>
                <strong>Función Objetivo:</strong><br>
                {operador} Z = {c0}x₀ + {c1}x₁
            </div>
        """, unsafe_allow_html=True)
    
    # Área principal
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.markdown("<div class='section-header'>📏 Restricciones</div>", unsafe_allow_html=True)
        
        # Inicializar el optimizador
        optimizer = LinearOptimizer()
        
        # Contenedor para restricciones dinámicas
        if 'num_constraints' not in st.session_state:
            st.session_state.num_constraints = 2
        
        constraints_data = []
        
        for i in range(st.session_state.num_constraints):
            with st.expander(f"**Restricción {i+1}**", expanded=True):
                col_a, col_b = st.columns(2)
                with col_a:
                    a_coeff = st.number_input(f"Coef. x₀", value=1.0, step=0.1, key=f"a_{i}")
                with col_b:
                    b_coeff = st.number_input(f"Coef. x₁", value=1.0, step=0.1, key=f"b_{i}")
                
                col_op, col_rhs = st.columns(2)
                with col_op:
                    operator = st.selectbox("Operador", ["<=", ">=", "="], key=f"op_{i}")
                with col_rhs:
                    rhs = st.number_input("Valor", value=10.0, step=0.1, key=f"rhs_{i}")
                
                constraints_data.append((a_coeff, b_coeff, operator, rhs))
                optimizer.add_constraint([a_coeff, b_coeff], operator, rhs)
        
        # Botones para agregar/quitar restricciones
        col_add, col_remove = st.columns(2)
        with col_add:
            if st.button("➕ Agregar Restricción"):
                st.session_state.num_constraints += 1
                st.rerun()
        
        with col_remove:
            if st.button("➖ Quitar Restricción") and st.session_state.num_constraints > 1:
                st.session_state.num_constraints -= 1
                st.rerun()
        
        # Configurar optimizador
        optimizer.set_objective([c0, c1], "Maximizar" in tipo_optimizacion)
        
        # Mostrar resumen de restricciones
        st.markdown("### 📋 Resumen del Modelo")
        modelo_text = f"**{operador}** Z = {c0}x₀ + {c1}x₁\n\n**Sujeto a:**\n"
        for i, (a, b, op, rhs) in enumerate(constraints_data):
            modelo_text += f"• {a}x₀ + {b}x₁ {op} {rhs}\n"
        modelo_text += "• x₀, x₁ ≥ 0"
        
        # Formatear el texto para HTML
        modelo_html = modelo_text.replace('**', '<strong>')
        modelo_html = modelo_html.replace('</strong><strong>', '</strong>')
        modelo_html = modelo_html.replace('\n', '<br>')
        
        st.markdown(f"""
            <div class='metric-card'>
                {modelo_html}
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if "Gráfico" in metodo:
            st.markdown("<div class='section-header'>📊 Análisis Gráfico</div>", unsafe_allow_html=True)
            
            # Resolver el problema
            solution, optimal_value, vertices = optimizer.solve_graphical()
            
            if solution:
                # Crear y mostrar gráfico
                fig = create_plotly_graph(optimizer)
                st.plotly_chart(fig, use_container_width=True)
                
                # Resultados
                st.markdown("### 🎯 Resultados Óptimos")
                
                col_res1, col_res2, col_res3 = st.columns(3)
                with col_res1:
                    st.metric("**Valor Óptimo**", f"{optimal_value:.3f}")
                with col_res2:
                    st.metric(f"**{x0_label}**", f"{solution[0]:.3f}")
                with col_res3:
                    st.metric(f"**{x1_label}**", f"{solution[1]:.3f}")
                
                # Tabla de vértices
                st.markdown("### 📍 Análisis de Vértices")
                vertices_data = []
                for i, vertex in enumerate(vertices):
                    z_value = c0 * vertex[0] + c1 * vertex[1]
                    if not ("Maximizar" in tipo_optimizacion):
                        z_value = -z_value  # Ajustar para minimización
                    
                    is_optimal = abs(vertex[0] - solution[0]) < 1e-6 and abs(vertex[1] - solution[1]) < 1e-6
                    vertices_data.append({
                        "Vértice": f"P{i}",
                        f"{x0_label}": f"{vertex[0]:.3f}",
                        f"{x1_label}": f"{vertex[1]:.3f}",
                        "Valor Z": f"{z_value:.3f}",
                        "Óptimo": "✅" if is_optimal else "❌"
                    })
                
                df_vertices = pd.DataFrame(vertices_data)
                st.dataframe(df_vertices, use_container_width=True)
                
            else:
                st.error("⚠️ No se encontró solución factible. Verifique las restricciones.")
        
        else:
            st.markdown("<div class='section-header'>📊 Método Simplex</div>", unsafe_allow_html=True)
            st.info("🚧 **Próximamente:** El método Simplex estará disponible en la siguiente versión.")
            
            # Placeholder para futuras implementaciones
            st.markdown("""
                **Características planeadas:**
                - ✅ Solución de problemas con múltiples variables
                - ✅ Análisis de sensibilidad
                - ✅ Detección de soluciones múltiples
                - ✅ Casos especiales (ilimitado, infactible)
                - ✅ Exportación de resultados
            """)

    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #666; padding: 1rem;'>
            <p>🔬 <strong>Optimizador de Programación Lineal</strong> - Versión 2.0</p>
            <p>Desarrollado con ❤️ usando Streamlit y Plotly</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()