import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_groq import ChatGroq

# Configuración de la página
st.set_page_config("LLMs con DataFrames", layout="wide")
st.title("📊 LLMs con DataFrames")

# Inicializar el modelo
model = ChatGroq(
    model="llama-3.3-70b-versatile", 
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=st.secrets["groq"]["API_KEY"],
)

# Inicializar estados de sesión
if "messages" not in st.session_state:
    st.session_state.messages = []
if "df" not in st.session_state:
    st.session_state.df = None
if "agent" not in st.session_state:
    st.session_state.agent = None
if "df_clean" not in st.session_state:
    st.session_state.df_clean = None

def reloadChat():
    st.session_state.messages = []
    st.session_state.df = None
    st.session_state.agent = None
    st.session_state.df_clean = None

def clean_dataframe(df):
    """Limpia el dataframe para evitar problemas de serialización"""
    df_clean = df.copy()
    
    # Limitar longitud de textos en columnas object
    for col in df_clean.select_dtypes(include=['object']).columns:
        df_clean[col] = df_clean[col].astype(str).str.slice(0, 100)
    
    # Reemplazar valores problemáticos
    df_clean = df_clean.replace([np.nan, None], 'No especificado')
    
    return df_clean

# Uploader de archivos
file = st.file_uploader("Elige un archivo CSV o XLS", type=["csv", "xlsx", "xls"], on_change=reloadChat)

if file is not None:
    try:
        if file.name.endswith(".csv"):
            df = pd.read_csv(file)
        elif file.name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file)
        
        st.session_state.df = df
        st.session_state.df_clean = clean_dataframe(df)
        
        # Muestra preview del dataframe limpio
        st.subheader("📋 Vista previa de los datos")
        st.dataframe(st.session_state.df_clean.head())
        
        # Crear el agente con el dataframe original
        st.session_state.agent = create_pandas_dataframe_agent(
            model, 
            df,  # Usa el dataframe original para análisis
            allow_dangerous_code=True,
            verbose=True,
            max_iterations=3,
            early_stopping_method="force"
        )
        
        # Muestra información del dataset
        with st.expander("📊 Información del Dataset"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"*Filas:* {len(df)}")
                st.write(f"*Columnas:* {len(df.columns)}")
            with col2:
                st.write("*Columnas numéricas:*")
                st.write(list(df.select_dtypes(include=['number']).columns))
                st.write("*Columnas categóricas:*")
                st.write(list(df.select_dtypes(include=['object']).columns))
        
    except Exception as e:
        st.error(f"Error al cargar el archivo: {str(e)}")

# Visualizaciones
def create_visualizations(df):
    """Crea visualizaciones automáticas mejoradas"""
    st.subheader("📈 Visualizaciones Automáticas")
    
    # Configurar estilo
    plt.style.use('default')
    sns.set_palette("husl")
    plt.rcParams['font.size'] = 10
    plt.rcParams['figure.figsize'] = (12, 8)
    
    # Gráficas para columnas numéricas
    numeric_cols = df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        st.write(f"### 📐 Análisis de {col}")
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Histograma
        df[col].hist(ax=axes[0], bins=10, edgecolor='black', alpha=0.7)
        axes[0].set_title(f'Distribución de {col}', fontsize=12, pad=20)
        axes[0].set_xlabel(col, fontsize=10)
        axes[0].set_ylabel('Frecuencia', fontsize=10)
        axes[0].grid(False)
        
        # Boxplot
        df.boxplot(column=col, ax=axes[1])
        axes[1].set_title(f'Boxplot de {col}', fontsize=12, pad=20)
        axes[1].grid(False)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Estadísticas
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Media", f"{df[col].mean():.2f}")
        with col2:
            st.metric("Mediana", f"{df[col].median():.2f}")
        with col3:
            st.metric("Desviación", f"{df[col].std():.2f}")
        with col4:
            st.metric("Rango", f"{df[col].min():.2f} - {df[col].max():.2f}")
    
    # Gráficas para columnas categóricas
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols[:5]:  # Máximo 5 columnas categóricas
        if df[col].nunique() <= 15:  # Solo si tiene pocas categorías
            st.write(f"### 📊 Análisis de {col}")
            
            value_counts = df[col].value_counts()
            
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # Gráfico de barras horizontal 
            value_counts.plot(kind='barh', ax=axes[0], edgecolor='black', alpha=0.7)
            axes[0].set_title(f'Distribución de {col}', fontsize=12, pad=20)
            axes[0].set_xlabel('Frecuencia', fontsize=10)
            axes[0].set_ylabel(col, fontsize=10)
            axes[0].grid(False)
            
            # Gráfico de pie
            if len(value_counts) <= 8:  # Solo pie chart para pocas categorías
                axes[1].pie(value_counts.values, labels=value_counts.index, 
                           autopct='%1.1f%%', startangle=90)
                axes[1].set_title(f'Proporción de {col}', fontsize=12, pad=20)
            else:
                axes[1].text(0.5, 0.5, 'Demasiadas categorías\npara gráfico de pie', 
                            ha='center', va='center', fontsize=12)
                axes[1].set_title(f'Proporción de {col}', fontsize=12, pad=20)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Tabla de frecuencias
            st.write("*Frecuencias:*")
            freq_df = pd.DataFrame({
                'Categoría': value_counts.index,
                'Frecuencia': value_counts.values,
                'Porcentaje': (value_counts.values / len(df) * 100).round(2)
            })
            st.dataframe(freq_df)

# Función segura para ejecutar el agente
def safe_agent_execute(prompt):
    try:
        response = st.session_state.agent.run(prompt)
        return response
    except Exception as e:
        error_msg = str(e)
        if "iteration" in error_msg.lower() or "limit" in error_msg.lower():
            return "He completado el análisis básico. Para un análisis más profundo, por favor formula preguntas más específicas."
        return f"Error: {error_msg}. Intenta reformular tu pregunta."

# Chat input
if prompt := st.chat_input("💬 Escribe tu pregunta sobre los datos"):
    if st.session_state.agent is None:
        st.warning("⏳ Por favor, carga primero un archivo CSV o Excel.")
    else:
        # Agregar mensaje del usuario
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Prompt final
        prompt_final = f"""Eres un experto analista de datos. Responde en español.

Pregunta: {prompt}

Instrucciones CRÍTICAS:
1.⁠ ⁠Responde solo con análisis textual, NO intentes crear gráficas con código
2.⁠ ⁠Limita tu respuesta a máximo 2-3 acciones/iteraciones
3.⁠ ⁠Proporciona estadísticas descriptivas claras
4.⁠ ⁠Para datos categóricos, menciona frecuencias y porcentajes
5.⁠ ⁠Para datos numéricos, menciona medidas de tendencia central
6.⁠ ⁠Si el análisis requiere más pasos, sugiere preguntas específicas para profundizar

Datos disponibles: {len(st.session_state.df)} filas, {len(st.session_state.df.columns)} columnas
"""
        
        with st.spinner("🔍 Analizando datos..."):
            try:
                # Ejecutar el agente
                response = safe_agent_execute(prompt_final)
                
                # Mostrar respuesta
                with st.chat_message("assistant"):
                    st.markdown(response)
                
                # Guardar en historial
                st.session_state.messages.append({"role": "assistant", "content": response})
                
                # Mostrar visualizaciones automáticas si se pide análisis
                if any(word in prompt.lower() for word in ['análisis', 'gráfic', 'visualiz', 'chart', 'plot', 'estadístic']):
                    with st.spinner("🖼️ Generando visualizaciones..."):
                        create_visualizations(st.session_state.df_clean)
                
            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                with st.chat_message("assistant"):
                    st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Botón para mostrar visualizaciones automáticas
if st.session_state.df_clean is not None:
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📈 Mostrar Visualizaciones Automáticas", type="primary"):
            create_visualizations(st.session_state.df_clean)
    with col2:
        if st.button("🧹 Limpiar Conversación"):
            reloadChat()

# Mostrar historial de conversación
if st.session_state.messages:
    st.subheader("💬 Historial de Conversación")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Footer
st.markdown("---")
st.caption("💡 Consejo: Formula preguntas específicas como '¿Cuál es la distribución por género?' o '¿Qué edades tienen los participantes?'")