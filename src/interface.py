import streamlit as st
import pandas as pd

# --- Interface visual para carregar pista e parâmetros ---
def main():
    st.title("Lap Time Simulator - Copa Truck Brasil")

    st.sidebar.header("Configuração da Simulação")
    # Upload do arquivo CSV (pista)
    file_pista = st.sidebar.file_uploader("Selecione o arquivo da pista (.csv):", type=["csv"])

    if file_pista is not None:
        pista_df = pd.read_csv(file_pista)
        st.subheader("Dados da pista carregados")
        st.dataframe(pista_df.head())

        # Parâmetros do veículo
        st.sidebar.subheader("Parâmetros do Veículo")
        massa = st.sidebar.slider("Massa (kg)", min_value=2000, max_value=10000, value=6000)

        # Botão para iniciar simulação
        if st.sidebar.button("Iniciar Simulação"):
            st.success("Iniciando simulação...")
            # Aqui será implementado o acionamento do backend
            st.write(f"Simulação com massa = {massa} kg")
            st.write(f"Simulação iniciou com pista: {file_pista.name}")
            # Futuro: chamar função de simulação
    else:
        st.info("Por favor, carregue o arquivo CSV da pista para continuar.")

if __name__ == "__main__":
    main()
