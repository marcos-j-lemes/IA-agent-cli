import chromadb
import json

# 1. Carrega os dados (assumindo que você salvou o JSON acima em 'dataset.json')
with open('dataset.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 2. Inicia o client do ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="comandos_sistema")

# 3. Prepara para inserir no formato que o ChromaDB gosta
ids = [item["id"] for item in data]
documents = [item["text"] for item in data] # O que o modelo vai ler e vetorizar
metadatas = [item["metadata"] for item in data] # Onde estão os comandos reais

# 4. Adiciona na base de dados
collection.add(
    ids=ids,
    documents=documents,
    metadatas=metadatas
)

# --- EXEMPLO DE USO DEPOIS ---
def achar_comando(pergunta_do_usuario):
    results = collection.query(
        query_texts=[pergunta_do_usuario],
        n_results=1 # Traz a melhor correspondência
    )
    
    # Extraindo os dados do resultado
    melhor_resultado = results['metadatas'][0][0]
    
    return f"""
    Categoria identificada: {melhor_resultado['categoria']}
    Comando para executar: {melhor_resultado['comando']}
    Descrição: {melhor_resultado['descricao']}
    """

# Testando...
print(achar_comando("Meu servidor tá lento, como vejo o que tá comendo a cpu?"))
# Saída esperada aponta para: top -b -n 1 -o +%CPU | head -20