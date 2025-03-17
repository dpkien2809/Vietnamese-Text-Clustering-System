import streamlit as st
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer

def main():
    data = joblib.load("kmeans_with_labels.pkl")
    model = data["model"]
    labels = data["labels"]
    cluster_centers = model.cluster_centers_

    texts = [
    "tô nhỏ",
    "ít",
    "ko phộng",
    "không phộng",
    "ko đậu phộng",
    "không đậu phộng",
    "không đậu đậu phộng",
    "không hành",
    "không hành tây",
    "ko hành",
    "không tất cả hành",
    "không hành phi",
    "ko rau ko hành",
    "ko rau không hành",
    "không rau ko hành",
    "ít cay",
    "ít cay 1 xíu",
    "ít cay tẹo tẹo",
    "nhiều cay",
    "nhìu cay",
    "nhiều cay một chút",
    "nhiều sốt",
    "1 nhiều sốt",
    "nhiều bún",
    "nhiều bún 1 tô thay carol bằng giá sống",
    "ít bún",
    "ít mè th",
    "ít mè",
    "không mè",
    "ko mè",
    "rau trụng",
    "rau sống",
    "có rau nêm",
    "chỉ ăn tía tô",
    "tía tô",
    "không rau",
    "ko rau",
    "không rau nêm",
    "ko rau nêm",
    "không rau răm",
    "ko rau răm",
    "ko răm",
    "rau để riêng",
    "rau riêng",
    "rau, bún để riêng",
    "rau, bún khô để riêng",
    "gọi trà đá",
    "gọi trà",
    "1 trà đá",
    "trà đá",
    "gọi thêm trà đá",
    "trà không đá",
    "không đá",
    "cam ít đá",
    "hỏi uống cam ít đá",
    "cam ít đường",
    "ít đường",
    "có rau có hành",
    "có rau nêm và hành tây",
    "không rau và hành",
    "không rau không hành",
    "ko rau ko hành",
    "ko rau ko hành",
    "không rau ko hành",
    "ko rau không hành",
    "ko hành - rau thơm",
    "ko rau thơm - hành",
    "không hành - rau thơm",
    "không rau thơm - hành",
    "không rau, không hành, không trứng",
    "ko rau, ko hành, ko trứng",
    "không hành",
    "ko hành",
    "không chả bò",
    "ko chả bò",
    "chén trứng thêm",
    "chén trứng",
    "một chén trứng thêm",
    "1 chén trứng thêm",
    "chỉ ăn bắp và chả bò",
    "bắp và chả bò",
    "chỉ ăn bắp",
    "chỉ bắp hoa",
    "không ăn bắp",
    "ko ăn bắp",
    "không bắp",
    "ko bắp",
    "ko bắp bò",
    "không bắp bò",
    "thêm bắp",
    "gọi thêm bắp",
    "chỉ ăn gân",
    "gân",
    "chỉ ăn huyết",
    "huyết",
    "chỉ ăn giò móng",
    "giò móng",
    "chỉ ăn giò nạc",
    "giò nạc",
    "chỉ ăn giò khoanh",
    "giò khoanh",
    "chỉ ăn giò gân",
    "giò gân",
    "không ăn tái",
    "ko ăn tái",
    "không tái",
    "ko tái",
    "kg tái",
    "không ăn cua",
    "ko ăn cua",
    "không cua",
    "ko cua",
    "không ăn chả cua",
    "ko ăn chả cua",
    "không chả cua",
    "ko chả cua",
    "kg chả cua",
    "không ăn trứng gà",
    "ko ăn trứng gà",
    "không trứng gà",
    "ko trứng gà",
    "đổi tái thành bắp",
    "thay tái thành bắp",
    "ăn tại chỗ",
    "mang về",
    ]

    cluster_ids = list(set(labels))

    sbert_model = SentenceTransformer('keepitreal/vietnamese-sbert')

    representative_samples = {}
    embeddings = sbert_model.encode(texts, batch_size=1)  

    for cluster_id in cluster_ids:
        cluster_data = embeddings[np.where(labels == cluster_id)]
        center = cluster_centers[cluster_id]
        distances = np.linalg.norm(cluster_data - center, axis=1)
        representative_index = np.argmin(distances)
        representative_samples[cluster_id] = texts[np.where(labels == cluster_id)[0][representative_index]]

    st.title("Real-time Text Clustering with Streamlit")
    user_input = st.text_input("Enter text to cluster:")

    if user_input:
        embedding = sbert_model.encode([user_input], batch_size=1)
        cluster_id = model.predict(embedding)[0]
        st.write(f"**Cluster ID:** {cluster_id}")
        st.write(f"**Representative Text:** {representative_samples.get(cluster_id, 'No representative text available')}")

if __name__ == "__main__":
    main()
