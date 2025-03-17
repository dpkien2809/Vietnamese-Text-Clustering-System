# Vietnamese Text Clustering System

## Overview
This project focuses on processing and clustering Vietnamese text notes using Natural Language Processing (NLP) techniques. It leverages Sentence-BERT for vectorization and KMeans for clustering, enabling efficient topic identification and representative note selection.

## Features
- **Text Preprocessing**: Tokenization, spelling correction, and normalization.
- **Vectorization**: Using Sentence-BERT to generate meaningful vector representations of text.
- **Clustering**: Applying KMeans to group similar text notes.
- **Demo Interface**: A Streamlit-based UI allowing users to input text and receive cluster information.

## Technologies Used
- **Python**
- **Hugging Face Transformers** (for Sentence-BERT)
- **scikit-learn** (for KMeans clustering)
- **Streamlit** (for demo UI)

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/your-repo/vietnamese-text-clustering.git
   cd vietnamese-text-clustering
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the demo:
   ```sh
   streamlit run finale.py
   ```

## Usage
1. Input a Vietnamese text note.
2. The system processes and assigns it to a relevant cluster.
3. It displays the cluster ID and a representative note from that cluster.

## Next Steps
- Expand dataset diversity to improve clustering accuracy.
- Optimize parameters for better performance.
- Enhance the user interface for real-world applications.

## Contributors
- **GaMinD** - Technical Consultant
- **DoPhucKien** - co-developer, co-logic programming.
