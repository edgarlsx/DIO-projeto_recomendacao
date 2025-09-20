import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

# Carregar modelo pré-treinado (ResNet50) sem a última camada de classificação
def load_model():
    model = models.resnet50(pretrained=True)
    model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove a camada final
    model.eval()
    return model

# Pré-processamento da imagem
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Tamanho padrão para ResNet
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(image_path).convert('RGB')
    return transform(img).unsqueeze(0)  # Adiciona dimensão de batch

# Extrair vetores de características
def extract_features(model, image_path):
    img_tensor = preprocess_image(image_path)
    with torch.no_grad():
        features = model(img_tensor)
    return features.squeeze().numpy()  # Remove dimensões extras

# Construir a base de dados de imagens
def build_feature_database(model, image_folder):
    features_db = []
    image_paths = []

    for filename in os.listdir(image_folder):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')):
            path = os.path.join(image_folder, filename)
            features = extract_features(model, path)
            features_db.append(features)
            image_paths.append(path)
    
    return np.array(features_db), image_paths

# Fazer recomendação com base em uma imagem de consulta
def recommend_similar_images(query_image_path, features_db, image_paths, model, top_k=5):
    query_features = extract_features(model, query_image_path)
    similarities = cosine_similarity([query_features], features_db)[0]
    indices = similarities.argsort()[-top_k:][::-1]

    print(f"\nImagem consultada: {query_image_path}")
    print("Imagens recomendadas (mais similares):")
    for idx in indices:
        print(f"{image_paths[idx]} (Similaridade: {similarities[idx]:.4f})")

# ---------- MAIN ----------
if __name__ == "__main__":
    # Caminho para a pasta de imagens (base de dados)
    image_folder = "imagens_base"  # Ex: ./imagens_base/
    
    # Caminho da imagem de consulta (ex: relógio pesquisado pelo usuário)
    query_image = "consulta/query.jpg"  # Ex: ./consulta/query.jpg

    print("Carregando modelo e extraindo características...")
    model = load_model()

    print("Construindo base de imagens...")
    features_db, image_paths = build_feature_database(model, image_folder)

    print("Realizando recomendação por similaridade visual...")
    recommend_similar_images(query_image, features_db, image_paths, model, top_k=5)
