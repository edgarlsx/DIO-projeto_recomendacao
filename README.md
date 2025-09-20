# 🧠 Sistema de Recomendação de Imagens por Similaridade Visual

Este projeto implementa um **sistema de recomendação de imagens** baseado em **similaridade visual**, utilizando **Deep Learning** com a arquitetura **ResNet50 pré-treinada**.

O objetivo é permitir que, ao buscar um produto (como um relógio ou camiseta), o sistema recomende **outros produtos visualmente semelhantes**, com base em características como **cor, textura, formato**, entre outros — **sem utilizar dados textuais** como marca, preço ou nome.

---

## 🚀 Funcionalidades

- 🔍 Classificação de imagens por **similaridade visual**
- 🧠 Extração de features com **ResNet50**
- 📊 Cálculo de similaridade com **distância cosseno**
- 🖼️ Recomendação das imagens mais similares ao input do usuário

---

## 🖼️ Exemplos de Aplicação

- Recomendação de produtos similares em **e-commerce**
- Busca visual baseada em imagens
- Organização de bancos de imagens por aparência
- Agrupamento visual de objetos (roupas, sapatos, acessórios, etc.)

---

## 🧰 Tecnologias utilizadas

- [Python 3.8+](https://www.python.org/)
- [PyTorch](https://pytorch.org/)
- [Torchvision](https://pytorch.org/vision/stable/index.html)
- [scikit-learn](https://scikit-learn.org/)
- [Pillow (PIL)](https://python-pillow.org/)

---

## 📁 Estrutura do Projeto

projeto_recomendacao/

│

├── recommender.py # Código principal do sistema

├── imagens_base/ # Imagens que compõem a base de dados

│ ├── relogio1.jpg

│ ├── camiseta2.jpg

│ └── ...

├── consulta/ # Imagem enviada pelo usuário

│ └── query.jpg

└── README.md # Este arquivo

---

## ⚙️ Como usar

### Clone este repositório

git clone https://github.com/seu-usuario/seu-repositorio.git
cd seu-repositorio

---

Instale as dependências.

pip install torch torchvision scikit-learn pillow

---

Execute o sistema.

python recommender.py

---

Resultado

O terminal exibirá as imagens mais similares à consulta, ordenadas pela similaridade visual (com seus respectivos valores).

---

Resultados esperados

Dado um relógio pesquisado pelo usuário, por exemplo:

Imagem consultada: consulta/query.jpg

Imagens recomendadas (mais similares):

imagens_base/relogio3.jpg (Similaridade: 0.9812)

imagens_base/relogio2.jpg (Similaridade: 0.9731)

imagens_base/relogio1.jpg (Similaridade: 0.9687)

...
