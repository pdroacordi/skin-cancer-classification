import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Caminhos para os dados
metadata_path = 'E:\\FAG\\PIBIC\\python\\skincancer\\data\\HAM10000_metadata.csv'
image_dir_1 = 'E:\\FAG\\PIBIC\\python\\skincancer\\data\\HAM10000_images_part_1'
image_dir_2 = 'E:\\FAG\\PIBIC\\python\\skincancer\\data\\HAM10000_images_part_2'

# Carregar o arquivo de metadados
metadata = pd.read_csv(metadata_path)

# Adicionar a extensão ".jpg" aos IDs das imagens
metadata['image_file'] = metadata['image_id'] + ".jpg"

# Caminho completo das imagens
metadata['image_path'] = metadata['image_file'].apply(
    lambda x: os.path.join(image_dir_1, x) if os.path.exists(os.path.join(image_dir_1, x))
    else os.path.join(image_dir_2, x)
)

# Verificar se todas as imagens foram mapeadas corretamente
if not all(metadata['image_path'].apply(os.path.exists)):
    raise FileNotFoundError("Algumas imagens listadas no CSV não foram encontradas nas pastas especificadas.")

# Divisão estratificada baseada nas classes (dx)
train_metadata, val_metadata = train_test_split(
    metadata, test_size=0.3, random_state=42, stratify=metadata['dx']
)

# Criar os arquivos .txt com os caminhos das imagens
train_metadata['image_path'].to_csv("../res/train_files.txt", index=False, header=False)
val_metadata['image_path'].to_csv("../res/val_files.txt", index=False, header=False)

# Exibir estatísticas
print(f"Total de imagens para treino: {len(train_metadata)}")
print(f"Total de imagens para validação: {len(val_metadata)}")

# Verificar a distribuição das classes nos conjuntos
print("\nDistribuição no conjunto de treinamento:")
print(train_metadata['dx'].value_counts())

print("\nDistribuição no conjunto de validação:")
print(val_metadata['dx'].value_counts())

print("Arquivos 'train_files.txt' e 'val_files.txt' criados com sucesso!")
