import json
from datetime import datetime

# Carregar dados
with open('experiment.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f'Total de experimentos: {len(data)}')

# Função para extrair timestamp de ordenação
def get_sort_key(item):
    try:
        body = json.loads(item['body'])
        info = body.get('general_info', {})
        return info.get('start_date') or info.get('end_date') or 0
    except:
        return 0

# Ordenar por data crescente
data_sorted = sorted(data, key=get_sort_key)

# Verificar primeira e última data
first_body = json.loads(data_sorted[0]['body'])
last_body = json.loads(data_sorted[-1]['body'])
first_date = first_body.get('general_info', {}).get('start_date')
last_date = last_body.get('general_info', {}).get('start_date')

print(f'Primeira data: {datetime.fromtimestamp(first_date)}')
print(f'Ultima data: {datetime.fromtimestamp(last_date)}')

# Salvar arquivo ordenado
with open('experiment.json', 'w', encoding='utf-8') as f:
    json.dump(data_sorted, f, ensure_ascii=False)

print('Arquivo ordenado salvo!')
