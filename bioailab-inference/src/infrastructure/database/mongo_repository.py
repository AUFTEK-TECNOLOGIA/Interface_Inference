"""
Repositório MongoDB para dados de experimentos.

Responsabilidade única: acesso ao banco de dados MongoDB.
"""

import re
from typing import Any, Dict, List, Optional

from bson import ObjectId
from pymongo import MongoClient


class MongoRepository:
    """Repositório para acesso a dados de experimentos no MongoDB."""
    
    _TENANT_RE = re.compile(r"^[a-z0-9_\-]+$")
    
    def __init__(self, mongo_uri: str, db_prefix: str = "bioailab_"):
        """
        Inicializa o repositório.
        
        Args:
            mongo_uri: URI de conexão do MongoDB
            db_prefix: Prefixo para nome do database por tenant
        """
        self._client = MongoClient(mongo_uri, tz_aware=True)
        self._db_prefix = db_prefix
        self._db_cache: Dict[str, Any] = {}
    
    def get_db(self, tenant: str):
        """
        Obtém o banco de dados específico para o tenant.
        
        Args:
            tenant: Identificador do tenant
        
        Returns:
            Database do MongoDB para o tenant
        
        Raises:
            ValueError: Se o tenant for inválido
        """
        if not tenant or not self._TENANT_RE.match(tenant):
            raise ValueError(f"Tenant inválido: {tenant}")
        
        db_name = f"{self._db_prefix}{tenant}"
        
        if db_name not in self._db_cache:
            self._db_cache[db_name] = self._client[db_name]
        
        return self._db_cache[db_name]
    
    def get_experiment(self, tenant: str, exp_id: str) -> Optional[Dict[str, Any]]:
        """
        Busca um experimento por ID.
        
        Args:
            tenant: Identificador do tenant
            exp_id: ID do experimento
        
        Returns:
            Documento do experimento ou None
        """
        db = self.get_db(tenant)
        col = db["experiments"]
        
        # Tentativa 1: _id como string literal
        doc = col.find_one({"_id": exp_id})
        if doc:
            return doc
        
        # Tentativa 2: _id como ObjectId
        try:
            doc = col.find_one({"_id": ObjectId(exp_id)})
            if doc:
                return doc
        except Exception:
            pass
        
        # Tentativa 3: campo 'id'
        return col.find_one({"id": exp_id})
    
    def get_experiment_data(
        self,
        tenant: str,
        exp_id: str,
        limit: int = 10_000
    ) -> List[Dict[str, Any]]:
        """
        Busca os dados de análise de um experimento.
        
        Args:
            tenant: Identificador do tenant
            exp_id: ID do experimento
            limit: Número máximo de documentos
        
        Returns:
            Lista de documentos ordenados por timestamp
        """
        db = self.get_db(tenant)
        return list(
            db["data_analise"]
            .find({"experiment_id": exp_id})
            .sort("timestamp", 1)
            .limit(limit)
        )

    def get_lab_results(
        self,
        tenant: str,
        exp_id: str,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Busca resultados de laboratório (lab_results) por experimentId.

        Observação: a coleção armazena `experimentId` (camelCase) conforme o CRM.
        """
        db = self.get_db(tenant)
        return list(
            db["lab_results"]
            .find({"experimentId": exp_id})
            .sort("analysisDate", -1)
            .limit(limit)
        )
    
    def get_lab_results_batch(
        self,
        tenant: str,
        experiment_ids: List[str],
        limit_per_exp: int = 5,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Busca lab_results para múltiplos experimentos em uma única query.
        
        Otimização: usa $in para buscar todos de uma vez, depois agrupa por experimentId.
        Muito mais eficiente que N queries individuais.
        
        Args:
            tenant: Identificador do tenant
            experiment_ids: Lista de IDs de experimentos
            limit_per_exp: Limite de resultados por experimento
        
        Returns:
            Dict mapeando experiment_id -> lista de lab_results
        """
        if not experiment_ids:
            return {}
        
        db = self.get_db(tenant)
        
        # Buscar todos os lab_results dos experimentos em uma query
        # Ordenar por analysisDate descendente para pegar os mais recentes
        cursor = db["lab_results"].find(
            {"experimentId": {"$in": experiment_ids}},
            # Projeção: apenas campos necessários (Fase 5 - otimização)
            {
                "experimentId": 1,
                "analysisDate": 1,
                "coliformesTotaisNmp": 1,
                "coliformesTotaisUfc": 1,
                "ecoliNmp": 1,
                "ecoliUfc": 1,
                "label": 1,
                "bacteria": 1,
                "type": 1,
                "count": 1,
                "value": 1,
                "unit": 1,
                "tagUnidade": 1,
                "method": 1,
                "presence": 1,
            }
        ).sort("analysisDate", -1)
        
        # Agrupar por experimentId, limitando por experimento
        result: Dict[str, List[Dict[str, Any]]] = {exp_id: [] for exp_id in experiment_ids}
        for doc in cursor:
            exp_id = doc.get("experimentId")
            if exp_id and exp_id in result:
                if len(result[exp_id]) < limit_per_exp:
                    result[exp_id].append(doc)
        
        return result
    
    def get_experiment_ids_with_lab_results(
        self,
        tenant: str,
        experiment_ids: List[str],
    ) -> set[str]:
        """
        Retorna o conjunto de experiment_ids que possuem lab_results.
        
        Usa uma única consulta agregada com $in para eficiência.
        """
        if not experiment_ids:
            return set()
        
        db = self.get_db(tenant)
        pipeline = [
            {"$match": {"experimentId": {"$in": experiment_ids}}},
            {"$group": {"_id": "$experimentId"}},
        ]
        results = db["lab_results"].aggregate(pipeline)
        return {doc["_id"] for doc in results}
    
    def close(self):
        """Fecha a conexão com o MongoDB."""
        self._client.close()

    def list_experiments_by_protocol(
        self,
        tenant: str,
        protocol_id: str,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """
        Lista experimentos que pertencem a um protocolId específico.
        
        Busca diretamente na coleção experiments pelo campo protocolId.
        
        Args:
            tenant: Identificador do tenant
            protocol_id: ID do protocolo
            limit: Número máximo de experimentos
        
        Returns:
            Lista de experimentos com informações resumidas
        """
        db = self.get_db(tenant)
        
        # Buscar experimentos pelo protocolId
        exp_docs = list(
            db["experiments"]
            .find({"protocolId": protocol_id})
            .sort("startDate", -1)
            .limit(limit)
        )
        
        experiments = []
        for exp_doc in exp_docs:
            # O _id do experimento pode estar em diferentes campos
            exp_id = str(exp_doc.get("_id"))
            
            # Contar pontos de dados na coleção data_analise
            data_count = db["data_analise"].count_documents({"experiment_id": exp_id})
            
            # Se não tem dados, pular
            if data_count == 0:
                continue
            
            # Buscar lab_results
            lab_results = self.get_lab_results(tenant, exp_id, limit=5)
            has_lab_results = len(lab_results) > 0
            
            # Extrair labels dos lab_results
            labels = []
            for lr in lab_results:
                label = lr.get("label") or lr.get("bacteria") or lr.get("type")
                if label and label not in labels:
                    labels.append(label)
            
            # Extrair data de início
            start_date = exp_doc.get("startDate")
            created_at = None
            if start_date:
                try:
                    ts = int(start_date)
                    from datetime import datetime
                    created_at = datetime.fromtimestamp(ts).isoformat()
                except (ValueError, TypeError):
                    created_at = start_date
            
            experiments.append({
                "experiment_id": exp_id,
                "name": exp_doc.get("nome") or exp_doc.get("name"),
                "description": exp_doc.get("description"),
                "created_at": created_at,
                "status": exp_doc.get("status"),
                "diluicao": exp_doc.get("diluicao"),
                "serial_number": exp_doc.get("serialNumber"),
                "data_points": data_count,
                "has_lab_results": has_lab_results,
                "labels": labels,
            })
        
        return experiments

    def list_protocol_ids(
        self,
        tenant: str,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Lista os protocolIds disponíveis para um tenant.
        
        Args:
            tenant: Identificador do tenant
            limit: Número máximo de IDs
        
        Returns:
            Lista de protocolIds com informações
        """
        db = self.get_db(tenant)
        
        # Buscar protocolIds únicos dos experimentos
        pipeline = [
            {"$group": {
                "_id": "$protocolId",
                "count": {"$sum": 1},
            }},
            {"$sort": {"count": -1}},
            {"$limit": limit},
        ]
        
        results = list(db["experiments"].aggregate(pipeline))
        
        protocols = []
        for r in results:
            protocol_id = r["_id"]
            if not protocol_id:
                continue
            protocols.append({
                "protocol_id": protocol_id,
                "experiment_count": r["count"],
            })
        
        return protocols

    def list_experiments_by_analysis(
        self,
        tenant: str,
        analysis_id: str,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """
        Lista experimentos que possuem dados para um analysisId específico.
        
        Busca na coleção data_analise os experiment_ids únicos que têm
        dados para o analysisId informado.
        
        Args:
            tenant: Identificador do tenant
            analysis_id: ID da análise (ex: "coliformes", "ecoli")
            limit: Número máximo de experimentos
        
        Returns:
            Lista de experimentos com informações resumidas
        """
        db = self.get_db(tenant)
        
        # Busca experiment_ids únicos na coleção data_analise
        # Campo é analysis_id (snake_case) conforme estrutura do banco
        pipeline = [
            {"$match": {"analysis_id": analysis_id}},
            {"$group": {
                "_id": "$experiment_id",
                "first_timestamp": {"$min": "$timestamp"},
                "last_timestamp": {"$max": "$timestamp"},
                "count": {"$sum": 1},
            }},
            {"$sort": {"last_timestamp": -1}},
            {"$limit": limit},
        ]
        
        results = list(db["data_analise"].aggregate(pipeline))
        
        # Enriquece com dados do experimento e lab_results
        experiments = []
        for r in results:
            exp_id = r["_id"]
            if not exp_id:
                continue
            
            # Buscar dados do experimento
            exp_doc = self.get_experiment(tenant, exp_id)
            
            # Buscar lab_results
            lab_results = self.get_lab_results(tenant, exp_id, limit=5)
            has_lab_results = len(lab_results) > 0
            
            # Extrair labels dos lab_results
            labels = []
            for lr in lab_results:
                label = lr.get("label") or lr.get("bacteria") or lr.get("type")
                if label and label not in labels:
                    labels.append(label)
            
            experiments.append({
                "experiment_id": exp_id,
                "name": exp_doc.get("name") if exp_doc else None,
                "description": exp_doc.get("description") if exp_doc else None,
                "created_at": exp_doc.get("createdAt") or exp_doc.get("created_at") if exp_doc else None,
                "first_data": r.get("first_timestamp"),
                "last_data": r.get("last_timestamp"),
                "data_points": r.get("count", 0),
                "has_lab_results": has_lab_results,
                "labels": labels,
            })
        
        return experiments

    def list_analysis_ids(
        self,
        tenant: str,
        limit: int = 100,
    ) -> List[str]:
        """
        Lista os analysisIds disponíveis para um tenant.
        
        Args:
            tenant: Identificador do tenant
            limit: Número máximo de IDs
        
        Returns:
            Lista de analysisIds únicos
        """
        db = self.get_db(tenant)
        
        # Busca analysis_ids únicos na coleção data_analise
        # Campo é analysis_id (snake_case) conforme estrutura do banco
        result = db["data_analise"].distinct("analysis_id")
        return sorted([str(r) for r in result if r])[:limit]
