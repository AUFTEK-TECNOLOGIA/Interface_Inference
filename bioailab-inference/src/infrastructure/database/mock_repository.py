"""
Repositorio de mock para dados de experimentos.

Lê arquivos JSON em resources/<tenant>/datasets/mock/experiment.json
e normaliza para o formato esperado pelo pipeline (similar ao Mongo).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class MockRepository:
    """Repositorio para dados mockados."""

    def __init__(self, resources_dir: Path):
        self._resources_dir = Path(resources_dir)
        self._cache: dict[str, list[dict[str, Any]]] = {}
        self._bacteria_map: dict[str, str] | None = None

    def _bacteria_file(self) -> Path:
        repo_root = self._resources_dir.parent
        return repo_root / "apps" / "API_legacy" / "data" / "bacteria_filtered.json"

    def _load_bacteria_map(self) -> dict[str, str]:
        if self._bacteria_map is not None:
            return self._bacteria_map
        path = self._bacteria_file()
        mapping: dict[str, str] = {}
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    for item in data:
                        if not isinstance(item, dict):
                            continue
                        body = item.get("body")
                        if not body:
                            continue
                        try:
                            payload = json.loads(body)
                        except Exception:
                            continue
                        if not isinstance(payload, dict):
                            continue
                        uid = str(payload.get("UUID") or "").strip()
                        name = str(payload.get("name") or "").strip()
                        if uid and name:
                            mapping[uid] = name
            except Exception:
                mapping = {}
        self._bacteria_map = mapping
        return mapping

    @staticmethod
    def _normalize_label(name: str) -> str:
        n = str(name or "").strip().lower()
        if not n:
            return ""
        if "escherichia coli" in n or n in ("e. coli", "e coli", "ecoli"):
            return "e_coli"
        if "coliformes totais" in n or n == "coliformes":
            return "coliformes_totais"
        # slug simples
        return (
            n.replace(".", "")
            .replace("/", "_")
            .replace("\\", "_")
            .replace(" ", "_")
            .replace("-", "_")
        )

    def _label_from_calibration_id(self, calib_id: str) -> tuple[str, str]:
        mapping = self._load_bacteria_map()
        name = mapping.get(str(calib_id) or "", "")
        label = self._normalize_label(name) if name else ""
        return label, name

    def _mock_file(self, tenant: str) -> Path:
        return self._resources_dir / tenant / "datasets" / "mock" / "experiment.json"

    def _load(self, tenant: str) -> list[dict[str, Any]]:
        if tenant in self._cache:
            return self._cache[tenant]
        path = self._mock_file(tenant)
        if not path.exists():
            self._cache[tenant] = []
            return []
        data = json.loads(path.read_text(encoding="utf-8"))
        items = data if isinstance(data, list) else []
        
        # Parse items - handle both raw format and {statusCode, body, headers} format
        parsed: list[dict[str, Any]] = []
        for d in items:
            if not isinstance(d, dict):
                continue
            # If item has 'body' key, it's wrapped format - parse the body
            if "body" in d and isinstance(d.get("body"), str):
                try:
                    body = json.loads(d["body"])
                    if isinstance(body, dict):
                        parsed.append(body)
                except (json.JSONDecodeError, TypeError):
                    pass
            # Otherwise use as-is (raw format)
            elif "experiment_UUID" in d or "general_info" in d:
                parsed.append(d)
        
        self._cache[tenant] = parsed
        return self._cache[tenant]

    @staticmethod
    def _strip_prefix(exp_id: str) -> str:
        exp_id = str(exp_id or "")
        return exp_id[len("mock:") :] if exp_id.startswith("mock:") else exp_id

    @staticmethod
    def _channel_key(name: str) -> str:
        key = str(name or "").strip().lower()
        if key == "clr":
            return "clear"
        if key == "nir":
            return "nir"
        if key.startswith("f"):
            return key
        return key

    @staticmethod
    def _map_unit(unit: str) -> str:
        u = str(unit or "").strip()
        if u.upper() in ("NPM/ML", "NMP/ML"):
            return "NMP/100mL"
        return u

    def _find_raw(self, tenant: str, exp_id: str) -> Optional[dict[str, Any]]:
        raw_id = self._strip_prefix(exp_id)
        for item in self._load(tenant):
            if str(item.get("experiment_UUID") or "") == raw_id:
                return item
        return None

    def _normalize_experiment(self, raw: dict[str, Any], exp_id: str) -> dict[str, Any]:
        info = raw.get("general_info") if isinstance(raw.get("general_info"), dict) else {}
        comments = info.get("comments") if isinstance(info.get("comments"), list) else []
        name = comments[0] if comments else exp_id
        analysis_id = info.get("analysis_UUID") or info.get("protocol_UUID")
        return {
            "_id": exp_id,
            "id": exp_id,
            "name": name,
            "description": "Experimento mockado",
            "serialNumber": raw.get("serial_number"),
            "protocolId": info.get("protocol_UUID"),
            "analysisId": analysis_id,
            "startDate": info.get("start_date"),
            "endDate": info.get("end_date"),
            "status": info.get("status"),
        }

    def _build_lab_results(self, raw: dict[str, Any], exp_id: str) -> list[dict[str, Any]]:
        info = raw.get("general_info") if isinstance(raw.get("general_info"), dict) else {}
        calibration = raw.get("calibration") if isinstance(raw.get("calibration"), dict) else {}
        analysis_date = info.get("end_date") or info.get("start_date")

        results: list[dict[str, Any]] = []
        for calib_id, item in calibration.items():
            if not isinstance(item, dict):
                continue

            label, name = self._label_from_calibration_id(str(calib_id))
            def _add_result(count: Any, unit: Any, suffix: str | None = None) -> None:
                if count is None:
                    return
                mapped_unit = self._map_unit(unit)
                lab: dict[str, Any] = {
                    "id": f"mock_lab_result_{exp_id}_{calib_id}{'_' + suffix if suffix else ''}",
                    "experimentId": exp_id,
                    "analysisDate": analysis_date,
                    "count": count,
                    "unit": mapped_unit,
                    "tagUnidade": mapped_unit,
                }
                if label:
                    lab["label"] = label
                if name:
                    lab["bacteria"] = name

                unit_upper = str(mapped_unit or "").upper()
                if label == "e_coli":
                    if "UFC" in unit_upper:
                        lab["ecoliUfc"] = count
                    else:
                        lab["ecoliNmp"] = count
                elif label == "coliformes_totais":
                    if "UFC" in unit_upper:
                        lab["coliformesTotaisUfc"] = count
                    else:
                        lab["coliformesTotaisNmp"] = count
                else:
                    if "UFC" in unit_upper:
                        lab["coliformesTotaisUfc"] = count
                    else:
                        lab["coliformesTotaisNmp"] = count
                results.append(lab)

            # Suporta formato simples (count/unit) e formato expandido (nmp/ufc)
            if "nmp" in item or "ufc" in item:
                nmp = item.get("nmp") if isinstance(item.get("nmp"), dict) else None
                ufc = item.get("ufc") if isinstance(item.get("ufc"), dict) else None
                if nmp:
                    _add_result(nmp.get("count"), nmp.get("unit"), suffix="nmp")
                if ufc:
                    _add_result(ufc.get("count"), ufc.get("unit"), suffix="ufc")
                continue

            _add_result(item.get("count"), item.get("unit"))

        return results

    def _build_experiment_data(self, raw: dict[str, Any], exp_id: str) -> list[dict[str, Any]]:
        timestamps = raw.get("timestamps") if isinstance(raw.get("timestamps"), list) else []

        def sensor_from_raw(key: str) -> dict[str, list]:
            data = raw.get(key) if isinstance(raw.get(key), dict) else {}
            out: dict[str, list] = {}
            for ch, values in data.items():
                if isinstance(values, list):
                    out[self._channel_key(ch)] = values
            return out

        spectral = {
            "fluorescence": sensor_from_raw("spectral_uv"),
            "nephelometry": sensor_from_raw("spectral_vis_1"),
            "turbidimetry": sensor_from_raw("spectral_vis_2"),
        }

        temp = raw.get("temperature") if isinstance(raw.get("temperature"), dict) else {}
        temp_sample = temp.get("sample") if isinstance(temp.get("sample"), list) else []

        info = raw.get("general_info") if isinstance(raw.get("general_info"), dict) else {}
        analysis_id = info.get("analysis_UUID") or info.get("protocol_UUID")

        docs: list[dict[str, Any]] = []
        for i, ts in enumerate(timestamps):
            row: dict[str, Any] = {
                "experiment_id": exp_id,
                "analysis_id": analysis_id,
                "analysisId": analysis_id,
                "timestamp": ts,
                "spectral": {},
            }
            if i < len(temp_sample):
                row["temperatures"] = {"sample": temp_sample[i]}
            for sensor_name, channels in spectral.items():
                if not channels:
                    continue
                row["spectral"][sensor_name] = {
                    "sensorType": "AS7341",
                }
                if sensor_name == "fluorescence":
                    row["spectral"][sensor_name]["gain"] = 512
                    row["spectral"][sensor_name]["integrationTime"] = 700.0
                else:
                    row["spectral"][sensor_name]["gain"] = 1
                    row["spectral"][sensor_name]["integrationTime"] = 300.0
                for ch, values in channels.items():
                    if i < len(values):
                        row["spectral"][sensor_name][ch] = values[i]
                if sensor_name in ("turbidimetry", "nephelometry"):
                    reference = {
                        "gain": row["spectral"][sensor_name]["gain"],
                        "timeMs": row["spectral"][sensor_name]["integrationTime"],
                    }
                    for ch in channels.keys():
                        if ch in row["spectral"][sensor_name]:
                            reference[ch] = row["spectral"][sensor_name][ch]
                    row["spectral"][sensor_name]["reference"] = reference
            docs.append(row)
        return docs

    def get_experiment(self, tenant: str, exp_id: str) -> Optional[Dict[str, Any]]:
        raw = self._find_raw(tenant, exp_id)
        if not raw:
            return None
        raw_id = self._strip_prefix(exp_id)
        return self._normalize_experiment(raw, raw_id)

    def get_experiment_data(self, tenant: str, exp_id: str, limit: int = 10_000) -> List[Dict[str, Any]]:
        raw = self._find_raw(tenant, exp_id)
        if not raw:
            return []
        raw_id = self._strip_prefix(exp_id)
        data = self._build_experiment_data(raw, raw_id)
        return data[:limit]

    def get_lab_results(self, tenant: str, exp_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        raw = self._find_raw(tenant, exp_id)
        if not raw:
            return []
        raw_id = self._strip_prefix(exp_id)
        results = self._build_lab_results(raw, raw_id)
        return results[:limit]

    def get_lab_results_batch(
        self,
        tenant: str,
        experiment_ids: List[str],
        limit_per_exp: int = 5,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Busca lab_results para múltiplos experimentos.
        
        Para mock, ainda precisa iterar, mas mantém interface consistente.
        
        Returns:
            Dict mapeando experiment_id -> lista de lab_results
        """
        result: Dict[str, List[Dict[str, Any]]] = {}
        for exp_id in experiment_ids:
            lab = self.get_lab_results(tenant, exp_id, limit=limit_per_exp)
            result[exp_id] = lab
        return result

    def get_experiment_ids_with_lab_results(
        self,
        tenant: str,
        experiment_ids: List[str],
    ) -> set[str]:
        """
        Retorna o conjunto de experiment_ids que possuem lab_results.
        """
        if not experiment_ids:
            return set()
        
        result = set()
        for exp_id in experiment_ids:
            raw = self._find_raw(tenant, exp_id)
            if raw:
                raw_id = self._strip_prefix(exp_id)
                lab = self._build_lab_results(raw, raw_id)
                if lab:
                    result.add(exp_id)
        return result

    def list_protocol_ids(self, tenant: str, limit: int = 100) -> List[Dict[str, Any]]:
        counts: dict[str, int] = {}
        for item in self._load(tenant):
            info = item.get("general_info") if isinstance(item.get("general_info"), dict) else {}
            pid = info.get("protocol_UUID")
            if not pid:
                continue
            counts[pid] = counts.get(pid, 0) + 1
        protocols = [{"protocol_id": k, "experiment_count": v} for k, v in counts.items()]
        protocols.sort(key=lambda x: x["experiment_count"], reverse=True)
        return protocols[:limit]

    def list_analysis_ids(self, tenant: str, limit: int = 100) -> List[str]:
        ids = set()
        for item in self._load(tenant):
            info = item.get("general_info") if isinstance(item.get("general_info"), dict) else {}
            aid = info.get("analysis_UUID") or info.get("protocol_UUID")
            if aid:
                ids.add(str(aid))
        return sorted(list(ids))[:limit]

    def list_experiments_by_protocol(self, tenant: str, protocol_id: str, limit: int = 1000) -> List[Dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for item in self._load(tenant):
            info = item.get("general_info") if isinstance(item.get("general_info"), dict) else {}
            pid = info.get("protocol_UUID")
            if pid != protocol_id:
                continue
            raw_id = str(item.get("experiment_UUID") or "")
            exp_id = f"mock:{raw_id}"
            exp = self._normalize_experiment(item, raw_id)
            lab = self._build_lab_results(item, raw_id)
            labels = []
            for lr in lab:
                label = lr.get("label")
                if label and label not in labels:
                    labels.append(label)
            data_points = len(item.get("timestamps") or [])
            results.append(
                {
                    "experiment_id": exp_id,
                    "name": exp.get("name"),
                    "description": exp.get("description"),
                    "created_at": exp.get("startDate"),
                    "status": exp.get("status"),
                    "diluicao": exp.get("diluicao"),
                    "serial_number": exp.get("serialNumber"),
                    "data_points": data_points,
                    "has_lab_results": bool(lab),
                    "labels": labels,
                    "source": "mock",
                }
            )
        return results[:limit]

    def list_experiments_by_analysis(self, tenant: str, analysis_id: str, limit: int = 1000) -> List[Dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for item in self._load(tenant):
            info = item.get("general_info") if isinstance(item.get("general_info"), dict) else {}
            aid = info.get("analysis_UUID") or info.get("protocol_UUID")
            if aid != analysis_id:
                continue
            raw_id = str(item.get("experiment_UUID") or "")
            exp_id = f"mock:{raw_id}"
            exp = self._normalize_experiment(item, raw_id)
            lab = self._build_lab_results(item, raw_id)
            labels = []
            for lr in lab:
                label = lr.get("label")
                if label and label not in labels:
                    labels.append(label)
            data_points = len(item.get("timestamps") or [])
            results.append(
                {
                    "experiment_id": exp_id,
                    "name": exp.get("name"),
                    "description": exp.get("description"),
                    "created_at": exp.get("startDate"),
                    "first_data": None,
                    "last_data": None,
                    "data_points": data_points,
                    "has_lab_results": bool(lab),
                    "labels": labels,
                    "source": "mock",
                }
            )
        return results[:limit]

    def close(self) -> None:
        """Compatibilidade com MongoRepository."""
        return None

    # ==========================================================================
    # Métodos de edição de lab results (calibration)
    # ==========================================================================

    def _save_experiments(self, tenant: str, experiments: list[dict[str, Any]]) -> None:
        """Salva os experimentos de volta ao arquivo JSON."""
        path = self._mock_file(tenant)
        
        # Read the original file to preserve the wrapped format
        original_data = json.loads(path.read_text(encoding="utf-8"))
        
        # Build a map from experiment_UUID to wrapped item
        uuid_to_wrapped: dict[str, dict[str, Any]] = {}
        for item in original_data:
            if isinstance(item, dict) and "body" in item and isinstance(item.get("body"), str):
                try:
                    body = json.loads(item["body"])
                    if isinstance(body, dict):
                        exp_uuid = body.get("experiment_UUID")
                        if exp_uuid:
                            uuid_to_wrapped[exp_uuid] = item
                except (json.JSONDecodeError, TypeError):
                    pass
        
        # Update the wrapped items with modified experiments
        for exp in experiments:
            exp_uuid = exp.get("experiment_UUID")
            if exp_uuid and exp_uuid in uuid_to_wrapped:
                wrapped = uuid_to_wrapped[exp_uuid]
                wrapped["body"] = json.dumps(exp)
        
        # Write back
        path.write_text(json.dumps(original_data, indent=2, ensure_ascii=False), encoding="utf-8")
        
        # Clear cache
        if tenant in self._cache:
            del self._cache[tenant]

    def get_raw_experiment(self, tenant: str, exp_id: str) -> Optional[dict[str, Any]]:
        """Retorna o experimento raw (sem normalização)."""
        return self._find_raw(tenant, exp_id)

    def get_calibration(self, tenant: str, exp_id: str) -> dict[str, Any]:
        """Retorna apenas os dados de calibration (lab results) do experimento."""
        raw = self._find_raw(tenant, exp_id)
        if not raw:
            return {}
        return raw.get("calibration") or {}

    def update_calibration(
        self,
        tenant: str,
        exp_id: str,
        calibration: dict[str, Any],
    ) -> bool:
        """Atualiza os dados de calibration (lab results) do experimento."""
        raw_id = self._strip_prefix(exp_id)
        experiments = self._load(tenant)
        
        for exp in experiments:
            if str(exp.get("experiment_UUID") or "") == raw_id:
                exp["calibration"] = calibration
                self._save_experiments(tenant, experiments)
                return True
        return False

    def update_lab_result(
        self,
        tenant: str,
        exp_id: str,
        calibration_id: str,
        count: Optional[float] = None,
        unit: Optional[str] = None,
    ) -> bool:
        """Atualiza um lab result específico dentro do calibration."""
        raw_id = self._strip_prefix(exp_id)
        experiments = self._load(tenant)
        
        for exp in experiments:
            if str(exp.get("experiment_UUID") or "") == raw_id:
                calibration = exp.get("calibration")
                if not isinstance(calibration, dict):
                    calibration = {}
                    exp["calibration"] = calibration
                
                if calibration_id not in calibration:
                    calibration[calibration_id] = {}
                
                item = calibration[calibration_id]
                if count is not None:
                    item["count"] = count
                if unit is not None:
                    item["unit"] = unit
                
                self._save_experiments(tenant, experiments)
                return True
        return False

    def add_lab_result(
        self,
        tenant: str,
        exp_id: str,
        calibration_id: str,
        count: float,
        unit: str,
    ) -> bool:
        """Adiciona um novo lab result ao experimento."""
        raw_id = self._strip_prefix(exp_id)
        experiments = self._load(tenant)
        
        for exp in experiments:
            if str(exp.get("experiment_UUID") or "") == raw_id:
                calibration = exp.get("calibration")
                if not isinstance(calibration, dict):
                    calibration = {}
                    exp["calibration"] = calibration
                
                calibration[calibration_id] = {
                    "count": count,
                    "unit": unit,
                }
                
                self._save_experiments(tenant, experiments)
                return True
        return False

    def delete_lab_result(
        self,
        tenant: str,
        exp_id: str,
        calibration_id: str,
    ) -> bool:
        """Remove um lab result do experimento."""
        raw_id = self._strip_prefix(exp_id)
        experiments = self._load(tenant)
        
        for exp in experiments:
            if str(exp.get("experiment_UUID") or "") == raw_id:
                calibration = exp.get("calibration")
                if isinstance(calibration, dict) and calibration_id in calibration:
                    del calibration[calibration_id]
                    self._save_experiments(tenant, experiments)
                    return True
                return False
        return False

    def list_bacteria_options(self) -> list[dict[str, str]]:
        """Lista todas as opções de bactérias disponíveis para seleção."""
        mapping = self._load_bacteria_map()
        return [{"id": k, "name": v} for k, v in mapping.items()]
