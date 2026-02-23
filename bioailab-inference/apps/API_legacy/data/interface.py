#!/usr/bin/env python3
"""
Experiment Viewer GUI

Requisitos:
    pip install matplotlib

Esse script abre vários JSONs (experiments + tabelas auxiliares)
e permite filtrar por usuário, protocolo e análise, navegar entre
experimentos e visualizar séries temporais dos sensores, além de
exportar o resultado filtrado em três arquivos:
  - experiments_filtered.json  (sem ‘features’, e com o objeto completo de protocolo)
  - protocols_filtered.json    (só os protocolos usados)
  - calibrations_filtered.json (só as calibrações usadas pelos experimentos)
"""

import os
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class ExperimentViewer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Experiment Viewer")
        self.geometry("1200x700")

        # Estado
        self.tables               = {}   # armazena cada tabela auxiliar carregada
        self.experiments          = []  # lista completa de experimentos
        self.filtered_experiments = []  # depois dos filtros
        self.current              = None # experimento selecionado
        self.canvas               = None # plot atual

        self.user_filtered_experiments     = []
        self.protocol_filtered_experiments = []
        self.filtered_experiments          = []

        # --- Widgets ---
        self.btn_load = ttk.Button(self,
            text="Load Data (experiments + tables…)",
            command=self.load_data
        )
        # filtros em cascata: user → protocol → analysis
        self.ctrl_filters = ttk.Frame(self)
        self.user_var     = tk.StringVar()
        self.protocol_var = tk.StringVar()
        self.analysis_var = tk.StringVar()
        self.cb_user     = ttk.Combobox(self.ctrl_filters,
                                        textvariable=self.user_var,
                                        state="readonly", width=30)
        self.cb_protocol = ttk.Combobox(self.ctrl_filters,
                                        textvariable=self.protocol_var,
                                        state="readonly", width=30)
        self.cb_analysis = ttk.Combobox(self.ctrl_filters,
                                        textvariable=self.analysis_var,
                                        state="readonly", width=30)
        self.cb_user.bind("<<ComboboxSelected>>",     self.on_user_filter)
        self.cb_protocol.bind("<<ComboboxSelected>>", self.on_protocol_filter)
        self.cb_analysis.bind("<<ComboboxSelected>>", self.on_analysis_filter)

        # lista de experimentos
        self.lb_expts   = tk.Listbox(self, exportselection=False)
        self.sb_expts_v = ttk.Scrollbar(self, orient=tk.VERTICAL,
                                        command=self.lb_expts.yview)
        self.lb_expts.configure(yscrollcommand=self.sb_expts_v.set)
        self.lb_expts.bind("<<ListboxSelect>>", self.on_select_expt)

        # notebook: abas de metadata e plots
        self.nb = ttk.Notebook(self)
        # └ metadata
        self.frame_meta = ttk.Frame(self.nb)
        self.txt_meta   = tk.Text(self.frame_meta, wrap=tk.NONE)
        self.sb_meta_v  = ttk.Scrollbar(self.frame_meta,
                                        orient=tk.VERTICAL,
                                        command=self.txt_meta.yview)
        self.sb_meta_h  = ttk.Scrollbar(self.frame_meta,
                                        orient=tk.HORIZONTAL,
                                        command=self.txt_meta.xview)
        self.txt_meta.configure(yscrollcommand=self.sb_meta_v.set,
                                xscrollcommand=self.sb_meta_h.set)
        self.txt_meta.config(state=tk.DISABLED)
        # └ plots
        self.frame_plot    = ttk.Frame(self.nb)
        self.ctrl_plot     = ttk.Frame(self.frame_plot)
        self.category_var  = tk.StringVar()
        self.channel_var   = tk.StringVar()
        self.cb_category   = ttk.Combobox(self.ctrl_plot,
                                          textvariable=self.category_var,
                                          state="readonly")
        self.cb_channel    = ttk.Combobox(self.ctrl_plot,
                                          textvariable=self.channel_var,
                                          state="readonly")
        self.cb_category.bind("<<ComboboxSelected>>", self.on_category)
        self.cb_channel .bind("<<ComboboxSelected>>", self.on_channel)

        # botão de exportar
        self.btn_export = ttk.Button(self,
            text="Export Filtered + Protocols/Calibrations…",
            command=self.export_filtered
        )

        self.lbl_status = ttk.Label(self, text="", foreground="green")
        self._layout()

    def _layout(self):
        # linha 0: Load
        self.btn_load.grid(row=0, column=0, columnspan=2,
                           sticky="ew", padx=5, pady=5)
        # linha 1: filtros
        self.ctrl_filters.grid(row=1, column=0, columnspan=2,
                               sticky="ew", padx=5)
        ttk.Label(self.ctrl_filters, text="User:").grid(row=0, column=0, padx=5)
        self.cb_user   .grid(row=0, column=1, padx=5)
        ttk.Label(self.ctrl_filters, text="Protocol:").grid(row=0, column=2, padx=5)
        self.cb_protocol.grid(row=0, column=3, padx=5)
        ttk.Label(self.ctrl_filters, text="Analysis:").grid(row=0, column=4, padx=5)
        self.cb_analysis.grid(row=0, column=5, padx=5)

        # linha 2: lista de experimentos + notebook
        self.lb_expts  .grid(row=2, column=0, sticky="nsew",
                             padx=(5,0), pady=5)
        self.sb_expts_v.grid(row=2, column=0, sticky="nse",
                             padx=(0,5), pady=5)
        self.nb       .grid(row=2, column=1, sticky="nsew",
                            padx=5, pady=5)

        # abas
        self.nb.add(self.frame_meta, text="Metadata")
        self.txt_meta .grid(row=0, column=0, sticky="nsew")
        self.sb_meta_v.grid(row=0, column=1, sticky="nsew")
        self.sb_meta_h.grid(row=1, column=0, sticky="ew")
        self.frame_meta.grid_rowconfigure(0, weight=1)
        self.frame_meta.grid_columnconfigure(0, weight=1)

        self.nb.add(self.frame_plot, text="Plots")
        self.ctrl_plot.grid(row=0, column=0, sticky="ew", pady=5)
        ttk.Label(self.ctrl_plot, text="Category:").grid(row=0, column=0, padx=5)
        self.cb_category.grid(row=0, column=1, padx=5)
        ttk.Label(self.ctrl_plot, text="Channel:").grid(row=0, column=2, padx=5)
        self.cb_channel.grid(row=0, column=3, padx=5)
        self.frame_plot.grid_rowconfigure(1, weight=1)
        self.frame_plot.grid_columnconfigure(0, weight=1)

        # linha 3: export
        self.btn_export.grid(row=3, column=0, columnspan=2,
                             sticky="ew", padx=5, pady=5)
        
        # linha 4: status
        self.lbl_status.grid(row=4, column=0, columnspan=2,
                             sticky="ew", padx=5, pady=(0,5))
        
        # redimensionamento
        self.grid_rowconfigure(2, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=3)

    def load_data(self):
        paths = filedialog.askopenfilenames(
            title="Select JSON files",
            filetypes=[("JSON files","*.json"),("All files","*.*")]
        )
        if not paths:
            return
        # limpa estado
        self.tables.clear()
        self.experiments.clear()
        self.filtered_experiments.clear()

        for p in paths:
            try:
                raw = json.load(open(p, "r", encoding="utf-8"))
            except Exception as e:
                messagebox.showwarning("Load", f"Falha ao ler {p}:\n{e}")
                continue
            # desembrulha wrapper Lambda HTTP
            if isinstance(raw, dict) and "body" in raw:
                b = raw["body"]
                raw = json.loads(b) if isinstance(b, str) else b

            name = os.path.splitext(os.path.basename(p))[0].lower()
            if isinstance(raw, list):
                if "experiment" in name:
                    self.experiments = raw
                    self.filtered_experiments = raw.copy()
                else:
                    self.tables[name] = raw
            else:
                messagebox.showwarning("Load",
                    f"{p} não é um array JSON e foi ignorado.")

        if not self.experiments:
            return messagebox.showerror("Load",
                "Nenhum experimento carregado.")

        # popula filtro de usuários
        self._populate_user_filter()
        messagebox.showinfo("Load",
            f"{len(self.experiments)} experimentos e "
            f"{len(self.tables)} tabelas carregadas.")

    def _populate_user_filter(self):
        uuids = sorted({ex["general_info"].get("user_UUID")
                        for ex in self.experiments})
        self.cb_user["values"] = uuids
        self.user_var.set("")
        self.cb_protocol["values"] = []
        self.protocol_var.set("")
        self.cb_analysis["values"] = []
        self.analysis_var.set("")
        self._update_experiment_list(self.experiments)

    def on_user_filter(self, _):
        u = self.user_var.get()

        # limpa filtros abaixo
        self.protocol_var.set("")
        self.cb_protocol["values"] = []
        self.analysis_var.set("")
        self.cb_analysis["values"] = []

        # base: todos experimentos do usuário
        self.user_filtered_experiments = [
            ex for ex in self.experiments
            if ex["general_info"].get("user_UUID") == u
        ]

        self._populate_protocol_filter()
        # lista mostra somente base usuário
        self._update_experiment_list(self.user_filtered_experiments)

    def _populate_protocol_filter(self):
        protables = self.tables.get("protocol") or []
        self.table_protocol = {p["UUID"]: p for p in protables}

        uuids = sorted({
            ex["general_info"].get("protocol_UUID")
            for ex in self.user_filtered_experiments
        })
        names = []
        self.protocol_map = {}
        for pu in uuids:
            p  = self.table_protocol.get(pu, {})
            nm = p.get("name", pu)
            names.append(nm)
            self.protocol_map[nm] = pu

        self.cb_protocol["values"] = names


    def on_protocol_filter(self, _):
        nm = self.protocol_var.get()
        pu = self.protocol_map.get(nm)

        # limpa filtro abaixo
        self.analysis_var.set("")
        self.cb_analysis["values"] = []

        # base: só protocolos dentro do usuário
        self.protocol_filtered_experiments = [
            ex for ex in self.user_filtered_experiments
            if ex["general_info"].get("protocol_UUID") == pu
        ]

        self._populate_analysis_filter()
        self._update_experiment_list(self.protocol_filtered_experiments)

    def _populate_analysis_filter(self):
        pu    = self.protocol_map.get(self.protocol_var.get())
        proto = self.table_protocol.get(pu, {}) or {}
        # mantém só as análises ativas
        actives = [a for a in proto.get("analysis", []) if a.get("active")]

        # primeiro “Todos”, depois cada nome
        names = ["Todos"] + [a["name"] for a in actives]
        self.analysis_map = {a["name"]: a["uuid"] for a in actives}

        self.cb_analysis["values"] = names
        # reseta seleção
        self.analysis_var.set("Todos")

    def on_analysis_filter(self, _):
        sel = self.analysis_var.get()
        if sel == "Todos" or not sel:
            # mostra todos os já filtrados por protocolo
            self.filtered_experiments = self.protocol_filtered_experiments.copy()
        else:
            # filtra somente pela análise escolhida
            au = self.analysis_map.get(sel)
            self.filtered_experiments = [
                ex for ex in self.protocol_filtered_experiments
                if ex["general_info"].get("analysis_UUID") == au
            ]
        self._update_experiment_list(self.filtered_experiments)


    def _update_experiment_list(self, arr):
        self.lb_expts.delete(0, tk.END)
        for i, ex in enumerate(arr, start=1):
            self.lb_expts.insert(tk.END, f"{i}: {ex.get('experiment_UUID','n/a')}")

    def on_select_expt(self, _):
        sel = self.lb_expts.curselection()
        if not sel:
            return
        self.current = self.filtered_experiments[sel[0]]
        self._show_meta()
        self._reset_plot_controls()

    def _show_meta(self):
        txt = json.dumps(self.current, indent=2, ensure_ascii=False)
        self.txt_meta.config(state=tk.NORMAL)
        self.txt_meta.delete("1.0", tk.END)
        self.txt_meta.insert(tk.END, txt)
        self.txt_meta.config(state=tk.DISABLED)

    def _reset_plot_controls(self):
        meta_keys = {
            "experiment_UUID","serial_number",
            "calibration","features",
            "general_info","light_sensor_config"
        }
        cats = [k for k in self.current.keys() if k not in meta_keys]
        self.cb_category["values"] = cats
        self.category_var.set("")
        self.cb_channel["values"] = []
        self.channel_var.set("")
        self._clear_canvas()

    def on_category(self, _):
        cat = self.category_var.get()
        chs = list(self.current.get(cat, {}).keys())
        self.cb_channel["values"] = chs
        self.channel_var.set("")
        self._clear_canvas()

    def on_channel(self, _):
        cat = self.category_var.get()
        ch  = self.channel_var.get()
        x_epochs = self.current.get("timestamps", [])
        y        = self.current.get(cat, {}).get(ch, [])
        if not x_epochs:
            return

        # pega o primeiro tempo e calcula diferença em segundos
        t0 = x_epochs[0]
        deltas_sec = [t - t0 for t in x_epochs]

        # converte para unidades inteiras (minutos, se os dados forem a cada 60s)
        # ou mantenha em segundos, apenas troque o divisor
        x_rel = [int(delta / 60) for delta in deltas_sec]

        self._plot_series(x_rel, y, f"{cat}.{ch} (min desde t0)")


    def _plot_series(self, x, y, title):
        self._clear_canvas()
        fig = Figure(figsize=(6,4), dpi=100)
        ax  = fig.add_subplot(111)
        ax.plot(x, y, marker=".", linestyle="-")
        ax.set_title(title)
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("Value")
        fig.tight_layout()

        self.canvas = FigureCanvasTkAgg(fig, master=self.frame_plot)
        self.canvas.draw()
        widget = self.canvas.get_tk_widget()
        widget.grid(row=1, column=0, sticky="nsew")

    def _clear_canvas(self):
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
            self.canvas = None


    def export_filtered(self):
        if not self.filtered_experiments:
            return messagebox.showwarning("Export", "Nenhum experimento selecionado.")
        folder = filedialog.askdirectory(title="Selecione a pasta de exportação")
        if not folder:
            return

        # indica início
        self.btn_export.config(text="Saving...", state="disabled")
        self.lbl_status.config(text="Exporting files, please wait…")
        self.config(cursor="watch")
        self.update_idletasks()

        try:
            # 1) experiments_filtered.json (timestamps em minutos relativos)
            wrappers = []
            for ex in self.filtered_experiments:
                ne = {k: v for k, v in ex.items() if k != "features"}
                ts = ex.get("timestamps", [])
                if ts:
                    t0 = ts[0]
                    # converte cada epoch em minutos inteiros desde t0
                    ne["timestamps"] = [int((t - t0) / 60) for t in ts]
                wrappers.append({
                    "statusCode": 200,
                    "body": json.dumps(ne, ensure_ascii=False),
                    "headers": {
                        "Access-Control-Allow-Origin": "*",
                        "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
                        "Access-Control-Allow-Headers": "Content-Type"
                    }
                })
            path_exp = os.path.join(folder, "experiments_filtered.json")
            with open(path_exp, "w", encoding="utf-8") as f:
                json.dump(wrappers, f, ensure_ascii=False, indent=2)

            # 2) protocols_filtered.json
            prot_used = sorted({ex["general_info"]["protocol_UUID"] for ex in self.filtered_experiments})
            prot_out = []
            for pu in prot_used:
                p = self.table_protocol.get(pu, {})
                prot_out.append({k: v for k, v in p.items() if k != "configurations"})
            wrapped_prot = [{
                "statusCode": 200,
                "body": json.dumps(p, ensure_ascii=False),
                "headers": {
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
                    "Access-Control-Allow-Headers": "Content-Type"
                }
            } for p in prot_out]
            path_prot = os.path.join(folder, "protocols_filtered.json")
            with open(path_prot, "w", encoding="utf-8") as f:
                json.dump(wrapped_prot, f, ensure_ascii=False, indent=2)

            # 3) culture_media_filtered.json
            media_table = self.tables.get("culture_media", [])
            media_uuids = {p.get("cultureMediaUUID") for p in prot_out if p.get("cultureMediaUUID")}
            culture_out = [m for m in media_table if m["UUID"] in media_uuids]
            wrapped_culture = [{
                "statusCode": 200,
                "body": json.dumps(m, ensure_ascii=False),
                "headers": {
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
                    "Access-Control-Allow-Headers": "Content-Type"
                }
            } for m in culture_out]
            path_cult = os.path.join(folder, "culture_media_filtered.json")
            with open(path_cult, "w", encoding="utf-8") as f:
                json.dump(wrapped_culture, f, ensure_ascii=False, indent=2)

            # 4) bacteria_filtered.json
            bacteria_table = self.tables.get("bacteria", [])
            bact_uuids = {
                cid
                for ex in self.filtered_experiments
                for cid in ex.get("calibration", {}).keys()
            }
            bacteria_out = [b for b in bacteria_table if b.get("UUID") in bact_uuids]
            wrapped_bacteria = [{
                "statusCode": 200,
                "body": json.dumps(b, ensure_ascii=False),
                "headers": {
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
                    "Access-Control-Allow-Headers": "Content-Type"
                }
            } for b in bacteria_out]
            path_bact = os.path.join(folder, "bacteria_filtered.json")
            with open(path_bact, "w", encoding="utf-8") as f:
                json.dump(wrapped_bacteria, f, ensure_ascii=False, indent=2)

            # 5) devices_filtered.json (filtra por serial_number dos experiments)
            # ———————————————————————————————————————————————————————

            # 1) coleta todos os serial_number usados nos experiments filtrados
            serials_used = {
                ex.get("serial_number")
                for ex in self.filtered_experiments
                if ex.get("serial_number")
            }

            # 2) busca a lista de devices com fallback na key "device" ou "devices"
            devices_table = (
                self.tables.get("devices")
                or self.tables.get("device")
                or []
            )

            # 3) filtra apenas os devices cujo serial_number está em serials_used
            filtered_devices = [
                dev for dev in devices_table
                if dev.get("serial_number") in serials_used
            ]

            # (Opcional) DEBUG: mostrar no console quantos serials e devices foram encontrados
            print(f"Serials usados: {serials_used}")
            print(f"Total devices carregados: {len(devices_table)}")
            print(f"Devices filtrados: {len(filtered_devices)}")

            # 4) embala cada device no envelope HTTP
            wrapped_devices = [{
                "statusCode": 200,
                "body": json.dumps(dev, ensure_ascii=False),
                "headers": {
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
                    "Access-Control-Allow-Headers": "Content-Type"
                }
            } for dev in filtered_devices]

            # 5) grava o arquivo
            path_dev = os.path.join(folder, "devices_filtered.json")
            with open(path_dev, "w", encoding="utf-8") as f:
                json.dump(wrapped_devices, f, ensure_ascii=False, indent=2)


        finally:
            # restaura estado da UI
            self.btn_export.config(text="Export Filtered + Protocols/Calibrations…", state="normal")
            self.lbl_status.config(text="Export concluído com sucesso.")
            self.config(cursor="")
            self.update_idletasks()

        # info final
        messagebox.showinfo(
            "Export concluído",
            f"• {len(wrappers)} envelopes → {os.path.basename(path_exp)}\n"
            f"• {len(wrapped_prot)} protocolos → {os.path.basename(path_prot)}\n"
            f"• {len(wrapped_culture)} meios de cultura → {os.path.basename(path_cult)}\n"
            f"• {len(wrapped_bacteria)} bactérias → {os.path.basename(path_bact)}\n"
            f"• {len(wrapped_devices)} devices → {os.path.basename(path_dev)}"
        )




if __name__ == "__main__":
    # pip install matplotlib
    app = ExperimentViewer()
    app.mainloop()
