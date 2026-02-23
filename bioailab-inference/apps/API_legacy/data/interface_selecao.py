#!/usr/bin/env python3
"""
Experiment Viewer GUI

Requisitos:
    pip install matplotlib

Esse script abre vários JSONs (experiments + tabelas auxiliares)
e permite filtrar por usuário, protocolo, análise e unidades,
escolher via checkboxes quais experimentos exportar, navegar entre
experimentos e visualizar, para cada um, todos os gráficos de canais
(F1…F8, CLR, NIR — cada um com as três medições UV/VIS1/VIS2) além do
gráfico de temperatura da amostra, com possibilidade de cortar um número
global de pontos do início e do fim. Filtra apenas experimentos que tenham
todas as unidades selecionadas. Exibe contadores de total, vistos, não vistos
e selecionados. Inclui dropdown para selecionar canal individual ou "All".
Também exporta cinco arquivos JSON filtrados.
"""

import os
import json
import copy
import tkinter as tk
import tkinter.font as tkFont

from tkinter import ttk, filedialog, messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

HTTP_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET,POST,PUT,DELETE,OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type"
}

class ExperimentViewer(tk.Tk):
    def __init__(self):
        super().__init__()

        self.font_bold   = tkFont.Font(weight="bold")
        self.font_normal = tkFont.Font(weight="normal")

        self.title("Experiment Viewer")
        self.geometry("1300x850")

        # ─── Estado ───────────────────────────────────────────────────────
        self.tables = {}                   # dados auxiliares carregados
        self.experiments = []              # lista de todos experimentos
        self.current = None                # experimento selecionado
        self.canvas = None                 # matplotlib canvas

        # filtros intermediários
        self.user_filtered = []
        self.protocol_filtered = []
        self.analysis_filtered = []
        self.unit_filtered = []
        self.expt_vars = []

        # Lista de checkboxes (var, experiment)
        self.expt_vars = []

        # Conjunto de vistos
        self.viewed_ids = set()

        # Variáveis globais de trim
        self.trim_start_var = tk.IntVar(value=0)
        self.trim_end_var   = tk.IntVar(value=0)
        self.trim_start_var.trace_add('write', lambda *a: self._plot_all_channels())
        self.trim_end_var.trace_add('write',   lambda *a: self._plot_all_channels())

        # ─── Widgets ───────────────────────────────────────────────────────
        self.btn_load = ttk.Button(self, text="Load JSON files…", command=self.load_data)

        # filtros em cascata
        self.ctrl_filters = ttk.Frame(self)
        self.user_var     = tk.StringVar()
        self.cb_user      = ttk.Combobox(self.ctrl_filters, textvariable=self.user_var,
                                         state="readonly", width=25)
        self.cb_user.bind("<<ComboboxSelected>>", self.on_user_filter)

        self.protocol_var = tk.StringVar()
        self.cb_protocol  = ttk.Combobox(self.ctrl_filters, textvariable=self.protocol_var,
                                         state="readonly", width=25)
        self.cb_protocol.bind("<<ComboboxSelected>>", self.on_protocol_filter)

        self.analysis_var = tk.StringVar()
        self.cb_analysis  = ttk.Combobox(self.ctrl_filters, textvariable=self.analysis_var,
                                         state="readonly", width=25)
        self.cb_analysis.bind("<<ComboboxSelected>>", self.on_analysis_filter)

        # Listbox para múltiplas unidades
        self.lb_unit = tk.Listbox(self.ctrl_filters, selectmode='multiple', height=2,
                                  exportselection=False, width=10)
        for u in ("UFC/mL", "NPM/mL"):
            self.lb_unit.insert(tk.END, u)
        self.lb_unit.bind("<<ListboxSelect>>", lambda e: self._apply_filters())

        # Canvas e scrollbar para checkboxes
        self.expt_canvas = tk.Canvas(self)
        self.expt_scroll = ttk.Scrollbar(self, orient="vertical",
                                         command=self.expt_canvas.yview)
        self.expt_frame  = ttk.Frame(self.expt_canvas)
        self.expt_canvas.configure(yscrollcommand=self.expt_scroll.set)
        self.expt_canvas.create_window((0,0), window=self.expt_frame, anchor="nw")
        self.expt_frame.bind("<Configure>",
            lambda e: self.expt_canvas.configure(scrollregion=self.expt_canvas.bbox("all"))
        )

        # Notebook para abas: Metadata e Plots
        self.nb = ttk.Notebook(self)
        # ─ Metadata
        self.frame_meta = ttk.Frame(self.nb)
        self.txt_meta   = tk.Text(self.frame_meta, wrap=tk.NONE)
        self.sb_meta_v  = ttk.Scrollbar(self.frame_meta, orient=tk.VERTICAL,
                                        command=self.txt_meta.yview)
        self.sb_meta_h  = ttk.Scrollbar(self.frame_meta, orient=tk.HORIZONTAL,
                                        command=self.txt_meta.xview)
        self.txt_meta.configure(yscrollcommand=self.sb_meta_v.set,
                                xscrollcommand=self.sb_meta_h.set)
        self.txt_meta.config(state=tk.DISABLED)
        # ─ Plots
        self.frame_plot = ttk.Frame(self.nb)
        self.ctrl_plot  = ttk.Frame(self.frame_plot)
        ttk.Label(self.ctrl_plot, text="Trim start:").grid(row=0, column=0, padx=5)
        tk.Spinbox(self.ctrl_plot, from_=0, to=10000,
                   textvariable=self.trim_start_var, width=5
        ).grid(row=0, column=1, padx=5)
        ttk.Label(self.ctrl_plot, text="Trim end:").grid(row=0, column=2, padx=5)
        tk.Spinbox(self.ctrl_plot, from_=0, to=10000,
                   textvariable=self.trim_end_var, width=5
        ).grid(row=0, column=3, padx=5)

        # dropdown de canal
        self.channel_var = tk.StringVar(value="All")
        channels = [f"F{i}" for i in range(1,9)] + ["CLR","NIR"]
        self.cb_channel = ttk.Combobox(self.ctrl_plot, textvariable=self.channel_var,
                                       state="readonly", values=["All"]+channels, width=10)
        self.cb_channel.grid(row=0, column=4, padx=10)
        self.cb_channel.bind("<<ComboboxSelected>>",
                             lambda e: self._plot_all_channels())

        # export e contadores
        self.btn_export = ttk.Button(self, text="Export Filtered…",
                                     command=self.export_filtered)
        self.lbl_counts = ttk.Label(self, text="Total: 0 | Viewed: 0 | Not Viewed: 0 | Selected: 0",
                                    foreground="blue")
        self.lbl_status = ttk.Label(self, text="", foreground="green")

        self._layout()

    def _layout(self):
        # linha 0: Load
        self.btn_load.grid(row=0, column=0, columnspan=2,
                           sticky="ew", padx=5, pady=5)

        # linha 1: filtros
        self.ctrl_filters.grid(row=1, column=0, columnspan=2,
                               sticky="ew", padx=5, pady=(0,5))
        ttk.Label(self.ctrl_filters, text="User:").grid(row=0, column=0, padx=5)
        self.cb_user.grid(row=0, column=1, padx=5)
        ttk.Label(self.ctrl_filters, text="Protocol:").grid(row=0, column=2, padx=5)
        self.cb_protocol.grid(row=0, column=3, padx=5)
        ttk.Label(self.ctrl_filters, text="Analysis:").grid(row=0, column=4, padx=5)
        self.cb_analysis.grid(row=0, column=5, padx=5)
        ttk.Label(self.ctrl_filters, text="Units:").grid(row=0, column=6, padx=5)
        self.lb_unit.grid(row=0, column=7, padx=5)

        # linha 2: checkboxes + notebook
        self.expt_canvas.grid(row=2, column=0, sticky="nsew",
                              padx=(5,0), pady=5)
        self.expt_scroll.grid(row=2, column=0, sticky="nse",
                              padx=(0,5), pady=5)
        self.nb.grid(row=2, column=1, sticky="nsew",
                     padx=5, pady=5)

        # configura abas
        self.nb.add(self.frame_meta, text="Metadata")
        self.txt_meta.grid(row=0, column=0, sticky="nsew")
        self.sb_meta_v.grid(row=0, column=1, sticky="ns")
        self.sb_meta_h.grid(row=1, column=0, sticky="ew")
        self.frame_meta.grid_rowconfigure(0, weight=1)
        self.frame_meta.grid_columnconfigure(0, weight=1)

        self.nb.add(self.frame_plot, text="Plots")
        self.ctrl_plot.grid(row=0, column=0, sticky="ew", pady=5)
        self.frame_plot.grid_rowconfigure(1, weight=1)
        self.frame_plot.grid_columnconfigure(0, weight=1)

        # linha 3: contadores
        self.lbl_counts.grid(row=3, column=0, columnspan=2,
                             sticky="ew", padx=5, pady=(0,5))
        # linha 4: export + status
        self.btn_export.grid(row=4, column=0, sticky="ew", padx=5, pady=5)
        self.lbl_status.grid(row=4, column=1, sticky="ew", padx=5, pady=5)

        # ajuste redimensionamento
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

        # reset
        self.tables.clear()
        self.experiments.clear()
        for w in self.expt_frame.winfo_children():
            w.destroy()
        self.txt_meta.config(state=tk.NORMAL)
        self.txt_meta.delete("1.0", tk.END)
        self.txt_meta.config(state=tk.DISABLED)
        self._clear_canvas()
        self.expt_vars.clear()
        self.viewed_ids.clear()

        # load
        for p in paths:
            try:
                raw = json.load(open(p, "r", encoding="utf-8"))
            except Exception as e:
                messagebox.showwarning("Load", f"Falha ao ler {p}:\n{e}")
                continue
            if isinstance(raw, dict) and "body" in raw:
                raw = json.loads(raw["body"] or "{}")
            name = os.path.splitext(os.path.basename(p))[0].lower()
            if isinstance(raw, list):
                if "experiment" in name:
                    self.experiments = raw
                else:
                    self.tables[name] = raw

        if not self.experiments:
            messagebox.showerror("Load", "Nenhum experimento carregado.")
            return

        # popula users
        users = sorted({ex["general_info"].get("user_UUID") for ex in self.experiments})
        self.cb_user["values"] = users
        self.cb_user.set("")
        self.cb_protocol.set("")
        self.cb_analysis.set("")
        self.lb_unit.selection_clear(0, tk.END)

        self._build_checkboxes(self.experiments)

    def on_user_filter(self, _):
        u = self.user_var.get()
        self.user_filtered = [ex for ex in self.experiments
                              if ex["general_info"].get("user_UUID")==u]
        protables = self.tables.get("protocol", [])
        self.table_protocol = {p["UUID"]:p for p in protables}
        names = sorted({
            self.table_protocol.get(
                ex["general_info"].get("protocol_UUID"),{}
            ).get("name",
                  ex["general_info"].get("protocol_UUID"))
            for ex in self.user_filtered})
        self.cb_protocol["values"] = names
        self.cb_protocol.set("")
        self.cb_analysis.set("")
        self.lb_unit.selection_clear(0, tk.END)
        self._apply_filters()

    def on_protocol_filter(self, _):
        nm = self.protocol_var.get()
        pu = next((pu for pu,p in self.table_protocol.items() if p.get("name")==nm), None)
        self.protocol_filtered = [ex for ex in self.user_filtered
                                  if ex["general_info"].get("protocol_UUID")==pu]
        proto = self.table_protocol.get(pu, {})
        actives = [a for a in proto.get("analysis",[]) if a.get("active")]
        self.analysis_map = {a["name"]:a["uuid"] for a in actives}
        names = ["Todos"] + [a["name"] for a in actives]
        self.cb_analysis["values"] = names
        self.cb_analysis.set("Todos")
        self.lb_unit.selection_clear(0, tk.END)
        self._apply_filters()

    def on_analysis_filter(self, _):
        sel = self.analysis_var.get()
        base = self.protocol_filtered
        if sel in ("","Todos"):
            self.analysis_filtered = base.copy()
        else:
            au = self.analysis_map.get(sel)
            self.analysis_filtered = [ex for ex in base
                                      if ex["general_info"].get("analysis_UUID")==au]
        self.lb_unit.selection_clear(0, tk.END)
        self._apply_filters()

    def _apply_filters(self):
        base = (self.analysis_filtered or
                self.protocol_filtered or
                self.user_filtered or
                self.experiments)
        sel_idxs = self.lb_unit.curselection()
        if sel_idxs:
            sel_units = {self.lb_unit.get(i) for i in sel_idxs}
            self.unit_filtered = []
            for ex in base:
                units = {c.get("unit") for c in ex.get("calibration",{}).values()}
                if sel_units.issubset(units):
                    self.unit_filtered.append(ex)
        else:
            self.unit_filtered = base
        self._build_checkboxes(self.unit_filtered)

    def _build_checkboxes(self, arr):
        # limpa tudo…
        for w in self.expt_frame.winfo_children():
            w.destroy()
        self.expt_vars.clear()

        for i, ex in enumerate(arr, start=1):
            var = tk.BooleanVar(value=True)
            frm = ttk.Frame(self.expt_frame)
            cb  = ttk.Checkbutton(frm, variable=var, command=self.update_counts)

            lbl = ttk.Label(
                frm,
                text=f"{i}: {ex.get('experiment_UUID','n/a')}",
                cursor="hand2",
                font=self.font_bold    # ← começa em negrito
            )
            lbl.bind("<Button-1>", lambda e, ex=ex: self._select_experiment(ex))

            cb.pack(side="left")
            lbl.pack(side="left", padx=5)
            frm.pack(anchor="w", pady=1, fill="x")

            # armazena também o label
            self.expt_vars.append((var, ex, lbl))

        self.update_counts()


    def update_counts(self):
        total    = len(self.expt_vars)
        selected = sum(var.get() for var, _, _ in self.expt_vars)
        viewed   = len(self.viewed_ids & {ex["experiment_UUID"] for _, ex, _ in self.expt_vars})
        not_view = total - viewed
        self.lbl_counts.config(
            text=f"Total: {total} | Viewed: {viewed} | Not Viewed: {not_view} | Selected: {selected}"
        )



    def _select_experiment(self, ex):
        self.current = ex
        self.viewed_ids.add(ex.get("experiment_UUID"))

        # procura o label desse experimento e tira o bold
        for var, e, lbl in self.expt_vars:
            if e is ex:
                lbl.config(font=self.font_normal)
                break

        self._show_meta()
        self._plot_all_channels()
        self.update_counts()


    def _show_meta(self):
        txt = json.dumps(self.current, indent=2, ensure_ascii=False)
        self.txt_meta.config(state=tk.NORMAL)
        self.txt_meta.delete("1.0", tk.END)
        self.txt_meta.insert(tk.END, txt)
        self.txt_meta.config(state=tk.DISABLED)

    def _plot_all_channels(self):
        if not self.current:
            return
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
            self.canvas = None

        data = self.current
        ts = data.get("timestamps", [])
        if not ts:
            return
        t0 = ts[0]
        x = [(t - t0)/60 for t in ts]

        trim_s = self.trim_start_var.get()
        trim_e = self.trim_end_var.get()

        all_channels = [f"F{i}" for i in range(1,9)] + ["CLR","NIR"]
        sensors  = [("spectral_uv","UV"),("spectral_vis_1","VIS1"),("spectral_vis_2","VIS2")]

        sel_ch = self.channel_var.get()

        if sel_ch == "All":
            fig = Figure(figsize=(12,8), dpi=100)
            axes = fig.subplots(3,4).flatten()
            for idx, ch in enumerate(all_channels):
                ax = axes[idx]
                for key,label in sensors:
                    y = data.get(key,{}).get(ch,[])
                    if not y: continue
                    mn, mx = min(y), max(y)
                    span = mx-mn or 1.0
                    y_norm = [(v-mn)/span for v in y]
                    if trim_s>0:
                        first = y_norm[min(trim_s, len(y_norm)-1)]
                        y_norm = [first]*min(trim_s,len(y_norm)) + y_norm[trim_s:]
                    if trim_e>0:
                        last = y_norm[-min(trim_e,len(y_norm))-1]
                        y_norm = y_norm[:-trim_e] + [last]*min(trim_e,len(y_norm))
                    ax.plot(x, y_norm, marker=".", linestyle="-", label=label)
                ax.set_title(ch); ax.set_xlabel("min desde t₀"); ax.set_ylabel("norm.")
                ax.legend(fontsize=6)

            ax = axes[10]
            temp = data.get("temperature",{}).get("sample",[])
            if temp:
                if trim_s>0:
                    first_t = temp[min(trim_s,len(temp)-1)]
                    temp = [first_t]*min(trim_s,len(temp)) + temp[trim_s:]
                if trim_e>0:
                    last_t = temp[-min(trim_e,len(temp))-1]
                    temp = temp[:-trim_e] + [last_t]*min(trim_e,len(temp))
                ax.plot(x, temp, marker=".", linestyle="-")
            ax.set_title("temperature.sample"); ax.set_xlabel("min desde t₀"); ax.set_ylabel("°C")
            axes[11].axis("off")
        else:
            fig = Figure(figsize=(8,6), dpi=100)
            ax = fig.add_subplot(111)
            for key,label in sensors:
                y = data.get(key,{}).get(sel_ch,[])
                if not y: continue
                mn, mx = min(y), max(y)
                span = mx-mn or 1.0
                y_norm = [(v-mn)/span for v in y]
                if trim_s>0:
                    first = y_norm[min(trim_s, len(y_norm)-1)]
                    y_norm = [first]*min(trim_s,len(y_norm)) + y_norm[trim_s:]
                if trim_e>0:
                    last = y_norm[-min(trim_e,len(y_norm))-1]
                    y_norm = y_norm[:-trim_e] + [last]*min(trim_e,len(y_norm))
                ax.plot(x, y_norm, marker=".", linestyle="-", label=label)
            ax.set_title(f"Channel {sel_ch}")
            ax.set_xlabel("min desde t₀"); ax.set_ylabel("norm.")
            ax.legend(fontsize=8)

        self.canvas = FigureCanvasTkAgg(fig, master=self.frame_plot)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew")

    def _clear_canvas(self):
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
            self.canvas = None

    def export_filtered(self):
        to_export = [ex for var, ex, _ in self.expt_vars if var.get()]
        if not to_export:
            return messagebox.showwarning("Export", "Nenhum experimento selecionado.")
        folder = filedialog.askdirectory(title="Select export folder")
        if not folder:
            return

        self.btn_export.config(text="Saving...", state="disabled")
        self.lbl_status.config(text="Exporting…")
        self.update_idletasks()

        try:
            # 1) experiments_filtered.json
            wrappers = []
            for ex in to_export:
                ne = copy.deepcopy(ex)
                ne.pop("features", None)
                wrappers.append({
                    "statusCode": 200,
                    "body": json.dumps(ne, ensure_ascii=False),
                    "headers": HTTP_HEADERS
                })
            path_exp = os.path.join(folder, "experiments_filtered.json")
            with open(path_exp, "w", encoding="utf-8") as f:
                json.dump(wrappers, f, ensure_ascii=False, indent=2)

            # 2) protocols_filtered.json
            prot_used = sorted({ex["general_info"]["protocol_UUID"] for ex in to_export})
            prot_out  = []
            for pu in prot_used:
                p = self.table_protocol.get(pu, {})
                prot_out.append({k:v for k,v in p.items() if k!="configurations"})
            wrapped_prot = [{
                "statusCode":200, "body":json.dumps(p, ensure_ascii=False), "headers":HTTP_HEADERS
            } for p in prot_out]
            path_prot = os.path.join(folder, "protocols_filtered.json")
            with open(path_prot, "w", encoding="utf-8") as f:
                json.dump(wrapped_prot, f, ensure_ascii=False, indent=2)

            # 3) culture_media_filtered.json
            media = self.tables.get("culture_media", [])
            media_uuids = {p.get("cultureMediaUUID") for p in prot_out if p.get("cultureMediaUUID")}
            cult_out = [m for m in media if m.get("UUID") in media_uuids]
            wrapped_cult = [{
                "statusCode":200, "body":json.dumps(m, ensure_ascii=False), "headers":HTTP_HEADERS
            } for m in cult_out]
            path_cult = os.path.join(folder, "culture_media_filtered.json")
            with open(path_cult, "w", encoding="utf-8") as f:
                json.dump(wrapped_cult, f, ensure_ascii=False, indent=2)

            # 4) bacteria_filtered.json
            bact = self.tables.get("bacteria", [])
            bact_uuids = {cid for ex in to_export for cid in ex.get("calibration",{}).keys()}
            bact_out = [b for b in bact if b.get("UUID") in bact_uuids]
            wrapped_bact = [{
                "statusCode":200, "body":json.dumps(b, ensure_ascii=False), "headers":HTTP_HEADERS
            } for b in bact_out]
            path_bact = os.path.join(folder, "bacteria_filtered.json")
            with open(path_bact, "w", encoding="utf-8") as f:
                json.dump(wrapped_bact, f, ensure_ascii=False, indent=2)

            # 5) devices_filtered.json
            serials = {ex.get("serial_number") for ex in to_export if ex.get("serial_number")}
            devs = self.tables.get("devices") or self.tables.get("device") or []
            dev_out = [d for d in devs if d.get("serial_number") in serials]
            wrapped_devs = [{
                "statusCode":200, "body":json.dumps(d, ensure_ascii=False), "headers":HTTP_HEADERS
            } for d in dev_out]
            path_dev = os.path.join(folder, "devices_filtered.json")
            with open(path_dev, "w", encoding="utf-8") as f:
                json.dump(wrapped_devs, f, ensure_ascii=False, indent=2)

        finally:
            self.btn_export.config(text="Export Filtered…", state="normal")
            self.lbl_status.config(text="Export complete")
            self.update_idletasks()

        messagebox.showinfo("Export completed",
            f"• {len(wrappers)} experiments → {os.path.basename(path_exp)}\n"
            f"• {len(wrapped_prot)} protocols → {os.path.basename(path_prot)}\n"
            f"• {len(wrapped_cult)} cultures → {os.path.basename(path_cult)}\n"
            f"• {len(wrapped_bact)} bacteria → {os.path.basename(path_bact)}\n"
            f"• {len(wrapped_devs)} devices → {os.path.basename(path_dev)}"
        )

if __name__ == "__main__":
    app = ExperimentViewer()
    app.mainloop()
