"""
Лабораторная работа №3: Верификация пользователя.
Двухфакторная аутентификация: Проверка ID + пароль + БХЧ (биометрический вектор Хаара).
Хранение базы данных пользователей в JSON (biometrics_users.json).

Требования:
 a) Ввод пароля (формирование БХЧ) и ID. Проверка: пароль + БХЧ (с погрешностью) + ID;
 b) Вердикт: действительно ли пользователь является тем, за кого себя выдает;
 c) Оповещение при отсутствии пользователя в БД.
"""

import json
import math
import os
import tkinter as tk
from tkinter import ttk, messagebox

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np

JSON_FILE = "biometrics_users.json"
HAAR_N = 16  # Размерность вектора БХЧ (N = 16)


# =====================================================================
# 1. МАТЕМАТИЧЕСКИЙ БЛОК: СИНТЕЗ f(t) И ВЫЧИСЛЕНИЕ БХЧ (ВЕКТОР ХААРА)
# =====================================================================

def generate_haar_matrix(N):
    """Генерация матрицы Хаара H*(n) размерности N x N."""
    n = int(math.log2(N))
    H = np.zeros((N, N), dtype=float)
    H[0, :] = 1.0
    row = 1
    for r in range(n):
        amp = 2 ** (r / 2.0)
        num_intervals = 2 ** r
        len_sub = N // num_intervals
        half_len = len_sub // 2
        for m in range(num_intervals):
            start = m * len_sub
            mid = start + half_len
            end = start + len_sub
            if row < N:
                H[row, start:mid] = amp
                H[row, mid:end] = -amp
                row += 1
    return H


def calculate_bhch_vector(events, total_time, N=HAAR_N):
    """
    Построение функции f(t) (пауза=0, удержание=1, наложение=2)
    и получение вектора БХЧ (Хаара) V = (1/N) * H* * F.
    """
    samples_count = 512
    t_vals = np.linspace(0, total_time, samples_count)
    f_vals = np.zeros(samples_count, dtype=float)

    for idx, t in enumerate(t_vals):
        active_count = sum(1 for e in events if e['down'] <= t <= e['up'])
        if active_count == 0:
            f_vals[idx] = 0.0
        elif active_count == 1:
            f_vals[idx] = 1.0
        else:
            f_vals[idx] = 2.0

    chunk_size = samples_count // N
    F_discrete = np.zeros(N)
    for i in range(N):
        F_discrete[i] = np.mean(f_vals[i * chunk_size : (i + 1) * chunk_size])

    H_star = generate_haar_matrix(N)
    V = (1.0 / N) * np.dot(H_star, F_discrete)
    return V, t_vals, f_vals


# =====================================================================
# 2. МОДУЛЬ БАЗЫ ДАННЫХ НА ОСНОВЕ JSON
# =====================================================================

class BiometricsJSONDB:
    def __init__(self, filepath=JSON_FILE):
        self.filepath = filepath
        self.users = []
        self.load_data()

    def load_data(self):
        """Загрузка пользователей из файла JSON."""
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        self.users = data
            except Exception as e:
                print(f"Ошибка при чтении {self.filepath}: {e}")
                self.users = []
        else:
            self.users = []
            self.save_data()

    def save_data(self):
        """Сохранение всех пользователей в JSON."""
        try:
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump(self.users, f, ensure_ascii=False, indent=2)
        except Exception as e:
            messagebox.showerror("Ошибка сохранения", f"Не удалось сохранить данные в JSON: {e}")

    def get_user_by_id(self, user_id):
        """Поиск пользователя по ID."""
        for u in self.users:
            if u.get("id") == user_id:
                return (
                    u["id"],
                    u["username"],
                    u["phrase"],
                    json.dumps(u["template_vector"]),
                    u["attempts_count"]
                )
        return None

    def get_all_users(self):
        """Получение списка всех пользователей для таблицы."""
        result = []
        for u in sorted(self.users, key=lambda x: x.get("id", 0)):
            result.append((
                u.get("id"),
                u.get("username"),
                u.get("phrase"),
                u.get("attempts_count", 1)
            ))
        return result

    def register_or_update(self, username, phrase, vector):
        """Регистрация нового пользователя или усреднение эталона при дообучении."""
        vec_list = vector.tolist() if isinstance(vector, np.ndarray) else list(vector)

        # Поиск существующего пользователя по логину
        existing = None
        for u in self.users:
            if u.get("username") == username:
                existing = u
                break

        if existing is None:
            # Новый ID = максимум + 1
            new_id = max([u.get("id", 0) for u in self.users], default=0) + 1
            new_user = {
                "id": new_id,
                "username": username,
                "phrase": phrase,
                "template_vector": vec_list,
                "attempts_count": 1
            }
            self.users.append(new_user)
        else:
            old_vec = np.array(existing["template_vector"])
            count = existing.get("attempts_count", 1)
            new_count = count + 1
            updated = (old_vec * count + np.array(vec_list)) / new_count

            existing["template_vector"] = updated.tolist()
            existing["attempts_count"] = new_count
            existing["phrase"] = phrase

        self.save_data()


# =====================================================================
# 3. ГРАФИЧЕСКИЙ ИНТЕРФЕЙС (GUI НА TKINTER)
# =====================================================================

class VerificationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Лабораторная работа №3: Верификация пользователя по ID, паролю и БХЧ (JSON БД)")
        self.root.geometry("1120x840")

        self.db = BiometricsJSONDB()

        # Входные параметры верификации
        self.id_var = tk.StringVar(value="1")
        self.threshold_var = tk.DoubleVar(value=0.25)

        # Переменные для регистрации (вкладка 2)
        self.reg_name_var = tk.StringVar(value="Баженов")
        self.reg_phrase_var = tk.StringVar(value="password1234")

        # Перехват аппаратных событий набора
        self.active_keys = {}
        self.current_events = []

        self._build_ui()
        self.refresh_users_table()

    def _build_ui(self):
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # ---------------- ВКЛАДКА 1: ВЕРИФИКАЦИЯ ----------------
        tab_verify = ttk.Frame(notebook)
        notebook.add(tab_verify, text="Верификация пользователя")

        # Верхняя панель: ввод ID и порога
        top_box = ttk.LabelFrame(tab_verify, text="1. Предъявление идентификатора (ID) и параметров допуска", padding=10)
        top_box.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        ttk.Label(top_box, text="ID пользователя в системе:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Entry(top_box, textvariable=self.id_var, font=("Segoe UI", 11, "bold"), width=10).grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)

        ttk.Label(top_box, text="Погрешность допуска (Порог ε):").grid(row=0, column=2, sticky=tk.W, padx=(20, 5), pady=2)
        ttk.Entry(top_box, textvariable=self.threshold_var, font=("Segoe UI", 10), width=8).grid(row=0, column=3, sticky=tk.W, padx=5, pady=2)

        # Панель ввода пароля
        input_box = ttk.LabelFrame(tab_verify, text="2. Ввод пароля (в процессе набора формируется БХЧ)", padding=10)
        input_box.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        ttk.Label(input_box, text="Введите пароль в обычном темпе и нажмите Enter или кнопку «Верифицировать»:").pack(anchor=tk.W)
        self.verify_entry = ttk.Entry(input_box, font=("Courier New", 13), width=65)
        self.verify_entry.pack(anchor=tk.W, pady=6, fill=tk.X)

        self.verify_entry.bind("<KeyPress>", self.on_key_press)
        self.verify_entry.bind("<KeyRelease>", self.on_key_release)
        self.verify_entry.bind("<Return>", self.on_verify_click)

        btn_box = ttk.Frame(input_box)
        btn_box.pack(anchor=tk.W, pady=2)
        ttk.Button(btn_box, text="Пройти верификацию (Enter)", command=self.on_verify_click).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_box, text="Сбросить ввод", command=self.clear_verify_input).pack(side=tk.LEFT, padx=5)

        # Результат вердикта системы
        self.lbl_verdict = tk.Label(
            tab_verify, text="ОЖИДАНИЕ ВВОДА ДАННЫХ", font=("Segoe UI", 12, "bold"),
            bg="#EEEEEE", fg="#555555", relief=tk.GROOVE, padx=10, pady=8
        )
        self.lbl_verdict.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        # Графическое сравнение векторов (Matplotlib)
        self.fig = Figure(figsize=(9, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=tab_verify)
        self.canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, padx=10, pady=5)

        # ---------------- ВКЛАДКА 2: БАЗА ПОЛЬЗОВАТЕЛЕЙ ----------------
        tab_db = ttk.Frame(notebook)
        notebook.add(tab_db, text="База данных пользователей / Обучение эталона")

        # Регистрация нового эталона
        reg_box = ttk.LabelFrame(tab_db, text="Добавление / Обучение пользователя (сохранение в JSON)", padding=10)
        reg_box.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        ttk.Label(reg_box, text="ФИО / Логин:").grid(row=0, column=0, padx=5, pady=2)
        ttk.Entry(reg_box, textvariable=self.reg_name_var, width=20).grid(row=0, column=1, padx=5, pady=2)

        ttk.Label(reg_box, text="Пароль:").grid(row=0, column=2, padx=5, pady=2)
        ttk.Entry(reg_box, textvariable=self.reg_phrase_var, width=20).grid(row=0, column=3, padx=5, pady=2)

        ttk.Label(reg_box, text="Контрольный набор:").grid(row=1, column=0, padx=5, pady=2)
        self.reg_entry = ttk.Entry(reg_box, font=("Courier New", 11), width=40)
        self.reg_entry.grid(row=1, column=1, columnspan=3, sticky=tk.W, padx=5, pady=4)
        self.reg_entry.bind("<KeyPress>", self.on_reg_key_press)
        self.reg_entry.bind("<KeyRelease>", self.on_reg_key_release)

        ttk.Button(reg_box, text="Записать эталон в JSON", command=self.on_register_click).grid(row=1, column=4, padx=10, pady=4)

        # Таблица зарегистрированных пользователей
        table_box = ttk.LabelFrame(tab_db, text=f"Зарегистрированные пользователи ({JSON_FILE})", padding=10)
        table_box.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, padx=10, pady=5)

        cols = ("id", "username", "phrase", "attempts")
        self.tree = ttk.Treeview(table_box, columns=cols, show="headings", height=10)
        self.tree.heading("id", text="ID пользователя")
        self.tree.heading("username", text="ФИО / Логин")
        self.tree.heading("phrase", text="Текстовый пароль")
        self.tree.heading("attempts", text="Число образцов в эталоне")
        self.tree.column("id", width=120, anchor=tk.CENTER)
        self.tree.column("username", width=220)
        self.tree.column("phrase", width=220)
        self.tree.column("attempts", width=180, anchor=tk.CENTER)
        self.tree.pack(fill=tk.BOTH, expand=True)

        ttk.Button(table_box, text="Обновить список пользователей", command=self.refresh_users_table).pack(anchor=tk.W, pady=5)

    # ---------------- ПЕРЕХВАТ КЛАВИШ ДЛЯ ВЕРИФИКАЦИИ ----------------
    def on_key_press(self, event):
        t = event.time / 1000.0
        k = event.char
        if not k or (ord(k) < 32 and ord(k) != 13):
            return
        if k not in self.active_keys:
            self.active_keys[k] = t

    def on_key_release(self, event):
        t = event.time / 1000.0
        k = event.char
        if k in self.active_keys:
            t_down = self.active_keys.pop(k)
            self.current_events.append({'key': k, 'down': t_down, 'up': t})

    def clear_verify_input(self):
        self.active_keys.clear()
        self.current_events.clear()
        self.verify_entry.delete(0, tk.END)

    # ---------------- ГЛАВНЫЙ АЛГОРИТМ ВЕРИФИКАЦИИ (a, b, c) ----------------
    def on_verify_click(self, event=None):
        raw_id = self.id_var.get().strip()
        entered_password = self.verify_entry.get()

        # 1. Проверка корректности введенного ID
        if not raw_id.isdigit():
            messagebox.showerror("Ошибка", "ID пользователя должен быть положительным числом!")
            return
        user_id = int(raw_id)

        # 2. Требование (c): Проверка наличия пользователя в БД
        user_record = self.db.get_user_by_id(user_id)
        if user_record is None:
            self.lbl_verdict.config(
                text=f"ОПОВЕЩЕНИЕ: Пользователь с ID={user_id} отсутствует в базе данных!",
                bg="#FFCDD2", fg="#B71C1C"
            )
            messagebox.showwarning(
                "Оповещение системы",
                f"Пользователь с идентификатором ID={user_id} в базе данных не найден!\n"
                "Доступ невозможен."
            )
            self.clear_verify_input()
            return

        u_id, username, stored_phrase, template_json, attempts_count = user_record
        template_vector = np.array(json.loads(template_json))

        # 3. Проверка совпадения текстового пароля
        if entered_password != stored_phrase:
            self.lbl_verdict.config(
                text=f"ОТКАЗ В ДОСТУПЕ: Неверный текстовый пароль для пользователя {username} (ID={user_id})!",
                bg="#FFCDD2", fg="#B71C1C"
            )
            messagebox.showerror("Аутентификация не пройдена", "Введен неверный пароль!")
            self.clear_verify_input()
            return

        if len(self.current_events) < 2:
            messagebox.showwarning("Ошибка", "Слишком мало событий клавиатурного ввода для формирования БХЧ!")
            self.clear_verify_input()
            return

        # 4. Формирование БХЧ (вектора Хаара V) из процесса набора
        events = sorted(self.current_events, key=lambda x: x['down'])
        t0 = events[0]['down']
        for e in events:
            e['down'] -= t0
            e['up'] -= t0
        total_time = max(e['up'] for e in events)

        current_vector, _, _ = calculate_bhch_vector(events, total_time, N=HAAR_N)

        # 5. Проверка совпадения БХЧ с эталоном с заданной погрешностью (Евклидово расстояние)
        dist = float(np.linalg.norm(current_vector - template_vector))
        threshold = self.threshold_var.get()

        # 6. Требование (b): Выдача вердикта подлинности
        is_genuine = dist <= threshold

        if is_genuine:
            status_text = (
                f"ВЕРИФИКАЦИЯ ПРОЙДЕНА УСПЕШНО!\n"
                f"Пользователь действительно является тем, за кого себя выдает: [{username}, ID={user_id}]\n"
                f"Евклидово расстояние БХЧ d = {dist:.4f} <= Порог ε = {threshold:.4f}"
            )
            self.lbl_verdict.config(text=status_text, bg="#C8E6C9", fg="#1B5E20")
            messagebox.showinfo("Подлинность подтверждена", status_text)
        else:
            status_text = (
                f"ОТКАЗ В ВЕРИФИКАЦИИ! (Возможна попытка подлога биометрии)\n"
                f"Пароль верен, но БХЧ почерка не соответствует эталону пользователя [{username}, ID={user_id}]!\n"
                f"Евклидово расстояние БХЧ d = {dist:.4f} > Порог ε = {threshold:.4f}"
            )
            self.lbl_verdict.config(text=status_text, bg="#FFCDD2", fg="#B71C1C")
            messagebox.showwarning("Отказ в доступе", status_text)

        # Отрисовка графического сравнения векторов
        self.plot_vector_comparison(current_vector, template_vector, username, dist, threshold, is_genuine)
        self.clear_verify_input()

    # ---------------- ВИЗУАЛИЗАЦИЯ СРАВНЕНИЯ ВЕКТОРОВ ----------------
    def plot_vector_comparison(self, current_v, template_v, username, dist, threshold, is_genuine):
        self.fig.clear()
        ax = self.fig.add_subplot(111)

        indices = np.arange(len(current_v))
        width = 0.38

        ax.bar(indices - width/2, template_v, width=width, color="#1976D2", label="Эталон в БД (V_эталон)", alpha=0.85)
        ax.bar(indices + width/2, current_v, width=width, color="#4CAF50" if is_genuine else "#F44336",
               label="Предъявленный ввод (V_текущий)", alpha=0.85)

        ax.set_xticks(indices)
        ax.set_xticklabels([f"v{i}" for i in indices], fontsize=8)
        ax.set_ylabel("Амплитуда коэффициента")
        ax.set_title(
            f"Сравнение БХЧ для пользователя: {username} | Евклидово расстояние d = {dist:.4f} (Порог ε = {threshold:.2f})\n"
            f"Вердикт: {'ПОДТВЕРЖДЕН (Свой)' if is_genuine else 'ОТКЛОНЕН (Чужой / Имитатор)'}",
            fontsize=10, fontweight="bold", color="#1B5E20" if is_genuine else "#B71C1C"
        )
        ax.grid(axis='y', linestyle="--", alpha=0.6)
        ax.legend(loc="upper right")

        self.fig.tight_layout()
        self.canvas.draw()

    # ---------------- РЕГИСТРАЦИЯ ЭТАЛОНОВ ----------------
    def on_reg_key_press(self, event):
        t = event.time / 1000.0
        k = event.char
        if not hasattr(self, 'reg_active'):
            self.reg_active = {}
            self.reg_events = []
        if k and (ord(k) >= 32 or ord(k) == 13) and k not in self.reg_active:
            self.reg_active[k] = t

    def on_reg_key_release(self, event):
        t = event.time / 1000.0
        k = event.char
        if hasattr(self, 'reg_active') and k in self.reg_active:
            t_down = self.reg_active.pop(k)
            self.reg_events.append({'key': k, 'down': t_down, 'up': t})

    def on_register_click(self):
        name = self.reg_name_var.get().strip()
        phrase = self.reg_phrase_var.get().strip()
        typed = self.reg_entry.get()

        if not name or not phrase:
            messagebox.showwarning("Ошибка", "Заполните имя и пароль!")
            return
        if typed != phrase:
            messagebox.showwarning("Ошибка", "Контрольный набор не совпадает с паролем!")
            return
        if not hasattr(self, 'reg_events') or len(self.reg_events) < 2:
            messagebox.showwarning("Ошибка", "Нет событий набора!")
            return

        events = sorted(self.reg_events, key=lambda x: x['down'])
        t0 = events[0]['down']
        for e in events:
            e['down'] -= t0
            e['up'] -= t0
        total_time = max(e['up'] for e in events)

        vector, _, _ = calculate_bhch_vector(events, total_time, N=HAAR_N)
        self.db.register_or_update(name, phrase, vector)

        messagebox.showinfo("Успех", f"Пользователь '{name}' сохранен/обновлен в JSON!")
        self.reg_active.clear()
        self.reg_events.clear()
        self.reg_entry.delete(0, tk.END)
        self.refresh_users_table()

    def refresh_users_table(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        for r in self.db.get_all_users():
            self.tree.insert("", tk.END, values=r)


if __name__ == "__main__":
    root = tk.Tk()
    app = VerificationApp(root)
    root.mainloop()