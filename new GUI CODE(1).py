import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from tkinter import filedialog, messagebox
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from io import StringIO
import numpy as np

class DataAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Data Analysis GUI")
        self.root.geometry("900x700")
        self.root.resizable(False, False)

        # Apply a modern theme
        self.style = ttk.Style("solar")
        self.root.configure(bg="#f8f9fa")  # 3D white background color

        # Frame for title
        title_frame = ttk.Frame(root, padding=20, bootstyle="dark")
        title_frame.pack(fill=X)

        title_label = ttk.Label(
            title_frame, text="Data Analysis Application", font=("Helvetica", 24, "bold"), bootstyle="light-inverse"
        )
        title_label.pack()

        # Main frame
        main_frame = ttk.Frame(root, padding=30)
        main_frame.pack(fill=BOTH, expand=True)

        # Buttons Panel
        button_frame = ttk.Frame(main_frame, padding=10, bootstyle="secondary")
        button_frame.grid(row=0, column=0, sticky="nw")

        self.create_button(button_frame, "Load Dataset", self.load_dataset).pack(pady=10)
        self.create_button(button_frame, "Show Summary", self.show_summary).pack(pady=10)
        self.create_button(button_frame, "Visualize Data", self.visualize_data).pack(pady=10)
        self.create_button(button_frame, "Export Data", self.export_data).pack(pady=10)
        self.create_button(button_frame, "Predict with Model", self.predict_with_model).pack(pady=10)

        # Placeholder for dynamic content
        self.content_frame = ttk.Frame(main_frame, padding=10, bootstyle="info")
        self.content_frame.grid(row=0, column=1, sticky="nsew")

        # Configure column weights
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)

        self.data = None
        self.model = None

    def create_button(self, parent, text, command):
        return ttk.Button(
            parent,
            text=text,
            command=command,
            bootstyle="secondary",
            style="Custom.TButton",  # Custom style for buttons
            width=25,
        )

    def load_dataset(self):
        # Sample data
        sample_data = """No,year,month,day,hour,PM2.5,PM10,SO2,NO2,CO,O3,TEMP,PRES,DEWP,RAIN,wd,WSPM,station
1,2013,3,1,0,4,4,4,7,300,77,-0.7,1023,-18.8,0,NNW,4.4,Aotizhongxin
2,2013,3,1,1,8,8,4,7,300,77,-1.1,1023.2,-18.2,0,N,4.7,Aotizhongxin
3,2013,3,1,2,7,7,5,10,300,73,-1.1,1023.5,-18.2,0,NNW,5.6,Aotizhongxin"""

        try:
            self.data = pd.read_csv(StringIO(sample_data))
            messagebox.showinfo("Success", "Sample dataset loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load dataset: {e}")

    def show_summary(self):
        if self.data is None:
            messagebox.showwarning("Warning", "Please load a dataset first!")
            return

        summary = self.data.describe()

        for widget in self.content_frame.winfo_children():
            widget.destroy()

        text = ttk.Text(self.content_frame, wrap="none", width=100, height=30)
        text.insert("1.0", summary.to_string())
        text.pack(fill=BOTH, expand=True)

    def visualize_data(self):
        if self.data is None:
            messagebox.showwarning("Warning", "Please load a dataset first!")
            return

        columns = list(self.data.columns)
        col_window = ttk.Toplevel(self.root)
        col_window.title("Select Columns to Visualize")

        ttk.Label(col_window, text="Select Column:", bootstyle=INFO).pack()
        selected_col = ttk.StringVar(col_window)
        selected_col.set(columns[0])
        dropdown = ttk.OptionMenu(col_window, selected_col, *columns)
        dropdown.pack()

        def plot_column():
            col = selected_col.get()
            plt.figure(figsize=(8, 5))
            if pd.api.types.is_numeric_dtype(self.data[col]):
                sns.histplot(self.data[col], kde=True, color="blue")
                plt.title(f"Histogram of {col}")
            else:
                sns.countplot(x=self.data[col], palette="pastel")
                plt.title(f"Countplot of {col}")
            plt.show()

        ttk.Button(col_window, text="Visualize", command=plot_column, bootstyle=PRIMARY).pack(pady=10)

    def predict_with_model(self):
        if self.data is None:
            messagebox.showwarning("Warning", "Please load a dataset first!")
            return

        try:
            numeric_data = self.data.select_dtypes(include=[np.number])
            X = numeric_data.drop("PM2.5", axis=1)
            y = numeric_data["PM2.5"]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.model.fit(X_train, y_train)

            predictions = self.model.predict(X_test)
            mse = mean_squared_error(y_test, predictions)

            results_text = f"Mean Squared Error: {mse:.2f}\n\nPredictions:\n{predictions}"

            for widget in self.content_frame.winfo_children():
                widget.destroy()

            text = ttk.Text(self.content_frame, wrap="none", width=80, height=20)
            text.insert("1.0", results_text)
            text.pack(fill=BOTH, expand=True)

            # Plot actual vs predicted values
            plt.figure(figsize=(10, 6))
            plt.scatter(y_test, predictions, color="black", alpha=0.7, label="Predictions")
            plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red", linestyle="--", label="Ideal Fit")
            plt.xlabel("Actual Values")
            plt.ylabel("Predicted Values")
            plt.title("Actual vs Predicted Values")
            plt.legend()
            plt.show()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to train model: {e}")

    def export_data(self):
        if self.data is None:
            messagebox.showwarning("Warning", "Please load a dataset first!")
            return

        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if file_path:
            try:
                self.data.to_csv(file_path, index=False)
                messagebox.showinfo("Success", "Dataset exported successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export dataset: {e}")

if __name__ == "__main__":
    root = ttk.Window(themename="solar")

    # Create a custom button style
    custom_style = ttk.Style()
    custom_style.configure("Custom.TButton",
                           font=("Helvetica", 12, "bold"),
                           foreground="white",  # White text color
                           background="black",  # Black background color
                           bordercolor="red",   # Red border color
                           focusthickness=3,
                           focusthicknesscolor="red")  # Active red background

    app = DataAnalysisApp(root)
    root.mainloop()
