from numpy.core.numeric import True_
from sklearn import metrics
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, roc_curve, auc, confusion_matrix, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

def main():
    st.title("Stress Level Dataset Model Training with SVM & Naive Bayes")
    st.sidebar.title("- Content to Show -")

if __name__ == '__main__':
    main()


@st.cache_data(persist= True)
def load():
    data= pd.read_csv("Stress-Lysis.csv")
    label= LabelEncoder()
    for i in data.columns:
        data[i] = label.fit_transform(data[i])
    return data
df = load()

option = st.sidebar.radio("Choose action", ("Display Data + EDA", "Classification Result"))

if option == "Display Data + EDA":
    st.subheader("Show Stress-Lysis dataset")
    st.write(df)

    # Visualisasi distribusi data
    st.subheader("Distribution of Data")
    fig, ax = plt.subplots(figsize=(10, 8))
    df.hist(ax=ax)
    plt.tight_layout()
    st.pyplot(fig)
    st.write("Distribusi kelas dalam data ini cukup baik untuk tujuan klasifikasi. Kelas-kelas stres rendah, sedang, dan tinggi memiliki frekuensi yang relatif seimbang, sehingga model pembelajaran mesin dapat mempelajari setiap kelas dengan representasi yang memadai.")

    # Box Plot untuk identifikasi outlier
    st.subheader("Box Plot for Outlier Identification")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, ax=ax)
    plt.title('Outlier Data')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    st.write(" Dari grafik ini, kita bisa melihat rentang data, nilai tengah (median), dan penyebaran data untuk masing-masing variabel. Untuk kelembaban, rentangnya berkisar antara 10 hingga 30 dengan beberapa data yang mendekati nilai maksimum. Suhu berkisar antara 80 hingga 100, menunjukkan distribusi yang lebih sempit dibanding kelembaban. Jumlah langkah memiliki penyebaran yang paling luas, dengan rentang dari 0 hingga hampir 200, mencerminkan variasi besar dalam aktivitas fisik. Tingkat stres memiliki rentang yang sangat kecil, mengindikasikan bahwa datanya lebih terfokus pada nilai-nilai tertentu, tanpa banyak variasi. Tidak ada outlier yang jelas terlihat dalam grafik ini, menunjukkan bahwa semua data berada dalam batasan yang diharapkan untuk masing-masing variabel.")

    # Periksa korelasi antar kolom
    st.subheader("Correlation Matrix")
    correlation_matrix = df.corr()
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='Reds', fmt='.2f', linewidths=0.5, ax=ax)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    st.pyplot(fig)
    st.write("semua fitur (Humidity, Temperature, dan Step count) memiliki korelasi yang kuat dengan tingkat stres (Stress Level). Terutama, kelembaban tubuh (Humidity) dan suhu tubuh (Temperature) menunjukkan korelasi yang sangat tinggi dengan tingkat stres. Hal ini mengindikasikan bahwa perubahan pada kelembaban dan suhu tubuh dapat menjadi indikator yang sangat baik dalam menentukan tingkat stres seseorang. Jumlah langkah yang diambil juga berhubungan erat dengan tingkat stres, meskipun tidak sekuat dua fitur lainnya.")

    # Scatter plot dengan fit garis regresi
    st.subheader("Scatter plot with Regression Line")
    data_encoded = df.copy()
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    sns.regplot(x='Humidity', y='Stress Level', data=data_encoded, scatter_kws={'s': 50}, order=1)
    plt.title('Humidity vs Stress Level')

    plt.subplot(1, 3, 2)
    sns.regplot(x='Temperature', y='Stress Level', data=data_encoded, scatter_kws={'s': 50}, order=1)
    plt.title('Temperature vs Stress Level')

    plt.subplot(1, 3, 3)
    sns.regplot(x='Step count', y='Stress Level', data=data_encoded, scatter_kws={'s': 50}, order=1)
    plt.title('Step count vs Stress Level')

    plt.tight_layout()
    st.pyplot(plt)
    st.write("Hasil scatterplot dengan garis regresi menunjukkan bahwa terdapat hubungan positif antara setiap fitur (Humidity, Temperature, Step count) dan tingkat stres (Stress Level). Secara spesifik, kelembaban (Humidity) memiliki korelasi positif yang jelas dengan tingkat stres, dengan dua kelompok data yang terpisah menunjukkan batasan tertentu dalam nilai kelembaban yang membedakan tingkat stres rendah, sedang, dan tinggi. Suhu (Temperature) juga menunjukkan tren positif, meskipun data lebih tersebar, menunjukkan bahwa tingkat stres cenderung meningkat dengan kenaikan suhu, meskipun korelasinya mungkin lebih lemah dibandingkan dengan kelembaban. Sementara itu, jumlah langkah (Step count) menunjukkan tren positif yang sangat jelas, menandakan bahwa semakin banyak langkah yang diambil, semakin tinggi tingkat stres, dengan kelompok data yang jelas untuk setiap tingkat stres.")

    # Feature Importance
    st.subheader("Feature Importance")
    X = df.drop('Stress Level', axis=1)
    y = df['Stress Level']

    model = RandomForestClassifier()
    model.fit(X, y)

    importances = model.feature_importances_
    feature_importances = pd.Series(importances, index=X.columns)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_importances, y=feature_importances.index, palette='viridis')
    plt.title('Feature Importance')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    st.pyplot(plt)
    st.write("Berdasarkan grafik 'Feature Importance' yang dihasilkan oleh model machine learning seperti Random Forest, terlihat bahwa fitur 'Humidity' memiliki pengaruh terbesar terhadap prediksi 'Stress Level', diikuti oleh 'Temperature' dan 'Step count'. Hal ini menunjukkan bahwa kelembaban tubuh merupakan indikator paling signifikan dalam menentukan tingkat stres seseorang. Suhu tubuh juga memberikan kontribusi penting, sementara jumlah langkah yang diambil meskipun lebih rendah pengaruhnya, tetap berperan dalam prediksi stres. Model ini mengungkapkan betapa pentingnya kondisi fisik dalam memahami dan memprediksi tingkat stres, dengan kelembaban tubuh menjadi faktor dominan.")

    # Pair Plot of Numerical Features Colored by Stress Level
    st.subheader("Pair Plot of Numerical Features Colored by Stress Level")
    numerical_features = df.drop("Stress Level", axis=1)
    numerical_features["Stress Level"] = df["Stress Level"]
    plt.figure(figsize=(15, 10))
    sns.pairplot(numerical_features, hue="Stress Level", palette="viridis")
    plt.suptitle("Pair Plot of Numerical Features Colored by Stress Level", y=1.02)
    plt.tight_layout()
    st.pyplot(plt)
    st.write("Grafik tersebut merupakan pair plot yang menampilkan hubungan antara tiga fitur numerik yaitu kelembaban (Humidity), suhu (Temperature), dan jumlah langkah (Step count) dengan tingkat stres yang ditandai oleh warna berbeda: ungu untuk stres level 0, biru untuk stres level 1, dan kuning untuk stres level 2. Dari grafik ini, terlihat bahwa setiap tingkat stres memiliki rentang kelembaban, suhu, dan jumlah langkah yang berbeda. Stres level 0 cenderung terjadi pada kelembaban rendah (sekitar 10-15%), suhu rendah (sekitar 80-85 derajat), dan jumlah langkah yang rendah (0-100). Stres level 1 terjadi pada rentang menengah untuk ketiga fitur tersebut, sementara stres level 2 terjadi pada kelembaban tinggi (25-30%), suhu tinggi (95-100 derajat), dan jumlah langkah yang tinggi (150-200). Hubungan linier yang kuat antara suhu dan kelembaban serta pola blok pada grafik jumlah langkah menunjukkan bahwa faktor-faktor lingkungan ini sangat mempengaruhi tingkat stres seseorang, dengan tingkat stres meningkat seiring dengan peningkatan suhu dan kelembaban, serta peningkatan jumlah langkah.")
else:
    st.sidebar.markdown(" ")
    st.sidebar.title("Complete the configuration first to start classification")
    class_names=["Low Sress", "Medium Stress", "High Stress"]

    def plot_metrics(metrics_list):
        if "Confusion Matrix" in metrics_list:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)  # Compute confusion matrix
            
            fig, ax = plt.subplots(figsize=(10, 7))  # Increase figure size
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=class_names, yticklabels=class_names, annot_kws={"size": 14})
            ax.set_xlabel('Predicted labels')
            ax.set_ylabel('True labels')
            ax.set_title('Confusion Matrix')
            
            st.pyplot(fig)  # Display the plot in Streamlit

        if "ROC Curve" in metrics_list:
            st.subheader("ROC Curve")
            fig, ax = plt.subplots()
            
            for i in range(len(class_names)):
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, label=f'Class {class_names[i]} (AUC = {roc_auc:.2f})')
            
            ax.plot([0, 1], [0, 1], 'k--')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC Curve')
            ax.legend(loc='lower right')
            
            st.pyplot(fig)  # Display the plot in Streamlit

        if "Precision-Recall Curve" in metrics_list:
            st.subheader("Precision-Recall Curve")
            fig, ax = plt.subplots()
            
            for i in range(len(class_names)):
                precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
                avg_precision = average_precision_score(y_test_bin[:, i], y_score[:, i])
                ax.plot(recall, precision, label=f'Class {class_names[i]} (AP = {avg_precision:.2f})')
            
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title('Precision-Recall Curve')
            ax.legend(loc='lower left')
            
            st.pyplot(fig)  # Display the plot in Streamlit


    st.sidebar.subheader("Choose classifier")
    classifier = st.sidebar.selectbox("Classifier", ("Support Vector Machine (SVM)", "Naive Bayes"))

    if classifier == "Support Vector Machine (SVM)":
        st.sidebar.subheader("Hyperparameters")
        param = st.sidebar.selectbox("Parameter Used", ("Auto Tunning", "Manual"))
        if param == "Manual":
            C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key="C")
            kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key="kernel") 
            gamma = st.sidebar.radio("Gamma (Kernal coefficient", ("scale", "auto"), key="gamma")

    metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))

    if st.sidebar.button("Classify", key="classify"):

        # Pisahkan atribut dan label
        x = df.drop('Stress Level', axis=1)
        y = df['Stress Level']

        # Bagi dataset menjadi data pelatihan dan pengujian
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        if classifier == "Support Vector Machine (SVM)":
            st.subheader("Support Vector Machine (SVM) results")
            if param == "Manual":
                var_c = C
                var_gamma = gamma
                var_kernel = kernel

                # untuk display streamlit
                param_dict = {
                    "C": C,
                    "gamma": gamma,
                    "kernel": kernel
                }
                st.write("Parameter used:", param_dict)
            else:
                # Definisikan hyperparameter grid
                param_grid = {
                    'C': [0.1, 1, 10, 100],
                    'gamma': [1, 0.1, 0.01, 0.001],
                    'kernel': ['rbf', 'linear', 'poly', 'sigmoid']
                }

                # Inisialisasi model SVM
                svm_model = SVC()

                # Buat objek GridSearchCV
                grid_search = GridSearchCV(estimator=svm_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

                # Latih model pada dataset dengan GridSearch
                grid_search.fit(x_train, y_train)

                # Print hasil pencarian grid
                st.write("Best Estimator:", grid_search.best_estimator_)

                var_c = grid_search.best_params_['C']
                var_gamma = grid_search.best_params_['gamma']
                var_kernel = grid_search.best_params_['kernel']
        
            model = SVC(C=var_c, kernel=var_kernel, gamma=var_gamma)
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)

            # untuk grafik AUC
            y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
            y_score = model.decision_function(x_test)  # Obtain scores

        else:
            st.subheader("Naive Bayes (GaussianNB) results")
            # Inisialisasi model Naive Bayes
            nb_model = GaussianNB()

            # Definisikan hyperparameter grid (tidak ada hyperparameter yang dapat dioptimalkan untuk Naive Bayes)
            param_grid = {}

            # Buat objek GridSearchCV
            grid_search = GridSearchCV(estimator=nb_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

            # Latih model pada dataset dengan GridSearch
            grid_search.fit(x_train, y_train)

            # Print hasil pencarian grid (karena tidak ada hyperparameter yang dioptimalkan, hanya mencetak model default)
            print("Best Estimator:", grid_search.best_estimator_)

            # Evaluasi model terbaik
            model = grid_search.best_estimator_
            y_pred = model.predict(x_test)

            # untuk grafik AUC
            y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
            y_score = model.predict_proba(x_test)


        # Hitung dan print hasil evaluasi
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        st.write("Accuracy: ", round(accuracy, 7))
        st.write("Precision: ", round(precision, 7))
        st.write("Recall: ", round(recall, 7))
        st.write("F1-Score: ", round(f1, 7))

        plot_metrics(metrics)
        st.write("selesai proses")
        
