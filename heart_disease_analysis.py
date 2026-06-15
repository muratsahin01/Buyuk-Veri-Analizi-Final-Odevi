# =============================================================================
# BÜYÜK VERİ ANALİZİ - FİNAL PROJESİ
# Heart Disease Risk Factors and Patient Health Survey
# Veri Seti: Heart_Dataset_Cleaned.csv
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report,
                              roc_auc_score, roc_curve)
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# BÖLÜM A: VERİ SETİNİ TANIMA
# ─────────────────────────────────────────────────────────────────────────────

df = pd.read_csv('Heart_Dataset_Cleaned.csv', encoding='utf-8-sig')

print("=" * 60)
print("BÖLÜM A — VERİ SETİNİ TANIMA")
print("=" * 60)
print(f"\nSatır sayısı  : {df.shape[0]}")
print(f"Sütun sayısı  : {df.shape[1]}")
print(f"\nDeğişken isimleri:\n{df.columns.tolist()}")
print(f"\nEksik değer sayısı:\n{df.isnull().sum()}")
print(f"\nTekrarlayan kayıt: {df.duplicated().sum()}")

# Değişken türleri
categorical_cols = df.select_dtypes(include='object').columns.tolist()
numerical_cols   = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
print(f"\nKategorik değişkenler ({len(categorical_cols)}): {categorical_cols}")
print(f"Sayısal değişkenler   ({len(numerical_cols)}): {numerical_cols}")
print(f"\nHedef değişken: Heart patient")

# ─────────────────────────────────────────────────────────────────────────────
# BÖLÜM B: TANIMLAYICI İSTATİSTİKLER
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("BÖLÜM B — TANIMLAYICI İSTATİSTİKLER")
print("=" * 60)

print("\n--- Sayısal Değişkenler ---")
print(df[['Height (cm)', 'Weight', 'BMI']].describe().round(2))

print("\n--- Hedef Değişken Dağılımı ---")
hp = df['Heart patient'].value_counts()
print(hp)
print(f"Kalp hastası oranı: %{hp['Yes'] / len(df) * 100:.2f}")

print("\n--- Cinsiyet Dağılımı ---")
print(df['Gender'].value_counts())

print("\n--- Yaş Grubu Dağılımı ---")
print(df['Age'].value_counts())

print("\n--- Kan Basıncı Kategorileri ---")
print(df['Blood Pressure'].value_counts())

print("\n--- Tüm Kategorik Değişkenler (Frekans / Yüzde) ---")
for col in categorical_cols:
    freq = df[col].value_counts()
    pct  = df[col].value_counts(normalize=True).mul(100).round(2)
    print(f"\n{col}:\n{pd.DataFrame({'Frekans': freq, 'Yüzde (%)': pct})}")

print(f"\nEn yaygın yaş grubu    : {df['Age'].mode()[0]}")
print(f"En yaygın kan basıncı  : {df['Blood Pressure'].mode()[0]}")

print("\n--- Aykırı Değer Kontrolü (IQR Yöntemi) ---")
for col in numerical_cols:
    q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_out = ((df[col] < lower) | (df[col] > upper)).sum()
    print(f"{col}: sınır [{lower:.2f}, {upper:.2f}] — aykırı: {n_out}")

# ─────────────────────────────────────────────────────────────────────────────
# BÖLÜM C: GÖRSELLEŞTİRME
# ─────────────────────────────────────────────────────────────────────────────

YES_COLOR = '#e74c3c'
NO_COLOR  = '#3498db'
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.spines.top']   = False
plt.rcParams['axes.spines.right'] = False

# Grafik 1 — Kalp Hastası Dağılımı
fig, ax = plt.subplots(figsize=(7, 5))
counts = df['Heart patient'].value_counts()
ax.bar(counts.index, counts.values, color=[NO_COLOR, YES_COLOR], width=0.5, edgecolor='white')
for i, (idx, val) in enumerate(counts.items()):
    ax.text(i, val + 3, f'{val} (%{val/len(df)*100:.1f})', ha='center', fontsize=11, fontweight='bold')
ax.set_title('Kalp Hastası Dağılımı', fontsize=14, fontweight='bold')
ax.set_xlabel('Kalp Hastası'); ax.set_ylabel('Kişi Sayısı')
ax.set_ylim(0, 260)
plt.tight_layout(); plt.savefig('plot1_heart_distribution.png', dpi=150); plt.close()

# Grafik 2 — Yaş Grubu ve Kalp Hastası
fig, ax = plt.subplots(figsize=(8, 5))
age_order = ['< 35', '35–50', '> 50']
cross = pd.crosstab(df['Age'], df['Heart patient']).reindex(age_order)
pct   = cross.div(cross.sum(axis=1), axis=0) * 100
bottom = np.zeros(len(pct))
for i, col in enumerate(pct.columns):
    bars = ax.bar(pct.index, pct[col], bottom=bottom, color=[NO_COLOR, YES_COLOR][i], label=col, edgecolor='white')
    for bar, val, bot in zip(bars, pct[col], bottom):
        if val > 5:
            ax.text(bar.get_x() + bar.get_width()/2, bot + val/2, f'%{val:.0f}',
                    ha='center', va='center', color='white', fontweight='bold', fontsize=10)
    bottom += pct[col].values
ax.set_title('Yaş Grubuna Göre Kalp Hastası Oranı', fontsize=14, fontweight='bold')
ax.set_xlabel('Yaş Grubu'); ax.set_ylabel('Yüzde (%)'); ax.set_ylim(0, 115)
ax.legend(title='Kalp Hastası')
plt.tight_layout(); plt.savefig('plot2_age_heart.png', dpi=150); plt.close()

# Grafik 3 — Cinsiyet ve Kalp Hastası
fig, ax = plt.subplots(figsize=(7, 5))
gc = pd.crosstab(df['Gender'], df['Heart patient'])
gp = gc.div(gc.sum(axis=1), axis=0) * 100
bottom = np.zeros(len(gp))
for i, col in enumerate(gp.columns):
    bars = ax.bar(gp.index, gp[col], bottom=bottom, color=[NO_COLOR, YES_COLOR][i], label=col, edgecolor='white', width=0.5)
    for bar, val, bot in zip(bars, gp[col], bottom):
        if val > 5:
            ax.text(bar.get_x() + bar.get_width()/2, bot + val/2, f'%{val:.0f}',
                    ha='center', va='center', color='white', fontweight='bold')
    bottom += gp[col].values
ax.set_title('Cinsiyete Göre Kalp Hastası Oranı', fontsize=14, fontweight='bold')
ax.set_xlabel('Cinsiyet'); ax.set_ylabel('Yüzde (%)'); ax.set_ylim(0, 115)
ax.legend(title='Kalp Hastası')
plt.tight_layout(); plt.savefig('plot3_gender_heart.png', dpi=150); plt.close()

# Grafik 4 — BMI Boxplot
fig, ax = plt.subplots(figsize=(8, 5))
bp = ax.boxplot([df[df['Heart patient']=='No']['BMI'], df[df['Heart patient']=='Yes']['BMI']],
                patch_artist=True, medianprops=dict(color='white', linewidth=2.5))
ax.set_xticks([1, 2])
ax.set_xticklabels(['Kalp Hastası Değil', 'Kalp Hastası'])
bp['boxes'][0].set_facecolor(NO_COLOR); bp['boxes'][1].set_facecolor(YES_COLOR)
ax.set_title('Kalp Hastası Durumuna Göre BMI Dağılımı', fontsize=14, fontweight='bold')
ax.set_ylabel('BMI (kg/m²)')
plt.tight_layout(); plt.savefig('plot4_bmi_boxplot.png', dpi=150); plt.close()

# Grafik 5 — Kan Basıncı ve Kalp Hastası
fig, ax = plt.subplots(figsize=(9, 5))
bp_order = ['Hypotension', 'Normal', 'Pre-hypertension', 'Hypertension']
bpc = pd.crosstab(df['Blood Pressure'], df['Heart patient']).reindex(bp_order)
bpp = bpc.div(bpc.sum(axis=1), axis=0) * 100
bottom = np.zeros(len(bpp))
for i, col in enumerate(bpp.columns):
    bars = ax.bar(bpp.index, bpp[col], bottom=bottom, color=[NO_COLOR, YES_COLOR][i], label=col, edgecolor='white')
    for bar, val, bot in zip(bars, bpp[col], bottom):
        if val > 5:
            ax.text(bar.get_x() + bar.get_width()/2, bot + val/2, f'%{val:.0f}',
                    ha='center', va='center', color='white', fontweight='bold', fontsize=10)
    bottom += bpp[col].values
ax.set_title('Kan Basıncı Kategorisine Göre Kalp Hastası Oranı', fontsize=14, fontweight='bold')
ax.set_xlabel('Kan Basıncı'); ax.set_ylabel('Yüzde (%)'); ax.set_ylim(0, 115)
ax.legend(title='Kalp Hastası')
plt.tight_layout(); plt.savefig('plot5_bp_heart.png', dpi=150); plt.close()

# Grafik 6 — Fiziksel Aktivite ve Kalp Hastası
fig, ax = plt.subplots(figsize=(10, 5))
pa_order  = ['Rarely / Never', 'Less than 2 days per week', '2–4 days per week', '5 or more days per week']
pa_labels = ['Hiç/Nadiren', 'Haftada <2 Gün', 'Haftada 2–4 Gün', 'Haftada 5+ Gün']
pac = pd.crosstab(df['Physical activity'], df['Heart patient']).reindex(pa_order)
pap = pac.div(pac.sum(axis=1), axis=0) * 100
bottom = np.zeros(len(pap))
for i, col in enumerate(pap.columns):
    bars = ax.bar(pa_labels, pap[col], bottom=bottom, color=[NO_COLOR, YES_COLOR][i], label=col, edgecolor='white')
    for bar, val, bot in zip(bars, pap[col], bottom):
        if val > 5:
            ax.text(bar.get_x() + bar.get_width()/2, bot + val/2, f'%{val:.0f}',
                    ha='center', va='center', color='white', fontweight='bold', fontsize=10)
    bottom += pap[col].values
ax.set_title('Fiziksel Aktivite Düzeyine Göre Kalp Hastası Oranı', fontsize=14, fontweight='bold')
ax.set_xlabel('Fiziksel Aktivite'); ax.set_ylabel('Yüzde (%)'); ax.set_ylim(0, 115)
ax.legend(title='Kalp Hastası')
plt.tight_layout(); plt.savefig('plot6_activity_heart.png', dpi=150); plt.close()

# Grafik 7 — Sigara Kullanımı ve Kalp Hastası
fig, ax = plt.subplots(figsize=(8, 5))
smoke_order = ['Never', 'Occasionally', 'Regularly']
sc = pd.crosstab(df['Smoke or Tobacco'], df['Heart patient']).reindex(smoke_order)
sp = sc.div(sc.sum(axis=1), axis=0) * 100
bottom = np.zeros(len(sp))
for i, col in enumerate(sp.columns):
    bars = ax.bar(sp.index, sp[col], bottom=bottom, color=[NO_COLOR, YES_COLOR][i], label=col, edgecolor='white')
    for bar, val, bot in zip(bars, sp[col], bottom):
        if val > 5:
            ax.text(bar.get_x() + bar.get_width()/2, bot + val/2, f'%{val:.0f}',
                    ha='center', va='center', color='white', fontweight='bold', fontsize=10)
    bottom += sp[col].values
ax.set_title('Sigara Kullanımına Göre Kalp Hastası Oranı', fontsize=14, fontweight='bold')
ax.set_xlabel('Sigara / Tütün Kullanımı'); ax.set_ylabel('Yüzde (%)'); ax.set_ylim(0, 115)
ax.legend(title='Kalp Hastası')
plt.tight_layout(); plt.savefig('plot8_smoke_heart.png', dpi=150); plt.close()

# Grafik 8 — Diyabet ve Kalp Hastası
fig, ax = plt.subplots(figsize=(8, 5))
diab_order = ['Normal', 'Prediabetes', 'Diabetes']
dc = pd.crosstab(df['Diabetes'], df['Heart patient']).reindex(diab_order)
dp = dc.div(dc.sum(axis=1), axis=0) * 100
bottom = np.zeros(len(dp))
for i, col in enumerate(dp.columns):
    bars = ax.bar(dp.index, dp[col], bottom=bottom, color=[NO_COLOR, YES_COLOR][i], label=col, edgecolor='white')
    for bar, val, bot in zip(bars, dp[col], bottom):
        if val > 5:
            ax.text(bar.get_x() + bar.get_width()/2, bot + val/2, f'%{val:.0f}',
                    ha='center', va='center', color='white', fontweight='bold', fontsize=10)
    bottom += dp[col].values
ax.set_title('Diyabet Durumuna Göre Kalp Hastası Oranı', fontsize=14, fontweight='bold')
ax.set_xlabel('Diyabet'); ax.set_ylabel('Yüzde (%)'); ax.set_ylim(0, 115)
ax.legend(title='Kalp Hastası')
plt.tight_layout(); plt.savefig('plot9_diabetes_heart.png', dpi=150); plt.close()

# Grafik 9 — BMI Histogramı
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(df[df['Heart patient']=='No']['BMI'], bins=20, color=NO_COLOR, alpha=0.7, edgecolor='white', label='Kalp Hastası Değil')
ax.hist(df[df['Heart patient']=='Yes']['BMI'], bins=20, color=YES_COLOR, alpha=0.7, edgecolor='white', label='Kalp Hastası')
ax.axvline(25, color='orange', linestyle='--', linewidth=1, label='Fazla kilo sınırı (25)')
ax.axvline(30, color='purple', linestyle='--', linewidth=1, label='Obezite sınırı (30)')
ax.set_title('BMI Dağılımı (Kalp Hastası Durumuna Göre)', fontsize=14, fontweight='bold')
ax.set_xlabel('BMI (kg/m²)'); ax.set_ylabel('Kişi Sayısı')
ax.legend()
plt.tight_layout(); plt.savefig('plot10_bmi_histogram.png', dpi=150); plt.close()

# Grafik 10 — Korelasyon Matrisi
df_enc = df.copy()
le = LabelEncoder()
for col in df_enc.select_dtypes(include='object').columns:
    df_enc[col] = le.fit_transform(df_enc[col])

fig, ax = plt.subplots(figsize=(14, 10))
corr = df_enc.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            square=True, linewidths=0.5, ax=ax, annot_kws={'size': 7})
ax.set_title('Değişkenler Arası Korelasyon Matrisi', fontsize=14, fontweight='bold')
plt.tight_layout(); plt.savefig('plot7_correlation.png', dpi=150); plt.close()

print("\nTüm grafikler kaydedildi.")

# ─────────────────────────────────────────────────────────────────────────────
# BÖLÜM D: RİSK FAKTÖRLERİ ANALİZİ
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("BÖLÜM D — RİSK FAKTÖRLERİ ANALİZİ")
print("=" * 60)

risk_factors = [
    'Age', 'Gender', 'BMI', 'Blood Pressure', 'Diabetes',
    'Family Heart problems History', 'Physical activity', 'Food habits',
    'Sleep at night', 'Depression', 'Smoke or Tobacco', 'Alcohol',
    'Feel stressed', 'Medicines'
]

for factor in risk_factors:
    if factor in numerical_cols:
        g = df.groupby('Heart patient')[factor]
        print(f"\n[{factor}] — Evet ort: {g.get_group('Yes').mean():.2f} | Hayır ort: {g.get_group('No').mean():.2f}")
    else:
        ct = pd.crosstab(df[factor], df['Heart patient'], normalize='index') * 100
        print(f"\n[{factor}] (% satır içi):\n{ct.round(1)}")

# ─────────────────────────────────────────────────────────────────────────────
# BÖLÜM E: MAKİNE ÖĞRENMESİ MODELİ
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("BÖLÜM E — MAKİNE ÖĞRENMESİ MODELİ")
print("=" * 60)

X = df_enc.drop('Heart patient', axis=1)
y = df_enc['Heart patient']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nEğitim seti: {len(X_train)} örnek | Test seti: {len(X_test)} örnek")

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree'      : DecisionTreeClassifier(max_depth=5, random_state=42),
    'Random Forest'      : RandomForestClassifier(n_estimators=100, random_state=42),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    results[name] = {
        'model'    : model,
        'y_pred'   : y_pred,
        'y_proba'  : y_proba,
        'accuracy' : accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall'   : recall_score(y_test, y_pred),
        'f1'       : f1_score(y_test, y_pred),
        'roc_auc'  : roc_auc_score(y_test, y_proba),
        'cm'       : confusion_matrix(y_test, y_pred),
    }
    print(f"\n{name}")
    print("-" * 40)
    print(classification_report(y_test, y_pred, target_names=['No', 'Yes']))
    print(f"ROC-AUC : {results[name]['roc_auc']:.4f}")

# Confusion Matrix Grafikleri
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, (name, res) in zip(axes, results.items()):
    cm = res['cm']
    ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.set_title(name, fontsize=12, fontweight='bold')
    ax.set_xlabel('Tahmin Edilen'); ax.set_ylabel('Gerçek')
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['No', 'Yes']); ax.set_yticklabels(['No', 'Yes'])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=16,
                    fontweight='bold', color='white' if cm[i, j] > cm.max()/2 else 'black')
plt.suptitle('Karmaşıklık Matrisleri', fontsize=14, fontweight='bold')
plt.tight_layout(); plt.savefig('plot_confusion_matrices.png', dpi=150); plt.close()

# ROC Eğrileri
fig, ax = plt.subplots(figsize=(8, 6))
colors_roc = ['#3498db', '#e67e22', '#27ae60']
for (name, res), color in zip(results.items(), colors_roc):
    fpr, tpr, _ = roc_curve(y_test, res['y_proba'])
    ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={res['roc_auc']:.3f})")
ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Rastgele Tahmin')
ax.set_xlabel('Yanlış Pozitif Oranı'); ax.set_ylabel('Doğru Pozitif Oranı')
ax.set_title('ROC Eğrileri — Model Karşılaştırması', fontsize=14, fontweight='bold')
ax.legend(loc='lower right')
plt.tight_layout(); plt.savefig('plot_roc_curves.png', dpi=150); plt.close()

# Değişken Önem Dereceleri (Random Forest)
rf = results['Random Forest']['model']
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(9, 8))
colors_fi = ['#e74c3c' if v >= importances.quantile(0.75) else '#3498db' for v in importances]
ax.barh(importances.index, importances.values, color=colors_fi, edgecolor='white')
ax.set_title('Random Forest — Değişken Önem Dereceleri', fontsize=14, fontweight='bold')
ax.set_xlabel('Önem Derecesi')
plt.tight_layout(); plt.savefig('plot_feature_importance.png', dpi=150); plt.close()

print("\nEn önemli 5 değişken (Random Forest):")
print(importances.tail(5)[::-1])

# Grafik 14 — Model Performans Karşılaştırması
fig, ax = plt.subplots(figsize=(12, 6))
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
x_idx = np.arange(len(metrics_names))
width = 0.25
for i, (name, color) in enumerate(zip(results.keys(), ['#3498db', '#e67e22', '#27ae60'])):
    scores = [results[name][k] for k in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']]
    bars = ax.bar(x_idx + i * width, scores, width, color=color, edgecolor='white', label=name)
    for bar, val in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.01, f'{val:.2f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.set_title('Model Performans Karşılaştırması', fontsize=14, fontweight='bold')
ax.set_ylabel('Skor'); ax.set_xticks(x_idx + width); ax.set_xticklabels(metrics_names)
ax.set_ylim(0, 1.15); ax.legend(loc='upper right')
plt.tight_layout(); plt.savefig('plot14_model_comparison.png', dpi=150); plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# BÖLÜM F: MODEL YORUMLAMA (Özet)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("BÖLÜM F — MODEL YORUMLAMA")
print("=" * 60)
best_acc  = max(results, key=lambda k: results[k]['accuracy'])
best_auc  = max(results, key=lambda k: results[k]['roc_auc'])
print(f"\nEn yüksek Accuracy : {best_acc} ({results[best_acc]['accuracy']:.4f})")
print(f"En yüksek ROC-AUC   : {best_auc} ({results[best_auc]['roc_auc']:.4f})")
print("Recall değerleri düşük — sınıf dengesizliği nedeniyle Accuracy tek başına yeterli değildir.")

print("\n" + "=" * 60)
print("Analiz tamamlandı. Tüm grafikler ve sonuçlar kaydedildi.")
print("=" * 60)
