```python
from google.colab import drive
drive.mount('/content/gdrive')
```

    Drive already mounted at /content/gdrive; to attempt to forcibly remount, call drive.mount("/content/gdrive", force_remount=True).
    

# HOME CREDIT INDONESIA SCORECARD MODEL
## Problem Statement:

Home Credit saat ini sedang menggunakan berbagai macam metode statistik dan Machine Learning untuk membuat prediksi skor kredit. Sekarang, kami meminta anda untuk membuka potensi maksimal dari data kami. **Dengan melakukannya, kita dapat memastikan pelanggan yang mampu melakukan pelunasan tidak ditolak ketika melakukan pengajuan pinjaman, dan pinjaman datap diberikan dengan principal, maturity, dan repayment calendar yang akan memotivsi pelanggan untuk sukses.** Evaluasi akan dilakukan dengan mengecek seberapa dalam pemahaman analisa yang anda kerjakan. Sebagai catatan, anda perlu menggunakan **setidaknya 2 model Machine Learning dimana salah satunya adalah Logistic Regression**. Setelah itu, buatlah slide presentasi yang mengandung analisa hasil pemodelan secara end-to-end beserta rekomendasi bisnisnya (maksimal 10 halaman)

Point utama:
1. Predict probability variable TARGET (sesuai contoh submission)
2. Pakai 2 model, salah satunya Logistic Regression
3. Cari top 2 insight (relate satu atau lebih variable dengan target atau yang lainnya) dan solusi
4. PPT

## Import Library dan Load Data
Pada tahap ini akan dilakukan import beberapa library Python yang diperlukan serta melakukan pembacaan data.

### Update Library


```python
!pip install --upgrade scikit-learn
```

    Requirement already satisfied: scikit-learn in /usr/local/lib/python3.10/dist-packages (1.4.0)
    Requirement already satisfied: numpy<2.0,>=1.19.5 in /usr/local/lib/python3.10/dist-packages (from scikit-learn) (1.23.5)
    Requirement already satisfied: scipy>=1.6.0 in /usr/local/lib/python3.10/dist-packages (from scikit-learn) (1.11.4)
    Requirement already satisfied: joblib>=1.2.0 in /usr/local/lib/python3.10/dist-packages (from scikit-learn) (1.3.2)
    Requirement already satisfied: threadpoolctl>=2.0.0 in /usr/local/lib/python3.10/dist-packages (from scikit-learn) (3.2.0)
    

### Import Library


```python
import os
import sys

import numpy as np
import pandas as pd

import matplotlib.pylab as plt
from matplotlib_venn import venn2
import seaborn as sns

import gc

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')
```

### Load Data
Terdapat 9 data yang diberikan dengan data utama berupa data training dan data testing. Untuk sementara, analisis awal hanya akan menggunakan kedua data tersebut untuk membentuk model baseline yang dapat dikembangkan lebih jauh selanjutnya.

Berikut adalah ukuran data training dan data testing.


```python
%%time
train_df = pd.read_csv('/content/gdrive/MyDrive/RAKAMIN/application_train.csv')
test_df = pd.read_csv('/content/gdrive/MyDrive/RAKAMIN/application_test.csv')

print(f'Training data shape: {train_df.shape}')
print(f'Testing data shape: {test_df.shape}')
```

    Training data shape: (307511, 122)
    Testing data shape: (48744, 121)
    CPU times: user 6.28 s, sys: 2.17 s, total: 8.45 s
    Wall time: 10.4 s
    

Data training berjumlah 307511 dengan jumlah features sebanyak 122 sementara data test berjumlah 48733 dengan 121 fitur (tidak ada TARGET pada data test).



```python
# Training dataset
train_df.head()
```





  <div id="df-c5e753a4-fc32-435d-91ff-6e5d98ac83ae" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SK_ID_CURR</th>
      <th>TARGET</th>
      <th>NAME_CONTRACT_TYPE</th>
      <th>CODE_GENDER</th>
      <th>FLAG_OWN_CAR</th>
      <th>FLAG_OWN_REALTY</th>
      <th>CNT_CHILDREN</th>
      <th>AMT_INCOME_TOTAL</th>
      <th>AMT_CREDIT</th>
      <th>AMT_ANNUITY</th>
      <th>...</th>
      <th>FLAG_DOCUMENT_18</th>
      <th>FLAG_DOCUMENT_19</th>
      <th>FLAG_DOCUMENT_20</th>
      <th>FLAG_DOCUMENT_21</th>
      <th>AMT_REQ_CREDIT_BUREAU_HOUR</th>
      <th>AMT_REQ_CREDIT_BUREAU_DAY</th>
      <th>AMT_REQ_CREDIT_BUREAU_WEEK</th>
      <th>AMT_REQ_CREDIT_BUREAU_MON</th>
      <th>AMT_REQ_CREDIT_BUREAU_QRT</th>
      <th>AMT_REQ_CREDIT_BUREAU_YEAR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100002</td>
      <td>1</td>
      <td>Cash loans</td>
      <td>M</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>202500.0</td>
      <td>406597.5</td>
      <td>24700.5</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100003</td>
      <td>0</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>N</td>
      <td>0</td>
      <td>270000.0</td>
      <td>1293502.5</td>
      <td>35698.5</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>100004</td>
      <td>0</td>
      <td>Revolving loans</td>
      <td>M</td>
      <td>Y</td>
      <td>Y</td>
      <td>0</td>
      <td>67500.0</td>
      <td>135000.0</td>
      <td>6750.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>100006</td>
      <td>0</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>135000.0</td>
      <td>312682.5</td>
      <td>29686.5</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>100007</td>
      <td>0</td>
      <td>Cash loans</td>
      <td>M</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>121500.0</td>
      <td>513000.0</td>
      <td>21865.5</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 122 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-c5e753a4-fc32-435d-91ff-6e5d98ac83ae')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-c5e753a4-fc32-435d-91ff-6e5d98ac83ae button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-c5e753a4-fc32-435d-91ff-6e5d98ac83ae');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-3ab6c296-5491-467f-b6a4-ae3cf8a6f855">
  <button class="colab-df-quickchart" onclick="quickchart('df-3ab6c296-5491-467f-b6a4-ae3cf8a6f855')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-3ab6c296-5491-467f-b6a4-ae3cf8a6f855 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>
    </div>
  </div>





```python
# Testing dataset
test_df.head()
```





  <div id="df-c9ac5ff1-1b08-4e10-961c-27fdfe7721c3" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SK_ID_CURR</th>
      <th>NAME_CONTRACT_TYPE</th>
      <th>CODE_GENDER</th>
      <th>FLAG_OWN_CAR</th>
      <th>FLAG_OWN_REALTY</th>
      <th>CNT_CHILDREN</th>
      <th>AMT_INCOME_TOTAL</th>
      <th>AMT_CREDIT</th>
      <th>AMT_ANNUITY</th>
      <th>AMT_GOODS_PRICE</th>
      <th>...</th>
      <th>FLAG_DOCUMENT_18</th>
      <th>FLAG_DOCUMENT_19</th>
      <th>FLAG_DOCUMENT_20</th>
      <th>FLAG_DOCUMENT_21</th>
      <th>AMT_REQ_CREDIT_BUREAU_HOUR</th>
      <th>AMT_REQ_CREDIT_BUREAU_DAY</th>
      <th>AMT_REQ_CREDIT_BUREAU_WEEK</th>
      <th>AMT_REQ_CREDIT_BUREAU_MON</th>
      <th>AMT_REQ_CREDIT_BUREAU_QRT</th>
      <th>AMT_REQ_CREDIT_BUREAU_YEAR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100001</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>135000.0</td>
      <td>568800.0</td>
      <td>20560.5</td>
      <td>450000.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100005</td>
      <td>Cash loans</td>
      <td>M</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>99000.0</td>
      <td>222768.0</td>
      <td>17370.0</td>
      <td>180000.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>100013</td>
      <td>Cash loans</td>
      <td>M</td>
      <td>Y</td>
      <td>Y</td>
      <td>0</td>
      <td>202500.0</td>
      <td>663264.0</td>
      <td>69777.0</td>
      <td>630000.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>100028</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>2</td>
      <td>315000.0</td>
      <td>1575000.0</td>
      <td>49018.5</td>
      <td>1575000.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>100038</td>
      <td>Cash loans</td>
      <td>M</td>
      <td>Y</td>
      <td>N</td>
      <td>1</td>
      <td>180000.0</td>
      <td>625500.0</td>
      <td>32067.0</td>
      <td>625500.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 121 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-c9ac5ff1-1b08-4e10-961c-27fdfe7721c3')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-c9ac5ff1-1b08-4e10-961c-27fdfe7721c3 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-c9ac5ff1-1b08-4e10-961c-27fdfe7721c3');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-21951741-3533-496c-aaee-07e26bdda746">
  <button class="colab-df-quickchart" onclick="quickchart('df-21951741-3533-496c-aaee-07e26bdda746')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-21951741-3533-496c-aaee-07e26bdda746 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>
    </div>
  </div>




### Keputusan Awal

Terlihat bahwa pada kedua dataset terdapat kolom gender. Untuk menghindari bias, kolom ini akan dihapus.


```python
train_df = train_df.drop(columns = 'CODE_GENDER')
test_df = test_df.drop(columns = 'CODE_GENDER')
```

## Exploratory Data Analysis

### Target Distribution (Optional)
Hal pertama yang perlu dilihat adalah distribusi dari kolom target dimana:
* TARGET 0 -> Pinjaman dibayar tepat waktu
* TARGET 1 -> Menandakan bahwa client mengalami kesulitan untuk membayar


```python
fig, ax = plt.subplots(figsize=(10,5))
sns.countplot(x=train_df['TARGET'])
plt.show()
```


    
![png](README_files/README_18_0.png)
    


Data tidak seimbang, lebih banyak TARGET 0 dibanding 1

Selanjutnya, kita lihat apakah data training dan test yang diberikan memiliki observasi yang overlap (terdapat data yang berada di kedua dataset sekaligus) atau tidak


```python
fig, ax = plt.subplots(figsize=(10,5))
set1 = set(train_df.SK_ID_CURR.unique())
set2 = set(test_df.SK_ID_CURR.unique())

venn2([set1, set2], ('train', 'test'))
plt.show()
```


    
![png](README_files/README_21_0.png)
    


Terlihat pada gambar di atas bahwa tidak ada data yang overlap

### Menangani Missing Values (Optional)
Selanjutnya adalah terkait penanganan missing values. Optional: Tidak ditangani di sini, hanya melihat adanya missing values

Berikut tampilan beberapa data pertama dari dataset training dan dataset testing.


```python
# Function to calculate missing values by column
def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()

        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)

        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total'})

        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[mis_val_table_ren_columns.iloc[:, 1] != 0]
        mis_val_table_ren_columns = mis_val_table_ren_columns.sort_values('% of Total', ascending=False).round(1)

        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")

        # Return the dataframe with missing information
        return mis_val_table_ren_columns
```


```python
# Missing values statistics
missing_values = missing_values_table(train_df)
missing_values.head(10)

del missing_values
gc.collect()
```

    Your selected dataframe has 121 columns.
    There are 67 columns that have missing values.
    




    137



Terlihat bahwa terdapat 67 kolom pada data training. Keberadaan missing value tersebut akan mempengaruhi model yang akan dibangun nantinya. Terdapat beberapa cara yang dapat digunakan untuk mengataasi hal tersebut seperti mengisi data yang kosong tersebut atau pun menghapus kolom yang memiliki persentasi missing value yang besar. Namun, karena saat ini belum diketahui apakah kolom-kolom tersebut memiliki peran dalam memprediksi target, untuk saat ini data tidak akan diubah terlebih dahulu.

### Tipe Kolom (Optional)
Berikut adalah tipe kolom pada dataset training. Optional: Hanya sebagai insight dan tidak berpengaruh ke step selanjutnya.


```python
train_df.dtypes.value_counts()
```




    float64    65
    int64      41
    object     15
    dtype: int64




```python
train_df.dtypes.head(10)
```




    SK_ID_CURR              int64
    TARGET                  int64
    NAME_CONTRACT_TYPE     object
    FLAG_OWN_CAR           object
    FLAG_OWN_REALTY        object
    CNT_CHILDREN            int64
    AMT_INCOME_TOTAL      float64
    AMT_CREDIT            float64
    AMT_ANNUITY           float64
    AMT_GOODS_PRICE       float64
    dtype: object



Terdapat 65 kolom dengan tipe data `float64` dan 41 kolom dengantipe data `int64` yang keduanya merupakan data numerik. Lalu, terdapat juga 16 kolom dengan tipe `object` yang berisikan string dan merupakan data categorical seperti kode gender/jenis kelamin, nama kontrak, dsb.

Berikut adalah jumlah dari kategori pada masing-masing kolom yang bersifat categorical.


```python
train_df.select_dtypes('object').apply(pd.Series.nunique, axis = 0)
```




    NAME_CONTRACT_TYPE             2
    FLAG_OWN_CAR                   2
    FLAG_OWN_REALTY                2
    NAME_TYPE_SUITE                7
    NAME_INCOME_TYPE               8
    NAME_EDUCATION_TYPE            5
    NAME_FAMILY_STATUS             6
    NAME_HOUSING_TYPE              6
    OCCUPATION_TYPE               18
    WEEKDAY_APPR_PROCESS_START     7
    ORGANIZATION_TYPE             58
    FONDKAPREMONT_MODE             4
    HOUSETYPE_MODE                 3
    WALLSMATERIAL_MODE             7
    EMERGENCYSTATE_MODE            2
    dtype: int64




```python
train_df.select_dtypes('object').apply(lambda x: x.unique(), axis=0).head(5)
```




    NAME_CONTRACT_TYPE                        [Cash loans, Revolving loans]
    FLAG_OWN_CAR                                                     [N, Y]
    FLAG_OWN_REALTY                                                  [Y, N]
    NAME_TYPE_SUITE       [Unaccompanied, Family, Spouse, partner, Child...
    NAME_INCOME_TYPE      [Working, State servant, Commercial associate,...
    dtype: object



### Encoding
Salah satu model yang diminta untuk dibuat pada problem statement adalah **logistic regression**. Model tersebut tidak bisa menangani variabel yang bersifat categorical. Oleh karena itu, diperlukan proses encoding pada variabel-variabel tersebut sehingga dapat direpresentasikan dengan angka.

Pada proses ini, data kategorikal yang hanya memiliki 2 unique values (atau kurang) akan melalui proses label encoding (menggunakan `LabelEncoder` dari Scikit-learn) sedangkan yang lainnya akan melalui proses one-hot encoding (menggunakan metode `get_dummies(df)` dari pandas).


```python
from sklearn import preprocessing
```


```python
# instansiasi objek label encoder
lab_en = preprocessing.LabelEncoder()
lab_en_count = 0

for col in train_df:
  if train_df[col].dtype == 'object':
    # If 2 or fewer unique categories
    if len(list(train_df[col].unique())) <= 2:
      print(f'label encoding: {col}')
      # Train on the training data
      lab_en.fit(train_df[col])
      # Transform
      train_df[col] = lab_en.transform(train_df[col])
      test_df[col] = lab_en.transform(test_df[col])

      lab_en_count += 1

print('%d columns were label encoded.' % lab_en_count)
```

    label encoding: NAME_CONTRACT_TYPE
    label encoding: FLAG_OWN_CAR
    label encoding: FLAG_OWN_REALTY
    3 columns were label encoded.
    


```python
# one-hot encoding pada kolom categorical lainnya
train_df = pd.get_dummies(train_df)
test_df = pd.get_dummies(test_df)

print('Training shape: ', train_df.shape)
print('Testing shape: ', test_df.shape)
```

    Training shape:  (307511, 240)
    Testing shape:  (48744, 237)
    

Terlihat bahwa pada setelah proses di atas, ukuran data training dan data testing berubah. Hal tersebut diakibatkan oleh proses one-hot encoding yang menghasilkan kolom-kolom baru. Terdapat ketidaksesuaian antara kolom yang dihasilkan pada data training dan data testing dimana seharusnya data training hanya memiliki tambahan satu kolom saja (kolom TARGET). Perlu dilakukan alignment antara data testing dan data training.


```python
# Simpan kolom TARGET dari data training terlebih dahulu
train_labels = train_df['TARGET']

# Align the training and testing data, keep only columns present in both dataframes
train_df, test_df = train_df.align(test_df, join = 'inner', axis = 1)
print('Training shape before: ', train_df.shape)
print('Testing shape before: ', test_df.shape)

# Kembalikan kolom target pada data training
train_df['TARGET'] = train_labels

print('Training shape after: ', train_df.shape)
print('Testing shape after: ', test_df.shape)
```

    Training shape before:  (307511, 237)
    Testing shape before:  (48744, 237)
    Training shape after:  (307511, 238)
    Testing shape after:  (48744, 237)
    

### Outlier

Pertama, kita lihat nilai variance dari masing-masing kolom


```python
from tqdm import tqdm # feedback
```


```python
train_df.var().sort_values().tail(20)
```




    REGION_RATING_CLIENT          2.591155e-01
    CNT_CHILDREN                  5.214593e-01
    AMT_REQ_CREDIT_BUREAU_QRT     6.305244e-01
    CNT_FAM_MEMBERS               8.293409e-01
    AMT_REQ_CREDIT_BUREAU_MON     8.390604e-01
    AMT_REQ_CREDIT_BUREAU_YEAR    3.494264e+00
    OBS_60_CNT_SOCIAL_CIRCLE      5.663464e+00
    OBS_30_CNT_SOCIAL_CIRCLE      5.764747e+00
    HOUR_APPR_PROCESS_START       1.066566e+01
    OWN_CAR_AGE                   1.426785e+02
    DAYS_LAST_PHONE_CHANGE        6.836123e+05
    DAYS_ID_PUBLISH               2.278441e+06
    DAYS_REGISTRATION             1.241073e+07
    DAYS_BIRTH                    1.904440e+07
    AMT_ANNUITY                   2.100684e+08
    SK_ID_CURR                    1.056582e+10
    DAYS_EMPLOYED                 1.995884e+10
    AMT_INCOME_TOTAL              5.622739e+10
    AMT_GOODS_PRICE               1.364907e+11
    AMT_CREDIT                    1.619988e+11
    dtype: float64



Terlihat bahwa terdapat beberapa kolom dengan nilai variance yang sangat tinggi.


```python
# Ambil 10 kolom dengan variance tertinggi
highest_var_cols = ['AMT_CREDIT', 'AMT_GOODS_PRICE', 'AMT_INCOME_TOTAL', 'DAYS_EMPLOYED', 'AMT_ANNUITY',
                   'DAYS_BIRTH', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH', 'DAYS_LAST_PHONE_CHANGE', 'OWN_CAR_AGE']

# Plot pie charts for integer columns and distribution plots for float columns
for col in tqdm(highest_var_cols):
  train_df[col].plot.hist(title = col)
  plt.xlabel(col)
  plt.show()
del highest_var_cols
gc.collect()
```

      0%|          | 0/10 [00:00<?, ?it/s]


    
![png](README_files/README_45_1.png)
    


     10%|█         | 1/10 [00:00<00:07,  1.19it/s]


    
![png](README_files/README_45_3.png)
    


     20%|██        | 2/10 [00:01<00:05,  1.47it/s]


    
![png](README_files/README_45_5.png)
    


     30%|███       | 3/10 [00:02<00:04,  1.53it/s]


    
![png](README_files/README_45_7.png)
    


     40%|████      | 4/10 [00:02<00:03,  1.51it/s]


    
![png](README_files/README_45_9.png)
    


     50%|█████     | 5/10 [00:03<00:04,  1.24it/s]


    
![png](README_files/README_45_11.png)
    


     60%|██████    | 6/10 [00:05<00:03,  1.04it/s]


    
![png](README_files/README_45_13.png)
    


     70%|███████   | 7/10 [00:06<00:03,  1.03s/it]


    
![png](README_files/README_45_15.png)
    


     80%|████████  | 8/10 [00:07<00:01,  1.04it/s]


    
![png](README_files/README_45_17.png)
    


     90%|█████████ | 9/10 [00:07<00:00,  1.21it/s]


    
![png](README_files/README_45_19.png)
    


    100%|██████████| 10/10 [00:07<00:00,  1.25it/s]
    




    30091



Dapat dilihat bahwa data-data berikut memiliki nilai variance yang besar. Terlihat juga bahwa pada plot distribusi data terdapat ketimpangan. Selanjutnya, akan kita cek kolom DAYS_EMPLOYED.


```python
(train_df['DAYS_EMPLOYED']).describe()
```




    count    307511.000000
    mean      63815.045904
    std      141275.766519
    min      -17912.000000
    25%       -2760.000000
    50%       -1213.000000
    75%        -289.000000
    max      365243.000000
    Name: DAYS_EMPLOYED, dtype: float64




```python
# Plot a histogram of the 'your_column' data
plt.hist(train_df['DAYS_EMPLOYED'], bins=100, color='blue', edgecolor='black')

# Add labels and title
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of your_column')

# Show the plot
plt.show()
```


    
![png](README_files/README_48_0.png)
    


Terlihat bahwa persebaran data tersebut tidak masuk akal. Data atau kolom tersebut menunjukan jumlah total hari kerja dari seorang client dan nilai terbesarnya mencapai 365243 hari (setara kurang lebih 1000 tahun). Selain itu, perhitungan DAYS_EMPLOYED seharusnya bernilai negatif. Kita lihat apakah ada data di antara value 5000 sampai 350000.


```python
# Use boolean indexing to filter values within the specified range
filtered_values = train_df['DAYS_EMPLOYED'][(train_df['DAYS_EMPLOYED'] >= 5000) & (train_df['DAYS_EMPLOYED'] <= 350000)]

# Get the count of values within the specified range
len(filtered_values)
```




    0



Sudah dipastikan bahwa tidak ada data DAYS_EMPLOYED lainnya di antara nilai 5000 - 350000. Karena sudah terlihat bahwa data outlier seluruhnya bernilai 365243, kita hanya perlu mengganti data dengan value tersebut dengan missing value untuk nanti dihandle menggunakan imputer


```python
# Replace the anomalous values with nan
train_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace = True)

train_df["DAYS_EMPLOYED"].plot.hist(title = 'Days Employment Histogram')
plt.xlabel('Days Employment')
```




    Text(0.5, 0, 'Days Employment')




    
![png](README_files/README_52_1.png)
    


Hal yang sama diperlukan juga untuk dilakukan pada dataset testing


```python
(test_df['DAYS_EMPLOYED']).describe()
```




    count     48744.000000
    mean      67485.366322
    std      144348.507136
    min      -17463.000000
    25%       -2910.000000
    50%       -1293.000000
    75%        -296.000000
    max      365243.000000
    Name: DAYS_EMPLOYED, dtype: float64




```python
# Replace the anomalous values with nan
test_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace = True)
```

### Correlation (Optional)
Penghitungan korelasi antara fitur lain dengan target. Optional: Hanya untuk mencari insight. Run ketika ada perubahan dari dataset saja karena hasilnya tidak berubah jika dataset tidak berubah dan juga tidak berpengaruh ke step selanjutnya.


```python
# Find correlations with the target and sort
correlations = train_df.corr()['TARGET'].sort_values()

# Display correlations
print('Most Positive Correlations:\n', correlations.tail(20))
print('\nMost Negative Correlations:\n', correlations.head(20))
```

    Most Positive Correlations:
     OCCUPATION_TYPE_Drivers                              0.030303
    DEF_60_CNT_SOCIAL_CIRCLE                             0.031276
    DEF_30_CNT_SOCIAL_CIRCLE                             0.032248
    LIVE_CITY_NOT_WORK_CITY                              0.032518
    OWN_CAR_AGE                                          0.037612
    DAYS_REGISTRATION                                    0.041975
    OCCUPATION_TYPE_Laborers                             0.043019
    FLAG_DOCUMENT_3                                      0.044346
    REG_CITY_NOT_LIVE_CITY                               0.044395
    FLAG_EMP_PHONE                                       0.045982
    NAME_EDUCATION_TYPE_Secondary / secondary special    0.049824
    REG_CITY_NOT_WORK_CITY                               0.050994
    DAYS_ID_PUBLISH                                      0.051457
    DAYS_LAST_PHONE_CHANGE                               0.055218
    NAME_INCOME_TYPE_Working                             0.057481
    REGION_RATING_CLIENT                                 0.058899
    REGION_RATING_CLIENT_W_CITY                          0.060893
    DAYS_EMPLOYED                                        0.074958
    DAYS_BIRTH                                           0.078239
    TARGET                                               1.000000
    Name: TARGET, dtype: float64
    
    Most Negative Correlations:
     EXT_SOURCE_3                           -0.178919
    EXT_SOURCE_2                           -0.160472
    EXT_SOURCE_1                           -0.155317
    NAME_EDUCATION_TYPE_Higher education   -0.056593
    NAME_INCOME_TYPE_Pensioner             -0.046209
    ORGANIZATION_TYPE_XNA                  -0.045987
    FLOORSMAX_AVG                          -0.044003
    FLOORSMAX_MEDI                         -0.043768
    FLOORSMAX_MODE                         -0.043226
    EMERGENCYSTATE_MODE_No                 -0.042201
    HOUSETYPE_MODE_block of flats          -0.040594
    AMT_GOODS_PRICE                        -0.039645
    REGION_POPULATION_RELATIVE             -0.037227
    ELEVATORS_AVG                          -0.034199
    ELEVATORS_MEDI                         -0.033863
    FLOORSMIN_AVG                          -0.033614
    FLOORSMIN_MEDI                         -0.033394
    WALLSMATERIAL_MODE_Panel               -0.033119
    LIVINGAREA_AVG                         -0.032997
    LIVINGAREA_MEDI                        -0.032739
    Name: TARGET, dtype: float64
    


```python
print('Columns with highest correlation (absolute):\n', abs(correlations).sort_values(ascending=False).head(20))
del correlations
gc.collect()
```

    Columns with highest correlation (absolute):
     TARGET                                               1.000000
    EXT_SOURCE_3                                         0.178919
    EXT_SOURCE_2                                         0.160472
    EXT_SOURCE_1                                         0.155317
    DAYS_BIRTH                                           0.078239
    DAYS_EMPLOYED                                        0.074958
    REGION_RATING_CLIENT_W_CITY                          0.060893
    REGION_RATING_CLIENT                                 0.058899
    NAME_INCOME_TYPE_Working                             0.057481
    NAME_EDUCATION_TYPE_Higher education                 0.056593
    DAYS_LAST_PHONE_CHANGE                               0.055218
    DAYS_ID_PUBLISH                                      0.051457
    REG_CITY_NOT_WORK_CITY                               0.050994
    NAME_EDUCATION_TYPE_Secondary / secondary special    0.049824
    NAME_INCOME_TYPE_Pensioner                           0.046209
    ORGANIZATION_TYPE_XNA                                0.045987
    FLAG_EMP_PHONE                                       0.045982
    REG_CITY_NOT_LIVE_CITY                               0.044395
    FLAG_DOCUMENT_3                                      0.044346
    FLOORSMAX_AVG                                        0.044003
    Name: TARGET, dtype: float64
    




    9528



**Perlu diperhatikan** bahwa beberapa kolom memiliki nilai negatif seperti kolom DAYS_BIRTH dan DAYS_EMPLOYED. Hal tersebut menandakan bahwa korelasi sebenarnya dari DAYS_BIRTH terbalik atau negatif. Terlihat bahwa secara magnitude, dapat diurutkan kolom dengan korelasi terbesar dengan target yaitu kolom:

1. EXT_SOURCE_3
2. EXT_SOURCE_2
3. EXT_SOURCE_1
4. DAYS_BIRTH
5. DAYS_EMPLOYED
6. REGION_RATING_CLIENT_W_CITY
7. REGION_RATING_CLIENT
8.  NAME_INCOME_TYPE
9.   NAME_EDUCATION_TYPE

Beberapa insight:
1. Seseorang dengan EXT_SOURCE (diasumsikan external source income atau pendapatan eksternal) memiliki kecenderungan yang cukup besar untuk membayar
2. Seseorang yang telah pensiun cenderung untuk membayar, begitu juga dengan usia yang lebih tua
3. Pendidikan cukup memiliki korelasi yang besar dimana seseorang dengan status  pendidikan Higher education cenderung untuk membayar sedangkan seseorang dengan status pendidikan Secondary / secondary special cenderung untuk sebaliknya
4.

### Pengecekan ulang outlier pada kolom dengan korelasi yang tinggi (Optional)

Selanjutnya akan dilakukan pengecekan ulang terhadap kolom-kolom tersebut untuk memastikan bahwa tidak ada outlier atau anomali pada data tersebut


```python
def draw_categorical_pie(cols):
    # Identify one-hot-encoded columns that start with 'Category'
    one_hot_columns = [col for col in train_df.columns if col.startswith(cols)]

    # Create a new DataFrame containing only the one-hot-encoded columns
    one_hot_df = train_df[one_hot_columns]

    # Calculate the sum of each one-hot-encoded column
    category_counts = one_hot_df.sum()

    # Plot a pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(category_counts, labels=None, autopct='%1.1f%%', startangle=90)
    plt.title('Distribution of One-Hot-Encoded Categories')

    # Display labels at the bottom using legend
    plt.legend(category_counts.index, title="Categories", loc="lower center", bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=len(category_counts))

    plt.show()

columns = ['EXT_SOURCE_3', 'EXT_SOURCE_2', 'EXT_SOURCE_1',
            'DAYS_BIRTH', 'DAYS_EMPLOYED',
            'REGION_RATING_CLIENT_W_CITY', 'REGION_RATING_CLIENT',
            'NAME_INCOME_TYPE',
            'NAME_EDUCATION_TYPE']

for cols in tqdm(columns):
    if cols == 'NAME_INCOME_TYPE':
        draw_categorical_pie(cols)
    elif cols == 'NAME_EDUCATION_TYPE':
        draw_categorical_pie(cols)
    else:
        train_df[cols].plot.hist(title = cols)
        plt.xlabel(cols)
        plt.show()
```

      0%|          | 0/9 [00:00<?, ?it/s]


    
![png](README_files/README_62_1.png)
    


     11%|█         | 1/9 [00:00<00:01,  4.10it/s]


    
![png](README_files/README_62_3.png)
    


     22%|██▏       | 2/9 [00:00<00:01,  3.85it/s]


    
![png](README_files/README_62_5.png)
    


     33%|███▎      | 3/9 [00:00<00:01,  3.73it/s]


    
![png](README_files/README_62_7.png)
    


     44%|████▍     | 4/9 [00:01<00:01,  3.57it/s]


    
![png](README_files/README_62_9.png)
    


     56%|█████▌    | 5/9 [00:01<00:01,  3.55it/s]


    
![png](README_files/README_62_11.png)
    


     67%|██████▋   | 6/9 [00:01<00:00,  3.54it/s]


    
![png](README_files/README_62_13.png)
    


     78%|███████▊  | 7/9 [00:02<00:00,  3.21it/s]


    
![png](README_files/README_62_15.png)
    


     89%|████████▉ | 8/9 [00:02<00:00,  2.27it/s]


    
![png](README_files/README_62_17.png)
    


    100%|██████████| 9/9 [00:03<00:00,  2.83it/s]
    

Distribusi data terlihat normal. Beberapa insight yang didapat:
1. Urutan jumlah client berdasarkan income type: Working, commercial_associate, Pensioner, state_servant, diikuti yang lainnya
3. Urutan jumlah client berdasarkan status pendidikan: Secondary/secondary special, Higher education, Incomplete higher, Lower secondary, dan terakhir Academic degree


```python
numeric_cols = columns = ['EXT_SOURCE_3',
'EXT_SOURCE_2',
'EXT_SOURCE_1',
'DAYS_BIRTH',
'DAYS_EMPLOYED',
'REGION_RATING_CLIENT_W_CITY',
'REGION_RATING_CLIENT']
missing_values = missing_values_table(train_df[numeric_cols])
missing_values
```

    Your selected dataframe has 7 columns.
    There are 4 columns that have missing values.
    





  <div id="df-1b5f49be-296b-48b5-b86f-4c6c5b2c2afb" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Missing Values</th>
      <th>% of Total</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>EXT_SOURCE_1</th>
      <td>173378</td>
      <td>56.4</td>
    </tr>
    <tr>
      <th>EXT_SOURCE_3</th>
      <td>60965</td>
      <td>19.8</td>
    </tr>
    <tr>
      <th>DAYS_EMPLOYED</th>
      <td>55374</td>
      <td>18.0</td>
    </tr>
    <tr>
      <th>EXT_SOURCE_2</th>
      <td>660</td>
      <td>0.2</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-1b5f49be-296b-48b5-b86f-4c6c5b2c2afb')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-1b5f49be-296b-48b5-b86f-4c6c5b2c2afb button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-1b5f49be-296b-48b5-b86f-4c6c5b2c2afb');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-79b9f89e-ecbd-4c8c-a67d-f4030058c481">
  <button class="colab-df-quickchart" onclick="quickchart('df-79b9f89e-ecbd-4c8c-a67d-f4030058c481')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-79b9f89e-ecbd-4c8c-a67d-f4030058c481 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>
    </div>
  </div>




Kolom EXT_SOURCE_1 memiliki banyak missing values, **ingat untuk menggunakan imputer untuk mengganti missing values**

### Persentase (Optional)


```python
def percent_defaulted(cols_prefix):
    # Filter columns that start with the specified prefix
    columns = [col for col in train_df.columns if col.startswith(cols_prefix)]

    category_counts = train_df.groupby('TARGET')[columns].sum()
    category_1_counts = category_counts.loc[1]
    category_0_counts = category_counts.loc[0]

    percentage_df = (100 * category_1_counts/(category_0_counts+category_1_counts)).sort_values(ascending=False)
    print(f'Percentage of defaulted client per {cols_prefix}:')
    print(percentage_df)
    print(f'Percentage of non-defaulted client per {cols_prefix}:')
    print(100-percentage_df)
```


```python
percent_defaulted('NAME_INCOME_TYPE')
```

    Percentage of defaulted client per NAME_INCOME_TYPE:
    NAME_INCOME_TYPE_Unemployed              36.363636
    NAME_INCOME_TYPE_Working                  9.588472
    NAME_INCOME_TYPE_Commercial associate     7.484257
    NAME_INCOME_TYPE_State servant            5.754965
    NAME_INCOME_TYPE_Pensioner                5.386366
    NAME_INCOME_TYPE_Businessman              0.000000
    NAME_INCOME_TYPE_Student                  0.000000
    dtype: float64
    Percentage of non-defaulted client per NAME_INCOME_TYPE:
    NAME_INCOME_TYPE_Unemployed               63.636364
    NAME_INCOME_TYPE_Working                  90.411528
    NAME_INCOME_TYPE_Commercial associate     92.515743
    NAME_INCOME_TYPE_State servant            94.245035
    NAME_INCOME_TYPE_Pensioner                94.613634
    NAME_INCOME_TYPE_Businessman             100.000000
    NAME_INCOME_TYPE_Student                 100.000000
    dtype: float64
    

Poin menarik:
1. Seluruh bussinessman dan seluruh student membayar, tapi jumlah client tersebut sedikit. **Belum pasti jika client jenis ini lebih ramai semuanya akan membayar juga, tetapi ada baiknya dilakukan campaign dan juga taktik khusus seperti pinjaman pendidikan dengan metode bayar yang berbeda atau pinjaman bisnis dengan metode pembayaran berupa royalti  atau yang lainnya (gaya sharktank, jatuhnya jadi seperti investasi)**
2. Client unemployed paling sulit membayar, jumlah client pun sedikit  **Perlu ada taktik seperti planning supaya client jenis ini lebih sering membayar**
3. Pensioner cenderung untuk membayar dan jumlah client tersebut juga banyak (18% dari total), **ada baiknya dilakukan campaign lebih terhadap tipe ini supaya lebih ramai pensiunan yang menggunakan jasa Home Credit**
4. Sisanya biasa saja dan hanya perlu ditingkatkan untuk state servant dan commercial associate, sementara working class client bisa digunakan taktik yang lebih umum


```python
percent_defaulted('NAME_EDUCATION_TYPE')
```

    Percentage of defaulted client per NAME_EDUCATION_TYPE:
    NAME_EDUCATION_TYPE_Lower secondary                  10.927673
    NAME_EDUCATION_TYPE_Secondary / secondary special     8.939929
    NAME_EDUCATION_TYPE_Incomplete higher                 8.484966
    NAME_EDUCATION_TYPE_Higher education                  5.355115
    NAME_EDUCATION_TYPE_Academic degree                   1.829268
    dtype: float64
    Percentage of non-defaulted client per NAME_EDUCATION_TYPE:
    NAME_EDUCATION_TYPE_Lower secondary                  89.072327
    NAME_EDUCATION_TYPE_Secondary / secondary special    91.060071
    NAME_EDUCATION_TYPE_Incomplete higher                91.515034
    NAME_EDUCATION_TYPE_Higher education                 94.644885
    NAME_EDUCATION_TYPE_Academic degree                  98.170732
    dtype: float64
    

Poin menarik:
1. client dengan status pendidikan lower secondary (SMP/Sederajat) cenderung sulit untuk membayar dan jumlahnya pun sedikit, **perlu plan yang lebih cocok**
2. Semakin tinggi derajat pendidikan client semakin sering membayar, yang cukup menarik adalah client dengan higher education berjumlah cukup banyak (terbanyak kedua) dan lebih sering membayar (posisi kedua terakhir pada list kesulitan membayar), sehingga **pelanggan dengan status pendidikan Higher Education dapat dijadikan target utama**

## Model Baseline

Pertama-tama akan dibuat model baseline. Model dibangun dengan menggunakan data mentah (dan tanpa kolom gender)

### Preprocessing
Preprocessing data meliputi penanganan missing value dan scaling data (normalisasi)


```python
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
```


```python
def preprocess(train_df, test_df):
    # Drop the target from the training data
    if 'TARGET' in train_df:
        train = train_df.drop(columns = ['TARGET'])
    else:
        train = train_df.copy()

    # Feature names
    features = list(train.columns)

    # Copy of the testing data
    test = test_df.copy()

    # Median imputation of missing values
    imputer = SimpleImputer(strategy = 'median')

    # Scale each feature to 0-1
    scaler = MinMaxScaler(feature_range = (0, 1))

    # Fit on the training data
    imputer.fit(train)

    # Transform both training and testing data
    train = imputer.transform(train)
    test = imputer.transform(test)

    # Repeat with the scaler
    scaler.fit(train)
    train = scaler.transform(train)
    test = scaler.transform(test)

    print('Training data shape: ', train.shape)
    print('Testing data shape: ', test.shape)
    return train, test, features
```

Untuk validasi akan menggunakan metode KFold Cross Validation dan metrics yang digunakan adalah ROC-AUC. KFold Cross Validation digunakan karena data training tidak begitu besar, sehingga jika menggunakan split train/validation biasa ditakutkan akan terjadi overfitting.


```python
train, test, features = preprocess(train_df, test_df)
```

    Training data shape:  (307511, 237)
    Testing data shape:  (48744, 237)
    

### Model Validation Method
K-Fold Cross-Validation dengan Metrics ROC-AUC


```python
# K-Fold Cross-Validation dengan Metrics ROC-AUC
from sklearn.model_selection import KFold
from sklearn.metrics import RocCurveDisplay, auc

# Set Up K-Fold Cross-Validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
```


```python
def roc_auc_plot(X, y, cv=None, classifier=None):
  """
    X : Train data
    y : Target
    returns trained classifier (yang terakhir)
  """
  tprs = []
  aucs = []
  mean_fpr = np.linspace(0, 1, 100)
  fig, ax = plt.subplots(figsize=(6, 6))
  for fold, (train, test) in enumerate(cv.split(X, y)):
      classifier.fit(X[train], y[train])
      viz = RocCurveDisplay.from_estimator(
          classifier,
          X[test],
          y[test],
          name=f"ROC fold {fold}",
          alpha=0.3,
          lw=1,
          ax=ax,
          plot_chance_level=(fold == 4),
      )
      interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
      interp_tpr[0] = 0.0
      tprs.append(interp_tpr)
      aucs.append(viz.roc_auc)

  mean_tpr = np.mean(tprs, axis=0)
  mean_tpr[-1] = 1.0
  mean_auc = auc(mean_fpr, mean_tpr)
  std_auc = np.std(aucs)
  ax.plot(
      mean_fpr,
      mean_tpr,
      color="b",
      label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
      lw=2,
      alpha=0.8,
  )

  std_tpr = np.std(tprs, axis=0)
  tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
  tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
  ax.fill_between(
      mean_fpr,
      tprs_lower,
      tprs_upper,
      color="grey",
      alpha=0.2,
      label=r"$\pm$ 1 std. dev.",
  )

  ax.set(
      xlabel="False Positive Rate",
      ylabel="True Positive Rate",
      title=f"Mean ROC curve with variability"#\n(Positive label '{target_names[1]}')",
  )
  ax.legend(loc="lower right")
  plt.show()
  return classifier
```

### Logistic Regression
score = 0.68


```python
from sklearn.linear_model import LogisticRegression
```


```python
# Make the model with the specified regularization parameter
log_reg = LogisticRegression(C = 0.0001)
```


```python
log_reg = roc_auc_plot(train, train_labels, cv=kfold, classifier=log_reg)
```


    
![png](README_files/README_85_0.png)
    



```python
# Make predictions
# Make sure to select the second column only
log_reg_pred = log_reg.predict_proba(test)[:, 1]

# Submission dataframe
submit = test_df[['SK_ID_CURR']]
submit['TARGET'] = log_reg_pred

submit.head()
```





  <div id="df-ade28d04-3b7a-4225-90f3-1cfa9d09dcf3" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SK_ID_CURR</th>
      <th>TARGET</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100001</td>
      <td>0.069942</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100005</td>
      <td>0.100977</td>
    </tr>
    <tr>
      <th>2</th>
      <td>100013</td>
      <td>0.064911</td>
    </tr>
    <tr>
      <th>3</th>
      <td>100028</td>
      <td>0.074376</td>
    </tr>
    <tr>
      <th>4</th>
      <td>100038</td>
      <td>0.106937</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-ade28d04-3b7a-4225-90f3-1cfa9d09dcf3')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-ade28d04-3b7a-4225-90f3-1cfa9d09dcf3 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-ade28d04-3b7a-4225-90f3-1cfa9d09dcf3');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-c75cd1aa-3ee2-402b-9027-ddd4569de3e0">
  <button class="colab-df-quickchart" onclick="quickchart('df-c75cd1aa-3ee2-402b-9027-ddd4569de3e0')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-c75cd1aa-3ee2-402b-9027-ddd4569de3e0 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>
    </div>
  </div>





```python
plt.hist(log_reg_pred)
```




    (array([5.1600e+02, 5.6050e+03, 1.2859e+04, 1.2866e+04, 9.9150e+03,
            4.8210e+03, 1.7310e+03, 3.7200e+02, 5.5000e+01, 4.0000e+00]),
     array([0.035263  , 0.04817443, 0.06108587, 0.07399731, 0.08690875,
            0.09982018, 0.11273162, 0.12564306, 0.1385545 , 0.15146593,
            0.16437737]),
     <BarContainer object of 10 artists>)




    
![png](README_files/README_87_1.png)
    


Seluruh prediksi bernilai < 0.5


```python
# Save the submission to a csv file
submit.to_csv('/content/gdrive/MyDrive/RAKAMIN/log_reg_baseline.csv', index = False)
```

#### unused


```python
# scoring = ['roc_auc', 'f1']
# # Perform Cross-Validation
# scores = cross_validate(log_reg, train, train_labels, cv=kfold, scoring=scoring, return_estimator=True) # new


# Print Average Scores
# print("Average ROC AUC: {:.2f}".format(np.mean(scores['test_roc_auc'])))
# print("Average F1 Score: {:.2f}".format(np.mean(scores['test_f1'])))

# Train on the training data
# log_reg.fit(train, train_labels)

# print(scores)

# fitted_estimators = scores['estimator']
# log_reg = fitted_estimators[np.argmax(scores['test_roc_auc'])]
```

### SGD
score = 0.72


```python
from sklearn.linear_model import SGDClassifier
```


```python
sgd_classifier = SGDClassifier(loss='modified_huber', random_state=2024)
```


```python
sgd_classifier = roc_auc_plot(train, train_labels, cv=kfold, classifier=sgd_classifier)
```


    
![png](README_files/README_95_0.png)
    



```python
# Make predictions on the test data
predictions = sgd_classifier.predict_proba(test)[:, 1]

# Make a submission dataframe
submit = test_df[['SK_ID_CURR']]
submit['TARGET'] = predictions
print(submit.head())
```

       SK_ID_CURR    TARGET
    0      100001  0.013302
    1      100005  0.084539
    2      100013  0.000000
    3      100028  0.000000
    4      100038  0.026888
    


```python
# Save the submission dataframe
submit.to_csv('/content/gdrive/MyDrive/RAKAMIN/sgd_classifier_baseline.csv', index = False)
```

### Naive Bayes
Score: 0.58 +- 0.01


```python
from sklearn.naive_bayes import GaussianNB
```


```python
gnb_clf = GaussianNB()
```


```python
gnb_clf = roc_auc_plot(train, train_labels, cv=kfold, classifier=gnb_clf)
```


    
![png](README_files/README_101_0.png)
    



```python
# Make predictions on the test data
predictions = gnb_clf.predict_proba(test)[:, 1]

# Make a submission dataframe
submit = test_df[['SK_ID_CURR']]
submit['TARGET'] = predictions
print(submit.head())
```

       SK_ID_CURR        TARGET
    0      100001  1.000000e+00
    1      100005  1.000000e+00
    2      100013  1.000000e+00
    3      100028  2.340385e-09
    4      100038  1.000000e+00
    


```python
# Save the submission dataframe
submit.to_csv('/content/gdrive/MyDrive/RAKAMIN/gnb_clf_baseline.csv', index = False)
```

### BernoulliNB
Score: 0.62


```python
from sklearn.naive_bayes import BernoulliNB
```


```python
bnb = BernoulliNB()
```


```python
bnb = roc_auc_plot(train, train_labels, cv=kfold, classifier=bnb)
```


    
![png](README_files/README_107_0.png)
    



```python
# Make predictions on the test data
predictions = bnb.predict_proba(test)[:, 1]

# Make a submission dataframe
submit = test_df[['SK_ID_CURR']]
submit['TARGET'] = predictions
print(submit.head())
```

       SK_ID_CURR    TARGET
    0      100001  0.467928
    1      100005  0.459403
    2      100013  0.113088
    3      100028  0.018208
    4      100038  0.487367
    


```python
# Save the submission dataframe
submit.to_csv('/content/gdrive/MyDrive/RAKAMIN/bnb_baseline.csv', index = False)
```

### LightGBM
Score: 0.75


```python
!pip install lightgbm
```

    Requirement already satisfied: lightgbm in /usr/local/lib/python3.10/dist-packages (4.1.0)
    Requirement already satisfied: numpy in /usr/local/lib/python3.10/dist-packages (from lightgbm) (1.23.5)
    Requirement already satisfied: scipy in /usr/local/lib/python3.10/dist-packages (from lightgbm) (1.11.4)
    


```python
from lightgbm import LGBMClassifier
```


```python
lgb = LGBMClassifier(n_estimators=1000, objective = 'binary',
                     class_weight = 'balanced', learning_rate = 0.05,
                     reg_alpha = 0.1, reg_lambda = 0.1,
                     subsample = 0.8, n_jobs = -1, random_state = 50)
```


```python
lgb = roc_auc_plot(train, train_labels, cv=kfold, classifier=lgb)
```

    [LightGBM] [Info] Number of positive: 19876, number of negative: 226132
    [LightGBM] [Info] Auto-choosing row-wise multi-threading, the overhead of testing was 0.187051 seconds.
    You can set `force_row_wise=true` to remove the overhead.
    And if memory is not enough, you can set `force_col_wise=true`.
    [LightGBM] [Info] Total Bins 11618
    [LightGBM] [Info] Number of data points in the train set: 246008, number of used features: 231
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.500000 -> initscore=0.000000
    [LightGBM] [Info] Start training from score 0.000000
    [LightGBM] [Info] Number of positive: 19888, number of negative: 226121
    [LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.405717 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 11616
    [LightGBM] [Info] Number of data points in the train set: 246009, number of used features: 229
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.500000 -> initscore=0.000000
    [LightGBM] [Info] Start training from score 0.000000
    [LightGBM] [Info] Number of positive: 19743, number of negative: 226266
    [LightGBM] [Info] Auto-choosing row-wise multi-threading, the overhead of testing was 0.393152 seconds.
    You can set `force_row_wise=true` to remove the overhead.
    And if memory is not enough, you can set `force_col_wise=true`.
    [LightGBM] [Info] Total Bins 11600
    [LightGBM] [Info] Number of data points in the train set: 246009, number of used features: 229
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.500000 -> initscore=0.000000
    [LightGBM] [Info] Start training from score 0.000000
    [LightGBM] [Info] Number of positive: 19921, number of negative: 226088
    [LightGBM] [Info] Auto-choosing row-wise multi-threading, the overhead of testing was 0.187767 seconds.
    You can set `force_row_wise=true` to remove the overhead.
    And if memory is not enough, you can set `force_col_wise=true`.
    [LightGBM] [Info] Total Bins 11615
    [LightGBM] [Info] Number of data points in the train set: 246009, number of used features: 229
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.500000 -> initscore=-0.000000
    [LightGBM] [Info] Start training from score -0.000000
    [LightGBM] [Info] Number of positive: 19872, number of negative: 226137
    [LightGBM] [Info] Auto-choosing row-wise multi-threading, the overhead of testing was 0.199573 seconds.
    You can set `force_row_wise=true` to remove the overhead.
    And if memory is not enough, you can set `force_col_wise=true`.
    [LightGBM] [Info] Total Bins 11615
    [LightGBM] [Info] Number of data points in the train set: 246009, number of used features: 229
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.500000 -> initscore=-0.000000
    [LightGBM] [Info] Start training from score -0.000000
    


    
![png](README_files/README_114_1.png)
    



```python
# Make predictions on the test data
predictions = lgb.predict_proba(test)[:, 1]

# Make a submission dataframe
submit = test_df[['SK_ID_CURR']]
submit['TARGET'] = predictions
print(submit.head())
```

       SK_ID_CURR    TARGET
    0      100001  0.161719
    1      100005  0.197959
    2      100013  0.035123
    3      100028  0.141083
    4      100038  0.498024
    


```python
# Save the submission dataframe
submit.to_csv('/content/gdrive/MyDrive/RAKAMIN/lgb_baseline.csv', index = False)
```

### Model Tidak Digunakan
Classifier di bawah ini tidak digunakan karena hasilnya kurang memuaskan, fitting terlalu lama, dan/atau makan RAM

#### Random Forest
score = 0.69144, fit sedikit lama


```python
# from sklearn.ensemble import RandomForestClassifier

# # Make the random forest classifier
# random_forest = RandomForestClassifier(n_estimators = 100, random_state = 50, verbose = 1, n_jobs = -1)
```


```python
# random_forest = roc_auc_plot(train, train_labels, cv=kfold, classifier=random_forest)

# Perform Cross-Validation
# scores = cross_validate(random_forest, train, train_labels, cv=kfold, scoring=scoring, return_estimator=True) # new

# # Print Average Scores
# print("Average ROC AUC: {:.2f}".format(np.mean(scores['test_roc_auc'])))
# print("Average F1 Score: {:.2f}".format(np.mean(scores['test_f1'])))

# # Train on the training data
# random_forest.fit(train, train_labels)
```

    [Parallel(n_jobs=-1)]: Using backend ThreadingBackend with 2 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  46 tasks      | elapsed:  1.2min
    [Parallel(n_jobs=-1)]: Done 100 out of 100 | elapsed:  2.5min finished
    [Parallel(n_jobs=2)]: Using backend ThreadingBackend with 2 concurrent workers.
    [Parallel(n_jobs=2)]: Done  46 tasks      | elapsed:    1.1s
    [Parallel(n_jobs=2)]: Done 100 out of 100 | elapsed:    2.3s finished
    [Parallel(n_jobs=-1)]: Using backend ThreadingBackend with 2 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  46 tasks      | elapsed:  1.2min
    [Parallel(n_jobs=-1)]: Done 100 out of 100 | elapsed:  2.6min finished
    [Parallel(n_jobs=2)]: Using backend ThreadingBackend with 2 concurrent workers.
    [Parallel(n_jobs=2)]: Done  46 tasks      | elapsed:    0.8s
    [Parallel(n_jobs=2)]: Done 100 out of 100 | elapsed:    1.8s finished
    [Parallel(n_jobs=-1)]: Using backend ThreadingBackend with 2 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  46 tasks      | elapsed:  1.2min
    [Parallel(n_jobs=-1)]: Done 100 out of 100 | elapsed:  2.5min finished
    [Parallel(n_jobs=2)]: Using backend ThreadingBackend with 2 concurrent workers.
    [Parallel(n_jobs=2)]: Done  46 tasks      | elapsed:    1.1s
    [Parallel(n_jobs=2)]: Done 100 out of 100 | elapsed:    2.1s finished
    [Parallel(n_jobs=-1)]: Using backend ThreadingBackend with 2 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  46 tasks      | elapsed:  1.2min
    [Parallel(n_jobs=-1)]: Done 100 out of 100 | elapsed:  2.5min finished
    [Parallel(n_jobs=2)]: Using backend ThreadingBackend with 2 concurrent workers.
    [Parallel(n_jobs=2)]: Done  46 tasks      | elapsed:    0.9s
    [Parallel(n_jobs=2)]: Done 100 out of 100 | elapsed:    1.9s finished
    [Parallel(n_jobs=-1)]: Using backend ThreadingBackend with 2 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  46 tasks      | elapsed:  1.1min
    [Parallel(n_jobs=-1)]: Done 100 out of 100 | elapsed:  2.4min finished
    [Parallel(n_jobs=2)]: Using backend ThreadingBackend with 2 concurrent workers.
    [Parallel(n_jobs=2)]: Done  46 tasks      | elapsed:    0.9s
    [Parallel(n_jobs=2)]: Done 100 out of 100 | elapsed:    2.2s finished
    


    
![png](README_files/README_120_1.png)
    


Average ROC AUC: 0.71

Average F1 Score: 0.00


```python
# # Extract feature importances
# feature_importance_values = random_forest.feature_importances_
# feature_importances = pd.DataFrame({'feature': features, 'importance': feature_importance_values})

# # Make predictions on the test data
# predictions = random_forest.predict_proba(test)[:, 1]
```

    [Parallel(n_jobs=2)]: Using backend ThreadingBackend with 2 concurrent workers.
    [Parallel(n_jobs=2)]: Done  46 tasks      | elapsed:    1.0s
    [Parallel(n_jobs=2)]: Done 100 out of 100 | elapsed:    2.0s finished
    


```python
# # Make a submission dataframe
# submit = test_df[['SK_ID_CURR']]
# submit['TARGET'] = predictions
# print(submit.head())
# # Save the submission dataframe
# submit.to_csv('random_forest_baseline.csv', index = False)
```

       SK_ID_CURR  TARGET
    0      100001    0.11
    1      100005    0.10
    2      100013    0.04
    3      100028    0.06
    4      100038    0.14
    


```python
# # Sort the DataFrame by 'importance' in descending order
# feature_importances = feature_importances.sort_values(by='importance', ascending=False)[:30]
```


```python
# # Increase figure size
# plt.figure(figsize=(15, 6))

# # Plot the barplot
# plt.title('Features Importance', fontsize=16)
# s = sns.barplot(x='feature', y='importance', data=feature_importances)

# # Rotate labels and adjust font size
# s.set_xticklabels(s.get_xticklabels(), rotation=45, ha='right', fontsize=11)

# plt.show()
```


    
![png](README_files/README_125_0.png)
    


#### KNN
*fit terlalu lama* Score = 0.55377


```python
# from sklearn.neighbors import KNeighborsClassifier
```


```python
# knn = KNeighborsClassifier(n_neighbors=3)

# knn = roc_auc_plot(train, train_labels, cv=kfold, classifier=knn)
```


    
![png](README_files/README_128_0.png)
    


Mean ROC AUC = 0.56


```python
# # Make predictions on the test data
# predictions = knn.predict_proba(test)[:, 1]
```


```python
# # Make a submission dataframe
# submit = test_df[['SK_ID_CURR']]
# submit['TARGET'] = predictions
# print(submit.head())
# # Save the submission dataframe
# submit.to_csv('/content/gdrive/MyDrive/RAKAMIN/knn_baseline.csv', index = False)
```

       SK_ID_CURR  TARGET
    0      100001     0.2
    1      100005     0.2
    2      100013     0.0
    3      100028     0.0
    4      100038     0.2
    

model terbaik sebelum feature engineering = sgd_sclassifier

#### Gaussian Process Classifier


```python
# from sklearn.gaussian_process import GaussianProcessClassifier
# from sklearn.gaussian_process.kernels import RBF
```


```python
# gpc_clf = GaussianProcessClassifier(1.0 * RBF(1.0), random_state=42)
```


```python
# gpc_clf = roc_auc_plot(train, train_labels, cv=kfold, classifier=gpc_clf)
```


```python
# # Make predictions on the test data
# predictions = gpc_clf.predict_proba(test)[:, 1]

# # Make a submission dataframe
# submit = test_df[['SK_ID_CURR']]
# submit['TARGET'] = predictions
# print(submit.head())
```


```python
# # Save the submission dataframe
# submit.to_csv('/content/gdrive/MyDrive/RAKAMIN/gpc_clf_baseline.csv', index = False)
```

#### AdaBoostClassifier



```python
from sklearn.ensemble import AdaBoostClassifier
```


```python
abc = AdaBoostClassifier(n_estimators=100, algorithm="SAMME", random_state=0)
```


```python
abc = roc_auc_plot(train, train_labels, cv=kfold, classifier=abc)
```


    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    <ipython-input-65-3aa68027684d> in <cell line: 1>()
    ----> 1 abc = roc_auc_plot(train, train_labels, cv=kfold, classifier=abc)
    

    <ipython-input-40-c265a08ea214> in roc_auc_plot(X, y, cv, classifier)
         10   fig, ax = plt.subplots(figsize=(6, 6))
         11   for fold, (train, test) in enumerate(cv.split(X, y)):
    ---> 12       classifier.fit(X[train], y[train])
         13       viz = RocCurveDisplay.from_estimator(
         14           classifier,
    

    /usr/local/lib/python3.10/dist-packages/sklearn/base.py in wrapper(estimator, *args, **kwargs)
       1349                 )
       1350             ):
    -> 1351                 return fit_method(estimator, *args, **kwargs)
       1352 
       1353         return wrapper
    

    /usr/local/lib/python3.10/dist-packages/sklearn/ensemble/_weight_boosting.py in fit(self, X, y, sample_weight)
        167 
        168             # Boosting step
    --> 169             sample_weight, estimator_weight, estimator_error = self._boost(
        170                 iboost, X, y, sample_weight, random_state
        171             )
    

    /usr/local/lib/python3.10/dist-packages/sklearn/ensemble/_weight_boosting.py in _boost(self, iboost, X, y, sample_weight, random_state)
        587 
        588         else:  # elif self.algorithm == "SAMME":
    --> 589             return self._boost_discrete(iboost, X, y, sample_weight, random_state)
        590 
        591     # TODO(1.6): Remove function. The `_boost_real` function won't be used any
    

    /usr/local/lib/python3.10/dist-packages/sklearn/ensemble/_weight_boosting.py in _boost_discrete(self, iboost, X, y, sample_weight, random_state)
        654         estimator = self._make_estimator(random_state=random_state)
        655 
    --> 656         estimator.fit(X, y, sample_weight=sample_weight)
        657 
        658         y_predict = estimator.predict(X)
    

    /usr/local/lib/python3.10/dist-packages/sklearn/base.py in wrapper(estimator, *args, **kwargs)
       1349                 )
       1350             ):
    -> 1351                 return fit_method(estimator, *args, **kwargs)
       1352 
       1353         return wrapper
    

    /usr/local/lib/python3.10/dist-packages/sklearn/tree/_classes.py in fit(self, X, y, sample_weight, check_input)
       1007         """
       1008 
    -> 1009         super()._fit(
       1010             X,
       1011             y,
    

    /usr/local/lib/python3.10/dist-packages/sklearn/tree/_classes.py in _fit(self, X, y, sample_weight, check_input, missing_values_in_feature_mask)
        470             )
        471 
    --> 472         builder.build(self.tree_, X, y, sample_weight, missing_values_in_feature_mask)
        473 
        474         if self.n_outputs_ == 1 and is_classifier(self):
    

    KeyboardInterrupt: 



    
![png](README_files/README_142_1.png)
    



```python
# Make predictions on the test data
predictions = abc.predict_proba(test)[:, 1]

# Make a submission dataframe
submit = test_df[['SK_ID_CURR']]
submit['TARGET'] = predictions
print(submit.head())
```


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    <ipython-input-66-e764ebc5d9c9> in <cell line: 2>()
          1 # Make predictions on the test data
    ----> 2 predictions = abc.predict_proba(test)[:, 1]
          3 
          4 # Make a submission dataframe
          5 submit = test_df[['SK_ID_CURR']]
    

    /usr/local/lib/python3.10/dist-packages/sklearn/ensemble/_weight_boosting.py in predict_proba(self, X)
        906             return np.ones((_num_samples(X), 1))
        907 
    --> 908         decision = self.decision_function(X)
        909         return self._compute_proba_from_decision(decision, n_classes)
        910 
    

    /usr/local/lib/python3.10/dist-packages/sklearn/ensemble/_weight_boosting.py in decision_function(self, X)
        790             )
        791         else:  # self.algorithm == "SAMME"
    --> 792             pred = sum(
        793                 np.where(
        794                     (estimator.predict(X) == classes).T,
    

    /usr/local/lib/python3.10/dist-packages/sklearn/ensemble/_weight_boosting.py in <genexpr>(.0)
        792             pred = sum(
        793                 np.where(
    --> 794                     (estimator.predict(X) == classes).T,
        795                     w,
        796                     -1 / (n_classes - 1) * w,
    

    /usr/local/lib/python3.10/dist-packages/sklearn/tree/_classes.py in predict(self, X, check_input)
        534         if is_classifier(self):
        535             if self.n_outputs_ == 1:
    --> 536                 return self.classes_.take(np.argmax(proba, axis=1), axis=0)
        537 
        538             else:
    

    AttributeError: 'list' object has no attribute 'take'



```python
# Save the submission dataframe
submit.to_csv('/content/gdrive/MyDrive/RAKAMIN/abc_baseline.csv', index = False)
```

## Feature Engineering


### Domain knowledge engineering
Terdapat beberapa data tambahan yang dapat digunakan untuk menentukan apakah seseorang akan kesulitan membayar atau tidak. Beberapa contohnya dari tabel application:
1. DAYS_EMPLOYED_PERCENT: rasio antara lama kerja client terhadap umurnya
2. INCOME_CREDIT_PERCENT: rasio antara pendapaatan client dengan jumlah pinjaman
3. INCOME_PER_PERSON : membagi pendapatan client dengan jumlah keluarga
4. ANNUITY_INCOME_PERCENT: rasio annuity dan pendapatan client
5. PAYMENT_RATE: lamanya pembayaran dalam bulan (annuity adalah pembayaran per bulan)


```python
def add_features(df):
    print(f'before: {df.shape}')
    df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    print(f'after: {df.shape}')
```


```python
add_features(train_df)
add_features(test_df)
```

    before: (307511, 238)
    after: (307511, 243)
    before: (48744, 237)
    after: (48744, 242)
    

### Polynomial features
Polynomial features untuk beberapa kolom yang sekiranya memiliki korelasi (belum analisis, baru berdasarkan domain knowledge)


```python
features_to_engineer = ['EXT_SOURCE_3', 'EXT_SOURCE_2', 'EXT_SOURCE_1',
                        'DAYS_BIRTH', 'DAYS_EMPLOYED', 'AMT_INCOME_TOTAL',
                        'AMT_CREDIT', 'AMT_ANNUITY', 'CNT_FAM_MEMBERS']
```


```python
poly_train = train_df[features_to_engineer]
poly_test = test_df[features_to_engineer]
```


```python
# imputer for handling missing values
imputer = SimpleImputer(strategy = 'median')

# Need to impute missing values
poly_train = imputer.fit_transform(poly_train)
poly_test = imputer.transform(poly_test)

from sklearn.preprocessing import PolynomialFeatures

# Create the polynomial object with specified degree
poly_transformer = PolynomialFeatures(degree = 3)
```


```python
# Train the polynomial features
# IMPORTANT: RUN ONCE
poly_transformer.fit(poly_train)

# Transform the features
poly_train = poly_transformer.transform(poly_train)
poly_test = poly_transformer.transform(poly_test)
```


```python
print('Polynomial train df shape: ', poly_train.shape)
print('Polynomial test df shape: ', poly_test.shape)
```

    Polynomial train df shape:  (307511, 220)
    Polynomial test df shape:  (48744, 220)
    


```python
poly_transformer.get_feature_names_out(features_to_engineer)[:10]
```




    array(['1', 'EXT_SOURCE_3', 'EXT_SOURCE_2', 'EXT_SOURCE_1', 'DAYS_BIRTH',
           'DAYS_EMPLOYED', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY',
           'CNT_FAM_MEMBERS'], dtype=object)




```python
# Create a dataframe of the features
poly_train = pd.DataFrame(poly_train, columns = poly_transformer.get_feature_names_out(features_to_engineer))
# Add in the target
poly_train['TARGET'] = train_labels
```


```python
# Put test features into dataframe
poly_test = pd.DataFrame(poly_test,
            columns = poly_transformer.get_feature_names_out(features_to_engineer))
```


```python
# Merge polynomial features into training dataframe
poly_train['SK_ID_CURR'] = train_df['SK_ID_CURR']
app_train_pd = train_df.merge(poly_train, on = 'SK_ID_CURR', how = 'left')

# Merge polnomial features into testing dataframe
poly_test['SK_ID_CURR'] = test_df['SK_ID_CURR']
app_test_pd = test_df.merge(poly_test, on = 'SK_ID_CURR', how = 'left')

# Align the dataframes
app_train_pd, app_test_pd = app_train_pd.align(app_test_pd, join = 'inner', axis = 1)
app_train_pd['TARGET'] = train_df['TARGET']

# Print out the new shapes
print('Training data with polynomial features shape: ', app_train_pd.shape)
print('Testing data with polynomial features shape:  ', app_test_pd.shape)
```

    Training data with polynomial features shape:  (307511, 463)
    Testing data with polynomial features shape:   (48744, 462)
    

### Optional


```python
del poly_train, poly_test
gc.collect()
```




    5111




```python
# Find the correlations with the target
poly_corrs = app_train_pd.corr()['TARGET'].sort_values()

# Display most negative and most positive
print(poly_corrs.head(10))
print(poly_corrs.tail(10))
del poly_corrs
gc.collect()
```

    EXT_SOURCE_3 EXT_SOURCE_2                -0.193939
    EXT_SOURCE_3 EXT_SOURCE_2 EXT_SOURCE_1   -0.189605
    EXT_SOURCE_3_x                           -0.178919
    EXT_SOURCE_3 EXT_SOURCE_2^2              -0.176428
    EXT_SOURCE_3^2 EXT_SOURCE_2              -0.172282
    EXT_SOURCE_2 EXT_SOURCE_1                -0.166625
    EXT_SOURCE_3 EXT_SOURCE_1                -0.164065
    EXT_SOURCE_2_x                           -0.160472
    EXT_SOURCE_2_y                           -0.160295
    EXT_SOURCE_2^2 EXT_SOURCE_1              -0.156867
    Name: TARGET, dtype: float64
    EXT_SOURCE_2 DAYS_BIRTH CNT_FAM_MEMBERS    0.123111
    EXT_SOURCE_3^2 DAYS_BIRTH                  0.141777
    EXT_SOURCE_2^2 DAYS_BIRTH                  0.149313
    EXT_SOURCE_3 DAYS_BIRTH                    0.150109
    EXT_SOURCE_3 EXT_SOURCE_1 DAYS_BIRTH       0.151816
    EXT_SOURCE_2 EXT_SOURCE_1 DAYS_BIRTH       0.155891
    EXT_SOURCE_2 DAYS_BIRTH                    0.156873
    EXT_SOURCE_3 EXT_SOURCE_2 DAYS_BIRTH       0.181283
    TARGET                                     1.000000
    1                                               NaN
    Name: TARGET, dtype: float64
    




    0



## Model setelah feature engineering

### Preprocessing


```python
# preprocess dataset
train, test, features = preprocess(app_train_pd, app_test_pd)
```

    Training data shape:  (307511, 462)
    Testing data shape:  (48744, 462)
    

### Logistic Regression
Score: 0.73


```python
# Make the model with the specified regularization parameter
log_reg = LogisticRegression(C = 0.0001)

log_reg = roc_auc_plot(train, train_labels, cv=kfold, classifier=log_reg)
```


    
![png](README_files/README_166_0.png)
    



```python
# Make predictions
log_reg_pred = log_reg.predict_proba(test)[:, 1]
```


```python
# Submission dataframe
submit = test_df[['SK_ID_CURR']]
submit['TARGET'] = log_reg_pred

print(submit.head())
plt.hist(log_reg_pred)
```

       SK_ID_CURR    TARGET
    0      100001  0.059451
    1      100005  0.117069
    2      100013  0.040917
    3      100028  0.066997
    4      100038  0.123094
    




    (array([2.5210e+03, 1.1979e+04, 1.3941e+04, 1.0053e+04, 5.8800e+03,
            2.8600e+03, 1.1440e+03, 3.0300e+02, 6.0000e+01, 3.0000e+00]),
     array([0.01287276, 0.03637096, 0.05986916, 0.08336736, 0.10686556,
            0.13036376, 0.15386196, 0.17736016, 0.20085836, 0.22435656,
            0.24785476]),
     <BarContainer object of 10 artists>)




    
![png](README_files/README_168_2.png)
    



```python
# Save the submission to a csv file
submit.to_csv('/content/gdrive/MyDrive/RAKAMIN/log_reg_feature_engineered.csv', index = False)
```

### SGD
Score: 0.72 +- 0.01


```python
sgd_classifier = SGDClassifier(loss='modified_huber', random_state=2024)
```


```python
sgd_classifier = roc_auc_plot(train, train_labels, cv=kfold, classifier=sgd_classifier)
```


    
![png](README_files/README_172_0.png)
    



```python
# Make predictions on the test data
predictions = sgd_classifier.predict_proba(test)[:, 1]

# Make a submission dataframe
submit = test_df[['SK_ID_CURR']]
submit['TARGET'] = predictions
print(submit.head())
```

       SK_ID_CURR    TARGET
    0      100001  0.000000
    1      100005  0.154365
    2      100013  0.000000
    3      100028  0.000000
    4      100038  0.041681
    


```python
# Save the submission dataframe
submit.to_csv('/content/gdrive/MyDrive/RAKAMIN/sgd_classifier_feature_engineered.csv', index = False)
```

### Naive Bayes
Score: 0.66 +- 0.04


```python
gnb_clf = GaussianNB()
```


```python
gnb_clf = roc_auc_plot(train, train_labels, cv=kfold, classifier=gnb_clf)
```


    
![png](README_files/README_177_0.png)
    



```python
# Make predictions on the test data
predictions = gnb_clf.predict_proba(test)[:, 1]

# Make a submission dataframe
submit = test_df[['SK_ID_CURR']]
submit['TARGET'] = predictions
print(submit.head())
```

       SK_ID_CURR        TARGET
    0      100001  9.115344e-16
    1      100005  1.000000e+00
    2      100013  3.947801e-98
    3      100028  1.649292e-86
    4      100038  1.000000e+00
    


```python
# Save the submission dataframe
submit.to_csv('/content/gdrive/MyDrive/RAKAMIN/gnb_clf_feature_engineered.csv', index = False)
```

### BernoulliNB
Score: 0.62


```python
bnb = BernoulliNB()
```


```python
bnb = roc_auc_plot(train, train_labels, cv=kfold, classifier=bnb)
```


    
![png](README_files/README_182_0.png)
    



```python
# Make predictions on the test data
predictions = bnb.predict_proba(test)[:, 1]

# Make a submission dataframe
submit = test_df[['SK_ID_CURR']]
submit['TARGET'] = predictions
print(submit.head())
```

       SK_ID_CURR    TARGET
    0      100001  0.452341
    1      100005  0.443863
    2      100013  0.106945
    3      100028  0.017120
    4      100038  0.471707
    


```python
# Save the submission dataframe
submit.to_csv('/content/gdrive/MyDrive/RAKAMIN/bnb_feature_engineered.csv', index = False)
```

### LightGBM
Score: 0.76


```python
lgb = LGBMClassifier(n_estimators=1000, objective = 'binary',
                     class_weight = 'balanced', learning_rate = 0.05,
                     reg_alpha = 0.1, reg_lambda = 0.1,
                     subsample = 0.8, n_jobs = -1, random_state = 50)
```


```python
lgb = roc_auc_plot(train, train_labels, cv=kfold, classifier=lgb)
```

    [LightGBM] [Info] Number of positive: 19876, number of negative: 226132
    [LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 2.134608 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 67779
    [LightGBM] [Info] Number of data points in the train set: 246008, number of used features: 455
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.500000 -> initscore=0.000000
    [LightGBM] [Info] Start training from score 0.000000
    [LightGBM] [Info] Number of positive: 19888, number of negative: 226121
    [LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 1.802055 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 67779
    [LightGBM] [Info] Number of data points in the train set: 246009, number of used features: 453
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.500000 -> initscore=0.000000
    [LightGBM] [Info] Start training from score 0.000000
    [LightGBM] [Info] Number of positive: 19743, number of negative: 226266
    [LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 1.823152 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 67773
    [LightGBM] [Info] Number of data points in the train set: 246009, number of used features: 453
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.500000 -> initscore=0.000000
    [LightGBM] [Info] Start training from score 0.000000
    [LightGBM] [Info] Number of positive: 19921, number of negative: 226088
    [LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 1.638235 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 67783
    [LightGBM] [Info] Number of data points in the train set: 246009, number of used features: 453
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.500000 -> initscore=-0.000000
    [LightGBM] [Info] Start training from score -0.000000
    [LightGBM] [Info] Number of positive: 19872, number of negative: 226137
    [LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 1.874431 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 67800
    [LightGBM] [Info] Number of data points in the train set: 246009, number of used features: 453
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.500000 -> initscore=-0.000000
    [LightGBM] [Info] Start training from score -0.000000
    


    
![png](README_files/README_187_1.png)
    



```python
# Make predictions on the test data
predictions = lgb.predict_proba(test)[:, 1]

# Make a submission dataframe
submit = test_df[['SK_ID_CURR']]
submit['TARGET'] = predictions
print(submit.head())
```

       SK_ID_CURR    TARGET
    0      100001  0.161095
    1      100005  0.242755
    2      100013  0.025411
    3      100028  0.119149
    4      100038  0.536400
    


```python
# Save the submission dataframe
submit.to_csv('/content/gdrive/MyDrive/RAKAMIN/lgb_feature_engineered.csv', index = False)
```

## Kesimpulan Model Baseline dan Setelah Feature Engineering
SGD, Logistic Regression, dan LightGBM konsisten menghasilkan score lebih besar.

Score untuk dikalahkan: 0.76 dari LightGBM.

Feature Engineering berhasil meningkatkan ROC-AUC beberapa model.

Langkah selanjutnya:
1. Penggabungan data dengan tabel lain
2. Feature engineering (lagi) dan feature selection
3. Hyperparameter tuning

## Penggabungan dengan data previous application dan bureau
Penggabungan akan menggunakan agregat untuk data previous application dan bureau

### Agregat


```python
def agg_df(df, group_by, df_name):
  group_ids = df[group_by]
  numeric_df = df.select_dtypes('number')
  numeric_df[group_by] = group_ids

  # Group by the specified variable and calculate the statistics
  agg = numeric_df.groupby(group_by).agg(['count', 'mean', 'max', 'min', 'sum']).reset_index()
  # Need to create new column names
  column_names = [group_by]

  # Iterate through the variables names
  for var in agg.columns.levels[0]:
      # Skip the grouping variable
      if var != group_by:
          # Iterate through the stat names
          for stat in agg.columns.levels[1][:-1]:
              # Make a new column name for the variable and stat
              column_names.append('%s_%s_%s' % (df_name, var, stat))

  agg.columns = column_names

  return agg
```


```python
def count_categorical(df, group_by, df_name):
  # Select the categorical columns
  categorical = pd.get_dummies(df.select_dtypes('object'))

  # Make sure to put the identifying id on the column
  categorical[group_by] = df[group_by]

  # Groupby the group var and calculate the sum and mean
  categorical = categorical.groupby(group_by).agg(['sum', 'mean'])

  column_names = []

  # Iterate through the columns in level 0
  for var in categorical.columns.levels[0]:
      # Iterate through the stats in level 1
      for stat in ['count', 'count_norm']:
          # Make a new column name
          column_names.append('%s_%s_%s' % (df_name, var, stat))

  categorical.columns = column_names

  return categorical
```

### Load data Burreau
Data pinjaman sebelumnya dari institusi lain


```python
# Read in bureau
bureau = pd.read_csv('/content/gdrive/MyDrive/RAKAMIN/bureau.csv')
bureau.head()
```





  <div id="df-cd29f76c-1dfc-47e8-b30b-7c52a741fd57" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SK_ID_CURR</th>
      <th>SK_ID_BUREAU</th>
      <th>CREDIT_ACTIVE</th>
      <th>CREDIT_CURRENCY</th>
      <th>DAYS_CREDIT</th>
      <th>CREDIT_DAY_OVERDUE</th>
      <th>DAYS_CREDIT_ENDDATE</th>
      <th>DAYS_ENDDATE_FACT</th>
      <th>AMT_CREDIT_MAX_OVERDUE</th>
      <th>CNT_CREDIT_PROLONG</th>
      <th>AMT_CREDIT_SUM</th>
      <th>AMT_CREDIT_SUM_DEBT</th>
      <th>AMT_CREDIT_SUM_LIMIT</th>
      <th>AMT_CREDIT_SUM_OVERDUE</th>
      <th>CREDIT_TYPE</th>
      <th>DAYS_CREDIT_UPDATE</th>
      <th>AMT_ANNUITY</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>215354</td>
      <td>5714462</td>
      <td>Closed</td>
      <td>currency 1</td>
      <td>-497</td>
      <td>0</td>
      <td>-153.0</td>
      <td>-153.0</td>
      <td>NaN</td>
      <td>0</td>
      <td>91323.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>Consumer credit</td>
      <td>-131</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>215354</td>
      <td>5714463</td>
      <td>Active</td>
      <td>currency 1</td>
      <td>-208</td>
      <td>0</td>
      <td>1075.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>225000.0</td>
      <td>171342.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>Credit card</td>
      <td>-20</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>215354</td>
      <td>5714464</td>
      <td>Active</td>
      <td>currency 1</td>
      <td>-203</td>
      <td>0</td>
      <td>528.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>464323.5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>Consumer credit</td>
      <td>-16</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>215354</td>
      <td>5714465</td>
      <td>Active</td>
      <td>currency 1</td>
      <td>-203</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>90000.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>Credit card</td>
      <td>-16</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>215354</td>
      <td>5714466</td>
      <td>Active</td>
      <td>currency 1</td>
      <td>-629</td>
      <td>0</td>
      <td>1197.0</td>
      <td>NaN</td>
      <td>77674.5</td>
      <td>0</td>
      <td>2700000.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>Consumer credit</td>
      <td>-21</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-cd29f76c-1dfc-47e8-b30b-7c52a741fd57')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-cd29f76c-1dfc-47e8-b30b-7c52a741fd57 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-cd29f76c-1dfc-47e8-b30b-7c52a741fd57');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-0cb50f49-9310-4f74-84fd-6ec36fe313c6">
  <button class="colab-df-quickchart" onclick="quickchart('df-0cb50f49-9310-4f74-84fd-6ec36fe313c6')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-0cb50f49-9310-4f74-84fd-6ec36fe313c6 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>
    </div>
  </div>





```python
bureau_agg = agg_df(bureau.drop(columns = ['SK_ID_BUREAU']), group_by = 'SK_ID_CURR', df_name = 'bureau')
bureau_agg.head()
```





  <div id="df-9aa35b0f-058e-4f6d-a42a-6537e367bf14" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SK_ID_CURR</th>
      <th>bureau_DAYS_CREDIT_count</th>
      <th>bureau_DAYS_CREDIT_mean</th>
      <th>bureau_DAYS_CREDIT_max</th>
      <th>bureau_DAYS_CREDIT_min</th>
      <th>bureau_DAYS_CREDIT_sum</th>
      <th>bureau_CREDIT_DAY_OVERDUE_count</th>
      <th>bureau_CREDIT_DAY_OVERDUE_mean</th>
      <th>bureau_CREDIT_DAY_OVERDUE_max</th>
      <th>bureau_CREDIT_DAY_OVERDUE_min</th>
      <th>...</th>
      <th>bureau_DAYS_CREDIT_UPDATE_count</th>
      <th>bureau_DAYS_CREDIT_UPDATE_mean</th>
      <th>bureau_DAYS_CREDIT_UPDATE_max</th>
      <th>bureau_DAYS_CREDIT_UPDATE_min</th>
      <th>bureau_DAYS_CREDIT_UPDATE_sum</th>
      <th>bureau_AMT_ANNUITY_count</th>
      <th>bureau_AMT_ANNUITY_mean</th>
      <th>bureau_AMT_ANNUITY_max</th>
      <th>bureau_AMT_ANNUITY_min</th>
      <th>bureau_AMT_ANNUITY_sum</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100001</td>
      <td>7</td>
      <td>-735.000000</td>
      <td>-49</td>
      <td>-1572</td>
      <td>-5145</td>
      <td>7</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>7</td>
      <td>-93.142857</td>
      <td>-6</td>
      <td>-155</td>
      <td>-652</td>
      <td>7</td>
      <td>3545.357143</td>
      <td>10822.5</td>
      <td>0.0</td>
      <td>24817.5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100002</td>
      <td>8</td>
      <td>-874.000000</td>
      <td>-103</td>
      <td>-1437</td>
      <td>-6992</td>
      <td>8</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>8</td>
      <td>-499.875000</td>
      <td>-7</td>
      <td>-1185</td>
      <td>-3999</td>
      <td>7</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>100003</td>
      <td>4</td>
      <td>-1400.750000</td>
      <td>-606</td>
      <td>-2586</td>
      <td>-5603</td>
      <td>4</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>4</td>
      <td>-816.000000</td>
      <td>-43</td>
      <td>-2131</td>
      <td>-3264</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>100004</td>
      <td>2</td>
      <td>-867.000000</td>
      <td>-408</td>
      <td>-1326</td>
      <td>-1734</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>2</td>
      <td>-532.000000</td>
      <td>-382</td>
      <td>-682</td>
      <td>-1064</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>100005</td>
      <td>3</td>
      <td>-190.666667</td>
      <td>-62</td>
      <td>-373</td>
      <td>-572</td>
      <td>3</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>3</td>
      <td>-54.333333</td>
      <td>-11</td>
      <td>-121</td>
      <td>-163</td>
      <td>3</td>
      <td>1420.500000</td>
      <td>4261.5</td>
      <td>0.0</td>
      <td>4261.5</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 61 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-9aa35b0f-058e-4f6d-a42a-6537e367bf14')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-9aa35b0f-058e-4f6d-a42a-6537e367bf14 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-9aa35b0f-058e-4f6d-a42a-6537e367bf14');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-15eead5d-a84d-49c2-9a2d-299c476a066f">
  <button class="colab-df-quickchart" onclick="quickchart('df-15eead5d-a84d-49c2-9a2d-299c476a066f')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-15eead5d-a84d-49c2-9a2d-299c476a066f button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>
    </div>
  </div>





```python
bureau_counts = count_categorical(bureau, group_by = 'SK_ID_CURR', df_name = 'bureau')
bureau_counts.head()
```





  <div id="df-09043f1b-488c-41f7-a806-377f98407fdf" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bureau_CREDIT_ACTIVE_Active_count</th>
      <th>bureau_CREDIT_ACTIVE_Active_count_norm</th>
      <th>bureau_CREDIT_ACTIVE_Bad debt_count</th>
      <th>bureau_CREDIT_ACTIVE_Bad debt_count_norm</th>
      <th>bureau_CREDIT_ACTIVE_Closed_count</th>
      <th>bureau_CREDIT_ACTIVE_Closed_count_norm</th>
      <th>bureau_CREDIT_ACTIVE_Sold_count</th>
      <th>bureau_CREDIT_ACTIVE_Sold_count_norm</th>
      <th>bureau_CREDIT_CURRENCY_currency 1_count</th>
      <th>bureau_CREDIT_CURRENCY_currency 1_count_norm</th>
      <th>...</th>
      <th>bureau_CREDIT_TYPE_Microloan_count</th>
      <th>bureau_CREDIT_TYPE_Microloan_count_norm</th>
      <th>bureau_CREDIT_TYPE_Mobile operator loan_count</th>
      <th>bureau_CREDIT_TYPE_Mobile operator loan_count_norm</th>
      <th>bureau_CREDIT_TYPE_Mortgage_count</th>
      <th>bureau_CREDIT_TYPE_Mortgage_count_norm</th>
      <th>bureau_CREDIT_TYPE_Real estate loan_count</th>
      <th>bureau_CREDIT_TYPE_Real estate loan_count_norm</th>
      <th>bureau_CREDIT_TYPE_Unknown type of loan_count</th>
      <th>bureau_CREDIT_TYPE_Unknown type of loan_count_norm</th>
    </tr>
    <tr>
      <th>SK_ID_CURR</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>100001</th>
      <td>3</td>
      <td>0.428571</td>
      <td>0</td>
      <td>0.0</td>
      <td>4</td>
      <td>0.571429</td>
      <td>0</td>
      <td>0.0</td>
      <td>7</td>
      <td>1.0</td>
      <td>...</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>100002</th>
      <td>2</td>
      <td>0.250000</td>
      <td>0</td>
      <td>0.0</td>
      <td>6</td>
      <td>0.750000</td>
      <td>0</td>
      <td>0.0</td>
      <td>8</td>
      <td>1.0</td>
      <td>...</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>100003</th>
      <td>1</td>
      <td>0.250000</td>
      <td>0</td>
      <td>0.0</td>
      <td>3</td>
      <td>0.750000</td>
      <td>0</td>
      <td>0.0</td>
      <td>4</td>
      <td>1.0</td>
      <td>...</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>100004</th>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.0</td>
      <td>2</td>
      <td>1.000000</td>
      <td>0</td>
      <td>0.0</td>
      <td>2</td>
      <td>1.0</td>
      <td>...</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>100005</th>
      <td>2</td>
      <td>0.666667</td>
      <td>0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0.333333</td>
      <td>0</td>
      <td>0.0</td>
      <td>3</td>
      <td>1.0</td>
      <td>...</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 46 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-09043f1b-488c-41f7-a806-377f98407fdf')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-09043f1b-488c-41f7-a806-377f98407fdf button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-09043f1b-488c-41f7-a806-377f98407fdf');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-51174bfb-c970-4916-9856-6609643ca2de">
  <button class="colab-df-quickchart" onclick="quickchart('df-51174bfb-c970-4916-9856-6609643ca2de')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-51174bfb-c970-4916-9856-6609643ca2de button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>
    </div>
  </div>





```python
bureau_new = bureau_agg.merge(bureau_counts, right_index = True, left_on = 'SK_ID_CURR', how = 'outer')

# Merge to include the SK_ID_CURR
# bureau_new = bureau_new.merge(bureau[['SK_ID_BUREAU', 'SK_ID_CURR']], on = 'SK_ID_CURR', how = 'left')
bureau_new.head()
```





  <div id="df-a0e8114b-6a5c-4ae3-be3c-a127d78497b7" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SK_ID_CURR</th>
      <th>bureau_DAYS_CREDIT_count</th>
      <th>bureau_DAYS_CREDIT_mean</th>
      <th>bureau_DAYS_CREDIT_max</th>
      <th>bureau_DAYS_CREDIT_min</th>
      <th>bureau_DAYS_CREDIT_sum</th>
      <th>bureau_CREDIT_DAY_OVERDUE_count</th>
      <th>bureau_CREDIT_DAY_OVERDUE_mean</th>
      <th>bureau_CREDIT_DAY_OVERDUE_max</th>
      <th>bureau_CREDIT_DAY_OVERDUE_min</th>
      <th>...</th>
      <th>bureau_CREDIT_TYPE_Microloan_count</th>
      <th>bureau_CREDIT_TYPE_Microloan_count_norm</th>
      <th>bureau_CREDIT_TYPE_Mobile operator loan_count</th>
      <th>bureau_CREDIT_TYPE_Mobile operator loan_count_norm</th>
      <th>bureau_CREDIT_TYPE_Mortgage_count</th>
      <th>bureau_CREDIT_TYPE_Mortgage_count_norm</th>
      <th>bureau_CREDIT_TYPE_Real estate loan_count</th>
      <th>bureau_CREDIT_TYPE_Real estate loan_count_norm</th>
      <th>bureau_CREDIT_TYPE_Unknown type of loan_count</th>
      <th>bureau_CREDIT_TYPE_Unknown type of loan_count_norm</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100001</td>
      <td>7</td>
      <td>-735.000000</td>
      <td>-49</td>
      <td>-1572</td>
      <td>-5145</td>
      <td>7</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100002</td>
      <td>8</td>
      <td>-874.000000</td>
      <td>-103</td>
      <td>-1437</td>
      <td>-6992</td>
      <td>8</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>100003</td>
      <td>4</td>
      <td>-1400.750000</td>
      <td>-606</td>
      <td>-2586</td>
      <td>-5603</td>
      <td>4</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>100004</td>
      <td>2</td>
      <td>-867.000000</td>
      <td>-408</td>
      <td>-1326</td>
      <td>-1734</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>100005</td>
      <td>3</td>
      <td>-190.666667</td>
      <td>-62</td>
      <td>-373</td>
      <td>-572</td>
      <td>3</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 107 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-a0e8114b-6a5c-4ae3-be3c-a127d78497b7')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-a0e8114b-6a5c-4ae3-be3c-a127d78497b7 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-a0e8114b-6a5c-4ae3-be3c-a127d78497b7');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-c97cd6b2-c488-4eb5-ab08-b0de587467f5">
  <button class="colab-df-quickchart" onclick="quickchart('df-c97cd6b2-c488-4eb5-ab08-b0de587467f5')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-c97cd6b2-c488-4eb5-ab08-b0de587467f5 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>
    </div>
  </div>





```python
train_df.head()
```





  <div id="df-0d97bed2-1445-4b97-bf0e-11a32671b953" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SK_ID_CURR</th>
      <th>NAME_CONTRACT_TYPE</th>
      <th>FLAG_OWN_CAR</th>
      <th>FLAG_OWN_REALTY</th>
      <th>CNT_CHILDREN</th>
      <th>AMT_INCOME_TOTAL</th>
      <th>AMT_CREDIT</th>
      <th>AMT_ANNUITY</th>
      <th>AMT_GOODS_PRICE</th>
      <th>REGION_POPULATION_RELATIVE</th>
      <th>...</th>
      <th>WALLSMATERIAL_MODE_Stone, brick</th>
      <th>WALLSMATERIAL_MODE_Wooden</th>
      <th>EMERGENCYSTATE_MODE_No</th>
      <th>EMERGENCYSTATE_MODE_Yes</th>
      <th>TARGET</th>
      <th>DAYS_EMPLOYED_PERC</th>
      <th>INCOME_CREDIT_PERC</th>
      <th>INCOME_PER_PERSON</th>
      <th>ANNUITY_INCOME_PERC</th>
      <th>PAYMENT_RATE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100002</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>202500.0</td>
      <td>406597.5</td>
      <td>24700.5</td>
      <td>351000.0</td>
      <td>0.018801</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0.067329</td>
      <td>0.498036</td>
      <td>202500.0</td>
      <td>0.121978</td>
      <td>0.060749</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100003</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>270000.0</td>
      <td>1293502.5</td>
      <td>35698.5</td>
      <td>1129500.0</td>
      <td>0.003541</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.070862</td>
      <td>0.208736</td>
      <td>135000.0</td>
      <td>0.132217</td>
      <td>0.027598</td>
    </tr>
    <tr>
      <th>2</th>
      <td>100004</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>67500.0</td>
      <td>135000.0</td>
      <td>6750.0</td>
      <td>135000.0</td>
      <td>0.010032</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.011814</td>
      <td>0.500000</td>
      <td>67500.0</td>
      <td>0.100000</td>
      <td>0.050000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>100006</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>135000.0</td>
      <td>312682.5</td>
      <td>29686.5</td>
      <td>297000.0</td>
      <td>0.008019</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.159905</td>
      <td>0.431748</td>
      <td>67500.0</td>
      <td>0.219900</td>
      <td>0.094941</td>
    </tr>
    <tr>
      <th>4</th>
      <td>100007</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>121500.0</td>
      <td>513000.0</td>
      <td>21865.5</td>
      <td>513000.0</td>
      <td>0.028663</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.152418</td>
      <td>0.236842</td>
      <td>121500.0</td>
      <td>0.179963</td>
      <td>0.042623</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 243 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-0d97bed2-1445-4b97-bf0e-11a32671b953')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-0d97bed2-1445-4b97-bf0e-11a32671b953 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-0d97bed2-1445-4b97-bf0e-11a32671b953');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-fd2f2711-0b09-47eb-bf4e-48dac1ac6116">
  <button class="colab-df-quickchart" onclick="quickchart('df-fd2f2711-0b09-47eb-bf4e-48dac1ac6116')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-fd2f2711-0b09-47eb-bf4e-48dac1ac6116 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>
    </div>
  </div>





```python
print("-----------------------")
# Merge with bureau
train_df = train_df.merge(bureau_new, on = 'SK_ID_CURR', how = 'left')
train_df.head()
```

    -----------------------
    





  <div id="df-97e1307c-e9d6-4224-a18f-4645054775a2" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SK_ID_CURR</th>
      <th>NAME_CONTRACT_TYPE</th>
      <th>FLAG_OWN_CAR</th>
      <th>FLAG_OWN_REALTY</th>
      <th>CNT_CHILDREN</th>
      <th>AMT_INCOME_TOTAL</th>
      <th>AMT_CREDIT</th>
      <th>AMT_ANNUITY</th>
      <th>AMT_GOODS_PRICE</th>
      <th>REGION_POPULATION_RELATIVE</th>
      <th>...</th>
      <th>bureau_CREDIT_TYPE_Microloan_count</th>
      <th>bureau_CREDIT_TYPE_Microloan_count_norm</th>
      <th>bureau_CREDIT_TYPE_Mobile operator loan_count</th>
      <th>bureau_CREDIT_TYPE_Mobile operator loan_count_norm</th>
      <th>bureau_CREDIT_TYPE_Mortgage_count</th>
      <th>bureau_CREDIT_TYPE_Mortgage_count_norm</th>
      <th>bureau_CREDIT_TYPE_Real estate loan_count</th>
      <th>bureau_CREDIT_TYPE_Real estate loan_count_norm</th>
      <th>bureau_CREDIT_TYPE_Unknown type of loan_count</th>
      <th>bureau_CREDIT_TYPE_Unknown type of loan_count_norm</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100002</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>202500.0</td>
      <td>406597.5</td>
      <td>24700.5</td>
      <td>351000.0</td>
      <td>0.018801</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100003</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>270000.0</td>
      <td>1293502.5</td>
      <td>35698.5</td>
      <td>1129500.0</td>
      <td>0.003541</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>100004</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>67500.0</td>
      <td>135000.0</td>
      <td>6750.0</td>
      <td>135000.0</td>
      <td>0.010032</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>100006</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>135000.0</td>
      <td>312682.5</td>
      <td>29686.5</td>
      <td>297000.0</td>
      <td>0.008019</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>100007</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>121500.0</td>
      <td>513000.0</td>
      <td>21865.5</td>
      <td>513000.0</td>
      <td>0.028663</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 349 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-97e1307c-e9d6-4224-a18f-4645054775a2')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-97e1307c-e9d6-4224-a18f-4645054775a2 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-97e1307c-e9d6-4224-a18f-4645054775a2');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-2ae76e0b-0e52-460f-963b-73009d141502">
  <button class="colab-df-quickchart" onclick="quickchart('df-2ae76e0b-0e52-460f-963b-73009d141502')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-2ae76e0b-0e52-460f-963b-73009d141502 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>
    </div>
  </div>





```python
test_df = test_df.merge(bureau_new, on = 'SK_ID_CURR', how = 'left')
test_df.head()
```





  <div id="df-cb52c3a9-a09a-4daf-bf1c-3679485d277f" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SK_ID_CURR</th>
      <th>NAME_CONTRACT_TYPE</th>
      <th>FLAG_OWN_CAR</th>
      <th>FLAG_OWN_REALTY</th>
      <th>CNT_CHILDREN</th>
      <th>AMT_INCOME_TOTAL</th>
      <th>AMT_CREDIT</th>
      <th>AMT_ANNUITY</th>
      <th>AMT_GOODS_PRICE</th>
      <th>REGION_POPULATION_RELATIVE</th>
      <th>...</th>
      <th>bureau_CREDIT_TYPE_Microloan_count</th>
      <th>bureau_CREDIT_TYPE_Microloan_count_norm</th>
      <th>bureau_CREDIT_TYPE_Mobile operator loan_count</th>
      <th>bureau_CREDIT_TYPE_Mobile operator loan_count_norm</th>
      <th>bureau_CREDIT_TYPE_Mortgage_count</th>
      <th>bureau_CREDIT_TYPE_Mortgage_count_norm</th>
      <th>bureau_CREDIT_TYPE_Real estate loan_count</th>
      <th>bureau_CREDIT_TYPE_Real estate loan_count_norm</th>
      <th>bureau_CREDIT_TYPE_Unknown type of loan_count</th>
      <th>bureau_CREDIT_TYPE_Unknown type of loan_count_norm</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100001</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>135000.0</td>
      <td>568800.0</td>
      <td>20560.5</td>
      <td>450000.0</td>
      <td>0.018850</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100005</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>99000.0</td>
      <td>222768.0</td>
      <td>17370.0</td>
      <td>180000.0</td>
      <td>0.035792</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>100013</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>202500.0</td>
      <td>663264.0</td>
      <td>69777.0</td>
      <td>630000.0</td>
      <td>0.019101</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>100028</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>315000.0</td>
      <td>1575000.0</td>
      <td>49018.5</td>
      <td>1575000.0</td>
      <td>0.026392</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>100038</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>180000.0</td>
      <td>625500.0</td>
      <td>32067.0</td>
      <td>625500.0</td>
      <td>0.010032</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 348 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-cb52c3a9-a09a-4daf-bf1c-3679485d277f')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-cb52c3a9-a09a-4daf-bf1c-3679485d277f button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-cb52c3a9-a09a-4daf-bf1c-3679485d277f');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-ae5c5220-5a10-4466-9f3e-f8224ae6fe92">
  <button class="colab-df-quickchart" onclick="quickchart('df-ae5c5220-5a10-4466-9f3e-f8224ae6fe92')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-ae5c5220-5a10-4466-9f3e-f8224ae6fe92 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>
    </div>
  </div>





```python
del bureau_agg, bureau_counts, bureau, bureau_new
gc.collect()
```




    0



### Load Data Previous Application


```python
prev_app = pd.read_csv('/content/gdrive/MyDrive/RAKAMIN/previous_application.csv')
prev_app.head()
```





  <div id="df-0a2923c2-7048-453a-bfec-59e4e28dc6a8" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SK_ID_PREV</th>
      <th>SK_ID_CURR</th>
      <th>NAME_CONTRACT_TYPE</th>
      <th>AMT_ANNUITY</th>
      <th>AMT_APPLICATION</th>
      <th>AMT_CREDIT</th>
      <th>AMT_DOWN_PAYMENT</th>
      <th>AMT_GOODS_PRICE</th>
      <th>WEEKDAY_APPR_PROCESS_START</th>
      <th>HOUR_APPR_PROCESS_START</th>
      <th>...</th>
      <th>NAME_SELLER_INDUSTRY</th>
      <th>CNT_PAYMENT</th>
      <th>NAME_YIELD_GROUP</th>
      <th>PRODUCT_COMBINATION</th>
      <th>DAYS_FIRST_DRAWING</th>
      <th>DAYS_FIRST_DUE</th>
      <th>DAYS_LAST_DUE_1ST_VERSION</th>
      <th>DAYS_LAST_DUE</th>
      <th>DAYS_TERMINATION</th>
      <th>NFLAG_INSURED_ON_APPROVAL</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2030495</td>
      <td>271877</td>
      <td>Consumer loans</td>
      <td>1730.430</td>
      <td>17145.0</td>
      <td>17145.0</td>
      <td>0.0</td>
      <td>17145.0</td>
      <td>SATURDAY</td>
      <td>15</td>
      <td>...</td>
      <td>Connectivity</td>
      <td>12.0</td>
      <td>middle</td>
      <td>POS mobile with interest</td>
      <td>365243.0</td>
      <td>-42.0</td>
      <td>300.0</td>
      <td>-42.0</td>
      <td>-37.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2802425</td>
      <td>108129</td>
      <td>Cash loans</td>
      <td>25188.615</td>
      <td>607500.0</td>
      <td>679671.0</td>
      <td>NaN</td>
      <td>607500.0</td>
      <td>THURSDAY</td>
      <td>11</td>
      <td>...</td>
      <td>XNA</td>
      <td>36.0</td>
      <td>low_action</td>
      <td>Cash X-Sell: low</td>
      <td>365243.0</td>
      <td>-134.0</td>
      <td>916.0</td>
      <td>365243.0</td>
      <td>365243.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2523466</td>
      <td>122040</td>
      <td>Cash loans</td>
      <td>15060.735</td>
      <td>112500.0</td>
      <td>136444.5</td>
      <td>NaN</td>
      <td>112500.0</td>
      <td>TUESDAY</td>
      <td>11</td>
      <td>...</td>
      <td>XNA</td>
      <td>12.0</td>
      <td>high</td>
      <td>Cash X-Sell: high</td>
      <td>365243.0</td>
      <td>-271.0</td>
      <td>59.0</td>
      <td>365243.0</td>
      <td>365243.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2819243</td>
      <td>176158</td>
      <td>Cash loans</td>
      <td>47041.335</td>
      <td>450000.0</td>
      <td>470790.0</td>
      <td>NaN</td>
      <td>450000.0</td>
      <td>MONDAY</td>
      <td>7</td>
      <td>...</td>
      <td>XNA</td>
      <td>12.0</td>
      <td>middle</td>
      <td>Cash X-Sell: middle</td>
      <td>365243.0</td>
      <td>-482.0</td>
      <td>-152.0</td>
      <td>-182.0</td>
      <td>-177.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1784265</td>
      <td>202054</td>
      <td>Cash loans</td>
      <td>31924.395</td>
      <td>337500.0</td>
      <td>404055.0</td>
      <td>NaN</td>
      <td>337500.0</td>
      <td>THURSDAY</td>
      <td>9</td>
      <td>...</td>
      <td>XNA</td>
      <td>24.0</td>
      <td>high</td>
      <td>Cash Street: high</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 37 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-0a2923c2-7048-453a-bfec-59e4e28dc6a8')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-0a2923c2-7048-453a-bfec-59e4e28dc6a8 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-0a2923c2-7048-453a-bfec-59e4e28dc6a8');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-02e1393b-2ec6-455a-a43e-c97fded3a517">
  <button class="colab-df-quickchart" onclick="quickchart('df-02e1393b-2ec6-455a-a43e-c97fded3a517')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-02e1393b-2ec6-455a-a43e-c97fded3a517 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>
    </div>
  </div>




#### Outlier


```python
# Days 365.243 values -> nan
prev_app['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace= True)
prev_app['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace= True)
prev_app['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace= True)
prev_app['DAYS_LAST_DUE'].replace(365243, np.nan, inplace= True)
prev_app['DAYS_TERMINATION'].replace(365243, np.nan, inplace= True)
```

#### Aggregate


```python
prev_app_agg = agg_df(prev_app.drop(columns = ['SK_ID_PREV']), group_by = 'SK_ID_CURR', df_name = 'prev_app')
prev_app_agg.head()
```





  <div id="df-2e7b4032-e3cc-4d3d-a1fc-e11f91b186d9" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SK_ID_CURR</th>
      <th>prev_app_AMT_ANNUITY_count</th>
      <th>prev_app_AMT_ANNUITY_mean</th>
      <th>prev_app_AMT_ANNUITY_max</th>
      <th>prev_app_AMT_ANNUITY_min</th>
      <th>prev_app_AMT_ANNUITY_sum</th>
      <th>prev_app_AMT_APPLICATION_count</th>
      <th>prev_app_AMT_APPLICATION_mean</th>
      <th>prev_app_AMT_APPLICATION_max</th>
      <th>prev_app_AMT_APPLICATION_min</th>
      <th>...</th>
      <th>prev_app_DAYS_TERMINATION_count</th>
      <th>prev_app_DAYS_TERMINATION_mean</th>
      <th>prev_app_DAYS_TERMINATION_max</th>
      <th>prev_app_DAYS_TERMINATION_min</th>
      <th>prev_app_DAYS_TERMINATION_sum</th>
      <th>prev_app_NFLAG_INSURED_ON_APPROVAL_count</th>
      <th>prev_app_NFLAG_INSURED_ON_APPROVAL_mean</th>
      <th>prev_app_NFLAG_INSURED_ON_APPROVAL_max</th>
      <th>prev_app_NFLAG_INSURED_ON_APPROVAL_min</th>
      <th>prev_app_NFLAG_INSURED_ON_APPROVAL_sum</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100001</td>
      <td>1</td>
      <td>3951.000</td>
      <td>3951.000</td>
      <td>3951.000</td>
      <td>3951.000</td>
      <td>1</td>
      <td>24835.50</td>
      <td>24835.5</td>
      <td>24835.5</td>
      <td>...</td>
      <td>1</td>
      <td>-1612.000000</td>
      <td>-1612.0</td>
      <td>-1612.0</td>
      <td>-1612.0</td>
      <td>1</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100002</td>
      <td>1</td>
      <td>9251.775</td>
      <td>9251.775</td>
      <td>9251.775</td>
      <td>9251.775</td>
      <td>1</td>
      <td>179055.00</td>
      <td>179055.0</td>
      <td>179055.0</td>
      <td>...</td>
      <td>1</td>
      <td>-17.000000</td>
      <td>-17.0</td>
      <td>-17.0</td>
      <td>-17.0</td>
      <td>1</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>100003</td>
      <td>3</td>
      <td>56553.990</td>
      <td>98356.995</td>
      <td>6737.310</td>
      <td>169661.970</td>
      <td>3</td>
      <td>435436.50</td>
      <td>900000.0</td>
      <td>68809.5</td>
      <td>...</td>
      <td>3</td>
      <td>-1047.333333</td>
      <td>-527.0</td>
      <td>-1976.0</td>
      <td>-3142.0</td>
      <td>3</td>
      <td>0.666667</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>100004</td>
      <td>1</td>
      <td>5357.250</td>
      <td>5357.250</td>
      <td>5357.250</td>
      <td>5357.250</td>
      <td>1</td>
      <td>24282.00</td>
      <td>24282.0</td>
      <td>24282.0</td>
      <td>...</td>
      <td>1</td>
      <td>-714.000000</td>
      <td>-714.0</td>
      <td>-714.0</td>
      <td>-714.0</td>
      <td>1</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>100005</td>
      <td>1</td>
      <td>4813.200</td>
      <td>4813.200</td>
      <td>4813.200</td>
      <td>4813.200</td>
      <td>2</td>
      <td>22308.75</td>
      <td>44617.5</td>
      <td>0.0</td>
      <td>...</td>
      <td>1</td>
      <td>-460.000000</td>
      <td>-460.0</td>
      <td>-460.0</td>
      <td>-460.0</td>
      <td>1</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 96 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-2e7b4032-e3cc-4d3d-a1fc-e11f91b186d9')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-2e7b4032-e3cc-4d3d-a1fc-e11f91b186d9 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-2e7b4032-e3cc-4d3d-a1fc-e11f91b186d9');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-b81bf621-9edc-4efc-9bc6-6d8830a31f7e">
  <button class="colab-df-quickchart" onclick="quickchart('df-b81bf621-9edc-4efc-9bc6-6d8830a31f7e')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-b81bf621-9edc-4efc-9bc6-6d8830a31f7e button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>
    </div>
  </div>





```python
prev_app_count = count_categorical(prev_app, group_by = 'SK_ID_CURR', df_name = 'prev_app')
prev_app_count.head()
```





  <div id="df-adfbda11-b364-4e56-845e-43465c3e8c79" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>prev_app_NAME_CONTRACT_TYPE_Cash loans_count</th>
      <th>prev_app_NAME_CONTRACT_TYPE_Cash loans_count_norm</th>
      <th>prev_app_NAME_CONTRACT_TYPE_Consumer loans_count</th>
      <th>prev_app_NAME_CONTRACT_TYPE_Consumer loans_count_norm</th>
      <th>prev_app_NAME_CONTRACT_TYPE_Revolving loans_count</th>
      <th>prev_app_NAME_CONTRACT_TYPE_Revolving loans_count_norm</th>
      <th>prev_app_NAME_CONTRACT_TYPE_XNA_count</th>
      <th>prev_app_NAME_CONTRACT_TYPE_XNA_count_norm</th>
      <th>prev_app_WEEKDAY_APPR_PROCESS_START_FRIDAY_count</th>
      <th>prev_app_WEEKDAY_APPR_PROCESS_START_FRIDAY_count_norm</th>
      <th>...</th>
      <th>prev_app_PRODUCT_COMBINATION_POS industry without interest_count</th>
      <th>prev_app_PRODUCT_COMBINATION_POS industry without interest_count_norm</th>
      <th>prev_app_PRODUCT_COMBINATION_POS mobile with interest_count</th>
      <th>prev_app_PRODUCT_COMBINATION_POS mobile with interest_count_norm</th>
      <th>prev_app_PRODUCT_COMBINATION_POS mobile without interest_count</th>
      <th>prev_app_PRODUCT_COMBINATION_POS mobile without interest_count_norm</th>
      <th>prev_app_PRODUCT_COMBINATION_POS other with interest_count</th>
      <th>prev_app_PRODUCT_COMBINATION_POS other with interest_count_norm</th>
      <th>prev_app_PRODUCT_COMBINATION_POS others without interest_count</th>
      <th>prev_app_PRODUCT_COMBINATION_POS others without interest_count_norm</th>
    </tr>
    <tr>
      <th>SK_ID_CURR</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>100001</th>
      <td>0</td>
      <td>0.000000</td>
      <td>1</td>
      <td>1.000000</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>1</td>
      <td>1.000000</td>
      <td>...</td>
      <td>0</td>
      <td>0.0</td>
      <td>1</td>
      <td>1.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>100002</th>
      <td>0</td>
      <td>0.000000</td>
      <td>1</td>
      <td>1.000000</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>1</td>
      <td>1.0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>100003</th>
      <td>1</td>
      <td>0.333333</td>
      <td>2</td>
      <td>0.666667</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0.333333</td>
      <td>...</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>100004</th>
      <td>0</td>
      <td>0.000000</td>
      <td>1</td>
      <td>1.000000</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>1</td>
      <td>1.000000</td>
      <td>...</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>1</td>
      <td>1.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>100005</th>
      <td>1</td>
      <td>0.500000</td>
      <td>1</td>
      <td>0.500000</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0.500000</td>
      <td>...</td>
      <td>0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0.5</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 286 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-adfbda11-b364-4e56-845e-43465c3e8c79')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-adfbda11-b364-4e56-845e-43465c3e8c79 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-adfbda11-b364-4e56-845e-43465c3e8c79');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-5bb7de38-9c61-4592-a0dd-ec26eb338ea7">
  <button class="colab-df-quickchart" onclick="quickchart('df-5bb7de38-9c61-4592-a0dd-ec26eb338ea7')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-5bb7de38-9c61-4592-a0dd-ec26eb338ea7 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>
    </div>
  </div>





```python
prev_app_new = prev_app_agg.merge(prev_app_count, right_index = True, left_on = 'SK_ID_CURR', how = 'outer')

# Merge to include the SK_ID_CURR
# prev_app_new = prev_app_new.merge(prev_app[['SK_ID_PREV', 'SK_ID_CURR']], on = 'SK_ID_PREV', how = 'left')
prev_app_new.head()
```





  <div id="df-96e7a7d6-4906-4ae7-974b-7f37d4a47008" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SK_ID_CURR</th>
      <th>prev_app_AMT_ANNUITY_count</th>
      <th>prev_app_AMT_ANNUITY_mean</th>
      <th>prev_app_AMT_ANNUITY_max</th>
      <th>prev_app_AMT_ANNUITY_min</th>
      <th>prev_app_AMT_ANNUITY_sum</th>
      <th>prev_app_AMT_APPLICATION_count</th>
      <th>prev_app_AMT_APPLICATION_mean</th>
      <th>prev_app_AMT_APPLICATION_max</th>
      <th>prev_app_AMT_APPLICATION_min</th>
      <th>...</th>
      <th>prev_app_PRODUCT_COMBINATION_POS industry without interest_count</th>
      <th>prev_app_PRODUCT_COMBINATION_POS industry without interest_count_norm</th>
      <th>prev_app_PRODUCT_COMBINATION_POS mobile with interest_count</th>
      <th>prev_app_PRODUCT_COMBINATION_POS mobile with interest_count_norm</th>
      <th>prev_app_PRODUCT_COMBINATION_POS mobile without interest_count</th>
      <th>prev_app_PRODUCT_COMBINATION_POS mobile without interest_count_norm</th>
      <th>prev_app_PRODUCT_COMBINATION_POS other with interest_count</th>
      <th>prev_app_PRODUCT_COMBINATION_POS other with interest_count_norm</th>
      <th>prev_app_PRODUCT_COMBINATION_POS others without interest_count</th>
      <th>prev_app_PRODUCT_COMBINATION_POS others without interest_count_norm</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100001</td>
      <td>1</td>
      <td>3951.000</td>
      <td>3951.000</td>
      <td>3951.000</td>
      <td>3951.000</td>
      <td>1</td>
      <td>24835.50</td>
      <td>24835.5</td>
      <td>24835.5</td>
      <td>...</td>
      <td>0</td>
      <td>0.0</td>
      <td>1</td>
      <td>1.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100002</td>
      <td>1</td>
      <td>9251.775</td>
      <td>9251.775</td>
      <td>9251.775</td>
      <td>9251.775</td>
      <td>1</td>
      <td>179055.00</td>
      <td>179055.0</td>
      <td>179055.0</td>
      <td>...</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>1</td>
      <td>1.0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>100003</td>
      <td>3</td>
      <td>56553.990</td>
      <td>98356.995</td>
      <td>6737.310</td>
      <td>169661.970</td>
      <td>3</td>
      <td>435436.50</td>
      <td>900000.0</td>
      <td>68809.5</td>
      <td>...</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>100004</td>
      <td>1</td>
      <td>5357.250</td>
      <td>5357.250</td>
      <td>5357.250</td>
      <td>5357.250</td>
      <td>1</td>
      <td>24282.00</td>
      <td>24282.0</td>
      <td>24282.0</td>
      <td>...</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>1</td>
      <td>1.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>100005</td>
      <td>1</td>
      <td>4813.200</td>
      <td>4813.200</td>
      <td>4813.200</td>
      <td>4813.200</td>
      <td>2</td>
      <td>22308.75</td>
      <td>44617.5</td>
      <td>0.0</td>
      <td>...</td>
      <td>0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0.5</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 382 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-96e7a7d6-4906-4ae7-974b-7f37d4a47008')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-96e7a7d6-4906-4ae7-974b-7f37d4a47008 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-96e7a7d6-4906-4ae7-974b-7f37d4a47008');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-ec190d8b-f8c5-48ff-87c8-8d820e73797b">
  <button class="colab-df-quickchart" onclick="quickchart('df-ec190d8b-f8c5-48ff-87c8-8d820e73797b')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-ec190d8b-f8c5-48ff-87c8-8d820e73797b button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>
    </div>
  </div>





```python
train_df.head()
```





  <div id="df-e3f4f5b9-04ae-49a6-9e20-3b3dec8e081b" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SK_ID_CURR</th>
      <th>NAME_CONTRACT_TYPE</th>
      <th>FLAG_OWN_CAR</th>
      <th>FLAG_OWN_REALTY</th>
      <th>CNT_CHILDREN</th>
      <th>AMT_INCOME_TOTAL</th>
      <th>AMT_CREDIT</th>
      <th>AMT_ANNUITY</th>
      <th>AMT_GOODS_PRICE</th>
      <th>REGION_POPULATION_RELATIVE</th>
      <th>...</th>
      <th>bureau_CREDIT_TYPE_Microloan_count</th>
      <th>bureau_CREDIT_TYPE_Microloan_count_norm</th>
      <th>bureau_CREDIT_TYPE_Mobile operator loan_count</th>
      <th>bureau_CREDIT_TYPE_Mobile operator loan_count_norm</th>
      <th>bureau_CREDIT_TYPE_Mortgage_count</th>
      <th>bureau_CREDIT_TYPE_Mortgage_count_norm</th>
      <th>bureau_CREDIT_TYPE_Real estate loan_count</th>
      <th>bureau_CREDIT_TYPE_Real estate loan_count_norm</th>
      <th>bureau_CREDIT_TYPE_Unknown type of loan_count</th>
      <th>bureau_CREDIT_TYPE_Unknown type of loan_count_norm</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100002</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>202500.0</td>
      <td>406597.5</td>
      <td>24700.5</td>
      <td>351000.0</td>
      <td>0.018801</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100003</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>270000.0</td>
      <td>1293502.5</td>
      <td>35698.5</td>
      <td>1129500.0</td>
      <td>0.003541</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>100004</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>67500.0</td>
      <td>135000.0</td>
      <td>6750.0</td>
      <td>135000.0</td>
      <td>0.010032</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>100006</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>135000.0</td>
      <td>312682.5</td>
      <td>29686.5</td>
      <td>297000.0</td>
      <td>0.008019</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>100007</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>121500.0</td>
      <td>513000.0</td>
      <td>21865.5</td>
      <td>513000.0</td>
      <td>0.028663</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 349 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-e3f4f5b9-04ae-49a6-9e20-3b3dec8e081b')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-e3f4f5b9-04ae-49a6-9e20-3b3dec8e081b button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-e3f4f5b9-04ae-49a6-9e20-3b3dec8e081b');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-6add5d48-f817-4de8-bd0b-d243aaa76484">
  <button class="colab-df-quickchart" onclick="quickchart('df-6add5d48-f817-4de8-bd0b-d243aaa76484')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-6add5d48-f817-4de8-bd0b-d243aaa76484 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>
    </div>
  </div>





```python
print("-----------------------")
# Merge with bureau
train_df = train_df.merge(prev_app_new, on = 'SK_ID_CURR', how = 'left')
train_df.head()
```

    -----------------------
    





  <div id="df-39423114-6f36-4597-a2bb-8839c98e6c03" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SK_ID_CURR</th>
      <th>NAME_CONTRACT_TYPE</th>
      <th>FLAG_OWN_CAR</th>
      <th>FLAG_OWN_REALTY</th>
      <th>CNT_CHILDREN</th>
      <th>AMT_INCOME_TOTAL</th>
      <th>AMT_CREDIT</th>
      <th>AMT_ANNUITY</th>
      <th>AMT_GOODS_PRICE</th>
      <th>REGION_POPULATION_RELATIVE</th>
      <th>...</th>
      <th>prev_app_PRODUCT_COMBINATION_POS industry without interest_count</th>
      <th>prev_app_PRODUCT_COMBINATION_POS industry without interest_count_norm</th>
      <th>prev_app_PRODUCT_COMBINATION_POS mobile with interest_count</th>
      <th>prev_app_PRODUCT_COMBINATION_POS mobile with interest_count_norm</th>
      <th>prev_app_PRODUCT_COMBINATION_POS mobile without interest_count</th>
      <th>prev_app_PRODUCT_COMBINATION_POS mobile without interest_count_norm</th>
      <th>prev_app_PRODUCT_COMBINATION_POS other with interest_count</th>
      <th>prev_app_PRODUCT_COMBINATION_POS other with interest_count_norm</th>
      <th>prev_app_PRODUCT_COMBINATION_POS others without interest_count</th>
      <th>prev_app_PRODUCT_COMBINATION_POS others without interest_count_norm</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100002</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>202500.0</td>
      <td>406597.5</td>
      <td>24700.5</td>
      <td>351000.0</td>
      <td>0.018801</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100003</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>270000.0</td>
      <td>1293502.5</td>
      <td>35698.5</td>
      <td>1129500.0</td>
      <td>0.003541</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>100004</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>67500.0</td>
      <td>135000.0</td>
      <td>6750.0</td>
      <td>135000.0</td>
      <td>0.010032</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>100006</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>135000.0</td>
      <td>312682.5</td>
      <td>29686.5</td>
      <td>297000.0</td>
      <td>0.008019</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>100007</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>121500.0</td>
      <td>513000.0</td>
      <td>21865.5</td>
      <td>513000.0</td>
      <td>0.028663</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.166667</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 730 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-39423114-6f36-4597-a2bb-8839c98e6c03')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-39423114-6f36-4597-a2bb-8839c98e6c03 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-39423114-6f36-4597-a2bb-8839c98e6c03');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-4a40863d-83b6-4111-8ad6-7fb32129eabe">
  <button class="colab-df-quickchart" onclick="quickchart('df-4a40863d-83b6-4111-8ad6-7fb32129eabe')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-4a40863d-83b6-4111-8ad6-7fb32129eabe button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>
    </div>
  </div>





```python
test_df = test_df.merge(prev_app_new, on = 'SK_ID_CURR', how = 'left')
test_df.head()
```





  <div id="df-9dab5df0-8930-4b3f-a19c-802833d56a54" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SK_ID_CURR</th>
      <th>NAME_CONTRACT_TYPE</th>
      <th>FLAG_OWN_CAR</th>
      <th>FLAG_OWN_REALTY</th>
      <th>CNT_CHILDREN</th>
      <th>AMT_INCOME_TOTAL</th>
      <th>AMT_CREDIT</th>
      <th>AMT_ANNUITY</th>
      <th>AMT_GOODS_PRICE</th>
      <th>REGION_POPULATION_RELATIVE</th>
      <th>...</th>
      <th>prev_app_PRODUCT_COMBINATION_POS industry without interest_count</th>
      <th>prev_app_PRODUCT_COMBINATION_POS industry without interest_count_norm</th>
      <th>prev_app_PRODUCT_COMBINATION_POS mobile with interest_count</th>
      <th>prev_app_PRODUCT_COMBINATION_POS mobile with interest_count_norm</th>
      <th>prev_app_PRODUCT_COMBINATION_POS mobile without interest_count</th>
      <th>prev_app_PRODUCT_COMBINATION_POS mobile without interest_count_norm</th>
      <th>prev_app_PRODUCT_COMBINATION_POS other with interest_count</th>
      <th>prev_app_PRODUCT_COMBINATION_POS other with interest_count_norm</th>
      <th>prev_app_PRODUCT_COMBINATION_POS others without interest_count</th>
      <th>prev_app_PRODUCT_COMBINATION_POS others without interest_count_norm</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100001</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>135000.0</td>
      <td>568800.0</td>
      <td>20560.5</td>
      <td>450000.0</td>
      <td>0.018850</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100005</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>99000.0</td>
      <td>222768.0</td>
      <td>17370.0</td>
      <td>180000.0</td>
      <td>0.035792</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.50</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>100013</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>202500.0</td>
      <td>663264.0</td>
      <td>69777.0</td>
      <td>630000.0</td>
      <td>0.019101</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.25</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>100028</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>315000.0</td>
      <td>1575000.0</td>
      <td>49018.5</td>
      <td>1575000.0</td>
      <td>0.026392</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.20</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>100038</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>180000.0</td>
      <td>625500.0</td>
      <td>32067.0</td>
      <td>625500.0</td>
      <td>0.010032</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.50</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 729 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-9dab5df0-8930-4b3f-a19c-802833d56a54')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-9dab5df0-8930-4b3f-a19c-802833d56a54 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-9dab5df0-8930-4b3f-a19c-802833d56a54');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-9211eea1-c66d-42e1-bea9-4081fb6bec9b">
  <button class="colab-df-quickchart" onclick="quickchart('df-9211eea1-c66d-42e1-bea9-4081fb6bec9b')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-9211eea1-c66d-42e1-bea9-4081fb6bec9b button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>
    </div>
  </div>





```python
del prev_app_agg, prev_app_count, prev_app, prev_app_new
gc.collect()
```




    0



## Feature Selection

### Drop variabel collinear
Beberapa variabbel dapat memiliki korelasi yang terlalu besar dengan variabel lainnya. Hal ini bisa mengurangi performa dari model yang dibangun.

*Note: Cukup lama jadi run sekali saja tiap ada perubahan lagi ke dataset*


```python
# Calculate all correlations in dataframe
corrs = train_df.corr()
corrs = corrs.sort_values('TARGET', ascending = False)
```





  <div id="df-4e6667ad-b191-44b7-a2a0-086ba1d968ea" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TARGET</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>TARGET</th>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>prev_app_DAYS_FIRST_DRAWING_max</th>
      <td>0.096222</td>
    </tr>
    <tr>
      <th>prev_app_DAYS_FIRST_DRAWING_mean</th>
      <td>0.096119</td>
    </tr>
    <tr>
      <th>prev_app_DAYS_FIRST_DRAWING_min</th>
      <td>0.095787</td>
    </tr>
    <tr>
      <th>bureau_DAYS_CREDIT_mean</th>
      <td>0.089729</td>
    </tr>
    <tr>
      <th>DAYS_BIRTH</th>
      <td>0.078239</td>
    </tr>
    <tr>
      <th>prev_app_NAME_CONTRACT_STATUS_Refused_count_norm</th>
      <td>0.077671</td>
    </tr>
    <tr>
      <th>bureau_CREDIT_ACTIVE_Active_count_norm</th>
      <td>0.077356</td>
    </tr>
    <tr>
      <th>bureau_DAYS_CREDIT_min</th>
      <td>0.075248</td>
    </tr>
    <tr>
      <th>DAYS_EMPLOYED</th>
      <td>0.074958</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-4e6667ad-b191-44b7-a2a0-086ba1d968ea')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-4e6667ad-b191-44b7-a2a0-086ba1d968ea button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-4e6667ad-b191-44b7-a2a0-086ba1d968ea');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-07d53ebf-5ac1-45f7-b4e7-ef45de3439f5">
  <button class="colab-df-quickchart" onclick="quickchart('df-07d53ebf-5ac1-45f7-b4e7-ef45de3439f5')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-07d53ebf-5ac1-45f7-b4e7-ef45de3439f5 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>
    </div>
  </div>





```python
# Set the threshold
threshold = 0.8

# Empty dictionary to hold correlated variables
above_threshold_vars = {}

# For each column, record the variables that are above the threshold
for col in corrs:
    above_threshold_vars[col] = list(corrs.index[corrs[col] > threshold])
```


```python
# Track columns to remove and columns already examined
cols_to_remove = []
cols_seen = []
cols_to_remove_pair = []

# Iterate through columns and correlated columns
for key, value in above_threshold_vars.items():
    # Keep track of columns already examined
    cols_seen.append(key)
    for x in value:
        if x == key:
            continue
        else:
            # Only want to remove one in a pair
            if x not in cols_seen:
                cols_to_remove.append(x)
                cols_to_remove_pair.append(key)

cols_to_remove = list(set(cols_to_remove))
print('Number of columns to remove: ', len(cols_to_remove))
```

    Number of columns to remove:  196
    


```python
print(cols_to_remove)
```

    ['prev_app_RATE_DOWN_PAYMENT_count', 'prev_app_DAYS_TERMINATION_count', 'prev_app_AMT_GOODS_PRICE_max', 'bureau_CREDIT_TYPE_Mobile operator loan_count_norm', 'prev_app_NAME_GOODS_CATEGORY_XNA_count_norm', 'bureau_AMT_CREDIT_MAX_OVERDUE_sum', 'bureau_CREDIT_TYPE_Interbank credit_count_norm', 'prev_app_NAME_CASH_LOAN_PURPOSE_XAP_count', 'prev_app_NAME_PORTFOLIO_POS_count_norm', 'prev_app_CHANNEL_TYPE_Car dealer_count_norm', 'prev_app_NAME_CASH_LOAN_PURPOSE_XNA_count', 'APARTMENTS_MEDI', 'prev_app_DAYS_FIRST_DRAWING_sum', 'prev_app_HOUR_APPR_PROCESS_START_count', 'prev_app_RATE_INTEREST_PRIMARY_min', 'prev_app_DAYS_TERMINATION_sum', 'LANDAREA_MODE', 'prev_app_NAME_PORTFOLIO_Cards_count', 'prev_app_AMT_DOWN_PAYMENT_max', 'prev_app_PRODUCT_COMBINATION_Card X-Sell_count_norm', 'bureau_CREDIT_TYPE_Cash loan (non-earmarked)_count_norm', 'prev_app_AMT_APPLICATION_max', 'prev_app_NAME_PORTFOLIO_Cash_count', 'prev_app_NAME_CONTRACT_TYPE_Consumer loans_count', 'prev_app_RATE_DOWN_PAYMENT_min', 'prev_app_DAYS_LAST_DUE_count', 'bureau_CREDIT_ACTIVE_Bad debt_count_norm', 'prev_app_NAME_SELLER_INDUSTRY_Clothing_count_norm', 'NONLIVINGAPARTMENTS_MODE', 'bureau_DAYS_CREDIT_ENDDATE_count', 'prev_app_AMT_CREDIT_sum', 'prev_app_AMT_GOODS_PRICE_sum', 'prev_app_CODE_REJECT_REASON_CLIENT_count', 'prev_app_DAYS_LAST_DUE_min', 'bureau_AMT_CREDIT_MAX_OVERDUE_max', 'prev_app_PRODUCT_COMBINATION_Card Street_count', 'prev_app_NAME_PAYMENT_TYPE_XNA_count', 'LIVINGAPARTMENTS_MEDI', 'prev_app_NAME_SELLER_INDUSTRY_Furniture_count_norm', 'bureau_AMT_CREDIT_SUM_DEBT_sum', 'prev_app_NAME_YIELD_GROUP_XNA_count', 'LIVE_REGION_NOT_WORK_REGION', 'prev_app_CODE_REJECT_REASON_XAP_count', 'bureau_DAYS_ENDDATE_FACT_sum', 'LANDAREA_MEDI', 'prev_app_NAME_SELLER_INDUSTRY_Clothing_count', 'ELEVATORS_AVG', 'prev_app_FLAG_LAST_APPL_PER_CONTRACT_N_count_norm', 'prev_app_AMT_DOWN_PAYMENT_sum', 'prev_app_NAME_SELLER_INDUSTRY_Connectivity_count', 'FLOORSMIN_MEDI', 'prev_app_NAME_PORTFOLIO_Cards_count_norm', 'prev_app_NAME_PRODUCT_TYPE_XNA_count_norm', 'EMERGENCYSTATE_MODE_No', 'bureau_AMT_CREDIT_SUM_sum', 'prev_app_DAYS_TERMINATION_min', 'prev_app_NAME_PAYMENT_TYPE_Cash through the bank_count', 'LIVINGAREA_MODE', 'NONLIVINGAREA_MODE', 'OBS_60_CNT_SOCIAL_CIRCLE', 'bureau_AMT_CREDIT_SUM_LIMIT_count', 'prev_app_PRODUCT_COMBINATION_POS mobile with interest_count_norm', 'prev_app_NAME_CONTRACT_STATUS_Approved_count', 'bureau_CNT_CREDIT_PROLONG_count', 'prev_app_DAYS_LAST_DUE_1ST_VERSION_min', 'bureau_AMT_ANNUITY_sum', 'bureau_AMT_CREDIT_SUM_OVERDUE_sum', 'FLOORSMAX_MODE', 'prev_app_PRODUCT_COMBINATION_Cash_count', 'prev_app_NAME_PORTFOLIO_XNA_count', 'prev_app_RATE_INTEREST_PRIMARY_sum', 'prev_app_DAYS_FIRST_DUE_min', 'prev_app_NAME_CONTRACT_TYPE_XNA_count_norm', 'prev_app_NFLAG_INSURED_ON_APPROVAL_max', 'bureau_DAYS_CREDIT_min', 'prev_app_CODE_REJECT_REASON_XNA_count_norm', 'prev_app_CODE_REJECT_REASON_HC_count', 'YEARS_BUILD_MODE', 'prev_app_NFLAG_LAST_APPL_IN_DAY_count', 'ELEVATORS_MEDI', 'prev_app_DAYS_FIRST_DRAWING_max', 'YEARS_BEGINEXPLUATATION_MODE', 'prev_app_AMT_GOODS_PRICE_count', 'bureau_DAYS_ENDDATE_FACT_min', 'bureau_CREDIT_CURRENCY_currency 4_count_norm', 'TOTALAREA_MODE', 'prev_app_NAME_CLIENT_TYPE_Repeater_count', 'prev_app_AMT_GOODS_PRICE_mean', 'CNT_FAM_MEMBERS', 'bureau_CREDIT_TYPE_Loan for purchase of shares (margin lending)_count_norm', 'bureau_CREDIT_TYPE_Consumer credit_count', 'bureau_DAYS_ENDDATE_FACT_count', 'bureau_AMT_CREDIT_SUM_max', 'REGION_RATING_CLIENT_W_CITY', 'FLOORSMAX_MEDI', 'prev_app_RATE_DOWN_PAYMENT_max', 'DEF_60_CNT_SOCIAL_CIRCLE', 'prev_app_PRODUCT_COMBINATION_POS household with interest_count', 'prev_app_DAYS_FIRST_DUE_max', 'prev_app_DAYS_FIRST_DUE_count', 'prev_app_CNT_PAYMENT_sum', 'prev_app_DAYS_LAST_DUE_sum', 'prev_app_NAME_SELLER_INDUSTRY_Connectivity_count_norm', 'prev_app_AMT_CREDIT_min', 'bureau_CREDIT_DAY_OVERDUE_count', 'prev_app_NAME_PRODUCT_TYPE_XNA_count', 'prev_app_RATE_INTEREST_PRIVILEGED_min', 'prev_app_AMT_APPLICATION_sum', 'prev_app_AMT_CREDIT_count', 'prev_app_FLAG_LAST_APPL_PER_CONTRACT_Y_count', 'prev_app_HOUR_APPR_PROCESS_START_min', 'LIVINGAPARTMENTS_MODE', 'prev_app_NAME_SELLER_INDUSTRY_XNA_count_norm', 'bureau_AMT_CREDIT_MAX_OVERDUE_min', 'prev_app_CNT_PAYMENT_count', 'prev_app_RATE_INTEREST_PRIVILEGED_sum', 'BASEMENTAREA_MODE', 'bureau_CREDIT_CURRENCY_currency 1_count', 'prev_app_DAYS_FIRST_DUE_sum', 'prev_app_NAME_PORTFOLIO_Cars_count_norm', 'LIVE_CITY_NOT_WORK_CITY', 'prev_app_RATE_INTEREST_PRIMARY_max', 'LIVINGAREA_MEDI', 'prev_app_CHANNEL_TYPE_Credit and cash offices_count', 'prev_app_DAYS_TERMINATION_max', 'NONLIVINGAREA_MEDI', 'bureau_AMT_CREDIT_SUM_LIMIT_sum', 'prev_app_DAYS_FIRST_DUE_mean', 'prev_app_HOUR_APPR_PROCESS_START_max', 'prev_app_NAME_CASH_LOAN_PURPOSE_XNA_count_norm', 'prev_app_CODE_REJECT_REASON_CLIENT_count_norm', 'prev_app_AMT_APPLICATION_count', 'prev_app_CHANNEL_TYPE_Car dealer_count', 'prev_app_PRODUCT_COMBINATION_POS mobile with interest_count', 'prev_app_NAME_GOODS_CATEGORY_Animals_count_norm', 'prev_app_DAYS_LAST_DUE_mean', 'prev_app_NFLAG_INSURED_ON_APPROVAL_count', 'prev_app_DAYS_FIRST_DRAWING_min', 'prev_app_NAME_CASH_LOAN_PURPOSE_Hobby_count_norm', 'prev_app_NAME_PORTFOLIO_XNA_count_norm', 'bureau_CNT_CREDIT_PROLONG_sum', 'YEARS_BUILD_MEDI', 'APARTMENTS_MODE', 'bureau_DAYS_CREDIT_ENDDATE_sum', 'bureau_CREDIT_ACTIVE_Closed_count', 'prev_app_NAME_SELLER_INDUSTRY_Furniture_count', 'prev_app_NAME_SELLER_INDUSTRY_XNA_count', 'prev_app_NAME_GOODS_CATEGORY_XNA_count', 'prev_app_DAYS_LAST_DUE_1ST_VERSION_count', 'prev_app_NAME_PRODUCT_TYPE_x-sell_count', 'NONLIVINGAPARTMENTS_MEDI', 'prev_app_NAME_CASH_LOAN_PURPOSE_XAP_count_norm', 'prev_app_CODE_REJECT_REASON_SYSTEM_count_norm', 'ELEVATORS_MODE', 'prev_app_NAME_YIELD_GROUP_XNA_count_norm', 'prev_app_NAME_PORTFOLIO_POS_count', 'prev_app_DAYS_TERMINATION_mean', 'prev_app_DAYS_LAST_DUE_1ST_VERSION_mean', 'prev_app_DAYS_LAST_DUE_1ST_VERSION_max', 'prev_app_NAME_PORTFOLIO_Cash_count_norm', 'prev_app_RATE_INTEREST_PRIVILEGED_count', 'BASEMENTAREA_MEDI', 'AMT_GOODS_PRICE', 'ENTRANCES_MODE', 'prev_app_PRODUCT_COMBINATION_Cash_count_norm', 'LIVINGAPARTMENTS_AVG', 'bureau_AMT_CREDIT_SUM_count', 'prev_app_RATE_DOWN_PAYMENT_sum', 'prev_app_NAME_CONTRACT_STATUS_Unused offer_count_norm', 'ORGANIZATION_TYPE_XNA', 'bureau_DAYS_CREDIT_UPDATE_count', 'prev_app_HOUR_APPR_PROCESS_START_sum', 'prev_app_PRODUCT_COMBINATION_Card Street_count_norm', 'prev_app_AMT_CREDIT_max', 'prev_app_SELLERPLACE_AREA_sum', 'YEARS_BEGINEXPLUATATION_MEDI', 'bureau_AMT_CREDIT_SUM_OVERDUE_count', 'prev_app_AMT_DOWN_PAYMENT_min', 'ENTRANCES_MEDI', 'prev_app_CNT_PAYMENT_max', 'prev_app_SELLERPLACE_AREA_count', 'FLOORSMIN_MODE', 'prev_app_RATE_INTEREST_PRIVILEGED_max', 'prev_app_AMT_ANNUITY_max', 'bureau_DAYS_CREDIT_UPDATE_sum', 'prev_app_DAYS_LAST_DUE_1ST_VERSION_sum', 'bureau_CREDIT_DAY_OVERDUE_sum', 'prev_app_NFLAG_LAST_APPL_IN_DAY_sum', 'LIVINGAREA_AVG', 'COMMONAREA_MODE', 'COMMONAREA_MEDI', 'prev_app_DAYS_DECISION_count', 'bureau_AMT_CREDIT_SUM_DEBT_count', 'prev_app_SELLERPLACE_AREA_min', 'prev_app_NAME_CONTRACT_TYPE_Cash loans_count', 'prev_app_AMT_CREDIT_mean']
    

### Langsung ke sini


```python
cols_to_remove = ['prev_app_RATE_DOWN_PAYMENT_count', 'prev_app_DAYS_TERMINATION_count', 'prev_app_AMT_GOODS_PRICE_max', 'bureau_CREDIT_TYPE_Mobile operator loan_count_norm', 'prev_app_NAME_GOODS_CATEGORY_XNA_count_norm', 'bureau_AMT_CREDIT_MAX_OVERDUE_sum', 'bureau_CREDIT_TYPE_Interbank credit_count_norm', 'prev_app_NAME_CASH_LOAN_PURPOSE_XAP_count', 'prev_app_NAME_PORTFOLIO_POS_count_norm', 'prev_app_CHANNEL_TYPE_Car dealer_count_norm', 'prev_app_NAME_CASH_LOAN_PURPOSE_XNA_count', 'APARTMENTS_MEDI', 'prev_app_DAYS_FIRST_DRAWING_sum', 'prev_app_HOUR_APPR_PROCESS_START_count', 'prev_app_RATE_INTEREST_PRIMARY_min', 'prev_app_DAYS_TERMINATION_sum', 'LANDAREA_MODE', 'prev_app_NAME_PORTFOLIO_Cards_count', 'prev_app_AMT_DOWN_PAYMENT_max', 'prev_app_PRODUCT_COMBINATION_Card X-Sell_count_norm', 'bureau_CREDIT_TYPE_Cash loan (non-earmarked)_count_norm', 'prev_app_AMT_APPLICATION_max', 'prev_app_NAME_PORTFOLIO_Cash_count', 'prev_app_NAME_CONTRACT_TYPE_Consumer loans_count', 'prev_app_RATE_DOWN_PAYMENT_min', 'prev_app_DAYS_LAST_DUE_count', 'bureau_CREDIT_ACTIVE_Bad debt_count_norm', 'prev_app_NAME_SELLER_INDUSTRY_Clothing_count_norm', 'NONLIVINGAPARTMENTS_MODE', 'bureau_DAYS_CREDIT_ENDDATE_count', 'prev_app_AMT_CREDIT_sum', 'prev_app_AMT_GOODS_PRICE_sum', 'prev_app_CODE_REJECT_REASON_CLIENT_count', 'prev_app_DAYS_LAST_DUE_min', 'bureau_AMT_CREDIT_MAX_OVERDUE_max', 'prev_app_PRODUCT_COMBINATION_Card Street_count', 'prev_app_NAME_PAYMENT_TYPE_XNA_count', 'LIVINGAPARTMENTS_MEDI', 'prev_app_NAME_SELLER_INDUSTRY_Furniture_count_norm', 'bureau_AMT_CREDIT_SUM_DEBT_sum', 'prev_app_NAME_YIELD_GROUP_XNA_count', 'LIVE_REGION_NOT_WORK_REGION', 'prev_app_CODE_REJECT_REASON_XAP_count', 'bureau_DAYS_ENDDATE_FACT_sum', 'LANDAREA_MEDI', 'prev_app_NAME_SELLER_INDUSTRY_Clothing_count', 'ELEVATORS_AVG', 'prev_app_FLAG_LAST_APPL_PER_CONTRACT_N_count_norm', 'prev_app_AMT_DOWN_PAYMENT_sum', 'prev_app_NAME_SELLER_INDUSTRY_Connectivity_count', 'FLOORSMIN_MEDI', 'prev_app_NAME_PORTFOLIO_Cards_count_norm', 'prev_app_NAME_PRODUCT_TYPE_XNA_count_norm', 'EMERGENCYSTATE_MODE_No', 'bureau_AMT_CREDIT_SUM_sum', 'prev_app_DAYS_TERMINATION_min', 'prev_app_NAME_PAYMENT_TYPE_Cash through the bank_count', 'LIVINGAREA_MODE', 'NONLIVINGAREA_MODE', 'OBS_60_CNT_SOCIAL_CIRCLE', 'bureau_AMT_CREDIT_SUM_LIMIT_count', 'prev_app_PRODUCT_COMBINATION_POS mobile with interest_count_norm', 'prev_app_NAME_CONTRACT_STATUS_Approved_count', 'bureau_CNT_CREDIT_PROLONG_count', 'prev_app_DAYS_LAST_DUE_1ST_VERSION_min', 'bureau_AMT_ANNUITY_sum', 'bureau_AMT_CREDIT_SUM_OVERDUE_sum', 'FLOORSMAX_MODE', 'prev_app_PRODUCT_COMBINATION_Cash_count', 'prev_app_NAME_PORTFOLIO_XNA_count', 'prev_app_RATE_INTEREST_PRIMARY_sum', 'prev_app_DAYS_FIRST_DUE_min', 'prev_app_NAME_CONTRACT_TYPE_XNA_count_norm', 'prev_app_NFLAG_INSURED_ON_APPROVAL_max', 'bureau_DAYS_CREDIT_min', 'prev_app_CODE_REJECT_REASON_XNA_count_norm', 'prev_app_CODE_REJECT_REASON_HC_count', 'YEARS_BUILD_MODE', 'prev_app_NFLAG_LAST_APPL_IN_DAY_count', 'ELEVATORS_MEDI', 'prev_app_DAYS_FIRST_DRAWING_max', 'YEARS_BEGINEXPLUATATION_MODE', 'prev_app_AMT_GOODS_PRICE_count', 'bureau_DAYS_ENDDATE_FACT_min', 'bureau_CREDIT_CURRENCY_currency 4_count_norm', 'TOTALAREA_MODE', 'prev_app_NAME_CLIENT_TYPE_Repeater_count', 'prev_app_AMT_GOODS_PRICE_mean', 'CNT_FAM_MEMBERS', 'bureau_CREDIT_TYPE_Loan for purchase of shares (margin lending)_count_norm', 'bureau_CREDIT_TYPE_Consumer credit_count', 'bureau_DAYS_ENDDATE_FACT_count', 'bureau_AMT_CREDIT_SUM_max', 'REGION_RATING_CLIENT_W_CITY', 'FLOORSMAX_MEDI', 'prev_app_RATE_DOWN_PAYMENT_max', 'DEF_60_CNT_SOCIAL_CIRCLE', 'prev_app_PRODUCT_COMBINATION_POS household with interest_count', 'prev_app_DAYS_FIRST_DUE_max', 'prev_app_DAYS_FIRST_DUE_count', 'prev_app_CNT_PAYMENT_sum', 'prev_app_DAYS_LAST_DUE_sum', 'prev_app_NAME_SELLER_INDUSTRY_Connectivity_count_norm', 'prev_app_AMT_CREDIT_min', 'bureau_CREDIT_DAY_OVERDUE_count', 'prev_app_NAME_PRODUCT_TYPE_XNA_count', 'prev_app_RATE_INTEREST_PRIVILEGED_min', 'prev_app_AMT_APPLICATION_sum', 'prev_app_AMT_CREDIT_count', 'prev_app_FLAG_LAST_APPL_PER_CONTRACT_Y_count', 'prev_app_HOUR_APPR_PROCESS_START_min', 'LIVINGAPARTMENTS_MODE', 'prev_app_NAME_SELLER_INDUSTRY_XNA_count_norm', 'bureau_AMT_CREDIT_MAX_OVERDUE_min', 'prev_app_CNT_PAYMENT_count', 'prev_app_RATE_INTEREST_PRIVILEGED_sum', 'BASEMENTAREA_MODE', 'bureau_CREDIT_CURRENCY_currency 1_count', 'prev_app_DAYS_FIRST_DUE_sum', 'prev_app_NAME_PORTFOLIO_Cars_count_norm', 'LIVE_CITY_NOT_WORK_CITY', 'prev_app_RATE_INTEREST_PRIMARY_max', 'LIVINGAREA_MEDI', 'prev_app_CHANNEL_TYPE_Credit and cash offices_count', 'prev_app_DAYS_TERMINATION_max', 'NONLIVINGAREA_MEDI', 'bureau_AMT_CREDIT_SUM_LIMIT_sum', 'prev_app_DAYS_FIRST_DUE_mean', 'prev_app_HOUR_APPR_PROCESS_START_max', 'prev_app_NAME_CASH_LOAN_PURPOSE_XNA_count_norm', 'prev_app_CODE_REJECT_REASON_CLIENT_count_norm', 'prev_app_AMT_APPLICATION_count', 'prev_app_CHANNEL_TYPE_Car dealer_count', 'prev_app_PRODUCT_COMBINATION_POS mobile with interest_count', 'prev_app_NAME_GOODS_CATEGORY_Animals_count_norm', 'prev_app_DAYS_LAST_DUE_mean', 'prev_app_NFLAG_INSURED_ON_APPROVAL_count', 'prev_app_DAYS_FIRST_DRAWING_min', 'prev_app_NAME_CASH_LOAN_PURPOSE_Hobby_count_norm', 'prev_app_NAME_PORTFOLIO_XNA_count_norm', 'bureau_CNT_CREDIT_PROLONG_sum', 'YEARS_BUILD_MEDI', 'APARTMENTS_MODE', 'bureau_DAYS_CREDIT_ENDDATE_sum', 'bureau_CREDIT_ACTIVE_Closed_count', 'prev_app_NAME_SELLER_INDUSTRY_Furniture_count', 'prev_app_NAME_SELLER_INDUSTRY_XNA_count', 'prev_app_NAME_GOODS_CATEGORY_XNA_count', 'prev_app_DAYS_LAST_DUE_1ST_VERSION_count', 'prev_app_NAME_PRODUCT_TYPE_x-sell_count', 'NONLIVINGAPARTMENTS_MEDI', 'prev_app_NAME_CASH_LOAN_PURPOSE_XAP_count_norm', 'prev_app_CODE_REJECT_REASON_SYSTEM_count_norm', 'ELEVATORS_MODE', 'prev_app_NAME_YIELD_GROUP_XNA_count_norm', 'prev_app_NAME_PORTFOLIO_POS_count', 'prev_app_DAYS_TERMINATION_mean', 'prev_app_DAYS_LAST_DUE_1ST_VERSION_mean', 'prev_app_DAYS_LAST_DUE_1ST_VERSION_max', 'prev_app_NAME_PORTFOLIO_Cash_count_norm', 'prev_app_RATE_INTEREST_PRIVILEGED_count', 'BASEMENTAREA_MEDI', 'AMT_GOODS_PRICE', 'ENTRANCES_MODE', 'prev_app_PRODUCT_COMBINATION_Cash_count_norm', 'LIVINGAPARTMENTS_AVG', 'bureau_AMT_CREDIT_SUM_count', 'prev_app_RATE_DOWN_PAYMENT_sum', 'prev_app_NAME_CONTRACT_STATUS_Unused offer_count_norm', 'ORGANIZATION_TYPE_XNA', 'bureau_DAYS_CREDIT_UPDATE_count', 'prev_app_HOUR_APPR_PROCESS_START_sum', 'prev_app_PRODUCT_COMBINATION_Card Street_count_norm', 'prev_app_AMT_CREDIT_max', 'prev_app_SELLERPLACE_AREA_sum', 'YEARS_BEGINEXPLUATATION_MEDI', 'bureau_AMT_CREDIT_SUM_OVERDUE_count', 'prev_app_AMT_DOWN_PAYMENT_min', 'ENTRANCES_MEDI', 'prev_app_CNT_PAYMENT_max', 'prev_app_SELLERPLACE_AREA_count', 'FLOORSMIN_MODE', 'prev_app_RATE_INTEREST_PRIVILEGED_max', 'prev_app_AMT_ANNUITY_max', 'bureau_DAYS_CREDIT_UPDATE_sum', 'prev_app_DAYS_LAST_DUE_1ST_VERSION_sum', 'bureau_CREDIT_DAY_OVERDUE_sum', 'prev_app_NFLAG_LAST_APPL_IN_DAY_sum', 'LIVINGAREA_AVG', 'COMMONAREA_MODE', 'COMMONAREA_MEDI', 'prev_app_DAYS_DECISION_count', 'bureau_AMT_CREDIT_SUM_DEBT_count', 'prev_app_SELLERPLACE_AREA_min', 'prev_app_NAME_CONTRACT_TYPE_Cash loans_count', 'prev_app_AMT_CREDIT_mean']
```


```python
train_coll_removed = train_df.drop(columns = cols_to_remove)
test_coll_removed = test_df.drop(columns = cols_to_remove)

print('Training Colls Removed Shape: ', train_coll_removed.shape)
print('Testing Colls Removed Shape: ', test_coll_removed.shape)
```

    Training Colls Removed Shape:  (307511, 534)
    Testing Colls Removed Shape:  (48744, 533)
    

Save file supaya tidak perlu kalkulasi lagi


```python
train_coll_removed.to_csv('/content/gdrive/MyDrive/RAKAMIN/train_app_bureau_prev_colls_removed.csv', index = False)
test_coll_removed.to_csv('/content/gdrive/MyDrive/RAKAMIN/test_app_bureau_prev_colls_removed.csv', index = False)
```

## Hyperparameter Tuning


```python
%%time

train_coll_removed = pd.read_csv('/content/gdrive/MyDrive/RAKAMIN/train_app_bureau_prev_colls_removed.csv')
test_coll_removed = pd.read_csv('/content/gdrive/MyDrive/RAKAMIN/test_app_bureau_prev_colls_removed.csv')

print(f'Training data shape: {train_coll_removed.shape}')
print(f'Testing data shape: {test_coll_removed.shape}')
```

    Training data shape: (307511, 534)
    Testing data shape: (48744, 533)
    CPU times: user 22.3 s, sys: 3.51 s, total: 25.8 s
    Wall time: 27.9 s
    


```python
train, test, features = preprocess(train_coll_removed, test_coll_removed)
```

    Training data shape:  (307511, 533)
    Testing data shape:  (48744, 533)
    

### Logreg, SGD, dan LightGBM
**Logreg score: 0.71**

**SGD score: 0.74 +- 0.01**

**LightGBM score: 0.77**


```python
# Run kalau belum define train labels
train_labels = train_coll_removed['TARGET']
```


```python
# Make the model with the specified regularization parameter
log_reg = LogisticRegression(C = 0.0001)
```


```python
log_reg = roc_auc_plot(train, train_labels, cv=kfold, classifier=log_reg)
```


    
![png](README_files/README_233_0.png)
    



```python
# Make predictions
# Make sure to select the second column only
log_reg_pred = log_reg.predict_proba(test)[:, 1]

# Submission dataframe
submit = test_coll_removed[['SK_ID_CURR']]
submit['TARGET'] = log_reg_pred

submit.head()
```





  <div id="df-56441f30-fe64-446e-aeb0-13d155499beb" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SK_ID_CURR</th>
      <th>TARGET</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100001</td>
      <td>0.067107</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100005</td>
      <td>0.116815</td>
    </tr>
    <tr>
      <th>2</th>
      <td>100013</td>
      <td>0.055242</td>
    </tr>
    <tr>
      <th>3</th>
      <td>100028</td>
      <td>0.078205</td>
    </tr>
    <tr>
      <th>4</th>
      <td>100038</td>
      <td>0.116298</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-56441f30-fe64-446e-aeb0-13d155499beb')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-56441f30-fe64-446e-aeb0-13d155499beb button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-56441f30-fe64-446e-aeb0-13d155499beb');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-1c57707d-b07d-4666-815a-c56bd1ec51ff">
  <button class="colab-df-quickchart" onclick="quickchart('df-1c57707d-b07d-4666-815a-c56bd1ec51ff')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-1c57707d-b07d-4666-815a-c56bd1ec51ff button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>
    </div>
  </div>





```python
plt.hist(log_reg_pred)
```




    (array([4.5000e+02, 7.4120e+03, 1.5865e+04, 1.3724e+04, 7.3440e+03,
            2.8110e+03, 8.8600e+02, 2.0300e+02, 4.3000e+01, 6.0000e+00]),
     array([0.02998749, 0.04596552, 0.06194356, 0.07792159, 0.09389962,
            0.10987765, 0.12585569, 0.14183372, 0.15781175, 0.17378978,
            0.18976782]),
     <BarContainer object of 10 artists>)




    
![png](README_files/README_235_1.png)
    



```python
# Save the submission to a csv file
submit.to_csv('/content/gdrive/MyDrive/RAKAMIN/log_reg_data_final.csv', index = False)
```


```python
sgd_classifier = SGDClassifier(loss='modified_huber', random_state=2024)
```


```python
sgd_classifier = roc_auc_plot(train, train_labels, cv=kfold, classifier=sgd_classifier)
```


    
![png](README_files/README_238_0.png)
    



```python
# Make predictions on the test data
predictions = sgd_classifier.predict_proba(test)[:, 1]

# Make a submission dataframe
submit = test_coll_removed[['SK_ID_CURR']]
submit['TARGET'] = predictions
print(submit.head())
```

       SK_ID_CURR    TARGET
    0      100001  0.000000
    1      100005  0.221357
    2      100013  0.061636
    3      100028  0.092986
    4      100038  0.273369
    


```python
# Save the submission to a csv file
submit.to_csv('/content/gdrive/MyDrive/RAKAMIN/sgd_data_final.csv', index = False)
```


```python
lgb = LGBMClassifier(n_estimators=1000, objective = 'binary',
                     class_weight = 'balanced', learning_rate = 0.05,
                     reg_alpha = 0.1, reg_lambda = 0.1,
                     subsample = 0.8, n_jobs = -1, random_state = 50)
```


```python
lgb = roc_auc_plot(train, train_labels, cv=kfold, classifier=lgb)
```

    [LightGBM] [Info] Number of positive: 19876, number of negative: 226132
    [LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 1.206577 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 30709
    [LightGBM] [Info] Number of data points in the train set: 246008, number of used features: 512
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.500000 -> initscore=0.000000
    [LightGBM] [Info] Start training from score 0.000000
    [LightGBM] [Info] Number of positive: 19888, number of negative: 226121
    [LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 1.325580 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 30782
    [LightGBM] [Info] Number of data points in the train set: 246009, number of used features: 509
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.500000 -> initscore=0.000000
    [LightGBM] [Info] Start training from score 0.000000
    [LightGBM] [Info] Number of positive: 19743, number of negative: 226266
    [LightGBM] [Info] Auto-choosing row-wise multi-threading, the overhead of testing was 0.843186 seconds.
    You can set `force_row_wise=true` to remove the overhead.
    And if memory is not enough, you can set `force_col_wise=true`.
    [LightGBM] [Info] Total Bins 30759
    [LightGBM] [Info] Number of data points in the train set: 246009, number of used features: 510
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.500000 -> initscore=0.000000
    [LightGBM] [Info] Start training from score 0.000000
    [LightGBM] [Info] Number of positive: 19921, number of negative: 226088
    [LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 1.381916 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 30786
    [LightGBM] [Info] Number of data points in the train set: 246009, number of used features: 508
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.500000 -> initscore=-0.000000
    [LightGBM] [Info] Start training from score -0.000000
    [LightGBM] [Info] Number of positive: 19872, number of negative: 226137
    [LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 1.288178 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 30758
    [LightGBM] [Info] Number of data points in the train set: 246009, number of used features: 508
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.500000 -> initscore=-0.000000
    [LightGBM] [Info] Start training from score -0.000000
    


    
![png](README_files/README_242_1.png)
    



```python
# Make predictions on the test data
predictions = lgb.predict_proba(test)[:, 1]

# Make a submission dataframe
submit = test_coll_removed[['SK_ID_CURR']]
submit['TARGET'] = predictions
print(submit.head())
```

       SK_ID_CURR    TARGET
    0      100001  0.170696
    1      100005  0.431899
    2      100013  0.041342
    3      100028  0.178551
    4      100038  0.680210
    


```python
# Save the submission dataframe
submit.to_csv('/content/gdrive/MyDrive/RAKAMIN/lgb_data_final.csv', index = False)
```

### Tuning and Result
Belum dapat hasil karena terlalu lama. *Cek metode lain*


```python
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score
```


```python
#List Hyperparameters yang akan diuji
# penalty = ['l1', 'l2']
C = np.logspace(-3, 1, 5)
print(C)
```

    [1.e-03 1.e-02 1.e-01 1.e+00 1.e+01]
    


```python
#Menjadikan ke dalam bentuk dictionary
hyperparameters = dict(C=C)
#Membuat Object Logistic Regression
logreg = LogisticRegression()
#Memasukan ke Grid Search
#CV itu Cross Validation
#Menggunakan 5-Fold CV
clf = GridSearchCV(logreg, hyperparameters, cv=5)
#Fitting Model
best_model = clf.fit(train,train_labels)
#Nilai hyperparameters terbaik
# print('Best Penalty:', best_model.best_estimator_.get_params()['penalty'])
print('Best C:', best_model.best_estimator_.get_params()['C'])
#Prediksi menggunakan model baru
y_pred = best_model.predict(test)
y_test = test_coll_removed['TARGET']
#Check performa dari model
print(classification_report(y_test, y_pred))
roc_auc_score(y_test, y_pred)
```


    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    <ipython-input-30-d256962b955b> in <cell line: 10>()
          8 clf = GridSearchCV(logreg, hyperparameters, cv=5)
          9 #Fitting Model
    ---> 10 best_model = clf.fit(train,train_labels)
         11 #Nilai hyperparameters terbaik
         12 # print('Best Penalty:', best_model.best_estimator_.get_params()['penalty'])
    

    /usr/local/lib/python3.10/dist-packages/sklearn/base.py in wrapper(estimator, *args, **kwargs)
       1349                 )
       1350             ):
    -> 1351                 return fit_method(estimator, *args, **kwargs)
       1352 
       1353         return wrapper
    

    /usr/local/lib/python3.10/dist-packages/sklearn/model_selection/_search.py in fit(self, X, y, **params)
        968                 return results
        969 
    --> 970             self._run_search(evaluate_candidates)
        971 
        972             # multimetric is determined here because in the case of a callable
    

    /usr/local/lib/python3.10/dist-packages/sklearn/model_selection/_search.py in _run_search(self, evaluate_candidates)
       1525     def _run_search(self, evaluate_candidates):
       1526         """Search all candidates in param_grid"""
    -> 1527         evaluate_candidates(ParameterGrid(self.param_grid))
       1528 
       1529 
    

    /usr/local/lib/python3.10/dist-packages/sklearn/model_selection/_search.py in evaluate_candidates(candidate_params, cv, more_results)
        914                     )
        915 
    --> 916                 out = parallel(
        917                     delayed(_fit_and_score)(
        918                         clone(base_estimator),
    

    /usr/local/lib/python3.10/dist-packages/sklearn/utils/parallel.py in __call__(self, iterable)
         65             for delayed_func, args, kwargs in iterable
         66         )
    ---> 67         return super().__call__(iterable_with_config)
         68 
         69 
    

    /usr/local/lib/python3.10/dist-packages/joblib/parallel.py in __call__(self, iterable)
       1861             output = self._get_sequential_output(iterable)
       1862             next(output)
    -> 1863             return output if self.return_generator else list(output)
       1864 
       1865         # Let's create an ID that uniquely identifies the current call. If the
    

    /usr/local/lib/python3.10/dist-packages/joblib/parallel.py in _get_sequential_output(self, iterable)
       1790                 self.n_dispatched_batches += 1
       1791                 self.n_dispatched_tasks += 1
    -> 1792                 res = func(*args, **kwargs)
       1793                 self.n_completed_tasks += 1
       1794                 self.print_progress()
    

    /usr/local/lib/python3.10/dist-packages/sklearn/utils/parallel.py in __call__(self, *args, **kwargs)
        127             config = {}
        128         with config_context(**config):
    --> 129             return self.function(*args, **kwargs)
    

    /usr/local/lib/python3.10/dist-packages/sklearn/model_selection/_validation.py in _fit_and_score(estimator, X, y, scorer, train, test, verbose, parameters, fit_params, score_params, return_train_score, return_parameters, return_n_test_samples, return_times, return_estimator, split_progress, candidate_progress, error_score)
        888             estimator.fit(X_train, **fit_params)
        889         else:
    --> 890             estimator.fit(X_train, y_train, **fit_params)
        891 
        892     except Exception:
    

    /usr/local/lib/python3.10/dist-packages/sklearn/base.py in wrapper(estimator, *args, **kwargs)
       1349                 )
       1350             ):
    -> 1351                 return fit_method(estimator, *args, **kwargs)
       1352 
       1353         return wrapper
    

    /usr/local/lib/python3.10/dist-packages/sklearn/linear_model/_logistic.py in fit(self, X, y, sample_weight)
       1294             n_threads = 1
       1295 
    -> 1296         fold_coefs_ = Parallel(n_jobs=self.n_jobs, verbose=self.verbose, prefer=prefer)(
       1297             path_func(
       1298                 X,
    

    /usr/local/lib/python3.10/dist-packages/sklearn/utils/parallel.py in __call__(self, iterable)
         65             for delayed_func, args, kwargs in iterable
         66         )
    ---> 67         return super().__call__(iterable_with_config)
         68 
         69 
    

    /usr/local/lib/python3.10/dist-packages/joblib/parallel.py in __call__(self, iterable)
       1861             output = self._get_sequential_output(iterable)
       1862             next(output)
    -> 1863             return output if self.return_generator else list(output)
       1864 
       1865         # Let's create an ID that uniquely identifies the current call. If the
    

    /usr/local/lib/python3.10/dist-packages/joblib/parallel.py in _get_sequential_output(self, iterable)
       1790                 self.n_dispatched_batches += 1
       1791                 self.n_dispatched_tasks += 1
    -> 1792                 res = func(*args, **kwargs)
       1793                 self.n_completed_tasks += 1
       1794                 self.print_progress()
    

    /usr/local/lib/python3.10/dist-packages/sklearn/utils/parallel.py in __call__(self, *args, **kwargs)
        127             config = {}
        128         with config_context(**config):
    --> 129             return self.function(*args, **kwargs)
    

    /usr/local/lib/python3.10/dist-packages/sklearn/linear_model/_logistic.py in _logistic_regression_path(X, y, pos_class, Cs, fit_intercept, max_iter, tol, verbose, solver, coef, class_weight, dual, penalty, intercept_scaling, multi_class, random_state, check_input, max_squared_sum, sample_weight, l1_ratio, n_threads)
        453                 np.searchsorted(np.array([0, 1, 2, 3]), verbose)
        454             ]
    --> 455             opt_res = optimize.minimize(
        456                 func,
        457                 w0,
    

    /usr/local/lib/python3.10/dist-packages/scipy/optimize/_minimize.py in minimize(fun, x0, args, method, jac, hess, hessp, bounds, constraints, tol, callback, options)
        708                                  **options)
        709     elif meth == 'l-bfgs-b':
    --> 710         res = _minimize_lbfgsb(fun, x0, args, jac, bounds,
        711                                callback=callback, **options)
        712     elif meth == 'tnc':
    

    /usr/local/lib/python3.10/dist-packages/scipy/optimize/_lbfgsb_py.py in _minimize_lbfgsb(fun, x0, args, jac, bounds, disp, maxcor, ftol, gtol, eps, maxfun, maxiter, iprint, callback, maxls, finite_diff_rel_step, **unknown_options)
        363             # until the completion of the current minimization iteration.
        364             # Overwrite f and g:
    --> 365             f, g = func_and_grad(x)
        366         elif task_str.startswith(b'NEW_X'):
        367             # new iteration
    

    /usr/local/lib/python3.10/dist-packages/scipy/optimize/_differentiable_functions.py in fun_and_grad(self, x)
        283         if not np.array_equal(x, self.x):
        284             self._update_x_impl(x)
    --> 285         self._update_fun()
        286         self._update_grad()
        287         return self.f, self.g
    

    /usr/local/lib/python3.10/dist-packages/scipy/optimize/_differentiable_functions.py in _update_fun(self)
        249     def _update_fun(self):
        250         if not self.f_updated:
    --> 251             self._update_fun_impl()
        252             self.f_updated = True
        253 
    

    /usr/local/lib/python3.10/dist-packages/scipy/optimize/_differentiable_functions.py in update_fun()
        153 
        154         def update_fun():
    --> 155             self.f = fun_wrapped(self.x)
        156 
        157         self._update_fun_impl = update_fun
    

    /usr/local/lib/python3.10/dist-packages/scipy/optimize/_differentiable_functions.py in fun_wrapped(x)
        135             # Overwriting results in undefined behaviour because
        136             # fun(self.x) will change self.x, with the two no longer linked.
    --> 137             fx = fun(np.copy(x), *args)
        138             # Make sure the function returns a true scalar
        139             if not np.isscalar(fx):
    

    /usr/local/lib/python3.10/dist-packages/scipy/optimize/_optimize.py in __call__(self, x, *args)
         75     def __call__(self, x, *args):
         76         """ returns the function value """
    ---> 77         self._compute_if_needed(x, *args)
         78         return self._value
         79 
    

    /usr/local/lib/python3.10/dist-packages/scipy/optimize/_optimize.py in _compute_if_needed(self, x, *args)
         69         if not np.all(x == self.x) or self._value is None or self.jac is None:
         70             self.x = np.asarray(x).copy()
    ---> 71             fg = self.fun(x, *args)
         72             self.jac = fg[1]
         73             self._value = fg[0]
    

    /usr/local/lib/python3.10/dist-packages/sklearn/linear_model/_linear_loss.py in loss_gradient(self, coef, X, y, sample_weight, l2_reg_strength, n_threads, raw_prediction)
        294             grad[:n_features] = X.T @ grad_pointwise + l2_reg_strength * weights
        295             if self.fit_intercept:
    --> 296                 grad[-1] = grad_pointwise.sum()
        297         else:
        298             grad = np.empty((n_classes, n_dof), dtype=weights.dtype, order="F")
    

    /usr/local/lib/python3.10/dist-packages/numpy/core/_methods.py in _sum(a, axis, dtype, out, keepdims, initial, where)
         44     return umr_minimum(a, axis, None, out, keepdims, initial, where)
         45 
    ---> 46 def _sum(a, axis=None, dtype=None, out=None, keepdims=False,
         47          initial=_NoValue, where=True):
         48     return umr_sum(a, axis, dtype, out, keepdims, initial, where)
    

    KeyboardInterrupt: 



```python
#List Hyperparameters yang akan diuji
loss = ['hinge', 'log_loss', 'modified_huber', 'squared_error', 'huber']
learning_rate = ['constant', 'optimal', 'invscaling', 'adaptive']
```


```python
#Menjadikan ke dalam bentuk dictionary
hyperparameters = dict(loss=loss, learning_rate=learning_rate)
#Membuat Object Logistic Regression
sgd = SGDClassifier()
#Memasukan ke Grid Search
#CV itu Cross Validation
#Menggunakan 5-Fold CV
clf = GridSearchCV(sgd, hyperparameters, cv=5)
#Fitting Model
best_model = clf.fit(train,train_labels)
#Nilai hyperparameters terbaik
print('Best Loss:', best_model.best_estimator_.get_params()['loss'])
print('Best Learning Rate:', best_model.best_estimator_.get_params()['learning_rate'])
#Prediksi menggunakan model baru
y_pred = best_model.predict(test)
#Check performa dari model
print(classification_report(y_test, y_pred))
roc_auc_score(y_test, y_pred)
```

# Kesimpulan
1. Pada data akhir (setelah menggabung data application, previous application, dan bureau serta telah melakukan feature engineering dan feature selection), LightGBM menghasilkan score terbaik dengan nilai ROC-AUC sebesar 0.77
2. Logistic Regression mengalami penurunan nilai ROC-AUC pada data akhir, kemungkinan terjadi overfit

# Saran
1. Diperlukan campaign lebih untuk para pensiunan karena umumnya mereka tidak mengalami kesulitan membayar
2. Jadikan pelanggan dengan status pendidikan Higher Education sebagai target utama, karena pelanggan jenis ini umumnya tidak mengalami kesulitan membayar


# Referensi:
1. https://www.kaggle.com/code/willkoehrsen/start-here-a-gentle-introduction
2. https://www.kaggle.com/code/jsaguiar/lightgbm-with-simple-features
