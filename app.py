import streamlit as st
import pickle
import numpy as np
import pandas as pd
# App title
st.title("SEGMENTASI PELANGGAN")
st.markdown('''
- Parameter untuk menentukan cluster : Pendapatan Pertahun dan Skor Pengeluaran
- ID Pelanggan, Jenis Kelamin dan Usia hanya sebagai identitas data
- Clustering menggunakan KMeans
- Jumlah cluster : 5
''')
st.write('---')

# Sidebar
st.sidebar.subheader('INPUT PARAMETERS')

id_p = st.sidebar.number_input('ID Pelanggan :', value=int(), step=int())

jk = st.sidebar.selectbox('Jenis Kelamin :', ('Pria', 'Wanita'))
if jk == 'Pria':
    jk_v = 1
else:
    jk_v = 0

usia = st.sidebar.number_input('Usia :', value=int(), step=int())
pp = st.sidebar.number_input('Pendapatan Pertahun (Rp) :', value=int(), step=int())
sp = st.sidebar.slider('Skor Pengeluaran:', 0, 100)   

pp = pp/10000000
data = np.array([id_p,jk_v,usia,pp,sp])
data = data.reshape(1,-1)
df_data = pd.DataFrame(data, columns = ['id_pelanggan','	jenis_kelamin','	usia','pendapatan_pertahun','skor_pengeluaran'])
 
fitur = np.array([pp,sp])
fitur = fitur.reshape(1,-1)
df_fitur = pd.DataFrame(fitur, columns = ['pendapatan_pertahun','skor_pengeluaran'])
filename = 'clustering.sav'
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.predict(fitur)
if result == 0:
    r = 'HASIL CLUSTER : PELANGGAN KATEGORI 1  \n Penghasilan Rendah, Skor Pengeluaran Tinggi '
elif result == 1:
    r = 'HASIL CLUSTER : PELANGGAN KATEGORI 2  \n Penghasilan Menengah, Skor Pengeluaran Menengah'
elif result == 2:
    r = 'HASIL CLUSTER : PELANGGAN KATEGORI 3  \n Penghasilan Tinggi, Skor Pengeluaran Rendah'  
elif result == 3:
    r = 'HASIL CLUSTER : PELANGGAN KATEGORI 4  \n Penghasilan Rendah, Skor Pengeluaran Rendah'
else:
    r = 'HASIL CLUSTER : PELANGGAN KATEGORI 5  \n Penghasilan Tinggi, Skor Pengeluaran Tinggi'
#result = fitur.reshape(1,-1)
df_result = pd.DataFrame(result, columns = ['labels'])

if st.button('Process') :
    st.header(r)
    st.write('---')
    st.subheader('Data Pelanggan')
    st.write(df_data)
    st.subheader('Fitur Clustering')
    st.write(df_fitur)
    st.subheader('Hasil Clustering')
    st.write(df_result)
    


    


